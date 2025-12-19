# ADR-003: Phase 2 Code Review Fixes

**Status**: Accepted
**Date**: 2025-12-18
**Decision makers**: Development Team, Claude Opus 4.5

## Context

A comprehensive code and logic review of Phase 2 implementation identified several gaps between the design specification and implementation. These issues were categorized by priority:

- **High Priority**: Issues that could cause production problems
- **Medium Priority**: Missing features that reduce system capability
- **Low Priority**: Technical debt and optimization opportunities

The review identified 14 issues across the Risk Management Engine, LLM Clients, Trading Decision Agent, and API layer.

## Decision

We decided to address all identified issues immediately, before proceeding to Phase 3 orchestration, to ensure a solid foundation for the production system.

### Issues Addressed

#### 1. Volatility Spike Circuit Breaker (Risk Engine)

**Problem**: Design specified volatility spike detection (ATR > 3x average), but it was not implemented.

**Solution**: Added `update_volatility()` method to detect spikes and reduce position sizes by 50% during high volatility periods. Added 15-minute cooldown when spike detected.

```python
def update_volatility(self, current_atr: float, avg_atr_20: float) -> bool:
    spike_detected = current_atr > (avg_atr_20 * self.volatility_spike_multiplier)
    if spike_detected:
        self._risk_state.volatility_spike_active = True
        self._apply_cooldown(self.volatility_spike_cooldown_min, "Volatility spike")
    return spike_detected
```

#### 2. Risk State Persistence (Risk Engine)

**Problem**: RiskState was in-memory only. Application restart lost consecutive loss counts, drawdown tracking, and circuit breaker states.

**Solution**: Added `to_dict()` and `from_dict()` serialization methods to RiskState. Added async `persist_state()` and `load_state()` methods that use database storage.

**Migration**: Created `003_risk_state_and_indexes.sql` with `risk_state` and `risk_state_history` tables.

#### 3. Rate Limiting for LLM APIs (LLM Clients)

**Problem**: No rate limiting implemented. Risk of hitting API limits during parallel model queries.

**Solution**: Created `RateLimiter` class with sliding window algorithm. Added `generate_with_retry()` method with exponential backoff retry logic.

```python
class RateLimiter:
    async def acquire(self) -> float:
        # Sliding window algorithm
        # Returns wait time if rate limited
```

#### 4. Thread-Safe Output Caching (Base Agent)

**Problem**: Output caching used `_last_output` instance variable, not thread-safe for concurrent async operations.

**Solution**: Added `asyncio.Lock` for thread-safe cache access. Implemented TTL-based cache expiration per symbol.

```python
self._cache: dict[str, tuple[AgentOutput, datetime]] = {}
self._cache_lock = asyncio.Lock()

async def get_latest_output(self, symbol: str, max_age_seconds: int):
    async with self._cache_lock:
        # Thread-safe cache access
```

#### 5. Entry Strictness Consumption (Risk Engine)

**Problem**: Regime Detection Agent outputs `entry_strictness` but it was never used.

**Solution**: Added `entry_strictness` parameter to `validate_trade()`. Applied confidence adjustment based on strictness level:
- `relaxed`: -0.05 (lower threshold)
- `normal`: 0.0 (no change)
- `strict`: +0.05 (higher threshold)
- `very_strict`: +0.10 (much higher threshold)

#### 6. Correlated Position Checking (Risk Engine)

**Problem**: No correlation checking between positions. BTC/USDT and XRP/USDT are highly correlated.

**Solution**: Added `PAIR_CORRELATIONS` matrix and `_validate_correlation()` method. Default max correlated exposure: 40%.

```python
PAIR_CORRELATIONS = {
    ('BTC/USDT', 'XRP/USDT'): 0.75,
    ('BTC/USDT', 'XRP/BTC'): 0.60,
    ('XRP/USDT', 'XRP/BTC'): 0.85,
}
```

#### 7. Confidence Boost Application (Trading Decision)

**Problem**: Consensus confidence boost was calculated but not applied to output.

**Solution**: Applied boost based on agreement level:
- Unanimous (6/6): +0.15
- Strong majority (5/6): +0.10
- Majority (4/6): +0.05
- Split (<4/6): +0.00

#### 8. Daily/Weekly Reset Automation (Risk Engine)

**Problem**: `reset_daily()` and `reset_weekly()` required manual invocation.

**Solution**: Added `_check_and_reset_periods()` called at the start of `validate_trade()`. Automatically resets when crossing day/week boundaries.

#### 9. TradingDecisionAgent Base Class (Architecture)

**Problem**: Agent bypassed `super().__init__()`, losing stats tracking and cache functionality.

**Solution**: Fixed to call base class init with first available client:

```python
first_client = next(iter(llm_clients.values()), None)
super().__init__(llm_client=first_client, ...)
```

#### 10. Model Timeout Handling (Trading Decision)

**Problem**: `asyncio.gather` with timeout lost all results if any model timed out.

**Solution**: Changed to `asyncio.wait` with `return_when=ALL_COMPLETED`:

```python
done, pending = await asyncio.wait(tasks.keys(), timeout=self.timeout_seconds)
for task in pending:
    task.cancel()  # Cancel slow models but keep completed results
```

#### 11. POST Trigger Endpoints (API)

**Problem**: Only GET endpoints existed for agents. No way to force fresh analysis.

**Solution**: Added POST endpoints:
- `POST /api/v1/agents/ta/{symbol}/run`
- `POST /api/v1/agents/regime/{symbol}/run`

#### 12. Confidence Thresholds from Config (Risk Engine)

**Problem**: Confidence thresholds hardcoded instead of loaded from config.

**Solution**: Now loads from `risk.yaml`:

```yaml
confidence:
  base_minimum: 0.60
  after_3_losses: 0.70
  after_5_losses: 0.80
```

## Consequences

### Positive

1. **Improved Reliability**: State persistence ensures risk limits survive restarts
2. **Better Rate Control**: Rate limiting prevents API quota exhaustion
3. **Thread Safety**: Concurrent operations won't corrupt cache
4. **Full Design Compliance**: All design features now implemented
5. **Partial Results**: Model timeouts no longer lose all results

### Negative

1. **Increased Complexity**: More code to maintain
2. **Database Dependency**: Risk state now requires database connection
3. **Latency**: Rate limiting may add delay to API calls

### Risks Mitigated

1. **Risk of Lost State**: State now persisted
2. **Risk of API Limits**: Rate limiting implemented
3. **Risk of Race Conditions**: Thread-safe caching
4. **Risk of Correlation**: Position correlation checking active

## Alternatives Considered

### 1. Redis for State Persistence

**Rejected**: TimescaleDB already available, adding Redis increases infrastructure complexity.

### 2. External Rate Limiter (e.g., aiolimiter library)

**Rejected**: Simple sliding window sufficient for our needs. Avoiding external dependency.

### 3. Delay Fixes to Phase 4

**Rejected**: These are foundation issues that compound over time. Better to fix now.

---

## Addendum: Deep Audit Fixes (2025-12-18)

A secondary deep audit identified additional issues that were addressed:

#### 13. Model Outcome Tracking (Trading Decision)

**Problem**: `model_comparisons` table stored decisions but never tracked trade outcomes (P&L), making A/B analysis impossible.

**Solution**:
- `_store_model_comparisons()` now stores `price_at_decision`
- Added `update_comparison_outcomes()` method to populate `was_correct` and `outcome_pnl_pct` after 1h/4h/24h

```python
async def update_comparison_outcomes(
    self, symbol: str, timestamp_from: datetime, timestamp_to: datetime,
    price_1h: float, price_4h: float, price_24h: float
) -> int:
    # Updates model_comparisons with outcome data
```

#### 14. `_last_output` Attribute Declaration (Base Agent)

**Problem**: `_last_output` was used in agents but never declared in `__init__`, causing potential AttributeError.

**Solution**: Added `self._last_output: Optional[AgentOutput] = None` in `BaseAgent.__init__()` and added `@property last_output` getter.

#### 15. Zero-Equity Drawdown Edge Case (Risk Engine)

**Problem**: `update_drawdown()` could fail or produce incorrect results when `peak_equity = 0` or `current_equity < 0`.

**Solution**: Enhanced `update_drawdown()` with edge case handling:
- Zero/zero equity → 0% drawdown (initial state)
- Negative equity → >100% drawdown calculation
- New peak detection → Reset drawdown to 0%

```python
def update_drawdown(self):
    if self.current_equity <= 0 and self.peak_equity <= 0:
        self.current_drawdown_pct = 0.0
        return
    # ... rest of method
```

**Tests Added**: 10 new unit tests covering these edge cases (378 total tests, was 368).

---

## Related

- [Phase 2 Deep Audit Review](../../development/reviews/phase-2/phase-2-deep-audit-review.md)
- [Phase 2 Code Review](../../development/reviews/phase-2/phase-2-code-logic-review.md)
- [Phase 2 Feature Documentation](../../development/features/phase-2-core-agents.md)
- [ADR-002: Phase 2 Core Agents Architecture](ADR-002-phase2-core-agents-architecture.md)

---

*ADR-003 - December 2025 (Updated with Deep Audit fixes)*
