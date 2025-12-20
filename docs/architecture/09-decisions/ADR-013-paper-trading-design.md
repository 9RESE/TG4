# ADR-013: Paper Trading Design

**Status**: Accepted
**Date**: 2025-12-19
**Decision Makers**: Development Team
**Context**: Phase 6 Implementation

---

## Context

TripleGain is an LLM-assisted trading system that requires extensive testing before live trading. The system had partial paper trading infrastructure (config existed but was never read, mock mode used hardcoded values) but lacked complete implementation for safe testing.

### Problem Statement

1. **Safety Risk**: Without proper paper trading, the system could accidentally execute real trades during testing
2. **Realism Gap**: Hardcoded mock values (2s delay, static prices) don't reflect real market conditions
3. **Data Mixing**: No isolation between paper and live trading data
4. **Session Loss**: No persistence of paper trading state across restarts

### Requirements

1. Paper trading must be the **enforced default** with explicit opt-in for live trading
2. Simulation must be configurable (slippage, fees, delays)
3. Paper and live data must be completely isolated
4. Sessions must persist across restarts
5. Full API access for monitoring and control

---

## Decision

### 1. Dual-Confirmation for Live Trading

**Decision**: Require BOTH environment variable AND config file to agree on "live" mode.

**Rationale**:
- Single point of failure is too risky for financial systems
- Prevents accidental live trading from config typos or env var pollution
- Explicit confirmation string adds third layer of safety

```python
# Requires ALL three:
# 1. TRIPLEGAIN_TRADING_MODE=live (env)
# 2. execution.yaml trading_mode: live (config)
# 3. TRIPLEGAIN_CONFIRM_LIVE_TRADING=I_UNDERSTAND_THE_RISKS (env)
```

**Alternatives Considered**:
- Single env var: Too easy to accidentally set
- Config only: Could be accidentally committed
- Runtime flag: Could be forgotten

### 2. OrderStatus.REJECTED vs OrderStatus.ERROR

**Decision**: Add `OrderStatus.REJECTED` for business logic rejections (insufficient balance, position limits).

**Rationale**:
- `ERROR` implies system/technical failure requiring investigation
- `REJECTED` implies expected business rule enforcement
- Different handling in monitoring/alerting systems
- Clearer audit trail for compliance

**Impact**: Added new enum value, updated paper executor and tests.

### 3. Thread-Safe Statistics

**Decision**: Use `asyncio.Lock` to protect concurrent counter increments.

**Rationale**:
- Paper executor may process multiple orders concurrently
- Python's `+=` is not atomic for concurrent access
- Lock overhead is minimal (<1ms) for infrequent operations
- Prevents race conditions in statistics reporting

**Alternatives Considered**:
- `threading.Lock`: Not appropriate for async context
- `asyncio.Queue`: Overkill for simple counters
- No locking: Acceptable for non-critical stats, but incorrect values could mask bugs

### 4. Async Price Source

**Decision**: Create separate `get_price_async()` method for async contexts.

**Rationale**:
- Database queries must be awaited in async context
- Sync `get_price()` method remains for backwards compatibility
- Clear API contract for callers
- Prevents blocking the event loop with sync database calls

**Implementation**:
```python
# Async context (coordinator, executor)
price = await source.get_price_async(symbol)

# Sync context (testing, CLI)
price = source.get_price(symbol)
```

### 5. Session Persistence Schema

**Decision**: Use existing migration 005 schema with `id`, `current_balances`, `initial_balances` columns.

**Rationale**:
- Schema was already designed for paper trading
- Column names match domain language
- JSONB for balances allows flexible multi-asset support
- Upsert pattern prevents duplicate sessions

**Key Design Choices**:
- `id` as session identifier (not auto-increment)
- Status enum: `active`, `paused`, `ended`
- Nullable `ended_at` for active sessions
- JSONB for balance dictionaries

### 6. Price Cache Timestamp Comparison

**Decision**: Reject price updates with older timestamps than cached values.

**Rationale**:
- WebSocket messages may arrive out of order
- Stale prices could cause incorrect order fills
- Timestamp comparison is O(1) with minimal overhead
- Returns boolean to indicate if update was applied

**Implementation**:
```python
def update_price(self, symbol: str, price: Decimal, timestamp: datetime) -> bool:
    if symbol in self._cache_time:
        if self._cache_time[symbol] > timestamp:
            return False  # Rejected stale price
    self._cache[symbol] = price
    self._cache_time[symbol] = timestamp
    return True
```

### 7. Order History Persistence Before Trimming

**Decision**: Persist orders to database before removing from in-memory history.

**Rationale**:
- Memory-limited history (1000 orders) could lose data
- Database provides permanent audit trail
- Enables historical analysis beyond memory window
- Graceful degradation if DB unavailable (logged warning, data lost)

### 8. Rate Limiting for Paper Trading Endpoints

**Decision**: Add paper trading endpoints to rate limit tiers.

**Configuration**:
- `/paper/trade`: Expensive tier (stricter limits)
- `/paper/reset`: Expensive tier (critical operation)
- `/paper/*`: Moderate tier (general endpoints)

**Rationale**:
- Prevents abuse of trade execution
- Protects system from rapid reset attacks
- Consistent with existing security model

---

## Consequences

### Positive

1. **Safety**: Cannot accidentally trade live without explicit triple confirmation
2. **Realism**: Configurable simulation matches real market conditions
3. **Isolation**: Complete separation of paper and live data
4. **Persistence**: Sessions survive restarts, enabling long-term testing
5. **Auditability**: Full trade history with proper status codes
6. **Consistency**: Thread-safe statistics, no race conditions

### Negative

1. **Complexity**: More code to maintain (4 new files, ~1500 lines)
2. **Lock Overhead**: Minimal but non-zero for statistics
3. **Migration Required**: Database migration 005 must be run

### Neutral

1. **API Surface**: 6 new endpoints to document and test
2. **Config Growth**: Additional config section for paper trading

---

## Compliance

### Arc42 Alignment

- **Section 6 (Runtime)**: Paper executor integrates with coordinator's execution flow
- **Section 9 (Decisions)**: This ADR documents key decisions
- **Section 10 (Quality)**: 53 new tests ensure reliability

### Quality Attributes

| Attribute | Impact |
|-----------|--------|
| Safety | HIGH - Dual confirmation prevents accidents |
| Reliability | HIGH - Thread-safe, persistent |
| Testability | HIGH - 53 unit tests, full coverage |
| Maintainability | MEDIUM - Well-structured but adds complexity |

---

## Related Decisions

- [ADR-010: Execution Layer Robustness](ADR-010-execution-layer-robustness.md) - Base execution infrastructure
- [ADR-011: API Security Fixes](ADR-011-api-security-fixes.md) - Rate limiting framework
- [ADR-012: Configuration Integration](ADR-012-configuration-integration-fixes.md) - Config loading patterns

---

## References

- [Phase 6 Feature Documentation](../../development/features/phase-6-paper-trading.md)
- [Phase 6 Implementation Plan](../../development/TripleGain-implementation-plan/phase-6-paper-trading-plan.md)
- [Phase 6 Code Review](../../development/reviews/phase-6/phase-6-code-review.md)
