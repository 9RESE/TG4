# ADR-006: Consolidated Code Review Fixes

**Status**: Accepted
**Date**: 2025-12-19
**Decision Makers**: Development Team

## Context

A comprehensive consolidated code review identified 13 P0 (critical) and 12 P1 (high-priority) issues across the codebase. These issues covered security vulnerabilities, race conditions, logic errors, and missing validations that needed to be addressed before production deployment.

The review document: `docs/development/reviews/full/review-2/FINAL-REPORT-CONSOLIDATED.md`

## Decision

### P0 Critical Fixes Implemented

#### 1. API Security Layer (5 issues)

**Problems**:
- No authentication on API endpoints
- No rate limiting
- No CORS configuration
- No request size limits
- No async timeouts

**Solution**: Created comprehensive `triplegain/src/api/security.py` module:

```python
# Rate limiting tiers
RATE_LIMIT_DEFAULT = 60      # requests/min
RATE_LIMIT_EXPENSIVE = 5     # LLM calls
RATE_LIMIT_MODERATE = 30     # Database queries

# Security middleware stack
app.add_middleware(RequestSizeLimitMiddleware, max_size=1_048_576)  # 1MB
app.add_middleware(TimeoutMiddleware, timeout=45.0)  # 45s
app.add_middleware(RateLimitMiddleware, config=security_config)
app.add_middleware(CORSMiddleware, ...)
```

Features:
- API key authentication with role-based access control
- Tiered rate limiting (5/30/60 requests per minute by endpoint)
- CORS configurable via environment variables
- 1MB request size limit
- 45s request timeout

#### 2. Trading Decision Consensus Logic (P0-AGT-01)

**Problem**: 3-way tie (2-2-2) picked action alphabetically, potentially executing BUY on 33% consensus.

**Solution**: Force HOLD when consensus ≤ 50%:

```python
# triplegain/src/agents/trading_decision.py
if consensus_strength <= 0.5:
    winning_action = 'HOLD'
    logger.info(f"Forcing HOLD due to weak consensus: {consensus_strength:.0%}")
```

**Rationale**: A 50% or lower consensus indicates models strongly disagree. Acting on uncertain signals increases risk.

#### 3. DCA Rounding Overflow (P0-AGT-02)

**Problem**: `quantize(Decimal('0.01'))` with default ROUND_HALF_EVEN could round up, causing total to exceed original amount.

**Solution**: Use ROUND_DOWN:

```python
# triplegain/src/agents/portfolio_rebalance.py
from decimal import ROUND_DOWN
rounded_batch_amount = base_batch_amount.quantize(Decimal('0.01'), rounding=ROUND_DOWN)
```

#### 4. LLM Response JSON Validation (P0-LLM-02)

**Problem**: No handling for markdown-wrapped JSON responses (```json...```).

**Solution**: Added robust `parse_json_response()` in `triplegain/src/llm/clients/base.py`:

```python
def parse_json_response(response_text: str) -> tuple[Optional[dict], Optional[str]]:
    """
    Parse JSON from LLM response, handling:
    - Plain JSON
    - Markdown-wrapped JSON (```json\n{...}\n```)
    - JSON embedded in text
    """
```

#### 5. Circuit Breaker Race Condition (P0-RSK-01)

**Problem**: `validate_trade()` could receive stale external state, bypassing circuit breaker.

**Solution**:
1. Added `threading.Lock()` for thread safety
2. Circuit breaker checks always use internal state

```python
# triplegain/src/risk/rules_engine.py
with self._state_lock:
    # CRITICAL: Circuit breaker checks ALWAYS use internal state
    if internal_state.trading_halted:
        return RiskValidation(status=ValidationStatus.HALTED, ...)
```

#### 6. Leverage in Exposure Calculation (P0-RSK-02)

**Problem**: Exposure calculation ignored leverage. $1000 at 5x = 10% exposure on $10k portfolio (should be 50%).

**Solution**:

```python
# triplegain/src/risk/rules_engine.py
actual_exposure_usd = proposal.size_usd * proposal.leverage
position_exposure = (actual_exposure_usd / float(state.current_equity)) * 100
```

#### 7. Risk Engine Input Validation (P0-RSK-03)

**Problem**: No validation on TradeProposal inputs. Negative size, zero entry price would cause undefined behavior.

**Solution**: Added `TradeProposalValidationError` and validation in `__post_init__()`:

```python
@dataclass
class TradeProposal:
    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        # Validates: symbol, side, size_usd, entry_price, stop_loss,
        # take_profit, leverage, confidence
        if self.size_usd <= 0:
            errors.append(f"Size must be positive, got {self.size_usd}")
        # ... comprehensive validation
```

### P1 High-Priority Fixes Implemented

#### 1. MessageBus Deadlock (P1-ORC-01)

**Problem**: `publish()` held lock while calling handlers. If handler published a message → deadlock.

**Solution**: Two-phase publish - copy subscribers under lock, call handlers outside:

```python
async def publish(self, message: Message) -> int:
    # Phase 1: Under lock
    async with self._lock:
        subscriptions = list(self._subscriptions.get(message.topic, []))

    # Phase 2: Outside lock - handlers can safely publish
    for sub in subscriptions:
        await sub.handler(message)
```

#### 2. TA Fallback Confidence (P1-AGT-02)

**Problem**: Fallback confidence of 0.4 too high for heuristic-based analysis.

**Solution**: Reduced to 0.25:

```python
# triplegain/src/agents/technical_analysis.py
'confidence': 0.25,  # Conservative confidence for heuristic fallback
```

## Consequences

### Positive

1. **Security**: Complete API protection stack (auth, rate limiting, CORS, limits, timeouts)
2. **Correctness**: Thread-safe risk engine with proper validation
3. **Reliability**: No deadlocks in message bus, conservative fallbacks
4. **Risk Management**: Leverage properly factored into exposure calculations
5. **Decision Quality**: Weak consensus forces HOLD instead of acting on uncertainty

### Negative

1. **Complexity**: Additional security middleware layer
2. **Configuration**: Environment variables required for JWT, CORS
3. **Test Updates**: 4 tests needed updating to reflect corrected behavior

### Risks Mitigated

- **P0 Security**: Unauthorized access, DDoS, CSRF, memory exhaustion
- **P0 Logic**: Trading on 33% consensus, exposure underestimation, circuit breaker bypass
- **P0 Data**: Input validation prevents undefined behavior
- **P1 Concurrency**: MessageBus deadlock eliminated

## Implementation

- **Files Modified**: 10 source files + 2 test files
- **New File**: `triplegain/src/api/security.py`
- **Tests**: 917 passing (1 new test added)
- **Coverage**: 87%

## Related Documents

- [Consolidated Review](../../development/reviews/full/review-2/FINAL-REPORT-CONSOLIDATED.md)
- [ADR-005: Security Robustness Fixes](ADR-005-security-robustness-fixes.md)
- [Risk Management Standards](../../team/standards/risk-management-standards.md)

---

*ADR-006 - December 2025*
