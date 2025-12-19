# TripleGain Implementation - Issues & Action Items

**Generated**: 2025-12-19
**Total Issues**: 9 (0 P0, 1 P1, 5 P2, 3 P3)
**Status**: Ready for Paper Trading (all issues non-blocking)

---

## Quick Fix Summary

| Priority | Count | Effort | Category |
|----------|-------|--------|----------|
| P0 (Critical) | 0 | - | - |
| P1 (High) | 1 | 30 min | Security |
| P2 (Medium) | 5 | 4 hr | Code Quality |
| P3 (Low) | 3 | 3 hr | Improvements |

**Total Fix Time**: ~7.5 hours

---

## P1 - High Priority (Fix Before Paper Trading)

### P1-1: API Exception Details Exposed

**File**: `triplegain/src/api/app.py`
**Lines**: 207-209, 251-253, 298-300
**Impact**: Security - Information disclosure vulnerability
**Effort**: 30 minutes

**Current Code**:
```python
except Exception as e:
    logger.error(f"Failed to calculate indicators: {e}", exc_info=True)
    raise HTTPException(status_code=500, detail=str(e))
```

**Fixed Code**:
```python
except Exception as e:
    logger.error("Failed to calculate indicators", exc_info=True)
    raise HTTPException(status_code=500, detail="Internal server error")
```

**Locations to fix**:
- Line 207-209 (indicators endpoint)
- Line 251-253 (snapshot endpoint)
- Line 298-300 (debug prompt endpoint)

---

## P2 - Medium Priority (Fix During Paper Trading)

### P2-1: Supertrend Initial State Logic

**File**: `triplegain/src/data/indicator_library.py`
**Lines**: 932-937
**Impact**: Indicator accuracy - First Supertrend value may not reflect true trend
**Effort**: 30 minutes

**Current Code**:
```python
if closes[period] > upper_band[period]:
    supertrend[period] = lower_band[period]
    direction[period] = 1
else:
    supertrend[period] = upper_band[period]
    direction[period] = -1
```

**Fixed Code**:
```python
mid_band = (upper_band[period] + lower_band[period]) / 2
if closes[period] > mid_band:
    supertrend[period] = lower_band[period]
    direction[period] = 1
else:
    supertrend[period] = upper_band[period]
    direction[period] = -1
```

---

### P2-2: Async Error Handling Silent Failures

**File**: `triplegain/src/data/market_snapshot.py`
**Lines**: 332-347
**Impact**: Reliability - Agents may process stale/incomplete data
**Effort**: 1 hour

**Current Code**:
```python
results = await asyncio.gather(
    *candle_tasks, data_24h_task, order_book_task,
    return_exceptions=True
)
```

**Fixed Code**:
```python
results = await asyncio.gather(
    *candle_tasks, data_24h_task, order_book_task,
    return_exceptions=True
)

# Check for complete failure
candle_results = results[:len(timeframes)]
candle_failures = sum(1 for r in candle_results if isinstance(r, Exception))
if candle_failures >= len(timeframes):
    logger.error(f"All {len(timeframes)} candle fetches failed")
    raise RuntimeError("All candle fetches failed - cannot build snapshot")

# Log individual failures
for i, result in enumerate(candle_results):
    if isinstance(result, Exception):
        logger.warning(f"Candle fetch {i} failed: {result}")
```

---

### P2-3: Truncation Not Logged

**File**: `triplegain/src/llm/prompt_builder.py`
**Lines**: 143-150
**Impact**: Observability - Silent data loss in prompts
**Effort**: 30 minutes

**Current Code**:
```python
if estimated_tokens > max_budget:
    user_message = self.truncate_to_budget(...)
```

**Fixed Code**:
```python
if estimated_tokens > max_budget:
    logger.warning(
        f"Truncating prompt from {estimated_tokens} to {max_budget} tokens "
        f"(agent={agent_name})"
    )
    user_message = self.truncate_to_budget(...)
```

---

### P2-4: Type Coercion in Config Validation

**File**: `triplegain/src/utils/config.py`
**Lines**: 174-175
**Impact**: Reliability - Valid configs may fail validation
**Effort**: 30 minutes

**Current Code**:
```python
rsi_period = rsi.get('period', 14)
if not isinstance(rsi_period, int) or rsi_period <= 0:
```

**Fixed Code**:
```python
try:
    rsi_period = int(rsi.get('period', 14))
except (TypeError, ValueError):
    errors.append("RSI period must be a valid integer")
    rsi_period = 14  # Use default for remaining checks

if rsi_period <= 0:
    errors.append("RSI period must be positive")
```

---

### P2-5: Missing Symbol Format Validation

**File**: `triplegain/src/api/app.py`
**Lines**: 171-172, 217-218
**Impact**: Robustness - Invalid symbols passed to queries
**Effort**: 1 hour

**Add Helper Function** (at top of file):
```python
import re

SYMBOL_PATTERN = re.compile(r'^[A-Z]{2,5}/[A-Z]{2,5}$')

def validate_symbol(symbol: str) -> str:
    """Validate and return normalized symbol."""
    if not SYMBOL_PATTERN.match(symbol.upper()):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid symbol format: {symbol}. Expected format: XXX/YYY"
        )
    return symbol.upper()
```

**Update Endpoints**:
```python
@app.get("/api/v1/indicators/{symbol}/{timeframe}")
async def get_indicators(symbol: str, timeframe: str):
    symbol = validate_symbol(symbol)
    # ... rest of function

@app.get("/api/v1/snapshot/{symbol}")
async def get_snapshot(symbol: str):
    symbol = validate_symbol(symbol)
    # ... rest of function
```

---

## P3 - Low Priority (Future Improvements)

### P3-1: Token Estimation Accuracy

**File**: `triplegain/src/llm/prompt_builder.py`
**Lines**: 69, 160-168
**Impact**: Accuracy - May exceed token budget by ~10-15%
**Effort**: 2 hours

**Current**: Uses `CHARS_PER_TOKEN = 3.5` heuristic

**Improvement Options**:
1. Use `tiktoken` library for accurate OpenAI/Claude estimation
2. Add per-tier calibration values
3. Add safety margin (e.g., 0.9 * max_budget)

**Recommended Approach** (lowest effort):
```python
# Add 10% safety margin
if estimated_tokens > max_budget * 0.9:
    user_message = self.truncate_to_budget(
        user_message,
        int(max_budget * 0.9)
    )
```

---

### P3-2: Conflict Resolution Timeout Not Enforced

**File**: `triplegain/src/orchestration/coordinator.py`
**Impact**: Reliability - LLM call could exceed timeout
**Effort**: 30 minutes

**Current Code**:
```python
response = await self._call_llm_for_resolution(prompt)
```

**Fixed Code**:
```python
try:
    response = await asyncio.wait_for(
        self._call_llm_for_resolution(prompt),
        timeout=self._max_resolution_time_ms / 1000.0
    )
except asyncio.TimeoutError:
    logger.warning(f"Conflict resolution timed out after {self._max_resolution_time_ms}ms")
    return self._get_fallback_resolution(conflict)
```

---

### P3-3: Degradation Recovery Event Not Published

**File**: `triplegain/src/orchestration/coordinator.py`
**Impact**: Observability - No notification when system recovers
**Effort**: 30 minutes

**Add to degradation recovery logic**:
```python
if previous_level != DegradationLevel.NORMAL and new_level == DegradationLevel.NORMAL:
    await self._message_bus.publish(
        topic=MessageTopic.SYSTEM_EVENTS,
        message={
            "event": "degradation_recovered",
            "previous_level": previous_level.name,
            "timestamp": datetime.utcnow().isoformat()
        },
        priority=MessagePriority.HIGH
    )
    logger.info(f"System recovered from {previous_level.name} to NORMAL")
```

---

## Test Coverage Gaps (Non-Blocking)

### Execution Module (Currently 61%)

**Files needing additional tests**:
- `order_manager.py`: Need 40-50 edge case tests
- `position_tracker.py`: Need 60-80 edge case tests

**Test scenarios to add**:
1. Kraken API error handling (network, rate limit, invalid response)
2. Partial fill scenarios
3. Order cancellation race conditions
4. Leverage >1 P&L calculations
5. Concurrent position updates
6. Trailing stop lifecycle

### Orchestration Module (Coordinator 57%)

**Test scenarios to add**:
1. Task scheduling failures
2. LLM unavailability fallback
3. State persistence recovery
4. Graceful degradation transitions
5. Circuit breaker state transitions

---

## Implementation Priority Order

**Week 1** (Paper Trading Start):
1. P1-1: API Exception Details (30 min) - **MUST DO**
2. P2-5: Symbol Validation (1 hr) - **SHOULD DO**
3. P2-3: Truncation Logging (30 min) - **SHOULD DO**

**Week 2** (During Paper Trading):
4. P2-1: Supertrend Initial State (30 min)
5. P2-2: Async Error Handling (1 hr)
6. P2-4: Type Coercion (30 min)

**Week 3+** (As Needed):
7. P3-1: Token Estimation (2 hr)
8. P3-2: Timeout Enforcement (30 min)
9. P3-3: Recovery Events (30 min)

---

## Verification Checklist

After fixing each issue, verify:

- [ ] P1-1: Test API with intentional error, verify generic response
- [ ] P2-1: Check first Supertrend value matches expected direction
- [ ] P2-2: Test with mock database failures, verify proper error
- [ ] P2-3: Check logs when prompt exceeds budget
- [ ] P2-4: Test config with string values where int expected
- [ ] P2-5: Test API with invalid symbols (lowercase, no slash, etc.)
- [ ] P3-1: Verify token estimates are within 10% of actual
- [ ] P3-2: Test conflict resolution with very slow LLM
- [ ] P3-3: Check SYSTEM_EVENTS when recovering from degradation

---

**Document Version**: 1.0
**Last Updated**: 2025-12-19
