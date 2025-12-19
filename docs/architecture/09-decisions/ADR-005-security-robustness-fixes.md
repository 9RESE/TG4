# ADR-005: Security and Robustness Fixes

**Status**: Accepted
**Date**: 2025-12-19
**Decision Makers**: Development Team

## Context

After comprehensive code reviews of the Phase 3 implementation, several security and robustness issues were identified that needed addressing before production deployment:

1. **API Exception Details Exposure**: Error messages returned exception details to clients
2. **Missing Input Validation**: Symbol format validation was inconsistent
3. **Timeout Enforcement**: Conflict resolution had no timeout limits
4. **System State Visibility**: Degradation recovery events weren't published
5. **Token Estimation Accuracy**: No safety margin for BPE encoding variations
6. **Portfolio Edge Cases**: Several edge cases in DCA and allocation handling

## Decision

### 1. Generic API Error Messages

**Problem**: Stack traces and internal details exposed in error responses.

**Solution**: Replace all `raise HTTPException(status_code=500, detail=str(e))` with generic messages like `"Internal server error during TA analysis"`. Full details logged server-side with `exc_info=True`.

**Affected Files**:
- `triplegain/src/api/routes_orchestration.py` (15+ endpoints)
- `triplegain/src/api/routes_agents.py` (6 endpoints)

### 2. Centralized Symbol Validation

**Problem**: Symbol format validation was ad-hoc and inconsistent.

**Solution**: Created `triplegain/src/api/validation.py` with:
- `validate_symbol()` - Returns (is_valid, error_message)
- `validate_symbol_or_raise()` - Raises HTTPException(400) if invalid
- `normalize_symbol()` - Converts BTC_USDT → BTC/USDT

**Design Choices**:
- Accepts both slash and underscore separators for URL compatibility
- Normalizes all symbols to standard slash format internally
- Strict mode enforces supported symbols only (BTC/USDT, XRP/USDT, etc.)

### 3. Conflict Resolution Timeout

**Problem**: LLM conflict resolution had no timeout enforcement.

**Solution**: Wrap LLM call with `asyncio.wait_for()`:
```python
response = await asyncio.wait_for(
    self._call_llm_for_resolution(prompt),
    timeout=timeout_seconds,
)
```

Returns conservative "wait" action on timeout.

### 4. Degradation Recovery Events

**Problem**: System published events when degrading but not when recovering.

**Solution**: Added `_publish_degradation_event()` method that fires on all state changes:
```python
event_type = "degradation_recovery" if new_level == NORMAL else "degradation_increased"
```

### 5. Token Estimation Safety Margin

**Problem**: Token estimates could undercount, causing truncation.

**Solution**: Added 10% safety margin:
```python
TOKEN_SAFETY_MARGIN = 1.10
return int(base_estimate * self.TOKEN_SAFETY_MARGIN)
```

Updated `truncate_to_budget()` to account for margin when calculating max chars.

### 6. Portfolio Edge Cases

**DCA Batch Rounding**: Ensures batches sum to original total with first batch getting remainder.

**DCA Sub-Minimum Trades**: Auto-reduces batch count when individual batches would be below minimum.

**Target Allocation Validation**: Logs warning when allocations don't sum to 100%.

**Hodl Bag Warning**: Logs warning when hodl bags exceed available balance (clamps to 0).

**Zero Equity Handling**: Returns target allocation percentages to prevent false rebalancing trigger.

**LLM Fallback Transparency**: Added `used_fallback_strategy` field to RebalanceOutput.

## Consequences

### Positive

1. **Security**: No internal details exposed to clients
2. **Consistency**: Centralized validation ensures uniform behavior
3. **Reliability**: Timeout enforcement prevents hanging operations
4. **Observability**: Recovery events enable monitoring
5. **Accuracy**: Safety margin reduces token limit errors
6. **Correctness**: Edge cases handled gracefully

### Negative

1. **Debugging**: Generic error messages require log inspection
2. **Complexity**: Additional validation layer in request processing
3. **Performance**: Small overhead from safety margin calculations

### Risks Mitigated

- Information disclosure (OWASP A01)
- Denial of service via hanging LLM calls
- Silent failures in degradation recovery
- Token limit exceeded errors
- Portfolio calculation errors at edge cases

## Related Documents

- [Phase 3 Features v1.4](../../development/features/phase-3-orchestration.md)
- [Comprehensive Code Review](../../development/reviews/full/comprehensive-implementation-review.md)
- [Issues Action Items](../../development/reviews/full/issues-action-items.md)

## Implementation

- **PR**: Implemented in main branch
- **Tests**: 916 tests passing
- **Coverage**: 87%

---

*ADR-005 - December 2025*
