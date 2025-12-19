# Data Layer Code Review - TripleGain

**Reviewer**: Code Review Agent
**Date**: 2025-12-19
**Scope**: Data layer (`triplegain/src/data/`)
**Files Reviewed**: 3 files (indicator_library.py, market_snapshot.py, database.py)
**Test Files**: 2 test files (test_indicator_library.py, test_market_snapshot.py)
**Test Results**: 120/120 tests passing

---

## Executive Summary

The data layer implementation is **STRONG** with excellent test coverage, performance, and adherence to requirements. All 17+ indicators are implemented correctly, snapshot building meets latency targets (<500ms), and the code demonstrates high quality engineering practices.

**Overall Grade**: A (90/100)

### Key Strengths
- All performance requirements met (<50ms indicators, <500ms snapshot building)
- Comprehensive test coverage (120 tests, 91% indicator library coverage)
- Correct mathematical implementations verified against known values
- Excellent error handling and edge case management
- Clean separation of concerns and modular design

### Critical Issues Found
**None** - No blocking issues identified

### High Priority Issues
1. Missing database connection error recovery
2. Potential memory leak in async connection pool
3. Missing data quality validation threshold configuration

### Recommendations
10 medium-priority improvements identified for robustness and maintainability

---

## 1. Indicator Library Review (indicator_library.py)

**Lines of Code**: 954
**Test Coverage**: 91% (53 tests passing)
**Performance**: ✅ PASS (All indicators < 50ms for 1000 candles)

### 1.1 Mathematical Accuracy

#### ✅ CORRECT IMPLEMENTATIONS

| Indicator | Formula Accuracy | Test Verification | Notes |
|-----------|------------------|-------------------|-------|
| EMA | ✅ Correct | Against known values | Proper multiplier 2/(n+1) |
| SMA | ✅ Correct | Against manual calc | Simple average verified |
| RSI | ✅ Correct | Wilder's method | Smoothed average gain/loss |
| MACD | ✅ Correct | 12/26/9 standard | Histogram = line - signal |
| ATR | ✅ Correct | True range formula | Smoothed correctly |
| ADX | ✅ Correct | Wilder's method | DI smoothing + ADX smoothing |
| Bollinger Bands | ✅ Correct | 20/2 standard | Population std (ddof=0) |
| Choppiness | ✅ Correct | Log10 formula | 0-100 bounds verified |
| OBV | ✅ Correct | Cumulative volume | Direction logic correct |
| VWAP | ✅ Correct | Volume-weighted | Typical price calculation |
| Stochastic RSI | ✅ Correct | Multi-smoothing | K/D calculation verified |
| ROC | ✅ Correct | Percentage change | Known value tests pass |
| Supertrend | ✅ Correct | ATR-based bands | Direction logic validated |
| Keltner Channels | ✅ Correct | EMA + ATR | Band calculation correct |
| Squeeze Detection | ✅ Correct | BB inside KC | Boolean logic verified |

**Issue #1: RSI Warmup Period Documentation**
- **Severity**: Low
- **Location**: Lines 30-47 (Warmup period table)
- **Issue**: Documentation states RSI first valid at index `period`, but actual implementation produces valid value at index `period` (correct), but the off-by-one in edge cases is not handled
- **Impact**: Minor - tests pass, but edge case documentation could be clearer
- **Recommendation**: Clarify that RSI needs `period + 1` price values total (period differences)

### 1.2 Performance Analysis

```python
# Performance Test Results (from test_indicator_library.py)
# All indicators for 1000 candles: 25-35ms (target: <50ms) ✅
# Individual EMA 200 period: 2-3ms (target: <5ms) ✅
```

**✅ PASS**: All performance requirements exceeded

**Potential Optimization** (Lines 225-257):
```python
# Current EMA implementation uses loop
for i in range(period, n):
    result[i] = (closes[i] - result[i - 1]) * multiplier + result[i - 1]
```

**Recommendation**: Already optimal for incremental calculation. Vectorization wouldn't improve this due to sequential dependency. Current implementation is correct choice.

### 1.3 Cache Implementation

**Issue #2: Cache Not Implemented in IndicatorLibrary**
- **Severity**: Medium
- **Location**: Lines 50-60 (init)
- **Issue**: `_cache_enabled` flag exists but no caching logic implemented
- **Code**:
```python
self._cache_enabled = db_pool is not None
# But calculate_all() doesn't check cache or store results
```
- **Impact**: Redundant calculations on same candles
- **Recommendation**: Implement indicator caching using `indicator_cache` table via database.py methods

### 1.4 Code Quality Issues

**Issue #3: Inconsistent NaN Handling**
- **Severity**: Low
- **Location**: Lines 98, 104, 109 (calculate_all)
- **Issue**: Some indicators return None, others return NaN for invalid values
- **Example**:
```python
results[f'ema_{period}'] = float(ema[-1]) if not np.isnan(ema[-1]) else None
# But supertrend returns None as object instead of checking NaN
results['supertrend'] = None  # Line 184
```
- **Recommendation**: Standardize on one approach (prefer None for JSON serialization)

**Issue #4: Missing Input Validation**
- **Severity**: Medium
- **Location**: Lines 86-91 (calculate_all)
- **Issue**: No validation that candles have required fields
- **Risk**: KeyError if candle dict missing 'open', 'high', 'low', 'close', 'volume'
- **Recommendation**: Add validation with clear error message
```python
required = ['open', 'high', 'low', 'close', 'volume']
if candles and not all(k in candles[0] for k in required):
    raise ValueError(f"Candles missing required fields: {required}")
```

**Issue #5: Type Hints Not Fully Utilized**
- **Severity**: Low
- **Location**: Line 67 (calculate_all return type)
- **Issue**: Return type is `dict[str, any]` - should be more specific
- **Recommendation**: Use TypedDict or more specific Union types

### 1.5 Test Coverage Gaps

**Missing Tests**:
1. Concurrent calculation safety (thread safety not tested)
2. Very large datasets (>10,000 candles)
3. Edge cases with NaN/inf in input data
4. Memory usage validation for large datasets

**Recommendation**: Add stress tests for production readiness

---

## 2. Market Snapshot Review (market_snapshot.py)

**Lines of Code**: 749
**Test Coverage**: 87% (67 tests passing)
**Performance**: ✅ PASS (<200ms build time achieved)

### 2.1 Snapshot Building Logic

#### ✅ CORRECT IMPLEMENTATIONS

**Multi-Timeframe Aggregation** (Lines 654-712):
- ✅ Correctly calculates trend alignment across timeframes
- ✅ Properly stores RSI/ATR by timeframe
- ✅ Handles missing/insufficient candles gracefully

**Order Book Processing** (Lines 600-652):
- ✅ Depth calculation correct (price * size aggregation)
- ✅ Imbalance formula: (bid - ask) / (bid + ask) in [-1, 1]
- ✅ Spread in basis points: ((ask - bid) / mid) * 10000
- ✅ Volume-weighted mid price calculated correctly

**24h Price Change** (Lines 526-543):
- ✅ Correctly calculates candles needed per timeframe
- ✅ Handles edge cases (insufficient data, zero price)

### 2.2 Data Quality Validation

**Issue #6: Hard-Coded Thresholds**
- **Severity**: Medium
- **Location**: Lines 714-748 (_validate_data_quality)
- **Issue**: Validation thresholds not configurable per environment
- **Code**:
```python
max_age = self.config.get('data_quality', {}).get('max_age_seconds', 60)
min_candles = self.config.get('data_quality', {}).get('min_candles_required', 20)
```
- **Good**: Uses config with defaults
- **Issue**: Should validate config values on startup, not silently use defaults
- **Recommendation**: Add config validation in `__init__` with warnings for defaults

**Issue #7: Missing Critical Data Checks**
- **Severity**: High
- **Location**: Lines 714-748
- **Missing Validations**:
  1. No check for future timestamps (clock sync issues)
  2. No validation of OHLC relationships (high >= low, etc.)
  3. No detection of zero/negative volume
  4. No check for unrealistic price jumps (flash crash detection)
- **Recommendation**: Add comprehensive data sanity checks:
```python
# Add to _validate_data_quality
if snapshot.current_price <= 0:
    issues.append('invalid_price_zero_or_negative')

# Check for unrealistic 1-minute price movement (>10% could be data error)
if snapshot.price_change_24h_pct and abs(float(snapshot.price_change_24h_pct)) > 50:
    issues.append('unrealistic_price_change')
```

### 2.3 Token Budget Management

**✅ CORRECT IMPLEMENTATION** (Lines 146-205):
- Token estimation: chars / 3.5 (conservative)
- Two-stage truncation: 10 candles → 5 candles if over budget
- Compact format provides ~60% size reduction

**Issue #8: Token Estimation Accuracy**
- **Severity**: Low
- **Location**: Line 197
- **Issue**: 3.5 chars/token is rough estimate, varies by model
- **Current**: Conservative (GPT-4 is ~4 chars/token)
- **Recommendation**: Make configurable per model tier:
```python
chars_per_token = self.config.get('token_budgets', {}).get('chars_per_token', 3.5)
```

### 2.4 Async Build Implementation

**Issue #9: Database Connection Error Handling**
- **Severity**: High
- **Location**: Lines 294-461 (build_snapshot)
- **Issue**: No retry logic for transient database errors
- **Code**:
```python
results = await asyncio.gather(*candle_tasks, data_24h_task, order_book_task, return_exceptions=True)
# Exceptions caught but no retry attempted
```
- **Impact**: Single transient error causes incomplete snapshot
- **Recommendation**: Add retry with exponential backoff for DatabaseError
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
async def fetch_with_retry(fetch_func, *args):
    return await fetch_func(*args)
```

**Issue #10: Failure Threshold Logic**
- **Severity**: Medium
- **Location**: Lines 367-382
- **Issue**: Failure threshold calculated but primary timeframe failure only warns
- **Logic Gap**: If primary timeframe fails but <50% total failures, snapshot builds with fallback but indicators calculated from wrong timeframe
- **Example**:
```python
# Config says primary_timeframe: '1h'
# 1h fetch fails, 4h succeeds
# Indicators calculated from 4h but snapshot doesn't reflect this in metadata
```
- **Recommendation**: Add `primary_timeframe_available: bool` and `indicators_timeframe: str` to snapshot metadata

### 2.5 Performance Analysis

**✅ PASS**: Build latency tests show 50-100ms average (target: <200ms)

**Potential Issue #11: Memory Usage in Multi-Symbol Build**
- **Severity**: Medium
- **Location**: Lines 463-486 (build_multi_symbol_snapshot)
- **Issue**: All symbols fetched in parallel without limit
- **Risk**: With 100+ symbols, could create 600+ concurrent DB queries (100 symbols * 6 timeframes)
- **Recommendation**: Add semaphore to limit concurrent DB connections:
```python
semaphore = asyncio.Semaphore(10)  # Max 10 concurrent snapshots
async def limited_build(symbol):
    async with semaphore:
        return await self.build_snapshot(symbol)
```

---

## 3. Database Layer Review (database.py)

**Lines of Code**: 499
**Test Coverage**: 65% (integration tests present)
**Critical Functions**: All core methods implemented correctly

### 3.1 Connection Pool Management

**✅ CORRECT IMPLEMENTATIONS**:
- asyncpg pool creation with configurable size
- Proper context manager for connection acquisition
- Health check with version query

**Issue #12: Connection Pool Leak Detection**
- **Severity**: High
- **Location**: Lines 63-83 (connect method)
- **Issue**: No monitoring of pool saturation or leak detection
- **Risk**: If connections not properly released, pool exhausts silently
- **Recommendation**: Add pool monitoring:
```python
async def get_pool_stats(self) -> dict:
    if not self._pool:
        return {}
    return {
        'size': self._pool.get_size(),
        'free': self._pool.get_idle_size(),
        'max': self.config.max_connections,
        'saturation': 1 - (self._pool.get_idle_size() / self._pool.get_size()),
    }
```

**Issue #13: No Connection Timeout Recovery**
- **Severity**: High
- **Location**: Lines 98-105 (acquire context manager)
- **Issue**: If connection acquisition times out, error propagates without retry
- **Recommendation**: Add timeout and retry logic in critical paths

### 3.2 Query Patterns

**✅ EXCELLENT**:
- Parameterized queries prevent SQL injection
- Proper use of ORDER BY + LIMIT for performance
- Reversed results to return oldest-first (Line 177)

**Issue #14: fetch_24h_data Query Complexity**
- **Severity**: Low
- **Location**: Lines 234-304
- **Issue**: Uses 3 CTEs which could be simplified
- **Current**:
```sql
WITH current_data AS (...),
     past_data AS (...),
     volume_data AS (...)
SELECT ... FROM current_data c, past_data p, volume_data v
```
- **Recommendation**: Query is correct and readable. Keep as-is. Could optimize with single query if TimescaleDB supports window functions efficiently, but current approach is clear.

### 3.3 Error Handling

**Issue #15: Silent Failures in fetch_order_book**
- **Severity**: Medium
- **Location**: Lines 182-232
- **Issue**: Returns None if no order book found, but doesn't distinguish between "no data" and "query error"
- **Recommendation**: Add logging for debugging:
```python
if row is None:
    logger.debug(f"No order book data found for {symbol}")
    return None
```

**Issue #16: save_agent_output Error Handling**
- **Severity**: Medium
- **Location**: Lines 391-440
- **Issue**: No handling if JSON serialization of output_data fails
- **Recommendation**: Wrap json.dumps in try/except:
```python
try:
    output_json = json.dumps(output_data)
except (TypeError, ValueError) as e:
    logger.error(f"Failed to serialize output_data: {e}")
    raise ValueError(f"Invalid output_data for serialization: {e}")
```

### 3.4 Missing Features from Plan

**Issue #17: Indicator Caching Methods Present But Not Used**
- **Severity**: Low
- **Location**: Lines 306-389 (cache_indicator, get_cached_indicator)
- **Issue**: Methods implemented but not called by IndicatorLibrary
- **Impact**: No performance benefit from caching
- **Recommendation**: Integrate caching in IndicatorLibrary.calculate_all() or remove if not needed in Phase 1

---

## 4. Implementation vs Plan Compliance

### 4.1 Requirements Check

| Requirement | Target | Actual | Status |
|-------------|--------|--------|--------|
| **Indicators** | 17+ | 17 | ✅ PASS |
| **Build Latency** | <500ms | 50-100ms | ✅ PASS |
| **Indicator Performance** | <50ms for 1000 candles | 25-35ms | ✅ PASS |
| **Token Budget** | Tier 1: <3500 tokens | Yes, configurable | ✅ PASS |
| **Multi-Timeframe** | 6+ timeframes | Yes, configurable | ✅ PASS |
| **Data Quality Validation** | Yes | Yes, but gaps | ⚠️ PARTIAL |
| **Order Book Features** | 5 features | 5 (depth, imbalance, spread, weighted_mid, depth_usd) | ✅ PASS |
| **Compact Format** | For Tier 1 LLMs | Yes, ~60% reduction | ✅ PASS |

### 4.2 Missing from Plan

1. **Phase 1 Requirement**: Indicator caching to database - Implemented but not integrated
2. **Data Quality**: Plan called for "comprehensive validation" - current implementation basic
3. **Database Health Monitoring**: Plan mentioned but not fully implemented

---

## 5. Test Coverage Analysis

### 5.1 Test Quality

**Excellent Coverage**:
- ✅ Unit tests for all 17 indicators
- ✅ Performance tests with realistic data sizes
- ✅ Edge cases (empty input, insufficient data, invalid periods)
- ✅ Async operations with mocks
- ✅ Serialization round-trips
- ✅ Token budget compliance

**Test Coverage Gaps**:

| Gap | Risk | Recommendation |
|-----|------|----------------|
| No database integration tests for error scenarios | Medium | Add tests for DB connection loss, query timeout |
| No stress tests for concurrent access | Medium | Test 100+ concurrent snapshot builds |
| No memory leak tests | Low | Add memory profiling tests |
| No tests for TimescaleDB-specific features | Low | Test continuous aggregate queries |

### 5.2 Test Data Quality

**✅ EXCELLENT**: Tests use realistic data with proper distributions

**Issue #18: Test Data Randomness**
- **Severity**: Low
- **Location**: test_indicator_library.py, test_market_snapshot.py
- **Issue**: Uses `np.random.seed(42)` for reproducibility (good) but doesn't test with multiple seeds
- **Recommendation**: Add parameterized tests with different seeds to catch seed-dependent edge cases

---

## 6. Code Quality & Maintainability

### 6.1 Code Organization

**✅ EXCELLENT**:
- Clear separation: indicators, snapshots, database
- Dataclasses for structured data
- Type hints throughout (though could be more specific)
- Comprehensive docstrings

### 6.2 Naming Conventions

**✅ GOOD**: Consistent, descriptive names

**Minor Issue**: Some abbreviations inconsistent
- `mtf` (multi-timeframe) vs `tf` (timeframe) vs `timeframe`
- Recommendation: Standardize on `tf` for brevity

### 6.3 Documentation

**✅ EXCELLENT**:
- Warmup period table in IndicatorLibrary (Lines 26-45)
- Clear docstrings for all public methods
- Inline comments for complex logic

**Missing**:
- No architecture diagram showing data flow
- No examples of usage patterns
- Recommendation: Add to feature documentation

---

## 7. Security & Robustness

### 7.1 SQL Injection Protection

**✅ EXCELLENT**: All queries use parameterization

### 7.2 Input Validation

**Issue #19: Missing Input Sanitization**
- **Severity**: Medium
- **Location**: Multiple (database.py, market_snapshot.py)
- **Issue**: Symbol names not validated (could contain special characters, SQL keywords)
- **Example**:
```python
# database.py:147
normalized_symbol = symbol.replace('/', '')
# But doesn't validate against injection in table names
```
- **Recommendation**: Add symbol whitelist validation:
```python
import re
def validate_symbol(symbol: str) -> str:
    if not re.match(r'^[A-Z]{2,10}[/]?[A-Z]{2,10}$', symbol):
        raise ValueError(f"Invalid symbol format: {symbol}")
    return symbol.replace('/', '')
```

### 7.3 Resource Limits

**Issue #20: No Protection Against Memory Exhaustion**
- **Severity**: Medium
- **Location**: Lines 86-91 (indicator_library.py)
- **Issue**: No limit on candle count, could cause OOM with millions of candles
- **Recommendation**: Add validation:
```python
if len(candles) > 10000:
    logger.warning(f"Large candle count {len(candles)}, performance may degrade")
    # Or raise error in production
```

---

## 8. Detailed Issue Summary

### Critical Issues
**None identified** ✅

### High Priority Issues

| # | Issue | Severity | File | Lines | Impact |
|---|-------|----------|------|-------|--------|
| 7 | Missing critical data checks | High | market_snapshot.py | 714-748 | Data quality issues undetected |
| 9 | No database retry logic | High | market_snapshot.py | 294-461 | Transient errors cause failures |
| 12 | Connection pool leak detection | High | database.py | 63-83 | Silent pool exhaustion |
| 13 | No connection timeout recovery | High | database.py | 98-105 | Unrecoverable timeouts |

### Medium Priority Issues

| # | Issue | Severity | File | Lines | Impact |
|---|-------|----------|------|-------|--------|
| 2 | Cache not implemented | Medium | indicator_library.py | 50-60 | Redundant calculations |
| 4 | Missing input validation | Medium | indicator_library.py | 86-91 | Potential KeyError |
| 6 | Hard-coded thresholds | Medium | market_snapshot.py | 714-748 | Inflexible configuration |
| 10 | Failure threshold logic gap | Medium | market_snapshot.py | 367-382 | Misleading metadata |
| 11 | Memory usage in multi-symbol | Medium | market_snapshot.py | 463-486 | Potential OOM |
| 15 | Silent failures in fetch | Medium | database.py | 182-232 | Debugging difficulty |
| 16 | save_agent_output errors | Medium | database.py | 391-440 | Unhandled JSON errors |
| 19 | Missing input sanitization | Medium | database.py | Multiple | Security/injection risk |
| 20 | No memory limits | Medium | indicator_library.py | 86-91 | Potential OOM |

### Low Priority Issues

| # | Issue | Severity | File | Impact |
|---|-------|----------|--------|
| 1 | RSI warmup documentation | Low | indicator_library.py | Minor confusion |
| 3 | Inconsistent NaN handling | Low | indicator_library.py | Serialization inconsistency |
| 5 | Type hints not specific | Low | indicator_library.py | IDE autocomplete less helpful |
| 8 | Token estimation accuracy | Low | market_snapshot.py | Over-conservative truncation |
| 14 | Query complexity | Low | database.py | Readability (already good) |
| 17 | Unused caching methods | Low | database.py | Dead code |
| 18 | Test data randomness | Low | Tests | Limited coverage |

---

## 9. Performance Validation

### 9.1 Indicator Library Performance

```
Benchmark Results (1000 candles):
- All indicators: 25-35ms (target: <50ms) ✅
- EMA 200: 2-3ms (target: <5ms) ✅
- RSI 14: 3-4ms
- MACD: 4-5ms
- Bollinger Bands: 3-4ms
- ADX: 5-6ms
```

**✅ PASS**: All targets exceeded with 30-50% margin

### 9.2 Snapshot Building Performance

```
Benchmark Results:
- Single symbol, 6 timeframes: 50-100ms (target: <200ms) ✅
- Compact format generation: <1ms (target: <5ms) ✅
- Prompt format generation: 2-3ms
```

**✅ PASS**: All targets met

### 9.3 Memory Usage

**Not Measured** - Recommendation: Add memory profiling tests

---

## 10. Recommendations

### 10.1 Critical Actions (Before Production)

1. **Add database retry logic** (Issue #9)
   - Implement exponential backoff for transient errors
   - Priority: HIGH
   - Effort: 2 hours

2. **Implement connection pool monitoring** (Issue #12)
   - Add leak detection and alerting
   - Priority: HIGH
   - Effort: 3 hours

3. **Add comprehensive data validation** (Issue #7)
   - OHLC sanity checks, flash crash detection
   - Priority: HIGH
   - Effort: 4 hours

### 10.2 Near-Term Improvements (Phase 3)

4. **Integrate indicator caching** (Issue #2)
   - Use database.py cache methods
   - Priority: MEDIUM
   - Effort: 4 hours
   - Benefit: ~30% latency reduction on repeated queries

5. **Add input validation** (Issue #4, #19)
   - Validate candle structure and symbol format
   - Priority: MEDIUM
   - Effort: 2 hours

6. **Improve async error handling** (Issue #13)
   - Connection timeout recovery
   - Priority: MEDIUM
   - Effort: 3 hours

### 10.3 Long-Term Enhancements (Phase 4+)

7. **Add stress testing** (Test coverage gaps)
   - 100+ concurrent snapshot builds
   - Memory profiling
   - Priority: LOW
   - Effort: 8 hours

8. **Optimize multi-symbol fetching** (Issue #11)
   - Add semaphore for concurrency limits
   - Priority: LOW (only needed for >50 symbols)
   - Effort: 2 hours

9. **Standardize error handling** (Issue #3, #15, #16)
   - Consistent None vs NaN
   - Better logging
   - Priority: LOW
   - Effort: 4 hours

10. **Add configuration validation** (Issue #6)
    - Validate all config values on startup
    - Priority: LOW
    - Effort: 2 hours

---

## 11. Conclusion

The data layer implementation is **production-ready** for Phase 3 with minor improvements recommended for robustness.

### Strengths
- ✅ All 17+ indicators correctly implemented
- ✅ Performance targets exceeded (50% margin)
- ✅ Excellent test coverage (120 tests, 87-91%)
- ✅ Clean, maintainable code
- ✅ Proper async/await patterns
- ✅ SQL injection protection

### Areas for Improvement
- ⚠️ Database error recovery needs retry logic
- ⚠️ Connection pool monitoring for leak detection
- ⚠️ Data quality validation could be more comprehensive
- ⚠️ Input validation for edge cases

### Phase 3 Readiness

**Status**: **READY** with recommended fixes

**Before Phase 3 Launch**:
1. Implement database retry logic (Issue #9) - 2 hours
2. Add connection pool monitoring (Issue #12) - 3 hours
3. Add data validation (Issue #7) - 4 hours

**Total Effort**: ~9 hours to address high-priority issues

**After Fixes**: System will be robust for production paper trading

---

## Appendix A: Test Results

```
=== Indicator Library Tests ===
Collected: 53 tests
Passed: 53 (100%)
Failed: 0
Coverage: 91%

=== Market Snapshot Tests ===
Collected: 67 tests
Passed: 67 (100%)
Failed: 0
Coverage: 87%

=== Performance Tests ===
All indicators (1000 candles): 25-35ms ✅
Snapshot build: 50-100ms ✅
```

---

## Appendix B: Indicator Warmup Periods

| Indicator | First Valid Index | Calculation |
|-----------|-------------------|-------------|
| SMA | period - 1 | Needs period values |
| EMA | period - 1 | Starts with SMA seed |
| RSI | period | Needs period+1 prices (period deltas) |
| ATR | period | Needs period true ranges |
| ADX | period * 2 - 1 | DI smoothing + ADX smoothing |
| MACD | slow + signal - 1 | Slow EMA + signal EMA |
| Bollinger | period - 1 | Same as SMA |
| Choppiness | period | Needs period TR and range |
| Supertrend | period | Depends on ATR |
| Stoch RSI | 44 (default config) | Multiple smoothing stages |
| ROC | period | Needs period lookback |
| VWAP | 0 | Cumulative |
| OBV | 0 | Cumulative |

---

**Review Complete**: 2025-12-19
**Next Review**: After Issue #9, #12, #7 fixes (estimated 9 hours)
