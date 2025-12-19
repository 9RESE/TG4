# Data Layer Fix Action Plan

**Created**: 2025-12-19
**Review**: data-layer-review-2025-12-19.md
**Estimated Total Time**: 9 hours (high priority) + 13 hours (medium priority)

---

## High Priority Fixes (Before Production)

### Issue #9: Database Retry Logic
**Time**: 2 hours | **Priority**: HIGH | **Impact**: Critical

**Problem**: Transient database errors cause complete snapshot failure without retry

**Solution**:
```python
# Add to market_snapshot.py
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import asyncpg

class RetryableDBError(Exception):
    """Errors that should trigger retry"""
    pass

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((asyncpg.PostgresConnectionError, asyncpg.QueryTimeoutError))
)
async def fetch_with_retry(fetch_func, *args, **kwargs):
    """Wrap database fetch with retry logic"""
    try:
        return await fetch_func(*args, **kwargs)
    except Exception as e:
        logger.warning(f"Database fetch failed: {e}, retrying...")
        raise

# Update build_snapshot to use retry wrapper
candle_tasks = [
    fetch_with_retry(self.db.fetch_candles, symbol, tf, lookback_config.get(tf, 50))
    for tf in timeframes
]
```

**Testing**:
- Add test with mock database failures
- Verify 3 retries with exponential backoff
- Confirm success after transient failure
- Verify permanent failure after 3 attempts

**Files Modified**:
- `triplegain/src/data/market_snapshot.py`
- `triplegain/tests/unit/test_market_snapshot.py`

---

### Issue #12: Connection Pool Monitoring
**Time**: 3 hours | **Priority**: HIGH | **Impact**: Critical

**Problem**: No detection of connection pool leaks or saturation

**Solution**:
```python
# Add to database.py

async def get_pool_stats(self) -> dict:
    """Get current pool statistics"""
    if not self._pool:
        return {
            'status': 'disconnected',
            'connected': False,
        }

    size = self._pool.get_size()
    idle = self._pool.get_idle_size()

    return {
        'status': 'healthy',
        'connected': True,
        'total_connections': size,
        'idle_connections': idle,
        'active_connections': size - idle,
        'max_connections': self.config.max_connections,
        'min_connections': self.config.min_connections,
        'saturation_pct': ((size - idle) / size * 100) if size > 0 else 0,
        'is_saturated': (size - idle) >= size * 0.9,  # 90% threshold
    }

async def check_pool_health(self) -> dict:
    """Enhanced health check with pool stats"""
    base_health = await self.check_health()
    if base_health['connected']:
        pool_stats = await self.get_pool_stats()
        base_health.update(pool_stats)

        # Warn if saturated
        if pool_stats.get('is_saturated'):
            logger.warning(
                f"Connection pool saturated: {pool_stats['active_connections']}/{pool_stats['total_connections']} "
                f"connections active"
            )

    return base_health

# Add periodic monitoring task
async def monitor_pool_health(self, interval_seconds: int = 60):
    """Background task to monitor pool health"""
    while self.is_connected:
        try:
            stats = await self.get_pool_stats()
            if stats.get('saturation_pct', 0) > 80:
                logger.warning(f"Pool saturation high: {stats['saturation_pct']:.1f}%")
            await asyncio.sleep(interval_seconds)
        except Exception as e:
            logger.error(f"Pool monitoring error: {e}")
            await asyncio.sleep(interval_seconds)
```

**Testing**:
- Test pool stats with various connection states
- Simulate saturation and verify warning
- Test monitoring task

**Files Modified**:
- `triplegain/src/data/database.py`
- `triplegain/tests/unit/test_database.py`

---

### Issue #7: Comprehensive Data Validation
**Time**: 4 hours | **Priority**: HIGH | **Impact**: High

**Problem**: Missing critical data quality checks (OHLC sanity, flash crash detection)

**Solution**:
```python
# Add to market_snapshot.py

def _validate_data_quality(self, snapshot: MarketSnapshot) -> list[str]:
    """
    Enhanced data quality validation.

    Checks:
    - Stale data
    - Missing indicators
    - Insufficient candles
    - Price sanity (OHLC relationships)
    - Unrealistic price movements
    - Zero/negative volumes
    - Future timestamps
    """
    issues = []

    # Existing checks
    max_age = self.config.get('data_quality', {}).get('max_age_seconds', 60)
    if snapshot.data_age_seconds > max_age:
        issues.append('stale_data')

    if not snapshot.indicators:
        issues.append('no_indicators')

    min_candles = self.config.get('data_quality', {}).get('min_candles_required', 20)
    for tf, candles in snapshot.candles.items():
        if len(candles) < min_candles:
            issues.append(f'insufficient_{tf}_candles')

    if snapshot.order_book is None:
        issues.append('no_order_book')

    # NEW: Price sanity checks
    if snapshot.current_price <= 0:
        issues.append('invalid_price_zero_or_negative')

    # NEW: Check for unrealistic price movement (>50% in 24h could be data error)
    if snapshot.price_change_24h_pct:
        abs_change = abs(float(snapshot.price_change_24h_pct))
        if abs_change > 50:
            issues.append('unrealistic_price_change_24h')
            logger.warning(
                f"Unrealistic 24h price change for {snapshot.symbol}: "
                f"{snapshot.price_change_24h_pct:.2f}%"
            )

    # NEW: Validate OHLC relationships in recent candles
    for tf, candles in snapshot.candles.items():
        if candles:
            recent = candles[-1]
            if recent.high < recent.low:
                issues.append(f'invalid_ohlc_{tf}_high_less_than_low')
            if recent.high < recent.close or recent.high < recent.open:
                issues.append(f'invalid_ohlc_{tf}_high_less_than_close_or_open')
            if recent.low > recent.close or recent.low > recent.open:
                issues.append(f'invalid_ohlc_{tf}_low_greater_than_close_or_open')
            if recent.volume <= 0:
                issues.append(f'invalid_volume_{tf}_zero_or_negative')

    # NEW: Check for future timestamps (clock sync issues)
    from datetime import datetime, timezone, timedelta
    now = datetime.now(timezone.utc)
    future_threshold = timedelta(minutes=5)  # Allow 5 min clock skew

    if snapshot.timestamp > now + future_threshold:
        issues.append('future_timestamp_detected')
        logger.error(
            f"Future timestamp detected for {snapshot.symbol}: "
            f"{snapshot.timestamp} vs now {now}"
        )

    return issues
```

**Testing**:
- Test invalid OHLC (high < low)
- Test unrealistic price changes
- Test zero/negative volumes
- Test future timestamps
- Test normal data passes validation

**Files Modified**:
- `triplegain/src/data/market_snapshot.py`
- `triplegain/tests/unit/test_market_snapshot.py`

---

### Issue #13: Connection Timeout Recovery
**Time**: Included in Issue #9 (2 hours)

**Solution**: Handled by retry logic in Issue #9

---

## Medium Priority Fixes (During Phase 3)

### Issue #2: Integrate Indicator Caching
**Time**: 4 hours | **Priority**: MEDIUM | **Benefit**: ~30% latency reduction

**Problem**: Database cache methods exist but not used by IndicatorLibrary

**Solution**:
```python
# Update indicator_library.py

async def calculate_all_with_cache(
    self,
    symbol: str,
    timeframe: str,
    candles: list[dict]
) -> dict[str, any]:
    """
    Calculate all indicators with database caching.

    Checks cache for recent calculations first.
    Falls back to calculate_all() if cache miss.
    """
    if not self._cache_enabled or not candles:
        return self.calculate_all(symbol, timeframe, candles)

    # Get latest candle timestamp
    latest_ts = candles[-1].get('timestamp')
    if not latest_ts:
        return self.calculate_all(symbol, timeframe, candles)

    # Check cache for each indicator
    cached_results = {}
    uncached_indicators = []

    indicator_names = [
        'rsi_14', 'macd', 'atr_14', 'adx_14',
        'ema_9', 'ema_21', 'ema_50', 'ema_200',
        'bollinger_bands', 'obv', 'vwap', 'supertrend',
    ]

    for ind_name in indicator_names:
        cached_value = await self.db.get_cached_indicator(
            symbol, timeframe, ind_name, latest_ts, max_age_seconds=300
        )
        if cached_value is not None:
            cached_results[ind_name] = cached_value
        else:
            uncached_indicators.append(ind_name)

    # If all cached, return cached results
    if not uncached_indicators:
        logger.debug(f"Cache hit for all indicators: {symbol}/{timeframe}")
        return cached_results

    # Otherwise, calculate fresh
    logger.debug(f"Cache miss for {len(uncached_indicators)} indicators: {symbol}/{timeframe}")
    results = self.calculate_all(symbol, timeframe, candles)

    # Store in cache asynchronously (don't wait)
    asyncio.create_task(self._cache_results(symbol, timeframe, latest_ts, results))

    return results

async def _cache_results(
    self,
    symbol: str,
    timeframe: str,
    timestamp: datetime,
    results: dict
):
    """Store indicator results in cache"""
    try:
        for ind_name, value in results.items():
            if value is not None and not isinstance(value, dict):
                await self.db.cache_indicator(
                    symbol, timeframe, ind_name, timestamp, float(value)
                )
    except Exception as e:
        logger.warning(f"Failed to cache indicators: {e}")
```

**Testing**:
- Test cache hit scenario
- Test cache miss scenario
- Test partial cache hit
- Measure latency improvement (expect ~30%)

**Files Modified**:
- `triplegain/src/data/indicator_library.py`
- `triplegain/tests/unit/test_indicator_library.py`

---

### Issue #4: Input Validation
**Time**: 2 hours | **Priority**: MEDIUM | **Impact**: Medium

**Problem**: Missing validation for candle structure

**Solution**:
```python
# Add to indicator_library.py

def calculate_all(
    self,
    symbol: str,
    timeframe: str,
    candles: list[dict]
) -> dict[str, any]:
    """Calculate all configured indicators with input validation."""
    start_time = time.perf_counter()

    if not candles:
        logger.debug(f"No candles provided for {symbol}/{timeframe}")
        return {}

    # NEW: Validate candle structure
    required_fields = ['open', 'high', 'low', 'close', 'volume']
    first_candle = candles[0]

    missing_fields = [f for f in required_fields if f not in first_candle]
    if missing_fields:
        raise ValueError(
            f"Candles missing required fields {missing_fields}. "
            f"Required: {required_fields}, Got: {list(first_candle.keys())}"
        )

    # Validate data types (at least for first candle)
    for field in required_fields:
        try:
            float(first_candle[field])
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Invalid data type for field '{field}': {first_candle[field]}. "
                f"Expected numeric value."
            )

    # Rest of existing implementation...
```

**Testing**:
- Test with missing fields
- Test with invalid data types
- Test with valid data

**Files Modified**:
- `triplegain/src/data/indicator_library.py`
- `triplegain/tests/unit/test_indicator_library.py`

---

### Issue #11: Concurrency Limits
**Time**: 2 hours | **Priority**: MEDIUM | **Impact**: Medium

**Problem**: Multi-symbol build can exhaust DB connections with 100+ symbols

**Solution**:
```python
# Update market_snapshot.py

async def build_multi_symbol_snapshot(
    self,
    symbols: list[str],
    max_concurrent: int = 10
) -> dict[str, MarketSnapshot]:
    """
    Build snapshots for multiple symbols with concurrency control.

    Args:
        symbols: List of trading pairs
        max_concurrent: Maximum concurrent snapshot builds (default: 10)

    Returns:
        Dict of symbol -> MarketSnapshot
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def limited_build(symbol: str) -> tuple[str, MarketSnapshot]:
        """Build snapshot with semaphore limit"""
        async with semaphore:
            try:
                snapshot = await self.build_snapshot(symbol)
                return (symbol, snapshot)
            except Exception as e:
                logger.error(f"Failed to build snapshot for {symbol}: {e}")
                return (symbol, e)

    # Build all snapshots with concurrency limit
    tasks = [limited_build(symbol) for symbol in symbols]
    results = await asyncio.gather(*tasks)

    # Separate successes from failures
    snapshots = {}
    for symbol, result in results:
        if isinstance(result, Exception):
            logger.error(f"Skipping {symbol} due to error: {result}")
            continue
        snapshots[symbol] = result

    logger.info(
        f"Built {len(snapshots)}/{len(symbols)} snapshots "
        f"(max_concurrent={max_concurrent})"
    )

    return snapshots
```

**Testing**:
- Test with 50 symbols
- Verify only 10 concurrent connections
- Test failure handling

**Files Modified**:
- `triplegain/src/data/market_snapshot.py`
- `triplegain/tests/unit/test_market_snapshot.py`

---

### Issue #19: Symbol Validation
**Time**: 2 hours | **Priority**: MEDIUM | **Impact**: Security

**Problem**: Symbol names not validated, potential SQL injection in table names

**Solution**:
```python
# Add to database.py

import re

def validate_symbol(symbol: str) -> str:
    """
    Validate and normalize symbol format.

    Args:
        symbol: Trading pair (e.g., "BTC/USDT" or "BTCUSDT")

    Returns:
        Normalized symbol without slash

    Raises:
        ValueError: If symbol format is invalid
    """
    # Remove slash for normalization
    normalized = symbol.replace('/', '')

    # Validate format: 2-10 uppercase letters, optional slash, 2-10 uppercase letters
    # Examples: BTCUSDT, BTC/USDT, XRPUSDT
    if not re.match(r'^[A-Z]{2,10}[A-Z]{2,10}$', normalized):
        raise ValueError(
            f"Invalid symbol format: '{symbol}'. "
            f"Expected format: XXX/YYY or XXXYYY (2-10 uppercase letters each)"
        )

    return normalized

# Update all methods to use validation
async def fetch_candles(
    self,
    symbol: str,
    timeframe: str,
    limit: int = 100,
    end_time: Optional[datetime] = None
) -> list[dict]:
    """Fetch candles with symbol validation"""
    normalized_symbol = validate_symbol(symbol)  # NEW
    # Rest of existing implementation...
```

**Testing**:
- Test valid symbols (BTC/USDT, BTCUSDT)
- Test invalid symbols (btc/usdt, BTC, 1234, SQL injection attempts)
- Verify all database methods use validation

**Files Modified**:
- `triplegain/src/data/database.py`
- `triplegain/tests/unit/test_database.py`

---

### Other Medium Priority Issues

**Issue #6: Configuration Validation** (2 hours)
**Issue #10: Failure Threshold Logic** (2 hours)
**Issue #15: Silent Failures Logging** (1 hour)
**Issue #16: JSON Serialization Errors** (1 hour)
**Issue #20: Memory Limits** (1 hour)

*Details available in main review document*

---

## Implementation Order

### Phase 1: High Priority (Week 1)
1. Issue #9: Database retry logic (2 hours)
2. Issue #12: Pool monitoring (3 hours)
3. Issue #7: Data validation (4 hours)
4. Testing & validation (2 hours)

**Total**: 11 hours (1.5 days)

### Phase 2: Medium Priority (Week 2-3)
5. Issue #2: Indicator caching (4 hours)
6. Issue #4: Input validation (2 hours)
7. Issue #11: Concurrency limits (2 hours)
8. Issue #19: Symbol validation (2 hours)
9. Testing & validation (3 hours)

**Total**: 13 hours (1.5 days)

### Phase 3: Low Priority (Ongoing)
- Issue #3, #5, #6, #8, #14, #15, #16, #17, #18, #20
- As time permits during Phase 3 operation

---

## Testing Strategy

### After Each Fix
1. Run unit tests: `pytest triplegain/tests/unit/test_*.py -v`
2. Run integration tests: `pytest triplegain/tests/integration/ -v`
3. Run coverage: `pytest --cov=triplegain/src/data --cov-report=html`
4. Performance benchmarks: `pytest triplegain/tests/unit/test_indicator_library.py::TestPerformance -v`

### Before Production Deploy
1. Full test suite: `pytest triplegain/tests/ -v`
2. Load testing with 100+ concurrent snapshots
3. Memory profiling for leak detection
4. Database failure simulation

---

## Success Criteria

### Phase 1 Complete When:
- ✅ All high priority issues fixed
- ✅ Retry logic handles 3 transient failures
- ✅ Pool monitoring detects saturation
- ✅ Data validation catches bad OHLC
- ✅ All tests passing (100%)
- ✅ Coverage maintained at 85%+

### Phase 2 Complete When:
- ✅ Indicator caching reduces latency by 20%+
- ✅ Input validation prevents KeyErrors
- ✅ Concurrency limits prevent pool exhaustion
- ✅ Symbol validation prevents injection
- ✅ All tests passing (100%)

---

## Risk Mitigation

### During Implementation
- Feature flags for new code paths
- Gradual rollout (caching optional initially)
- Monitoring at each step
- Rollback plan for each change

### Testing
- Test in isolation first
- Integration testing before merge
- Performance regression testing
- Load testing before production

---

**Action Plan Ready**: 2025-12-19
**Estimated Completion**: 24 hours of development + testing
**Next Review**: After Phase 1 complete (11 hours)
