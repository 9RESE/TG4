# ADR-007: Phase 1 Foundation Review Fixes

**Status**: Accepted
**Date**: 2025-12-19
**Updated**: 2025-12-19 (P3 fixes added)
**Decision Makers**: Development Team

## Context

A comprehensive Phase 1 Foundation code review (Review 4) identified 14 issues across the foundation layer. The review covered:
- Database layer (`database.py`)
- Indicator library (`indicator_library.py`)
- Market snapshot builder (`market_snapshot.py`)
- Prompt template system (`prompt_builder.py`)
- Configuration loader (`config.py`)

Review document: `docs/development/reviews/full/review-4/findings/phase-1-findings.md`

## Decision

### Issues Found Summary

| Priority | Count | Fixed |
|----------|-------|-------|
| P0 (Critical) | 0 | - |
| P1 (High) | 3 | 2 |
| P2 (Medium) | 6 | 6 |
| P3 (Low) | 5 | 5 |

### P1 Fixes Implemented

#### 1. Stochastic RSI Smoothing Bug (P1.2)

**Problem**: The Stochastic RSI calculation overwrote raw K values with smoothed values in-place, causing %D to use already-smoothed K values instead of raw stochastic RSI values.

**File**: `triplegain/src/data/indicator_library.py:809-858`

**Solution**: Separated raw stochastic values from smoothed K values:

```python
# Calculate raw stochastic RSI values first
raw_stoch = np.full(n, np.nan)
for i in range(rsi_period + stoch_period - 1, n):
    raw_stoch[i] = 100 * (rsi[i] - rsi_min) / (rsi_max - rsi_min)

# Smooth raw stochastic to get %K (using SMA)
k = np.full(n, np.nan)
for i in range(rsi_period + stoch_period + k_period - 2, n):
    k[i] = np.nanmean(raw_stoch[i - k_period + 1:i + 1])

# Calculate %D (SMA of %K) - now uses correct %K values
d = np.full(n, np.nan)
for i in range(rsi_period + stoch_period + k_period + d_period - 3, n):
    d[i] = np.nanmean(k[i - d_period + 1:i + 1])
```

**Impact**: Corrected Stochastic RSI signals for trading decisions.

#### 2. Order Book Data Structure Mismatch (P1.3)

**Problem**: `_process_order_book()` expected raw bid/ask lists with 'price'/'size' keys, but database returned pre-computed values with different structure.

**File**: `triplegain/src/data/market_snapshot.py:600-672`

**Solution**: Added handling for both formats:

```python
def _process_order_book(self, order_book: dict) -> OrderBookFeatures:
    bids = order_book.get('bids', [])
    asks = order_book.get('asks', [])

    # Check if we have pre-computed values from database
    if not bids and not asks and 'bid_price' in order_book:
        return OrderBookFeatures(
            bid_depth_usd=Decimal(str(order_book.get('bid_volume_total', 0) or 0)),
            # ... use database format
        )

    # Otherwise calculate from raw lists
    # ... original calculation
```

**Impact**: Order book features now correctly extracted regardless of data source.

### P2 Fixes Implemented

#### 1. Transaction Context Manager (P2.1)

**File**: `triplegain/src/data/database.py:107-122`

Added atomic transaction support:

```python
@asynccontextmanager
async def transaction(self) -> AsyncIterator[asyncpg.Connection]:
    """Execute operations in an atomic transaction."""
    async with self._pool.acquire() as connection:
        async with connection.transaction():
            yield connection
```

#### 2. Supertrend Direction Initialization (P2.2)

**File**: `triplegain/src/data/indicator_library.py:930-932`

Changed from zeros to NaN for warmup period:

```python
supertrend = np.full(n, np.nan)
direction = np.full(n, np.nan)  # Was: np.zeros(n)
```

**Rationale**: NaN clearly indicates invalid warmup values vs 0 which could be misinterpreted.

#### 3. Async Timeout Handling (P2.3)

**File**: `triplegain/src/data/market_snapshot.py:331-358`

Added configurable timeout for snapshot building:

```python
snapshot_timeout = self.config.get('snapshot_timeout', 30.0)
try:
    results = await asyncio.wait_for(
        asyncio.gather(..., return_exceptions=True),
        timeout=snapshot_timeout
    )
except asyncio.TimeoutError:
    raise RuntimeError(f"Snapshot building timed out for {symbol}")
```

#### 4. Template Validation Strictness (P2.4)

**File**: `triplegain/src/llm/prompt_builder.py:295-296`

Changed keyword requirement from 50% to 67%:

```python
# Require at least 2/3 of keywords to be present (was: 1/2)
if len(missing) > len(required_keywords) // 3:
    errors.append(f"Template missing key concepts: {missing}")
```

#### 5. Scientific Notation Float Parsing (P2.5)

**File**: `triplegain/src/utils/config.py:159-188`

Fixed float parsing to handle scientific notation:

```python
# Try float (including scientific notation like 1e-5, 2.5E10)
try:
    float_val = float(value)
    if math.isfinite(float_val):
        return float_val
except ValueError:
    pass
```

#### 6. Missing Template Fallback (P2.6)

**File**: `triplegain/src/llm/prompt_builder.py:115-118`

Added fallback instead of empty system prompt:

```python
system_prompt = self._templates.get(agent_name)
if not system_prompt:
    logger.warning(f"No template found for agent {agent_name}")
    system_prompt = f"You are the {agent_name} agent. Analyze data and provide structured output."
```

### P3 Fixes Implemented

#### 1. Database Reconnection Logic (P3.1)

**File**: `triplegain/src/data/database.py:135-237`

Added robust reconnection with exponential backoff:

```python
async def reconnect(self) -> None:
    """Reconnect with exponential backoff."""
    for attempt in range(1, self.config.max_retries + 1):
        delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
        jitter = delay * random.uniform(0, 0.25)
        await asyncio.sleep(delay + jitter)
        await self.connect()

async def execute_with_retry(self, operation, query, *args):
    """Execute with automatic retry on connection errors."""
    # Catches asyncpg.PostgresConnectionError and InterfaceError
```

#### 2. VWAP NaN Handling (P3.2)

**File**: `triplegain/src/data/indicator_library.py:629-685`

Updated VWAP calculation to handle NaN values:

```python
# Treat NaN volumes as zero contribution
volumes_clean = np.where(np.isnan(volumes), 0, volumes)

# Handle NaN in prices - use close as fallback
typical_price = np.where(
    np.isnan(highs) | np.isnan(lows),
    closes,
    (highs + lows + closes) / 3
)
```

#### 3. Timestamp Normalization (P3.3)

**File**: `triplegain/src/data/market_snapshot.py:31-77`

Added utility function for consistent timestamp handling:

```python
def normalize_timestamp(value: Any, default: Optional[datetime] = None) -> datetime:
    """Normalize datetime, ISO strings, Unix timestamps to UTC datetime."""
    if isinstance(value, datetime):
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value
    if isinstance(value, str):
        return datetime.fromisoformat(value.replace('Z', '+00:00'))
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)
```

#### 4. Thread-Safe ConfigLoader (P3.4)

**File**: `triplegain/src/utils/config.py:286-327`

Added thread safety with double-checked locking:

```python
_config_lock = threading.Lock()

def get_config_loader(config_dir=None) -> ConfigLoader:
    if _config_loader is None:
        with _config_lock:
            if _config_loader is None:  # Double-check inside lock
                _config_loader = ConfigLoader(config_dir)
    return _config_loader

def reset_config_loader() -> None:
    """Reset for test isolation."""
    with _config_lock:
        _config_loader = None
```

#### 5. Token Estimation Documentation & Configuration (P3.5)

**File**: `triplegain/src/llm/prompt_builder.py:61-111`

Made token estimation configurable with documentation:

```python
class PromptBuilder:
    """
    Token Estimation Accuracy:
        - Model: GPT ~4 chars/token, Claude ~3.5
        - Content: Code ~3, prose ~4-5, JSON ~3.5
        Configure via: token_estimation.chars_per_token, safety_margin
    """

    def __init__(self, config: dict):
        token_config = config.get('token_estimation', {})
        self.chars_per_token = token_config.get('chars_per_token', 3.5)
        self.safety_margin = token_config.get('safety_margin', 1.10)
```

### Issues Not Fixed (P1.1 Only)

#### P1.1: Float Conversion for Financial Values

**Decision**: Not fixed. The indicator library uses NumPy arrays which require float for vectorized operations. Converting to Decimal would require significant refactoring and introduce performance penalties.

**Mitigation**: The precision loss is acceptable for indicator calculations which are used for relative comparisons, not exact financial accounting.

## Consequences

### Positive

1. **Correctness**: Stochastic RSI now produces accurate signals
2. **Reliability**: Order book processing works with all data formats
3. **Robustness**: Async timeouts prevent hanging operations
4. **Safety**: Transaction support for multi-statement atomicity
5. **Usability**: Better float parsing for configuration values
6. **Resilience**: Automatic reconnection with exponential backoff (P3.1)
7. **Data Quality**: VWAP handles NaN values gracefully (P3.2)
8. **Consistency**: Unified timestamp handling across codebase (P3.3)
9. **Thread Safety**: ConfigLoader safe for concurrent access (P3.4)
10. **Configurability**: Token estimation tunable per deployment (P3.5)

### Negative

1. **Test Updates**: 1 test updated for Supertrend direction change
2. **Breaking Change**: Stochastic RSI values will differ from previous (now correct)

### Risks Mitigated

- **Trading Risk**: Incorrect Stochastic RSI signals could have led to bad trades
- **Data Risk**: Order book mismatch could have caused zero-value features
- **Availability Risk**: Missing timeout could hang entire snapshot process
- **Connection Risk**: Database disconnections now handled gracefully (P3.1)
- **Data Integrity Risk**: NaN propagation in VWAP now prevented (P3.2)
- **Concurrency Risk**: Thread-safe config prevents race conditions (P3.4)

## Implementation

- **Files Modified**: 6 source files + 1 test file
- **Tests**: 917 passing (all existing + updated)
- **Coverage**: 87% maintained
- **Version**: v0.3.2 (P3 fixes)

## Related Documents

- [Phase 1 Foundation Review Plan](../../development/reviews/full/review-4/phase-1-foundation.md)
- [Phase 1 Findings](../../development/reviews/full/review-4/findings/phase-1-findings.md)
- [ADR-006: Consolidated Review Fixes](ADR-006-consolidated-review-fixes.md)

---

*ADR-007 - December 2025*
*Updated: 2025-12-19 - P3 fixes implemented*
