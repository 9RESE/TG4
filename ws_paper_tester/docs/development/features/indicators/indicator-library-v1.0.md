# Centralized Indicator Library v1.0

**Status:** IMPLEMENTED (Phases 0-7)
**Date:** 2025-12-15
**Version:** 1.0.0
**Plan:** `docs/development/plans/indicator-library-refactoring-v1.0.md`

---

## Executive Summary

This release introduces a centralized technical indicator library that consolidates ~45 duplicated indicator functions from 8 strategy modules into a unified `ws_tester/indicators/` package. The library provides:

- **27 indicator functions** across 7 modules
- **5 structured return types** (NamedTuples)
- **Flexible input handling** (accepts both Candle objects and raw price lists)
- **46 comprehensive tests** with golden fixture regression testing

---

## Architecture

### Package Structure

```
ws_tester/indicators/
├── __init__.py          # Public API exports (27 functions, 5 types)
├── _types.py            # Type definitions (PriceInput, NamedTuples, extractors)
├── moving_averages.py   # SMA, EMA (single value and series)
├── oscillators.py       # RSI (Wilder's), ADX, MACD
├── volatility.py        # ATR, Bollinger Bands, volatility percentage
├── correlation.py       # Rolling Pearson correlation
├── volume.py            # Volume ratio, VPIN, micro-price
├── flow.py              # Trade flow analysis
└── trend.py             # Trend slope, trailing stops
```

### Test Structure

```
tests/
├── fixtures/
│   └── indicator_test_data.py  # Golden fixtures (50 test candles, expected values)
└── test_indicators.py          # 46 tests (unit + regression)
```

---

## API Reference

### Type Definitions (`_types.py`)

```python
# Flexible input type - accepts Candles or raw price lists
PriceInput = Union[List[Candle], Tuple[Candle, ...], List[float]]

# Structured return types
class BollingerResult(NamedTuple):
    sma: Optional[float]
    upper: Optional[float]
    lower: Optional[float]
    std_dev: Optional[float]

class ATRResult(NamedTuple):
    atr: Optional[float]
    atr_pct: Optional[float]
    tr_series: List[float]

class TradeFlowResult(NamedTuple):
    buy_volume: float
    sell_volume: float
    imbalance: float
    total_volume: float
    trade_count: int
    valid: bool

class TrendResult(NamedTuple):
    slope_pct: float
    is_trending: bool

class CorrelationTrendResult(NamedTuple):
    slope: float
    is_declining: bool
    direction: str  # 'declining', 'stable', 'improving'
```

### Moving Averages (`moving_averages.py`)

| Function | Description | Returns |
|----------|-------------|---------|
| `calculate_sma(data, period)` | Simple Moving Average | `Optional[float]` |
| `calculate_sma_series(data, period)` | SMA for all valid points | `List[float]` |
| `calculate_ema(data, period)` | Exponential Moving Average | `Optional[float]` |
| `calculate_ema_series(data, period)` | EMA for all valid points | `List[float]` |

### Oscillators (`oscillators.py`)

| Function | Description | Returns |
|----------|-------------|---------|
| `calculate_rsi(data, period=14)` | RSI with Wilder's smoothing | `Optional[float]` |
| `calculate_rsi_series(data, period=14)` | RSI series | `List[float]` |
| `calculate_adx(data, period=14)` | Average Directional Index | `Optional[float]` |
| `calculate_macd(data, fast=12, slow=26, signal=9)` | MACD values | `Dict[str, Optional[float]]` |
| `calculate_macd_with_history(data, ...)` | MACD with crossover detection | `Dict[str, Any]` |

### Volatility (`volatility.py`)

| Function | Description | Returns |
|----------|-------------|---------|
| `calculate_volatility(data, lookback=20)` | Std dev of returns * 100 | `float` |
| `calculate_atr(data, period=14, rich_output=False)` | Average True Range | `Optional[float]` or `ATRResult` |
| `calculate_bollinger_bands(data, period=20, num_std=2.0)` | Bollinger Bands | `BollingerResult` |
| `calculate_z_score(price, sma, std_dev)` | Z-score | `float` |
| `get_volatility_regime(volatility_pct, config)` | Regime classification | `Tuple[str, float, float]` |

### Correlation (`correlation.py`)

| Function | Description | Returns |
|----------|-------------|---------|
| `calculate_rolling_correlation(prices_a, prices_b, window=20)` | Pearson correlation on returns | `Optional[float]` |
| `calculate_correlation_trend(history, lookback=10)` | Correlation trend direction | `CorrelationTrendResult` |

### Volume (`volume.py`)

| Function | Description | Returns |
|----------|-------------|---------|
| `calculate_volume_ratio(data, lookback=20)` | Current vs average volume | `float` |
| `calculate_volume_spike(volumes, lookback=20, recent=3)` | Recent vs historical volume | `float` |
| `calculate_micro_price(orderbook)` | Volume-weighted mid-price | `float` |
| `calculate_vpin(trades, bucket_count=50)` | Volume-sync informed trading | `float` |

### Flow (`flow.py`)

| Function | Description | Returns |
|----------|-------------|---------|
| `calculate_trade_flow(trades, lookback=50)` | Buy/sell volume imbalance | `TradeFlowResult` |
| `check_trade_flow_confirmation(trades_or_imbalance, direction, threshold=0.10)` | Signal confirmation | `Tuple[bool, Dict]` |

### Trend (`trend.py`)

| Function | Description | Returns |
|----------|-------------|---------|
| `calculate_trend_slope(data, period=20)` | Linear regression slope | `TrendResult` |
| `detect_trend_strength(prices, lookback=10, threshold=0.7)` | Directional consistency | `Tuple[bool, str, float]` |
| `calculate_trailing_stop(entry, highest, lowest, side, activation_pct, trail_pct)` | Trailing stop price | `Optional[float]` |

---

## Usage Examples

### Basic Usage

```python
from ws_tester.indicators import (
    calculate_sma, calculate_ema, calculate_rsi,
    calculate_volatility, calculate_bollinger_bands,
    calculate_rolling_correlation,
)

# Works with both Candle objects and raw price lists
candles = [...]  # List of Candle objects
prices = [c.close for c in candles]

# Either input works
sma_from_candles = calculate_sma(candles, 20)
sma_from_prices = calculate_sma(prices, 20)  # Same result

# RSI with Wilder's smoothing (industry standard)
rsi = calculate_rsi(prices, period=14)

# Bollinger Bands with structured output
bb = calculate_bollinger_bands(prices, period=20, num_std=2.0)
print(f"SMA: {bb.sma}, Upper: {bb.upper}, Lower: {bb.lower}")

# Cross-asset correlation
xrp_prices = [...]
btc_prices = [...]
correlation = calculate_rolling_correlation(xrp_prices, btc_prices, window=20)
```

### Strategy Integration

```python
from ws_tester.indicators import (
    calculate_rsi, calculate_volatility, get_volatility_regime,
    calculate_trade_flow, check_trade_flow_confirmation,
)

def generate_signal(self, snapshot):
    candles = snapshot.candles['XRP/USDT']

    # Calculate indicators
    rsi = calculate_rsi(candles, period=14)
    volatility = calculate_volatility(candles, lookback=20)
    regime, thresh_mult, size_mult = get_volatility_regime(volatility, self.config)

    # Trade flow confirmation
    flow = calculate_trade_flow(snapshot.trades, lookback=50)
    confirmed, flow_data = check_trade_flow_confirmation(
        flow.imbalance, 'buy', threshold=0.10
    )

    # Build signal with indicators
    return Signal(
        action='buy',
        indicators={
            'rsi': rsi,
            'volatility': volatility,
            'regime': regime,
            'flow_imbalance': flow.imbalance,
            'flow_confirmed': confirmed,
        }
    )
```

---

## Implementation Details

### RSI: Wilder's Smoothing Method

The library uses Wilder's smoothing (industry standard) for RSI calculation:

```
avg_gain = (prev_avg_gain * (period - 1) + current_gain) / period
avg_loss = (prev_avg_loss * (period - 1) + current_loss) / period
```

This differs from simple averaging and produces smoother, less volatile RSI values.

### Correlation: Returns-Based Calculation

Rolling correlation uses percentage returns rather than raw prices to avoid spurious correlation from trending prices:

```python
returns_a = [(a[i] - a[i-1]) / a[i-1] for i in range(1, len(a))]
returns_b = [(b[i] - b[i-1]) / b[i-1] for i in range(1, len(b))]
correlation = covariance(returns_a, returns_b) / (std(returns_a) * std(returns_b))
```

### Volatility: Population Variance

Volatility calculation uses population variance (N, not N-1) for consistency with original implementations:

```python
variance = sum((r - mean) ** 2 for r in returns) / len(returns)
volatility_pct = sqrt(variance) * 100
```

---

## Testing

### Test Categories

1. **Unit Tests** - Known input/output verification
2. **Edge Cases** - Empty data, insufficient data, extreme values
3. **Regression Tests** - Golden fixtures with 1e-10 tolerance
4. **Integration** - All 46 tests pass

### Running Tests

```bash
cd ws_paper_tester
python -m pytest tests/test_indicators.py -v
```

### Golden Fixtures

Test fixtures in `tests/fixtures/indicator_test_data.py` capture expected behavior:

- 50 synthetic XRP/USDT candles with realistic price movement
- 50 synthetic BTC/USDT candles for correlation testing
- 100 synthetic trades for flow calculations
- Pre-calculated expected values for all indicators

---

## Code Review Findings (Addressed)

The following issues were identified and fixed during code review:

| Issue | Resolution |
|-------|------------|
| Type annotation `Tuple` too vague in `volume.py` | Changed to `Tuple[Trade, ...]` |
| Missing type hints in `flow.py` | Added comprehensive Union types |
| Correlation zero-handling misalignment | Fixed to process returns in lockstep |
| Population variance undocumented | Added explanatory comment |
| `extract_volumes` in public exports | Removed from `__all__` (internal helper) |
| Missing RSI edge case tests | Added all-losses and flat market tests |
| Incomplete module docstring | Updated to list all NamedTuples |

---

## Migration Guide (Phase 8 - Future)

When migrating strategies to use the centralized library:

1. **Update imports:**
   ```python
   from ws_tester.indicators import calculate_rsi, calculate_volatility
   ```

2. **Update return type handling:**
   ```python
   # Old: bb = calculate_bollinger(prices)  # tuple
   # New: bb = calculate_bollinger_bands(prices)  # BollingerResult
   print(bb.sma, bb.upper, bb.lower)  # Named access
   ```

3. **Remove duplicated functions** from strategy `indicators.py`

4. **Keep strategy-specific functions** in local module (e.g., WaveTrend oscillator)

5. **Run tests** after each strategy migration

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-15 | Initial release: 27 functions, 46 tests, Phases 0-7 complete |

---

## References

- Plan document: `docs/development/plans/indicator-library-refactoring-v1.0.md`
- Test fixtures: `tests/fixtures/indicator_test_data.py`
- Test suite: `tests/test_indicators.py`

---

*Document generated: 2025-12-15*
*Library version: 1.0.0*
