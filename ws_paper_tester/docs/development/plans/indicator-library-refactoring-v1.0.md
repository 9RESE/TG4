# Indicator Library Refactoring Plan v1.0

**Status:** IMPLEMENTED (Phases 0-7 Complete)
**Date:** 2025-12-15
**Author:** Claude Code
**Scope:** ws_paper_tester/strategies/ indicator consolidation
**Implementation:** See `docs/development/features/indicators/indicator-library-v1.0.md`

---

## Executive Summary

This plan consolidates redundant indicator calculations across 9 trading strategies into a centralized `ws_tester/indicators/` library, eliminating approximately 60-70% of duplicated code while maintaining full behavioral compatibility.

---

## Problem Statement

### Current State: Indicator Redundancy

Each strategy implements its own indicator calculations in separate `indicators.py` files, resulting in:
- **5 implementations** of RSI
- **6 implementations** of volatility calculation
- **5+ implementations** of rolling correlation
- **3 implementations** each of EMA, SMA, ATR, ADX
- **45+ total duplicated functions** across 8 strategy modules

### Redundancy Matrix

| Indicator | Impl Count | Strategies |
|-----------|------------|------------|
| RSI | 5 | mean_reversion, grid_rsi, ratio_trading, momentum_scalping (2x) |
| Volatility | 6 | mean_reversion, grid_rsi, ratio_trading, order_flow, momentum_scalping, market_making |
| Correlation | 5+ | mean_reversion, wavetrend, grid_rsi, ratio_trading, market_making, whale_sentiment |
| EMA | 3 | wavetrend, momentum_scalping, whale_sentiment |
| SMA | 3 | mean_reversion, wavetrend, whale_sentiment |
| ATR | 3 | grid_rsi, momentum_scalping, whale_sentiment |
| ADX | 3 | mean_reversion, grid_rsi, momentum_scalping |
| Trade Flow | 3 | grid_rsi, wavetrend, whale_sentiment |
| Bollinger Bands | 2 | mean_reversion, ratio_trading |
| Trailing Stop | 2 | ratio_trading, market_making |

### Interface Incompatibilities

| Function | Issue | Resolution |
|----------|-------|------------|
| `calculate_rsi` | Input: candles vs closes; Return: float vs Optional; Period: 14 vs 7 | Unified interface with type detection |
| `calculate_volatility` | ratio_trading uses price list, others use candles | Accept both via `extract_closes()` helper |
| `calculate_bollinger_bands` | Different return tuple order (lower,mid,upper vs sma,upper,lower,std) | Standardize to NamedTuple |
| `calculate_atr` | whale_sentiment returns Dict, others return float | Rich return with `rich_output` flag |
| `calculate_correlation` | Param names vary (lookback vs window) | Standardize to `window` |
| `calculate_trade_flow` | grid_rsi returns Tuple, others return Dict | Standardize to NamedTuple |

---

## Solution Architecture

### Target Directory Structure

```
ws_paper_tester/
├── ws_tester/
│   ├── indicators/                    # NEW: Centralized library
│   │   ├── __init__.py               # Public API exports
│   │   ├── _types.py                 # Internal type definitions
│   │   ├── moving_averages.py        # SMA, EMA (single + series)
│   │   ├── oscillators.py            # RSI, ADX, MACD
│   │   ├── volatility.py             # ATR, volatility, Bollinger Bands
│   │   ├── correlation.py            # Rolling Pearson correlation
│   │   ├── volume.py                 # Volume ratio, VPIN, micro-price
│   │   ├── flow.py                   # Trade flow, imbalance
│   │   └── trend.py                  # Trend slope, trailing stops
│   └── types.py                      # Existing (Candle, DataSnapshot, etc.)
└── strategies/
    └── [each strategy]/
        └── indicators.py             # Reduced to strategy-specific functions only
```

### Type Definitions (`_types.py`)

```python
from typing import Union, List, Tuple, NamedTuple, Optional
from ws_tester.types import Candle

# Flexible input type for price data
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

class TrendResult(NamedTuple):
    slope_pct: float
    is_trending: bool
```

---

## Implementation Phases

### Phase 1: Core Infrastructure

**Files to create:**
- `ws_tester/indicators/__init__.py`
- `ws_tester/indicators/_types.py`

**Tasks:**
1. Create `indicators/` package with `__init__.py`
2. Define helper types in `_types.py`:
   - `PriceInput` union type
   - `BollingerResult`, `ATRResult`, `TradeFlowResult`, `TrendResult` NamedTuples
3. Implement `extract_closes(data: PriceInput) -> List[float]` helper
4. Implement `extract_hlc(data: PriceInput) -> Tuple[List[float], List[float], List[float]]` helper

---

### Phase 2: Moving Averages Module

**File:** `ws_tester/indicators/moving_averages.py`

**Functions:**
```python
def calculate_sma(data: PriceInput, period: int) -> Optional[float]
def calculate_sma_series(data: PriceInput, period: int) -> List[float]
def calculate_ema(data: PriceInput, period: int) -> Optional[float]
def calculate_ema_series(data: PriceInput, period: int) -> List[float]
```

**Source implementations:**
- `mean_reversion/indicators.py:14-19` (SMA)
- `wavetrend/indicators.py:45-99` (EMA single + series)
- `whale_sentiment/indicators.py:44-87` (EMA, SMA)
- `momentum_scalping/indicators.py:13-67` (EMA single + series)

---

### Phase 3: Oscillators Module

**File:** `ws_tester/indicators/oscillators.py`

**Functions:**
```python
def calculate_rsi(data: PriceInput, period: int = 14) -> Optional[float]
def calculate_rsi_series(data: PriceInput, period: int = 14) -> List[float]
def calculate_adx(candles: PriceInput, period: int = 14) -> Optional[float]
def calculate_macd(
    data: PriceInput,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> Dict[str, Optional[float]]
def calculate_macd_with_history(
    data: PriceInput,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    history_length: int = 2
) -> Dict[str, Any]
```

**Source implementations:**
- `mean_reversion/indicators.py:22-57` (RSI, period 14)
- `grid_rsi_reversion/indicators.py:12-54` (RSI, period 14)
- `ratio_trading/indicators.py:72-109` (RSI, period 14)
- `momentum_scalping/indicators.py:70-162` (RSI, period 7, + series)
- `mean_reversion/indicators.py:204-310` (ADX)
- `grid_rsi_reversion/indicators.py:91-190` (ADX)
- `momentum_scalping/indicators.py:165-310` (MACD)

**Note:** RSI default period is 14; strategies needing period 7 (momentum_scalping) will pass explicit parameter.

---

### Phase 4: Volatility Module

**File:** `ws_tester/indicators/volatility.py`

**Functions:**
```python
def calculate_volatility(data: PriceInput, lookback: int = 20) -> float
def calculate_atr(
    candles: PriceInput,
    period: int = 14,
    rich_output: bool = False
) -> Union[Optional[float], ATRResult]
def calculate_bollinger_bands(
    data: PriceInput,
    period: int = 20,
    num_std: float = 2.0
) -> BollingerResult
def calculate_z_score(price: float, sma: float, std_dev: float) -> float
def get_volatility_regime(
    volatility_pct: float,
    config: Dict[str, Any]
) -> Tuple[str, float, float]
```

**Source implementations:**
- `mean_reversion/indicators.py:81-102` (volatility)
- `grid_rsi_reversion/indicators.py:625-653` (volatility)
- `ratio_trading/indicators.py:48-69` (volatility - uses price list)
- `order_flow/indicators.py:12-33` (volatility)
- `momentum_scalping/indicators.py:376-400` (volatility)
- `market_making/calculations.py:155-180` (volatility)
- `grid_rsi_reversion/indicators.py:57-88` (ATR)
- `whale_sentiment/indicators.py:90-146` (ATR - returns dict)
- `mean_reversion/indicators.py:60-78` (Bollinger Bands)
- `ratio_trading/indicators.py:11-38` (Bollinger Bands)

---

### Phase 5: Correlation Module

**File:** `ws_tester/indicators/correlation.py`

**Functions:**
```python
def calculate_rolling_correlation(
    prices_a: PriceInput,
    prices_b: PriceInput,
    window: int = 20
) -> Optional[float]
def calculate_correlation_trend(
    correlation_history: List[float],
    lookback: int = 10
) -> Tuple[float, bool, str]
```

**Source implementations:**
- `mean_reversion/indicators.py:139-201` (correlation, uses candles, lookback=50)
- `wavetrend/indicators.py:396-462` (correlation, uses prices, window=20)
- `grid_rsi_reversion/indicators.py:473-530` (correlation, uses prices, lookback=20)
- `ratio_trading/indicators.py:205-257` (correlation, uses prices, lookback=20)
- `whale_sentiment/indicators.py:correlation` (uses prices, window=20)
- `market_making/calculations.py:552-599` (correlation, uses prices, window=20)
- `ratio_trading/indicators.py:260-290` (correlation_trend)

---

### Phase 6: Volume & Flow Modules

**File:** `ws_tester/indicators/volume.py`

**Functions:**
```python
def calculate_volume_ratio(data: PriceInput, lookback: int = 20) -> float
def calculate_volume_spike(
    volumes: List[float],
    lookback: int = 20,
    recent_count: int = 3
) -> float
def calculate_micro_price(ob: OrderbookSnapshot) -> float
def calculate_vpin(trades: Tuple, bucket_count: int = 50) -> float
```

**File:** `ws_tester/indicators/flow.py`

**Functions:**
```python
def calculate_trade_flow(trades: Tuple, lookback: int = 50) -> TradeFlowResult
def check_trade_flow_confirmation(
    trades_or_imbalance: Union[Tuple, float],
    direction: str,
    threshold: float = 0.10,
    lookback: int = 50
) -> Tuple[bool, Dict[str, Any]]
```

**Source implementations:**
- `grid_rsi_reversion/indicators.py:405-450` (volume_ratio)
- `momentum_scalping/indicators.py:313-373` (volume_ratio, volume_spike)
- `order_flow/indicators.py:36-58` (micro_price)
- `market_making/calculations.py:14-35` (micro_price)
- `order_flow/indicators.py:61-120` (VPIN)
- `grid_rsi_reversion/indicators.py:358-404` (trade_flow - returns tuple)
- `wavetrend/indicators.py:465-522` (trade_flow - returns dict)
- `whale_sentiment/indicators.py:trade_flow` (returns dict)

---

### Phase 7: Trend Module

**File:** `ws_tester/indicators/trend.py`

**Functions:**
```python
def calculate_trend_slope(
    candles: PriceInput,
    period: int = 20
) -> TrendResult
def detect_trend_strength(
    prices: List[float],
    lookback: int = 10,
    threshold: float = 0.7
) -> Tuple[bool, str, float]
def calculate_trailing_stop(
    entry_price: float,
    highest_price: float,
    lowest_price: Optional[float],
    side: str,
    activation_pct: float,
    trail_distance_pct: float
) -> Optional[float]
```

**Source implementations:**
- `mean_reversion/indicators.py:105-136` (trend_slope, period=50)
- `market_making/calculations.py:200-240` (trend_slope, lookback=20, returns tuple)
- `ratio_trading/indicators.py:112-153` (detect_trend_strength)
- `ratio_trading/indicators.py:156-180` (trailing_stop - has lowest_price)
- `market_making/calculations.py:121-152` (trailing_stop - no lowest_price)

---

### Phase 8: Strategy Migration

**Migration approach:** Clean break - update all imports immediately, no deprecation wrappers.

**Migration order (by complexity):**

1. **order_flow** - 4 functions (2 unique: VPIN, volume_anomaly)
2. **whale_sentiment** - 12 functions (unique: fear/greed, volume spike detection)
3. **wavetrend** - 11 functions (unique: WaveTrend oscillator)
4. **mean_reversion** - 7 functions (all common)
5. **grid_rsi_reversion** - 11 functions (unique: adaptive RSI zones, grid R:R)
6. **ratio_trading** - 11 functions (unique: position decay, USD conversion)
7. **momentum_scalping** - 15+ functions (unique: EMA alignment, momentum signal)
8. **market_making** - 18 functions (unique: A-S optimal spread, circuit breaker)

**Per-strategy tasks:**
1. Update imports: `from ws_tester.indicators import calculate_rsi, calculate_volatility, ...`
2. Remove duplicated functions from local `indicators.py`
3. Keep strategy-specific functions in local module
4. Update any call sites for new return types (e.g., `BollingerResult`)
5. Run strategy tests to verify behavior unchanged

---

### Phase 9: Comprehensive Testing

**New test files:**
- `tests/test_indicators.py` - Main test suite
- `tests/fixtures/indicator_test_data.py` - Test fixtures and golden data

**Test categories:**

1. **Unit tests** for each indicator function
   - Known input/output pairs
   - Mathematical correctness verification

2. **Edge cases:**
   - Empty data (`[]`, `()`)
   - Insufficient data (fewer candles than period)
   - Single element
   - Extreme values (very large, very small, negative prices)
   - All same values (zero variance scenarios)
   - NaN/Inf handling

3. **Regression tests:**
   - Capture golden fixtures from current implementations before removal
   - Assert new implementations match within `1e-10` floating-point tolerance
   - Test both Candle input and List[float] input paths

4. **Integration tests:**
   - Each strategy's `generate_signal()` produces same outputs
   - Full signal generation workflow unchanged

5. **Performance benchmarks:**
   - Measure latency of critical indicators
   - Ensure no regression from current performance

**Validation workflow:**
1. Create test fixtures capturing current behavior (before any changes)
2. Implement centralized indicators with comprehensive tests
3. Run regression tests to verify exact behavioral match
4. Migrate each strategy, run its test suite after each migration
5. Final full integration test run

---

## Files Summary

### New Files to Create (11)

| File | Description |
|------|-------------|
| `ws_tester/indicators/__init__.py` | Public API exports |
| `ws_tester/indicators/_types.py` | Type definitions (NamedTuples, PriceInput) |
| `ws_tester/indicators/moving_averages.py` | SMA, EMA implementations |
| `ws_tester/indicators/oscillators.py` | RSI, ADX, MACD implementations |
| `ws_tester/indicators/volatility.py` | ATR, volatility, Bollinger Bands |
| `ws_tester/indicators/correlation.py` | Rolling correlation |
| `ws_tester/indicators/volume.py` | Volume ratio, VPIN, micro-price |
| `ws_tester/indicators/flow.py` | Trade flow analysis |
| `ws_tester/indicators/trend.py` | Trend slope, trailing stops |
| `tests/test_indicators.py` | Comprehensive test suite |
| `tests/fixtures/indicator_test_data.py` | Test fixtures and golden data |

### Files to Modify (8 strategy indicator files)

| File | Functions to Remove |
|------|---------------------|
| `strategies/mean_reversion/indicators.py` | 7 functions |
| `strategies/grid_rsi_reversion/indicators.py` | 8 functions |
| `strategies/ratio_trading/indicators.py` | 6 functions |
| `strategies/momentum_scalping/indicators.py` | 10 functions |
| `strategies/wavetrend/indicators.py` | 4 functions |
| `strategies/order_flow/indicators.py` | 2 functions |
| `strategies/whale_sentiment/indicators.py` | 3 functions |
| `strategies/market_making/calculations.py` | 5 functions |

---

## Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Subtle behavioral differences | Medium | High | Golden test fixtures, regression tests |
| Breaking live trading | Low | Critical | Paper tester migration first, extensive validation |
| Performance regression | Low | Medium | Benchmark critical paths before/after |
| Import cycles | Low | Medium | indicators/ imports only ws_tester.types |
| Merge conflicts during migration | Medium | Low | Single atomic migration per strategy |

---

## Success Criteria

1. All existing tests pass (zero regressions)
2. No behavioral changes in indicator outputs (within `1e-10` float tolerance)
3. ~60% reduction in total indicator code lines
4. Single source of truth for each indicator algorithm
5. Comprehensive test coverage for indicator library (>90%)
6. Clear API documentation for unified interfaces

---

## Appendix A: Function Signature Reference

### Final Unified API

```python
# ws_tester/indicators/__init__.py

# Moving Averages
from .moving_averages import (
    calculate_sma,
    calculate_sma_series,
    calculate_ema,
    calculate_ema_series,
)

# Oscillators
from .oscillators import (
    calculate_rsi,
    calculate_rsi_series,
    calculate_adx,
    calculate_macd,
    calculate_macd_with_history,
)

# Volatility
from .volatility import (
    calculate_volatility,
    calculate_atr,
    calculate_bollinger_bands,
    calculate_z_score,
    get_volatility_regime,
)

# Correlation
from .correlation import (
    calculate_rolling_correlation,
    calculate_correlation_trend,
)

# Volume
from .volume import (
    calculate_volume_ratio,
    calculate_volume_spike,
    calculate_micro_price,
    calculate_vpin,
)

# Flow
from .flow import (
    calculate_trade_flow,
    check_trade_flow_confirmation,
)

# Trend
from .trend import (
    calculate_trend_slope,
    detect_trend_strength,
    calculate_trailing_stop,
)

# Types
from ._types import (
    PriceInput,
    BollingerResult,
    ATRResult,
    TradeFlowResult,
    TrendResult,
)
```

---

## Appendix B: Strategy-Specific Functions (Not Migrated)

These functions remain in their respective strategy indicator files:

### WaveTrend Strategy
- `calculate_wavetrend()` - Custom oscillator
- `classify_zone()` - WaveTrendZone enum classification
- `detect_crossover()` - WT1/WT2 crossover detection
- `detect_divergence()` - Price/WaveTrend divergence
- `calculate_confidence()` - WaveTrend-specific confidence

### Whale Sentiment Strategy
- `detect_volume_spike()` - Whale activity detection
- `classify_whale_signal()` - WhaleSignal enum classification
- `calculate_fear_greed_proxy()` - Sentiment indicator
- `classify_sentiment_zone()` - SentimentZone classification
- `validate_volume_spike()` - False positive filtering
- `calculate_composite_confidence()` - Multi-signal confidence

### Grid RSI Reversion Strategy
- `get_adaptive_rsi_zones()` - Volatility-adjusted RSI thresholds
- `classify_rsi_zone()` - RSIZone enum classification
- `calculate_rsi_confidence()` - RSI-based confidence
- `calculate_position_size_multiplier()` - Position sizing
- `calculate_grid_rr_ratio()` - Grid risk:reward calculation

### Ratio Trading Strategy
- `check_position_decay()` - Time-based exit logic
- `get_btc_price_usd()` - Price fetching
- `convert_usd_to_xrp()` - Currency conversion

### Momentum Scalping Strategy
- `check_ema_alignment()` - EMA ribbon analysis
- `check_momentum_signal()` - Signal generation
- `check_5m_trend_alignment()` - Multi-timeframe filter

### Market Making Strategy
- `calculate_optimal_spread()` - Avellaneda-Stoikov formula
- `calculate_reservation_price()` - Inventory adjustment
- `check_fee_profitability()` - Fee analysis
- `check_position_decay()` - Stale position detection
- `check_circuit_breaker()` - Loss protection
- `update_circuit_breaker_on_fill()` - State update
- `get_trading_session()` - Session classification
- `get_session_multipliers()` - Session adjustments
- `check_correlation_pause()` - XRP/BTC correlation logic

### Order Flow Strategy
- `check_volume_anomaly()` - Wash trading detection

---

*Document generated: 2025-12-15*
*Plan version: 1.0*
















