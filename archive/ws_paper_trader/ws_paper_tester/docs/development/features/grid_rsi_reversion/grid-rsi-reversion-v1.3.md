# Grid RSI Reversion Strategy v1.3.0

**Document Version:** 1.3
**Created:** 2025-12-14
**Updated:** 2025-12-17
**Author:** Strategy Development
**Status:** Implemented - Ready for Optimization
**Strategy Version:** 1.3.0

---

## Overview

Grid RSI Reversion combines grid trading mechanics with RSI-based mean reversion signals. Grid levels provide primary entry signals, while RSI acts as a confidence modifier to enhance signal quality and position sizing.

**Version:** 1.3.0
**Release Date:** 2025-12-17
**Previous Version:** 1.2.0

## What's New in v1.3.0

### Major Changes

| Change | v1.2.x | v1.3.0 | Rationale |
|--------|--------|--------|-----------|
| **Configurable Timeframe** | Hardcoded 5m | `candle_timeframe_minutes` config | Enable timeframe optimization |
| **Optimizer Enhancements** | Basic CLI | Full CLI with --timeframes, --focus, epilog | Parity with EMA9 optimizer |
| **Parameter Grids** | Limited | Expanded with adaptive features | More comprehensive optimization |
| **Time Estimation** | Fixed | Period-based multipliers | Accurate run time estimates |

### New Configuration Parameters

```python
# v1.3.0: Timeframe Settings
'candle_timeframe_minutes': 5,      # Supported: 5, 60, 1440
'min_candles_required': 25,         # Minimum for indicator calculation

# Optimizer parameter grids now include:
'use_adaptive_rsi': [True, False],  # Toggle adaptive RSI zones
'use_atr_spacing': [True, False],   # Toggle ATR-based spacing
'adx_threshold': [25, 30, 35],      # Trend filter threshold
```

---

## Strategy Specifications

| Property | Value |
|----------|-------|
| Name | grid_rsi_reversion |
| Version | 1.3.0 |
| Pairs | XRP/USDT, BTC/USDT, XRP/BTC |
| Timeframes | 5m (default), 1h, 1d (configurable) |
| Style | Mean-reversion grid trading with RSI confidence |

---

## Key Differentiators from Momentum Scalping

| Aspect | Momentum Scalping | Grid RSI Reversion |
|--------|-------------------|-------------------|
| Entry Logic | Momentum crossovers | Price at grid levels |
| RSI Role | Hard filter | Confidence modifier |
| Position Style | Single position | Multi-level accumulation |
| Exit Logic | Fixed TP/SL | Cycle completion + stop |
| Hold Time | 1-5 minutes | Until cycle complete |
| Best Market | Trending with pullbacks | Range-bound/mean-reverting |

---

## Architecture

### Module Structure

```
strategies/grid_rsi_reversion/
├── __init__.py       # Public API exports
├── config.py         # Configuration and enums (v1.3.0: timeframe settings)
├── validation.py     # Config validation
├── grid.py           # Grid level management
├── indicators.py     # RSI, ATR, ADX calculations
├── signal.py         # Main generate_signal() (v1.3.0: dynamic timeframe)
├── exits.py          # Exit condition checks
├── risk.py           # Position and accumulation limits
├── regimes.py        # Volatility/session classification
└── lifecycle.py      # on_start, on_fill, on_stop
```

### Required Interface

```python
from strategies.grid_rsi_reversion import (
    STRATEGY_NAME,      # "grid_rsi_reversion"
    STRATEGY_VERSION,   # "1.3.0"
    SYMBOLS,           # ["XRP/USDT", "BTC/USDT", "XRP/BTC"]
    CONFIG,            # Default configuration dict
    generate_signal,   # Main signal generation function
    on_start,          # Lifecycle: strategy initialization
    on_fill,           # Lifecycle: fill handling
    on_stop,           # Lifecycle: shutdown
)
```

---

## Configuration

### v1.3.0 Default Configuration

```python
CONFIG = {
    # Timeframe Settings (v1.3.0)
    'candle_timeframe_minutes': 5,      # Primary timeframe
    'min_candles_required': 25,         # Minimum for indicators

    # Grid Settings
    'grid_type': 'geometric',           # 'arithmetic' or 'geometric'
    'num_grids': 15,                    # Grid levels per side
    'grid_spacing_pct': 1.5,            # Spacing between levels (%)
    'range_pct': 7.5,                   # Range from center (%)
    'recenter_after_cycles': 5,         # Recenter after N cycles
    'min_recenter_interval': 3600,      # Min seconds between recenters

    # RSI Settings
    'rsi_period': 14,                   # RSI calculation period
    'rsi_oversold': 30,                 # Oversold threshold
    'rsi_overbought': 70,               # Overbought threshold
    'use_adaptive_rsi': True,           # Adjust zones by volatility
    'rsi_zone_expansion': 5,            # Max zone expansion
    'rsi_extreme_multiplier': 1.3,      # Size multiplier at extremes

    # ATR Settings
    'atr_period': 14,                   # ATR calculation period
    'use_atr_spacing': True,            # Dynamic grid spacing
    'atr_multiplier': 0.3,              # ATR spacing multiplier

    # Position Sizing
    'position_size_usd': 20.0,          # Base size per level
    'max_position_usd': 100.0,          # Max per symbol
    'max_accumulation_levels': 5,       # Max filled levels

    # Risk Management (REC-004: Widened stops)
    'stop_loss_pct': 8.0,               # Stop below lowest grid
    'max_drawdown_pct': 10.0,           # Max portfolio drawdown
    'use_trend_filter': True,           # Pause in trends
    'adx_threshold': 30,                # ADX trend threshold

    # Circuit Breaker
    'use_circuit_breaker': True,
    'max_consecutive_losses': 3,
    'circuit_breaker_minutes': 15,

    # Trade Flow (REC-003)
    'use_trade_flow_confirmation': True,
    'min_volume_ratio': 0.8,
    'flow_confirmation_threshold': 0.2,

    # Correlation (REC-005)
    'use_real_correlation': True,
    'correlation_block_threshold': 0.85,
    'correlation_lookback': 20,
}
```

### Per-Symbol Configurations

#### XRP/USDT (High Suitability)
```python
'XRP/USDT': {
    'grid_type': 'geometric',
    'num_grids': 15,
    'grid_spacing_pct': 1.5,
    'position_size_usd': 25.0,
    'stop_loss_pct': 5.0,           # REC-004
}
```

#### BTC/USDT (Medium-High Suitability)
```python
'BTC/USDT': {
    'grid_type': 'arithmetic',
    'num_grids': 20,
    'grid_spacing_pct': 1.5,        # REC-009: Wider for better R:R
    'position_size_usd': 50.0,
    'stop_loss_pct': 10.0,          # REC-004
    'rsi_oversold': 35,             # Relaxed
    'rsi_overbought': 65,
}
```

#### XRP/BTC (Medium Suitability)
```python
'XRP/BTC': {
    'grid_type': 'geometric',
    'num_grids': 10,                # Fewer due to liquidity
    'grid_spacing_pct': 2.5,        # REC-006: Wider
    'position_size_usd': 10.0,      # REC-006: Smaller
    'stop_loss_pct': 8.0,
    'min_volume_usd': 100_000_000,  # REC-006: Liquidity check
}
```

---

## Optimization

### Command-Line Interface

```bash
# Quick optimization (~108 runs)
python optimize_grid_rsi.py --symbol XRP/USDT --period 3m --quick

# Full optimization (2000+ runs)
python optimize_grid_rsi.py --symbol XRP/USDT --period 6m

# Focused optimization modes
python optimize_grid_rsi.py --focus grid       # Grid structure
python optimize_grid_rsi.py --focus rsi        # RSI parameters
python optimize_grid_rsi.py --focus risk       # Risk management
python optimize_grid_rsi.py --focus timeframes # Timeframe comparison
python optimize_grid_rsi.py --focus adaptive   # Adaptive features

# Parallel execution
python optimize_grid_rsi.py --symbol BTC/USDT --period 3m --parallel --workers 8

# Specific timeframes
python optimize_grid_rsi.py --symbol XRP/USDT --period 3m --timeframes 5,60
```

### Parameter Grids

#### Quick Mode (~108 combinations)

| Parameter | Values |
|-----------|--------|
| `num_grids` | 10, 15, 20 |
| `grid_spacing_pct` | 1.0, 1.5, 2.0 |
| `range_pct` | 7.5, 10.0 |
| `rsi_period` | 12, 14 |
| `rsi_oversold` | 25, 30 |
| `rsi_overbought` | 70, 75 |
| `stop_loss_pct` | 5.0, 8.0 |
| `max_accumulation_levels` | 4, 5 |
| `adx_threshold` | 25, 30 |
| `candle_timeframe_minutes` | 5, 60 |

#### Full Mode (~2000+ combinations)

All quick mode parameters plus:
- Extended ranges for each parameter
- `rsi_extreme_multiplier`: 1.0, 1.2, 1.3
- `use_adaptive_rsi`: True, False
- `use_atr_spacing`: True, False
- `candle_timeframe_minutes`: 5, 60, 1440

---

## Signal Generation Flow

```
1. Get candles for configured timeframe (v1.3.0)
2. Check trend filter (ADX < threshold)
3. Check volatility regime (pause if EXTREME)
4. Calculate indicators (RSI, ATR, ADX)
5. Check existing position exits:
   - Grid stop loss
   - Max drawdown
   - Cycle completion (matched sell)
6. Check grid entry conditions:
   - Price at/near grid level
   - Grid level not filled
   - RSI zone (confidence modifier)
7. Check risk limits:
   - Max accumulation
   - Position limits
   - Correlation exposure
   - Trade flow confirmation (REC-003)
   - Liquidity validation (REC-006)
8. Generate Signal or None
```

---

## Grid Mechanics

### Grid Level Setup

**Geometric Spacing:**
```python
buy_price[i] = center / (1 + spacing_pct/100)^i
sell_price[i] = center * (1 + spacing_pct/100)^i
```

**Arithmetic Spacing:**
```python
spacing = center * (spacing_pct / 100)
buy_price[i] = center - (i * spacing)
sell_price[i] = center + (i * spacing)
```

### Cycle Completion

A cycle completes when a buy order is filled and subsequently the matching sell level is triggered.

### Grid Recentering (REC-008)

Grids recenter when:
1. N cycles completed (default: 5)
2. Minimum time passed (default: 1 hour)
3. ADX below threshold (trend check before recenter)

---

## Risk Management

### Accumulation Limits
- Max unfilled buy levels: `max_accumulation_levels` (default: 5)
- Prevents over-accumulation in trending markets

### Trend Filter
- ADX > 30 pauses grid trading
- Grid strategies underperform in strong trends

### Stop Loss (REC-004)
- Per-symbol stop loss (5-10% based on volatility)
- Below lowest grid level

### R:R Ratio Calculation (REC-007)
```python
rr_ratio = grid_spacing_pct / stop_loss_pct
# Example: 1.5% spacing / 8% stop = 0.19:1 R:R
```

### Trade Flow Confirmation (REC-003)
- Volume ratio vs average
- Flow imbalance threshold
- Rejects trades against flow

### Liquidity Validation (REC-006)
- 24h volume minimum for XRP/BTC
- Prevents trading in illiquid conditions

---

## Supported Timeframes

| Timeframe | Minutes | Supported | Notes |
|-----------|---------|-----------|-------|
| 5 minute | 5 | **Yes** | Default - best for crypto |
| 1 hour | 60 | **Yes** | Fewer signals, cleaner |
| 1 day | 1440 | **Yes** | Position trading |

---

## Version History

### v1.3.0 (2025-12-17)
**Feature Release** - Configurable timeframe support

- **NEW**: `candle_timeframe_minutes` config parameter
- **NEW**: `_get_candles_for_timeframe()` helper for dynamic timeframe selection
- **NEW**: Optimizer `--timeframes` CLI flag
- **NEW**: Optimizer `--focus adaptive` mode
- **NEW**: Optimizer `--focus timeframes` mode
- **IMPROVED**: Expanded parameter grids with adaptive features
- **IMPROVED**: Period-based time estimation in optimizer
- **IMPROVED**: Detailed CLI help with examples (epilog)
- **IMPROVED**: Timeframe tracking in indicators dict

### v1.2.0 (2025-12-16)
**Deep Review v2.1 Implementation** (REC-009 through REC-011)

- **REC-009**: BTC/USDT grid_spacing_pct increased from 1.0% to 1.5%
- **REC-010**: Aligned adx_recenter_threshold to match main threshold (30)
- **REC-011**: Documented VPIN for regime detection as future enhancement
- Compliance score: 97% per deep-review-v2.1

### v1.1.0 (2025-12-15)
**Deep Review v2.0 Implementation** (REC-001 through REC-008)

- **REC-001**: Signal rejection tracking verified
- **REC-002**: Complete indicator logging on all paths
- **REC-003**: Trade flow confirmation
- **REC-004**: Widened stop-loss (5-10% per symbol)
- **REC-005**: Real correlation monitoring
- **REC-006**: Liquidity validation for XRP/BTC
- **REC-007**: Explicit R:R ratio calculation
- **REC-008**: Trend check before grid recentering

### v1.0.0 (2025-12-14)
- Initial implementation based on master-plan-v1.0.md research

---

## Testing Recommendations

### Paper Trading Validation (24-48 hours)

1. Monitor grid cycle completion rate (target: >= 60%)
2. Track accumulation events and recovery
3. Verify stop loss triggers correctly
4. Check drawdown stays within limits (< 10%)

### Key Metrics to Track

| Metric | Target | Warning |
|--------|--------|---------|
| Cycle Completion Rate | >= 60% | < 40% |
| Average Cycle Time | < 4 hours | > 8 hours |
| Max Accumulation Depth | <= 5 levels | > 7 levels |
| Portfolio Drawdown | < 10% | > 15% |
| Win Rate | >= 55% | < 45% |

---

## References

- [Master Plan v1.0](../../review/grid_rsi_reversion/master-plan-v1.0.md) - Research and planning
- [EMA-9 Strategy v2.0](../ema9_trend_flip/ema9-trend-flip-v2.0.md) - Reference implementation
- [Optimization System](../optimization/optimization-system-v1.0.md) - Framework docs
- [Strategy Development Guide](../../strategy-development-guide.md) - Framework guidelines
