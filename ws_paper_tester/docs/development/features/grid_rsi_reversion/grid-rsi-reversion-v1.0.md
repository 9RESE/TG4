# Grid RSI Reversion Strategy v1.0.0

**Document Version:** 1.0
**Created:** 2025-12-14
**Author:** Strategy Development
**Status:** Implemented - Ready for Testing
**Strategy Version:** 1.0.0

---

## Overview

Grid RSI Reversion combines grid trading mechanics with RSI-based mean reversion signals. Grid levels provide primary entry signals, while RSI acts as a confidence modifier to enhance signal quality and position sizing.

### Key Differentiators from Momentum Scalping

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
├── config.py         # Configuration and enums
├── validation.py     # Config validation
├── grid.py           # Grid level management
├── indicators.py     # RSI, ATR, ADX calculations
├── signal.py         # Main generate_signal()
├── exits.py          # Exit condition checks
├── risk.py           # Position and accumulation limits
├── regimes.py        # Volatility/session classification
└── lifecycle.py      # on_start, on_fill, on_stop
```

### Required Interface

```python
from strategies.grid_rsi_reversion import (
    STRATEGY_NAME,      # "grid_rsi_reversion"
    STRATEGY_VERSION,   # "1.0.0"
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

### Default Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `grid_type` | `geometric` | Grid spacing type (geometric/arithmetic) |
| `num_grids` | `15` | Grid levels per side |
| `grid_spacing_pct` | `1.5` | Spacing between levels (%) |
| `range_pct` | `7.5` | Range from center (%) |
| `rsi_period` | `14` | RSI calculation period |
| `rsi_oversold` | `30` | RSI oversold threshold |
| `rsi_overbought` | `70` | RSI overbought threshold |
| `position_size_usd` | `20.0` | Base size per grid level |
| `max_position_usd` | `100.0` | Max total position per symbol |
| `max_accumulation_levels` | `5` | Max filled grid levels before pause |
| `stop_loss_pct` | `3.0` | Stop loss below lowest grid level |
| `adx_threshold` | `30` | ADX threshold for trend filter |

### Per-Symbol Overrides

#### XRP/USDT (High Suitability)
- Geometric grid spacing
- 15 grid levels, 1.5% spacing
- $25 position size, $100 max

#### BTC/USDT (Medium-High Suitability)
- Arithmetic grid spacing
- 20 grid levels, 1.0% spacing
- $50 position size, $150 max
- Relaxed RSI (35/65)

#### XRP/BTC (Medium Suitability)
- Geometric grid spacing
- 10 grid levels, 2.0% spacing
- $15 position size, $60 max
- 120s cooldown (vs 60s default)

---

## Signal Generation Flow

```
1. Initialize state + grid levels (per symbol)
2. Check trend filter (ADX < 30)
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
8. Generate Signal or None
```

---

## Grid Mechanics

### Grid Level Setup

Grids are created around a center price with levels above (sell) and below (buy).

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

A cycle completes when a buy order is filled and subsequently the matching sell level is triggered. Each completed cycle represents profit capture.

### Grid Recentering

Grids recenter automatically when:
1. N cycles have completed (default: 5)
2. Minimum time has passed (default: 1 hour)

---

## Risk Management

### Accumulation Limits

- Maximum unfilled buy levels before pausing: `max_accumulation_levels` (default: 5)
- Prevents over-accumulation in trending markets

### Trend Filter

- ADX > 30 pauses grid trading
- Grid strategies perform poorly in strong trends

### Position Limits

- Per-symbol maximum: `max_position_usd`
- Total portfolio maximum: `max_total_long_exposure`
- Correlation adjustment: Same-direction reduction

### Circuit Breaker

- Triggers after 3 consecutive losses
- 15-minute cooldown before resuming

---

## Adaptive Features

### Adaptive RSI Zones

RSI thresholds expand based on ATR volatility:
```python
expansion = min(5, atr_pct * 2)
adaptive_oversold = max(15, base_oversold - expansion)
adaptive_overbought = min(85, base_overbought + expansion)
```

### ATR-Based Grid Spacing

When enabled, grid spacing uses the larger of:
- Configured `grid_spacing_pct`
- ATR-derived spacing: `(ATR / price) * 100 * atr_multiplier`

### Volatility Regimes

| Regime | Volatility | Grid Adjustment | Size Adjustment |
|--------|-----------|-----------------|-----------------|
| LOW | < 0.3% | 0.8x spacing | 1.0x size |
| MEDIUM | 0.3-0.8% | 1.0x spacing | 1.0x size |
| HIGH | 0.8-1.5% | 1.3x spacing | 0.8x size |
| EXTREME | > 1.5% | Pause trading | - |

---

## Exit Conditions

### Priority Order

1. **Grid Stop Loss** - Below lowest grid level
2. **Max Drawdown** - Position drawdown exceeds threshold
3. **Grid Cycle Sell** - Price reaches matched sell level
4. **Stale Position** - No cycles in extended time (2 hours)

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

## Implementation Notes

### State Management

The strategy maintains per-symbol state:
- `grid_levels[symbol]` - List of grid level dicts
- `grid_metadata[symbol]` - Grid statistics and settings
- `position_entries[symbol]` - Entry tracking
- `position_by_symbol[symbol]` - Current position size

### Grid Level Structure

```python
{
    'level_index': 0,
    'price': 2.3153,
    'side': 'buy',
    'size': 20.0,
    'filled': False,
    'fill_price': None,
    'fill_time': None,
    'order_id': 'XRP_USDT_abc123_buy_0',
    'matched_order_id': 'XRP_USDT_abc123_sell_0',
}
```

---

## Version History

### v1.0.0 (2025-12-14)
- Initial implementation
- Grid level setup with geometric/arithmetic spacing
- RSI confidence calculation for entries
- Adaptive RSI zones based on ATR
- Grid cycle completion tracking
- Stop loss below lowest grid level
- Trend filter using ADX
- Volatility regime classification
- Per-symbol configuration

---

## References

- [Master Plan v1.0](../../review/grid_rsi_reversion/master-plan-v1.0.md) - Research and planning document
- [Strategy Development Guide](../../strategy-development-guide.md) - Framework guidelines
