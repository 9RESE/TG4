# EMA-9 Trend Flip Strategy v1.0.0

## Overview

The EMA-9 Trend Flip strategy is a trend-following approach based on the 9-period Exponential Moving Average (EMA). Entry signals are generated when price "flips" from consistently opening on one side of the EMA to opening on the opposite side, indicating a potential trend change.

**Implementation Date:** 2025-12-15
**Based On:** ws_paper_tester/docs/development/plans/ema9/ema-9-strategy-analysis.md (Option 3)

## Strategy Specifications

| Property | Value |
|----------|-------|
| Name | ema9_trend_flip |
| Version | 1.0.0 |
| Pairs | BTC/USDT |
| Timeframe | 1H (built from 1m candles) |
| Style | Trend-following EMA flip detection |

## Core Concepts

### EMA-9 Flip Detection

The strategy monitors when price crosses from one side of the EMA to the other:

1. **Consecutive Tracking**: Track N consecutive candles opening above or below EMA
2. **Flip Signal**: When candle opens on opposite side after N consecutive candles
3. **Buffer Zone**: Small percentage buffer to reduce whipsaw false signals

### Timeframe Aggregation

Since DataSnapshot only provides 1-minute candles, the strategy:

1. Builds 1H candles from 60 consecutive 1-minute candles
2. Calculates EMA-9 on hourly candle open prices
3. Evaluates flip conditions at the hourly level

### Why 1H Timeframe?

Per the EMA-9 Strategy Analysis research:
- **Weekly/Daily**: Too long, opportunity cost concerns
- **4H**: Reasonable but still infrequent signals
- **1H**: Good balance of signal frequency vs quality (Recommended)
- **15m/5m**: More signals but higher noise

## Entry Logic

### Long Entry Conditions

1. **Previous N candles** opened below EMA-9 (with buffer)
2. **Current candle** opens above EMA-9 (with buffer)
3. **No existing position** in the same direction
4. **Cooldown period** has elapsed since last signal

### Short Entry Conditions

1. **Previous N candles** opened above EMA-9 (with buffer)
2. **Current candle** opens below EMA-9 (with buffer)
3. **No existing position** in the same direction
4. **Cooldown period** has elapsed since last signal

## Exit Logic

| Priority | Exit Type | Condition |
|----------|-----------|-----------|
| 1 | Stop Loss | Position loss exceeds SL% |
| 2 | Take Profit | Position profit exceeds TP% |
| 3 | EMA Flip | Price flips to opposite side of EMA |
| 4 | Max Hold Time | Position exceeds maximum hold duration |

## Configuration

### Default Configuration

```python
CONFIG = {
    # EMA Settings
    'ema_period': 9,                    # EMA period
    'consecutive_candles': 3,           # Min consecutive candles before flip
    'buffer_pct': 0.1,                  # Buffer % to reduce whipsaws
    'use_open_price': True,             # Use candle open price

    # Timeframe Settings
    'candle_timeframe_minutes': 60,     # 1H candles
    'min_candles_required': 15,         # Minimum candles for EMA calculation

    # Position Sizing (USD)
    'position_size_usd': 50.0,          # Trade size
    'max_position_usd': 100.0,          # Maximum position exposure
    'min_trade_size_usd': 10.0,         # Minimum trade size

    # Risk Management - 2:1 R:R
    'stop_loss_pct': 1.0,               # Stop loss percentage
    'take_profit_pct': 2.0,             # Take profit percentage
    'use_atr_stops': False,             # Use ATR-based stops
    'atr_stop_mult': 1.5,               # ATR multiplier for SL
    'atr_tp_mult': 3.0,                 # ATR multiplier for TP

    # Exit Conditions
    'exit_on_flip': True,               # Exit on opposing EMA flip
    'max_hold_hours': 72,               # Maximum hold time (3 days)

    # Cooldown Mechanisms
    'cooldown_minutes': 30,             # Min minutes between signals
    'cooldown_after_loss_minutes': 60,  # Extended cooldown after loss

    # Signal Tracking
    'track_rejections': True,
}
```

### Per-Symbol Configuration

| Symbol | Position Size | SL% | TP% | Consecutive |
|--------|---------------|-----|-----|-------------|
| BTC/USDT | $50 | 1.0% | 2.0% | 3 |

## Risk Management

### Risk-Reward Ratio

- **Default**: 2:1 (2% take profit, 1% stop loss)
- **ATR-Based** (optional): 3.0x ATR for TP, 1.5x ATR for SL

### Position Limits

- **Max Position**: $100 USD total exposure
- **Min Trade Size**: $10 USD

### Cooldown Mechanisms

| Condition | Cooldown |
|-----------|----------|
| After any signal | 30 minutes |
| After losing trade | 60 minutes |

## Module Structure

```
ws_paper_tester/strategies/ema9_trend_flip/
├── __init__.py      # Public API exports
├── config.py        # Enums, CONFIG, SYMBOL_CONFIGS
├── indicators.py    # EMA calculation, candle building, ATR
├── signal.py        # Main generate_signal function
├── exits.py         # Exit condition logic
├── risk.py          # Position limits, entry signal creation
└── lifecycle.py     # on_start, on_fill, on_stop callbacks
```

## Compliance

### Strategy Development Guide v2.0 Compliance

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| Signal Rejection Tracking | Done | `track_rejection()` in risk.py |
| Per-Symbol Configuration | Done | `SYMBOL_CONFIGS` in config.py |
| R:R Ratio >= 1:1 | Done | 2:1 ratio (2% TP / 1% SL) |
| Position Sizing (USD) | Done | All sizes in USD |
| Indicator Logging | Done | `state['indicators']` on all paths |
| Lifecycle Callbacks | Done | on_start, on_fill, on_stop |
| Cooldown Mechanisms | Done | Time-based + extended after loss |
| Maximum Hold Time | Done | 72 hours default |

## Usage

```python
from ws_paper_tester.strategies.ema9_trend_flip import (
    generate_signal, on_start, on_fill, on_stop,
    CONFIG, SYMBOL_CONFIGS
)

# Initialize
state = {}
on_start(CONFIG, state)

# Generate signals (called on each tick)
signal = generate_signal(data, CONFIG, state)

# Handle fills
on_fill(fill_data, state)

# Shutdown
on_stop(state)
```

## Indicators Logged

The strategy logs the following indicators in `state['indicators']`:

| Indicator | Description |
|-----------|-------------|
| `symbol` | Trading symbol |
| `status` | Current status (active, warming_up, cooldown, etc.) |
| `price` | Current market price |
| `ema_9` | Current EMA-9 value |
| `current_position` | Candle position relative to EMA (above/below/neutral) |
| `prev_position` | Previous consecutive position |
| `prev_consecutive_count` | Count of consecutive candles |
| `atr` | Average True Range |
| `position_side` | Current position (long/short/None) |
| `position_size` | Current position size |
| `entry_price` | Entry price if in position |
| `trade_count` | Total trades taken |
| `win_count` | Winning trades |
| `loss_count` | Losing trades |
| `total_pnl` | Total P&L |

## Signal Rejection Reasons

| Reason | Description |
|--------|-------------|
| `warming_up` | Insufficient candles for EMA calculation |
| `no_price_data` | No price data available |
| `insufficient_candles` | Not enough data for indicators |
| `no_flip_signal` | No EMA flip detected |
| `existing_position` | Already in position |
| `max_position` | Maximum position limit reached |
| `time_cooldown` | Cooldown period active |
| `buffer_not_met` | Price within buffer zone |

## Test Coverage

The strategy includes comprehensive unit tests (26 tests):

- Strategy discovery and loading
- EMA calculation accuracy
- Hourly candle building from 1-minute data
- Candle position classification
- ATR calculation
- Signal generation edge cases
- Position tracking and fill handling
- Exit condition checks
- Cooldown mechanisms
- Position size limits

## References

- EMA-9 Strategy Analysis: `ws_paper_tester/docs/development/plans/ema9/ema-9-strategy-analysis.md`
- Strategy Development Guide: `ws_paper_tester/docs/development/strategy-development-guide.md`

## Version History

### v1.0.0 (2025-12-15)
- Initial implementation based on EMA-9 Strategy Analysis (Option 3)
- 1H timeframe with candle aggregation from 1-minute data
- EMA-9 calculation on hourly opens
- Flip detection with consecutive candle confirmation
- Buffer percentage to reduce whipsaws
- Exit on opposing EMA flip
- ATR-based or fixed percentage stops
- 2:1 risk-reward ratio
- Time-based cooldown mechanisms
- Maximum hold time exit condition
