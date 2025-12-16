# EMA-9 Trend Flip Strategy v2.0.0

## Overview

The EMA-9 Trend Flip strategy is a trend-following approach based on the 9-period Exponential Moving Average (EMA). Entry signals are generated when price "flips" from consistently being on one side of the EMA to the opposite side, indicating a potential trend change.

**Version:** 2.0.0
**Release Date:** 2025-12-16
**Previous Version:** 1.0.1

## What's New in v2.0

### Major Changes

| Change | v1.x | v2.0 | Rationale |
|--------|------|------|-----------|
| **strict_candle_mode** | Optional (default: False) | **Required** (always True) | 50% win rate vs 12.6%, 0.48% DD vs 50.44% |
| **Exit Strategy** | Flip + Take Profit + Max Hold | **Flip Only** | Flip IS the exit signal |
| **max_hold_hours** | 72 hours | **Removed** | Hold until flip - no time limit |
| **entry_clearance_pct** | 0.1% | **Removed** | Redundant with strict_candle_mode |
| **take_profit_pct** | 2.0% | **Removed** | Flip IS the profit exit |
| **consecutive_candles** | 2-5 | **1-5** | Support immediate flip detection |

### Optimization Results (6-month BTC/USDT backtest)

| Metric | strict_candle_mode=False | strict_candle_mode=True |
|--------|--------------------------|-------------------------|
| Average Win Rate | 12.6% | **50.0%** |
| Average Max DD | 50.44% | **0.48%** |
| Average P&L | -0.42% | **+0.09%** |
| Trade Count | 11.5 avg | 2.0 avg |

## Strategy Specifications

| Property | Value |
|----------|-------|
| Name | ema9_trend_flip |
| Version | 2.0.0 |
| Pairs | BTC/USDT |
| Timeframe | 5m, 1H, 1D (supported) |
| Style | Trend-following EMA flip detection |

## Core Concepts

### Strict Candle Mode (Required)

The key v2.0 improvement. Instead of checking only the candle OPEN price, strict mode requires the **entire candle** (including wicks) to be clear of the EMA:

```
LEGACY MODE (v1.x):                 STRICT MODE (v2.0):
Only checks OPEN price              Checks entire candle

    High ────┬────                      High ────┬────
             │                                   │
      Open ──┼── > EMA? ✓                 Low ───┴── > EMA? ✓
             │    (ignores wick)               (whole candle must clear)
    Low  ────┴──── touches EMA
    EMA  ──────────────────             EMA  ──────────────────
```

**Why this matters:**
- Legacy mode triggers on candles that touch/cross EMA during their timeframe
- Strict mode only triggers when there's decisive separation from EMA
- Result: Fewer but much higher quality signals

### Exit Strategy: Flip Only

The EMA flip **IS** the exit signal. There is no take-profit target or time limit:

1. **Enter** when price flips to opposite side of EMA
2. **Hold** until price flips back (regardless of time)
3. **Exit** on the flip (this IS your profit/loss realization)

Stop loss remains for catastrophic protection only.

## Entry Logic

### Long Entry Conditions

1. **Previous N candles** entirely below EMA (strict mode)
2. **Current candle** entirely above EMA (strict mode)
3. **No existing position**
4. **Cooldown period** has elapsed

### Short Entry Conditions

1. **Previous N candles** entirely above EMA (strict mode)
2. **Current candle** entirely below EMA (strict mode)
3. **No existing position**
4. **Cooldown period** has elapsed

## Exit Logic

| Priority | Exit Type | Condition |
|----------|-----------|-----------|
| 1 | **EMA Flip** | Price flips to opposite side (PRIMARY) |
| 2 | Stop Loss | ATR-based or % stop hit (PROTECTION ONLY) |

**Removed in v2.0:**
- ~~Take Profit~~ (flip IS the profit exit)
- ~~Max Hold Time~~ (hold until flip)

## Configuration

### v2.0 Default Configuration

```python
CONFIG = {
    # EMA Settings
    'ema_period': 9,                    # EMA period (9 is optimal)
    'consecutive_candles': 2,           # Min consecutive candles (1-5)
    'buffer_pct': 0.0,                  # Buffer % (0 with strict mode)
    'use_open_price': True,             # Use candle open price

    # v2.0: Strict Candle Mode - REQUIRED
    'strict_candle_mode': True,         # Whole candle must clear EMA

    # Timeframe Settings
    'candle_timeframe_minutes': 60,     # Supported: 5, 60, 1440
    'min_candles_required': 15,         # Minimum candles for EMA

    # Position Sizing (USD)
    'position_size_usd': 50.0,
    'max_position_usd': 100.0,
    'min_trade_size_usd': 10.0,

    # Risk Management - Stop Loss for PROTECTION ONLY
    'stop_loss_pct': 2.5,               # Fallback if ATR unavailable
    'use_atr_stops': True,              # ATR adapts to volatility
    'atr_stop_mult': 2.0,               # 2x ATR = stop distance

    # Exit Conditions
    'exit_confirmation_candles': 1,     # 1=immediate, 2+=confirmation

    # Cooldown Mechanisms
    'cooldown_minutes': 30,
    'cooldown_after_loss_minutes': 60,

    # Circuit Breaker
    'use_circuit_breaker': True,
    'max_consecutive_losses': 3,
    'circuit_breaker_minutes': 30,

    # Tracking
    'track_rejections': True,
}
```

### Removed in v2.0

```python
# REMOVED - These no longer exist:
# 'take_profit_pct': 2.0,           # Flip IS the profit exit
# 'max_hold_hours': 72,             # Hold until flip
# 'entry_clearance_pct': 0.1,       # Redundant with strict mode
# 'exit_on_flip': True,             # Always exits on flip (core strategy)
```

## ATR-Based Stops Explained

ATR (Average True Range) measures market volatility. ATR-based stops adapt to current conditions:

| Market State | ATR | Stop Distance (2x ATR) |
|--------------|-----|------------------------|
| Calm | $300 | $600 from entry |
| Normal | $500 | $1000 from entry |
| Volatile | $800 | $1600 from entry |

**Why ATR > Fixed %:**
- Fixed 2.5% at $100k = always $2500
- In volatile markets, this gets hit by normal noise
- ATR stops expand/contract with volatility

**Fallback:** If ATR can't be calculated, uses `stop_loss_pct` as fallback.

## Supported Timeframes

| Timeframe | Minutes | Supported | Notes |
|-----------|---------|-----------|-------|
| 1 minute | 1 | Yes | Very noisy |
| 5 minute | 5 | **Yes** | More signals |
| 15 minute | 15 | **No** | Not in database |
| 30 minute | 30 | **No** | Not in database |
| 1 hour | 60 | **Yes** | Recommended |
| 4 hour | 240 | **No** | Not in database |
| 1 day | 1440 | **Yes** | Fewer signals |

## Parameter Reference

### Signal Detection

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `ema_period` | 9 | 7-21 | EMA calculation period |
| `consecutive_candles` | 2 | 1-5 | Candles on one side before flip |
| `strict_candle_mode` | True | True | **REQUIRED** - whole candle must clear EMA |
| `buffer_pct` | 0.0 | 0.0-0.2 | Buffer zone (use 0 with strict mode) |

### Risk Management

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `stop_loss_pct` | 2.5 | 1.5-5.0 | Fallback % stop |
| `use_atr_stops` | True | True/False | ATR adapts to volatility |
| `atr_stop_mult` | 2.0 | 1.5-3.0 | ATR × multiplier = stop distance |

### Exit

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `exit_confirmation_candles` | 1 | 1-3 | Candles to confirm flip before exit |

### Cooldown

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cooldown_minutes` | 30 | Time between signals |
| `cooldown_after_loss_minutes` | 60 | Extended cooldown after loss |
| `use_circuit_breaker` | True | Stop after consecutive losses |
| `max_consecutive_losses` | 3 | Losses before circuit breaker |

## Module Structure

```
ws_paper_tester/strategies/ema9_trend_flip/
├── __init__.py      # Public API exports
├── config.py        # CONFIG, SYMBOL_CONFIGS, enums
├── indicators.py    # EMA, ATR, candle position logic
├── signal.py        # Main generate_signal function
├── exits.py         # Exit condition logic (flip + stop loss)
├── risk.py          # Position limits, entry signal creation
└── lifecycle.py     # on_start, on_fill, on_stop callbacks
```

## Signal Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    ENTRY SIGNAL FLOW                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  consecutive_candles (N) candles on one side (strict mode)      │
│           ↓                                                     │
│  Current candle flips to other side (strict mode)               │
│           ↓                                                     │
│  cooldown_minutes → Check time since last signal                │
│           ↓                                                     │
│  circuit_breaker → Check not in cooldown from losses            │
│           ↓                                                     │
│  ✅ ENTRY SIGNAL                                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    EXIT SIGNAL FLOW                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Check ATR/% stop_loss → Emergency exit (protection only)       │
│           ↓                                                     │
│  Check EMA flip (opposite direction)                            │
│           ↓                                                     │
│  exit_confirmation_candles → Wait for confirmation              │
│           ↓                                                     │
│  ✅ EXIT SIGNAL                                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

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

## Optimization

### Quick Mode

```bash
cd ws_paper_tester/optimization
python optimize_ema9.py --symbol BTC/USDT --period 6m --quick
```

Quick mode parameters (24 combinations):
- `consecutive_candles`: [1, 2, 3]
- `exit_confirmation_candles`: [1, 2]
- `stop_loss_pct`: [2.0, 3.0]
- `candle_timeframe_minutes`: [5, 60]

### Focused Modes

```bash
# Signal quality tuning
python optimize_ema9.py --focus signal_quality

# Stop loss optimization
python optimize_ema9.py --focus stop_loss

# Timeframe comparison
python optimize_ema9.py --focus timeframes
```

## Version History

### v2.0.0 (2025-12-16)
**Major Release** - Optimization-driven improvements

- **BREAKING**: `strict_candle_mode` now required (always True)
- **BREAKING**: Removed `max_hold_hours` - hold until flip
- **BREAKING**: Removed `entry_clearance_pct` - redundant with strict mode
- **BREAKING**: Removed `take_profit_pct` - flip IS the exit
- Added `consecutive_candles=1` option for immediate flip detection
- Updated optimizer grids to use supported timeframes only (5, 60, 1440)
- Improved ATR stop documentation
- Exit strategy simplified to flip-only + protection stop

### v1.0.1 (2025-12-15)
- Circuit breaker integration
- Database warmup support
- Complete `on_fill()` handler
- Config validation in `on_start()`
- Structured logging

### v1.0.0 (2025-12-15)
- Initial implementation
- 1H timeframe with candle aggregation
- EMA-9 calculation on hourly opens
- Flip detection with consecutive candle confirmation

## References

- EMA-9 Strategy Analysis: `ws_paper_tester/docs/development/plans/ema9/ema-9-strategy-analysis.md`
- Optimization System: `ws_paper_tester/docs/development/features/optimization/optimization-system-v1.0.md`
- Strategy Development Guide: `ws_paper_tester/docs/development/strategy-development-guide.md`
