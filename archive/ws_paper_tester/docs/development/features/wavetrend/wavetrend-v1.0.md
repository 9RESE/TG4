# WaveTrend Oscillator Strategy v1.0.0

## Overview

The WaveTrend Oscillator strategy trades based on the LazyBear WaveTrend indicator, a momentum oscillator that identifies overbought/oversold conditions with cleaner signals than RSI. It uses a dual-line crossover mechanism (WT1/WT2) similar to MACD.

**Implementation Date:** 2025-12-14

## Strategy Specifications

| Property | Value |
|----------|-------|
| Name | wavetrend |
| Version | 1.0.0 |
| Pairs | XRP/USDT, BTC/USDT, XRP/BTC |
| Timeframe | 5m (adaptable to hourly) |
| Style | Zone-based momentum reversal |

## Core Concepts

### WaveTrend Oscillator Formula

The WaveTrend oscillator is calculated as follows:

1. **HLC3** = (High + Low + Close) / 3 (Typical Price)
2. **ESA** = EMA(HLC3, channel_length) (Exponential Smoothed Average)
3. **D** = EMA(|HLC3 - ESA|, channel_length) (Average Deviation)
4. **CI** = (HLC3 - ESA) / (0.015 * D) (Channel Index)
5. **WT1** = EMA(CI, average_length) (WaveTrend Line 1)
6. **WT2** = SMA(WT1, ma_length) (WaveTrend Line 2 / Signal)

### Zone Classification

| Zone | WT1 Range | Description |
|------|-----------|-------------|
| Extreme Overbought | ≥ 80 | Very high probability of reversal |
| Overbought | ≥ 60 | High probability of reversal |
| Neutral | -60 to 60 | No clear signal |
| Oversold | ≤ -60 | High probability of reversal |
| Extreme Oversold | ≤ -80 | Very high probability of reversal |

## Entry Logic

### Long Entry Conditions

1. **Bullish Crossover**: WT1 crosses above WT2
2. **Zone Confirmation**: Crossover occurs from/in oversold zone (WT1 ≤ -60)
3. **Divergence Bonus**: +10% confidence if bullish divergence detected

### Short Entry Conditions

1. **Bearish Crossover**: WT1 crosses below WT2
2. **Zone Confirmation**: Crossover occurs from/in overbought zone (WT1 ≥ 60)
3. **Divergence Bonus**: +10% confidence if bearish divergence detected

## Exit Logic

| Priority | Exit Type | Condition |
|----------|-----------|-----------|
| 1 | Stop Loss | Position loss exceeds SL% |
| 2 | Take Profit | Position profit exceeds TP% |
| 3 | Crossover Reversal | Opposite crossover signal |
| 4 | Extreme Zone | Long at extreme OB, Short at extreme OS |

## Configuration

### Default Configuration

```python
CONFIG = {
    # WaveTrend Indicator
    'wt_channel_length': 10,
    'wt_average_length': 21,
    'wt_ma_length': 4,

    # Zone Thresholds
    'wt_overbought': 60,
    'wt_oversold': -60,
    'wt_extreme_overbought': 80,
    'wt_extreme_oversold': -80,

    # Signal Settings
    'require_zone_exit': True,
    'use_divergence': True,
    'divergence_lookback': 14,

    # Position Sizing (USD)
    'position_size_usd': 25.0,
    'max_position_usd': 75.0,
    'max_position_per_symbol_usd': 50.0,

    # Risk Management
    'stop_loss_pct': 1.5,
    'take_profit_pct': 3.0,  # 2:1 R:R ratio

    # Cooldown
    'cooldown_seconds': 60.0,
}
```

### Per-Symbol Configuration

| Symbol | OB | OS | Extreme OB | Extreme OS | Position Size | SL% | TP% |
|--------|----|----|------------|------------|---------------|-----|-----|
| XRP/USDT | 60 | -60 | 75 | -75 | $25 | 1.5% | 3.0% |
| BTC/USDT | 65 | -65 | 80 | -80 | $50 | 1.0% | 2.0% |
| XRP/BTC | 55 | -55 | 70 | -70 | $15 | 2.0% | 4.0% |

## Risk Management

### Confidence Calculation

| Condition | Base Confidence | Bonus |
|-----------|-----------------|-------|
| Bullish crossover in neutral | 0.55 | - |
| Bullish crossover from oversold | 0.75 | - |
| Bullish crossover from extreme oversold | 0.75 | +0.05 |
| Bullish divergence detected | - | +0.10 |
| **Maximum Long Confidence** | 0.92 | |
| **Maximum Short Confidence** | 0.88 | |

### Circuit Breaker

- **Max Consecutive Losses**: 3
- **Cooldown Duration**: 30 minutes

### Correlation Management

- **Max Total Long Exposure**: $100
- **Max Total Short Exposure**: $100
- **Same Direction Multiplier**: 0.75 (reduce size when multiple pairs same direction)

## Session Awareness

| Session | Hours (UTC) | Size Multiplier |
|---------|-------------|-----------------|
| Asia | 00:00 - 08:00 | 0.8 |
| Europe | 08:00 - 14:00 | 1.0 |
| US/Europe Overlap | 14:00 - 17:00 | 1.1 |
| US | 17:00 - 21:00 | 1.0 |
| Off Hours | 21:00 - 24:00 | 0.5 |

## Module Structure

```
ws_paper_tester/strategies/wavetrend/
├── __init__.py      # Public API exports
├── config.py        # Enums, CONFIG, SYMBOL_CONFIGS
├── indicators.py    # WaveTrend calculation, zone classification, divergence
├── signal.py        # Main generate_signal function
├── exits.py         # Exit signal logic
├── risk.py          # Position limits, fee checks, correlation management
├── regimes.py       # Session awareness, zone regime adjustments
├── lifecycle.py     # on_start, on_fill, on_stop callbacks
└── validation.py    # Configuration validation
```

## Compliance

### Strategy Development Guide v2.0 Compliance

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| Volatility Regime Classification | N/A | Uses zone-based instead of volatility |
| Circuit Breaker Protection | ✅ | `check_circuit_breaker()` in risk.py |
| Signal Rejection Tracking | ✅ | `track_rejection()` in signal.py |
| Per-Symbol Configuration | ✅ | `SYMBOL_CONFIGS` in config.py |
| Correlation Monitoring | ✅ | `check_correlation_exposure()` in risk.py |
| R:R Ratio ≥ 1:1 | ✅ | 2:1 ratio (3% TP / 1.5% SL) |
| Position Sizing (USD) | ✅ | All sizes in USD |
| Indicator Logging | ✅ | `state['indicators']` on all paths |

## Usage

```python
from ws_paper_tester.strategies.wavetrend import (
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

## References

- [LazyBear WaveTrend Oscillator](https://www.tradingview.com/script/2KE8wTuF-Indicator-WaveTrend-Oscillator-WT/)
- Master Plan: `ws_paper_tester/docs/development/review/wavetrend/master-plan-v1.0.md`
- Strategy Development Guide: `ws_paper_tester/docs/development/strategy-development-guide.md`
