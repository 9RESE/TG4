# Ratio Trading Strategy v1.0 - XRP/BTC Mean Reversion

## Overview

The Ratio Trading strategy uses mean reversion on the XRP/BTC pair to accumulate both XRP and BTC over time. It trades based on deviations from a moving average using Bollinger Bands.

## Strategy Logic

### Core Concept
- Track the XRP/BTC price ratio over time
- Calculate a moving average (SMA) and standard deviation
- Use Bollinger Bands to identify overextended conditions
- Buy XRP when the ratio is low (XRP cheap vs BTC)
- Sell XRP when the ratio is high (XRP expensive vs BTC)
- Target mean reversion for profit taking

### Entry Conditions

**Buy Signal (Accumulate XRP):**
- Z-score < -1.0 (price below lower Bollinger Band)
- Available position capacity
- Cooldown period elapsed

**Sell Signal (Accumulate BTC):**
- Z-score > +1.0 (price above upper Bollinger Band)
- XRP position or holdings available to sell
- Cooldown period elapsed

### Exit Conditions
- Take profit: Price reverts toward SMA
- Stop loss: Price continues against position
- Partial exit when z-score approaches 0

## Configuration

```python
CONFIG = {
    # Mean reversion parameters
    'lookback_periods': 20,        # Periods for moving average
    'bollinger_std': 2.0,          # Standard deviations for bands
    'entry_threshold': 1.0,        # Entry at N std devs from mean
    'exit_threshold': 0.5,         # Exit at N std devs (closer to mean)

    # Position sizing (in XRP)
    'position_size_xrp': 30.0,     # Base size per trade in XRP
    'max_position_xrp': 200.0,     # Maximum XRP exposure

    # Risk management
    'stop_loss_pct': 0.6,          # Stop loss percentage
    'take_profit_pct': 0.5,        # Take profit percentage

    # Cooldown
    'cooldown_seconds': 60.0,      # Minimum time between trades
    'min_candles': 10,             # Minimum candles before trading
}
```

## Indicators Logged

| Indicator | Description |
|-----------|-------------|
| `price` | Current XRP/BTC price |
| `sma` | Simple Moving Average |
| `upper_band` | Upper Bollinger Band |
| `lower_band` | Lower Bollinger Band |
| `z_score` | Standard deviations from mean |
| `band_width_pct` | Band width as percentage |
| `position_xrp` | Current XRP position |
| `xrp_accumulated` | Total XRP accumulated |
| `btc_accumulated` | Total BTC accumulated |

## Accumulation Tracking

The strategy tracks dual-asset accumulation:

- **XRP Accumulated**: Total XRP bought through the strategy
- **BTC Accumulated**: Total BTC received from selling XRP

This allows measuring progress toward the goal of growing both holdings.

## Comparison with Other Strategies

| Strategy | Approach | Best For |
|----------|----------|----------|
| **Ratio Trading** | Mean reversion | Balanced accumulation, range-bound markets |
| Market Making | Spread capture | Frequent small profits, inventory balancing |
| Order Flow | Momentum | Trending markets, directional moves |

## Example Trade Flow

1. XRP/BTC at 0.0000220 (below -1σ from SMA 0.0000224)
2. Strategy buys 30 XRP with BTC
3. XRP/BTC rises to 0.0000224 (at SMA)
4. Strategy sells 15 XRP (partial exit at mean)
5. XRP/BTC rises to 0.0000228 (above +1σ)
6. Strategy sells remaining XRP for BTC

Result: Accumulated XRP during dip, accumulated BTC during rally.

## Risk Considerations

1. **Trending Markets**: Mean reversion fails in strong trends
2. **Volatility Expansion**: Band width should be monitored
3. **Liquidity**: XRP/BTC has lower volume than USDT pairs
4. **Correlation Risk**: Both assets may move together vs USD

## Files

- Strategy: `strategies/ratio_trading.py`
- Symbol: `XRP/BTC`
- Version: 1.0.0
