# Ratio Trading Strategy v2.0 - XRP/BTC Mean Reversion

**Version:** 2.0.0
**Last Updated:** 2025-12-14
**Status:** Production Ready
**Review Reference:** `docs/development/review/ratio_trading/ratio-trading-strategy-review-v1.0.md`

## Overview

The Ratio Trading strategy uses mean reversion on the XRP/BTC pair to accumulate both XRP and BTC over time. It trades based on deviations from a moving average using Bollinger Bands with volatility-adaptive thresholds and comprehensive risk management.

**Important:** This strategy is designed ONLY for crypto-to-crypto ratio pairs (XRP/BTC). It is NOT suitable for USDT-denominated pairs. For USDT pairs, use the `mean_reversion.py` strategy instead.

## Version 2.0.0 Changes

Major refactor implementing all recommendations from the strategy review:

| Change | Reference | Description |
|--------|-----------|-------------|
| USD Position Sizing | REC-002 | Converted from XRP to USD-based sizing for platform compliance |
| 1:1 R:R Ratio | REC-003 | Fixed to 0.6%/0.6% (was 0.5%/0.6%) |
| Volatility Regimes | REC-004 | Adaptive thresholds based on market volatility |
| Circuit Breaker | REC-005 | Protection against consecutive losses |
| Per-Pair PnL | REC-006 | Comprehensive trade tracking per symbol |
| Config Validation | REC-007 | Startup configuration validation |
| Spread Monitoring | REC-008 | Filters trades when spreads are too wide |
| Trade Flow | REC-010 | Optional market microstructure confirmation |
| Code Refactor | - | Modular functions, rejection tracking, metadata |

## Strategy Logic

### Core Concept

- Track the XRP/BTC price ratio over time
- Calculate a moving average (SMA) and standard deviation
- Use Bollinger Bands to identify overextended conditions
- Adjust entry thresholds based on volatility regime
- Buy XRP when the ratio is low (XRP cheap vs BTC)
- Sell XRP when the ratio is high (XRP expensive vs BTC)
- Target price-based take profit for mean reversion

### Entry Conditions

**Buy Signal (Accumulate XRP):**
- Z-score < -effective_entry_threshold (adjusted for regime)
- Spread within acceptable limits
- Available position capacity
- Circuit breaker not active
- Cooldown period elapsed
- (Optional) Trade flow confirms buy pressure

**Sell Signal (Accumulate BTC):**
- Z-score > +effective_entry_threshold (adjusted for regime)
- Spread within acceptable limits
- XRP position or holdings available to sell
- Circuit breaker not active
- Cooldown period elapsed
- (Optional) Trade flow confirms sell pressure

### Exit Conditions

- Take profit: Price moves 0.6% in favorable direction
- Stop loss: Price moves 0.6% against position
- Mean reversion exit: Partial exit when |z-score| < exit_threshold

### Volatility Regimes

| Regime | Volatility Range | Threshold Mult | Size Mult | Notes |
|--------|------------------|----------------|-----------|-------|
| LOW | < 0.2% | 0.8x | 1.0x | Tighter entry thresholds |
| MEDIUM | 0.2% - 0.5% | 1.0x | 1.0x | Standard parameters |
| HIGH | 0.5% - 1.0% | 1.3x | 0.8x | Wider thresholds, smaller size |
| EXTREME | > 1.0% | - | - | Trading paused (configurable) |

## Configuration

```python
CONFIG = {
    # Core Ratio Trading Parameters
    'lookback_periods': 20,           # Periods for moving average
    'bollinger_std': 2.0,             # Standard deviations for bands
    'entry_threshold': 1.0,           # Entry at N std devs from mean
    'exit_threshold': 0.5,            # Exit at N std devs (closer to mean)

    # Position Sizing - USD-based (REC-002)
    'position_size_usd': 15.0,        # Base size per trade in USD
    'max_position_usd': 50.0,         # Maximum position exposure in USD
    'min_trade_size_usd': 5.0,        # Minimum USD per trade

    # Risk Management - 1:1 R:R (REC-003)
    'stop_loss_pct': 0.6,             # Stop loss percentage
    'take_profit_pct': 0.6,           # Take profit percentage

    # Cooldown Mechanisms
    'cooldown_seconds': 30.0,         # Minimum time between trades
    'min_candles': 10,                # Minimum candles before trading

    # Volatility Parameters (REC-004)
    'use_volatility_regimes': True,   # Enable regime-based adjustments
    'volatility_lookback': 20,        # Candles for volatility calculation
    'regime_low_threshold': 0.2,      # Below = LOW regime
    'regime_medium_threshold': 0.5,   # Below = MEDIUM regime
    'regime_high_threshold': 1.0,     # Below = HIGH regime, above = EXTREME
    'regime_extreme_pause': True,     # Pause trading in EXTREME regime

    # Circuit Breaker (REC-005)
    'use_circuit_breaker': True,      # Enable consecutive loss protection
    'max_consecutive_losses': 3,      # Max losses before cooldown
    'circuit_breaker_minutes': 15,    # Cooldown after max losses

    # Spread Monitoring (REC-008)
    'use_spread_filter': True,        # Enable spread filtering
    'max_spread_pct': 0.10,           # Max spread % for XRP/BTC
    'min_profitability_mult': 0.5,    # TP must exceed spread * this

    # Trade Flow Confirmation (REC-010)
    'use_trade_flow_confirmation': False,  # Disabled by default
    'trade_flow_threshold': 0.10,          # Minimum trade flow alignment

    # Rejection Tracking
    'track_rejections': True,         # Enable rejection tracking
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
| `volatility_pct` | Current volatility percentage |
| `volatility_regime` | Current regime (LOW/MEDIUM/HIGH/EXTREME) |
| `regime_threshold_mult` | Applied threshold multiplier |
| `regime_size_mult` | Applied size multiplier |
| `base_entry_threshold` | Base entry threshold |
| `effective_entry_threshold` | Regime-adjusted threshold |
| `position_usd` | Current position in USD |
| `position_xrp` | Current XRP position (for reference) |
| `xrp_accumulated` | Total XRP accumulated |
| `btc_accumulated` | Total BTC accumulated |
| `current_spread_pct` | Current spread percentage |
| `trade_flow` | Trade flow imbalance |
| `consecutive_losses` | Current consecutive loss count |
| `pnl_symbol` | Per-symbol P&L |
| `trades_symbol` | Per-symbol trade count |

## Rejection Tracking

The strategy tracks all signal rejections for analysis:

| Rejection Reason | Description |
|------------------|-------------|
| `circuit_breaker` | Trading paused due to consecutive losses |
| `time_cooldown` | Within cooldown period between trades |
| `warming_up` | Insufficient candle history |
| `regime_pause` | Paused due to EXTREME volatility regime |
| `no_price_data` | No price available for symbol |
| `max_position` | Maximum position size reached |
| `insufficient_size` | Calculated size below minimum |
| `trade_flow_not_aligned` | Trade flow doesn't confirm signal |
| `spread_too_wide` | Current spread exceeds maximum |
| `no_signal_conditions` | No entry/exit conditions met |

## Accumulation Tracking

The strategy tracks dual-asset accumulation (unique to ratio trading):

- **XRP Accumulated**: Total XRP bought through the strategy
- **BTC Accumulated**: Total BTC received from selling XRP

This allows measuring progress toward the goal of growing both holdings.

## Signal Metadata

Signals include metadata for enhanced logging:

```python
metadata={
    'strategy': 'ratio_trading',
    'signal_type': 'entry' | 'exit',
    'z_score': float,
    'regime': 'LOW' | 'MEDIUM' | 'HIGH',
}
```

## Lifecycle Callbacks

### on_start()
- Validates configuration (REC-007)
- Initializes all state variables
- Logs startup information including feature flags and R:R ratio

### on_fill()
- Updates position tracking (USD and XRP)
- Tracks per-pair P&L (REC-006)
- Tracks wins/losses for circuit breaker (REC-005)
- Updates accumulation totals (XRP and BTC)

### on_stop()
- Logs comprehensive summary:
  - Total P&L, trades, win rate
  - Per-symbol performance
  - Accumulated XRP and BTC
  - Rejection summary (top 5 reasons)

## Risk Considerations

1. **Trending Markets**: Mean reversion fails in strong trends
2. **Volatility Expansion**: Regime classification helps adapt
3. **Liquidity**: XRP/BTC has lower volume than USDT pairs (wider spreads)
4. **Correlation Risk**: Both assets may move together vs USD
5. **Cointegration**: Strategy assumes mean reversion without testing cointegration

## Comparison with Other Strategies

| Strategy | Approach | Position Sizing | Pairs | Best For |
|----------|----------|-----------------|-------|----------|
| **Ratio Trading** | Mean reversion | USD | XRP/BTC | Balanced accumulation, range-bound |
| Mean Reversion | Mean reversion | USD | XRP/USDT, BTC/USDT | USD growth |
| Market Making | Spread capture | USD | All | Frequent small profits |
| Order Flow | Momentum | USD | All | Trending markets |

## Example Trade Flow

1. XRP/BTC at 0.0000218 (z=-1.2, below effective threshold in MEDIUM regime)
2. Spread check passes (0.04% < 0.10%)
3. Strategy generates BUY signal for $15 USD worth
4. Circuit breaker not active, cooldown elapsed
5. Signal executed: Buy XRP with BTC
6. XRP/BTC rises to 0.0000231 (+0.6%)
7. Take profit triggered automatically
8. Result: Accumulated XRP during dip, profit realized

## Files

- Strategy: `strategies/ratio_trading.py`
- Symbol: `XRP/BTC`
- Version: 2.0.0
- Review: `docs/development/review/ratio_trading/ratio-trading-strategy-review-v1.0.md`

## Strategy Development Guide Compliance

| Requirement | Status | Evidence |
|-------------|--------|----------|
| STRATEGY_NAME | PASS | `"ratio_trading"` |
| STRATEGY_VERSION | PASS | `"2.0.0"` |
| SYMBOLS list | PASS | `["XRP/BTC"]` |
| CONFIG dict | PASS | 25+ parameters |
| generate_signal() | PASS | Correct signature |
| Size in USD | PASS | `position_size_usd` (REC-002) |
| Stop loss below entry (buy) | PASS | Correct calculation |
| Stop loss above entry (sell) | PASS | Correct calculation |
| R:R ratio >= 1:1 | PASS | 0.6%/0.6% = 1:1 (REC-003) |
| Signal metadata | PASS | Includes strategy, type, z-score, regime |
| on_start() | PASS | Config validation, state init |
| on_fill() | PASS | Position tracking, per-pair PnL |
| on_stop() | PASS | Comprehensive summary |
| Per-pair PnL | PASS | REC-006 implemented |
| Config validation | PASS | REC-007 implemented |
| Volatility regimes | PASS | REC-004 implemented |
| Circuit breaker | PASS | REC-005 implemented |
