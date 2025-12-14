# Ratio Trading Strategy v3.0 - XRP/BTC Mean Reversion

**Version:** 3.0.0
**Last Updated:** 2025-12-14
**Status:** Production Ready
**Review Reference:** `docs/development/review/ratio_trading/ratio-trading-strategy-review-v3.1.md`

## Overview

The Ratio Trading strategy uses mean reversion on the XRP/BTC pair to accumulate both XRP and BTC over time. It trades based on deviations from a moving average using Bollinger Bands with volatility-adaptive thresholds and comprehensive risk management.

**Important:** This strategy is designed ONLY for crypto-to-crypto ratio pairs (XRP/BTC). It is NOT suitable for USDT-denominated pairs. For USDT pairs, use the `mean_reversion.py` strategy instead.

**WARNING - Trend Continuation Risk:** Bollinger Band touches can signal trend CONTINUATION rather than reversal. Price exceeding the bands may indicate strong momentum, not necessarily a mean reversion opportunity. The volatility regime system and trend filter help mitigate this risk.

**WARNING - Correlation Stability:** XRP/BTC correlation has been declining (~24.86% over 90 days as of 2025). The strategy includes rolling correlation monitoring to warn when the relationship may be weakening. Consider pausing if correlation falls below historical norms.

## Version 3.0.0 Changes

Enhancement refactor implementing all recommendations from the v3.1 review:

| Change | Reference | Description |
|--------|-----------|-------------|
| Dynamic BTC Price | REC-018 | Uses real-time BTC/USD price for conversions instead of hardcoded $100k |
| On_start Fix | REC-019 | Confirmed correct config value display in startup output |
| Separate Exit Tracking | REC-020 | Exit reasons tracked separately from signal rejections |
| Correlation Monitoring | REC-021 | Rolling correlation with warning and optional pause |
| Hedge Ratio Config | REC-022 | Config placeholders for future hedge ratio optimization |
| ExitReason Enum | v3.0 | New enum for intentional exit categorization |
| Correlation Warning | v3.0 | Warns when XRP/BTC correlation weakens |
| Enhanced Summary | v3.0 | Exit statistics with P&L breakdown in on_stop |

## Strategy Logic

### Core Concept

- Track the XRP/BTC price ratio over time
- Calculate a moving average (SMA) and standard deviation
- Use Bollinger Bands to identify overextended conditions
- Adjust entry thresholds based on volatility regime
- Monitor rolling correlation for relationship stability
- Filter signals with RSI and trend detection
- Buy XRP when the ratio is low (XRP cheap vs BTC)
- Sell XRP when the ratio is high (XRP expensive vs BTC)
- Protect profits with trailing stops
- Exit stale positions via position decay

### Entry Conditions

**Buy Signal (Accumulate XRP):**
- Z-score < -effective_entry_threshold (1.5x regime multiplier)
- RSI < rsi_oversold (if RSI confirmation enabled)
- No strong downtrend detected (if trend filter enabled)
- Correlation above pause threshold (if correlation pause enabled)
- Spread within acceptable limits
- Available position capacity
- Circuit breaker not active
- Cooldown period elapsed
- (Optional) Trade flow confirms buy pressure

**Sell Signal (Accumulate BTC):**
- Z-score > +effective_entry_threshold (1.5x regime multiplier)
- RSI > rsi_overbought (if RSI confirmation enabled)
- No strong uptrend detected (if trend filter enabled)
- Correlation above pause threshold (if correlation pause enabled)
- Spread within acceptable limits
- XRP position or holdings available to sell
- Circuit breaker not active
- Cooldown period elapsed
- (Optional) Trade flow confirms sell pressure

### Exit Conditions

- **Take profit:** Price moves 0.6% in favorable direction
- **Stop loss:** Price moves 0.6% against position
- **Trailing stop:** Activated at 0.3% profit, trails 0.2% from high
- **Position decay:** Partial exit after 5 minutes if near mean
- **Mean reversion exit:** Partial exit when |z-score| < exit_threshold

### New Features in v3.0

#### Dynamic BTC Price for USD Conversion (REC-018)

Replaces hardcoded $100,000 BTC price with real-time data:
- Checks market data for BTC/USDT or BTC/USD prices
- Falls back to configurable default if unavailable
- Used in on_fill for accurate XRP/BTC/USD conversions
- Tracked in state for consistency across calls

#### Separate Exit Tracking (REC-020)

Intentional exits are now tracked separately from signal rejections:
- New `ExitReason` enum: TRAILING_STOP, POSITION_DECAY, TAKE_PROFIT, STOP_LOSS, MEAN_REVERSION, CORRELATION_EXIT
- `_track_exit()` function for exit tracking
- P&L breakdown by exit reason in summary
- Exit counts by symbol for detailed analysis

#### Rolling Correlation Monitoring (REC-021)

Monitors XRP/BTC price correlation to detect relationship changes:
- Calculates Pearson correlation over configurable lookback
- Warning threshold (default 0.5) triggers correlation warnings
- Pause threshold (default 0.3) can optionally stop trading
- Correlation history stored for analysis
- Warns when XRP may be decoupling from BTC

## Configuration

```python
CONFIG = {
    # Core Ratio Trading Parameters - REC-013: Higher entry threshold
    'lookback_periods': 20,           # Periods for moving average
    'bollinger_std': 2.0,             # Standard deviations for bands
    'entry_threshold': 1.5,           # Entry at N std devs from mean
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

    # RSI Confirmation Filter - REC-014
    'use_rsi_confirmation': True,     # Enable RSI filter
    'rsi_period': 14,                 # RSI calculation period
    'rsi_oversold': 35,               # RSI oversold level for buy
    'rsi_overbought': 65,             # RSI overbought level for sell

    # Trend Detection Warning - REC-015
    'use_trend_filter': True,         # Enable trend filtering
    'trend_lookback': 10,             # Candles to check for trend
    'trend_strength_threshold': 0.7,  # % of candles = strong trend

    # Trailing Stops
    'use_trailing_stop': True,        # Enable trailing stops
    'trailing_activation_pct': 0.3,   # Activate at 0.3% profit
    'trailing_distance_pct': 0.2,     # Trail 0.2% from high/low

    # Position Decay
    'use_position_decay': True,       # Enable position decay
    'position_decay_minutes': 5,      # Start decay after 5 minutes
    'position_decay_tp_mult': 0.5,    # Reduce TP target to 50%

    # Rejection Tracking
    'track_rejections': True,         # Enable rejection tracking

    # Correlation Monitoring - REC-021 (NEW in v3.0)
    'use_correlation_monitoring': True,   # Enable correlation monitoring
    'correlation_lookback': 20,           # Periods for correlation calculation
    'correlation_warning_threshold': 0.5, # Warn if correlation below this
    'correlation_pause_threshold': 0.3,   # Pause trading if below this
    'correlation_pause_enabled': False,   # Whether to pause on low correlation

    # Dynamic BTC Price - REC-018 (NEW in v3.0)
    'btc_price_fallback': 100000.0,       # Fallback BTC/USD price if unavailable
    'btc_price_symbols': ['BTC/USDT', 'BTC/USD'],  # Symbols to check for BTC price

    # Hedge Ratio - REC-022 (Future Enhancement)
    'use_hedge_ratio': False,         # Enable hedge ratio optimization
    'hedge_ratio_lookback': 50,       # Periods for hedge ratio calculation
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
| `base_entry_threshold` | Base entry threshold (1.5) |
| `effective_entry_threshold` | Regime-adjusted threshold |
| `position_usd` | Current position in USD |
| `position_xrp` | Current XRP position |
| `xrp_accumulated` | Total XRP accumulated |
| `btc_accumulated` | Total BTC accumulated |
| `xrp_accumulated_value_usd` | USD cost of XRP acquisitions |
| `btc_accumulated_value_usd` | USD value of BTC acquisitions |
| `rsi` | RSI value |
| `use_rsi_confirmation` | RSI filter enabled status |
| `is_strong_trend` | Trend detection result |
| `trend_direction` | Trend direction (up/down/neutral) |
| `trend_strength` | Trend strength (0-1) |
| `position_minutes_held` | Time position held (if decay enabled) |
| `position_decayed` | Position decay status |
| `current_spread_pct` | Current spread percentage |
| `trade_flow` | Trade flow imbalance |
| `consecutive_losses` | Current consecutive loss count |
| `pnl_symbol` | Per-symbol P&L |
| `trades_symbol` | Per-symbol trade count |
| `btc_price_usd` | Current BTC/USD price (v3.0) |
| `correlation` | Rolling XRP/BTC correlation (v3.0) |
| `use_correlation_monitoring` | Correlation monitoring enabled (v3.0) |
| `correlation_warnings` | Count of low correlation warnings (v3.0) |

## Exit Tracking (REC-020)

The strategy now tracks intentional exits separately from rejections:

| Exit Reason | Description |
|-------------|-------------|
| `trailing_stop` | Exit triggered by trailing stop |
| `position_decay` | Exit triggered by position decay |
| `take_profit` | Exit at take profit target |
| `stop_loss` | Exit at stop loss level |
| `mean_reversion` | Exit when z-score returns to threshold |
| `correlation_exit` | Exit due to low correlation |

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
| `rsi_not_confirmed` | RSI doesn't confirm signal |
| `strong_trend_detected` | Strong trend blocks mean reversion |
| `no_signal_conditions` | No entry/exit conditions met |
| `correlation_too_low` | Correlation below pause threshold (v3.0) |

## Enhanced Accumulation Tracking (REC-016)

The strategy tracks comprehensive dual-asset accumulation:

| Metric | Description |
|--------|-------------|
| `xrp_accumulated` | Total XRP bought through strategy |
| `btc_accumulated` | Total BTC received from selling XRP |
| `xrp_accumulated_value_usd` | USD cost basis of XRP purchases |
| `btc_accumulated_value_usd` | USD value of BTC acquisitions |
| `total_trades_xrp_bought` | Number of XRP purchase trades |
| `total_trades_btc_bought` | Number of BTC acquisition trades |
| `avg_xrp_buy_value_usd` | Average USD per XRP trade |
| `avg_btc_buy_value_usd` | Average USD per BTC trade |

## Lifecycle Callbacks

### on_start()
- Validates configuration (REC-007)
- Stores config values in state for on_fill access
- Initializes all state variables including v3.0 fields
- Logs startup information including v3.0 feature flags
- Displays correlation monitoring thresholds

### on_fill()
- Updates position tracking (USD and XRP)
- Uses dynamic BTC price for accurate conversions (REC-018)
- Tracks per-pair P&L (REC-006)
- Tracks wins/losses for circuit breaker
- Updates accumulation totals with USD values (REC-016)
- Manages trailing stop price tracking

### on_stop()
- Logs comprehensive summary with enhanced metrics:
  - Total P&L, trades, win rate
  - Per-symbol performance
  - Accumulated XRP and BTC with USD cost basis
  - Trade counts per asset type
  - Exit statistics with P&L by reason (REC-020)
  - Correlation monitoring summary (REC-021)
  - Rejection summary (top 5 reasons)

## Risk Considerations

1. **Trending Markets**: Mean reversion fails in strong trends (mitigated by trend filter)
2. **Volatility Expansion**: Regime classification helps adapt
3. **Liquidity**: XRP/BTC has lower volume than USDT pairs (wider spreads)
4. **Correlation Risk**: Both assets may move together vs USD (now monitored)
5. **RSI False Signals**: RSI can remain overbought/oversold in trends
6. **Trailing Stop Whipsaws**: Volatile markets may trigger premature exits
7. **Correlation Breakdown**: XRP may decouple from BTC (new warning system)

## Files

- Strategy: `strategies/ratio_trading.py`
- Symbol: `XRP/BTC`
- Version: 3.0.0
- Review: `docs/development/review/ratio_trading/ratio-trading-strategy-review-v3.1.md`

## Strategy Development Guide Compliance

| Requirement | Status | Evidence |
|-------------|--------|----------|
| STRATEGY_NAME | PASS | `"ratio_trading"` |
| STRATEGY_VERSION | PASS | `"3.0.0"` |
| SYMBOLS list | PASS | `["XRP/BTC"]` |
| CONFIG dict | PASS | 40+ parameters |
| generate_signal() | PASS | Correct signature |
| Size in USD | PASS | `position_size_usd` (REC-002) |
| Stop loss below entry (buy) | PASS | Correct calculation |
| Stop loss above entry (sell) | PASS | Correct calculation |
| R:R ratio >= 1:1 | PASS | 0.6%/0.6% = 1:1 (REC-003) |
| Signal metadata | PASS | Includes strategy, type, z-score, regime, exit_reason |
| on_start() | PASS | Config validation, state init, feature logging |
| on_fill() | PASS | Position tracking, per-pair PnL, dynamic BTC price |
| on_stop() | PASS | Comprehensive summary with exit statistics |
| Per-pair PnL | PASS | REC-006 implemented |
| Config validation | PASS | REC-007 implemented |
| Volatility regimes | PASS | REC-004 implemented |
| Circuit breaker | PASS | REC-005 implemented |
| RSI confirmation | PASS | REC-014 implemented |
| Trend filter | PASS | REC-015 implemented |
| Enhanced metrics | PASS | REC-016 implemented |
| Trailing stops | PASS | Implemented |
| Position decay | PASS | Implemented |
| Dynamic BTC price | PASS | REC-018 implemented |
| Separate exit tracking | PASS | REC-020 implemented |
| Correlation monitoring | PASS | REC-021 implemented |

## Migration from v2.1

No breaking changes. All new features are additive. To match v2.1 behavior exactly:

```python
# Disable v3.0 features in config.yaml
strategy_overrides:
  ratio_trading:
    use_correlation_monitoring: false
    correlation_pause_enabled: false
```

To enable correlation-based trading pause:

```python
# Enable conservative correlation protection
strategy_overrides:
  ratio_trading:
    correlation_pause_enabled: true
    correlation_pause_threshold: 0.3
```
