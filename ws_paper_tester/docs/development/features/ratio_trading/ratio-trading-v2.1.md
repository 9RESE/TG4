# Ratio Trading Strategy v2.1 - XRP/BTC Mean Reversion

**Version:** 2.1.0
**Last Updated:** 2025-12-14
**Status:** Production Ready
**Review Reference:** `docs/development/review/ratio_trading/ratio-trading-strategy-review-v2.0.md`

## Overview

The Ratio Trading strategy uses mean reversion on the XRP/BTC pair to accumulate both XRP and BTC over time. It trades based on deviations from a moving average using Bollinger Bands with volatility-adaptive thresholds and comprehensive risk management.

**Important:** This strategy is designed ONLY for crypto-to-crypto ratio pairs (XRP/BTC). It is NOT suitable for USDT-denominated pairs. For USDT pairs, use the `mean_reversion.py` strategy instead.

**WARNING - Trend Continuation Risk:** Bollinger Band touches can signal trend CONTINUATION rather than reversal. Price exceeding the bands may indicate strong momentum, not necessarily a mean reversion opportunity. The volatility regime system and new trend filter help mitigate this risk.

## Version 2.1.0 Changes

Enhancement refactor implementing all recommendations from the v2.0 review:

| Change | Reference | Description |
|--------|-----------|-------------|
| Higher Entry Threshold | REC-013 | Increased from 1.0 to 1.5 std for more selective entries |
| RSI Confirmation | REC-014 | Optional RSI filter for signal quality (now enabled) |
| Trend Detection | REC-015 | Blocks mean reversion signals in strong trends |
| Enhanced Accumulation Metrics | REC-016 | Tracks USD value and trade counts for acquisitions |
| Trailing Stops | Mean Rev Pattern | Locks in profits with trailing stop mechanism |
| Position Decay | Mean Rev Pattern | Forces exit on stale positions near mean |
| Fixed max_losses Bug | Code Fix | Uses config value instead of hardcoded constant |

## Strategy Logic

### Core Concept

- Track the XRP/BTC price ratio over time
- Calculate a moving average (SMA) and standard deviation
- Use Bollinger Bands to identify overextended conditions
- Adjust entry thresholds based on volatility regime
- Filter signals with RSI and trend detection
- Buy XRP when the ratio is low (XRP cheap vs BTC)
- Sell XRP when the ratio is high (XRP expensive vs BTC)
- Protect profits with trailing stops
- Exit stale positions via position decay

### Entry Conditions

**Buy Signal (Accumulate XRP):**
- Z-score < -effective_entry_threshold (now 1.5x regime multiplier)
- RSI < rsi_oversold (if RSI confirmation enabled)
- No strong downtrend detected (if trend filter enabled)
- Spread within acceptable limits
- Available position capacity
- Circuit breaker not active
- Cooldown period elapsed
- (Optional) Trade flow confirms buy pressure

**Sell Signal (Accumulate BTC):**
- Z-score > +effective_entry_threshold (now 1.5x regime multiplier)
- RSI > rsi_overbought (if RSI confirmation enabled)
- No strong uptrend detected (if trend filter enabled)
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

### New Features in v2.1

#### RSI Confirmation Filter (REC-014)

Adds RSI as a secondary confirmation for entries:
- Buy signals require RSI < 35 (oversold)
- Sell signals require RSI > 65 (overbought)
- Prevents entries when momentum doesn't confirm deviation

#### Trend Detection Warning (REC-015)

Detects strong trends to avoid mean reversion failures:
- Analyzes last 10 candles for directional bias
- If 70%+ candles move in same direction = strong trend
- Blocks buy signals in downtrends, sell signals in uptrends
- Prevents catching falling knives

#### Trailing Stops

Locks in profits when position moves favorably:
- Activates at 0.3% profit from entry
- Trails 0.2% behind the highest price reached
- Automatically exits if price retraces to trailing stop

#### Position Decay

Forces exits on positions held too long:
- Starts after 5 minutes of holding
- Triggers partial exit if price is somewhat near mean
- Prevents positions from becoming stale

#### Enhanced Accumulation Metrics (REC-016)

Tracks detailed acquisition statistics:
- USD value at time of XRP acquisition
- USD value at time of BTC acquisition
- Number of trades for each asset type
- Average trade values for analysis

## Configuration

```python
CONFIG = {
    # Core Ratio Trading Parameters - REC-013: Higher entry threshold
    'lookback_periods': 20,           # Periods for moving average
    'bollinger_std': 2.0,             # Standard deviations for bands
    'entry_threshold': 1.5,           # Entry at N std devs from mean (was 1.0)
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

    # RSI Confirmation Filter - REC-014 (NEW in v2.1)
    'use_rsi_confirmation': True,     # Enable RSI filter
    'rsi_period': 14,                 # RSI calculation period
    'rsi_oversold': 35,               # RSI oversold level for buy
    'rsi_overbought': 65,             # RSI overbought level for sell

    # Trend Detection Warning - REC-015 (NEW in v2.1)
    'use_trend_filter': True,         # Enable trend filtering
    'trend_lookback': 10,             # Candles to check for trend
    'trend_strength_threshold': 0.7,  # % of candles = strong trend

    # Trailing Stops (NEW in v2.1)
    'use_trailing_stop': True,        # Enable trailing stops
    'trailing_activation_pct': 0.3,   # Activate at 0.3% profit
    'trailing_distance_pct': 0.2,     # Trail 0.2% from high/low

    # Position Decay (NEW in v2.1)
    'use_position_decay': True,       # Enable position decay
    'position_decay_minutes': 5,      # Start decay after 5 minutes
    'position_decay_tp_mult': 0.5,    # Reduce TP target to 50%

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
| `base_entry_threshold` | Base entry threshold (1.5) |
| `effective_entry_threshold` | Regime-adjusted threshold |
| `position_usd` | Current position in USD |
| `position_xrp` | Current XRP position |
| `xrp_accumulated` | Total XRP accumulated |
| `btc_accumulated` | Total BTC accumulated |
| `xrp_accumulated_value_usd` | USD cost of XRP acquisitions |
| `btc_accumulated_value_usd` | USD value of BTC acquisitions |
| `rsi` | RSI value (v2.1) |
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
| `rsi_not_confirmed` | RSI doesn't confirm signal (v2.1) |
| `strong_trend_detected` | Strong trend blocks mean reversion (v2.1) |
| `position_decayed` | Position exited due to decay (v2.1) |
| `no_signal_conditions` | No entry/exit conditions met |

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
- Initializes all state variables including new v2.1 fields
- Logs startup information including v2.1 feature flags

### on_fill()
- Updates position tracking (USD and XRP)
- Tracks per-pair P&L (REC-006)
- Tracks wins/losses for circuit breaker (uses config, not hardcoded)
- Updates accumulation totals with USD values (REC-016)
- Manages trailing stop price tracking

### on_stop()
- Logs comprehensive summary with enhanced metrics:
  - Total P&L, trades, win rate
  - Per-symbol performance
  - Accumulated XRP and BTC with USD cost basis
  - Trade counts per asset type
  - Rejection summary (top 5 reasons)

## Risk Considerations

1. **Trending Markets**: Mean reversion fails in strong trends (mitigated by trend filter)
2. **Volatility Expansion**: Regime classification helps adapt
3. **Liquidity**: XRP/BTC has lower volume than USDT pairs (wider spreads)
4. **Correlation Risk**: Both assets may move together vs USD
5. **RSI False Signals**: RSI can remain overbought/oversold in trends
6. **Trailing Stop Whipsaws**: Volatile markets may trigger premature exits

## Files

- Strategy: `strategies/ratio_trading.py`
- Symbol: `XRP/BTC`
- Version: 2.1.0
- Review: `docs/development/review/ratio_trading/ratio-trading-strategy-review-v2.0.md`

## Strategy Development Guide Compliance

| Requirement | Status | Evidence |
|-------------|--------|----------|
| STRATEGY_NAME | PASS | `"ratio_trading"` |
| STRATEGY_VERSION | PASS | `"2.1.0"` |
| SYMBOLS list | PASS | `["XRP/BTC"]` |
| CONFIG dict | PASS | 35+ parameters |
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
| RSI confirmation | PASS | REC-014 implemented |
| Trend filter | PASS | REC-015 implemented |
| Enhanced metrics | PASS | REC-016 implemented |
| Trailing stops | PASS | Implemented from mean reversion |
| Position decay | PASS | Implemented from mean reversion |

## Migration from v2.0

No breaking changes. All new features are additive and enabled by default. To match v2.0 behavior exactly:

```python
# Disable v2.1 features in config.yaml
strategy_overrides:
  ratio_trading:
    entry_threshold: 1.0            # Restore original threshold
    use_rsi_confirmation: false
    use_trend_filter: false
    use_trailing_stop: false
    use_position_decay: false
```
