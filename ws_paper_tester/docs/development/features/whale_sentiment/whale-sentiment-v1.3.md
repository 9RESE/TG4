# Whale Sentiment Strategy v1.3.0

**Implementation Date:** December 15, 2025
**Status:** Deep Review v3.0 Implementation Complete
**Research References:** See deep-review-v3.0.md Section 7

## Overview

The Whale Sentiment Strategy combines institutional activity detection (via volume spike analysis) with price deviation sentiment indicators to identify contrarian trading opportunities. The strategy operates on the principle that extreme market fear or greed, particularly when coupled with large-holder activity, often precedes price reversals.

**Key Changes in v1.3.0:**
- RSI COMPLETELY REMOVED from all code paths (zones now use price deviation only)
- ATR-based volatility regime classification added
- Extended fear period detection prevents capital exhaustion
- Dynamic confidence threshold adjusts to market conditions
- BTC/USDT stop loss widened for December 2025 volatility

## Version 1.3.0 Changes (Deep Review v3.0)

| REC ID | Change | Rationale |
|--------|--------|-----------|
| REC-021 | RSI COMPLETELY REMOVED | Academic evidence; consistency with confidence removal |
| REC-022 | BTC/USDT stop loss widened | 2.0% SL / 4.0% TP for Dec 2025 volatility |
| REC-023 | Volatility regime classification | ATR-based parameter adjustments |
| REC-025 | Extended fear period detection | Prevent capital exhaustion in prolonged extremes |
| REC-026 | Short size multiplier increased | 0.60x (from 0.50x) for extreme fear market |
| REC-027 | Dynamic confidence threshold | Adjusts based on sentiment and volatility |

### Deferred Recommendations

| REC ID | Description | Effort | Status |
|--------|-------------|--------|--------|
| REC-024 | Backtest confidence weights | High | Documented for future |

## Key Features

### 1. Volume Spike Detection (PRIMARY Signal - 55% Weight)

Volume spike detection is the PRIMARY signal:
- Volume spikes >= 2x average detected as whale activity
- False positive filtering (price movement, spread, trade count)
- Classification: Accumulation (spike + price up), Distribution (spike + price down), Neutral

**Research Support:** "The Moby Dick Effect" (Magner & Sanhueza, 2025) validates whale contagion effects 6-24 hours after transfers.

### 2. Price Deviation Sentiment (PRIMARY - REC-021)

Price deviation from recent high/low is now the ONLY sentiment signal:
- **Extreme Fear:** Price >= 8% below recent high
- **Fear:** Price >= 5% below recent high
- **Greed:** Price >= 5% above recent low
- **Extreme Greed:** Price >= 8% above recent low
- **Neutral:** Neither fear nor greed conditions met

**Note:** RSI has been COMPLETELY REMOVED from the strategy per REC-021.

### 3. Volatility Regime Classification (REC-023)

ATR-based volatility regime adjusts strategy parameters:

| Regime | ATR % | Size Mult | Stop Mult | Cooldown Mult |
|--------|-------|-----------|-----------|---------------|
| Low | < 1.5% | 1.1x | 0.8x | 0.8x |
| Medium | 1.5-3.5% | 1.0x | 1.0x | 1.0x |
| High | > 3.5% | 0.75x | 1.5x | 1.5x |

### 4. Extended Fear Period Detection (REC-025)

Prevents capital exhaustion during prolonged extreme sentiment:
- **7+ days in extreme zone:** 30% position size reduction
- **14+ days in extreme zone:** Entry pause (exits only)
- Resets when sentiment exits extreme zone

### 5. Dynamic Confidence Threshold (REC-027)

Confidence threshold adjusts based on conditions:
- **Base threshold:** 0.50
- **Extreme sentiment bonus:** -0.05 (easier entry)
- **High volatility penalty:** +0.05 (harder entry)
- **Effective range:** 0.40 - 0.60

### 6. Risk Management

- Stricter circuit breaker (2 consecutive losses, 45 min cooldown)
- Wider stops for counter-trend entries (2.5% default)
- REC-026: Higher short exposure (0.60x multiplier for extreme fear)
- Cross-pair correlation management
- REC-022: BTC/USDT widened to 2.0% SL / 4.0% TP

## Module Structure

```
strategies/whale_sentiment/
    __init__.py          # Public API exports
    config.py            # Metadata, CONFIG, SYMBOL_CONFIGS, enums
    indicators.py        # Volume spike, fear/greed, ATR, composite
    signal.py            # generate_signal, _evaluate_symbol
    regimes.py           # Sentiment + volatility regime classification
    risk.py              # Circuit breaker, position limits, correlation
    exits.py             # Exit signal logic
    lifecycle.py         # on_start, on_fill, on_stop
    validation.py        # Config validation
    persistence.py       # Candle data persistence
```

## Configuration

### Confidence Weights (REC-021 Updated)

| Parameter | v1.2.0 | v1.3.0 | Rationale |
|-----------|--------|--------|-----------|
| `weight_volume_spike` | 0.55 | 0.55 | PRIMARY signal |
| `weight_rsi_sentiment` | 0.00 | **DEPRECATED** | Completely removed |
| `weight_price_deviation` | 0.35 | 0.35 | PRIMARY sentiment |
| `weight_trade_flow` | 0.10 | 0.10 | Confirmation |
| `weight_divergence` | 0.00 | **DEPRECATED** | Removed with RSI |

### Position Sizing

| Parameter | v1.2.0 | v1.3.0 | Description |
|-----------|--------|--------|-------------|
| `position_size_usd` | 25.0 | 25.0 | Base position size |
| `max_position_usd` | 150.0 | 150.0 | Total position limit |
| `max_position_per_symbol_usd` | 75.0 | 75.0 | Per-symbol limit |
| `short_size_multiplier` | 0.50 | **0.60** | REC-026: Increased for extreme fear |

### Price Deviation Thresholds (REC-021)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fear_deviation_pct` | -5.0 | Fear zone threshold |
| `greed_deviation_pct` | 5.0 | Greed zone threshold |
| `extreme_fear_deviation_pct` | -8.0 | Extreme fear threshold |
| `extreme_greed_deviation_pct` | 8.0 | Extreme greed threshold |

### Volatility Regime (REC-023)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `volatility_low_threshold` | 1.5 | ATR% below = low volatility |
| `volatility_high_threshold` | 3.5 | ATR% above = high volatility |
| `volatility_high_size_mult` | 0.75 | Size reduction in high vol |
| `volatility_high_stop_mult` | 1.5 | Stop widening in high vol |
| `volatility_high_cooldown_mult` | 1.5 | Cooldown extension in high vol |

### Extended Fear Period (REC-025)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_extended_fear_detection` | true | Enable feature |
| `extended_fear_threshold_hours` | 168 | 7 days = reduce size |
| `extended_fear_pause_hours` | 336 | 14 days = pause entries |
| `extended_fear_size_reduction` | 0.70 | 30% size reduction |

### Dynamic Confidence (REC-027)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_dynamic_confidence` | true | Enable feature |
| `confidence_extreme_bonus` | -0.05 | Easier in extremes |
| `confidence_high_volatility_penalty` | 0.05 | Harder in high vol |

## Symbol-Specific Configurations

### XRP/USDT
- Standard thresholds (2x volume spike)
- 2.5% stop loss, 5.0% take profit
- Standard price deviation thresholds (-5%/+5% regular, -8%/+8% extreme)
- High suitability for contrarian plays

### BTC/USDT (REC-022 Updated)
- Higher volume threshold (2.5x)
- Larger position sizes (50 USD base)
- **REC-022:** 2.0% stop loss, 4.0% take profit (widened for Dec 2025)
- Higher price deviation thresholds (-7%/+7% regular, -10%/+10% extreme)

### XRP/BTC (Disabled by Default)
- Requires `enable_xrpbtc: true` to activate
- 7-10x lower liquidity than USD pairs
- Highest volume threshold (3x) if enabled
- Largest price deviation thresholds (-8%/+8% regular, -12%/+12% extreme)

## Signal Generation Flow

1. **Check Config Validity** - Validations
2. **Check Circuit Breaker** - 2 consecutive losses triggers 45 min pause
3. **Check Cooldown** - Adjusted by volatility regime (REC-023)
4. **For Each Symbol:**
   - Load persisted candles if available
   - Warmup check with progress indicator
   - Calculate indicators (volume spike, fear/greed, ATR)
   - Classify volatility regime (REC-023)
   - Check extended fear period (REC-025)
   - Check exits for existing positions
   - Classify sentiment regime (REC-021: price deviation only)
   - Check for extended fear pause (REC-025)
   - Validate contrarian opportunity
   - Validate volume spike (false positive filter)
   - Check trade flow confirmation
   - Check correlation limits
   - Calculate composite confidence
   - Check dynamic confidence threshold (REC-027)
   - Apply all size adjustments (volatility, extended fear, sentiment)
   - Generate signal if all conditions met

## New Indicator Fields (v1.3.0)

| Field | Type | Description |
|-------|------|-------------|
| `volatility_regime` | string | 'low', 'medium', 'high', 'unknown' |
| `volatility_size_mult` | float | Size multiplier from volatility |
| `atr_pct` | float | ATR as percentage of price |
| `extended_fear_active` | bool | Is extended fear period active |
| `hours_in_extreme` | float | Hours in extreme sentiment zone |
| `extended_fear_paused` | bool | Are entries paused |
| `min_confidence` | float | Dynamic confidence threshold |
| `confidence_margin` | float | confidence - min_confidence |

## Compliance Checklist

| Requirement | Status |
|-------------|--------|
| `STRATEGY_NAME` defined | Yes (`whale_sentiment`) |
| `STRATEGY_VERSION` defined | Yes (`1.3.0`) |
| `SYMBOLS` list defined | Yes (XRP/USDT, BTC/USDT) |
| `CONFIG` dict defined | Yes (75+ keys) |
| `generate_signal()` function | Yes |
| `on_start()` callback | Yes |
| `on_fill()` callback | Yes |
| `on_stop()` callback | Yes |
| Signal uses USD sizing | Yes |
| R:R >= 1:1 validation | Yes (blocking) |
| Position limit checks | Yes |
| RSI completely removed | Yes (REC-021) |
| Volatility regime | Yes (REC-023) |
| Extended fear detection | Yes (REC-025) |
| Dynamic confidence | Yes (REC-027) |

## Future Improvements

### Deferred from Deep Review v3.0:
1. **REC-024:** Backtest-validated confidence weights

### Additional Ideas:
2. External whale data integration (Whale Alert API)
3. Social sentiment API integration
4. On-chain metrics integration
5. Adaptive thresholds based on market conditions
6. XRP/BTC re-enablement monitoring (golden cross printed)

---

**Document Version:** 1.3.0
**Author:** Deep Review v3.0 Implementation
**Platform Version:** WebSocket Paper Tester v1.0.2+
**Review Reference:** deep-review-v3.0.md
