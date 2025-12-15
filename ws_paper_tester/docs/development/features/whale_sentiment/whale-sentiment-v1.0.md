# Whale Sentiment Strategy v1.0.0

**Implementation Date:** December 2025
**Status:** Initial Release
**Based On:** master-plan-v1.0.md

## Overview

The Whale Sentiment Strategy combines institutional activity detection (via volume spike analysis) with market sentiment indicators (RSI, price deviation) to identify contrarian trading opportunities. The strategy operates on the principle that extreme market fear or greed, particularly when coupled with large-holder activity, often precedes price reversals.

## Key Features

### 1. Volume Spike Detection (Whale Proxy)
- Volume spikes >= 2x average detected as whale activity
- False positive filtering (price movement, spread, trade count)
- Classification: Accumulation (spike + price up), Distribution (spike + price down), Neutral

### 2. Sentiment Classification
- RSI-based sentiment zones (Extreme Fear < 25, Fear < 40, Greed > 60, Extreme Greed > 75)
- Price deviation from recent high/low as supplementary signal
- Composite sentiment zone classification

### 3. Contrarian Mode
- Default: Buy fear, sell greed
- Optional: Momentum following mode
- Signal alignment with whale activity direction

### 4. Risk Management
- Stricter circuit breaker (2 consecutive losses, 45 min cooldown)
- Wider stops for counter-trend entries (2.5% default)
- Lower short exposure limit (75 USD vs 100 USD for longs)
- Cross-pair correlation management

## Module Structure

```
strategies/whale_sentiment/
    __init__.py          # Public API exports
    config.py            # Metadata, CONFIG, SYMBOL_CONFIGS, enums
    indicators.py        # RSI, volume spike, fear/greed, composite
    signal.py            # generate_signal, _evaluate_symbol
    regimes.py           # Sentiment regime classification
    risk.py              # Circuit breaker, position limits, correlation
    exits.py             # Exit signal logic
    lifecycle.py         # on_start, on_fill, on_stop
    validation.py        # Config validation
```

## Configuration

### Core Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `volume_spike_mult` | 2.0 | Volume spike threshold multiplier |
| `volume_window` | 288 | 24h in 5m candles |
| `rsi_period` | 14 | RSI calculation period |
| `rsi_extreme_fear` | 25 | Extreme fear threshold |
| `rsi_extreme_greed` | 75 | Extreme greed threshold |
| `contrarian_mode` | true | Buy fear, sell greed |

### Position Sizing

| Parameter | Default | Description |
|-----------|---------|-------------|
| `position_size_usd` | 25.0 | Base position size |
| `max_position_usd` | 150.0 | Total position limit |
| `max_position_per_symbol_usd` | 75.0 | Per-symbol limit |
| `short_size_multiplier` | 0.75 | Reduce short sizes (squeeze risk) |

### Risk Management

| Parameter | Default | Description |
|-----------|---------|-------------|
| `stop_loss_pct` | 2.5 | Stop loss percentage |
| `take_profit_pct` | 5.0 | Take profit percentage |
| `max_consecutive_losses` | 2 | Circuit breaker trigger |
| `circuit_breaker_minutes` | 45 | Cooldown duration |

## Symbol-Specific Configurations

### XRP/USDT
- Standard thresholds (2x volume spike)
- 2.5% stop loss, 5.0% take profit
- High suitability for contrarian plays

### BTC/USDT
- Higher volume threshold (2.5x)
- More extreme RSI thresholds (22/78)
- Larger position sizes (50 USD base)
- Tighter stops (1.5%)

### XRP/BTC
- Highest volume threshold (3x) due to low liquidity
- Smallest position sizes (15 USD)
- Widest stops (3.0%)

## Signal Generation Flow

1. **Check Circuit Breaker** - 2 consecutive losses triggers 45 min pause
2. **Check Cooldown** - 120 seconds between signals
3. **For Each Symbol:**
   - Warmup check (310 candles minimum)
   - Calculate indicators (RSI, volume spike, fear/greed)
   - Check exits for existing positions
   - Classify sentiment regime
   - Validate contrarian opportunity
   - Validate volume spike (false positive filter)
   - Check trade flow confirmation
   - Check correlation limits
   - Calculate composite confidence
   - Generate signal if confidence >= 0.55

## Entry Conditions

### Long Entry (Contrarian Mode)
- Sentiment zone: FEAR or EXTREME_FEAR
- Whale signal: ACCUMULATION or NEUTRAL
- Trade flow: Not strongly negative (imbalance >= -0.10)
- Confidence >= 0.55

### Short Entry (Contrarian Mode)
- Sentiment zone: GREED or EXTREME_GREED
- Whale signal: DISTRIBUTION or NEUTRAL
- Trade flow: Not strongly positive (imbalance <= 0.10)
- Confidence >= 0.55

## Exit Conditions

1. **Stop Loss** - Price drops below stop level
2. **Take Profit** - Price reaches profit target
3. **Trailing Stop** (Optional) - Activates after 50% of TP
4. **Sentiment Reversal** - Sentiment shifts opposite to entry

## Confidence Calculation

Weighted composite of:
- Volume spike (30%): Higher contribution for stronger spikes
- RSI sentiment (25%): Extreme zones get full weight
- Price deviation (20%): Implicit in sentiment classification
- Trade flow (15%): Confirms market microstructure alignment
- Divergence bonus (10%): RSI divergence adds confirmation

Maximum confidence capped at 0.90 (0.85 for shorts).

## Warmup Requirements

**Minimum Warmup:** 310 candles @ 5 minutes = ~26 hours

Required for:
- Volume spike baseline (288 candles = 24h)
- RSI calculation stability
- Price deviation reference points

## Implementation Notes

### False Positive Filtering
- Volume spike without price movement rejected
- Wide spreads during spikes rejected
- Low trade counts during spikes rejected

### Contrarian Risk Management
- Stricter circuit breaker (catching falling knives)
- Wider stops (counter-trend entries)
- Lower short exposure (squeeze risk)
- Requires volume spike OR moderate sentiment alignment

### Cross-Pair Correlation
- Real-time rolling correlation calculation
- Same-direction trades blocked if correlation > 0.85
- Size reduction for correlated positions

## Compliance Checklist

| Requirement | Status |
|-------------|--------|
| `STRATEGY_NAME` defined | Yes (`whale_sentiment`) |
| `STRATEGY_VERSION` defined | Yes (`1.0.0`) |
| `SYMBOLS` list defined | Yes |
| `CONFIG` dict defined | Yes (56 keys) |
| `generate_signal()` function | Yes |
| `on_start()` callback | Yes |
| `on_fill()` callback | Yes |
| `on_stop()` callback | Yes |
| Signal uses USD sizing | Yes |
| R:R >= 1:1 validation | Yes (blocking) |
| Position limit checks | Yes |
| Minimum trade size check | Yes |
| Data null checks | Yes |
| Circuit breaker | Yes |
| Rejection tracking | Yes |
| Warmup documentation | Yes |

## Testing Notes

For faster testing, config.yaml override reduces:
- `min_candle_buffer`: 150 (vs 310 default)
- `cooldown_seconds`: 60 (vs 120 default)
- `volume_spike_mult`: 1.8 (vs 2.0 default)

## Future Improvements

1. External whale data integration (Whale Alert API)
2. Social sentiment API integration
3. On-chain metrics integration
4. Adaptive thresholds based on market conditions
5. Machine learning for pattern recognition

---

**Document Version:** 1.0.0
**Author:** Implementation Agent
**Platform Version:** WebSocket Paper Tester v1.0.2+
