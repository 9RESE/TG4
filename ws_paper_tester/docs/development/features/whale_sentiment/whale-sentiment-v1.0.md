# Whale Sentiment Strategy v1.1.0

**Implementation Date:** December 2025
**Status:** Deep Review v1.0 Implementation Complete
**Research References:** See deep-review-v1.0.md Section 7

## Overview

The Whale Sentiment Strategy combines institutional activity detection (via volume spike analysis) with market sentiment indicators (RSI, price deviation) to identify contrarian trading opportunities. The strategy operates on the principle that extreme market fear or greed, particularly when coupled with large-holder activity, often precedes price reversals.

## Version 1.1.0 Changes (Deep Review v1.0)

| REC ID | Change | Rationale |
|--------|--------|-----------|
| REC-001 | Volume weight 40%, RSI weight 15% | RSI proven ineffective in crypto per academic research |
| REC-003 | Clarified trade flow logic | Intentionally lenient for contrarian mode |
| REC-005 | Enhanced indicator logging | Better debugging on circuit breaker/cooldown paths |
| REC-007 | XRP/BTC disabled by default | 7-10x lower liquidity than USD pairs |
| REC-008 | Short multiplier 0.50x | Reduced from 0.75x for crypto squeeze risk |
| REC-009 | Updated research references | Points to deep-review-v1.0.md |
| REC-010 | UTC timezone documented | Session boundaries are UTC-only |

### Deferred Recommendations

| REC ID | Description | Effort | Documented In |
|--------|-------------|--------|---------------|
| REC-002 | Candle data persistence | Medium | config.py header |
| REC-004 | Volatility regime classification | High | config.py header |
| REC-006 | Backtest confidence weights | High | config.py header |

## Key Features

### 1. Volume Spike Detection (Whale Proxy)
- Volume spikes >= 2x average detected as whale activity (PRIMARY signal per REC-001)
- False positive filtering (price movement, spread, trade count)
- Classification: Accumulation (spike + price up), Distribution (spike + price down), Neutral

### 2. Sentiment Classification
- RSI-based sentiment zones (Extreme Fear < 25, Fear < 40, Greed > 60, Extreme Greed > 75)
- Price deviation from recent high/low as supplementary signal
- Composite sentiment zone classification
- **Note:** RSI weight reduced to 15% per REC-001 due to ineffectiveness in crypto markets

### 3. Contrarian Mode
- Default: Buy fear, sell greed
- Optional: Momentum following mode
- Signal alignment with whale activity direction
- **REC-003:** Trade flow logic is intentionally lenient to allow contrarian entries

### 4. Risk Management
- Stricter circuit breaker (2 consecutive losses, 45 min cooldown)
- Wider stops for counter-trend entries (2.5% default)
- **REC-008:** Lower short exposure (0.50x multiplier, down from 0.75x)
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

### Confidence Weights (REC-001 Updated)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `weight_volume_spike` | **0.40** | Volume spike contribution (increased from 0.30) |
| `weight_rsi_sentiment` | **0.15** | RSI sentiment contribution (reduced from 0.25) |
| `weight_price_deviation` | 0.20 | Price deviation contribution |
| `weight_trade_flow` | 0.15 | Trade flow confirmation |
| `weight_divergence` | 0.10 | RSI divergence bonus |

### Position Sizing

| Parameter | Default | Description |
|-----------|---------|-------------|
| `position_size_usd` | 25.0 | Base position size |
| `max_position_usd` | 150.0 | Total position limit |
| `max_position_per_symbol_usd` | 75.0 | Per-symbol limit |
| `short_size_multiplier` | **0.50** | Reduce short sizes (REC-008: reduced from 0.75) |

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

### XRP/BTC (REC-007: Disabled by Default)
- **Not included in SYMBOLS by default** due to 7-10x lower liquidity
- Configuration retained in SYMBOL_CONFIGS for optional re-enablement
- Highest volume threshold (3x) if enabled
- Smallest position sizes (15 USD)
- Widest stops (3.0%)

## Session Boundaries (REC-010: UTC Only)

All session times are **UTC-only**. No DST adjustment is performed.

| Session | Start (UTC) | End (UTC) | Size Multiplier |
|---------|-------------|-----------|-----------------|
| Asia | 00:00 | 08:00 | 0.8x |
| Europe | 08:00 | 14:00 | 1.0x |
| US-Europe Overlap | 14:00 | 17:00 | 1.1x |
| US | 17:00 | 21:00 | 1.0x |
| Off Hours | 21:00 | 24:00 | 0.5x |

## Signal Generation Flow

1. **Check Circuit Breaker** - 2 consecutive losses triggers 45 min pause
   - **REC-005:** Enhanced logging with elapsed/remaining cooldown times
2. **Check Cooldown** - 120 seconds between signals
   - **REC-005:** Enhanced logging with last signal time
3. **For Each Symbol:**
   - Warmup check (310 candles minimum)
   - Calculate indicators (RSI, volume spike, fear/greed)
   - Check exits for existing positions
   - Classify sentiment regime
   - Validate contrarian opportunity
   - Validate volume spike (false positive filter)
   - Check trade flow confirmation (**REC-003:** lenient for contrarian)
   - Check correlation limits
   - Calculate composite confidence (**REC-001:** volume-weighted)
   - Generate signal if confidence >= 0.55

## Trade Flow Confirmation (REC-003 Clarification)

For **contrarian mode**, trade flow logic is intentionally lenient:

- **BUY signals (in fear):** Accept mild selling pressure (imbalance >= -0.10)
  - Rationale: Contrarian buys occur during panic selling. Requiring positive flow would reject valid entries.

- **SHORT signals (in greed):** Accept mild buying pressure (imbalance <= +0.10)
  - Rationale: Contrarian shorts occur during FOMO buying. Requiring negative flow would reject valid entries.

This differs from momentum strategies which would require flow alignment.

## Confidence Calculation (REC-001 Updated)

Weighted composite of:
- **Volume spike (40%):** Primary signal, higher contribution for stronger spikes
- RSI sentiment (15%): Reduced weight due to crypto ineffectiveness
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

**REC-002 Note:** Candle persistence mechanism is documented for future implementation to reduce warmup impact after restarts.

## Compliance Checklist

| Requirement | Status |
|-------------|--------|
| `STRATEGY_NAME` defined | Yes (`whale_sentiment`) |
| `STRATEGY_VERSION` defined | Yes (`1.1.0`) |
| `SYMBOLS` list defined | Yes (XRP/USDT, BTC/USDT) |
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
| Enhanced indicator logging | Yes (REC-005) |

## Testing Notes

For faster testing, config.yaml override reduces:
- `min_candle_buffer`: 150 (vs 310 default)
- `cooldown_seconds`: 60 (vs 120 default)
- `volume_spike_mult`: 1.8 (vs 2.0 default)

## Future Improvements

### Deferred from Deep Review v1.0:
1. **REC-002:** Candle data persistence for faster warmup
2. **REC-004:** Volatility regime classification
3. **REC-006:** Backtest-validated confidence weights

### Additional Ideas:
4. External whale data integration (Whale Alert API)
5. Social sentiment API integration
6. On-chain metrics integration
7. Adaptive thresholds based on market conditions
8. Machine learning for pattern recognition

---

**Document Version:** 1.1.0
**Author:** Deep Review Implementation
**Platform Version:** WebSocket Paper Tester v1.0.2+
**Review Reference:** deep-review-v1.0.md
