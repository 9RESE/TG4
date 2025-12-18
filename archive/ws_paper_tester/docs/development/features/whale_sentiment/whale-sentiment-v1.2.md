# Whale Sentiment Strategy v1.2.0

**Implementation Date:** December 2025
**Status:** Deep Review v2.0 Implementation Complete
**Research References:** See deep-review-v2.0.md Section 7

## Overview

The Whale Sentiment Strategy combines institutional activity detection (via volume spike analysis) with price deviation sentiment indicators to identify contrarian trading opportunities. The strategy operates on the principle that extreme market fear or greed, particularly when coupled with large-holder activity, often precedes price reversals.

**Key Change in v1.2.0:** RSI has been completely removed from confidence calculation based on academic research showing RSI ineffectiveness in cryptocurrency markets.

## Version 1.2.0 Changes (Deep Review v2.0)

| REC ID | Change | Rationale |
|--------|--------|-----------|
| REC-011 | Candle data persistence | Eliminates 25+ hour warmup after restarts |
| REC-012 | Warmup progress indicator | Shows completion %, ETA during warmup |
| REC-013 | RSI REMOVED from confidence | Academic evidence of ineffectiveness in crypto |
| REC-016 | XRP/BTC re-enablement guard | Explicit flag required to trade low-liquidity pair |
| REC-017 | UTC timezone validation | Warns if server not in UTC for session accuracy |
| REC-018 | Trade flow expected indicator | Clarifies contrarian flow expectations |
| REC-019 | Per-symbol volume window | Configurable volume baseline per pair |
| REC-020 | Extracted magic numbers | Configurable confidence calculation parameters |

### Deferred Recommendations

| REC ID | Description | Effort | Status |
|--------|-------------|--------|--------|
| REC-014 | Volatility regime classification | High | Documented for future |
| REC-015 | Backtest confidence weights | High | Documented for future |

## Key Features

### 1. Volume Spike Detection (PRIMARY Signal - 55% Weight)

Per REC-013, volume spike detection is now the PRIMARY signal:
- Volume spikes >= 2x average detected as whale activity
- False positive filtering (price movement, spread, trade count)
- Classification: Accumulation (spike + price up), Distribution (spike + price down), Neutral

**Research Support:** "The Moby Dick Effect" (Magner & Sanhueza, 2025) validates whale contagion effects on crypto returns.

### 2. Price Deviation Sentiment (35% Weight)

Price deviation from recent high/low now serves as the PRIMARY sentiment signal:
- Fear: Price significantly below recent high
- Greed: Price significantly above recent low
- Composite sentiment zone classification

**Note:** RSI has been REMOVED from sentiment calculation per REC-013 due to academic evidence of ineffectiveness in crypto markets.

### 3. Trade Flow Confirmation (10% Weight)

Validates signal with market microstructure:
- REC-003: Trade flow logic is intentionally lenient for contrarian mode
- REC-018: New indicators show expected flow direction

### 4. Candle Data Persistence (REC-011)

New feature eliminates warmup delay:
- Saves candle buffer to disk every new candle
- Reloads on startup if data is fresh (< 4 hours old)
- Location: `data/candles/{symbol}_5m.json`
- Graceful fallback to fresh warmup if data corrupted

### 5. Risk Management

- Stricter circuit breaker (2 consecutive losses, 45 min cooldown)
- Wider stops for counter-trend entries (2.5% default)
- REC-008: Lower short exposure (0.50x multiplier)
- Cross-pair correlation management

## Module Structure

```
strategies/whale_sentiment/
    __init__.py          # Public API exports
    config.py            # Metadata, CONFIG, SYMBOL_CONFIGS, enums
    indicators.py        # Volume spike, fear/greed, composite
    signal.py            # generate_signal, _evaluate_symbol
    regimes.py           # Sentiment regime classification
    risk.py              # Circuit breaker, position limits, correlation
    exits.py             # Exit signal logic
    lifecycle.py         # on_start, on_fill, on_stop
    validation.py        # Config validation
    persistence.py       # NEW: Candle data persistence (REC-011)
```

## Configuration

### Confidence Weights (REC-013 Updated)

| Parameter | v1.1.0 | v1.2.0 | Rationale |
|-----------|--------|--------|-----------|
| `weight_volume_spike` | 0.40 | **0.55** | PRIMARY signal |
| `weight_rsi_sentiment` | 0.15 | **0.00** | REMOVED (ineffective) |
| `weight_price_deviation` | 0.20 | **0.35** | PRIMARY sentiment |
| `weight_trade_flow` | 0.15 | **0.10** | Reduced contribution |
| `weight_divergence` | 0.10 | **0.00** | Removed with RSI |

### Position Sizing

| Parameter | Default | Description |
|-----------|---------|-------------|
| `position_size_usd` | 25.0 | Base position size |
| `max_position_usd` | 150.0 | Total position limit |
| `max_position_per_symbol_usd` | 75.0 | Per-symbol limit |
| `short_size_multiplier` | 0.50 | Reduce short sizes (REC-008) |

### Risk Management

| Parameter | Default | Description |
|-----------|---------|-------------|
| `stop_loss_pct` | 2.5 | Stop loss percentage |
| `take_profit_pct` | 5.0 | Take profit percentage |
| `max_consecutive_losses` | 2 | Circuit breaker trigger |
| `circuit_breaker_minutes` | 45 | Cooldown duration |

### Candle Persistence (REC-011)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_candle_persistence` | true | Enable persistence |
| `candle_persistence_dir` | data/candles | Save directory |
| `max_candle_age_hours` | 4.0 | Max age for reload |
| `candle_save_interval_candles` | 1 | Save every N candles |

### XRP/BTC Guard (REC-016)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_xrpbtc` | false | Must be true to trade XRP/BTC |

### Timezone Validation (REC-017)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `require_utc_timezone` | true | Check server timezone |
| `timezone_warning_only` | true | Warn vs block if not UTC |

## Symbol-Specific Configurations

### XRP/USDT
- Standard thresholds (2x volume spike)
- 2.5% stop loss, 5.0% take profit
- High suitability for contrarian plays

### BTC/USDT
- Higher volume threshold (2.5x)
- Larger position sizes (50 USD base)
- Tighter stops (1.5%) - more predictable moves

### XRP/BTC (Disabled by Default)
- **REC-016:** Requires `enable_xrpbtc: true` to activate
- 7-10x lower liquidity than USD pairs
- Highest volume threshold (3x) if enabled

## Session Boundaries (UTC Only)

All session times are **UTC-only**. REC-017 validates timezone on startup.

| Session | Start (UTC) | End (UTC) | Size Multiplier |
|---------|-------------|-----------|-----------------|
| Asia | 00:00 | 08:00 | 0.8x |
| Europe | 08:00 | 14:00 | 1.0x |
| US-Europe Overlap | 14:00 | 17:00 | 1.1x |
| US | 17:00 | 21:00 | 1.0x |
| Off Hours | 21:00 | 24:00 | 0.5x |

## Signal Generation Flow

1. **Check Config Validity** - REC-016/017 validations
2. **Check Circuit Breaker** - 2 consecutive losses triggers 45 min pause
3. **Check Cooldown** - 120 seconds between signals
4. **For Each Symbol:**
   - Load persisted candles if available (REC-011)
   - Warmup check with progress indicator (REC-012)
   - Calculate indicators (volume spike, fear/greed - NO RSI per REC-013)
   - Check exits for existing positions
   - Classify sentiment regime
   - Validate contrarian opportunity
   - Validate volume spike (false positive filter)
   - Check trade flow confirmation (REC-018: shows expected flow)
   - Check correlation limits
   - Calculate composite confidence (REC-013: volume-weighted)
   - Generate signal if confidence >= 0.50

## Trade Flow Confirmation (REC-003/REC-018)

For **contrarian mode**, trade flow logic is intentionally lenient:

- **BUY signals (in fear):** Accept mild selling pressure (imbalance >= -0.10)
- **SHORT signals (in greed):** Accept mild buying pressure (imbalance <= +0.10)

REC-018 adds indicators showing:
- `trade_flow_expected`: What flow direction is expected ('positive' or 'negative')
- `trade_flow_mode`: 'contrarian' or 'momentum'

## Warmup Requirements

### With Persistence (REC-011)
- Reload time: ~1-2 seconds
- Data must be < 4 hours old
- Falls back to fresh warmup if file missing/corrupted

### Fresh Warmup (Fallback)
- **Minimum Warmup:** 310 candles @ 5 minutes = ~26 hours
- **REC-012:** Progress indicator shows:
  - `warmup_pct`: Completion percentage
  - `warmup_eta_hours`: Estimated time remaining

## Compliance Checklist

| Requirement | Status |
|-------------|--------|
| `STRATEGY_NAME` defined | Yes (`whale_sentiment`) |
| `STRATEGY_VERSION` defined | Yes (`1.2.0`) |
| `SYMBOLS` list defined | Yes (XRP/USDT, BTC/USDT) |
| `CONFIG` dict defined | Yes (65+ keys) |
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
| Enhanced indicator logging | Yes |
| Candle persistence | Yes (REC-011) |
| Warmup progress | Yes (REC-012) |
| RSI removed | Yes (REC-013) |
| XRP/BTC guard | Yes (REC-016) |
| Timezone validation | Yes (REC-017) |

## Future Improvements

### Deferred from Deep Review v2.0:
1. **REC-014:** Volatility regime classification (ATR-based)
2. **REC-015:** Backtest-validated confidence weights

### Additional Ideas:
3. External whale data integration (Whale Alert API)
4. Social sentiment API integration
5. On-chain metrics integration
6. Adaptive thresholds based on market conditions

---

**Document Version:** 1.2.0
**Author:** Deep Review v2.0 Implementation
**Platform Version:** WebSocket Paper Tester v1.0.2+
**Review Reference:** deep-review-v2.0.md
