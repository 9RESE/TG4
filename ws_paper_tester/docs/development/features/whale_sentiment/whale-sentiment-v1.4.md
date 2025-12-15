# Whale Sentiment Strategy v1.4.0

**Implementation Date:** December 15, 2025
**Status:** Deep Review v4.0 Implementation Complete
**Research References:** See deep-review-v4.0.md Section 7

## Overview

The Whale Sentiment Strategy combines institutional activity detection (via volume spike analysis) with price deviation sentiment indicators to identify contrarian trading opportunities. The strategy operates on the principle that extreme market fear or greed, particularly when coupled with large-holder activity, often precedes price reversals.

**Key Changes in v1.4.0:**
- CRITICAL bug fix: Undefined function reference in signal metadata (REC-030)
- EXTREME volatility regime added with trading pause (REC-031)
- Deprecated RSI code removed per clean code principles (REC-032)
- Scope and Limitations documentation added (REC-033)

## Version 1.4.0 Changes (Deep Review v4.0)

| REC ID | Priority | Change | Rationale |
|--------|----------|--------|-----------|
| REC-030 | CRITICAL | Fixed `_classify_volatility_regime` reference | Bug: function did not exist |
| REC-031 | MEDIUM | Added EXTREME volatility regime (ATR > 6%) | Guide v2.0 Section 15 compliance |
| REC-032 | MEDIUM | Removed deprecated RSI code | Clean code principles |
| REC-033 | LOW | Added scope documentation | Guide v2.0 Section 26 compliance |

### Compliance Score Improvement

| Version | Compliance | Notes |
|---------|------------|-------|
| v1.3.0 | 89% | 8 of 9 requirements |
| v1.4.0 | 100% | All 9 requirements met |

## Scope and Limitations (REC-033)

### Strategy Scope

**Intended Use:**
- Contrarian trading during extreme market sentiment periods
- Detection of institutional activity via volume spike analysis
- Counter-trend entries in high-fear or high-greed environments

**Supported Trading Pairs:**
- XRP/USDT (HIGH suitability)
- BTC/USDT (MEDIUM-HIGH suitability)
- XRP/BTC (MEDIUM suitability, disabled by default)

**Market Conditions Where Strategy Performs Best:**
1. High-volatility periods with clear sentiment extremes
2. Markets with identifiable whale activity patterns
3. Assets with sufficient liquidity for contrarian positions
4. Periods following sharp price movements (>5% from recent high/low)

### Strategy Limitations

**Known Limitations:**

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| Off-exchange whale activity not detectable | May miss some institutional moves | Extended fear period detection |
| Extended sentiment regimes | Capital exhaustion risk | REC-025: Extended fear pause after 14 days |
| False positive volume spikes | Spurious signals | Multi-filter validation (spread, trades, price move) |
| Correlation across pairs | Concentrated exposure | Real-time correlation blocking |
| Extreme volatility (ATR > 6%) | High loss risk | REC-031: EXTREME regime trading pause |

**Conditions Where Strategy Should NOT Trade:**

1. **EXTREME Volatility Regime** - ATR > 6% triggers automatic pause
2. **Extended Extreme Sentiment** - 14+ days in extreme zone pauses entries
3. **High Correlation** - Correlation > 0.85 between pairs blocks new positions
4. **Neutral Sentiment** - No contrarian opportunity in neutral zones
5. **Circuit Breaker Active** - 2 consecutive losses triggers 45-min pause

**Assumptions:**

| Assumption | Validity | Failure Mode |
|------------|----------|--------------|
| Whale activity detectable via volume | Moderate | Off-exchange activity may be missed |
| Extreme sentiment precedes reversals | Moderate | Can persist longer than expected |
| Volume spikes correlate with institutional moves | Moderate | False positives possible |
| Price deviation reflects market sentiment | High | Established technical analysis principle |

### Risk Profile

**Risk Characteristics:**
- **Direction:** Counter-trend (higher individual trade risk)
- **Win Rate Target:** >33% (with 2:1 R:R ratio)
- **Maximum Consecutive Losses:** 2 before circuit breaker
- **Position Risk:** 2-3% stop loss per trade
- **Portfolio Risk:** Max 150 USD total exposure

**This strategy is NOT suitable for:**
- Trending markets without sentiment extremes
- Low-liquidity trading pairs
- Accounts requiring >90% win rate
- Risk-averse traders uncomfortable with counter-trend positions

## Key Features

### 1. Volume Spike Detection (PRIMARY Signal - 55% Weight)

Volume spike detection is the PRIMARY signal:
- Volume spikes >= 2x average detected as whale activity
- False positive filtering (price movement, spread, trade count)
- Classification: Accumulation (spike + price up), Distribution (spike + price down), Neutral

**Research Support:** "The Moby Dick Effect" (Magner & Sanhueza, 2025) validates whale contagion effects 6-24 hours after transfers.

### 2. Price Deviation Sentiment (PRIMARY - REC-021)

Price deviation from recent high/low is the ONLY sentiment signal:
- **Extreme Fear:** Price >= 8% below recent high
- **Fear:** Price >= 5% below recent high
- **Greed:** Price >= 5% above recent low
- **Extreme Greed:** Price >= 8% above recent low
- **Neutral:** Neither fear nor greed conditions met

### 3. Volatility Regime Classification (REC-023 + REC-031)

ATR-based volatility regime adjusts strategy parameters:

| Regime | ATR % | Size Mult | Stop Mult | Cooldown Mult | Should Pause |
|--------|-------|-----------|-----------|---------------|--------------|
| Low | < 1.5% | 1.1x | 0.8x | 0.8x | No |
| Medium | 1.5-3.5% | 1.0x | 1.0x | 1.0x | No |
| High | 3.5-6.0% | 0.75x | 1.5x | 1.5x | No |
| **EXTREME** | > 6.0% | 0.0x | 2.0x | 3.0x | **Yes** |

**REC-031:** EXTREME regime added in v1.4.0 - trading pauses entirely when ATR exceeds 6%.

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
- Higher short exposure (0.60x multiplier for extreme fear)
- Cross-pair correlation management
- BTC/USDT widened to 2.0% SL / 4.0% TP

## Module Structure

```
strategies/whale_sentiment/
    __init__.py          # Public API exports
    config.py            # Metadata, CONFIG, SYMBOL_CONFIGS, enums
    indicators.py        # Volume spike, fear/greed, ATR, composite (RSI removed)
    signal.py            # generate_signal, _evaluate_symbol
    regimes.py           # Sentiment + volatility regime classification
    risk.py              # Circuit breaker, position limits, correlation
    exits.py             # Exit signal logic
    lifecycle.py         # on_start, on_fill, on_stop
    validation.py        # Config validation
    persistence.py       # Candle data persistence
```

## Configuration

### Volatility Regime (REC-023 + REC-031)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `volatility_low_threshold` | 1.5 | ATR% below = low volatility |
| `volatility_high_threshold` | 3.5 | ATR% above = high volatility |
| `volatility_extreme_threshold` | 6.0 | **REC-031:** ATR% above = EXTREME (pause) |
| `volatility_high_size_mult` | 0.75 | Size reduction in high vol |
| `volatility_high_stop_mult` | 1.5 | Stop widening in high vol |
| `volatility_high_cooldown_mult` | 1.5 | Cooldown extension in high vol |

### Signal Rejection Reasons (v1.4.0)

New rejection reason added for EXTREME volatility:

| Reason | Description |
|--------|-------------|
| `extreme_volatility` | **REC-031:** Trading paused - ATR exceeds extreme threshold |

## New Indicator Fields (v1.4.0)

| Field | Type | Description |
|-------|------|-------------|
| `volatility_should_pause` | bool | **REC-031:** Should trading pause for volatility |
| `volatility_pause_reason` | string | Reason for volatility pause |

## Compliance Checklist (v1.4.0)

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Section 15: Volatility Regime | **Yes** | REC-031 adds EXTREME |
| Section 16: Circuit Breaker | Yes | Stricter than required |
| Section 17: Signal Rejection Tracking | Yes | 19 rejection reasons |
| Section 18: Trade Flow Confirmation | Yes | Contrarian-aware |
| Section 22: Per-Symbol Configuration | Yes | 3 pairs configured |
| Section 24: Correlation Monitoring | Yes | Real-time blocking |
| R:R Ratio >= 1:1 | Yes | All pairs 2:1 |
| USD-Based Sizing | Yes | All signals |
| Indicator Logging | Yes | All code paths |
| **Section 26: Scope Documentation** | **Yes** | REC-033 added |

**Overall Compliance: 100%** (All requirements met)

## Future Improvements

### Deferred from Deep Review v3.0/v4.0:
1. **REC-024:** Backtest-validated confidence weights

### Additional Ideas:
2. External whale data integration (Whale Alert API)
3. Social sentiment API integration
4. On-chain metrics integration
5. Adaptive thresholds based on market conditions
6. XRP/BTC re-enablement monitoring (golden cross printed)

---

**Document Version:** 1.4.0
**Author:** Deep Review v4.0 Implementation
**Platform Version:** WebSocket Paper Tester v1.4.0+
**Review Reference:** deep-review-v4.0.md
