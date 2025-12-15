# Pair Analysis: Momentum Scalping Strategy v2.0

**Review Date:** 2025-12-14
**Review Version:** v2.0 (December 2025 Market Data)

---

## Overview

This document analyzes the three trading pairs supported by the momentum scalping strategy:
1. XRP/USDT (Primary)
2. BTC/USDT (Secondary)
3. XRP/BTC (Cross-pair - Conditional)

---

## 1. XRP/USDT

### 1.1 Market Characteristics (December 2025)

| Metric | Value | Source |
|--------|-------|--------|
| Current Price | ~$2.02 | Market data |
| Market Cap | $125.1 billion | CoinGape |
| Market Cap Share | 4.63% | Global crypto |
| 90-day BTC Correlation | 0.84 (declining) | AMBCrypto |
| YTD Performance | +20% | AMBCrypto |
| Liquidity Rank | 4th globally | Market data |

### 1.2 Current Configuration (v2.1.0)

| Parameter | Value | Line Reference |
|-----------|-------|----------------|
| `ema_fast_period` | 8 | `config.py:265` |
| `ema_slow_period` | 21 | `config.py:266` |
| `ema_filter_period` | 50 | `config.py:267` |
| `rsi_period` | 8 | `config.py:268` (REC-003) |
| `position_size_usd` | 25.0 | `config.py:269` |
| `take_profit_pct` | 0.8% | `config.py:270` |
| `stop_loss_pct` | 0.4% | `config.py:271` |
| `volume_spike_threshold` | 1.5x | `config.py:272` |

### 1.3 Suitability Assessment (v2.1.0)

| Criteria | Rating | Notes |
|----------|--------|-------|
| Liquidity | EXCELLENT | 4th largest, minimal slippage |
| Volatility Match | GOOD | Supports 0.8% TP targets |
| Spread vs TP | ACCEPTABLE | ~0.15% spread vs 0.8% TP |
| R:R Ratio | GOOD | 2:1 (0.8%/0.4%) |
| RSI Period | OPTIMIZED | Period 8 (was 7) |
| Independence | INCREASING | Growing real-world use cases |

**Overall Suitability: HIGH** (Primary trading pair)

### 1.4 XRP Independence Factors (December 2025)

Research indicates XRP is becoming more independent from Bitcoin:

1. **$1 billion GTreasury deal** - Access to $120 trillion payments market
2. **Ripple institutional acquisitions** - Three major acquisitions in 2025
3. **Regulatory clarity** - Post-SEC settlement
4. **TradFi integration expanding** - TVL up 54% in 2025

**Implication:** Correlation monitoring remains critical as XRP may decouple further.

---

## 2. BTC/USDT

### 2.1 Market Characteristics (December 2025)

| Metric | Value | Source |
|--------|-------|--------|
| Current Price | $90,000-$100,000 | CoinDesk |
| 2025 ATH | $126,198 (October) | CoinCodex |
| 30-day Volatility | 49% (annualized) | Volmex BVIV |
| Price Volatility | 3.27% (30-day) | CoinCodex |
| RSI (Daily) | 44.94 | CoinCodex |
| Fear & Greed | 23 (Extreme Fear) | Market data |
| Liquidity Rank | #1 globally | Market data |

### 2.2 Current Configuration (v2.1.0)

| Parameter | Value | Line Reference |
|-----------|-------|----------------|
| `ema_fast_period` | 8 | `config.py:281` |
| `ema_slow_period` | 21 | `config.py:282` |
| `ema_filter_period` | 50 | `config.py:283` |
| `rsi_period` | 9 | `config.py:284` |
| `position_size_usd` | 50.0 | `config.py:285` |
| `take_profit_pct` | 0.6% | `config.py:286` |
| `stop_loss_pct` | 0.3% | `config.py:287` |
| `volume_spike_threshold` | 1.8x | `config.py:288` |

### 2.3 Suitability Assessment (v2.1.0)

| Criteria | Rating | Notes |
|----------|--------|-------|
| Liquidity | EXCELLENT | Deepest in crypto, minimal slippage |
| Volatility Match | GOOD | Compressing volatility favorable |
| Spread vs TP | EXCELLENT | ~0.03% spread vs 0.6% TP |
| R:R Ratio | GOOD | 2:1 (0.6%/0.3%) |
| ADX Filter | ACTIVE | Threshold at 30 (REC-002) |
| RSI (Daily) | NEUTRAL | 44.94 supports scalping |

**Overall Suitability: HIGH** (Secondary trading pair)

### 2.4 BTC Market Context (December 2025)

| Factor | Status | Implication |
|--------|--------|-------------|
| Volatility | Compressing | MEDIUM regime favorable |
| Trend Strength | Moderate | ADX filter provides protection |
| Market Sentiment | Extreme Fear | Contrarian bullish signal |
| Price Zone | High ($90K+) | Potential for trending moves |

**Recommendation:** Continue trading with ADX filter at 30. Monitor for volatility expansion.

---

## 3. XRP/BTC

### 3.1 Market Characteristics (December 2025)

| Metric | Value | Source |
|--------|-------|--------|
| 90-day Correlation | 0.84 | AMBCrypto |
| Correlation Trend | -24.86% decline | AMBCrypto |
| Recent Correlation Range | 0.40-0.67 | Market analysis |
| Liquidity | 7-10x lower than USD pairs | Strategy research |
| XRP vs BTC Volatility | XRP 1.55x more volatile | Research |
| Pause Threshold | 0.60 | `config.py:205` |

### 3.2 Current Configuration (v2.1.0)

| Parameter | Value | Line Reference |
|-----------|-------|----------------|
| `ema_fast_period` | 8 | `config.py:297` |
| `ema_slow_period` | 21 | `config.py:298` |
| `ema_filter_period` | 50 | `config.py:299` |
| `rsi_period` | 9 | `config.py:300` |
| `position_size_usd` | 15.0 | `config.py:301` |
| `take_profit_pct` | 1.2% | `config.py:302` |
| `stop_loss_pct` | 0.6% | `config.py:303` |
| `volume_spike_threshold` | 2.0x | `config.py:304` |
| `cooldown_trades` | 15 | `config.py:305` |

### 3.3 Correlation Protection (v2.1.0)

| Parameter | Value | Line Reference |
|-----------|-------|----------------|
| `use_correlation_monitoring` | True | `config.py:202` |
| `correlation_lookback` | 100 | `config.py:203` (REC-008) |
| `correlation_warn_threshold` | 0.60 | `config.py:204` |
| `correlation_pause_threshold` | 0.60 | `config.py:205` (REC-001) |
| `correlation_pause_enabled` | True | `config.py:206` |

### 3.4 Suitability Assessment (v2.1.0)

| Criteria | Rating | Notes |
|----------|--------|-------|
| Liquidity | POOR | 7-10x lower, slippage risk |
| Correlation | CONDITIONAL | 0.60 pause threshold active |
| Volatility Match | GOOD | 1.2% TP for ratio volatility |
| R:R Ratio | GOOD | 2:1 (1.2%/0.6%) |
| Signal Reliability | CONDITIONAL | Depends on correlation |

**Overall Suitability: CONDITIONAL**

Trading only permitted when:
- XRP-BTC correlation > 0.60 (checked via `should_pause_for_low_correlation()`)
- Implementation: `risk.py:369-395`

### 3.5 XRP Independence Analysis

**December 2025 Research Findings:**

> "XRP's correlation with Bitcoin is continuing to weaken. This weakening correlation with Bitcoin highlights its growing independence in 2025, fueled by Ripple's expanding real-world footprint."

**Key Independence Drivers:**
1. Ripple GTreasury ($120T payments market access)
2. Three major TradFi acquisitions in 2025
3. TVL surge 54% (outpacing BTC's 33%)
4. XRP outperforming BTC by 1.13x in 2025

**Implication:** XRP/BTC pair may face increasing periods of trading pause as correlation continues to decline. The 0.60 threshold provides appropriate protection.

---

## 4. Cross-Pair Comparison

### 4.1 Configuration Comparison (v2.1.0)

| Parameter | XRP/USDT | BTC/USDT | XRP/BTC | Rationale |
|-----------|----------|----------|---------|-----------|
| RSI Period | 8 | 9 | 9 | XRP faster, BTC/ratio slower |
| Position Size | $25 | $50 | $15 | Risk-adjusted |
| Take Profit | 0.8% | 0.6% | 1.2% | Volatility-scaled |
| Stop Loss | 0.4% | 0.3% | 0.6% | Maintains 2:1 R:R |
| Volume Threshold | 1.5x | 1.8x | 2.0x | Liquidity-adjusted |
| Cooldown | 5 | 5 | 15 | Quality over quantity |

### 4.2 Risk Assessment by Pair (v2.1.0)

| Pair | Signal Quality | Execution Risk | Special Risk | Overall Risk |
|------|----------------|----------------|--------------|--------------|
| XRP/USDT | HIGH | LOW | Independence | **LOW** |
| BTC/USDT | HIGH | LOW | Trending | **LOW** |
| XRP/BTC | CONDITIONAL | MEDIUM | Correlation | **MEDIUM** |

### 4.3 Recommended Trading Priority

1. **XRP/USDT** - Primary focus, excellent risk-adjusted returns
2. **BTC/USDT** - Secondary, with ADX filter active
3. **XRP/BTC** - Conditional only when correlation > 0.60

---

## 5. Session Impact by Pair

### 5.1 Best Trading Sessions

| Pair | Best Session | Worst Session | Notes |
|------|--------------|---------------|-------|
| XRP/USDT | US_EUROPE_OVERLAP | OFF_HOURS | Retail activity peak |
| BTC/USDT | US_EUROPE_OVERLAP | ASIA | Institutional flow |
| XRP/BTC | EUROPE | OFF_HOURS | Tightest spreads |

### 5.2 Session Multiplier Assessment

| Session | Threshold Mult | Size Mult | Line Reference |
|---------|----------------|-----------|----------------|
| ASIA | 1.2 | 0.8 | `config.py:147-148` |
| EUROPE | 1.0 | 1.0 | `config.py:149` |
| US | 1.0 | 1.0 | `config.py:150` |
| US_EUROPE_OVERLAP | 0.9 | 1.1 | `config.py:151` |
| OFF_HOURS | 1.4 | 0.5 | `config.py:152` |

**Assessment:** Session multipliers are well-calibrated for all pairs.

---

## 6. Summary Recommendations (v2.1.0 Verified)

### 6.1 Trading Status

| Pair | Status | Condition |
|------|--------|-----------|
| XRP/USDT | ACTIVE | Primary pair |
| BTC/USDT | ACTIVE | ADX filter protects |
| XRP/BTC | CONDITIONAL | Correlation > 0.60 required |

### 6.2 Monitoring Requirements

1. **Daily:** Check XRP-BTC correlation status
2. **Weekly:** Review ADX levels for BTC trending behavior
3. **Monthly:** Reassess pair suitability based on correlation trends

### 6.3 Future Considerations

| Pair | Concern | Monitoring |
|------|---------|------------|
| XRP/USDT | Growing independence | Track correlation trend |
| BTC/USDT | Volatility expansion | Watch Fear & Greed Index |
| XRP/BTC | Correlation decline | May require threshold adjustment |

---

*Next: [Compliance Matrix](./compliance-matrix.md)*
