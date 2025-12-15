# Pair Analysis: Momentum Scalping Strategy

**Review Date:** 2025-12-14

---

## Overview

This document analyzes the three trading pairs supported by the momentum scalping strategy:
1. XRP/USDT (Primary)
2. BTC/USDT (Secondary)
3. XRP/BTC (Cross-pair)

---

## 1. XRP/USDT

### 1.1 Market Characteristics (December 2025)

| Metric | Value | Source |
|--------|-------|--------|
| Typical Intraday Volatility | ~5.1% | Strategy research |
| Typical Spread | 0.15% | Strategy research |
| Liquidity Rank | High | Tier-1 exchange pairs |
| Correlation with BTC | 0.84 (90-day), declining | Market data |
| YTD Performance | +20% (outperforming BTC) | Market analysis |

### 1.2 Current Configuration

```python
'XRP/USDT': {
    'ema_fast_period': 8,
    'ema_slow_period': 21,
    'ema_filter_period': 50,
    'rsi_period': 7,
    'position_size_usd': 25.0,
    'take_profit_pct': 0.8,     # 0.8%
    'stop_loss_pct': 0.4,       # 0.4%
    'volume_spike_threshold': 1.5,
}
```

### 1.3 Suitability Assessment

| Criteria | Rating | Notes |
|----------|--------|-------|
| Liquidity | EXCELLENT | Minimal slippage expected |
| Volatility Match | GOOD | 5.1% intraday supports 0.8% TP targets |
| Spread vs TP | ACCEPTABLE | 0.15% spread vs 0.8% TP (~19% of profit) |
| R:R Ratio | GOOD | 2:1 (0.8%/0.4%) |
| RSI Period | AGGRESSIVE | Period 7 may be too fast |

**Overall Suitability: HIGH**

### 1.4 Pair-Specific Recommendations

1. **No changes required** - Configuration is well-optimized
2. **Consider RSI period 8-9** - Reduce noise while maintaining responsiveness
3. **Monitor correlation** - XRP showing independence from BTC in 2025

---

## 2. BTC/USDT

### 2.1 Market Characteristics (December 2025)

| Metric | Value | Source |
|--------|-------|--------|
| ATH (2025) | $126,198 (October 2025) | Market data |
| Current Price Range | ~$90,000-$100,000 | Market data |
| Liquidity Rank | Deepest in crypto | Institutional flow |
| Typical Spread | 0.02-0.05% | Tight due to liquidity |
| ADX (Current) | ~24.81 | Technical indicators |
| Trending Behavior | Strong at extremes | Research finding |

### 2.2 Current Configuration

```python
'BTC/USDT': {
    'ema_fast_period': 8,
    'ema_slow_period': 21,
    'ema_filter_period': 50,
    'rsi_period': 9,            # Slower than XRP (good)
    'position_size_usd': 50.0,  # Higher due to lower volatility
    'take_profit_pct': 0.6,     # Conservative
    'stop_loss_pct': 0.3,       # Tight (viable with low spread)
    'volume_spike_threshold': 1.8,  # Higher (high normal volume)
}
```

### 2.3 Suitability Assessment

| Criteria | Rating | Notes |
|----------|--------|-------|
| Liquidity | EXCELLENT | Best in crypto, minimal slippage |
| Volatility Match | MODERATE | Lower volatility, smaller moves |
| Spread vs TP | EXCELLENT | 0.03% spread vs 0.6% TP (~5% of profit) |
| R:R Ratio | GOOD | 2:1 (0.6%/0.3%) |
| ADX Filter | CRITICAL | BTC trends strongly - filter essential |

**Overall Suitability: MEDIUM-HIGH**

### 2.4 Pair-Specific Concerns

**Critical Finding: BTC Trending Behavior**

Research indicates:
> "BTC tends to trend when it is at its maximum and bounce back when at the minimum."

Current ADX is 24.81 - just below the 25 threshold. This means:
- ADX filter may allow entries that subsequently fail
- BTC at/near ATH ($126K in Oct 2025) exhibits strong trending

**Recommendation:** Consider raising ADX threshold to 30 for BTC specifically.

### 2.5 Pair-Specific Recommendations

1. **Raise ADX threshold** - Consider 30 instead of 25
2. **RSI period 9 is appropriate** - Good balance for BTC
3. **Conservative TP (0.6%)** - Appropriate for lower volatility
4. **Monitor for strong trends** - May need to pause during BTC rallies/crashes

---

## 3. XRP/BTC

### 3.1 Market Characteristics (December 2025)

| Metric | Value | Source |
|--------|-------|--------|
| Liquidity | 7-10x lower than USD pairs | Strategy research |
| XRP-BTC Correlation | 0.40-0.67 (down from 0.85) | 2025 market data |
| XRP vs BTC Volatility | XRP 1.55x more volatile | Research |
| Institutional Interest | XRP gaining independence | Market analysis |
| Spread | Higher than USD pairs | Cross-pair nature |

### 3.2 Current Configuration

```python
'XRP/BTC': {
    'ema_fast_period': 8,
    'ema_slow_period': 21,
    'ema_filter_period': 50,
    'rsi_period': 9,            # Slower (good for noise)
    'position_size_usd': 15.0,  # Smaller (slippage risk)
    'take_profit_pct': 1.2,     # Wider (higher volatility)
    'stop_loss_pct': 0.6,       # Wider (maintains 2:1)
    'volume_spike_threshold': 2.0,  # Higher (need strong confirmation)
    'cooldown_trades': 15,      # Higher (fewer quality signals)
}
```

### 3.3 Suitability Assessment

| Criteria | Rating | Notes |
|----------|--------|-------|
| Liquidity | POOR | 7-10x lower, slippage risk |
| Correlation | CRITICAL ISSUE | Declined 24.86% over 90 days |
| Volatility Match | GOOD | 1.2% TP appropriate for ratio volatility |
| R:R Ratio | GOOD | 2:1 (1.2%/0.6%) |
| Signal Reliability | POOR | Low correlation = unreliable momentum |

**Overall Suitability: LOW (Currently)**

### 3.4 Critical Correlation Analysis

**Current State (December 2025):**

| Metric | Value | Implication |
|--------|-------|-------------|
| 90-day Correlation | ~0.84 | Still significant |
| 90-day Decline | -24.86% | Rapidly decoupling |
| Recent Range | 0.40-0.67 | Highly variable |
| Pause Threshold | 0.50 | May trigger frequently |

**Research Finding:**
> "XRP's correlation with Bitcoin is continuing to weaken... this 'weakening correlation with Bitcoin highlights its growing independence in 2025, fueled by Ripple's expanding real-world footprint.'"

**XRP Independence Factors:**
1. $1 billion GTreasury deal (access to $120T payments market)
2. Ripple institutional acquisitions
3. Regulatory clarity (post-SEC settlement)
4. TradFi integration expanding

### 3.5 Pair-Specific Recommendations

1. **PAUSE TRADING** - Until correlation stabilizes > 0.60
2. **If trading:** Reduce position size further (10 USD max)
3. **Increase correlation lookback** - Consider 100 candles vs 50
4. **Consider removing pair** - May not be suitable for momentum scalping in current market

---

## 4. Cross-Pair Comparison

### 4.1 Configuration Comparison

| Parameter | XRP/USDT | BTC/USDT | XRP/BTC | Rationale |
|-----------|----------|----------|---------|-----------|
| RSI Period | 7 | 9 | 9 | BTC/XRP-BTC slower for noise |
| Position Size | $25 | $50 | $15 | Risk-adjusted for volatility |
| Take Profit | 0.8% | 0.6% | 1.2% | Scaled to pair volatility |
| Stop Loss | 0.4% | 0.3% | 0.6% | Maintains 2:1 R:R |
| Volume Threshold | 1.5x | 1.8x | 2.0x | Higher for less liquid pairs |
| Cooldown | 5 | 5 | 15 | XRP/BTC fewer quality signals |

### 4.2 Risk Assessment by Pair

| Pair | Signal Quality | Execution Risk | Correlation Risk | Overall Risk |
|------|----------------|----------------|------------------|--------------|
| XRP/USDT | HIGH | LOW | MEDIUM | **LOW** |
| BTC/USDT | MEDIUM | LOW | N/A | **MEDIUM** |
| XRP/BTC | LOW | HIGH | HIGH | **HIGH** |

### 4.3 Recommended Trading Priority

1. **XRP/USDT** - Primary focus, best risk-adjusted returns
2. **BTC/USDT** - Secondary, with enhanced ADX filtering
3. **XRP/BTC** - PAUSE or disable until correlation improves

---

## 5. Session Impact by Pair

### 5.1 Best Trading Sessions

| Pair | Best Session | Worst Session | Notes |
|------|--------------|---------------|-------|
| XRP/USDT | US_EUROPE_OVERLAP | OFF_HOURS | High retail activity |
| BTC/USDT | US_EUROPE_OVERLAP | ASIA | Institutional flow |
| XRP/BTC | EUROPE | OFF_HOURS | Cross-pair spreads tightest |

### 5.2 Session Multiplier Assessment

Current configuration (from `config.py`):

```python
'session_size_multipliers': {
    'ASIA': 0.8,              # Appropriate
    'EUROPE': 1.0,            # Standard
    'US': 1.0,                # Standard
    'US_EUROPE_OVERLAP': 1.1, # Appropriate
    'OFF_HOURS': 0.5,         # Appropriately conservative
}
```

**Assessment:** Session multipliers are well-calibrated for all pairs.

---

## 6. Summary Recommendations

### 6.1 Immediate Actions

| Pair | Action | Reason |
|------|--------|--------|
| XRP/USDT | Continue trading | Well-optimized |
| BTC/USDT | Raise ADX to 30 | Trending behavior risk |
| XRP/BTC | **PAUSE** | Correlation breakdown |

### 6.2 Monitoring Requirements

1. **Daily:** Check XRP-BTC correlation
2. **Weekly:** Review ADX levels for BTC
3. **Monthly:** Reassess pair suitability

### 6.3 Parameter Tuning Candidates

| Pair | Parameter | Current | Suggested | Priority |
|------|-----------|---------|-----------|----------|
| XRP/USDT | rsi_period | 7 | 8 | LOW |
| BTC/USDT | adx_threshold | 25 | 30 | HIGH |
| XRP/BTC | correlation_pause | 0.50 | 0.55 | MEDIUM |

---

*Next: [Compliance Matrix](./compliance-matrix.md)*
