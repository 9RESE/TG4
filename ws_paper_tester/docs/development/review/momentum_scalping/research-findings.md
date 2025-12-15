# Research Findings: Momentum Scalping Strategy

**Review Date:** 2025-12-14

---

## 1. Academic Foundations

### 1.1 Momentum Trading Theory

Momentum trading exploits the tendency for assets that have performed well (or poorly) to continue in the same direction for a period. Academic research supports momentum as one of the few persistent market anomalies.

**Key Academic Finding (PMC/NIH Study):**
> "The long-only strategy results using RSI were relatively good, with 4 out of 10 indices making an overhold profit using the oversold RSI value as an entry signal. However, the returns were worse for strongly growing cryptocurrencies."

**Implication:** In strongly trending crypto markets, traditional RSI overbought/oversold signals underperform buy-and-hold. The strategy's regime-based RSI adjustment (REC-004) partially addresses this.

### 1.2 RSI Effectiveness in Cryptocurrency Markets

**Research Consensus:**

1. **RSI works best in ranging/sideways markets** - Not during strong trends
2. **Crypto sustains overbought/oversold longer** - Standard 70/30 thresholds less reliable
3. **Higher timeframes more reliable** - 1-minute RSI generates more false signals
4. **Combined indicators outperform** - RSI + MACD combination shows 73% win rate in backtests

**Academic Warning (PMC Study):**
> "The authors advise against treating RSI oversold level as a long signal, as results show the upward, asymmetric nature of the cryptocurrency market may make primary RSI applications ineffective."

**Strategy Alignment:**
- Uses RSI period 7 (scalping-optimized, but high noise)
- Combines with MACD for confirmation
- Regime-based RSI bands (75/25 in HIGH volatility) - ALIGNED with research

### 1.3 MACD Settings for 1-Minute Scalping

**Industry Standard for Scalping:**
- Fast: 6, Slow: 13, Signal: 5 (the strategy uses exactly this)
- Reacts within 1-2 candles (60-120 seconds)
- More responsive than traditional 12/26/9 settings

**Academic Finding (Tong Chio, 2022):**
> "MACD strategies on short timeframes underperform unless combined with additional momentum filters (like RSI or MFI)."

**Strategy Alignment:**
- MACD settings (6, 13, 5) - ALIGNED with industry best practices
- Combines MACD with RSI - ALIGNED with academic recommendation
- Uses volume confirmation - Additional filter beyond minimum recommendation

### 1.4 ADX Trend Strength Filtering

**Research Finding (2024 Market Data):**
> "In 2024, traders found that an ADX above 30 often signaled strong market moves. Bitcoin's ADX hit 35 during a major rally in April 2024, confirming a strong upward trend. More than 60% of professional traders used ADX daily."

**Academic/Industry Thresholds:**

| ADX Value | Interpretation | Scalping Suitability |
|-----------|----------------|---------------------|
| < 20 | Weak/no trend | IDEAL for mean-reversion |
| 20-25 | Trend developing | CAUTION |
| 25-30 | Moderate trend | AVOID scalping |
| 30-50 | Strong trend | DEFINITELY AVOID |
| > 50 | Very strong trend | Trend-following only |

**Strategy Configuration:**
- ADX threshold: 25 (current)
- Research suggests: 30 may be more appropriate for crypto volatility

---

## 2. Market Condition Analysis

### 2.1 Conditions Where Momentum Scalping Excels

1. **Range-bound markets** - Price oscillates between support/resistance
2. **Mean-reverting conditions** - Oversold/overbought leads to reversals
3. **High liquidity sessions** - US/Europe overlap (14:00-17:00 UTC)
4. **Moderate volatility** - MEDIUM regime classification
5. **Correlated asset pairs** - When XRP-BTC correlation > 0.70

### 2.2 Conditions Where Momentum Scalping Fails

**Market Condition Failures:**

1. **Strong trending markets** - Price continues beyond RSI extremes
2. **Low volatility choppy markets** - False signals, slippage > profit
3. **Correlation breakdown** - XRP/BTC signals become unreliable
4. **Extreme volatility** - Stops triggered before momentum plays out
5. **Thin liquidity (OFF_HOURS)** - Slippage erodes profits

**Technical Failures:**

1. **RSI overbought continuation** - Crypto can stay overbought for weeks
2. **News/event-driven moves** - Technical indicators lag fundamentals
3. **Flash crashes** - Stop losses may not protect at desired price
4. **Whipsaw conditions** - Rapid reversals trigger consecutive losses

### 2.3 Predictable Crash Conditions

**Research Finding (CEPR):**
> "The momentum strategy is most likely to crash when past returns are high. Capital available to momentum traders predicts sharp downturns in momentum profits."

**Implication:** After strong performance periods, the strategy may face increased drawdown risk. Circuit breaker (3 consecutive losses) provides partial protection.

---

## 3. Optimal Parameter Selection

### 3.1 RSI Period

| Period | Use Case | Noise Level | Strategy Fit |
|--------|----------|-------------|--------------|
| 7 | Ultra-fast scalping | HIGH | Current (aggressive) |
| 9 | Fast scalping | MODERATE | Recommended for BTC |
| 14 | Standard | LOW | Better for 5m+ timeframes |

**Recommendation:** Consider RSI period 9 for BTC/USDT (already configured in SYMBOL_CONFIGS)

### 3.2 MACD Settings

| Setting | Fast | Slow | Signal | Use Case |
|---------|------|------|--------|----------|
| Ultra-fast | 5 | 13 | 6 | Very aggressive |
| **Scalping** | **6** | **13** | **5** | **Current (optimal)** |
| Standard | 12 | 26 | 9 | Longer timeframes |

**Assessment:** Current MACD settings are industry-standard for 1-minute scalping.

### 3.3 Take-Profit / Stop-Loss Ratios

| Pair | TP% | SL% | R:R Ratio | Assessment |
|------|-----|-----|-----------|------------|
| XRP/USDT | 0.8% | 0.4% | 2:1 | GOOD |
| BTC/USDT | 0.6% | 0.3% | 2:1 | GOOD |
| XRP/BTC | 1.2% | 0.6% | 2:1 | GOOD |

**All pairs maintain minimum 2:1 R:R ratio** - COMPLIANT with best practices

### 3.4 Volume Spike Thresholds

| Pair | Current | Research Range | Assessment |
|------|---------|----------------|------------|
| XRP/USDT | 1.5x | 1.3-2.0x | APPROPRIATE |
| BTC/USDT | 1.8x | 1.5-2.5x | APPROPRIATE |
| XRP/BTC | 2.0x | 1.8-3.0x | APPROPRIATE |

---

## 4. Mathematical Models

### 4.1 RSI Calculation

The strategy uses Wilder's smoothing method for RSI:

```
RS = Average Gain / Average Loss (over N periods)
RSI = 100 - (100 / (1 + RS))

Smoothed average:
avg_gain = (prev_avg_gain * (period - 1) + current_gain) / period
```

**Assessment:** Implementation in `indicators.py:70-112` follows standard Wilder's method correctly.

### 4.2 MACD Calculation

```
MACD Line = EMA(fast) - EMA(slow)
Signal Line = EMA(MACD Line, signal_period)
Histogram = MACD Line - Signal Line
```

**Assessment:** Implementation in `indicators.py:165-226` is correct. Crossover detection properly identifies histogram sign changes.

### 4.3 Correlation Calculation

Uses Pearson correlation on returns:

```
correlation = covariance(returns_a, returns_b) / (std_a * std_b)
```

**Assessment:** Implementation in `indicators.py:587-648` is mathematically correct. Uses 50-candle lookback on 5m timeframe (~4 hours of data).

### 4.4 ADX Calculation

```
True Range = max(high-low, |high-prev_close|, |low-prev_close|)
+DM = high - prev_high (if positive and > -DM)
-DM = prev_low - low (if positive and > +DM)
ADX = smoothed(|+DI - -DI| / |+DI + -DI|) * 100
```

**Assessment:** Implementation in `indicators.py:654-760` uses Wilder's smoothing correctly.

---

## 5. Key Research Implications

### 5.1 For Current Strategy

1. **RSI alone unreliable in crypto** - Strategy correctly combines with MACD
2. **70/30 RSI too narrow** - REC-004 (regime RSI widening) is well-founded
3. **Strong trends defeat scalping** - ADX filter (REC-003) addresses this
4. **Correlation matters** - REC-001 correlation monitoring is critical
5. **2:1 R:R required** - All pairs compliant

### 5.2 Areas of Concern

1. **1-minute timeframe inherently noisy** - Higher false signal rate expected
2. **7-period RSI very fast** - May benefit from 9 for less noise
3. **ADX threshold 25 may be too low** - Research suggests 30 for crypto
4. **Correlation lookback 50 candles (5m)** - ~4 hours, may miss rapid changes

---

*Next: [Pair Analysis](./pair-analysis.md)*
