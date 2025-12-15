# Research Findings: Momentum Scalping Strategy v2.0

**Review Date:** 2025-12-14
**Review Version:** v2.0 (Updated with December 2025 research)

---

## 1. Academic Foundations

### 1.1 Momentum Trading Theory

Momentum trading exploits the tendency for assets that have performed well (or poorly) to continue in the same direction for a period. Academic research supports momentum as one of the few persistent market anomalies.

**Key Academic Finding (PMC/NIH Study):**
> "The long-only strategy results using RSI were relatively good, with 4 out of 10 indices making an overhold profit using the oversold RSI value as an entry signal. However, the returns were worse for strongly growing cryptocurrencies."

**Implication:** In strongly trending crypto markets, traditional RSI overbought/oversold signals underperform buy-and-hold. The strategy's regime-based RSI adjustment addresses this.

### 1.2 RSI Effectiveness in Cryptocurrency Markets (2025 Update)

**Research Consensus (December 2025):**

1. **RSI works best in ranging/sideways markets** - Not during strong trends
2. **Crypto sustains overbought/oversold longer** - Standard 70/30 thresholds less reliable
3. **Higher timeframes more reliable** - 1-minute RSI generates more false signals
4. **Combined indicators outperform** - RSI + MACD combination shows improved win rates

**2025 Research Finding:**
> "Cryptocurrency assets can sustain overbought conditions longer than traditional markets, rendering RSI less reliable during volatile periods. For example, NEAR Protocol experienced a 42% price surge on November 7, 2025, with RSI only briefly exceeding 80."

**Optimal RSI Settings (2025 Research):**
- **RSI 7 (Short Term)** - Ideal for scalping on 1-minute to 15-minute charts
- **RSI 14 (Standard)** - Better for swing trading and daily timeframes
- **RSI 9** - Good balance for crypto markets (less noise than 7)

**Strategy Alignment:**
- XRP/USDT: RSI period 8 (balanced for noise reduction) - ALIGNED
- BTC/USDT: RSI period 9 (appropriate for lower volatility) - ALIGNED
- Regime-based RSI bands (75/25 in HIGH volatility) - ALIGNED with research

### 1.3 MACD Settings for 1-Minute Scalping

**Industry Standard for Scalping (2025 Update):**
- Scalping: 5, 13, 1 (fastest response)
- Alternative: 6, 13, 5 (current strategy uses this - slightly smoother)
- Standard: 12, 26, 9 (not suitable for 1-minute)

**2025 Research Finding:**
> "The default MACD settings (12, 26, 9) are built for daily charts. For scalping, the preferred setting is 5, 13, 1. This configuration makes the MACD lines and histogram react much more quickly."

**Strategy Alignment:**
- MACD settings (6, 13, 5) - ALIGNED with industry practices
- Slightly less aggressive than 5/13/1 but provides smoother signals
- Combines MACD with RSI for confirmation - ALIGNED with best practices

### 1.4 ADX Trend Strength Filtering (2025 Update)

**Research Finding (2024-2025 Market Data):**
> "In 2024, traders found that an ADX above 30 often signaled strong market moves. Bitcoin's ADX hit 35 during a major rally in April 2024, confirming a strong upward trend. More than 60% of professional traders used ADX daily."

**Updated ADX Thresholds (Crypto-Specific):**

| ADX Value | Interpretation | Scalping Suitability |
|-----------|----------------|---------------------|
| < 20 | Weak/no trend | IDEAL for scalping |
| 20-25 | Trend developing | CAUTION |
| 25-30 | Moderate trend | REDUCE size |
| 30-50 | Strong trend | AVOID (current threshold) |
| > 50 | Very strong trend | Trend-following only |

**Strategy Configuration (v2.1.0):**
- ADX threshold: 30 (raised from 25 per REC-002)
- Applied to BTC/USDT only - ALIGNED with research

---

## 2. Market Condition Analysis (December 2025)

### 2.1 Current Bitcoin Market Status

| Metric | Value | Implication |
|--------|-------|-------------|
| BTC Price Range | $90,000-$100,000 | High price zone |
| 30-day Volatility | 49% (annualized) | Compressing |
| RSI (Daily) | 44.94 | Neutral zone |
| Price Volatility | 3.27% (30-day) | Moderate |
| Fear & Greed Index | 23 (Extreme Fear) | Contrarian signal |

**Market Condition Assessment:**
- Volatility compressing suggests MEDIUM regime favorable for scalping
- Neutral RSI indicates no strong directional bias
- Extreme Fear may precede volatility expansion - monitor closely

### 2.2 Conditions Where Momentum Scalping Excels

1. **Range-bound markets** - Price oscillates between support/resistance
2. **Mean-reverting conditions** - Oversold/overbought leads to reversals
3. **High liquidity sessions** - US/Europe overlap (14:00-17:00 UTC)
4. **Moderate volatility** - MEDIUM regime classification
5. **Correlated asset pairs** - When XRP-BTC correlation > 0.70

### 2.3 Conditions Where Momentum Scalping Fails

**Market Condition Failures:**

1. **Strong trending markets** - Price continues beyond RSI extremes (ADX filter protects)
2. **Low volatility choppy markets** - False signals, slippage > profit
3. **Correlation breakdown** - XRP/BTC signals unreliable (pause threshold protects)
4. **Extreme volatility** - Stops triggered before momentum plays out (regime pause)
5. **Thin liquidity (OFF_HOURS)** - Slippage erodes profits (session adjustment)

**Technical Failures:**

1. **RSI overbought continuation** - Crypto can stay overbought for weeks (regime RSI protects)
2. **News/event-driven moves** - Technical indicators lag fundamentals
3. **Flash crashes** - Stop losses may not protect at desired price
4. **Whipsaw conditions** - Rapid reversals trigger consecutive losses (circuit breaker)

### 2.4 Predictable Crash Conditions

**Research Finding (CEPR):**
> "The momentum strategy is most likely to crash when past returns are high. Capital available to momentum traders predicts sharp downturns in momentum profits."

**Implication:** After strong performance periods, the strategy may face increased drawdown risk. Circuit breaker (3 consecutive losses) provides partial protection.

---

## 3. Optimal Parameter Selection (v2.1.0 Verified)

### 3.1 RSI Period

| Period | Use Case | Noise Level | Strategy Assignment |
|--------|----------|-------------|---------------------|
| 7 | Ultra-fast scalping | HIGH | Replaced by 8 |
| 8 | Fast scalping (XRP) | MODERATE | XRP/USDT (REC-003) |
| 9 | Balanced scalping | MODERATE | BTC/USDT, XRP/BTC |
| 14 | Standard | LOW | Better for 5m+ |

**Implementation:** `config.py:268` (XRP), `config.py:284` (BTC), `config.py:300` (XRP/BTC)

### 3.2 MACD Settings

| Setting | Fast | Slow | Signal | Assessment |
|---------|------|------|--------|------------|
| Ultra-fast | 5 | 13 | 1 | More aggressive |
| **Current** | **6** | **13** | **5** | Good balance |
| Standard | 12 | 26 | 9 | Too slow |

**Implementation:** `config.py:87-89`

### 3.3 Take-Profit / Stop-Loss Ratios

| Pair | TP% | SL% | R:R Ratio | Line Reference |
|------|-----|-----|-----------|----------------|
| XRP/USDT | 0.8% | 0.4% | 2:1 | `config.py:270-271` |
| BTC/USDT | 0.6% | 0.3% | 2:1 | `config.py:286-287` |
| XRP/BTC | 1.2% | 0.6% | 2:1 | `config.py:302-303` |

**All pairs maintain minimum 2:1 R:R ratio** - COMPLIANT

### 3.4 Volume Spike Thresholds

| Pair | Current | Research Range | Line Reference |
|------|---------|----------------|----------------|
| XRP/USDT | 1.5x | 1.3-2.0x | `config.py:272` |
| BTC/USDT | 1.8x | 1.5-2.5x | `config.py:288` |
| XRP/BTC | 2.0x | 1.8-3.0x | `config.py:304` |

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

**Implementation:** `indicators.py:70-112` - Follows standard Wilder's method correctly.

### 4.2 MACD Calculation

```
MACD Line = EMA(fast) - EMA(slow)
Signal Line = EMA(MACD Line, signal_period)
Histogram = MACD Line - Signal Line
```

**Implementation:** `indicators.py:165-226` - Correct implementation with crossover detection.

### 4.3 Correlation Calculation

Uses Pearson correlation on returns:

```
correlation = covariance(returns_a, returns_b) / (std_a * std_b)
```

**Implementation:** `indicators.py:587-648` - 100-candle lookback on 5m timeframe (~8.3 hours).

### 4.4 ADX Calculation

```
True Range = max(high-low, |high-prev_close|, |low-prev_close|)
+DM = high - prev_high (if positive and > -DM)
-DM = prev_low - low (if positive and > +DM)
ADX = smoothed(|+DI - -DI| / |+DI + -DI|) * 100
```

**Implementation:** `indicators.py:654-760` - Uses Wilder's smoothing correctly.

### 4.5 ATR Calculation (v2.1.0)

```
TR = max(high-low, |high-prev_close|, |low-prev_close|)
ATR = SMA(TR, period)
```

**Implementation:** `indicators.py:407-438` - Used for trailing stop calculation.

---

## 5. Key Research Implications

### 5.1 Strategy Alignment Assessment

| Finding | Implementation | Status |
|---------|----------------|--------|
| RSI alone unreliable in crypto | RSI + MACD combination | ✅ ALIGNED |
| 70/30 RSI too narrow | Regime RSI widening (75/25) | ✅ ALIGNED |
| Strong trends defeat scalping | ADX filter at 30 | ✅ ALIGNED |
| Correlation matters | Correlation monitoring + pause | ✅ ALIGNED |
| 2:1 R:R required | All pairs compliant | ✅ ALIGNED |
| Trade flow confirmation | Imbalance filter added | ✅ ALIGNED |
| 1-minute inherently noisy | Multi-timeframe (5m) filter | ✅ ALIGNED |

### 5.2 Areas of Continued Monitoring

1. **XRP independence increasing** - Correlation may continue declining
2. **Market sentiment extreme fear** - Volatility expansion possible
3. **BTC volatility compressing** - Watch for breakout signals
4. **Session boundaries** - DST transitions (documented in regimes.py)

---

*Next: [Pair Analysis](./pair-analysis.md)*
