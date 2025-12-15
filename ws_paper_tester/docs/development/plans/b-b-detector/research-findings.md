# Market Regime Detection: Research Findings

## 1. Algorithm Categories

### 1.1 Rule-Based Methods (Recommended for Initial Implementation)

Rule-based methods are deterministic, interpretable, and computationally efficient. They're ideal for real-time trading systems.

#### A. Moving Average Regime Classification

**Simple MA Threshold:**
```
IF price > SMA(200) THEN regime = "BULL"
IF price < SMA(200) THEN regime = "BEAR"
```

**Dual MA Crossover:**
```
IF SMA(50) > SMA(200) THEN regime = "BULL"  (Golden Cross)
IF SMA(50) < SMA(200) THEN regime = "BEAR"  (Death Cross)
```

**Triple MA System (Recommended):**
```python
def classify_ma_regime(price, sma_20, sma_50, sma_200):
    if price > sma_20 > sma_50 > sma_200:
        return "STRONG_BULL"
    elif price > sma_50 > sma_200:
        return "BULL"
    elif price < sma_20 < sma_50 < sma_200:
        return "STRONG_BEAR"
    elif price < sma_50 < sma_200:
        return "BEAR"
    else:
        return "SIDEWAYS"
```

**Pros:** Simple, clear, widely understood
**Cons:** Lagging indicator, whipsaws in ranging markets

#### B. ADX-Based Trend Strength Detection

The **Average Directional Index (ADX)** is specifically designed to measure trend strength without regard to direction.

**ADX Thresholds (Industry Standard):**

| ADX Value | Market Condition | Strategy Recommendation |
|-----------|-----------------|------------------------|
| 0-15 | Absent or weak trend | Avoid trend strategies, use mean reversion |
| 15-20 | Developing trend | Prepare for breakout |
| 20-25 | Emerging trend | Early trend entry possible |
| 25-40 | Strong trend | Trend following optimal |
| 40-60 | Very strong trend | Ride the trend, tighten stops |
| 60+ | Extremely strong | Trend may be exhausting, watch for reversal |

**Directional Movement (+DI/-DI) for Direction:**
```python
def classify_adx_regime(adx, plus_di, minus_di):
    if adx < 20:
        return "SIDEWAYS", "RANGE_BOUND"
    elif adx >= 25:
        if plus_di > minus_di:
            return "TRENDING", "BULLISH"
        else:
            return "TRENDING", "BEARISH"
    else:
        return "TRANSITIONING", "UNCERTAIN"
```

**Pros:** Non-directional strength measure, industry standard
**Cons:** Lagging (14-period default), can stay elevated after trend ends

#### C. Choppiness Index (CHOP) - Consolidation Detection

The Choppiness Index specifically identifies sideways/ranging markets where trend strategies fail.

**Formula:**
```
CHOP = 100 * LOG10(SUM(ATR(1), n) / (HIGH_n - LOW_n)) / LOG10(n)
```

**Interpretation Thresholds:**

| CHOP Value | Market State | Interpretation |
|------------|-------------|----------------|
| > 61.8 | Choppy/Sideways | Market consolidating, avoid trend trades |
| 38.2 - 61.8 | Transitional | Market deciding direction |
| < 38.2 | Trending | Strong directional movement |

**Python Implementation:**
```python
def calculate_choppiness(highs, lows, closes, period=14):
    atr_sum = sum_atr(highs, lows, closes, period)
    high_low_range = max(highs[-period:]) - min(lows[-period:])

    if high_low_range == 0:
        return 50.0  # Neutral

    chop = 100 * math.log10(atr_sum / high_low_range) / math.log10(period)
    return max(0, min(100, chop))  # Clamp to 0-100
```

**Pros:** Specifically designed for consolidation detection, predictive of breakouts
**Cons:** Doesn't indicate direction, needs confirmation

#### D. RSI Regime Zones

RSI can indicate regime through extended time in zones:

```python
def classify_rsi_regime(rsi_history, lookback=20):
    avg_rsi = sum(rsi_history[-lookback:]) / lookback

    if avg_rsi > 60:
        return "BULLISH"
    elif avg_rsi < 40:
        return "BEARISH"
    else:
        return "NEUTRAL"
```

### 1.2 Statistical Methods

#### A. Hidden Markov Models (HMM)

HMMs are probabilistic models that assume the market exists in "hidden" states that influence observable data (returns, volatility).

**Features for HMM Training:**
1. **Log Returns** - `ln(close_t / close_{t-1})`
2. **Realized Volatility** - Rolling standard deviation of returns
3. **Volume Ratio** - Current volume / Average volume

**Typical 3-State Model:**
- **State 0:** Low volatility, positive drift (Bull)
- **State 1:** Low volatility, negative drift (Bear)
- **State 2:** High volatility, uncertain direction (Volatile)

**Implementation with hmmlearn:**
```python
from hmmlearn.hmm import GaussianHMM
import numpy as np

def train_regime_hmm(returns, volatility, n_states=3):
    features = np.column_stack([returns, volatility])

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=1000,
        random_state=42
    )
    model.fit(features)

    return model

def predict_regime(model, features):
    return model.predict(features)[-1]
```

**State Interpretation:**
```python
def label_states(model):
    # Sort states by mean return
    mean_returns = [model.means_[i][0] for i in range(model.n_components)]
    sorted_states = sorted(range(len(mean_returns)), key=lambda i: mean_returns[i])

    labels = {}
    labels[sorted_states[0]] = "BEAR"
    labels[sorted_states[-1]] = "BULL"
    labels[sorted_states[1]] = "SIDEWAYS" if len(sorted_states) == 3 else "NEUTRAL"

    return labels
```

**Pros:** Probabilistic, adapts to market structure, transition probabilities
**Cons:** Requires training data, computationally expensive, may overfit

#### B. Regime Detection via Clustering (Gaussian Mixture Models)

```python
from sklearn.mixture import GaussianMixture

def cluster_regimes(features, n_regimes=3):
    gmm = GaussianMixture(n_components=n_regimes, random_state=42)
    gmm.fit(features)
    labels = gmm.predict(features)
    return labels, gmm
```

### 1.3 Multi-Timeframe Analysis (MTF)

MTF analysis provides confluence-based confidence by checking alignment across timeframes.

**Timeframe Hierarchy:**
```
4H  →  Strategic direction (major trend)
1H  →  Tactical direction (intermediate)
15m →  Operational direction (short-term)
5m  →  Execution timing
1m  →  Entry precision
```

**Confluence Scoring:**
```python
def calculate_mtf_score(regimes: dict) -> dict:
    """
    regimes = {
        '1m': 'BULL', '5m': 'BULL', '15m': 'SIDEWAYS',
        '1h': 'BULL', '4h': 'BULL'
    }
    """
    weights = {'4h': 5, '1h': 4, '15m': 3, '5m': 2, '1m': 1}

    bull_score = sum(weights[tf] for tf, r in regimes.items() if r == 'BULL')
    bear_score = sum(weights[tf] for tf, r in regimes.items() if r == 'BEAR')
    total_weight = sum(weights.values())

    bull_pct = bull_score / total_weight * 100
    bear_pct = bear_score / total_weight * 100

    if bull_pct > 70:
        return {'regime': 'STRONG_BULL', 'confidence': bull_pct}
    elif bull_pct > 50:
        return {'regime': 'BULL', 'confidence': bull_pct}
    elif bear_pct > 70:
        return {'regime': 'STRONG_BEAR', 'confidence': bear_pct}
    elif bear_pct > 50:
        return {'regime': 'BEAR', 'confidence': bear_pct}
    else:
        return {'regime': 'SIDEWAYS', 'confidence': 100 - bull_pct - bear_pct}
```

---

## 2. Composite Scoring System (Recommended Approach)

### 2.1 Design Philosophy

Rather than relying on a single indicator, combine multiple indicators into a weighted composite score. This reduces false signals and provides confidence levels.

### 2.2 Indicator Weights

| Indicator | Weight | Rationale |
|-----------|--------|-----------|
| ADX/+DI/-DI | 25% | Primary trend strength & direction |
| Choppiness Index | 20% | Consolidation confirmation |
| MA Alignment | 20% | Price structure analysis |
| RSI Regime Zone | 15% | Momentum confirmation |
| Volume Analysis | 10% | Participation confirmation |
| External Sentiment | 10% | Market-wide context |

### 2.3 Scoring Algorithm

```python
class CompositeRegimeDetector:
    def __init__(self):
        self.weights = {
            'adx': 0.25,
            'chop': 0.20,
            'ma': 0.20,
            'rsi': 0.15,
            'volume': 0.10,
            'sentiment': 0.10
        }

    def calculate_regime(self, indicators: dict) -> dict:
        """
        indicators = {
            'adx': {'value': 28, 'plus_di': 25, 'minus_di': 18},
            'chop': 42.5,
            'ma': {'price': 2.30, 'sma_20': 2.28, 'sma_50': 2.25, 'sma_200': 2.15},
            'rsi': 58,
            'volume_ratio': 1.2,
            'fear_greed': 55
        }
        """
        scores = {}

        # ADX Score (-1 to +1, negative = bear, positive = bull)
        adx = indicators['adx']
        if adx['value'] < 20:
            scores['adx'] = 0  # No trend
        else:
            direction = 1 if adx['plus_di'] > adx['minus_di'] else -1
            strength = min(adx['value'] / 50, 1.0)  # Cap at 50
            scores['adx'] = direction * strength

        # Choppiness Score (-1 = choppy, +1 = trending)
        chop = indicators['chop']
        if chop > 61.8:
            scores['chop'] = -1.0  # Very choppy
        elif chop < 38.2:
            scores['chop'] = 1.0   # Trending
        else:
            scores['chop'] = (50 - chop) / 23.6  # Linear interpolation

        # MA Alignment Score
        ma = indicators['ma']
        if ma['price'] > ma['sma_20'] > ma['sma_50'] > ma['sma_200']:
            scores['ma'] = 1.0
        elif ma['price'] < ma['sma_20'] < ma['sma_50'] < ma['sma_200']:
            scores['ma'] = -1.0
        elif ma['sma_50'] > ma['sma_200']:
            scores['ma'] = 0.5
        elif ma['sma_50'] < ma['sma_200']:
            scores['ma'] = -0.5
        else:
            scores['ma'] = 0.0

        # RSI Score
        rsi = indicators['rsi']
        scores['rsi'] = (rsi - 50) / 50  # -1 to +1

        # Volume Score (high volume confirms trend)
        vol_ratio = indicators['volume_ratio']
        scores['volume'] = min(max(vol_ratio - 1, -1), 1)  # -1 to +1

        # Sentiment Score
        fg = indicators['fear_greed']
        scores['sentiment'] = (fg - 50) / 50  # -1 to +1

        # Weighted composite
        composite = sum(scores[k] * self.weights[k] for k in scores)

        # Classify regime
        if composite > 0.4:
            regime = 'STRONG_BULL'
        elif composite > 0.15:
            regime = 'BULL'
        elif composite < -0.4:
            regime = 'STRONG_BEAR'
        elif composite < -0.15:
            regime = 'BEAR'
        else:
            regime = 'SIDEWAYS'

        confidence = abs(composite) * 100

        return {
            'regime': regime,
            'composite_score': composite,
            'confidence': min(confidence, 100),
            'component_scores': scores,
            'is_trending': abs(composite) > 0.15,
            'trend_direction': 'UP' if composite > 0 else 'DOWN' if composite < 0 else 'NONE'
        }
```

---

## 3. External Data Sources Research

### 3.1 Fear & Greed Index

**Source:** Alternative.me (Free, no registration)

**API Endpoint:** `https://api.alternative.me/fng/`

**Components:**
- Volatility (25%)
- Market Momentum/Volume (25%)
- Social Media (15%)
- Surveys (15%)
- Dominance (10%)
- Trends (10%)

**Interpretation:**
| Value | Classification | Regime Implication |
|-------|---------------|-------------------|
| 0-25 | Extreme Fear | Potential bottom, contrarian buy |
| 25-45 | Fear | Bearish sentiment |
| 45-55 | Neutral | No clear sentiment |
| 55-75 | Greed | Bullish sentiment |
| 75-100 | Extreme Greed | Potential top, contrarian sell |

**Implementation:**
```python
import aiohttp

async def fetch_fear_greed():
    async with aiohttp.ClientSession() as session:
        async with session.get('https://api.alternative.me/fng/') as resp:
            data = await resp.json()
            return {
                'value': int(data['data'][0]['value']),
                'classification': data['data'][0]['value_classification']
            }
```

### 3.2 Bitcoin Dominance

**Source:** Can be calculated from Kraken ticker data or fetched from CoinGecko

**Formula:**
```
BTC_Dominance = BTC_Market_Cap / Total_Crypto_Market_Cap
```

**Trading Implications:**
| BTC Dominance | Trend | Interpretation |
|--------------|-------|----------------|
| > 60% Rising | Up | Bitcoin season - rotate to BTC |
| > 60% Falling | Down | Early altcoin season starting |
| < 40% Falling | Down | Peak altcoin season - caution |
| < 40% Rising | Up | Capital returning to BTC |

**CoinGecko Free API:**
```python
async def fetch_btc_dominance():
    url = "https://api.coingecko.com/api/v3/global"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            data = await resp.json()
            return data['data']['market_cap_percentage']['btc']
```

### 3.3 CoinyBubble Sentiment API (Alternative Free Option)

**Endpoint:** Free public API, no registration
**Note:** Attribution requested for commercial use

---

## 4. Recommendations

### 4.1 Phased Implementation Approach

**Phase 1 - Core Rule-Based System:**
1. Implement ADX/+DI/-DI calculations
2. Add Choppiness Index
3. Create MA alignment scoring
4. Build composite scoring system

**Phase 2 - Multi-Timeframe Analysis:**
1. Extend candle storage to multiple timeframes
2. Calculate indicators per timeframe
3. Implement confluence scoring

**Phase 3 - External Data Integration:**
1. Add Fear & Greed API client
2. Add BTC Dominance tracking
3. Integrate into composite score

**Phase 4 - Strategy Integration:**
1. Create parameter mapping tables
2. Implement regime-based config routing
3. Add regime-based position sizing

**Phase 5 - Advanced (Optional):**
1. Train HMM on historical data
2. Add machine learning regime classification
3. Backtest and optimize weights

### 4.2 Key Design Decisions

1. **Smoothing:** Apply 3-5 bar smoothing to regime output to prevent rapid switching
2. **Hysteresis:** Require confirmation (e.g., 3 consecutive readings) before regime change
3. **Confidence Thresholds:** Only act on high-confidence (>60%) regime classifications
4. **Fallback:** When uncertain, default to conservative "SIDEWAYS" classification

### 4.3 Risk Considerations

- **Lagging nature:** All technical indicators lag price; regime changes detected after they occur
- **Whipsaw risk:** Frequent regime changes can cause over-trading
- **Overfitting:** Tuned parameters may not generalize to future markets
- **External API reliability:** Fear & Greed API may have downtime

---

## 5. References

1. [LuxAlgo - Market Regimes Explained](https://www.luxalgo.com/blog/market-regimes-explained-build-winning-trading-strategies/)
2. [LuxAlgo - Choppiness Index](https://www.luxalgo.com/blog/choppiness-index-quantifying-consolidation/)
3. [QuantStart - HMM Regime Detection](https://www.quantstart.com/articles/market-regime-detection-using-hidden-markov-models-in-qstrader/)
4. [QuantInsti - Regime Adaptive Trading](https://blog.quantinsti.com/regime-adaptive-trading-python/)
5. [Charles Schwab - ADX and RSI](https://www.schwab.com/learn/story/spot-and-stick-to-trends-with-adx-and-rsi)
6. [Alternative.me Fear & Greed API](https://alternative.me/crypto/fear-and-greed-index/)
7. [CoinGecko API Documentation](https://www.coingecko.com/en/api)
8. [Medium - Creating Composite Indicators](https://medium.com/@corinneroosen/create-a-composite-indicator-for-algorithmic-trading-in-python-0a81920f905b)
9. [Medium - Multi-Timeframe Adaptive Strategy](https://medium.com/@FMZQuant/multi-timeframe-adaptive-market-regime-quantitative-trading-strategy-1b16309ddabb)

---

*Version: 1.0.0 | Created: 2025-12-15*
