# Whale Sentiment Strategy Master Plan v1.0

**Target Platform:** WebSocket Paper Tester v1.0.2+
**Supported Pairs:** XRP/USDT, BTC/USDT, XRP/BTC (Kraken)
**Primary Timeframe:** 1h (with 5m candle adaptation for tick-level execution)
**Style:** Contrarian sentiment + volume spike detection (whale proxy)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Research Findings](#2-research-findings)
   - 2.1 [Whale Tracking Fundamentals](#21-whale-tracking-fundamentals)
   - 2.2 [Sentiment Analysis Methodology](#22-sentiment-analysis-methodology)
   - 2.3 [RSI as Sentiment Indicator](#23-rsi-as-sentiment-indicator)
   - 2.4 [Contrarian Investing Theory](#24-contrarian-investing-theory)
3. [Source Code Analysis](#3-source-code-analysis)
   - 3.1 [Legacy Implementation Summary](#31-legacy-implementation-summary)
   - 3.2 [Component Mapping](#32-component-mapping)
   - 3.3 [WebSocket Adaptation Requirements](#33-websocket-adaptation-requirements)
4. [Pair-Specific Analysis](#4-pair-specific-analysis)
   - 4.1 [XRP/USDT](#41-xrpusdt)
   - 4.2 [BTC/USDT](#42-btcusdt)
   - 4.3 [XRP/BTC](#43-xrpbtc)
5. [Recommended Approach](#5-recommended-approach)
   - 5.1 [Core Architecture](#51-core-architecture)
   - 5.2 [Signal Generation Logic](#52-signal-generation-logic)
   - 5.3 [Risk Management Framework](#53-risk-management-framework)
6. [Known Pitfalls and Mitigations](#6-known-pitfalls-and-mitigations)
7. [Development Plan](#7-development-plan)
8. [Compliance Checklist](#8-compliance-checklist)

---

## 1. Executive Summary

The Whale Sentiment Strategy combines institutional activity detection (via volume spike analysis) with market sentiment indicators (RSI, price deviation) to identify contrarian trading opportunities. The strategy operates on the principle that extreme market fear or greed, particularly when coupled with large-holder activity, often precedes price reversals.

### Key Adaptations for ws_paper_tester

| Aspect | Legacy Implementation | ws_paper_tester Adaptation |
|--------|----------------------|---------------------------|
| Timeframe | 1h candles | 5m candles (12x aggregation proxy) |
| Whale Detection | External APIs (Whale Alert, Glassnode) | Volume spike proxy (2x-3x average) |
| Sentiment Data | Social APIs (Twitter, Reddit) | RSI + Price deviation proxy |
| Execution | Batch hourly signals | Tick-based (100ms) with cooldowns |
| External Data | Required | Optional (proxy-only mode functional) |

### Strategy Viability Assessment

**Strengths:**
- Contrarian approach captures panic/euphoria extremes
- Volume spikes provide actionable whale-proxy signals without external APIs
- RSI divergence adds confirmation layer
- Well-suited for volatile crypto markets

**Challenges:**
- Volume spike false positives (wash trading, exchange manipulation)
- Contrarian entries inherently counter-trend (requires tight risk management)
- 5m candles are noisier than hourly data
- Cross-pair correlation requires careful exposure management

**Recommendation:** Implement with enhanced false-positive filtering and strict position limits.

---

## 2. Research Findings

### 2.1 Whale Tracking Fundamentals

#### Academic Foundation

Whale tracking in cryptocurrency markets is grounded in **Smart Money Theory** - the premise that large holders (institutional investors, early adopters, experienced traders) possess informational advantages and their collective behavior can signal future price movements.

Key research findings from 2025:
- Whale deposits to exchanges surged 100%+ since January 2023, indicating strategic positioning
- In Ethereum, large holders tend to accumulate prior to price increases while retail reduces holdings
- Deep learning models trained on whale-alert data outperform traditional methods in predicting volatility spikes

#### Volume Spike as Whale Proxy

Without direct access to on-chain data, volume spikes serve as the primary whale activity indicator:

```
Volume Ratio = Current Volume / MA(Volume, 24 periods)
Whale Activity Detected = Volume Ratio >= 2.0x
```

**Detection Accuracy:**
- Volume spikes 300-1000% above average with minimal preceding news strongly indicate institutional activity
- Spikes correlated with price moves suggest directional intent (accumulation vs distribution)
- Spikes without price movement may indicate wash trading or manipulation

**Red Flags for False Positives:**
- Sudden spikes in low-liquidity environments
- Spikes during off-hours without corresponding news
- Repetitive spike patterns suggesting automated manipulation

#### Whale Threshold Calibration

Based on 2025 research:

| Asset | Whale Threshold | Volume Multiple | Notes |
|-------|----------------|-----------------|-------|
| BTC | 100+ BTC / $1M+ | 2.5x | More stable, higher threshold |
| XRP | 1M+ XRP / $1M+ | 2.0x | Higher retail participation |
| XRP/BTC | 500K+ XRP equivalent | 3.0x | Lower liquidity, stricter filter |

### 2.2 Sentiment Analysis Methodology

#### Fear & Greed Index Methodology

The Crypto Fear & Greed Index (Alternative.me / CoinMarketCap) uses weighted components:

| Component | Weight | Proxy Available |
|-----------|--------|-----------------|
| Volatility | 25% | Yes (price std dev) |
| Market Momentum/Volume | 25% | Yes (price deviation from MA) |
| Social Media Sentiment | 15% | No |
| Surveys | 15% | No |
| Bitcoin Dominance | 10% | Partial (BTC/pair correlation) |
| Search Trends | 10% | No |

**Proxy Approach (50% of Index):**
- Volatility proxy: 20-period price standard deviation
- Momentum proxy: Distance from 20-period SMA as percentage
- Combined proxy sentiment score: 0-100 scale mimicking F&G Index

**2025 Performance:**
- February 2025: Index dropped to 10 (extreme fear), followed by 25% rebound in weeks
- October 2025: Index at 22 preceded 70% BTC surge over 6 months
- Bitcoin ATH at $126,080 (October 2025) registered only 71 (Greed, not Extreme Greed) - suggesting institutional influence dampens extremes

#### Price Deviation Fear/Greed Proxy

```python
# Fear/Greed from price deviation
from_high_pct = (current_price - recent_high) / recent_high * 100
from_low_pct = (current_price - recent_low) / recent_low * 100

# Thresholds
Fear = from_high_pct <= -5%
Greed = from_low_pct >= +5%
```

### 2.3 RSI as Sentiment Indicator

#### RSI: Momentum vs Sentiment

The RSI is technically a **momentum oscillator** but serves dual purposes:

1. **Momentum Function:** Measures speed/magnitude of price changes
2. **Sentiment Function:** Extreme values (< 30 oversold, > 70 overbought) indicate emotional market states

**Key Distinction:**
- RSI > 50: Bullish sentiment/momentum
- RSI < 50: Bearish sentiment/momentum
- RSI < 30: Extreme fear (panic selling)
- RSI > 70: Extreme greed (FOMO buying)

#### Crypto-Specific RSI Considerations

From 2025 research:
- Crypto assets can remain overbought/oversold longer than traditional markets
- Adjusted thresholds recommended: 25/75 for extreme, 40/60 for moderate
- RSI divergences (price vs RSI) are particularly effective for reversal detection
- Failure swings (RSI fails to make new high/low) signal momentum exhaustion

**Best Practice:**
- Use RSI for sentiment classification, not standalone signals
- Combine with volume analysis for confirmation
- Use divergence detection for higher-confidence entries

### 2.4 Contrarian Investing Theory

#### Behavioral Finance Foundation

Contrarian investing is grounded in the observation that markets overreact to news and sentiment, creating mispricings:

- **Herding Behavior:** Retail investors follow crowds, amplifying price moves
- **Availability Bias:** Recent events disproportionately influence decisions
- **Loss Aversion:** Fear of losses causes panic selling at bottoms
- **FOMO:** Fear of missing out causes euphoric buying at tops

#### Smart Money Divergence

The most powerful contrarian signal occurs when:
1. Retail sentiment is extreme (fear or greed)
2. Large holders are moving opposite to retail

2025 Examples:
- **XRP (Sept-Nov 2025):** Whales accumulated 340M tokens while retail panic-sold as price dropped from $3.67 to $2.20
- **February 2025 Bybit breach:** Retail panic-sold, institutional buyers viewed as contrarian opportunity

#### Contrarian Mode Implementation

```python
# Contrarian Mode (default)
if contrarian_mode:
    if sentiment == 'fear':
        signal = 'bullish'  # Buy the fear
    elif sentiment == 'greed':
        signal = 'bearish'  # Sell the greed

# Momentum Mode (alternative)
else:
    if sentiment == 'fear':
        signal = 'bearish'  # Follow the trend
    elif sentiment == 'greed':
        signal = 'bullish'  # Follow the trend
```

---

## 3. Source Code Analysis

### 3.1 Legacy Implementation Summary

The legacy `WhaleSentiment` class (`src/strategies/whale_sentiment/strategy.py`) implements:

#### Configuration Parameters

```python
# Whale Detection
whale_threshold_btc = 100          # 100+ BTC
whale_threshold_xrp = 1_000_000    # 1M+ XRP
whale_threshold_usd = 1_000_000    # $1M+

# Sentiment Thresholds
bullish_threshold = 0.60           # >60% = bullish
bearish_threshold = 0.40           # <40% = bearish

# Volume Spike Detection
volume_spike_mult = 2.0            # 2x average volume
volume_window = 24                 # 24 bars (hourly)

# Fear/Greed Price Deviation
fear_deviation = -0.05             # -5% from recent high
greed_deviation = 0.05             # +5% from recent low

# RSI Settings
rsi_period = 14
rsi_fear = 25                      # Extreme fear
rsi_greed = 75                     # Extreme greed

# Contrarian Mode
contrarian_mode = True             # Buy fear, sell greed

# Position Sizing
base_size_pct = 0.10               # 10% per trade
```

#### Core Signal Generation Logic

1. **Volume Spike Detection:** Identify 2x+ volume as whale activity proxy
2. **Fear/Greed Calculation:** Price deviation from recent high/low
3. **RSI Sentiment:** Map RSI to sentiment categories
4. **External Data (Optional):** Parse whale alerts and social sentiment
5. **Composite Signal:** Weighted aggregation of all sources

#### Composite Signal Weights

| Signal Source | Weight | Notes |
|---------------|--------|-------|
| Volume Spike | 0.70 | Strong whale proxy |
| Fear/Greed (price) | 0.60 | Moderate sentiment |
| RSI Extreme | 0.75 | Strong sentiment |
| External Whale Data | 0.85 | Highest weight (when available) |
| External Sentiment | Variable | Based on score |

#### Signal Confidence Threshold

- Minimum confidence: **0.55** for signal generation
- Maximum confidence cap: **0.90** (never 100% confident)

### 3.2 Component Mapping

Map legacy implementation to ws_paper_tester modular structure:

| Module | Legacy Component | New File | Purpose |
|--------|-----------------|----------|---------|
| `config.py` | `__init__` parameters | `whale_sentiment/config.py` | Strategy metadata, defaults, enums |
| `indicators.py` | `_calculate_rsi`, `_detect_volume_spike`, `_calculate_fear_greed_proxy` | `whale_sentiment/indicators.py` | All indicator calculations |
| `signal.py` | `generate_signals`, `_generate_composite_signal` | `whale_sentiment/signal.py` | Main signal generation |
| `regimes.py` | Implicit sentiment classification | `whale_sentiment/regimes.py` | Sentiment regime classification |
| `risk.py` | Position sizing, confidence capping | `whale_sentiment/risk.py` | Risk management checks |
| `exits.py` | N/A (no explicit exits) | `whale_sentiment/exits.py` | Exit signal logic |
| `lifecycle.py` | State tracking | `whale_sentiment/lifecycle.py` | on_start, on_fill, on_stop |

#### New Components Required

1. **Validation Module:** Config validation (following wavetrend pattern)
2. **Enums:** SentimentZone, WhaleSignal, SignalDirection, RejectionReason
3. **Circuit Breaker:** Post-contrarian-loss protection
4. **Cross-Pair Correlation:** XRP-BTC correlation management

### 3.3 WebSocket Adaptation Requirements

#### Timeframe Adaptation

Legacy uses 1h candles; ws_paper_tester provides 1m and 5m candles:

| Parameter | 1h Original | 5m Adapted | Scaling Factor |
|-----------|------------|------------|----------------|
| `volume_window` | 24 | 288 | 12x (24 * 12) |
| `rsi_period` | 14 | 14 | No change |
| `recent_high_low` | 20 | 48 | ~2.5x (4h window) |
| `min_candle_buffer` | 50 | 300 | 6x |

**Warmup Calculation:**
- min_candle_buffer: 300 candles
- candle_timeframe: 5 minutes
- Warmup time: 300 * 5 = 1500 minutes = **25 hours minimum**

#### Tick-Based Execution Adaptation

```python
# Legacy: Evaluate once per hour
signal = strategy.generate_signals(data)

# ws_paper_tester: Evaluate every tick (100ms) with cooldown
def generate_signal(data, config, state):
    # Cooldown check (minimum 60 seconds between signals)
    if elapsed_since_last_signal < config['cooldown_seconds']:
        return None

    # Calculate indicators from candles (not ticks)
    indicators = calculate_indicators(data.candles_5m)

    # Generate signal if conditions met
    ...
```

#### Data Access Pattern

```python
# ws_paper_tester DataSnapshot access
def generate_signal(data: DataSnapshot, config, state):
    # 5-minute candles (preferred)
    candles = data.candles_5m.get(symbol, ())

    # Orderbook for volume validation
    ob = data.orderbooks.get(symbol)

    # Recent trades for flow analysis
    trades = data.trades.get(symbol, ())

    # Current price
    price = data.prices.get(symbol)
```

---

## 4. Pair-Specific Analysis

### 4.1 XRP/USDT

#### Market Characteristics (2025)

| Metric | Value | Implication |
|--------|-------|-------------|
| Daily Volatility | ~5.1% | Good for contrarian plays |
| Upbit Dominance | 16.87% | Korean retail influence |
| Whale Threshold | 1M+ XRP | ~$2M at current prices |
| Active Addresses | 31K (from 581K peak) | Declining retail, whale-dominated |
| ETF Inflows | $887M+ | Growing institutional presence |

#### Volume Pattern Analysis

From 2025 research:
- Whales accumulated 340M XRP (Sept-Nov 2025) while retail panic-sold
- A 75M XRP sell-off triggered 15% price drops, exposing thin order books
- Retail vs institutional divergence is pronounced

#### XRP/USDT Configuration

```python
SYMBOL_CONFIGS['XRP/USDT'] = {
    # Volume Spike (whale proxy)
    'volume_spike_mult': 2.0,          # Standard threshold
    'volume_window': 288,              # 24h in 5m candles

    # Sentiment Zones
    'rsi_fear': 25,                    # Extreme fear
    'rsi_greed': 75,                   # Extreme greed
    'fear_deviation_pct': -5.0,        # -5% from high
    'greed_deviation_pct': 5.0,        # +5% from low

    # Position Sizing
    'position_size_usd': 25.0,         # Base size
    'max_position_per_symbol_usd': 75.0,

    # Risk Management
    'stop_loss_pct': 2.5,              # Wider for contrarian
    'take_profit_pct': 5.0,            # 2:1 R:R ratio

    # Cooldown
    'cooldown_seconds': 120,           # 2 minutes (more conservative)
}
```

### 4.2 BTC/USDT

#### Market Characteristics (2025)

| Metric | Value | Implication |
|--------|-------|-------------|
| Daily Volatility | ~1.64% | Lower than XRP |
| ATH | $126,080 (Oct 2025) | Strong institutional support |
| Whale Threshold | 100+ BTC | ~$8M+ at current prices |
| ETF Influence | Dominant | Institutional flows drive price |
| Fear/Greed at ATH | 71 | Dampened extremes |

#### Volume Pattern Analysis

- Institutional activity signatures more subtle (custody solutions, OTC)
- Volume spike accuracy for whale detection: MEDIUM
- Stealth accumulation patterns require longer observation windows

#### BTC/USDT Configuration

```python
SYMBOL_CONFIGS['BTC/USDT'] = {
    # Volume Spike (whale proxy)
    'volume_spike_mult': 2.5,          # Higher threshold (more noise)
    'volume_window': 288,              # 24h in 5m candles

    # Sentiment Zones (wider due to institutional dampening)
    'rsi_fear': 22,                    # More extreme fear
    'rsi_greed': 78,                   # More extreme greed
    'fear_deviation_pct': -7.0,        # Larger deviation required
    'greed_deviation_pct': 7.0,

    # Position Sizing (larger due to lower volatility)
    'position_size_usd': 50.0,
    'max_position_per_symbol_usd': 100.0,

    # Risk Management (tighter due to lower volatility)
    'stop_loss_pct': 1.5,
    'take_profit_pct': 3.0,            # 2:1 R:R ratio

    # Cooldown
    'cooldown_seconds': 180,           # 3 minutes (less frequent)
}
```

### 4.3 XRP/BTC

#### Market Characteristics (2025)

| Metric | Value | Implication |
|--------|-------|-------------|
| Liquidity | 7-10x lower than USD pairs | Higher slippage risk |
| Correlation | Decoupling observed | XRP outperforming BTC |
| Key Resistance | 0.0000215 BTC | Critical breakout level |
| Volatility | Higher than either base | Both assets contribute |

#### Cross-Pair Sentiment Divergence

From research:
- XRP/BTC fell 97% from 2018 peak to November 2024
- Double bottom formation signaled reversal
- "Decoupling" narrative: XRP strength independent of BTC

#### XRP/BTC Configuration

```python
SYMBOL_CONFIGS['XRP/BTC'] = {
    # Volume Spike (stricter due to low liquidity)
    'volume_spike_mult': 3.0,          # Higher threshold
    'volume_window': 288,

    # Sentiment Zones (ratio pair dynamics)
    'rsi_fear': 28,
    'rsi_greed': 72,
    'fear_deviation_pct': -8.0,        # Larger moves
    'greed_deviation_pct': 8.0,

    # Position Sizing (smaller due to liquidity)
    'position_size_usd': 15.0,
    'max_position_per_symbol_usd': 40.0,

    # Risk Management (wider due to volatility)
    'stop_loss_pct': 3.0,
    'take_profit_pct': 6.0,            # 2:1 R:R ratio

    # Cooldown
    'cooldown_seconds': 240,           # 4 minutes
}
```

---

## 5. Recommended Approach

### 5.1 Core Architecture

#### Module Structure

```
ws_paper_tester/strategies/whale_sentiment/
    __init__.py              # Public API exports
    config.py                # Metadata, CONFIG, SYMBOL_CONFIGS, enums
    indicators.py            # RSI, volume spike, fear/greed, composite
    signal.py                # generate_signal, _evaluate_symbol
    regimes.py               # Sentiment regime classification
    risk.py                  # Circuit breaker, position limits, correlation
    exits.py                 # Exit signal logic
    lifecycle.py             # on_start, on_fill, on_stop
    validation.py            # Config validation
```

#### Enums

```python
class SentimentZone(Enum):
    """Market sentiment classification."""
    EXTREME_FEAR = auto()      # RSI < 25, large drop from high
    FEAR = auto()              # RSI < 40, moderate drop
    NEUTRAL = auto()           # RSI 40-60, range-bound
    GREED = auto()             # RSI > 60, moderate rise
    EXTREME_GREED = auto()     # RSI > 75, large rise from low

class WhaleSignal(Enum):
    """Volume spike classification."""
    ACCUMULATION = auto()      # Volume spike + price up
    DISTRIBUTION = auto()      # Volume spike + price down
    NEUTRAL = auto()           # No significant volume

class SignalDirection(Enum):
    """Composite signal direction."""
    BULLISH = auto()
    BEARISH = auto()
    NEUTRAL = auto()

class RejectionReason(Enum):
    """Signal rejection tracking."""
    CIRCUIT_BREAKER = "circuit_breaker"
    TIME_COOLDOWN = "time_cooldown"
    WARMING_UP = "warming_up"
    NO_PRICE_DATA = "no_price_data"
    MAX_POSITION = "max_position"
    INSUFFICIENT_SIZE = "insufficient_size"
    CORRELATION_LIMIT = "correlation_limit"
    NO_VOLUME_SPIKE = "no_volume_spike"
    NEUTRAL_SENTIMENT = "neutral_sentiment"
    INSUFFICIENT_CONFIDENCE = "insufficient_confidence"
    VOLUME_FALSE_POSITIVE = "volume_false_positive"
```

### 5.2 Signal Generation Logic

#### Signal Flow

```
1. Check circuit breaker
2. Check cooldown
3. For each symbol:
   a. Check warmup (min candles)
   b. Calculate indicators:
      - Volume spike detection
      - RSI calculation
      - Fear/greed proxy
      - Trade flow analysis
   c. Check existing positions for exits
   d. Classify sentiment regime
   e. Check entry conditions:
      - Volume spike present?
      - Sentiment extreme?
      - Contrarian direction aligns?
   f. Generate composite confidence
   g. Check position limits
   h. Check correlation exposure
   i. Generate Signal or None
```

#### Entry Logic

```python
# Contrarian Entry Conditions
if contrarian_mode:
    # Long Entry
    if (whale_signal == 'accumulation' or whale_signal == 'neutral') and \
       sentiment_zone in [SentimentZone.FEAR, SentimentZone.EXTREME_FEAR]:
        action = 'buy'
        confidence = calculate_composite_confidence(...)
        if confidence >= 0.55:
            return Signal(action='buy', ...)

    # Short Entry
    if (whale_signal == 'distribution' or whale_signal == 'neutral') and \
       sentiment_zone in [SentimentZone.GREED, SentimentZone.EXTREME_GREED]:
        action = 'short'
        confidence = calculate_composite_confidence(...)
        if confidence >= 0.55:
            return Signal(action='short', ...)
```

#### Composite Confidence Calculation

```python
def calculate_composite_confidence(
    whale_signal: WhaleSignal,
    sentiment_zone: SentimentZone,
    rsi: float,
    volume_ratio: float,
    trade_flow_imbalance: float,
    config: Dict
) -> Tuple[float, List[str]]:
    """
    Calculate weighted confidence from multiple signals.

    Weights:
    - Volume spike (whale proxy): 0.30
    - Sentiment zone (RSI): 0.25
    - Sentiment zone (price dev): 0.20
    - Trade flow confirmation: 0.15
    - Divergence bonus: 0.10

    Maximum confidence capped at 0.90.
    """
    confidence = 0.0
    reasons = []

    # Volume spike contribution
    if whale_signal in [WhaleSignal.ACCUMULATION, WhaleSignal.DISTRIBUTION]:
        vol_conf = min(0.30, 0.15 + (volume_ratio - 2.0) * 0.03)
        confidence += vol_conf
        reasons.append(f"Volume {volume_ratio:.1f}x ({whale_signal.name.lower()})")

    # RSI sentiment contribution
    if sentiment_zone == SentimentZone.EXTREME_FEAR:
        confidence += 0.25
        reasons.append(f"RSI extreme fear ({rsi:.0f})")
    elif sentiment_zone == SentimentZone.EXTREME_GREED:
        confidence += 0.25
        reasons.append(f"RSI extreme greed ({rsi:.0f})")
    elif sentiment_zone in [SentimentZone.FEAR, SentimentZone.GREED]:
        confidence += 0.15
        reasons.append(f"RSI {sentiment_zone.name.lower()} ({rsi:.0f})")

    # Trade flow confirmation
    if abs(trade_flow_imbalance) > 0.10:
        flow_conf = min(0.15, abs(trade_flow_imbalance) * 0.5)
        confidence += flow_conf
        reasons.append(f"Trade flow {trade_flow_imbalance:+.2f}")

    # Cap confidence
    max_conf = config.get('max_confidence', 0.90)
    confidence = min(confidence, max_conf)

    return confidence, reasons
```

### 5.3 Risk Management Framework

#### Circuit Breaker

Contrarian strategies are prone to multiple consecutive losses during trending markets. Enhanced circuit breaker:

```python
# Configuration
'max_consecutive_losses': 2,          # Stricter for contrarian
'circuit_breaker_minutes': 45,        # Longer cooldown
'circuit_breaker_pnl_threshold': -3.0, # Also trigger on % loss
```

#### Position Limits

```python
# Total exposure limits
'max_position_usd': 150.0,            # Total across all pairs
'max_position_per_symbol_usd': 75.0,  # Per symbol
'max_total_long_exposure': 100.0,     # Total long exposure
'max_total_short_exposure': 75.0,     # Lower short exposure (squeeze risk)

# Size adjustments
'short_size_multiplier': 0.75,        # Reduce short sizes
'high_correlation_size_mult': 0.60,   # Reduce when correlated
```

#### Contrarian-Specific Risk

```python
# Wider stops for counter-trend entries
'stop_loss_pct': 2.5,                 # Wider than momentum strategies
'take_profit_pct': 5.0,               # Maintain 2:1 R:R

# Trailing stop option (activate after 50% of TP reached)
'use_trailing_stop': True,
'trailing_stop_activation_pct': 50.0,  # Activate at 2.5% profit
'trailing_stop_distance_pct': 1.0,     # Trail 1% below peak
```

#### Volume Spike False Positive Filter

```python
def validate_volume_spike(
    volume_ratio: float,
    price_change_pct: float,
    spread_pct: float,
    trade_count: int,
    config: Dict
) -> Tuple[bool, str]:
    """
    Filter potential false positive volume spikes.

    Red flags:
    - Volume spike without price movement
    - Extremely wide spread (manipulation)
    - Very few trades (single large order vs distributed)
    """
    # No price movement with volume spike = suspicious
    if volume_ratio > 3.0 and abs(price_change_pct) < 0.1:
        return False, "volume_without_price_move"

    # Wide spread during spike = thin book
    if spread_pct > config.get('max_spread_pct', 0.5):
        return False, "wide_spread"

    # Too few trades = single large order (less predictive)
    if trade_count < config.get('min_spike_trades', 20):
        return False, "insufficient_trade_count"

    return True, "valid"
```

---

## 6. Known Pitfalls and Mitigations

### 6.1 Volume Spike False Positives

**Pitfall:** Exchange wash trading, market manipulation, or single large orders can create false volume signals.

**Mitigations:**
1. Require price movement correlation with volume spike
2. Check trade count (distributed vs concentrated)
3. Validate spread remains reasonable
4. Use trade flow imbalance as confirmation

### 6.2 Contrarian Trap (Catching Falling Knives)

**Pitfall:** Buying during fear can result in continued downside in trending markets.

**Mitigations:**
1. Strict circuit breaker (2 consecutive losses = pause)
2. Require whale accumulation signal, not just fear
3. Wider stop losses to account for trend continuation
4. Reduce position size during confirmed downtrends

### 6.3 Sentiment Divergence

**Pitfall:** Proxy indicators may diverge from actual market sentiment.

**Mitigations:**
1. Use multiple proxy signals (RSI + price deviation + trade flow)
2. Require minimum confidence threshold (0.55)
3. Log sentiment mismatches for analysis
4. Consider external data integration path

### 6.4 Cross-Pair Correlation Breakdown

**Pitfall:** XRP-BTC correlation can break down during altcoin seasons.

**Mitigations:**
1. Calculate real-time rolling correlation
2. Block same-direction trades when correlation > 0.85
3. Reduce position size when correlation is high
4. Track correlation in indicators for analysis

### 6.5 Warmup Period Issues

**Pitfall:** 25-hour warmup may miss early signals.

**Mitigations:**
1. Pre-load historical candle data if available
2. Use reduced confidence during early warmup
3. Clearly log warmup status in indicators

---

## 7. Development Plan

### Phase 1: Foundation

**Deliverables:**
- `config.py`: Strategy metadata, CONFIG, SYMBOL_CONFIGS, enums
- `validation.py`: Configuration validation with blocking errors
- `lifecycle.py`: State initialization, on_fill tracking, on_stop summary

**Dependencies:** None (parallel start)

### Phase 2: Indicators

**Deliverables:**
- `indicators.py`: RSI, volume spike, fear/greed proxy, composite confidence
- Volume spike validation (false positive filter)
- Trade flow analysis integration

**Dependencies:** Phase 1 (enums from config.py)

### Phase 3: Signal Generation

**Deliverables:**
- `signal.py`: Main generate_signal function
- Entry logic with contrarian mode
- Rejection tracking

**Dependencies:** Phase 1, Phase 2

### Phase 4: Risk Management

**Deliverables:**
- `risk.py`: Circuit breaker, position limits, correlation checks
- `exits.py`: Stop loss, take profit, sentiment reversal exits

**Dependencies:** Phase 1, Phase 2

### Phase 5: Regimes and Refinement

**Deliverables:**
- `regimes.py`: Sentiment regime classification
- Session awareness (optional)
- Performance optimization

**Dependencies:** All previous phases

### Phase 6: Integration and Testing

**Deliverables:**
- `__init__.py`: Public API exports
- Unit tests for all modules
- Integration test with ws_paper_tester
- Documentation updates

**Dependencies:** All previous phases

---

## 8. Compliance Checklist

### vs strategy-development-guide.md v1.0

| Requirement | Status | Notes |
|-------------|--------|-------|
| `STRATEGY_NAME` defined | Planned | `whale_sentiment` |
| `STRATEGY_VERSION` defined | Planned | `1.0.0` |
| `SYMBOLS` list defined | Planned | `["XRP/USDT", "BTC/USDT", "XRP/BTC"]` |
| `CONFIG` dict defined | Planned | Comprehensive defaults |
| `generate_signal()` function | Planned | Main entry point |
| `on_start()` optional | Planned | Config validation, state init |
| `on_fill()` optional | Planned | Position tracking, circuit breaker |
| `on_stop()` optional | Planned | Summary statistics |
| Signal uses USD sizing | Planned | Via config parameters |
| Stop loss below entry (long) | Planned | Automatic calculation |
| Stop loss above entry (short) | Planned | Automatic calculation |
| `state['indicators']` populated | Planned | All calculations logged |
| Bounded state buffers | Planned | Max 50 fills, 100 candle cache |
| Position limit checks | Planned | Total and per-symbol |
| Minimum trade size check | Planned | $5 USD minimum |
| Proper data null checks | Planned | All data.get() calls validated |

### Additional Requirements (from existing strategies)

| Requirement | Status | Notes |
|-------------|--------|-------|
| REC-001: Trade Flow Confirmation | Planned | Buy/sell volume imbalance |
| REC-002: Real Correlation Monitoring | Planned | Rolling correlation calculation |
| REC-006: Blocking R:R Validation | Planned | 2:1 minimum R:R |
| Circuit Breaker | Planned | 2 consecutive losses |
| Rejection Tracking | Planned | Per-symbol rejection counts |
| Warmup Documentation | Planned | 25 hours @ 5m candles |

---

## Appendix A: Research Sources

### Whale Tracking
- [AInvest: Whale Activity as Leading Indicator 2025](https://www.ainvest.com/news/whale-activity-leading-indicator-crypto-markets-insights-2025-chain-data-2512/)
- [WunderTrading: Crypto Whale Tracking Tools 2025](https://wundertrading.com/journal/en/learn/article/crypto-whale-tracking-tools)
- [Whale Alert](https://whale-alert.io/)

### Fear & Greed Index
- [CoinMarketCap Fear & Greed Index](https://coinmarketcap.com/charts/fear-and-greed-index/)
- [Alternative.me Crypto Fear & Greed](https://alternative.me/crypto/fear-and-greed-index/)
- [AInvest: Extreme Fear Contrarian Buy Signal](https://www.ainvest.com/news/extreme-fear-crypto-fear-greed-index-contrarian-buy-signal-2512/)

### Contrarian Investing
- [AInvest: Contrarian Opportunities in Crypto](https://www.ainvest.com/news/contrarian-opportunities-crypto-markets-decoding-retail-investor-sentiment-behavioral-finance-2510/)
- [CoinTelegraph: Smart Money Concepts](https://cointelegraph.com/news/smart-money-concepts-smc-in-crypto-trading-how-to-track-profit)

### XRP Analysis
- [AInvest: XRP Whales Accumulate 340M Tokens](https://247wallst.com/investing/2025/12/09/xrp-whales-accumulate-340m-tokens-while-retail-panic-sells-is-5-next/)
- [AInvest: XRP Institutional Momentum Q4 2025](https://www.ainvest.com/news/xrp-institutional-momentum-whale-activity-strategic-entry-point-q4-2025-2509/)

### BTC Dominance & Correlation
- [CoinDesk: Bitcoin Dominance Slides](https://www.coindesk.com/markets/2025/07/21/bitcoin-s-dominance-slides-by-most-in-3-years-as-btc-s-correlation-with-altcoins-weakens/)
- [TradingView: XRP Dominance Decoupling](https://www.tradingview.com/news/newsbtc:941d95800094b:0-xrp-dominance-explodes-decoupling-from-bitcoin-and-ethereum-has-begun/)

### Volume Manipulation Detection
- [Chainalysis: Crypto Market Manipulation 2025](https://www.chainalysis.com/blog/crypto-market-manipulation-wash-trading-pump-and-dump-2025/)
- [Nasdaq: Crypto Wash Trading Detection](https://www.nasdaq.com/articles/fintech/crypto-wash-trading-why-its-still-flying-under-the-radar-and-what-institutions-can-do-about-it)

---

**Document Version:** 1.0
**Created:** December 2025
**Author:** Research Agent
**Platform Version:** WebSocket Paper Tester v1.0.2+
