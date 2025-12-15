# WaveTrend Oscillator Strategy Master Plan v1.0

**Document Version:** 1.0
**Created:** 2025-12-14
**Author:** Strategy Research & Planning
**Status:** Research Complete - Ready for Implementation Planning
**Target Platform:** WebSocket Paper Tester v1.0.2+
**Guide Version:** Strategy Development Guide v1.0
**Source Strategy:** `src/strategies/wavetrend/strategy.py`

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Research Findings](#2-research-findings)
3. [Source Code Analysis](#3-source-code-analysis)
4. [Pair-Specific Analysis](#4-pair-specific-analysis)
5. [Recommended Approach](#5-recommended-approach)
6. [Development Plan](#6-development-plan)
7. [Compliance Checklist](#7-compliance-checklist)
8. [Research References](#8-research-references)

---

## 1. Executive Summary

### Overview

This document presents a comprehensive research plan for adapting the **WaveTrend Oscillator** strategy from the legacy implementation to the ws_paper_tester framework. The strategy is a momentum oscillator system that identifies overbought/oversold conditions with cleaner signals than RSI, using a dual-line crossover mechanism.

### Strategy Concept

The WaveTrend Oscillator, originally developed by LazyBear for TradingView, is a channel-based momentum indicator that calculates price deviation from a smoothed average, normalized to reduce noise. It produces two lines (WT1 and WT2) whose crossovers generate trading signals, with enhanced reliability when occurring in overbought or oversold zones.

**Core Principle:**
> "WaveTrend is a port of a famous TS/MT indicator. It generates cleaner signals than RSI, especially for volatile crypto markets."

### Key Differentiators from Existing Strategies

| Aspect | Momentum Scalping | Mean Reversion | **WaveTrend Oscillator** |
|--------|------------------|----------------|-------------------------|
| Primary Signal | RSI/MACD crossover | Price deviation from mean | WT1/WT2 crossover |
| Entry Trigger | Momentum acceleration | Oversold/Overbought RSI | Crossover in OB/OS zone |
| Zone System | RSI 70/30 | Deviation bands | WT 60/-60 (80/-80 extreme) |
| Divergence | Not primary | Not primary | **Primary confirmation** |
| Hold Time | 1-3 minutes | 3-10 minutes | **Variable (hourly signals)** |
| Timeframe | 1m primary | 1m/5m | **1h primary (15m fallback)** |

### Success Criteria

| Metric | Target | Rationale |
|--------|--------|-----------|
| Win Rate | >= 55% | Zone-filtered signals should have higher quality |
| Risk:Reward | >= 2:1 | Dual-line confirmation reduces false signals |
| Sharpe Ratio | >= 1.2 | Moderate frequency, higher quality trades |
| Max Drawdown | <= 5% | Capital preservation |
| Signal Frequency | 2-4/day | Hourly timeframe produces fewer signals |

### Risk Assessment Summary

| Risk Level | Category | Concern |
|------------|----------|---------|
| HIGH | Whipsaw | Neutral zone crossovers in sideways markets |
| HIGH | Lag | Multiple EMA smoothing layers create inherent lag |
| MEDIUM | False Divergence | Divergence can persist in strong trends |
| MEDIUM | Zone Sensitivity | 60/-60 vs 80/-80 threshold tuning critical |
| LOW | Liquidity | Target pairs are highly liquid |

---

## 2. Research Findings

### 2.1 WaveTrend Oscillator Fundamentals

#### What is the WaveTrend Oscillator?

The WaveTrend Oscillator is a momentum indicator created by LazyBear that identifies overbought and oversold conditions with cleaner signals than traditional oscillators. It's ranked among the **Top 10 TradingView indicators for 2025**.

**Key Characteristics:**
- **Dual-line system**: Similar to MACD's dual-line approach
- **Zone-based signals**: Enhanced reliability in extreme zones
- **Divergence detection**: Built-in divergence identification
- **Reduced noise**: Channel-based normalization reduces false signals

**Research Validation:**
> "This indicator is based on the renowned WaveTrend Oscillator by LazyBear, a favorite among professional traders for spotting trend reversals with precision."

#### The WaveTrend Formula

The WaveTrend calculation follows this sequence:

```
1. HLC3 = (High + Low + Close) / 3        # Typical Price
2. ESA = EMA(HLC3, channel_length)        # Exponential Smoothed Average
3. D = EMA(|HLC3 - ESA|, channel_length)  # Average Deviation
4. CI = (HLC3 - ESA) / (0.015 * D)        # Channel Index
5. WT1 = EMA(CI, average_length)          # WaveTrend Line 1 (tci)
6. WT2 = SMA(WT1, ma_length)              # WaveTrend Line 2 (Signal)
```

**Parameters (Default):**

| Parameter | Default | Purpose |
|-----------|---------|---------|
| channel_length | 10 | ESA and D calculation period |
| average_length | 21 | WT1 smoothing period |
| ma_length | 4 | WT2 signal line smoothing |

**Zone Levels:**

| Zone | Level | Signal Interpretation |
|------|-------|----------------------|
| Extreme Overbought | +80 | High-probability reversal zone |
| Overbought | +60 | Standard overbought zone |
| Neutral | -60 to +60 | No clear directional bias |
| Oversold | -60 | Standard oversold zone |
| Extreme Oversold | -80 | High-probability reversal zone |

### 2.2 WaveTrend vs RSI Comparison

#### Advantages Over RSI

1. **Better Signal Clarity**: WaveTrend smooths out noise while maintaining momentum connection
2. **Reduced False Signals**: Requires confirmation between oscillator lines
3. **Earlier Reversal Detection**: In high-volatility crypto markets, can predict reversals 2-3 candles earlier
4. **Visual Clarity**: Clear crossover signals with zone confirmation

#### Key Differences

| Aspect | RSI | WaveTrend |
|--------|-----|-----------|
| Bounded | Yes (0-100) | Unbounded (typically -100 to +100) |
| Calculation | Price change momentum | Channel deviation |
| Signal Type | Level crossings | Dual-line crossover |
| Divergence | Manual identification | Built-in detection |
| Noise Level | Higher on short timeframes | Reduced via multi-layer smoothing |

**Research Finding:**
> "Unlike traditional oscillators that can give you whiplash with false signals, the WaveTrend takes a different approach. It smooths out the noise while keeping you connected to real market momentum."

#### Limitations

1. **Unbounded Nature**: Doesn't reach traditional +-100 levels consistently
2. **Lag**: Multiple EMA layers introduce 2-5 candle delay
3. **Sideways Markets**: Can generate whipsaw signals in neutral zones
4. **Not Ideal for Reversals**: "Because it is an unbound oscillator, WaveTrend does not excel at revealing Positive/Negative reversals within a trend as pronounced as a bound oscillator like RSI"

### 2.3 Signal Generation Criteria

#### Entry Signals

##### Long Entry (Buy)

| Priority | Condition | Weight |
|----------|-----------|--------|
| Required | WT1 crosses above WT2 | Primary signal |
| Required | Crossover occurs below -60 (oversold) | Zone confirmation |
| Optional | Price divergence (higher low while WT shows higher low) | +10% confidence |
| Optional | Crossover in extreme oversold (-80) | +5% confidence |
| Optional | Previous zone was extreme oversold | +5% confidence |

**Entry Logic from Legacy Code:**
```python
# Bullish cross: WT1 crosses above WT2
if current_wt1 > current_wt2 and prev_wt1 <= prev_wt2:
    if 'oversold' in prev_zone:
        confidence = 0.75
    elif current_zone in ['oversold', 'extreme_oversold']:
        confidence = 0.70
    if divergence == 'bullish':
        confidence += 0.10
    if 'extreme' in prev_zone:
        confidence += 0.05
```

##### Short Entry (Sell)

| Priority | Condition | Weight |
|----------|-----------|--------|
| Required | WT1 crosses below WT2 | Primary signal |
| Required | Crossover occurs above +60 (overbought) | Zone confirmation |
| Optional | Price divergence (lower high while WT shows lower high) | +10% confidence |
| Optional | Crossover in extreme overbought (+80) | +5% confidence |
| Optional | Previous zone was extreme overbought | +5% confidence |

**Confidence Caps:**
- Long entries: Maximum 0.92
- Short entries: Maximum 0.88 (conservative approach)

#### Exit Signals

##### Exit Long Position

| Condition | Exit Type | Order Type |
|-----------|-----------|------------|
| WT1 crosses below WT2 (bearish crossover) | Signal exit | Market |
| Current zone reaches extreme overbought | Profit taking | Market |
| Stop loss triggered | Risk exit | Market |

##### Exit Short Position

| Condition | Exit Type | Order Type |
|-----------|-----------|------------|
| WT1 crosses above WT2 (bullish crossover) | Signal exit | Market |
| Current zone reaches extreme oversold | Profit taking | Market |
| Stop loss triggered | Risk exit | Market |

#### The `require_zone_exit` Option

The legacy strategy includes a `require_zone_exit` configuration:

- **When True**: Entry signals wait for price to exit the overbought/oversold zone before executing
- **When False**: Entry signals execute immediately on crossover in zone

**Research Recommendation:**
> "Wait for those clean signals in the extreme zones. Don't chase every crossover."

### 2.4 Divergence Detection Algorithm

#### Bullish Divergence

```
Price makes lower low + WaveTrend makes higher low = Bullish divergence
```

**Implementation from Legacy:**
```python
# Recent vs Prior period comparison
if recent_close_min < prior_close_min and recent_wt_min > prior_wt_min:
    if current_wt < oversold:  # More significant in oversold
        return 'bullish'
```

#### Bearish Divergence

```
Price makes higher high + WaveTrend makes lower high = Bearish divergence
```

**Implementation:**
```python
if recent_close_max > prior_close_max and recent_wt_max < prior_wt_max:
    if current_wt > overbought:  # More significant in overbought
        return 'bearish'
```

**Parameters:**
- `divergence_lookback`: 14 candles (default)
- Comparison: Recent N candles vs Prior N candles

### 2.5 Known Pitfalls and Failure Modes

#### Critical Pitfalls

##### 1. Whipsaw in Sideways Markets

**Problem:**
> "In choppy or sideways markets, the WaveTrend Oscillator may generate false signals, leading to whipsaw trading."

**Mitigation Strategies:**
- Only trade crossovers in overbought/oversold zones (avoid neutral zone signals)
- Add ADX filter (only trade when ADX > 25)
- Require volume confirmation
- Use longer timeframes (1h instead of 15m)

##### 2. Multiple EMA Smoothing Lag

**Problem:**
The WaveTrend formula includes three layers of smoothing:
1. ESA = EMA of HLC3
2. D = EMA of deviation
3. WT1 = EMA of CI
4. WT2 = SMA of WT1

This creates inherent lag of 2-5 candles depending on settings.

**Mitigation:**
- Accept lag as trade-off for cleaner signals
- Use tick-level execution once signal confirms
- Avoid overly long periods

##### 3. Zone Threshold Sensitivity

**Problem:**
The choice between 60/-60 and 80/-80 zones significantly impacts signal frequency and quality.

| Zone | Signals | Quality | Risk |
|------|---------|---------|------|
| 60/-60 | More frequent | Lower | Higher false signals |
| 80/-80 | Less frequent | Higher | May miss moves |

**Mitigation:**
- Use tiered approach: 60/-60 for standard, 80/-80 for extreme
- Adjust thresholds per pair based on volatility
- Consider hybrid: enter at 60, exit at 80

##### 4. Divergence False Positives

**Problem:**
> "In strong trends, divergence can persist for extended periods before (or without) reversal."

**Mitigation:**
- Use divergence as confidence boost, not standalone signal
- Require zone confirmation + divergence
- Consider trend filter (don't fade strong trends)

##### 5. Unbounded Oscillator Behavior

**Problem:**
Unlike RSI (0-100), WaveTrend is unbounded and can reach values beyond +-100 in extreme conditions, making static thresholds less reliable.

**Mitigation:**
- Use adaptive thresholds based on recent WT range
- Consider WaveTrend 3D variant with normalized levels
- Monitor for threshold adjustments per pair

##### 6. Candle Buffer Management

**Problem (Tick Environment):**
The legacy strategy uses DataFrame-based hourly candle calculations. WebSocket environment receives ticks and must accumulate candles.

**Mitigation:**
- Implement candle buffer accumulation from tick data
- Calculate minimum required bars: `max(channel_length, average_length, divergence_lookback * 2) + 10`
- With defaults: `max(10, 21, 28) + 10 = 38 candles minimum`

---

## 3. Source Code Analysis

### 3.1 Legacy Implementation Structure

**File:** `src/strategies/wavetrend/strategy.py`

**Class:** `WaveTrend(BaseStrategy)`

#### Key Components

| Method | Purpose | Adaptation Needed |
|--------|---------|-------------------|
| `__init__` | Parameter initialization | Convert to CONFIG dict |
| `_calculate_wavetrend` | Core WT calculation | Extract to indicators.py |
| `_get_zone` | Zone classification | Extract to indicators.py |
| `_check_crossover` | WT1/WT2 crossover detection | Extract to signals.py |
| `_check_divergence` | Price/WT divergence | Extract to indicators.py |
| `generate_signals` | Main signal logic | Adapt to generate_signal() |
| `on_order_filled` | Position sync | Adapt to on_fill() |
| `get_status` | Status reporting | Extract key indicators |

### 3.2 Legacy Parameters Mapping

**Configuration Parameters:**

| Legacy Parameter | Default | ws_paper_tester CONFIG |
|-----------------|---------|----------------------|
| `channel_length` | 10 | `wt_channel_length` |
| `average_length` | 21 | `wt_average_length` |
| `ma_length` | 4 | `wt_ma_length` |
| `overbought` | 60 | `wt_overbought` |
| `oversold` | -60 | `wt_oversold` |
| `extreme_overbought` | 80 | `wt_extreme_overbought` |
| `extreme_oversold` | -80 | `wt_extreme_oversold` |
| `require_zone_exit` | True | `require_zone_exit` |
| `use_divergence` | True | `use_divergence` |
| `divergence_lookback` | 14 | `divergence_lookback` |
| `base_size_pct` | 0.10 | `position_size_usd` (converted) |

### 3.3 Timeframe Adaptation

**Legacy Approach:**
- Primary: 1-hour candles (`{symbol}_1h`)
- Fallback: 15-minute candles (`{symbol}_15m`)
- Final fallback: Raw symbol data

**ws_paper_tester Adaptation:**

The ws_paper_tester provides:
- `data.candles_1m` - 1-minute candles
- `data.candles_5m` - 5-minute candles

**Recommended Approach:**

1. **Candle Aggregation**: Aggregate tick data to build 1-hour candles internally
2. **Buffer Management**: Maintain rolling buffer of last 50-100 1h candles
3. **Fallback**: Use 5m candles aggregated to simulate 15m if hourly unavailable

**Alternative Approach:**
- Calculate WaveTrend on 5m candles with adjusted parameters
- Faster signals but potentially noisier

### 3.4 Position Tracking Adaptation

**Legacy Pattern:**
```python
def on_order_filled(self, order: Dict[str, Any]) -> None:
    action = order.get('action', '')
    symbol = order.get('symbol', '')

    if action == 'buy':
        self.positions[symbol] = 'long'
    elif action == 'short':
        self.positions[symbol] = 'short'
    elif action in ['sell', 'cover', 'close']:
        self.positions.pop(symbol, None)
```

**ws_paper_tester Pattern:**
```python
def on_fill(fill: Dict[str, Any], state: Dict[str, Any]) -> None:
    side = fill.get('side', '')
    symbol = fill.get('symbol', '')
    value = fill.get('value', 0)

    if side == 'buy':
        state['position_side'] = 'long'
        state['position_by_symbol'][symbol] = state.get(...) + value
    elif side == 'sell':
        state['position_by_symbol'][symbol] -= value
        if state['position_by_symbol'][symbol] < 0.01:
            state['position_side'] = None
```

### 3.5 Order Type Consideration

**Legacy Strategy Uses:**
- `limit` orders for entries (better fill prices)
- `market` orders for exits (immediate execution)

**ws_paper_tester Guide v1.0:**
> "order_type: str = 'market' # Only 'market' supported"

**Adaptation Decision:**
For v1.0, use market orders for both entries and exits. Consider limit order support as future enhancement.

---

## 4. Pair-Specific Analysis

### 4.1 XRP/USDT Analysis

#### WaveTrend Characteristics

| Metric | Value | Implication |
|--------|-------|-------------|
| Intraday Volatility | 5.1% | Moderate - good for hourly signals |
| Typical Daily Range | ~$0.10 | Sufficient for WT zone transitions |
| Average Spread | 0.15% | Acceptable execution |
| 1H Candle Range | 1-3% | Adequate WT movement per candle |

#### WaveTrend-Specific Observations

- **Volatility Squeeze**: Bollinger Bands show periodic tight consolidation followed by expansion
- **Zone Transitions**: XRP tends to reach extreme zones during high-volume events
- **Reversal Reliability**: 5.1% intraday volatility suggests zone reversals are tradeable

#### Recommended Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| wt_overbought | 60 | Standard zone |
| wt_oversold | -60 | Standard zone |
| wt_extreme_overbought | 75 | Slightly lower for faster signals |
| wt_extreme_oversold | -75 | Slightly higher for faster signals |
| position_size_usd | $25 | Standard sizing |
| stop_loss_pct | 1.5% | Wider for hourly timeframe |
| take_profit_pct | 3.0% | 2:1 R:R ratio |

#### Suitability: HIGH

XRP/USDT is well-suited for WaveTrend:
- Sufficient volatility for zone transitions
- Good liquidity for execution
- Historical responsiveness to momentum signals

### 4.2 BTC/USDT Analysis

#### WaveTrend Characteristics

| Metric | Value | Implication |
|--------|-------|-------------|
| Intraday ATR | $770-1010 | High absolute volatility |
| Percentage Volatility | 1.64% daily | Lower % than altcoins |
| RSI Behavior | Currently ~60 | Not in extreme territory |
| Institutional Flow | 80% of volume | Systematic patterns |

#### WaveTrend-Specific Observations

- **Trending Behavior**: BTC exhibits strong trending at price extremes
- **ADX Consideration**: ADX > 30 indicates strong trend where WT signals may lag
- **Consolidation Phases**: Current market shows "volatility squeeze" structure

#### Recommended Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| wt_overbought | 65 | Higher threshold - BTC sustains overbought longer |
| wt_oversold | -65 | Higher threshold for same reason |
| wt_extreme_overbought | 80 | Standard extreme |
| wt_extreme_oversold | -80 | Standard extreme |
| position_size_usd | $50 | Larger due to lower % volatility |
| stop_loss_pct | 1.0% | Tighter - more predictable moves |
| take_profit_pct | 2.0% | 2:1 R:R ratio |

#### Suitability: MEDIUM-HIGH

BTC/USDT requires caution:
- Lower percentage volatility means fewer extreme zone visits
- Strong trending behavior can create extended WT divergences
- Consider ADX filter to avoid trend-following markets

### 4.3 XRP/BTC Analysis

#### WaveTrend Characteristics

| Metric | Value | Implication |
|--------|-------|-------------|
| Correlation with BTC | 0.84 (declining) | Increasing independence |
| XRP Relative Volatility | 1.55x BTC | Higher ratio volatility |
| Liquidity | 7-10x lower than USDT pairs | Execution risk |

#### WaveTrend-Specific Observations

- **Ratio Dynamics**: XRP/BTC measures relative strength, not absolute
- **Zone Transitions**: More extreme WT readings due to relative volatility
- **Declining Correlation**: -24.86% over 90 days suggests more independent movements

#### Recommended Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| wt_overbought | 55 | Lower - ratio pairs move differently |
| wt_oversold | -55 | Lower threshold |
| wt_extreme_overbought | 70 | Lower extreme |
| wt_extreme_oversold | -70 | Lower extreme |
| position_size_usd | $15 | Smaller - liquidity constraints |
| stop_loss_pct | 2.0% | Wider - higher volatility |
| take_profit_pct | 4.0% | Wider targets |
| cooldown_seconds | 120 | Longer - fewer signals |

#### Suitability: MEDIUM

XRP/BTC should be approached cautiously:
- Lower liquidity increases slippage risk
- Ratio interpretation differs from USDT pairs
- Consider as secondary/diversification pair

### 4.4 Cross-Pair Correlation Management

#### Correlation Matrix (December 2025)

| Pair A | Pair B | Correlation | WT Signal Conflict Risk |
|--------|--------|-------------|------------------------|
| XRP/USDT | BTC/USDT | 0.84 | HIGH - likely same direction |
| XRP/USDT | XRP/BTC | ~0.50 | MEDIUM |
| BTC/USDT | XRP/BTC | ~-0.30 | LOW - inverse relationship |

#### WT Signal Synchronization

When XRP/USDT and BTC/USDT generate WT signals simultaneously:
- Both likely in same market phase
- Total exposure risk increases
- Consider position limits

**Recommended Limits:**

| Limit Type | Value | Rationale |
|------------|-------|-----------|
| Max total long | $100 | Correlation risk |
| Max total short | $100 | Correlation risk |
| Same direction multiplier | 0.75x | Reduce overlapping exposure |

---

## 5. Recommended Approach

### 5.1 Strategy Architecture

#### Module Structure

Following the established pattern from momentum_scalping and order_flow strategies:

```
strategies/
  wavetrend/
    __init__.py         # Public API exports
    config.py           # Configuration, enums, per-symbol settings
    signal.py           # Core signal generation
    indicators.py       # WaveTrend calculation, zones, divergence
    regimes.py          # Volatility and zone classification
    risk.py             # Position limits, fee checks
    exits.py            # WT crossover exits, profit taking
    lifecycle.py        # on_start, on_fill, on_stop callbacks
    validation.py       # Configuration validation
  wavetrend.py          # Strategy entry point (imports from package)
```

#### Signal Generation Flow

```
1. Initialize state (on first call)
       ↓
2. Check circuit breaker / cooldowns
       ↓
3. Accumulate/update candle buffer
       ↓
4. Calculate WaveTrend indicators (WT1, WT2)
       ↓
5. Classify current zone (OB/OS/neutral/extreme)
       ↓
6. Check existing position exits first:
   - Crossover reversal exit
   - Extreme zone profit taking
       ↓
7. Check entry conditions:
   - Crossover detection
   - Zone requirement (if require_zone_exit)
   - Divergence bonus
       ↓
8. Check risk limits:
   - Position limits
   - Correlation exposure
       ↓
9. Generate Signal or None
```

### 5.2 Core Algorithm

#### WaveTrend Calculation (Pseudocode)

```python
def calculate_wavetrend(candles, config):
    """
    Calculate WaveTrend Oscillator values.

    Formula:
    1. HLC3 = (High + Low + Close) / 3
    2. ESA = EMA(HLC3, channel_length)
    3. D = EMA(|HLC3 - ESA|, channel_length)
    4. CI = (HLC3 - ESA) / (0.015 * D)
    5. WT1 = EMA(CI, average_length)
    6. WT2 = SMA(WT1, ma_length)
    """
    channel_length = config['wt_channel_length']  # 10
    average_length = config['wt_average_length']  # 21
    ma_length = config['wt_ma_length']           # 4

    hlc3 = [(c.high + c.low + c.close) / 3 for c in candles]

    # ESA - EMA of HLC3
    esa = calculate_ema(hlc3, channel_length)

    # D - EMA of absolute deviation
    deviation = [abs(h - e) for h, e in zip(hlc3, esa)]
    d = calculate_ema(deviation, channel_length)

    # CI - Channel Index (avoid division by zero)
    ci = [(hlc3[i] - esa[i]) / (0.015 * d[i] + 1e-10)
          for i in range(len(hlc3))]

    # WT1 - EMA of CI
    wt1_series = calculate_ema(ci, average_length)

    # WT2 - SMA of WT1
    wt2_series = calculate_sma(wt1_series, ma_length)

    return {
        'wt1': wt1_series[-1],
        'wt2': wt2_series[-1],
        'prev_wt1': wt1_series[-2],
        'prev_wt2': wt2_series[-2],
        'diff': wt1_series[-1] - wt2_series[-1],
    }
```

#### Zone Classification

```python
def classify_zone(wt1, config):
    """Determine current WaveTrend zone."""
    if wt1 >= config['wt_extreme_overbought']:
        return 'extreme_overbought'
    elif wt1 >= config['wt_overbought']:
        return 'overbought'
    elif wt1 <= config['wt_extreme_oversold']:
        return 'extreme_oversold'
    elif wt1 <= config['wt_oversold']:
        return 'oversold'
    return 'neutral'
```

#### Crossover Detection

```python
def detect_crossover(wt1, wt2, prev_wt1, prev_wt2):
    """Detect WT1/WT2 crossover."""
    # Bullish: WT1 crosses above WT2
    if wt1 > wt2 and prev_wt1 <= prev_wt2:
        return 'bullish'

    # Bearish: WT1 crosses below WT2
    if wt1 < wt2 and prev_wt1 >= prev_wt2:
        return 'bearish'

    return None
```

#### Divergence Detection

```python
def detect_divergence(closes, wt1_series, lookback, oversold, overbought):
    """Detect price/WaveTrend divergence."""
    n = lookback
    if len(closes) < n * 2 + 5:
        return None

    recent_close = closes[-n:]
    prior_close = closes[-2*n:-n]
    recent_wt = wt1_series[-n:]
    prior_wt = wt1_series[-2*n:-n]

    # Bullish divergence
    if min(recent_close) < min(prior_close) and min(recent_wt) > min(prior_wt):
        if wt1_series[-1] < oversold:
            return 'bullish'

    # Bearish divergence
    if max(recent_close) > max(prior_close) and max(recent_wt) < max(prior_wt):
        if wt1_series[-1] > overbought:
            return 'bearish'

    return None
```

### 5.3 Configuration Design

#### Default Configuration

```python
STRATEGY_NAME = "wavetrend"
STRATEGY_VERSION = "1.0.0"
SYMBOLS = ["XRP/USDT", "BTC/USDT", "XRP/BTC"]

CONFIG = {
    # === WaveTrend Indicator Settings ===
    'wt_channel_length': 10,        # ESA and D calculation period
    'wt_average_length': 21,        # WT1 smoothing period
    'wt_ma_length': 4,              # WT2 signal line smoothing

    # === Zone Thresholds ===
    'wt_overbought': 60,            # Standard overbought
    'wt_oversold': -60,             # Standard oversold
    'wt_extreme_overbought': 80,    # Extreme overbought
    'wt_extreme_oversold': -80,     # Extreme oversold

    # === Signal Settings ===
    'require_zone_exit': True,      # Wait for zone exit before entry
    'use_divergence': True,         # Include divergence in confidence
    'divergence_lookback': 14,      # Candles for divergence calculation

    # === Position Sizing ===
    'position_size_usd': 25.0,      # Base position size
    'max_position_usd': 75.0,       # Maximum total position
    'max_position_per_symbol_usd': 50.0,  # Per-symbol maximum
    'min_trade_size_usd': 5.0,      # Minimum trade size
    'short_size_multiplier': 0.8,   # Reduce short position size

    # === Risk Management ===
    'stop_loss_pct': 1.5,           # Stop loss percentage
    'take_profit_pct': 3.0,         # Take profit percentage (2:1 R:R)

    # === Confidence Caps ===
    'max_long_confidence': 0.92,    # Maximum confidence for longs
    'max_short_confidence': 0.88,   # Maximum confidence for shorts

    # === Candle Management ===
    'candle_timeframe_minutes': 60, # Primary timeframe (1 hour)
    'fallback_timeframe_minutes': 15,  # Fallback timeframe
    'min_candle_buffer': 50,        # Minimum candles required

    # === Cooldowns ===
    'cooldown_seconds': 60,         # Minimum time between signals

    # === Circuit Breaker ===
    'use_circuit_breaker': True,
    'max_consecutive_losses': 3,
    'circuit_breaker_minutes': 30,  # Longer cooldown for hourly strategy

    # === Fee Check ===
    'fee_rate': 0.001,
    'use_fee_check': True,
}
```

#### Per-Symbol Configuration

```python
SYMBOL_CONFIGS = {
    'XRP/USDT': {
        'wt_overbought': 60,
        'wt_oversold': -60,
        'wt_extreme_overbought': 75,
        'wt_extreme_oversold': -75,
        'position_size_usd': 25.0,
        'stop_loss_pct': 1.5,
        'take_profit_pct': 3.0,
    },
    'BTC/USDT': {
        'wt_overbought': 65,
        'wt_oversold': -65,
        'wt_extreme_overbought': 80,
        'wt_extreme_oversold': -80,
        'position_size_usd': 50.0,
        'stop_loss_pct': 1.0,
        'take_profit_pct': 2.0,
    },
    'XRP/BTC': {
        'wt_overbought': 55,
        'wt_oversold': -55,
        'wt_extreme_overbought': 70,
        'wt_extreme_oversold': -70,
        'position_size_usd': 15.0,
        'stop_loss_pct': 2.0,
        'take_profit_pct': 4.0,
        'cooldown_seconds': 120,
    },
}
```

### 5.4 Key Design Decisions

#### Decision 1: Timeframe Handling

**Decision:** Build 1-hour candles from tick data internally

**Rationale:**
- WaveTrend is optimized for hourly timeframes per legacy strategy
- Provides consistency with proven parameters
- Reduces noise compared to 1m/5m available data

**Implementation:**
- Accumulate ticks into OHLCV buckets
- Complete candle on boundary crossing
- Maintain rolling buffer of 50+ candles

#### Decision 2: Zone Requirement

**Decision:** `require_zone_exit = True` as default

**Rationale:**
- Research shows zone-filtered signals have higher quality
- Reduces whipsaw in neutral zone
- Accept fewer signals for higher win rate

#### Decision 3: Divergence Usage

**Decision:** Divergence as confidence bonus, not required

**Rationale:**
- Divergence can persist in strong trends
- Adding 10% confidence is meaningful but not deciding
- Allows signals without divergence to execute

#### Decision 4: Market Orders Only (v1.0)

**Decision:** Use market orders for both entries and exits

**Rationale:**
- ws_paper_tester v1.0 only supports market orders
- Limit orders can be added in future versions
- Market orders ensure execution in paper testing

#### Decision 5: Confidence Caps

**Decision:** Long max 0.92, Short max 0.88

**Rationale:**
- From legacy implementation
- Shorts carry additional risk in crypto markets
- Prevents overconfidence even with all confirmations

---

## 6. Development Plan

### Phase 1: Foundation (Core Infrastructure)

#### 1.1 Module Setup

- [ ] Create `strategies/wavetrend/` directory structure
- [ ] Implement `config.py` with CONFIG, SYMBOL_CONFIGS, enums
- [ ] Implement `validation.py` for configuration validation
- [ ] Create `__init__.py` with public API exports
- [ ] Create entry point `strategies/wavetrend.py`

#### 1.2 Indicator Implementation

- [ ] Implement `indicators.py`:
  - [ ] `calculate_ema()` - Exponential Moving Average
  - [ ] `calculate_sma()` - Simple Moving Average
  - [ ] `calculate_wavetrend()` - Full WT calculation (WT1, WT2)
  - [ ] `classify_zone()` - Zone classification
  - [ ] `detect_crossover()` - WT1/WT2 crossover detection
  - [ ] `detect_divergence()` - Price/WT divergence

#### 1.3 Candle Buffer Management

- [ ] Implement candle accumulation from tick data:
  - [ ] OHLCV bucket management
  - [ ] Hourly boundary detection
  - [ ] Rolling buffer maintenance
  - [ ] Minimum buffer validation

#### 1.4 Lifecycle Callbacks

- [ ] Implement `lifecycle.py`:
  - [ ] `on_start()` - Initialize state, validate config
  - [ ] `on_fill()` - Update position tracking
  - [ ] `on_stop()` - Log summary statistics

### Phase 2: Signal Generation

#### 2.1 Entry Signal Logic

- [ ] Implement `signal.py` core structure:
  - [ ] State initialization
  - [ ] Cooldown checks
  - [ ] WaveTrend calculation
  - [ ] Zone classification
  - [ ] Main signal generation loop

#### 2.2 Entry Conditions

- [ ] Long entry implementation:
  - [ ] Bullish crossover detection
  - [ ] Zone requirement check (`require_zone_exit`)
  - [ ] Divergence bonus calculation
  - [ ] Confidence scoring with caps

- [ ] Short entry implementation:
  - [ ] Bearish crossover detection
  - [ ] Zone requirement check
  - [ ] Divergence bonus calculation
  - [ ] Reduced position sizing (`short_size_multiplier`)

#### 2.3 Exit Logic

- [ ] Implement `exits.py`:
  - [ ] Crossover reversal exit (WT1/WT2 cross against position)
  - [ ] Extreme zone profit taking
  - [ ] Stop loss check
  - [ ] Take profit check

### Phase 3: Risk Management

#### 3.1 Position Limits

- [ ] Implement `risk.py`:
  - [ ] Position size calculation with confidence scaling
  - [ ] Max position checks (per-symbol, total)
  - [ ] Fee profitability check

#### 3.2 Circuit Breaker

- [ ] Consecutive loss tracking
- [ ] Extended circuit breaker cooldown (30 minutes)
- [ ] Reset on winning trade

#### 3.3 Correlation Management

- [ ] Total exposure tracking
- [ ] Same-direction size reduction
- [ ] Cross-pair limit enforcement

### Phase 4: Regime Integration

#### 4.1 Zone-Based Regimes

- [ ] Implement `regimes.py`:
  - [ ] WaveTrend zone classification
  - [ ] Zone transition tracking
  - [ ] Threshold adjustments per zone

#### 4.2 Volatility Awareness

- [ ] ATR-based volatility calculation
- [ ] Dynamic threshold adjustment option
- [ ] Extreme volatility pause option

### Phase 5: Testing & Validation

#### 5.1 Unit Tests

- [ ] Test WaveTrend calculation accuracy
- [ ] Test zone classification
- [ ] Test crossover detection
- [ ] Test divergence detection
- [ ] Test position tracking

#### 5.2 Integration Testing

- [ ] Paper trading session (48-72 hours)
- [ ] All sessions coverage
- [ ] All pairs coverage
- [ ] Performance metrics collection

#### 5.3 Metrics Validation

- [ ] Win rate >= 55%
- [ ] R:R >= 2:1 achieved
- [ ] Max drawdown <= 5%
- [ ] Signal frequency 2-4 per day per pair

### Phase 6: Documentation & Review

#### 6.1 Code Documentation

- [ ] Inline comments for WaveTrend formula
- [ ] Docstrings for all public functions
- [ ] Version history updates

#### 6.2 Strategy Documentation

- [ ] Create feature documentation in `/docs/development/features/wavetrend/`
- [ ] Update review document with implementation details
- [ ] Create BACKLOG.md for future enhancements (limit orders, adaptive zones)

---

## 7. Compliance Checklist

### Strategy Development Guide v1.0 Compliance

#### Section 2: Strategy Module Contract

| Requirement | Status | Notes |
|-------------|--------|-------|
| STRATEGY_NAME defined | PLANNED | "wavetrend" |
| STRATEGY_VERSION defined | PLANNED | "1.0.0" |
| SYMBOLS list defined | PLANNED | ["XRP/USDT", "BTC/USDT", "XRP/BTC"] |
| CONFIG dictionary defined | PLANNED | See Section 5.3 |
| generate_signal() function | PLANNED | Main entry point |

#### Section 3: Signal Generation

| Requirement | Status | Notes |
|-------------|--------|-------|
| Returns Signal or None | PLANNED | Required interface |
| Signal includes action, symbol, size, price, reason | PLANNED | Per guide spec |
| stop_loss and take_profit optional fields | PLANNED | Will be included |
| Informative reason field | PLANNED | Include WT1, zone, confidence |

#### Section 4: Stop Loss & Take Profit

| Requirement | Status | Notes |
|-------------|--------|-------|
| SL below entry for long | PLANNED | Calculated per-symbol |
| SL above entry for short | PLANNED | Inverted calculation |
| TP above entry for long | PLANNED | Calculated per-symbol |
| TP below entry for short | PLANNED | Inverted calculation |
| R:R ratio >= 1:1 | PLANNED | Target 2:1 |

#### Section 5: Position Management

| Requirement | Status | Notes |
|-------------|--------|-------|
| Size in USD (not base asset) | PLANNED | position_size_usd |
| Max position limits | PLANNED | Per-symbol and total |
| Partial close support | PLANNED | Exit positions fully |

#### Section 6: State Management

| Requirement | Status | Notes |
|-------------|--------|-------|
| State dict persistence | PLANNED | Via state parameter |
| Lazy initialization | PLANNED | On first generate_signal call |
| Bounded buffers | PLANNED | Max 100 candles stored |

#### Section 7: Logging Requirements

| Requirement | Status | Notes |
|-------------|--------|-------|
| state['indicators'] populated | PLANNED | WT1, WT2, zone, divergence |
| All code paths log indicators | PLANNED | Before every return |
| Informative signal reasons | PLANNED | Include key metrics |

#### Section 8: Data Access Patterns

| Requirement | Status | Notes |
|-------------|--------|-------|
| Use DataSnapshot correctly | PLANNED | Per guide patterns |
| Safe access with .get() | PLANNED | Handle missing data |
| Check candle count before calculations | PLANNED | Minimum 50 candles |

#### Section 9: Configuration Best Practices

| Requirement | Status | Notes |
|-------------|--------|-------|
| Structured CONFIG dict | PLANNED | Grouped by category |
| Sensible defaults | PLANNED | Based on legacy + research |
| Runtime validation | PLANNED | In validation.py |

#### Section 11: Common Pitfalls Avoided

| Pitfall | Status | Mitigation |
|---------|--------|------------|
| Signal on every tick | PLANNED | Zone requirement + cooldown |
| Not checking position | PLANNED | Max position check |
| Stop loss on wrong side | PLANNED | Long/short logic |
| Unbounded state growth | PLANNED | Bounded candle buffer |
| Missing data checks | PLANNED | Safe access patterns |
| Forgetting on_fill | PLANNED | Position tracking |
| Size confusion (USD vs base) | PLANNED | Document clearly |

#### Additional Requirements (from existing strategies)

| Requirement | Status | Notes |
|-------------|--------|-------|
| Volatility regime awareness | PLANNED | Zone-based regimes |
| Circuit breaker protection | PLANNED | 30-minute cooldown |
| Signal rejection tracking | PLANNED | RejectionReason enum |
| Per-symbol configuration | PLANNED | SYMBOL_CONFIGS dict |
| Fee profitability check | PLANNED | Before signal generation |
| Correlation management | PLANNED | Cross-pair limits |

---

## 8. Research References

### WaveTrend Oscillator Fundamentals

1. **Original WaveTrend Oscillator [WT] by LazyBear** - TradingView
   [https://www.tradingview.com/script/2KE8wTuF-Indicator-WaveTrend-Oscillator-WT/](https://www.tradingview.com/script/2KE8wTuF-Indicator-WaveTrend-Oscillator-WT/)

2. **Understanding the WaveTrend [LazyBear] Indicator** - Medium
   [https://medium.com/the-modern-scientist/understanding-the-wavetrend-lazybear-indicator-71254f4234ec](https://medium.com/the-modern-scientist/understanding-the-wavetrend-lazybear-indicator-71254f4234ec)

3. **WaveTrend Oscillator: Best Trading Signals** - Pineify Blog
   [https://pineify.app/resources/blog/wavetrend-oscillator-indicator-tradingview-pine-script](https://pineify.app/resources/blog/wavetrend-oscillator-indicator-tradingview-pine-script)

4. **Wave Trend Oscillator: Master Market Momentum** - ChartAlert
   [https://chartalert.in/2023/05/04/wave-trend-oscillator/](https://chartalert.in/2023/05/04/wave-trend-oscillator/)

### WaveTrend vs RSI Comparison

5. **3 TradingView Indicators Better Than RSI** - Edge Forex
   [https://edge-forex.com/3-tradingview-indicators-that-are-better-than-rsi/](https://edge-forex.com/3-tradingview-indicators-that-are-better-than-rsi/)

6. **Trading With The Wave Trend Oscillator** - CoinLoop Medium
   [https://medium.com/@coinloop/trading-with-the-wave-trend-oscilator-53ddc85293bf](https://medium.com/@coinloop/trading-with-the-wave-trend-oscilator-53ddc85293bf)

7. **TOP 10 BEST TRADINGVIEW INDICATORS FOR 2025** - TradingView
   [https://www.tradingview.com/chart/BTCUSD/9wGoeKnE-TOP-10-BEST-TRADINGVIEW-INDICATORS-FOR-2025/](https://www.tradingview.com/chart/BTCUSD/9wGoeKnE-TOP-10-BEST-TRADINGVIEW-INDICATORS-FOR-2025/)

### Avoiding False Signals

8. **WaveTrend 3D - Improved Implementation** - GitHub
   [https://github.com/artnaz/wavetrend-3d](https://github.com/artnaz/wavetrend-3d)

9. **Avoiding Whipsaw: Strategies to Minimize False Signals** - Above The Green Line
   [https://abovethegreenline.com/whipsaw-trading/](https://abovethegreenline.com/whipsaw-trading/)

10. **WaveTrend Oscillator Indicator** - The Forex Geek
    [https://theforexgeek.com/wavetrend-oscillator/](https://theforexgeek.com/wavetrend-oscillator/)

### Pair-Specific Research

11. **XRP/USDT Analysis** - TradingView
    [https://www.tradingview.com/symbols/XRPUSDT/](https://www.tradingview.com/symbols/XRPUSDT/)

12. **BTC/USDT Technical Analysis** - TradingView
    [https://www.tradingview.com/symbols/BTCUSDT/](https://www.tradingview.com/symbols/BTCUSDT/)

13. **Bitcoin Technical Analysis** - Investtech
    [https://www.investtech.com/main/market.php?CompanyID=99400001&product=241](https://www.investtech.com/main/market.php?CompanyID=99400001&product=241)

### Internal Documentation

14. Strategy Development Guide v1.0
15. Momentum Scalping Strategy Master Plan v1.0 (reference implementation)
16. Legacy WaveTrend Strategy (`src/strategies/wavetrend/strategy.py`)

---

## Appendix A: WaveTrend Formula Reference

### Complete Calculation Sequence

```
Input: OHLCV candle data

Step 1: Calculate Typical Price
    HLC3[i] = (High[i] + Low[i] + Close[i]) / 3

Step 2: Calculate Exponential Smoothed Average
    ESA[0] = SMA(HLC3, channel_length)  # Initialize with SMA
    ESA[i] = (HLC3[i] * k) + (ESA[i-1] * (1 - k))
    where k = 2 / (channel_length + 1)

Step 3: Calculate Average Deviation
    D[i] = EMA(|HLC3[i] - ESA[i]|, channel_length)

Step 4: Calculate Channel Index
    CI[i] = (HLC3[i] - ESA[i]) / (0.015 * D[i] + epsilon)
    where epsilon = 1e-10 (avoid division by zero)

Step 5: Calculate WaveTrend Line 1
    WT1[i] = EMA(CI, average_length)

Step 6: Calculate WaveTrend Line 2 (Signal Line)
    WT2[i] = SMA(WT1, ma_length)

Output: WT1, WT2, Diff = WT1 - WT2
```

### Default Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| channel_length | 10 | ESA and D smoothing period |
| average_length | 21 | WT1 (CI) smoothing period |
| ma_length | 4 | WT2 (signal) smoothing period |
| overbought | 60 | Standard overbought threshold |
| oversold | -60 | Standard oversold threshold |
| extreme_overbought | 80 | Extreme overbought threshold |
| extreme_oversold | -80 | Extreme oversold threshold |

---

## Appendix B: Zone Classification Reference

| Zone | WT1 Range | Signal Quality | Trade Direction |
|------|-----------|----------------|-----------------|
| Extreme Overbought | >= 80 | Highest (short) | Profit-take longs |
| Overbought | 60 to 79 | High (short) | Short entry zone |
| Upper Neutral | 0 to 59 | Low | Avoid new entries |
| Lower Neutral | -59 to 0 | Low | Avoid new entries |
| Oversold | -79 to -60 | High (long) | Long entry zone |
| Extreme Oversold | <= -80 | Highest (long) | Profit-take shorts |

---

## Appendix C: Confidence Scoring Reference

### Long Entry Confidence

| Condition | Base/Bonus | Cumulative |
|-----------|------------|------------|
| Bullish crossover (required) | 0.55 | 0.55 |
| Crossover from oversold zone | +0.20 | 0.75 |
| Currently in oversold | +0.15 | 0.70 |
| Bullish divergence | +0.10 | +0.10 |
| From extreme zone | +0.05 | +0.05 |
| **Maximum (capped)** | | **0.92** |

### Short Entry Confidence

| Condition | Base/Bonus | Cumulative |
|-----------|------------|------------|
| Bearish crossover (required) | 0.55 | 0.55 |
| Crossover from overbought zone | +0.15 | 0.70 |
| Currently in overbought | +0.10 | 0.65 |
| Bearish divergence | +0.10 | +0.10 |
| From extreme zone | +0.05 | +0.05 |
| **Maximum (capped)** | | **0.88** |

---

## Appendix D: Comparison with Legacy Strategy

| Aspect | Legacy Implementation | ws_paper_tester Adaptation |
|--------|----------------------|---------------------------|
| Class-based | `WaveTrend(BaseStrategy)` | Modular functions |
| Data format | `pd.DataFrame` | `DataSnapshot` (immutable) |
| Position tracking | `self.positions` dict | `state['position_by_symbol']` |
| Order callback | `on_order_filled()` | `on_fill()` |
| Order types | Limit (entry), Market (exit) | Market only (v1.0) |
| Timeframe | 1h primary (DataFrame key) | 1h candle accumulation |
| Configuration | `config` dict passed to `__init__` | `CONFIG` module constant |
| Multi-symbol | Loop in `generate_signals()` | Loop in `generate_signal()` |

---

**Document Version:** 1.0
**Created:** 2025-12-14
**Status:** Research Complete - Awaiting Implementation Approval
**Next Steps:** Phase 1 Foundation Development
