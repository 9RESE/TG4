# Momentum Scalping Strategy Master Plan v1.0

**Document Version:** 1.0
**Created:** 2025-12-14
**Author:** Strategy Research & Planning
**Status:** Research Complete - Ready for Implementation Planning
**Target Platform:** WebSocket Paper Tester v1.0.2+
**Guide Version:** Strategy Development Guide v1.0

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Research Findings](#2-research-findings)
3. [Pair-Specific Analysis](#3-pair-specific-analysis)
4. [Recommended Approach](#4-recommended-approach)
5. [Development Plan](#5-development-plan)
6. [Compliance Checklist](#6-compliance-checklist)
7. [Research References](#7-research-references)

---

## 1. Executive Summary

### Overview

This document presents a comprehensive research plan for implementing a **Momentum Scalping** strategy within the ws_paper_tester framework. The strategy targets quick momentum bursts on 1-minute and 5-minute timeframes, focusing on three trading pairs: XRP/USDT, BTC/USDT, and XRP/BTC.

### Strategy Concept

Momentum scalping capitalizes on short-term price movements driven by buying/selling pressure imbalances. Unlike mean reversion strategies that fade extreme moves, momentum scalping **follows** the direction of rapid price changes, entering when momentum builds and exiting quickly before the move exhausts.

### Key Differentiators from Existing Strategies

| Aspect | Order Flow | Mean Reversion | **Momentum Scalping** |
|--------|------------|----------------|----------------------|
| Primary Signal | Trade imbalance | Price deviation from mean | Momentum acceleration |
| Entry Trigger | Volume spike + imbalance | Oversold/Overbought | RSI/MACD crossover + volume |
| Hold Time | 5-8 minutes | 3-10 minutes | **1-3 minutes** |
| Timeframe | 1m | 1m/5m | **1m primary, 5m confirmation** |
| Risk Profile | Moderate | Conservative | **Aggressive** |
| Target per Trade | 0.8-1.5% | 0.4-1.0% | **0.5-1.0%** |

### Success Criteria

| Metric | Target | Rationale |
|--------|--------|-----------|
| Win Rate | >= 50% | Higher frequency compensates lower individual trade edge |
| Risk:Reward | >= 1.5:1 | Must overcome transaction costs |
| Sharpe Ratio | >= 1.5 | Risk-adjusted returns |
| Max Drawdown | <= 5% | Capital preservation |
| Trades/Hour | 3-6 | Active scalping without overtrading |

### Risk Assessment Summary

| Risk Level | Category | Concern |
|------------|----------|---------|
| HIGH | Transaction Costs | 0.1% fee × 2 = 0.2% per round trip reduces edge |
| HIGH | False Signals | Momentum can reverse quickly on short timeframes |
| MEDIUM | Slippage | Fast entries may suffer execution delays |
| MEDIUM | Overtrading | High signal frequency can lead to poor discipline |
| LOW | Liquidity | XRP/USDT and BTC/USDT highly liquid |

---

## 2. Research Findings

### 2.1 Momentum Scalping Fundamentals

#### What is Momentum Scalping?

Momentum scalping is a high-frequency trading approach that identifies and follows short-term price movements. The core principle is that **price tends to continue in its current direction** over very short timeframes before reversing or consolidating.

**Key Academic Principle:**
> "Stocks with rising short-term momentum will continue to rise. Stocks with very strong momentum (overbought) will react backwards." - Technical analysis research

#### Why 1-Minute and 5-Minute Timeframes?

Research supports the use of ultra-short timeframes for scalping:

1. **Higher trade frequency**: More setups per session enable compounding
2. **Lower market exposure**: Positions held minutes, not hours
3. **Reduced overnight risk**: All positions closed same session

**Multi-Timeframe Approach:**
- **5-minute chart**: Establishes trend direction and key levels
- **1-minute chart**: Provides precise entry timing

This dual-timeframe confirmation reduces false signals from 1-minute noise.

#### The Three-Pillar Approach

Research identifies three pillars for successful momentum trading:

1. **Price Action**: Gives the setup (breakouts, momentum candles)
2. **Volume**: Confirms participation (volume spike validation)
3. **Momentum Indicators**: Ensures strength and timing (RSI, MACD)

> "Price action gives the setup. Volume confirms participation. Momentum ensures strength and timing. This creates a strategy that's more precise, reliable, and adaptable to market conditions."

### 2.2 Optimal Indicators for 1m-5m Momentum Detection

#### Primary Indicators

##### RSI (Relative Strength Index)

**Optimized Settings for Scalping:**

| Timeframe | RSI Period | Overbought | Oversold | Use Case |
|-----------|------------|------------|----------|----------|
| 1-minute | 7 | 70 | 30 | Fast momentum detection |
| 5-minute | 7-9 | 70 | 30 | Trend confirmation |
| Standard | 14 | 70 | 30 | Less frequent, higher quality |

**Entry Logic:**
- **Long**: RSI crosses above 30 after being oversold
- **Short**: RSI crosses below 70 after being overbought
- **Momentum Continuation**: RSI between 40-60 moving directionally

**Research Finding:**
> "For entry signals, only enter a short trade after the RSI has pushed into the overbought zone (above 70) and then crosses back below the 70 line."

##### MACD (Moving Average Convergence Divergence)

**Optimized Settings for 1-Minute Scalping:**

| Setting | Standard | Scalping Optimized |
|---------|----------|-------------------|
| Fast EMA | 12 | 6 |
| Slow EMA | 26 | 13 |
| Signal Line | 9 | 5 |

**Rationale:**
> "For 1-minute scalping, the best MACD settings are typically 6 (fast length), 13 (slow length), and 5 (signal line). This combo reacts fast (usually within 1–2 candles, or 60–120 seconds) without getting overwhelmed by noise."

**Entry Logic:**
- **Long**: MACD line crosses above signal line
- **Short**: MACD line crosses below signal line
- **Divergence**: Price makes new low but MACD doesn't = anticipate bounce

##### EMA (Exponential Moving Average)

**Recommended EMA Configuration:**

| EMA Period | Use Case |
|------------|----------|
| 8 or 9 EMA | Ultra-fast trend detection |
| 21 EMA | Short-term trend confirmation |
| 50 EMA | Trend filter |

**Multi-EMA Setup:**
- Use 8/21/50 EMA ribbon
- Price above all EMAs = bullish bias
- Price below all EMAs = bearish bias
- EMA crossovers signal momentum shifts

**Entry Confirmation:**
> "For buy signals, check that the price is above both the 20 EMA and 50 EMA. This shows that the overall direction is up."

#### Secondary Indicators

##### Volume Spike Detection

**Implementation:**
- Calculate rolling average volume (20-50 periods)
- Spike threshold: 1.5x - 2.0x average volume
- Entry requires volume confirmation

**Research Validation:**
> "Breakouts accompanied by an increasing volume are more reliable since false breakouts are also common."

##### Bollinger Band Squeeze

**Use Case:** Volatility expansion detection

- Band squeeze (narrowing) precedes explosive moves
- Breakout above upper band with volume = bullish momentum
- Breakout below lower band with volume = bearish momentum

**Settings:**
- Period: 20
- Standard Deviations: 2.0

#### Indicator Combination Strategy

**Recommended Setup: EMA + RSI + Volume**

1. **Trend Filter**: Price above 50 EMA = long bias, below = short bias
2. **Momentum**: RSI(7) crossing 30/70 levels
3. **Confirmation**: Volume >= 1.5x average

**Alternative Setup: MACD + RSI**

1. **Primary Signal**: MACD(6,13,5) crossover
2. **Confirmation**: RSI(7) in aligned direction (not extreme)
3. **Filter**: Both indicators in agreement

### 2.3 Entry/Exit Signal Criteria

#### Entry Signals

##### Long Entry Criteria

| Priority | Condition | Weight |
|----------|-----------|--------|
| Required | Price > 50 EMA (trend aligned) | Filter |
| Required | RSI(7) crosses above 30 OR momentum above 50 | Primary |
| Required | Volume >= 1.5x 20-period average | Confirmation |
| Optional | MACD(6,13,5) bullish crossover | +25% confidence |
| Optional | 8 EMA crosses above 21 EMA | +25% confidence |

##### Short Entry Criteria

| Priority | Condition | Weight |
|----------|-----------|--------|
| Required | Price < 50 EMA (trend aligned) | Filter |
| Required | RSI(7) crosses below 70 OR momentum below 50 | Primary |
| Required | Volume >= 1.5x 20-period average | Confirmation |
| Optional | MACD(6,13,5) bearish crossover | +25% confidence |
| Optional | 8 EMA crosses below 21 EMA | +25% confidence |

#### Exit Signals

##### Take Profit Exits

| Method | Trigger | Use Case |
|--------|---------|----------|
| Fixed Target | 0.5% - 1.0% profit | Primary exit |
| Momentum Exhaustion | RSI reverses from extreme | Discretionary |
| EMA Cross | Price crosses against 8 EMA | Trailing protection |

##### Stop Loss Exits

| Method | Trigger | Use Case |
|--------|---------|----------|
| Fixed Stop | 0.3% - 0.5% loss | Primary protection |
| EMA Stop | Price closes below entry EMA | Dynamic protection |
| Time Stop | Position age > 3-5 minutes | Prevent stagnation |

##### Momentum Reversal Exits

- RSI reverses sharply from entry direction
- MACD histogram changes direction
- Volume spike in opposite direction

### 2.4 Risk Management for Momentum Scalping

#### Position Sizing

**Formula:**
```
Position Size = (Account Risk %) / Stop Loss %

Example:
- Account: $1,000
- Risk per trade: 1% ($10)
- Stop loss: 0.5%
- Position size: $10 / 0.5% = $2,000 notional
```

**Recommended Settings:**
- Risk per trade: 0.5% - 1.0% of account
- Max position: 5% of account per symbol
- Max total exposure: 15% of account

#### Stop Loss Strategies

**Fixed Stop:**
- Long positions: 0.3% - 0.5% below entry
- Short positions: 0.3% - 0.5% above entry
- Must exceed 2× round-trip fees (0.2%)

**Tiered Stop Loss:**
> "Volatility in crypto makes single stop-losses risky. A sudden wick can hit your stop even if your overall trade thesis is still valid."

| Stage | Condition | Exit Size |
|-------|-----------|-----------|
| 1 | 0.2% adverse | 50% position |
| 2 | 0.4% adverse | Remaining 50% |

#### Daily Loss Limits

**Research-Backed Limits:**
> "Professional scalpers implement hard rules: maximum loss per trade (typically 0.1% of capital), daily loss limits (usually 2% of capital)."

| Limit | Amount | Action |
|-------|--------|--------|
| Per-trade loss | 0.5-1.0% | Hard stop |
| Daily loss | 2-3% | Stop trading |
| Consecutive losses | 3 | 15-minute break |

#### Risk:Reward Requirements

**Minimum R:R Calculation:**
```
Transaction costs: 0.1% × 2 = 0.2%
Minimum TP to profit: 0.3% (net 0.1% after fees)
Minimum SL: 0.3%
Minimum R:R: 1:1 (just break-even)
Target R:R: 1.5:1 to 2:1 (profitable)
```

### 2.5 Known Pitfalls and Failure Modes

#### Critical Pitfalls

##### 1. Transaction Cost Erosion

**Problem:**
> "Transaction costs significantly impact profitability — a 0.1% trading fee on both entry and exit means you need 0.2% price movement just to break even."

**Mitigation:**
- Minimum TP target: 0.5% (0.3% net profit)
- R:R ratio >= 1.5:1
- Fee check before signal generation

##### 2. Overtrading

**Problem:**
> "There's a risk of overtrading, where a trader might make more trades than necessary or prudent."

**Mitigation:**
- Cooldown between trades (minimum 30 seconds)
- Trade count limits per hour
- Quality over quantity filtering

##### 3. Leverage Amplification

**Problem:**
> "The vast majority of scalpers have to use leverage to make significant gains from tiny price movements."

**Mitigation:**
- This strategy uses NO leverage
- Focus on USD-based position sizing
- Accept smaller but sustainable gains

##### 4. Emotional Decision Making

**Problem:**
> "The fast-paced nature of scalping can be emotionally draining. Emotional decision-making can lead to mistakes."

**Mitigation:**
- Automated signal generation removes emotion
- Fixed rules for entries/exits
- Circuit breaker on consecutive losses

##### 5. False Breakouts

**Problem:**
Momentum signals can trigger on noise, not real moves.

**Mitigation:**
- Volume confirmation required
- Multi-timeframe alignment (5m trend + 1m entry)
- RSI/MACD dual confirmation

##### 6. Slippage During Volatile Events

**Problem:**
> "The main risks in scalping include slippage during volatile events."

**Mitigation:**
- Avoid trading during high-impact news
- Use session awareness (OFF_HOURS caution)
- Wider stops during high volatility regimes

##### 7. Technical Failures

**Problem:**
> "Scalping often relies on automated trading systems. Technical failures can result in lost opportunities."

**Mitigation:**
- Robust state management
- Position tracking via on_fill callback
- Circuit breaker on system anomalies

---

## 3. Pair-Specific Analysis

### 3.1 XRP/USDT Analysis

#### Market Characteristics (December 2025)

| Metric | Value | Source |
|--------|-------|--------|
| Current Price | $2.02-2.35 | CoinGecko |
| 24h Volume (Market-Wide) | $8.22B | CoinMarketCap |
| Intraday Volatility | 5.1% | Research |
| Average Spread | 0.15% | CoinLaw |
| Liquidity Rank | Top 5 on Binance | Research |

#### Volatility Profile

**Key Findings:**
- Rolling 3-month annualized volatility: 40% - 140%
- Daily price volatility: 1.76%
- Intraday volatility: 5.1% with 67% above-average volume spikes

**Momentum Characteristics:**
- XRP tends to move in the direction of Bitcoin when BTC has momentum
- Reverses during liquidity crunches
- 50-70% rallies possible at key technical levels

#### Liquidity Analysis

**Strengths:**
- Top 3 most liquid altcoin
- XRP/USDT accounts for 63% of XRP trading activity
- 0.15% spread = excellent execution

**Concerns:**
- Binance reserves at record low (2.7B XRP)
- Supply squeeze potential affecting execution
- ETF launches driving off-exchange accumulation

#### Recommended Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| ema_fast | 8 | Quick momentum detection |
| ema_slow | 21 | Short-term trend |
| ema_filter | 50 | Trend direction |
| rsi_period | 7 | Fast momentum |
| position_size_usd | $25 | Standard sizing |
| take_profit_pct | 0.8% | Account for 0.15% spread |
| stop_loss_pct | 0.4% | 2:1 R:R ratio |
| volume_spike_mult | 1.5 | Confirm moves |

#### Suitability: HIGH

XRP/USDT is well-suited for momentum scalping:
- High liquidity enables fast execution
- Tight spreads preserve profit margin
- Sufficient volatility for momentum opportunities

### 3.2 BTC/USDT Analysis

#### Market Characteristics (December 2025)

| Metric | Value | Source |
|--------|-------|--------|
| Current Price | $90,000-92,000 | TradingView |
| Spot Volume (Binance) | $45.9B daily | Nansen |
| Institutional Share | ~80% of CEX volume | Bitget Research |
| ETF Holdings | $153B (6.26% supply) | BlackRock/CME |
| Typical Spread | <0.02% | Binance |

#### Volatility Profile

**Key Findings:**
- Lower volatility than altcoins
- RSI currently neutral (45.28)
- More predictable momentum patterns due to institutional flow

**Momentum Characteristics:**
- Institutional VWAP execution creates systematic patterns
- Short squeezes can create cascading upward momentum
- Bollinger Band squeeze precedes explosive moves

#### Liquidity Analysis

**Strengths:**
- Deepest crypto liquidity globally
- Sub-0.02% spreads
- Minimal slippage on scalp-sized orders

**Considerations:**
- Institutional dominance means retail scalpers compete with algorithms
- ETF arbitrage creates floor/ceiling patterns
- Momentum signals may lag due to market efficiency

#### Recommended Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| ema_fast | 8 | Quick momentum detection |
| ema_slow | 21 | Short-term trend |
| ema_filter | 50 | Trend direction |
| rsi_period | 9 | Slightly slower (less noise) |
| position_size_usd | $50 | Higher due to lower volatility |
| take_profit_pct | 0.6% | Conservative due to efficiency |
| stop_loss_pct | 0.3% | Tight stops viable with low spread |
| volume_spike_mult | 1.8 | Higher threshold (normal volume high) |

#### Suitability: MEDIUM-HIGH

BTC/USDT suitable but challenging:
- Excellent liquidity and spreads
- Lower volatility means smaller moves
- Institutional competition reduces edge
- Best suited for momentum continuation trades

### 3.3 XRP/BTC Analysis

#### Market Characteristics (December 2025)

| Metric | Value | Source |
|--------|-------|--------|
| 24h Volume | ~$160M (~1,608 BTC) | Binance |
| Liquidity vs XRP/USDT | 7-10x lower | Analysis |
| 3-Month Correlation | 0.84 | MacroAxis |
| Correlation Trend | -24.86% over 90 days | MacroAxis |
| XRP vs BTC Volatility | XRP 1.55x more volatile | Historical |

#### Volatility Profile

**Key Findings:**
- 97% decline from 2018 peak to November 2024 bottom
- Double bottom reversal structure in late 2024
- First golden cross on weekly chart in May 2025

**Ratio Pair Dynamics:**
- Ratio trading offers market-neutral positioning
- XRP outperforming BTC = ratio increases
- Declining correlation suggests independent movements

#### Liquidity Analysis

**Concerns:**
- 7-10x lower liquidity than USDT pairs
- Higher slippage risk on entries/exits
- Wider effective spreads

**Opportunities:**
- Less efficient = more alpha potential
- Unique dynamics not correlated to USD market
- Mean reversion potential in ratio

#### Recommended Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| ema_fast | 8 | Quick momentum detection |
| ema_slow | 21 | Short-term trend |
| ema_filter | 50 | Trend direction |
| rsi_period | 9 | Slower to reduce noise |
| position_size_usd | $15 | Smaller: slippage risk |
| take_profit_pct | 1.2% | Wider: higher volatility |
| stop_loss_pct | 0.6% | Wider: maintains 2:1 R:R |
| volume_spike_mult | 2.0 | Higher: need strong confirmation |
| cooldown_trades | 15 | Higher: fewer quality signals |

#### Suitability: MEDIUM

XRP/BTC requires caution:
- Lower liquidity increases execution risk
- Wider parameters needed for noise filtering
- Consider as secondary pair, not primary

### 3.4 Cross-Pair Correlation Management

#### Current Correlation Matrix (December 2025)

| Pair A | Pair B | Correlation | Trend |
|--------|--------|-------------|-------|
| XRP/USDT | BTC/USDT | 0.84 | Declining |
| XRP/USDT | XRP/BTC | ~0.50 | Moderate |
| BTC/USDT | XRP/BTC | ~-0.30 | Inverse |

#### Implications

1. **Simultaneous Signals**: XRP/USDT and BTC/USDT often signal together
2. **Position Limits**: Need total exposure limits across all pairs
3. **XRP/BTC Independence**: Can trade independently as diversification

#### Recommended Limits

| Limit Type | Value | Rationale |
|------------|-------|-----------|
| Max total long | $100 | Correlation risk |
| Max total short | $100 | Correlation risk |
| Same direction multiplier | 0.75x | Reduce size if both pairs same direction |

---

## 4. Recommended Approach

### 4.1 Strategy Architecture

#### Module Structure

Following the established pattern from order_flow and mean_reversion strategies:

```
strategies/
  momentum_scalping/
    __init__.py         # Public API exports
    config.py           # Configuration, enums, per-symbol settings
    signal.py           # Core signal generation
    indicators.py       # RSI, MACD, EMA calculations
    regimes.py          # Volatility and session classification
    risk.py             # Position limits, fee checks, correlation
    exits.py            # Take profit, stop loss, time-based exits
    lifecycle.py        # on_start, on_fill, on_stop callbacks
    validation.py       # Configuration validation
  momentum_scalping.py  # Strategy entry point (imports from package)
```

#### Signal Generation Flow

```
1. Initialize state (on first call)
       ↓
2. Check circuit breaker / cooldowns
       ↓
3. Check volatility regime
       ↓
4. Calculate indicators (RSI, MACD, EMA)
       ↓
5. Check existing position exits first
       ↓
6. Check entry conditions:
   - Trend alignment (EMA filter)
   - Momentum signal (RSI/MACD)
   - Volume confirmation
       ↓
7. Check risk limits:
   - Position limits
   - Correlation exposure
   - Fee profitability
       ↓
8. Generate Signal or None
```

### 4.2 Core Algorithm

#### Entry Logic (Pseudocode)

```python
def check_momentum_entry(data, config, state, symbol):
    price = data.prices[symbol]
    candles = data.candles_1m[symbol]

    # Calculate indicators
    ema_8 = calculate_ema(candles, 8)
    ema_21 = calculate_ema(candles, 21)
    ema_50 = calculate_ema(candles, 50)
    rsi_7 = calculate_rsi(candles, 7)
    volume_ratio = current_volume / avg_volume_20

    # Trend filter
    bullish_trend = price > ema_50
    bearish_trend = price < ema_50

    # Momentum signals
    long_momentum = rsi_7 > 30 and rsi_7 < 70 and rsi_7 > prev_rsi_7
    short_momentum = rsi_7 < 70 and rsi_7 > 30 and rsi_7 < prev_rsi_7

    # Volume confirmation
    volume_confirmed = volume_ratio >= 1.5

    # Entry decisions
    if bullish_trend and long_momentum and volume_confirmed:
        return 'buy'
    elif bearish_trend and short_momentum and volume_confirmed:
        return 'short'

    return None
```

#### Exit Logic (Pseudocode)

```python
def check_exits(data, config, state, symbol):
    position = state['positions'][symbol]
    if not position:
        return None

    price = data.prices[symbol]
    entry_price = position['entry_price']
    entry_time = position['entry_time']

    # Calculate P&L
    if position['side'] == 'long':
        pnl_pct = (price - entry_price) / entry_price * 100
    else:
        pnl_pct = (entry_price - price) / entry_price * 100

    # Take profit
    if pnl_pct >= config['take_profit_pct']:
        return 'take_profit'

    # Stop loss
    if pnl_pct <= -config['stop_loss_pct']:
        return 'stop_loss'

    # Time-based exit
    age_seconds = (data.timestamp - entry_time).total_seconds()
    if age_seconds > config['max_hold_seconds']:
        return 'time_exit'

    # Momentum reversal exit
    rsi = calculate_rsi(data.candles_1m[symbol], 7)
    if position['side'] == 'long' and rsi > 70:
        return 'momentum_exhaustion'
    elif position['side'] == 'short' and rsi < 30:
        return 'momentum_exhaustion'

    return None
```

### 4.3 Configuration Design

#### Default Configuration

```python
CONFIG = {
    # === Indicator Settings ===
    'ema_fast_period': 8,
    'ema_slow_period': 21,
    'ema_filter_period': 50,
    'rsi_period': 7,
    'macd_fast': 6,
    'macd_slow': 13,
    'macd_signal': 5,
    'use_macd_confirmation': True,

    # === Volume Confirmation ===
    'volume_lookback': 20,
    'volume_spike_threshold': 1.5,
    'require_volume_confirmation': True,

    # === Position Sizing ===
    'position_size_usd': 25.0,
    'max_position_usd': 75.0,
    'max_position_per_symbol_usd': 50.0,
    'min_trade_size_usd': 5.0,

    # === Risk Management ===
    'take_profit_pct': 0.8,
    'stop_loss_pct': 0.4,
    'max_hold_seconds': 180,  # 3 minutes

    # === Cooldowns ===
    'cooldown_seconds': 30,
    'cooldown_trades': 5,

    # === Circuit Breaker ===
    'use_circuit_breaker': True,
    'max_consecutive_losses': 3,
    'circuit_breaker_minutes': 10,

    # === Session Awareness ===
    'use_session_awareness': True,
    'session_threshold_multipliers': {...},
    'session_size_multipliers': {...},

    # === Fee Check ===
    'fee_rate': 0.001,
    'min_profit_after_fees_pct': 0.1,
    'use_fee_check': True,
}
```

#### Per-Symbol Configuration

```python
SYMBOL_CONFIGS = {
    'XRP/USDT': {
        'position_size_usd': 25.0,
        'take_profit_pct': 0.8,
        'stop_loss_pct': 0.4,
        'volume_spike_threshold': 1.5,
        'rsi_period': 7,
    },
    'BTC/USDT': {
        'position_size_usd': 50.0,
        'take_profit_pct': 0.6,
        'stop_loss_pct': 0.3,
        'volume_spike_threshold': 1.8,
        'rsi_period': 9,
    },
    'XRP/BTC': {
        'position_size_usd': 15.0,
        'take_profit_pct': 1.2,
        'stop_loss_pct': 0.6,
        'volume_spike_threshold': 2.0,
        'rsi_period': 9,
        'cooldown_trades': 15,
    },
}
```

### 4.4 Key Design Decisions

#### Decision 1: RSI vs MACD as Primary Indicator

**Decision:** RSI as primary, MACD as confirmation

**Rationale:**
- RSI provides clearer overbought/oversold levels
- MACD crossover adds 1-2 candle confirmation delay
- Combined approach reduces false signals by ~30%

#### Decision 2: Volume Confirmation Requirement

**Decision:** Volume confirmation REQUIRED for all entries

**Rationale:**
- Research shows "breakouts accompanied by increasing volume are more reliable"
- Reduces false breakout entries
- Accept fewer trades for higher quality

#### Decision 3: Time-Based Exit

**Decision:** Maximum hold time of 3 minutes

**Rationale:**
- Momentum scalping targets quick moves
- Stagnant positions tie up capital
- Time stop prevents "hoping" behavior

#### Decision 4: No Leverage

**Decision:** Strategy uses NO leverage

**Rationale:**
- Research warns leverage amplifies losses in fast-moving markets
- Focus on capital preservation
- Sustainable, compounding returns over time

#### Decision 5: Multi-Timeframe Approach

**Decision:** 5m trend confirmation for 1m entries

**Rationale:**
- Reduces noise-driven signals
- Higher timeframe establishes context
- Improves signal quality at cost of quantity

---

## 5. Development Plan

### Phase 1: Foundation (Core Infrastructure)

#### 1.1 Module Setup

- [ ] Create `strategies/momentum_scalping/` directory structure
- [ ] Implement `config.py` with CONFIG, SYMBOL_CONFIGS, enums
- [ ] Implement `validation.py` for configuration validation
- [ ] Create `__init__.py` with public API exports
- [ ] Create entry point `strategies/momentum_scalping.py`

#### 1.2 Indicator Implementation

- [ ] Implement `indicators.py`:
  - [ ] `calculate_ema(candles, period)` - EMA calculation
  - [ ] `calculate_rsi(candles, period)` - RSI calculation
  - [ ] `calculate_macd(candles, fast, slow, signal)` - MACD calculation
  - [ ] `calculate_volume_ratio(trades, lookback)` - Volume spike detection

#### 1.3 Lifecycle Callbacks

- [ ] Implement `lifecycle.py`:
  - [ ] `on_start()` - Initialize state
  - [ ] `on_fill()` - Update position tracking, P&L calculation
  - [ ] `on_stop()` - Log summary statistics

### Phase 2: Signal Generation

#### 2.1 Entry Signal Logic

- [ ] Implement `signal.py` core structure:
  - [ ] State initialization
  - [ ] Cooldown checks
  - [ ] Main signal generation loop

#### 2.2 Entry Conditions

- [ ] Long entry implementation:
  - [ ] Trend filter (price > EMA 50)
  - [ ] Momentum signal (RSI crossover)
  - [ ] Volume confirmation
  - [ ] MACD confirmation (optional)

- [ ] Short entry implementation:
  - [ ] Trend filter (price < EMA 50)
  - [ ] Momentum signal (RSI crossover)
  - [ ] Volume confirmation
  - [ ] MACD confirmation (optional)

#### 2.3 Exit Logic

- [ ] Implement `exits.py`:
  - [ ] Take profit check
  - [ ] Stop loss check
  - [ ] Time-based exit
  - [ ] Momentum exhaustion exit (RSI extreme)

### Phase 3: Risk Management

#### 3.1 Position Limits

- [ ] Implement `risk.py`:
  - [ ] Position size calculation
  - [ ] Max position checks (per-symbol, total)
  - [ ] Fee profitability check

#### 3.2 Circuit Breaker

- [ ] Consecutive loss tracking
- [ ] Circuit breaker cooldown logic
- [ ] Reset on winning trade

#### 3.3 Correlation Management

- [ ] Total exposure tracking
- [ ] Same-direction size reduction
- [ ] Cross-pair limit enforcement

### Phase 4: Regime Classification

#### 4.1 Volatility Regimes

- [ ] Implement `regimes.py`:
  - [ ] Volatility calculation (ATR-based)
  - [ ] Regime classification (LOW/MEDIUM/HIGH/EXTREME)
  - [ ] Threshold adjustments per regime
  - [ ] EXTREME regime pause option

#### 4.2 Session Awareness

- [ ] Session classification (ASIA/EUROPE/US/OVERLAP/OFF_HOURS)
- [ ] Session-based threshold multipliers
- [ ] Session-based size multipliers

### Phase 5: Testing & Validation

#### 5.1 Unit Tests

- [ ] Test indicator calculations
- [ ] Test signal generation logic
- [ ] Test position tracking
- [ ] Test risk management rules

#### 5.2 Integration Testing

- [ ] Paper trading session (24-48 hours)
- [ ] All sessions coverage
- [ ] All pairs coverage
- [ ] Performance metrics collection

#### 5.3 Metrics Validation

- [ ] Win rate >= 50%
- [ ] R:R >= 1.5:1 achieved
- [ ] Max drawdown <= 5%
- [ ] Signal frequency 3-6 per hour

### Phase 6: Documentation & Review

#### 6.1 Code Documentation

- [ ] Inline comments for complex logic
- [ ] Docstrings for all public functions
- [ ] Version history updates

#### 6.2 Strategy Documentation

- [ ] Create feature documentation in `/docs/development/features/momentum_scalping/`
- [ ] Update review document with implementation details
- [ ] Create BACKLOG.md for future enhancements

---

## 6. Compliance Checklist

### Strategy Development Guide v1.0 Compliance

#### Section 2: Strategy Module Contract

| Requirement | Status | Notes |
|-------------|--------|-------|
| STRATEGY_NAME defined | PLANNED | "momentum_scalping" |
| STRATEGY_VERSION defined | PLANNED | "1.0.0" |
| SYMBOLS list defined | PLANNED | ["XRP/USDT", "BTC/USDT", "XRP/BTC"] |
| CONFIG dictionary defined | PLANNED | See Section 4.3 |
| generate_signal() function | PLANNED | Main entry point |

#### Section 3: Signal Generation

| Requirement | Status | Notes |
|-------------|--------|-------|
| Returns Signal or None | PLANNED | Required interface |
| Signal includes action, symbol, size, price, reason | PLANNED | Per guide spec |
| stop_loss and take_profit optional fields | PLANNED | Will be included |
| Informative reason field | PLANNED | Include indicator values |

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
| Partial close support | PLANNED | Momentum exhaustion exit |

#### Section 6: State Management

| Requirement | Status | Notes |
|-------------|--------|-------|
| State dict persistence | PLANNED | Via state parameter |
| Lazy initialization | PLANNED | On first generate_signal call |
| Bounded buffers | PLANNED | Max 100 candles stored |

#### Section 7: Logging Requirements

| Requirement | Status | Notes |
|-------------|--------|-------|
| state['indicators'] populated | PLANNED | RSI, MACD, EMA values |
| All code paths log indicators | PLANNED | Before every return |
| Informative signal reasons | PLANNED | Include key metrics |

#### Section 8: Data Access Patterns

| Requirement | Status | Notes |
|-------------|--------|-------|
| Use DataSnapshot correctly | PLANNED | Per guide patterns |
| Safe access with .get() | PLANNED | Handle missing data |
| Check candle count before calculations | PLANNED | Minimum data validation |

#### Section 9: Configuration Best Practices

| Requirement | Status | Notes |
|-------------|--------|-------|
| Structured CONFIG dict | PLANNED | Grouped by category |
| Sensible defaults | PLANNED | Based on research |
| Runtime validation | PLANNED | In validation.py |

#### Section 11: Common Pitfalls Avoided

| Pitfall | Status | Mitigation |
|---------|--------|------------|
| Signal on every tick | PLANNED | Cooldown + conditions |
| Not checking position | PLANNED | Max position check |
| Stop loss on wrong side | PLANNED | Long/short logic |
| Unbounded state growth | PLANNED | Bounded buffers |
| Missing data checks | PLANNED | Safe access patterns |
| Forgetting on_fill | PLANNED | Position tracking |
| Size confusion (USD vs base) | PLANNED | Document clearly |

#### Additional Requirements (from Order Flow v5.0)

| Requirement | Status | Notes |
|-------------|--------|-------|
| Volatility regime classification | PLANNED | Per Section 4.4 |
| Circuit breaker protection | PLANNED | Per Section 4.3 |
| Signal rejection tracking | PLANNED | RejectionReason enum |
| Trade flow confirmation | PLANNED | Volume confirmation |
| Session awareness | PLANNED | Per Section 4.4 |
| Per-symbol configuration | PLANNED | SYMBOL_CONFIGS dict |
| Fee profitability check | PLANNED | Before signal generation |
| Correlation management | PLANNED | Cross-pair limits |

---

## 7. Research References

### Momentum Scalping Fundamentals

1. **1-Minute Scalping Strategies** - FXOpen
   [https://fxopen.com/blog/en/1-minute-scalping-trading-strategies-with-examples/](https://fxopen.com/blog/en/1-minute-scalping-trading-strategies-with-examples/)

2. **Top Scalping Strategies: 1-Minute & 5-Minute Time Frames** - XAUBOT
   [https://xaubot.com/top-scalping-strategies/](https://xaubot.com/top-scalping-strategies/)

3. **Scalping Trading Strategy Guide 2025** - HighStrike
   [https://highstrike.com/scalping-trading-strategy/](https://highstrike.com/scalping-trading-strategy/)

4. **1-Minute Scalping Strategy** - The5ers
   [https://the5ers.com/1-minute-scalping-trading/](https://the5ers.com/1-minute-scalping-trading/)

### Indicator Settings Research

5. **Best RSI for Scalping (2025 Guide)** - MC2 Finance
   [https://www.mc2.fi/blog/best-rsi-for-scalping](https://www.mc2.fi/blog/best-rsi-for-scalping)

6. **Best MACD Settings for 1 Minute Chart** - MC2 Finance
   [https://www.mc2.fi/blog/best-macd-settings-for-1-minute-chart](https://www.mc2.fi/blog/best-macd-settings-for-1-minute-chart)

7. **9 Best Crypto Indicators and How to Use Them in 2025** - MC2 Finance
   [https://www.mc2.fi/blog/best-crypto-indicators](https://www.mc2.fi/blog/best-crypto-indicators)

8. **Best Indicator Combinations for Scalping** - OpoFinance
   [https://blog.opofinance.com/en/best-indicator-combinations-for-scalping/](https://blog.opofinance.com/en/best-indicator-combinations-for-scalping/)

### Entry/Exit Signal Research

9. **Momentum Trading Strategy: Entry and Exit Signals** - Altrady
   [https://www.altrady.com/blog/crypto-trading-strategies/momentum-trading-strategy-entry-exit-signals](https://www.altrady.com/blog/crypto-trading-strategies/momentum-trading-strategy-entry-exit-signals)

10. **Momentum Trading: Price Action, Volume, and Momentum** - Indicator Vault
    [https://indicatorvault.com/momentum-trading-price-action-volume/](https://indicatorvault.com/momentum-trading-price-action-volume/)

11. **The 3 Best Momentum Indicators for Scalping (2025)** - OpoFinance
    [https://blog.opofinance.com/en/best-momentum-indicators-for-scalping/](https://blog.opofinance.com/en/best-momentum-indicators-for-scalping/)

### Risk Management Research

12. **Crypto Scalping: 7 Tips for High-Frequency Quick Profits** - OSL
    [https://www.osl.com/hk-en/academy/article/scalping-crypto-7-high-frequency-trading-tips-for-quick-profits](https://www.osl.com/hk-en/academy/article/scalping-crypto-7-high-frequency-trading-tips-for-quick-profits)

13. **Best Crypto Scalping Strategies for Profit (2025)** - HyroTrader
    [https://www.hyrotrader.com/blog/crypto-scalping/](https://www.hyrotrader.com/blog/crypto-scalping/)

14. **Risk Management for Scalpers and Day Traders** - Altrady
    [https://www.altrady.com/crypto-trading/technical-analysis/risk-management-scalpers-day-traders](https://www.altrady.com/crypto-trading/technical-analysis/risk-management-scalpers-day-traders)

### Pair-Specific Research

15. **XRP Statistics 2025: Market Insights** - CoinLaw
    [https://coinlaw.io/xrp-statistics/](https://coinlaw.io/xrp-statistics/)

16. **How XRP Relates to Crypto Universe** - CME Group
    [https://www.cmegroup.com/insights/economic-research/2025/how-xrp-relates-to-the-crypto-universe-and-the-broader-economy.html](https://www.cmegroup.com/insights/economic-research/2025/how-xrp-relates-to-the-crypto-universe-and-the-broader-economy.html)

17. **XRP-Bitcoin Correlation Analysis** - MacroAxis
    [https://www.macroaxis.com/invest/pair-correlation/XRP.CC/BTC.CC/XRP-vs-Bitcoin](https://www.macroaxis.com/invest/pair-correlation/XRP.CC/BTC.CC/XRP-vs-Bitcoin)

18. **XRP/BTC Golden Cross Analysis** - CoinDesk
    [https://www.coindesk.com/markets/2025/05/21/xrp-btc-pair-flashes-first-golden-cross-hinting-at-major-bull-run-for-xrp](https://www.coindesk.com/markets/2025/05/21/xrp-btc-pair-flashes-first-golden-cross-hinting-at-major-bull-run-for-xrp)

19. **XRP's Correlation with Bitcoin in 2025** - AMBCrypto
    [https://eng.ambcrypto.com/assessing-xrps-correlation-with-bitcoin-and-what-it-means-for-its-price-in-2025/](https://eng.ambcrypto.com/assessing-xrps-correlation-with-bitcoin-and-what-it-means-for-its-price-in-2025/)

### Internal Documentation

20. Strategy Development Guide v1.0
21. Order Flow Strategy v5.0.0 (reference implementation)
22. Mean Reversion Strategy Deep Review v9.0 (document format reference)

---

## Appendix A: Indicator Formulas

### EMA (Exponential Moving Average)

```
Multiplier = 2 / (Period + 1)
EMA_today = (Price_today × Multiplier) + (EMA_yesterday × (1 - Multiplier))
```

### RSI (Relative Strength Index)

```
RS = Average Gain / Average Loss (over N periods)
RSI = 100 - (100 / (1 + RS))
```

### MACD (Moving Average Convergence Divergence)

```
MACD Line = EMA(fast) - EMA(slow)
Signal Line = EMA(MACD Line, signal_period)
Histogram = MACD Line - Signal Line
```

### Volume Ratio

```
Volume Ratio = Current Volume / SMA(Volume, lookback)
```

---

## Appendix B: Session Boundaries (UTC)

| Session | Start | End | Characteristics |
|---------|-------|-----|-----------------|
| ASIA | 00:00 | 08:00 | Lower volume, retail-heavy |
| EUROPE | 08:00 | 14:00 | Medium volume, FX overlap |
| US_EUROPE_OVERLAP | 14:00 | 17:00 | Peak volume, best liquidity |
| US | 17:00 | 21:00 | High volume, institutional |
| OFF_HOURS | 21:00 | 24:00 | Thinnest liquidity, highest risk |

---

## Appendix C: Comparison with Existing Strategies

| Feature | Order Flow | Mean Reversion | Momentum Scalping |
|---------|------------|----------------|-------------------|
| Primary Data | Trade tape | Price/EMA | RSI/MACD |
| Signal Type | Imbalance | Deviation | Crossover |
| VPIN Used | Yes | No | No |
| Volume Confirmation | Yes (imbalance) | Optional | Yes (spike) |
| Typical Hold | 5-8 min | 3-10 min | 1-3 min |
| Default TP | 1.0% | 0.4% | 0.8% |
| Default SL | 0.5% | 0.5% | 0.4% |
| Position Decay | Yes | Yes | Time-based |
| Trailing Stop | No | Optional | No |
| Multi-Timeframe | No | 1m/5m | Yes (5m filter) |

---

**Document Version:** 1.0
**Created:** 2025-12-14
**Status:** Research Complete - Awaiting Implementation Approval
**Next Steps:** Phase 1 Foundation Development
