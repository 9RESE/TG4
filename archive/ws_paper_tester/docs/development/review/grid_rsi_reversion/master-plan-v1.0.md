# Grid RSI Reversion Strategy Master Plan v1.0

**Document Version:** 1.0
**Created:** 2025-12-14
**Implemented:** 2025-12-14
**Author:** Strategy Research & Planning
**Status:** IMPLEMENTED - Ready for Testing
**Target Platform:** WebSocket Paper Tester v1.0.2+
**Guide Version:** Strategy Development Guide v1.0

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

This document presents a comprehensive research plan for implementing a **Grid RSI Reversion** strategy within the ws_paper_tester framework. The strategy combines grid trading mechanics with RSI-based mean reversion signals, targeting three trading pairs: XRP/USDT, BTC/USDT, and XRP/BTC.

### Strategy Concept

Grid RSI Reversion is a hybrid approach that:
1. **Grid Trading**: Places buy orders at predetermined price levels below current price and sell orders above
2. **RSI Mean Reversion**: Uses RSI as a confidence modifier to enhance grid signals when RSI indicates oversold (for buys) or overbought (for sells) conditions

Unlike pure grid trading that executes mechanically at price levels, this strategy modulates signal confidence and position sizing based on RSI extremes, capitalizing on the mean-reverting nature of price oscillations within ranges.

### Key Differentiators from Existing Strategies

| Aspect | Order Flow | Mean Reversion | Momentum Scalping | **Grid RSI Reversion** |
|--------|------------|----------------|-------------------|------------------------|
| Primary Signal | Trade imbalance | Price deviation | RSI/MACD crossover | Grid level + RSI zone |
| Entry Trigger | Volume spike | Oversold/Overbought | Momentum acceleration | Price hits grid level |
| Hold Time | 5-8 minutes | 3-10 minutes | 1-3 minutes | **Cycle-based (variable)** |
| Timeframe | 1m | 1m/5m | 1m primary | **5m primary, 1m confirmation** |
| Risk Profile | Moderate | Conservative | Aggressive | **Conservative-Moderate** |
| Target per Trade | 0.8-1.5% | 0.4-1.0% | 0.5-1.0% | **Grid spacing (1-3%)** |
| Position Management | Single | Single | Single | **Multiple grid positions** |

### Success Criteria

| Metric | Target | Rationale |
|--------|--------|-----------|
| Grid Cycle Completion | >= 60% | Most grid orders should complete buy-sell cycles |
| Win Rate | >= 55% | Grid trading inherently wins on range completion |
| Risk:Reward | >= 1.5:1 | Grid spacing provides natural R:R |
| Max Drawdown | <= 10% | Grid accumulation can cause drawdown |
| Sharpe Ratio | >= 1.0 | Consistent, lower-variance returns |
| Capital Efficiency | >= 50% | Avoid excessive capital tied in grid levels |

### Risk Assessment Summary

| Risk Level | Category | Concern |
|------------|----------|---------|
| HIGH | Grid Accumulation | Continuous buying in trending down market |
| HIGH | Trend Markets | Grid strategies fail in strong trends |
| MEDIUM | Capital Lockup | Funds tied in unfilled grid levels |
| MEDIUM | Range Breakout | Price escaping grid boundaries |
| LOW | Liquidity | XRP/USDT and BTC/USDT highly liquid |
| LOW | Transaction Costs | Grid spacing > 1% exceeds fee impact |

---

## 2. Research Findings

### 2.1 Grid Trading Fundamentals

#### What is Grid Trading?

Grid trading is a systematic approach that places multiple buy and sell orders at predetermined price intervals, creating a "grid" of orders. The core principle is that markets oscillate within ranges, and the grid captures profits from these oscillations.

**Key Academic Principle:**
> "The advantage of a grid strategy is that it systematically 'buys the dip and sells the rip' in a choppy market without requiring any predictive forecasting." - [Grid Trading Strategy Guide](https://zignaly.com/crypto-trading/algorithmic-strategies/grid-trading)

#### Grid Spacing Approaches

**Arithmetic Grid (Fixed Spacing):**
- Equal dollar/price spacing between levels
- Example: $100 levels ($90,000, $90,100, $90,200...)
- Best for: Lower volatility periods, ranging markets

**Geometric Grid (Percentage Spacing):**
- Equal percentage spacing between levels
- Example: 1% levels ($90,000, $90,900, $91,809...)
- Best for: Volatile markets, matches how crypto moves in percentage terms

**Research Finding:**
> "Most successful traders use geometric grids (equal percentage spacing) because they match how crypto actually moves in percentage terms, not dollar terms." - [Wundertrading](https://wundertrading.com/journal/en/learn/article/best-grid-bot-settings)

#### Grid Configuration Best Practices

| Parameter | Conservative | Standard | Aggressive |
|-----------|--------------|----------|------------|
| Grid Levels | 10-15 | 15-25 | 25-40 |
| Grid Spacing | 2-3% | 1-2% | 0.5-1% |
| Capital per Level | 5-10% | 3-5% | 2-3% |
| Total Range | 10-15% | 15-25% | 25-40% |

### 2.2 Mean Reversion Theory

#### Ornstein-Uhlenbeck Process

The mathematical foundation for mean reversion trading is the Ornstein-Uhlenbeck (OU) process, which models how prices tend to revert to a long-term mean over time.

**Key Parameters:**
- **θ (theta)**: Long-term mean level - prices evolve around this
- **μ (mu)**: Speed of reversion - how fast prices return to mean
- **σ (sigma)**: Volatility - amplitude of random fluctuations

**Half-Life Formula:**
```
Half-Life = -ln(2) / theta
```

> "11.24 days is the half-life of mean reversion, meaning we anticipate the series to fully revert to the mean by 2× the half-life or 22.5 days." - [Half Life of Mean Reversion](https://flare9xblog.wordpress.com/2017/09/27/half-life-of-mean-reversion-ornstein-uhlenbeck-formula-for-mean-reverting-process/)

**Trading Implications:**
- Use half-life as lookback window for rolling statistics
- Set time stops at 2× half-life (regime change assumption)
- Scale position inversely to z-score deviation

#### RSI as Mean Reversion Indicator

The Relative Strength Index (RSI) is a momentum oscillator that measures overbought/oversold conditions, making it ideal for mean reversion confluence.

**RSI-Mean Reversion Logic:**
- RSI < 30: Oversold - expect price bounce (mean reversion UP)
- RSI > 70: Overbought - expect price pullback (mean reversion DOWN)
- RSI 40-60: Neutral zone - reduced mean reversion probability

**Research Finding:**
> "Mean reversion bots often use technical indicators like Bollinger Bands or RSI to signal overbought or oversold conditions. When the price moves significantly away from this average — either too high or too low — traders expect it to 'revert' back." - [3commas Mean Reversion](https://3commas.io/mean-reversion-trading-bot)

#### Larry Connors' RSI(2) Strategy

A well-documented mean reversion approach using short-period RSI:

> "When RSI(2) falls below 10, it is considered overselling and traders should look for buying opportunities. When RSI(2) rises above 90, it is considered overbuying and traders should look for selling opportunities." - [FMZQuant RSI2 Strategy](https://medium.com/@FMZQuant/larry-connors-rsi2-mean-reversion-strategy-861f5a3579e3)

### 2.3 RSI Reversion Confluence for Grid Trading

#### Combining Grid + RSI

The Grid RSI Reversion strategy enhances traditional grid trading by:

1. **RSI as Confidence Modifier**: Grid signals gain higher confidence when RSI is in the appropriate zone
2. **Position Sizing by RSI Extreme**: Larger positions when RSI indicates stronger mean reversion probability
3. **RSI Filter Mode (Optional)**: Only execute grid orders when RSI confirms

**RSI Confidence Calculation (from legacy code):**
```python
# For BUY signals
if rsi < oversold:
    confidence = 0.7 + min(0.4, (oversold - rsi) / 50)  # 0.7-1.0
elif rsi < oversold + 15:
    confidence = 0.5 + (oversold + 15 - rsi) / 30 * 0.3  # 0.5-0.8
else:
    confidence = max(0.2, 0.5 - 0.1)  # 0.2-0.4
```

#### Adaptive RSI Zones

During volatile markets, standard RSI thresholds (30/70) may trigger too frequently. The strategy uses ATR-based zone expansion:

**Zone Expansion Formula:**
```python
atr_pct = (current_atr / current_price) * 100
expansion = min(zone_expansion_limit, atr_pct * 2)
adaptive_oversold = max(15, base_oversold - expansion)
adaptive_overbought = min(85, base_overbought + expansion)
```

### 2.4 Entry/Exit Signal Criteria

#### Grid Buy Entry Conditions

| Priority | Condition | Weight |
|----------|-----------|--------|
| Required | Price <= Grid Level Price + Slippage Tolerance | Primary |
| Required | Grid Level Side == 'buy' | Primary |
| Required | Grid Level Not Filled | Primary |
| Optional | RSI < Adaptive Oversold | +20% confidence |
| Optional | RSI < Adaptive Oversold + 15 | +10% confidence |
| Optional | Volume Spike (1.5x avg) | +10% confidence |

#### Grid Sell Entry Conditions

| Priority | Condition | Weight |
|----------|-----------|--------|
| Required | Price >= Grid Level Price - Slippage Tolerance | Primary |
| Required | Grid Level Side == 'sell' | Primary |
| Required | Grid Level Not Filled | Primary |
| Optional | RSI > Adaptive Overbought | +20% confidence |
| Optional | RSI > Adaptive Overbought - 15 | +10% confidence |
| Optional | Volume Spike (1.5x avg) | +10% confidence |

#### Exit Conditions

| Exit Type | Trigger | Action |
|-----------|---------|--------|
| Grid Cycle Complete | Corresponding sell order fills | Close position, book profit |
| Stop Loss | Price < Lower Grid - Stop Distance | Close all positions |
| Take Profit | Price > Upper Grid + TP Distance | Close all positions |
| Max Drawdown | Drawdown > Threshold | Close all positions |
| Time-Based Recentering | Cycles completed > N | Recenter grid around current price |

### 2.5 Risk Management for Grid Strategies

#### Position Accumulation Limits

**Problem:** Grid strategies buy more as price falls, potentially accumulating large positions.

**Mitigation Strategies:**
1. **Max Positions per Level:** Limit fills per grid level
2. **Total Position Cap:** Maximum total position across all levels
3. **Capital Reserve:** Keep 20-30% capital unfilled for emergencies
4. **Dynamic Position Sizing:** Reduce size as accumulation increases

#### Stop-Loss Placement for Grids

**Below-Grid Stop:**
```
Stop Loss = Lower Grid Price - (ATR × Multiplier)
Example: $88,000 (lower grid) - ($500 × 2) = $87,000
```

**Percentage-Based Stop:**
```
Stop Loss = Entry Average Price × (1 - Stop Loss %)
Example: $90,000 × (1 - 3%) = $87,300
```

#### Capital Allocation Across Grid Levels

**Equal Allocation:**
```python
capital_per_level = total_capital / num_grids
Example: $1000 / 20 levels = $50 per level
```

**Weighted Allocation (Legacy Approach):**
```python
# RSI extreme multiplier from legacy code
if rsi < oversold:
    multiplier = rsi_extreme_multiplier  # 1.3x
elif rsi < oversold + 10:
    multiplier = 1.0 + (rsi_extreme_multiplier - 1.0) * 0.5  # 1.15x
else:
    multiplier = 1.0
```

### 2.6 Known Pitfalls and Failure Modes

#### Critical Pitfalls

##### 1. Trending Market Trap

**Problem:**
> "Grid trading performs best in volatile and sideways markets when prices fluctuate in a given range. A range that's too narrow will be broken quickly, stopping your strategy." - [Cloudzy Grid Trading](https://cloudzy.com/blog/best-coin-pairs-for-grid-trading/)

**Mitigation:**
- Trend detection before grid initialization
- Pause strategy when ADX > 30 (strong trend)
- Use wider grid range (±10-15% from center)

##### 2. Grid Range Breakout

**Problem:** Price escapes grid range, leaving unfilled orders and potential losses.

**Mitigation:**
- Monitor price vs grid boundaries
- Recenter grid after N completed cycles
- Use adaptive range based on ATR

##### 3. Over-Accumulation in Downtrends

**Problem:** Buying every dip in a sustained downtrend locks capital and increases drawdown.

**Mitigation:**
- Position limit per symbol
- Max grid accumulation threshold
- Reduce position size as accumulation increases

##### 4. RSI Divergence False Signals

**Problem:**
> "RSI can sometimes produce false signals, leading to premature entry or exit from trades. As a momentum oscillator, RSI is inherently a lagging indicator." - [TIOMarkets RSI Guide](https://tiomarkets.com/en/article/relative-strength-index-guide-in-mean-reversion-trading)

**Mitigation:**
- Use RSI as confidence modifier, not hard filter
- Combine with price action (grid levels)
- Multi-timeframe RSI confirmation

##### 5. Mean Reversion Failure (Regime Change)

**Problem:**
> "All trading carries risk, and mean reversion is no exception. One major risk is that prices might not revert to the mean quickly — or at all — especially if a new trend forms." - [3commas](https://3commas.io/mean-reversion-trading-bot)

**Mitigation:**
- Time stop at 2× half-life
- Volatility regime monitoring
- Circuit breaker on consecutive losses

---

## 3. Source Code Analysis

### 3.1 Legacy Source Files Overview

| File | Purpose | Lines | Translates? |
|------|---------|-------|-------------|
| `src/strategies/grid_base.py` | Core grid strategies including RSIMeanReversionGrid | ~2000 | Partial (extract RSI logic) |
| `src/strategies/grid_wrappers.py` | Adapters for orchestrator interface | ~1954 | Yes (modify for ws_tester) |
| `src/grid_ensemble_orchestrator.py` | Multi-strategy paper trading | ~879 | No (ws_tester has own executor) |

### 3.2 RSIMeanReversionGrid Class Analysis

**Location:** `src/strategies/grid_base.py:1014-1401`

#### Core Components to Extract

##### 1. RSI Calculation (Reuse)

```python
def _calculate_rsi(self, closes: np.ndarray) -> float:
    """Calculate RSI using Wilder's smoothing method."""
    if len(closes) < self.rsi_period + 1:
        return 50.0

    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    # Use exponential moving average for smoother RSI
    alpha = 1.0 / self.rsi_period
    avg_gain = np.mean(gains[:self.rsi_period])
    avg_loss = np.mean(losses[:self.rsi_period])

    for i in range(self.rsi_period, len(gains)):
        avg_gain = alpha * gains[i] + (1 - alpha) * avg_gain
        avg_loss = alpha * losses[i] + (1 - alpha) * avg_loss

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))
```

**Status:** Can be adapted from existing momentum_scalping indicators.py

##### 2. Confidence Calculation (Reuse with Modification)

```python
def _calculate_confidence(self, side: str, base_confidence: float = 0.5) -> float:
    """Calculate confidence based on RSI position."""
    if side == 'buy':
        if self.current_rsi < self.rsi_oversold:
            boost = min(0.4, (self.rsi_oversold - self.current_rsi) / 50)
            return min(1.0, 0.7 + boost)
        elif self.current_rsi < self.rsi_oversold + 15:
            boost = (self.rsi_oversold + 15 - self.current_rsi) / 30
            return min(1.0, base_confidence + boost * 0.3)
        else:
            return max(0.2, base_confidence - 0.1)
    # ... similar for sell
```

**Status:** Extract and adapt for ws_tester Signal confidence field

##### 3. Grid Setup Logic (Significant Rewrite)

```python
def _setup_grid(self, current_price: float = None):
    """Set up grid with smart buy/sell assignment."""
    price_range = self.upper_price - self.lower_price

    # ATR-based spacing
    if self.use_atr_spacing and self.current_atr > 0:
        spacing = self.current_atr * self.atr_multiplier
        self.num_grids = max(5, int(price_range / spacing))
    else:
        spacing = price_range / self.num_grids

    self.grid_levels = []
    for i in range(self.num_grids + 1):
        price = self.lower_price + (i * spacing)
        size = self._calculate_position_size(i, price)

        # Below current = buy, Above current = sell
        if price < reference_price * 0.998:
            side = 'buy'
        elif price > reference_price * 1.002:
            side = 'sell'
        else:
            side = 'buy'  # Accumulation bias

        self.grid_levels.append(GridLevel(...))
```

**Status:** Requires significant rewrite for ws_tester patterns (no GridLevel class, use state dict)

##### 4. Adaptive RSI Zones (Reuse)

```python
def _get_adaptive_rsi_zones(self) -> Tuple[float, float]:
    """Get RSI zones adjusted by volatility."""
    if not self.use_adaptive_rsi or self.current_atr <= 0:
        return self.rsi_oversold, self.rsi_overbought

    atr_pct = (self.current_atr / self.current_price * 100)
    expansion = min(self.rsi_zone_expansion, atr_pct * 2)

    adaptive_oversold = max(15, self.rsi_oversold - expansion)
    adaptive_overbought = min(85, self.rsi_overbought + expansion)

    return adaptive_oversold, adaptive_overbought
```

**Status:** Direct extraction viable

##### 5. Update/Signal Generation (Major Rewrite)

The legacy `update()` method returns a list of signal dicts with grid-specific fields (`grid_level`, `order_id`, `target_sell_price`). These need mapping to ws_tester's Signal dataclass.

**Legacy Signal Format:**
```python
{
    'action': 'buy',
    'price': level.price,
    'size': level.size,
    'grid_level': i,
    'order_id': level.order_id,
    'confidence': 0.7,
    'reason': 'Grid buy at level 5',
    'rsi': 28.5,
    'rsi_zone': 'oversold'
}
```

**ws_tester Signal Format:**
```python
Signal(
    action='buy',
    symbol='XRP/USDT',
    size=20.0,  # USD
    price=2.35,
    reason="RSI oversold (28.5), grid level 5",
    stop_loss=2.30,
    take_profit=2.40,
    metadata={'grid_level': 5, 'rsi_zone': 'oversold'}
)
```

### 3.3 RSIMeanReversionGridWrapper Analysis

**Location:** `src/strategies/grid_wrappers.py:645-866`

#### Key Adaptations Required

##### 1. Multi-Pair Support

The wrapper supports BTC and XRP via `secondary_grid`. For ws_tester, we'll handle multiple symbols in the signal generation loop instead.

##### 2. Size Conversion

```python
def _convert_size_to_pct(self, size_asset: float, price: float) -> float:
    """Convert absolute asset size to percentage of capital."""
    size_usd = size_asset * price
    reference_capital = max(total_capital * 2, 2000)
    size_pct = size_usd / reference_capital
    return max(0.01, min(0.25, size_pct))
```

**Status:** ws_tester uses USD-based sizing directly, simpler approach

##### 3. Lifecycle Callbacks

```python
def on_order_filled(self, order: Dict[str, Any]) -> None:
    """Callback when order is filled - sync grid state."""
    grid_level = order.get('grid_level')
    fill_price = order.get('price', 0)
    if grid_level is not None:
        self.grid_strategy.fill_order(grid_level, fill_price)
```

**Status:** Map to ws_tester's `on_fill()` pattern

### 3.4 Component Translation Matrix

| Legacy Component | ws_tester Equivalent | Translation Effort |
|------------------|---------------------|-------------------|
| `GridLevel` dataclass | `state['grid_levels']` list | Medium |
| `GridStats` dataclass | `state['stats']` dict | Low |
| `_calculate_rsi()` | `indicators.py calculate_rsi()` | Exists (reuse) |
| `_calculate_atr()` | `indicators.py calculate_atr()` | Exists (reuse) |
| `_setup_grid()` | `lifecycle.py on_start()` | High |
| `update()` | `signal.py generate_signal()` | High |
| `fill_order()` | `lifecycle.py on_fill()` | Medium |
| `get_status()` | `state['indicators']` logging | Low |
| `should_recenter()` | `regimes.py` function | Medium |
| `_get_adaptive_rsi_zones()` | `indicators.py` function | Low |

---

## 4. Pair-Specific Analysis

### 4.1 XRP/USDT Analysis

#### Market Characteristics

| Metric | Value | Source |
|--------|-------|--------|
| Current Price Range | $1.95-2.35 | TradingView (Dec 2025) |
| 24h Volume | $8.2B+ market-wide | CoinMarketCap |
| Intraday Volatility | 5.1% | Research |
| Typical Spread | 0.15% | Kraken |
| Range Behavior | Strong support $1.95-2.17 | TradingView |

#### Grid Trading Suitability

**Strengths:**
- High volatility (5.1% intraday) provides range opportunities
- Tight spreads preserve grid profits
- Strong support/resistance zones for grid boundaries
- XRP price "rejecting upwards" from support indicates mean reversion

**Concerns:**
- Can break out explosively (300%+ moves in late 2024)
- Consolidation periods can extend weeks
- Bollinger bandwidth compression signals impending breakouts

**Research Finding:**
> "XRP has shown a familiar pattern—strong early rallies followed by an extended consolidation phase. Since July, the token has been trading within a contracting triangle, defined by lower highs and higher lows, signaling a tightening market range." - [TradingView Analysis](https://www.tradingview.com/symbols/XRPUSDT/)

#### Recommended Grid Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Grid Type | Geometric | Matches crypto % movement |
| Num Grids | 15 | Balance between granularity and capital |
| Grid Spacing | 1.5% | Matches typical intraday swings |
| Total Range | ±7.5% | Covers typical consolidation |
| RSI Period | 14 | Standard mean reversion |
| RSI Oversold | 30 | Standard threshold |
| RSI Overbought | 70 | Standard threshold |
| Position Size | $20-25 USD per level | Per dev guide limits |
| Max Accumulation | 5 levels | Prevent over-exposure |

#### Suitability: HIGH

XRP/USDT is well-suited for Grid RSI Reversion during consolidation phases.

### 4.2 BTC/USDT Analysis

#### Market Characteristics

| Metric | Value | Source |
|--------|-------|--------|
| Current Price | $90,000-100,000 | TradingView (Dec 2025) |
| Spot Volume (Binance) | $45.9B daily | Nansen |
| Typical Spread | <0.02% | Binance |
| Institutional Share | ~80% CEX volume | Bitget Research |
| Monthly Volatility | 12-18% | Wundertrading |

#### Grid Trading Suitability

**Strengths:**
- Deepest liquidity globally
- Minimal slippage on grid orders
- Predictable support/resistance zones (ETF-driven)
- 12-18% monthly volatility ideal for grid range

**Concerns:**
- Institutional algorithms compete for range trades
- ETF arbitrage creates ceiling/floor patterns
- Lower volatility means smaller profit per cycle

**Research Finding:**
> "Bitcoin (BTC) leads as the top choice for grid trading, with $35+ billion daily volume and 12-18% monthly volatility providing sufficient movement for profitable execution while maintaining predictable support/resistance zones." - [Wundertrading](https://wundertrading.com/journal/en/learn/article/best-grid-bot-settings)

#### Recommended Grid Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Grid Type | Arithmetic | Works well for established ranges |
| Num Grids | 20 | Finer granularity for lower volatility |
| Grid Spacing | 1.0% | Matches BTC typical intraday moves |
| Total Range | ±10% | Conservative range |
| RSI Period | 14 | Standard |
| RSI Oversold | 35 | Slightly relaxed (BTC trends) |
| RSI Overbought | 65 | Slightly relaxed |
| Position Size | $40-50 USD per level | Higher due to lower volatility |
| Max Accumulation | 4 levels | More conservative |

#### Suitability: MEDIUM-HIGH

BTC/USDT is suitable but requires wider ranges and more conservative RSI thresholds due to institutional efficiency.

### 4.3 XRP/BTC Analysis

#### Market Characteristics

| Metric | Value | Source |
|--------|-------|--------|
| 24h Volume | ~$160M (~1,608 BTC) | Binance |
| Liquidity vs XRP/USDT | 7-10x lower | Analysis |
| 3-Month Correlation | 0.84 | MacroAxis |
| Correlation Trend | -24.86% over 90 days | MacroAxis (declining) |
| XRP vs BTC Volatility | XRP 1.55x more volatile | Historical |

#### Cross-Pair Dynamics

**Research Finding:**
> "XRP is more weakly correlated to BTC, ETH and SOL than they are to one another. XRP rallied nearly 500% last November after many years of underperformance relative to other cryptocurrencies." - [CME Group](https://www.cmegroup.com/insights/economic-research/2025/how-xrp-relates-to-the-crypto-universe-and-the-broader-economy.html)

**Pairs Trading Potential:**
> "By analyzing existing cross correlation between XRP and Bitcoin, you can compare the effects of market volatilities and check how they will diversify away market risk if combined in the same portfolio." - [MacroAxis](https://www.macroaxis.com/invest/pair-correlation/XRP.CC/BTC.CC/XRP-vs-Bitcoin)

#### Grid Trading Suitability

**Strengths:**
- Declining correlation (independent moves)
- Higher volatility ratio provides opportunities
- Less efficient market = more alpha potential
- Ratio mean reversion potential

**Concerns:**
- 7-10x lower liquidity (slippage risk)
- Wider effective spreads
- More difficult to exit large positions
- Correlation breakdowns can be extended

#### Recommended Grid Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Grid Type | Geometric | Match XRP volatility |
| Num Grids | 10 | Fewer due to liquidity |
| Grid Spacing | 2.0% | Wider to account for slippage |
| Total Range | ±10% | Conservative |
| RSI Period | 14 | Standard |
| RSI Oversold | 25 | More aggressive (ratio moves) |
| RSI Overbought | 75 | More aggressive |
| Position Size | $10-15 USD per level | Smaller due to slippage |
| Max Accumulation | 3 levels | Very conservative |
| Cooldown | 10-15 minutes | Fewer, higher-quality signals |

#### Suitability: MEDIUM

XRP/BTC is suitable as a secondary pair with reduced position sizing and wider parameters.

### 4.4 Cross-Pair Risk Management

#### Correlation Matrix (December 2025)

| Pair A | Pair B | Correlation | Trading Implication |
|--------|--------|-------------|---------------------|
| XRP/USDT | BTC/USDT | 0.84 | Simultaneous signals likely |
| XRP/USDT | XRP/BTC | ~0.50 | Moderate independence |
| BTC/USDT | XRP/BTC | ~-0.30 | Inverse relationship |

#### Position Limits

| Limit Type | Value | Rationale |
|------------|-------|-----------|
| Max Long per Symbol | $100 | Single symbol concentration |
| Max Long Total | $150 | Correlation exposure |
| Max Short per Symbol | N/A | Grid strategy is long-only |
| Same Direction Multiplier | 0.75x | Reduce size if XRP & BTC same direction |

---

## 5. Recommended Approach

### 5.1 Strategy Architecture

#### Module Structure

Following the established pattern from momentum_scalping:

```
strategies/
  grid_rsi_reversion/
    __init__.py           # Public API exports
    config.py             # Configuration, enums, per-symbol settings
    signal.py             # Core signal generation
    indicators.py         # RSI, ATR calculations (extend existing)
    grid.py               # Grid-specific logic (setup, levels, cycles)
    regimes.py            # Volatility and trend detection
    risk.py               # Position limits, accumulation checks
    exits.py              # Stop loss, take profit, cycle completion
    lifecycle.py          # on_start, on_fill, on_stop callbacks
    validation.py         # Configuration validation
  grid_rsi_reversion.py   # Strategy entry point (imports from package)
```

#### Signal Generation Flow

```
1. Initialize state + grid levels (on_start or first call)
       ↓
2. Check trend filter (ADX < 30, not strong trend)
       ↓
3. Check volatility regime (adjust RSI zones)
       ↓
4. Calculate indicators (RSI, ATR)
       ↓
5. Check existing position exits first:
   - Cycle completion (matched sell)
   - Stop loss
   - Max drawdown
       ↓
6. Check grid entry conditions:
   - Price at/near grid level
   - Grid level not filled
   - RSI zone (confidence modifier)
       ↓
7. Check risk limits:
   - Max accumulation
   - Position limits
   - Correlation exposure
       ↓
8. Generate Signal or None
```

### 5.2 Core Algorithm

#### Grid Level Structure (State-Based)

```python
state['grid_levels'] = [
    {
        'price': 2.20,
        'side': 'buy',
        'size': 20.0,  # USD
        'filled': False,
        'fill_price': None,
        'fill_time': None,
        'order_id': 'grid_001_buy_0',
        'matched_sell_id': 'grid_001_sell_0',  # For cycle tracking
    },
    # ... more levels
]
```

#### Signal Generation Pseudocode

```python
def generate_signal(data, config, state):
    # Initialize on first call
    if 'initialized' not in state:
        on_start(config, state)
        return None  # Wait for next tick

    for symbol in SYMBOLS:
        price = data.prices.get(symbol)
        if not price:
            continue

        candles = data.candles_5m.get(symbol, ())
        if len(candles) < config['min_candles']:
            continue

        # Calculate indicators
        rsi = calculate_rsi(candles, config['rsi_period'])
        atr = calculate_atr(candles, config['atr_period'])

        # Get adaptive RSI zones
        oversold, overbought = get_adaptive_rsi_zones(rsi, atr, price, config)

        # Check for exits first
        exit_signal = check_grid_exits(data, config, state, symbol)
        if exit_signal:
            return exit_signal

        # Check grid levels for entries
        for level in state['grid_levels'][symbol]:
            if level['filled']:
                continue

            # Check if price hit level
            if level['side'] == 'buy' and price <= level['price'] * 1.005:
                # Calculate confidence based on RSI
                confidence = calculate_rsi_confidence('buy', rsi, oversold, overbought)

                # Check risk limits
                if not check_accumulation_limit(state, symbol, config):
                    continue

                return Signal(
                    action='buy',
                    symbol=symbol,
                    size=level['size'],
                    price=price,
                    reason=f"Grid buy at ${level['price']:.2f}, RSI={rsi:.1f}",
                    stop_loss=calculate_grid_stop_loss(state, symbol, config),
                    take_profit=None,  # Grid uses matched sell, not TP
                    metadata={
                        'grid_level': level['order_id'],
                        'rsi_zone': 'oversold' if rsi < oversold else 'neutral',
                        'confidence': confidence,
                    }
                )

    return None
```

### 5.3 Configuration Design

#### Default Configuration

```python
CONFIG = {
    # === Grid Settings ===
    'grid_type': 'geometric',           # 'arithmetic' or 'geometric'
    'num_grids': 15,                    # Grid levels per side
    'grid_spacing_pct': 1.5,            # Spacing between levels (%)
    'range_pct': 7.5,                   # Range from center (±%)
    'recenter_after_cycles': 5,         # Recenter grid after N cycles

    # === RSI Settings ===
    'rsi_period': 14,
    'rsi_oversold': 30,
    'rsi_overbought': 70,
    'use_adaptive_rsi': True,
    'rsi_zone_expansion': 5,            # Max expansion by ATR
    'rsi_mode': 'confidence_only',      # 'confidence_only' or 'filter'
    'rsi_extreme_multiplier': 1.3,      # Size multiplier at RSI extremes

    # === ATR Settings ===
    'atr_period': 14,
    'use_atr_spacing': True,
    'atr_multiplier': 0.3,

    # === Position Sizing ===
    'position_size_usd': 20.0,          # Base size per grid level
    'max_position_usd': 100.0,          # Max total position per symbol
    'max_accumulation_levels': 5,       # Max filled grid levels

    # === Risk Management ===
    'stop_loss_pct': 3.0,               # Below lowest grid level
    'max_drawdown_pct': 10.0,
    'use_trend_filter': True,
    'adx_threshold': 30,                # Pause if ADX > threshold

    # === Fees ===
    'fee_rate': 0.001,                  # 0.1%

    # === Cooldowns ===
    'min_recenter_interval': 3600,      # 1 hour between recenters
}
```

#### Per-Symbol Configuration

```python
SYMBOL_CONFIGS = {
    'XRP/USDT': {
        'grid_type': 'geometric',
        'num_grids': 15,
        'grid_spacing_pct': 1.5,
        'position_size_usd': 25.0,
        'max_accumulation_levels': 5,
        'rsi_oversold': 30,
        'rsi_overbought': 70,
    },
    'BTC/USDT': {
        'grid_type': 'arithmetic',
        'num_grids': 20,
        'grid_spacing_pct': 1.0,
        'position_size_usd': 50.0,
        'max_accumulation_levels': 4,
        'rsi_oversold': 35,
        'rsi_overbought': 65,
    },
    'XRP/BTC': {
        'grid_type': 'geometric',
        'num_grids': 10,
        'grid_spacing_pct': 2.0,
        'position_size_usd': 15.0,
        'max_accumulation_levels': 3,
        'rsi_oversold': 25,
        'rsi_overbought': 75,
        'cooldown_minutes': 15,
    },
}
```

### 5.4 Key Design Decisions

#### Decision 1: RSI Mode

**Decision:** RSI as confidence modifier (not filter)

**Rationale:**
- Grid levels provide primary entry signal
- RSI enhances confidence and position sizing
- Avoids missing grid opportunities when RSI is neutral
- Legacy code Phase 26 uses this approach successfully

#### Decision 2: Grid Type Default

**Decision:** Geometric grid as default

**Rationale:**
- Crypto moves in percentages, not dollars
- Better capital distribution across price range
- Research supports geometric for volatile assets

#### Decision 3: Position Sizing

**Decision:** USD-based sizing per grid level

**Rationale:**
- Matches ws_tester Signal size convention
- Simpler than percentage-of-capital approach
- Easier to reason about total exposure

#### Decision 4: Multi-Position Management

**Decision:** Track each grid position independently via state

**Rationale:**
- Grid strategy requires tracking multiple simultaneous positions
- Cycle completion requires matching buy → sell pairs
- on_fill must update specific grid level state

#### Decision 5: Recentering Approach

**Decision:** Cycle-based recentering (not price-based)

**Rationale:**
- Price-based recentering causes thrashing in trends
- Cycle completion indicates successful range trading
- Minimum time interval prevents excessive recentering

---

## 6. Development Plan

### Phase 1: Foundation

#### 1.1 Module Setup

- [ ] Create `strategies/grid_rsi_reversion/` directory structure
- [ ] Implement `config.py` with CONFIG, SYMBOL_CONFIGS, enums
- [ ] Implement `validation.py` for configuration validation
- [ ] Create `__init__.py` with public API exports
- [ ] Create entry point `strategies/grid_rsi_reversion.py`

#### 1.2 Grid Infrastructure

- [ ] Implement `grid.py`:
  - [ ] `setup_grid_levels()` - Create grid level structure
  - [ ] `calculate_grid_prices()` - Arithmetic/geometric spacing
  - [ ] `get_unfilled_levels()` - Find tradeable levels
  - [ ] `should_recenter_grid()` - Recentering logic
  - [ ] `recenter_grid()` - Update grid around new center

#### 1.3 Lifecycle Callbacks

- [ ] Implement `lifecycle.py`:
  - [ ] `on_start()` - Initialize grid state
  - [ ] `on_fill()` - Update grid level status, track cycle completion
  - [ ] `on_stop()` - Log summary statistics

### Phase 2: Signal Generation

#### 2.1 Indicator Functions

- [ ] Implement/extend `indicators.py`:
  - [ ] `calculate_rsi()` - Reuse from momentum_scalping
  - [ ] `calculate_atr()` - Reuse from momentum_scalping
  - [ ] `get_adaptive_rsi_zones()` - Extract from legacy
  - [ ] `calculate_rsi_confidence()` - Port from legacy

#### 2.2 Entry Signal Logic

- [ ] Implement `signal.py`:
  - [ ] State initialization check
  - [ ] Trend filter (ADX check)
  - [ ] Grid level price matching
  - [ ] RSI confidence calculation
  - [ ] Signal generation with metadata

#### 2.3 Exit Logic

- [ ] Implement `exits.py`:
  - [ ] `check_cycle_completion()` - Matched sell for buy position
  - [ ] `check_grid_stop_loss()` - Below lowest grid level
  - [ ] `check_max_drawdown()` - Portfolio-level check
  - [ ] `check_time_based_exit()` - Stale position cleanup

### Phase 3: Risk Management

#### 3.1 Position Limits

- [ ] Implement `risk.py`:
  - [ ] `check_accumulation_limit()` - Max filled grid levels
  - [ ] `check_position_limits()` - Per-symbol and total
  - [ ] `check_trend_filter()` - ADX-based pause

#### 3.2 Correlation Management

- [ ] Cross-pair exposure tracking
- [ ] Same-direction size reduction
- [ ] Total exposure limits

### Phase 4: Regime Classification

#### 4.1 Volatility Regimes

- [ ] Implement `regimes.py`:
  - [ ] `classify_volatility_regime()` - ATR-based
  - [ ] `get_regime_adjustments()` - RSI zone expansion
  - [ ] `check_trend_strength()` - ADX calculation

#### 4.2 Session Awareness (Optional)

- [ ] Session classification (lower priority for grid)
- [ ] Session-based grid spacing adjustment

### Phase 5: Testing & Validation

#### 5.1 Unit Tests

- [ ] Test grid level calculation
- [ ] Test RSI confidence calculation
- [ ] Test cycle completion tracking
- [ ] Test accumulation limits

#### 5.2 Integration Testing

- [ ] Paper trading session (24-48 hours)
- [ ] All pairs coverage
- [ ] Range-bound market verification
- [ ] Trending market pause verification

#### 5.3 Metrics Validation

- [ ] Grid cycle completion >= 60%
- [ ] Win rate >= 55%
- [ ] Max drawdown <= 10%
- [ ] Capital efficiency >= 50%

### Phase 6: Documentation & Review

#### 6.1 Code Documentation

- [ ] Inline comments for grid logic
- [ ] Docstrings for all public functions
- [ ] Version history updates

#### 6.2 Strategy Documentation

- [ ] Create feature documentation
- [ ] Update review document
- [ ] Create BACKLOG.md for enhancements

---

## 7. Compliance Checklist

### Strategy Development Guide v1.0 Compliance

#### Section 2: Strategy Module Contract

| Requirement | Status | Notes |
|-------------|--------|-------|
| STRATEGY_NAME defined | PLANNED | "grid_rsi_reversion" |
| STRATEGY_VERSION defined | PLANNED | "1.0.0" |
| SYMBOLS list defined | PLANNED | ["XRP/USDT", "BTC/USDT", "XRP/BTC"] |
| CONFIG dictionary defined | PLANNED | See Section 5.3 |
| generate_signal() function | PLANNED | Main entry point |

#### Section 3: Signal Generation

| Requirement | Status | Notes |
|-------------|--------|-------|
| Returns Signal or None | PLANNED | Required interface |
| Signal includes action, symbol, size, price, reason | PLANNED | Per guide spec |
| stop_loss and take_profit optional fields | PLANNED | stop_loss included |
| Informative reason field | PLANNED | Include RSI, grid level |
| metadata field | PLANNED | Grid-specific data |

#### Section 4: Stop Loss & Take Profit

| Requirement | Status | Notes |
|-------------|--------|-------|
| SL below entry for long | PLANNED | Below lowest grid level |
| TP above entry for long | N/A | Grid uses cycle completion |
| R:R ratio documented | PLANNED | Grid spacing defines R:R |

#### Section 5: Position Management

| Requirement | Status | Notes |
|-------------|--------|-------|
| Size in USD (not base asset) | PLANNED | position_size_usd |
| Max position limits | PLANNED | Per-symbol and total |
| Multiple position support | PLANNED | Grid level tracking |

#### Section 6: State Management

| Requirement | Status | Notes |
|-------------|--------|-------|
| State dict persistence | PLANNED | Via state parameter |
| Lazy initialization | PLANNED | On first generate_signal |
| Bounded buffers | PLANNED | Max candle storage |
| Grid level state | PLANNED | state['grid_levels'] |

#### Section 7: Logging Requirements

| Requirement | Status | Notes |
|-------------|--------|-------|
| state['indicators'] populated | PLANNED | RSI, ATR, grid metrics |
| All code paths log indicators | PLANNED | Before every return |
| Informative signal reasons | PLANNED | Include key metrics |

#### Section 8: Data Access Patterns

| Requirement | Status | Notes |
|-------------|--------|-------|
| Use DataSnapshot correctly | PLANNED | Per guide patterns |
| Safe access with .get() | PLANNED | Handle missing data |
| Check candle count | PLANNED | Minimum data validation |

#### Section 9: Configuration Best Practices

| Requirement | Status | Notes |
|-------------|--------|-------|
| Structured CONFIG dict | PLANNED | Grouped by category |
| Sensible defaults | PLANNED | Based on research |
| Runtime validation | PLANNED | In validation.py |

#### Section 11: Common Pitfalls Avoided

| Pitfall | Status | Mitigation |
|---------|--------|------------|
| Signal on every tick | PLANNED | Only at grid levels |
| Not checking position | PLANNED | Accumulation limits |
| Stop loss on wrong side | PLANNED | Below grid for longs |
| Unbounded state growth | PLANNED | Bounded grid levels |
| Missing data checks | PLANNED | Safe access patterns |
| Forgetting on_fill | PLANNED | Grid state updates |
| Size confusion | PLANNED | USD-based sizing |

### Additional Requirements

| Requirement | Status | Notes |
|-------------|--------|-------|
| Volatility regime classification | PLANNED | ATR-based zones |
| Trend filter | PLANNED | ADX threshold |
| Per-symbol configuration | PLANNED | SYMBOL_CONFIGS |
| Cross-pair correlation | PLANNED | Exposure limits |

---

## 8. Research References

### Grid Trading Fundamentals

1. **Grid Trading Strategy 2025 Guide** - Zignaly
   [https://zignaly.com/crypto-trading/algorithmic-strategies/grid-trading](https://zignaly.com/crypto-trading/algorithmic-strategies/grid-trading)

2. **Best Grid Bot Settings** - Wundertrading
   [https://wundertrading.com/journal/en/learn/article/best-grid-bot-settings](https://wundertrading.com/journal/en/learn/article/best-grid-bot-settings)

3. **Grid Trading Strategy Setup Guide** - Cloudzy
   [https://cloudzy.com/blog/best-coin-pairs-for-grid-trading/](https://cloudzy.com/blog/best-coin-pairs-for-grid-trading/)

### Mean Reversion Theory

4. **Half-Life of Mean Reversion** - Flare9x Blog
   [https://flare9xblog.wordpress.com/2017/09/27/half-life-of-mean-reversion-ornstein-uhlenbeck-formula-for-mean-reverting-process/](https://flare9xblog.wordpress.com/2017/09/27/half-life-of-mean-reversion-ornstein-uhlenbeck-formula-for-mean-reverting-process/)

5. **Ornstein-Uhlenbeck for Crypto Mean Reversion** - Janelle Turing
   [https://janelleturing.medium.com/python-ornstein-uhlenbeck-for-crypto-mean-reversion-trading-287856264f7a](https://janelleturing.medium.com/python-ornstein-uhlenbeck-for-crypto-mean-reversion-trading-287856264f7a)

6. **Mean Reversion Trading Bot** - 3Commas
   [https://3commas.io/mean-reversion-trading-bot](https://3commas.io/mean-reversion-trading-bot)

### RSI Mean Reversion

7. **RSI in Mean Reversion Trading** - TIOMarkets
   [https://tiomarkets.com/en/article/relative-strength-index-guide-in-mean-reversion-trading](https://tiomarkets.com/en/article/relative-strength-index-guide-in-mean-reversion-trading)

8. **Larry Connors RSI2 Strategy** - FMZQuant
   [https://medium.com/@FMZQuant/larry-connors-rsi2-mean-reversion-strategy-861f5a3579e3](https://medium.com/@FMZQuant/larry-connors-rsi2-mean-reversion-strategy-861f5a3579e3)

9. **RSI Trading Strategy Backtest** - Quantified Strategies
   [https://www.quantifiedstrategies.com/rsi-trading-strategy/](https://www.quantifiedstrategies.com/rsi-trading-strategy/)

### Pair-Specific Research

10. **XRP-Bitcoin Correlation** - MacroAxis
    [https://www.macroaxis.com/invest/pair-correlation/XRP.CC/BTC.CC/XRP-vs-Bitcoin](https://www.macroaxis.com/invest/pair-correlation/XRP.CC/BTC.CC/XRP-vs-Bitcoin)

11. **How XRP Relates to the Crypto Universe** - CME Group
    [https://www.cmegroup.com/insights/economic-research/2025/how-xrp-relates-to-the-crypto-universe-and-the-broader-economy.html](https://www.cmegroup.com/insights/economic-research/2025/how-xrp-relates-to-the-crypto-universe-and-the-broader-economy.html)

12. **XRP/USDT Charts** - TradingView
    [https://www.tradingview.com/symbols/XRPUSDT/](https://www.tradingview.com/symbols/XRPUSDT/)

13. **XRP Volatility Analysis** - CoinDesk
    [https://www.coindesk.com/markets/2025/04/20/xrp-resembles-a-compressed-spring-poised-for-a-significant-move-as-key-volatility-indicator-mirrors-late-2024-pattern](https://www.coindesk.com/markets/2025/04/20/xrp-resembles-a-compressed-spring-poised-for-a-significant-move-as-key-volatility-indicator-mirrors-late-2024-pattern)

### Internal Documentation

14. Strategy Development Guide v1.0
15. Legacy `src/strategies/grid_base.py` (RSIMeanReversionGrid)
16. Legacy `src/strategies/grid_wrappers.py` (RSIMeanReversionGridWrapper)
17. Momentum Scalping Master Plan v1.0 (document format reference)

---

## Appendix A: Grid Level Data Structure

```python
# State structure for grid levels
state['grid_levels'] = {
    'XRP/USDT': [
        {
            'level_index': 0,
            'price': 2.15,
            'side': 'buy',
            'size': 25.0,
            'filled': False,
            'fill_price': None,
            'fill_time': None,
            'order_id': 'xrp_grid_buy_0',
            'matched_order_id': 'xrp_grid_sell_0',
            'original_side': 'buy',
        },
        # ... more levels
    ],
    'BTC/USDT': [...],
    'XRP/BTC': [...],
}

# Grid metadata
state['grid_metadata'] = {
    'XRP/USDT': {
        'center_price': 2.25,
        'upper_price': 2.42,
        'lower_price': 2.08,
        'grid_spacing': 0.023,
        'last_recenter_time': datetime(...),
        'cycles_completed': 5,
        'cycles_at_last_recenter': 3,
    },
}
```

---

## Appendix B: Signal Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    generate_signal()                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  1. Check initialization (on_start if needed)               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  2. Check trend filter (ADX < 30)                           │
│     └── If strong trend: return None (pause trading)        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  3. Calculate indicators:                                   │
│     - RSI (14 period)                                       │
│     - ATR (14 period)                                       │
│     - Adaptive RSI zones                                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  4. Check exits for existing positions:                     │
│     - Cycle completion (matched sell filled)                │
│     - Stop loss (below lowest grid)                         │
│     - Max drawdown                                          │
│     └── If exit triggered: return exit Signal               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  5. Check grid entries:                                     │
│     FOR each unfilled grid level:                           │
│       - Price within tolerance of level?                    │
│       - Calculate RSI confidence                            │
│       - Check accumulation limits                           │
│       └── If conditions met: return entry Signal            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  6. No signal generated:                                    │
│     - Update state['indicators'] for logging                │
│     - Return None                                           │
└─────────────────────────────────────────────────────────────┘
```

---

## Appendix C: Comparison with Legacy Implementation

| Feature | Legacy RSIMeanReversionGrid | ws_tester Grid RSI Reversion |
|---------|----------------------------|------------------------------|
| Grid Level Storage | `GridLevel` dataclass list | `state['grid_levels']` dict |
| Statistics | `GridStats` dataclass | `state['stats']` dict |
| Signal Format | Dict with grid fields | Signal dataclass + metadata |
| Position Sizing | Asset-based | USD-based |
| Multi-Symbol | Via wrapper secondary_grid | Native in generate_signal loop |
| RSI Mode | 'confidence_only' or 'filter' | 'confidence_only' default |
| Recentering | Cycle + time + price-based | Cycle + time-based |
| Stop Loss | Strategy-level | Signal-level stop_loss field |
| Execution | GridEnsembleOrchestrator | ws_tester PaperExecutor |

---

**Document Version:** 1.0
**Created:** 2025-12-14
**Implemented:** 2025-12-14
**Status:** IMPLEMENTED - Strategy Ready for Testing
**Next Steps:** Paper trading validation (24-48 hours)
