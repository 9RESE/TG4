# Market Regime Detector: Strategy Integration Guide

## Overview

This document explains how strategies consume regime data and adapt their behavior based on market conditions.

---

## 1. Accessing Regime Data

### 1.1 Via DataSnapshot

The `RegimeSnapshot` is attached to the `DataSnapshot` that strategies receive:

```python
def generate_signal(
    data: DataSnapshot,
    config: dict,
    state: dict
) -> Optional[Signal]:
    # Access regime data
    regime = data.regime

    if regime is None:
        # Fallback: regime detection not available
        # Use conservative defaults
        pass
    else:
        # Use regime data
        current_regime = regime.overall_regime
        confidence = regime.overall_confidence
        is_trending = regime.is_trending
```

### 1.2 RegimeSnapshot Properties

```python
# Core regime classification
regime.overall_regime        # MarketRegime enum: STRONG_BULL, BULL, SIDEWAYS, BEAR, STRONG_BEAR
regime.overall_confidence    # float 0.0-1.0
regime.is_trending           # bool
regime.trend_direction       # str: "UP", "DOWN", "NONE"

# Volatility context
regime.volatility_state      # VolatilityState enum: LOW, MEDIUM, HIGH, EXTREME

# Per-symbol breakdown
regime.symbol_regimes        # Dict[str, SymbolRegime]
symbol_regime = regime.symbol_regimes['XRP/USDT']
symbol_regime.regime         # This symbol's regime
symbol_regime.confidence     # This symbol's confidence
symbol_regime.adx            # Raw ADX value
symbol_regime.choppiness     # Raw Choppiness Index

# External sentiment
regime.external_sentiment    # Optional[ExternalSentiment]
if regime.external_sentiment:
    fear_greed = regime.external_sentiment.fear_greed_value  # 0-100
    btc_dom = regime.external_sentiment.btc_dominance        # percentage

# Multi-timeframe confluence
regime.mtf_confluence        # Optional[MTFConfluence]
if regime.mtf_confluence:
    aligned = regime.mtf_confluence.timeframes_aligned
    total = regime.mtf_confluence.total_timeframes

# Stability metrics
regime.regime_age_seconds    # How long in current regime
regime.recent_transitions    # Regime changes in last hour

# Helper methods
regime.is_favorable_for_trend_strategy()      # bool
regime.is_favorable_for_mean_reversion()      # bool
```

---

## 2. Strategy Adaptation Patterns

### 2.1 Gate Pattern (Enable/Disable)

The simplest adaptation: only trade when conditions are favorable.

```python
def generate_signal(data, config, state):
    regime = data.regime

    # Gate: Don't trade in unfavorable conditions
    if regime and not regime.is_favorable_for_mean_reversion():
        return None

    # Continue with normal signal generation...
```

### 2.2 Position Sizing Pattern

Adjust position sizes based on regime confidence and volatility.

```python
def generate_signal(data, config, state):
    regime = data.regime
    base_size = config['position_size_usd']

    # Adjust size based on regime
    if regime:
        # Reduce size in uncertain conditions
        size_multiplier = regime.overall_confidence

        # Further reduce in high volatility
        if regime.volatility_state == VolatilityState.HIGH:
            size_multiplier *= 0.7
        elif regime.volatility_state == VolatilityState.EXTREME:
            size_multiplier *= 0.3

        adjusted_size = base_size * size_multiplier
    else:
        adjusted_size = base_size * 0.5  # Conservative default

    # Use adjusted_size in signal
```

### 2.3 Threshold Adjustment Pattern

Require stronger signals in uncertain conditions.

```python
def generate_signal(data, config, state):
    regime = data.regime

    # Base thresholds
    rsi_oversold = config['rsi_oversold']  # e.g., 35
    rsi_overbought = config['rsi_overbought']  # e.g., 65

    # Tighten thresholds when trending (mean reversion less reliable)
    if regime and regime.is_trending:
        rsi_oversold -= 5      # Require RSI < 30 instead of < 35
        rsi_overbought += 5    # Require RSI > 70 instead of > 65

    # Use adjusted thresholds
    current_rsi = calculate_rsi(data.candles_1m[symbol])
    if current_rsi < rsi_oversold:
        # Oversold signal...
```

### 2.4 Stop-Loss/Take-Profit Adjustment

Widen stops in volatile conditions, tighten in calm conditions.

```python
def generate_signal(data, config, state):
    regime = data.regime

    base_stop_pct = config['stop_loss_pct']
    base_tp_pct = config['take_profit_pct']

    if regime:
        vol_state = regime.volatility_state

        if vol_state == VolatilityState.LOW:
            stop_pct = base_stop_pct * 0.8   # Tighter stop
            tp_pct = base_tp_pct * 0.8       # Quicker profit
        elif vol_state == VolatilityState.HIGH:
            stop_pct = base_stop_pct * 1.5   # Wider stop
            tp_pct = base_tp_pct * 1.5       # Larger target
        elif vol_state == VolatilityState.EXTREME:
            stop_pct = base_stop_pct * 2.0
            tp_pct = base_tp_pct * 2.0
        else:
            stop_pct = base_stop_pct
            tp_pct = base_tp_pct
    else:
        stop_pct = base_stop_pct
        tp_pct = base_tp_pct

    # Apply to signal
    return Signal(
        ...,
        stop_loss=entry_price * (1 - stop_pct / 100),
        take_profit=entry_price * (1 + tp_pct / 100)
    )
```

### 2.5 Direction Bias Pattern

Use regime to filter signal direction.

```python
def generate_signal(data, config, state):
    regime = data.regime

    # Calculate raw signals
    buy_signal = check_buy_conditions(data, config)
    sell_signal = check_sell_conditions(data, config)

    if regime:
        # In strong bull: only take buys
        if regime.overall_regime == MarketRegime.STRONG_BULL:
            sell_signal = False

        # In strong bear: only take sells
        elif regime.overall_regime == MarketRegime.STRONG_BEAR:
            buy_signal = False

        # In sideways: take both
        # (default behavior, no filter)

    if buy_signal:
        return Signal(action='buy', ...)
    elif sell_signal:
        return Signal(action='sell', ...)

    return None
```

### 2.6 Cooldown Adjustment Pattern

Trade less frequently in uncertain conditions.

```python
def generate_signal(data, config, state):
    regime = data.regime
    base_cooldown = config['cooldown_seconds']

    # Increase cooldown when uncertain
    if regime:
        if regime.overall_confidence < 0.5:
            cooldown = base_cooldown * 2
        elif regime.recent_transitions > 5:  # Unstable regime
            cooldown = base_cooldown * 1.5
        else:
            cooldown = base_cooldown
    else:
        cooldown = base_cooldown * 2

    # Check cooldown
    last_trade = state.get('last_signal_time')
    if last_trade:
        elapsed = (data.timestamp - last_trade).total_seconds()
        if elapsed < cooldown:
            return None

    # Continue signal generation...
```

---

## 3. Strategy-Specific Recommendations

### 3.1 Mean Reversion Strategy

**Favorable Regimes:** SIDEWAYS, low-to-medium volatility
**Unfavorable Regimes:** STRONG_BULL, STRONG_BEAR, EXTREME volatility

```python
# Mean Reversion specific adaptations
def generate_signal(data, config, state):
    regime = data.regime

    # Hard gate: Don't mean-revert in strong trends
    if regime:
        if regime.overall_regime in (MarketRegime.STRONG_BULL, MarketRegime.STRONG_BEAR):
            return None

        # Soft gate: Reduce activity in weak trends
        if regime.is_trending and regime.overall_confidence > 0.6:
            # Only take very strong mean-reversion signals
            config = {**config, 'deviation_threshold': config['deviation_threshold'] * 1.5}

        # Pause in extreme volatility (whipsaw risk)
        if regime.volatility_state == VolatilityState.EXTREME:
            return None

    # Normal signal generation with adjusted config...
```

### 3.2 Momentum/Trend Following Strategy

**Favorable Regimes:** BULL, STRONG_BULL, BEAR, STRONG_BEAR (trending)
**Unfavorable Regimes:** SIDEWAYS

```python
def generate_signal(data, config, state):
    regime = data.regime

    # Gate: Only trade in trending conditions
    if regime:
        if not regime.is_trending:
            return None

        if regime.overall_confidence < 0.4:
            return None  # Not confident enough in trend

        # Direction filter
        if regime.trend_direction == "UP":
            # Only take long signals
            pass
        elif regime.trend_direction == "DOWN":
            # Only take short signals
            pass

    # Continue with momentum logic...
```

### 3.3 Ratio Trading Strategy

**Special Considerations:**
- Monitor correlation between assets
- Consider BTC dominance for XRP/BTC pair
- Watch for decoupling events

```python
def generate_signal(data, config, state):
    regime = data.regime

    # Access XRP/BTC specific regime
    xrp_btc_regime = regime.symbol_regimes.get('XRP/BTC') if regime else None

    # Check external sentiment for BTC dominance context
    if regime and regime.external_sentiment:
        btc_dom = regime.external_sentiment.btc_dominance

        # High BTC dominance = altcoins underperforming
        if btc_dom > 60:
            # Be cautious with XRP/BTC longs
            # BTC likely outperforming, XRP/BTC ratio may fall
            pass

        # Low BTC dominance = altcoin season
        if btc_dom < 40:
            # More confident in XRP/BTC longs
            pass

    # Check correlation stability
    if xrp_btc_regime:
        # If XRP/BTC is trending independently, correlation may be breaking
        if xrp_btc_regime.regime in (MarketRegime.STRONG_BULL, MarketRegime.STRONG_BEAR):
            # Caution: correlation breakdown risk
            pass

    # Continue with ratio trading logic...
```

### 3.4 Whale Sentiment Strategy

**Special Considerations:**
- Volume spikes mean different things in different regimes
- Fear & Greed context for whale behavior interpretation

```python
def generate_signal(data, config, state):
    regime = data.regime

    # Get Fear & Greed for sentiment context
    fg_value = 50  # Neutral default
    if regime and regime.external_sentiment:
        fg_value = regime.external_sentiment.fear_greed_value

    # Interpret whale activity differently based on sentiment
    if fg_value < 25:  # Extreme Fear
        # Large buys in fear = smart money accumulation (bullish)
        # Large sells in fear = panic (may continue)
        pass
    elif fg_value > 75:  # Extreme Greed
        # Large sells in greed = smart money distribution (bearish)
        # Large buys in greed = FOMO (may continue short-term)
        pass

    # Adjust contrarian behavior based on regime
    if regime:
        if regime.overall_regime == MarketRegime.STRONG_BULL:
            # Don't fight strong trends, reduce contrarian trades
            pass
        elif regime.overall_regime == MarketRegime.SIDEWAYS:
            # Contrarian works well in ranges
            pass

    # Continue with whale sentiment logic...
```

### 3.5 Grid Trading Strategy

**Favorable Regimes:** SIDEWAYS, LOW volatility
**Unfavorable Regimes:** Strong trends (grid gets caught one-sided)

```python
def generate_signal(data, config, state):
    regime = data.regime

    if regime:
        # Perfect conditions for grid: sideways + low volatility
        if (regime.overall_regime == MarketRegime.SIDEWAYS and
            regime.volatility_state == VolatilityState.LOW):
            # Tighten grid spacing, more aggressive
            config = {**config, 'grid_spacing_pct': config['grid_spacing_pct'] * 0.8}

        # Unfavorable: strong trend (grid accumulates losing side)
        if regime.overall_regime in (MarketRegime.STRONG_BULL, MarketRegime.STRONG_BEAR):
            # Pause grid or widen spacing significantly
            return None

        # High volatility: widen grid to avoid rapid fills
        if regime.volatility_state in (VolatilityState.HIGH, VolatilityState.EXTREME):
            config = {**config, 'grid_spacing_pct': config['grid_spacing_pct'] * 1.5}

    # Continue with grid logic...
```

---

## 4. Parameter Mapping Tables

### 4.1 Mean Reversion Parameter Map

| Regime | Position Size | Stop Loss | Take Profit | Entry Threshold | Enabled |
|--------|--------------|-----------|-------------|-----------------|---------|
| STRONG_BULL | 0.3x | 2.0x | 1.5x | 2.0x | No |
| BULL | 0.6x | 1.5x | 1.2x | 1.5x | Yes |
| SIDEWAYS | 1.0x | 1.0x | 1.0x | 1.0x | Yes |
| BEAR | 0.6x | 1.5x | 1.2x | 1.5x | Yes |
| STRONG_BEAR | 0.3x | 2.0x | 1.5x | 2.0x | No |

### 4.2 Momentum Strategy Parameter Map

| Regime | Position Size | Stop Loss | Take Profit | Min Confidence | Enabled |
|--------|--------------|-----------|-------------|----------------|---------|
| STRONG_BULL | 1.2x | 1.0x | 1.5x | 0.4 | Yes |
| BULL | 1.0x | 1.0x | 1.2x | 0.5 | Yes |
| SIDEWAYS | 0.3x | 0.8x | 0.8x | 0.8 | No |
| BEAR | 1.0x | 1.0x | 1.2x | 0.5 | Yes |
| STRONG_BEAR | 1.2x | 1.0x | 1.5x | 0.4 | Yes |

### 4.3 Volatility Modifiers (Applied on top)

| Volatility | Position Size | Stop Loss | Take Profit |
|------------|--------------|-----------|-------------|
| LOW | 1.2x | 0.8x | 0.8x |
| MEDIUM | 1.0x | 1.0x | 1.0x |
| HIGH | 0.7x | 1.5x | 1.5x |
| EXTREME | 0.3x | 2.0x | 2.0x |

---

## 5. Implementation Example

### Complete Strategy with Regime Integration

```python
"""Example: Regime-Aware Mean Reversion Strategy"""

from ws_tester.regime.types import MarketRegime, VolatilityState

STRATEGY_NAME = "regime_aware_mean_reversion"
STRATEGY_VERSION = "1.0.0"
SYMBOLS = ["XRP/USDT", "BTC/USDT"]

CONFIG = {
    'position_size_usd': 20.0,
    'deviation_threshold': 2.0,
    'rsi_oversold': 30,
    'rsi_overbought': 70,
    'stop_loss_pct': 1.0,
    'take_profit_pct': 0.5,
    'cooldown_seconds': 30,
}

# Regime-based adjustments
REGIME_CONFIG = {
    MarketRegime.STRONG_BULL: {
        'enabled': False,
    },
    MarketRegime.BULL: {
        'position_size_mult': 0.6,
        'deviation_threshold_mult': 1.5,
    },
    MarketRegime.SIDEWAYS: {
        'position_size_mult': 1.0,
    },
    MarketRegime.BEAR: {
        'position_size_mult': 0.6,
        'deviation_threshold_mult': 1.5,
    },
    MarketRegime.STRONG_BEAR: {
        'enabled': False,
    },
}

VOLATILITY_CONFIG = {
    VolatilityState.LOW: {'position_mult': 1.2, 'stop_mult': 0.8},
    VolatilityState.MEDIUM: {'position_mult': 1.0, 'stop_mult': 1.0},
    VolatilityState.HIGH: {'position_mult': 0.7, 'stop_mult': 1.5},
    VolatilityState.EXTREME: {'position_mult': 0.0, 'stop_mult': 2.0},  # 0 = disabled
}


def generate_signal(data, config, state):
    """Generate regime-aware mean reversion signal."""
    regime = data.regime

    # Apply regime adjustments
    adjusted_config = apply_regime_adjustments(config, regime)

    # Check if strategy is enabled for current regime
    if adjusted_config.get('_disabled', False):
        return None

    # Check cooldown
    if not check_cooldown(data, state, adjusted_config['cooldown_seconds']):
        return None

    # Generate signals for each symbol
    for symbol in SYMBOLS:
        signal = check_symbol_signal(data, symbol, adjusted_config, regime)
        if signal:
            return signal

    return None


def apply_regime_adjustments(config, regime):
    """Apply regime-based config adjustments."""
    adjusted = config.copy()

    if regime is None:
        # Conservative defaults when no regime data
        adjusted['position_size_usd'] *= 0.5
        return adjusted

    # Get regime-specific adjustments
    regime_adj = REGIME_CONFIG.get(regime.overall_regime, {})

    # Check if disabled for this regime
    if not regime_adj.get('enabled', True):
        adjusted['_disabled'] = True
        return adjusted

    # Apply multipliers
    if 'position_size_mult' in regime_adj:
        adjusted['position_size_usd'] *= regime_adj['position_size_mult']

    if 'deviation_threshold_mult' in regime_adj:
        adjusted['deviation_threshold'] *= regime_adj['deviation_threshold_mult']

    # Apply volatility adjustments
    vol_adj = VOLATILITY_CONFIG.get(regime.volatility_state, {})

    if vol_adj.get('position_mult', 1.0) == 0:
        adjusted['_disabled'] = True
        return adjusted

    adjusted['position_size_usd'] *= vol_adj.get('position_mult', 1.0)
    adjusted['stop_loss_pct'] *= vol_adj.get('stop_mult', 1.0)
    adjusted['take_profit_pct'] *= vol_adj.get('stop_mult', 1.0)

    # Reduce size based on confidence
    adjusted['position_size_usd'] *= regime.overall_confidence

    return adjusted


def check_symbol_signal(data, symbol, config, regime):
    """Check for mean reversion signal on a symbol."""
    candles = data.candles_1m.get(symbol)
    if not candles or len(candles) < 50:
        return None

    # Calculate indicators
    closes = [c.close for c in candles]
    current_price = closes[-1]

    sma_20 = sum(closes[-20:]) / 20
    std_20 = (sum((c - sma_20)**2 for c in closes[-20:]) / 20) ** 0.5
    z_score = (current_price - sma_20) / std_20 if std_20 > 0 else 0

    rsi = calculate_rsi(closes)

    # Check oversold condition
    if z_score < -config['deviation_threshold'] and rsi < config['rsi_oversold']:
        return Signal(
            action='buy',
            symbol=symbol,
            size=config['position_size_usd'],
            price=current_price,
            reason=f"Mean reversion buy: z={z_score:.2f}, RSI={rsi:.1f}",
            stop_loss=current_price * (1 - config['stop_loss_pct'] / 100),
            take_profit=current_price * (1 + config['take_profit_pct'] / 100),
            metadata={
                'regime': regime.overall_regime.name if regime else 'UNKNOWN',
                'confidence': regime.overall_confidence if regime else 0.0,
            }
        )

    # Check overbought condition
    if z_score > config['deviation_threshold'] and rsi > config['rsi_overbought']:
        return Signal(
            action='sell',
            symbol=symbol,
            size=config['position_size_usd'],
            price=current_price,
            reason=f"Mean reversion sell: z={z_score:.2f}, RSI={rsi:.1f}",
            stop_loss=current_price * (1 + config['stop_loss_pct'] / 100),
            take_profit=current_price * (1 - config['take_profit_pct'] / 100),
            metadata={
                'regime': regime.overall_regime.name if regime else 'UNKNOWN',
                'confidence': regime.overall_confidence if regime else 0.0,
            }
        )

    return None
```

---

## 6. Testing Regime Integration

### 6.1 Unit Test Example

```python
def test_strategy_disabled_in_strong_trend():
    """Verify strategy disables in STRONG_BULL regime."""
    # Create mock regime
    regime = RegimeSnapshot(
        overall_regime=MarketRegime.STRONG_BULL,
        overall_confidence=0.8,
        is_trending=True,
        # ... other fields
    )

    # Create mock data with regime
    data = DataSnapshot(
        # ... price data that would normally trigger signal
        regime=regime
    )

    # Strategy should return None (disabled)
    signal = generate_signal(data, CONFIG, {})
    assert signal is None


def test_position_size_reduced_in_high_volatility():
    """Verify position size reduction in HIGH volatility."""
    regime = RegimeSnapshot(
        overall_regime=MarketRegime.SIDEWAYS,
        overall_confidence=0.7,
        volatility_state=VolatilityState.HIGH,
        # ...
    )

    adjusted = apply_regime_adjustments(CONFIG, regime)

    # Should be reduced by 0.7x (HIGH vol) * 0.7 (confidence)
    expected = CONFIG['position_size_usd'] * 0.7 * 0.7
    assert abs(adjusted['position_size_usd'] - expected) < 0.01
```

---

*Version: 1.0.0 | Created: 2025-12-15*
