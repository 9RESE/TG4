# Market Regime Detection System v1.0.0

## Summary

Implementation of a comprehensive Bull/Bear/Sideways Market Regime Detection System that analyzes market conditions and provides actionable regime classifications that strategies can use to optimize their parameters.

## Problem Statement

Trading strategies operate with static parameters regardless of market conditions. A strategy optimized for trending markets performs poorly in sideways markets, while mean-reversion strategies fail during strong trends. This system addresses the problem by:

1. **Detecting market regimes** - Classify current conditions as trending (bull/bear) or ranging (sideways)
2. **Adapting strategy parameters** - Automatically adjust position sizes, stop-losses, and entry thresholds
3. **Reducing drawdowns** - Avoid trading in unfavorable conditions
4. **Improving win rates** - Trade only when conditions favor the strategy type

## Features Implemented

### Phase 1: Core Indicators

#### 1.1 Choppiness Index
- **Location**: `ws_tester/indicators/choppiness.py`
- **Function**: `calculate_choppiness(data, period=14)`
- **Purpose**: Identifies consolidating/ranging markets vs trending markets
- **Output**: Value 0-100 where >61.8 = choppy/sideways, <38.2 = trending
- **Tests**: `tests/test_regime.py::TestChoppinessIndicator`

#### 1.2 Enhanced ADX with +DI/-DI
- **Location**: `ws_tester/indicators/oscillators.py`
- **Function**: `calculate_adx_with_di(data, period=14)`
- **Purpose**: Returns full ADX analysis including directional indicators
- **Output**: `ADXResult(adx, plus_di, minus_di, trend_strength)`
- **Tests**: `tests/test_regime.py::TestADXWithDI`

#### 1.3 MA Alignment Helper
- **Location**: `ws_tester/indicators/moving_averages.py`
- **Function**: `calculate_ma_alignment(price, sma_20, sma_50, sma_200)`
- **Purpose**: Analyze moving average alignment for trend classification
- **Output**: `MAAlignmentResult(score, alignment, ...)`
- **Tests**: `tests/test_regime.py::TestMAAlignment`

### Phase 2: Regime Detection Module

New module created at `ws_tester/regime/` with the following components:

#### 2.1 Types (`types.py`)
- `MarketRegime` enum: STRONG_BULL, BULL, SIDEWAYS, BEAR, STRONG_BEAR
- `VolatilityState` enum: LOW, MEDIUM, HIGH, EXTREME
- `TrendStrength` enum: ABSENT, WEAK, EMERGING, STRONG, VERY_STRONG
- `IndicatorScores`: Individual indicator contribution scores
- `SymbolRegime`: Per-symbol regime classification
- `MTFConfluence`: Multi-timeframe alignment data
- `ExternalSentiment`: Fear & Greed, BTC Dominance data
- `RegimeSnapshot`: Complete market regime state
- `RegimeAdjustments`: Strategy parameter multipliers

#### 2.2 Composite Scorer (`composite_scorer.py`)
- Weighted combination of indicators (ADX, Choppiness, MA, RSI, Volume, Sentiment)
- Default weights: ADX 25%, Chop 20%, MA 20%, RSI 15%, Volume 10%, Sentiment 10%
- Score smoothing to prevent whipsaw
- **Tests**: `tests/test_regime.py::TestCompositeScorer`

#### 2.3 External Data Fetcher (`external_data.py`)
- Fetches Fear & Greed Index from Alternative.me
- Fetches BTC Dominance from CoinGecko
- 5-minute caching with graceful fallback
- **Tests**: `tests/test_regime.py::TestExternalDataFetcher`

#### 2.4 MTF Analyzer (`mtf_analyzer.py`)
- Analyzes regime confluence across 1m, 5m, 15m, 1h timeframes
- Weighted alignment scoring (higher timeframes = more weight)
- **Tests**: `tests/test_regime.py::TestMTFAnalyzer`

#### 2.5 Parameter Router (`parameter_router.py`)
- Routes regime-based adjustments to strategies
- Pre-configured adjustments for mean_reversion, momentum_scalping, grid_trading, whale_sentiment
- Volatility modifiers applied automatically
- **Tests**: `tests/test_regime.py::TestParameterRouter`

#### 2.6 Main Detector (`detector.py`)
- Main orchestrator class `RegimeDetector`
- Hysteresis to prevent rapid regime switching
- Transition tracking for stability analysis
- **Tests**: `tests/test_regime.py::TestRegimeDetector`

### Phase 3: Integration

#### 3.1 DataSnapshot Extension
- Added optional `regime: Optional[RegimeSnapshot]` field to `DataSnapshot`
- Backward compatible - field defaults to None

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MARKET REGIME DETECTION SYSTEM                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐    │
│  │   Kraken WS      │     │  External APIs   │     │  Historical      │    │
│  │   Price Data     │     │  (F&G, BTC.D)    │     │  Data Store      │    │
│  └────────┬─────────┘     └────────┬─────────┘     └────────┬─────────┘    │
│           ↓                        ↓                        ↓              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      INDICATOR CALCULATION LAYER                     │   │
│  │  ADX+DI  |  CHOP  |  MA Alignment  |  RSI  |  Volume  |  ATR         │   │
│  └────────────────────────────────────┬────────────────────────────────┘   │
│                                       ↓                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     REGIME CLASSIFICATION LAYER                      │   │
│  │  Per-Symbol Classifier  |  MTF Confluence  |  Composite Score Engine │   │
│  └────────────────────────────────────┬────────────────────────────────┘   │
│                                       ↓                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        REGIME OUTPUT LAYER                           │   │
│  │  RegimeSnapshot: overall_regime, confidence, is_trending, volatility │   │
│  └────────────────────────────────────┬────────────────────────────────┘   │
│                                       ↓                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    STRATEGY PARAMETER ROUTER                         │   │
│  │  Position sizing  |  Stop-loss scaling  |  Entry threshold shifts   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Usage

### Basic Usage

```python
from ws_tester.regime import RegimeDetector, RegimeSnapshot

# Initialize detector
detector = RegimeDetector(symbols=['XRP/USDT', 'BTC/USDT'])

# On each tick
regime = await detector.detect(data_snapshot)

# Check regime conditions
if regime.is_favorable_for_mean_reversion():
    # Safe to use mean reversion strategy
    pass

if regime.should_reduce_exposure():
    # Reduce position sizes
    pass
```

### Strategy Integration

```python
def generate_signal(data, config, state):
    regime = data.regime
    if regime is None:
        return None

    # Get adjusted config for current regime
    if not detector.should_trade('mean_reversion', regime):
        return None

    adjustments = detector.get_strategy_adjustments('mean_reversion', regime)
    adjusted_config = detector.apply_adjustments(config, adjustments)

    # Use adjusted_config for signal generation
    size = adjusted_config['position_size_usd']
    # ...
```

### Configuration

```python
config = {
    'weights': {
        'adx': 0.25,
        'chop': 0.20,
        'ma': 0.20,
        'rsi': 0.15,
        'volume': 0.10,
        'sentiment': 0.10,
    },
    'thresholds': {
        'strong_bull': 0.4,
        'bull': 0.15,
        'bear': -0.15,
        'strong_bear': -0.4,
    },
    'smoothing_period': 3,
    'min_regime_duration': 60,  # seconds
    'confirmation_bars': 3,
    'external_enabled': True,
}

detector = RegimeDetector(symbols, config)
```

## Testing

Run all regime tests:
```bash
python -m pytest tests/test_regime.py -v
```

Test results: **39 tests passing**

## Files Changed

### New Files
- `ws_tester/indicators/choppiness.py` - Choppiness Index indicator
- `ws_tester/regime/__init__.py` - Module exports
- `ws_tester/regime/types.py` - Type definitions
- `ws_tester/regime/composite_scorer.py` - Composite scoring
- `ws_tester/regime/external_data.py` - External API fetcher
- `ws_tester/regime/mtf_analyzer.py` - Multi-timeframe analyzer
- `ws_tester/regime/parameter_router.py` - Parameter routing
- `ws_tester/regime/detector.py` - Main detector class
- `tests/test_regime.py` - Comprehensive test suite

### Modified Files
- `ws_tester/indicators/__init__.py` - Added new exports
- `ws_tester/indicators/_types.py` - Added ADXResult, MAAlignmentResult
- `ws_tester/indicators/oscillators.py` - Added calculate_adx_with_di
- `ws_tester/indicators/moving_averages.py` - Added MA alignment functions
- `ws_tester/types.py` - Added regime field to DataSnapshot

## Regime Classifications

| Regime | Score Range | Characteristics |
|--------|-------------|-----------------|
| STRONG_BULL | > 0.4 | Strong uptrend, high confidence |
| BULL | 0.15 to 0.4 | Uptrend |
| SIDEWAYS | -0.15 to 0.15 | Range-bound, no clear direction |
| BEAR | -0.4 to -0.15 | Downtrend |
| STRONG_BEAR | < -0.4 | Strong downtrend, high confidence |

## Strategy Adaptation Guidelines

| Strategy Type | Favorable Regimes | Unfavorable Regimes |
|--------------|-------------------|---------------------|
| Mean Reversion | SIDEWAYS | STRONG_BULL, STRONG_BEAR |
| Momentum/Trend | BULL, BEAR, STRONG_* | SIDEWAYS |
| Grid Trading | SIDEWAYS (low vol) | Strong trends |
| Whale Sentiment | All (with adjustments) | Extreme volatility |

## Future Enhancements

1. **Phase 4**: Strategy-specific implementations (update existing strategies)
2. **Phase 5**: Backtesting with regime data
3. **Phase 6**: Dashboard integration with regime visualization
4. **Optional**: Machine learning regime classification (HMM)

## References

- [Implementation Plan](../plans/b-b-detector/implementation-plan.md)
- [Architecture Design](../plans/b-b-detector/architecture-design.md)
- [Research Findings](../plans/b-b-detector/research-findings.md)
- [Strategy Integration Guide](../plans/b-b-detector/strategy-integration.md)

---

*Version: 1.0.0 | Created: 2025-12-15*
