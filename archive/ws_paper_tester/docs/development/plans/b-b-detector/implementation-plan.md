# Market Regime Detector: Implementation Plan

## Overview

This document outlines the step-by-step implementation plan for the Bull/Bear/Sideways Market Regime Detection System.

## Phase 1: Core Indicator Extensions

### 1.1 Add Choppiness Index Indicator

**File:** `ws_tester/indicators/choppiness.py`

**Tasks:**
- [ ] Implement `calculate_choppiness(highs, lows, closes, period=14)` function
- [ ] Add proper input validation and edge case handling
- [ ] Return float between 0-100
- [ ] Add comprehensive unit tests

**Algorithm:**
```python
def calculate_choppiness(highs, lows, closes, period=14):
    """
    Calculate Choppiness Index.

    Returns:
        float: 0-100 value where >61.8 = choppy/sideways, <38.2 = trending
    """
    # Sum of ATR over period
    atr_sum = sum(calculate_atr_single(h, l, c, prev_c)
                  for h, l, c, prev_c in zip(...))

    # High-Low range over period
    period_high = max(highs[-period:])
    period_low = min(lows[-period:])
    range_hl = period_high - period_low

    if range_hl == 0:
        return 50.0  # Neutral default

    chop = 100 * math.log10(atr_sum / range_hl) / math.log10(period)
    return max(0.0, min(100.0, chop))
```

**Tests Required:**
- Test with trending data (should return < 38.2)
- Test with choppy/ranging data (should return > 61.8)
- Test edge cases (zero range, insufficient data)
- Test with real historical data

### 1.2 Enhance ADX Implementation

**File:** `ws_tester/indicators/oscillators.py`

**Tasks:**
- [ ] Verify current ADX implementation returns +DI and -DI values
- [ ] Add `calculate_adx_with_di()` function if needed
- [ ] Ensure returns `{'adx': float, 'plus_di': float, 'minus_di': float}`
- [ ] Add comprehensive tests

### 1.3 Add MA Alignment Helper

**File:** `ws_tester/indicators/moving_averages.py`

**Tasks:**
- [ ] Add `calculate_ma_alignment(price, sma_20, sma_50, sma_200)` function
- [ ] Return alignment score and classification
- [ ] Add tests

```python
def calculate_ma_alignment(price, sma_20, sma_50, sma_200):
    """
    Calculate moving average alignment score.

    Returns:
        dict: {'score': float (-1 to 1), 'alignment': str}
    """
    if price > sma_20 > sma_50 > sma_200:
        return {'score': 1.0, 'alignment': 'PERFECT_BULL'}
    elif price < sma_20 < sma_50 < sma_200:
        return {'score': -1.0, 'alignment': 'PERFECT_BEAR'}
    elif sma_50 > sma_200:
        score = 0.5 if price > sma_50 else 0.25
        return {'score': score, 'alignment': 'BULL'}
    elif sma_50 < sma_200:
        score = -0.5 if price < sma_50 else -0.25
        return {'score': score, 'alignment': 'BEAR'}
    else:
        return {'score': 0.0, 'alignment': 'NEUTRAL'}
```

### 1.4 Update Indicators Index

**File:** `ws_tester/indicators/__init__.py`

**Tasks:**
- [ ] Export new functions
- [ ] Update docstrings

---

## Phase 2: Regime Detection Module

### 2.1 Create Module Structure

**Directory:** `ws_tester/regime/`

**Tasks:**
- [ ] Create directory structure:
  ```
  ws_tester/regime/
  ├── __init__.py
  ├── types.py
  ├── detector.py
  ├── composite_scorer.py
  ├── mtf_analyzer.py
  ├── parameter_router.py
  └── external_data.py
  ```
- [ ] Create `__init__.py` with public exports

### 2.2 Implement Types (`types.py`)

**Tasks:**
- [ ] Define `MarketRegime` enum
- [ ] Define `VolatilityState` enum
- [ ] Define `TrendStrength` enum
- [ ] Define `IndicatorScores` frozen dataclass
- [ ] Define `SymbolRegime` frozen dataclass
- [ ] Define `MTFConfluence` frozen dataclass
- [ ] Define `ExternalSentiment` frozen dataclass
- [ ] Define `RegimeSnapshot` frozen dataclass
- [ ] Define `RegimeAdjustments` dataclass
- [ ] Add type validation tests

### 2.3 Implement Composite Scorer (`composite_scorer.py`)

**Tasks:**
- [ ] Implement `CompositeScorer` class
- [ ] Add `_score_adx()` method
- [ ] Add `_score_chop()` method
- [ ] Add `_score_ma()` method
- [ ] Add `_score_rsi()` method
- [ ] Add `_score_volume()` method
- [ ] Add `_score_sentiment()` method
- [ ] Add `calculate_symbol_score()` method
- [ ] Add score smoothing via moving average
- [ ] Add comprehensive tests

**Test Cases:**
- Perfect bull market indicators → score > 0.4
- Perfect bear market indicators → score < -0.4
- Mixed/neutral indicators → score near 0
- Verify individual component scoring
- Verify weight application

### 2.4 Implement External Data Fetcher (`external_data.py`)

**Tasks:**
- [ ] Implement `ExternalDataFetcher` class
- [ ] Add Fear & Greed API integration
- [ ] Add BTC Dominance API integration (CoinGecko)
- [ ] Implement 5-minute caching
- [ ] Add error handling and fallbacks
- [ ] Add async timeout handling
- [ ] Add tests with mocked responses

**API Integration:**
```python
# Fear & Greed: https://api.alternative.me/fng/
# CoinGecko: https://api.coingecko.com/api/v3/global
```

### 2.5 Implement MTF Analyzer (`mtf_analyzer.py`)

**Tasks:**
- [ ] Implement `MTFAnalyzer` class
- [ ] Add `_classify_timeframe()` helper
- [ ] Add `analyze()` main method
- [ ] Add weighted confluence calculation
- [ ] Add tests

**Note:** Currently DataSnapshot only has 1m and 5m candles. For full MTF:
- Option A: Extend DataManager to build 15m, 1h, 4h candles
- Option B: Use only available timeframes (1m, 5m)
- **Recommendation:** Start with Option B, extend later

### 2.6 Implement Parameter Router (`parameter_router.py`)

**Tasks:**
- [ ] Implement `ParameterRouter` class
- [ ] Define default `REGIME_ADJUSTMENTS` mapping
- [ ] Add `get_adjustments()` method
- [ ] Add `apply_to_config()` method
- [ ] Add volatility-based modifiers
- [ ] Add per-strategy adjustment tables
- [ ] Add tests

### 2.7 Implement Main Detector (`detector.py`)

**Tasks:**
- [ ] Implement `RegimeDetector` class
- [ ] Wire up all sub-components
- [ ] Add `detect()` async method
- [ ] Add regime transition tracking
- [ ] Add regime age calculation
- [ ] Add hysteresis (minimum regime duration)
- [ ] Add logging for regime changes
- [ ] Add comprehensive integration tests

---

## Phase 3: Integration

### 3.1 Extend DataSnapshot

**File:** `ws_tester/types.py`

**Tasks:**
- [ ] Add optional `regime: Optional[RegimeSnapshot]` field
- [ ] Update any affected code
- [ ] Verify backward compatibility

### 3.2 Integrate into Main Loop

**File:** `ws_tester.py` (or main entry point)

**Tasks:**
- [ ] Initialize `RegimeDetector` on startup
- [ ] Call `regime_detector.detect()` each tick
- [ ] Attach regime to DataSnapshot
- [ ] Log regime status periodically

### 3.3 Update Strategy Interface

**File:** Strategy base/interface

**Tasks:**
- [ ] Document regime access pattern
- [ ] Add helper methods for regime-based decisions
- [ ] Update example strategies with regime awareness

### 3.4 Update Configuration

**File:** `config.yaml`

**Tasks:**
- [ ] Add `regime_detection` section
- [ ] Add weight configuration
- [ ] Add threshold configuration
- [ ] Add external data toggles

---

## Phase 4: Strategy Adaptation

### 4.1 Mean Reversion Strategy Updates

**File:** `strategies/mean_reversion/`

**Tasks:**
- [ ] Add regime check at signal generation start
- [ ] Disable/reduce activity in strong trends
- [ ] Adjust position sizing based on regime
- [ ] Add regime-based stop-loss widening
- [ ] Log regime influence on decisions

### 4.2 Ratio Trading Strategy Updates

**File:** `strategies/ratio_trading/`

**Tasks:**
- [ ] Add regime awareness for correlation breakdown detection
- [ ] Adjust entry thresholds based on market regime
- [ ] Consider BTC dominance in decisions

### 4.3 Whale Sentiment Strategy Updates

**File:** `strategies/whale_sentiment/`

**Tasks:**
- [ ] Use regime for context on volume spikes
- [ ] Adjust contrarian behavior based on regime
- [ ] Consider Fear & Greed in whale interpretation

### 4.4 Other Strategies

**Tasks:**
- [ ] Review each strategy for regime integration opportunities
- [ ] Document which regimes favor each strategy
- [ ] Add enable/disable logic based on regime

---

## Phase 5: Testing & Validation

### 5.1 Unit Tests

**File:** `tests/test_regime/`

**Tasks:**
- [ ] Test all type definitions
- [ ] Test composite scorer with various inputs
- [ ] Test MTF analyzer
- [ ] Test parameter router
- [ ] Test external data fetcher (mocked)
- [ ] Test main detector integration

### 5.2 Integration Tests

**Tasks:**
- [ ] Test regime detection with real market data
- [ ] Test strategy behavior changes with different regimes
- [ ] Test regime transition handling
- [ ] Test external API fallback behavior

### 5.3 Backtesting

**Tasks:**
- [ ] Run strategies with and without regime adaptation
- [ ] Compare win rates by regime
- [ ] Validate regime classification accuracy
- [ ] Document performance improvements

---

## Phase 6: Documentation & Monitoring

### 6.1 Documentation

**Tasks:**
- [ ] Add regime detection to strategy development guide
- [ ] Document configuration options
- [ ] Add troubleshooting guide
- [ ] Update CHANGELOG

### 6.2 Dashboard Updates

**File:** `ws_tester/dashboard/`

**Tasks:**
- [ ] Add regime indicator to dashboard
- [ ] Show per-symbol regimes
- [ ] Display external sentiment
- [ ] Add regime transition history

### 6.3 Logging

**Tasks:**
- [ ] Add structured logging for regime changes
- [ ] Log regime influence on signals
- [ ] Add regime metrics to monitoring

---

## Dependency Additions

**File:** `requirements.txt`

```
# Existing
aiohttp>=3.8.0  # For async HTTP (external APIs)

# New (if using HMM later)
hmmlearn>=0.3.0  # Hidden Markov Models (optional, Phase 6+)
scikit-learn>=1.0.0  # ML utilities (optional)
```

---

## Risk Mitigation

### Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| External API downtime | Cache with long TTL, graceful fallback |
| Regime whipsaw | Smoothing, hysteresis, minimum duration |
| Over-optimization | Start simple, validate with out-of-sample data |
| Strategy performance degradation | A/B testing, gradual rollout |
| Computational overhead | Lazy calculation, caching, efficient algorithms |

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Regime detection latency | < 10ms per symbol |
| Memory overhead | < 50MB additional |
| Strategy win rate improvement | > 10% |
| False regime transitions | < 5 per hour |
| External API success rate | > 95% |

---

## Checklist Summary

### Phase 1
- [ ] Choppiness Index implementation
- [ ] ADX enhancement
- [ ] MA alignment helper
- [ ] Indicator tests

### Phase 2
- [ ] Types module
- [ ] Composite scorer
- [ ] External data fetcher
- [ ] MTF analyzer
- [ ] Parameter router
- [ ] Main detector
- [ ] Module tests

### Phase 3
- [ ] DataSnapshot extension
- [ ] Main loop integration
- [ ] Strategy interface update
- [ ] Configuration update

### Phase 4
- [ ] Mean reversion adaptation
- [ ] Ratio trading adaptation
- [ ] Other strategy adaptations

### Phase 5
- [ ] Unit tests
- [ ] Integration tests
- [ ] Backtesting validation

### Phase 6
- [ ] Documentation updates
- [ ] Dashboard updates
- [ ] Logging enhancements

---

*Version: 1.0.0 | Created: 2025-12-15*
