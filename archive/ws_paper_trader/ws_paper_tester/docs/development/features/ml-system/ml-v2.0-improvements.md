# ML Trading System v2.1 Improvements

**Document Version**: 2.1
**Created**: 2025-12-17
**Updated**: 2025-12-17
**Status**: Implemented
**Author**: Claude Code

---

## Overview

This document describes the significant improvements made to the ML trading system in v2.0 and v2.1. These changes address critical issues with metrics calculation, data leakage, class imbalance, and add new feature engineering capabilities including regime detection.

## Summary of Changes

### 1. Sharpe Ratio Calculation Fix

**Problem**: The original Sharpe ratio calculation used `periods_per_year = 365 * 24 * 60 = 525,600` (1-minute bars) when calculating trade returns. This inflated Sharpe ratios by ~74x because trade returns should be annualized based on trades per year, not bars per year.

**Solution**: Updated `ws_paper_tester/ml/evaluation/metrics.py`:
- Changed `calculate_trading_metrics()` to use trade frequency for annualization
- Added `trades_per_year` parameter (auto-calculated from data if not provided)
- Added `trading_days` parameter to specify backtest period
- Updated `calculate_sortino_ratio()` and `calculate_calmar_ratio()` with same fix

**Impact**:
- Before: Sharpe ratio of 519 for 95 trades (unrealistic)
- After: Sharpe ratio correctly annualized based on actual trade frequency

### 2. Data Leakage Prevention

**Problem**: Labels were generated using future price data (`shift(-future_bars)`), but data was split AFTER label generation. This caused the last `future_bars` rows of training data to have labels based on validation prices, creating lookahead bias.

**Solution**: Updated `ws_paper_tester/ml/scripts/retrain_pipeline.py`:
- Added gap buffer equal to `lookahead` bars between splits
- Train set ends at `train_end - gap`
- Validation set: `[train_end, val_end - gap)`
- Test set: `[val_end, n - gap)`

**Impact**:
- Eliminates false signal from validation accuracy being higher than training accuracy
- Provides more realistic model performance estimates

### 3. Order Flow Features from Trades Table

**New File**: `ws_paper_tester/ml/features/order_flow_features.py`

Computes real order flow features from the `trades` table in TimescaleDB:

| Feature | Description |
|---------|-------------|
| `trade_imbalance` | (buy_volume - sell_volume) / total_volume |
| `vpin` | Volume-Synchronized Probability of Informed Trading |
| `order_flow_toxicity` | Combined toxicity score (vpin * abs(imbalance)) |
| `buy_sell_ratio` | buy_volume / sell_volume |
| `trade_intensity` | trades per minute |
| `avg_trade_size` | average trade size |
| `large_trade_ratio` | ratio of large trades (>2x avg size) |

**Usage**:
```python
from ml.features.order_flow_features import OrderFlowFeatureProvider

provider = OrderFlowFeatureProvider(db_url)
await provider.connect()
features = await provider.compute_features('XRP/USDT', lookback_minutes=60)
```

### 4. Multi-Timeframe Features

**New File**: `ws_paper_tester/ml/features/multi_timeframe.py`

Computes features across multiple timeframes (1m, 5m, 15m, 1h, 4h):

| Feature | Description |
|---------|-------------|
| `mtf_trend_alignment` | % of timeframes in same trend direction |
| `tf_divergence_score` | Measure of conflicting trends |
| `multi_resolution_volatility` | Weighted avg volatility across TFs |
| `dominant_trend` | Majority trend direction |
| `momentum_confluence` | Avg RSI deviation from neutral |

### 5. Automatic Retraining Pipeline

**New File**: `ws_paper_tester/ml/scripts/retrain_pipeline.py`

Complete pipeline for automatic model retraining:

```bash
python -m ws_paper_tester.ml.scripts.retrain_pipeline \
    --db-url 'postgresql://...' \
    --symbol 'XRP/USDT' \
    --days 365 \
    --enrich-order-flow \
    --enrich-mtf \
    --auto-deploy
```

Features:
- Loads historical data from TimescaleDB
- Optional order flow and MTF feature enrichment
- Time-ordered train/val/test split with gap buffer
- XGBoost training with early stopping
- Regularization parameters to prevent overfitting
- Model registration with versioning
- Auto-deployment if performance improves
- Results saved to `backtest_runs` table

### 6. ML Signal Strategy

**New Directory**: `ws_paper_tester/strategies/ml_signal/`

Trading strategy that uses ML models for signal generation:

- Loads models from ModelRegistry
- Extracts features from market snapshots
- Generates buy/sell/hold signals based on model predictions
- Includes confidence thresholds and risk management

### 7. Performance Tracker

**New File**: `ws_paper_tester/ml/evaluation/performance_tracker.py`

Tracks ML model performance using the `backtest_runs` table:

- Save performance records after training
- Query historical performance
- Compare model versions
- Generate performance reports

## Configuration Changes

### RetrainingConfig Defaults (v2.0)

```python
@dataclass
class RetrainingConfig:
    days: int = 365  # Was 30 - more data for better generalization
    n_estimators: int = 300
    max_depth: int = 4  # Was 6 - reduced for regularization
    learning_rate: float = 0.05  # Was 0.1 - slower learning
    min_child_weight: int = 10  # Prevents overfitting
    subsample: float = 0.7
    colsample_bytree: float = 0.7
    reg_alpha: float = 0.1  # L1 regularization
    reg_lambda: float = 1.0  # L2 regularization
    early_stopping_rounds: int = 30
    lookahead: int = 10  # Prediction horizon
```

### BacktestConfig Updates

```python
@dataclass
class BacktestConfig:
    # ... existing fields ...
    trading_days: int = 365  # NEW: For correct Sharpe annualization
```

## Files Modified

| File | Changes |
|------|---------|
| `ml/evaluation/metrics.py` | Fixed Sharpe/Sortino/Calmar annualization |
| `ml/evaluation/backtest.py` | Added `trading_days` to BacktestConfig |
| `ml/evaluation/__init__.py` | Added PerformanceTracker exports |
| `ml/models/classifier.py` | Fixed XGBoost early_stopping_rounds API |
| `ml/features/__init__.py` | Added order flow and MTF exports |
| `ml/features/order_flow_features.py` | NEW: Order flow feature provider |
| `ml/features/multi_timeframe.py` | NEW: Multi-timeframe features |
| `ml/scripts/retrain_pipeline.py` | NEW: Automatic retraining pipeline |
| `ml/evaluation/performance_tracker.py` | NEW: Performance tracking |
| `strategies/ml_signal/` | NEW: ML-based trading strategy |

## Known Limitations

1. **Database Schema**: Requires `candles` and `trades` tables in TimescaleDB
2. **GPU Memory**: Large models may require reducing batch size for 12GB VRAM
3. **Order Flow Data**: Requires trade-level data with buy/sell side information

## v2.1 Improvements (2025-12-17)

### 8. Class Weights for HOLD Bias Reduction

**Problem**: The model was heavily biased towards predicting HOLD (68-84% of labels), leading to few actionable signals.

**Solution**: Added `use_class_weights` parameter to `XGBoostClassifier`:
```python
class XGBoostClassifier:
    def __init__(self, ..., use_class_weights: bool = False):
        self.use_class_weights = use_class_weights

    @staticmethod
    def compute_class_weights(y: np.ndarray) -> np.ndarray:
        """Inverse frequency weighting to reduce majority class bias."""
        weights[cls] = total / (num_classes * counts[cls])
```

**Impact**:
- BUY/SELL signals now properly weighted during training
- Reduces over-prediction of HOLD class

### 9. Lower Confidence Threshold

**Problem**: Default confidence threshold of 0.6 was too high, resulting in 0 trades in many periods.

**Solution**: Lowered default `confidence_threshold` from 0.6 to 0.4 in `BacktestConfig`.

**Impact**:
- More trades generated for backtesting
- Better signal coverage while maintaining reasonable precision

### 10. Signal Precision Metrics

**New File**: `ws_paper_tester/ml/evaluation/metrics.py` (extended)

New `SignalPrecisionMetrics` class and `calculate_signal_precision_metrics()` function:

| Metric | Description |
|--------|-------------|
| `buy_precision` | When we predict BUY, how often is it correct? |
| `sell_precision` | When we predict SELL, how often is it correct? |
| `action_precision` | Precision for non-HOLD predictions |
| `false_buy_rate` | Rate of incorrect BUY signals |
| `false_sell_rate` | Rate of incorrect SELL signals |
| `hold_bias` | Fraction of predictions that are HOLD |

Also added `calculate_profit_by_signal()` for P&L breakdown by signal type.

### 11. Regime Detection

**New File**: `ws_paper_tester/ml/features/regime_detection.py`

Identifies market regimes to understand model performance across conditions:

| Regime | Description |
|--------|-------------|
| `TRENDING_UP` | Strong upward trend with momentum |
| `TRENDING_DOWN` | Strong downward trend with momentum |
| `RANGING` | Sideways price action with low volatility |
| `VOLATILE` | High volatility with no clear direction |
| `BREAKOUT` | Transitioning from ranging to trending |

Key features:
- ADX-based trend strength detection
- ATR-based volatility percentile calculation
- Support/resistance level tracking
- One-hot encoded regime features for ML input

**Usage**:
```python
from ml.features.regime_detection import RegimeDetector, analyze_performance_by_regime

detector = RegimeDetector()
state = detector.detect_regime(prices, highs, lows, volumes)
print(f"Current regime: {state.regime.value}, confidence: {state.confidence:.2f}")
```

### 12. Updated RetrainingConfig Defaults (v2.1)

```python
@dataclass
class RetrainingConfig:
    # ... v2.0 fields ...

    # v2.1 improvements
    use_class_weights: bool = True  # Reduce HOLD bias
    confidence_threshold: float = 0.4  # Lowered from 0.6
    enrich_regime: bool = True  # Add regime detection features
```

## Files Modified in v2.1

| File | Changes |
|------|---------|
| `ml/models/classifier.py` | Added `use_class_weights`, `compute_class_weights()`, `compute_sample_weights()` |
| `ml/evaluation/metrics.py` | Added `SignalPrecisionMetrics`, `calculate_signal_precision_metrics()`, `calculate_profit_by_signal()` |
| `ml/evaluation/backtest.py` | Updated default `confidence_threshold` to 0.4 |
| `ml/evaluation/__init__.py` | Exported new precision metric classes and functions |
| `ml/features/regime_detection.py` | NEW: Complete regime detection module |
| `ml/features/__init__.py` | Exported regime detection classes |
| `ml/scripts/retrain_pipeline.py` | Integrated all v2.1 improvements |

## Future Improvements

1. ~~Walk-forward validation for more realistic performance estimates~~ (Implemented in v2.0)
2. Online learning for continuous model updates
3. Multi-asset portfolio optimization
4. Attention-based models for longer-term dependencies
5. Ensemble models combining multiple timeframes

---

**Related Documents**:
- [ML Overview](../plans/ml-v1/00-overview.md)
- [Feature Engineering](../plans/ml-v1/03-feature-engineering.md)
- [ML Architecture](../plans/ml-v1/05-ml-architecture.md)
