"""
Evaluation module for ML models.

Provides metrics calculation, backtesting integration, and
performance analysis tools.

New in v2.0:
- PerformanceTracker: Track model performance in TimescaleDB
"""

from .metrics import (
    calculate_metrics,
    calculate_trading_metrics,
    calculate_per_class_metrics,
    confusion_matrix_report,
    TradingMetrics,
    calculate_sortino_ratio,
    calculate_calmar_ratio,
    signal_accuracy_by_confidence,
    SignalPrecisionMetrics,
    calculate_signal_precision_metrics,
    calculate_profit_by_signal
)
from .backtest import (
    backtest_model,
    simulate_trading,
    walk_forward_backtest,
    BacktestConfig,
    BacktestResult,
    Trade
)
from .performance_tracker import (
    PerformanceTracker,
    ModelPerformanceRecord,
    save_training_performance,
)

__all__ = [
    # Metrics
    "calculate_metrics",
    "calculate_trading_metrics",
    "calculate_per_class_metrics",
    "confusion_matrix_report",
    "TradingMetrics",
    "calculate_sortino_ratio",
    "calculate_calmar_ratio",
    "signal_accuracy_by_confidence",
    "SignalPrecisionMetrics",
    "calculate_signal_precision_metrics",
    "calculate_profit_by_signal",
    # Backtesting
    "backtest_model",
    "simulate_trading",
    "walk_forward_backtest",
    "BacktestConfig",
    "BacktestResult",
    "Trade",
    # Performance tracking
    "PerformanceTracker",
    "ModelPerformanceRecord",
    "save_training_performance",
]
