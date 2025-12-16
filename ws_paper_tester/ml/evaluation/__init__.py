"""
Evaluation module for ML models.

Provides metrics calculation, backtesting integration, and
performance analysis tools.
"""

from .metrics import (
    calculate_metrics,
    calculate_trading_metrics,
    calculate_per_class_metrics,
    confusion_matrix_report,
    TradingMetrics,
    calculate_sortino_ratio,
    calculate_calmar_ratio,
    signal_accuracy_by_confidence
)
from .backtest import (
    backtest_model,
    simulate_trading,
    walk_forward_backtest,
    BacktestConfig,
    BacktestResult,
    Trade
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
    # Backtesting
    "backtest_model",
    "simulate_trading",
    "walk_forward_backtest",
    "BacktestConfig",
    "BacktestResult",
    "Trade",
]
