"""
Evaluation Metrics for ML Trading Models

Provides both ML metrics (accuracy, F1) and trading metrics (Sharpe, drawdown).
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)


@dataclass
class TradingMetrics:
    """Container for trading performance metrics."""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    num_trades: int
    avg_trade_return: float
    avg_winner: float
    avg_loser: float
    largest_winner: float
    largest_loser: float


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = 'weighted'
) -> Dict[str, float]:
    """
    Calculate classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging method for multi-class metrics

    Returns:
        Dictionary with metrics
    """
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred, average=average),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
    }


def calculate_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    Calculate per-class metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names for each class

    Returns:
        Dictionary with per-class metrics
    """
    if class_names is None:
        class_names = ['sell', 'hold', 'buy']

    metrics = {}
    for cls_idx, cls_name in enumerate(class_names):
        mask = y_true == cls_idx
        if mask.sum() > 0:
            cls_pred = y_pred[mask]
            cls_true = y_true[mask]

            # Binary classification for this class
            binary_true = (cls_true == cls_idx).astype(int)
            binary_pred = (cls_pred == cls_idx).astype(int)

            metrics[cls_name] = {
                'accuracy': (cls_pred == cls_idx).mean(),
                'support': int(mask.sum()),
                'precision': precision_score(binary_true, binary_pred, zero_division=0),
                'recall': recall_score(binary_true, binary_pred, zero_division=0),
            }

    return metrics


def confusion_matrix_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = None,
    normalize: bool = False
) -> Dict[str, Any]:
    """
    Generate confusion matrix report.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names for each class
        normalize: Whether to normalize the matrix

    Returns:
        Dictionary with confusion matrix and statistics
    """
    if class_names is None:
        class_names = ['sell', 'hold', 'buy']

    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)

    # Calculate additional statistics
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    return {
        'confusion_matrix': cm.tolist(),
        'class_names': class_names,
        'classification_report': report,
        'overall_accuracy': accuracy_score(y_true, y_pred)
    }


def calculate_trading_metrics(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 365 * 24 * 60  # 1-minute bars
) -> TradingMetrics:
    """
    Calculate trading performance metrics from returns.

    Args:
        returns: Array of trade returns (percentage)
        risk_free_rate: Risk-free rate for Sharpe calculation
        periods_per_year: Number of periods in a year

    Returns:
        TradingMetrics dataclass
    """
    if len(returns) == 0:
        return TradingMetrics(
            total_return=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            num_trades=0,
            avg_trade_return=0.0,
            avg_winner=0.0,
            avg_loser=0.0,
            largest_winner=0.0,
            largest_loser=0.0
        )

    # Convert to numpy array
    returns = np.array(returns)

    # Total return
    total_return = (1 + returns / 100).prod() - 1

    # Sharpe ratio (annualized)
    excess_returns = returns - risk_free_rate / periods_per_year
    if excess_returns.std() > 0:
        sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(periods_per_year)
    else:
        sharpe = 0.0

    # Maximum drawdown
    cumulative = (1 + returns / 100).cumprod()
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0.0

    # Win rate
    winners = returns > 0
    losers = returns < 0
    win_rate = winners.sum() / len(returns) if len(returns) > 0 else 0.0

    # Profit factor
    gross_profit = returns[winners].sum() if winners.any() else 0
    gross_loss = abs(returns[losers].sum()) if losers.any() else 1e-10
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

    # Average trade metrics
    avg_trade_return = returns.mean()
    avg_winner = returns[winners].mean() if winners.any() else 0.0
    avg_loser = returns[losers].mean() if losers.any() else 0.0

    # Largest trades
    largest_winner = returns.max() if len(returns) > 0 else 0.0
    largest_loser = returns.min() if len(returns) > 0 else 0.0

    return TradingMetrics(
        total_return=total_return * 100,
        sharpe_ratio=sharpe,
        max_drawdown=max_drawdown * 100,
        win_rate=win_rate * 100,
        profit_factor=profit_factor,
        num_trades=len(returns),
        avg_trade_return=avg_trade_return,
        avg_winner=avg_winner,
        avg_loser=avg_loser,
        largest_winner=largest_winner,
        largest_loser=largest_loser
    )


def calculate_sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 365 * 24 * 60
) -> float:
    """
    Calculate Sortino ratio (downside risk-adjusted return).

    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate
        periods_per_year: Annualization factor

    Returns:
        Sortino ratio
    """
    excess_returns = returns - risk_free_rate / periods_per_year
    downside_returns = returns[returns < 0]

    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0.0

    downside_std = downside_returns.std()
    return (excess_returns.mean() / downside_std) * np.sqrt(periods_per_year)


def calculate_calmar_ratio(
    returns: np.ndarray,
    periods_per_year: int = 365 * 24 * 60
) -> float:
    """
    Calculate Calmar ratio (return / max drawdown).

    Args:
        returns: Array of returns
        periods_per_year: Annualization factor

    Returns:
        Calmar ratio
    """
    # Annualized return
    total_return = (1 + returns / 100).prod() - 1
    n_years = len(returns) / periods_per_year
    annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

    # Max drawdown
    cumulative = (1 + returns / 100).cumprod()
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 1e-10

    return annualized_return / max_drawdown if max_drawdown > 0 else 0.0


def signal_accuracy_by_confidence(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidences: np.ndarray,
    bins: int = 10
) -> Dict[str, List[float]]:
    """
    Calculate accuracy by confidence level.

    Useful for understanding model calibration.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        confidences: Prediction confidence scores
        bins: Number of confidence bins

    Returns:
        Dictionary with confidence bins and accuracies
    """
    bin_edges = np.linspace(0, 1, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    accuracies = []
    counts = []

    for i in range(bins):
        mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_accuracy = (y_true[mask] == y_pred[mask]).mean()
            accuracies.append(bin_accuracy)
            counts.append(mask.sum())
        else:
            accuracies.append(np.nan)
            counts.append(0)

    return {
        'bin_centers': bin_centers.tolist(),
        'accuracies': accuracies,
        'counts': counts,
        'expected_calibration_error': calculate_ece(
            y_true, y_pred, confidences, bins
        )
    }


def calculate_ece(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidences: np.ndarray,
    bins: int = 10
) -> float:
    """
    Calculate Expected Calibration Error.

    Measures how well-calibrated the model's confidence is.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        confidences: Prediction confidence scores
        bins: Number of confidence bins

    Returns:
        ECE score (lower is better, 0 = perfectly calibrated)
    """
    bin_edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    total_samples = len(y_true)

    for i in range(bins):
        mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_accuracy = (y_true[mask] == y_pred[mask]).mean()
            bin_confidence = confidences[mask].mean()
            bin_weight = mask.sum() / total_samples
            ece += bin_weight * abs(bin_accuracy - bin_confidence)

    return ece
