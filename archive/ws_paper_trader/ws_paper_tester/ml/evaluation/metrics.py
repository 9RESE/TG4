"""
Evaluation Metrics for ML Trading Models

Provides both ML metrics (accuracy, F1) and trading metrics (Sharpe, drawdown).
"""

from typing import Dict, Any, List, Optional, Union
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
    trades_per_year: Optional[int] = None,
    trading_days: int = 365
) -> TradingMetrics:
    """
    Calculate trading performance metrics from returns.

    Args:
        returns: Array of trade returns (percentage)
        risk_free_rate: Risk-free rate for Sharpe calculation (annual)
        trades_per_year: Expected trades per year for annualization.
            If None, calculated from len(returns) assuming data covers trading_days.
        trading_days: Number of days the returns data covers (default 365)

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

    # Calculate annualization factor based on trade frequency
    # For trade returns, we annualize based on trades per year, not bars per year
    if trades_per_year is None:
        # Estimate trades per year from the data
        trades_per_year = max(1, int(len(returns) * 365 / trading_days))

    # Sharpe ratio (annualized) - using trade frequency
    # Risk-free rate per trade = annual rate / trades per year
    rf_per_trade = risk_free_rate / trades_per_year if trades_per_year > 0 else 0
    excess_returns = returns - rf_per_trade
    if excess_returns.std() > 0:
        # Annualize: multiply by sqrt(trades_per_year)
        sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(trades_per_year)
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
    trades_per_year: Optional[int] = None,
    trading_days: int = 365
) -> float:
    """
    Calculate Sortino ratio (downside risk-adjusted return).

    Args:
        returns: Array of trade returns
        risk_free_rate: Annual risk-free rate
        trades_per_year: Expected trades per year for annualization
        trading_days: Number of days the returns data covers

    Returns:
        Sortino ratio
    """
    if trades_per_year is None:
        trades_per_year = max(1, int(len(returns) * 365 / trading_days))

    rf_per_trade = risk_free_rate / trades_per_year if trades_per_year > 0 else 0
    excess_returns = returns - rf_per_trade
    downside_returns = returns[returns < 0]

    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0.0

    downside_std = downside_returns.std()
    return (excess_returns.mean() / downside_std) * np.sqrt(trades_per_year)


def calculate_calmar_ratio(
    returns: np.ndarray,
    trading_days: int = 365
) -> float:
    """
    Calculate Calmar ratio (return / max drawdown).

    Args:
        returns: Array of trade returns
        trading_days: Number of days the returns data covers

    Returns:
        Calmar ratio
    """
    # Total return
    total_return = (1 + returns / 100).prod() - 1

    # Annualize based on actual trading period
    n_years = trading_days / 365
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


@dataclass
class SignalPrecisionMetrics:
    """Precision-focused metrics for buy/sell signals."""
    # Buy signal metrics
    buy_precision: float  # When we predict buy, how often is it correct?
    buy_recall: float  # Of all true buys, how many did we catch?
    buy_f1: float
    buy_count: int  # Number of buy predictions
    buy_true_count: int  # Number of actual buy labels

    # Sell signal metrics
    sell_precision: float
    sell_recall: float
    sell_f1: float
    sell_count: int
    sell_true_count: int

    # Combined actionable signal metrics
    action_precision: float  # Precision for non-hold predictions
    action_recall: float  # Recall for non-hold predictions
    action_f1: float
    action_count: int

    # Trading-relevant metrics
    false_buy_rate: float  # Rate of false buy signals (costly)
    false_sell_rate: float  # Rate of false sell signals (costly)
    hold_bias: float  # Fraction of predictions that are hold


def calculate_signal_precision_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidences: Optional[np.ndarray] = None,
    confidence_threshold: float = 0.0
) -> SignalPrecisionMetrics:
    """
    Calculate precision-focused metrics specifically for trading signals.

    These metrics focus on the quality of BUY and SELL signals,
    which are what actually generate trades and P&L.

    Labels: 0=SELL, 1=HOLD, 2=BUY

    Args:
        y_true: True labels
        y_pred: Predicted labels
        confidences: Optional confidence scores for filtering
        confidence_threshold: Minimum confidence to consider a prediction

    Returns:
        SignalPrecisionMetrics with detailed buy/sell analysis
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Apply confidence threshold if provided
    if confidences is not None and confidence_threshold > 0:
        confidences = np.array(confidences)
        mask = confidences >= confidence_threshold
        # Convert low-confidence predictions to HOLD
        y_pred_filtered = y_pred.copy()
        y_pred_filtered[~mask] = 1  # HOLD
    else:
        y_pred_filtered = y_pred

    # Buy signal metrics (class 2)
    buy_pred_mask = y_pred_filtered == 2
    buy_true_mask = y_true == 2
    buy_count = buy_pred_mask.sum()
    buy_true_count = buy_true_mask.sum()

    if buy_count > 0:
        buy_precision = (y_true[buy_pred_mask] == 2).sum() / buy_count
    else:
        buy_precision = 0.0

    if buy_true_count > 0:
        buy_recall = (y_pred_filtered[buy_true_mask] == 2).sum() / buy_true_count
    else:
        buy_recall = 0.0

    buy_f1 = 2 * buy_precision * buy_recall / (buy_precision + buy_recall) if (buy_precision + buy_recall) > 0 else 0.0

    # Sell signal metrics (class 0)
    sell_pred_mask = y_pred_filtered == 0
    sell_true_mask = y_true == 0
    sell_count = sell_pred_mask.sum()
    sell_true_count = sell_true_mask.sum()

    if sell_count > 0:
        sell_precision = (y_true[sell_pred_mask] == 0).sum() / sell_count
    else:
        sell_precision = 0.0

    if sell_true_count > 0:
        sell_recall = (y_pred_filtered[sell_true_mask] == 0).sum() / sell_true_count
    else:
        sell_recall = 0.0

    sell_f1 = 2 * sell_precision * sell_recall / (sell_precision + sell_recall) if (sell_precision + sell_recall) > 0 else 0.0

    # Combined actionable signal metrics (BUY or SELL, not HOLD)
    action_pred_mask = y_pred_filtered != 1
    action_true_mask = y_true != 1
    action_count = action_pred_mask.sum()
    action_true_count = action_true_mask.sum()

    if action_count > 0:
        # Correct action = predicted action matches true action (both non-hold)
        action_precision = ((y_pred_filtered == y_true) & action_pred_mask).sum() / action_count
    else:
        action_precision = 0.0

    if action_true_count > 0:
        action_recall = ((y_pred_filtered == y_true) & action_true_mask).sum() / action_true_count
    else:
        action_recall = 0.0

    action_f1 = 2 * action_precision * action_recall / (action_precision + action_recall) if (action_precision + action_recall) > 0 else 0.0

    # False signal rates (these are the costly mistakes)
    # False buy = predicted buy but actual was sell or hold
    if buy_count > 0:
        false_buy_rate = (y_true[buy_pred_mask] != 2).sum() / buy_count
    else:
        false_buy_rate = 0.0

    # False sell = predicted sell but actual was buy or hold
    if sell_count > 0:
        false_sell_rate = (y_true[sell_pred_mask] != 0).sum() / sell_count
    else:
        false_sell_rate = 0.0

    # Hold bias = fraction of predictions that are HOLD
    hold_bias = (y_pred_filtered == 1).sum() / len(y_pred_filtered) if len(y_pred_filtered) > 0 else 0.0

    return SignalPrecisionMetrics(
        buy_precision=float(buy_precision),
        buy_recall=float(buy_recall),
        buy_f1=float(buy_f1),
        buy_count=int(buy_count),
        buy_true_count=int(buy_true_count),
        sell_precision=float(sell_precision),
        sell_recall=float(sell_recall),
        sell_f1=float(sell_f1),
        sell_count=int(sell_count),
        sell_true_count=int(sell_true_count),
        action_precision=float(action_precision),
        action_recall=float(action_recall),
        action_f1=float(action_f1),
        action_count=int(action_count),
        false_buy_rate=float(false_buy_rate),
        false_sell_rate=float(false_sell_rate),
        hold_bias=float(hold_bias)
    )


def calculate_profit_by_signal(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    price_returns: np.ndarray,
    confidences: Optional[np.ndarray] = None,
    confidence_threshold: float = 0.4
) -> Dict[str, Any]:
    """
    Calculate actual profit/loss by signal type.

    This ties predictions directly to P&L, showing which signals make money.

    Args:
        y_true: True labels (0=sell, 1=hold, 2=buy)
        y_pred: Predicted labels
        price_returns: Actual price returns for each sample (percentage)
        confidences: Confidence scores
        confidence_threshold: Minimum confidence for action

    Returns:
        Dictionary with P&L breakdown by signal
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    price_returns = np.array(price_returns)

    # Apply confidence filter
    if confidences is not None:
        confidences = np.array(confidences)
        action_mask = confidences >= confidence_threshold
    else:
        action_mask = np.ones(len(y_pred), dtype=bool)

    results = {
        'buy_signals': {
            'count': 0,
            'correct': 0,
            'avg_return_when_correct': 0.0,
            'avg_return_when_wrong': 0.0,
            'total_return': 0.0
        },
        'sell_signals': {
            'count': 0,
            'correct': 0,
            'avg_return_when_correct': 0.0,
            'avg_return_when_wrong': 0.0,
            'total_return': 0.0
        },
        'hold_signals': {
            'count': 0,
            'missed_opportunities': 0,
            'avg_missed_return': 0.0
        }
    }

    # Buy signals
    buy_mask = (y_pred == 2) & action_mask
    if buy_mask.sum() > 0:
        buy_returns = price_returns[buy_mask]
        buy_correct = y_true[buy_mask] == 2

        results['buy_signals']['count'] = int(buy_mask.sum())
        results['buy_signals']['correct'] = int(buy_correct.sum())
        results['buy_signals']['total_return'] = float(buy_returns.sum())

        if buy_correct.sum() > 0:
            results['buy_signals']['avg_return_when_correct'] = float(buy_returns[buy_correct].mean())
        if (~buy_correct).sum() > 0:
            results['buy_signals']['avg_return_when_wrong'] = float(buy_returns[~buy_correct].mean())

    # Sell signals (profit = negative of price return for shorts)
    sell_mask = (y_pred == 0) & action_mask
    if sell_mask.sum() > 0:
        sell_returns = -price_returns[sell_mask]  # Invert for short positions
        sell_correct = y_true[sell_mask] == 0

        results['sell_signals']['count'] = int(sell_mask.sum())
        results['sell_signals']['correct'] = int(sell_correct.sum())
        results['sell_signals']['total_return'] = float(sell_returns.sum())

        if sell_correct.sum() > 0:
            results['sell_signals']['avg_return_when_correct'] = float(sell_returns[sell_correct].mean())
        if (~sell_correct).sum() > 0:
            results['sell_signals']['avg_return_when_wrong'] = float(sell_returns[~sell_correct].mean())

    # Hold signals - check for missed opportunities
    hold_mask = (y_pred == 1) | (~action_mask)
    if hold_mask.sum() > 0:
        # Missed opportunities = true signal was buy/sell but we held
        missed_mask = hold_mask & (y_true != 1)

        results['hold_signals']['count'] = int(hold_mask.sum())
        results['hold_signals']['missed_opportunities'] = int(missed_mask.sum())

        if missed_mask.sum() > 0:
            # Calculate what we could have made
            missed_returns = np.abs(price_returns[missed_mask])
            results['hold_signals']['avg_missed_return'] = float(missed_returns.mean())

    # Overall summary
    total_trades = results['buy_signals']['count'] + results['sell_signals']['count']
    total_correct = results['buy_signals']['correct'] + results['sell_signals']['correct']
    total_return = results['buy_signals']['total_return'] + results['sell_signals']['total_return']

    results['summary'] = {
        'total_trades': total_trades,
        'total_correct': total_correct,
        'overall_accuracy': total_correct / total_trades if total_trades > 0 else 0.0,
        'total_return': total_return,
        'avg_return_per_trade': total_return / total_trades if total_trades > 0 else 0.0
    }

    return results
