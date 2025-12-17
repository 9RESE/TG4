"""
Backtesting Module for ML Models

Simulates trading performance using ML model predictions
on historical data.
"""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd

from ..models.classifier import SignalClassifier
from ..models.predictor import LSTMPredictor
from .metrics import calculate_trading_metrics, TradingMetrics


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    initial_capital: float = 1000.0
    position_size_pct: float = 0.1  # 10% of capital per trade
    fee_rate: float = 0.001  # 0.1% per trade
    slippage_pct: float = 0.0005  # 0.05% slippage
    stop_loss_pct: float = 2.0  # 2% stop loss
    take_profit_pct: float = 4.0  # 4% take profit
    confidence_threshold: float = 0.6  # Minimum confidence for signal
    max_holding_periods: int = 60  # Maximum bars to hold position
    trading_days: int = 365  # Number of days in the backtest period


@dataclass
class Trade:
    """Single trade record."""
    entry_time: int
    exit_time: int
    entry_price: float
    exit_price: float
    side: str  # 'long' or 'short'
    size: float
    pnl: float
    return_pct: float
    exit_reason: str  # 'signal', 'stop_loss', 'take_profit', 'max_hold'


@dataclass
class BacktestResult:
    """Results from backtesting."""
    metrics: TradingMetrics
    equity_curve: List[float]
    trades: List[Trade]
    signals: List[Dict[str, Any]]
    config: BacktestConfig

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'metrics': {
                'total_return': self.metrics.total_return,
                'sharpe_ratio': self.metrics.sharpe_ratio,
                'max_drawdown': self.metrics.max_drawdown,
                'win_rate': self.metrics.win_rate,
                'profit_factor': self.metrics.profit_factor,
                'num_trades': self.metrics.num_trades,
                'avg_trade_return': self.metrics.avg_trade_return,
            },
            'equity_curve': self.equity_curve,
            'num_trades': len(self.trades),
            'num_signals': len(self.signals)
        }


def backtest_model(
    model: Union[SignalClassifier, LSTMPredictor],
    features: np.ndarray,
    prices: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    config: Optional[BacktestConfig] = None
) -> BacktestResult:
    """
    Backtest a trained ML model.

    Args:
        model: Trained signal classifier or LSTM predictor
        features: Feature array of shape (n_samples, n_features) or (n_samples, seq_len, n_features)
        prices: Price array of shape (n_samples,)
        timestamps: Optional timestamp array
        config: Backtest configuration

    Returns:
        BacktestResult with trading metrics
    """
    if config is None:
        config = BacktestConfig()

    # Get predictions
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(features)
    else:
        # For LSTM predictor
        probs = model.predict_proba(features)

    predictions = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)

    # Run simulation
    return simulate_trading(
        predictions=predictions,
        confidences=confidences,
        prices=prices,
        config=config
    )


def simulate_trading(
    predictions: np.ndarray,
    confidences: np.ndarray,
    prices: np.ndarray,
    config: BacktestConfig
) -> BacktestResult:
    """
    Simulate trading based on predictions.

    Args:
        predictions: Array of predictions (0=sell, 1=hold, 2=buy)
        confidences: Array of confidence scores
        prices: Array of prices
        config: Backtest configuration

    Returns:
        BacktestResult
    """
    n_samples = len(predictions)

    # Initialize state
    capital = config.initial_capital
    position = None  # Current position
    trades = []
    signals = []
    equity_curve = [capital]
    trade_returns = []

    for i in range(n_samples):
        price = prices[i]
        pred = predictions[i]
        conf = confidences[i]

        # Check if we have an open position
        if position is not None:
            # Check exit conditions
            exit_reason = None
            exit_price = price

            # Stop loss check
            if position['side'] == 'long':
                unrealized_return = (price - position['entry_price']) / position['entry_price'] * 100
                if unrealized_return <= -config.stop_loss_pct:
                    exit_reason = 'stop_loss'
                elif unrealized_return >= config.take_profit_pct:
                    exit_reason = 'take_profit'
            else:  # short
                unrealized_return = (position['entry_price'] - price) / position['entry_price'] * 100
                if unrealized_return <= -config.stop_loss_pct:
                    exit_reason = 'stop_loss'
                elif unrealized_return >= config.take_profit_pct:
                    exit_reason = 'take_profit'

            # Max holding period
            if i - position['entry_time'] >= config.max_holding_periods:
                exit_reason = 'max_hold'

            # Signal-based exit
            if exit_reason is None:
                # Exit long on sell signal
                if position['side'] == 'long' and pred == 0 and conf >= config.confidence_threshold:
                    exit_reason = 'signal'
                # Exit short on buy signal
                elif position['side'] == 'short' and pred == 2 and conf >= config.confidence_threshold:
                    exit_reason = 'signal'

            # Execute exit
            if exit_reason is not None:
                # Apply slippage
                if position['side'] == 'long':
                    exit_price = price * (1 - config.slippage_pct)
                else:
                    exit_price = price * (1 + config.slippage_pct)

                # Calculate P&L
                if position['side'] == 'long':
                    gross_pnl = (exit_price - position['entry_price']) * position['size']
                else:
                    gross_pnl = (position['entry_price'] - exit_price) * position['size']

                # Fees
                fees = exit_price * position['size'] * config.fee_rate
                net_pnl = gross_pnl - fees

                # Calculate return
                trade_return = net_pnl / (position['entry_price'] * position['size']) * 100

                # Record trade
                trade = Trade(
                    entry_time=position['entry_time'],
                    exit_time=i,
                    entry_price=position['entry_price'],
                    exit_price=exit_price,
                    side=position['side'],
                    size=position['size'],
                    pnl=net_pnl,
                    return_pct=trade_return,
                    exit_reason=exit_reason
                )
                trades.append(trade)
                trade_returns.append(trade_return)

                # Update capital
                capital += net_pnl
                position = None

        # Check for entry signal (only if no position)
        if position is None and conf >= config.confidence_threshold:
            # Buy signal
            if pred == 2:
                entry_price = price * (1 + config.slippage_pct)
                position_value = capital * config.position_size_pct
                size = position_value / entry_price

                # Entry fee
                fee = position_value * config.fee_rate
                capital -= fee

                position = {
                    'side': 'long',
                    'entry_time': i,
                    'entry_price': entry_price,
                    'size': size
                }

                signals.append({
                    'time': i,
                    'action': 'buy',
                    'price': price,
                    'confidence': conf
                })

            # Sell signal (for short)
            elif pred == 0:
                entry_price = price * (1 - config.slippage_pct)
                position_value = capital * config.position_size_pct
                size = position_value / entry_price

                # Entry fee
                fee = position_value * config.fee_rate
                capital -= fee

                position = {
                    'side': 'short',
                    'entry_time': i,
                    'entry_price': entry_price,
                    'size': size
                }

                signals.append({
                    'time': i,
                    'action': 'sell',
                    'price': price,
                    'confidence': conf
                })

        # Update equity curve
        if position is not None:
            # Mark-to-market
            if position['side'] == 'long':
                mtm = (price - position['entry_price']) * position['size']
            else:
                mtm = (position['entry_price'] - price) * position['size']
            equity_curve.append(capital + mtm)
        else:
            equity_curve.append(capital)

    # Close any remaining position at the end
    if position is not None:
        final_price = prices[-1]
        if position['side'] == 'long':
            gross_pnl = (final_price - position['entry_price']) * position['size']
        else:
            gross_pnl = (position['entry_price'] - final_price) * position['size']

        fees = final_price * position['size'] * config.fee_rate
        net_pnl = gross_pnl - fees
        trade_return = net_pnl / (position['entry_price'] * position['size']) * 100

        trade = Trade(
            entry_time=position['entry_time'],
            exit_time=len(prices) - 1,
            entry_price=position['entry_price'],
            exit_price=final_price,
            side=position['side'],
            size=position['size'],
            pnl=net_pnl,
            return_pct=trade_return,
            exit_reason='end_of_data'
        )
        trades.append(trade)
        trade_returns.append(trade_return)
        capital += net_pnl

    # Calculate metrics with correct annualization
    metrics = calculate_trading_metrics(
        np.array(trade_returns),
        trading_days=config.trading_days
    )

    return BacktestResult(
        metrics=metrics,
        equity_curve=equity_curve,
        trades=trades,
        signals=signals,
        config=config
    )


def walk_forward_backtest(
    model_class,
    features: np.ndarray,
    labels: np.ndarray,
    prices: np.ndarray,
    n_folds: int = 5,
    train_pct: float = 0.8,
    config: Optional[BacktestConfig] = None,
    model_params: Optional[Dict] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Walk-forward backtesting with model retraining.

    Trains model on expanding window and tests on out-of-sample data.

    Args:
        model_class: Model class to instantiate
        features: Full feature array
        labels: Full label array
        prices: Full price array
        n_folds: Number of test folds
        train_pct: Training data percentage per fold
        config: Backtest configuration
        model_params: Parameters for model initialization
        verbose: Print progress

    Returns:
        Combined backtest results across all folds
    """
    if config is None:
        config = BacktestConfig()
    if model_params is None:
        model_params = {}

    n_samples = len(features)
    fold_size = n_samples // n_folds

    all_trade_returns = []
    all_trades = []
    fold_results = []

    for fold in range(n_folds):
        # Determine train/test split for this fold
        test_start = fold * fold_size
        test_end = (fold + 1) * fold_size if fold < n_folds - 1 else n_samples

        # Train on all data before test period
        train_end = int(test_start * train_pct) if test_start > 0 else int(fold_size * train_pct)

        if train_end < 100:  # Need minimum training data
            continue

        X_train = features[:train_end]
        y_train = labels[:train_end]
        X_test = features[test_start:test_end]
        prices_test = prices[test_start:test_end]

        # Train model
        model = model_class(**model_params)
        model.fit(X_train, y_train, verbose=False)

        # Backtest on test period
        fold_result = backtest_model(model, X_test, prices_test, config=config)

        # Collect results
        all_trade_returns.extend([t.return_pct for t in fold_result.trades])
        all_trades.extend(fold_result.trades)
        fold_results.append({
            'fold': fold,
            'train_size': train_end,
            'test_size': test_end - test_start,
            'metrics': fold_result.metrics
        })

        if verbose:
            print(f"Fold {fold + 1}/{n_folds}: "
                  f"train={train_end}, test={test_end - test_start}, "
                  f"trades={fold_result.metrics.num_trades}, "
                  f"return={fold_result.metrics.total_return:.2f}%")

    # Calculate overall metrics using total trading days from config
    overall_metrics = calculate_trading_metrics(
        np.array(all_trade_returns),
        trading_days=config.trading_days
    )

    return {
        'overall_metrics': overall_metrics,
        'fold_results': fold_results,
        'all_trades': all_trades,
        'total_trades': len(all_trades)
    }
