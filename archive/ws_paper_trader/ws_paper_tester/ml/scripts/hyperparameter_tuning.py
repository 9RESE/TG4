#!/usr/bin/env python3
"""
Hyperparameter Tuning Script for ML Signal Classifier

This script systematically tests different parameter combinations to find
the best configuration for each asset pair and strategy.

WHAT IS HYPERPARAMETER TUNING?
==============================
Machine learning models have two types of parameters:
1. Model parameters - learned during training (e.g., tree splits, weights)
2. Hyperparameters - set BEFORE training (e.g., number of trees, learning rate)

Hyperparameter tuning is the process of finding the best hyperparameter values
by trying many combinations and comparing results.

GRID SEARCH vs RANDOM SEARCH
============================
- Grid Search: Try ALL combinations (thorough but slow)
- Random Search: Try random combinations (faster, often works well)

This script uses Grid Search for thoroughness, but limits the search space
to keep runtime reasonable.

Usage:
    # Tune for a specific symbol
    python -m ml.scripts.hyperparameter_tuning --symbol XRP/USDT

    # Tune for multiple symbols
    python -m ml.scripts.hyperparameter_tuning --symbol XRP/USDT BTC/USDT ETH/USDT

    # Use a specific strategy config
    python -m ml.scripts.hyperparameter_tuning --symbol XRP/USDT --strategy ema9_trend_flip

    # Quick mode (fewer combinations)
    python -m ml.scripts.hyperparameter_tuning --symbol XRP/USDT --quick

Output:
    - results/tuning/{symbol}_{strategy}_{timestamp}/
        - runs.csv          - All run results in CSV format
        - runs.json         - Detailed results for each run
        - report.md         - Human-readable summary report
        - best_config.json  - Best configuration found
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from itertools import product
import csv
import time

import numpy as np
import pandas as pd

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class TuningConfig:
    """
    Configuration for hyperparameter tuning.

    Each parameter has a list of values to try.
    Grid search will test ALL combinations.
    """
    # Label generation parameters
    threshold_pcts: List[float]    # Price change threshold for buy/sell
    lookahead_bars: List[int]      # How far ahead to look for price change

    # Model hyperparameters
    n_estimators: List[int]        # Number of trees in XGBoost
    max_depths: List[int]          # Maximum tree depth
    learning_rates: List[float]    # Learning rate (step size)
    subsamples: List[float]        # Fraction of data per tree

    # Data parameters - NOW TUNABLE!
    # More data = more patterns, but old data may not reflect current market
    history_days_options: List[int]  # Different history lengths to try


def get_available_data_range(db_url: str, symbol: str) -> Tuple[Optional[datetime], Optional[datetime], int]:
    """
    Query database to find available data range for a symbol.

    Returns:
        (min_date, max_date, total_days)
    """
    try:
        from sqlalchemy import create_engine, text

        engine = create_engine(db_url)
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT
                    MIN(timestamp) as min_ts,
                    MAX(timestamp) as max_ts,
                    COUNT(*) as total_candles
                FROM candles
                WHERE symbol = :symbol
            """), {'symbol': symbol})

            row = result.fetchone()
            if row and row[0]:
                min_date = row[0]
                max_date = row[1]
                total_days = (max_date - min_date).days
                return min_date, max_date, total_days

        return None, None, 0
    except Exception as e:
        print(f"Warning: Could not query data range: {e}")
        return None, None, 0


def suggest_history_options(total_days: int) -> List[int]:
    """
    Suggest history_days options based on available data.

    ML CONCEPT: Data quantity trade-offs
    - More data = more patterns to learn
    - But markets evolve, old data may not be relevant
    - We test multiple options to find the sweet spot
    """
    if total_days <= 7:
        return [total_days]
    elif total_days <= 30:
        return [7, 14, total_days]
    elif total_days <= 90:
        return [14, 30, 60, total_days]
    elif total_days <= 365:
        return [30, 60, 90, 180, total_days]
    else:
        # For very long histories, test various lengths
        return [30, 60, 90, 180, 365, total_days]


# Predefined tuning configurations
QUICK_CONFIG = TuningConfig(
    threshold_pcts=[0.3, 0.5, 0.7],
    lookahead_bars=[5, 10, 15],
    n_estimators=[100, 200],
    max_depths=[4, 6],
    learning_rates=[0.1],
    subsamples=[0.8],
    history_days_options=[30]  # Will be auto-updated based on available data
)

STANDARD_CONFIG = TuningConfig(
    threshold_pcts=[0.2, 0.3, 0.5, 0.7, 1.0],
    lookahead_bars=[5, 10, 15, 20],
    n_estimators=[100, 200, 300],
    max_depths=[4, 6, 8],
    learning_rates=[0.05, 0.1, 0.15],
    subsamples=[0.7, 0.8, 0.9],
    history_days_options=[30, 60, 90]  # Will be auto-updated
)

THOROUGH_CONFIG = TuningConfig(
    threshold_pcts=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0],
    lookahead_bars=[3, 5, 8, 10, 12, 15, 20],
    n_estimators=[100, 200, 300, 500],
    max_depths=[3, 4, 5, 6, 7, 8],
    learning_rates=[0.01, 0.05, 0.1, 0.15, 0.2],
    subsamples=[0.6, 0.7, 0.8, 0.9, 1.0],
    history_days_options=[30, 60, 90, 180, 365]  # Will be auto-updated
)


@dataclass
class RunResult:
    """Results from a single training run."""
    run_id: int
    params: Dict[str, Any]

    # Training metrics
    train_accuracy: float
    val_accuracy: float
    train_f1: float
    val_f1: float

    # Test metrics
    test_accuracy: float
    test_precision: float
    test_recall: float
    test_f1: float

    # Backtest metrics
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int

    # Metadata
    duration_seconds: float
    timestamp: str


def count_combinations(config: TuningConfig) -> int:
    """Count total number of parameter combinations."""
    return (
        len(config.threshold_pcts) *
        len(config.lookahead_bars) *
        len(config.n_estimators) *
        len(config.max_depths) *
        len(config.learning_rates) *
        len(config.subsamples) *
        len(config.history_days_options)
    )


def generate_param_grid(config: TuningConfig) -> List[Dict[str, Any]]:
    """
    Generate all parameter combinations from config.

    This is the core of grid search - we create every possible
    combination of parameters to test.

    Now includes history_days to find optimal data length!
    """
    combinations = []

    for threshold, lookahead, n_est, depth, lr, subsample, history in product(
        config.threshold_pcts,
        config.lookahead_bars,
        config.n_estimators,
        config.max_depths,
        config.learning_rates,
        config.subsamples,
        config.history_days_options
    ):
        combinations.append({
            'threshold_pct': threshold,
            'lookahead': lookahead,
            'n_estimators': n_est,
            'max_depth': depth,
            'learning_rate': lr,
            'subsample': subsample,
            'history_days': history
        })

    return combinations


def load_data_sync(db_url: str, symbol: str, days: int) -> Optional[pd.DataFrame]:
    """
    Load historical data from database (synchronous).

    Uses SQLAlchemy for synchronous access instead of asyncpg.
    """
    from sqlalchemy import create_engine, text
    from datetime import timedelta

    try:
        engine = create_engine(db_url)
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)

        query = text("""
            SELECT timestamp, open, high, low, close, volume
            FROM candles
            WHERE symbol = :symbol
              AND interval_minutes = 1
              AND timestamp >= :start_time
              AND timestamp < :end_time
            ORDER BY timestamp ASC
        """)

        with engine.connect() as conn:
            result = conn.execute(query, {
                'symbol': symbol,
                'start_time': start_time,
                'end_time': end_time
            })
            rows = result.fetchall()

        if not rows:
            return None

        df = pd.DataFrame(rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        # Convert Decimal to float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        return df

    except Exception as e:
        return None


def run_single_training(
    symbol: str,
    params: Dict[str, Any],
    db_url: str,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run a single training with given parameters.

    params should include 'history_days' - the amount of data to use.
    This is now tunable to find the optimal data length!

    Returns dictionary with all metrics.
    """
    from ml.scripts.train_signal_classifier import (
        extract_features,
        generate_labels,
        prepare_train_test_split,
        evaluate_model
    )
    from ml.models.classifier import XGBoostClassifier

    # Load data - history_days is now in params
    history_days = params.get('history_days', 30)
    df = load_data_sync(db_url, symbol, history_days)

    if df is None or len(df) < 100:
        return {'error': 'Insufficient data'}

    # Extract features
    features = extract_features(df, verbose=False)

    # Generate labels with current parameters
    labels = generate_labels(
        df,
        threshold_pct=params['threshold_pct'],
        lookahead=params['lookahead'],
        verbose=False
    )

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_train_test_split(
        features, labels, verbose=False
    )

    if len(X_train) == 0:
        return {'error': 'No valid training samples'}

    # Get prices for backtesting (use same split ratio)
    prices = df['close'].values
    # Remove warmup NaNs (same as features)
    valid_start = len(prices) - (len(X_train) + len(X_val) + len(X_test))
    prices_valid = prices[valid_start:]
    train_end = len(X_train)
    val_end = train_end + len(X_val)
    prices_test = prices_valid[val_end:]

    # Create and train model
    model = XGBoostClassifier(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        learning_rate=params['learning_rate'],
        subsample=params['subsample'],
        colsample_bytree=0.8,
        device='cpu'  # Use CPU for parallel runs
    )

    # Train (suppress output)
    train_metrics = model.fit(X_train, y_train, X_val, y_val, verbose=False)

    # Evaluate with prices for backtesting
    eval_results = evaluate_model(model, X_test, y_test, prices_test, verbose=False)

    # Extract metrics from nested structure
    classification = eval_results.get('classification', {})
    backtest = eval_results.get('backtest', None)

    return {
        'train_accuracy': train_metrics['train_accuracy'],
        'val_accuracy': train_metrics['val_accuracy'],
        'train_f1': train_metrics.get('train_f1', 0),
        'val_f1': train_metrics.get('val_f1', 0),
        'test_accuracy': classification.get('accuracy', 0),
        'test_precision': classification.get('precision', 0),
        'test_recall': classification.get('recall', 0),
        'test_f1': classification.get('f1_score', 0),
        'total_return': backtest.total_return if backtest else 0,
        'sharpe_ratio': backtest.sharpe_ratio if backtest else 0,
        'max_drawdown': backtest.max_drawdown if backtest else 0,
        'win_rate': backtest.win_rate if backtest else 0,
        'total_trades': backtest.num_trades if backtest else 0,
    }


def run_tuning(
    symbol: str,
    config: TuningConfig,
    db_url: str,
    output_dir: Path,
    strategy: str = 'default',
    verbose: bool = True
) -> List[RunResult]:
    """
    Run hyperparameter tuning for a symbol.

    This is the main tuning loop that:
    1. Generates all parameter combinations
    2. Trains a model for each combination
    3. Records all results
    4. Reports progress
    """
    results = []
    param_grid = generate_param_grid(config)
    total_runs = len(param_grid)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  HYPERPARAMETER TUNING: {symbol}")
        print(f"{'='*60}")
        print(f"Strategy: {strategy}")
        print(f"Total combinations to test: {total_runs}")
        print(f"Output directory: {output_dir}")
        print()

    # Progress tracking
    best_sharpe = float('-inf')
    best_accuracy = 0
    best_params = None

    for run_id, params in enumerate(param_grid, 1):
        start_time = time.time()

        if verbose:
            progress = run_id / total_runs * 100
            print(f"\r[{progress:5.1f}%] Run {run_id}/{total_runs}: "
                  f"days={params['history_days']}, "
                  f"thresh={params['threshold_pct']}, "
                  f"look={params['lookahead']}, "
                  f"trees={params['n_estimators']}", end='')

        try:
            metrics = run_single_training(
                symbol=symbol,
                params=params,
                db_url=db_url,
                verbose=False
            )

            if 'error' in metrics:
                if verbose:
                    print(f" - ERROR: {metrics['error']}")
                continue

            duration = time.time() - start_time

            result = RunResult(
                run_id=run_id,
                params=params,
                train_accuracy=metrics['train_accuracy'],
                val_accuracy=metrics['val_accuracy'],
                train_f1=metrics.get('train_f1', 0),
                val_f1=metrics.get('val_f1', 0),
                test_accuracy=metrics.get('test_accuracy', 0),
                test_precision=metrics.get('test_precision', 0),
                test_recall=metrics.get('test_recall', 0),
                test_f1=metrics.get('test_f1', 0),
                total_return=metrics.get('total_return', 0),
                sharpe_ratio=metrics.get('sharpe_ratio', 0),
                max_drawdown=metrics.get('max_drawdown', 0),
                win_rate=metrics.get('win_rate', 0),
                total_trades=metrics.get('total_trades', 0),
                duration_seconds=duration,
                timestamp=datetime.now().isoformat()
            )

            results.append(result)

            # Track best results
            if result.sharpe_ratio > best_sharpe:
                best_sharpe = result.sharpe_ratio
                best_params = params.copy()
                best_params['sharpe'] = best_sharpe
                best_params['accuracy'] = result.test_accuracy

            if result.test_accuracy > best_accuracy:
                best_accuracy = result.test_accuracy

            if verbose and run_id % 10 == 0:
                print(f" | Best Sharpe: {best_sharpe:.2f}, Best Acc: {best_accuracy*100:.1f}%")

        except Exception as e:
            if verbose:
                print(f" - FAILED: {str(e)[:50]}")
            continue

    if verbose:
        print(f"\n\nCompleted {len(results)}/{total_runs} runs successfully")
        if best_params:
            print(f"\nBest configuration found:")
            print(f"  Sharpe Ratio: {best_params.get('sharpe', 0):.2f}")
            print(f"  Accuracy: {best_params.get('accuracy', 0)*100:.1f}%")
            print(f"  Parameters: threshold={best_params['threshold_pct']}, "
                  f"lookahead={best_params['lookahead']}")

    return results


def save_results(
    results: List[RunResult],
    output_dir: Path,
    symbol: str,
    strategy: str
):
    """Save all results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert results to list of dicts
    results_data = []
    for r in results:
        data = asdict(r)
        # Flatten params
        params = data.pop('params')
        data.update(params)
        results_data.append(data)

    # Save as CSV
    csv_path = output_dir / 'runs.csv'
    if results_data:
        df = pd.DataFrame(results_data)
        df.to_csv(csv_path, index=False)
        print(f"  Saved CSV: {csv_path}")

    # Save as JSON (more detailed)
    json_path = output_dir / 'runs.json'
    with open(json_path, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2, default=str)
    print(f"  Saved JSON: {json_path}")

    return csv_path, json_path


def generate_report(
    results: List[RunResult],
    output_dir: Path,
    symbol: str,
    strategy: str,
    config: TuningConfig
) -> Path:
    """
    Generate a comprehensive markdown report.

    The report includes:
    - Summary statistics
    - Best configurations by different metrics
    - Parameter sensitivity analysis
    - Recommendations
    """
    report_path = output_dir / 'report.md'

    if not results:
        with open(report_path, 'w') as f:
            f.write(f"# Hyperparameter Tuning Report\n\n")
            f.write(f"**Symbol:** {symbol}\n")
            f.write(f"**Status:** No successful runs\n")
        return report_path

    # Sort results by different metrics
    by_sharpe = sorted(results, key=lambda x: x.sharpe_ratio, reverse=True)
    by_accuracy = sorted(results, key=lambda x: x.test_accuracy, reverse=True)
    by_return = sorted(results, key=lambda x: x.total_return, reverse=True)
    by_winrate = sorted(results, key=lambda x: x.win_rate, reverse=True)

    # Calculate statistics
    sharpes = [r.sharpe_ratio for r in results if not np.isinf(r.sharpe_ratio)]
    accuracies = [r.test_accuracy for r in results]
    returns = [r.total_return for r in results]

    with open(report_path, 'w') as f:
        # Header
        f.write(f"# Hyperparameter Tuning Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Summary
        f.write(f"## Summary\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| Symbol | {symbol} |\n")
        f.write(f"| Strategy | {strategy} |\n")
        f.write(f"| Total Runs | {len(results)} |\n")
        f.write(f"| History Days Tested | {config.history_days_options} |\n\n")

        # Statistics
        f.write(f"## Performance Statistics\n\n")
        f.write(f"| Metric | Min | Mean | Max | Std |\n")
        f.write(f"|--------|-----|------|-----|-----|\n")

        if sharpes:
            f.write(f"| Sharpe Ratio | {min(sharpes):.2f} | {np.mean(sharpes):.2f} | "
                   f"{max(sharpes):.2f} | {np.std(sharpes):.2f} |\n")
        f.write(f"| Accuracy | {min(accuracies)*100:.1f}% | {np.mean(accuracies)*100:.1f}% | "
               f"{max(accuracies)*100:.1f}% | {np.std(accuracies)*100:.1f}% |\n")
        f.write(f"| Return | {min(returns):.2f}% | {np.mean(returns):.2f}% | "
               f"{max(returns):.2f}% | {np.std(returns):.2f}% |\n\n")

        # Best configurations
        f.write(f"## Best Configurations\n\n")

        # By Sharpe
        f.write(f"### Top 5 by Sharpe Ratio\n\n")
        f.write(f"| Rank | Sharpe | Return | Accuracy | Days | Threshold | Lookahead | Trees | Depth |\n")
        f.write(f"|------|--------|--------|----------|------|-----------|-----------|-------|-------|\n")
        for i, r in enumerate(by_sharpe[:5], 1):
            f.write(f"| {i} | {r.sharpe_ratio:.2f} | {r.total_return:.2f}% | "
                   f"{r.test_accuracy*100:.1f}% | {r.params['history_days']} | "
                   f"{r.params['threshold_pct']} | {r.params['lookahead']} | "
                   f"{r.params['n_estimators']} | {r.params['max_depth']} |\n")
        f.write("\n")

        # By Accuracy
        f.write(f"### Top 5 by Accuracy\n\n")
        f.write(f"| Rank | Accuracy | F1 | Return | Threshold | Lookahead | Trees | Depth |\n")
        f.write(f"|------|----------|-----|--------|-----------|-----------|-------|-------|\n")
        for i, r in enumerate(by_accuracy[:5], 1):
            f.write(f"| {i} | {r.test_accuracy*100:.1f}% | {r.test_f1*100:.1f}% | "
                   f"{r.total_return:.2f}% | {r.params['threshold_pct']} | "
                   f"{r.params['lookahead']} | {r.params['n_estimators']} | "
                   f"{r.params['max_depth']} |\n")
        f.write("\n")

        # By Return
        f.write(f"### Top 5 by Total Return\n\n")
        f.write(f"| Rank | Return | Sharpe | Win Rate | Trades | Threshold | Lookahead |\n")
        f.write(f"|------|--------|--------|----------|--------|-----------|----------|\n")
        for i, r in enumerate(by_return[:5], 1):
            f.write(f"| {i} | {r.total_return:.2f}% | {r.sharpe_ratio:.2f} | "
                   f"{r.win_rate*100:.1f}% | {r.total_trades} | "
                   f"{r.params['threshold_pct']} | {r.params['lookahead']} |\n")
        f.write("\n")

        # Parameter Sensitivity Analysis
        f.write(f"## Parameter Sensitivity Analysis\n\n")
        f.write(f"Average Sharpe ratio by parameter value:\n\n")

        # Analyze each parameter - HISTORY_DAYS IS FIRST because it's most important!
        for param_name in ['history_days', 'threshold_pct', 'lookahead', 'n_estimators', 'max_depth', 'learning_rate']:
            f.write(f"### {param_name.replace('_', ' ').title()}\n\n")

            # Group results by this parameter
            param_groups = {}
            for r in results:
                val = r.params[param_name]
                if val not in param_groups:
                    param_groups[val] = []
                if not np.isinf(r.sharpe_ratio):
                    param_groups[val].append(r.sharpe_ratio)

            f.write(f"| Value | Avg Sharpe | Count |\n")
            f.write(f"|-------|------------|-------|\n")
            for val in sorted(param_groups.keys()):
                sharpes = param_groups[val]
                if sharpes:
                    f.write(f"| {val} | {np.mean(sharpes):.2f} | {len(sharpes)} |\n")
            f.write("\n")

        # Recommendations
        f.write(f"## Recommendations\n\n")

        best = by_sharpe[0] if by_sharpe else None
        if best:
            f.write(f"Based on the tuning results, the recommended configuration is:\n\n")
            f.write(f"```python\n")
            f.write(f"# Best configuration for {symbol}\n")
            f.write(f"config = {{\n")
            f.write(f"    'history_days': {best.params['history_days']},  # Optimal data length\n")
            f.write(f"    'threshold_pct': {best.params['threshold_pct']},\n")
            f.write(f"    'lookahead': {best.params['lookahead']},\n")
            f.write(f"    'n_estimators': {best.params['n_estimators']},\n")
            f.write(f"    'max_depth': {best.params['max_depth']},\n")
            f.write(f"    'learning_rate': {best.params['learning_rate']},\n")
            f.write(f"    'subsample': {best.params['subsample']},\n")
            f.write(f"}}\n")
            f.write(f"```\n\n")

            f.write(f"**Expected Performance:**\n")
            f.write(f"- Sharpe Ratio: {best.sharpe_ratio:.2f}\n")
            f.write(f"- Accuracy: {best.test_accuracy*100:.1f}%\n")
            f.write(f"- Total Return: {best.total_return:.2f}%\n")
            f.write(f"- Win Rate: {best.win_rate*100:.1f}%\n\n")

        # Notes
        f.write(f"## Notes\n\n")
        f.write(f"- Results are based on backtesting with historical data\n")
        f.write(f"- Past performance does not guarantee future results\n")
        f.write(f"- Consider running paper trading before live deployment\n")
        f.write(f"- Market conditions change; periodic re-tuning is recommended\n")

    print(f"  Saved Report: {report_path}")
    return report_path


def save_best_config(
    results: List[RunResult],
    output_dir: Path,
    symbol: str,
    strategy: str
) -> Optional[Path]:
    """Save the best configuration as a JSON file."""
    if not results:
        return None

    # Find best by Sharpe ratio
    best = max(results, key=lambda x: x.sharpe_ratio if not np.isinf(x.sharpe_ratio) else float('-inf'))

    config_path = output_dir / 'best_config.json'

    config_data = {
        'symbol': symbol,
        'strategy': strategy,
        'generated': datetime.now().isoformat(),
        'parameters': best.params,
        'expected_performance': {
            'sharpe_ratio': best.sharpe_ratio,
            'accuracy': best.test_accuracy,
            'total_return': best.total_return,
            'win_rate': best.win_rate,
            'max_drawdown': best.max_drawdown
        }
    }

    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)

    print(f"  Saved Best Config: {config_path}")
    return config_path


def main():
    parser = argparse.ArgumentParser(
        description='Hyperparameter tuning for ML signal classifier',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick tuning for one symbol
  python -m ml.scripts.hyperparameter_tuning --symbol XRP/USDT --quick

  # Standard tuning for multiple symbols
  python -m ml.scripts.hyperparameter_tuning --symbol XRP/USDT BTC/USDT

  # Thorough tuning with specific strategy
  python -m ml.scripts.hyperparameter_tuning --symbol XRP/USDT --strategy ema9_trend_flip --thorough
        """
    )

    parser.add_argument(
        '--symbol', '-s',
        nargs='+',
        required=True,
        help='Symbol(s) to tune (e.g., XRP/USDT BTC/USDT)'
    )

    parser.add_argument(
        '--strategy',
        default='default',
        help='Strategy name for organizing results (default: default)'
    )

    parser.add_argument(
        '--quick', '-q',
        action='store_true',
        help='Use quick tuning config (fewer combinations, faster)'
    )

    parser.add_argument(
        '--thorough', '-t',
        action='store_true',
        help='Use thorough tuning config (more combinations, slower)'
    )

    parser.add_argument(
        '--db-url',
        help='Database URL (or set DATABASE_URL env var)'
    )

    parser.add_argument(
        '--output-dir', '-o',
        default='results/tuning',
        help='Output directory for results (default: results/tuning)'
    )

    parser.add_argument(
        '--days',
        type=int,
        nargs='+',
        help='Specific history days to test (e.g., --days 30 60 90)'
    )

    parser.add_argument(
        '--auto-detect',
        action='store_true',
        default=True,
        help='Auto-detect available data and suggest history options (default: True)'
    )

    parser.add_argument(
        '--use-all-data',
        action='store_true',
        help='Use ALL available historical data (may be slow)'
    )

    args = parser.parse_args()

    # Get database URL
    db_url = args.db_url or os.environ.get('DATABASE_URL')
    if not db_url:
        print("ERROR: Database URL required.")
        print("Set DATABASE_URL environment variable or use --db-url")
        sys.exit(1)

    # Select base tuning config
    if args.quick:
        config = QUICK_CONFIG
        config_name = "QUICK"
    elif args.thorough:
        config = THOROUGH_CONFIG
        config_name = "THOROUGH"
    else:
        config = STANDARD_CONFIG
        config_name = "STANDARD"

    # Run tuning for each symbol
    all_results = {}

    for symbol in args.symbol:
        print(f"\n{'='*60}")
        print(f"  Processing: {symbol}")
        print(f"{'='*60}")

        # Auto-detect available data
        min_date, max_date, total_days = get_available_data_range(db_url, symbol)

        if total_days == 0:
            print(f"WARNING: No data found for {symbol}, skipping...")
            continue

        print(f"\nData available:")
        print(f"  Date range: {min_date} to {max_date}")
        print(f"  Total days: {total_days}")

        # Determine history_days_options to test
        if args.days:
            # User specified specific days to test
            history_options = [d for d in args.days if d <= total_days]
            print(f"  Using specified days: {history_options}")
        elif args.use_all_data:
            # Use all data plus some smaller windows
            history_options = suggest_history_options(total_days)
            print(f"  Using all available data with options: {history_options}")
        else:
            # Auto-suggest based on available data
            history_options = suggest_history_options(total_days)
            # Limit to reasonable number for standard config
            if config_name == "QUICK":
                history_options = history_options[:2]  # Just 2 options for quick
            elif config_name == "STANDARD":
                history_options = history_options[:4]  # Up to 4 options
            print(f"  Auto-detected history options: {history_options}")

        # Update config with history options
        from copy import deepcopy
        symbol_config = deepcopy(config)
        symbol_config.history_days_options = history_options

        total_combinations = count_combinations(symbol_config)
        print(f"\nUsing {config_name} tuning config")
        print(f"Total parameter combinations: {total_combinations}")
        print(f"Estimated time: ~{total_combinations * 3 / 60:.1f} minutes")

        # Create output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        symbol_safe = symbol.replace('/', '_')
        output_dir = Path(args.output_dir) / f"{symbol_safe}_{args.strategy}_{timestamp}"

        # Run tuning
        results = run_tuning(
            symbol=symbol,
            config=symbol_config,
            db_url=db_url,
            output_dir=output_dir,
            strategy=args.strategy,
            verbose=True
        )

        if results:
            # Save results
            print(f"\nSaving results...")
            save_results(results, output_dir, symbol, args.strategy)
            generate_report(results, output_dir, symbol, args.strategy, symbol_config)
            save_best_config(results, output_dir, symbol, args.strategy)

            all_results[symbol] = output_dir

    # Final summary
    print(f"\n{'='*60}")
    print(f"  TUNING COMPLETE")
    print(f"{'='*60}")
    print(f"\nResults saved to:")
    for symbol, path in all_results.items():
        print(f"  {symbol}: {path}")

    print(f"\nNext steps:")
    print(f"  1. Review the report.md files for each symbol")
    print(f"  2. Copy best_config.json to your strategy config")
    print(f"  3. Run paper trading to validate")


if __name__ == '__main__':
    main()
