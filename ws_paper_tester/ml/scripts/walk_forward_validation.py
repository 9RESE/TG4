#!/usr/bin/env python3
"""
Walk-Forward Validation Script

Performs realistic out-of-sample validation using expanding window training.
This gives much more realistic performance estimates than a single train/test split.

How it works:
1. Divide data into N folds chronologically
2. For each fold:
   - Train on all data BEFORE the fold
   - Test on the fold (truly out-of-sample)
3. Aggregate results across all folds

This prevents look-ahead bias and shows how the model would perform
in real trading conditions.

Usage:
    python -m ml.scripts.walk_forward_validation \
        --symbol XRP/USDT \
        --days 365 \
        --n-folds 5 \
        --db-url "postgresql://..."
"""

import os
import sys
import argparse
import asyncio
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd

# Set GPU environment before importing torch
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '10.3.0')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation."""
    db_url: str
    symbol: str
    interval_minutes: int = 1
    days: int = 365
    n_folds: int = 5
    min_train_days: int = 60  # Minimum training data

    # Label generation
    threshold_pct: float = 0.5
    lookahead: int = 10

    # Feature enrichment
    enrich_regime: bool = True

    # Model parameters (v2.1)
    use_class_weights: bool = True
    n_estimators: int = 300
    max_depth: int = 4
    learning_rate: float = 0.05
    min_child_weight: int = 10
    subsample: float = 0.7
    colsample_bytree: float = 0.7
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    early_stopping_rounds: int = 30

    # Backtest parameters
    confidence_threshold: float = 0.4
    position_size_pct: float = 0.1
    stop_loss_pct: float = 2.0
    take_profit_pct: float = 4.0


@dataclass
class FoldResult:
    """Results from a single fold."""
    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_samples: int
    test_samples: int

    # Classification
    train_accuracy: float
    test_accuracy: float

    # Signal precision
    buy_precision: float
    sell_precision: float
    buy_count: int
    sell_count: int
    hold_bias: float

    # Trading
    total_return: float
    sharpe_ratio: float
    win_rate: float
    num_trades: int
    max_drawdown: float


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward results."""
    config: WalkForwardConfig
    fold_results: List[FoldResult]

    # Aggregated metrics
    avg_test_accuracy: float
    std_test_accuracy: float
    avg_return: float
    std_return: float
    cumulative_return: float
    avg_sharpe: float
    avg_win_rate: float
    total_trades: int
    avg_buy_precision: float
    avg_sell_precision: float

    def print_summary(self):
        """Print formatted summary."""
        print("\n" + "=" * 70)
        print("WALK-FORWARD VALIDATION RESULTS")
        print("=" * 70)

        print(f"\nConfiguration:")
        print(f"  Symbol: {self.config.symbol}")
        print(f"  Data: {self.config.days} days, {self.config.n_folds} folds")
        print(f"  Class Weights: {self.config.use_class_weights}")
        print(f"  Confidence Threshold: {self.config.confidence_threshold}")

        print(f"\n{'Fold':<6} {'Train':<12} {'Test':<12} {'Accuracy':<10} {'Return':<10} {'Sharpe':<8} {'Trades':<8} {'WinRate':<8}")
        print("-" * 70)

        for r in self.fold_results:
            print(f"{r.fold:<6} {r.train_samples:<12,} {r.test_samples:<12,} "
                  f"{r.test_accuracy*100:<10.1f} {r.total_return:<10.2f} "
                  f"{r.sharpe_ratio:<8.2f} {r.num_trades:<8} {r.win_rate:<8.1f}")

        print("-" * 70)
        print(f"\nAggregated Results (Out-of-Sample):")
        print(f"  Avg Test Accuracy: {self.avg_test_accuracy*100:.1f}% (+/- {self.std_test_accuracy*100:.1f}%)")
        print(f"  Avg Return per Fold: {self.avg_return:.2f}% (+/- {self.std_return:.2f}%)")
        print(f"  Cumulative Return: {self.cumulative_return:.2f}%")
        print(f"  Avg Sharpe Ratio: {self.avg_sharpe:.2f}")
        print(f"  Avg Win Rate: {self.avg_win_rate:.1f}%")
        print(f"  Total Trades: {self.total_trades}")
        print(f"  Avg Buy Precision: {self.avg_buy_precision*100:.1f}%")
        print(f"  Avg Sell Precision: {self.avg_sell_precision*100:.1f}%")

        # Verdict
        print(f"\n{'='*70}")
        if self.cumulative_return > 0 and self.avg_sharpe > 0.5:
            print("VERDICT: Model shows POSITIVE out-of-sample performance")
        elif self.cumulative_return > 0:
            print("VERDICT: Model is MARGINALLY profitable (low Sharpe)")
        else:
            print("VERDICT: Model is NOT profitable out-of-sample")
        print("=" * 70)


class WalkForwardValidator:
    """
    Walk-forward validation for ML trading models.

    Uses expanding window: each fold trains on ALL prior data,
    ensuring truly out-of-sample testing.
    """

    def __init__(self, config: WalkForwardConfig):
        self.config = config
        self.pool = None

    async def connect(self):
        """Connect to database."""
        import asyncpg
        self.pool = await asyncpg.create_pool(
            self.config.db_url,
            min_size=2,
            max_size=10
        )
        logger.info("Connected to database")

    async def close(self):
        """Close database connection."""
        if self.pool:
            await self.pool.close()

    async def load_data(self) -> pd.DataFrame:
        """Load candle data from database."""
        logger.info(f"Loading {self.config.days} days of {self.config.symbol} data")

        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=self.config.days)

        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM candles
            WHERE symbol = $1
              AND interval_minutes = $2
              AND timestamp >= $3
              AND timestamp < $4
            ORDER BY timestamp ASC
        """

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                query,
                self.config.symbol,
                self.config.interval_minutes,
                start_time,
                end_time
            )

        if not rows:
            raise ValueError(f"No data found for {self.config.symbol}")

        df = pd.DataFrame([dict(row) for row in rows])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        logger.info(f"Loaded {len(df):,} candles")
        return df

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from OHLCV data."""
        from ml.features.extractor import FeatureExtractor

        extractor = FeatureExtractor()
        features_df = extractor.extract_features(df)

        # Add regime features if enabled
        if self.config.enrich_regime:
            from ml.features.regime_detection import RegimeDetector

            detector = RegimeDetector()
            regime_df = detector.detect_regime_series(
                features_df,
                close_col='close',
                high_col='high',
                low_col='low',
                volume_col='volume'
            )

            features_df['regime'] = regime_df['regime']
            features_df['regime_confidence'] = regime_df['regime_confidence']
            features_df['regime_volatility'] = regime_df['regime_volatility']
            features_df['regime_trend_strength'] = regime_df['regime_trend_strength']

            for regime_type in ['trending_up', 'trending_down', 'ranging', 'volatile', 'breakout']:
                features_df[f'regime_{regime_type}'] = (
                    features_df['regime'] == regime_type
                ).astype(float)

            features_df = features_df.drop(columns=['regime'])

        # Fill NaN defaults
        fill_defaults = {
            'regime_confidence': 0.0,
            'regime_volatility': 0.5,
            'regime_trend_strength': 0.0,
            'regime_trending_up': 0.0,
            'regime_trending_down': 0.0,
            'regime_ranging': 1.0,
            'regime_volatile': 0.0,
            'regime_breakout': 0.0,
        }
        features_df.fillna(fill_defaults, inplace=True)

        return features_df

    def generate_labels(self, df: pd.DataFrame) -> np.ndarray:
        """Generate classification labels."""
        from ml.features.labels import generate_classification_labels

        labeled_df = generate_classification_labels(
            df=df,
            future_bars=self.config.lookahead,
            threshold_pct=self.config.threshold_pct,
            price_col='close'
        )

        return labeled_df['label_class'].values

    def prepare_features(self, features: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Prepare feature matrix."""
        drop_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                     'rsi_zone', 'session']
        feature_cols = [c for c in features.columns
                       if c not in drop_cols and features[c].dtype in [np.float64, np.int64, np.float32, np.int32]]

        return features[feature_cols].values, feature_cols

    def train_and_evaluate_fold(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        prices_test: np.ndarray,
        test_days: int
    ) -> Dict[str, Any]:
        """Train model and evaluate on one fold."""
        from ml.models.classifier import XGBoostClassifier
        from ml.evaluation.metrics import calculate_metrics, calculate_signal_precision_metrics
        from ml.evaluation.backtest import backtest_model, BacktestConfig

        # Train model
        model = XGBoostClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            min_child_weight=self.config.min_child_weight,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            reg_alpha=self.config.reg_alpha,
            reg_lambda=self.config.reg_lambda,
            early_stopping_rounds=self.config.early_stopping_rounds,
            device='cpu',
            use_class_weights=self.config.use_class_weights
        )

        # Split some training data for validation (for early stopping)
        val_split = int(len(X_train) * 0.9)
        X_tr, X_val = X_train[:val_split], X_train[val_split:]
        y_tr, y_val = y_train[:val_split], y_train[val_split:]

        train_metrics = model.fit(X_tr, y_tr, X_val, y_val, verbose=False)

        # Evaluate on test
        predictions = model.predict(X_test)
        probs = model.predict_proba(X_test)
        confidences = probs.max(axis=1)

        class_metrics = calculate_metrics(y_test, predictions)
        signal_metrics = calculate_signal_precision_metrics(
            y_test, predictions, confidences,
            confidence_threshold=self.config.confidence_threshold
        )

        # Backtest
        backtest_result = backtest_model(
            model=model,
            features=X_test,
            prices=prices_test,
            config=BacktestConfig(
                initial_capital=1000.0,
                position_size_pct=self.config.position_size_pct,
                confidence_threshold=self.config.confidence_threshold,
                stop_loss_pct=self.config.stop_loss_pct,
                take_profit_pct=self.config.take_profit_pct,
                trading_days=test_days
            )
        )

        return {
            'train_accuracy': train_metrics['train_accuracy'],
            'test_accuracy': class_metrics['accuracy'],
            'buy_precision': signal_metrics.buy_precision,
            'sell_precision': signal_metrics.sell_precision,
            'buy_count': signal_metrics.buy_count,
            'sell_count': signal_metrics.sell_count,
            'hold_bias': signal_metrics.hold_bias,
            'total_return': backtest_result.metrics.total_return,
            'sharpe_ratio': backtest_result.metrics.sharpe_ratio,
            'win_rate': backtest_result.metrics.win_rate,
            'num_trades': backtest_result.metrics.num_trades,
            'max_drawdown': backtest_result.metrics.max_drawdown
        }

    async def run(self) -> WalkForwardResult:
        """Run walk-forward validation."""
        logger.info("=" * 60)
        logger.info("STARTING WALK-FORWARD VALIDATION")
        logger.info("=" * 60)

        await self.connect()

        try:
            # Load and prepare data
            df = await self.load_data()
            features_df = self.extract_features(df)
            labels = self.generate_labels(df)
            X, feature_cols = self.prepare_features(features_df)
            prices = features_df['close'].values
            timestamps = features_df['timestamp'].values

            # Remove NaN rows
            valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(labels)
            X = X[valid_mask]
            labels = labels[valid_mask]
            prices = prices[valid_mask]
            timestamps = timestamps[valid_mask]

            logger.info(f"Total samples after cleaning: {len(X):,}")
            logger.info(f"Features: {len(feature_cols)}")

            # Calculate fold boundaries
            n = len(X)
            fold_size = n // self.config.n_folds
            min_train_samples = self.config.min_train_days * 24 * 60  # 1-min bars
            gap = self.config.lookahead

            fold_results = []

            for fold in range(self.config.n_folds):
                # Test period for this fold
                test_start = fold * fold_size
                test_end = (fold + 1) * fold_size if fold < self.config.n_folds - 1 else n - gap

                # Training uses all data before test period (with gap)
                train_end = test_start - gap

                if train_end < min_train_samples:
                    logger.info(f"Fold {fold + 1}: Skipping (insufficient training data)")
                    continue

                # Prepare fold data
                X_train = X[:train_end]
                y_train = labels[:train_end]
                X_test = X[test_start:test_end]
                y_test = labels[test_start:test_end]
                prices_test = prices[test_start:test_end]

                # Calculate test days for this fold
                test_days = max(1, (test_end - test_start) // (24 * 60))

                logger.info(f"\nFold {fold + 1}/{self.config.n_folds}:")
                logger.info(f"  Train: {len(X_train):,} samples (0 to {train_end})")
                logger.info(f"  Test: {len(X_test):,} samples ({test_start} to {test_end})")
                logger.info(f"  Test days: {test_days}")

                # Train and evaluate
                metrics = self.train_and_evaluate_fold(
                    X_train, y_train, X_test, y_test, prices_test, test_days
                )

                # Record results
                result = FoldResult(
                    fold=fold + 1,
                    train_start=str(timestamps[0]),
                    train_end=str(timestamps[train_end - 1]),
                    test_start=str(timestamps[test_start]),
                    test_end=str(timestamps[test_end - 1]),
                    train_samples=len(X_train),
                    test_samples=len(X_test),
                    train_accuracy=metrics['train_accuracy'],
                    test_accuracy=metrics['test_accuracy'],
                    buy_precision=metrics['buy_precision'],
                    sell_precision=metrics['sell_precision'],
                    buy_count=metrics['buy_count'],
                    sell_count=metrics['sell_count'],
                    hold_bias=metrics['hold_bias'],
                    total_return=metrics['total_return'],
                    sharpe_ratio=metrics['sharpe_ratio'],
                    win_rate=metrics['win_rate'],
                    num_trades=metrics['num_trades'],
                    max_drawdown=metrics['max_drawdown']
                )
                fold_results.append(result)

                logger.info(f"  Accuracy: {metrics['test_accuracy']*100:.1f}%, "
                           f"Return: {metrics['total_return']:.2f}%, "
                           f"Sharpe: {metrics['sharpe_ratio']:.2f}, "
                           f"Trades: {metrics['num_trades']}")

            # Aggregate results
            if not fold_results:
                raise ValueError("No valid folds completed")

            returns = [r.total_return for r in fold_results]
            accuracies = [r.test_accuracy for r in fold_results]
            sharpes = [r.sharpe_ratio for r in fold_results]
            win_rates = [r.win_rate for r in fold_results]
            buy_precs = [r.buy_precision for r in fold_results]
            sell_precs = [r.sell_precision for r in fold_results]

            # Cumulative return (compounded)
            cumulative = 1.0
            for r in returns:
                cumulative *= (1 + r / 100)
            cumulative_return = (cumulative - 1) * 100

            result = WalkForwardResult(
                config=self.config,
                fold_results=fold_results,
                avg_test_accuracy=np.mean(accuracies),
                std_test_accuracy=np.std(accuracies),
                avg_return=np.mean(returns),
                std_return=np.std(returns),
                cumulative_return=cumulative_return,
                avg_sharpe=np.mean(sharpes),
                avg_win_rate=np.mean(win_rates),
                total_trades=sum(r.num_trades for r in fold_results),
                avg_buy_precision=np.mean(buy_precs),
                avg_sell_precision=np.mean(sell_precs)
            )

            return result

        finally:
            await self.close()


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Walk-Forward Validation for ML Trading Models',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--db-url', type=str, required=True,
                        help='PostgreSQL connection URL')
    parser.add_argument('--symbol', type=str, default='XRP/USDT',
                        help='Trading symbol')
    parser.add_argument('--days', type=int, default=365,
                        help='Days of history')
    parser.add_argument('--n-folds', type=int, default=5,
                        help='Number of test folds')
    parser.add_argument('--min-train-days', type=int, default=60,
                        help='Minimum training days per fold')
    parser.add_argument('--confidence-threshold', type=float, default=0.4,
                        help='Confidence threshold for trades')
    parser.add_argument('--no-class-weights', action='store_true',
                        help='Disable class weights')
    parser.add_argument('--no-regime', action='store_true',
                        help='Disable regime features')

    args = parser.parse_args()

    config = WalkForwardConfig(
        db_url=args.db_url,
        symbol=args.symbol,
        days=args.days,
        n_folds=args.n_folds,
        min_train_days=args.min_train_days,
        confidence_threshold=args.confidence_threshold,
        use_class_weights=not args.no_class_weights,
        enrich_regime=not args.no_regime
    )

    validator = WalkForwardValidator(config)
    result = await validator.run()

    result.print_summary()


if __name__ == '__main__':
    asyncio.run(main())
