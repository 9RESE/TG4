#!/usr/bin/env python3
"""
Automatic Retraining Pipeline

Automatically retrains ML models using the latest data from TimescaleDB.
Enriches features with order flow data and multi-timeframe analysis from
the historic database tables.

Features:
- Automatic data loading from TimescaleDB
- Order flow feature enrichment (VPIN, trade imbalance)
- Multi-timeframe feature enrichment (1m, 5m, 15m, 1h, 4h)
- Model version control via ModelRegistry
- Performance comparison with previous models
- Automatic deployment if performance improves
- Results saved to backtest_runs table

Usage:
    # Retrain with default settings
    python -m ml.scripts.retrain_pipeline --symbol XRP/USDT

    # Full retraining with all features
    python -m ml.scripts.retrain_pipeline --symbol XRP/USDT --enrich-order-flow --enrich-mtf

    # Auto-deploy if better than current
    python -m ml.scripts.retrain_pipeline --symbol XRP/USDT --auto-deploy

    # Scheduled daily retraining (cron example):
    # 0 2 * * * /path/to/python -m ml.scripts.retrain_pipeline --auto-deploy
"""

import os
import sys
import argparse
import asyncio
import json
import uuid
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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class RetrainingConfig:
    """Configuration for retraining pipeline."""
    db_url: str
    symbol: str
    interval_minutes: int = 1
    days: int = 365  # Default to 1 year for better generalization
    threshold_pct: float = 0.5
    lookahead: int = 10

    # Feature enrichment
    enrich_order_flow: bool = True
    order_flow_lookback_minutes: int = 60
    vpin_buckets: int = 50

    enrich_mtf: bool = True
    mtf_intervals: List[int] = None

    # Training - regularization to prevent overfitting
    train_pct: float = 0.7
    val_pct: float = 0.15
    n_estimators: int = 300  # More trees with early stopping
    max_depth: int = 4  # Shallower trees to reduce overfitting
    learning_rate: float = 0.05  # Slower learning
    min_child_weight: int = 10  # Regularization
    subsample: float = 0.7  # Use 70% of data per tree
    colsample_bytree: float = 0.7  # Use 70% of features per tree
    reg_alpha: float = 0.1  # L1 regularization
    reg_lambda: float = 1.0  # L2 regularization
    early_stopping_rounds: int = 30  # Stop if no improvement

    # Deployment
    auto_deploy: bool = False
    min_improvement_pct: float = 5.0  # Min improvement to auto-deploy

    # Output
    registry_path: str = "models/registry"
    model_name: str = "signal_classifier"

    def __post_init__(self):
        if self.mtf_intervals is None:
            self.mtf_intervals = [1, 5, 15, 60, 240]


@dataclass
class RetrainingResult:
    """Results from retraining."""
    model_version: str
    training_samples: int
    training_time_seconds: float

    # Metrics
    train_accuracy: float
    val_accuracy: float
    test_accuracy: float
    precision: float
    recall: float
    f1_score: float

    # Trading metrics
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    num_trades: int

    # Comparison
    previous_version: Optional[str] = None
    improvement_pct: Optional[float] = None
    deployed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RetrainingPipeline:
    """
    Automatic retraining pipeline for ML models.

    Connects to TimescaleDB, loads latest data, enriches with
    order flow and multi-timeframe features, trains models,
    and manages model versions.
    """

    def __init__(self, config: RetrainingConfig):
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

    async def load_candles(self) -> pd.DataFrame:
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

    async def extract_and_enrich_features(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """Extract base features and enrich with DB-backed features."""
        from ml.features.extractor import FeatureExtractor

        logger.info("Extracting base features")
        extractor = FeatureExtractor()
        features_df = extractor.extract_features(df)

        # Enrich with order flow features
        if self.config.enrich_order_flow:
            logger.info("Enriching with order flow features from trades table")
            from ml.features.order_flow_features import OrderFlowFeatureProvider

            provider = OrderFlowFeatureProvider(self.config.db_url)
            try:
                await provider.connect()
                timestamps = features_df['timestamp'].tolist()

                of_df = await provider.compute_features_for_candles(
                    self.config.symbol,
                    timestamps,
                    self.config.order_flow_lookback_minutes,
                    self.config.vpin_buckets
                )

                # Replace placeholder columns
                for col in ['trade_imbalance', 'vpin']:
                    if col in features_df.columns:
                        features_df = features_df.drop(columns=[col])

                # Merge
                features_df = features_df.merge(
                    of_df.reset_index(),
                    on='timestamp',
                    how='left'
                )

                logger.info(f"Added {len(of_df.columns)} order flow features")
            finally:
                await provider.close()

        # Enrich with multi-timeframe features
        if self.config.enrich_mtf:
            logger.info("Enriching with multi-timeframe features")
            from ml.features.multi_timeframe import MultiTimeframeFeatureProvider

            provider = MultiTimeframeFeatureProvider(self.config.db_url)
            try:
                await provider.connect()
                timestamps = features_df['timestamp'].tolist()

                mtf_df = await provider.compute_features_for_candles(
                    self.config.symbol,
                    timestamps,
                    self.config.mtf_intervals
                )

                # Merge
                features_df = features_df.merge(
                    mtf_df.reset_index(),
                    on='timestamp',
                    how='left'
                )

                logger.info(f"Added {len(mtf_df.columns)} MTF features")
            finally:
                await provider.close()

        # Fill NaN values
        fill_defaults = {
            'trade_imbalance': 0.0,
            'vpin': 0.5,
            'order_flow_toxicity': 0.5,
            'buy_sell_ratio': 1.0,
            'trade_intensity': 0.0,
            'avg_trade_size': 0.0,
            'large_trade_ratio': 0.0,
            'trade_count': 0,
            'mtf_trend_alignment': 0.0,
            'tf_divergence_score': 0.0,
            'multi_resolution_volatility': 0.0,
            'dominant_trend': 0.0,
            'momentum_confluence': 0.0
        }
        features_df.fillna(fill_defaults, inplace=True)

        logger.info(f"Total features: {len(features_df.columns)}")
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

    def prepare_data(
        self,
        features: pd.DataFrame,
        labels: np.ndarray
    ) -> Tuple[np.ndarray, ...]:
        """Prepare train/val/test splits with gap buffer to prevent data leakage.

        IMPORTANT: Labels are generated using future price data (lookahead).
        To prevent leakage, we add a gap of `lookahead` bars between splits
        so training labels don't depend on validation/test prices.
        """

        # Drop non-feature columns
        drop_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                     'rsi_zone', 'session']
        feature_cols = [c for c in features.columns
                       if c not in drop_cols and features[c].dtype in [np.float64, np.int64, np.float32, np.int32]]

        features_only = features[feature_cols]

        # Remove NaN rows
        valid_mask = ~features_only.isna().any(axis=1) & ~np.isnan(labels)
        features_clean = features_only[valid_mask].values
        labels_clean = labels[valid_mask]
        prices_clean = features.loc[valid_mask, 'close'].values

        # Time-ordered split WITH GAP to prevent lookahead bias
        # The gap equals the lookahead period used for label generation
        gap = self.config.lookahead

        n = len(features_clean)
        train_end = int(n * self.config.train_pct)
        val_end = int(n * (self.config.train_pct + self.config.val_pct))

        # Train: [0, train_end - gap) - removes last `gap` rows whose labels use val data
        # Val: [train_end, val_end - gap) - removes last `gap` rows whose labels use test data
        # Test: [val_end, n - gap) - removes last `gap` rows with no future data

        train_actual_end = train_end - gap
        val_actual_end = val_end - gap
        test_actual_end = n - gap  # Labels need future_bars of data after

        X_train = features_clean[:train_actual_end]
        X_val = features_clean[train_end:val_actual_end]
        X_test = features_clean[val_end:test_actual_end]

        y_train = labels_clean[:train_actual_end]
        y_val = labels_clean[train_end:val_actual_end]
        y_test = labels_clean[val_end:test_actual_end]

        prices_test = prices_clean[val_end:test_actual_end]

        logger.info(f"Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")
        logger.info(f"Gap buffer: {gap} bars to prevent lookahead bias")

        return (X_train, X_val, X_test, y_train, y_val, y_test,
                prices_test, feature_cols)

    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ):
        """Train XGBoost model with regularization to prevent overfitting."""
        from ml.models.classifier import XGBoostClassifier

        logger.info("Training XGBoost model with regularization")
        logger.info(f"  max_depth={self.config.max_depth}, "
                   f"learning_rate={self.config.learning_rate}, "
                   f"early_stopping={self.config.early_stopping_rounds}")

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
            device='cpu'  # Fallback to CPU if CUDA not available
        )

        metrics = model.fit(X_train, y_train, X_val, y_val, verbose=True)

        logger.info(f"Train accuracy: {metrics['train_accuracy']*100:.1f}%")
        logger.info(f"Val accuracy: {metrics['val_accuracy']*100:.1f}%")

        # Check for overfitting
        overfit_gap = metrics['train_accuracy'] - metrics['val_accuracy']
        if overfit_gap > 0.15:  # >15% gap
            logger.warning(f"Potential overfitting detected: {overfit_gap*100:.1f}% gap")

        return model, metrics

    def evaluate_model(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        prices_test: np.ndarray
    ) -> Dict[str, Any]:
        """Evaluate model on test set."""
        from ml.evaluation.metrics import calculate_metrics
        from ml.evaluation.backtest import backtest_model, BacktestConfig

        logger.info("Evaluating on test set")

        predictions = model.predict(X_test)
        metrics = calculate_metrics(y_test, predictions)

        # Run backtest with correct trading days for annualization
        # Test set is typically 20% of total days
        test_days = max(1, int(self.config.days * 0.2))
        backtest_result = backtest_model(
            model=model,
            features=X_test,
            prices=prices_test,
            config=BacktestConfig(
                initial_capital=1000.0,
                position_size_pct=0.1,
                confidence_threshold=0.6,
                stop_loss_pct=2.0,
                take_profit_pct=4.0,
                trading_days=test_days
            )
        )

        return {
            'classification': metrics,
            'backtest': backtest_result.metrics
        }

    async def save_to_backtest_runs(
        self,
        result: RetrainingResult,
        config: Dict[str, Any]
    ):
        """Save training run to backtest_runs table."""
        logger.info("Saving results to backtest_runs table")

        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO backtest_runs (
                    strategy_name, symbols, start_time, end_time,
                    parameters, metrics, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
                f"ml_retrain_{self.config.model_name}",
                [self.config.symbol],
                datetime.now(timezone.utc) - timedelta(days=self.config.days),
                datetime.now(timezone.utc),
                json.dumps(config),
                json.dumps(result.to_dict()),
                datetime.now(timezone.utc)
            )

    async def run(self) -> RetrainingResult:
        """Run the full retraining pipeline."""
        import time
        start_time = time.time()

        logger.info("=" * 60)
        logger.info("STARTING RETRAINING PIPELINE")
        logger.info("=" * 60)

        await self.connect()

        try:
            # Load data
            df = await self.load_candles()

            # Extract and enrich features
            features = await self.extract_and_enrich_features(df)

            # Generate labels
            labels = self.generate_labels(df)

            # Prepare data
            (X_train, X_val, X_test, y_train, y_val, y_test,
             prices_test, feature_cols) = self.prepare_data(features, labels)

            # Train model
            model, train_metrics = self.train_model(
                X_train, y_train, X_val, y_val
            )

            # Evaluate
            eval_metrics = self.evaluate_model(
                model, X_test, y_test, prices_test
            )

            training_time = time.time() - start_time

            # Register model
            from ml.integration.registry import ModelRegistry

            registry = ModelRegistry(self.config.registry_path)
            previous_version = registry.get_deployed_version(self.config.model_name)

            model_info = registry.register(
                model=model,
                name=self.config.model_name,
                model_type='xgboost',
                metrics={
                    'train_accuracy': train_metrics['train_accuracy'],
                    'val_accuracy': train_metrics['val_accuracy'],
                    'test_accuracy': eval_metrics['classification']['accuracy'],
                    'precision': eval_metrics['classification']['precision'],
                    'recall': eval_metrics['classification']['recall'],
                    'f1_score': eval_metrics['classification']['f1_score'],
                    'total_return': eval_metrics['backtest'].total_return,
                    'sharpe_ratio': eval_metrics['backtest'].sharpe_ratio,
                },
                config={
                    'symbol': self.config.symbol,
                    'days': self.config.days,
                    'enrich_order_flow': self.config.enrich_order_flow,
                    'enrich_mtf': self.config.enrich_mtf,
                    'n_estimators': self.config.n_estimators,
                    'max_depth': self.config.max_depth,
                },
                training_features=list(feature_cols),
                description=f"Retrained on {self.config.days} days, "
                           f"order_flow={self.config.enrich_order_flow}, "
                           f"mtf={self.config.enrich_mtf}",
                tags=['retrained', 'automated']
            )

            logger.info(f"Registered model version: {model_info.version}")

            # Compare with previous
            improvement_pct = None
            if previous_version:
                prev_info = registry.get_model_info(
                    self.config.model_name, previous_version
                )
                if prev_info and 'test_accuracy' in prev_info.metrics:
                    prev_acc = prev_info.metrics['test_accuracy']
                    curr_acc = eval_metrics['classification']['accuracy']
                    improvement_pct = (curr_acc - prev_acc) / prev_acc * 100
                    logger.info(f"Improvement over {previous_version}: {improvement_pct:.1f}%")

            # Auto-deploy if improved
            deployed = False
            if self.config.auto_deploy:
                should_deploy = (
                    improvement_pct is None or
                    improvement_pct >= self.config.min_improvement_pct
                )
                if should_deploy:
                    registry.deploy(self.config.model_name, model_info.version)
                    deployed = True
                    logger.info(f"Auto-deployed version {model_info.version}")
                else:
                    logger.info(f"Not deploying: improvement {improvement_pct:.1f}% "
                              f"< threshold {self.config.min_improvement_pct}%")

            # Build result
            result = RetrainingResult(
                model_version=model_info.version,
                training_samples=len(X_train),
                training_time_seconds=training_time,
                train_accuracy=train_metrics['train_accuracy'],
                val_accuracy=train_metrics['val_accuracy'],
                test_accuracy=eval_metrics['classification']['accuracy'],
                precision=eval_metrics['classification']['precision'],
                recall=eval_metrics['classification']['recall'],
                f1_score=eval_metrics['classification']['f1_score'],
                total_return=eval_metrics['backtest'].total_return,
                sharpe_ratio=eval_metrics['backtest'].sharpe_ratio,
                max_drawdown=eval_metrics['backtest'].max_drawdown,
                win_rate=eval_metrics['backtest'].win_rate,
                num_trades=eval_metrics['backtest'].num_trades,
                previous_version=previous_version,
                improvement_pct=improvement_pct,
                deployed=deployed
            )

            # Save to database
            await self.save_to_backtest_runs(result, asdict(self.config))

            logger.info("=" * 60)
            logger.info("RETRAINING COMPLETE")
            logger.info("=" * 60)
            logger.info(f"Model: {self.config.model_name}:{model_info.version}")
            logger.info(f"Test accuracy: {result.test_accuracy*100:.1f}%")
            logger.info(f"Total return: {result.total_return:.2f}%")
            logger.info(f"Sharpe ratio: {result.sharpe_ratio:.2f}")
            logger.info(f"Deployed: {result.deployed}")

            return result

        finally:
            await self.close()


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Automatic ML Model Retraining Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--db-url', type=str,
        default=os.getenv('DATABASE_URL'),
        help='PostgreSQL connection URL'
    )
    parser.add_argument(
        '--symbol', type=str, default='XRP/USDT',
        help='Trading symbol'
    )
    parser.add_argument(
        '--days', type=int, default=365,
        help='Days of history to use (default: 365 for better generalization)'
    )
    parser.add_argument(
        '--threshold', type=float, default=0.5,
        help='Label threshold percentage'
    )
    parser.add_argument(
        '--lookahead', type=int, default=10,
        help='Label lookahead bars'
    )
    parser.add_argument(
        '--enrich-order-flow', action='store_true',
        help='Enrich with order flow features from trades table'
    )
    parser.add_argument(
        '--enrich-mtf', action='store_true',
        help='Enrich with multi-timeframe features'
    )
    parser.add_argument(
        '--auto-deploy', action='store_true',
        help='Auto-deploy if performance improves'
    )
    parser.add_argument(
        '--min-improvement', type=float, default=5.0,
        help='Minimum improvement %% for auto-deploy'
    )
    parser.add_argument(
        '--registry-path', type=str, default='models/registry',
        help='Model registry path'
    )
    parser.add_argument(
        '--model-name', type=str, default='signal_classifier',
        help='Model name in registry'
    )

    args = parser.parse_args()

    if not args.db_url:
        print("ERROR: DATABASE_URL required")
        sys.exit(1)

    config = RetrainingConfig(
        db_url=args.db_url,
        symbol=args.symbol,
        days=args.days,
        threshold_pct=args.threshold,
        lookahead=args.lookahead,
        enrich_order_flow=args.enrich_order_flow,
        enrich_mtf=args.enrich_mtf,
        auto_deploy=args.auto_deploy,
        min_improvement_pct=args.min_improvement,
        registry_path=args.registry_path,
        model_name=args.model_name
    )

    pipeline = RetrainingPipeline(config)
    result = await pipeline.run()

    # Print summary
    print("\n" + "=" * 60)
    print("RETRAINING SUMMARY")
    print("=" * 60)
    print(f"Model Version: {result.model_version}")
    print(f"Training Time: {result.training_time_seconds:.1f}s")
    print(f"Training Samples: {result.training_samples:,}")
    print(f"Test Accuracy: {result.test_accuracy*100:.1f}%")
    print(f"F1 Score: {result.f1_score*100:.1f}%")
    print(f"Total Return: {result.total_return:.2f}%")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"Win Rate: {result.win_rate:.1f}%")
    print(f"Deployed: {result.deployed}")
    if result.improvement_pct is not None:
        print(f"Improvement: {result.improvement_pct:.1f}%")
    print("=" * 60)


if __name__ == '__main__':
    asyncio.run(main())
