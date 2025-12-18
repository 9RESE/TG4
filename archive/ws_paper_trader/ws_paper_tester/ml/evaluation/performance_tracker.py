"""
ML Model Performance Tracker

Tracks ML model performance using the backtest_runs table in TimescaleDB.
Provides analytics, comparison, and monitoring capabilities.

Features:
- Save model performance after training/backtesting
- Compare model versions over time
- Monitor deployed model performance
- Generate performance reports
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd

try:
    import asyncpg
except ImportError:
    asyncpg = None

logger = logging.getLogger(__name__)


@dataclass
class ModelPerformanceRecord:
    """Record of model performance for a specific run."""
    model_name: str
    model_version: str
    symbols: List[str]
    start_time: datetime
    end_time: datetime

    # Classification metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float

    # Trading metrics
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    num_trades: int
    profit_factor: float

    # Additional info
    training_samples: int = 0
    feature_count: int = 0
    enriched_order_flow: bool = False
    enriched_mtf: bool = False

    # Metadata
    config: Dict[str, Any] = None
    run_type: str = 'backtest'  # 'backtest', 'live', 'paper'

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PerformanceTracker:
    """
    Tracks ML model performance in TimescaleDB.

    Provides methods to:
    - Save performance records to backtest_runs table
    - Query historical performance
    - Compare model versions
    - Generate analytics
    """

    def __init__(self, db_url: str):
        if asyncpg is None:
            raise ImportError("asyncpg is required")

        self.db_url = db_url
        self.pool: Optional[asyncpg.Pool] = None

    async def connect(self):
        """Connect to database."""
        self.pool = await asyncpg.create_pool(
            self.db_url,
            min_size=2,
            max_size=5
        )
        logger.info("PerformanceTracker connected")

    async def close(self):
        """Close database connection."""
        if self.pool:
            await self.pool.close()

    async def save_performance(
        self,
        record: ModelPerformanceRecord
    ) -> str:
        """
        Save performance record to backtest_runs table.

        Returns:
            Run ID (UUID)
        """
        if not self.pool:
            raise RuntimeError("Not connected")

        run_id = str(uuid.uuid4())

        # Prepare parameters
        strategy_name = f"ml:{record.model_name}:{record.model_version}"

        parameters = {
            'model_name': record.model_name,
            'model_version': record.model_version,
            'training_samples': record.training_samples,
            'feature_count': record.feature_count,
            'enriched_order_flow': record.enriched_order_flow,
            'enriched_mtf': record.enriched_mtf,
            'run_type': record.run_type,
            **(record.config or {})
        }

        metrics = {
            'accuracy': record.accuracy,
            'precision': record.precision,
            'recall': record.recall,
            'f1_score': record.f1_score,
            'total_return': record.total_return,
            'sharpe_ratio': record.sharpe_ratio,
            'max_drawdown': record.max_drawdown,
            'win_rate': record.win_rate,
            'num_trades': record.num_trades,
            'profit_factor': record.profit_factor,
        }

        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO backtest_runs (
                    id, strategy_name, symbols, start_time, end_time,
                    parameters, metrics, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
                uuid.UUID(run_id),
                strategy_name,
                record.symbols,
                record.start_time,
                record.end_time,
                json.dumps(parameters),
                json.dumps(metrics),
                datetime.now(timezone.utc)
            )

        logger.info(f"Saved performance record: {run_id}")
        return run_id

    async def get_model_history(
        self,
        model_name: str,
        model_version: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get performance history for a model.

        Args:
            model_name: Name of the model
            model_version: Specific version (None = all versions)
            limit: Maximum records to return

        Returns:
            List of performance records
        """
        if not self.pool:
            raise RuntimeError("Not connected")

        if model_version:
            pattern = f"ml:{model_name}:{model_version}"
        else:
            pattern = f"ml:{model_name}:%"

        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT id, strategy_name, symbols, start_time, end_time,
                       parameters, metrics, created_at
                FROM backtest_runs
                WHERE strategy_name LIKE $1
                ORDER BY created_at DESC
                LIMIT $2
            """, pattern, limit)

        results = []
        for row in rows:
            results.append({
                'id': str(row['id']),
                'strategy_name': row['strategy_name'],
                'symbols': row['symbols'],
                'start_time': row['start_time'],
                'end_time': row['end_time'],
                'parameters': json.loads(row['parameters']) if row['parameters'] else {},
                'metrics': json.loads(row['metrics']) if row['metrics'] else {},
                'created_at': row['created_at'],
            })

        return results

    async def compare_versions(
        self,
        model_name: str,
        versions: List[str],
        metric: str = 'total_return'
    ) -> pd.DataFrame:
        """
        Compare performance across model versions.

        Args:
            model_name: Model name
            versions: List of versions to compare
            metric: Metric to compare

        Returns:
            DataFrame with comparison
        """
        results = []

        for version in versions:
            history = await self.get_model_history(model_name, version, limit=10)

            if history:
                metrics = [h['metrics'].get(metric, 0) for h in history]
                results.append({
                    'version': version,
                    'mean': np.mean(metrics),
                    'std': np.std(metrics),
                    'min': np.min(metrics),
                    'max': np.max(metrics),
                    'runs': len(history)
                })

        return pd.DataFrame(results)

    async def get_best_model(
        self,
        model_name: str,
        metric: str = 'sharpe_ratio',
        min_trades: int = 10
    ) -> Optional[Dict[str, Any]]:
        """
        Get the best performing model version.

        Args:
            model_name: Model name
            metric: Metric to optimize
            min_trades: Minimum trades required

        Returns:
            Best model info or None
        """
        if not self.pool:
            raise RuntimeError("Not connected")

        pattern = f"ml:{model_name}:%"

        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT strategy_name, parameters, metrics, created_at
                FROM backtest_runs
                WHERE strategy_name LIKE $1
                ORDER BY created_at DESC
                LIMIT 100
            """, pattern)

        best = None
        best_value = float('-inf')

        for row in rows:
            metrics = json.loads(row['metrics']) if row['metrics'] else {}
            num_trades = metrics.get('num_trades', 0)

            if num_trades < min_trades:
                continue

            value = metrics.get(metric, float('-inf'))

            if value > best_value:
                best_value = value
                params = json.loads(row['parameters']) if row['parameters'] else {}
                best = {
                    'version': params.get('model_version'),
                    'metric': metric,
                    'value': value,
                    'metrics': metrics,
                    'created_at': row['created_at']
                }

        return best

    async def get_performance_trend(
        self,
        model_name: str,
        days: int = 30
    ) -> pd.DataFrame:
        """
        Get performance trend over time.

        Args:
            model_name: Model name
            days: Number of days to look back

        Returns:
            DataFrame with performance over time
        """
        if not self.pool:
            raise RuntimeError("Not connected")

        pattern = f"ml:{model_name}:%"
        start_date = datetime.now(timezone.utc) - timedelta(days=days)

        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT strategy_name, metrics, created_at
                FROM backtest_runs
                WHERE strategy_name LIKE $1
                  AND created_at >= $2
                ORDER BY created_at ASC
            """, pattern, start_date)

        data = []
        for row in rows:
            metrics = json.loads(row['metrics']) if row['metrics'] else {}
            parts = row['strategy_name'].split(':')
            version = parts[2] if len(parts) > 2 else 'unknown'

            data.append({
                'date': row['created_at'].date(),
                'version': version,
                'accuracy': metrics.get('accuracy', 0),
                'total_return': metrics.get('total_return', 0),
                'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                'win_rate': metrics.get('win_rate', 0),
            })

        return pd.DataFrame(data)

    async def generate_report(
        self,
        model_name: str,
        output_format: str = 'dict'
    ) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.

        Args:
            model_name: Model name
            output_format: 'dict' or 'markdown'

        Returns:
            Performance report
        """
        history = await self.get_model_history(model_name, limit=100)

        if not history:
            return {'error': 'No performance data found'}

        # Aggregate metrics
        all_metrics = [h['metrics'] for h in history if h['metrics']]

        accuracies = [m.get('accuracy', 0) for m in all_metrics]
        returns = [m.get('total_return', 0) for m in all_metrics]
        sharpes = [m.get('sharpe_ratio', 0) for m in all_metrics]
        win_rates = [m.get('win_rate', 0) for m in all_metrics]

        # Get version counts
        version_counts = {}
        for h in history:
            params = h.get('parameters', {})
            version = params.get('model_version', 'unknown')
            version_counts[version] = version_counts.get(version, 0) + 1

        # Find best run
        best_run = max(history, key=lambda h: h['metrics'].get('sharpe_ratio', 0))

        report = {
            'model_name': model_name,
            'total_runs': len(history),
            'versions_tested': len(version_counts),
            'version_distribution': version_counts,
            'aggregate_metrics': {
                'accuracy': {
                    'mean': np.mean(accuracies),
                    'std': np.std(accuracies),
                    'min': np.min(accuracies),
                    'max': np.max(accuracies)
                },
                'total_return': {
                    'mean': np.mean(returns),
                    'std': np.std(returns),
                    'min': np.min(returns),
                    'max': np.max(returns)
                },
                'sharpe_ratio': {
                    'mean': np.mean(sharpes),
                    'std': np.std(sharpes),
                    'min': np.min(sharpes),
                    'max': np.max(sharpes)
                },
                'win_rate': {
                    'mean': np.mean(win_rates),
                    'std': np.std(win_rates),
                    'min': np.min(win_rates),
                    'max': np.max(win_rates)
                }
            },
            'best_run': {
                'id': best_run['id'],
                'version': best_run['parameters'].get('model_version'),
                'sharpe_ratio': best_run['metrics'].get('sharpe_ratio'),
                'total_return': best_run['metrics'].get('total_return'),
                'date': str(best_run['created_at'])
            },
            'recent_runs': [
                {
                    'id': h['id'],
                    'version': h['parameters'].get('model_version'),
                    'accuracy': h['metrics'].get('accuracy'),
                    'total_return': h['metrics'].get('total_return'),
                    'sharpe_ratio': h['metrics'].get('sharpe_ratio'),
                    'date': str(h['created_at'])
                }
                for h in history[:10]
            ]
        }

        if output_format == 'markdown':
            return self._format_markdown(report)

        return report

    def _format_markdown(self, report: Dict[str, Any]) -> str:
        """Format report as markdown."""
        md = f"""# ML Model Performance Report: {report['model_name']}

## Overview
- **Total Runs:** {report['total_runs']}
- **Versions Tested:** {report['versions_tested']}

## Aggregate Metrics

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Accuracy | {report['aggregate_metrics']['accuracy']['mean']:.3f} | {report['aggregate_metrics']['accuracy']['std']:.3f} | {report['aggregate_metrics']['accuracy']['min']:.3f} | {report['aggregate_metrics']['accuracy']['max']:.3f} |
| Total Return | {report['aggregate_metrics']['total_return']['mean']:.2f}% | {report['aggregate_metrics']['total_return']['std']:.2f}% | {report['aggregate_metrics']['total_return']['min']:.2f}% | {report['aggregate_metrics']['total_return']['max']:.2f}% |
| Sharpe Ratio | {report['aggregate_metrics']['sharpe_ratio']['mean']:.2f} | {report['aggregate_metrics']['sharpe_ratio']['std']:.2f} | {report['aggregate_metrics']['sharpe_ratio']['min']:.2f} | {report['aggregate_metrics']['sharpe_ratio']['max']:.2f} |
| Win Rate | {report['aggregate_metrics']['win_rate']['mean']:.1f}% | {report['aggregate_metrics']['win_rate']['std']:.1f}% | {report['aggregate_metrics']['win_rate']['min']:.1f}% | {report['aggregate_metrics']['win_rate']['max']:.1f}% |

## Best Run
- **Version:** {report['best_run']['version']}
- **Sharpe Ratio:** {report['best_run']['sharpe_ratio']:.2f}
- **Total Return:** {report['best_run']['total_return']:.2f}%
- **Date:** {report['best_run']['date']}

## Recent Runs
| Version | Accuracy | Return | Sharpe | Date |
|---------|----------|--------|--------|------|
"""
        for run in report['recent_runs']:
            md += f"| {run['version']} | {run['accuracy']:.3f} | {run['total_return']:.2f}% | {run['sharpe_ratio']:.2f} | {run['date'][:10]} |\n"

        return md


async def save_training_performance(
    db_url: str,
    model_name: str,
    model_version: str,
    symbols: List[str],
    metrics: Dict[str, Any],
    config: Dict[str, Any] = None
) -> str:
    """
    Convenience function to save training performance.

    Args:
        db_url: Database URL
        model_name: Model name
        model_version: Model version
        symbols: Symbols trained on
        metrics: Performance metrics
        config: Training config

    Returns:
        Run ID
    """
    tracker = PerformanceTracker(db_url)

    try:
        await tracker.connect()

        record = ModelPerformanceRecord(
            model_name=model_name,
            model_version=model_version,
            symbols=symbols,
            start_time=datetime.now(timezone.utc) - timedelta(days=30),
            end_time=datetime.now(timezone.utc),
            accuracy=metrics.get('accuracy', 0),
            precision=metrics.get('precision', 0),
            recall=metrics.get('recall', 0),
            f1_score=metrics.get('f1_score', 0),
            total_return=metrics.get('total_return', 0),
            sharpe_ratio=metrics.get('sharpe_ratio', 0),
            max_drawdown=metrics.get('max_drawdown', 0),
            win_rate=metrics.get('win_rate', 0),
            num_trades=metrics.get('num_trades', 0),
            profit_factor=metrics.get('profit_factor', 0),
            training_samples=config.get('training_samples', 0) if config else 0,
            feature_count=config.get('feature_count', 0) if config else 0,
            enriched_order_flow=config.get('enrich_order_flow', False) if config else False,
            enriched_mtf=config.get('enrich_mtf', False) if config else False,
            config=config,
            run_type='training'
        )

        return await tracker.save_performance(record)

    finally:
        await tracker.close()
