#!/usr/bin/env python3
"""
Base Optimizer Framework for Strategy Parameter Tuning.

Runs backtests in separate subprocesses for memory isolation.
Each run releases memory when the subprocess exits.

Usage:
    This is a base class - use strategy-specific optimizers that inherit from it.
"""

import asyncio
import gc
import json
import logging
import multiprocessing as mp
import os
import signal
import sys
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Generator
import csv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    """Result of a single optimization run."""
    run_id: int
    params: Dict[str, Any]
    symbol: str
    strategy: str

    # Performance metrics
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    max_drawdown_pct: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0

    # Meta
    duration_seconds: float = 0.0
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class OptimizationConfig:
    """Configuration for optimization run."""
    strategy_name: str
    symbol: str
    param_grid: Dict[str, List[Any]]

    # Data settings
    db_url: str = ""
    start_date: Optional[str] = None  # YYYY-MM-DD
    end_date: Optional[str] = None
    period: str = "3m"  # Default 3 months for faster optimization

    # Execution settings
    max_workers: int = 1  # Number of parallel workers (1 = sequential)
    starting_capital: float = 100.0
    timeout_per_run: int = 600  # 10 minutes max per run

    # Output settings
    output_dir: str = "optimization_results"
    save_all_trades: bool = False  # Set True to save full trade logs

    # Parallel execution settings
    parallel: bool = False  # Enable parallel execution
    chunk_size: int = 10  # Number of runs per chunk for parallel


def _run_single_backtest(args: Tuple) -> Dict[str, Any]:
    """
    Run a single backtest in isolated process.

    This function runs in a subprocess - all memory is released when it exits.
    """
    run_id, config_dict, params, symbol = args

    # Import here to avoid loading in main process
    import asyncio
    import sys
    from pathlib import Path

    # Add project root to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    result = {
        'run_id': run_id,
        'params': params,
        'symbol': symbol,
        'strategy': config_dict['strategy_name'],
        'error': None,
    }

    try:
        from data import HistoricalDataProvider
        from ws_tester.strategy_loader import discover_strategies
        from backtest_runner import BacktestExecutor, BacktestConfig
        from datetime import datetime, timezone, timedelta

        async def run():
            # Initialize provider
            provider = HistoricalDataProvider(config_dict['db_url'])
            await provider.connect()

            try:
                # Determine time range
                end_time = datetime.now(timezone.utc)
                if config_dict.get('end_date'):
                    end_time = datetime.fromisoformat(config_dict['end_date']).replace(tzinfo=timezone.utc)

                start_time = None
                if config_dict.get('start_date'):
                    start_time = datetime.fromisoformat(config_dict['start_date']).replace(tzinfo=timezone.utc)
                elif config_dict.get('period'):
                    periods = {
                        '1w': timedelta(weeks=1),
                        '2w': timedelta(weeks=2),
                        '1m': timedelta(days=30),
                        '3m': timedelta(days=90),
                        '6m': timedelta(days=180),
                        '1y': timedelta(days=365),
                    }
                    delta = periods.get(config_dict['period'], timedelta(days=90))
                    start_time = end_time - delta
                else:
                    start_time = end_time - timedelta(days=90)

                # Load strategy and apply parameter overrides
                strategies_path = project_root / "strategies"
                all_strategies = discover_strategies(str(strategies_path))

                strategy_name = config_dict['strategy_name']
                if strategy_name not in all_strategies:
                    raise ValueError(f"Strategy {strategy_name} not found")

                strategy = all_strategies[strategy_name]

                # Apply parameter overrides to strategy config
                if hasattr(strategy, 'config'):
                    for key, value in params.items():
                        if key in strategy.config:
                            strategy.config[key] = value

                # Create backtest config
                bt_config = BacktestConfig(
                    start_time=start_time,
                    end_time=end_time,
                    symbols=[symbol],
                    strategies=[strategy_name],
                    starting_capital=config_dict.get('starting_capital', 100.0),
                )

                # Run backtest
                executor = BacktestExecutor(bt_config, provider)
                bt_result = await executor.run_strategy(strategy, start_time, end_time)

                # Extract metrics
                return {
                    'total_pnl': bt_result.total_pnl,
                    'total_pnl_pct': bt_result.total_pnl_pct,
                    'total_trades': bt_result.total_trades,
                    'winning_trades': bt_result.winning_trades,
                    'losing_trades': bt_result.losing_trades,
                    'win_rate': bt_result.win_rate,
                    'max_drawdown_pct': bt_result.max_drawdown_pct,
                    'profit_factor': bt_result.profit_factor if bt_result.profit_factor != float('inf') else 999.0,
                }

            finally:
                await provider.close()

        # Run the async backtest
        metrics = asyncio.run(run())
        result.update(metrics)

    except Exception as e:
        result['error'] = f"{type(e).__name__}: {str(e)}"
        logger.error(f"Run {run_id} failed: {e}")
        traceback.print_exc()

    # Force garbage collection before subprocess exits
    gc.collect()

    return result


class BaseOptimizer(ABC):
    """
    Base class for strategy parameter optimization.

    Runs backtests in subprocesses for memory isolation.
    """

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.results: List[RunResult] = []
        self.output_path = Path(config.output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Generate run timestamp
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    @abstractmethod
    def get_param_grid(self) -> Dict[str, List[Any]]:
        """Return parameter grid for this strategy. Override in subclass."""
        pass

    def generate_param_combinations(self) -> Generator[Dict[str, Any], None, None]:
        """Generate all parameter combinations from grid."""
        grid = self.get_param_grid()
        keys = list(grid.keys())
        values = list(grid.values())

        for combo in product(*values):
            yield dict(zip(keys, combo))

    def count_combinations(self) -> int:
        """Count total parameter combinations."""
        grid = self.get_param_grid()
        count = 1
        for values in grid.values():
            count *= len(values)
        return count

    def run_optimization(self) -> List[RunResult]:
        """
        Run full optimization across parameter grid.

        Uses parallel execution if config.parallel=True and config.max_workers > 1.
        Each run executes in a subprocess for memory isolation.
        """
        if self.config.parallel and self.config.max_workers > 1:
            return self._run_parallel()
        else:
            return self._run_sequential()

    def _run_sequential(self) -> List[RunResult]:
        """Run optimization sequentially (one backtest at a time)."""
        param_combos = list(self.generate_param_combinations())
        total_runs = len(param_combos)

        logger.info(f"=" * 60)
        logger.info(f"OPTIMIZATION: {self.config.strategy_name}")
        logger.info(f"Symbol: {self.config.symbol}")
        logger.info(f"Total runs: {total_runs}")
        logger.info(f"Period: {self.config.period}")
        logger.info(f"Mode: SEQUENTIAL")
        logger.info(f"=" * 60)

        # Prepare config dict for subprocess
        config_dict = {
            'strategy_name': self.config.strategy_name,
            'symbol': self.config.symbol,
            'db_url': self.config.db_url,
            'start_date': self.config.start_date,
            'end_date': self.config.end_date,
            'period': self.config.period,
            'starting_capital': self.config.starting_capital,
        }

        # Prepare run arguments
        run_args = [
            (i, config_dict, params, self.config.symbol)
            for i, params in enumerate(param_combos)
        ]

        # Run sequentially with subprocess isolation
        ctx = mp.get_context('spawn')

        for i, args in enumerate(run_args):
            run_id = args[0]
            params = args[2]

            logger.info(f"\nRun {run_id + 1}/{total_runs}")
            logger.info(f"Params: {params}")

            start_time = datetime.now()

            try:
                with ctx.Pool(1) as pool:
                    async_result = pool.apply_async(_run_single_backtest, (args,))
                    result_dict = async_result.get(timeout=self.config.timeout_per_run)

                duration = (datetime.now() - start_time).total_seconds()

                run_result = RunResult(
                    run_id=result_dict['run_id'],
                    params=result_dict['params'],
                    symbol=result_dict['symbol'],
                    strategy=result_dict['strategy'],
                    total_pnl=result_dict.get('total_pnl', 0),
                    total_pnl_pct=result_dict.get('total_pnl_pct', 0),
                    total_trades=result_dict.get('total_trades', 0),
                    winning_trades=result_dict.get('winning_trades', 0),
                    losing_trades=result_dict.get('losing_trades', 0),
                    win_rate=result_dict.get('win_rate', 0),
                    max_drawdown_pct=result_dict.get('max_drawdown_pct', 0),
                    profit_factor=result_dict.get('profit_factor', 0),
                    duration_seconds=duration,
                    error=result_dict.get('error'),
                )

                self.results.append(run_result)

                if run_result.error:
                    logger.error(f"  ERROR: {run_result.error}")
                else:
                    logger.info(f"  P&L: ${run_result.total_pnl:.2f} ({run_result.total_pnl_pct:.1f}%)")
                    logger.info(f"  Trades: {run_result.total_trades}, Win Rate: {run_result.win_rate:.1f}%")
                    logger.info(f"  Max DD: {run_result.max_drawdown_pct:.1f}%, PF: {run_result.profit_factor:.2f}")

                self._save_run_result(run_result)

            except mp.TimeoutError:
                logger.error(f"  TIMEOUT after {self.config.timeout_per_run}s")
                run_result = RunResult(
                    run_id=run_id,
                    params=params,
                    symbol=self.config.symbol,
                    strategy=self.config.strategy_name,
                    error="TIMEOUT",
                )
                self.results.append(run_result)

            except Exception as e:
                logger.error(f"  FAILED: {e}")
                run_result = RunResult(
                    run_id=run_id,
                    params=params,
                    symbol=self.config.symbol,
                    strategy=self.config.strategy_name,
                    error=str(e),
                )
                self.results.append(run_result)

            gc.collect()

        self._generate_report()
        return self.results

    def _run_parallel(self) -> List[RunResult]:
        """
        Run optimization in parallel using multiple CPU cores.

        Memory-efficient: runs in batches to control memory usage.
        """
        param_combos = list(self.generate_param_combinations())
        total_runs = len(param_combos)
        workers = min(self.config.max_workers, total_runs)

        logger.info(f"=" * 60)
        logger.info(f"OPTIMIZATION: {self.config.strategy_name}")
        logger.info(f"Symbol: {self.config.symbol}")
        logger.info(f"Total runs: {total_runs}")
        logger.info(f"Period: {self.config.period}")
        logger.info(f"Mode: PARALLEL ({workers} workers)")
        logger.info(f"=" * 60)

        # Prepare config dict for subprocess
        config_dict = {
            'strategy_name': self.config.strategy_name,
            'symbol': self.config.symbol,
            'db_url': self.config.db_url,
            'start_date': self.config.start_date,
            'end_date': self.config.end_date,
            'period': self.config.period,
            'starting_capital': self.config.starting_capital,
        }

        # Prepare all run arguments
        run_args = [
            (i, config_dict, params, self.config.symbol)
            for i, params in enumerate(param_combos)
        ]

        # Process in chunks to manage memory
        chunk_size = self.config.chunk_size
        ctx = mp.get_context('spawn')
        start_time = datetime.now()

        completed = 0
        for chunk_start in range(0, total_runs, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_runs)
            chunk_args = run_args[chunk_start:chunk_end]

            logger.info(f"\nProcessing batch {chunk_start + 1}-{chunk_end} of {total_runs} ({workers} workers)...")

            try:
                with ctx.Pool(workers) as pool:
                    # Use imap_unordered for better performance
                    results_iter = pool.imap_unordered(
                        _run_single_backtest,
                        chunk_args,
                        chunksize=max(1, len(chunk_args) // workers)
                    )

                    # Collect results with timeout
                    chunk_results = []
                    for result_dict in results_iter:
                        run_result = RunResult(
                            run_id=result_dict['run_id'],
                            params=result_dict['params'],
                            symbol=result_dict['symbol'],
                            strategy=result_dict['strategy'],
                            total_pnl=result_dict.get('total_pnl', 0),
                            total_pnl_pct=result_dict.get('total_pnl_pct', 0),
                            total_trades=result_dict.get('total_trades', 0),
                            winning_trades=result_dict.get('winning_trades', 0),
                            losing_trades=result_dict.get('losing_trades', 0),
                            win_rate=result_dict.get('win_rate', 0),
                            max_drawdown_pct=result_dict.get('max_drawdown_pct', 0),
                            profit_factor=result_dict.get('profit_factor', 0),
                            error=result_dict.get('error'),
                        )
                        chunk_results.append(run_result)
                        self._save_run_result(run_result)
                        completed += 1

                        # Progress update
                        if completed % 10 == 0:
                            elapsed = (datetime.now() - start_time).total_seconds()
                            rate = completed / elapsed if elapsed > 0 else 0
                            eta = (total_runs - completed) / rate if rate > 0 else 0
                            logger.info(f"  Progress: {completed}/{total_runs} ({rate:.1f}/s, ETA: {eta/60:.1f}m)")

                    self.results.extend(chunk_results)

                    # Log batch summary
                    successful = [r for r in chunk_results if r.error is None]
                    if successful:
                        best = max(successful, key=lambda r: r.total_pnl)
                        logger.info(f"  Batch best: P&L=${best.total_pnl:.2f}, WR={best.win_rate:.1f}%")

            except Exception as e:
                logger.error(f"  Batch error: {e}")
                # Mark remaining in batch as failed
                for args in chunk_args:
                    run_id = args[0]
                    params = args[2]
                    if not any(r.run_id == run_id for r in self.results):
                        run_result = RunResult(
                            run_id=run_id,
                            params=params,
                            symbol=self.config.symbol,
                            strategy=self.config.strategy_name,
                            error=str(e),
                        )
                        self.results.append(run_result)

            # Force garbage collection between batches
            gc.collect()

        total_elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"\nTotal time: {total_elapsed:.1f}s ({total_elapsed/60:.1f}m)")
        logger.info(f"Average: {total_elapsed/total_runs:.2f}s per run")

        self._generate_report()
        return self.results

    def _save_run_result(self, result: RunResult):
        """Save individual run result to CSV (append mode)."""
        csv_file = self.output_path / f"{self.config.strategy_name}_{self.config.symbol.replace('/', '_')}_{self.run_timestamp}_runs.csv"

        file_exists = csv_file.exists()

        # Flatten params into separate columns
        flat_result = {
            'run_id': result.run_id,
            'symbol': result.symbol,
            'strategy': result.strategy,
            'total_pnl': result.total_pnl,
            'total_pnl_pct': result.total_pnl_pct,
            'total_trades': result.total_trades,
            'winning_trades': result.winning_trades,
            'losing_trades': result.losing_trades,
            'win_rate': result.win_rate,
            'max_drawdown_pct': result.max_drawdown_pct,
            'profit_factor': result.profit_factor,
            'duration_seconds': result.duration_seconds,
            'error': result.error,
            'timestamp': result.timestamp,
        }

        # Add params as columns
        for key, value in result.params.items():
            flat_result[f'param_{key}'] = value

        with open(csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=flat_result.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(flat_result)

    def _generate_report(self):
        """Generate final optimization report."""
        report_file = self.output_path / f"{self.config.strategy_name}_{self.config.symbol.replace('/', '_')}_{self.run_timestamp}_report.json"

        # Filter successful runs
        successful = [r for r in self.results if r.error is None and r.total_trades > 0]

        if not successful:
            logger.warning("No successful runs to analyze!")
            report = {
                'strategy': self.config.strategy_name,
                'symbol': self.config.symbol,
                'total_runs': len(self.results),
                'successful_runs': 0,
                'failed_runs': len(self.results),
                'error': 'No successful runs',
            }
        else:
            # Sort by different metrics
            by_pnl = sorted(successful, key=lambda r: r.total_pnl, reverse=True)
            by_win_rate = sorted(successful, key=lambda r: r.win_rate, reverse=True)
            by_profit_factor = sorted(successful, key=lambda r: r.profit_factor, reverse=True)
            by_risk_adjusted = sorted(
                successful,
                key=lambda r: r.total_pnl / max(r.max_drawdown_pct, 0.1),  # Return/Risk ratio
                reverse=True
            )

            report = {
                'strategy': self.config.strategy_name,
                'symbol': self.config.symbol,
                'period': self.config.period,
                'timestamp': self.run_timestamp,
                'total_runs': len(self.results),
                'successful_runs': len(successful),
                'failed_runs': len(self.results) - len(successful),

                # Best by P&L
                'best_by_pnl': {
                    'params': by_pnl[0].params,
                    'total_pnl': by_pnl[0].total_pnl,
                    'total_pnl_pct': by_pnl[0].total_pnl_pct,
                    'win_rate': by_pnl[0].win_rate,
                    'max_drawdown_pct': by_pnl[0].max_drawdown_pct,
                    'profit_factor': by_pnl[0].profit_factor,
                    'total_trades': by_pnl[0].total_trades,
                },

                # Best by win rate (min 5 trades)
                'best_by_win_rate': None,

                # Best by profit factor (min 5 trades)
                'best_by_profit_factor': None,

                # Best risk-adjusted (P&L / Max DD)
                'best_risk_adjusted': {
                    'params': by_risk_adjusted[0].params,
                    'total_pnl': by_risk_adjusted[0].total_pnl,
                    'total_pnl_pct': by_risk_adjusted[0].total_pnl_pct,
                    'win_rate': by_risk_adjusted[0].win_rate,
                    'max_drawdown_pct': by_risk_adjusted[0].max_drawdown_pct,
                    'profit_factor': by_risk_adjusted[0].profit_factor,
                    'risk_reward_ratio': by_risk_adjusted[0].total_pnl / max(by_risk_adjusted[0].max_drawdown_pct, 0.1),
                },

                # Top 5 overall
                'top_5_by_pnl': [
                    {
                        'rank': i + 1,
                        'params': r.params,
                        'total_pnl': r.total_pnl,
                        'win_rate': r.win_rate,
                        'max_drawdown_pct': r.max_drawdown_pct,
                        'profit_factor': r.profit_factor,
                        'total_trades': r.total_trades,
                    }
                    for i, r in enumerate(by_pnl[:5])
                ],

                # Parameter analysis
                'parameter_sensitivity': self._analyze_parameter_sensitivity(successful),
            }

            # Best by win rate (min 5 trades)
            qualified_wr = [r for r in by_win_rate if r.total_trades >= 5]
            if qualified_wr:
                report['best_by_win_rate'] = {
                    'params': qualified_wr[0].params,
                    'win_rate': qualified_wr[0].win_rate,
                    'total_pnl': qualified_wr[0].total_pnl,
                    'total_trades': qualified_wr[0].total_trades,
                }

            # Best by profit factor (min 5 trades)
            qualified_pf = [r for r in by_profit_factor if r.total_trades >= 5]
            if qualified_pf:
                report['best_by_profit_factor'] = {
                    'params': qualified_pf[0].params,
                    'profit_factor': qualified_pf[0].profit_factor,
                    'total_pnl': qualified_pf[0].total_pnl,
                    'total_trades': qualified_pf[0].total_trades,
                }

        # Save report
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"\n{'=' * 60}")
        logger.info("OPTIMIZATION COMPLETE")
        logger.info(f"{'=' * 60}")
        logger.info(f"Total runs: {len(self.results)}")
        logger.info(f"Successful: {len([r for r in self.results if r.error is None])}")
        logger.info(f"Report saved: {report_file}")

        if successful:
            best = by_pnl[0]
            logger.info(f"\nBEST PARAMETERS (by P&L):")
            for key, value in best.params.items():
                logger.info(f"  {key}: {value}")
            logger.info(f"\nBEST PERFORMANCE:")
            logger.info(f"  P&L: ${best.total_pnl:.2f} ({best.total_pnl_pct:.1f}%)")
            logger.info(f"  Trades: {best.total_trades}, Win Rate: {best.win_rate:.1f}%")
            logger.info(f"  Max DD: {best.max_drawdown_pct:.1f}%, Profit Factor: {best.profit_factor:.2f}")

        return report

    def _analyze_parameter_sensitivity(self, results: List[RunResult]) -> Dict[str, Any]:
        """Analyze which parameters have most impact on performance."""
        if not results:
            return {}

        sensitivity = {}
        grid = self.get_param_grid()

        for param_name, param_values in grid.items():
            if len(param_values) < 2:
                continue

            # Group results by parameter value
            value_performance = {}
            for value in param_values:
                matching = [r for r in results if r.params.get(param_name) == value]
                if matching:
                    avg_pnl = sum(r.total_pnl for r in matching) / len(matching)
                    avg_wr = sum(r.win_rate for r in matching) / len(matching)
                    value_performance[str(value)] = {
                        'avg_pnl': avg_pnl,
                        'avg_win_rate': avg_wr,
                        'count': len(matching),
                    }

            # Calculate variance in performance
            if value_performance:
                pnl_values = [v['avg_pnl'] for v in value_performance.values()]
                pnl_range = max(pnl_values) - min(pnl_values)

                # Find best value
                best_value = max(value_performance.items(), key=lambda x: x[1]['avg_pnl'])

                sensitivity[param_name] = {
                    'pnl_range': pnl_range,
                    'best_value': best_value[0],
                    'best_avg_pnl': best_value[1]['avg_pnl'],
                    'values': value_performance,
                }

        # Sort by impact (pnl_range)
        sorted_sensitivity = dict(
            sorted(sensitivity.items(), key=lambda x: x[1]['pnl_range'], reverse=True)
        )

        return sorted_sensitivity
