#!/usr/bin/env python3
"""
EMA-9 Trend Flip Strategy Optimizer.

Optimizes parameters for the EMA-9 trend flip strategy.
Runs backtests in subprocesses for memory isolation.

Usage:
    python optimize_ema9.py --symbol BTC/USDT --period 3m
    python optimize_ema9.py --symbol BTC/USDT --period 6m --quick
    python optimize_ema9.py --symbol BTC/USDT --db-url "postgresql://..."
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from optimization.base_optimizer import BaseOptimizer, OptimizationConfig


class EMA9Optimizer(BaseOptimizer):
    """
    Optimizer for EMA-9 Trend Flip Strategy.

    Key tunable parameters:
    - ema_period: EMA calculation period (7-12)
    - consecutive_candles: Confirmation bars required (2-5)
    - buffer_pct: Whipsaw reduction buffer (0.05-0.20%)
    - stop_loss_pct / take_profit_pct: Risk/Reward ratios
    - cooldown_minutes: Time between signals
    """

    def __init__(self, config: OptimizationConfig, quick_mode: bool = False):
        super().__init__(config)
        self.quick_mode = quick_mode

    def get_param_grid(self) -> Dict[str, List[Any]]:
        """
        Return parameter grid for EMA-9 strategy.

        Quick mode: ~24 combinations (~30 min)
        Full mode: ~180 combinations (~3-4 hours)
        """
        if self.quick_mode:
            # Quick grid - most impactful parameters only
            return {
                'ema_period': [8, 9, 10],
                'consecutive_candles': [2, 3, 4],
                'buffer_pct': [0.05, 0.10],
                'stop_loss_pct': [0.8, 1.0],
                'take_profit_pct': [1.6, 2.0],  # 2:1 R:R maintained
            }
        else:
            # Full grid - comprehensive search
            return {
                # EMA Settings
                'ema_period': [7, 8, 9, 10, 11, 12],
                'consecutive_candles': [2, 3, 4, 5],
                'buffer_pct': [0.05, 0.08, 0.10, 0.12, 0.15],

                # Risk Management
                'stop_loss_pct': [0.6, 0.8, 1.0, 1.2, 1.5],
                'take_profit_pct': [1.2, 1.5, 1.8, 2.0, 2.5, 3.0],

                # Cooldowns
                'cooldown_minutes': [15, 30, 45, 60],
            }

    def get_focused_grid(self, focus: str) -> Dict[str, List[Any]]:
        """
        Get focused grids for specific optimization goals.

        Args:
            focus: 'signal_quality', 'risk_reward', 'entry_timing'
        """
        if focus == 'signal_quality':
            # Focus on EMA settings that affect signal quality
            return {
                'ema_period': [7, 8, 9, 10, 11, 12, 13, 14],
                'consecutive_candles': [1, 2, 3, 4, 5, 6],
                'buffer_pct': [0.0, 0.05, 0.10, 0.15, 0.20],
                'stop_loss_pct': [1.0],
                'take_profit_pct': [2.0],
            }
        elif focus == 'risk_reward':
            # Focus on stop loss / take profit optimization
            return {
                'ema_period': [9],
                'consecutive_candles': [3],
                'buffer_pct': [0.10],
                'stop_loss_pct': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0],
                'take_profit_pct': [1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 4.0],
            }
        elif focus == 'entry_timing':
            # Focus on cooldown and consecutive candle optimization
            return {
                'ema_period': [9],
                'consecutive_candles': [1, 2, 3, 4, 5],
                'buffer_pct': [0.05, 0.10, 0.15],
                'stop_loss_pct': [1.0],
                'take_profit_pct': [2.0],
                'cooldown_minutes': [0, 15, 30, 45, 60, 90, 120],
            }
        else:
            return self.get_param_grid()


def main():
    parser = argparse.ArgumentParser(
        description='EMA-9 Trend Flip Strategy Optimizer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick optimization (24 runs, ~30 min)
    python optimize_ema9.py --symbol BTC/USDT --period 3m --quick

    # Full optimization (180+ runs, ~3-4 hours)
    python optimize_ema9.py --symbol BTC/USDT --period 6m

    # Focused optimization on signal quality
    python optimize_ema9.py --symbol BTC/USDT --period 3m --focus signal_quality

    # Focused optimization on risk/reward ratios
    python optimize_ema9.py --symbol BTC/USDT --period 3m --focus risk_reward
        """
    )

    parser.add_argument('--symbol', type=str, default='BTC/USDT',
                        help='Trading pair to optimize (default: BTC/USDT)')
    parser.add_argument('--period', type=str, default='3m',
                        help='Backtest period: 1w, 2w, 1m, 3m, 6m, 1y (default: 3m)')
    parser.add_argument('--start', type=str, default=None,
                        help='Start date (YYYY-MM-DD), overrides --period')
    parser.add_argument('--end', type=str, default=None,
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--db-url', type=str, default=os.getenv('DATABASE_URL'),
                        help='Database URL (or set DATABASE_URL env var)')
    parser.add_argument('--output', type=str, default='optimization_results',
                        help='Output directory (default: optimization_results)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode with reduced parameter grid')
    parser.add_argument('--focus', type=str, default=None,
                        choices=['signal_quality', 'risk_reward', 'entry_timing'],
                        help='Focus optimization on specific aspect')
    parser.add_argument('--capital', type=float, default=100.0,
                        help='Starting capital (default: 100)')
    parser.add_argument('--timeout', type=int, default=600,
                        help='Timeout per run in seconds (default: 600)')
    parser.add_argument('--parallel', action='store_true',
                        help='Enable parallel execution using multiple CPU cores')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers (default: 8)')
    parser.add_argument('--chunk-size', type=int, default=50,
                        help='Batch size for parallel processing (default: 50)')

    args = parser.parse_args()

    if not args.db_url:
        print("ERROR: Database URL required. Set DATABASE_URL or use --db-url")
        print("Example: DATABASE_URL=postgresql://trading:password@localhost:5433/kraken_data")
        sys.exit(1)

    # Create optimization config
    config = OptimizationConfig(
        strategy_name='ema9_trend_flip',
        symbol=args.symbol,
        param_grid={},  # Will be set by optimizer
        db_url=args.db_url,
        start_date=args.start,
        end_date=args.end,
        period=args.period,
        starting_capital=args.capital,
        timeout_per_run=args.timeout,
        output_dir=args.output,
        parallel=args.parallel,
        max_workers=args.workers,
        chunk_size=args.chunk_size,
    )

    # Create optimizer
    optimizer = EMA9Optimizer(config, quick_mode=args.quick)

    # Use focused grid if specified
    if args.focus:
        optimizer.get_param_grid = lambda: optimizer.get_focused_grid(args.focus)

    # Show what we're about to do
    total_combinations = optimizer.count_combinations()
    print(f"\nEMA-9 Trend Flip Strategy Optimizer")
    print(f"=" * 50)
    print(f"Symbol: {args.symbol}")
    print(f"Period: {args.period}")
    print(f"Mode: {'Quick' if args.quick else 'Full'}")
    if args.focus:
        print(f"Focus: {args.focus}")
    print(f"Parameter combinations: {total_combinations}")
    if args.parallel:
        # Parallel mode: estimate ~10s per run with workers
        est_time = (total_combinations / args.workers) * 10 / 60
        print(f"Execution: PARALLEL ({args.workers} workers, chunk={args.chunk_size})")
        print(f"Estimated time: ~{est_time:.0f} minutes")
    else:
        print(f"Execution: SEQUENTIAL")
        print(f"Estimated time: ~{total_combinations * 1.5:.0f} minutes")
    print(f"Output: {args.output}/")
    print(f"=" * 50)

    # Confirm with user
    response = input("\nProceed with optimization? [y/N]: ")
    if response.lower() != 'y':
        print("Cancelled.")
        sys.exit(0)

    # Run optimization
    results = optimizer.run_optimization()

    print(f"\nOptimization complete! Check {args.output}/ for results.")


if __name__ == '__main__':
    main()
