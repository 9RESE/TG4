#!/usr/bin/env python3
"""
WaveTrend Oscillator Strategy Optimizer.

Optimizes parameters for the WaveTrend oscillator strategy.
Runs backtests in subprocesses for memory isolation.

Usage:
    python optimize_wavetrend.py --symbol XRP/USDT --period 3m
    python optimize_wavetrend.py --symbol BTC/USDT --period 6m --quick
    python optimize_wavetrend.py --symbol XRP/BTC --focus zones
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from optimization.base_optimizer import BaseOptimizer, OptimizationConfig


class WaveTrendOptimizer(BaseOptimizer):
    """
    Optimizer for WaveTrend Oscillator Strategy.

    Key tunable parameters:
    - wt_channel_length / wt_average_length: Core indicator settings
    - wt_overbought / wt_oversold: Zone thresholds
    - divergence_lookback: Divergence detection window
    - stop_loss_pct / take_profit_pct: Risk/Reward ratios
    - require_zone_exit: Signal quality vs frequency
    """

    def __init__(self, config: OptimizationConfig, quick_mode: bool = False):
        super().__init__(config)
        self.quick_mode = quick_mode

    def get_param_grid(self) -> Dict[str, List[Any]]:
        """
        Return parameter grid for WaveTrend strategy.

        Quick mode: ~36 combinations (~45 min)
        Full mode: ~300+ combinations (~5-6 hours)
        """
        if self.quick_mode:
            # Quick grid - most impactful parameters only
            return {
                'wt_channel_length': [9, 10, 11],
                'wt_average_length': [18, 21, 24],
                'wt_overbought': [55, 60, 65],
                'wt_oversold': [-55, -60, -65],
                'stop_loss_pct': [1.5],
                'take_profit_pct': [3.0],
            }
        else:
            # Full grid - comprehensive search
            return {
                # Core WaveTrend Settings
                'wt_channel_length': [8, 9, 10, 11, 12],
                'wt_average_length': [15, 18, 21, 24, 28],
                'wt_ma_length': [3, 4, 5],

                # Zone Thresholds
                'wt_overbought': [50, 55, 60, 65, 70],
                'wt_oversold': [-50, -55, -60, -65, -70],

                # Risk Management
                'stop_loss_pct': [1.0, 1.5, 2.0],
                'take_profit_pct': [2.0, 2.5, 3.0, 4.0],

                # Signal Quality
                'require_zone_exit': [True, False],
            }

    def get_focused_grid(self, focus: str) -> Dict[str, List[Any]]:
        """
        Get focused grids for specific optimization goals.

        Args:
            focus: 'zones', 'indicator', 'risk_reward'
        """
        if focus == 'zones':
            # Focus on zone threshold optimization
            return {
                'wt_channel_length': [10],
                'wt_average_length': [21],
                'wt_overbought': [45, 50, 55, 60, 65, 70, 75],
                'wt_oversold': [-45, -50, -55, -60, -65, -70, -75],
                'wt_extreme_overbought': [70, 75, 80, 85],
                'wt_extreme_oversold': [-70, -75, -80, -85],
                'require_zone_exit': [True, False],
                'stop_loss_pct': [1.5],
                'take_profit_pct': [3.0],
            }
        elif focus == 'indicator':
            # Focus on core indicator settings
            return {
                'wt_channel_length': [6, 7, 8, 9, 10, 11, 12, 13, 14],
                'wt_average_length': [12, 15, 18, 21, 24, 28, 32],
                'wt_ma_length': [2, 3, 4, 5, 6],
                'wt_overbought': [60],
                'wt_oversold': [-60],
                'stop_loss_pct': [1.5],
                'take_profit_pct': [3.0],
            }
        elif focus == 'risk_reward':
            # Focus on risk/reward optimization
            return {
                'wt_channel_length': [10],
                'wt_average_length': [21],
                'wt_overbought': [60],
                'wt_oversold': [-60],
                'stop_loss_pct': [0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5],
                'take_profit_pct': [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0],
                'require_zone_exit': [True],
            }
        elif focus == 'divergence':
            # Focus on divergence settings
            return {
                'wt_channel_length': [10],
                'wt_average_length': [21],
                'wt_overbought': [60],
                'wt_oversold': [-60],
                'use_divergence': [True, False],
                'divergence_lookback': [8, 10, 12, 14, 16, 18, 20],
                'stop_loss_pct': [1.5],
                'take_profit_pct': [3.0],
            }
        else:
            return self.get_param_grid()


def main():
    parser = argparse.ArgumentParser(
        description='WaveTrend Oscillator Strategy Optimizer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick optimization (36 runs, ~45 min)
    python optimize_wavetrend.py --symbol XRP/USDT --period 3m --quick

    # Full optimization (300+ runs, ~5-6 hours)
    python optimize_wavetrend.py --symbol XRP/USDT --period 6m

    # Focus on zone thresholds
    python optimize_wavetrend.py --symbol XRP/USDT --period 3m --focus zones

    # Focus on indicator settings
    python optimize_wavetrend.py --symbol BTC/USDT --period 3m --focus indicator
        """
    )

    parser.add_argument('--symbol', type=str, default='XRP/USDT',
                        help='Trading pair to optimize (default: XRP/USDT)')
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
                        choices=['zones', 'indicator', 'risk_reward', 'divergence'],
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
        strategy_name='wavetrend',
        symbol=args.symbol,
        param_grid={},
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
    optimizer = WaveTrendOptimizer(config, quick_mode=args.quick)

    # Use focused grid if specified
    if args.focus:
        optimizer.get_param_grid = lambda: optimizer.get_focused_grid(args.focus)

    # Show what we're about to do
    total_combinations = optimizer.count_combinations()
    print(f"\nWaveTrend Oscillator Strategy Optimizer")
    print(f"=" * 50)
    print(f"Symbol: {args.symbol}")
    print(f"Period: {args.period}")
    print(f"Mode: {'Quick' if args.quick else 'Full'}")
    if args.focus:
        print(f"Focus: {args.focus}")
    print(f"Parameter combinations: {total_combinations}")
    if args.parallel:
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
