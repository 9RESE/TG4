#!/usr/bin/env python3
"""
Grid RSI Reversion Strategy Optimizer.

Optimizes parameters for the grid RSI reversion strategy.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from optimization.base_optimizer import BaseOptimizer, OptimizationConfig


class GridRSIOptimizer(BaseOptimizer):
    """Optimizer for Grid RSI Reversion Strategy."""

    def __init__(self, config: OptimizationConfig, quick_mode: bool = False):
        super().__init__(config)
        self.quick_mode = quick_mode

    def get_param_grid(self) -> Dict[str, List[Any]]:
        if self.quick_mode:
            return {
                'num_grids': [10, 15, 20],
                'grid_spacing_pct': [1.0, 1.5, 2.0],
                'rsi_period': [12, 14, 16],
                'rsi_oversold': [25, 30],
                'rsi_overbought': [70, 75],
                'stop_loss_pct': [5.0, 8.0],
            }
        else:
            return {
                'num_grids': [8, 10, 12, 15, 18, 20],
                'grid_spacing_pct': [0.8, 1.0, 1.2, 1.5, 2.0, 2.5],
                'range_pct': [5.0, 7.5, 10.0],
                'rsi_period': [10, 12, 14, 16, 18],
                'rsi_oversold': [20, 25, 30, 35],
                'rsi_overbought': [65, 70, 75, 80],
                'stop_loss_pct': [3.0, 5.0, 8.0, 10.0],
                'max_accumulation_levels': [3, 4, 5, 6],
            }

    def get_focused_grid(self, focus: str) -> Dict[str, List[Any]]:
        if focus == 'grid':
            return {
                'num_grids': [6, 8, 10, 12, 15, 18, 20, 25],
                'grid_spacing_pct': [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0],
                'range_pct': [5.0, 7.5, 10.0, 12.5, 15.0],
                'rsi_period': [14],
                'rsi_oversold': [30],
                'rsi_overbought': [70],
                'stop_loss_pct': [8.0],
            }
        elif focus == 'rsi':
            return {
                'num_grids': [15],
                'grid_spacing_pct': [1.5],
                'rsi_period': [8, 10, 12, 14, 16, 18, 20],
                'rsi_oversold': [20, 25, 30, 35, 40],
                'rsi_overbought': [60, 65, 70, 75, 80],
                'rsi_extreme_multiplier': [1.0, 1.2, 1.3, 1.5],
                'stop_loss_pct': [8.0],
            }
        elif focus == 'risk':
            return {
                'num_grids': [15],
                'grid_spacing_pct': [1.5],
                'rsi_period': [14],
                'rsi_oversold': [30],
                'rsi_overbought': [70],
                'stop_loss_pct': [3.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0],
                'max_accumulation_levels': [2, 3, 4, 5, 6, 7],
                'adx_threshold': [25, 30, 35],
            }
        return self.get_param_grid()


def main():
    parser = argparse.ArgumentParser(description='Grid RSI Reversion Strategy Optimizer')
    parser.add_argument('--symbol', type=str, default='XRP/USDT')
    parser.add_argument('--period', type=str, default='3m')
    parser.add_argument('--start', type=str, default=None)
    parser.add_argument('--end', type=str, default=None)
    parser.add_argument('--db-url', type=str, default=os.getenv('DATABASE_URL'))
    parser.add_argument('--output', type=str, default='optimization_results')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--focus', type=str, choices=['grid', 'rsi', 'risk'])
    parser.add_argument('--capital', type=float, default=100.0)
    parser.add_argument('--timeout', type=int, default=600)
    parser.add_argument('--parallel', action='store_true',
                        help='Enable parallel execution')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers')
    parser.add_argument('--chunk-size', type=int, default=50,
                        help='Batch size for parallel processing')

    args = parser.parse_args()

    if not args.db_url:
        print("ERROR: Database URL required.")
        sys.exit(1)

    config = OptimizationConfig(
        strategy_name='grid_rsi_reversion',
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

    optimizer = GridRSIOptimizer(config, quick_mode=args.quick)
    if args.focus:
        optimizer.get_param_grid = lambda: optimizer.get_focused_grid(args.focus)

    total = optimizer.count_combinations()
    print(f"\nGrid RSI Reversion Optimizer - {args.symbol}")
    if args.parallel:
        est_time = (total / args.workers) * 10 / 60
        print(f"Combinations: {total}, PARALLEL ({args.workers} workers), Est: ~{est_time:.0f} min")
    else:
        print(f"Combinations: {total}, Est. time: ~{total * 1.5:.0f} min")

    if input("\nProceed? [y/N]: ").lower() != 'y':
        sys.exit(0)

    optimizer.run_optimization()


if __name__ == '__main__':
    main()
