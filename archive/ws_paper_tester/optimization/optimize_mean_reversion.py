#!/usr/bin/env python3
"""
Mean Reversion Strategy Optimizer.

Optimizes parameters for the mean reversion strategy.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from optimization.base_optimizer import BaseOptimizer, OptimizationConfig


class MeanReversionOptimizer(BaseOptimizer):
    """Optimizer for Mean Reversion Strategy."""

    def __init__(self, config: OptimizationConfig, quick_mode: bool = False):
        super().__init__(config)
        self.quick_mode = quick_mode

    def get_param_grid(self) -> Dict[str, List[Any]]:
        if self.quick_mode:
            return {
                'lookback_candles': [15, 20, 25],
                'deviation_threshold': [0.4, 0.5, 0.6],
                'rsi_period': [12, 14, 16],
                'rsi_oversold': [30, 35],
                'rsi_overbought': [65, 70],
                'stop_loss_pct': [0.4, 0.5],
                'take_profit_pct': [0.4, 0.5],
            }
        else:
            return {
                'lookback_candles': [10, 15, 20, 25, 30],
                'deviation_threshold': [0.3, 0.4, 0.5, 0.6, 0.8, 1.0],
                'bb_period': [15, 20, 25],
                'bb_std_dev': [1.5, 2.0, 2.5],
                'rsi_period': [10, 12, 14, 16, 18],
                'rsi_oversold': [25, 30, 35, 40],
                'rsi_overbought': [60, 65, 70, 75],
                'stop_loss_pct': [0.3, 0.4, 0.5, 0.6],
                'take_profit_pct': [0.3, 0.4, 0.5, 0.6, 0.8],
            }

    def get_focused_grid(self, focus: str) -> Dict[str, List[Any]]:
        if focus == 'rsi':
            return {
                'lookback_candles': [20],
                'deviation_threshold': [0.5],
                'rsi_period': [8, 10, 12, 14, 16, 18, 20],
                'rsi_oversold': [20, 25, 30, 35, 40],
                'rsi_overbought': [60, 65, 70, 75, 80],
                'stop_loss_pct': [0.5],
                'take_profit_pct': [0.5],
            }
        elif focus == 'deviation':
            return {
                'lookback_candles': [15, 20, 25, 30],
                'deviation_threshold': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0],
                'bb_std_dev': [1.5, 2.0, 2.5, 3.0],
                'rsi_period': [14],
                'rsi_oversold': [35],
                'rsi_overbought': [65],
                'stop_loss_pct': [0.5],
                'take_profit_pct': [0.5],
            }
        elif focus == 'risk_reward':
            return {
                'lookback_candles': [20],
                'deviation_threshold': [0.5],
                'rsi_period': [14],
                'rsi_oversold': [35],
                'rsi_overbought': [65],
                'stop_loss_pct': [0.3, 0.4, 0.5, 0.6, 0.8, 1.0],
                'take_profit_pct': [0.3, 0.4, 0.5, 0.6, 0.8, 1.0],
            }
        return self.get_param_grid()


def main():
    parser = argparse.ArgumentParser(description='Mean Reversion Strategy Optimizer')
    parser.add_argument('--symbol', type=str, default='XRP/USDT')
    parser.add_argument('--period', type=str, default='3m')
    parser.add_argument('--start', type=str, default=None)
    parser.add_argument('--end', type=str, default=None)
    parser.add_argument('--db-url', type=str, default=os.getenv('DATABASE_URL'))
    parser.add_argument('--output', type=str, default='optimization_results')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--focus', type=str, choices=['rsi', 'deviation', 'risk_reward'])
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
        print("ERROR: Database URL required. Set DATABASE_URL or use --db-url")
        sys.exit(1)

    config = OptimizationConfig(
        strategy_name='mean_reversion',
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

    optimizer = MeanReversionOptimizer(config, quick_mode=args.quick)
    if args.focus:
        optimizer.get_param_grid = lambda: optimizer.get_focused_grid(args.focus)

    total = optimizer.count_combinations()
    print(f"\nMean Reversion Optimizer - {args.symbol}")
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
