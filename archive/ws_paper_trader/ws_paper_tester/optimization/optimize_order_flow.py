#!/usr/bin/env python3
"""
Order Flow Strategy Optimizer.

Optimizes parameters for the order flow strategy.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from optimization.base_optimizer import BaseOptimizer, OptimizationConfig


class OrderFlowOptimizer(BaseOptimizer):
    """Optimizer for Order Flow Strategy."""

    def __init__(self, config: OptimizationConfig, quick_mode: bool = False):
        super().__init__(config)
        self.quick_mode = quick_mode

    def get_param_grid(self) -> Dict[str, List[Any]]:
        if self.quick_mode:
            return {
                'buy_imbalance_threshold': [0.25, 0.30, 0.35],
                'sell_imbalance_threshold': [0.20, 0.25, 0.30],
                'volume_spike_mult': [1.8, 2.0, 2.2],
                'take_profit_pct': [0.8, 1.0, 1.2],
                'stop_loss_pct': [0.4, 0.5, 0.6],
            }
        else:
            return {
                'buy_imbalance_threshold': [0.20, 0.25, 0.30, 0.35, 0.40],
                'sell_imbalance_threshold': [0.15, 0.20, 0.25, 0.30, 0.35],
                'volume_spike_mult': [1.5, 1.8, 2.0, 2.2, 2.5],
                'lookback_trades': [30, 40, 50, 60, 75],
                'take_profit_pct': [0.6, 0.8, 1.0, 1.2, 1.5],
                'stop_loss_pct': [0.3, 0.4, 0.5, 0.6],
                'vpin_high_threshold': [0.6, 0.65, 0.7, 0.75],
            }

    def get_focused_grid(self, focus: str) -> Dict[str, List[Any]]:
        if focus == 'imbalance':
            return {
                'buy_imbalance_threshold': [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45],
                'sell_imbalance_threshold': [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
                'use_asymmetric_thresholds': [True, False],
                'volume_spike_mult': [2.0],
                'take_profit_pct': [1.0],
                'stop_loss_pct': [0.5],
            }
        elif focus == 'vpin':
            return {
                'buy_imbalance_threshold': [0.30],
                'sell_imbalance_threshold': [0.25],
                'volume_spike_mult': [2.0],
                'use_vpin': [True],
                'vpin_bucket_count': [30, 40, 50, 60],
                'vpin_high_threshold': [0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
                'vpin_lookback_trades': [150, 200, 250, 300],
                'take_profit_pct': [1.0],
                'stop_loss_pct': [0.5],
            }
        elif focus == 'risk_reward':
            return {
                'buy_imbalance_threshold': [0.30],
                'sell_imbalance_threshold': [0.25],
                'volume_spike_mult': [2.0],
                'take_profit_pct': [0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0],
                'stop_loss_pct': [0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0],
            }
        return self.get_param_grid()


def main():
    parser = argparse.ArgumentParser(description='Order Flow Strategy Optimizer')
    parser.add_argument('--symbol', type=str, default='XRP/USDT')
    parser.add_argument('--period', type=str, default='3m')
    parser.add_argument('--start', type=str, default=None)
    parser.add_argument('--end', type=str, default=None)
    parser.add_argument('--db-url', type=str, default=os.getenv('DATABASE_URL'))
    parser.add_argument('--output', type=str, default='optimization_results')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--focus', type=str, choices=['imbalance', 'vpin', 'risk_reward'])
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
        strategy_name='order_flow',
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

    optimizer = OrderFlowOptimizer(config, quick_mode=args.quick)
    if args.focus:
        optimizer.get_param_grid = lambda: optimizer.get_focused_grid(args.focus)

    total = optimizer.count_combinations()
    print(f"\nOrder Flow Optimizer - {args.symbol}")
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
