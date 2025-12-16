#!/usr/bin/env python3
"""
Market Making Strategy Optimizer.

Optimizes parameters for the market making strategy.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from optimization.base_optimizer import BaseOptimizer, OptimizationConfig


class MarketMakingOptimizer(BaseOptimizer):
    """Optimizer for Market Making Strategy."""

    def __init__(self, config: OptimizationConfig, quick_mode: bool = False):
        super().__init__(config)
        self.quick_mode = quick_mode

    def get_param_grid(self) -> Dict[str, List[Any]]:
        if self.quick_mode:
            return {
                'min_spread_pct': [0.05, 0.08, 0.10],
                'imbalance_threshold': [0.08, 0.10, 0.12],
                'inventory_skew': [0.3, 0.5, 0.7],
                'take_profit_pct': [0.4, 0.5, 0.6],
                'stop_loss_pct': [0.4, 0.5, 0.6],
            }
        else:
            return {
                'min_spread_pct': [0.03, 0.05, 0.08, 0.10, 0.12],
                'imbalance_threshold': [0.05, 0.08, 0.10, 0.12, 0.15],
                'inventory_skew': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                'take_profit_pct': [0.3, 0.4, 0.5, 0.6],
                'stop_loss_pct': [0.3, 0.4, 0.5, 0.6],
                'trade_flow_threshold': [0.10, 0.15, 0.20],
                'gamma': [0.05, 0.1, 0.15, 0.2],  # A-S risk aversion
            }

    def get_focused_grid(self, focus: str) -> Dict[str, List[Any]]:
        if focus == 'spread':
            return {
                'min_spread_pct': [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15],
                'imbalance_threshold': [0.10],
                'inventory_skew': [0.5],
                'take_profit_pct': [0.5],
                'stop_loss_pct': [0.5],
            }
        elif focus == 'inventory':
            return {
                'min_spread_pct': [0.08],
                'imbalance_threshold': [0.10],
                'inventory_skew': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                'max_inventory': [50, 75, 100, 150, 200],
                'gamma': [0.05, 0.1, 0.15, 0.2, 0.3],
                'take_profit_pct': [0.5],
                'stop_loss_pct': [0.5],
            }
        elif focus == 'risk_reward':
            return {
                'min_spread_pct': [0.08],
                'imbalance_threshold': [0.10],
                'inventory_skew': [0.5],
                'take_profit_pct': [0.2, 0.3, 0.4, 0.5, 0.6, 0.8],
                'stop_loss_pct': [0.2, 0.3, 0.4, 0.5, 0.6, 0.8],
            }
        return self.get_param_grid()


def main():
    parser = argparse.ArgumentParser(description='Market Making Strategy Optimizer')
    parser.add_argument('--symbol', type=str, default='XRP/USDT')
    parser.add_argument('--period', type=str, default='3m')
    parser.add_argument('--start', type=str, default=None)
    parser.add_argument('--end', type=str, default=None)
    parser.add_argument('--db-url', type=str, default=os.getenv('DATABASE_URL'))
    parser.add_argument('--output', type=str, default='optimization_results')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--focus', type=str, choices=['spread', 'inventory', 'risk_reward'])
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
        strategy_name='market_making',
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

    optimizer = MarketMakingOptimizer(config, quick_mode=args.quick)
    if args.focus:
        optimizer.get_param_grid = lambda: optimizer.get_focused_grid(args.focus)

    total = optimizer.count_combinations()
    print(f"\nMarket Making Optimizer - {args.symbol}")
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
