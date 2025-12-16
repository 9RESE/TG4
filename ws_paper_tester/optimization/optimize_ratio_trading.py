#!/usr/bin/env python3
"""
Ratio Trading Strategy Optimizer.

Optimizes parameters for the XRP/BTC ratio trading strategy.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from optimization.base_optimizer import BaseOptimizer, OptimizationConfig


class RatioTradingOptimizer(BaseOptimizer):
    """Optimizer for Ratio Trading Strategy (XRP/BTC only)."""

    def __init__(self, config: OptimizationConfig, quick_mode: bool = False):
        super().__init__(config)
        self.quick_mode = quick_mode

    def get_param_grid(self) -> Dict[str, List[Any]]:
        if self.quick_mode:
            return {
                'lookback_periods': [15, 20, 25],
                'bollinger_std': [1.5, 2.0, 2.5],
                'entry_threshold': [1.0, 1.5, 2.0],
                'exit_threshold': [0.3, 0.5, 0.7],
                'stop_loss_pct': [0.5, 0.6, 0.8],
                'take_profit_pct': [0.5, 0.6, 0.8],
            }
        else:
            return {
                'lookback_periods': [10, 15, 20, 25, 30],
                'bollinger_std': [1.5, 2.0, 2.5, 3.0],
                'entry_threshold': [0.8, 1.0, 1.2, 1.5, 1.8, 2.0],
                'exit_threshold': [0.2, 0.3, 0.4, 0.5, 0.7],
                'stop_loss_pct': [0.4, 0.5, 0.6, 0.8, 1.0],
                'take_profit_pct': [0.4, 0.5, 0.6, 0.8, 1.0],
                'rsi_period': [10, 12, 14, 16],
                'rsi_oversold': [30, 35, 40],
                'rsi_overbought': [60, 65, 70],
            }

    def get_focused_grid(self, focus: str) -> Dict[str, List[Any]]:
        if focus == 'bollinger':
            return {
                'lookback_periods': [10, 15, 20, 25, 30, 35, 40],
                'bollinger_std': [1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
                'entry_threshold': [1.0, 1.5, 2.0],
                'exit_threshold': [0.5],
                'stop_loss_pct': [0.6],
                'take_profit_pct': [0.6],
            }
        elif focus == 'thresholds':
            return {
                'lookback_periods': [20],
                'bollinger_std': [2.0],
                'entry_threshold': [0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5],
                'exit_threshold': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8],
                'stop_loss_pct': [0.6],
                'take_profit_pct': [0.6],
            }
        elif focus == 'rsi':
            return {
                'lookback_periods': [20],
                'bollinger_std': [2.0],
                'entry_threshold': [1.5],
                'exit_threshold': [0.5],
                'use_rsi_confirmation': [True],
                'rsi_period': [8, 10, 12, 14, 16, 18, 20],
                'rsi_oversold': [25, 30, 35, 40],
                'rsi_overbought': [60, 65, 70, 75],
                'stop_loss_pct': [0.6],
                'take_profit_pct': [0.6],
            }
        elif focus == 'risk_reward':
            return {
                'lookback_periods': [20],
                'bollinger_std': [2.0],
                'entry_threshold': [1.5],
                'exit_threshold': [0.5],
                'stop_loss_pct': [0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2],
                'take_profit_pct': [0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2],
            }
        return self.get_param_grid()


def main():
    parser = argparse.ArgumentParser(description='Ratio Trading Strategy Optimizer')
    # Note: ratio_trading is XRP/BTC only
    parser.add_argument('--symbol', type=str, default='XRP/BTC')
    parser.add_argument('--period', type=str, default='3m')
    parser.add_argument('--start', type=str, default=None)
    parser.add_argument('--end', type=str, default=None)
    parser.add_argument('--db-url', type=str, default=os.getenv('DATABASE_URL'))
    parser.add_argument('--output', type=str, default='optimization_results')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--focus', type=str, choices=['bollinger', 'thresholds', 'rsi', 'risk_reward'])
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

    # Force XRP/BTC for ratio trading
    if args.symbol != 'XRP/BTC':
        print(f"Note: Ratio trading is XRP/BTC only, overriding {args.symbol} -> XRP/BTC")
        args.symbol = 'XRP/BTC'

    config = OptimizationConfig(
        strategy_name='ratio_trading',
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

    optimizer = RatioTradingOptimizer(config, quick_mode=args.quick)
    if args.focus:
        optimizer.get_param_grid = lambda: optimizer.get_focused_grid(args.focus)

    total = optimizer.count_combinations()
    print(f"\nRatio Trading Optimizer - {args.symbol}")
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
