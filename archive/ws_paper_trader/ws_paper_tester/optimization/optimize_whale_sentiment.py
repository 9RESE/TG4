#!/usr/bin/env python3
"""
Whale Sentiment Strategy Optimizer.

Optimizes parameters for the whale sentiment strategy.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from optimization.base_optimizer import BaseOptimizer, OptimizationConfig


class WhaleSentimentOptimizer(BaseOptimizer):
    """Optimizer for Whale Sentiment Strategy."""

    def __init__(self, config: OptimizationConfig, quick_mode: bool = False):
        super().__init__(config)
        self.quick_mode = quick_mode

    def get_param_grid(self) -> Dict[str, List[Any]]:
        if self.quick_mode:
            return {
                'volume_spike_mult': [1.8, 2.0, 2.5],
                'fear_deviation_pct': [-4.0, -5.0, -6.0],
                'greed_deviation_pct': [4.0, 5.0, 6.0],
                'stop_loss_pct': [2.0, 2.5, 3.0],
                'take_profit_pct': [4.0, 5.0, 6.0],
            }
        else:
            return {
                'volume_spike_mult': [1.5, 1.8, 2.0, 2.2, 2.5, 3.0],
                'fear_deviation_pct': [-3.0, -4.0, -5.0, -6.0, -7.0],
                'greed_deviation_pct': [3.0, 4.0, 5.0, 6.0, 7.0],
                'extreme_fear_deviation_pct': [-6.0, -8.0, -10.0],
                'extreme_greed_deviation_pct': [6.0, 8.0, 10.0],
                'price_lookback': [36, 48, 60, 72],
                'stop_loss_pct': [1.5, 2.0, 2.5, 3.0],
                'take_profit_pct': [3.0, 4.0, 5.0, 6.0],
                'min_confidence': [0.45, 0.50, 0.55],
            }

    def get_focused_grid(self, focus: str) -> Dict[str, List[Any]]:
        if focus == 'volume':
            return {
                'volume_spike_mult': [1.2, 1.5, 1.8, 2.0, 2.2, 2.5, 2.8, 3.0, 3.5],
                'volume_window': [144, 200, 288, 350],  # 12h, 16h, 24h, 29h in 5m candles
                'min_spike_trades': [10, 15, 20, 25, 30],
                'fear_deviation_pct': [-5.0],
                'greed_deviation_pct': [5.0],
                'stop_loss_pct': [2.5],
                'take_profit_pct': [5.0],
            }
        elif focus == 'sentiment':
            return {
                'volume_spike_mult': [2.0],
                'fear_deviation_pct': [-3.0, -4.0, -5.0, -6.0, -7.0, -8.0],
                'greed_deviation_pct': [3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                'extreme_fear_deviation_pct': [-6.0, -7.0, -8.0, -9.0, -10.0],
                'extreme_greed_deviation_pct': [6.0, 7.0, 8.0, 9.0, 10.0],
                'price_lookback': [24, 36, 48, 60, 72, 96],
                'stop_loss_pct': [2.5],
                'take_profit_pct': [5.0],
            }
        elif focus == 'risk_reward':
            return {
                'volume_spike_mult': [2.0],
                'fear_deviation_pct': [-5.0],
                'greed_deviation_pct': [5.0],
                'stop_loss_pct': [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
                'take_profit_pct': [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            }
        return self.get_param_grid()


def main():
    parser = argparse.ArgumentParser(description='Whale Sentiment Strategy Optimizer')
    parser.add_argument('--symbol', type=str, default='XRP/USDT')
    parser.add_argument('--period', type=str, default='3m')
    parser.add_argument('--start', type=str, default=None)
    parser.add_argument('--end', type=str, default=None)
    parser.add_argument('--db-url', type=str, default=os.getenv('DATABASE_URL'))
    parser.add_argument('--output', type=str, default='optimization_results')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--focus', type=str, choices=['volume', 'sentiment', 'risk_reward'])
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
        strategy_name='whale_sentiment',
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

    optimizer = WhaleSentimentOptimizer(config, quick_mode=args.quick)
    if args.focus:
        optimizer.get_param_grid = lambda: optimizer.get_focused_grid(args.focus)

    total = optimizer.count_combinations()
    print(f"\nWhale Sentiment Optimizer - {args.symbol}")
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
