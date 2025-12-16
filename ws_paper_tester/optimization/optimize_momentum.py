#!/usr/bin/env python3
"""
Momentum Scalping Strategy Optimizer.

Optimizes parameters for the momentum scalping strategy.
Runs backtests in subprocesses for memory isolation.

Usage:
    python optimize_momentum.py --symbol XRP/USDT --period 3m
    python optimize_momentum.py --symbol BTC/USDT --period 6m --quick
    python optimize_momentum.py --symbol XRP/USDT --focus indicators
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from optimization.base_optimizer import BaseOptimizer, OptimizationConfig


class MomentumScalpingOptimizer(BaseOptimizer):
    """
    Optimizer for Momentum Scalping Strategy.

    Key tunable parameters:
    - ema_fast_period / ema_slow_period: Trend detection
    - rsi_period / rsi_overbought / rsi_oversold: Momentum indicators
    - take_profit_pct / stop_loss_pct: Risk/Reward
    - volume_spike_threshold: Volume confirmation
    - max_hold_seconds: Position duration
    """

    def __init__(self, config: OptimizationConfig, quick_mode: bool = False):
        super().__init__(config)
        self.quick_mode = quick_mode

    def get_param_grid(self) -> Dict[str, List[Any]]:
        """
        Return parameter grid for Momentum Scalping strategy.

        Quick mode: ~48 combinations (~1 hour)
        Full mode: ~500+ combinations (~8-10 hours)
        """
        if self.quick_mode:
            # Quick grid - most impactful parameters only
            return {
                'ema_fast_period': [6, 8, 10],
                'ema_slow_period': [18, 21, 24],
                'rsi_period': [6, 7, 8],
                'take_profit_pct': [0.6, 0.8, 1.0],
                'stop_loss_pct': [0.3, 0.4, 0.5],
                'volume_spike_threshold': [1.3, 1.5, 1.8],
            }
        else:
            # Full grid - comprehensive search
            return {
                # EMA Settings
                'ema_fast_period': [5, 6, 7, 8, 9, 10, 12],
                'ema_slow_period': [15, 18, 21, 24, 28],
                'ema_filter_period': [40, 50, 60],

                # RSI Settings
                'rsi_period': [5, 6, 7, 8, 9, 10],
                'rsi_overbought': [65, 70, 75],
                'rsi_oversold': [25, 30, 35],

                # Risk Management
                'take_profit_pct': [0.5, 0.6, 0.8, 1.0, 1.2],
                'stop_loss_pct': [0.25, 0.3, 0.4, 0.5, 0.6],
                'max_hold_seconds': [120, 180, 240, 300],

                # Volume
                'volume_spike_threshold': [1.2, 1.5, 1.8, 2.0],
            }

    def get_focused_grid(self, focus: str) -> Dict[str, List[Any]]:
        """
        Get focused grids for specific optimization goals.

        Args:
            focus: 'indicators', 'risk_reward', 'timing', 'volume'
        """
        if focus == 'indicators':
            # Focus on EMA and RSI settings
            return {
                'ema_fast_period': [4, 5, 6, 7, 8, 9, 10, 11, 12],
                'ema_slow_period': [12, 15, 18, 21, 24, 28, 32],
                'rsi_period': [4, 5, 6, 7, 8, 9, 10, 11, 12],
                'take_profit_pct': [0.8],
                'stop_loss_pct': [0.4],
                'volume_spike_threshold': [1.5],
            }
        elif focus == 'risk_reward':
            # Focus on stop loss / take profit optimization
            return {
                'ema_fast_period': [8],
                'ema_slow_period': [21],
                'rsi_period': [7],
                'take_profit_pct': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5],
                'stop_loss_pct': [0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8],
                'volume_spike_threshold': [1.5],
            }
        elif focus == 'timing':
            # Focus on hold time and cooldown
            return {
                'ema_fast_period': [8],
                'ema_slow_period': [21],
                'rsi_period': [7],
                'take_profit_pct': [0.8],
                'stop_loss_pct': [0.4],
                'max_hold_seconds': [60, 90, 120, 150, 180, 240, 300, 360, 480],
                'cooldown_seconds': [15, 20, 30, 45, 60],
                'volume_spike_threshold': [1.5],
            }
        elif focus == 'volume':
            # Focus on volume confirmation settings
            return {
                'ema_fast_period': [8],
                'ema_slow_period': [21],
                'rsi_period': [7],
                'take_profit_pct': [0.8],
                'stop_loss_pct': [0.4],
                'volume_spike_threshold': [1.0, 1.2, 1.3, 1.5, 1.7, 2.0, 2.5],
                'volume_lookback': [10, 15, 20, 25, 30],
                'require_volume_confirmation': [True, False],
            }
        elif focus == 'macd':
            # Focus on MACD confirmation
            return {
                'ema_fast_period': [8],
                'ema_slow_period': [21],
                'rsi_period': [7],
                'take_profit_pct': [0.8],
                'stop_loss_pct': [0.4],
                'macd_fast': [4, 5, 6, 7, 8],
                'macd_slow': [10, 12, 13, 15, 17],
                'macd_signal': [4, 5, 6, 7],
                'use_macd_confirmation': [True, False],
                'volume_spike_threshold': [1.5],
            }
        else:
            return self.get_param_grid()


def main():
    parser = argparse.ArgumentParser(
        description='Momentum Scalping Strategy Optimizer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick optimization (48 runs, ~1 hour)
    python optimize_momentum.py --symbol XRP/USDT --period 3m --quick

    # Full optimization (500+ runs, ~8-10 hours)
    python optimize_momentum.py --symbol XRP/USDT --period 6m

    # Focus on indicator settings
    python optimize_momentum.py --symbol XRP/USDT --period 3m --focus indicators

    # Focus on risk/reward ratios
    python optimize_momentum.py --symbol BTC/USDT --period 3m --focus risk_reward

    # Focus on volume confirmation
    python optimize_momentum.py --symbol XRP/USDT --period 3m --focus volume
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
                        choices=['indicators', 'risk_reward', 'timing', 'volume', 'macd'],
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
        strategy_name='momentum_scalping',
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
    optimizer = MomentumScalpingOptimizer(config, quick_mode=args.quick)

    # Use focused grid if specified
    if args.focus:
        optimizer.get_param_grid = lambda: optimizer.get_focused_grid(args.focus)

    # Show what we're about to do
    total_combinations = optimizer.count_combinations()
    print(f"\nMomentum Scalping Strategy Optimizer")
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
