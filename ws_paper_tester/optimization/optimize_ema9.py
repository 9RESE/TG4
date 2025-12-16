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
        Return parameter grid for EMA-9 strategy v2.0.

        Quick mode: ~72 combinations
        Full mode: ~500+ combinations

        v2.0 NEW parameters:
        - strict_candle_mode: Require whole candle above/below EMA
        - entry_clearance_pct: Min clearance from EMA for entry quality
        - exit_confirmation_candles: Confirmation before exit (anti-whipsaw)
        """
        if self.quick_mode:
            # Quick grid - most impactful parameters only
            # NOTE: No take_profit_pct (flip IS the profit exit)
            # NOTE: No max_hold_hours (hold until flip - that's the strategy)
            return {
                'ema_period': [9],
                'consecutive_candles': [1, 2, 3],  # 1=immediate flip, 2-3=confirmation
                'buffer_pct': [0.0],               # Use strict mode instead of buffer

                # v2.0: Strict candle mode is REQUIRED (proven superior)
                # - 50% win rate vs 12.6% in legacy mode
                # - 0.48% max DD vs 50.44% in legacy mode
                'strict_candle_mode': [True],

                # v2.0 NEW: Exit confirmation
                'exit_confirmation_candles': [1, 2],  # 1=immediate, 2=require confirmation

                # Risk Management - ATR-based stops adapt to volatility
                'stop_loss_pct': [2.0, 3.0],      # Fallback if ATR unavailable
                'use_atr_stops': [True],          # ATR-based stops (recommended)

                # Timeframe - ONLY supported timeframes (5m, 1h, 1d)
                # NOTE: 15m is NOT supported by backtest_runner
                'candle_timeframe_minutes': [5, 60],  # 5m and 1h
            }
        else:
            # Full grid - comprehensive search
            # NOTE: No take_profit_pct (flip IS the profit exit)
            # NOTE: No max_hold_hours (hold until flip - that's the strategy)
            return {
                # EMA Settings
                'ema_period': [7, 9, 12, 21],
                'consecutive_candles': [1, 2, 3, 4],  # 1=immediate flip
                'buffer_pct': [0.0],                  # Use strict mode instead of buffer

                # v2.0: Strict candle mode is REQUIRED (proven superior)
                # Optimization results: 50% win rate vs 12.6%, 0.48% DD vs 50.44%
                'strict_candle_mode': [True],

                # v2.0 NEW: Exit confirmation
                'exit_confirmation_candles': [1, 2, 3],

                # Timeframe - ONLY supported timeframes
                # Supported: 1m, 5m, 60m (1h), 1440m (1d)
                # NOT supported: 15m, 240m (4h)
                'candle_timeframe_minutes': [5, 60, 1440],  # 5m, 1h, 1d

                # Risk Management - ATR adapts to volatility, % is fallback
                'stop_loss_pct': [2.0, 2.5, 3.0, 4.0],  # Fallback if ATR unavailable
                'use_atr_stops': [True, False],          # ATR-based vs percentage stops
                'atr_stop_mult': [2.0, 2.5],             # ATR multiplier options

                # Cooldowns
                'cooldown_minutes': [15, 30, 60],
            }

    def get_focused_grid(self, focus: str) -> Dict[str, List[Any]]:
        """
        Get focused grids for specific optimization goals.

        NOTE: No take_profit_pct (flip IS the profit exit)
        NOTE: No exit_on_flip toggle (always exit on flip - it's the strategy)

        v2.0: All focused grids now include strict_candle_mode

        Args:
            focus: 'signal_quality', 'stop_loss', 'entry_timing', 'strict_mode', 'timeframes'
        """
        if focus == 'signal_quality':
            # Focus on EMA settings that affect signal quality
            return {
                'ema_period': [7, 9, 12, 14, 21],
                'consecutive_candles': [2, 3, 4, 5],
                'buffer_pct': [0.0],
                'strict_candle_mode': [True],     # v2.0: Required (proven superior)
                'entry_clearance_pct': [0.0],     # Strict mode already filters enough
                'exit_confirmation_candles': [1, 2],
                'stop_loss_pct': [2.5],           # Fixed wide stop for this test
                'use_atr_stops': [True],
                'candle_timeframe_minutes': [60], # 1h default
            }
        elif focus == 'stop_loss':
            # Focus on stop loss optimization (protection settings)
            # NOTE: Stop is for protection only - flip is the profit exit
            return {
                'ema_period': [9],
                'consecutive_candles': [2],
                'buffer_pct': [0.0],
                'strict_candle_mode': [True],     # v2.0: Required
                'entry_clearance_pct': [0.0],     # Strict mode already filters
                'exit_confirmation_candles': [2], # Use confirmation
                'stop_loss_pct': [1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
                'use_atr_stops': [True, False],
                'atr_stop_mult': [1.5, 2.0, 2.5, 3.0],
                'candle_timeframe_minutes': [60], # 1h default
            }
        elif focus == 'entry_timing':
            # Focus on cooldown and consecutive candle optimization
            return {
                'ema_period': [9],
                'consecutive_candles': [2, 3, 4, 5],
                'buffer_pct': [0.0],
                'strict_candle_mode': [True],     # v2.0: Required
                'entry_clearance_pct': [0.0],     # Strict mode already filters
                'exit_confirmation_candles': [1, 2, 3],
                'stop_loss_pct': [2.5],           # Fixed wide stop for this test
                'use_atr_stops': [True],
                'cooldown_minutes': [0, 15, 30, 60, 120],
                'candle_timeframe_minutes': [60], # 1h default
            }
        elif focus == 'strict_mode':
            # DEPRECATED: Strict mode is now REQUIRED (proven superior)
            # This grid now focuses on exit confirmation tuning
            return {
                'ema_period': [9],
                'consecutive_candles': [2, 3],
                'buffer_pct': [0.0],
                'strict_candle_mode': [True],     # Required - no longer comparing
                'entry_clearance_pct': [0.0],     # Strict mode already filters
                'exit_confirmation_candles': [1, 2, 3],
                'stop_loss_pct': [2.0, 2.5, 3.0],
                'use_atr_stops': [True],
                'candle_timeframe_minutes': [5, 60, 1440],  # All supported TFs
            }
        elif focus == 'timeframes':
            # Focus on different SUPPORTED timeframes only
            # Supported: 1m, 5m, 60m (1h), 1440m (1d)
            # NOT supported: 15m, 30m, 240m (4h)
            return {
                'ema_period': [9],
                'consecutive_candles': [2, 3],
                'buffer_pct': [0.0],
                'strict_candle_mode': [True],     # Required
                'entry_clearance_pct': [0.0],     # Strict mode already filters
                'exit_confirmation_candles': [1, 2],
                'stop_loss_pct': [2.0, 3.0],
                'use_atr_stops': [True],
                'candle_timeframe_minutes': [5, 60, 1440],  # Supported TFs only
            }
        else:
            return self.get_param_grid()


def main():
    parser = argparse.ArgumentParser(
        description='EMA-9 Trend Flip Strategy Optimizer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick optimization (~54 runs)
    python optimize_ema9.py --symbol BTC/USDT --period 3m --quick

    # Full optimization (many runs)
    python optimize_ema9.py --symbol BTC/USDT --period 6m

    # Focused optimization on signal quality
    python optimize_ema9.py --symbol BTC/USDT --period 3m --focus signal_quality

    # Focused optimization on stop loss settings (protection only)
    python optimize_ema9.py --symbol BTC/USDT --period 3m --focus stop_loss

Note: The EMA flip IS the profit exit. There is no take_profit parameter.
      Stop loss is for catastrophic protection only.
        """
    )

    parser.add_argument('--symbol', type=str, default='BTC/USDT',
                        help='Trading pair to optimize (default: BTC/USDT)')
    parser.add_argument('--period', type=str, default='6m',
                        help='Backtest period: 1w, 2w, 1m, 3m, 6m, 1y (default: 6m for better signal coverage)')
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
                        choices=['signal_quality', 'stop_loss', 'entry_timing', 'strict_mode', 'timeframes'],
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
        # Parallel mode: estimate ~45s per run with workers (includes DB load overhead)
        # Adjust for period length: 1m=0.5x, 1w=0.5x, 2w=0.7x, 3m=1x, 6m=1.5x, 1y=2x
        period_multiplier = {'1w': 0.5, '2w': 0.7, '1m': 0.5, '3m': 1.0, '6m': 1.5, '1y': 2.0}.get(args.period, 1.0)
        est_time = (total_combinations / args.workers) * 45 * period_multiplier / 60
        print(f"Execution: PARALLEL ({args.workers} workers, chunk={args.chunk_size})")
        print(f"Estimated time: ~{est_time:.0f} minutes")
    else:
        # Sequential mode: ~90s per run
        period_multiplier = {'1w': 0.5, '2w': 0.7, '1m': 0.5, '3m': 1.0, '6m': 1.5, '1y': 2.0}.get(args.period, 1.0)
        print(f"Execution: SEQUENTIAL")
        print(f"Estimated time: ~{total_combinations * 1.5 * period_multiplier:.0f} minutes")
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
