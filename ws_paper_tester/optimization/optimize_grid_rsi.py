#!/usr/bin/env python3
"""
Grid RSI Reversion Strategy Optimizer.

Optimizes parameters for the grid RSI reversion strategy.
Runs backtests in subprocesses for memory isolation.

Usage:
    python optimize_grid_rsi.py --symbol XRP/USDT --period 3m
    python optimize_grid_rsi.py --symbol XRP/USDT --period 6m --quick
    python optimize_grid_rsi.py --symbol BTC/USDT --db-url "postgresql://..."
    python optimize_grid_rsi.py --symbol XRP/USDT --focus grid --parallel
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from optimization.base_optimizer import BaseOptimizer, OptimizationConfig


class GridRSIOptimizer(BaseOptimizer):
    """
    Optimizer for Grid RSI Reversion Strategy.

    Key tunable parameters:
    - num_grids: Number of grid levels (8-25)
    - grid_spacing_pct: Spacing between levels (0.5-3.0%)
    - range_pct: Grid range from center (5-15%)
    - rsi_period: RSI calculation period (10-20)
    - rsi_oversold/overbought: RSI thresholds
    - stop_loss_pct: Stop loss percentage (3-15%)
    - adx_threshold: Trend filter threshold (20-40)
    - candle_timeframe_minutes: Candle timeframe (5, 60, 1440)
    """

    def __init__(
        self,
        config: OptimizationConfig,
        quick_mode: bool = False,
        timeframes: List[int] = None
    ):
        super().__init__(config)
        self.quick_mode = quick_mode
        # Allow timeframe override from command line
        self.timeframes = timeframes  # None means use defaults

    def get_param_grid(self) -> Dict[str, List[Any]]:
        """
        Return parameter grid for Grid RSI strategy.

        Quick mode: ~108 combinations
        Full mode: ~2000+ combinations

        Key parameters:
        - Grid structure: num_grids, grid_spacing_pct, range_pct
        - RSI settings: rsi_period, rsi_oversold, rsi_overbought
        - Risk management: stop_loss_pct, max_accumulation_levels, adx_threshold
        - Timeframe: candle_timeframe_minutes (5m, 1h, 1d)
        """
        # Use CLI timeframe override if provided, otherwise use defaults
        quick_timeframes = self.timeframes if self.timeframes else [5, 60]
        full_timeframes = self.timeframes if self.timeframes else [5, 60, 1440]

        if self.quick_mode:
            # Quick grid - ~108 combinations for faster iteration
            # 3 * 3 * 2 * 2 * 3 * 1 = 108 combinations
            return {
                # Grid structure (key parameters to vary)
                'num_grids': [10, 15, 20],
                'grid_spacing_pct': [1.0, 1.5, 2.0],

                # RSI settings (key parameters to vary)
                'rsi_oversold': [25, 30],
                'rsi_overbought': [70, 75],

                # Risk management (key parameter to vary)
                'stop_loss_pct': [5.0, 8.0, 10.0],

                # Fixed defaults for quick mode
                'range_pct': [7.5],
                'rsi_period': [14],
                'max_accumulation_levels': [5],
                'adx_threshold': [30],
                'use_adaptive_rsi': [True],
                'use_atr_spacing': [False],

                # Single timeframe for quick mode (use first from CLI or default to 5m)
                'candle_timeframe_minutes': [quick_timeframes[0]] if quick_timeframes else [5],
            }
        else:
            # Full grid - comprehensive parameter search
            return {
                # Grid structure
                'num_grids': [8, 10, 12, 15, 18, 20],
                'grid_spacing_pct': [0.8, 1.0, 1.2, 1.5, 2.0, 2.5],
                'range_pct': [5.0, 7.5, 10.0, 12.5],

                # RSI settings
                'rsi_period': [10, 12, 14, 16, 18],
                'rsi_oversold': [20, 25, 30, 35],
                'rsi_overbought': [65, 70, 75, 80],
                'rsi_extreme_multiplier': [1.0, 1.2, 1.3],

                # Risk management
                'stop_loss_pct': [3.0, 5.0, 8.0, 10.0],
                'max_accumulation_levels': [3, 4, 5, 6],
                'adx_threshold': [25, 30, 35],

                # Adaptive features
                'use_adaptive_rsi': [True, False],
                'use_atr_spacing': [True, False],

                # Timeframe - use CLI override or default
                'candle_timeframe_minutes': full_timeframes,
            }

    def get_focused_grid(self, focus: str) -> Dict[str, List[Any]]:
        """
        Get focused grids for specific optimization goals.

        Args:
            focus: 'grid', 'rsi', 'risk', 'timeframes', 'adaptive'

        Returns:
            Parameter grid focused on specific aspect
        """
        # Use CLI timeframe override if provided
        default_single_tf = self.timeframes if self.timeframes else [5]
        default_all_tf = self.timeframes if self.timeframes else [5, 60, 1440]

        if focus == 'grid':
            # Focus on grid structure parameters
            return {
                'num_grids': [6, 8, 10, 12, 15, 18, 20, 25],
                'grid_spacing_pct': [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0],
                'range_pct': [5.0, 7.5, 10.0, 12.5, 15.0],
                'rsi_period': [14],
                'rsi_oversold': [30],
                'rsi_overbought': [70],
                'stop_loss_pct': [8.0],
                'max_accumulation_levels': [5],
                'adx_threshold': [30],
                'use_adaptive_rsi': [True],
                'candle_timeframe_minutes': default_single_tf,
            }
        elif focus == 'rsi':
            # Focus on RSI parameter optimization
            return {
                'num_grids': [15],
                'grid_spacing_pct': [1.5],
                'range_pct': [7.5],
                'rsi_period': [8, 10, 12, 14, 16, 18, 20],
                'rsi_oversold': [20, 25, 30, 35, 40],
                'rsi_overbought': [60, 65, 70, 75, 80],
                'rsi_extreme_multiplier': [1.0, 1.2, 1.3, 1.5],
                'stop_loss_pct': [8.0],
                'max_accumulation_levels': [5],
                'adx_threshold': [30],
                'use_adaptive_rsi': [True, False],
                'candle_timeframe_minutes': default_single_tf,
            }
        elif focus == 'risk':
            # Focus on risk management parameters
            return {
                'num_grids': [15],
                'grid_spacing_pct': [1.5],
                'range_pct': [7.5],
                'rsi_period': [14],
                'rsi_oversold': [30],
                'rsi_overbought': [70],
                'stop_loss_pct': [3.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0],
                'max_accumulation_levels': [2, 3, 4, 5, 6, 7],
                'adx_threshold': [20, 25, 30, 35, 40],
                'use_adaptive_rsi': [True],
                'candle_timeframe_minutes': default_single_tf,
            }
        elif focus == 'timeframes':
            # Focus on different timeframes
            return {
                'num_grids': [12, 15, 18],
                'grid_spacing_pct': [1.0, 1.5, 2.0],
                'range_pct': [7.5, 10.0],
                'rsi_period': [14],
                'rsi_oversold': [30],
                'rsi_overbought': [70],
                'stop_loss_pct': [5.0, 8.0],
                'max_accumulation_levels': [4, 5],
                'adx_threshold': [30],
                'use_adaptive_rsi': [True],
                'candle_timeframe_minutes': default_all_tf,
            }
        elif focus == 'adaptive':
            # Focus on adaptive feature combinations
            return {
                'num_grids': [15],
                'grid_spacing_pct': [1.5],
                'range_pct': [7.5],
                'rsi_period': [12, 14, 16],
                'rsi_oversold': [25, 30],
                'rsi_overbought': [70, 75],
                'rsi_zone_expansion': [3, 5, 7],
                'stop_loss_pct': [8.0],
                'max_accumulation_levels': [5],
                'adx_threshold': [30],
                'use_adaptive_rsi': [True, False],
                'use_atr_spacing': [True, False],
                'atr_multiplier': [0.2, 0.3, 0.4],
                'candle_timeframe_minutes': default_single_tf,
            }
        else:
            return self.get_param_grid()


def main():
    parser = argparse.ArgumentParser(
        description='Grid RSI Reversion Strategy Optimizer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick optimization (~108 runs)
    python optimize_grid_rsi.py --symbol XRP/USDT --period 3m --quick

    # Full optimization (many runs)
    python optimize_grid_rsi.py --symbol XRP/USDT --period 6m

    # Focused optimization on grid structure
    python optimize_grid_rsi.py --symbol XRP/USDT --period 3m --focus grid

    # Focused optimization on RSI parameters
    python optimize_grid_rsi.py --symbol XRP/USDT --period 3m --focus rsi

    # Focused optimization on risk management
    python optimize_grid_rsi.py --symbol XRP/USDT --period 3m --focus risk

    # Parallel execution for faster results
    python optimize_grid_rsi.py --symbol BTC/USDT --period 3m --parallel --workers 8

    # Specific timeframes only
    python optimize_grid_rsi.py --symbol XRP/USDT --period 3m --timeframes 5,60

Note: Grid RSI is a mean-reversion strategy. It performs best in ranging markets.
      The ADX threshold filters out trending markets where grids underperform.
        """
    )

    parser.add_argument('--symbol', type=str, default='XRP/USDT',
                        help='Trading pair to optimize (default: XRP/USDT)')
    parser.add_argument('--period', type=str, default='3m',
                        help='Backtest period: 1w, 2w, 1m, 3m, 6m, 12m/1y, 2y, 3y, 4y, 5y, all (default: 3m)')
    parser.add_argument('--start', type=str, default=None,
                        help='Start date (YYYY-MM-DD), overrides --period')
    parser.add_argument('--end', type=str, default=None,
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--db-url', type=str, default=os.getenv('DATABASE_URL'),
                        help='Database URL (or set DATABASE_URL env var)')
    parser.add_argument('--output', type=str, default='optimization_results',
                        help='Output directory (default: optimization_results)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode with reduced parameter grid (~108 combinations)')
    parser.add_argument('--focus', type=str, default=None,
                        choices=['grid', 'rsi', 'risk', 'timeframes', 'adaptive'],
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
    parser.add_argument('--timeframes', type=str, default=None,
                        help='Comma-separated timeframes in minutes (e.g., "5,60,1440" for 5m,1h,1d). '
                             'Supported: 5, 60, 1440. Default: 5,60 in quick mode, 5,60,1440 in full mode')
    parser.add_argument('-y', '--yes', action='store_true',
                        help='Skip confirmation prompt (for scripted use)')

    args = parser.parse_args()

    # Parse timeframes if provided
    timeframe_override = None
    if args.timeframes:
        try:
            timeframe_override = [int(t.strip()) for t in args.timeframes.split(',')]
            # Validate timeframes
            supported = {5, 60, 1440}
            invalid = set(timeframe_override) - supported
            if invalid:
                print(f"ERROR: Unsupported timeframes: {invalid}")
                print(f"Supported timeframes: 5 (5m), 60 (1h), 1440 (1d)")
                sys.exit(1)
        except ValueError:
            print(f"ERROR: Invalid timeframes format: {args.timeframes}")
            print("Use comma-separated integers, e.g., '5,60,1440'")
            sys.exit(1)

    if not args.db_url:
        print("ERROR: Database URL required. Set DATABASE_URL or use --db-url")
        print("Example: DATABASE_URL=postgresql://trading:password@localhost:5433/kraken_data")
        sys.exit(1)

    # Create optimization config
    config = OptimizationConfig(
        strategy_name='grid_rsi_reversion',
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
    optimizer = GridRSIOptimizer(config, quick_mode=args.quick, timeframes=timeframe_override)

    # Use focused grid if specified
    if args.focus:
        optimizer.get_param_grid = lambda: optimizer.get_focused_grid(args.focus)

    # Show what we're about to do
    total_combinations = optimizer.count_combinations()

    # Period-based time estimation multipliers
    period_multiplier = {
        '1w': 0.3, '2w': 0.5, '1m': 0.5, '2m': 0.7, '3m': 1.0,
        '6m': 1.5, '9m': 2.0, '12m': 2.5, '1y': 2.5,
        '2y': 4.0, '3y': 5.0, '4y': 6.0, '5y': 7.0, 'all': 10.0
    }.get(args.period.lower(), 1.0)

    print(f"\nGrid RSI Reversion Strategy Optimizer")
    print(f"=" * 50)
    print(f"Symbol: {args.symbol}")
    print(f"Period: {args.period}")
    print(f"Mode: {'Quick' if args.quick else 'Full'}")
    if args.focus:
        print(f"Focus: {args.focus}")
    print(f"Parameter combinations: {total_combinations}")

    if args.parallel:
        # Parallel mode: estimate ~45s per run with workers
        est_time = (total_combinations / args.workers) * 45 * period_multiplier / 60
        print(f"Execution: PARALLEL ({args.workers} workers, chunk={args.chunk_size})")
        print(f"Estimated time: ~{est_time:.0f} minutes")
    else:
        # Sequential mode: ~90s per run
        est_time = total_combinations * 1.5 * period_multiplier
        print(f"Execution: SEQUENTIAL")
        print(f"Estimated time: ~{est_time:.0f} minutes")

    print(f"Output: {args.output}/")
    print(f"=" * 50)

    # Confirm with user (skip if --yes flag)
    if not args.yes:
        response = input("\nProceed with optimization? [y/N]: ")
        if response.lower() != 'y':
            print("Cancelled.")
            sys.exit(0)

    # Run optimization
    results = optimizer.run_optimization()

    print(f"\nOptimization complete! Check {args.output}/ for results.")


if __name__ == '__main__':
    main()
