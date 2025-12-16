#!/usr/bin/env python3
"""
Batch Optimization Runner.

Runs multiple optimization jobs sequentially, one strategy/symbol at a time.
Memory is released between jobs by running each as a subprocess.

Usage:
    # Run quick optimization for all strategies on XRP/USDT
    python run_optimization.py --quick

    # Run specific strategy and symbol
    python run_optimization.py --strategy ema9 --symbol BTC/USDT

    # Run all strategies for all symbols (full optimization)
    python run_optimization.py --full

    # Run with custom period
    python run_optimization.py --period 6m --quick
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Configuration - All 9 Strategies
STRATEGIES = {
    # === Trend-Following Strategies ===
    'ema9': {
        'script': 'optimize_ema9.py',
        'symbols': ['BTC/USDT'],  # EMA-9 is optimized for BTC on 1H
        'description': 'EMA-9 Trend Flip Strategy (1H timeframe)',
    },
    'wavetrend': {
        'script': 'optimize_wavetrend.py',
        'symbols': ['XRP/USDT', 'BTC/USDT', 'XRP/BTC'],
        'description': 'WaveTrend Oscillator Strategy (5m timeframe)',
    },

    # === Scalping Strategies ===
    'momentum': {
        'script': 'optimize_momentum.py',
        'symbols': ['XRP/USDT', 'BTC/USDT', 'XRP/BTC'],
        'description': 'Momentum Scalping Strategy (1m timeframe)',
    },

    # === Mean Reversion Strategies ===
    'mean_reversion': {
        'script': 'optimize_mean_reversion.py',
        'symbols': ['XRP/USDT', 'BTC/USDT', 'XRP/BTC'],
        'description': 'Mean Reversion Strategy (RSI + Bollinger)',
    },
    'grid_rsi': {
        'script': 'optimize_grid_rsi.py',
        'symbols': ['XRP/USDT', 'BTC/USDT', 'XRP/BTC'],
        'description': 'Grid RSI Reversion Strategy',
    },

    # === Sentiment/Flow Strategies ===
    'whale_sentiment': {
        'script': 'optimize_whale_sentiment.py',
        'symbols': ['XRP/USDT', 'BTC/USDT'],  # XRP/BTC disabled by default
        'description': 'Whale Sentiment Strategy (contrarian)',
    },
    'order_flow': {
        'script': 'optimize_order_flow.py',
        'symbols': ['XRP/USDT', 'BTC/USDT', 'XRP/BTC'],
        'description': 'Order Flow Strategy (VPIN + imbalance)',
    },

    # === Market Making ===
    'market_making': {
        'script': 'optimize_market_making.py',
        'symbols': ['XRP/USDT', 'BTC/USDT', 'XRP/BTC'],
        'description': 'Market Making Strategy (spread capture)',
    },

    # === Ratio/Pair Trading ===
    'ratio_trading': {
        'script': 'optimize_ratio_trading.py',
        'symbols': ['XRP/BTC'],  # XRP/BTC only
        'description': 'Ratio Trading Strategy (XRP/BTC pair)',
    },
}

# Default symbols if not specified per-strategy
DEFAULT_SYMBOLS = ['XRP/USDT', 'BTC/USDT', 'XRP/BTC']


def run_optimization_job(
    strategy: str,
    symbol: str,
    period: str,
    db_url: str,
    output_dir: str,
    quick: bool = False,
    focus: str = None,
    timeout: int = 600,
    parallel: bool = False,
    workers: int = 8,
    chunk_size: int = 50,
) -> Dict[str, Any]:
    """
    Run a single optimization job as subprocess.

    Returns:
        Dict with job results including status, output path, and timing.
    """
    script_dir = Path(__file__).parent
    script_path = script_dir / STRATEGIES[strategy]['script']

    if not script_path.exists():
        return {
            'strategy': strategy,
            'symbol': symbol,
            'status': 'error',
            'error': f"Script not found: {script_path}",
        }

    # Build command
    cmd = [
        sys.executable,
        str(script_path),
        '--symbol', symbol,
        '--period', period,
        '--db-url', db_url,
        '--output', output_dir,
        '--timeout', str(timeout),
    ]

    if quick:
        cmd.append('--quick')

    if focus:
        cmd.extend(['--focus', focus])

    if parallel:
        cmd.append('--parallel')
        cmd.extend(['--workers', str(workers)])
        cmd.extend(['--chunk-size', str(chunk_size)])

    print(f"\n{'=' * 60}")
    print(f"RUNNING: {strategy} on {symbol}")
    print(f"Command: {' '.join(cmd[:6])}...")  # Don't print db url
    print(f"{'=' * 60}\n")

    start_time = datetime.now()

    try:
        # Run with auto-confirmation (pipe 'y' to stdin)
        result = subprocess.run(
            cmd,
            input='y\n',
            capture_output=False,
            text=True,
            timeout=timeout * 100,  # Allow for many runs
        )

        duration = (datetime.now() - start_time).total_seconds()

        return {
            'strategy': strategy,
            'symbol': symbol,
            'status': 'success' if result.returncode == 0 else 'failed',
            'return_code': result.returncode,
            'duration_seconds': duration,
        }

    except subprocess.TimeoutExpired:
        return {
            'strategy': strategy,
            'symbol': symbol,
            'status': 'timeout',
            'error': 'Job timed out',
        }
    except Exception as e:
        return {
            'strategy': strategy,
            'symbol': symbol,
            'status': 'error',
            'error': str(e),
        }


def generate_summary_report(
    results: List[Dict[str, Any]],
    output_dir: str,
) -> str:
    """Generate overall summary report from all optimization jobs."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = Path(output_dir) / f"optimization_batch_report_{timestamp}.json"

    # Collect results from individual runs
    summary = {
        'timestamp': timestamp,
        'total_jobs': len(results),
        'successful_jobs': len([r for r in results if r['status'] == 'success']),
        'failed_jobs': len([r for r in results if r['status'] != 'success']),
        'jobs': results,
        'best_params_by_strategy': {},
    }

    # Try to read individual reports and find best params
    output_path = Path(output_dir)
    for result in results:
        if result['status'] == 'success':
            strategy = result['strategy']
            symbol = result['symbol'].replace('/', '_')

            # Find the most recent report file
            pattern = f"{STRATEGIES[strategy]['script'].replace('optimize_', '').replace('.py', '')}*_{symbol}_*_report.json"
            report_files = list(output_path.glob(f"*{strategy}*{symbol}*_report.json"))

            if report_files:
                latest_report = max(report_files, key=lambda f: f.stat().st_mtime)
                try:
                    with open(latest_report) as f:
                        report_data = json.load(f)
                        if 'best_by_pnl' in report_data:
                            key = f"{strategy}_{result['symbol']}"
                            summary['best_params_by_strategy'][key] = {
                                'params': report_data['best_by_pnl']['params'],
                                'total_pnl': report_data['best_by_pnl']['total_pnl'],
                                'win_rate': report_data['best_by_pnl']['win_rate'],
                                'report_file': str(latest_report),
                            }
                except Exception:
                    pass

    # Save summary
    with open(report_path, 'w') as f:
        json.dump(summary, f, indent=2)

    return str(report_path)


def print_available_jobs():
    """Print available strategy/symbol combinations."""
    print("\nAvailable Optimization Jobs:")
    print("=" * 60)
    for name, config in STRATEGIES.items():
        print(f"\n{name}: {config['description']}")
        print(f"  Symbols: {', '.join(config['symbols'])}")
        print(f"  Script: {config['script']}")


def main():
    parser = argparse.ArgumentParser(
        description='Batch Strategy Parameter Optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick optimization for all strategies on their default symbols
    python run_optimization.py --quick

    # Run specific strategy on all its symbols
    python run_optimization.py --strategy wavetrend --quick

    # Run specific strategy on specific symbol
    python run_optimization.py --strategy momentum --symbol XRP/USDT --quick

    # Full optimization (WARNING: takes many hours)
    python run_optimization.py --full

    # Custom period
    python run_optimization.py --period 6m --strategy ema9 --quick

    # List available jobs
    python run_optimization.py --list
        """
    )

    parser.add_argument('--strategy', type=str, default=None,
                        choices=list(STRATEGIES.keys()),
                        help='Specific strategy to optimize')
    parser.add_argument('--symbol', type=str, default=None,
                        help='Specific symbol to optimize')
    parser.add_argument('--period', type=str, default='3m',
                        help='Backtest period (default: 3m)')
    parser.add_argument('--db-url', type=str, default=os.getenv('DATABASE_URL'),
                        help='Database URL (or set DATABASE_URL env var)')
    parser.add_argument('--output', type=str, default='optimization_results',
                        help='Output directory (default: optimization_results)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode with reduced parameter grids')
    parser.add_argument('--full', action='store_true',
                        help='Full optimization (all strategies, all symbols)')
    parser.add_argument('--focus', type=str, default=None,
                        help='Focus optimization (strategy-specific)')
    parser.add_argument('--timeout', type=int, default=600,
                        help='Timeout per run in seconds (default: 600)')
    parser.add_argument('--parallel', action='store_true',
                        help='Enable parallel execution for each strategy')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers (default: 8)')
    parser.add_argument('--chunk-size', type=int, default=50,
                        help='Batch size for parallel processing (default: 50)')
    parser.add_argument('--list', action='store_true',
                        help='List available optimization jobs')

    args = parser.parse_args()

    if args.list:
        print_available_jobs()
        sys.exit(0)

    if not args.db_url:
        print("ERROR: Database URL required. Set DATABASE_URL or use --db-url")
        print("Example: DATABASE_URL=postgresql://trading:password@localhost:5433/kraken_data")
        sys.exit(1)

    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Determine jobs to run
    jobs: List[Tuple[str, str]] = []

    if args.strategy and args.symbol:
        # Specific strategy and symbol
        jobs.append((args.strategy, args.symbol))
    elif args.strategy:
        # Specific strategy, all its symbols
        for symbol in STRATEGIES[args.strategy]['symbols']:
            jobs.append((args.strategy, symbol))
    elif args.symbol:
        # Specific symbol, all strategies that support it
        for name, config in STRATEGIES.items():
            if args.symbol in config['symbols']:
                jobs.append((name, args.symbol))
    elif args.full:
        # All strategies, all their symbols
        for name, config in STRATEGIES.items():
            for symbol in config['symbols']:
                jobs.append((name, symbol))
    else:
        # Default: all strategies, their default first symbol only
        for name, config in STRATEGIES.items():
            jobs.append((name, config['symbols'][0]))

    # Show planned jobs
    print(f"\nOptimization Batch Runner")
    print(f"=" * 60)
    print(f"Jobs to run: {len(jobs)}")
    print(f"Period: {args.period}")
    print(f"Mode: {'Quick' if args.quick else 'Full'}")
    if args.parallel:
        print(f"Execution: PARALLEL ({args.workers} workers per job)")
    else:
        print(f"Execution: SEQUENTIAL")
    print(f"Output: {args.output}/")
    print(f"\nPlanned jobs:")
    for i, (strategy, symbol) in enumerate(jobs, 1):
        print(f"  {i}. {strategy} on {symbol}")

    print(f"\n{'=' * 60}")

    # Confirm
    response = input("\nProceed with batch optimization? [y/N]: ")
    if response.lower() != 'y':
        print("Cancelled.")
        sys.exit(0)

    # Run jobs
    results = []
    for i, (strategy, symbol) in enumerate(jobs, 1):
        print(f"\n[{i}/{len(jobs)}] Starting {strategy} on {symbol}...")

        result = run_optimization_job(
            strategy=strategy,
            symbol=symbol,
            period=args.period,
            db_url=args.db_url,
            output_dir=args.output,
            quick=args.quick,
            focus=args.focus,
            timeout=args.timeout,
            parallel=args.parallel,
            workers=args.workers,
            chunk_size=args.chunk_size,
        )

        results.append(result)

        # Print status
        if result['status'] == 'success':
            print(f"\n[{i}/{len(jobs)}] COMPLETED: {strategy} on {symbol}")
            print(f"  Duration: {result.get('duration_seconds', 0):.1f}s")
        else:
            print(f"\n[{i}/{len(jobs)}] FAILED: {strategy} on {symbol}")
            print(f"  Status: {result['status']}")
            if 'error' in result:
                print(f"  Error: {result['error']}")

    # Generate summary report
    report_path = generate_summary_report(results, args.output)

    # Final summary
    print(f"\n{'=' * 60}")
    print("BATCH OPTIMIZATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total jobs: {len(results)}")
    print(f"Successful: {len([r for r in results if r['status'] == 'success'])}")
    print(f"Failed: {len([r for r in results if r['status'] != 'success'])}")
    print(f"\nSummary report: {report_path}")
    print(f"Results directory: {args.output}/")


if __name__ == '__main__':
    main()
