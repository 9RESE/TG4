#!/usr/bin/env python3
"""
Optimization Results Analyzer.

Analyzes and compares optimization results to find best parameters.
Can compare across multiple runs and generate consolidated reports.

Usage:
    python analyze_results.py                  # Analyze all results
    python analyze_results.py --strategy ema9  # Analyze specific strategy
    python analyze_results.py --compare        # Compare best params across symbols
"""

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import csv


def load_all_reports(results_dir: str) -> List[Dict[str, Any]]:
    """Load all JSON report files from the results directory."""
    reports = []
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"Results directory not found: {results_dir}")
        return []

    for report_file in results_path.glob("*_report.json"):
        try:
            with open(report_file) as f:
                data = json.load(f)
                data['_file'] = str(report_file)
                reports.append(data)
        except Exception as e:
            print(f"Error loading {report_file}: {e}")

    return reports


def load_all_runs(results_dir: str) -> List[Dict[str, Any]]:
    """Load all CSV run files and return as list of dicts."""
    runs = []
    results_path = Path(results_dir)

    for csv_file in results_path.glob("*_runs.csv"):
        try:
            with open(csv_file) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row['_file'] = str(csv_file)
                    # Convert numeric fields
                    for key in ['total_pnl', 'total_pnl_pct', 'win_rate',
                               'max_drawdown_pct', 'profit_factor', 'total_trades']:
                        if key in row and row[key]:
                            try:
                                row[key] = float(row[key])
                            except ValueError:
                                pass
                    runs.append(row)
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")

    return runs


def find_best_params(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Find best parameters across all reports."""
    best_by_strategy = defaultdict(list)

    for report in reports:
        strategy = report.get('strategy', 'unknown')
        symbol = report.get('symbol', 'unknown')

        if 'best_by_pnl' in report and report['best_by_pnl']:
            best = report['best_by_pnl']
            best_by_strategy[strategy].append({
                'symbol': symbol,
                'params': best.get('params', {}),
                'total_pnl': best.get('total_pnl', 0),
                'total_pnl_pct': best.get('total_pnl_pct', 0),
                'win_rate': best.get('win_rate', 0),
                'max_drawdown_pct': best.get('max_drawdown_pct', 0),
                'profit_factor': best.get('profit_factor', 0),
                'total_trades': best.get('total_trades', 0),
                'timestamp': report.get('timestamp', ''),
                'file': report.get('_file', ''),
            })

    return dict(best_by_strategy)


def compare_parameter_values(runs: List[Dict[str, Any]], param_name: str) -> Dict[str, Any]:
    """Analyze performance across different values of a parameter."""
    value_performance = defaultdict(list)

    for run in runs:
        # Find param value
        param_key = f'param_{param_name}'
        if param_key not in run:
            continue

        value = run[param_key]
        pnl = run.get('total_pnl', 0)

        if isinstance(pnl, (int, float)):
            value_performance[str(value)].append(pnl)

    # Calculate statistics
    results = {}
    for value, pnls in value_performance.items():
        results[value] = {
            'avg_pnl': sum(pnls) / len(pnls) if pnls else 0,
            'min_pnl': min(pnls) if pnls else 0,
            'max_pnl': max(pnls) if pnls else 0,
            'count': len(pnls),
        }

    return dict(sorted(results.items(), key=lambda x: x[1]['avg_pnl'], reverse=True))


def print_strategy_summary(strategy: str, results: List[Dict]) -> None:
    """Print summary for a strategy."""
    print(f"\n{'=' * 70}")
    print(f"STRATEGY: {strategy}")
    print(f"{'=' * 70}")

    if not results:
        print("  No results found.")
        return

    # Sort by P&L
    sorted_results = sorted(results, key=lambda x: x.get('total_pnl', 0), reverse=True)

    print(f"\n{'Symbol':<12} {'P&L':>10} {'P&L%':>8} {'WinRate':>8} {'MaxDD%':>8} {'PF':>8} {'Trades':>8}")
    print("-" * 70)

    for r in sorted_results:
        pf = r.get('profit_factor', 0)
        pf_str = f"{pf:.2f}" if pf < 100 else "99+"
        print(f"{r['symbol']:<12} ${r.get('total_pnl', 0):>9.2f} {r.get('total_pnl_pct', 0):>7.1f}% "
              f"{r.get('win_rate', 0):>7.1f}% {r.get('max_drawdown_pct', 0):>7.1f}% "
              f"{pf_str:>8} {r.get('total_trades', 0):>8}")

    # Show best params
    best = sorted_results[0]
    print(f"\nBest performing configuration ({best['symbol']}):")
    print(f"  Parameters: {json.dumps(best.get('params', {}), indent=4)}")


def print_parameter_analysis(runs: List[Dict], strategy: str) -> None:
    """Print parameter sensitivity analysis."""
    # Find all param columns
    param_names = set()
    for run in runs:
        for key in run.keys():
            if key.startswith('param_'):
                param_names.add(key.replace('param_', ''))

    if not param_names:
        print("  No parameters found to analyze.")
        return

    print(f"\n{'=' * 70}")
    print(f"PARAMETER SENSITIVITY ANALYSIS: {strategy}")
    print(f"{'=' * 70}")

    for param in sorted(param_names):
        analysis = compare_parameter_values(runs, param)
        if not analysis:
            continue

        print(f"\n{param}:")
        print(f"  {'Value':<15} {'Avg P&L':>12} {'Min':>10} {'Max':>10} {'Runs':>6}")
        print(f"  {'-' * 55}")

        for value, stats in list(analysis.items())[:5]:  # Top 5 values
            print(f"  {value:<15} ${stats['avg_pnl']:>11.2f} ${stats['min_pnl']:>9.2f} ${stats['max_pnl']:>9.2f} {stats['count']:>6}")


def generate_consolidated_report(
    results_dir: str,
    output_file: Optional[str] = None,
) -> str:
    """Generate consolidated report from all optimization runs."""
    reports = load_all_reports(results_dir)
    runs = load_all_runs(results_dir)

    best_params = find_best_params(reports)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    consolidated = {
        'timestamp': timestamp,
        'total_reports_analyzed': len(reports),
        'total_runs_analyzed': len(runs),
        'best_params_by_strategy': {},
        'recommendations': [],
    }

    # Process each strategy
    for strategy, strategy_results in best_params.items():
        consolidated['best_params_by_strategy'][strategy] = {}

        for result in strategy_results:
            symbol = result['symbol']
            consolidated['best_params_by_strategy'][strategy][symbol] = {
                'params': result['params'],
                'metrics': {
                    'total_pnl': result['total_pnl'],
                    'total_pnl_pct': result['total_pnl_pct'],
                    'win_rate': result['win_rate'],
                    'max_drawdown_pct': result['max_drawdown_pct'],
                    'profit_factor': result['profit_factor'],
                    'total_trades': result['total_trades'],
                },
            }

            # Generate recommendations
            if result['total_pnl'] > 0 and result['win_rate'] > 50:
                consolidated['recommendations'].append({
                    'strategy': strategy,
                    'symbol': symbol,
                    'recommendation': 'VIABLE',
                    'reason': f"Positive P&L (${result['total_pnl']:.2f}) with {result['win_rate']:.1f}% win rate",
                })
            elif result['total_pnl'] < 0:
                consolidated['recommendations'].append({
                    'strategy': strategy,
                    'symbol': symbol,
                    'recommendation': 'NEEDS_WORK',
                    'reason': f"Negative P&L (${result['total_pnl']:.2f})",
                })

    # Save report
    if output_file is None:
        output_file = Path(results_dir) / f"consolidated_report_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump(consolidated, f, indent=2)

    return str(output_file)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze Optimization Results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze all results in default directory
    python analyze_results.py

    # Analyze specific strategy
    python analyze_results.py --strategy ema9

    # Analyze specific symbol
    python analyze_results.py --symbol XRP/USDT

    # Generate consolidated report
    python analyze_results.py --consolidate

    # Custom results directory
    python analyze_results.py --dir /path/to/results
        """
    )

    parser.add_argument('--dir', type=str, default='optimization_results',
                        help='Results directory (default: optimization_results)')
    parser.add_argument('--strategy', type=str, default=None,
                        help='Filter by strategy name')
    parser.add_argument('--symbol', type=str, default=None,
                        help='Filter by symbol')
    parser.add_argument('--consolidate', action='store_true',
                        help='Generate consolidated JSON report')
    parser.add_argument('--params', action='store_true',
                        help='Show parameter sensitivity analysis')

    args = parser.parse_args()

    # Load data
    reports = load_all_reports(args.dir)
    runs = load_all_runs(args.dir)

    if not reports and not runs:
        print(f"No optimization results found in {args.dir}")
        print("Run optimization first: python run_optimization.py --quick")
        return

    print(f"\nOptimization Results Analysis")
    print(f"=" * 70)
    print(f"Reports found: {len(reports)}")
    print(f"Individual runs: {len(runs)}")

    # Filter by strategy if specified
    if args.strategy:
        reports = [r for r in reports if r.get('strategy') == args.strategy or args.strategy in r.get('_file', '')]
        runs = [r for r in runs if r.get('strategy') == args.strategy or args.strategy in str(r.get('_file', ''))]

    # Filter by symbol if specified
    if args.symbol:
        symbol_normalized = args.symbol.replace('/', '_')
        reports = [r for r in reports if r.get('symbol') == args.symbol or symbol_normalized in r.get('_file', '')]
        runs = [r for r in runs if r.get('symbol') == args.symbol or symbol_normalized in str(r.get('_file', ''))]

    # Find best params
    best_params = find_best_params(reports)

    # Print summaries
    for strategy, results in best_params.items():
        print_strategy_summary(strategy, results)

    # Parameter analysis
    if args.params and runs:
        strategies = set(r.get('strategy', 'unknown') for r in runs)
        for strategy in strategies:
            strategy_runs = [r for r in runs if r.get('strategy') == strategy]
            print_parameter_analysis(strategy_runs, strategy)

    # Consolidated report
    if args.consolidate:
        report_path = generate_consolidated_report(args.dir)
        print(f"\n{'=' * 70}")
        print(f"Consolidated report saved: {report_path}")

    print(f"\n{'=' * 70}")
    print("Analysis complete!")


if __name__ == '__main__':
    main()
