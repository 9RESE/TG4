#!/usr/bin/env python3
"""
Phase 31: Unified Trading Platform with Dual Portfolio Support
Single entry point for all trading strategies with experiment support.

NEW: Dual Portfolio Mode
- 70% USDT Accumulation: Quick trades, profit in USDT
- 30% Crypto Accumulation: Long-term BTC/XRP holding

Usage:
    # Paper trading with default config
    python unified_trader.py paper

    # DUAL PORTFOLIO MODE (70/30 split, $10,000 starting)
    python unified_trader.py dual
    python unified_trader.py dual --preset usdt_aggressive
    python unified_trader.py dual --preset crypto_aggressive

    # Paper trading with custom duration
    python unified_trader.py paper --duration 120 --interval 60

    # Run an experiment with parameter overrides
    python unified_trader.py experiment --preset aggressive

    # List available strategies
    python unified_trader.py list

    # Enable/disable strategies
    python unified_trader.py config --enable mean_reversion_vwap --disable grid_arithmetic

    # Create new config template
    python unified_trader.py init-config

    # Analyze previous experiment logs
    python unified_trader.py analyze --experiment exp_20251212_123456

    # Compare experiments
    python unified_trader.py compare exp_1 exp_2 exp_3
"""

import argparse
import os
import sys
import json
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# Ensure src is in path
sys.path.insert(0, os.path.dirname(__file__))

from unified_orchestrator import UnifiedOrchestrator
from strategy_registry import StrategyRegistry, create_unified_config_template
from portfolio import Portfolio


DEFAULT_CONFIG_PATH = "strategies_config/unified.yaml"
DUAL_PORTFOLIO_CONFIG_PATH = "strategies_config/dual_portfolio.yaml"


# Dual portfolio presets
DUAL_PRESETS = {
    'balanced': {
        'description': 'Default 70/30 split - balanced approach',
        'usdt_pct': 0.70,
        'crypto_pct': 0.30,
    },
    'usdt_aggressive': {
        'description': 'Maximize USDT accumulation (85/15)',
        'usdt_pct': 0.85,
        'crypto_pct': 0.15,
    },
    'crypto_aggressive': {
        'description': 'Maximize crypto accumulation during bear market (50/50)',
        'usdt_pct': 0.50,
        'crypto_pct': 0.50,
    },
    'usdt_only': {
        'description': '100% USDT trading - no crypto holding',
        'usdt_pct': 1.0,
        'crypto_pct': 0.0,
    },
    'crypto_only': {
        'description': '100% crypto accumulation - DCA only',
        'usdt_pct': 0.0,
        'crypto_pct': 1.0,
    },
}


def cmd_paper(args):
    """Run paper trading mode."""
    print("\n" + "="*70)
    print("UNIFIED TRADING PLATFORM - Paper Trading Mode")
    print("="*70)

    # Load config
    config_path = args.config or DEFAULT_CONFIG_PATH
    if not os.path.exists(config_path):
        print(f"Config not found: {config_path}")
        print("Creating default config...")
        create_unified_config_template(config_path)

    # Load starting balance from config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    starting_balance = config.get('global', {}).get('starting_balance', {
        'USDT': 2000.0,
        'XRP': 0.0,
        'BTC': 0.0
    })

    portfolio = Portfolio(starting_balance)

    # Create orchestrator
    experiment_id = args.experiment_id or f"paper_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    orchestrator = UnifiedOrchestrator(
        portfolio=portfolio,
        config_path=config_path,
        experiment_id=experiment_id
    )

    # Apply any CLI overrides
    if args.enable:
        for name in args.enable:
            orchestrator.add_strategy(name)
            print(f"Enabled: {name}")

    if args.disable:
        for name in args.disable:
            orchestrator.remove_strategy(name)
            print(f"Disabled: {name}")

    # Run trading loop
    orchestrator.run_loop(
        duration_minutes=args.duration,
        interval_seconds=args.interval
    )


def cmd_dual(args):
    """Run dual portfolio mode - 70% USDT accumulation / 30% Crypto accumulation."""
    print("\n" + "="*70)
    print("UNIFIED TRADING PLATFORM - Dual Portfolio Mode")
    print("="*70)

    # Load dual portfolio config
    config_path = args.config or DUAL_PORTFOLIO_CONFIG_PATH
    if not os.path.exists(config_path):
        print(f"Dual portfolio config not found: {config_path}")
        print("Please ensure dual_portfolio.yaml exists in strategies_config/")
        return

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Get starting balance (default $10,000 USDT)
    starting_balance = config.get('global', {}).get('starting_balance', {
        'USDT': 10000.0,
        'XRP': 0.0,
        'BTC': 0.0
    })

    # Apply preset if specified
    preset_name = args.preset or 'balanced'
    preset = DUAL_PRESETS.get(preset_name, DUAL_PRESETS['balanced'])

    usdt_pct = preset['usdt_pct']
    crypto_pct = preset['crypto_pct']

    # Phase 32: Per-strategy isolated portfolios
    global_config = config.get('global', {})
    use_isolated_portfolios = global_config.get('use_isolated_portfolios', True)
    per_strategy_capital = global_config.get('per_strategy_capital', 1000.0)

    print(f"\nPreset: {preset_name}")
    print(f"Description: {preset['description']}")
    print(f"Starting Balance: ${starting_balance.get('USDT', 10000):,.2f} USDT")
    print(f"\nPortfolio Split:")
    print(f"  USDT Accumulation: {usdt_pct*100:.0f}% (${starting_balance.get('USDT', 10000) * usdt_pct:,.2f})")
    print(f"  Crypto Accumulation: {crypto_pct*100:.0f}% (${starting_balance.get('USDT', 10000) * crypto_pct:,.2f})")
    if use_isolated_portfolios:
        print(f"\nPhase 32: Isolated Portfolios ENABLED")
        print(f"  Per-strategy capital: ${per_strategy_capital:,.0f}")

    # Create portfolio
    portfolio = Portfolio(starting_balance)

    # Create experiment ID
    experiment_id = args.experiment_id or f"dual_{preset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Create orchestrator with dual portfolio mode
    orchestrator = UnifiedOrchestrator(
        portfolio=portfolio,
        config_path=config_path,
        experiment_id=experiment_id,
        dual_portfolio_mode=True,
        usdt_allocation=usdt_pct,
        crypto_allocation=crypto_pct,
        use_isolated_portfolios=use_isolated_portfolios,
        per_strategy_capital=per_strategy_capital
    )

    # Show enabled strategies by portfolio
    print("\n" + "-"*70)
    print("USDT Portfolio Strategies:")
    usdt_strats = [name for name, strat in orchestrator.strategies.items()
                   if orchestrator.registry.get_params(name).get('portfolio') == 'usdt']
    for name in usdt_strats:
        params = orchestrator.registry.get_params(name)
        alloc = params.get('allocation_pct', 0) * 100
        print(f"  - {name} ({alloc:.0f}%)")

    print("\nCrypto Portfolio Strategies:")
    crypto_strats = [name for name, strat in orchestrator.strategies.items()
                     if orchestrator.registry.get_params(name).get('portfolio') == 'crypto']
    for name in crypto_strats:
        params = orchestrator.registry.get_params(name)
        alloc = params.get('allocation_pct', 0) * 100
        print(f"  - {name} ({alloc:.0f}%)")
    print("-"*70)

    print(f"\nExperiment ID: {experiment_id}")
    print("="*70 + "\n")

    # Run trading loop
    orchestrator.run_loop(
        duration_minutes=args.duration,
        interval_seconds=args.interval
    )


def cmd_experiment(args):
    """Run an experiment with parameter overrides."""
    print("\n" + "="*70)
    print("UNIFIED TRADING PLATFORM - Experiment Mode")
    print("="*70)

    config_path = args.config or DEFAULT_CONFIG_PATH
    if not os.path.exists(config_path):
        print(f"Config not found: {config_path}")
        return

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Get experiment preset if specified
    preset = None
    if args.preset:
        experiments = config.get('experiments', {})
        if args.preset in experiments:
            preset = experiments[args.preset]
            print(f"\nUsing experiment preset: {args.preset}")
            print(f"Description: {preset.get('description', 'N/A')}")
        else:
            print(f"Unknown preset: {args.preset}")
            print(f"Available: {list(experiments.keys())}")
            return

    # Create experiment ID
    experiment_id = args.experiment_id or f"exp_{args.preset or 'custom'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    starting_balance = config.get('global', {}).get('starting_balance', {
        'USDT': 2000.0,
        'XRP': 0.0,
        'BTC': 0.0
    })

    portfolio = Portfolio(starting_balance)

    orchestrator = UnifiedOrchestrator(
        portfolio=portfolio,
        config_path=config_path,
        experiment_id=experiment_id
    )

    # Apply preset overrides
    if preset:
        # Enable only specific strategies if specified
        if 'enable_only' in preset:
            orchestrator.registry.disable_all()
            for name in preset['enable_only']:
                orchestrator.registry.enable(name)
            # Reinitialize with new enabled set
            orchestrator._initialize()

        # Apply parameter overrides
        if 'overrides' in preset:
            for strat_name, params in preset['overrides'].items():
                for param, value in params.items():
                    orchestrator.set_experiment_param(strat_name, param, value)
                    print(f"  Override: {strat_name}.{param} = {value}")

    # Apply CLI parameter overrides
    if args.override:
        for override in args.override:
            try:
                strat, param, value = override.split(':')
                # Try to parse value as number
                try:
                    value = float(value)
                    if value == int(value):
                        value = int(value)
                except:
                    pass
                orchestrator.set_experiment_param(strat, param, value)
                print(f"  CLI Override: {strat}.{param} = {value}")
            except:
                print(f"  Invalid override format: {override} (use strategy:param:value)")

    print(f"\nExperiment ID: {experiment_id}")
    print("-"*70)

    # Run experiment
    orchestrator.run_loop(
        duration_minutes=args.duration,
        interval_seconds=args.interval
    )


def cmd_list(args):
    """List available strategies."""
    config_path = args.config or DEFAULT_CONFIG_PATH
    registry = StrategyRegistry(config_path if os.path.exists(config_path) else None)

    print("\n" + "="*70)
    print("AVAILABLE STRATEGIES")
    print("="*70)

    status = registry.get_status()

    # Group by category
    categories = {}
    for name, info in status['strategies'].items():
        cat = info.get('category', 'general')
        if cat not in categories:
            categories[cat] = []
        categories[cat].append((name, info))

    for category, strategies in sorted(categories.items()):
        print(f"\n[{category.upper()}]")
        for name, info in sorted(strategies):
            status_icon = "+" if info['enabled'] else "-"
            desc = info.get('description', '(no description)')[:50]
            print(f"  {status_icon} {name:30s} {desc}")

    print("\n" + "-"*70)
    print(f"Total: {status['total_registered']} strategies, {status['enabled']} enabled")
    print("="*70 + "\n")


def cmd_config(args):
    """Manage strategy configuration."""
    config_path = args.config or DEFAULT_CONFIG_PATH

    if not os.path.exists(config_path):
        print(f"Config not found: {config_path}")
        print("Run 'unified_trader.py init-config' to create one.")
        return

    with open(config_path) as f:
        config = yaml.safe_load(f)

    modified = False

    # Enable strategies
    if args.enable:
        for name in args.enable:
            if name in config.get('strategies', {}):
                config['strategies'][name]['enabled'] = True
                print(f"Enabled: {name}")
                modified = True
            else:
                print(f"Unknown strategy: {name}")

    # Disable strategies
    if args.disable:
        for name in args.disable:
            if name in config.get('strategies', {}):
                config['strategies'][name]['enabled'] = False
                print(f"Disabled: {name}")
                modified = True
            else:
                print(f"Unknown strategy: {name}")

    # Set parameters
    if args.set:
        for setting in args.set:
            try:
                path, value = setting.split('=')
                parts = path.split('.')

                # Navigate to the right location
                target = config
                for part in parts[:-1]:
                    if part not in target:
                        target[part] = {}
                    target = target[part]

                # Try to parse value as number or bool
                if value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
                else:
                    try:
                        value = float(value)
                        if value == int(value):
                            value = int(value)
                    except:
                        pass

                target[parts[-1]] = value
                print(f"Set: {path} = {value}")
                modified = True
            except:
                print(f"Invalid setting format: {setting} (use path.to.param=value)")

    # Save if modified
    if modified:
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"\nConfig saved to {config_path}")
    else:
        print("No changes made.")


def cmd_init_config(args):
    """Create a new configuration template."""
    config_path = args.output or DEFAULT_CONFIG_PATH

    if os.path.exists(config_path) and not args.force:
        print(f"Config already exists: {config_path}")
        print("Use --force to overwrite.")
        return

    create_unified_config_template(config_path)
    print(f"\nCreated config: {config_path}")
    print("Edit this file to enable/disable strategies and tune parameters.")


def cmd_analyze(args):
    """Analyze experiment logs."""
    log_dir = Path("logs")

    if args.experiment:
        # Find specific experiment
        experiment_files = list(log_dir.glob(f"**/*{args.experiment}*"))
        if not experiment_files:
            print(f"No logs found for experiment: {args.experiment}")
            return
    else:
        # List recent experiments
        print("\n" + "="*70)
        print("RECENT EXPERIMENTS")
        print("="*70)

        orchestrator_logs = sorted(log_dir.glob("orchestrator/*.jsonl"), reverse=True)[:10]
        for log_file in orchestrator_logs:
            # Read first and last lines
            with open(log_file) as f:
                lines = f.readlines()
                if lines:
                    start = json.loads(lines[0])
                    end = json.loads(lines[-1]) if len(lines) > 1 else {}

                    exp_id = start.get('experiment_id', 'unknown')
                    timestamp = start.get('timestamp', 'unknown')[:19]

                    total_trades = end.get('combined_trades', 0)
                    total_pnl = end.get('combined_pnl', 0)

                    print(f"\n  {exp_id}")
                    print(f"    Time: {timestamp}")
                    print(f"    Trades: {total_trades}, PnL: ${total_pnl:.2f}")

        print("\n" + "="*70)
        print("Use --experiment <id> to analyze a specific experiment")
        return

    # Detailed analysis
    print("\n" + "="*70)
    print(f"EXPERIMENT ANALYSIS: {args.experiment}")
    print("="*70)

    for log_file in experiment_files:
        print(f"\nFile: {log_file}")
        print("-"*50)

        with open(log_file) as f:
            for line in f:
                entry = json.loads(line)
                entry_type = entry.get('type', '')

                if entry_type == 'session_end':
                    summary = entry.get('summary', {})
                    print(f"\nStrategy: {entry.get('strategy', 'orchestrator')}")
                    print(f"  Trades: {summary.get('trades', {}).get('total', 0)}")
                    print(f"  Win Rate: {summary.get('trades', {}).get('win_rate', 0):.1f}%")
                    print(f"  PnL: ${summary.get('pnl', {}).get('net', 0):.2f}")

                elif entry_type == 'unified_session_end':
                    print(f"\nCombined Results:")
                    print(f"  Total Strategies: {entry.get('total_strategies', 0)}")
                    print(f"  Combined Trades: {entry.get('combined_trades', 0)}")
                    print(f"  Combined PnL: ${entry.get('combined_pnl', 0):.2f}")


def cmd_compare(args):
    """Compare multiple experiments."""
    log_dir = Path("logs")

    print("\n" + "="*70)
    print("EXPERIMENT COMPARISON")
    print("="*70)

    results = []

    for exp_id in args.experiments:
        # Find orchestrator log
        matches = list(log_dir.glob(f"orchestrator/*{exp_id}*"))
        if not matches:
            print(f"Not found: {exp_id}")
            continue

        log_file = matches[0]
        with open(log_file) as f:
            lines = f.readlines()
            if lines:
                end = json.loads(lines[-1])
                if end.get('type') == 'unified_session_end':
                    results.append({
                        'experiment': exp_id,
                        'strategies': end.get('total_strategies', 0),
                        'trades': end.get('combined_trades', 0),
                        'pnl': end.get('combined_pnl', 0)
                    })

    if not results:
        print("No valid experiments found.")
        return

    # Print comparison table
    print(f"\n{'Experiment':<30} {'Strategies':>10} {'Trades':>10} {'PnL':>12}")
    print("-"*70)

    for r in results:
        pnl_str = f"${r['pnl']:.2f}"
        print(f"{r['experiment']:<30} {r['strategies']:>10} {r['trades']:>10} {pnl_str:>12}")

    # Find best
    if len(results) > 1:
        best = max(results, key=lambda x: x['pnl'])
        print("-"*70)
        print(f"Best: {best['experiment']} (${best['pnl']:.2f})")


def main():
    parser = argparse.ArgumentParser(
        description="Unified Trading Platform - Phase 24",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s paper                         # Run paper trading
  %(prog)s paper --duration 120          # Run for 2 hours
  %(prog)s experiment --preset aggressive # Run aggressive experiment
  %(prog)s list                          # List all strategies
  %(prog)s config --enable grid_arithmetic  # Enable a strategy
  %(prog)s analyze                       # View recent experiments
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Paper trading
    paper_parser = subparsers.add_parser('paper', help='Run paper trading')
    paper_parser.add_argument('--config', type=str, help='Path to config file')
    paper_parser.add_argument('--duration', type=int, default=60, help='Duration in minutes')
    paper_parser.add_argument('--interval', type=int, default=300, help='Decision interval in seconds')
    paper_parser.add_argument('--experiment-id', type=str, help='Custom experiment ID')
    paper_parser.add_argument('--enable', nargs='+', help='Enable specific strategies')
    paper_parser.add_argument('--disable', nargs='+', help='Disable specific strategies')

    # Dual portfolio mode (NEW)
    dual_parser = subparsers.add_parser('dual', help='Run dual portfolio mode (70% USDT / 30% Crypto)')
    dual_parser.add_argument('--config', type=str, help='Path to dual portfolio config')
    dual_parser.add_argument('--preset', type=str, default='balanced',
                            choices=['balanced', 'usdt_aggressive', 'crypto_aggressive', 'usdt_only', 'crypto_only'],
                            help='Portfolio preset (default: balanced)')
    dual_parser.add_argument('--duration', type=int, default=60, help='Duration in minutes')
    dual_parser.add_argument('--interval', type=int, default=300, help='Decision interval in seconds')
    dual_parser.add_argument('--experiment-id', type=str, help='Custom experiment ID')

    # Experiment
    exp_parser = subparsers.add_parser('experiment', help='Run an experiment')
    exp_parser.add_argument('--config', type=str, help='Path to config file')
    exp_parser.add_argument('--preset', type=str, help='Experiment preset name')
    exp_parser.add_argument('--override', nargs='+', help='Parameter overrides (strategy:param:value)')
    exp_parser.add_argument('--duration', type=int, default=60, help='Duration in minutes')
    exp_parser.add_argument('--interval', type=int, default=300, help='Decision interval in seconds')
    exp_parser.add_argument('--experiment-id', type=str, help='Custom experiment ID')

    # List strategies
    list_parser = subparsers.add_parser('list', help='List available strategies')
    list_parser.add_argument('--config', type=str, help='Path to config file')

    # Config management
    config_parser = subparsers.add_parser('config', help='Manage configuration')
    config_parser.add_argument('--config', type=str, help='Path to config file')
    config_parser.add_argument('--enable', nargs='+', help='Enable strategies')
    config_parser.add_argument('--disable', nargs='+', help='Disable strategies')
    config_parser.add_argument('--set', nargs='+', help='Set parameters (path.to.param=value)')

    # Init config
    init_parser = subparsers.add_parser('init-config', help='Create config template')
    init_parser.add_argument('--output', type=str, help='Output path')
    init_parser.add_argument('--force', action='store_true', help='Overwrite existing')

    # Analyze
    analyze_parser = subparsers.add_parser('analyze', help='Analyze experiment logs')
    analyze_parser.add_argument('--experiment', type=str, help='Experiment ID to analyze')

    # Compare
    compare_parser = subparsers.add_parser('compare', help='Compare experiments')
    compare_parser.add_argument('experiments', nargs='+', help='Experiment IDs to compare')

    args = parser.parse_args()

    if args.command == 'paper':
        cmd_paper(args)
    elif args.command == 'dual':
        cmd_dual(args)
    elif args.command == 'experiment':
        cmd_experiment(args)
    elif args.command == 'list':
        cmd_list(args)
    elif args.command == 'config':
        cmd_config(args)
    elif args.command == 'init-config':
        cmd_init_config(args)
    elif args.command == 'analyze':
        cmd_analyze(args)
    elif args.command == 'compare':
        cmd_compare(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
