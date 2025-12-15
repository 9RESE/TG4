#!/usr/bin/env python3
"""
WebSocket Paper Trading Tester
Lightweight, strategy-focused paper trading with live WebSocket data.

Usage:
    python ws_tester.py                         # Run with default strategies
    python ws_tester.py --duration 60           # Run for 60 minutes
    python ws_tester.py --strategies mm,of      # Only specific strategies
    python ws_tester.py --simulated             # Use simulated data
    python ws_tester.py --no-dashboard          # Disable web dashboard
    python ws_tester.py --config config.yaml    # Use specific config file
"""

import asyncio
import argparse
import time
import hashlib
import signal as signal_module  # Renamed to avoid conflict with Signal type
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

# Try to import yaml for config loading
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file. If None, looks for config.yaml in project root.

    Returns:
        Configuration dictionary
    """
    if not YAML_AVAILABLE:
        print("[Config] Warning: PyYAML not installed. Using defaults.")
        return {}

    # Default config location
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        print(f"[Config] Config file not found: {config_path}")
        return {}

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            print(f"[Config] Loaded configuration from {config_path}")
            return config or {}
    except Exception as e:
        print(f"[Config] Error loading config: {e}")
        return {}

import dataclasses

from ws_tester.data_layer import KrakenWSClient, DataManager, SimulatedDataManager
from ws_tester.strategy_loader import discover_strategies, get_all_symbols
from ws_tester.executor import PaperExecutor
from ws_tester.portfolio import PortfolioManager, STARTING_CAPITAL
from ws_tester.logger import TesterLogger, LogConfig
from ws_tester.types import DataSnapshot
from ws_tester.regime import RegimeDetector


class WebSocketPaperTester:
    """Main application coordinator."""

    def __init__(
        self,
        symbols: list = None,
        strategies_dir: str = "strategies",
        log_config: LogConfig = None,
        starting_capital: float = STARTING_CAPITAL,
        enable_dashboard: bool = True,
        simulated: bool = False,
        config: Dict[str, Any] = None,
        strategy_overrides: Dict[str, Dict] = None,
    ):
        self.session_id = f"wst_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.strategies_dir = strategies_dir
        self.starting_capital = starting_capital
        self.starting_assets = (config or {}).get('starting_assets', {})
        self.enable_dashboard = enable_dashboard
        self.simulated = simulated
        self._running = False
        self.config = config or {}

        # Initialize logger with config
        logging_config = self.config.get('logging', {})
        self.logger = TesterLogger(
            self.session_id,
            log_config or LogConfig(
                base_dir=logging_config.get('base_dir', str(Path(__file__).parent / "logs")),
                buffer_size=logging_config.get('buffer_size', 100),
                enable_aggregated=logging_config.get('enable_aggregated', True),
                console_output=logging_config.get('console_output', True),
            )
        )

        # Load strategies
        strategies_path = Path(__file__).parent / strategies_dir
        self.strategies = discover_strategies(str(strategies_path))

        if not self.strategies:
            print("[ERROR] No strategies loaded. Add strategy files to strategies/")
            return

        # Apply strategy overrides from config (HIGH-003)
        config_overrides = self.config.get('strategy_overrides', {})
        all_overrides = {**config_overrides, **(strategy_overrides or {})}
        self._apply_strategy_overrides(all_overrides)

        # Determine symbols from config, strategies, or use defaults (LOW-003)
        config_symbols = self.config.get('symbols', [])
        if symbols:
            self.symbols = symbols
        elif config_symbols:
            self.symbols = config_symbols
        else:
            strategy_symbols = get_all_symbols(self.strategies)
            if strategy_symbols:
                self.symbols = strategy_symbols
            else:
                default_symbols = self.config.get('general', {}).get('default_symbols', ['XRP/USD', 'BTC/USD'])
                print(f"[Config] Warning: Using fallback symbols {default_symbols}")
                self.symbols = default_symbols

        # Initialize all components (portfolio manager, data manager, executor)
        self._initialize_components()

    def _apply_strategy_overrides(self, overrides: Dict[str, Dict]):
        """Apply configuration overrides to loaded strategies."""
        for strategy_name, config_updates in overrides.items():
            if strategy_name in self.strategies:
                strategy = self.strategies[strategy_name]
                strategy.config.update(config_updates)
                print(f"[Config] Applied overrides to strategy '{strategy_name}': {list(config_updates.keys())}")

    def _initialize_components(self):
        """Initialize portfolio manager, data manager, executor, regime detector, and stats."""
        # Initialize portfolio manager with all strategies and starting assets
        strategy_names = list(self.strategies.keys())
        self.portfolio_manager = PortfolioManager(
            strategy_names,
            self.starting_capital,
            starting_assets=self.starting_assets
        )

        # Initialize data manager
        if self.simulated:
            self.data_manager = SimulatedDataManager(self.symbols)
        else:
            self.data_manager = DataManager(self.symbols)

        # Initialize executor with configurable parameters
        # Note: config.yaml uses 'execution' key (HIGH-002 fix)
        executor_config = self.config.get('execution', {})
        self.executor = PaperExecutor(
            self.portfolio_manager,
            max_short_leverage=executor_config.get('max_short_leverage', 2.0),
            slippage_rate=executor_config.get('slippage_rate', 0.0005),
            fee_rate=executor_config.get('fee_rate', 0.001),
        )

        # Initialize regime detector for market regime analysis
        regime_config = self.config.get('regime_detection', {})
        self.regime_detector = RegimeDetector(
            symbols=self.symbols,
            config=regime_config
        )

        # Stats
        self.tick_count = 0
        self.signal_count = 0
        self.fill_count = 0
        self.start_time = None

        self.logger.log_system("strategies_loaded", details={
            "count": len(self.strategies),
            "names": list(self.strategies.keys()),
            "symbols": self.symbols,
            "starting_capital": self.starting_capital,
            "starting_assets": self.starting_assets,
        })

    async def run(
        self,
        duration_minutes: int = 60,
        interval_ms: int = 100,
    ):
        """Main async run loop."""
        if not self.strategies:
            return

        self._running = True
        self.start_time = time.time()

        print(f"\n{'='*60}")
        print("WebSocket Paper Trading Tester")
        print(f"{'='*60}")
        print(f"Session: {self.session_id}")
        print(f"Mode: {'Simulated' if self.simulated else 'Live'}")
        print(f"Symbols: {self.symbols}")
        print(f"Strategies: {list(self.strategies.keys())}")
        print(f"Starting Capital: ${self.starting_capital} USDT per strategy")
        if self.starting_assets:
            assets_str = ', '.join(f"{v} {k}" for k, v in self.starting_assets.items() if v > 0)
            print(f"Starting Assets: {assets_str}")
        print(f"Duration: {duration_minutes} min | Interval: {interval_ms}ms")
        print(f"{'='*60}")

        # Start dashboard if enabled
        dashboard_task = None
        broadcast_task = None

        # LOW-002: Cache dashboard functions to avoid import inside loop
        self._dashboard_add_trade = None
        self._dashboard_update_state = None

        if self.enable_dashboard:
            try:
                import uvicorn
                from ws_tester.dashboard.server import app, publisher, add_trade, update_state

                # LOW-002: Cache these functions to avoid repeated imports in loop
                self._dashboard_add_trade = add_trade
                self._dashboard_update_state = update_state

                # Get dashboard config with defaults
                dashboard_host = self.config.get('dashboard', {}).get('host', '127.0.0.1')
                dashboard_port = self.config.get('dashboard', {}).get('port', 8787)

                config = uvicorn.Config(
                    app,
                    host=dashboard_host,
                    port=dashboard_port,
                    log_level="warning"
                )
                server = uvicorn.Server(config)

                dashboard_task = asyncio.create_task(server.serve())
                broadcast_task = asyncio.create_task(publisher.broadcast_loop())

                print(f"\nDashboard: http://{dashboard_host}:{dashboard_port}")
            except ImportError:
                print("\n[WARN] FastAPI/uvicorn not installed. Dashboard disabled.")
                self.enable_dashboard = False

        print(f"\nPress Ctrl+C to stop\n")

        self.logger.log_system("session_start", details={
            "session_id": self.session_id,
            "symbols": self.symbols,
            "strategies": list(self.strategies.keys()),
            "duration_minutes": duration_minutes,
            "simulated": self.simulated,
        })

        # Start strategy on_start callbacks
        for name, strategy in self.strategies.items():
            strategy.on_start()

        # Connect to WebSocket or run simulated
        ws_client = None
        data_task = None

        if not self.simulated:
            ws_client = KrakenWSClient(self.symbols)
            connected = await ws_client.connect()

            if connected:
                await ws_client.subscribe(['trade', 'ticker', 'book'])
                data_task = asyncio.create_task(
                    ws_client.run_forever(self.data_manager.on_message)
                )
                self.logger.log_system("ws_connected", details={"url": ws_client.WS_URL})
            else:
                print("[ERROR] Failed to connect to WebSocket. Switching to simulated mode.")
                self.simulated = True
                self.data_manager = SimulatedDataManager(self.symbols)

        # Main trading loop
        end_time = self.start_time + (duration_minutes * 60)
        last_report = self.start_time
        last_dashboard_update = self.start_time

        try:
            while self._running and time.time() < end_time:
                self.tick_count += 1

                # Generate simulated data if needed
                if self.simulated:
                    await self.data_manager.simulate_tick()

                # Get immutable data snapshot (async for full thread safety)
                snapshot = await self.data_manager.get_snapshot_async()

                if not snapshot or not snapshot.prices:
                    await asyncio.sleep(interval_ms / 1000)
                    continue

                # Calculate market regime and populate snapshot
                try:
                    regime_snapshot = await self.regime_detector.detect(snapshot)
                    snapshot = dataclasses.replace(snapshot, regime=regime_snapshot)
                except Exception as e:
                    self.logger.log_error("regime_detector", str(e))
                    # Continue with None regime on error

                # Hash for reproducibility/logging
                data_hash = hashlib.sha256(
                    str(sorted(snapshot.prices.items())).encode()
                ).hexdigest()[:16]

                # Check stop-loss / take-profit
                stop_signals = self.executor.check_stops(snapshot)
                for strategy_name, sig in stop_signals:
                    fill = self.executor.execute(sig, strategy_name, snapshot)
                    if fill:
                        self.fill_count += 1
                        portfolio = self.portfolio_manager.get_portfolio(strategy_name)
                        self.logger.log_fill(
                            fill,
                            f"auto-{self.tick_count}",
                            strategy_name,
                            portfolio.to_dict(snapshot.prices) if portfolio else {},
                        )

                # Update position tracking
                self.executor.update_position_tracking(snapshot)

                # Run each strategy
                for name, strategy in self.strategies.items():
                    start_us = time.perf_counter_ns() // 1000

                    try:
                        signal = strategy.generate_signal(snapshot)
                    except Exception as e:
                        self.logger.log_error(name, str(e))
                        continue

                    latency_us = (time.perf_counter_ns() // 1000) - start_us

                    # Log signal (or no-signal)
                    correlation_id = self.logger.log_signal(
                        name,
                        signal,
                        data_hash,
                        indicators=strategy.state.get('indicators', {}),
                        latency_us=latency_us
                    )

                    if signal:
                        self.signal_count += 1
                        signal.metadata = signal.metadata or {}
                        signal.metadata['strategy'] = name

                        # Execute
                        fill = self.executor.execute(signal, name, snapshot)

                        if fill:
                            self.fill_count += 1

                            # Notify strategy
                            strategy.on_fill(fill.__dict__)

                            # Log fill
                            portfolio = self.portfolio_manager.get_portfolio(name)
                            portfolio_dict = portfolio.to_dict(snapshot.prices) if portfolio else {}
                            self.logger.log_fill(
                                fill,
                                correlation_id,
                                name,
                                portfolio_dict,
                            )

                            # Log aggregated entry (complete audit trail)
                            self.logger.log_aggregated(
                                correlation_id=correlation_id,
                                data={
                                    'timestamp': snapshot.timestamp.isoformat() if snapshot.timestamp else None,
                                    'prices': snapshot.prices,
                                    'data_hash': data_hash,
                                },
                                strategy=name,
                                signal=signal,
                                execution={
                                    'fill_id': fill.fill_id,
                                    'symbol': fill.symbol,
                                    'side': fill.side,
                                    'size': fill.size,
                                    'price': fill.price,
                                    'fee': fill.fee,
                                    'pnl': fill.pnl,
                                },
                                portfolio=portfolio_dict,
                            )

                            # Update dashboard (LOW-002: imports moved outside loop)
                            if self.enable_dashboard and self._dashboard_add_trade:
                                self._dashboard_add_trade({
                                    'timestamp': fill.timestamp.isoformat(),
                                    'strategy': name,
                                    'symbol': fill.symbol,
                                    'side': fill.side,
                                    'size': fill.size,
                                    'price': fill.price,
                                    'pnl': fill.pnl,
                                    'reason': fill.signal_reason,
                                })

                # Update dashboard state (LOW-002: imports moved outside loop)
                if self.enable_dashboard and (time.time() - last_dashboard_update) > 1.0 and self._dashboard_update_state:
                    self._dashboard_update_state(
                        prices=snapshot.prices,
                        strategies=self.portfolio_manager.get_leaderboard(snapshot.prices),
                        aggregate=self.portfolio_manager.get_aggregate(snapshot.prices),
                        session_info={
                            'session_id': self.session_id,
                            'runtime_minutes': (time.time() - self.start_time) / 60,
                            'ticks': self.tick_count,
                            'signals': self.signal_count,
                            'fills': self.fill_count,
                        }
                    )
                    last_dashboard_update = time.time()

                # Periodic status report (console + log)
                if (time.time() - last_report) > 30:
                    self._print_status(snapshot)

                    # Log status to file
                    leaderboard = self.portfolio_manager.get_leaderboard(snapshot.prices)
                    self.logger.log_status(
                        tick_count=self.tick_count,
                        signal_count=self.signal_count,
                        fill_count=self.fill_count,
                        prices=snapshot.prices,
                        portfolios=leaderboard,
                    )

                    # Log portfolio snapshot for each strategy
                    for name, strategy in self.strategies.items():
                        portfolio = self.portfolio_manager.get_portfolio(name)
                        if portfolio:
                            portfolio_dict = portfolio.to_dict(snapshot.prices)
                            # Get per-symbol stats from portfolio
                            symbol_stats = {
                                sym: {
                                    'pnl': portfolio_dict.get('pnl_by_symbol', {}).get(sym, 0),
                                    'trades': portfolio_dict.get('trades_by_symbol', {}).get(sym, 0),
                                }
                                for sym in strategy.symbols
                            }
                            self.logger.log_portfolio_snapshot(
                                strategy=name,
                                portfolio=portfolio_dict,
                                prices=snapshot.prices,
                                symbol_stats=symbol_stats,
                            )

                    last_report = time.time()

                await asyncio.sleep(interval_ms / 1000)

        except KeyboardInterrupt:
            print("\n\nStopping...")

        finally:
            self._running = False

            # Stop strategies
            for strategy in self.strategies.values():
                strategy.on_stop()

            # Cleanup
            if data_task:
                data_task.cancel()
            if ws_client:
                await ws_client.close()

            # Final report
            final_snapshot = self.data_manager.get_snapshot()
            self._print_final_report(final_snapshot)

            self.logger.log_system("session_end", details={
                "duration_minutes": (time.time() - self.start_time) / 60,
                "ticks": self.tick_count,
                "signals": self.signal_count,
                "fills": self.fill_count,
                "portfolios": self.portfolio_manager.get_leaderboard(
                    final_snapshot.prices if final_snapshot else {}
                ),
            })

            self.logger.close()

            # Cancel dashboard tasks
            if dashboard_task:
                dashboard_task.cancel()
            if broadcast_task:
                broadcast_task.cancel()

    def _print_status(self, snapshot: DataSnapshot):
        """Print periodic status update."""
        runtime = (time.time() - self.start_time) / 60
        aggregate = self.portfolio_manager.get_aggregate(snapshot.prices)

        print(f"\n--- Status @ {datetime.now().strftime('%H:%M:%S')} ({runtime:.1f} min) ---")
        print(f"Ticks: {self.tick_count} | Signals: {self.signal_count} | Fills: {self.fill_count}")
        print(f"Total Equity: ${aggregate['total_equity']:.2f} | P&L: ${aggregate['total_pnl']:+.2f}")
        print(f"Prices: {', '.join(f'{k}={v:.6f}' for k, v in snapshot.prices.items())}")

        # Leaderboard
        leaderboard = self.portfolio_manager.get_leaderboard(snapshot.prices)
        print("\nLeaderboard:")
        for i, s in enumerate(leaderboard[:5], 1):
            print(f"  {i}. {s['strategy']}: ${s['equity']:.2f} (P&L: ${s['pnl']:+.2f}, {s['trades']} trades)")
        print()

    def _print_final_report(self, snapshot: Optional[DataSnapshot]):
        """Print final session report."""
        prices = snapshot.prices if snapshot else {}
        aggregate = self.portfolio_manager.get_aggregate(prices)

        print(f"\n{'='*60}")
        print("SESSION COMPLETE")
        print(f"{'='*60}")
        print(f"Duration: {(time.time() - self.start_time) / 60:.1f} minutes")
        print(f"Mode: {'Simulated' if self.simulated else 'Live'}")
        print(f"Ticks: {self.tick_count}")
        print(f"Signals: {self.signal_count}")
        print(f"Fills: {self.fill_count}")
        print()
        print(f"Total Capital: ${aggregate['total_capital']:.2f}")
        print(f"Final Equity:  ${aggregate['total_equity']:.2f}")
        print(f"Total P&L:     ${aggregate['total_pnl']:+.2f} ({aggregate['total_roi_pct']:+.2f}%)")
        print(f"Win Rate:      {aggregate['win_rate']:.1f}%")
        print()
        print("Strategy Performance:")
        print("-" * 60)

        leaderboard = self.portfolio_manager.get_leaderboard(prices)
        for i, s in enumerate(leaderboard, 1):
            status = "WIN" if s['pnl'] > 0 else ("LOSS" if s['pnl'] < 0 else "FLAT")
            print(f"  {i}. {s['strategy']:<20} ${s['equity']:>8.2f}  "
                  f"P&L: ${s['pnl']:>+7.2f} ({s['roi_pct']:>+5.2f}%)  "
                  f"Trades: {s['trades']:>3}  Win: {s['win_rate']:>5.1f}%  [{status}]")

        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="WebSocket Paper Trading Tester")
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config.yaml file (default: config.yaml in project root)')
    parser.add_argument('--duration', type=int, default=None,
                       help='Duration in minutes (default: from config or 60)')
    parser.add_argument('--interval', type=int, default=None,
                       help='Loop interval in ms (default: from config or 100)')
    parser.add_argument('--symbols', type=str, default=None,
                       help='Comma-separated symbols (default: from config or strategies)')
    parser.add_argument('--strategies-dir', type=str, default='strategies',
                       help='Directory containing strategy files (default: strategies)')
    parser.add_argument('--capital', type=float, default=None,
                       help='Starting capital per strategy (default: from config or $100)')
    parser.add_argument('--no-dashboard', action='store_true',
                       help='Disable web dashboard')
    parser.add_argument('--simulated', action='store_true',
                       help='Use simulated data instead of live WebSocket')

    args = parser.parse_args()

    # Load config file (HIGH-003)
    config = load_config(args.config)

    # CLI arguments take precedence over config file
    general_config = config.get('general', {})

    duration = args.duration or general_config.get('duration_minutes', 60)
    interval = args.interval or general_config.get('interval_ms', 100)
    capital = args.capital or general_config.get('starting_capital', 100.0)

    symbols = None
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',')]

    # Determine simulated mode from config or CLI
    data_config = config.get('data', {})
    simulated = args.simulated or (data_config.get('source', 'kraken') == 'simulated')

    # Determine dashboard setting from config or CLI
    dashboard_config = config.get('dashboard', {})
    enable_dashboard = not args.no_dashboard and dashboard_config.get('enabled', True)

    tester = WebSocketPaperTester(
        symbols=symbols,
        strategies_dir=args.strategies_dir,
        starting_capital=capital,
        enable_dashboard=enable_dashboard,
        simulated=simulated,
        config=config,
    )

    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        tester._running = False

    signal_module.signal(signal_module.SIGINT, signal_handler)

    asyncio.run(tester.run(
        duration_minutes=duration,
        interval_ms=interval,
    ))


if __name__ == "__main__":
    main()
