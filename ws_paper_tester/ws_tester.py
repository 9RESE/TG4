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
"""

import asyncio
import argparse
import time
import hashlib
import signal
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from ws_tester.data_layer import KrakenWSClient, DataManager, SimulatedDataManager
from ws_tester.strategy_loader import discover_strategies, get_all_symbols
from ws_tester.executor import PaperExecutor
from ws_tester.portfolio import PortfolioManager, STARTING_CAPITAL
from ws_tester.logger import TesterLogger, LogConfig
from ws_tester.types import DataSnapshot


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
    ):
        self.session_id = f"wst_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.strategies_dir = strategies_dir
        self.starting_capital = starting_capital
        self.enable_dashboard = enable_dashboard
        self.simulated = simulated
        self._running = False

        # Initialize logger
        self.logger = TesterLogger(
            self.session_id,
            log_config or LogConfig(base_dir=str(Path(__file__).parent / "logs"))
        )

        # Load strategies
        strategies_path = Path(__file__).parent / strategies_dir
        self.strategies = discover_strategies(str(strategies_path))

        if not self.strategies:
            print("[ERROR] No strategies loaded. Add strategy files to strategies/")
            return

        # Determine symbols from strategies if not specified
        if symbols:
            self.symbols = symbols
        else:
            self.symbols = get_all_symbols(self.strategies) or ['XRP/USD', 'BTC/USD']

        # Initialize portfolio manager with all strategies
        strategy_names = list(self.strategies.keys())
        self.portfolio_manager = PortfolioManager(strategy_names, starting_capital)

        # Initialize data manager
        if simulated:
            self.data_manager = SimulatedDataManager(self.symbols)
        else:
            self.data_manager = DataManager(self.symbols)

        # Initialize executor
        self.executor = PaperExecutor(self.portfolio_manager)

        # Stats
        self.tick_count = 0
        self.signal_count = 0
        self.fill_count = 0
        self.start_time = None

        self.logger.log_system("strategies_loaded", details={
            "count": len(self.strategies),
            "names": list(self.strategies.keys()),
            "symbols": self.symbols,
            "starting_capital": starting_capital,
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
        print(f"Starting Capital: ${self.starting_capital} per strategy")
        print(f"Duration: {duration_minutes} min | Interval: {interval_ms}ms")
        print(f"{'='*60}")

        # Start dashboard if enabled
        dashboard_task = None
        broadcast_task = None

        if self.enable_dashboard:
            try:
                import uvicorn
                from ws_tester.dashboard.server import app, publisher

                config = uvicorn.Config(
                    app,
                    host="0.0.0.0",
                    port=8080,
                    log_level="warning"
                )
                server = uvicorn.Server(config)

                dashboard_task = asyncio.create_task(server.serve())
                broadcast_task = asyncio.create_task(publisher.broadcast_loop())

                print(f"\nDashboard: http://localhost:8080")
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

                # Get immutable data snapshot
                snapshot = self.data_manager.get_snapshot()

                if not snapshot or not snapshot.prices:
                    await asyncio.sleep(interval_ms / 1000)
                    continue

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
                            self.logger.log_fill(
                                fill,
                                correlation_id,
                                name,
                                portfolio.to_dict(snapshot.prices) if portfolio else {},
                            )

                            # Update dashboard
                            if self.enable_dashboard:
                                from ws_tester.dashboard.server import add_trade
                                add_trade({
                                    'timestamp': fill.timestamp.isoformat(),
                                    'strategy': name,
                                    'symbol': fill.symbol,
                                    'side': fill.side,
                                    'size': fill.size,
                                    'price': fill.price,
                                    'pnl': fill.pnl,
                                    'reason': fill.signal_reason,
                                })

                # Update dashboard state
                if self.enable_dashboard and (time.time() - last_dashboard_update) > 1.0:
                    from ws_tester.dashboard.server import update_state
                    update_state(
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

                # Periodic console status report
                if (time.time() - last_report) > 30:
                    self._print_status(snapshot)
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
    parser.add_argument('--duration', type=int, default=60,
                       help='Duration in minutes (default: 60)')
    parser.add_argument('--interval', type=int, default=100,
                       help='Loop interval in ms (default: 100)')
    parser.add_argument('--symbols', type=str, default=None,
                       help='Comma-separated symbols (default: from strategies)')
    parser.add_argument('--strategies-dir', type=str, default='strategies',
                       help='Directory containing strategy files (default: strategies)')
    parser.add_argument('--capital', type=float, default=100.0,
                       help='Starting capital per strategy (default: $100)')
    parser.add_argument('--no-dashboard', action='store_true',
                       help='Disable web dashboard')
    parser.add_argument('--simulated', action='store_true',
                       help='Use simulated data instead of live WebSocket')

    args = parser.parse_args()

    symbols = None
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',')]

    tester = WebSocketPaperTester(
        symbols=symbols,
        strategies_dir=args.strategies_dir,
        starting_capital=args.capital,
        enable_dashboard=not args.no_dashboard,
        simulated=args.simulated,
    )

    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        tester._running = False

    signal.signal(signal.SIGINT, signal_handler)

    asyncio.run(tester.run(
        duration_minutes=args.duration,
        interval_ms=args.interval,
    ))


if __name__ == "__main__":
    main()
