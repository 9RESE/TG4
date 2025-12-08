"""
Phase 14: Live Paper Trading Loop with Dashboard
Live Launch + Dashboard + Final Opportunistic Tuning.
Runs every 10 minutes, logs trades, yield, and portfolio state.
Features:
- Real-time regime dashboard (fear/greed display)
- Auto-yield accrual with compounding (6.5% avg APY)
- Softened opportunistic short thresholds (RSI>68, ATR>4.2%)
- Enhanced equity curve with yield metrics
"""
import os
import sys
import time
import signal
import csv
from datetime import datetime
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_fetcher import DataFetcher
from portfolio import Portfolio
from orchestrator import RLOrchestrator
from dashboard import LiveDashboard


class LivePaperTrader:
    """
    Phase 14: Live paper trading loop with dashboard integration.
    Fetches real-time data, runs RL decisions, displays regime dashboard.
    """

    def __init__(self, initial_balance: dict = None, log_dir: str = 'logs',
                 enable_dashboard: bool = True, enable_plots: bool = False):
        self.initial_balance = initial_balance or {
            'USDT': 1000.0,
            'XRP': 500.0,
            'BTC': 0.0
        }
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Initialize components
        self.fetcher = DataFetcher()
        self.portfolio = Portfolio(self.initial_balance.copy())
        self.data = {}
        self.orchestrator = None

        # Phase 14: Dashboard integration
        self.enable_dashboard = enable_dashboard
        self.dashboard = LiveDashboard(enable_plots=enable_plots) if enable_dashboard else None

        # Logging files
        self.trades_file = self.log_dir / 'trades.csv'
        self.equity_file = self.log_dir / 'equity_curve.csv'

        # Initialize CSV files
        self._init_csv_files()

        # Running state
        self.running = False
        self.cycle_count = 0

        # Register signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _init_csv_files(self):
        """Initialize CSV log files with headers."""
        # Trades log
        if not self.trades_file.exists():
            with open(self.trades_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'action', 'asset', 'amount', 'price',
                    'leverage', 'confidence', 'mode', 'volatility',
                    'rsi_xrp', 'rsi_btc', 'portfolio_value', 'pnl'
                ])

        # Equity curve - Phase 14: Added fear/greed tracking
        if not self.equity_file.exists():
            with open(self.equity_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'portfolio_value', 'usdt', 'xrp', 'btc',
                    'xrp_price', 'btc_price', 'mode', 'volatility',
                    'margin_exposure', 'etf_exposure', 'yield_earned', 'total_yield',
                    'fear_greed_index', 'fear_greed_label'
                ])

    def _handle_shutdown(self, signum, frame):
        """Handle graceful shutdown."""
        print("\n[SHUTDOWN] Received signal, stopping...")
        self.running = False

    def fetch_latest_data(self) -> bool:
        """Fetch latest OHLCV data for all symbols."""
        try:
            for sym in ['XRP/USDT', 'BTC/USDT']:
                df = self.fetcher.fetch_ohlcv('kraken', sym, '1h', 500)
                if not df.empty:
                    self.data[sym] = df
                else:
                    print(f"[WARN] Empty data for {sym}")
                    return False
            return True
        except Exception as e:
            print(f"[ERROR] Data fetch failed: {e}")
            return False

    def get_current_prices(self) -> dict:
        """Get current prices from latest data."""
        prices = {'USDT': 1.0}
        for sym in ['XRP/USDT', 'BTC/USDT']:
            if sym in self.data and len(self.data[sym]) > 0:
                asset = sym.split('/')[0]
                prices[asset] = self.data[sym]['close'].iloc[-1]
        return prices

    def log_trade(self, result: dict, prices: dict):
        """Log a trade to trades.csv."""
        if not result.get('executed'):
            return

        timestamp = datetime.now().isoformat()
        portfolio_value = self.portfolio.get_total_usd(prices)

        with open(self.trades_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                result.get('action_type', 'unknown'),
                result.get('asset', 'unknown'),
                result.get('amount', result.get('collateral', 0)),
                prices.get(result.get('asset', 'USDT'), 1.0),
                result.get('leverage', 0),
                result.get('confidence', 0),
                result.get('mode', 'unknown'),
                result.get('volatility', 0),
                result.get('rsi', {}).get('XRP', 50),
                result.get('rsi', {}).get('BTC', 50),
                portfolio_value,
                result.get('margin_pnl', 0)
            ])

    def log_equity(self, prices: dict, result: dict = None):
        """Log portfolio state to equity_curve.csv. Phase 14: Added fear/greed."""
        timestamp = datetime.now().isoformat()
        portfolio_value = self.portfolio.get_total_usd(prices)

        # Get margin/ETF exposure
        margin_exposure = 0
        etf_exposure = 0
        if self.orchestrator:
            for pos in self.orchestrator.kraken.positions.values():
                margin_exposure += pos.get('size', 0) * prices.get('XRP', 2.0)
            etf_exposure = sum(self.orchestrator.bitrue.etf_holdings.values())

        # Phase 14: Get yield and fear/greed stats
        yield_earned = result.get('yield_earned', 0) if result else 0
        total_yield = self.portfolio.total_yield_earned if hasattr(self.portfolio, 'total_yield_earned') else 0

        # Calculate fear/greed index
        volatility = result.get('volatility', 0) if result else 0
        rsi = result.get('rsi', {}) if result else {}
        rsi_xrp = rsi.get('XRP', 50)
        rsi_btc = rsi.get('BTC', 50)

        fg_index, fg_label = 50, 'Neutral'
        if self.dashboard:
            fg_index, fg_label = self.dashboard.get_fear_greed_index(volatility, rsi_xrp, rsi_btc)

        with open(self.equity_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                portfolio_value,
                self.portfolio.balances.get('USDT', 0),
                self.portfolio.balances.get('XRP', 0),
                self.portfolio.balances.get('BTC', 0),
                prices.get('XRP', 0),
                prices.get('BTC', 0),
                result.get('mode', 'unknown') if result else 'init',
                volatility,
                margin_exposure,
                etf_exposure,
                yield_earned,
                total_yield,
                fg_index,
                fg_label
            ])

    def run_cycle(self) -> dict:
        """Run one trading cycle with dashboard update."""
        self.cycle_count += 1

        # Fetch latest data
        if not self.fetch_latest_data():
            print("[SKIP] Data fetch failed, skipping cycle")
            return {'action': 'skip', 'reason': 'data_fetch_failed'}

        prices = self.get_current_prices()

        # Initialize orchestrator if needed
        if self.orchestrator is None:
            try:
                self.orchestrator = RLOrchestrator(self.portfolio, self.data)
                if not self.orchestrator.enabled:
                    print("[WARN] RL model not loaded, running in observation mode")
            except Exception as e:
                print(f"[ERROR] Orchestrator init failed: {e}")
                return {'action': 'error', 'reason': str(e)}
        else:
            # Update orchestrator's data
            self.orchestrator.data = self.data
            if self.orchestrator.env:
                self.orchestrator.env.data = self.data

        # Get current portfolio value
        portfolio_value = self.portfolio.get_total_usd(prices)

        # Run RL decision
        if self.orchestrator.enabled:
            result = self.orchestrator.decide_and_execute(prices)
            self.orchestrator.check_and_manage_positions(prices)
            self.orchestrator.update_env_step()

            # Phase 14: Update dashboard
            if self.dashboard:
                self.dashboard.update(
                    portfolio_value=portfolio_value,
                    mode=result.get('mode', 'unknown'),
                    volatility=result.get('volatility', 0),
                    rsi_xrp=result.get('rsi', {}).get('XRP', 50),
                    rsi_btc=result.get('rsi', {}).get('BTC', 50),
                    yield_earned=result.get('yield_earned', 0),
                    prices=prices
                )

            # Log to CSV
            self.log_trade(result, prices)
            self.log_equity(prices, result)

            return result
        else:
            # Observation mode - just log equity
            self.log_equity(prices)
            return {'action': 'observe', 'reason': 'model_not_loaded'}

    def run(self, interval_minutes: int = 10, max_cycles: int = None):
        """
        Run the live paper trading loop.

        Args:
            interval_minutes: Minutes between cycles (default 10)
            max_cycles: Maximum cycles to run (None = infinite)
        """
        self.running = True
        interval_seconds = interval_minutes * 60

        print(f"\n{'#'*60}")
        print(f"# PHASE 14: LIVE LAUNCH + DASHBOARD")
        print(f"# Interval: {interval_minutes} minutes")
        print(f"# Max cycles: {max_cycles or 'infinite'}")
        print(f"# Logs: {self.log_dir}")
        print(f"# Yield APY: 6.5% avg (Kraken 6% + Bitrue 7%)")
        print(f"# Opportunistic thresholds: RSI>68, ATR>4.2%, conf>0.78")
        print(f"# Dashboard: {'enabled' if self.enable_dashboard else 'disabled'}")
        print(f"{'#'*60}")

        # Initial data fetch and equity log
        if self.fetch_latest_data():
            prices = self.get_current_prices()
            self.log_equity(prices)

        while self.running:
            try:
                result = self.run_cycle()

                # Check max cycles
                if max_cycles and self.cycle_count >= max_cycles:
                    print(f"\n[DONE] Reached max cycles ({max_cycles})")
                    break

                # Sleep until next cycle
                if self.running:
                    print(f"\n[SLEEP] Next cycle in {interval_minutes} minutes...")
                    for _ in range(interval_seconds):
                        if not self.running:
                            break
                        time.sleep(1)

            except Exception as e:
                print(f"[ERROR] Cycle failed: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(60)  # Wait 1 minute on error

        print("\n[STOPPED] Live paper trading stopped")
        self._print_summary()

        # Close dashboard
        if self.dashboard:
            self.dashboard.print_summary()
            self.dashboard.close()

    def _print_summary(self):
        """Print trading session summary. Phase 14: Added fear/greed summary."""
        print(f"\n{'='*60}")
        print("SESSION SUMMARY")
        print(f"{'='*60}")
        print(f"Total cycles: {self.cycle_count}")

        # Load and summarize equity curve
        if self.equity_file.exists():
            try:
                df = pd.read_csv(self.equity_file)
                if len(df) > 0:
                    initial = df['portfolio_value'].iloc[0]
                    final = df['portfolio_value'].iloc[-1]
                    pnl = final - initial
                    roi = (pnl / initial) * 100 if initial > 0 else 0

                    print(f"Initial value: ${initial:.2f}")
                    print(f"Final value:   ${final:.2f}")
                    print(f"Total P&L:     ${pnl:+.2f} ({roi:+.1f}%)")
                    print(f"Max value:     ${df['portfolio_value'].max():.2f}")
                    print(f"Min value:     ${df['portfolio_value'].min():.2f}")

                    # Phase 14: Yield summary
                    if 'total_yield' in df.columns:
                        total_yield = df['total_yield'].iloc[-1]
                        print(f"\n--- YIELD STATS ---")
                        print(f"Total yield:   ${total_yield:.4f}")

                    # Phase 14: Fear/Greed summary
                    if 'fear_greed_label' in df.columns:
                        print(f"\n--- FEAR/GREED DISTRIBUTION ---")
                        fg_counts = df['fear_greed_label'].value_counts()
                        for label, count in fg_counts.items():
                            pct = (count / len(df)) * 100
                            print(f"  {label}: {pct:.1f}%")
            except Exception as e:
                print(f"Could not load summary: {e}")

        # Load and summarize trades
        if self.trades_file.exists():
            try:
                df = pd.read_csv(self.trades_file)
                print(f"\nTotal trades: {len(df)}")
                if len(df) > 0:
                    print(f"Leveraged trades: {(df['leverage'] > 0).sum()}")
            except Exception as e:
                print(f"Could not load trades: {e}")


def main():
    """Main entry point for live paper trading."""
    import argparse

    parser = argparse.ArgumentParser(description='Phase 14: Live Launch + Dashboard')
    parser.add_argument('--interval', type=int, default=10,
                        help='Minutes between cycles (default: 10)')
    parser.add_argument('--cycles', type=int, default=None,
                        help='Max cycles (default: infinite)')
    parser.add_argument('--usdt', type=float, default=1000.0,
                        help='Starting USDT (default: 1000)')
    parser.add_argument('--xrp', type=float, default=500.0,
                        help='Starting XRP (default: 500)')
    parser.add_argument('--btc', type=float, default=0.0,
                        help='Starting BTC (default: 0)')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='Log directory (default: logs)')
    parser.add_argument('--no-dashboard', action='store_true',
                        help='Disable terminal dashboard')
    parser.add_argument('--plots', action='store_true',
                        help='Enable matplotlib plots (requires display)')

    args = parser.parse_args()

    initial_balance = {
        'USDT': args.usdt,
        'XRP': args.xrp,
        'BTC': args.btc
    }

    trader = LivePaperTrader(
        initial_balance,
        args.log_dir,
        enable_dashboard=not args.no_dashboard,
        enable_plots=args.plots
    )
    trader.run(interval_minutes=args.interval, max_cycles=args.cycles)


if __name__ == '__main__':
    main()
