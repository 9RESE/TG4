"""
Phase 14: Live Regime Dashboard
Real-time visualization of portfolio value, trading mode, and fear/greed indicators.
Designed for terminal-friendly display with optional matplotlib plots.
"""
import os
import sys
from datetime import datetime
from typing import List, Dict, Optional

# Optional matplotlib import for graphical display
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.animation import FuncAnimation
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class LiveDashboard:
    """
    Real-time trading dashboard with regime visualization.
    Supports both terminal output and matplotlib plots.
    """

    def __init__(self, enable_plots: bool = False):
        """
        Initialize the dashboard.

        Args:
            enable_plots: Enable matplotlib plots (requires display)
        """
        self.enable_plots = enable_plots and HAS_MATPLOTLIB
        self.portfolio_history: List[float] = []
        self.mode_history: List[str] = []
        self.timestamp_history: List[datetime] = []
        self.yield_history: List[float] = []
        self.rsi_history: Dict[str, List[float]] = {'XRP': [], 'BTC': []}
        self.volatility_history: List[float] = []

        # Fear/Greed thresholds
        self.FEAR_THRESHOLD = 0.04  # High volatility = fear
        self.GREED_THRESHOLD = 0.02  # Low volatility = greed

        # Mode colors for display
        self.MODE_COLORS = {
            'defensive': '\033[93m',  # Yellow
            'offensive': '\033[92m',  # Green
            'bear': '\033[91m',       # Red
            'unknown': '\033[0m'      # Reset
        }
        self.RESET = '\033[0m'

        if self.enable_plots:
            plt.ion()  # Interactive mode
            self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 8))
            self.fig.suptitle('TG4 Live Trading Dashboard - Phase 14')

    def get_fear_greed_index(self, volatility: float, rsi_xrp: float, rsi_btc: float) -> tuple:
        """
        Calculate fear/greed index based on volatility and RSI.

        Returns:
            (index: float 0-100, label: str)
            0-25: Extreme Fear, 25-45: Fear, 45-55: Neutral,
            55-75: Greed, 75-100: Extreme Greed
        """
        # Volatility component (0-50, higher vol = more fear)
        if volatility >= self.FEAR_THRESHOLD:
            vol_score = 10 + (volatility - self.FEAR_THRESHOLD) * 500  # Fear zone
            vol_score = min(vol_score, 25)
        elif volatility <= self.GREED_THRESHOLD:
            vol_score = 75 + (self.GREED_THRESHOLD - volatility) * 500  # Greed zone
            vol_score = min(vol_score, 90)
        else:
            # Neutral zone
            vol_score = 50

        # RSI component (average of XRP and BTC)
        avg_rsi = (rsi_xrp + rsi_btc) / 2
        if avg_rsi > 70:
            rsi_score = 75 + (avg_rsi - 70) * 0.83  # Extreme greed
        elif avg_rsi > 55:
            rsi_score = 55 + (avg_rsi - 55) * 1.33  # Greed
        elif avg_rsi < 30:
            rsi_score = 25 - (30 - avg_rsi) * 0.83  # Extreme fear
        elif avg_rsi < 45:
            rsi_score = 45 - (45 - avg_rsi) * 1.33  # Fear
        else:
            rsi_score = 50  # Neutral

        # Combined index (weighted average)
        index = 0.6 * vol_score + 0.4 * rsi_score
        index = max(0, min(100, index))

        # Label
        if index < 25:
            label = "Extreme Fear"
        elif index < 45:
            label = "Fear"
        elif index < 55:
            label = "Neutral"
        elif index < 75:
            label = "Greed"
        else:
            label = "Extreme Greed"

        return index, label

    def update(self, portfolio_value: float, mode: str, volatility: float,
               rsi_xrp: float, rsi_btc: float, yield_earned: float = 0,
               prices: Dict[str, float] = None):
        """
        Update dashboard with latest data.

        Args:
            portfolio_value: Current portfolio value in USD
            mode: Current trading mode (defensive/offensive/bear)
            volatility: Current ATR% volatility
            rsi_xrp: XRP RSI value
            rsi_btc: BTC RSI value
            yield_earned: USDT yield earned this cycle
            prices: Current asset prices
        """
        now = datetime.now()
        self.portfolio_history.append(portfolio_value)
        self.mode_history.append(mode)
        self.timestamp_history.append(now)
        self.yield_history.append(yield_earned)
        self.volatility_history.append(volatility)
        self.rsi_history['XRP'].append(rsi_xrp)
        self.rsi_history['BTC'].append(rsi_btc)

        # Calculate fear/greed
        fg_index, fg_label = self.get_fear_greed_index(volatility, rsi_xrp, rsi_btc)

        # Terminal output
        self._print_terminal_update(portfolio_value, mode, volatility,
                                    rsi_xrp, rsi_btc, fg_index, fg_label,
                                    yield_earned, prices)

        # Matplotlib update
        if self.enable_plots and len(self.portfolio_history) > 1:
            self._update_plots(fg_index, fg_label)

    def _print_terminal_update(self, portfolio_value: float, mode: str,
                               volatility: float, rsi_xrp: float, rsi_btc: float,
                               fg_index: float, fg_label: str,
                               yield_earned: float, prices: Dict[str, float]):
        """Print formatted terminal update."""
        mode_color = self.MODE_COLORS.get(mode.lower(), self.RESET)

        # Fear/Greed color
        if fg_index < 25:
            fg_color = '\033[91m'  # Red
        elif fg_index < 45:
            fg_color = '\033[93m'  # Yellow
        elif fg_index < 55:
            fg_color = '\033[0m'   # White
        elif fg_index < 75:
            fg_color = '\033[92m'  # Green
        else:
            fg_color = '\033[96m'  # Cyan

        print(f"\n{'─'*60}")
        print(f"│ {'TG4 LIVE DASHBOARD':^56} │")
        print(f"{'─'*60}")
        print(f"│ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):42} │")
        print(f"│ Portfolio: ${portfolio_value:,.2f}{'':>32} │")
        print(f"│ Mode: {mode_color}{mode.upper():10}{self.RESET}{'':>37} │")
        print(f"│ Fear/Greed: {fg_color}{fg_index:5.1f} ({fg_label}){self.RESET}{'':>24} │")
        print(f"{'─'*60}")
        print(f"│ Volatility (ATR%): {volatility*100:6.2f}%{'':>28} │")
        print(f"│ RSI XRP: {rsi_xrp:5.1f}   RSI BTC: {rsi_btc:5.1f}{'':>20} │")
        if prices:
            print(f"│ XRP: ${prices.get('XRP', 0):,.4f}   BTC: ${prices.get('BTC', 0):,.2f}{'':>10} │")
        if yield_earned > 0:
            total_yield = sum(self.yield_history)
            print(f"│ Yield This Cycle: ${yield_earned:.4f}  Total: ${total_yield:.4f}{'':>10} │")
        print(f"{'─'*60}")

    def _update_plots(self, fg_index: float, fg_label: str):
        """Update matplotlib plots."""
        if not self.enable_plots:
            return

        try:
            # Clear all axes
            for ax in self.axes.flat:
                ax.clear()

            # Plot 1: Portfolio Value
            ax1 = self.axes[0, 0]
            ax1.plot(self.timestamp_history, self.portfolio_history, 'b-', linewidth=2)
            ax1.set_title(f'Portfolio Value: ${self.portfolio_history[-1]:,.2f}')
            ax1.set_ylabel('USD')
            ax1.grid(True, alpha=0.3)
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

            # Plot 2: Fear/Greed Gauge
            ax2 = self.axes[0, 1]
            colors = ['#FF0000', '#FF6600', '#FFFF00', '#66FF00', '#00FF00']
            ax2.barh([0], [fg_index], color=colors[int(fg_index/25)] if fg_index < 100 else colors[-1])
            ax2.set_xlim(0, 100)
            ax2.set_title(f'Fear/Greed: {fg_index:.1f} ({fg_label})')
            ax2.set_yticks([])
            ax2.axvline(x=25, color='gray', linestyle='--', alpha=0.5)
            ax2.axvline(x=50, color='gray', linestyle='--', alpha=0.5)
            ax2.axvline(x=75, color='gray', linestyle='--', alpha=0.5)

            # Plot 3: RSI
            ax3 = self.axes[1, 0]
            ax3.plot(self.timestamp_history, self.rsi_history['XRP'], 'c-', label='XRP RSI')
            ax3.plot(self.timestamp_history, self.rsi_history['BTC'], 'm-', label='BTC RSI')
            ax3.axhline(y=70, color='r', linestyle='--', alpha=0.5)
            ax3.axhline(y=30, color='g', linestyle='--', alpha=0.5)
            ax3.axhline(y=68, color='orange', linestyle=':', alpha=0.5, label='Opportunistic (68)')
            ax3.set_title('RSI Indicators')
            ax3.set_ylabel('RSI')
            ax3.set_ylim(0, 100)
            ax3.legend(loc='upper right')
            ax3.grid(True, alpha=0.3)
            ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

            # Plot 4: Volatility
            ax4 = self.axes[1, 1]
            ax4.plot(self.timestamp_history, [v*100 for v in self.volatility_history], 'r-')
            ax4.axhline(y=4.2, color='orange', linestyle='--', alpha=0.5, label='Opportunistic (4.2%)')
            ax4.axhline(y=4.0, color='red', linestyle='--', alpha=0.5, label='Bear (4.0%)')
            ax4.set_title('Volatility (ATR%)')
            ax4.set_ylabel('%')
            ax4.legend(loc='upper right')
            ax4.grid(True, alpha=0.3)
            ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

            plt.tight_layout()
            plt.pause(0.1)

        except Exception as e:
            print(f"[DASHBOARD] Plot update error: {e}")

    def get_mode_distribution(self) -> Dict[str, float]:
        """Get percentage distribution of trading modes."""
        if not self.mode_history:
            return {}

        total = len(self.mode_history)
        distribution = {}
        for mode in set(self.mode_history):
            count = self.mode_history.count(mode)
            distribution[mode] = (count / total) * 100

        return distribution

    def print_summary(self):
        """Print session summary."""
        if not self.portfolio_history:
            print("[DASHBOARD] No data to summarize")
            return

        initial = self.portfolio_history[0]
        final = self.portfolio_history[-1]
        pnl = final - initial
        roi = (pnl / initial) * 100 if initial > 0 else 0
        total_yield = sum(self.yield_history)

        distribution = self.get_mode_distribution()

        print(f"\n{'═'*60}")
        print(f"│ {'SESSION SUMMARY':^56} │")
        print(f"{'═'*60}")
        print(f"│ Duration: {len(self.portfolio_history)} cycles{'':>36} │")
        print(f"│ Initial Value: ${initial:,.2f}{'':>32} │")
        print(f"│ Final Value:   ${final:,.2f}{'':>32} │")
        print(f"│ Total P&L:     ${pnl:+,.2f} ({roi:+.2f}%){'':>20} │")
        print(f"│ Total Yield:   ${total_yield:.4f}{'':>32} │")
        print(f"│ Max Value:     ${max(self.portfolio_history):,.2f}{'':>32} │")
        print(f"│ Min Value:     ${min(self.portfolio_history):,.2f}{'':>32} │")
        print(f"{'─'*60}")
        print(f"│ Mode Distribution:{'':>38} │")
        for mode, pct in distribution.items():
            print(f"│   {mode.capitalize():12}: {pct:5.1f}%{'':>32} │")
        print(f"{'═'*60}")

    def close(self):
        """Close dashboard and cleanup."""
        if self.enable_plots and HAS_MATPLOTLIB:
            plt.ioff()
            plt.close('all')


def demo_dashboard():
    """Demo the dashboard with sample data."""
    import random

    dashboard = LiveDashboard(enable_plots=HAS_MATPLOTLIB)

    print("Running dashboard demo (10 cycles)...")
    portfolio_value = 2200.0

    for i in range(10):
        # Simulate data
        volatility = random.uniform(0.02, 0.06)
        rsi_xrp = random.uniform(30, 70)
        rsi_btc = random.uniform(35, 65)

        if volatility > 0.04:
            mode = 'bear' if rsi_xrp > 65 else 'defensive'
        elif volatility < 0.02 and rsi_xrp < 35:
            mode = 'offensive'
        else:
            mode = 'defensive'

        # Simulate P&L
        if mode == 'defensive':
            portfolio_value += random.uniform(-5, 10)
        elif mode == 'offensive':
            portfolio_value += random.uniform(-20, 50)
        else:
            portfolio_value += random.uniform(-30, 40)

        yield_earned = portfolio_value * 0.065 / 365 / 24 * 0.166 if mode == 'defensive' else 0

        dashboard.update(
            portfolio_value=portfolio_value,
            mode=mode,
            volatility=volatility,
            rsi_xrp=rsi_xrp,
            rsi_btc=rsi_btc,
            yield_earned=yield_earned,
            prices={'XRP': 2.35, 'BTC': 98500}
        )

        import time
        time.sleep(1)

    dashboard.print_summary()
    dashboard.close()


if __name__ == '__main__':
    demo_dashboard()
