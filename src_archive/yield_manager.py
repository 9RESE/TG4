"""
Phase 13: Yield Manager - Real rate simulation for USDT lending yields.
Simulates Kraken/Bitrue flexible lending rates (~6-7% APY Dec 2025).
Future: Pull real-time rates via ccxt API.
"""


class YieldManager:
    """
    Manages USDT yield accrual from Kraken/Bitrue flexible lending.
    Phase 13: Real rate simulation with averaged rates.
    """

    def __init__(self):
        # Dec 2025 baseline rates (paper simulation)
        self.kraken_rate = 0.06  # ~6% APY Kraken flexible lending
        self.bitrue_rate = 0.07  # ~7% APY Bitrue flexible lending
        self.avg_rate = (self.kraken_rate + self.bitrue_rate) / 2  # 6.5% avg

        # Tracking
        self.total_yield_accrued = 0.0
        self.accrual_count = 0

    def accrue(self, usdt_balance: float, hours: float = 4) -> float:
        """
        Accrue yield on USDT balance.

        Args:
            usdt_balance: Current USDT balance
            hours: Hours since last accrual (default 4h for backtest steps)

        Returns:
            float: Yield earned this period
        """
        if usdt_balance < 100:
            return 0.0

        # Calculate yield: (avg_rate / 365 / 24) * hours * balance
        hourly_rate = self.avg_rate / 365 / 24
        yield_earned = usdt_balance * hourly_rate * hours

        self.total_yield_accrued += yield_earned
        self.accrual_count += 1

        if yield_earned > 0.01:  # Only log meaningful yields
            print(f"YIELD ACCRUED: +${yield_earned:.4f} USDT (@ ~{self.avg_rate*100:.1f}% APY avg)")

        return yield_earned

    def get_stats(self) -> dict:
        """Get yield statistics."""
        return {
            'total_yield': self.total_yield_accrued,
            'accrual_count': self.accrual_count,
            'avg_per_accrual': self.total_yield_accrued / max(self.accrual_count, 1),
            'kraken_rate': self.kraken_rate,
            'bitrue_rate': self.bitrue_rate,
            'avg_rate': self.avg_rate
        }

    def set_rates(self, kraken: float = None, bitrue: float = None):
        """
        Update lending rates (for future real-time API pulls).

        Args:
            kraken: Kraken flexible lending APY
            bitrue: Bitrue flexible lending APY
        """
        if kraken is not None:
            self.kraken_rate = kraken
        if bitrue is not None:
            self.bitrue_rate = bitrue
        self.avg_rate = (self.kraken_rate + self.bitrue_rate) / 2
        print(f"RATES UPDATED: Kraken {self.kraken_rate*100:.1f}%, Bitrue {self.bitrue_rate*100:.1f}%, Avg {self.avg_rate*100:.1f}%")

    def simulate_monthly_yield(self, usdt_balance: float) -> float:
        """Simulate monthly yield for projections."""
        hourly_rate = self.avg_rate / 365 / 24
        return usdt_balance * hourly_rate * 24 * 30  # 30 days

    def simulate_annual_yield(self, usdt_balance: float) -> float:
        """Simulate annual yield for projections."""
        return usdt_balance * self.avg_rate
