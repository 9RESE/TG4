import pandas as pd
from typing import Dict

class Portfolio:
    def __init__(self, starting_balances: Dict[str, float]):
        self.balances = starting_balances.copy()
        self.history = []

    def update(self, asset: str, amount: float):
        self.balances[asset] = self.balances.get(asset, 0.0) + amount
        if self.balances[asset] < 1e-8:
            self.balances[asset] = 0.0

    def record_snapshot(self, prices: Dict[str, float]):
        total_usd = sum(self.balances.get(asset, 0) * prices.get(asset, 0) for asset in self.balances)
        snapshot = {
            'timestamp': pd.Timestamp.now(),
            'balances': self.balances.copy(),
            'total_usd': total_usd
        }
        self.history.append(snapshot)

    def get_total_usd(self, prices: Dict[str, float]) -> float:
        return sum(self.balances.get(a, 0) * prices.get(a, 0) for a in self.balances)

    def __str__(self):
        return f"Portfolio: { {k: round(v, 4) for k, v in self.balances.items()} }"
