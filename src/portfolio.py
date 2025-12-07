import pandas as pd
from typing import Dict


class Portfolio:
    """
    Portfolio manager for BTC, XRP, USDT holdings.
    Phase 7: Simplified to 3 assets with margin position tracking.
    """

    def __init__(self, starting_balances: Dict[str, float]):
        # Core assets only: BTC, XRP, USDT
        self.balances = {
            'BTC': starting_balances.get('BTC', 0.0),
            'XRP': starting_balances.get('XRP', 0.0),
            'USDT': starting_balances.get('USDT', 0.0)
        }
        self.history = []

        # Margin positions: {asset: {'size': x, 'entry': y, 'leverage': z, 'direction': 'long/short'}}
        self.margin_positions = {}

    def update(self, asset: str, amount: float):
        """Update spot balance for an asset"""
        if asset not in ['BTC', 'XRP', 'USDT']:
            return  # Ignore non-core assets
        self.balances[asset] = self.balances.get(asset, 0.0) + amount
        if self.balances[asset] < 1e-8:
            self.balances[asset] = 0.0

    def record_snapshot(self, prices: Dict[str, float]):
        """Record portfolio state for history tracking"""
        total_usd = self.get_total_usd(prices)
        snapshot = {
            'timestamp': pd.Timestamp.now(),
            'balances': self.balances.copy(),
            'margin_positions': self.margin_positions.copy(),
            'total_usd': total_usd
        }
        self.history.append(snapshot)

    def get_total_usd(self, prices: Dict[str, float]) -> float:
        """Calculate total portfolio value in USD including margin P&L"""
        spot_value = sum(self.balances.get(a, 0) * prices.get(a, 1.0) for a in self.balances)

        # Add unrealized margin P&L
        margin_pnl = 0.0
        for asset, pos in self.margin_positions.items():
            if pos.get('size', 0) > 0:
                current_price = prices.get(asset, pos['entry'])
                if pos['direction'] == 'long':
                    margin_pnl += (current_price - pos['entry']) * pos['size']
                else:
                    margin_pnl += (pos['entry'] - current_price) * pos['size']

        return spot_value + margin_pnl

    def __str__(self):
        return f"Portfolio: { {k: round(v, 4) for k, v in self.balances.items() if v > 0} }"

    def open_margin_position(self, asset: str, usdt_collateral: float, leverage: float,
                             price: float, direction: str = 'long'):
        """
        Open a leveraged margin position.

        Args:
            asset: BTC or XRP
            usdt_collateral: USDT to use as margin collateral
            leverage: Leverage multiplier (e.g., 10 for Kraken)
            price: Entry price
            direction: 'long' or 'short'

        Returns:
            bool: Success status
        """
        if self.balances.get('USDT', 0) < usdt_collateral:
            return False

        if asset not in ['BTC', 'XRP']:
            return False

        # Calculate position size
        exposure = usdt_collateral * leverage
        size = exposure / price

        # Lock collateral
        self.update('USDT', -usdt_collateral)

        # Record position
        self.margin_positions[asset] = {
            'size': size,
            'entry': price,
            'leverage': leverage,
            'direction': direction,
            'collateral': usdt_collateral
        }

        print(f"MARGIN OPEN: {direction.upper()} {size:.4f} {asset} @ ${price:.2f} ({leverage}x, ${usdt_collateral:.2f} collateral)")
        return True

    def close_margin_position(self, asset: str, price: float):
        """
        Close a margin position and realize P&L.

        Args:
            asset: BTC or XRP
            price: Exit price

        Returns:
            float: Realized P&L
        """
        if asset not in self.margin_positions:
            return 0.0

        pos = self.margin_positions[asset]
        size = pos['size']
        entry = pos['entry']
        direction = pos['direction']
        collateral = pos['collateral']

        # Calculate P&L
        if direction == 'long':
            pnl = (price - entry) * size
        else:
            pnl = (entry - price) * size

        # Return collateral + P&L
        total_return = collateral + pnl
        self.update('USDT', max(total_return, 0))  # Can't go below 0 (liquidation)

        # Clear position
        del self.margin_positions[asset]

        pnl_pct = (pnl / collateral) * 100 if collateral > 0 else 0
        print(f"MARGIN CLOSE: {asset} @ ${price:.2f} | P&L: ${pnl:.2f} ({pnl_pct:+.1f}%)")
        return pnl

    def get_margin_exposure(self) -> float:
        """Get total margin exposure in USD"""
        return sum(pos['size'] * pos['entry'] for pos in self.margin_positions.values())

    def check_liquidation(self, prices: Dict[str, float], liquidation_threshold: float = 0.8):
        """
        Check if any positions should be liquidated.
        Liquidation occurs when losses exceed (1 - threshold) of collateral.
        """
        liquidated = []
        for asset, pos in list(self.margin_positions.items()):
            current_price = prices.get(asset, pos['entry'])
            if pos['direction'] == 'long':
                pnl = (current_price - pos['entry']) * pos['size']
            else:
                pnl = (pos['entry'] - current_price) * pos['size']

            # Check if loss exceeds threshold
            if pnl < -pos['collateral'] * liquidation_threshold:
                print(f"LIQUIDATION: {asset} position liquidated!")
                del self.margin_positions[asset]
                liquidated.append(asset)

        return liquidated
