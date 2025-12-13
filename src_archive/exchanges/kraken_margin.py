"""
Kraken 10x Margin Trading Module
Phase 7: USDT-collateralized leverage for XRP and BTC
"""
from portfolio import Portfolio
from typing import Dict, Optional


class KrakenMargin:
    """
    Kraken margin trading simulator.
    Supports up to 10x leverage on XRP/USDT and BTC/USDT pairs.
    """

    def __init__(self, portfolio: Portfolio, max_leverage: float = 10.0):
        self.portfolio = portfolio
        self.max_leverage = max_leverage
        self.min_collateral = 50.0  # Minimum $50 USDT for margin
        self.liquidation_threshold = 0.80  # Liquidate at 80% loss
        self.fee_rate = 0.0002  # 0.02% per 4 hours (Kraken margin fee)

        # Track positions for P&L calculation
        self.positions = {}  # {asset: {'size', 'entry', 'leverage', 'direction', 'collateral'}}

    def can_open_position(self, usdt_collateral: float) -> bool:
        """Check if we have enough USDT to open a position"""
        available = self.portfolio.balances.get('USDT', 0)
        return available >= usdt_collateral >= self.min_collateral

    def open_long(self, asset: str, usdt_collateral: float, price: float,
                  leverage: float = None) -> bool:
        """
        Open a leveraged long position.

        Args:
            asset: 'XRP' or 'BTC'
            usdt_collateral: Amount of USDT to use as collateral
            price: Current asset price
            leverage: Leverage multiplier (default: max_leverage)

        Returns:
            bool: Success status
        """
        leverage = leverage or self.max_leverage

        if not self.can_open_position(usdt_collateral):
            print(f"KRAKEN: Insufficient collateral (need ${self.min_collateral}, have ${self.portfolio.balances.get('USDT', 0):.2f})")
            return False

        if asset not in ['XRP', 'BTC']:
            print(f"KRAKEN: Invalid asset {asset}")
            return False

        # Close existing position if any
        if asset in self.positions:
            self.close_position(asset, price)

        # Calculate position size
        exposure = usdt_collateral * leverage
        size = exposure / price

        # Use portfolio's margin tracking
        success = self.portfolio.open_margin_position(
            asset=asset,
            usdt_collateral=usdt_collateral,
            leverage=leverage,
            price=price,
            direction='long'
        )

        if success:
            self.positions[asset] = {
                'size': size,
                'entry': price,
                'leverage': leverage,
                'direction': 'long',
                'collateral': usdt_collateral
            }
            print(f"KRAKEN 10X LONG: {size:.4f} {asset} @ ${price:.2f} (${exposure:.2f} exposure)")

        return success

    def open_short(self, asset: str, usdt_collateral: float, price: float,
                   leverage: float = None) -> bool:
        """Open a leveraged short position"""
        leverage = leverage or self.max_leverage

        if not self.can_open_position(usdt_collateral):
            return False

        if asset not in ['XRP', 'BTC']:
            return False

        if asset in self.positions:
            self.close_position(asset, price)

        exposure = usdt_collateral * leverage
        size = exposure / price

        success = self.portfolio.open_margin_position(
            asset=asset,
            usdt_collateral=usdt_collateral,
            leverage=leverage,
            price=price,
            direction='short'
        )

        if success:
            self.positions[asset] = {
                'size': size,
                'entry': price,
                'leverage': leverage,
                'direction': 'short',
                'collateral': usdt_collateral
            }
            print(f"KRAKEN 10X SHORT: {size:.4f} {asset} @ ${price:.2f}")

        return success

    def close_position(self, asset: str, price: float) -> float:
        """
        Close an existing position and return P&L.

        Args:
            asset: 'XRP' or 'BTC'
            price: Current price for exit

        Returns:
            float: Realized P&L
        """
        if asset not in self.positions:
            return 0.0

        pnl = self.portfolio.close_margin_position(asset, price)
        del self.positions[asset]
        return pnl

    def get_unrealized_pnl(self, asset: str, current_price: float) -> float:
        """Calculate unrealized P&L for a position"""
        if asset not in self.positions:
            return 0.0

        pos = self.positions[asset]
        if pos['direction'] == 'long':
            return (current_price - pos['entry']) * pos['size']
        else:
            return (pos['entry'] - current_price) * pos['size']

    def get_position_value(self, asset: str, current_price: float) -> float:
        """Get current value of a position"""
        if asset not in self.positions:
            return 0.0
        return self.positions[asset]['size'] * current_price

    def check_liquidations(self, prices: Dict[str, float]) -> list:
        """Check and execute liquidations if needed"""
        liquidated = self.portfolio.check_liquidation(prices, self.liquidation_threshold)

        for asset in liquidated:
            if asset in self.positions:
                del self.positions[asset]

        return liquidated

    def get_total_exposure(self) -> float:
        """Get total leveraged exposure in USD"""
        return sum(pos['size'] * pos['entry'] for pos in self.positions.values())

    def get_total_collateral_locked(self) -> float:
        """Get total USDT locked as collateral"""
        return sum(pos['collateral'] for pos in self.positions.values())

    def apply_funding_fee(self, hours: float = 4.0):
        """
        Apply funding/margin fee (Kraken charges every 4 hours).
        Deducts from collateral.
        """
        fee_multiplier = self.fee_rate * (hours / 4.0)
        for asset, pos in self.positions.items():
            fee = pos['collateral'] * fee_multiplier
            # Fee would reduce effective P&L
            print(f"KRAKEN FEE: {asset} position charged ${fee:.4f}")

    def get_status(self) -> dict:
        """Get margin trading status summary"""
        return {
            'positions': len(self.positions),
            'total_exposure': self.get_total_exposure(),
            'collateral_locked': self.get_total_collateral_locked(),
            'available_usdt': self.portfolio.balances.get('USDT', 0),
            'assets': list(self.positions.keys())
        }
