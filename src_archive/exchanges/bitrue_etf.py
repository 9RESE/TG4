"""
Bitrue 3x Leveraged ETF Module
Phase 7: XRP3L/XRP3S and BTC3L/BTC3S ETF trading
"""
from portfolio import Portfolio
from typing import Dict


class BitrueETF:
    """
    Bitrue leveraged ETF simulator.
    Trades 3x leveraged tokens: XRP3L (3x long), XRP3S (3x short), etc.
    """

    def __init__(self, portfolio: Portfolio):
        self.portfolio = portfolio
        self.leverage = 3.0
        self.min_trade = 10.0  # Minimum $10 USDT trade
        self.fee_rate = 0.001  # 0.1% trading fee

        # ETF holdings (simulated as synthetic positions)
        self.etf_holdings = {
            'XRP3L': 0.0,  # 3x Long XRP
            'XRP3S': 0.0,  # 3x Short XRP
            'BTC3L': 0.0,  # 3x Long BTC
            'BTC3S': 0.0,  # 3x Short BTC
        }

        # Entry prices for P&L tracking
        self.entry_prices = {}

    def buy_etf(self, etf_symbol: str, usdt_amount: float, underlying_price: float) -> bool:
        """
        Buy a leveraged ETF token.

        Args:
            etf_symbol: 'XRP3L', 'XRP3S', 'BTC3L', 'BTC3S'
            usdt_amount: Amount of USDT to spend
            underlying_price: Current price of underlying asset (XRP or BTC)

        Returns:
            bool: Success status
        """
        if etf_symbol not in self.etf_holdings:
            print(f"BITRUE: Invalid ETF {etf_symbol}")
            return False

        if usdt_amount < self.min_trade:
            print(f"BITRUE: Minimum trade ${self.min_trade}")
            return False

        if self.portfolio.balances.get('USDT', 0) < usdt_amount:
            print(f"BITRUE: Insufficient USDT")
            return False

        # Apply fee
        net_amount = usdt_amount * (1 - self.fee_rate)

        # Calculate ETF tokens (simplified: 1 token = $1 notional)
        tokens = net_amount

        # Deduct USDT
        self.portfolio.update('USDT', -usdt_amount)

        # Add to holdings
        self.etf_holdings[etf_symbol] += tokens
        self.entry_prices[etf_symbol] = underlying_price

        direction = 'LONG' if etf_symbol.endswith('L') else 'SHORT'
        underlying = 'XRP' if 'XRP' in etf_symbol else 'BTC'
        print(f"BITRUE 3X {direction}: Bought ${tokens:.2f} {etf_symbol} (underlying {underlying} @ ${underlying_price:.4f})")

        return True

    def sell_etf(self, etf_symbol: str, token_amount: float = None,
                 underlying_price: float = None) -> float:
        """
        Sell ETF tokens and realize P&L.

        Args:
            etf_symbol: ETF to sell
            token_amount: Amount to sell (None = sell all)
            underlying_price: Current underlying price

        Returns:
            float: USDT received
        """
        if etf_symbol not in self.etf_holdings:
            return 0.0

        holdings = self.etf_holdings[etf_symbol]
        if holdings <= 0:
            return 0.0

        sell_amount = token_amount if token_amount else holdings
        sell_amount = min(sell_amount, holdings)

        # Calculate P&L based on 3x leverage
        entry = self.entry_prices.get(etf_symbol, underlying_price)
        if underlying_price and entry:
            is_long = etf_symbol.endswith('L')
            price_change = (underlying_price - entry) / entry

            if is_long:
                value_change = 1 + (price_change * self.leverage)
            else:
                value_change = 1 - (price_change * self.leverage)

            # ETF value can't go below 0
            value_change = max(value_change, 0)
            proceeds = sell_amount * value_change
        else:
            proceeds = sell_amount

        # Apply fee
        net_proceeds = proceeds * (1 - self.fee_rate)

        # Update holdings
        self.etf_holdings[etf_symbol] -= sell_amount
        if self.etf_holdings[etf_symbol] < 0.01:
            self.etf_holdings[etf_symbol] = 0.0
            if etf_symbol in self.entry_prices:
                del self.entry_prices[etf_symbol]

        # Credit USDT
        self.portfolio.update('USDT', net_proceeds)

        pnl = net_proceeds - sell_amount
        print(f"BITRUE SELL: {sell_amount:.2f} {etf_symbol} for ${net_proceeds:.2f} (P&L: ${pnl:+.2f})")

        return net_proceeds

    def get_etf_value(self, etf_symbol: str, underlying_price: float) -> float:
        """Calculate current value of ETF holdings"""
        holdings = self.etf_holdings.get(etf_symbol, 0)
        if holdings <= 0:
            return 0.0

        entry = self.entry_prices.get(etf_symbol, underlying_price)
        if not entry:
            return holdings

        is_long = etf_symbol.endswith('L')
        price_change = (underlying_price - entry) / entry

        if is_long:
            value_change = 1 + (price_change * self.leverage)
        else:
            value_change = 1 - (price_change * self.leverage)

        return holdings * max(value_change, 0)

    def get_total_etf_value(self, prices: Dict[str, float]) -> float:
        """Get total value of all ETF holdings"""
        xrp_price = prices.get('XRP', 2.0)
        btc_price = prices.get('BTC', 90000.0)

        total = 0.0
        for symbol in ['XRP3L', 'XRP3S']:
            total += self.get_etf_value(symbol, xrp_price)
        for symbol in ['BTC3L', 'BTC3S']:
            total += self.get_etf_value(symbol, btc_price)

        return total

    def close_all(self, prices: Dict[str, float]) -> float:
        """Close all ETF positions"""
        total_proceeds = 0.0
        xrp_price = prices.get('XRP', 2.0)
        btc_price = prices.get('BTC', 90000.0)

        for symbol in list(self.etf_holdings.keys()):
            if self.etf_holdings[symbol] > 0:
                underlying = xrp_price if 'XRP' in symbol else btc_price
                total_proceeds += self.sell_etf(symbol, underlying_price=underlying)

        return total_proceeds

    def get_status(self) -> dict:
        """Get ETF trading status"""
        return {
            'holdings': {k: v for k, v in self.etf_holdings.items() if v > 0},
            'entry_prices': self.entry_prices.copy()
        }
