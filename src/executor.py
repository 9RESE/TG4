from data_fetcher import DataFetcher
from portfolio import Portfolio
from risk_manager import RiskManager
import time

class Executor:
    """Paper trading executor with realistic simulation"""

    def __init__(self, portfolio: Portfolio):
        self.fetcher = DataFetcher()
        self.portfolio = portfolio
        self.risk = RiskManager()
        self.trade_log = []

    def place_paper_order(self, symbol: str, side: str, amount: float, leverage: float = 1.0):
        """
        Execute a paper trade with simulated fees and slippage.

        Args:
            symbol: Trading pair (e.g., 'XRP/USDT')
            side: 'buy' or 'sell'
            amount: Amount of base asset to trade
            leverage: Leverage multiplier (1.0 = spot)

        Returns:
            bool: True if order executed successfully
        """
        prices = self.fetcher.get_best_price(symbol)
        if not prices:
            print(f"No price data for {symbol}")
            return False

        avg_price = sum(prices.values()) / len(prices)

        # Simulate fees 0.1%, slippage 0.05%
        if side == 'buy':
            execution_price = avg_price * 1.0015  # Pay more when buying
        else:
            execution_price = avg_price * 0.9985  # Get less when selling

        base_asset = symbol.split('/')[0]
        quote_asset = symbol.split('/')[1]
        cost = amount * execution_price / leverage

        if side == 'buy':
            if self.portfolio.balances.get(quote_asset, 0) >= cost:
                if leverage > 1:
                    success = self.portfolio.open_margin_position(base_asset, amount, leverage, execution_price, 'long')
                else:
                    self.portfolio.update(quote_asset, -cost)
                    self.portfolio.update(base_asset, amount)
                    success = True

                if success:
                    self._log_trade(symbol, side, amount, execution_price, leverage)
                    print(f"EXECUTED: BUY {amount:.4f} {base_asset} @ {execution_price:.4f} (lev {leverage}x)")
                    return True
            else:
                print(f"Insufficient {quote_asset} balance for buy order")

        elif side == 'sell':
            holding = self.portfolio.balances.get(base_asset, 0)
            if holding >= amount:
                self.portfolio.update(base_asset, -amount)
                self.portfolio.update(quote_asset, amount * execution_price * 0.999)  # Fee on proceeds
                self._log_trade(symbol, side, amount, execution_price, leverage)
                print(f"EXECUTED: SELL {amount:.4f} {base_asset} @ {execution_price:.4f}")
                return True
            else:
                print(f"Insufficient {base_asset} balance for sell order")

        return False

    def close_position(self, asset: str, direction: str = 'long'):
        """Close an open margin position"""
        prices = self.fetcher.get_best_price(f"{asset}/USDT")
        if prices:
            avg_price = sum(prices.values()) / len(prices)
            self.portfolio.close_margin_position(asset, avg_price, direction)
            print(f"CLOSED: {direction.upper()} position on {asset} @ {avg_price:.4f}")

    def _log_trade(self, symbol: str, side: str, amount: float, price: float, leverage: float):
        """Log trade for history tracking"""
        self.trade_log.append({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'symbol': symbol,
            'side': side,
            'amount': amount,
            'price': price,
            'leverage': leverage,
            'value': amount * price
        })

    def get_trade_history(self):
        """Return trade history"""
        return self.trade_log
