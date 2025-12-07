import numpy as np

class RiskManager:
    def __init__(self, max_drawdown=0.20, max_leverage=5.0):
        self.max_dd = max_drawdown
        self.max_lev = max_leverage

    def position_size(self, portfolio_value, volatility, confidence=0.8):
        # Kelly-inspired
        kelly = confidence - (1 - confidence) / 2.0
        size = kelly / volatility
        return min(size, 0.1) * portfolio_value  # cap 10%

    def check_liquidation(self, entry_price, current_price, leverage, direction='long'):
        if direction == 'long':
            liq_price = entry_price * (1 - 0.9 / leverage)
            return current_price <= liq_price
        else:
            liq_price = entry_price * (1 + 0.9 / leverage)
            return current_price >= liq_price
