import vectorbt as vbt
import pandas as pd
import numpy as np
from typing import Dict

class Backtester:
    def __init__(self, price_data: Dict[str, pd.DataFrame]):
        self.price_data = price_data  # dict of symbol -> OHLCV DataFrame

    def run_momentum_strategy(self, symbol: str = 'XRP/USDT', rsi_window: int = 14, rsi_upper: int = 70, rsi_lower: int = 30):
        close = self.price_data[symbol]['close']

        rsi = vbt.IndicatorFactory.from_talib('RSI').run(close, timeperiod=rsi_window).real
        entries = rsi.crosses_below(rsi_lower)  # Buy on oversold
        exits = rsi.crosses_above(rsi_upper)    # Sell on overbought

        # Bias toward holding XRP/RLUSD longer  only exit on strong overbought
        pf = vbt.Portfolio.from_signals(
            close, entries, exits,
            init_cash=1000.0,
            fees=0.001,  # 0.1% avg fee
            freq='1h'
        )
        return pf

    def run_with_lstm_signals(self, symbol: str, signals: pd.Series):
        close = self.price_data[symbol]['close']
        pf = vbt.Portfolio.from_signals(
            close, signals, ~signals,  # long when True, exit when False
            init_cash=1000.0,
            fees=0.001,
            freq='1h'
        )
        return pf
