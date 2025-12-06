import ccxt
import pandas as pd
import time
import os

class DataFetcher:
    def __init__(self):
        self.exchanges = {
            'kraken': ccxt.kraken(),
            'blofin': ccxt.blofin(),
            'bitrue': ccxt.bitrue()
        }

    def fetch_ohlcv(self, exchange_name: str, symbol: str, timeframe: str = '1h', limit: int = 1000):
        exchange = self.exchanges[exchange_name]
        try:
            data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            print(f"Error fetching {symbol} from {exchange_name}: {e}")
            return pd.DataFrame()

    def get_best_price(self, symbol: str):
        prices = {}
        for name, ex in self.exchanges.items():
            try:
                ticker = ex.fetch_ticker(symbol)
                prices[name] = (ticker['bid'] + ticker['ask']) / 2
            except:
                pass
        return prices
