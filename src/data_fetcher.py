import ccxt
import pandas as pd
import time
import os

class DataFetcher:
    def __init__(self):
        self.exchanges = {
            'kraken': ccxt.kraken(),
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

    def fetch_rlusd_pairs(self):
        """Fetch RLUSD-specific trading pairs from Kraken"""
        symbols = ['XRP/RLUSD', 'RLUSD/USDT', 'BTC/RLUSD', 'RLUSD/USD']
        data = {}
        for sym in symbols:
            df = self.fetch_ohlcv('kraken', sym, '1h', 2000)
            if not df.empty:
                data[sym] = df
                print(f"Fetched {sym}: {len(df)} candles")
        return data
