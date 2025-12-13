import ccxt
import pandas as pd
import time
import os
from typing import Dict, List, Optional, Any


class DataFetcher:
    """
    Multi-exchange OHLCV data fetcher with support for multiple timeframes.

    Phase 25: Added support for scalping timeframes (5m, 15m) and
    multi-timeframe data fetching for the intraday_scalper strategy.
    """

    # Supported timeframes by exchange
    KRAKEN_TIMEFRAMES = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w']

    def __init__(self):
        self.exchanges = {
            'kraken': ccxt.kraken(),
            'bitrue': ccxt.bitrue()
        }
        # Cache for rate limiting
        self._last_fetch: Dict[str, float] = {}
        self._cache: Dict[str, pd.DataFrame] = {}
        self._cache_ttl = 60  # Cache for 60 seconds

    def fetch_ohlcv(self, exchange_name: str, symbol: str, timeframe: str = '1h',
                    limit: int = 1000, use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch OHLCV data from exchange.

        Args:
            exchange_name: Name of the exchange ('kraken', 'bitrue')
            symbol: Trading pair (e.g., 'XRP/USD', 'BTC/USD')
            timeframe: Candle timeframe ('1m', '5m', '15m', '30m', '1h', '4h', '1d')
            limit: Number of candles to fetch
            use_cache: Whether to use cached data if available

        Returns:
            DataFrame with OHLCV data indexed by timestamp
        """
        cache_key = f"{exchange_name}:{symbol}:{timeframe}"

        # Check cache
        if use_cache and cache_key in self._cache:
            last_fetch = self._last_fetch.get(cache_key, 0)
            if time.time() - last_fetch < self._cache_ttl:
                return self._cache[cache_key].copy()

        exchange = self.exchanges.get(exchange_name)
        if not exchange:
            print(f"Unknown exchange: {exchange_name}")
            return pd.DataFrame()

        try:
            data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            # Update cache
            self._cache[cache_key] = df.copy()
            self._last_fetch[cache_key] = time.time()

            return df

        except Exception as e:
            print(f"Error fetching {symbol} ({timeframe}) from {exchange_name}: {e}")
            return pd.DataFrame()

    def fetch_multi_timeframe(self, exchange_name: str, symbol: str,
                              timeframes: List[str] = None,
                              limits: Dict[str, int] = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple timeframes at once.

        Args:
            exchange_name: Name of the exchange
            symbol: Trading pair
            timeframes: List of timeframes to fetch (default: ['5m', '15m', '1h'])
            limits: Dict of timeframe -> limit (default: optimized per timeframe)

        Returns:
            Dict mapping timeframe suffix to DataFrame
            e.g., {'XRP/USDT_5m': df, 'XRP/USDT_15m': df, 'XRP/USDT': df}
        """
        if timeframes is None:
            timeframes = ['5m', '15m', '1h']

        if limits is None:
            # Default limits per timeframe
            limits = {
                '1m': 500,
                '5m': 500,
                '15m': 300,
                '30m': 200,
                '1h': 500,
                '4h': 200,
                '1d': 100
            }

        result = {}

        for tf in timeframes:
            limit = limits.get(tf, 500)

            # Convert symbol to exchange format (USDT -> USD for Kraken)
            exchange_symbol = symbol.replace('USDT', 'USD') if exchange_name == 'kraken' else symbol

            df = self.fetch_ohlcv(exchange_name, exchange_symbol, tf, limit)

            if not df.empty:
                # Store with timeframe suffix for 5m/15m, plain for 1h
                if tf == '1h':
                    result[symbol] = df
                else:
                    result[f"{symbol}_{tf}"] = df

        return result

    def fetch_scalper_data(self, symbols: List[str] = None,
                           exchange_name: str = 'kraken') -> Dict[str, pd.DataFrame]:
        """
        Fetch data optimized for the IntraDayScalper strategy.

        Fetches 5m, 15m, and 1h data for each symbol to allow the scalper
        to use the most appropriate timeframe.

        Args:
            symbols: List of symbols to fetch (default: XRP/USDT, BTC/USDT)
            exchange_name: Exchange to use

        Returns:
            Dict with data for each symbol and timeframe
        """
        if symbols is None:
            symbols = ['XRP/USDT', 'BTC/USDT']

        all_data = {}

        for symbol in symbols:
            tf_data = self.fetch_multi_timeframe(
                exchange_name,
                symbol,
                timeframes=['5m', '15m', '1h'],
                limits={'5m': 500, '15m': 300, '1h': 500}
            )
            all_data.update(tf_data)

        return all_data

    def get_best_price(self, symbol: str) -> Dict[str, float]:
        """Get the best (mid) price from all exchanges."""
        prices = {}
        for name, ex in self.exchanges.items():
            try:
                ticker = ex.fetch_ticker(symbol)
                prices[name] = (ticker['bid'] + ticker['ask']) / 2
            except:
                pass
        return prices

    def get_ticker(self, exchange_name: str, symbol: str) -> Optional[Dict[str, Any]]:
        """Get full ticker information from exchange."""
        exchange = self.exchanges.get(exchange_name)
        if not exchange:
            return None

        try:
            return exchange.fetch_ticker(symbol)
        except Exception as e:
            print(f"Error fetching ticker {symbol} from {exchange_name}: {e}")
            return None

    def fetch_rlusd_pairs(self) -> Dict[str, pd.DataFrame]:
        """Fetch RLUSD-specific trading pairs from Kraken."""
        symbols = ['XRP/RLUSD', 'RLUSD/USDT', 'BTC/RLUSD', 'RLUSD/USD']
        data = {}
        for sym in symbols:
            df = self.fetch_ohlcv('kraken', sym, '1h', 2000)
            if not df.empty:
                data[sym] = df
                print(f"Fetched {sym}: {len(df)} candles")
        return data

    def clear_cache(self):
        """Clear the data cache."""
        self._cache.clear()
        self._last_fetch.clear()
