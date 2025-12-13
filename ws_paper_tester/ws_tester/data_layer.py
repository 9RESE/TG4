"""
Data layer for WebSocket Paper Trading Tester.
Handles WebSocket connections and market data management.
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any, Tuple
from collections import deque

import websockets

from .types import DataSnapshot, Candle, Trade, OrderbookSnapshot


class KrakenWSClient:
    """
    WebSocket client for Kraken exchange.
    Handles connection, subscription, and message routing.
    """

    WS_URL = "wss://ws.kraken.com/v2"

    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.ws = None
        self._running = False
        self._reconnect_delay = 1.0
        self._max_reconnect_delay = 60.0
        self._message_callback: Optional[Callable] = None

        # Convert symbols to Kraken format
        self._kraken_symbols = [self._to_kraken_symbol(s) for s in symbols]

    def _to_kraken_symbol(self, symbol: str) -> str:
        """Convert standard symbol to Kraken format."""
        # XRP/USD -> XRP/USD (Kraken uses this format in v2)
        return symbol

    def _from_kraken_symbol(self, kraken_symbol: str) -> str:
        """Convert Kraken symbol to standard format."""
        return kraken_symbol

    async def connect(self):
        """Connect to WebSocket."""
        try:
            self.ws = await websockets.connect(
                self.WS_URL,
                ping_interval=30,
                ping_timeout=10,
                close_timeout=5
            )
            self._running = True
            self._reconnect_delay = 1.0
            print(f"[WS] Connected to {self.WS_URL}")
            return True
        except Exception as e:
            print(f"[WS] Connection failed: {e}")
            return False

    async def subscribe(self, channels: List[str]):
        """Subscribe to channels for all symbols."""
        if not self.ws:
            return

        for channel in channels:
            if channel == 'trade':
                msg = {
                    "method": "subscribe",
                    "params": {
                        "channel": "trade",
                        "symbol": self._kraken_symbols
                    }
                }
            elif channel == 'ticker':
                msg = {
                    "method": "subscribe",
                    "params": {
                        "channel": "ticker",
                        "symbol": self._kraken_symbols
                    }
                }
            elif channel == 'book':
                msg = {
                    "method": "subscribe",
                    "params": {
                        "channel": "book",
                        "symbol": self._kraken_symbols,
                        "depth": 10
                    }
                }
            elif channel == 'ohlc':
                msg = {
                    "method": "subscribe",
                    "params": {
                        "channel": "ohlc",
                        "symbol": self._kraken_symbols,
                        "interval": 1  # 1 minute
                    }
                }
            else:
                continue

            await self.ws.send(json.dumps(msg))
            print(f"[WS] Subscribed to {channel} for {self._kraken_symbols}")

    async def run_forever(self, callback: Callable):
        """Run message loop with auto-reconnect."""
        self._message_callback = callback

        while self._running:
            try:
                if not self.ws or self.ws.closed:
                    if await self.connect():
                        await self.subscribe(['trade', 'ticker', 'book'])
                    else:
                        await asyncio.sleep(self._reconnect_delay)
                        self._reconnect_delay = min(
                            self._reconnect_delay * 2,
                            self._max_reconnect_delay
                        )
                        continue

                async for message in self.ws:
                    try:
                        data = json.loads(message)
                        if self._message_callback:
                            await self._message_callback(data)
                    except json.JSONDecodeError:
                        pass
                    except Exception as e:
                        print(f"[WS] Message handler error: {e}")

            except websockets.ConnectionClosed:
                print("[WS] Connection closed, reconnecting...")
                await asyncio.sleep(self._reconnect_delay)
            except Exception as e:
                print(f"[WS] Error: {e}")
                await asyncio.sleep(self._reconnect_delay)

    async def close(self):
        """Close WebSocket connection."""
        self._running = False
        if self.ws:
            await self.ws.close()
            print("[WS] Connection closed")


class DataManager:
    """
    Manages market data from WebSocket feed.
    Builds candles, maintains orderbooks, and creates snapshots.

    Thread Safety (MED-007):
    - Uses asyncio.Lock for candle building operations
    - Provides thread-safe snapshots via copy
    """

    MAX_CANDLES = 100  # Keep last 100 candles per timeframe
    MAX_TRADES = 100   # Keep last 100 trades per symbol

    def __init__(self, symbols: List[str]):
        self.symbols = symbols

        # Current prices
        self._prices: Dict[str, float] = {}

        # Orderbooks
        self._orderbooks: Dict[str, Dict] = {}

        # Trade tape
        self._trades: Dict[str, deque] = {s: deque(maxlen=self.MAX_TRADES) for s in symbols}

        # OHLC candles
        self._candles_1m: Dict[str, deque] = {s: deque(maxlen=self.MAX_CANDLES) for s in symbols}
        self._candles_5m: Dict[str, deque] = {s: deque(maxlen=self.MAX_CANDLES) for s in symbols}

        # Current building candles
        self._current_candle_1m: Dict[str, Dict] = {}
        self._current_candle_5m: Dict[str, Dict] = {}

        # Last update timestamps
        self._last_update: float = 0

        # Lock for thread-safe candle building (MED-007)
        self._candle_lock = asyncio.Lock()

    async def on_message(self, data: dict):
        """Process incoming WebSocket message."""
        if not isinstance(data, dict):
            return

        channel = data.get('channel')

        if channel == 'trade':
            await self._handle_trade(data)
        elif channel == 'ticker':
            await self._handle_ticker(data)
        elif channel == 'book':
            await self._handle_book(data)
        elif channel == 'ohlc':
            await self._handle_ohlc(data)

        self._last_update = time.time()

    async def _handle_trade(self, data: dict):
        """Handle trade message."""
        trades_data = data.get('data', [])

        for trade in trades_data:
            symbol = trade.get('symbol', '')
            if symbol not in self.symbols:
                continue

            # Parse trade
            price = float(trade.get('price', 0))
            size = float(trade.get('qty', 0))
            side = trade.get('side', 'buy')
            timestamp_str = trade.get('timestamp', '')

            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            except (ValueError, AttributeError) as e:
                # HIGH-007: Replace bare except with specific exceptions
                timestamp = datetime.now()

            trade_obj = Trade(
                timestamp=timestamp,
                price=price,
                size=size,
                side=side
            )

            self._trades[symbol].append(trade_obj)
            self._prices[symbol] = price

            # Update building candles (async for thread safety - MED-007)
            await self._update_building_candle_async(symbol, price, size, timestamp)

    async def _handle_ticker(self, data: dict):
        """Handle ticker message."""
        ticker_data = data.get('data', [])

        for ticker in ticker_data:
            symbol = ticker.get('symbol', '')
            if symbol not in self.symbols:
                continue

            # Extract last price
            last = ticker.get('last', 0)
            if last:
                self._prices[symbol] = float(last)

    async def _handle_book(self, data: dict):
        """Handle orderbook message."""
        book_data = data.get('data', [])

        for book in book_data:
            symbol = book.get('symbol', '')
            if symbol not in self.symbols:
                continue

            bids = book.get('bids', [])
            asks = book.get('asks', [])

            # Parse orderbook
            parsed_bids = []
            parsed_asks = []

            for bid in bids:
                price = float(bid.get('price', 0))
                qty = float(bid.get('qty', 0))
                if price > 0 and qty > 0:
                    parsed_bids.append((price, qty))

            for ask in asks:
                price = float(ask.get('price', 0))
                qty = float(ask.get('qty', 0))
                if price > 0 and qty > 0:
                    parsed_asks.append((price, qty))

            # Sort bids descending, asks ascending
            parsed_bids.sort(key=lambda x: x[0], reverse=True)
            parsed_asks.sort(key=lambda x: x[0])

            self._orderbooks[symbol] = {
                'bids': parsed_bids[:10],
                'asks': parsed_asks[:10]
            }

            # Update price from orderbook mid
            if parsed_bids and parsed_asks:
                mid = (parsed_bids[0][0] + parsed_asks[0][0]) / 2
                self._prices[symbol] = mid

    async def _handle_ohlc(self, data: dict):
        """Handle OHLC candle message."""
        ohlc_data = data.get('data', [])

        for candle in ohlc_data:
            symbol = candle.get('symbol', '')
            if symbol not in self.symbols:
                continue

            interval = candle.get('interval', 1)

            try:
                timestamp = datetime.fromisoformat(
                    candle.get('timestamp', '').replace('Z', '+00:00')
                )
            except (ValueError, AttributeError, TypeError) as e:
                # HIGH-007: Replace bare except with specific exceptions
                timestamp = datetime.now()

            candle_obj = Candle(
                timestamp=timestamp,
                open=float(candle.get('open', 0)),
                high=float(candle.get('high', 0)),
                low=float(candle.get('low', 0)),
                close=float(candle.get('close', 0)),
                volume=float(candle.get('volume', 0))
            )

            if interval == 1:
                self._candles_1m[symbol].append(candle_obj)
            elif interval == 5:
                self._candles_5m[symbol].append(candle_obj)

            # Update price
            self._prices[symbol] = candle_obj.close

    async def _update_building_candle_async(
        self,
        symbol: str,
        price: float,
        volume: float,
        timestamp: datetime
    ):
        """Update the currently building candle from trades. Thread-safe (MED-007)."""
        async with self._candle_lock:
            self._update_building_candle_internal(symbol, price, volume, timestamp)

    def _update_building_candle(
        self,
        symbol: str,
        price: float,
        volume: float,
        timestamp: datetime
    ):
        """Update the currently building candle from trades (sync version for SimulatedDataManager)."""
        self._update_building_candle_internal(symbol, price, volume, timestamp)

    def _update_building_candle_internal(
        self,
        symbol: str,
        price: float,
        volume: float,
        timestamp: datetime
    ):
        """Internal candle building logic."""
        # 1-minute candle
        minute_key = timestamp.replace(second=0, microsecond=0)

        if symbol not in self._current_candle_1m:
            self._current_candle_1m[symbol] = {
                'timestamp': minute_key,
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'volume': volume
            }
        else:
            candle = self._current_candle_1m[symbol]
            if candle['timestamp'] != minute_key:
                # New candle - save old one
                old_candle = Candle(
                    timestamp=candle['timestamp'],
                    open=candle['open'],
                    high=candle['high'],
                    low=candle['low'],
                    close=candle['close'],
                    volume=candle['volume']
                )
                self._candles_1m[symbol].append(old_candle)

                # Start new candle
                self._current_candle_1m[symbol] = {
                    'timestamp': minute_key,
                    'open': price,
                    'high': price,
                    'low': price,
                    'close': price,
                    'volume': volume
                }
            else:
                # Update existing candle
                candle['high'] = max(candle['high'], price)
                candle['low'] = min(candle['low'], price)
                candle['close'] = price
                candle['volume'] += volume

        # 5-minute candle
        five_min_key = timestamp.replace(
            minute=(timestamp.minute // 5) * 5,
            second=0,
            microsecond=0
        )

        if symbol not in self._current_candle_5m:
            self._current_candle_5m[symbol] = {
                'timestamp': five_min_key,
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'volume': volume
            }
        else:
            candle = self._current_candle_5m[symbol]
            if candle['timestamp'] != five_min_key:
                # New candle - save old one
                old_candle = Candle(
                    timestamp=candle['timestamp'],
                    open=candle['open'],
                    high=candle['high'],
                    low=candle['low'],
                    close=candle['close'],
                    volume=candle['volume']
                )
                self._candles_5m[symbol].append(old_candle)

                # Start new candle
                self._current_candle_5m[symbol] = {
                    'timestamp': five_min_key,
                    'open': price,
                    'high': price,
                    'low': price,
                    'close': price,
                    'volume': volume
                }
            else:
                # Update existing candle
                candle['high'] = max(candle['high'], price)
                candle['low'] = min(candle['low'], price)
                candle['close'] = price
                candle['volume'] += volume

    def get_snapshot(self) -> Optional[DataSnapshot]:
        """
        Create an immutable snapshot of current market data.
        Thread-safe copy of data (MED-007).
        """
        if not self._prices:
            return None

        # Build orderbook snapshots
        orderbooks = {}
        for symbol, ob in self._orderbooks.items():
            orderbooks[symbol] = OrderbookSnapshot(
                bids=tuple(tuple(b) for b in ob.get('bids', [])),
                asks=tuple(tuple(a) for a in ob.get('asks', []))
            )

        # Build trade tuples (copy for thread safety)
        trades = {}
        for symbol, trade_deque in self._trades.items():
            trades[symbol] = tuple(trade_deque)

        # Build candle tuples (copy for thread safety - MED-007)
        # Note: For full thread safety in async context, use get_snapshot_async()
        candles_1m = {}
        for symbol, candle_deque in self._candles_1m.items():
            candles_1m[symbol] = tuple(candle_deque)

        candles_5m = {}
        for symbol, candle_deque in self._candles_5m.items():
            candles_5m[symbol] = tuple(candle_deque)

        return DataSnapshot(
            timestamp=datetime.now(),
            prices=dict(self._prices),
            candles_1m=candles_1m,
            candles_5m=candles_5m,
            orderbooks=orderbooks,
            trades=trades
        )

    async def get_snapshot_async(self) -> Optional[DataSnapshot]:
        """
        Create an immutable snapshot of current market data.
        Fully thread-safe async version (MED-007).
        """
        if not self._prices:
            return None

        # Build orderbook snapshots
        orderbooks = {}
        for symbol, ob in self._orderbooks.items():
            orderbooks[symbol] = OrderbookSnapshot(
                bids=tuple(tuple(b) for b in ob.get('bids', [])),
                asks=tuple(tuple(a) for a in ob.get('asks', []))
            )

        # Build trade tuples
        trades = {}
        for symbol, trade_deque in self._trades.items():
            trades[symbol] = tuple(trade_deque)

        # Build candle tuples under lock (MED-007)
        async with self._candle_lock:
            candles_1m = {}
            for symbol, candle_deque in self._candles_1m.items():
                candles_1m[symbol] = tuple(candle_deque)

            candles_5m = {}
            for symbol, candle_deque in self._candles_5m.items():
                candles_5m[symbol] = tuple(candle_deque)

        return DataSnapshot(
            timestamp=datetime.now(),
            prices=dict(self._prices),
            candles_1m=candles_1m,
            candles_5m=candles_5m,
            orderbooks=orderbooks,
            trades=trades
        )

    def get_price(self, symbol: str) -> float:
        """Get current price for symbol."""
        return self._prices.get(symbol, 0.0)

    def get_prices(self) -> Dict[str, float]:
        """Get all current prices."""
        return dict(self._prices)

    def is_stale(self, max_age_seconds: float = 10.0) -> bool:
        """Check if data is stale."""
        if self._last_update == 0:
            return True
        return (time.time() - self._last_update) > max_age_seconds


class SimulatedDataManager(DataManager):
    """
    Simulated data manager for testing without live WebSocket.
    Generates random price movements.
    """

    def __init__(self, symbols: List[str], initial_prices: Dict[str, float] = None):
        super().__init__(symbols)

        # Set initial prices
        default_prices = {
            'XRP/USD': 2.35,
            'BTC/USD': 104500.0,
            'ETH/USD': 3900.0,
            'XRP/BTC': 0.0000225,
        }

        self._prices = initial_prices or {s: default_prices.get(s, 100.0) for s in symbols}
        self._volatility = {s: 0.001 for s in symbols}  # 0.1% per tick

        # Initialize orderbooks
        for symbol in symbols:
            price = self._prices[symbol]
            spread = price * 0.001  # 0.1% spread

            self._orderbooks[symbol] = {
                'bids': [(price - spread/2 - i*spread/10, 100.0) for i in range(10)],
                'asks': [(price + spread/2 + i*spread/10, 100.0) for i in range(10)]
            }

    async def simulate_tick(self):
        """Generate simulated price movement."""
        import random

        for symbol in self.symbols:
            price = self._prices[symbol]
            volatility = self._volatility[symbol]

            # Random walk
            change = random.gauss(0, volatility)
            new_price = price * (1 + change)
            self._prices[symbol] = new_price

            # Update orderbook
            spread = new_price * 0.001
            self._orderbooks[symbol] = {
                'bids': [(new_price - spread/2 - i*spread/10, 100.0 + random.random()*50) for i in range(10)],
                'asks': [(new_price + spread/2 + i*spread/10, 100.0 + random.random()*50) for i in range(10)]
            }

            # Generate simulated trade
            side = 'buy' if change > 0 else 'sell'
            trade = Trade(
                timestamp=datetime.now(),
                price=new_price,
                size=random.uniform(10, 100),
                side=side
            )
            self._trades[symbol].append(trade)

            # Update building candle
            self._update_building_candle(symbol, new_price, trade.size, trade.timestamp)

        self._last_update = time.time()
