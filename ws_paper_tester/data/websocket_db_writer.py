"""
WebSocket Database Writer - Write WebSocket data to TimescaleDB in real-time.

This module provides:
- DatabaseWriter: Buffered async writer for trades and candles
- WebSocketDBIntegration: Integration layer for KrakenWSClient
- integrate_db_writer: Utility to hook DB writing into existing WS client
"""

import asyncio
import logging
from collections import deque
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

try:
    import asyncpg
except ImportError:
    asyncpg = None

from .types import TradeRecord, CandleRecord

logger = logging.getLogger(__name__)


class DatabaseWriter:
    """
    Asynchronous database writer with buffering for efficient batch inserts.

    Features:
    - Buffered writes to reduce database round-trips
    - Automatic flush on buffer size or time interval
    - Connection pooling for concurrent writes
    - Graceful error handling with retry logic

    Usage:
        db_writer = DatabaseWriter(db_url)
        await db_writer.start()

        await db_writer.write_trade(trade_record)
        await db_writer.write_candle(candle_record)

        await db_writer.stop()
    """

    def __init__(
        self,
        db_url: str,
        trade_buffer_size: int = 100,
        trade_flush_interval: float = 5.0,
        candle_flush_interval: float = 1.0,
        pool_min_size: int = 2,
        pool_max_size: int = 10,
    ):
        """
        Initialize DatabaseWriter.

        Args:
            db_url: PostgreSQL connection URL
            trade_buffer_size: Number of trades to buffer before flushing
            trade_flush_interval: Max seconds between trade flushes
            candle_flush_interval: Max seconds between candle flushes
            pool_min_size: Minimum connection pool size
            pool_max_size: Maximum connection pool size
        """
        if asyncpg is None:
            raise ImportError("asyncpg is required for DatabaseWriter. Install with: pip install asyncpg")

        self.db_url = db_url
        self.trade_buffer_size = trade_buffer_size
        self.trade_flush_interval = trade_flush_interval
        self.candle_flush_interval = candle_flush_interval
        self.pool_min_size = pool_min_size
        self.pool_max_size = pool_max_size

        self.pool: Optional[asyncpg.Pool] = None
        self.trade_buffer: deque[TradeRecord] = deque()
        self.candle_buffer: deque[CandleRecord] = deque()

        self._running = False
        self._flush_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

        # Statistics
        self._trades_written = 0
        self._candles_written = 0
        self._flush_count = 0
        self._error_count = 0

    async def start(self):
        """Initialize database connection and start flush task."""
        self.pool = await asyncpg.create_pool(
            self.db_url,
            min_size=self.pool_min_size,
            max_size=self.pool_max_size,
            command_timeout=60
        )

        self._running = True
        self._flush_task = asyncio.create_task(self._periodic_flush())

        logger.info("DatabaseWriter started")

    async def stop(self):
        """Stop writer and flush remaining data."""
        self._running = False

        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        # Final flush
        await self._flush_trades()
        await self._flush_candles()

        if self.pool:
            await self.pool.close()

        logger.info(
            f"DatabaseWriter stopped. "
            f"Trades: {self._trades_written}, Candles: {self._candles_written}, "
            f"Flushes: {self._flush_count}, Errors: {self._error_count}"
        )

    async def write_trade(self, trade: TradeRecord):
        """
        Buffer a trade for batch insertion.

        Flushes immediately if buffer is full.
        """
        async with self._lock:
            self.trade_buffer.append(trade)

            if len(self.trade_buffer) >= self.trade_buffer_size:
                await self._flush_trades()

    async def write_candle(self, candle: CandleRecord):
        """
        Buffer a candle for insertion.

        Candles are upserted to handle updates to the current candle.
        """
        async with self._lock:
            self.candle_buffer.append(candle)

    async def _periodic_flush(self):
        """Periodic flush task."""
        while self._running:
            try:
                await asyncio.sleep(min(self.trade_flush_interval, self.candle_flush_interval))

                async with self._lock:
                    if self.trade_buffer:
                        await self._flush_trades()
                    if self.candle_buffer:
                        await self._flush_candles()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Periodic flush error: {e}")
                self._error_count += 1

    async def _flush_trades(self):
        """Flush trade buffer to database."""
        if not self.trade_buffer:
            return

        trades = list(self.trade_buffer)
        self.trade_buffer.clear()

        try:
            async with self.pool.acquire() as conn:
                # Use COPY for efficient bulk insert
                await conn.copy_records_to_table(
                    'trades',
                    records=[
                        (t.symbol, t.timestamp, t.price, t.volume, t.side, None, None)
                        for t in trades
                    ],
                    columns=['symbol', 'timestamp', 'price', 'volume', 'side', 'order_type', 'misc']
                )

            self._trades_written += len(trades)
            self._flush_count += 1
            logger.debug(f"Flushed {len(trades)} trades to database")

        except Exception as e:
            logger.error(f"Failed to flush trades: {e}")
            self._error_count += 1
            # Re-add to buffer for retry (prepend to maintain order)
            self.trade_buffer.extendleft(reversed(trades))

    async def _flush_candles(self):
        """Flush candle buffer to database."""
        if not self.candle_buffer:
            return

        candles = list(self.candle_buffer)
        self.candle_buffer.clear()

        try:
            async with self.pool.acquire() as conn:
                await conn.executemany(
                    """
                    INSERT INTO candles
                        (symbol, timestamp, interval_minutes, open, high, low, close,
                         volume, quote_volume, trade_count, vwap)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    ON CONFLICT (timestamp, symbol, interval_minutes)
                    DO UPDATE SET
                        high = GREATEST(candles.high, EXCLUDED.high),
                        low = LEAST(candles.low, EXCLUDED.low),
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume,
                        trade_count = EXCLUDED.trade_count,
                        vwap = EXCLUDED.vwap
                    """,
                    [c.to_tuple() for c in candles]
                )

            self._candles_written += len(candles)
            self._flush_count += 1
            logger.debug(f"Flushed {len(candles)} candles to database")

        except Exception as e:
            logger.error(f"Failed to flush candles: {e}")
            self._error_count += 1
            self.candle_buffer.extendleft(reversed(candles))

    async def update_sync_status(self, symbol: str, data_type: str, timestamp: datetime):
        """Update the sync status for gap detection."""
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO data_sync_status (symbol, data_type, newest_timestamp, last_sync_at)
                    VALUES ($1, $2, $3, NOW())
                    ON CONFLICT (symbol, data_type) DO UPDATE SET
                        newest_timestamp = GREATEST(data_sync_status.newest_timestamp, EXCLUDED.newest_timestamp),
                        last_sync_at = NOW()
                    """,
                    symbol, data_type, timestamp
                )
        except Exception as e:
            logger.error(f"Failed to update sync status: {e}")
            self._error_count += 1

    def get_stats(self) -> dict:
        """Get writer statistics."""
        return {
            'trades_written': self._trades_written,
            'candles_written': self._candles_written,
            'flush_count': self._flush_count,
            'error_count': self._error_count,
            'trade_buffer_size': len(self.trade_buffer),
            'candle_buffer_size': len(self.candle_buffer),
        }


class WebSocketDBIntegration:
    """
    Integration layer between Kraken WebSocket client and database writer.

    Hooks into the existing ws_paper_tester WebSocket client to persist data.
    Handles message parsing and format conversion.
    """

    def __init__(self, db_writer: DatabaseWriter):
        """
        Initialize WebSocket DB integration.

        Args:
            db_writer: DatabaseWriter instance for data persistence
        """
        self.db_writer = db_writer

        # Track current candles for proper OHLC handling
        self._current_candles: dict[tuple[str, int], CandleRecord] = {}

    async def on_trade(self, symbol: str, trade_data: dict):
        """
        Handle incoming trade from WebSocket.

        Called by KrakenWSClient on each trade message.

        Args:
            symbol: Trading pair (e.g., 'XRP/USDT')
            trade_data: Trade data from Kraken WebSocket
                {
                    'price': '0.5234',
                    'qty': '100.5',
                    'timestamp': '2024-01-15T10:30:45.123456Z',
                    'side': 'buy'
                }
        """
        try:
            timestamp_str = trade_data.get('timestamp', '')
            if timestamp_str:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            else:
                timestamp = datetime.now(timezone.utc)

            trade = TradeRecord(
                symbol=symbol,
                timestamp=timestamp,
                price=Decimal(str(trade_data.get('price', 0))),
                volume=Decimal(str(trade_data.get('qty', 0))),
                side=trade_data.get('side', 'buy')
            )

            await self.db_writer.write_trade(trade)

        except Exception as e:
            logger.error(f"Error processing trade for {symbol}: {e}")

    async def on_ohlc(self, symbol: str, ohlc_data: dict, interval: int = 1):
        """
        Handle incoming OHLC candle from WebSocket.

        Args:
            symbol: Trading pair (e.g., 'XRP/USDT')
            ohlc_data: OHLC data from Kraken WebSocket
                {
                    'timestamp': '2024-01-15T10:30:00Z',
                    'open': '0.5200',
                    'high': '0.5250',
                    'low': '0.5190',
                    'close': '0.5234',
                    'volume': '50000.5',
                    'trades': 150,
                    'vwap': '0.5220'
                }
            interval: Candle interval in minutes
        """
        try:
            timestamp_str = ohlc_data.get('timestamp', '')
            if timestamp_str:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            else:
                timestamp = datetime.now(timezone.utc)

            candle = CandleRecord(
                symbol=symbol,
                timestamp=timestamp,
                interval_minutes=interval,
                open=Decimal(str(ohlc_data.get('open', 0))),
                high=Decimal(str(ohlc_data.get('high', 0))),
                low=Decimal(str(ohlc_data.get('low', 0))),
                close=Decimal(str(ohlc_data.get('close', 0))),
                volume=Decimal(str(ohlc_data.get('volume', 0))),
                trade_count=int(ohlc_data.get('trades', 0)),
                vwap=Decimal(str(ohlc_data['vwap'])) if ohlc_data.get('vwap') else None
            )

            # Track for duplicate detection
            key = (symbol, interval)
            prev_candle = self._current_candles.get(key)

            # Only write if this is a new candle or an update to current
            if prev_candle is None or prev_candle.timestamp != candle.timestamp:
                # Previous candle is complete, flush it
                if prev_candle is not None:
                    await self.db_writer.write_candle(prev_candle)

                self._current_candles[key] = candle
            else:
                # Update current candle
                self._current_candles[key] = candle

        except Exception as e:
            logger.error(f"Error processing OHLC for {symbol}: {e}")

    async def flush_current_candles(self):
        """Flush all current candles (call on shutdown)."""
        for candle in self._current_candles.values():
            await self.db_writer.write_candle(candle)
        self._current_candles.clear()


def integrate_db_writer(ws_client, db_writer: DatabaseWriter) -> WebSocketDBIntegration:
    """
    Integrate database writer with existing WebSocket client.

    This function wraps the WebSocket client's handlers to persist data
    to the database while maintaining existing functionality.

    Usage:
        db_writer = DatabaseWriter(db_url)
        await db_writer.start()

        ws_client = KrakenWSClient(...)
        integration = integrate_db_writer(ws_client, db_writer)

        await ws_client.connect()
        # ... run your trading logic ...

        await integration.flush_current_candles()
        await db_writer.stop()

    Args:
        ws_client: KrakenWSClient instance
        db_writer: DatabaseWriter instance

    Returns:
        WebSocketDBIntegration instance for managing the integration
    """
    integration = WebSocketDBIntegration(db_writer)

    # Store original handlers
    original_on_trade = getattr(ws_client, 'on_trade', None)
    original_on_ohlc = getattr(ws_client, 'on_ohlc', None)

    async def wrapped_on_trade(symbol: str, trade_data: dict):
        # Write to database
        await integration.on_trade(symbol, trade_data)
        # Call original handler
        if original_on_trade:
            if asyncio.iscoroutinefunction(original_on_trade):
                await original_on_trade(symbol, trade_data)
            else:
                original_on_trade(symbol, trade_data)

    async def wrapped_on_ohlc(symbol: str, ohlc_data: dict, interval: int = 1):
        # Write to database
        await integration.on_ohlc(symbol, ohlc_data, interval)
        # Call original handler
        if original_on_ohlc:
            if asyncio.iscoroutinefunction(original_on_ohlc):
                await original_on_ohlc(symbol, ohlc_data, interval)
            else:
                original_on_ohlc(symbol, ohlc_data, interval)

    # Monkey-patch handlers
    ws_client.on_trade = wrapped_on_trade
    ws_client.on_ohlc = wrapped_on_ohlc

    logger.info("Database writer integrated with WebSocket client")

    return integration
