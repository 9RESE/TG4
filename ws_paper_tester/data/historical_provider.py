"""
Historical Data Provider - Query historical data for backtesting and strategy warmup.

This module provides efficient access to historical candle data stored in TimescaleDB
for use in backtesting, strategy warmup, and analysis.

Usage:
    provider = HistoricalDataProvider(db_url)
    await provider.connect()

    # Get candles for a time range
    candles = await provider.get_candles('XRP/USDT', 1, start, end)

    # Get latest N candles for indicator warmup
    warmup_data = await provider.get_warmup_data('XRP/USDT', 5, 200)

    # Replay candles for backtesting
    async for candle in provider.replay_candles('XRP/USDT', 1, start, end):
        await strategy.on_candle(candle)

    await provider.close()
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import List, Optional, AsyncIterator

try:
    import asyncpg
except ImportError:
    asyncpg = None

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Candle:
    """
    Candle data structure optimized for database operations.

    REC-006: Design rationale - There are two Candle types in the system:

    1. `Candle` (this class) - Lightweight database-optimized type:
       - Used by HistoricalDataProvider for query results
       - Includes `from_row()` for efficient asyncpg Record conversion
       - Compatible with the existing ws_paper_tester strategy framework
       - Contains essential OHLCV fields and common computed properties

    2. `HistoricalCandle` (in types.py) - Full domain type:
       - Includes additional fields (quote_volume)
       - Additional computed properties (upper_wick, lower_wick)
       - Used for internal data representation and storage

    Both types are frozen dataclasses for thread safety.
    For most use cases (strategy warmup, backtesting), use this `Candle` type
    via HistoricalDataProvider methods.
    """
    symbol: str
    timestamp: datetime
    interval_minutes: int
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    trade_count: int
    vwap: Optional[Decimal]

    @classmethod
    def from_row(cls, row: asyncpg.Record) -> 'Candle':
        """Create Candle from database row."""
        return cls(
            symbol=row['symbol'],
            timestamp=row['timestamp'],
            interval_minutes=row['interval_minutes'],
            open=row['open'],
            high=row['high'],
            low=row['low'],
            close=row['close'],
            volume=row['volume'],
            trade_count=row['trade_count'] or 0,
            vwap=row['vwap']
        )

    @property
    def typical_price(self) -> Decimal:
        """Typical price (HLC average)."""
        return (self.high + self.low + self.close) / 3

    @property
    def range(self) -> Decimal:
        """Candle range (high - low)."""
        return self.high - self.low

    @property
    def body_size(self) -> Decimal:
        """Absolute size of candle body."""
        return abs(self.close - self.open)

    @property
    def is_bullish(self) -> bool:
        """True if close > open."""
        return self.close > self.open

    def to_dict(self) -> dict:
        """Convert to dictionary for compatibility."""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp,
            'interval_minutes': self.interval_minutes,
            'open': float(self.open),
            'high': float(self.high),
            'low': float(self.low),
            'close': float(self.close),
            'volume': float(self.volume),
            'trade_count': self.trade_count,
            'vwap': float(self.vwap) if self.vwap else None,
        }


class HistoricalDataProvider:
    """
    Provides historical candle data for:
    - Strategy warmup (loading indicator history on startup)
    - Backtesting (replaying historical data through strategies)
    - Analysis (querying historical patterns)

    Automatically routes queries to the appropriate view/table based on
    the requested interval (base candles table or continuous aggregates).
    """

    # Mapping of interval minutes to continuous aggregate views
    INTERVAL_VIEWS = {
        1: 'candles',
        5: 'candles_5m',
        15: 'candles_15m',
        30: 'candles_30m',
        60: 'candles_1h',
        240: 'candles_4h',
        720: 'candles_12h',
        1440: 'candles_1d',
        10080: 'candles_1w',
    }

    def __init__(
        self,
        db_url: str,
        pool_min_size: int = 2,
        pool_max_size: int = 10,
    ):
        """
        Initialize HistoricalDataProvider.

        Args:
            db_url: PostgreSQL connection URL
            pool_min_size: Minimum connection pool size
            pool_max_size: Maximum connection pool size
        """
        if asyncpg is None:
            raise ImportError("asyncpg is required. Install with: pip install asyncpg")

        self.db_url = db_url
        self.pool_min_size = pool_min_size
        self.pool_max_size = pool_max_size
        self.pool: Optional[asyncpg.Pool] = None

    async def connect(self):
        """Establish database connection."""
        self.pool = await asyncpg.create_pool(
            self.db_url,
            min_size=self.pool_min_size,
            max_size=self.pool_max_size
        )
        logger.info("HistoricalDataProvider connected")

    async def close(self):
        """Close database connection."""
        if self.pool:
            await self.pool.close()

    def _get_view_for_interval(self, interval_minutes: int) -> str:
        """Get the appropriate view/table for the interval."""
        # REC-001: Validate interval to prevent any potential SQL issues
        if not isinstance(interval_minutes, int) or interval_minutes < 1:
            raise ValueError(f"Invalid interval: {interval_minutes}")

        if interval_minutes in self.INTERVAL_VIEWS:
            return self.INTERVAL_VIEWS[interval_minutes]

        # For non-standard intervals, use base candles table
        return 'candles'

    def _ensure_connected(self):
        """
        Ensure database connection is established.

        Raises:
            RuntimeError: If not connected to database.
        """
        if not self.pool:
            raise RuntimeError(
                "HistoricalDataProvider not connected. "
                "Call await provider.connect() first."
            )

    async def get_candles(
        self,
        symbol: str,
        interval_minutes: int,
        start: datetime,
        end: datetime,
        limit: Optional[int] = None
    ) -> List[Candle]:
        """
        Query historical candles.

        Args:
            symbol: Trading pair (e.g., 'XRP/USDT')
            interval_minutes: Candle interval (1, 5, 15, 60, etc.)
            start: Start time (inclusive)
            end: End time (exclusive)
            limit: Maximum number of candles to return

        Returns:
            List of Candle objects, sorted by timestamp ascending

        Raises:
            RuntimeError: If not connected to database.
        """
        self._ensure_connected()  # REC-002
        view = self._get_view_for_interval(interval_minutes)

        query = f"""
            SELECT symbol, timestamp, {interval_minutes} as interval_minutes,
                   open, high, low, close, volume, trade_count, vwap
            FROM {view}
            WHERE symbol = $1
              AND timestamp >= $2
              AND timestamp < $3
        """

        if view == 'candles':
            query = f"""
                SELECT symbol, timestamp, interval_minutes,
                       open, high, low, close, volume, trade_count, vwap
                FROM {view}
                WHERE symbol = $1
                  AND timestamp >= $2
                  AND timestamp < $3
                  AND interval_minutes = $4
            """
            params = [symbol, start, end, interval_minutes]
        else:
            params = [symbol, start, end]

        query += " ORDER BY timestamp ASC"

        if limit:
            query += f" LIMIT {limit}"

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        return [Candle.from_row(row) for row in rows]

    async def get_latest_candles(
        self,
        symbol: str,
        interval_minutes: int,
        count: int
    ) -> List[Candle]:
        """
        Get the N most recent candles.

        Args:
            symbol: Trading pair
            interval_minutes: Candle interval
            count: Number of candles to retrieve

        Returns:
            List of Candle objects, sorted by timestamp ascending

        Raises:
            RuntimeError: If not connected to database.
        """
        self._ensure_connected()  # REC-002
        view = self._get_view_for_interval(interval_minutes)

        if view == 'candles':
            query = f"""
                SELECT symbol, timestamp, interval_minutes,
                       open, high, low, close, volume, trade_count, vwap
                FROM {view}
                WHERE symbol = $1 AND interval_minutes = $2
                ORDER BY timestamp DESC
                LIMIT $3
            """
            params = [symbol, interval_minutes, count]
        else:
            query = f"""
                SELECT symbol, timestamp, {interval_minutes} as interval_minutes,
                       open, high, low, close, volume, trade_count, vwap
                FROM {view}
                WHERE symbol = $1
                ORDER BY timestamp DESC
                LIMIT $2
            """
            params = [symbol, count]

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        # Reverse to ascending order
        return [Candle.from_row(row) for row in reversed(rows)]

    async def replay_candles(
        self,
        symbol: str,
        interval_minutes: int,
        start: datetime,
        end: datetime,
        speed: float = 1.0
    ) -> AsyncIterator[Candle]:
        """
        Replay historical candles as a stream for backtesting.

        Args:
            symbol: Trading pair
            interval_minutes: Candle interval
            start: Replay start time
            end: Replay end time
            speed: Replay speed multiplier (1.0 = real-time, 0 = instant)

        Yields:
            Candle objects in chronological order
        """
        candles = await self.get_candles(symbol, interval_minutes, start, end)

        for i, candle in enumerate(candles):
            yield candle

            # Simulate time delay between candles
            if speed > 0 and i < len(candles) - 1:
                next_candle = candles[i + 1]
                time_diff = (next_candle.timestamp - candle.timestamp).total_seconds()
                await asyncio.sleep(time_diff / speed)

    async def get_warmup_data(
        self,
        symbol: str,
        interval_minutes: int,
        warmup_periods: int
    ) -> List[Candle]:
        """
        Get historical data for strategy warmup.

        This provides enough historical candles to initialize indicators
        (e.g., 200 candles for a 200-period moving average).

        Args:
            symbol: Trading pair
            interval_minutes: Candle interval
            warmup_periods: Number of periods needed for warmup

        Returns:
            List of Candle objects
        """
        return await self.get_latest_candles(symbol, interval_minutes, warmup_periods)

    async def get_data_range(self, symbol: str) -> dict:
        """
        Get the available data range for a symbol.

        Returns:
            Dict with 'oldest', 'newest' timestamps and 'total_candles' count

        Raises:
            RuntimeError: If not connected to database.
        """
        self._ensure_connected()  # REC-002
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT
                    MIN(timestamp) as oldest,
                    MAX(timestamp) as newest,
                    COUNT(*) as total_candles
                FROM candles
                WHERE symbol = $1 AND interval_minutes = 1
                """,
                symbol
            )

        return {
            'oldest': row['oldest'],
            'newest': row['newest'],
            'total_candles': row['total_candles']
        }

    async def get_multi_timeframe_candles(
        self,
        symbol: str,
        end_time: datetime,
        intervals: List[int] = None,
        lookback_candles: int = 100
    ) -> dict[int, List[Candle]]:
        """
        Get candles for multiple timeframes at once.

        Useful for MTF analysis where you need aligned data across timeframes.

        Args:
            symbol: Trading pair
            end_time: End time for all timeframes
            intervals: List of intervals to fetch (default: [1, 5, 15, 60, 240])
            lookback_candles: Number of candles per interval

        Returns:
            Dict mapping interval -> list of candles

        Raises:
            RuntimeError: If not connected to database.
        """
        self._ensure_connected()  # REC-002
        if intervals is None:
            intervals = [1, 5, 15, 60, 240]

        result = {}

        # Determine lookback for each interval
        tasks = []
        for interval in intervals:
            lookback = timedelta(minutes=interval * lookback_candles)
            start_time = end_time - lookback
            tasks.append(self.get_candles(symbol, interval, start_time, end_time))

        candle_lists = await asyncio.gather(*tasks)

        for interval, candles in zip(intervals, candle_lists):
            result[interval] = candles

        return result

    async def get_symbols(self) -> List[str]:
        """
        Get list of all symbols with data.

        Returns:
            List of symbol strings

        Raises:
            RuntimeError: If not connected to database.
        """
        self._ensure_connected()  # REC-002
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT DISTINCT symbol
                FROM data_sync_status
                ORDER BY symbol
                """
            )

        return [row['symbol'] for row in rows]

    async def get_sync_status(self, symbol: str) -> Optional[dict]:
        """
        Get sync status for a symbol.

        Returns:
            Dict with sync status or None if no data

        Raises:
            RuntimeError: If not connected to database.
        """
        self._ensure_connected()  # REC-002
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT symbol, data_type, oldest_timestamp, newest_timestamp,
                       last_sync_at, total_records
                FROM data_sync_status
                WHERE symbol = $1 AND data_type = 'candles_1m'
                """,
                symbol
            )

        if not row:
            return None

        return {
            'symbol': row['symbol'],
            'data_type': row['data_type'],
            'oldest': row['oldest_timestamp'],
            'newest': row['newest_timestamp'],
            'last_sync': row['last_sync_at'],
            'total_records': row['total_records'],
        }

    async def health_check(self) -> dict:
        """
        Check provider health and data status.

        Returns:
            Dict with health information
        """
        status = {
            'connected': self.pool is not None,
            'symbols': [],
            'total_candles': 0,
            'oldest_data': None,
            'newest_data': None,
        }

        if not self.pool:
            return status

        try:
            async with self.pool.acquire() as conn:
                # Get symbol count and data range
                row = await conn.fetchrow(
                    """
                    SELECT
                        COUNT(DISTINCT symbol) as symbol_count,
                        MIN(oldest_timestamp) as oldest,
                        MAX(newest_timestamp) as newest,
                        SUM(total_records) as total_records
                    FROM data_sync_status
                    WHERE data_type = 'candles_1m'
                    """
                )

                status['symbols'] = await self.get_symbols()
                status['total_candles'] = row['total_records'] or 0
                status['oldest_data'] = row['oldest']
                status['newest_data'] = row['newest']

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            status['error'] = str(e)

        return status


async def main():
    """Example usage of HistoricalDataProvider."""
    import os

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # REC-004: No default password - require explicit configuration
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        print("ERROR: DATABASE_URL environment variable is required.")
        print("Example: DATABASE_URL=postgresql://trading:YOUR_PASSWORD@localhost:5432/kraken_data")
        return

    provider = HistoricalDataProvider(db_url)

    try:
        await provider.connect()

        # Health check
        health = await provider.health_check()
        print("\nHealth Check:")
        print(f"  Connected: {health['connected']}")
        print(f"  Symbols: {health['symbols']}")
        print(f"  Total candles: {health['total_candles']:,}")
        print(f"  Oldest data: {health['oldest_data']}")
        print(f"  Newest data: {health['newest_data']}")

        # Get warmup data example
        if health['symbols']:
            symbol = health['symbols'][0]
            warmup = await provider.get_warmup_data(symbol, 5, 100)
            print(f"\nWarmup data for {symbol} (5m, 100 candles):")
            print(f"  First: {warmup[0].timestamp if warmup else 'N/A'}")
            print(f"  Last: {warmup[-1].timestamp if warmup else 'N/A'}")
            print(f"  Count: {len(warmup)}")

    finally:
        await provider.close()


if __name__ == '__main__':
    asyncio.run(main())
