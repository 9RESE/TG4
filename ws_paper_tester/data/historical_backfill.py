"""
Historical Trades Backfill - Fetch complete trade history from Kraken REST API.

This module fetches the entire trade history for symbols using Kraken's
public Trades API endpoint, which provides data from market inception.

Usage:
    python -m ws_paper_tester.data.historical_backfill --symbols XRP/USDT BTC/USDT
"""

import asyncio
import logging
import os
from datetime import datetime, timezone
from decimal import Decimal
from typing import AsyncIterator, List, Optional, Tuple

try:
    import aiohttp
except ImportError:
    aiohttp = None

try:
    import asyncpg
except ImportError:
    asyncpg = None

from .types import PAIR_MAP  # REC-005: Use centralized pair mapping

logger = logging.getLogger(__name__)


class KrakenTradesBackfill:
    """
    Fetch and store complete trade history from Kraken REST API.

    The Kraken Trades API returns up to 1000 trades per request with
    pagination via the 'since' parameter (nanosecond timestamp).

    Rate limiting: 1 request per second for public endpoints.
    """

    BASE_URL = 'https://api.kraken.com'

    # Rate limiting: 1 request per second for public endpoints
    RATE_LIMIT_DELAY = 1.1

    # REC-005: Use centralized PAIR_MAP from types.py
    PAIR_MAP = PAIR_MAP

    def __init__(
        self,
        db_url: str,
        rate_limit_delay: float = 1.1,
        max_retries: int = 3,
    ):
        """
        Initialize KrakenTradesBackfill.

        Args:
            db_url: PostgreSQL connection URL
            rate_limit_delay: Seconds between API requests
            max_retries: Maximum retries on API error
        """
        if asyncpg is None:
            raise ImportError("asyncpg is required. Install with: pip install asyncpg")
        if aiohttp is None:
            raise ImportError("aiohttp is required. Install with: pip install aiohttp")

        self.db_url = db_url
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self.pool: Optional[asyncpg.Pool] = None
        self.session: Optional[aiohttp.ClientSession] = None

    async def connect(self):
        """Initialize connections."""
        self.pool = await asyncpg.create_pool(
            self.db_url,
            min_size=2,
            max_size=10
        )
        self.session = aiohttp.ClientSession()
        logger.info("Connected to database and initialized HTTP session")

    async def close(self):
        """Close connections."""
        if self.session:
            await self.session.close()
        if self.pool:
            await self.pool.close()

    async def fetch_trades_page(
        self,
        pair: str,
        since: int = 0
    ) -> Tuple[List[list], int]:
        """
        Fetch a page of trades from Kraken API.

        Args:
            pair: Kraken pair name (e.g., 'XRPUSDT')
            since: Starting timestamp (nanoseconds)

        Returns:
            Tuple of (trades list, last timestamp for pagination)
        """
        url = f'{self.BASE_URL}/0/public/Trades'
        params = {'pair': pair, 'since': since}

        for attempt in range(self.max_retries):
            try:
                async with self.session.get(url, params=params, timeout=30) as response:
                    data = await response.json()

                    if data.get('error'):
                        errors = data['error']
                        # Handle rate limiting
                        if any('EAPI:Rate limit' in e for e in errors):
                            logger.warning("Rate limited, backing off...")
                            await asyncio.sleep(5 * (attempt + 1))
                            continue
                        raise Exception(f"Kraken API error: {errors}")

                    result = data['result']

                    # Get trades (key is the pair name, varies by pair)
                    trades_key = [k for k in result.keys() if k != 'last'][0] if len(result) > 1 else None
                    trades = result.get(trades_key, []) if trades_key else []
                    last = int(result.get('last', 0))

                    return trades, last

            except asyncio.TimeoutError:
                logger.warning(f"Timeout fetching trades, attempt {attempt + 1}/{self.max_retries}")
                await asyncio.sleep(2 * (attempt + 1))
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                logger.warning(f"Error fetching trades: {e}, attempt {attempt + 1}/{self.max_retries}")
                await asyncio.sleep(2 * (attempt + 1))

        return [], 0

    async def fetch_all_trades(
        self,
        symbol: str,
        start_since: int = 0,
        end_timestamp: Optional[datetime] = None,
    ) -> AsyncIterator[List[list]]:
        """
        Generator that fetches all trades for a symbol.

        Args:
            symbol: Our symbol format (e.g., 'XRP/USDT')
            start_since: Starting timestamp (0 for beginning)
            end_timestamp: Optional end time to stop fetching

        Yields:
            Batches of trade records
        """
        pair = self.PAIR_MAP.get(symbol)
        if not pair:
            raise ValueError(f"Unknown symbol: {symbol}")

        since = start_since
        total_fetched = 0

        while True:
            try:
                trades, last = await self.fetch_trades_page(pair, since)

                if not trades:
                    logger.info(f"{symbol}: No more trades after {since}")
                    break

                total_fetched += len(trades)

                # Check if we've passed the end timestamp
                if end_timestamp and trades:
                    last_trade_time = datetime.fromtimestamp(float(trades[-1][2]), tz=timezone.utc)
                    if last_trade_time >= end_timestamp:
                        # Filter trades to end time
                        trades = [
                            t for t in trades
                            if datetime.fromtimestamp(float(t[2]), tz=timezone.utc) < end_timestamp
                        ]
                        if trades:
                            yield trades
                        break

                logger.info(
                    f"{symbol}: Fetched {len(trades)} trades "
                    f"(total: {total_fetched:,}, since: {since})"
                )

                yield trades

                # Update since for next page
                since = last

                # Rate limiting
                await asyncio.sleep(self.rate_limit_delay)

            except Exception as e:
                logger.error(f"Error fetching trades: {e}")
                await asyncio.sleep(5)  # Back off on error

    async def store_trades(self, symbol: str, trades: List[list]):
        """
        Store trades in database with validation.

        Kraken trade format: [price, volume, time, side, type, misc]

        REC-003: Validates trade data before storage:
        - Rejects invalid prices (<=0, NaN)
        - Rejects invalid volumes (<=0, NaN)
        - Handles malformed timestamps gracefully
        - Continues processing after invalid records
        """
        if not trades:
            return

        records = []
        skipped = 0

        for trade in trades:
            try:
                price, volume, timestamp, side, order_type, misc = trade[:6]

                # Validate and convert price
                try:
                    price_decimal = Decimal(str(price))
                    if price_decimal <= 0:
                        logger.debug(f"Skipping trade with invalid price: {price}")
                        skipped += 1
                        continue
                except Exception:
                    logger.debug(f"Skipping trade with malformed price: {price}")
                    skipped += 1
                    continue

                # Validate and convert volume
                try:
                    volume_decimal = Decimal(str(volume))
                    if volume_decimal <= 0:
                        logger.debug(f"Skipping trade with invalid volume: {volume}")
                        skipped += 1
                        continue
                except Exception:
                    logger.debug(f"Skipping trade with malformed volume: {volume}")
                    skipped += 1
                    continue

                # Validate and convert timestamp
                try:
                    trade_time = datetime.fromtimestamp(float(timestamp), tz=timezone.utc)
                except (ValueError, OSError, OverflowError):
                    logger.debug(f"Skipping trade with invalid timestamp: {timestamp}")
                    skipped += 1
                    continue

                records.append((
                    symbol,
                    trade_time,
                    price_decimal,
                    volume_decimal,
                    'buy' if side == 'b' else 'sell',
                    order_type,
                    misc
                ))

            except (ValueError, TypeError, IndexError) as e:
                logger.warning(f"Skipping malformed trade record: {e}")
                skipped += 1
                continue

        if skipped > 0:
            logger.info(f"Skipped {skipped} invalid trades for {symbol}")

        if not records:
            return

        async with self.pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO trades
                    (symbol, timestamp, price, volume, side, order_type, misc)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT DO NOTHING
                """,
                records
            )

    async def build_candles_from_trades(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime
    ):
        """Build 1-minute candles from stored trades."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO candles (symbol, timestamp, interval_minutes, open, high, low, close, volume, trade_count, vwap)
                SELECT
                    symbol,
                    time_bucket('1 minute', timestamp) AS timestamp,
                    1 AS interval_minutes,
                    first(price, timestamp) AS open,
                    max(price) AS high,
                    min(price) AS low,
                    last(price, timestamp) AS close,
                    sum(volume) AS volume,
                    count(*) AS trade_count,
                    sum(price * volume) / nullif(sum(volume), 0) AS vwap
                FROM trades
                WHERE symbol = $1
                  AND timestamp >= $2
                  AND timestamp < $3
                GROUP BY symbol, time_bucket('1 minute', timestamp)
                ON CONFLICT (timestamp, symbol, interval_minutes) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume,
                    trade_count = EXCLUDED.trade_count,
                    vwap = EXCLUDED.vwap
                """,
                symbol, start_time, end_time
            )

    async def backfill_symbol(
        self,
        symbol: str,
        since: int = 0,
        end_timestamp: Optional[datetime] = None,
        build_candles: bool = True,
    ) -> int:
        """
        Backfill complete trade history for a symbol.

        Args:
            symbol: Trading pair (e.g., 'XRP/USDT')
            since: Starting timestamp (0 for complete history)
            end_timestamp: Optional end time to stop
            build_candles: Whether to build candles from trades

        Returns:
            Total trades imported
        """
        logger.info(f"Starting backfill for {symbol} from {since}")

        total_trades = 0
        batch_count = 0
        first_timestamp = None
        last_timestamp = None

        async for trades in self.fetch_all_trades(symbol, since, end_timestamp):
            await self.store_trades(symbol, trades)
            total_trades += len(trades)
            batch_count += 1

            # Track progress
            if trades:
                if first_timestamp is None:
                    first_timestamp = datetime.fromtimestamp(
                        float(trades[0][2]),
                        tz=timezone.utc
                    )
                last_timestamp = datetime.fromtimestamp(
                    float(trades[-1][2]),
                    tz=timezone.utc
                )

            # Build candles periodically (every 100 batches)
            if build_candles and batch_count % 100 == 0 and first_timestamp and last_timestamp:
                logger.info(f"{symbol}: Building candles up to {last_timestamp}")
                await self.build_candles_from_trades(symbol, first_timestamp, last_timestamp)

        # Final candle build
        if build_candles and first_timestamp and last_timestamp:
            logger.info(f"{symbol}: Final candle build")
            await self.build_candles_from_trades(symbol, first_timestamp, last_timestamp)

        # Update sync status
        if last_timestamp:
            await self._update_sync_status(symbol, since, first_timestamp, last_timestamp, total_trades)

        logger.info(f"{symbol}: Backfill complete. Total trades: {total_trades:,}")
        return total_trades

    async def _update_sync_status(
        self,
        symbol: str,
        since: int,
        first_timestamp: datetime,
        last_timestamp: datetime,
        total_trades: int
    ):
        """Update sync status after backfill."""
        async with self.pool.acquire() as conn:
            # Update trades sync status
            await conn.execute(
                """
                INSERT INTO data_sync_status
                    (symbol, data_type, oldest_timestamp, newest_timestamp, last_kraken_since, total_records)
                VALUES ($1, 'trades', $2, $3, $4, $5)
                ON CONFLICT (symbol, data_type) DO UPDATE SET
                    oldest_timestamp = LEAST(data_sync_status.oldest_timestamp, EXCLUDED.oldest_timestamp),
                    newest_timestamp = GREATEST(data_sync_status.newest_timestamp, EXCLUDED.newest_timestamp),
                    last_kraken_since = EXCLUDED.last_kraken_since,
                    total_records = data_sync_status.total_records + EXCLUDED.total_records,
                    last_sync_at = NOW()
                """,
                symbol, first_timestamp, last_timestamp, since, total_trades
            )

            # Update candles sync status
            await conn.execute(
                """
                INSERT INTO data_sync_status
                    (symbol, data_type, oldest_timestamp, newest_timestamp, total_records)
                VALUES ($1, 'candles_1m', $2, $3, 0)
                ON CONFLICT (symbol, data_type) DO UPDATE SET
                    oldest_timestamp = LEAST(data_sync_status.oldest_timestamp, EXCLUDED.oldest_timestamp),
                    newest_timestamp = GREATEST(data_sync_status.newest_timestamp, EXCLUDED.newest_timestamp),
                    last_sync_at = NOW()
                """,
                symbol, first_timestamp, last_timestamp
            )

    async def get_resume_point(self, symbol: str) -> int:
        """Get the point to resume backfill from."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT last_kraken_since
                FROM data_sync_status
                WHERE symbol = $1 AND data_type = 'trades'
                """,
                symbol
            )
            return row['last_kraken_since'] if row and row['last_kraken_since'] else 0


async def main():
    """Run historical backfill from command line."""
    import argparse

    parser = argparse.ArgumentParser(description='Fetch complete trade history from Kraken')
    parser.add_argument('--symbols', nargs='+', default=['XRP/USDT', 'BTC/USDT', 'XRP/BTC'],
                        help='Trading pairs to backfill')
    # REC-004: No default password - require explicit configuration
    parser.add_argument('--db-url', type=str,
                        default=os.getenv('DATABASE_URL'),
                        help='PostgreSQL connection URL (required, or set DATABASE_URL env var)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from last sync point')
    parser.add_argument('--no-candles', action='store_true',
                        help='Skip candle building (just import trades)')

    args = parser.parse_args()

    # REC-004: Require database URL - no default credentials
    if not args.db_url:
        parser.error(
            "--db-url or DATABASE_URL environment variable is required.\n"
            "Example: DATABASE_URL=postgresql://trading:YOUR_PASSWORD@localhost:5432/kraken_data"
        )

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    backfill = KrakenTradesBackfill(args.db_url)

    try:
        await backfill.connect()

        for symbol in args.symbols:
            # Resume from last position if requested
            since = 0
            if args.resume:
                since = await backfill.get_resume_point(symbol)
                if since:
                    logger.info(f"Resuming {symbol} from {since}")

            await backfill.backfill_symbol(
                symbol,
                since=since,
                build_candles=not args.no_candles
            )

    finally:
        await backfill.close()


if __name__ == '__main__':
    asyncio.run(main())
