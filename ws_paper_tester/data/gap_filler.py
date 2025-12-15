"""
Gap Filler - Detect and fill data gaps on startup.

This module runs on program startup to detect and fill any missing data
between the last stored record and the current time.

Strategy:
1. Query data_sync_status for each symbol
2. Identify gaps between newest_timestamp and now
3. Use OHLC API for small gaps (< 12 hours)
4. Use Trades API for large gaps (>= 12 hours)
5. Update sync status and refresh continuous aggregates

Usage:
    from ws_paper_tester.data import run_gap_filler

    results = await run_gap_filler(db_url)
"""

import asyncio
import logging
import os
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import List, Optional

try:
    import aiohttp
except ImportError:
    aiohttp = None

try:
    import asyncpg
except ImportError:
    asyncpg = None

from .types import DataGap

logger = logging.getLogger(__name__)


class GapFiller:
    """
    Detect and fill gaps in historical data on startup.

    Strategy:
    1. Query data_sync_status for each symbol
    2. Identify gaps between newest_timestamp and now
    3. Use OHLC API for small gaps (< 12 hours)
    4. Use Trades API for large gaps (>= 12 hours)
    5. Update sync status after filling

    The OHLC API is faster but limited to 720 candles per request,
    while the Trades API can fetch complete history but is slower.
    """

    KRAKEN_BASE_URL = 'https://api.kraken.com'

    # Default symbols to check
    DEFAULT_SYMBOLS = ['XRP/USDT', 'BTC/USDT', 'XRP/BTC']

    # Kraken pair mapping
    PAIR_MAP = {
        'XRP/USDT': 'XRPUSDT',
        'BTC/USDT': 'XBTUSDT',
        'XRP/BTC': 'XRPXBT',
        'ETH/USDT': 'ETHUSDT',
        'SOL/USDT': 'SOLUSDT',
    }

    def __init__(
        self,
        db_url: str,
        symbols: Optional[List[str]] = None,
        rate_limit_delay: float = 1.1,
        max_retries: int = 3,
    ):
        """
        Initialize GapFiller.

        Args:
            db_url: PostgreSQL connection URL
            symbols: List of symbols to check (default: XRP/USDT, BTC/USDT, XRP/BTC)
            rate_limit_delay: Seconds between API requests
            max_retries: Maximum retries on API error
        """
        if asyncpg is None:
            raise ImportError("asyncpg is required. Install with: pip install asyncpg")
        if aiohttp is None:
            raise ImportError("aiohttp is required. Install with: pip install aiohttp")

        self.db_url = db_url
        self.symbols = symbols or self.DEFAULT_SYMBOLS
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self.pool: Optional[asyncpg.Pool] = None
        self.session: Optional[aiohttp.ClientSession] = None

    async def start(self):
        """Initialize connections."""
        self.pool = await asyncpg.create_pool(self.db_url, min_size=2, max_size=10)
        self.session = aiohttp.ClientSession()
        logger.info("GapFiller initialized")

    async def stop(self):
        """Close connections."""
        if self.session:
            await self.session.close()
        if self.pool:
            await self.pool.close()

    async def detect_gaps(self, min_gap_minutes: int = 2) -> List[DataGap]:
        """
        Detect gaps for all symbols.

        Args:
            min_gap_minutes: Minimum gap size to report (default: 2 minutes)

        Returns:
            List of DataGap objects describing missing data
        """
        gaps = []
        now = datetime.now(timezone.utc)

        async with self.pool.acquire() as conn:
            for symbol in self.symbols:
                # Check 1-minute candles
                row = await conn.fetchrow(
                    """
                    SELECT newest_timestamp
                    FROM data_sync_status
                    WHERE symbol = $1 AND data_type = 'candles_1m'
                    """,
                    symbol
                )

                if row and row['newest_timestamp']:
                    last_timestamp = row['newest_timestamp']
                    gap_duration = now - last_timestamp

                    # Only report gaps > min_gap_minutes
                    if gap_duration > timedelta(minutes=min_gap_minutes):
                        gaps.append(DataGap(
                            symbol=symbol,
                            data_type='candles_1m',
                            start_time=last_timestamp,
                            end_time=now,
                            duration=gap_duration
                        ))
                        logger.info(
                            f"Gap detected: {symbol} 1m candles "
                            f"from {last_timestamp} to {now} ({gap_duration})"
                        )
                else:
                    # No data at all - need full backfill
                    # Start from 30 days ago as minimum
                    start = now - timedelta(days=30)
                    gaps.append(DataGap(
                        symbol=symbol,
                        data_type='candles_1m',
                        start_time=start,
                        end_time=now,
                        duration=timedelta(days=30)
                    ))
                    logger.warning(f"No data found for {symbol} - need full backfill")

        return gaps

    async def fill_gap_ohlc(self, gap: DataGap) -> int:
        """
        Fill a small gap using OHLC REST API.

        The OHLC endpoint returns up to 720 candles, which covers
        12 hours of 1-minute data.

        Args:
            gap: DataGap to fill

        Returns:
            Number of candles inserted
        """
        pair = self.PAIR_MAP.get(gap.symbol)
        if not pair:
            logger.error(f"Unknown symbol: {gap.symbol}")
            return 0

        since = int(gap.start_time.timestamp())

        for attempt in range(self.max_retries):
            try:
                url = f'{self.KRAKEN_BASE_URL}/0/public/OHLC'
                params = {
                    'pair': pair,
                    'interval': 1,  # 1-minute candles
                    'since': since
                }

                async with self.session.get(url, params=params, timeout=30) as response:
                    data = await response.json()

                    if data.get('error'):
                        errors = data['error']
                        if any('EAPI:Rate limit' in e for e in errors):
                            logger.warning("Rate limited, backing off...")
                            await asyncio.sleep(5 * (attempt + 1))
                            continue
                        logger.error(f"Kraken OHLC API error: {errors}")
                        return 0

                    result = data['result']
                    pair_key = [k for k in result.keys() if k != 'last'][0] if len(result) > 1 else None
                    candles = result.get(pair_key, []) if pair_key else []

                if not candles:
                    logger.info(f"No OHLC data returned for {gap.symbol}")
                    return 0

                # Convert and insert candles
                records = []
                for c in candles:
                    timestamp, open_, high, low, close, vwap, volume, count = c[:8]

                    # Filter to gap period
                    candle_time = datetime.fromtimestamp(int(timestamp), tz=timezone.utc)
                    if candle_time < gap.start_time or candle_time >= gap.end_time:
                        continue

                    records.append((
                        gap.symbol,
                        candle_time,
                        1,  # interval_minutes
                        Decimal(str(open_)),
                        Decimal(str(high)),
                        Decimal(str(low)),
                        Decimal(str(close)),
                        Decimal(str(volume)),
                        None,  # quote_volume
                        int(count),
                        Decimal(str(vwap))
                    ))

                if records:
                    async with self.pool.acquire() as conn:
                        await conn.executemany(
                            """
                            INSERT INTO candles
                                (symbol, timestamp, interval_minutes, open, high, low, close,
                                 volume, quote_volume, trade_count, vwap)
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                            ON CONFLICT (timestamp, symbol, interval_minutes) DO UPDATE SET
                                high = GREATEST(candles.high, EXCLUDED.high),
                                low = LEAST(candles.low, EXCLUDED.low),
                                close = EXCLUDED.close,
                                volume = EXCLUDED.volume,
                                trade_count = EXCLUDED.trade_count,
                                vwap = EXCLUDED.vwap
                            """,
                            records
                        )

                logger.info(f"Filled {len(records)} candles for {gap.symbol} via OHLC API")
                return len(records)

            except asyncio.TimeoutError:
                logger.warning(f"Timeout fetching OHLC, attempt {attempt + 1}/{self.max_retries}")
                await asyncio.sleep(2 * (attempt + 1))
            except Exception as e:
                logger.error(f"Error fetching OHLC for {gap.symbol}: {e}")
                await asyncio.sleep(2 * (attempt + 1))

        return 0

    async def fill_gap_trades(self, gap: DataGap) -> int:
        """
        Fill a large gap using Trades REST API.

        This fetches raw trades and builds candles from them.
        Slower but can handle gaps of any size.

        Args:
            gap: DataGap to fill

        Returns:
            Number of trades processed
        """
        pair = self.PAIR_MAP.get(gap.symbol)
        if not pair:
            logger.error(f"Unknown symbol: {gap.symbol}")
            return 0

        since = int(gap.start_time.timestamp() * 1_000_000_000)  # Nanoseconds
        total_trades = 0

        while True:
            try:
                url = f'{self.KRAKEN_BASE_URL}/0/public/Trades'
                params = {'pair': pair, 'since': since}

                async with self.session.get(url, params=params, timeout=30) as response:
                    data = await response.json()

                    if data.get('error'):
                        errors = data['error']
                        if any('EAPI:Rate limit' in e for e in errors):
                            logger.warning("Rate limited, backing off...")
                            await asyncio.sleep(5)
                            continue
                        logger.error(f"Kraken Trades API error: {errors}")
                        break

                    result = data['result']
                    pair_key = [k for k in result.keys() if k != 'last'][0] if len(result) > 1 else None
                    trades = result.get(pair_key, []) if pair_key else []
                    last = result.get('last', 0)

                if not trades:
                    break

                # Check if we've passed the gap end time
                last_trade_time = datetime.fromtimestamp(float(trades[-1][2]), tz=timezone.utc)
                if last_trade_time >= gap.end_time:
                    # Filter trades to gap period
                    trades = [
                        t for t in trades
                        if datetime.fromtimestamp(float(t[2]), tz=timezone.utc) < gap.end_time
                    ]

                    if trades:
                        await self._store_trades_and_build_candles(gap.symbol, trades)
                        total_trades += len(trades)
                    break

                await self._store_trades_and_build_candles(gap.symbol, trades)
                total_trades += len(trades)

                since = int(last)
                await asyncio.sleep(self.rate_limit_delay)

            except asyncio.TimeoutError:
                logger.warning("Timeout fetching trades, retrying...")
                await asyncio.sleep(2)
            except Exception as e:
                logger.error(f"Error fetching trades for {gap.symbol}: {e}")
                break

        logger.info(f"Filled gap for {gap.symbol} with {total_trades} trades")
        return total_trades

    async def _store_trades_and_build_candles(self, symbol: str, trades: list):
        """Store trades and build 1-minute candles."""
        if not trades:
            return

        # Store raw trades
        trade_records = [
            (
                symbol,
                datetime.fromtimestamp(float(t[2]), tz=timezone.utc),
                Decimal(str(t[0])),
                Decimal(str(t[1])),
                'buy' if t[3] == 'b' else 'sell',
                t[4],
                t[5]
            )
            for t in trades
        ]

        async with self.pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO trades (symbol, timestamp, price, volume, side, order_type, misc)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT DO NOTHING
                """,
                trade_records
            )

            # Build candles from these trades
            min_time = min(t[1] for t in trade_records)
            max_time = max(t[1] for t in trade_records)

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
                  AND timestamp <= $3
                GROUP BY symbol, time_bucket('1 minute', timestamp)
                ON CONFLICT (timestamp, symbol, interval_minutes) DO UPDATE SET
                    high = GREATEST(candles.high, EXCLUDED.high),
                    low = LEAST(candles.low, EXCLUDED.low),
                    close = EXCLUDED.close,
                    volume = candles.volume + EXCLUDED.volume,
                    trade_count = candles.trade_count + EXCLUDED.trade_count
                """,
                symbol, min_time, max_time
            )

    async def update_sync_status(self, symbol: str, newest_time: datetime):
        """Update sync status after filling."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO data_sync_status (symbol, data_type, newest_timestamp, last_sync_at)
                VALUES ($1, 'candles_1m', $2, NOW())
                ON CONFLICT (symbol, data_type) DO UPDATE SET
                    newest_timestamp = GREATEST(data_sync_status.newest_timestamp, EXCLUDED.newest_timestamp),
                    last_sync_at = NOW()
                """,
                symbol, newest_time
            )

    async def refresh_continuous_aggregates(self):
        """Refresh all continuous aggregates after gap fill."""
        aggregates = [
            'candles_5m', 'candles_15m', 'candles_30m',
            'candles_1h', 'candles_4h', 'candles_12h',
            'candles_1d', 'candles_1w'
        ]

        async with self.pool.acquire() as conn:
            for agg in aggregates:
                try:
                    await conn.execute(f"CALL refresh_continuous_aggregate('{agg}', NULL, NULL)")
                    logger.info(f"Refreshed continuous aggregate: {agg}")
                except Exception as e:
                    # May fail if aggregate doesn't exist yet
                    logger.debug(f"Could not refresh {agg}: {e}")

    async def fill_all_gaps(self, max_concurrent: int = 3) -> dict:
        """
        Main entry point: detect and fill all gaps.

        Args:
            max_concurrent: Maximum concurrent gap fills

        Returns:
            Summary of gap filling results
        """
        results = {
            'gaps_detected': 0,
            'gaps_filled': 0,
            'candles_inserted': 0,
            'trades_processed': 0,
            'errors': []
        }

        gaps = await self.detect_gaps()
        results['gaps_detected'] = len(gaps)

        if not gaps:
            logger.info("No gaps detected - data is up to date")
            return results

        logger.info(f"Detected {len(gaps)} gaps to fill")

        # Fill gaps with limited concurrency
        semaphore = asyncio.Semaphore(max_concurrent)

        async def fill_with_limit(gap: DataGap):
            async with semaphore:
                try:
                    if gap.is_small:
                        count = await self.fill_gap_ohlc(gap)
                        results['candles_inserted'] += count
                    else:
                        count = await self.fill_gap_trades(gap)
                        results['trades_processed'] += count

                    await self.update_sync_status(gap.symbol, gap.end_time)
                    results['gaps_filled'] += 1

                except Exception as e:
                    error_msg = f"Failed to fill gap for {gap.symbol}: {e}"
                    logger.error(error_msg)
                    results['errors'].append(error_msg)

        await asyncio.gather(*[fill_with_limit(gap) for gap in gaps])

        # Refresh continuous aggregates
        await self.refresh_continuous_aggregates()

        logger.info(
            f"Gap fill complete: {results['gaps_filled']}/{results['gaps_detected']} gaps filled, "
            f"{results['candles_inserted']} candles, {results['trades_processed']} trades"
        )

        return results


async def run_gap_filler(
    db_url: str,
    symbols: Optional[List[str]] = None,
) -> dict:
    """
    Convenience function to run gap filler.

    Args:
        db_url: PostgreSQL connection URL
        symbols: Optional list of symbols to check

    Returns:
        Dictionary with gap filling results

    Usage:
        results = await run_gap_filler(db_url)
    """
    filler = GapFiller(db_url, symbols=symbols)

    try:
        await filler.start()
        return await filler.fill_all_gaps()
    finally:
        await filler.stop()


async def main():
    """Run gap filler from command line."""
    import argparse

    parser = argparse.ArgumentParser(description='Detect and fill data gaps')
    parser.add_argument('--symbols', nargs='+', default=None,
                        help='Trading pairs to check')
    parser.add_argument('--db-url', type=str,
                        default=os.getenv('DATABASE_URL', 'postgresql://trading:password@localhost:5432/kraken_data'),
                        help='PostgreSQL connection URL')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    results = await run_gap_filler(args.db_url, symbols=args.symbols)

    print("\n" + "=" * 60)
    print("Gap Filler Results")
    print("=" * 60)
    print(f"  Gaps detected: {results['gaps_detected']}")
    print(f"  Gaps filled: {results['gaps_filled']}")
    print(f"  Candles inserted: {results['candles_inserted']}")
    print(f"  Trades processed: {results['trades_processed']}")
    if results['errors']:
        print(f"  Errors: {len(results['errors'])}")
        for err in results['errors']:
            print(f"    - {err}")
    print("=" * 60)


if __name__ == '__main__':
    asyncio.run(main())
