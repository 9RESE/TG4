"""
Bulk CSV Importer - Import Kraken historical CSV files into TimescaleDB.

This module handles the initial bulk import of historical data from
Kraken's downloadable CSV files (available at support.kraken.com).

Usage:
    python -m ws_paper_tester.data.bulk_csv_importer --dir ./data/kraken_csv
"""

import asyncio
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

try:
    import asyncpg
except ImportError:
    asyncpg = None

try:
    import pandas as pd
except ImportError:
    pd = None

logger = logging.getLogger(__name__)


class BulkCSVImporter:
    """
    Import Kraken historical CSV files into TimescaleDB.

    Kraken provides downloadable OHLCVT CSV files at:
    https://support.kraken.com/hc/en-us/articles/360047124832

    CSV format: timestamp, open, high, low, close, volume, trades
    - timestamp: Unix timestamp (seconds)
    - OHLC: Decimal prices
    - volume: Base asset volume
    - trades: Number of trades in the candle
    """

    # Kraken CSV column mapping
    CSV_COLUMNS = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'trades']

    # Symbol mapping (Kraken CSV naming to our format)
    SYMBOL_MAP = {
        'XRPUSDT': 'XRP/USDT',
        'XBTUSDT': 'BTC/USDT',
        'BTCUSDT': 'BTC/USDT',
        'XRPXBT': 'XRP/BTC',
        'XRPBTC': 'XRP/BTC',
        'ETHUSDT': 'ETH/USDT',
        'SOLUSDT': 'SOL/USDT',
        # Add more mappings as needed
    }

    # Interval mapping (filename suffix to minutes)
    INTERVAL_MAP = {
        '1': 1,
        '5': 5,
        '15': 15,
        '30': 30,
        '60': 60,
        '240': 240,
        '720': 720,
        '1440': 1440,
        '10080': 10080,
    }

    def __init__(self, db_url: str, batch_size: int = 10000):
        """
        Initialize BulkCSVImporter.

        Args:
            db_url: PostgreSQL connection URL
            batch_size: Number of records per batch insert
        """
        if asyncpg is None:
            raise ImportError("asyncpg is required. Install with: pip install asyncpg")
        if pd is None:
            raise ImportError("pandas is required. Install with: pip install pandas")

        self.db_url = db_url
        self.batch_size = batch_size
        self.pool: Optional[asyncpg.Pool] = None

    async def connect(self):
        """Establish database connection pool."""
        self.pool = await asyncpg.create_pool(
            self.db_url,
            min_size=2,
            max_size=10,
            command_timeout=300
        )
        logger.info("Connected to TimescaleDB")

    async def close(self):
        """Close database connection pool."""
        if self.pool:
            await self.pool.close()

    async def import_csv_file(
        self,
        filepath: Path,
        symbol: str,
        interval_minutes: int
    ) -> int:
        """
        Import a single CSV file into the candles table.

        Args:
            filepath: Path to CSV file
            symbol: Trading pair symbol (e.g., 'XRP/USDT')
            interval_minutes: Candle interval in minutes

        Returns:
            Number of rows imported
        """
        logger.info(f"Importing {filepath} for {symbol} ({interval_minutes}m)")

        # Read CSV file
        df = pd.read_csv(
            filepath,
            names=self.CSV_COLUMNS,
            header=None
        )

        if df.empty:
            logger.warning(f"Empty CSV file: {filepath}")
            return 0

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)

        # Calculate VWAP (approximation from OHLC)
        df['vwap'] = (df['high'] + df['low'] + df['close']) / 3

        # Add symbol and interval
        df['symbol'] = symbol
        df['interval_minutes'] = interval_minutes

        # Prepare records for bulk insert
        records = [
            (
                row['symbol'],
                row['timestamp'].to_pydatetime(),
                row['interval_minutes'],
                float(row['open']),
                float(row['high']),
                float(row['low']),
                float(row['close']),
                float(row['volume']),
                None,  # quote_volume (not in CSV)
                int(row['trades']) if pd.notna(row['trades']) else None,
                float(row['vwap'])
            )
            for _, row in df.iterrows()
        ]

        # Bulk insert with conflict handling in batches
        total_inserted = 0
        async with self.pool.acquire() as conn:
            for i in range(0, len(records), self.batch_size):
                batch = records[i:i + self.batch_size]
                await conn.executemany(
                    """
                    INSERT INTO candles
                        (symbol, timestamp, interval_minutes, open, high, low, close,
                         volume, quote_volume, trade_count, vwap)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    ON CONFLICT (timestamp, symbol, interval_minutes)
                    DO UPDATE SET
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume,
                        trade_count = EXCLUDED.trade_count,
                        vwap = EXCLUDED.vwap
                    """,
                    batch
                )
                total_inserted += len(batch)
                logger.debug(f"Inserted batch {i//self.batch_size + 1}, total: {total_inserted}")

            # Update sync status
            if records:
                oldest = min(r[1] for r in records)
                newest = max(r[1] for r in records)
                await conn.execute(
                    """
                    INSERT INTO data_sync_status
                        (symbol, data_type, oldest_timestamp, newest_timestamp, total_records)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (symbol, data_type) DO UPDATE SET
                        oldest_timestamp = LEAST(data_sync_status.oldest_timestamp, EXCLUDED.oldest_timestamp),
                        newest_timestamp = GREATEST(data_sync_status.newest_timestamp, EXCLUDED.newest_timestamp),
                        total_records = data_sync_status.total_records + EXCLUDED.total_records,
                        last_sync_at = NOW()
                    """,
                    symbol, f'candles_{interval_minutes}m', oldest, newest, len(records)
                )

        logger.info(f"Imported {total_inserted} candles from {filepath}")
        return total_inserted

    async def import_directory(self, directory: Path, only_1m: bool = True) -> dict:
        """
        Import all CSV files from a directory.

        Expected structure:
            directory/
                XRPUSDT_1.csv
                XRPUSDT_5.csv
                XRPUSDT_60.csv
                BTCUSDT_1.csv
                ...

        Args:
            directory: Path to directory containing CSV files
            only_1m: If True, only import 1-minute data (others via continuous aggregates)

        Returns:
            Dictionary of {symbol: {interval: count}}
        """
        results = {}
        csv_files = list(directory.glob('*.csv'))

        logger.info(f"Found {len(csv_files)} CSV files in {directory}")

        for filepath in csv_files:
            # Parse filename: XRPUSDT_1.csv -> symbol=XRP/USDT, interval=1
            parts = filepath.stem.split('_')
            if len(parts) != 2:
                logger.warning(f"Skipping unrecognized file: {filepath}")
                continue

            pair_code, interval_str = parts

            # Map to our symbol format
            symbol = self.SYMBOL_MAP.get(pair_code.upper())
            if not symbol:
                logger.warning(f"Unknown pair code: {pair_code}")
                continue

            # Map interval
            interval = self.INTERVAL_MAP.get(interval_str)
            if not interval:
                logger.warning(f"Unknown interval: {interval_str}")
                continue

            # Only import 1-minute data by default (others computed via continuous aggregates)
            if only_1m and interval != 1:
                logger.info(f"Skipping {filepath} (only 1m data needed, use --all for all intervals)")
                continue

            try:
                count = await self.import_csv_file(filepath, symbol, interval)

                if symbol not in results:
                    results[symbol] = {}
                results[symbol][interval] = count

            except Exception as e:
                logger.error(f"Failed to import {filepath}: {e}")

        return results


async def main():
    """Run bulk CSV import from command line."""
    import argparse

    parser = argparse.ArgumentParser(description='Import Kraken historical CSV files')
    parser.add_argument('--dir', type=str, default='./data/kraken_csv',
                        help='Directory containing CSV files')
    parser.add_argument('--db-url', type=str,
                        default=os.getenv('DATABASE_URL', 'postgresql://trading:password@localhost:5432/kraken_data'),
                        help='PostgreSQL connection URL')
    parser.add_argument('--all', action='store_true',
                        help='Import all intervals (not just 1m)')
    parser.add_argument('--batch-size', type=int, default=10000,
                        help='Batch size for inserts')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    csv_dir = Path(args.dir)
    if not csv_dir.exists():
        logger.error(f"Directory not found: {csv_dir}")
        return

    importer = BulkCSVImporter(args.db_url, batch_size=args.batch_size)

    try:
        await importer.connect()
        results = await importer.import_directory(csv_dir, only_1m=not args.all)

        print("\n" + "=" * 60)
        print("Import Summary")
        print("=" * 60)
        for symbol, intervals in results.items():
            for interval, count in intervals.items():
                print(f"  {symbol} {interval}m: {count:,} candles")
        print("=" * 60)

        total = sum(sum(intervals.values()) for intervals in results.values())
        print(f"Total: {total:,} candles imported")

    finally:
        await importer.close()


if __name__ == '__main__':
    asyncio.run(main())
