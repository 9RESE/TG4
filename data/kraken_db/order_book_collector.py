"""
Order Book Depth Collector - Collect and store order book snapshots from Kraken.

This module provides:
- OrderBookCollector: Periodic REST API order book snapshots
- WebSocketOrderBookCollector: Real-time order book via WebSocket (optional)

Order book data is critical for:
- Spread analysis and optimization
- Liquidity depth assessment
- Market microstructure analysis
- Slippage estimation

Usage:
    collector = OrderBookCollector(db_url)
    await collector.start()
    await collector.run_collection_loop()  # Runs indefinitely
    await collector.stop()
"""

import asyncio
import logging
import os
from datetime import datetime, timezone
from decimal import Decimal
from typing import List, Optional
import json

try:
    import aiohttp
except ImportError:
    aiohttp = None

try:
    import asyncpg
except ImportError:
    asyncpg = None

from .types import PAIR_MAP, DEFAULT_SYMBOLS

logger = logging.getLogger(__name__)


class OrderBookCollector:
    """
    Collect order book depth snapshots from Kraken REST API.

    Stores periodic snapshots of the order book for analysis.
    Default: Every 60 seconds for each symbol.

    Features:
    - Configurable depth (10, 25, 100, 500 levels)
    - Spread calculation
    - Mid-price tracking
    - Order book imbalance metrics
    """

    KRAKEN_BASE_URL = 'https://api.kraken.com'
    PAIR_MAP = PAIR_MAP
    DEFAULT_SYMBOLS = DEFAULT_SYMBOLS

    def __init__(
        self,
        db_url: str,
        symbols: Optional[List[str]] = None,
        snapshot_interval: float = 60.0,  # seconds
        depth: int = 25,  # Number of levels to capture
        rate_limit_delay: float = 1.1,
    ):
        """
        Initialize OrderBookCollector.

        Args:
            db_url: PostgreSQL connection URL
            symbols: List of symbols to collect (default: XRP/USDT, BTC/USDT, XRP/BTC)
            snapshot_interval: Seconds between snapshots per symbol
            depth: Number of order book levels (10, 25, 100, 500)
            rate_limit_delay: Seconds between API calls
        """
        if asyncpg is None:
            raise ImportError("asyncpg is required. Install with: pip install asyncpg")
        if aiohttp is None:
            raise ImportError("aiohttp is required. Install with: pip install aiohttp")

        self.db_url = db_url
        self.symbols = symbols or self.DEFAULT_SYMBOLS
        self.snapshot_interval = snapshot_interval
        self.depth = depth
        self.rate_limit_delay = rate_limit_delay

        self.pool: Optional[asyncpg.Pool] = None
        self.session: Optional[aiohttp.ClientSession] = None
        self._running = False

        # Statistics
        self._snapshots_collected = 0
        self._errors = 0

    async def start(self):
        """Initialize connections and create table if needed."""
        self.pool = await asyncpg.create_pool(
            self.db_url,
            min_size=2,
            max_size=5
        )
        self.session = aiohttp.ClientSession()

        # Create table if not exists
        await self._create_table()

        self._running = True
        logger.info(f"OrderBookCollector started for {self.symbols}")

    async def stop(self):
        """Close connections."""
        self._running = False
        if self.session:
            await self.session.close()
        if self.pool:
            await self.pool.close()
        logger.info(
            f"OrderBookCollector stopped. "
            f"Snapshots: {self._snapshots_collected}, Errors: {self._errors}"
        )

    async def _create_table(self):
        """Create order_book_snapshots table if it doesn't exist."""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS order_book_snapshots (
                    id BIGSERIAL,
                    symbol VARCHAR(20) NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    bid_price DECIMAL(20, 10),
                    ask_price DECIMAL(20, 10),
                    spread DECIMAL(20, 10),
                    spread_pct DECIMAL(10, 6),
                    mid_price DECIMAL(20, 10),
                    bid_volume_total DECIMAL(20, 10),
                    ask_volume_total DECIMAL(20, 10),
                    imbalance DECIMAL(10, 6),
                    depth_levels INTEGER,
                    bids JSONB,
                    asks JSONB,
                    PRIMARY KEY (timestamp, symbol)
                );

                -- Create hypertable if TimescaleDB is available
                SELECT create_hypertable('order_book_snapshots', 'timestamp',
                    chunk_time_interval => INTERVAL '1 day',
                    if_not_exists => TRUE
                );

                -- Index for symbol queries
                CREATE INDEX IF NOT EXISTS idx_order_book_symbol_ts
                    ON order_book_snapshots (symbol, timestamp DESC);

                -- Compression policy (after 7 days)
                ALTER TABLE order_book_snapshots SET (
                    timescaledb.compress,
                    timescaledb.compress_segmentby = 'symbol',
                    timescaledb.compress_orderby = 'timestamp DESC'
                );
                SELECT add_compression_policy('order_book_snapshots',
                    INTERVAL '7 days', if_not_exists => TRUE);

                -- Retention policy (30 days - order books are large)
                SELECT add_retention_policy('order_book_snapshots',
                    INTERVAL '30 days', if_not_exists => TRUE);
            """)
            logger.info("Order book table ready")

    async def fetch_order_book(self, symbol: str) -> Optional[dict]:
        """
        Fetch order book from Kraken REST API.

        Args:
            symbol: Our symbol format (e.g., 'XRP/USDT')

        Returns:
            Order book data or None on error
        """
        pair = self.PAIR_MAP.get(symbol)
        if not pair:
            logger.error(f"Unknown symbol: {symbol}")
            return None

        url = f'{self.KRAKEN_BASE_URL}/0/public/Depth'
        params = {'pair': pair, 'count': self.depth}

        try:
            async with self.session.get(url, params=params, timeout=10) as response:
                data = await response.json()

                if data.get('error'):
                    logger.error(f"Kraken API error: {data['error']}")
                    return None

                result = data['result']
                # Get the pair key (varies by pair)
                pair_key = [k for k in result.keys()][0] if result else None

                if pair_key:
                    return result[pair_key]

        except asyncio.TimeoutError:
            logger.warning(f"Timeout fetching order book for {symbol}")
        except Exception as e:
            logger.error(f"Error fetching order book for {symbol}: {e}")

        return None

    async def store_snapshot(self, symbol: str, order_book: dict):
        """
        Process and store order book snapshot.

        Args:
            symbol: Trading pair
            order_book: Raw order book from Kraken
                {'bids': [[price, volume, timestamp], ...],
                 'asks': [[price, volume, timestamp], ...]}
        """
        try:
            timestamp = datetime.now(timezone.utc)

            bids = order_book.get('bids', [])
            asks = order_book.get('asks', [])

            if not bids or not asks:
                logger.warning(f"Empty order book for {symbol}")
                return

            # Calculate metrics
            best_bid = Decimal(str(bids[0][0]))
            best_ask = Decimal(str(asks[0][0]))
            spread = best_ask - best_bid
            mid_price = (best_bid + best_ask) / 2
            spread_pct = (spread / mid_price) * 100 if mid_price > 0 else Decimal(0)

            # Total volume at all levels
            bid_volume_total = sum(Decimal(str(b[1])) for b in bids)
            ask_volume_total = sum(Decimal(str(a[1])) for a in asks)

            # Imbalance: (bid_vol - ask_vol) / (bid_vol + ask_vol)
            total_vol = bid_volume_total + ask_volume_total
            imbalance = (bid_volume_total - ask_volume_total) / total_vol if total_vol > 0 else Decimal(0)

            # Store compact representation of levels
            bids_json = [[str(b[0]), str(b[1])] for b in bids]
            asks_json = [[str(a[0]), str(a[1])] for a in asks]

            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO order_book_snapshots
                        (symbol, timestamp, bid_price, ask_price, spread, spread_pct,
                         mid_price, bid_volume_total, ask_volume_total, imbalance,
                         depth_levels, bids, asks)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                    ON CONFLICT (timestamp, symbol) DO NOTHING
                """,
                    symbol, timestamp, best_bid, best_ask, spread, spread_pct,
                    mid_price, bid_volume_total, ask_volume_total, imbalance,
                    len(bids), json.dumps(bids_json), json.dumps(asks_json)
                )

            self._snapshots_collected += 1
            logger.debug(
                f"{symbol} order book: spread={spread:.8f} ({spread_pct:.4f}%), "
                f"imbalance={imbalance:.4f}"
            )

        except Exception as e:
            logger.error(f"Error storing order book for {symbol}: {e}")
            self._errors += 1

    async def collect_once(self):
        """Collect order book snapshot for all symbols once."""
        for symbol in self.symbols:
            order_book = await self.fetch_order_book(symbol)
            if order_book:
                await self.store_snapshot(symbol, order_book)
            await asyncio.sleep(self.rate_limit_delay)

    async def run_collection_loop(self):
        """Run continuous collection loop."""
        logger.info(
            f"Starting order book collection loop "
            f"(interval: {self.snapshot_interval}s, depth: {self.depth} levels)"
        )

        while self._running:
            try:
                await self.collect_once()

                # Log stats periodically
                if self._snapshots_collected % 100 == 0:
                    logger.info(
                        f"Order book stats: {self._snapshots_collected} snapshots, "
                        f"{self._errors} errors"
                    )

                # Wait for next collection cycle
                await asyncio.sleep(self.snapshot_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Collection loop error: {e}")
                self._errors += 1
                await asyncio.sleep(5)  # Back off on error

    def get_stats(self) -> dict:
        """Get collector statistics."""
        return {
            'snapshots_collected': self._snapshots_collected,
            'errors': self._errors,
            'symbols': self.symbols,
            'snapshot_interval': self.snapshot_interval,
            'depth': self.depth,
        }


async def run_order_book_collector(
    db_url: str,
    symbols: Optional[List[str]] = None,
    snapshot_interval: float = 60.0,
    depth: int = 25,
) -> None:
    """
    Convenience function to run order book collector.

    Args:
        db_url: PostgreSQL connection URL
        symbols: List of symbols to collect
        snapshot_interval: Seconds between snapshots
        depth: Number of order book levels
    """
    collector = OrderBookCollector(
        db_url,
        symbols=symbols,
        snapshot_interval=snapshot_interval,
        depth=depth,
    )

    try:
        await collector.start()
        await collector.run_collection_loop()
    finally:
        await collector.stop()


async def main():
    """Run order book collector from command line."""
    import argparse

    parser = argparse.ArgumentParser(description='Collect Kraken order book depth')
    parser.add_argument('--symbols', nargs='+', default=None,
                        help='Trading pairs to collect')
    parser.add_argument('--db-url', type=str,
                        default=os.getenv('DATABASE_URL'),
                        help='PostgreSQL connection URL')
    parser.add_argument('--interval', type=float, default=60.0,
                        help='Seconds between snapshots (default: 60)')
    parser.add_argument('--depth', type=int, default=25,
                        choices=[10, 25, 100, 500],
                        help='Number of order book levels (default: 25)')
    parser.add_argument('--once', action='store_true',
                        help='Collect once and exit')

    args = parser.parse_args()

    if not args.db_url:
        parser.error(
            "--db-url or DATABASE_URL environment variable is required.\n"
            "Example: DATABASE_URL=postgresql://trading:password@localhost:5433/kraken_data"
        )

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    collector = OrderBookCollector(
        args.db_url,
        symbols=args.symbols,
        snapshot_interval=args.interval,
        depth=args.depth,
    )

    try:
        await collector.start()

        if args.once:
            await collector.collect_once()
            print(f"\nCollected {collector._snapshots_collected} snapshots")
        else:
            await collector.run_collection_loop()

    finally:
        await collector.stop()


if __name__ == '__main__':
    asyncio.run(main())
