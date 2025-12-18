"""
Private Data Collector - Collect account trade history and ledger entries from Kraken.

This module provides:
- PrivateDataCollector: Fetch and store personal trade and ledger history

Data collected:
- Trade executions (your fills, not market trades)
- Ledger entries (deposits, withdrawals, trades, margin, staking)

Critical for:
- P&L tracking and analysis
- Fee calculation
- Position tracking
- Tax reporting

Usage:
    collector = PrivateDataCollector(db_url, api_key, api_secret)
    await collector.start()
    await collector.sync_all()  # Full sync
    await collector.stop()
"""

import asyncio
import base64
import hashlib
import hmac
import logging
import os
import time
import urllib.parse
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional

try:
    import aiohttp
except ImportError:
    aiohttp = None

try:
    import asyncpg
except ImportError:
    asyncpg = None

logger = logging.getLogger(__name__)


class PrivateDataCollector:
    """
    Collect private account data from Kraken API.

    Fetches and stores:
    - TradesHistory: Your executed trades with fees
    - Ledgers: All account ledger entries

    Requires API key with permissions:
    - Query Funds
    - Query Closed Orders & Trades
    """

    KRAKEN_BASE_URL = 'https://api.kraken.com'

    def __init__(
        self,
        db_url: str,
        api_key: str,
        api_secret: str,
        rate_limit_delay: float = 2.0,  # Private endpoints have stricter limits
    ):
        """
        Initialize PrivateDataCollector.

        Args:
            db_url: PostgreSQL connection URL
            api_key: Kraken API key
            api_secret: Kraken API secret (base64 encoded)
            rate_limit_delay: Seconds between API calls
        """
        if asyncpg is None:
            raise ImportError("asyncpg is required. Install with: pip install asyncpg")
        if aiohttp is None:
            raise ImportError("aiohttp is required. Install with: pip install aiohttp")

        self.db_url = db_url
        self.api_key = api_key
        self.api_secret = api_secret
        self.rate_limit_delay = rate_limit_delay

        self.pool: Optional[asyncpg.Pool] = None
        self.session: Optional[aiohttp.ClientSession] = None

        # Statistics
        self._trades_synced = 0
        self._ledgers_synced = 0
        self._errors = 0

    def _get_signature(self, urlpath: str, data: dict) -> str:
        """
        Generate Kraken API signature.

        Args:
            urlpath: API endpoint path (e.g., '/0/private/TradesHistory')
            data: POST data dictionary (must include 'nonce')

        Returns:
            Base64 encoded HMAC-SHA512 signature
        """
        postdata = urllib.parse.urlencode(data)
        encoded = (str(data['nonce']) + postdata).encode()
        message = urlpath.encode() + hashlib.sha256(encoded).digest()
        mac = hmac.new(base64.b64decode(self.api_secret), message, hashlib.sha512)
        return base64.b64encode(mac.digest()).decode()

    async def _private_request(
        self,
        endpoint: str,
        data: Optional[dict] = None
    ) -> Optional[dict]:
        """
        Make authenticated request to Kraken private API.

        Args:
            endpoint: API endpoint (e.g., 'TradesHistory')
            data: Additional POST data

        Returns:
            API response result or None on error
        """
        urlpath = f'/0/private/{endpoint}'
        url = f'{self.KRAKEN_BASE_URL}{urlpath}'

        post_data = data or {}
        post_data['nonce'] = str(int(time.time() * 1000))

        headers = {
            'API-Key': self.api_key,
            'API-Sign': self._get_signature(urlpath, post_data),
            'Content-Type': 'application/x-www-form-urlencoded'
        }

        try:
            async with self.session.post(
                url,
                data=post_data,
                headers=headers,
                timeout=30
            ) as response:
                result = await response.json()

                if result.get('error'):
                    errors = result['error']
                    # Check for rate limiting
                    if any('EAPI:Rate limit' in str(e) for e in errors):
                        logger.warning("Rate limited, backing off...")
                        await asyncio.sleep(10)
                        return None
                    logger.error(f"Kraken API error: {errors}")
                    self._errors += 1
                    return None

                return result.get('result', {})

        except asyncio.TimeoutError:
            logger.warning(f"Timeout on {endpoint}")
            self._errors += 1
        except Exception as e:
            logger.error(f"Error on {endpoint}: {e}")
            self._errors += 1

        return None

    async def start(self):
        """Initialize connections and create tables."""
        self.pool = await asyncpg.create_pool(
            self.db_url,
            min_size=2,
            max_size=5
        )
        self.session = aiohttp.ClientSession()

        await self._create_tables()
        logger.info("PrivateDataCollector started")

    async def stop(self):
        """Close connections."""
        if self.session:
            await self.session.close()
        if self.pool:
            await self.pool.close()
        logger.info(
            f"PrivateDataCollector stopped. "
            f"Trades: {self._trades_synced}, Ledgers: {self._ledgers_synced}, "
            f"Errors: {self._errors}"
        )

    async def _create_tables(self):
        """Create tables for private data."""
        async with self.pool.acquire() as conn:
            # Trade execution history table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS execution_history (
                    trade_id VARCHAR(50) PRIMARY KEY,
                    order_id VARCHAR(50),
                    pair VARCHAR(20) NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    type VARCHAR(10) NOT NULL,
                    order_type VARCHAR(20),
                    price DECIMAL(20, 10) NOT NULL,
                    cost DECIMAL(20, 10) NOT NULL,
                    fee DECIMAL(20, 10) NOT NULL,
                    volume DECIMAL(20, 10) NOT NULL,
                    margin DECIMAL(20, 10),
                    leverage VARCHAR(10),
                    misc VARCHAR(100),
                    position_status VARCHAR(20),
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );

                CREATE INDEX IF NOT EXISTS idx_execution_timestamp
                    ON execution_history (timestamp DESC);
                CREATE INDEX IF NOT EXISTS idx_execution_pair
                    ON execution_history (pair, timestamp DESC);
            """)

            # Ledger entries table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS ledger_entries (
                    ledger_id VARCHAR(50) PRIMARY KEY,
                    ref_id VARCHAR(50),
                    timestamp TIMESTAMPTZ NOT NULL,
                    type VARCHAR(20) NOT NULL,
                    subtype VARCHAR(20),
                    asset VARCHAR(20) NOT NULL,
                    amount DECIMAL(30, 10) NOT NULL,
                    fee DECIMAL(20, 10) NOT NULL,
                    balance DECIMAL(30, 10) NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );

                CREATE INDEX IF NOT EXISTS idx_ledger_timestamp
                    ON ledger_entries (timestamp DESC);
                CREATE INDEX IF NOT EXISTS idx_ledger_type
                    ON ledger_entries (type, timestamp DESC);
                CREATE INDEX IF NOT EXISTS idx_ledger_asset
                    ON ledger_entries (asset, timestamp DESC);
            """)

            # Balance history table (for tracking balance over time)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS balance_history (
                    id BIGSERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL,
                    asset VARCHAR(20) NOT NULL,
                    balance DECIMAL(30, 10) NOT NULL,
                    UNIQUE (timestamp, asset)
                );

                CREATE INDEX IF NOT EXISTS idx_balance_asset_ts
                    ON balance_history (asset, timestamp DESC);
            """)

            logger.info("Private data tables ready")

    async def fetch_trades_history(
        self,
        start: Optional[int] = None,
        end: Optional[int] = None,
        ofs: int = 0,
    ) -> Dict:
        """
        Fetch trade history page.

        Args:
            start: Starting unix timestamp
            end: Ending unix timestamp
            ofs: Result offset for pagination

        Returns:
            Trade history response
        """
        data = {'ofs': ofs}
        if start:
            data['start'] = start
        if end:
            data['end'] = end

        return await self._private_request('TradesHistory', data)

    async def fetch_ledgers(
        self,
        asset: Optional[str] = None,
        type_: Optional[str] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
        ofs: int = 0,
    ) -> Dict:
        """
        Fetch ledger entries page.

        Args:
            asset: Filter by asset (e.g., 'XRP', 'USDT')
            type_: Filter by type ('all', 'deposit', 'withdrawal', 'trade', 'margin')
            start: Starting unix timestamp
            end: Ending unix timestamp
            ofs: Result offset for pagination

        Returns:
            Ledger response
        """
        data = {'ofs': ofs}
        if asset:
            data['asset'] = asset
        if type_:
            data['type'] = type_
        if start:
            data['start'] = start
        if end:
            data['end'] = end

        return await self._private_request('Ledgers', data)

    async def sync_trades_history(self, since_timestamp: Optional[datetime] = None):
        """
        Sync all trade history to database.

        Args:
            since_timestamp: Only sync trades after this time
        """
        logger.info("Syncing trade history...")

        start = int(since_timestamp.timestamp()) if since_timestamp else None
        ofs = 0
        total = 0

        while True:
            result = await self.fetch_trades_history(start=start, ofs=ofs)
            if not result:
                break

            trades = result.get('trades', {})
            if not trades:
                break

            # Store trades
            for trade_id, trade in trades.items():
                await self._store_trade(trade_id, trade)
                total += 1

            # Check if more pages
            count = result.get('count', 0)
            if ofs + len(trades) >= count:
                break

            ofs += len(trades)
            await asyncio.sleep(self.rate_limit_delay)

        self._trades_synced += total
        logger.info(f"Synced {total} trades")

    async def _store_trade(self, trade_id: str, trade: dict):
        """Store a single trade execution."""
        try:
            timestamp = datetime.fromtimestamp(
                float(trade.get('time', 0)),
                tz=timezone.utc
            )

            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO execution_history
                        (trade_id, order_id, pair, timestamp, type, order_type,
                         price, cost, fee, volume, margin, leverage, misc, position_status)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                    ON CONFLICT (trade_id) DO UPDATE SET
                        cost = EXCLUDED.cost,
                        fee = EXCLUDED.fee
                """,
                    trade_id,
                    trade.get('ordertxid'),
                    trade.get('pair'),
                    timestamp,
                    trade.get('type'),  # buy/sell
                    trade.get('ordertype'),  # market/limit
                    Decimal(str(trade.get('price', 0))),
                    Decimal(str(trade.get('cost', 0))),
                    Decimal(str(trade.get('fee', 0))),
                    Decimal(str(trade.get('vol', 0))),
                    Decimal(str(trade.get('margin', 0))) if trade.get('margin') else None,
                    trade.get('leverage'),
                    trade.get('misc'),
                    trade.get('posstatus'),
                )

        except Exception as e:
            logger.error(f"Error storing trade {trade_id}: {e}")
            self._errors += 1

    async def sync_ledgers(self, since_timestamp: Optional[datetime] = None):
        """
        Sync all ledger entries to database.

        Args:
            since_timestamp: Only sync entries after this time
        """
        logger.info("Syncing ledger entries...")

        start = int(since_timestamp.timestamp()) if since_timestamp else None
        ofs = 0
        total = 0

        while True:
            result = await self.fetch_ledgers(start=start, ofs=ofs)
            if not result:
                break

            ledgers = result.get('ledger', {})
            if not ledgers:
                break

            # Store ledger entries
            for ledger_id, entry in ledgers.items():
                await self._store_ledger(ledger_id, entry)
                total += 1

            # Check if more pages
            count = result.get('count', 0)
            if ofs + len(ledgers) >= count:
                break

            ofs += len(ledgers)
            await asyncio.sleep(self.rate_limit_delay)

        self._ledgers_synced += total
        logger.info(f"Synced {total} ledger entries")

    async def _store_ledger(self, ledger_id: str, entry: dict):
        """Store a single ledger entry."""
        try:
            timestamp = datetime.fromtimestamp(
                float(entry.get('time', 0)),
                tz=timezone.utc
            )

            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO ledger_entries
                        (ledger_id, ref_id, timestamp, type, subtype, asset,
                         amount, fee, balance)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    ON CONFLICT (ledger_id) DO NOTHING
                """,
                    ledger_id,
                    entry.get('refid'),
                    timestamp,
                    entry.get('type'),
                    entry.get('subtype'),
                    entry.get('asset'),
                    Decimal(str(entry.get('amount', 0))),
                    Decimal(str(entry.get('fee', 0))),
                    Decimal(str(entry.get('balance', 0))),
                )

        except Exception as e:
            logger.error(f"Error storing ledger {ledger_id}: {e}")
            self._errors += 1

    async def fetch_current_balance(self) -> Dict[str, Decimal]:
        """Fetch current account balance."""
        result = await self._private_request('Balance')
        if not result:
            return {}

        balances = {}
        for asset, amount in result.items():
            balances[asset] = Decimal(str(amount))

        return balances

    async def snapshot_balance(self):
        """Take a snapshot of current balances."""
        balances = await self.fetch_current_balance()
        if not balances:
            return

        timestamp = datetime.now(timezone.utc)

        async with self.pool.acquire() as conn:
            for asset, balance in balances.items():
                if balance != 0:  # Only store non-zero balances
                    await conn.execute("""
                        INSERT INTO balance_history (timestamp, asset, balance)
                        VALUES ($1, $2, $3)
                        ON CONFLICT (timestamp, asset) DO UPDATE SET
                            balance = EXCLUDED.balance
                    """, timestamp, asset, balance)

        logger.info(f"Snapshotted {len(balances)} balances")

    async def sync_all(self):
        """Sync all private data."""
        logger.info("Starting full private data sync...")

        await self.sync_trades_history()
        await asyncio.sleep(self.rate_limit_delay)

        await self.sync_ledgers()
        await asyncio.sleep(self.rate_limit_delay)

        await self.snapshot_balance()

        logger.info("Full sync complete")

    def get_stats(self) -> dict:
        """Get collector statistics."""
        return {
            'trades_synced': self._trades_synced,
            'ledgers_synced': self._ledgers_synced,
            'errors': self._errors,
        }


async def run_private_data_sync(
    db_url: str,
    api_key: str,
    api_secret: str,
) -> dict:
    """
    Convenience function to run a full private data sync.

    Returns:
        Statistics dictionary
    """
    collector = PrivateDataCollector(db_url, api_key, api_secret)

    try:
        await collector.start()
        await collector.sync_all()
        return collector.get_stats()
    finally:
        await collector.stop()


async def main():
    """Run private data collector from command line."""
    import argparse

    parser = argparse.ArgumentParser(description='Sync Kraken private account data')
    parser.add_argument('--db-url', type=str,
                        default=os.getenv('DATABASE_URL'),
                        help='PostgreSQL connection URL')
    parser.add_argument('--api-key', type=str,
                        default=os.getenv('KRAKEN_API_KEY'),
                        help='Kraken API key')
    parser.add_argument('--api-secret', type=str,
                        default=os.getenv('KRAKEN_API_SECRET'),
                        help='Kraken API secret')
    parser.add_argument('--trades-only', action='store_true',
                        help='Only sync trade history')
    parser.add_argument('--ledgers-only', action='store_true',
                        help='Only sync ledger entries')
    parser.add_argument('--balance-only', action='store_true',
                        help='Only snapshot balance')

    args = parser.parse_args()

    if not args.db_url:
        parser.error("--db-url or DATABASE_URL environment variable is required")
    if not args.api_key:
        parser.error("--api-key or KRAKEN_API_KEY environment variable is required")
    if not args.api_secret:
        parser.error("--api-secret or KRAKEN_API_SECRET environment variable is required")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    collector = PrivateDataCollector(
        args.db_url,
        args.api_key,
        args.api_secret,
    )

    try:
        await collector.start()

        if args.trades_only:
            await collector.sync_trades_history()
        elif args.ledgers_only:
            await collector.sync_ledgers()
        elif args.balance_only:
            await collector.snapshot_balance()
        else:
            await collector.sync_all()

        stats = collector.get_stats()
        print("\n" + "=" * 60)
        print("Private Data Sync Results")
        print("=" * 60)
        print(f"  Trades synced: {stats['trades_synced']}")
        print(f"  Ledgers synced: {stats['ledgers_synced']}")
        print(f"  Errors: {stats['errors']}")
        print("=" * 60)

    finally:
        await collector.stop()


if __name__ == '__main__':
    asyncio.run(main())
