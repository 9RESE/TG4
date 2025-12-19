"""
Database Connection Pool - Async PostgreSQL/TimescaleDB connection management.

This module provides:
- Connection pool creation and management
- Async query execution
- Candle and order book fetching from continuous aggregates
- Indicator caching
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Any, AsyncIterator, Optional

try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False
    asyncpg = None

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database connection configuration."""
    host: str = 'localhost'
    port: int = 5432
    database: str = 'triplegain'
    user: str = 'postgres'
    password: str = ''
    min_connections: int = 5
    max_connections: int = 20
    command_timeout: int = 60


class DatabasePool:
    """
    Async database connection pool for TimescaleDB.

    Provides connection pooling and common query methods.
    """

    def __init__(self, config: DatabaseConfig):
        """
        Initialize DatabasePool.

        Args:
            config: Database configuration
        """
        if not ASYNCPG_AVAILABLE:
            raise RuntimeError("asyncpg is not installed. Install with: pip install asyncpg")

        self.config = config
        self._pool: Optional[asyncpg.Pool] = None
        self._connected = False

    async def connect(self) -> None:
        """Create the connection pool."""
        if self._pool is not None:
            return

        try:
            self._pool = await asyncpg.create_pool(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.user,
                password=self.config.password,
                min_size=self.config.min_connections,
                max_size=self.config.max_connections,
                command_timeout=self.config.command_timeout,
            )
            self._connected = True
            logger.info(f"Database pool created: {self.config.host}:{self.config.port}/{self.config.database}")
        except Exception as e:
            logger.error(f"Failed to create database pool: {e}")
            raise

    async def disconnect(self) -> None:
        """Close the connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            self._connected = False
            logger.info("Database pool closed")

    @property
    def is_connected(self) -> bool:
        """Check if pool is connected."""
        return self._connected and self._pool is not None

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[asyncpg.Connection]:
        """Acquire a connection from the pool."""
        if not self._pool:
            raise RuntimeError("Database pool not connected")

        async with self._pool.acquire() as connection:
            yield connection

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[asyncpg.Connection]:
        """
        Execute operations in an atomic transaction.

        Usage:
            async with db.transaction() as conn:
                await conn.execute("INSERT ...")
                await conn.execute("UPDATE ...")
        """
        if not self._pool:
            raise RuntimeError("Database pool not connected")

        async with self._pool.acquire() as connection:
            async with connection.transaction():
                yield connection

    async def fetch_candles(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 100,
        end_time: Optional[datetime] = None
    ) -> list[dict]:
        """
        Fetch candles from TimescaleDB continuous aggregates.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT" or "BTCUSDT")
            timeframe: Candle timeframe (e.g., "1m", "1h", "1d")
            limit: Maximum number of candles to fetch
            end_time: End time for the query (defaults to now)

        Returns:
            List of candle dictionaries (oldest first)
        """
        if end_time is None:
            end_time = datetime.now(timezone.utc)

        # Map timeframe to aggregate table name
        timeframe_map = {
            '1m': 'candles_1m',
            '5m': 'candles_5m',
            '15m': 'candles_15m',
            '30m': 'candles_30m',
            '1h': 'candles_1h',
            '4h': 'candles_4h',
            '12h': 'candles_12h',
            '1d': 'candles_1d',
            '1w': 'candles_1w',
        }

        table_name = timeframe_map.get(timeframe)
        if not table_name:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        # Normalize symbol format (remove slash if present)
        normalized_symbol = symbol.replace('/', '')

        query = f"""
            SELECT
                timestamp,
                open,
                high,
                low,
                close,
                volume
            FROM {table_name}
            WHERE symbol = $1
                AND timestamp <= $2
            ORDER BY timestamp DESC
            LIMIT $3
        """

        async with self.acquire() as conn:
            rows = await conn.fetch(query, normalized_symbol, end_time, limit)

        # Convert to list of dicts (oldest first)
        candles = [
            {
                'timestamp': row['timestamp'],
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume']),
            }
            for row in reversed(rows)
        ]

        return candles

    async def fetch_order_book(self, symbol: str) -> Optional[dict]:
        """
        Fetch the latest order book snapshot from the database.

        Args:
            symbol: Trading pair

        Returns:
            Order book dict with bids, asks, and spread info, or None if not found
        """
        normalized_symbol = symbol.replace('/', '')

        query = """
            SELECT
                timestamp,
                bid_price,
                ask_price,
                spread,
                spread_pct,
                mid_price,
                bid_volume_total,
                ask_volume_total,
                imbalance,
                depth_levels,
                bids,
                asks
            FROM order_book_snapshots
            WHERE symbol = $1
            ORDER BY timestamp DESC
            LIMIT 1
        """

        async with self.acquire() as conn:
            row = await conn.fetchrow(query, normalized_symbol)

        if row is None:
            return None

        return {
            'timestamp': row['timestamp'],
            'bid_price': float(row['bid_price']) if row['bid_price'] else None,
            'ask_price': float(row['ask_price']) if row['ask_price'] else None,
            'spread': float(row['spread']) if row['spread'] else None,
            'spread_pct': float(row['spread_pct']) if row['spread_pct'] else None,
            'mid_price': float(row['mid_price']) if row['mid_price'] else None,
            'bid_volume_total': float(row['bid_volume_total']) if row['bid_volume_total'] else None,
            'ask_volume_total': float(row['ask_volume_total']) if row['ask_volume_total'] else None,
            'imbalance': float(row['imbalance']) if row['imbalance'] else None,
            'bids': row['bids'],
            'asks': row['asks'],
        }

    async def fetch_24h_data(
        self,
        symbol: str,
        current_time: Optional[datetime] = None
    ) -> dict:
        """
        Fetch 24h price and volume data.

        Args:
            symbol: Trading pair
            current_time: Reference time (defaults to now)

        Returns:
            Dict with price_24h_ago, volume_24h, price_change_24h_pct
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        time_24h_ago = current_time - timedelta(hours=24)
        normalized_symbol = symbol.replace('/', '')

        # Query for price 24h ago and volume sum
        query = """
            WITH current_data AS (
                SELECT close as current_price
                FROM candles_1h
                WHERE symbol = $1 AND timestamp <= $2
                ORDER BY timestamp DESC
                LIMIT 1
            ),
            past_data AS (
                SELECT close as past_price
                FROM candles_1h
                WHERE symbol = $1 AND timestamp <= $3
                ORDER BY timestamp DESC
                LIMIT 1
            ),
            volume_data AS (
                SELECT COALESCE(SUM(volume), 0) as total_volume
                FROM candles_1h
                WHERE symbol = $1
                    AND timestamp > $3
                    AND timestamp <= $2
            )
            SELECT
                c.current_price,
                p.past_price as price_24h_ago,
                v.total_volume as volume_24h,
                CASE
                    WHEN p.past_price > 0
                    THEN ((c.current_price - p.past_price) / p.past_price) * 100
                    ELSE 0
                END as price_change_24h_pct
            FROM current_data c, past_data p, volume_data v
        """

        async with self.acquire() as conn:
            row = await conn.fetchrow(query, normalized_symbol, current_time, time_24h_ago)

        if row is None:
            return {
                'price_24h_ago': None,
                'volume_24h': None,
                'price_change_24h_pct': None,
            }

        return {
            'price_24h_ago': float(row['price_24h_ago']) if row['price_24h_ago'] else None,
            'volume_24h': float(row['volume_24h']) if row['volume_24h'] else None,
            'price_change_24h_pct': float(row['price_change_24h_pct']) if row['price_change_24h_pct'] else None,
        }

    async def cache_indicator(
        self,
        symbol: str,
        timeframe: str,
        indicator_name: str,
        timestamp: datetime,
        value: float,
        metadata: Optional[dict] = None
    ) -> None:
        """
        Cache a computed indicator value.

        Args:
            symbol: Trading pair
            timeframe: Candle timeframe
            indicator_name: Name of the indicator
            timestamp: Timestamp for the indicator value
            value: Indicator value
            metadata: Optional metadata dict
        """
        query = """
            INSERT INTO indicator_cache
                (symbol, timeframe, indicator_name, timestamp, value, metadata)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (symbol, timeframe, indicator_name, timestamp)
            DO UPDATE SET value = $5, metadata = $6, computed_at = NOW()
        """

        async with self.acquire() as conn:
            await conn.execute(
                query,
                symbol.replace('/', ''),
                timeframe,
                indicator_name,
                timestamp,
                value,
                metadata
            )

    async def get_cached_indicator(
        self,
        symbol: str,
        timeframe: str,
        indicator_name: str,
        timestamp: datetime,
        max_age_seconds: int = 60
    ) -> Optional[float]:
        """
        Get a cached indicator value if fresh enough.

        Args:
            symbol: Trading pair
            timeframe: Candle timeframe
            indicator_name: Name of the indicator
            timestamp: Timestamp for the indicator value
            max_age_seconds: Maximum age of cached value

        Returns:
            Cached value or None if not found/stale
        """
        query = """
            SELECT value, computed_at
            FROM indicator_cache
            WHERE symbol = $1
                AND timeframe = $2
                AND indicator_name = $3
                AND timestamp = $4
                AND computed_at >= NOW() - $5 * INTERVAL '1 second'
        """

        async with self.acquire() as conn:
            row = await conn.fetchrow(
                query,
                symbol.replace('/', ''),
                timeframe,
                indicator_name,
                timestamp,
                max_age_seconds
            )

        if row is None:
            return None

        return float(row['value'])

    async def save_agent_output(
        self,
        agent_name: str,
        output_type: str,
        output_data: dict,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        model_used: Optional[str] = None,
        prompt_hash: Optional[str] = None,
        latency_ms: Optional[int] = None,
        tokens_input: Optional[int] = None,
        tokens_output: Optional[int] = None,
        cost_usd: Optional[Decimal] = None,
        error_message: Optional[str] = None
    ) -> str:
        """
        Save an agent output to the database.

        Returns:
            UUID of the created record
        """
        query = """
            INSERT INTO agent_outputs (
                agent_name, symbol, timeframe, output_type, output_data,
                model_used, prompt_hash, latency_ms, tokens_input, tokens_output,
                cost_usd, error_message
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            RETURNING id
        """

        import json
        async with self.acquire() as conn:
            row = await conn.fetchrow(
                query,
                agent_name,
                symbol.replace('/', '') if symbol else None,
                timeframe,
                output_type,
                json.dumps(output_data),
                model_used,
                prompt_hash,
                latency_ms,
                tokens_input,
                tokens_output,
                cost_usd,
                error_message
            )

        return str(row['id'])

    async def check_health(self) -> dict:
        """
        Check database connectivity and health.

        Returns:
            Health status dict
        """
        if not self.is_connected:
            return {
                'status': 'unhealthy',
                'connected': False,
                'error': 'Pool not connected'
            }

        try:
            async with self.acquire() as conn:
                result = await conn.fetchval("SELECT 1")
                version = await conn.fetchval("SELECT version()")

            return {
                'status': 'healthy',
                'connected': True,
                'version': version,
                'pool_size': self._pool.get_size() if self._pool else 0,
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'connected': False,
                'error': str(e)
            }


def create_pool_from_config(config: dict) -> DatabasePool:
    """
    Create a DatabasePool from configuration dictionary.

    Args:
        config: Database configuration dict

    Returns:
        DatabasePool instance (not yet connected)
    """
    conn_config = config.get('connection', {})

    db_config = DatabaseConfig(
        host=conn_config.get('host', 'localhost'),
        port=int(conn_config.get('port', 5432)),
        database=conn_config.get('database', 'triplegain'),
        user=conn_config.get('user', 'postgres'),
        password=conn_config.get('password', ''),
        min_connections=int(conn_config.get('min_connections', 5)),
        max_connections=int(conn_config.get('max_connections', 20)),
        command_timeout=int(conn_config.get('command_timeout', 60)),
    )

    return DatabasePool(db_config)
