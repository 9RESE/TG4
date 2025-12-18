"""
Integration tests for the Database module.

These tests require a running TimescaleDB instance.
Run with: pytest triplegain/tests/integration/ -v

Environment variables required:
- TEST_DB_HOST (default: localhost)
- TEST_DB_PORT (default: 5433)
- TEST_DB_NAME (default: kraken_data)
- TEST_DB_USER (default: trading)
- TEST_DB_PASSWORD (required)
"""

import os
import pytest
from datetime import datetime, timezone, timedelta
from decimal import Decimal

# Skip all tests if asyncpg is not available
pytest.importorskip("asyncpg")

from triplegain.src.data.database import (
    DatabaseConfig,
    DatabasePool,
    create_pool_from_config,
)


# =============================================================================
# Test Configuration
# =============================================================================

def get_test_db_config() -> DatabaseConfig:
    """Get test database configuration from environment."""
    # Load from .env file if exists
    env_file = os.path.join(
        os.path.dirname(__file__),
        '../../../data/kraken_db/.env'
    )
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    if key not in os.environ:
                        os.environ[key] = value

    password = os.environ.get('DB_PASSWORD', '')
    if not password:
        pytest.skip("DB_PASSWORD environment variable not set")

    return DatabaseConfig(
        host=os.environ.get('TEST_DB_HOST', 'localhost'),
        port=int(os.environ.get('TEST_DB_PORT', '5433')),
        database=os.environ.get('TEST_DB_NAME', 'kraken_data'),
        user=os.environ.get('TEST_DB_USER', 'trading'),
        password=password,
        min_connections=1,
        max_connections=5,
        command_timeout=30,
    )


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
async def db_pool():
    """Create and connect a database pool for testing."""
    config = get_test_db_config()
    pool = DatabasePool(config)

    await pool.connect()
    yield pool
    await pool.disconnect()


# =============================================================================
# Connection Tests
# =============================================================================

class TestDatabaseConnection:
    """Tests for database connection and pool management."""

    @pytest.mark.asyncio
    async def test_connect_and_disconnect(self):
        """Test pool connection and disconnection."""
        config = get_test_db_config()
        pool = DatabasePool(config)

        assert not pool.is_connected

        await pool.connect()
        assert pool.is_connected

        await pool.disconnect()
        assert not pool.is_connected

    @pytest.mark.asyncio
    async def test_health_check(self, db_pool):
        """Test database health check."""
        health = await db_pool.check_health()

        assert health['status'] == 'healthy'
        assert health['connected'] is True
        assert 'version' in health
        assert 'PostgreSQL' in health['version']

    @pytest.mark.asyncio
    async def test_acquire_connection(self, db_pool):
        """Test acquiring a connection from the pool."""
        async with db_pool.acquire() as conn:
            result = await conn.fetchval("SELECT 1")
            assert result == 1

    @pytest.mark.asyncio
    async def test_multiple_queries(self, db_pool):
        """Test executing multiple queries."""
        async with db_pool.acquire() as conn:
            result1 = await conn.fetchval("SELECT 1 + 1")
            result2 = await conn.fetchval("SELECT 2 * 3")

        assert result1 == 2
        assert result2 == 6


# =============================================================================
# Candle Fetching Tests
# =============================================================================

class TestFetchCandles:
    """Tests for fetching candle data from continuous aggregates."""

    @pytest.mark.asyncio
    async def test_fetch_candles_1h(self, db_pool):
        """Test fetching 1h candles."""
        candles = await db_pool.fetch_candles('BTCUSDT', '1h', limit=10)

        # Should return a list (may be empty if no data)
        assert isinstance(candles, list)

        if candles:
            # Check candle structure
            candle = candles[0]
            assert 'timestamp' in candle
            assert 'open' in candle
            assert 'high' in candle
            assert 'low' in candle
            assert 'close' in candle
            assert 'volume' in candle

    @pytest.mark.asyncio
    async def test_fetch_candles_multiple_timeframes(self, db_pool):
        """Test fetching candles for multiple timeframes."""
        timeframes = ['5m', '15m', '1h', '4h', '1d']

        for tf in timeframes:
            candles = await db_pool.fetch_candles('XRPUSDT', tf, limit=5)
            assert isinstance(candles, list)

    @pytest.mark.asyncio
    async def test_fetch_candles_with_symbol_slash(self, db_pool):
        """Test fetching candles with symbol containing slash."""
        # Symbol with slash should be normalized
        candles = await db_pool.fetch_candles('XRP/USDT', '1h', limit=5)
        assert isinstance(candles, list)

    @pytest.mark.asyncio
    async def test_fetch_candles_invalid_timeframe(self, db_pool):
        """Test fetching candles with invalid timeframe."""
        with pytest.raises(ValueError, match="Unsupported timeframe"):
            await db_pool.fetch_candles('BTCUSDT', '2h', limit=5)

    @pytest.mark.asyncio
    async def test_fetch_candles_ordering(self, db_pool):
        """Test that candles are returned oldest first."""
        candles = await db_pool.fetch_candles('XRPUSDT', '1h', limit=10)

        if len(candles) >= 2:
            # Timestamps should be ascending (oldest first)
            for i in range(1, len(candles)):
                assert candles[i]['timestamp'] >= candles[i-1]['timestamp']


# =============================================================================
# 24h Data Tests
# =============================================================================

class TestFetch24hData:
    """Tests for fetching 24h price and volume data."""

    @pytest.mark.asyncio
    async def test_fetch_24h_data(self, db_pool):
        """Test fetching 24h data."""
        data = await db_pool.fetch_24h_data('XRPUSDT')

        assert isinstance(data, dict)
        assert 'price_24h_ago' in data
        assert 'volume_24h' in data
        assert 'price_change_24h_pct' in data

    @pytest.mark.asyncio
    async def test_fetch_24h_data_with_slash(self, db_pool):
        """Test fetching 24h data with symbol containing slash."""
        data = await db_pool.fetch_24h_data('XRP/USDT')
        assert isinstance(data, dict)


# =============================================================================
# Order Book Tests
# =============================================================================

class TestFetchOrderBook:
    """Tests for fetching order book data."""

    @pytest.mark.asyncio
    async def test_fetch_order_book(self, db_pool):
        """Test fetching order book snapshot."""
        order_book = await db_pool.fetch_order_book('XRPUSDT')

        # May be None if no order book data exists
        if order_book is not None:
            assert isinstance(order_book, dict)

    @pytest.mark.asyncio
    async def test_fetch_order_book_nonexistent_symbol(self, db_pool):
        """Test fetching order book for non-existent symbol."""
        order_book = await db_pool.fetch_order_book('NONEXISTENT')

        # Should return None for missing data
        assert order_book is None


# =============================================================================
# Pool Configuration Tests
# =============================================================================

class TestPoolConfiguration:
    """Tests for pool configuration from dict."""

    @pytest.mark.asyncio
    async def test_create_pool_from_config_dict(self):
        """Test creating pool from configuration dictionary."""
        test_config = get_test_db_config()

        config_dict = {
            'connection': {
                'host': test_config.host,
                'port': test_config.port,
                'database': test_config.database,
                'user': test_config.user,
                'password': test_config.password,
                'min_connections': 1,
                'max_connections': 3,
            }
        }

        pool = create_pool_from_config(config_dict)

        await pool.connect()
        assert pool.is_connected

        health = await pool.check_health()
        assert health['status'] == 'healthy'

        await pool.disconnect()
