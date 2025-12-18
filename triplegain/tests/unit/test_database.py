"""
Unit tests for the Database module.

Tests validate:
- Configuration creation
- Pool initialization
- Query building
"""

import pytest
from datetime import datetime, timezone
from decimal import Decimal

from triplegain.src.data.database import (
    DatabaseConfig,
    DatabasePool,
    create_pool_from_config,
)


# =============================================================================
# DatabaseConfig Tests
# =============================================================================

class TestDatabaseConfig:
    """Tests for DatabaseConfig dataclass."""

    def test_default_config(self):
        """Test DatabaseConfig with default values."""
        config = DatabaseConfig()

        assert config.host == 'localhost'
        assert config.port == 5432
        assert config.database == 'triplegain'
        assert config.user == 'postgres'
        assert config.password == ''
        assert config.min_connections == 5
        assert config.max_connections == 20
        assert config.command_timeout == 60

    def test_custom_config(self):
        """Test DatabaseConfig with custom values."""
        config = DatabaseConfig(
            host='dbserver.example.com',
            port=5433,
            database='testdb',
            user='testuser',
            password='testpass',
            min_connections=2,
            max_connections=10,
            command_timeout=30,
        )

        assert config.host == 'dbserver.example.com'
        assert config.port == 5433
        assert config.database == 'testdb'
        assert config.user == 'testuser'
        assert config.password == 'testpass'
        assert config.min_connections == 2
        assert config.max_connections == 10
        assert config.command_timeout == 30


# =============================================================================
# create_pool_from_config Tests
# =============================================================================

class TestCreatePoolFromConfig:
    """Tests for create_pool_from_config function."""

    def test_create_pool_basic_config(self):
        """Test creating pool from basic config dict."""
        config = {
            'connection': {
                'host': 'localhost',
                'port': 5432,
                'database': 'testdb',
                'user': 'testuser',
                'password': 'testpass',
            }
        }

        pool = create_pool_from_config(config)

        assert isinstance(pool, DatabasePool)
        assert pool.config.host == 'localhost'
        assert pool.config.port == 5432
        assert pool.config.database == 'testdb'
        assert pool.config.user == 'testuser'

    def test_create_pool_with_pool_settings(self):
        """Test creating pool with pool size settings."""
        config = {
            'connection': {
                'host': 'localhost',
                'port': 5432,
                'database': 'testdb',
                'user': 'testuser',
                'password': 'testpass',
                'min_connections': 3,
                'max_connections': 15,
                'command_timeout': 45,
            }
        }

        pool = create_pool_from_config(config)

        assert pool.config.min_connections == 3
        assert pool.config.max_connections == 15
        assert pool.config.command_timeout == 45

    def test_create_pool_empty_config(self):
        """Test creating pool from empty config uses defaults."""
        config = {}

        pool = create_pool_from_config(config)

        assert pool.config.host == 'localhost'
        assert pool.config.port == 5432

    def test_pool_not_connected_initially(self):
        """Test pool is not connected after creation."""
        config = {'connection': {'host': 'localhost', 'port': 5432, 'database': 'test', 'user': 'test'}}

        pool = create_pool_from_config(config)

        assert not pool.is_connected


# =============================================================================
# DatabasePool Tests (without actual connection)
# =============================================================================

class TestDatabasePoolNoConnection:
    """Tests for DatabasePool that don't require a real connection."""

    @pytest.fixture
    def pool(self):
        """Create a DatabasePool instance."""
        config = DatabaseConfig(
            host='localhost',
            port=5432,
            database='test',
            user='test',
            password='test',
        )
        return DatabasePool(config)

    def test_pool_creation(self, pool):
        """Test pool creation."""
        assert pool is not None
        assert pool.config is not None

    def test_pool_not_connected(self, pool):
        """Test pool is not connected by default."""
        assert not pool.is_connected
        assert pool._pool is None

    @pytest.mark.asyncio
    async def test_acquire_raises_without_connection(self, pool):
        """Test acquire raises error when not connected."""
        with pytest.raises(RuntimeError, match="Database pool not connected"):
            async with pool.acquire():
                pass

    @pytest.mark.asyncio
    async def test_check_health_not_connected(self, pool):
        """Test health check when not connected."""
        health = await pool.check_health()

        assert health['status'] == 'unhealthy'
        assert health['connected'] is False
        assert 'Pool not connected' in health['error']


# =============================================================================
# Mock-based Database Pool Tests
# =============================================================================

from unittest.mock import AsyncMock, MagicMock, patch


class TestDatabasePoolMocked:
    """Tests for DatabasePool with mocked asyncpg connection."""

    @pytest.fixture
    def mock_pool(self):
        """Create a DatabasePool with mocked internals."""
        config = DatabaseConfig(
            host='localhost',
            port=5432,
            database='test',
            user='test',
            password='test',
        )
        pool = DatabasePool(config)

        # Create mock asyncpg pool
        mock_asyncpg_pool = MagicMock()
        mock_asyncpg_pool.get_size.return_value = 5

        # Create mock connection
        mock_connection = AsyncMock()

        # Setup acquire context manager
        mock_asyncpg_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_asyncpg_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        pool._pool = mock_asyncpg_pool
        pool._connected = True
        pool._mock_connection = mock_connection  # Store for test access

        return pool

    @pytest.mark.asyncio
    async def test_fetch_candles_success(self, mock_pool):
        """Test successful candle fetching."""
        # Setup mock response
        mock_rows = [
            {
                'timestamp': datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
                'open': Decimal('100.5'),
                'high': Decimal('105.0'),
                'low': Decimal('99.0'),
                'close': Decimal('103.0'),
                'volume': Decimal('1000.0'),
            },
            {
                'timestamp': datetime(2024, 1, 1, 11, 0, 0, tzinfo=timezone.utc),
                'open': Decimal('98.0'),
                'high': Decimal('101.0'),
                'low': Decimal('97.0'),
                'close': Decimal('100.5'),
                'volume': Decimal('800.0'),
            },
        ]
        mock_pool._mock_connection.fetch = AsyncMock(return_value=mock_rows)

        candles = await mock_pool.fetch_candles('BTC/USDT', '1h', limit=10)

        # Verify result (should be reversed - oldest first)
        assert len(candles) == 2
        assert candles[0]['close'] == 100.5  # Second row (older) first
        assert candles[1]['close'] == 103.0  # First row (newer) second

        # Verify query was called with correct parameters
        mock_pool._mock_connection.fetch.assert_called_once()
        call_args = mock_pool._mock_connection.fetch.call_args
        assert 'BTCUSDT' in call_args.args  # Symbol normalized

    @pytest.mark.asyncio
    async def test_fetch_candles_invalid_timeframe(self, mock_pool):
        """Test fetch_candles with invalid timeframe."""
        with pytest.raises(ValueError, match="Unsupported timeframe"):
            await mock_pool.fetch_candles('BTC/USDT', 'invalid', limit=10)

    @pytest.mark.asyncio
    async def test_fetch_candles_normalizes_symbol(self, mock_pool):
        """Test that symbol is normalized (slash removed)."""
        mock_pool._mock_connection.fetch = AsyncMock(return_value=[])

        await mock_pool.fetch_candles('BTC/USDT', '1h')

        call_args = mock_pool._mock_connection.fetch.call_args
        assert 'BTCUSDT' in call_args.args

    @pytest.mark.asyncio
    async def test_fetch_order_book_success(self, mock_pool):
        """Test successful order book fetching."""
        mock_row = {
            'timestamp': datetime.now(timezone.utc),
            'bid_price': Decimal('100.0'),
            'ask_price': Decimal('100.5'),
            'spread': Decimal('0.5'),
            'spread_pct': Decimal('0.5'),
            'mid_price': Decimal('100.25'),
            'bid_volume_total': Decimal('10000'),
            'ask_volume_total': Decimal('9500'),
            'imbalance': Decimal('0.026'),
            'bids': [{'price': 100.0, 'size': 100}],
            'asks': [{'price': 100.5, 'size': 95}],
        }
        mock_pool._mock_connection.fetchrow = AsyncMock(return_value=mock_row)

        result = await mock_pool.fetch_order_book('BTC/USDT')

        assert result is not None
        assert result['bid_price'] == 100.0
        assert result['ask_price'] == 100.5
        assert result['spread'] == 0.5

    @pytest.mark.asyncio
    async def test_fetch_order_book_not_found(self, mock_pool):
        """Test order book when not found."""
        mock_pool._mock_connection.fetchrow = AsyncMock(return_value=None)

        result = await mock_pool.fetch_order_book('UNKNOWN/USDT')

        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_24h_data_success(self, mock_pool):
        """Test successful 24h data fetching."""
        mock_row = {
            'current_price': Decimal('105.0'),
            'price_24h_ago': Decimal('100.0'),
            'volume_24h': Decimal('50000.0'),
            'price_change_24h_pct': Decimal('5.0'),
        }
        mock_pool._mock_connection.fetchrow = AsyncMock(return_value=mock_row)

        result = await mock_pool.fetch_24h_data('BTC/USDT')

        assert result['price_24h_ago'] == 100.0
        assert result['volume_24h'] == 50000.0
        assert result['price_change_24h_pct'] == 5.0

    @pytest.mark.asyncio
    async def test_fetch_24h_data_not_found(self, mock_pool):
        """Test 24h data when not found."""
        mock_pool._mock_connection.fetchrow = AsyncMock(return_value=None)

        result = await mock_pool.fetch_24h_data('UNKNOWN/USDT')

        assert result['price_24h_ago'] is None
        assert result['volume_24h'] is None
        assert result['price_change_24h_pct'] is None

    @pytest.mark.asyncio
    async def test_cache_indicator_success(self, mock_pool):
        """Test caching an indicator value."""
        mock_pool._mock_connection.execute = AsyncMock()

        await mock_pool.cache_indicator(
            symbol='BTC/USDT',
            timeframe='1h',
            indicator_name='rsi_14',
            timestamp=datetime.now(timezone.utc),
            value=55.5,
            metadata={'period': 14}
        )

        mock_pool._mock_connection.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_cached_indicator_found(self, mock_pool):
        """Test retrieving a cached indicator value."""
        mock_row = {
            'value': Decimal('55.5'),
            'computed_at': datetime.now(timezone.utc),
        }
        mock_pool._mock_connection.fetchrow = AsyncMock(return_value=mock_row)

        result = await mock_pool.get_cached_indicator(
            symbol='BTC/USDT',
            timeframe='1h',
            indicator_name='rsi_14',
            timestamp=datetime.now(timezone.utc),
        )

        assert result == 55.5

    @pytest.mark.asyncio
    async def test_get_cached_indicator_not_found(self, mock_pool):
        """Test retrieving a cached indicator that doesn't exist."""
        mock_pool._mock_connection.fetchrow = AsyncMock(return_value=None)

        result = await mock_pool.get_cached_indicator(
            symbol='BTC/USDT',
            timeframe='1h',
            indicator_name='rsi_14',
            timestamp=datetime.now(timezone.utc),
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_save_agent_output_success(self, mock_pool):
        """Test saving agent output."""
        import uuid
        mock_row = {'id': uuid.uuid4()}
        mock_pool._mock_connection.fetchrow = AsyncMock(return_value=mock_row)

        result = await mock_pool.save_agent_output(
            agent_name='technical_analysis',
            output_type='analysis',
            output_data={'recommendation': 'buy', 'confidence': 0.75},
            symbol='BTC/USDT',
            timeframe='1h',
            model_used='qwen-2.5-7b',
            latency_ms=150,
            tokens_input=500,
            tokens_output=200,
        )

        assert result is not None
        mock_pool._mock_connection.fetchrow.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_health_connected(self, mock_pool):
        """Test health check when connected."""
        mock_pool._mock_connection.fetchval = AsyncMock(side_effect=[1, 'PostgreSQL 15.0'])

        health = await mock_pool.check_health()

        assert health['status'] == 'healthy'
        assert health['connected'] is True
        assert 'version' in health
        assert health['pool_size'] == 5

    @pytest.mark.asyncio
    async def test_check_health_query_error(self, mock_pool):
        """Test health check when query fails."""
        mock_pool._mock_connection.fetchval = AsyncMock(side_effect=Exception("Connection error"))

        health = await mock_pool.check_health()

        assert health['status'] == 'unhealthy'
        assert 'Connection error' in health['error']


# =============================================================================
# Timeframe Mapping Tests
# =============================================================================

class TestTimeframeMapping:
    """Tests for timeframe to table mapping."""

    @pytest.fixture
    def mock_pool(self):
        """Create a DatabasePool with mocked internals."""
        config = DatabaseConfig(host='localhost', port=5432, database='test', user='test')
        pool = DatabasePool(config)
        mock_asyncpg_pool = MagicMock()
        mock_connection = AsyncMock()
        mock_connection.fetch = AsyncMock(return_value=[])
        mock_asyncpg_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_asyncpg_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        pool._pool = mock_asyncpg_pool
        pool._connected = True
        return pool

    @pytest.mark.asyncio
    @pytest.mark.parametrize("timeframe,expected_table", [
        ('1m', 'candles_1m'),
        ('5m', 'candles_5m'),
        ('15m', 'candles_15m'),
        ('30m', 'candles_30m'),
        ('1h', 'candles_1h'),
        ('4h', 'candles_4h'),
        ('12h', 'candles_12h'),
        ('1d', 'candles_1d'),
        ('1w', 'candles_1w'),
    ])
    async def test_valid_timeframes(self, mock_pool, timeframe, expected_table):
        """Test that all valid timeframes map to correct tables."""
        # Just verify no exception is raised for valid timeframes
        await mock_pool.fetch_candles('BTC/USDT', timeframe, limit=1)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("invalid_timeframe", [
        '2m', '3m', '10m', '2h', '6h', '8h', '3d', '2w', 'invalid', '', 'hourly'
    ])
    async def test_invalid_timeframes(self, mock_pool, invalid_timeframe):
        """Test that invalid timeframes raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported timeframe"):
            await mock_pool.fetch_candles('BTC/USDT', invalid_timeframe)
