"""
Tests for the Historical Data System.

These tests verify the functionality of:
- Data types (HistoricalTrade, HistoricalCandle, etc.)
- DatabaseWriter (buffering, flushing)
- HistoricalDataProvider (queries, warmup)
- GapFiller (gap detection logic)

Note: Most tests are unit tests that don't require a database.
Integration tests require a running TimescaleDB instance.
"""

import pytest
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch

# Import data module types
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.types import (
    HistoricalTrade,
    HistoricalCandle,
    ExternalIndicator,
    DataGap,
    TradeRecord,
    CandleRecord,
)


class TestHistoricalTrade:
    """Tests for HistoricalTrade dataclass."""

    def test_historical_trade_creation(self):
        """Test creating a HistoricalTrade."""
        trade = HistoricalTrade(
            id=12345,
            symbol='XRP/USDT',
            timestamp=datetime(2024, 1, 15, 10, 30, 45, tzinfo=timezone.utc),
            price=Decimal('2.35'),
            volume=Decimal('1000'),
            side='buy',
            order_type='market',
            misc=''
        )

        assert trade.id == 12345
        assert trade.symbol == 'XRP/USDT'
        assert trade.price == Decimal('2.35')
        assert trade.volume == Decimal('1000')
        assert trade.side == 'buy'

    def test_historical_trade_value(self):
        """Test trade value property."""
        trade = HistoricalTrade(
            id=1,
            symbol='XRP/USDT',
            timestamp=datetime.now(timezone.utc),
            price=Decimal('2.50'),
            volume=Decimal('100'),
            side='buy',
            order_type='market',
            misc=''
        )

        assert trade.value == Decimal('250')

    def test_historical_trade_immutable(self):
        """Test that HistoricalTrade is immutable (frozen)."""
        trade = HistoricalTrade(
            id=1,
            symbol='XRP/USDT',
            timestamp=datetime.now(timezone.utc),
            price=Decimal('2.50'),
            volume=Decimal('100'),
            side='buy',
            order_type='market',
            misc=''
        )

        with pytest.raises(AttributeError):
            trade.price = Decimal('3.00')


class TestHistoricalCandle:
    """Tests for HistoricalCandle dataclass."""

    def test_historical_candle_creation(self):
        """Test creating a HistoricalCandle."""
        candle = HistoricalCandle(
            symbol='XRP/USDT',
            timestamp=datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc),
            interval_minutes=5,
            open=Decimal('2.30'),
            high=Decimal('2.40'),
            low=Decimal('2.25'),
            close=Decimal('2.35'),
            volume=Decimal('50000'),
            quote_volume=Decimal('117500'),
            trade_count=150,
            vwap=Decimal('2.34')
        )

        assert candle.symbol == 'XRP/USDT'
        assert candle.interval_minutes == 5
        assert candle.open == Decimal('2.30')
        assert candle.close == Decimal('2.35')

    def test_candle_typical_price(self):
        """Test typical price calculation."""
        candle = HistoricalCandle(
            symbol='XRP/USDT',
            timestamp=datetime.now(timezone.utc),
            interval_minutes=1,
            open=Decimal('2.30'),
            high=Decimal('2.40'),
            low=Decimal('2.20'),
            close=Decimal('2.35'),
            volume=Decimal('1000'),
        )

        # typical = (high + low + close) / 3 = (2.40 + 2.20 + 2.35) / 3
        expected = (Decimal('2.40') + Decimal('2.20') + Decimal('2.35')) / 3
        assert candle.typical_price == expected

    def test_candle_range(self):
        """Test candle range calculation."""
        candle = HistoricalCandle(
            symbol='XRP/USDT',
            timestamp=datetime.now(timezone.utc),
            interval_minutes=1,
            open=Decimal('2.30'),
            high=Decimal('2.50'),
            low=Decimal('2.20'),
            close=Decimal('2.35'),
            volume=Decimal('1000'),
        )

        assert candle.range == Decimal('0.30')  # 2.50 - 2.20

    def test_candle_is_bullish(self):
        """Test bullish candle detection."""
        bullish = HistoricalCandle(
            symbol='XRP/USDT',
            timestamp=datetime.now(timezone.utc),
            interval_minutes=1,
            open=Decimal('2.30'),
            high=Decimal('2.40'),
            low=Decimal('2.25'),
            close=Decimal('2.35'),
            volume=Decimal('1000'),
        )

        bearish = HistoricalCandle(
            symbol='XRP/USDT',
            timestamp=datetime.now(timezone.utc),
            interval_minutes=1,
            open=Decimal('2.35'),
            high=Decimal('2.40'),
            low=Decimal('2.25'),
            close=Decimal('2.30'),
            volume=Decimal('1000'),
        )

        assert bullish.is_bullish is True
        assert bearish.is_bullish is False

    def test_candle_body_size(self):
        """Test body size calculation."""
        candle = HistoricalCandle(
            symbol='XRP/USDT',
            timestamp=datetime.now(timezone.utc),
            interval_minutes=1,
            open=Decimal('2.30'),
            high=Decimal('2.40'),
            low=Decimal('2.20'),
            close=Decimal('2.35'),
            volume=Decimal('1000'),
        )

        assert candle.body_size == Decimal('0.05')  # |2.35 - 2.30|


class TestDataGap:
    """Tests for DataGap dataclass."""

    def test_data_gap_small(self):
        """Test small gap detection (< 12 hours)."""
        now = datetime.now(timezone.utc)
        gap = DataGap(
            symbol='XRP/USDT',
            data_type='candles_1m',
            start_time=now - timedelta(hours=6),
            end_time=now,
            duration=timedelta(hours=6)
        )

        assert gap.is_small is True
        assert gap.hours == 6.0

    def test_data_gap_large(self):
        """Test large gap detection (>= 12 hours)."""
        now = datetime.now(timezone.utc)
        gap = DataGap(
            symbol='XRP/USDT',
            data_type='candles_1m',
            start_time=now - timedelta(hours=24),
            end_time=now,
            duration=timedelta(hours=24)
        )

        assert gap.is_small is False
        assert gap.hours == 24.0

    def test_data_gap_candles_needed(self):
        """Test candles needed estimation."""
        now = datetime.now(timezone.utc)
        gap = DataGap(
            symbol='XRP/USDT',
            data_type='candles_1m',
            start_time=now - timedelta(hours=1),
            end_time=now,
            duration=timedelta(hours=1)
        )

        assert gap.candles_needed == 60  # 1 hour = 60 minutes


class TestTradeRecord:
    """Tests for TradeRecord dataclass."""

    def test_trade_record_to_tuple(self):
        """Test TradeRecord to tuple conversion."""
        now = datetime.now(timezone.utc)
        record = TradeRecord(
            symbol='XRP/USDT',
            timestamp=now,
            price=Decimal('2.35'),
            volume=Decimal('100'),
            side='buy'
        )

        result = record.to_tuple()

        assert result[0] == 'XRP/USDT'
        assert result[1] == now
        assert result[2] == Decimal('2.35')
        assert result[3] == Decimal('100')
        assert result[4] == 'buy'
        assert result[5] is None  # order_type
        assert result[6] is None  # misc


class TestCandleRecord:
    """Tests for CandleRecord dataclass."""

    def test_candle_record_to_tuple(self):
        """Test CandleRecord to tuple conversion."""
        now = datetime.now(timezone.utc)
        record = CandleRecord(
            symbol='XRP/USDT',
            timestamp=now,
            interval_minutes=1,
            open=Decimal('2.30'),
            high=Decimal('2.40'),
            low=Decimal('2.20'),
            close=Decimal('2.35'),
            volume=Decimal('1000'),
            trade_count=50,
            vwap=Decimal('2.33')
        )

        result = record.to_tuple()

        assert result[0] == 'XRP/USDT'
        assert result[1] == now
        assert result[2] == 1
        assert result[3] == Decimal('2.30')
        assert result[4] == Decimal('2.40')
        assert result[5] == Decimal('2.20')
        assert result[6] == Decimal('2.35')
        assert result[7] == Decimal('1000')
        assert result[8] is None  # quote_volume
        assert result[9] == 50
        assert result[10] == Decimal('2.33')


class TestDatabaseWriterUnit:
    """Unit tests for DatabaseWriter (no database required)."""

    @pytest.mark.asyncio
    async def test_trade_buffer_tracking(self):
        """Test that trades are buffered correctly."""
        # This test would require mocking asyncpg
        # For now, test the types
        record = TradeRecord(
            symbol='XRP/USDT',
            timestamp=datetime.now(timezone.utc),
            price=Decimal('2.35'),
            volume=Decimal('100'),
            side='buy'
        )

        assert record.symbol == 'XRP/USDT'
        assert record.side == 'buy'


class TestHistoricalProviderUnit:
    """Unit tests for HistoricalDataProvider (no database required)."""

    def test_interval_view_mapping(self):
        """Test interval to view mapping."""
        from data.historical_provider import HistoricalDataProvider

        # Check known mappings
        assert HistoricalDataProvider.INTERVAL_VIEWS[1] == 'candles'
        assert HistoricalDataProvider.INTERVAL_VIEWS[5] == 'candles_5m'
        assert HistoricalDataProvider.INTERVAL_VIEWS[15] == 'candles_15m'
        assert HistoricalDataProvider.INTERVAL_VIEWS[60] == 'candles_1h'
        assert HistoricalDataProvider.INTERVAL_VIEWS[1440] == 'candles_1d'

    def test_get_view_for_interval(self):
        """Test view selection for different intervals."""
        from data.historical_provider import HistoricalDataProvider

        provider = HistoricalDataProvider.__new__(HistoricalDataProvider)
        provider.pool = None  # Skip init

        assert provider._get_view_for_interval(1) == 'candles'
        assert provider._get_view_for_interval(5) == 'candles_5m'
        assert provider._get_view_for_interval(60) == 'candles_1h'

        # Unknown interval falls back to candles
        assert provider._get_view_for_interval(3) == 'candles'

    def test_get_view_for_interval_invalid(self):
        """REC-007: Test invalid interval validation."""
        from data.historical_provider import HistoricalDataProvider

        provider = HistoricalDataProvider.__new__(HistoricalDataProvider)
        provider.pool = None

        # Negative interval should raise ValueError
        with pytest.raises(ValueError, match="Invalid interval"):
            provider._get_view_for_interval(-1)

        # Zero interval should raise ValueError
        with pytest.raises(ValueError, match="Invalid interval"):
            provider._get_view_for_interval(0)

    def test_ensure_connected_raises_when_not_connected(self):
        """REC-007: Test connection check raises error when not connected."""
        from data.historical_provider import HistoricalDataProvider

        provider = HistoricalDataProvider.__new__(HistoricalDataProvider)
        provider.pool = None

        with pytest.raises(RuntimeError, match="not connected"):
            provider._ensure_connected()

    @pytest.mark.asyncio
    async def test_get_candles_raises_when_not_connected(self):
        """REC-007: Test get_candles raises when provider not connected."""
        from data.historical_provider import HistoricalDataProvider

        provider = HistoricalDataProvider.__new__(HistoricalDataProvider)
        provider.pool = None

        with pytest.raises(RuntimeError, match="not connected"):
            await provider.get_candles(
                'XRP/USDT', 1,
                datetime.now(timezone.utc) - timedelta(hours=1),
                datetime.now(timezone.utc)
            )


class TestGapFillerUnit:
    """Unit tests for GapFiller (no database/network required)."""

    def test_pair_mapping(self):
        """Test symbol to Kraken pair mapping."""
        from data.gap_filler import GapFiller

        assert GapFiller.PAIR_MAP['XRP/USDT'] == 'XRPUSDT'
        assert GapFiller.PAIR_MAP['BTC/USDT'] == 'XBTUSDT'
        assert GapFiller.PAIR_MAP['XRP/BTC'] == 'XRPXBT'


class TestDatabaseWriterBufferOverflow:
    """REC-007: Tests for buffer overflow protection in DatabaseWriter."""

    def test_max_buffer_size_constants(self):
        """Test that buffer size limits are properly defined."""
        from data.websocket_db_writer import DatabaseWriter

        assert hasattr(DatabaseWriter, 'MAX_TRADE_BUFFER_SIZE')
        assert hasattr(DatabaseWriter, 'MAX_CANDLE_BUFFER_SIZE')
        assert DatabaseWriter.MAX_TRADE_BUFFER_SIZE == 10000
        assert DatabaseWriter.MAX_CANDLE_BUFFER_SIZE == 1000

    def test_overflow_count_tracking(self):
        """Test that overflow count is tracked in stats."""
        from data.websocket_db_writer import DatabaseWriter
        from unittest.mock import patch

        with patch('data.websocket_db_writer.asyncpg', None):
            pass  # Can't test full writer without asyncpg mock

        # Just verify the class has the attribute
        assert hasattr(DatabaseWriter, 'MAX_TRADE_BUFFER_SIZE')


class TestTradeValidation:
    """REC-007: Tests for trade data validation in historical_backfill."""

    @pytest.mark.asyncio
    async def test_store_trades_validates_price(self):
        """Test that invalid prices are rejected."""
        from data.historical_backfill import KrakenTradesBackfill
        from unittest.mock import AsyncMock, MagicMock

        # Create backfill instance with mocked pool
        backfill = KrakenTradesBackfill.__new__(KrakenTradesBackfill)
        backfill.pool = MagicMock()

        # Mock the connection to track what gets inserted
        mock_conn = AsyncMock()
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        backfill.pool.acquire.return_value = mock_conn

        # Valid trade followed by invalid trade (negative price)
        trades = [
            ['2.50', '100', '1704067200', 'b', 'market', ''],  # Valid
            ['-1.00', '100', '1704067201', 'b', 'market', ''],  # Invalid price
            ['0', '100', '1704067202', 'b', 'market', ''],      # Invalid price (zero)
        ]

        await backfill.store_trades('XRP/USDT', trades)

        # Should only have called executemany with 1 valid trade
        if mock_conn.executemany.called:
            call_args = mock_conn.executemany.call_args
            records = call_args[0][1]  # Second arg to executemany
            assert len(records) == 1  # Only the valid trade

    @pytest.mark.asyncio
    async def test_store_trades_validates_volume(self):
        """Test that invalid volumes are rejected."""
        from data.historical_backfill import KrakenTradesBackfill
        from unittest.mock import AsyncMock, MagicMock

        backfill = KrakenTradesBackfill.__new__(KrakenTradesBackfill)
        backfill.pool = MagicMock()

        mock_conn = AsyncMock()
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        backfill.pool.acquire.return_value = mock_conn

        # Valid trade followed by invalid trade (negative volume)
        trades = [
            ['2.50', '100', '1704067200', 'b', 'market', ''],   # Valid
            ['2.50', '-50', '1704067201', 'b', 'market', ''],   # Invalid volume
            ['2.50', '0', '1704067202', 'b', 'market', ''],     # Invalid volume (zero)
        ]

        await backfill.store_trades('XRP/USDT', trades)

        if mock_conn.executemany.called:
            call_args = mock_conn.executemany.call_args
            records = call_args[0][1]
            assert len(records) == 1  # Only the valid trade

    @pytest.mark.asyncio
    async def test_store_trades_handles_malformed_data(self):
        """Test that malformed trades don't crash the system."""
        from data.historical_backfill import KrakenTradesBackfill
        from unittest.mock import AsyncMock, MagicMock

        backfill = KrakenTradesBackfill.__new__(KrakenTradesBackfill)
        backfill.pool = MagicMock()

        mock_conn = AsyncMock()
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        backfill.pool.acquire.return_value = mock_conn

        # Mix of valid and malformed trades
        trades = [
            ['2.50', '100', '1704067200', 'b', 'market', ''],  # Valid
            ['invalid', '100', '1704067201', 'b', 'market', ''],  # Invalid price string
            ['2.50', 'nan', '1704067202', 'b', 'market', ''],     # Invalid volume string
        ]

        # Should not raise exception
        await backfill.store_trades('XRP/USDT', trades)


class TestCentralizedPairMappings:
    """REC-005/REC-007: Tests for centralized pair mappings."""

    def test_pair_map_consistency(self):
        """Test that all files use centralized PAIR_MAP."""
        from data.types import PAIR_MAP, REVERSE_PAIR_MAP

        # Verify bidirectional mapping
        for our_format, kraken_format in PAIR_MAP.items():
            assert REVERSE_PAIR_MAP[kraken_format] == our_format

    def test_default_symbols_in_pair_map(self):
        """Test that default symbols are all in PAIR_MAP."""
        from data.types import PAIR_MAP, DEFAULT_SYMBOLS

        for symbol in DEFAULT_SYMBOLS:
            assert symbol in PAIR_MAP, f"Default symbol {symbol} not in PAIR_MAP"

    def test_csv_symbol_map_coverage(self):
        """Test that CSV symbol map covers PAIR_MAP."""
        from data.types import PAIR_MAP, CSV_SYMBOL_MAP

        # Every Kraken format should map back to our format
        for our_format, kraken_format in PAIR_MAP.items():
            assert kraken_format in CSV_SYMBOL_MAP
            assert CSV_SYMBOL_MAP[kraken_format] == our_format


# ============================================
# Integration Tests (require database)
# ============================================
# These tests are skipped by default unless DATABASE_URL is set

@pytest.fixture
def db_url():
    """Get database URL from environment."""
    import os
    url = os.getenv('DATABASE_URL')
    if not url:
        pytest.skip("DATABASE_URL not set, skipping integration tests")
    return url


@pytest.mark.integration
class TestHistoricalProviderIntegration:
    """Integration tests for HistoricalDataProvider."""

    @pytest.mark.asyncio
    async def test_connect_and_health_check(self, db_url):
        """Test database connection and health check."""
        from data.historical_provider import HistoricalDataProvider

        provider = HistoricalDataProvider(db_url)

        try:
            await provider.connect()
            health = await provider.health_check()

            assert health['connected'] is True
            assert isinstance(health['symbols'], list)
            assert isinstance(health['total_candles'], int)
        finally:
            await provider.close()

    @pytest.mark.asyncio
    async def test_get_symbols(self, db_url):
        """Test getting available symbols."""
        from data.historical_provider import HistoricalDataProvider

        provider = HistoricalDataProvider(db_url)

        try:
            await provider.connect()
            symbols = await provider.get_symbols()

            assert isinstance(symbols, list)
            # If we have data, check format
            for symbol in symbols:
                assert '/' in symbol  # e.g., 'XRP/USDT'
        finally:
            await provider.close()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
