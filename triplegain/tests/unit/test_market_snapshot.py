"""
Unit tests for the Market Snapshot Builder.

Tests validate:
- Snapshot structure and completeness
- Indicator inclusion
- Multi-timeframe analysis
- Data quality validation
- Prompt format conversion (full and compact)
- Performance requirements (<200ms build time)
"""

import pytest
import numpy as np
from datetime import datetime, timezone, timedelta
from decimal import Decimal
import json
import time

from triplegain.src.data.market_snapshot import (
    MarketSnapshot,
    MarketSnapshotBuilder,
    CandleSummary,
    OrderBookFeatures,
    MultiTimeframeState,
)
from triplegain.src.data.indicator_library import IndicatorLibrary


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def indicator_config() -> dict:
    """Indicator configuration for testing."""
    return {
        'ema': {'periods': [9, 21, 50, 200]},
        'sma': {'periods': [20, 50, 200]},
        'rsi': {'period': 14},
        'macd': {'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
        'atr': {'period': 14},
        'bollinger_bands': {'period': 20, 'std_dev': 2.0},
        'adx': {'period': 14},
        'choppiness': {'period': 14},
    }


@pytest.fixture
def snapshot_config() -> dict:
    """Snapshot builder configuration for testing."""
    return {
        'candle_lookback': {
            '1m': 60,
            '5m': 48,
            '15m': 32,
            '1h': 48,
            '4h': 30,
            '1d': 30,
        },
        'include_indicators': [
            'rsi_14', 'macd', 'ema_9', 'ema_21', 'ema_50', 'ema_200',
            'atr_14', 'adx_14', 'bollinger_bands', 'obv', 'choppiness_14'
        ],
        'order_book': {
            'enabled': True,
            'depth_levels': 10,
        },
        'data_quality': {
            'max_age_seconds': 60,
            'min_candles_required': 20,
        },
        'token_budgets': {
            'tier1_local': 3500,
            'tier2_api': 8000,
        }
    }


@pytest.fixture
def indicator_library(indicator_config) -> IndicatorLibrary:
    """Create indicator library instance."""
    return IndicatorLibrary(indicator_config)


@pytest.fixture
def sample_candles() -> list[dict]:
    """Generate sample candle data for testing."""
    np.random.seed(42)
    n = 100
    base_price = 45000.0
    now = datetime.now(timezone.utc)

    candles = []
    for i in range(n):
        price = base_price + np.random.randn() * 100
        candles.append({
            'timestamp': now - timedelta(hours=n - i),
            'open': price + np.random.randn() * 10,
            'high': price + abs(np.random.randn() * 50),
            'low': price - abs(np.random.randn() * 50),
            'close': price + np.random.randn() * 20,
            'volume': abs(np.random.randn() * 1000) + 500,
        })

    return candles


@pytest.fixture
def sample_order_book() -> dict:
    """Generate sample order book data."""
    return {
        'bids': [
            {'price': 45000.0, 'size': 1.5},
            {'price': 44990.0, 'size': 2.0},
            {'price': 44980.0, 'size': 3.0},
            {'price': 44970.0, 'size': 1.0},
            {'price': 44960.0, 'size': 2.5},
        ],
        'asks': [
            {'price': 45010.0, 'size': 1.2},
            {'price': 45020.0, 'size': 2.3},
            {'price': 45030.0, 'size': 1.8},
            {'price': 45040.0, 'size': 2.0},
            {'price': 45050.0, 'size': 3.0},
        ]
    }


@pytest.fixture
def mock_db_pool():
    """Mock database pool for testing."""
    class MockPool:
        async def acquire(self):
            return MockConnection()

    class MockConnection:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def fetch(self, query, *args):
            return []

    return MockPool()


# =============================================================================
# CandleSummary Tests
# =============================================================================

class TestCandleSummary:
    """Tests for CandleSummary dataclass."""

    def test_candle_summary_creation(self):
        """Test creating a CandleSummary."""
        now = datetime.now(timezone.utc)
        candle = CandleSummary(
            timestamp=now,
            open=Decimal('45000'),
            high=Decimal('45500'),
            low=Decimal('44500'),
            close=Decimal('45200'),
            volume=Decimal('1000'),
        )

        assert candle.timestamp == now
        assert candle.open == Decimal('45000')
        assert candle.close == Decimal('45200')

    def test_candle_to_compact(self):
        """Test compact representation of candle."""
        now = datetime.now(timezone.utc)
        candle = CandleSummary(
            timestamp=now,
            open=Decimal('45000'),
            high=Decimal('45500'),
            low=Decimal('44500'),
            close=Decimal('45200'),
            volume=Decimal('1000'),
        )

        compact = candle.to_compact()
        assert 't' in compact
        assert 'o' in compact
        assert 'h' in compact
        assert 'l' in compact
        assert 'c' in compact
        assert 'v' in compact


# =============================================================================
# OrderBookFeatures Tests
# =============================================================================

class TestOrderBookFeatures:
    """Tests for OrderBookFeatures dataclass."""

    def test_order_book_features_creation(self):
        """Test creating OrderBookFeatures."""
        features = OrderBookFeatures(
            bid_depth_usd=Decimal('2500000'),
            ask_depth_usd=Decimal('2200000'),
            imbalance=Decimal('0.12'),
            spread_bps=Decimal('3.5'),
            weighted_mid=Decimal('45005'),
        )

        assert features.bid_depth_usd == Decimal('2500000')
        assert features.imbalance == Decimal('0.12')

    def test_order_book_imbalance_bounds(self):
        """Test imbalance is within expected range."""
        features = OrderBookFeatures(
            bid_depth_usd=Decimal('1000000'),
            ask_depth_usd=Decimal('1000000'),
            imbalance=Decimal('0.0'),
            spread_bps=Decimal('2.0'),
            weighted_mid=Decimal('45000'),
        )

        # Imbalance should be between -1 and 1
        assert -1 <= float(features.imbalance) <= 1


# =============================================================================
# MultiTimeframeState Tests
# =============================================================================

class TestMultiTimeframeState:
    """Tests for MultiTimeframeState dataclass."""

    def test_mtf_state_creation(self):
        """Test creating MultiTimeframeState."""
        state = MultiTimeframeState(
            trend_alignment_score=Decimal('0.67'),
            aligned_bullish_count=4,
            aligned_bearish_count=1,
            total_timeframes=5,
            rsi_by_timeframe={'1h': Decimal('55'), '4h': Decimal('62')},
            atr_by_timeframe={'1h': Decimal('500'), '4h': Decimal('800')},
        )

        assert state.trend_alignment_score == Decimal('0.67')
        assert state.aligned_bullish_count == 4
        assert state.total_timeframes == 5

    def test_mtf_alignment_score_bounds(self):
        """Test alignment score is within expected range."""
        state = MultiTimeframeState(
            trend_alignment_score=Decimal('0.5'),
            aligned_bullish_count=2,
            aligned_bearish_count=2,
            total_timeframes=4,
            rsi_by_timeframe={},
            atr_by_timeframe={},
        )

        # Score should be between -1 and 1
        assert -1 <= float(state.trend_alignment_score) <= 1


# =============================================================================
# MarketSnapshot Tests
# =============================================================================

class TestMarketSnapshot:
    """Tests for MarketSnapshot dataclass."""

    def test_snapshot_creation(self):
        """Test creating a MarketSnapshot."""
        now = datetime.now(timezone.utc)
        snapshot = MarketSnapshot(
            timestamp=now,
            symbol='BTC/USDT',
            current_price=Decimal('45250.50'),
        )

        assert snapshot.symbol == 'BTC/USDT'
        assert snapshot.current_price == Decimal('45250.50')
        assert snapshot.candles == {}
        assert snapshot.indicators == {}

    def test_snapshot_with_indicators(self, indicator_library, sample_candles):
        """Test snapshot with indicators."""
        now = datetime.now(timezone.utc)
        indicators = indicator_library.calculate_all('BTC/USDT', '1h', sample_candles)

        snapshot = MarketSnapshot(
            timestamp=now,
            symbol='BTC/USDT',
            current_price=Decimal('45250.50'),
            indicators=indicators,
        )

        assert 'rsi_14' in snapshot.indicators or 'ema_9' in snapshot.indicators

    def test_to_prompt_format(self):
        """Test conversion to prompt format."""
        now = datetime.now(timezone.utc)
        snapshot = MarketSnapshot(
            timestamp=now,
            symbol='BTC/USDT',
            current_price=Decimal('45250.50'),
            indicators={'rsi_14': 62.5, 'ema_9': 45123.45},
        )

        prompt_str = snapshot.to_prompt_format(token_budget=4000)

        # Should be valid JSON
        data = json.loads(prompt_str)
        assert 'symbol' in data
        assert data['symbol'] == 'BTC/USDT'

    def test_to_compact_format(self):
        """Test conversion to compact format for Tier 1 LLM."""
        now = datetime.now(timezone.utc)
        snapshot = MarketSnapshot(
            timestamp=now,
            symbol='BTC/USDT',
            current_price=Decimal('45250.50'),
            indicators={'rsi_14': 62.5, 'macd': {'histogram': 29.7}},
        )

        compact = snapshot.to_compact_format()

        # Should be valid JSON
        data = json.loads(compact)
        assert 'sym' in data
        assert data['sym'] == 'BTC/USDT'
        # Should be compact (short keys)
        assert 'px' in data

    def test_prompt_format_token_budget(self):
        """Test that prompt fits within token budget."""
        now = datetime.now(timezone.utc)
        snapshot = MarketSnapshot(
            timestamp=now,
            symbol='BTC/USDT',
            current_price=Decimal('45250.50'),
            indicators={'rsi_14': 62.5},
        )

        prompt = snapshot.to_prompt_format(token_budget=3500)
        # Rough estimate: 3.5 chars per token
        estimated_tokens = len(prompt) / 3.5

        assert estimated_tokens < 3500

    def test_data_quality_flags(self):
        """Test data quality flag tracking."""
        now = datetime.now(timezone.utc)
        snapshot = MarketSnapshot(
            timestamp=now,
            symbol='BTC/USDT',
            current_price=Decimal('45250.50'),
            missing_data_flags=['no_order_book', 'stale_1h_candles'],
        )

        assert 'no_order_book' in snapshot.missing_data_flags


# =============================================================================
# MarketSnapshotBuilder Tests
# =============================================================================

class TestMarketSnapshotBuilder:
    """Tests for MarketSnapshotBuilder."""

    def test_builder_creation(self, indicator_library, snapshot_config):
        """Test creating a builder."""
        builder = MarketSnapshotBuilder(
            db_pool=None,
            indicator_library=indicator_library,
            config=snapshot_config,
        )

        assert builder is not None
        assert builder.indicators is indicator_library

    def test_process_order_book(self, indicator_library, snapshot_config, sample_order_book):
        """Test order book processing."""
        builder = MarketSnapshotBuilder(
            db_pool=None,
            indicator_library=indicator_library,
            config=snapshot_config,
        )

        features = builder._process_order_book(sample_order_book)

        assert features is not None
        assert features.bid_depth_usd > 0
        assert features.ask_depth_usd > 0
        assert features.spread_bps > 0

    def test_calculate_mtf_state(self, indicator_library, snapshot_config, sample_candles):
        """Test multi-timeframe state calculation."""
        builder = MarketSnapshotBuilder(
            db_pool=None,
            indicator_library=indicator_library,
            config=snapshot_config,
        )

        candles_by_tf = {
            '1h': sample_candles,
            '4h': sample_candles[:50],  # Fewer candles for higher TF
        }

        mtf_state = builder._calculate_mtf_state(candles_by_tf)

        assert mtf_state is not None
        assert -1 <= float(mtf_state.trend_alignment_score) <= 1

    def test_validate_data_quality(self, indicator_library, snapshot_config):
        """Test data quality validation."""
        builder = MarketSnapshotBuilder(
            db_pool=None,
            indicator_library=indicator_library,
            config=snapshot_config,
        )

        now = datetime.now(timezone.utc)
        snapshot = MarketSnapshot(
            timestamp=now,
            symbol='BTC/USDT',
            current_price=Decimal('45250.50'),
            data_age_seconds=30,
        )

        issues = builder._validate_data_quality(snapshot)

        assert isinstance(issues, list)

    def test_validate_stale_data(self, indicator_library, snapshot_config):
        """Test stale data detection."""
        builder = MarketSnapshotBuilder(
            db_pool=None,
            indicator_library=indicator_library,
            config=snapshot_config,
        )

        now = datetime.now(timezone.utc)
        snapshot = MarketSnapshot(
            timestamp=now,
            symbol='BTC/USDT',
            current_price=Decimal('45250.50'),
            data_age_seconds=120,  # > 60s threshold
        )

        issues = builder._validate_data_quality(snapshot)

        assert 'stale_data' in issues


# =============================================================================
# Integration Tests (without actual DB)
# =============================================================================

class TestSnapshotIntegration:
    """Integration tests for snapshot building (mocked DB)."""

    def test_build_snapshot_from_candles(self, indicator_library, snapshot_config, sample_candles):
        """Test building snapshot from provided candles."""
        builder = MarketSnapshotBuilder(
            db_pool=None,
            indicator_library=indicator_library,
            config=snapshot_config,
        )

        now = datetime.now(timezone.utc)
        snapshot = builder.build_snapshot_from_candles(
            symbol='BTC/USDT',
            candles_by_tf={'1h': sample_candles},
            order_book=None,
        )

        assert snapshot.symbol == 'BTC/USDT'
        assert snapshot.current_price is not None
        assert snapshot.indicators != {}

    def test_snapshot_serialization_roundtrip(self, indicator_library, snapshot_config, sample_candles):
        """Test snapshot can be serialized and has expected structure."""
        builder = MarketSnapshotBuilder(
            db_pool=None,
            indicator_library=indicator_library,
            config=snapshot_config,
        )

        snapshot = builder.build_snapshot_from_candles(
            symbol='BTC/USDT',
            candles_by_tf={'1h': sample_candles},
            order_book=None,
        )

        # Test full format
        full_json = snapshot.to_prompt_format()
        full_data = json.loads(full_json)

        assert full_data['symbol'] == 'BTC/USDT'
        assert 'indicators' in full_data

        # Test compact format
        compact_json = snapshot.to_compact_format()
        compact_data = json.loads(compact_json)

        assert compact_data['sym'] == 'BTC/USDT'


# =============================================================================
# Performance Tests
# =============================================================================

class TestSnapshotPerformance:
    """Performance tests for snapshot building."""

    def test_build_snapshot_latency(self, indicator_library, snapshot_config, sample_candles):
        """Test snapshot build time is under 200ms."""
        builder = MarketSnapshotBuilder(
            db_pool=None,
            indicator_library=indicator_library,
            config=snapshot_config,
        )

        start = time.time()

        for _ in range(10):
            builder.build_snapshot_from_candles(
                symbol='BTC/USDT',
                candles_by_tf={'1h': sample_candles},
                order_book=None,
            )

        elapsed = (time.time() - start) * 1000 / 10  # Average per build

        assert elapsed < 200, f"Build took {elapsed:.2f}ms, expected <200ms"

    def test_compact_format_latency(self, indicator_library, snapshot_config, sample_candles):
        """Test compact format generation is fast."""
        builder = MarketSnapshotBuilder(
            db_pool=None,
            indicator_library=indicator_library,
            config=snapshot_config,
        )

        snapshot = builder.build_snapshot_from_candles(
            symbol='BTC/USDT',
            candles_by_tf={'1h': sample_candles},
            order_book=None,
        )

        start = time.time()
        for _ in range(100):
            snapshot.to_compact_format()
        elapsed = (time.time() - start) * 1000 / 100

        assert elapsed < 5, f"Compact format took {elapsed:.2f}ms, expected <5ms"
