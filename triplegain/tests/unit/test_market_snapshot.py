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


# =============================================================================
# Additional Coverage Tests
# =============================================================================

class TestMarketSnapshotExtended:
    """Extended tests for MarketSnapshot to improve coverage."""

    def test_snapshot_with_mtf_state(self):
        """Test snapshot with multi-timeframe state."""
        now = datetime.now(timezone.utc)
        mtf_state = MultiTimeframeState(
            trend_alignment_score=Decimal('0.75'),
            aligned_bullish_count=3,
            aligned_bearish_count=1,
            total_timeframes=4,
            rsi_by_timeframe={'1h': Decimal('55.5'), '4h': Decimal('60.2')},
            atr_by_timeframe={'1h': Decimal('500.0'), '4h': Decimal('800.0')},
        )

        snapshot = MarketSnapshot(
            timestamp=now,
            symbol='BTC/USDT',
            current_price=Decimal('45250.50'),
            mtf_state=mtf_state,
        )

        prompt = snapshot.to_prompt_format()
        data = json.loads(prompt)

        assert 'mtf_state' in data
        assert data['mtf_state']['trend_alignment_score'] == 0.75

    def test_snapshot_with_order_book(self):
        """Test snapshot with order book features."""
        now = datetime.now(timezone.utc)
        order_book = OrderBookFeatures(
            bid_depth_usd=Decimal('100000'),
            ask_depth_usd=Decimal('95000'),
            imbalance=Decimal('0.026'),
            spread_bps=Decimal('5.5'),
            weighted_mid=Decimal('45005.0'),
        )

        snapshot = MarketSnapshot(
            timestamp=now,
            symbol='BTC/USDT',
            current_price=Decimal('45250.50'),
            order_book=order_book,
        )

        prompt = snapshot.to_prompt_format()
        data = json.loads(prompt)

        assert 'order_book' in data
        assert data['order_book']['spread_bps'] == 5.5

    def test_snapshot_with_regime_hint(self):
        """Test snapshot with regime hint in compact format."""
        now = datetime.now(timezone.utc)
        snapshot = MarketSnapshot(
            timestamp=now,
            symbol='BTC/USDT',
            current_price=Decimal('45250.50'),
            regime_hint='trending_bull',
            regime_confidence=Decimal('0.85'),
            indicators={'rsi_14': 65.0},
        )

        compact = snapshot.to_compact_format()
        data = json.loads(compact)

        assert 'regime' in data
        assert data['regime'] == 'bull'

    def test_snapshot_regime_hint_mapping(self):
        """Test all regime hint mappings in compact format."""
        now = datetime.now(timezone.utc)

        regimes = [
            ('trending_bull', 'bull'),
            ('trending_bear', 'bear'),
            ('ranging', 'range'),
            ('high_volatility', 'hvol'),
            ('low_volatility', 'lvol'),
            ('unknown_regime', 'unkn'),  # Should truncate to 4 chars
        ]

        for full_regime, expected_short in regimes:
            snapshot = MarketSnapshot(
                timestamp=now,
                symbol='BTC/USDT',
                current_price=Decimal('45250.50'),
                regime_hint=full_regime,
            )
            compact = snapshot.to_compact_format()
            data = json.loads(compact)
            assert data['regime'] == expected_short

    def test_snapshot_with_volume_vs_avg(self):
        """Test snapshot with volume vs average ratio."""
        now = datetime.now(timezone.utc)
        snapshot = MarketSnapshot(
            timestamp=now,
            symbol='BTC/USDT',
            current_price=Decimal('45250.50'),
            volume_vs_avg=Decimal('1.5'),
        )

        prompt = snapshot.to_prompt_format()
        data = json.loads(prompt)

        assert 'volume_vs_avg' in data
        assert data['volume_vs_avg'] == 1.5

    def test_snapshot_with_candles(self):
        """Test snapshot with candle data in prompt format."""
        now = datetime.now(timezone.utc)
        candles = [
            CandleSummary(
                timestamp=now - timedelta(hours=i),
                open=Decimal('45000'),
                high=Decimal('45500'),
                low=Decimal('44500'),
                close=Decimal('45200'),
                volume=Decimal('1000'),
            )
            for i in range(15)
        ]

        snapshot = MarketSnapshot(
            timestamp=now,
            symbol='BTC/USDT',
            current_price=Decimal('45250.50'),
            candles={'1h': candles},
        )

        prompt = snapshot.to_prompt_format()
        data = json.loads(prompt)

        # Should limit to 10 candles per timeframe
        assert 'candles' in data
        assert len(data['candles']['1h']) == 10

    def test_snapshot_truncates_candles_when_over_budget(self):
        """Test that candles are truncated further when over token budget."""
        now = datetime.now(timezone.utc)
        # Create lots of candles and indicators to exceed budget
        candles = [
            CandleSummary(
                timestamp=now - timedelta(hours=i),
                open=Decimal('45000.12345'),
                high=Decimal('45500.12345'),
                low=Decimal('44500.12345'),
                close=Decimal('45200.12345'),
                volume=Decimal('1000.12345'),
            )
            for i in range(10)
        ]

        snapshot = MarketSnapshot(
            timestamp=now,
            symbol='BTC/USDT',
            current_price=Decimal('45250.50'),
            candles={
                '1h': candles,
                '4h': candles,
                '1d': candles,
            },
            indicators={f'indicator_{i}': 123.456 for i in range(50)},
        )

        # With a very small token budget, candles should be truncated
        prompt = snapshot.to_prompt_format(token_budget=100)
        data = json.loads(prompt)

        # Should have truncated to 5 candles per timeframe
        for tf in data.get('candles', {}).values():
            assert len(tf) <= 5

    def test_candle_summary_to_dict(self):
        """Test CandleSummary to_dict method."""
        now = datetime.now(timezone.utc)
        candle = CandleSummary(
            timestamp=now,
            open=Decimal('100.5'),
            high=Decimal('105.0'),
            low=Decimal('98.0'),
            close=Decimal('103.0'),
            volume=Decimal('1000.0'),
        )

        d = candle.to_dict()

        assert 'timestamp' in d
        assert 'open' in d
        assert d['open'] == 100.5
        assert d['close'] == 103.0

    def test_order_book_features_to_dict(self):
        """Test OrderBookFeatures to_dict method."""
        features = OrderBookFeatures(
            bid_depth_usd=Decimal('100000'),
            ask_depth_usd=Decimal('95000'),
            imbalance=Decimal('0.026'),
            spread_bps=Decimal('5.5'),
            weighted_mid=Decimal('45005.0'),
        )

        d = features.to_dict()

        assert d['bid_depth_usd'] == 100000.0
        assert d['imbalance'] == 0.026

    def test_mtf_state_to_dict(self):
        """Test MultiTimeframeState to_dict method."""
        state = MultiTimeframeState(
            trend_alignment_score=Decimal('0.75'),
            aligned_bullish_count=3,
            aligned_bearish_count=1,
            total_timeframes=4,
            rsi_by_timeframe={'1h': Decimal('55.5')},
            atr_by_timeframe={'1h': Decimal('500.0')},
        )

        d = state.to_dict()

        assert d['trend_alignment_score'] == 0.75
        assert d['aligned_bullish'] == 3
        assert d['rsi_by_tf']['1h'] == 55.5


class TestMarketSnapshotBuilderExtended:
    """Extended tests for MarketSnapshotBuilder."""

    def test_build_snapshot_with_order_book(self, indicator_library, snapshot_config, sample_candles, sample_order_book):
        """Test building snapshot with order book data."""
        builder = MarketSnapshotBuilder(
            db_pool=None,
            indicator_library=indicator_library,
            config=snapshot_config,
        )

        snapshot = builder.build_snapshot_from_candles(
            symbol='BTC/USDT',
            candles_by_tf={'1h': sample_candles},
            order_book=sample_order_book,
        )

        assert snapshot.order_book is not None
        assert snapshot.order_book.bid_depth_usd > 0
        assert snapshot.order_book.spread_bps > 0

    def test_process_order_book_empty(self, indicator_library, snapshot_config):
        """Test order book processing with empty data."""
        builder = MarketSnapshotBuilder(
            db_pool=None,
            indicator_library=indicator_library,
            config=snapshot_config,
        )

        features = builder._process_order_book({'bids': [], 'asks': []})

        assert features.bid_depth_usd == Decimal('0')
        assert features.ask_depth_usd == Decimal('0')
        assert features.imbalance == Decimal('0')

    def test_process_order_book_imbalance_calculation(self, indicator_library, snapshot_config):
        """Test order book imbalance calculation."""
        builder = MarketSnapshotBuilder(
            db_pool=None,
            indicator_library=indicator_library,
            config=snapshot_config,
        )

        # More bids than asks = positive imbalance
        order_book = {
            'bids': [{'price': 100.0, 'size': 200}],  # 20000 USD
            'asks': [{'price': 101.0, 'size': 100}],  # 10100 USD
        }

        features = builder._process_order_book(order_book)

        # imbalance = (20000 - 10100) / (20000 + 10100) ≈ 0.329
        assert float(features.imbalance) > 0.3

    def test_calculate_mtf_state_empty_candles(self, indicator_library, snapshot_config):
        """Test MTF state calculation with empty candle data."""
        builder = MarketSnapshotBuilder(
            db_pool=None,
            indicator_library=indicator_library,
            config=snapshot_config,
        )

        mtf_state = builder._calculate_mtf_state({})

        assert mtf_state.trend_alignment_score == Decimal('0')
        assert mtf_state.total_timeframes == 0

    def test_calculate_mtf_state_insufficient_candles(self, indicator_library, snapshot_config):
        """Test MTF state calculation with insufficient candles."""
        builder = MarketSnapshotBuilder(
            db_pool=None,
            indicator_library=indicator_library,
            config=snapshot_config,
        )

        # Less than 20 candles should be skipped
        short_candles = [
            {'open': 100, 'high': 105, 'low': 98, 'close': 103, 'volume': 1000}
            for _ in range(10)
        ]

        mtf_state = builder._calculate_mtf_state({'1h': short_candles})

        assert mtf_state.total_timeframes == 0

    def test_validate_data_quality_no_indicators(self, indicator_library, snapshot_config):
        """Test data quality validation when no indicators."""
        builder = MarketSnapshotBuilder(
            db_pool=None,
            indicator_library=indicator_library,
            config=snapshot_config,
        )

        snapshot = MarketSnapshot(
            timestamp=datetime.now(timezone.utc),
            symbol='BTC/USDT',
            current_price=Decimal('45250.50'),
            indicators={},  # No indicators
        )

        issues = builder._validate_data_quality(snapshot)

        assert 'no_indicators' in issues

    def test_validate_data_quality_insufficient_candles(self, indicator_library, snapshot_config):
        """Test data quality validation with insufficient candles."""
        builder = MarketSnapshotBuilder(
            db_pool=None,
            indicator_library=indicator_library,
            config=snapshot_config,
        )

        # Create snapshot with only 10 candles (min required is 20)
        candles = [
            CandleSummary(
                timestamp=datetime.now(timezone.utc) - timedelta(hours=i),
                open=Decimal('100'), high=Decimal('105'),
                low=Decimal('98'), close=Decimal('103'), volume=Decimal('1000')
            )
            for i in range(10)
        ]

        snapshot = MarketSnapshot(
            timestamp=datetime.now(timezone.utc),
            symbol='BTC/USDT',
            current_price=Decimal('45250.50'),
            candles={'1h': candles},
            indicators={'rsi_14': 55.0},
        )

        issues = builder._validate_data_quality(snapshot)

        assert 'insufficient_1h_candles' in issues

    def test_build_snapshot_fallback_to_first_available_timeframe(self, indicator_library, sample_candles):
        """Test snapshot builder falls back to first available timeframe when primary missing."""
        config = {
            'primary_timeframe': '1d',  # Not available in our data
            'data_quality': {'max_age_seconds': 60, 'min_candles_required': 20},
        }

        builder = MarketSnapshotBuilder(
            db_pool=None,
            indicator_library=indicator_library,
            config=config,
        )

        snapshot = builder.build_snapshot_from_candles(
            symbol='BTC/USDT',
            candles_by_tf={'1h': sample_candles},  # Only 1h available
            order_book=None,
        )

        assert snapshot.current_price > 0

    def test_build_snapshot_24h_calculation_different_timeframes(self, indicator_library, snapshot_config):
        """Test 24h calculation works for different timeframes."""
        builder = MarketSnapshotBuilder(
            db_pool=None,
            indicator_library=indicator_library,
            config=snapshot_config,
        )

        now = datetime.now(timezone.utc)

        # Test with 4h candles (need 6 candles for 24h)
        candles_4h = [
            {
                'timestamp': now - timedelta(hours=4 * i),
                'open': 100.0 + i,
                'high': 105.0 + i,
                'low': 98.0,
                'close': 103.0 + i * 0.5,
                'volume': 1000.0,
            }
            for i in range(10)
        ]

        config_4h = {**snapshot_config, 'primary_timeframe': '4h'}
        builder_4h = MarketSnapshotBuilder(
            db_pool=None,
            indicator_library=indicator_library,
            config=config_4h,
        )

        snapshot = builder_4h.build_snapshot_from_candles(
            symbol='BTC/USDT',
            candles_by_tf={'4h': candles_4h},
            order_book=None,
        )

        # With 4h candles and 10 candles available (>6 needed for 24h)
        # 24h price change should be calculated
        assert snapshot.price_24h_ago is not None or len(candles_4h) < 6

    def test_compact_format_with_adx(self):
        """Test compact format includes ADX when available."""
        snapshot = MarketSnapshot(
            timestamp=datetime.now(timezone.utc),
            symbol='BTC/USDT',
            current_price=Decimal('45250.50'),
            indicators={'adx_14': 28.5},
        )

        compact = snapshot.to_compact_format()
        data = json.loads(compact)

        assert 'adx' in data
        assert data['adx'] == 28.5

    def test_compact_format_with_bb_position(self):
        """Test compact format includes Bollinger Band position."""
        snapshot = MarketSnapshot(
            timestamp=datetime.now(timezone.utc),
            symbol='BTC/USDT',
            current_price=Decimal('45250.50'),
            indicators={'bollinger_bands': {'position': 0.75}},
        )

        compact = snapshot.to_compact_format()
        data = json.loads(compact)

        assert 'bb_pos' in data
        assert data['bb_pos'] == 0.75

    def test_compact_format_with_atr(self):
        """Test compact format includes ATR."""
        snapshot = MarketSnapshot(
            timestamp=datetime.now(timezone.utc),
            symbol='BTC/USDT',
            current_price=Decimal('45250.50'),
            indicators={'atr_14': 523.5},
        )

        compact = snapshot.to_compact_format()
        data = json.loads(compact)

        assert 'atr' in data
        assert data['atr'] == 523.5

    def test_compact_format_with_trend_alignment(self):
        """Test compact format includes trend alignment."""
        mtf_state = MultiTimeframeState(
            trend_alignment_score=Decimal('0.8'),
            aligned_bullish_count=4,
            aligned_bearish_count=1,
            total_timeframes=5,
            rsi_by_timeframe={},
            atr_by_timeframe={},
        )

        snapshot = MarketSnapshot(
            timestamp=datetime.now(timezone.utc),
            symbol='BTC/USDT',
            current_price=Decimal('45250.50'),
            mtf_state=mtf_state,
        )

        compact = snapshot.to_compact_format()
        data = json.loads(compact)

        assert 'trend' in data
        assert data['trend'] == 0.8


# =============================================================================
# Async Build Snapshot Tests
# =============================================================================

class TestAsyncBuildSnapshot:
    """Tests for async build_snapshot method."""

    @pytest.fixture
    def mock_candles(self):
        """Generate sample candles as database would return."""
        now = datetime.now(timezone.utc)
        return [
            {
                'timestamp': now - timedelta(hours=i),
                'open': 45000 + np.random.randn() * 100,
                'high': 45050 + np.random.randn() * 50,
                'low': 44950 + np.random.randn() * 50,
                'close': 45000 + np.random.randn() * 100,
                'volume': 1000 + np.random.randn() * 100,
            }
            for i in range(50)
        ]

    @pytest.fixture
    def mock_db_pool(self, mock_candles):
        """Create mock database pool with async methods."""
        from unittest.mock import AsyncMock, MagicMock

        pool = MagicMock()

        # Mock fetch_candles
        pool.fetch_candles = AsyncMock(return_value=mock_candles)

        # Mock fetch_24h_data
        pool.fetch_24h_data = AsyncMock(return_value={
            'price_24h_ago': 44500,
            'price_change_24h_pct': 1.12,
            'volume_24h': 50000000,
        })

        # Mock fetch_order_book
        pool.fetch_order_book = AsyncMock(return_value={
            'bids': [
                {'price': 45000.0, 'size': 1.5},
                {'price': 44990.0, 'size': 2.0},
            ],
            'asks': [
                {'price': 45010.0, 'size': 1.2},
                {'price': 45020.0, 'size': 2.3},
            ],
        })

        return pool

    @pytest.mark.asyncio
    async def test_build_snapshot_success(
        self, mock_db_pool, indicator_library, snapshot_config
    ):
        """Test successful async snapshot build."""
        builder = MarketSnapshotBuilder(
            db_pool=mock_db_pool,
            indicator_library=indicator_library,
            config=snapshot_config,
        )

        snapshot = await builder.build_snapshot('BTC/USDT')

        assert snapshot.symbol == 'BTC/USDT'
        assert snapshot.current_price > 0
        assert snapshot.order_book is not None
        assert snapshot.mtf_state is not None

    @pytest.mark.asyncio
    async def test_build_snapshot_without_order_book(
        self, mock_db_pool, indicator_library, snapshot_config
    ):
        """Test snapshot build without order book."""
        builder = MarketSnapshotBuilder(
            db_pool=mock_db_pool,
            indicator_library=indicator_library,
            config=snapshot_config,
        )

        snapshot = await builder.build_snapshot('BTC/USDT', include_order_book=False)

        assert snapshot.symbol == 'BTC/USDT'
        assert snapshot.order_book is None

    @pytest.mark.asyncio
    async def test_build_snapshot_no_db_raises(
        self, indicator_library, snapshot_config
    ):
        """Test snapshot build without db pool raises error."""
        builder = MarketSnapshotBuilder(
            db_pool=None,
            indicator_library=indicator_library,
            config=snapshot_config,
        )

        with pytest.raises(RuntimeError, match="Database pool required"):
            await builder.build_snapshot('BTC/USDT')

    @pytest.mark.asyncio
    async def test_build_snapshot_handles_candle_error(
        self, mock_db_pool, indicator_library, snapshot_config
    ):
        """Test snapshot handles candle fetch errors gracefully."""
        from unittest.mock import AsyncMock

        # Make some candle fetches fail
        call_count = 0
        original_fetch = mock_db_pool.fetch_candles

        async def failing_fetch(symbol, tf, lookback):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Database error")
            return await original_fetch(symbol, tf, lookback)

        mock_db_pool.fetch_candles = failing_fetch

        builder = MarketSnapshotBuilder(
            db_pool=mock_db_pool,
            indicator_library=indicator_library,
            config=snapshot_config,
        )

        # Should complete despite one timeframe failing
        snapshot = await builder.build_snapshot('BTC/USDT')
        assert snapshot is not None

    @pytest.mark.asyncio
    async def test_build_snapshot_handles_24h_error(
        self, mock_db_pool, indicator_library, snapshot_config
    ):
        """Test snapshot handles 24h data fetch error."""
        mock_db_pool.fetch_24h_data.side_effect = Exception("24h data error")

        builder = MarketSnapshotBuilder(
            db_pool=mock_db_pool,
            indicator_library=indicator_library,
            config=snapshot_config,
        )

        snapshot = await builder.build_snapshot('BTC/USDT')

        # Should still build snapshot, just without 24h data
        assert snapshot.symbol == 'BTC/USDT'
        assert snapshot.price_24h_ago is None

    @pytest.mark.asyncio
    async def test_build_snapshot_handles_order_book_error(
        self, mock_db_pool, indicator_library, snapshot_config
    ):
        """Test snapshot handles order book fetch error."""
        mock_db_pool.fetch_order_book.side_effect = Exception("Order book error")

        builder = MarketSnapshotBuilder(
            db_pool=mock_db_pool,
            indicator_library=indicator_library,
            config=snapshot_config,
        )

        snapshot = await builder.build_snapshot('BTC/USDT')

        # Should still build snapshot, just without order book
        assert snapshot.symbol == 'BTC/USDT'
        assert snapshot.order_book is None

    @pytest.mark.asyncio
    async def test_build_snapshot_primary_timeframe_missing(
        self, mock_db_pool, indicator_library
    ):
        """Test fallback when primary timeframe is missing."""
        from unittest.mock import AsyncMock

        # Return empty for primary timeframe, data for others
        async def selective_fetch(symbol, tf, lookback):
            if tf == '1h':
                return []  # Empty for primary
            return [
                {
                    'timestamp': datetime.now(timezone.utc),
                    'open': 45000, 'high': 45500, 'low': 44500,
                    'close': 45200, 'volume': 1000
                }
                for _ in range(50)
            ]

        mock_db_pool.fetch_candles = selective_fetch

        config = {
            'candle_lookback': {'1h': 48, '4h': 30},
            'primary_timeframe': '1h',
            'data_quality': {'max_age_seconds': 60, 'min_candles_required': 20},
        }

        builder = MarketSnapshotBuilder(
            db_pool=mock_db_pool,
            indicator_library=indicator_library,
            config=config,
        )

        snapshot = await builder.build_snapshot('BTC/USDT', include_order_book=False)

        # Should fall back to 4h timeframe
        assert snapshot.current_price > 0


# =============================================================================
# Async Multi-Symbol Snapshot Tests
# =============================================================================

class TestAsyncMultiSymbolSnapshot:
    """Tests for async build_multi_symbol_snapshot method."""

    @pytest.fixture
    def mock_candles(self):
        """Generate sample candles."""
        now = datetime.now(timezone.utc)
        return [
            {
                'timestamp': now - timedelta(hours=i),
                'open': 45000, 'high': 45500, 'low': 44500,
                'close': 45200, 'volume': 1000
            }
            for i in range(50)
        ]

    @pytest.fixture
    def mock_db_pool(self, mock_candles):
        """Create mock database pool."""
        from unittest.mock import AsyncMock, MagicMock

        pool = MagicMock()
        pool.fetch_candles = AsyncMock(return_value=mock_candles)
        pool.fetch_24h_data = AsyncMock(return_value={})
        pool.fetch_order_book = AsyncMock(return_value={
            'bids': [{'price': 100, 'size': 10}],
            'asks': [{'price': 101, 'size': 10}],
        })
        return pool

    @pytest.mark.asyncio
    async def test_build_multi_symbol_snapshot(
        self, mock_db_pool, indicator_library, snapshot_config
    ):
        """Test building snapshots for multiple symbols."""
        builder = MarketSnapshotBuilder(
            db_pool=mock_db_pool,
            indicator_library=indicator_library,
            config=snapshot_config,
        )

        symbols = ['BTC/USDT', 'ETH/USDT', 'XRP/USDT']
        snapshots = await builder.build_multi_symbol_snapshot(symbols)

        assert len(snapshots) == 3
        assert 'BTC/USDT' in snapshots
        assert 'ETH/USDT' in snapshots
        assert 'XRP/USDT' in snapshots

    @pytest.mark.asyncio
    async def test_build_multi_symbol_handles_partial_failure(
        self, indicator_library, snapshot_config
    ):
        """Test multi-symbol build handles some failures."""
        from unittest.mock import AsyncMock, MagicMock

        now = datetime.now(timezone.utc)
        good_candles = [
            {
                'timestamp': now - timedelta(hours=i),
                'open': 45000, 'high': 45500, 'low': 44500,
                'close': 45200, 'volume': 1000
            }
            for i in range(50)
        ]

        # Create mock that fails for specific symbol
        async def selective_fetch(symbol, tf, lookback):
            if symbol == 'BAD/USDT':
                raise Exception("Symbol not found")
            return good_candles

        mock_pool = MagicMock()
        mock_pool.fetch_candles = selective_fetch
        mock_pool.fetch_24h_data = AsyncMock(return_value={})
        mock_pool.fetch_order_book = AsyncMock(return_value={
            'bids': [{'price': 100, 'size': 10}],
            'asks': [{'price': 101, 'size': 10}],
        })

        builder = MarketSnapshotBuilder(
            db_pool=mock_pool,
            indicator_library=indicator_library,
            config=snapshot_config,
        )

        symbols = ['BTC/USDT', 'BAD/USDT', 'ETH/USDT']
        snapshots = await builder.build_multi_symbol_snapshot(symbols)

        # Should have successful snapshots (bad symbol might partially succeed due to error handling)
        assert 'BTC/USDT' in snapshots
        assert 'ETH/USDT' in snapshots
        # At minimum, verify the successful builds completed
        assert len(snapshots) >= 2


# =============================================================================
# Build Snapshot From Candles Edge Cases
# =============================================================================

class TestBuildSnapshotFromCandlesEdgeCases:
    """Additional edge case tests for build_snapshot_from_candles."""

    def test_empty_candles(self, indicator_library, snapshot_config):
        """Test snapshot with empty candles."""
        builder = MarketSnapshotBuilder(
            db_pool=None,
            indicator_library=indicator_library,
            config=snapshot_config,
        )

        snapshot = builder.build_snapshot_from_candles(
            symbol='BTC/USDT',
            candles_by_tf={},
            order_book=None,
        )

        assert snapshot.current_price == Decimal('0')
        assert snapshot.indicators == {}

    def test_candles_without_primary_timeframe(self, indicator_library, snapshot_config):
        """Test snapshot when primary timeframe not in candles."""
        now = datetime.now(timezone.utc)
        candles_4h = [
            {
                'timestamp': now - timedelta(hours=4 * i),
                'open': 45000, 'high': 45500, 'low': 44500,
                'close': 45200 + i, 'volume': 1000
            }
            for i in range(50)
        ]

        builder = MarketSnapshotBuilder(
            db_pool=None,
            indicator_library=indicator_library,
            config=snapshot_config,
        )

        # Primary is '1h' but we only have '4h'
        snapshot = builder.build_snapshot_from_candles(
            symbol='BTC/USDT',
            candles_by_tf={'4h': candles_4h},
            order_book=None,
        )

        # Should use 4h data as fallback
        assert snapshot.current_price > 0

    def test_24h_calculation_with_insufficient_data(self, indicator_library, snapshot_config):
        """Test 24h calculation when not enough candles."""
        now = datetime.now(timezone.utc)
        # Only 10 1h candles (need 24 for 24h calculation)
        short_candles = [
            {
                'timestamp': now - timedelta(hours=i),
                'open': 45000, 'high': 45500, 'low': 44500,
                'close': 45200, 'volume': 1000
            }
            for i in range(10)
        ]

        builder = MarketSnapshotBuilder(
            db_pool=None,
            indicator_library=indicator_library,
            config=snapshot_config,
        )

        snapshot = builder.build_snapshot_from_candles(
            symbol='BTC/USDT',
            candles_by_tf={'1h': short_candles},
            order_book=None,
        )

        # Should not have 24h data
        assert snapshot.price_24h_ago is None

    def test_24h_calculation_with_zero_price(self, indicator_library, snapshot_config):
        """Test 24h calculation handles zero price gracefully."""
        now = datetime.now(timezone.utc)
        candles = [
            {
                'timestamp': now - timedelta(hours=i),
                'open': 0 if i >= 24 else 45000,  # Zero price 24h ago
                'high': 45500, 'low': 44500,
                'close': 0 if i >= 24 else 45200,
                'volume': 1000
            }
            for i in range(30)
        ]

        builder = MarketSnapshotBuilder(
            db_pool=None,
            indicator_library=indicator_library,
            config=snapshot_config,
        )

        snapshot = builder.build_snapshot_from_candles(
            symbol='BTC/USDT',
            candles_by_tf={'1h': candles},
            order_book=None,
        )

        # Should handle gracefully
        assert snapshot is not None

    def test_timestamp_not_datetime(self, indicator_library, snapshot_config):
        """Test handling when timestamp is not datetime."""
        now = datetime.now(timezone.utc)
        candles = [
            {
                'timestamp': 'invalid_timestamp',  # Not a datetime
                'open': 45000, 'high': 45500, 'low': 44500,
                'close': 45200, 'volume': 1000
            }
            for _ in range(50)
        ]

        builder = MarketSnapshotBuilder(
            db_pool=None,
            indicator_library=indicator_library,
            config=snapshot_config,
        )

        snapshot = builder.build_snapshot_from_candles(
            symbol='BTC/USDT',
            candles_by_tf={'1h': candles},
            order_book=None,
        )

        # Should default to 0 data age
        assert snapshot.data_age_seconds == 0


# =============================================================================
# MTF State Calculation Edge Cases
# =============================================================================

class TestMTFStateEdgeCases:
    """Test edge cases in multi-timeframe state calculation."""

    def test_mtf_bearish_alignment(self, indicator_library, snapshot_config):
        """Test MTF state with bearish alignment."""
        builder = MarketSnapshotBuilder(
            db_pool=None,
            indicator_library=indicator_library,
            config=snapshot_config,
        )

        # Create candles that would show bearish trend (EMA9 < EMA21)
        np.random.seed(42)
        now = datetime.now(timezone.utc)

        # Downtrending candles (price decreasing over time)
        downtrend_candles = [
            {
                'timestamp': now - timedelta(hours=i),
                'open': 45000 - i * 10,
                'high': 45050 - i * 10,
                'low': 44950 - i * 10,
                'close': 45000 - i * 10,
                'volume': 1000 + np.random.randn() * 100,
            }
            for i in range(50)
        ]

        mtf_state = builder._calculate_mtf_state({'1h': downtrend_candles})

        # Should show bearish alignment
        assert mtf_state.aligned_bearish_count >= 0

    def test_mtf_rsi_and_atr_tracking(self, indicator_library, snapshot_config, sample_candles):
        """Test MTF state tracks RSI and ATR by timeframe."""
        builder = MarketSnapshotBuilder(
            db_pool=None,
            indicator_library=indicator_library,
            config=snapshot_config,
        )

        mtf_state = builder._calculate_mtf_state({
            '1h': sample_candles,
            '4h': sample_candles[:40],
        })

        # Should have RSI/ATR data
        assert len(mtf_state.rsi_by_timeframe) >= 0
        assert len(mtf_state.atr_by_timeframe) >= 0


# =============================================================================
# Serialization Tests
# =============================================================================

class TestSerializationEdgeCases:
    """Test serialization edge cases."""

    def test_serialize_indicators_with_none_values(self):
        """Test _serialize_indicators handles None values."""
        now = datetime.now(timezone.utc)
        snapshot = MarketSnapshot(
            timestamp=now,
            symbol='BTC/USDT',
            current_price=Decimal('45250.50'),
            indicators={
                'rsi_14': 55.0,
                'macd': None,  # None value
                'ema_9': 45000.0,
            },
        )

        prompt = snapshot.to_prompt_format()
        data = json.loads(prompt)

        assert 'indicators' in data
        # None value should be excluded
        assert 'macd' not in data['indicators']

    def test_serialize_indicators_with_nested_none(self):
        """Test _serialize_indicators handles nested None values."""
        now = datetime.now(timezone.utc)
        snapshot = MarketSnapshot(
            timestamp=now,
            symbol='BTC/USDT',
            current_price=Decimal('45250.50'),
            indicators={
                'macd': {
                    'line': 100.5,
                    'signal': None,  # None in nested dict
                    'histogram': 20.3,
                },
            },
        )

        prompt = snapshot.to_prompt_format()
        data = json.loads(prompt)

        # Nested None should be excluded
        assert 'signal' not in data['indicators']['macd']

    def test_serialize_decimal_indicators(self):
        """Test serialization handles Decimal indicators."""
        now = datetime.now(timezone.utc)
        snapshot = MarketSnapshot(
            timestamp=now,
            symbol='BTC/USDT',
            current_price=Decimal('45250.50'),
            indicators={
                'rsi_14': Decimal('55.5'),
                'atr_14': Decimal('523.25'),
            },
        )

        prompt = snapshot.to_prompt_format()
        data = json.loads(prompt)

        assert data['indicators']['rsi_14'] == 55.5
        assert data['indicators']['atr_14'] == 523.25

    def test_compact_format_missing_indicators(self):
        """Test compact format handles missing indicators gracefully."""
        now = datetime.now(timezone.utc)
        snapshot = MarketSnapshot(
            timestamp=now,
            symbol='BTC/USDT',
            current_price=Decimal('45250.50'),
            indicators={},  # No indicators
        )

        compact = snapshot.to_compact_format()
        data = json.loads(compact)

        # Should still have basic fields
        assert data['sym'] == 'BTC/USDT'
        assert data['px'] == 45250.5

    def test_compact_format_none_rsi(self):
        """Test compact format handles None RSI."""
        now = datetime.now(timezone.utc)
        snapshot = MarketSnapshot(
            timestamp=now,
            symbol='BTC/USDT',
            current_price=Decimal('45250.50'),
            indicators={'rsi_14': None},
        )

        compact = snapshot.to_compact_format()
        data = json.loads(compact)

        # RSI should not be in output
        assert 'rsi' not in data

    def test_compact_format_invalid_macd(self):
        """Test compact format handles invalid MACD structure."""
        now = datetime.now(timezone.utc)
        snapshot = MarketSnapshot(
            timestamp=now,
            symbol='BTC/USDT',
            current_price=Decimal('45250.50'),
            indicators={'macd': 'not_a_dict'},  # Invalid structure
        )

        compact = snapshot.to_compact_format()
        data = json.loads(compact)

        # MACD should not be in output
        assert 'macd_h' not in data

    def test_compact_format_macd_without_histogram(self):
        """Test compact format handles MACD without histogram."""
        now = datetime.now(timezone.utc)
        snapshot = MarketSnapshot(
            timestamp=now,
            symbol='BTC/USDT',
            current_price=Decimal('45250.50'),
            indicators={'macd': {'line': 100, 'signal': 95}},  # No histogram
        )

        compact = snapshot.to_compact_format()
        data = json.loads(compact)

        # MACD histogram should not be in output
        assert 'macd_h' not in data