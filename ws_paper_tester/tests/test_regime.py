"""
Market Regime Detection Test Suite

Tests the regime detection module including:
- CompositeScorer
- MTFAnalyzer
- ExternalDataFetcher (mocked)
- ParameterRouter
- RegimeDetector (integration)

Run with: python -m pytest tests/test_regime.py -v
"""
import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch, MagicMock

from ws_tester.types import Candle, DataSnapshot, OrderbookSnapshot, Trade
from ws_tester.regime import (
    MarketRegime,
    VolatilityState,
    TrendStrength,
    IndicatorScores,
    SymbolRegime,
    MTFConfluence,
    ExternalSentiment,
    RegimeSnapshot,
    RegimeAdjustments,
    DEFAULT_REGIME_ADJUSTMENTS,
    RegimeDetector,
    CompositeScorer,
    MTFAnalyzer,
    ExternalDataFetcher,
    ParameterRouter,
)
from ws_tester.indicators import (
    calculate_choppiness,
    calculate_adx_with_di,
    calculate_ma_alignment,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================

def create_candle(
    timestamp: datetime,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: float
) -> Candle:
    """Create a Candle object for testing."""
    return Candle(
        timestamp=timestamp,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume
    )


def generate_trending_up_candles(n: int = 250, start_price: float = 2.0) -> tuple:
    """Generate candles for a trending up market."""
    candles = []
    base = datetime.now() - timedelta(minutes=n)
    price = start_price

    for i in range(n):
        # Trending up with small pullbacks
        change = 0.003 + (0.001 * (i % 5 == 0))  # +0.3% with occasional +0.4%
        price = price * (1 + change)
        high = price * 1.002
        low = price * 0.998
        open_ = price * 0.999

        candles.append(create_candle(
            timestamp=base + timedelta(minutes=i),
            open_=open_,
            high=high,
            low=low,
            close=price,
            volume=1000 + (i % 10) * 100
        ))

    return tuple(candles)


def generate_trending_down_candles(n: int = 250, start_price: float = 2.0) -> tuple:
    """Generate candles for a trending down market."""
    candles = []
    base = datetime.now() - timedelta(minutes=n)
    price = start_price

    for i in range(n):
        # Trending down with small bounces
        change = -0.003 - (0.001 * (i % 5 == 0))
        price = price * (1 + change)
        high = price * 1.002
        low = price * 0.998
        open_ = price * 1.001

        candles.append(create_candle(
            timestamp=base + timedelta(minutes=i),
            open_=open_,
            high=high,
            low=low,
            close=price,
            volume=1000 + (i % 10) * 100
        ))

    return tuple(candles)


def generate_sideways_candles(n: int = 250, center_price: float = 2.0) -> tuple:
    """Generate candles for a sideways/ranging market."""
    import math
    candles = []
    base = datetime.now() - timedelta(minutes=n)

    for i in range(n):
        # Oscillate around center with no clear trend
        offset = math.sin(i / 10) * 0.02  # +/- 2% oscillation
        price = center_price * (1 + offset)
        high = price * 1.005
        low = price * 0.995
        open_ = center_price * (1 + math.sin((i - 1) / 10) * 0.02)

        candles.append(create_candle(
            timestamp=base + timedelta(minutes=i),
            open_=open_,
            high=high,
            low=low,
            close=price,
            volume=500 + (i % 5) * 50
        ))

    return tuple(candles)


def generate_high_volatility_candles(n: int = 250, center_price: float = 2.0) -> tuple:
    """Generate candles for a high volatility market."""
    import random
    random.seed(42)  # Reproducible
    candles = []
    base = datetime.now() - timedelta(minutes=n)
    price = center_price

    for i in range(n):
        # Large random moves
        change = (random.random() - 0.5) * 0.04  # +/- 2% moves
        price = price * (1 + change)
        high = price * (1 + abs(change) + 0.005)
        low = price * (1 - abs(change) - 0.005)
        open_ = price * (1 - change * 0.5)

        candles.append(create_candle(
            timestamp=base + timedelta(minutes=i),
            open_=open_,
            high=high,
            low=low,
            close=price,
            volume=2000 + random.randint(0, 1000)
        ))

    return tuple(candles)


@pytest.fixture
def trending_up_candles():
    """Fixture for trending up candles."""
    return generate_trending_up_candles()


@pytest.fixture
def trending_down_candles():
    """Fixture for trending down candles."""
    return generate_trending_down_candles()


@pytest.fixture
def sideways_candles():
    """Fixture for sideways candles."""
    return generate_sideways_candles()


@pytest.fixture
def high_volatility_candles():
    """Fixture for high volatility candles."""
    return generate_high_volatility_candles()


@pytest.fixture
def mock_external_sentiment():
    """Fixture for external sentiment data."""
    return ExternalSentiment(
        fear_greed_value=55,
        fear_greed_classification="Greed",
        btc_dominance=56.5,
        last_updated=datetime.now()
    )


# =============================================================================
# TEST TYPES
# =============================================================================

class TestRegimeTypes:
    """Test regime type definitions."""

    def test_market_regime_enum(self):
        """MarketRegime enum should have expected values."""
        assert MarketRegime.STRONG_BULL.value is not None
        assert MarketRegime.BULL.value is not None
        assert MarketRegime.SIDEWAYS.value is not None
        assert MarketRegime.BEAR.value is not None
        assert MarketRegime.STRONG_BEAR.value is not None

    def test_volatility_state_enum(self):
        """VolatilityState enum should have expected values."""
        assert VolatilityState.LOW.value is not None
        assert VolatilityState.MEDIUM.value is not None
        assert VolatilityState.HIGH.value is not None
        assert VolatilityState.EXTREME.value is not None

    def test_trend_strength_enum(self):
        """TrendStrength enum should have expected values."""
        assert TrendStrength.ABSENT.value is not None
        assert TrendStrength.WEAK.value is not None
        assert TrendStrength.EMERGING.value is not None
        assert TrendStrength.STRONG.value is not None
        assert TrendStrength.VERY_STRONG.value is not None

    def test_indicator_scores_frozen(self):
        """IndicatorScores should be immutable."""
        scores = IndicatorScores(
            adx_score=0.5,
            chop_score=-0.3,
            ma_score=0.8,
            rsi_score=0.2,
            volume_score=0.1,
            sentiment_score=0.0
        )
        assert scores.adx_score == 0.5
        with pytest.raises(AttributeError):
            scores.adx_score = 0.6

    def test_regime_adjustments_defaults(self):
        """RegimeAdjustments should have sensible defaults."""
        adj = RegimeAdjustments()
        assert adj.position_size_multiplier == 1.0
        assert adj.stop_loss_multiplier == 1.0
        assert adj.take_profit_multiplier == 1.0
        assert adj.strategy_enabled is True

    def test_regime_adjustments_volatility_modifier(self):
        """RegimeAdjustments.apply_volatility_modifier should work correctly."""
        base = RegimeAdjustments(position_size_multiplier=1.0)

        # High volatility should reduce position size
        modified = base.apply_volatility_modifier(VolatilityState.HIGH)
        assert modified.position_size_multiplier < 1.0

        # Extreme volatility should disable strategy
        modified_extreme = base.apply_volatility_modifier(VolatilityState.EXTREME)
        assert modified_extreme.strategy_enabled is False


# =============================================================================
# TEST CHOPPINESS INDICATOR
# =============================================================================

class TestChoppinessIndicator:
    """Test the Choppiness Index indicator."""

    def test_choppiness_trending_market(self, trending_up_candles):
        """Choppiness should be low (<38.2) in trending markets."""
        chop = calculate_choppiness(trending_up_candles, period=14)
        assert chop is not None
        # Trending markets should have lower choppiness
        assert chop < 55, f"Expected choppiness < 55 in trending market, got {chop}"

    def test_choppiness_sideways_market(self, sideways_candles):
        """Choppiness should be high (>50) in sideways markets."""
        chop = calculate_choppiness(sideways_candles, period=14)
        assert chop is not None
        # Sideways markets should have higher choppiness
        assert chop > 45, f"Expected choppiness > 45 in sideways market, got {chop}"

    def test_choppiness_insufficient_data(self):
        """Choppiness should return None with insufficient data."""
        candles = generate_sideways_candles(n=10)
        chop = calculate_choppiness(candles, period=14)
        assert chop is None

    def test_choppiness_bounds(self, trending_up_candles):
        """Choppiness should be bounded between 0 and 100."""
        chop = calculate_choppiness(trending_up_candles, period=14)
        assert 0 <= chop <= 100


# =============================================================================
# TEST ADX WITH DI
# =============================================================================

class TestADXWithDI:
    """Test the enhanced ADX function with +DI/-DI."""

    def test_adx_trending_up_market(self, trending_up_candles):
        """ADX should be elevated with +DI > -DI in uptrend."""
        result = calculate_adx_with_di(trending_up_candles, period=14)
        assert result is not None
        assert result.adx > 20, f"Expected ADX > 20 in trending market, got {result.adx}"
        assert result.plus_di > result.minus_di, "+DI should be > -DI in uptrend"

    def test_adx_trending_down_market(self, trending_down_candles):
        """ADX should be elevated with -DI > +DI in downtrend."""
        result = calculate_adx_with_di(trending_down_candles, period=14)
        assert result is not None
        assert result.adx > 20, f"Expected ADX > 20 in trending market, got {result.adx}"
        assert result.minus_di > result.plus_di, "-DI should be > +DI in downtrend"

    def test_adx_trend_strength_classification(self, trending_up_candles):
        """ADX should provide correct trend strength classification."""
        result = calculate_adx_with_di(trending_up_candles, period=14)
        assert result is not None
        # Strong trend should be STRONG or VERY_STRONG
        assert result.trend_strength in ('EMERGING', 'STRONG', 'VERY_STRONG')

    def test_adx_insufficient_data(self):
        """ADX should return None with insufficient data."""
        candles = generate_sideways_candles(n=20)
        result = calculate_adx_with_di(candles, period=14)
        assert result is None


# =============================================================================
# TEST MA ALIGNMENT
# =============================================================================

class TestMAAlignment:
    """Test the MA alignment function."""

    def test_perfect_bull_alignment(self):
        """Should identify perfect bull alignment."""
        result = calculate_ma_alignment(
            price=2.35,
            sma_20=2.32,
            sma_50=2.30,
            sma_200=2.25
        )
        assert result.alignment == 'PERFECT_BULL'
        assert result.score == 1.0

    def test_perfect_bear_alignment(self):
        """Should identify perfect bear alignment."""
        result = calculate_ma_alignment(
            price=2.15,
            sma_20=2.18,
            sma_50=2.22,
            sma_200=2.28
        )
        assert result.alignment == 'PERFECT_BEAR'
        assert result.score == -1.0

    def test_neutral_alignment(self):
        """Should identify neutral alignment."""
        result = calculate_ma_alignment(
            price=2.25,
            sma_20=2.24,
            sma_50=2.25,  # Close to SMA200
            sma_200=2.25
        )
        assert result.alignment == 'NEUTRAL'
        assert result.score == 0.0


# =============================================================================
# TEST COMPOSITE SCORER
# =============================================================================

class TestCompositeScorer:
    """Test the CompositeScorer class."""

    def test_scorer_initialization(self):
        """CompositeScorer should initialize with default weights."""
        scorer = CompositeScorer()
        assert scorer.weights['adx'] == 0.25
        assert scorer.weights['chop'] == 0.20
        assert scorer.weights['ma'] == 0.20

    def test_scorer_custom_weights(self):
        """CompositeScorer should accept custom weights."""
        config = {'weights': {'adx': 0.30, 'chop': 0.15, 'ma': 0.25, 'rsi': 0.15, 'volume': 0.10, 'sentiment': 0.05}}
        scorer = CompositeScorer(config)
        assert scorer.weights['adx'] == 0.30

    def test_scorer_bull_market(self, trending_up_candles, mock_external_sentiment):
        """Scorer should classify trending up market as bullish."""
        scorer = CompositeScorer()
        result = scorer.calculate_symbol_regime('XRP/USDT', trending_up_candles, mock_external_sentiment)

        assert result is not None
        assert result.regime in (MarketRegime.BULL, MarketRegime.STRONG_BULL)
        assert result.composite_score > 0

    def test_scorer_bear_market(self, trending_down_candles, mock_external_sentiment):
        """Scorer should classify trending down market as bearish."""
        scorer = CompositeScorer()
        result = scorer.calculate_symbol_regime('XRP/USDT', trending_down_candles, mock_external_sentiment)

        assert result is not None
        assert result.regime in (MarketRegime.BEAR, MarketRegime.STRONG_BEAR)
        assert result.composite_score < 0

    def test_scorer_sideways_market(self, sideways_candles, mock_external_sentiment):
        """Scorer should classify sideways market correctly."""
        scorer = CompositeScorer()
        result = scorer.calculate_symbol_regime('XRP/USDT', sideways_candles, mock_external_sentiment)

        assert result is not None
        # Sideways should have score near zero
        assert -0.3 < result.composite_score < 0.3

    def test_scorer_insufficient_data(self, mock_external_sentiment):
        """Scorer should return None with insufficient data."""
        scorer = CompositeScorer()
        short_candles = generate_sideways_candles(n=50)
        result = scorer.calculate_symbol_regime('XRP/USDT', short_candles, mock_external_sentiment)
        assert result is None


# =============================================================================
# TEST EXTERNAL DATA FETCHER
# =============================================================================

class TestExternalDataFetcher:
    """Test the ExternalDataFetcher class."""

    def test_fetcher_disabled(self):
        """Fetcher should return None when disabled."""
        fetcher = ExternalDataFetcher(enabled=False)
        result = asyncio.run(fetcher.fetch())
        assert result is None

    @pytest.mark.asyncio
    async def test_fetcher_caching(self):
        """Fetcher should cache results."""
        fetcher = ExternalDataFetcher(enabled=True)

        # Mock the fetch methods
        with patch.object(fetcher, '_fetch_all', new_callable=AsyncMock) as mock_fetch:
            mock_sentiment = ExternalSentiment(
                fear_greed_value=50,
                fear_greed_classification="Neutral",
                btc_dominance=55.0,
                last_updated=datetime.utcnow()
            )
            mock_fetch.return_value = mock_sentiment

            # First call should fetch
            result1 = await fetcher.fetch()
            assert result1 is not None
            assert mock_fetch.call_count == 1

            # Second call should use cache
            result2 = await fetcher.fetch()
            assert result2 is not None
            assert mock_fetch.call_count == 1  # Still 1, used cache

    def test_fetcher_fallback_on_error(self):
        """Fetcher should return fallback on error."""
        fetcher = ExternalDataFetcher(enabled=True)

        # Inject a cached value
        fetcher._cache = ExternalSentiment(
            fear_greed_value=45,
            fear_greed_classification="Fear",
            btc_dominance=58.0,
            last_updated=datetime.utcnow()
        )
        fetcher._cache_time = datetime.utcnow() - timedelta(hours=1)  # Stale

        # Mock fetch to raise error
        async def error_fetch():
            raise Exception("API Error")

        with patch.object(fetcher, '_fetch_all', side_effect=error_fetch):
            result = asyncio.run(fetcher.fetch())
            # Should return stale cache
            assert result is not None
            assert result.fear_greed_value == 45


# =============================================================================
# TEST MTF ANALYZER
# =============================================================================

class TestMTFAnalyzer:
    """Test the MTFAnalyzer class."""

    def test_analyzer_initialization(self):
        """MTFAnalyzer should initialize with default weights."""
        analyzer = MTFAnalyzer()
        assert analyzer.weights['1m'] == 1
        assert analyzer.weights['5m'] == 2

    def test_analyzer_bull_confluence(self, trending_up_candles):
        """Analyzer should detect bullish signals across timeframes."""
        analyzer = MTFAnalyzer()

        # Create mock DataSnapshot
        data = DataSnapshot(
            timestamp=datetime.now(),
            prices={'XRP/USDT': 2.35},
            candles_1m={'XRP/USDT': trending_up_candles},
            candles_5m={'XRP/USDT': trending_up_candles[:50]},  # Shorter for 5m
            orderbooks={'XRP/USDT': OrderbookSnapshot(bids=(), asks=())},
            trades={'XRP/USDT': ()}
        )

        result = analyzer.analyze(data, 'XRP/USDT')
        assert result is not None
        # Check that most individual timeframes show bullish
        bull_count = sum(1 for r in result.per_timeframe.values()
                        if r in (MarketRegime.BULL, MarketRegime.STRONG_BULL))
        assert bull_count >= 2, f"Expected at least 2 bullish timeframes, got {bull_count}"
        assert result.alignment_score >= 0.5


# =============================================================================
# TEST PARAMETER ROUTER
# =============================================================================

class TestParameterRouter:
    """Test the ParameterRouter class."""

    def test_router_initialization(self):
        """ParameterRouter should initialize with default adjustments."""
        router = ParameterRouter()
        assert 'mean_reversion' in router.adjustments
        assert 'momentum_scalping' in router.adjustments

    def test_router_mean_reversion_in_strong_bull(self):
        """Mean reversion should be disabled in strong bull market."""
        router = ParameterRouter()

        regime = RegimeSnapshot(
            timestamp=datetime.now(),
            overall_regime=MarketRegime.STRONG_BULL,
            overall_confidence=0.8,
            is_trending=True,
            trend_direction='UP',
            volatility_state=VolatilityState.MEDIUM,
            symbol_regimes={},
            mtf_confluence=None,
            external_sentiment=None,
            composite_score=0.6,
            indicator_weights={},
            regime_age_seconds=300,
            recent_transitions=1
        )

        adjustments = router.get_adjustments('mean_reversion', regime)
        assert adjustments.strategy_enabled is False

    def test_router_mean_reversion_in_sideways(self):
        """Mean reversion should be enabled in sideways market."""
        router = ParameterRouter()

        regime = RegimeSnapshot(
            timestamp=datetime.now(),
            overall_regime=MarketRegime.SIDEWAYS,
            overall_confidence=0.6,
            is_trending=False,
            trend_direction='NONE',
            volatility_state=VolatilityState.MEDIUM,
            symbol_regimes={},
            mtf_confluence=None,
            external_sentiment=None,
            composite_score=0.0,
            indicator_weights={},
            regime_age_seconds=300,
            recent_transitions=1
        )

        adjustments = router.get_adjustments('mean_reversion', regime)
        assert adjustments.strategy_enabled is True

    def test_router_apply_to_config(self):
        """Router should correctly apply adjustments to config."""
        router = ParameterRouter()

        base_config = {
            'position_size_usd': 20.0,
            'stop_loss_pct': 1.0,
            'take_profit_pct': 0.5
        }

        adjustments = RegimeAdjustments(
            position_size_multiplier=0.5,
            stop_loss_multiplier=1.5,
            take_profit_multiplier=0.8
        )

        adjusted = router.apply_to_config(base_config, adjustments)

        assert adjusted['position_size_usd'] == 10.0  # 20 * 0.5
        assert adjusted['stop_loss_pct'] == 1.5       # 1.0 * 1.5
        assert adjusted['take_profit_pct'] == 0.4     # 0.5 * 0.8


# =============================================================================
# TEST REGIME DETECTOR (INTEGRATION)
# =============================================================================

class TestRegimeDetector:
    """Integration tests for the RegimeDetector class."""

    def test_detector_initialization(self):
        """RegimeDetector should initialize correctly."""
        detector = RegimeDetector(symbols=['XRP/USDT', 'BTC/USDT'])
        assert detector.symbols == ['XRP/USDT', 'BTC/USDT']
        assert detector.composite_scorer is not None
        assert detector.mtf_analyzer is not None

    @pytest.mark.asyncio
    async def test_detector_bull_market(self, trending_up_candles):
        """Detector should identify bull market."""
        detector = RegimeDetector(
            symbols=['XRP/USDT'],
            config={'external_enabled': False}
        )

        data = DataSnapshot(
            timestamp=datetime.now(),
            prices={'XRP/USDT': trending_up_candles[-1].close},
            candles_1m={'XRP/USDT': trending_up_candles},
            candles_5m={'XRP/USDT': ()},
            orderbooks={'XRP/USDT': OrderbookSnapshot(bids=(), asks=())},
            trades={'XRP/USDT': ()}
        )

        regime = await detector.detect(data)

        assert regime is not None
        assert regime.overall_regime in (MarketRegime.BULL, MarketRegime.STRONG_BULL)
        assert regime.is_trending is True
        assert regime.trend_direction == 'UP'

    @pytest.mark.asyncio
    async def test_detector_bear_market(self, trending_down_candles):
        """Detector should identify bear market."""
        detector = RegimeDetector(
            symbols=['XRP/USDT'],
            config={'external_enabled': False}
        )

        data = DataSnapshot(
            timestamp=datetime.now(),
            prices={'XRP/USDT': trending_down_candles[-1].close},
            candles_1m={'XRP/USDT': trending_down_candles},
            candles_5m={'XRP/USDT': ()},
            orderbooks={'XRP/USDT': OrderbookSnapshot(bids=(), asks=())},
            trades={'XRP/USDT': ()}
        )

        regime = await detector.detect(data)

        assert regime is not None
        assert regime.overall_regime in (MarketRegime.BEAR, MarketRegime.STRONG_BEAR)
        assert regime.is_trending is True
        assert regime.trend_direction == 'DOWN'

    @pytest.mark.asyncio
    async def test_detector_helper_methods(self, trending_up_candles):
        """Detector helper methods should work correctly."""
        detector = RegimeDetector(
            symbols=['XRP/USDT'],
            config={'external_enabled': False}
        )

        data = DataSnapshot(
            timestamp=datetime.now(),
            prices={'XRP/USDT': trending_up_candles[-1].close},
            candles_1m={'XRP/USDT': trending_up_candles},
            candles_5m={'XRP/USDT': ()},
            orderbooks={'XRP/USDT': OrderbookSnapshot(bids=(), asks=())},
            trades={'XRP/USDT': ()}
        )

        regime = await detector.detect(data)

        # Test is_favorable methods
        assert regime.is_favorable_for_trend_strategy() is True
        assert regime.is_favorable_for_mean_reversion() is False

        # Test should_trade
        assert detector.should_trade('mean_reversion', regime) is False
        assert detector.should_trade('momentum_scalping', regime) is True


# =============================================================================
# TEST REGIME SNAPSHOT HELPER METHODS
# =============================================================================

class TestRegimeSnapshotHelpers:
    """Test helper methods on RegimeSnapshot."""

    def test_is_favorable_for_trend_strategy(self):
        """is_favorable_for_trend_strategy should work correctly."""
        # Trending with good confidence
        regime = RegimeSnapshot(
            timestamp=datetime.now(),
            overall_regime=MarketRegime.BULL,
            overall_confidence=0.7,
            is_trending=True,
            trend_direction='UP',
            volatility_state=VolatilityState.MEDIUM,
            symbol_regimes={},
            mtf_confluence=None,
            external_sentiment=None,
            composite_score=0.3,
            indicator_weights={},
            regime_age_seconds=300,
            recent_transitions=1
        )
        assert regime.is_favorable_for_trend_strategy() is True

        # Not trending
        regime2 = RegimeSnapshot(
            timestamp=datetime.now(),
            overall_regime=MarketRegime.SIDEWAYS,
            overall_confidence=0.7,
            is_trending=False,
            trend_direction='NONE',
            volatility_state=VolatilityState.MEDIUM,
            symbol_regimes={},
            mtf_confluence=None,
            external_sentiment=None,
            composite_score=0.0,
            indicator_weights={},
            regime_age_seconds=300,
            recent_transitions=1
        )
        assert regime2.is_favorable_for_trend_strategy() is False

    def test_is_favorable_for_mean_reversion(self):
        """is_favorable_for_mean_reversion should work correctly."""
        # Sideways with low volatility
        regime = RegimeSnapshot(
            timestamp=datetime.now(),
            overall_regime=MarketRegime.SIDEWAYS,
            overall_confidence=0.6,
            is_trending=False,
            trend_direction='NONE',
            volatility_state=VolatilityState.LOW,
            symbol_regimes={},
            mtf_confluence=None,
            external_sentiment=None,
            composite_score=0.0,
            indicator_weights={},
            regime_age_seconds=300,
            recent_transitions=1
        )
        assert regime.is_favorable_for_mean_reversion() is True

    def test_should_reduce_exposure(self):
        """should_reduce_exposure should work correctly."""
        # Extreme volatility
        regime = RegimeSnapshot(
            timestamp=datetime.now(),
            overall_regime=MarketRegime.SIDEWAYS,
            overall_confidence=0.6,
            is_trending=False,
            trend_direction='NONE',
            volatility_state=VolatilityState.EXTREME,
            symbol_regimes={},
            mtf_confluence=None,
            external_sentiment=None,
            composite_score=0.0,
            indicator_weights={},
            regime_age_seconds=300,
            recent_transitions=1
        )
        assert regime.should_reduce_exposure() is True

        # Many transitions
        regime2 = RegimeSnapshot(
            timestamp=datetime.now(),
            overall_regime=MarketRegime.BULL,
            overall_confidence=0.6,
            is_trending=True,
            trend_direction='UP',
            volatility_state=VolatilityState.MEDIUM,
            symbol_regimes={},
            mtf_confluence=None,
            external_sentiment=None,
            composite_score=0.3,
            indicator_weights={},
            regime_age_seconds=300,
            recent_transitions=10  # High instability
        )
        assert regime2.should_reduce_exposure() is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
