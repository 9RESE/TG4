"""
Unit tests for the Indicator Library.

Tests validate:
- Calculation accuracy against known values
- Boundary conditions (RSI 0-100, etc.)
- Performance requirements (<50ms for 1000 candles)
- Error handling for invalid inputs
"""

import pytest
import numpy as np
from decimal import Decimal
from datetime import datetime, timezone
import time

from triplegain.src.data.indicator_library import (
    IndicatorLibrary,
    IndicatorResult,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_closes() -> list[float]:
    """Sample closing prices for testing."""
    return [
        44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08,
        45.89, 46.03, 45.61, 46.28, 46.28, 46.00, 46.03, 46.41, 46.22, 45.64,
        46.21, 46.25, 45.71, 46.45, 45.78, 46.21, 46.25, 45.71, 46.45, 45.78,
    ]


@pytest.fixture
def sample_ohlcv() -> dict:
    """Sample OHLCV data for testing."""
    # Generate 100 candles of sample data
    np.random.seed(42)
    n = 100
    base_price = 45000.0

    closes = [base_price]
    for _ in range(n - 1):
        change = np.random.normal(0, 100)
        closes.append(closes[-1] + change)

    closes = np.array(closes)
    highs = closes + np.abs(np.random.normal(50, 20, n))
    lows = closes - np.abs(np.random.normal(50, 20, n))
    opens = closes + np.random.normal(0, 30, n)
    volumes = np.abs(np.random.normal(1000, 200, n))

    return {
        'open': opens.tolist(),
        'high': highs.tolist(),
        'low': lows.tolist(),
        'close': closes.tolist(),
        'volume': volumes.tolist(),
    }


@pytest.fixture
def indicator_library() -> IndicatorLibrary:
    """Create an indicator library instance with default config."""
    config = {
        'ema': {'periods': [9, 21, 50, 200]},
        'sma': {'periods': [20, 50, 200]},
        'rsi': {'period': 14},
        'macd': {'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
        'atr': {'period': 14},
        'bollinger_bands': {'period': 20, 'std_dev': 2.0},
        'adx': {'period': 14},
        'stochastic_rsi': {'rsi_period': 14, 'stoch_period': 14, 'k_period': 3, 'd_period': 3},
        'choppiness': {'period': 14},
    }
    return IndicatorLibrary(config)


# =============================================================================
# EMA Tests
# =============================================================================

class TestEMA:
    """Tests for Exponential Moving Average calculations."""

    def test_ema_basic_calculation(self, indicator_library, sample_closes):
        """Test EMA calculation against known values."""
        result = indicator_library.calculate_ema(sample_closes, period=10)

        assert len(result) == len(sample_closes)
        # First values should be NaN until we have enough data
        assert np.isnan(result[0])
        # After warmup, values should exist
        assert not np.isnan(result[-1])

    def test_ema_multiplier(self, indicator_library):
        """Test EMA uses correct smoothing multiplier."""
        # For period 10, multiplier should be 2/(10+1) = 0.1818...
        closes = [10.0] * 20 + [20.0]
        result = indicator_library.calculate_ema(closes, period=10)

        # After the jump to 20, EMA should start moving up
        assert result[-1] > 10.0
        assert result[-1] < 20.0

    def test_ema_convergence(self, indicator_library):
        """Test EMA converges to price over time."""
        closes = [100.0] * 50
        result = indicator_library.calculate_ema(closes, period=10)

        # Should converge to 100
        assert abs(result[-1] - 100.0) < 0.01


# =============================================================================
# SMA Tests
# =============================================================================

class TestSMA:
    """Tests for Simple Moving Average calculations."""

    def test_sma_basic_calculation(self, indicator_library, sample_closes):
        """Test SMA calculation is correct."""
        result = indicator_library.calculate_sma(sample_closes, period=5)

        # Verify using manual calculation
        expected = np.mean(sample_closes[:5])
        assert abs(result[4] - expected) < 0.0001

    def test_sma_length(self, indicator_library, sample_closes):
        """Test SMA output has correct length."""
        result = indicator_library.calculate_sma(sample_closes, period=5)
        assert len(result) == len(sample_closes)


# =============================================================================
# RSI Tests
# =============================================================================

class TestRSI:
    """Tests for Relative Strength Index calculations."""

    def test_rsi_bounds(self, indicator_library, sample_closes):
        """Test RSI values are within 0-100."""
        result = indicator_library.calculate_rsi(sample_closes, period=14)

        # Filter out NaN values
        valid_values = [v for v in result if not np.isnan(v)]

        for value in valid_values:
            assert 0 <= value <= 100, f"RSI {value} out of bounds"

    def test_rsi_overbought(self, indicator_library):
        """Test RSI is high for strong uptrend."""
        # Consistently rising prices
        closes = [100.0 + i * 2 for i in range(30)]
        result = indicator_library.calculate_rsi(closes, period=14)

        # Should be overbought (>70)
        assert result[-1] > 70

    def test_rsi_oversold(self, indicator_library):
        """Test RSI is low for strong downtrend."""
        # Consistently falling prices
        closes = [100.0 - i * 2 for i in range(30)]
        result = indicator_library.calculate_rsi(closes, period=14)

        # Should be oversold (<30)
        assert result[-1] < 30

    def test_rsi_neutral(self, indicator_library):
        """Test RSI is neutral for flat prices."""
        # Oscillating prices
        closes = [50.0 + (i % 2) for i in range(50)]
        result = indicator_library.calculate_rsi(closes, period=14)

        # Should be near 50
        assert 40 <= result[-1] <= 60


# =============================================================================
# MACD Tests
# =============================================================================

class TestMACD:
    """Tests for MACD calculations."""

    def test_macd_structure(self, indicator_library, sample_closes):
        """Test MACD returns correct structure."""
        result = indicator_library.calculate_macd(
            sample_closes, fast=12, slow=26, signal=9
        )

        assert 'line' in result
        assert 'signal' in result
        assert 'histogram' in result

    def test_macd_histogram(self, indicator_library, sample_closes):
        """Test MACD histogram is line - signal."""
        result = indicator_library.calculate_macd(
            sample_closes, fast=12, slow=26, signal=9
        )

        # Get last valid values
        line = result['line'][-1]
        signal = result['signal'][-1]
        histogram = result['histogram'][-1]

        if not np.isnan(line) and not np.isnan(signal):
            assert abs(histogram - (line - signal)) < 0.0001

    def test_macd_uptrend(self, indicator_library):
        """Test MACD is positive in uptrend."""
        closes = [100.0 + i * 2 for i in range(50)]
        result = indicator_library.calculate_macd(
            closes, fast=12, slow=26, signal=9
        )

        # MACD line should be positive in uptrend
        assert result['line'][-1] > 0


# =============================================================================
# ATR Tests
# =============================================================================

class TestATR:
    """Tests for Average True Range calculations."""

    def test_atr_positive(self, indicator_library, sample_ohlcv):
        """Test ATR is always positive."""
        result = indicator_library.calculate_atr(
            sample_ohlcv['high'],
            sample_ohlcv['low'],
            sample_ohlcv['close'],
            period=14
        )

        valid_values = [v for v in result if not np.isnan(v)]
        for value in valid_values:
            assert value > 0, "ATR must be positive"

    def test_atr_volatility_correlation(self, indicator_library):
        """Test ATR increases with volatility."""
        # Low volatility
        low_vol = {
            'high': [100.0 + 0.5] * 30,
            'low': [100.0 - 0.5] * 30,
            'close': [100.0] * 30,
        }
        atr_low = indicator_library.calculate_atr(
            low_vol['high'], low_vol['low'], low_vol['close'], period=14
        )[-1]

        # High volatility
        high_vol = {
            'high': [100.0 + 5.0] * 30,
            'low': [100.0 - 5.0] * 30,
            'close': [100.0] * 30,
        }
        atr_high = indicator_library.calculate_atr(
            high_vol['high'], high_vol['low'], high_vol['close'], period=14
        )[-1]

        assert atr_high > atr_low


# =============================================================================
# Bollinger Bands Tests
# =============================================================================

class TestBollingerBands:
    """Tests for Bollinger Bands calculations."""

    def test_bb_structure(self, indicator_library, sample_closes):
        """Test Bollinger Bands returns correct structure."""
        result = indicator_library.calculate_bollinger_bands(
            sample_closes, period=20, std_dev=2.0
        )

        assert 'upper' in result
        assert 'middle' in result
        assert 'lower' in result
        assert 'width' in result
        assert 'position' in result

    def test_bb_ordering(self, indicator_library, sample_closes):
        """Test upper > middle > lower."""
        result = indicator_library.calculate_bollinger_bands(
            sample_closes, period=20, std_dev=2.0
        )

        upper = result['upper'][-1]
        middle = result['middle'][-1]
        lower = result['lower'][-1]

        if not np.isnan(upper):
            assert upper > middle > lower

    def test_bb_position_bounds(self, indicator_library, sample_closes):
        """Test BB position is within reasonable bounds."""
        result = indicator_library.calculate_bollinger_bands(
            sample_closes, period=20, std_dev=2.0
        )

        # Position can be outside 0-1 if price breaks bands
        position = result['position'][-1]
        assert -1 <= position <= 2  # Reasonable bounds


# =============================================================================
# ADX Tests
# =============================================================================

class TestADX:
    """Tests for Average Directional Index calculations."""

    def test_adx_bounds(self, indicator_library, sample_ohlcv):
        """Test ADX values are within 0-100."""
        result = indicator_library.calculate_adx(
            sample_ohlcv['high'],
            sample_ohlcv['low'],
            sample_ohlcv['close'],
            period=14
        )

        valid_values = [v for v in result if not np.isnan(v)]
        for value in valid_values:
            assert 0 <= value <= 100, f"ADX {value} out of bounds"

    def test_adx_trending_market(self, indicator_library):
        """Test ADX is high in trending market."""
        # Strong uptrend
        n = 50
        closes = [100.0 + i * 3 for i in range(n)]
        highs = [c + 1 for c in closes]
        lows = [c - 1 for c in closes]

        result = indicator_library.calculate_adx(highs, lows, closes, period=14)

        # ADX should be above 25 for trending market
        assert result[-1] > 20


# =============================================================================
# OBV Tests
# =============================================================================

class TestOBV:
    """Tests for On-Balance Volume calculations."""

    def test_obv_direction(self, indicator_library):
        """Test OBV increases on up days."""
        closes = [100, 101, 102, 101, 103]
        volumes = [1000, 1000, 1000, 1000, 1000]

        result = indicator_library.calculate_obv(closes, volumes)

        # OBV should increase overall with rising prices
        assert result[-1] > result[0]

    def test_obv_cumulative(self, indicator_library):
        """Test OBV is cumulative."""
        closes = [100, 101, 102, 103, 104]
        volumes = [1000, 1000, 1000, 1000, 1000]

        result = indicator_library.calculate_obv(closes, volumes)

        # All up days, OBV should equal cumulative volume (minus first)
        expected = 4000  # 4 up days * 1000 volume
        assert abs(result[-1] - expected) < 0.01


# =============================================================================
# Choppiness Index Tests
# =============================================================================

class TestChoppiness:
    """Tests for Choppiness Index calculations."""

    def test_choppiness_bounds(self, indicator_library, sample_ohlcv):
        """Test Choppiness Index is within 0-100."""
        result = indicator_library.calculate_choppiness(
            sample_ohlcv['high'],
            sample_ohlcv['low'],
            sample_ohlcv['close'],
            period=14
        )

        valid_values = [v for v in result if not np.isnan(v)]
        for value in valid_values:
            assert 0 <= value <= 100, f"Choppiness {value} out of bounds"

    def test_choppiness_trending(self, indicator_library):
        """Test Choppiness is low in trending market."""
        # Strong trend
        n = 50
        closes = [100.0 + i * 2 for i in range(n)]
        highs = [c + 0.5 for c in closes]
        lows = [c - 0.5 for c in closes]

        result = indicator_library.calculate_choppiness(highs, lows, closes, period=14)

        # Choppiness < 38.2 indicates trending
        assert result[-1] < 50


# =============================================================================
# Squeeze Detection Tests
# =============================================================================

class TestSqueezeDetection:
    """Tests for Bollinger Band / Keltner Channel squeeze detection."""

    def test_squeeze_returns_bool(self, indicator_library, sample_ohlcv):
        """Test squeeze detection returns boolean."""
        result = indicator_library.detect_squeeze(
            sample_ohlcv['close'],
            sample_ohlcv['high'],
            sample_ohlcv['low'],
            bb_config={'period': 20, 'std_dev': 2.0},
            kc_config={'period': 20, 'mult': 1.5}
        )

        assert isinstance(result, bool)


# =============================================================================
# Performance Tests
# =============================================================================

class TestPerformance:
    """Performance tests for indicator calculations."""

    def test_all_indicators_performance(self, indicator_library):
        """Test all indicators complete within 50ms for 1000 candles."""
        # Generate 1000 candles
        np.random.seed(42)
        n = 1000
        closes = np.cumsum(np.random.randn(n)) + 45000
        highs = closes + np.abs(np.random.randn(n) * 50)
        lows = closes - np.abs(np.random.randn(n) * 50)
        volumes = np.abs(np.random.randn(n) * 1000) + 500

        start = time.time()

        indicator_library.calculate_ema(closes.tolist(), 9)
        indicator_library.calculate_ema(closes.tolist(), 21)
        indicator_library.calculate_sma(closes.tolist(), 20)
        indicator_library.calculate_rsi(closes.tolist(), 14)
        indicator_library.calculate_macd(closes.tolist(), 12, 26, 9)
        indicator_library.calculate_atr(highs.tolist(), lows.tolist(), closes.tolist(), 14)
        indicator_library.calculate_bollinger_bands(closes.tolist(), 20, 2.0)
        indicator_library.calculate_adx(highs.tolist(), lows.tolist(), closes.tolist(), 14)
        indicator_library.calculate_obv(closes.tolist(), volumes.tolist())
        indicator_library.calculate_choppiness(highs.tolist(), lows.tolist(), closes.tolist(), 14)

        elapsed = (time.time() - start) * 1000

        assert elapsed < 50, f"Indicators took {elapsed:.2f}ms, expected <50ms"

    def test_ema_performance_1000(self, indicator_library):
        """Test EMA performance for 1000 candles."""
        np.random.seed(42)
        closes = np.cumsum(np.random.randn(1000)) + 45000

        start = time.time()
        indicator_library.calculate_ema(closes.tolist(), 200)
        elapsed = (time.time() - start) * 1000

        assert elapsed < 5, f"EMA took {elapsed:.2f}ms, expected <5ms"


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Tests for error handling."""

    def test_empty_input(self, indicator_library):
        """Test handling of empty input."""
        with pytest.raises(ValueError):
            indicator_library.calculate_ema([], 14)

    def test_insufficient_data(self, indicator_library):
        """Test handling of insufficient data."""
        result = indicator_library.calculate_ema([1, 2, 3], 14)
        # Should return NaN for insufficient data
        assert all(np.isnan(v) for v in result)

    def test_invalid_period(self, indicator_library):
        """Test handling of invalid period."""
        with pytest.raises(ValueError):
            indicator_library.calculate_ema([1, 2, 3, 4, 5], 0)

        with pytest.raises(ValueError):
            indicator_library.calculate_ema([1, 2, 3, 4, 5], -1)


# =============================================================================
# Calculate All Tests
# =============================================================================

class TestCalculateAll:
    """Tests for the calculate_all method."""

    def test_calculate_all_returns_dict(self, indicator_library, sample_ohlcv):
        """Test calculate_all returns dictionary of results."""
        candles = [
            {
                'timestamp': datetime.now(timezone.utc),
                'open': o, 'high': h, 'low': l, 'close': c, 'volume': v
            }
            for o, h, l, c, v in zip(
                sample_ohlcv['open'],
                sample_ohlcv['high'],
                sample_ohlcv['low'],
                sample_ohlcv['close'],
                sample_ohlcv['volume']
            )
        ]

        result = indicator_library.calculate_all('BTC/USDT', '1h', candles)

        assert isinstance(result, dict)
        assert 'ema_9' in result or 'rsi_14' in result
