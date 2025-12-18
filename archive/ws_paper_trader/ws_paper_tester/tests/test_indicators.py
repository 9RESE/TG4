"""
Indicator Library Test Suite

Tests the centralized indicator library against golden fixtures to ensure
behavioral compatibility with original strategy implementations.

Run with: python -m pytest tests/test_indicators.py -v
"""
import pytest
from datetime import datetime, timedelta

# Import indicator library
from ws_tester.indicators import (
    # Types
    PriceInput, BollingerResult, ATRResult, TradeFlowResult, TrendResult,
    extract_closes, extract_hlc, is_candle_data,
    # Moving averages
    calculate_sma, calculate_sma_series,
    calculate_ema, calculate_ema_series,
    # Oscillators
    calculate_rsi, calculate_rsi_series,
    calculate_adx, calculate_macd, calculate_macd_with_history,
    # Volatility
    calculate_volatility, calculate_atr, calculate_bollinger_bands,
    calculate_z_score, get_volatility_regime,
    # Correlation
    calculate_rolling_correlation, calculate_correlation_trend,
    # Volume
    calculate_volume_ratio, calculate_volume_spike,
    calculate_micro_price, calculate_vpin,
    # Flow
    calculate_trade_flow, check_trade_flow_confirmation,
    # Trend
    calculate_trend_slope, detect_trend_strength, calculate_trailing_stop,
)

# Import golden fixtures
from tests.fixtures.indicator_test_data import (
    TOLERANCE,
    TEST_CANDLES, TEST_CLOSES,
    BTC_TEST_CANDLES, BTC_TEST_CLOSES,
    TEST_TRADES, TEST_ORDERBOOK,
    GOLDEN_SMA, GOLDEN_EMA, GOLDEN_RSI,
    GOLDEN_VOLATILITY, GOLDEN_ATR, GOLDEN_BOLLINGER,
    GOLDEN_ADX, GOLDEN_CORRELATION,
    GOLDEN_TRADE_FLOW, GOLDEN_MICRO_PRICE, GOLDEN_TREND_SLOPE,
    GOLDEN_SUMMARY,
)


def assert_close(actual, expected, tolerance=TOLERANCE, msg=""):
    """Assert two values are close within tolerance."""
    if expected is None:
        assert actual is None, f"{msg}: Expected None, got {actual}"
    elif actual is None:
        pytest.fail(f"{msg}: Expected {expected}, got None")
    else:
        assert abs(actual - expected) < tolerance, \
            f"{msg}: Expected {expected}, got {actual}, diff={abs(actual-expected)}"


class TestHelperFunctions:
    """Test helper functions in _types.py"""

    def test_extract_closes_from_candles(self):
        """extract_closes should extract close prices from Candle objects."""
        closes = extract_closes(TEST_CANDLES)
        assert len(closes) == len(TEST_CANDLES)
        assert closes[0] == TEST_CANDLES[0].close

    def test_extract_closes_from_list(self):
        """extract_closes should return list unchanged for float list."""
        input_list = [1.0, 2.0, 3.0]
        result = extract_closes(input_list)
        assert result == input_list

    def test_extract_hlc_from_candles(self):
        """extract_hlc should extract high, low, close from Candles."""
        highs, lows, closes = extract_hlc(TEST_CANDLES)
        assert len(highs) == len(TEST_CANDLES)
        assert highs[0] == TEST_CANDLES[0].high
        assert lows[0] == TEST_CANDLES[0].low
        assert closes[0] == TEST_CANDLES[0].close

    def test_is_candle_data(self):
        """is_candle_data should correctly identify candle vs float data."""
        assert is_candle_data(TEST_CANDLES) is True
        assert is_candle_data(TEST_CLOSES) is False
        assert is_candle_data([]) is False


class TestMovingAverages:
    """Test SMA and EMA calculations against golden fixtures."""

    def test_sma_period_20_golden(self):
        """SMA with period 20 should match golden fixture."""
        result = calculate_sma(TEST_CLOSES, 20)
        expected = GOLDEN_SMA['period_20']['expected']
        assert_close(result, expected, msg="SMA(20)")

    def test_sma_series(self):
        """SMA series should produce correct length."""
        series = calculate_sma_series(TEST_CLOSES, 20)
        assert len(series) == len(TEST_CLOSES) - 20 + 1
        # Last value should match single calculation
        assert_close(series[-1], calculate_sma(TEST_CLOSES, 20), msg="SMA series last")

    def test_sma_insufficient_data(self):
        """SMA should return None with insufficient data."""
        result = calculate_sma(TEST_CLOSES[:3], 5)
        assert result is None

    def test_ema_period_21_golden(self):
        """EMA with period 21 should match golden fixture."""
        result = calculate_ema(TEST_CLOSES, 21)
        expected = GOLDEN_EMA['period_21']['expected']
        assert_close(result, expected, msg="EMA(21)")

    def test_ema_series(self):
        """EMA series should produce correct length."""
        series = calculate_ema_series(TEST_CLOSES, 21)
        assert len(series) == len(TEST_CLOSES) - 21 + 1
        assert_close(series[-1], calculate_ema(TEST_CLOSES, 21), msg="EMA series last")

    def test_ema_insufficient_data(self):
        """EMA should return None with insufficient data."""
        result = calculate_ema(TEST_CLOSES[:5], 10)
        assert result is None

    def test_candle_input_support(self):
        """Moving averages should accept Candle objects directly."""
        sma_from_candles = calculate_sma(TEST_CANDLES, 20)
        sma_from_closes = calculate_sma(TEST_CLOSES, 20)
        assert_close(sma_from_candles, sma_from_closes, msg="Candle vs Close input")


class TestOscillators:
    """Test RSI, ADX, MACD calculations against golden fixtures."""

    def test_rsi_wilder_period_14_golden(self):
        """RSI with Wilder's smoothing should match golden fixture."""
        result = calculate_rsi(TEST_CLOSES, 14)
        expected = GOLDEN_RSI['wilder_period_14']['expected']
        assert_close(result, expected, msg="RSI(14) Wilder")

    def test_rsi_series(self):
        """RSI series should produce values."""
        series = calculate_rsi_series(TEST_CLOSES, 14)
        assert len(series) > 0
        assert_close(series[-1], calculate_rsi(TEST_CLOSES, 14), msg="RSI series last")

    def test_rsi_insufficient_data(self):
        """RSI should return None with insufficient data."""
        result = calculate_rsi(TEST_CLOSES[:10], 14)
        assert result is None

    def test_rsi_extreme_values(self):
        """RSI should handle extreme price movements."""
        # All gains - RSI should be 100
        all_up = [1.0 + i*0.01 for i in range(20)]
        rsi = calculate_rsi(all_up, 14)
        assert rsi == 100.0

        # All losses - RSI should be 0
        all_down = [1.0 - i*0.01 for i in range(20)]
        rsi = calculate_rsi(all_down, 14)
        assert rsi == 0.0

    def test_rsi_flat_market(self):
        """RSI should return 50 when there's no price movement."""
        flat = [1.0] * 20
        rsi = calculate_rsi(flat, 14)
        assert rsi == 50.0  # Neutral when avg_gain == avg_loss == 0

    def test_adx_period_14_golden(self):
        """ADX should match golden fixture."""
        result = calculate_adx(TEST_CANDLES, 14)
        expected = GOLDEN_ADX['period_14']['expected']
        assert_close(result, expected, msg="ADX(14)")

    def test_adx_insufficient_data(self):
        """ADX should return None with insufficient data."""
        result = calculate_adx(TEST_CANDLES[:20], 14)
        assert result is None

    def test_macd_basic(self):
        """MACD should return dict with expected keys."""
        result = calculate_macd(TEST_CLOSES)
        assert 'macd' in result
        assert 'signal' in result
        assert 'histogram' in result

    def test_macd_with_history(self):
        """MACD with history should include crossover detection."""
        result = calculate_macd_with_history(TEST_CLOSES)
        assert 'bullish_crossover' in result
        assert 'bearish_crossover' in result
        assert 'histogram_history' in result


class TestVolatility:
    """Test volatility calculations against golden fixtures."""

    def test_volatility_lookback_20_golden(self):
        """Volatility should match golden fixture."""
        result = calculate_volatility(TEST_CANDLES, 20)
        expected = GOLDEN_VOLATILITY['lookback_20']['expected']
        assert_close(result, expected, msg="Volatility(20)")

    def test_volatility_insufficient_data(self):
        """Volatility should return 0.0 with insufficient data."""
        result = calculate_volatility(TEST_CANDLES[:10], 20)
        assert result == 0.0

    def test_atr_wilder_golden(self):
        """ATR with Wilder's smoothing should match golden fixture."""
        result = calculate_atr(TEST_CANDLES, 14, rich_output=False)
        # Implementation uses Wilder's smoothing
        expected = GOLDEN_ATR['wilder_period_14']['expected']['atr']
        assert_close(result, expected, msg="ATR(14) Wilder")

    def test_atr_rich_output(self):
        """ATR with rich output should return ATRResult."""
        result = calculate_atr(TEST_CANDLES, 14, rich_output=True)
        assert isinstance(result, ATRResult)
        assert result.atr is not None
        assert result.atr_pct is not None
        assert len(result.tr_series) > 0

    def test_bollinger_bands_golden(self):
        """Bollinger Bands should match golden fixture."""
        result = calculate_bollinger_bands(TEST_CLOSES, 20, 2.0)
        expected = GOLDEN_BOLLINGER['period_20_std_2']['expected']

        assert isinstance(result, BollingerResult)
        assert_close(result.sma, expected[0], msg="BB SMA")
        assert_close(result.upper, expected[1], msg="BB Upper")
        assert_close(result.lower, expected[2], msg="BB Lower")
        assert_close(result.std_dev, expected[3], msg="BB StdDev")

    def test_bollinger_insufficient_data(self):
        """Bollinger Bands should return None values with insufficient data."""
        result = calculate_bollinger_bands(TEST_CLOSES[:10], 20)
        assert result.sma is None
        assert result.upper is None

    def test_z_score(self):
        """Z-score should calculate correctly."""
        z = calculate_z_score(price=105, sma=100, std_dev=5)
        assert z == 1.0

        z = calculate_z_score(price=100, sma=100, std_dev=0)
        assert z == 0.0  # Avoid division by zero

    def test_volatility_regime(self):
        """Volatility regime should classify correctly."""
        config = {'regime_low_threshold': 0.3, 'regime_medium_threshold': 0.8, 'regime_high_threshold': 1.5}

        regime, thresh_mult, size_mult = get_volatility_regime(0.2, config)
        assert regime == "LOW"

        regime, thresh_mult, size_mult = get_volatility_regime(0.5, config)
        assert regime == "MEDIUM"

        regime, thresh_mult, size_mult = get_volatility_regime(1.0, config)
        assert regime == "HIGH"

        regime, thresh_mult, size_mult = get_volatility_regime(2.0, config)
        assert regime == "EXTREME"
        assert size_mult == 0.0  # Should pause trading


class TestCorrelation:
    """Test correlation calculations against golden fixtures."""

    def test_rolling_correlation_golden(self):
        """Rolling correlation should match golden fixture."""
        result = calculate_rolling_correlation(TEST_CLOSES, BTC_TEST_CLOSES, 20)
        expected = GOLDEN_CORRELATION['window_20']['expected']
        assert_close(result, expected, msg="Correlation(20)")

    def test_correlation_range(self):
        """Correlation should be between -1 and 1."""
        result = calculate_rolling_correlation(TEST_CLOSES, BTC_TEST_CLOSES, 20)
        assert result is not None
        assert -1.0 <= result <= 1.0

    def test_correlation_insufficient_data(self):
        """Correlation should return None with insufficient data."""
        result = calculate_rolling_correlation(TEST_CLOSES[:10], BTC_TEST_CLOSES[:10], 20)
        assert result is None

    def test_correlation_trend(self):
        """Correlation trend should detect declining correlations."""
        # Declining correlation history
        declining = [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45]
        result = calculate_correlation_trend(declining, 10)

        assert result.direction == 'declining'
        assert result.is_declining is True
        assert result.slope < 0


class TestVolume:
    """Test volume calculations."""

    def test_volume_ratio(self):
        """Volume ratio should handle candle data."""
        result = calculate_volume_ratio(TEST_CANDLES, 20)
        assert isinstance(result, float)
        assert result > 0

    def test_volume_spike(self):
        """Volume spike should calculate correctly."""
        volumes = [100.0] * 20 + [300.0, 300.0, 300.0]  # Spike at end
        result = calculate_volume_spike(volumes, 20, 3)
        assert result > 2.5  # Should detect spike

    def test_micro_price_golden(self):
        """Micro price should match golden fixture."""
        result = calculate_micro_price(TEST_ORDERBOOK)
        expected = GOLDEN_MICRO_PRICE['standard']['expected']
        assert_close(result, expected, msg="Micro price")

    def test_vpin(self):
        """VPIN should return value between 0 and 1."""
        result = calculate_vpin(TEST_TRADES, 20)
        assert 0.0 <= result <= 1.0


class TestFlow:
    """Test trade flow calculations."""

    def test_trade_flow_golden(self):
        """Trade flow should match golden fixture."""
        result = calculate_trade_flow(TEST_TRADES, 50)
        expected = GOLDEN_TRADE_FLOW['lookback_50']['expected']

        assert isinstance(result, TradeFlowResult)
        assert result.valid == expected['valid']
        # Allow some tolerance for accumulated floating point
        assert abs(result.imbalance - expected['imbalance']) < 0.01

    def test_trade_flow_insufficient(self):
        """Trade flow should handle insufficient trades."""
        result = calculate_trade_flow(TEST_TRADES[:3], 50)
        assert result.valid is False
        assert result.trade_count == 3

    def test_flow_confirmation_buy(self):
        """Flow confirmation for buy should work correctly."""
        # Positive imbalance should confirm buy
        confirmed, data = check_trade_flow_confirmation(0.2, 'buy')
        assert confirmed is True

        # Strong negative imbalance should not confirm buy
        confirmed, data = check_trade_flow_confirmation(-0.3, 'buy')
        assert confirmed is False

    def test_flow_confirmation_short(self):
        """Flow confirmation for short should work correctly."""
        # Negative imbalance should confirm short
        confirmed, data = check_trade_flow_confirmation(-0.2, 'short')
        assert confirmed is True

        # Strong positive imbalance should not confirm short
        confirmed, data = check_trade_flow_confirmation(0.3, 'short')
        assert confirmed is False


class TestTrend:
    """Test trend calculations against golden fixtures."""

    def test_trend_slope_golden(self):
        """Trend slope should match golden fixture."""
        result = calculate_trend_slope(TEST_CANDLES, 20)
        expected = GOLDEN_TREND_SLOPE['period_20']['expected']

        assert isinstance(result, TrendResult)
        assert_close(result.slope_pct, expected, msg="Trend slope(20)")
        assert result.is_trending is True

    def test_trend_slope_insufficient(self):
        """Trend slope should handle insufficient data."""
        result = calculate_trend_slope(TEST_CANDLES[:10], 20)
        assert result.slope_pct == 0.0
        assert result.is_trending is False

    def test_trend_strength(self):
        """Trend strength should detect strong trends."""
        # Strong uptrend
        uptrend = [1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.10]
        is_strong, direction, strength = detect_trend_strength(uptrend, 10, 0.7)
        assert is_strong is True
        assert direction == 'up'

    def test_trailing_stop_long(self):
        """Trailing stop for long should activate correctly."""
        # Entry at 100, highest at 110 (10% profit), activation at 5%
        result = calculate_trailing_stop(
            entry_price=100.0,
            highest_price=110.0,
            lowest_price=None,
            side='long',
            activation_pct=5.0,
            trail_distance_pct=2.0
        )
        assert result is not None
        assert result == 110.0 * 0.98  # 2% below highest

    def test_trailing_stop_not_activated(self):
        """Trailing stop should return None if not activated."""
        result = calculate_trailing_stop(
            entry_price=100.0,
            highest_price=102.0,  # Only 2% profit
            lowest_price=None,
            side='long',
            activation_pct=5.0,  # Requires 5%
            trail_distance_pct=2.0
        )
        assert result is None


class TestGoldenSummary:
    """Verify all golden summary values match new implementation."""

    def test_all_golden_values(self):
        """All golden summary values should match within tolerance."""
        # SMA
        result = calculate_sma(TEST_CLOSES, 20)
        assert_close(result, GOLDEN_SUMMARY['sma_20'], msg="Golden SMA(20)")

        # EMA
        result = calculate_ema(TEST_CLOSES, 21)
        assert_close(result, GOLDEN_SUMMARY['ema_21'], msg="Golden EMA(21)")

        # RSI
        result = calculate_rsi(TEST_CLOSES, 14)
        assert_close(result, GOLDEN_SUMMARY['rsi_14_wilder'], msg="Golden RSI(14)")

        # Volatility
        result = calculate_volatility(TEST_CANDLES, 20)
        assert_close(result, GOLDEN_SUMMARY['volatility_20'], msg="Golden Volatility(20)")

        # ATR Wilder's smoothing
        result = calculate_atr(TEST_CANDLES, 14)
        assert_close(result, GOLDEN_SUMMARY['atr_14_wilder_value'], msg="Golden ATR(14) Wilder")

        # ADX
        result = calculate_adx(TEST_CANDLES, 14)
        assert_close(result, GOLDEN_SUMMARY['adx_14'], msg="Golden ADX(14)")

        # Correlation
        result = calculate_rolling_correlation(TEST_CLOSES, BTC_TEST_CLOSES, 20)
        assert_close(result, GOLDEN_SUMMARY['correlation_20'], msg="Golden Correlation(20)")

        # Micro Price
        result = calculate_micro_price(TEST_ORDERBOOK)
        assert_close(result, GOLDEN_SUMMARY['micro_price'], msg="Golden Micro Price")

        # Trend Slope
        result = calculate_trend_slope(TEST_CANDLES, 20)
        assert_close(result.slope_pct, GOLDEN_SUMMARY['trend_slope_20'], msg="Golden Trend Slope(20)")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
