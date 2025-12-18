"""
Mean Reversion Strategy - Indicator Calculations

Contains strategy-specific indicator functions for mean reversion analysis.
Common indicators are imported from the centralized ws_tester.indicators library.

Technical indicators:
- Simple Moving Average (SMA)
- Relative Strength Index (RSI)
- Bollinger Bands
- Volatility (standard deviation of returns)
- ADX (Average Directional Index) - v4.3.0 REC-003
"""
from typing import List, Tuple, Optional

# Import common indicators from centralized library
from ws_tester.indicators import (
    calculate_sma as _calculate_sma_lib,
    calculate_rsi as _calculate_rsi_lib,
    calculate_bollinger_bands as _calculate_bollinger_bands_lib,
    calculate_volatility,
    calculate_trend_slope as _calculate_trend_slope_lib,
    calculate_rolling_correlation,
    calculate_adx,
    BollingerResult,
    TrendResult,
)


def calculate_sma(candles: List, period: int) -> float:
    """
    Calculate simple moving average.

    Thin wrapper around ws_tester.indicators.calculate_sma that
    returns 0.0 instead of None for backward compatibility.
    """
    result = _calculate_sma_lib(candles, period)
    return result if result is not None else 0.0


def calculate_rsi(candles: List, period: int = 14) -> float:
    """
    Calculate RSI indicator using Wilder's smoothing.

    Thin wrapper around ws_tester.indicators.calculate_rsi.
    Note: The library uses Wilder's smoothing (industry standard).
    """
    return _calculate_rsi_lib(candles, period)


def calculate_bollinger_bands(
    candles: List,
    period: int = 20,
    std_dev: float = 2.0
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Calculate Bollinger Bands.

    Thin wrapper around ws_tester.indicators.calculate_bollinger_bands that
    returns (lower, sma, upper) tuple for backward compatibility.
    """
    result = _calculate_bollinger_bands_lib(candles, period, std_dev)
    return result.lower, result.sma, result.upper


def calculate_trend_slope(candles: List, period: int = 50) -> float:
    """
    Calculate the slope of price trend over given period.

    Returns slope as percentage change per candle.
    Positive = uptrend, Negative = downtrend, Near zero = ranging.

    Thin wrapper around ws_tester.indicators.calculate_trend_slope
    that returns just the slope_pct value for backward compatibility.
    """
    result = _calculate_trend_slope_lib(candles, period)
    return result.slope_pct


def calculate_correlation(
    xrp_candles: List,
    btc_candles: List,
    lookback: int = 50
) -> Optional[float]:
    """
    Calculate rolling Pearson correlation between XRP and BTC price movements.

    REC-005 (v4.0.0): Added for XRP/BTC ratio trading analysis.
    Research shows XRP correlation with BTC declining (24.86% over 90 days),
    which affects ratio mean reversion timing.

    Uses ws_tester.indicators.calculate_rolling_correlation with closes
    extracted from candles.

    Args:
        xrp_candles: XRP/USDT candles
        btc_candles: BTC/USDT candles
        lookback: Number of candles for correlation

    Returns:
        Correlation coefficient (-1 to +1), None if insufficient data
    """
    if len(xrp_candles) < lookback + 1 or len(btc_candles) < lookback + 1:
        return None

    # Extract closes for the library function
    xrp_closes = [c.close for c in xrp_candles]
    btc_closes = [c.close for c in btc_candles]

    return calculate_rolling_correlation(xrp_closes, btc_closes, lookback)


# Re-export for backward compatibility
__all__ = [
    'calculate_sma',
    'calculate_rsi',
    'calculate_bollinger_bands',
    'calculate_volatility',
    'calculate_trend_slope',
    'calculate_correlation',
    'calculate_adx',
]
