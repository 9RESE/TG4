"""
Choppiness Index Calculation

The Choppiness Index is designed to identify consolidating/ranging markets
vs trending markets. It measures the "choppiness" of price action.

Formula:
    CHOP = 100 * LOG10(SUM(ATR, n) / (HIGH_n - LOW_n)) / LOG10(n)

Interpretation:
    - > 61.8: Choppy/Sideways market (consolidation)
    - < 38.2: Trending market (strong directional movement)
    - 38.2 - 61.8: Transitional zone

References:
    - LuxAlgo: https://www.luxalgo.com/blog/choppiness-index-quantifying-consolidation/
"""
import logging
import math
from typing import Optional, List

from ._types import PriceInput, extract_hlc

logger = logging.getLogger(__name__)


def calculate_choppiness(data: PriceInput, period: int = 14) -> Optional[float]:
    """
    Calculate Choppiness Index.

    The Choppiness Index measures whether the market is trending or ranging.
    It is bounded between 0 and 100.

    Formula:
        CHOP = 100 * LOG10(SUM(ATR_1, period) / (Highest High - Lowest Low)) / LOG10(period)

    Args:
        data: Must be Candle data with high, low, close attributes
        period: Choppiness period (default 14)

    Returns:
        Choppiness Index value (0-100) or None if insufficient data
        - > 61.8: Choppy/ranging market
        - < 38.2: Trending market
        - 38.2-61.8: Transitional

    Example:
        >>> chop = calculate_choppiness(candles, period=14)
        >>> if chop and chop > 61.8:
        ...     print("Market is ranging - avoid trend strategies")
    """
    highs, lows, closes = extract_hlc(data)

    if len(closes) < period + 1:
        return None

    # Calculate True Range series for each bar
    tr_series: List[float] = []
    for i in range(1, len(closes)):
        high = highs[i]
        low = lows[i]
        prev_close = closes[i - 1]

        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )
        tr_series.append(tr)

    if len(tr_series) < period:
        return None

    # Get the last 'period' True Range values
    tr_sum = sum(tr_series[-period:])

    # Get the highest high and lowest low over the period
    period_highs = highs[-period:]
    period_lows = lows[-period:]
    highest_high = max(period_highs)
    lowest_low = min(period_lows)
    range_hl = highest_high - lowest_low

    # Handle edge case of zero range (flat market)
    if range_hl == 0:
        logger.debug(
            f"Zero price range detected over {period} periods - "
            "returning neutral chop value (50.0). This indicates a flat/stale market."
        )
        return 50.0  # Neutral default for flat market

    # Calculate Choppiness Index
    # CHOP = 100 * LOG10(ATR_SUM / HL_RANGE) / LOG10(period)
    try:
        chop = 100 * math.log10(tr_sum / range_hl) / math.log10(period)
    except (ValueError, ZeroDivisionError):
        return 50.0  # Neutral default on math errors

    # Clamp to 0-100 range
    return max(0.0, min(100.0, chop))


def calculate_choppiness_series(data: PriceInput, period: int = 14) -> List[float]:
    """
    Calculate Choppiness Index series for the entire price history.

    Args:
        data: Must be Candle data with high, low, close attributes
        period: Choppiness period (default 14)

    Returns:
        List of Choppiness Index values
    """
    highs, lows, closes = extract_hlc(data)

    if len(closes) < period + 1:
        return []

    # Calculate True Range series
    tr_series: List[float] = []
    for i in range(1, len(closes)):
        high = highs[i]
        low = lows[i]
        prev_close = closes[i - 1]

        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )
        tr_series.append(tr)

    # Calculate Choppiness for each valid window
    chop_series: List[float] = []
    for i in range(period - 1, len(tr_series)):
        # Sum of TR over window
        tr_sum = sum(tr_series[i - period + 1:i + 1])

        # High-Low range over window (offset by 1 because TR starts at index 1)
        window_start = i - period + 2  # +2 because highs/lows are full length
        window_end = i + 2
        period_highs = highs[window_start:window_end]
        period_lows = lows[window_start:window_end]

        if not period_highs or not period_lows:
            chop_series.append(50.0)
            continue

        highest_high = max(period_highs)
        lowest_low = min(period_lows)
        range_hl = highest_high - lowest_low

        if range_hl == 0:
            chop_series.append(50.0)
            continue

        try:
            chop = 100 * math.log10(tr_sum / range_hl) / math.log10(period)
            chop_series.append(max(0.0, min(100.0, chop)))
        except (ValueError, ZeroDivisionError):
            chop_series.append(50.0)

    return chop_series


def is_choppy(chop_value: float, threshold: float = 61.8) -> bool:
    """
    Check if market is in a choppy/ranging state.

    Args:
        chop_value: Choppiness Index value
        threshold: Threshold for choppy classification (default 61.8)

    Returns:
        True if market is choppy/ranging
    """
    return chop_value > threshold


def is_trending_chop(chop_value: float, threshold: float = 38.2) -> bool:
    """
    Check if market is in a trending state based on Choppiness Index.

    Args:
        chop_value: Choppiness Index value
        threshold: Threshold for trending classification (default 38.2)

    Returns:
        True if market is trending
    """
    return chop_value < threshold


def get_choppiness_state(chop_value: float) -> str:
    """
    Get the market state classification based on Choppiness Index.

    Args:
        chop_value: Choppiness Index value

    Returns:
        State string: 'CHOPPY', 'TRENDING', or 'TRANSITIONAL'
    """
    if chop_value > 61.8:
        return 'CHOPPY'
    elif chop_value < 38.2:
        return 'TRENDING'
    else:
        return 'TRANSITIONAL'
