"""
Moving Average Calculations

Contains SMA (Simple Moving Average) and EMA (Exponential Moving Average)
implementations that accept both Candle objects and raw price lists.

Source implementations:
- SMA: mean_reversion/indicators.py:14-19, wavetrend/indicators.py:102-116
- EMA: momentum_scalping/indicators.py:13-67, wavetrend/indicators.py:45-99
"""
from typing import List, Optional

from ._types import PriceInput, extract_closes


def calculate_sma(data: PriceInput, period: int) -> Optional[float]:
    """
    Calculate Simple Moving Average.

    SMA = sum(closes[-period:]) / period

    Args:
        data: Either List/Tuple of Candle objects or List[float] prices
        period: SMA period

    Returns:
        SMA value or None if insufficient data

    Example:
        >>> closes = [2.30, 2.31, 2.32, 2.33, 2.34]
        >>> calculate_sma(closes, 3)
        2.33  # (2.32 + 2.33 + 2.34) / 3
    """
    closes = extract_closes(data)

    if len(closes) < period:
        return None

    return sum(closes[-period:]) / period


def calculate_sma_series(data: PriceInput, period: int) -> List[float]:
    """
    Calculate SMA series for the entire price history.

    Returns a rolling SMA for each point where sufficient data exists.

    Args:
        data: Either List/Tuple of Candle objects or List[float] prices
        period: SMA period

    Returns:
        List of SMA values (length = len(data) - period + 1)
    """
    closes = extract_closes(data)

    if len(closes) < period:
        return []

    sma_series = []
    for i in range(period - 1, len(closes)):
        sma = sum(closes[i - period + 1:i + 1]) / period
        sma_series.append(sma)

    return sma_series


def calculate_ema(data: PriceInput, period: int) -> Optional[float]:
    """
    Calculate Exponential Moving Average.

    Formula:
        Multiplier = 2 / (Period + 1)
        EMA_today = (Price_today * Multiplier) + (EMA_yesterday * (1 - Multiplier))

    Initialization: First EMA value is the SMA of the first 'period' values.

    Args:
        data: Either List/Tuple of Candle objects or List[float] prices
        period: EMA period

    Returns:
        EMA value or None if insufficient data

    Example:
        >>> closes = [2.30, 2.31, 2.32, 2.33, 2.34, 2.35]
        >>> ema = calculate_ema(closes, 3)  # Returns latest EMA
    """
    closes = extract_closes(data)

    if len(closes) < period:
        return None

    # Initialize EMA with SMA for the first 'period' values
    sma = sum(closes[:period]) / period
    multiplier = 2.0 / (period + 1)

    ema = sma
    for price in closes[period:]:
        ema = (price * multiplier) + (ema * (1 - multiplier))

    return ema


def calculate_ema_series(data: PriceInput, period: int) -> List[float]:
    """
    Calculate EMA series for the entire price history.

    Args:
        data: Either List/Tuple of Candle objects or List[float] prices
        period: EMA period

    Returns:
        List of EMA values (first value at index 0 is SMA of first 'period' values)
    """
    closes = extract_closes(data)

    if len(closes) < period:
        return []

    # Initialize EMA with SMA for the first 'period' values
    sma = sum(closes[:period]) / period
    multiplier = 2.0 / (period + 1)

    ema_series = [sma]
    ema = sma

    for price in closes[period:]:
        ema = (price * multiplier) + (ema * (1 - multiplier))
        ema_series.append(ema)

    return ema_series
