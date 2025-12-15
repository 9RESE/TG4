"""
Moving Average Calculations

Contains SMA (Simple Moving Average) and EMA (Exponential Moving Average)
implementations that accept both Candle objects and raw price lists.

Source implementations:
- SMA: mean_reversion/indicators.py:14-19, wavetrend/indicators.py:102-116
- EMA: momentum_scalping/indicators.py:13-67, wavetrend/indicators.py:45-99
"""
from typing import List, Optional

from ._types import PriceInput, MAAlignmentResult, extract_closes


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


def calculate_ma_alignment(
    price: float,
    sma_20: float,
    sma_50: float,
    sma_200: float
) -> MAAlignmentResult:
    """
    Calculate moving average alignment for trend regime classification.

    The alignment of price relative to multiple SMAs indicates trend strength:
    - Perfect Bull: Price > SMA20 > SMA50 > SMA200 (strongest uptrend)
    - Bull: SMA50 > SMA200, price above key SMAs
    - Neutral: No clear alignment
    - Bear: SMA50 < SMA200, price below key SMAs
    - Perfect Bear: Price < SMA20 < SMA50 < SMA200 (strongest downtrend)

    The score ranges from -1.0 (perfect bear) to +1.0 (perfect bull).

    Args:
        price: Current price
        sma_20: 20-period Simple Moving Average
        sma_50: 50-period Simple Moving Average
        sma_200: 200-period Simple Moving Average

    Returns:
        MAAlignmentResult with score, alignment classification, and component flags

    Example:
        >>> result = calculate_ma_alignment(2.35, 2.32, 2.30, 2.25)
        >>> print(f"Alignment: {result.alignment}, Score: {result.score}")
        Alignment: PERFECT_BULL, Score: 1.0
    """
    price_above_sma20 = price > sma_20
    price_above_sma50 = price > sma_50
    price_above_sma200 = price > sma_200
    sma50_above_sma200 = sma_50 > sma_200

    # Perfect Bull: Price > SMA20 > SMA50 > SMA200
    if price > sma_20 > sma_50 > sma_200:
        return MAAlignmentResult(
            score=1.0,
            alignment='PERFECT_BULL',
            price_above_sma20=True,
            price_above_sma50=True,
            price_above_sma200=True,
            sma50_above_sma200=True
        )

    # Perfect Bear: Price < SMA20 < SMA50 < SMA200
    if price < sma_20 < sma_50 < sma_200:
        return MAAlignmentResult(
            score=-1.0,
            alignment='PERFECT_BEAR',
            price_above_sma20=False,
            price_above_sma50=False,
            price_above_sma200=False,
            sma50_above_sma200=False
        )

    # Bull trend (SMA50 > SMA200)
    if sma50_above_sma200:
        if price > sma_50:
            score = 0.7
            alignment = 'BULL'
        elif price > sma_200:
            score = 0.4
            alignment = 'BULL'
        else:
            # Price below SMA200 but golden cross still intact
            score = 0.1
            alignment = 'WEAK_BULL'

        return MAAlignmentResult(
            score=score,
            alignment=alignment,
            price_above_sma20=price_above_sma20,
            price_above_sma50=price_above_sma50,
            price_above_sma200=price_above_sma200,
            sma50_above_sma200=True
        )

    # Bear trend (SMA50 < SMA200)
    if sma_50 < sma_200:
        if price < sma_50:
            score = -0.7
            alignment = 'BEAR'
        elif price < sma_200:
            score = -0.4
            alignment = 'BEAR'
        else:
            # Price above SMA200 but death cross still intact
            score = -0.1
            alignment = 'WEAK_BEAR'

        return MAAlignmentResult(
            score=score,
            alignment=alignment,
            price_above_sma20=price_above_sma20,
            price_above_sma50=price_above_sma50,
            price_above_sma200=price_above_sma200,
            sma50_above_sma200=False
        )

    # Neutral (SMA50 ≈ SMA200)
    return MAAlignmentResult(
        score=0.0,
        alignment='NEUTRAL',
        price_above_sma20=price_above_sma20,
        price_above_sma50=price_above_sma50,
        price_above_sma200=price_above_sma200,
        sma50_above_sma200=sma50_above_sma200
    )


def calculate_ma_alignment_from_data(
    data: PriceInput,
    sma_20_period: int = 20,
    sma_50_period: int = 50,
    sma_200_period: int = 200
) -> Optional[MAAlignmentResult]:
    """
    Calculate moving average alignment directly from price data.

    Convenience function that calculates SMAs internally.

    Args:
        data: Either List/Tuple of Candle objects or List[float] prices
        sma_20_period: Short-term SMA period (default 20)
        sma_50_period: Medium-term SMA period (default 50)
        sma_200_period: Long-term SMA period (default 200)

    Returns:
        MAAlignmentResult or None if insufficient data
    """
    closes = extract_closes(data)

    if len(closes) < sma_200_period:
        return None

    price = closes[-1]
    sma_20 = sum(closes[-sma_20_period:]) / sma_20_period
    sma_50 = sum(closes[-sma_50_period:]) / sma_50_period
    sma_200 = sum(closes[-sma_200_period:]) / sma_200_period

    return calculate_ma_alignment(price, sma_20, sma_50, sma_200)
