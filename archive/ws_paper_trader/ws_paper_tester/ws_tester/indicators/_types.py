"""
Indicator Library - Internal Type Definitions

Contains:
- PriceInput: Flexible input type for price data (Candles or List[float])
- Result NamedTuples: Structured return types for complex indicators
  - BollingerResult: Bollinger Bands output (sma, upper, lower, std_dev)
  - ATRResult: ATR with rich output (atr, atr_pct, tr_series)
  - TradeFlowResult: Trade flow analysis (buy/sell volumes, imbalance)
  - TrendResult: Trend slope output (slope_pct, is_trending)
  - CorrelationTrendResult: Correlation trend (slope, is_declining, direction)
- Helper functions: extract_closes(), extract_hlc(), extract_volumes(), is_candle_data()
"""
from typing import Union, List, Tuple, NamedTuple, Optional

# Import Candle type from ws_tester
from ws_tester.types import Candle

# =============================================================================
# FLEXIBLE INPUT TYPE
# =============================================================================
# Allows indicators to accept either Candle objects or raw price lists
PriceInput = Union[List[Candle], Tuple[Candle, ...], List[float]]


# =============================================================================
# RESULT NAMED TUPLES
# =============================================================================
class BollingerResult(NamedTuple):
    """Structured result for Bollinger Bands calculation."""
    sma: Optional[float]
    upper: Optional[float]
    lower: Optional[float]
    std_dev: Optional[float]


class ATRResult(NamedTuple):
    """Structured result for ATR calculation with rich output."""
    atr: Optional[float]
    atr_pct: Optional[float]
    tr_series: List[float]


class TradeFlowResult(NamedTuple):
    """Structured result for trade flow calculation."""
    buy_volume: float
    sell_volume: float
    imbalance: float
    total_volume: float = 0.0
    trade_count: int = 0
    valid: bool = False


class TrendResult(NamedTuple):
    """Structured result for trend slope calculation."""
    slope_pct: float
    is_trending: bool


class CorrelationTrendResult(NamedTuple):
    """Structured result for correlation trend calculation."""
    slope: float
    is_declining: bool
    direction: str  # 'declining', 'stable', 'improving'


class ADXResult(NamedTuple):
    """Structured result for ADX calculation with directional indicators."""
    adx: float
    plus_di: float
    minus_di: float
    trend_strength: str  # 'ABSENT', 'WEAK', 'EMERGING', 'STRONG', 'VERY_STRONG'


class MAAlignmentResult(NamedTuple):
    """Structured result for moving average alignment analysis."""
    score: float  # -1.0 to +1.0
    alignment: str  # 'PERFECT_BULL', 'BULL', 'NEUTRAL', 'BEAR', 'PERFECT_BEAR'
    price_above_sma20: bool
    price_above_sma50: bool
    price_above_sma200: bool
    sma50_above_sma200: bool


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def extract_closes(data: PriceInput) -> List[float]:
    """
    Extract closing prices from candles or return price list directly.

    Handles both Candle objects and raw List[float] inputs, allowing
    indicator functions to accept either format.

    Args:
        data: Either List/Tuple of Candle objects or List[float] prices

    Returns:
        List of closing prices (floats)

    Example:
        >>> candles = [Candle(..., close=2.30), Candle(..., close=2.31)]
        >>> extract_closes(candles)
        [2.30, 2.31]

        >>> prices = [2.30, 2.31, 2.32]
        >>> extract_closes(prices)
        [2.30, 2.31, 2.32]
    """
    if not data:
        return []

    # Check if first element is a Candle (has .close attribute)
    first = data[0]
    if hasattr(first, 'close'):
        return [c.close for c in data]

    # Already a list of floats
    return list(data)


def extract_hlc(data: PriceInput) -> Tuple[List[float], List[float], List[float]]:
    """
    Extract High, Low, Close prices from candles.

    For raw float lists, returns (data, data, data) as approximation.

    Args:
        data: Either List/Tuple of Candle objects or List[float] prices

    Returns:
        Tuple of (highs, lows, closes) lists

    Example:
        >>> candles = [Candle(..., high=2.32, low=2.28, close=2.30)]
        >>> highs, lows, closes = extract_hlc(candles)
        >>> highs[0], lows[0], closes[0]
        (2.32, 2.28, 2.30)
    """
    if not data:
        return [], [], []

    first = data[0]
    if hasattr(first, 'close') and hasattr(first, 'high') and hasattr(first, 'low'):
        highs = [c.high for c in data]
        lows = [c.low for c in data]
        closes = [c.close for c in data]
        return highs, lows, closes

    # Raw float list - use as all three
    prices = list(data)
    return prices, prices, prices


def extract_volumes(data: PriceInput) -> List[float]:
    """
    Extract volumes from candles.

    For raw float lists, returns empty list.

    Args:
        data: Either List/Tuple of Candle objects or List[float] prices

    Returns:
        List of volume values, or empty list if input is raw prices
    """
    if not data:
        return []

    first = data[0]
    if hasattr(first, 'volume'):
        return [c.volume for c in data]

    return []


def is_candle_data(data: PriceInput) -> bool:
    """
    Check if input is candle data (has OHLCV attributes).

    Args:
        data: Input data to check

    Returns:
        True if data contains Candle objects, False if raw floats
    """
    if not data:
        return False

    first = data[0]
    return hasattr(first, 'close') and hasattr(first, 'high') and hasattr(first, 'low')
