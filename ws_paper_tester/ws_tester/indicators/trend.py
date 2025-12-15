"""
Trend Calculations

Contains trend slope detection, trend strength, and trailing stop logic.

Source implementations:
- Trend slope: mean_reversion/indicators.py:105-136, market_making/calculations.py:335-381
- Trend strength: ratio_trading/indicators.py:112-153
- Trailing stop: ratio_trading/indicators.py:156-180, market_making/calculations.py:121-152
"""
from typing import List, Optional, Tuple

from ._types import PriceInput, TrendResult, extract_closes


def calculate_trend_slope(data: PriceInput, period: int = 20) -> TrendResult:
    """
    Calculate price trend using linear regression slope.

    Uses simple linear regression on closing prices to determine
    trend direction and strength.

    Args:
        data: Either List/Tuple of Candle objects or List[float] prices
        period: Number of periods to analyze

    Returns:
        TrendResult(slope_pct, is_trending)
        - slope_pct: Price change per candle as percentage
        - is_trending: True if sufficient data was available
    """
    closes = extract_closes(data)

    if len(closes) < period:
        return TrendResult(slope_pct=0.0, is_trending=False)

    closes = closes[-period:]
    if len(closes) < 2:
        return TrendResult(slope_pct=0.0, is_trending=False)

    # Simple linear regression: y = mx + b
    n = len(closes)
    sum_x = sum(range(n))
    sum_y = sum(closes)
    sum_xy = sum(i * closes[i] for i in range(n))
    sum_x2 = sum(i * i for i in range(n))

    denominator = n * sum_x2 - sum_x * sum_x
    if denominator == 0:
        return TrendResult(slope_pct=0.0, is_trending=False)

    slope = (n * sum_xy - sum_x * sum_y) / denominator

    # Convert to percentage change per candle
    avg_price = sum_y / n if n > 0 else 1.0
    slope_pct = (slope / avg_price) * 100 if avg_price > 0 else 0.0

    return TrendResult(slope_pct=slope_pct, is_trending=True)


def detect_trend_strength(
    prices: List[float],
    lookback: int = 10,
    threshold: float = 0.7
) -> Tuple[bool, str, float]:
    """
    Detect if there's a strong trend in recent price action.

    Counts directional moves to determine trend consistency.

    Args:
        prices: List of closing prices (oldest first)
        lookback: Number of candles for comparison
        threshold: Minimum ratio to be considered strong trend (0-1)

    Returns:
        Tuple of (is_strong_trend, direction, strength)
        - is_strong_trend: True if trend is strong enough
        - direction: 'up', 'down', or 'neutral'
        - strength: 0.0 to 1.0 (% of candles in same direction)
    """
    if len(prices) < lookback + 1:
        return False, 'neutral', 0.0

    recent = prices[-(lookback + 1):]
    up_moves = 0
    down_moves = 0

    for i in range(1, len(recent)):
        if recent[i] > recent[i - 1]:
            up_moves += 1
        elif recent[i] < recent[i - 1]:
            down_moves += 1

    total_moves = up_moves + down_moves
    if total_moves == 0:
        return False, 'neutral', 0.0

    up_strength = up_moves / total_moves
    down_strength = down_moves / total_moves

    if up_strength >= threshold:
        return True, 'up', up_strength
    elif down_strength >= threshold:
        return True, 'down', down_strength

    return False, 'neutral', max(up_strength, down_strength)


def calculate_trailing_stop(
    entry_price: float,
    highest_price: float,
    lowest_price: Optional[float],
    side: str,
    activation_pct: float,
    trail_distance_pct: float
) -> Optional[float]:
    """
    Calculate trailing stop price.

    Trailing stop activates once profit reaches activation_pct,
    then trails at trail_distance_pct from the best price.

    Args:
        entry_price: Original entry price
        highest_price: Highest price since entry (for longs)
        lowest_price: Lowest price since entry (for shorts, can be None)
        side: 'long' or 'short'
        activation_pct: Minimum profit % to activate trailing
        trail_distance_pct: Distance from high/low to trail

    Returns:
        Trailing stop price or None if not activated
    """
    if side == 'long':
        # Long: profit when price increases
        profit_pct = (highest_price - entry_price) / entry_price * 100
        if profit_pct >= activation_pct:
            return highest_price * (1 - trail_distance_pct / 100)

    elif side == 'short':
        # Short: profit when price decreases
        # For shorts, we track lowest_price (or use highest_price if lowest not available)
        reference_price = lowest_price if lowest_price is not None else highest_price
        profit_pct = (entry_price - reference_price) / entry_price * 100
        if profit_pct >= activation_pct:
            return reference_price * (1 + trail_distance_pct / 100)

    return None
