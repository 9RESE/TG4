"""
EMA-9 Trend Flip Strategy - Indicators

Technical indicator calculations including EMA, ATR, and candle position logic.
"""
from typing import Dict, List, Any, Optional, Tuple

from ws_tester.types import Candle


def calculate_ema(prices: List[float], period: int) -> Optional[float]:
    """
    Calculate Exponential Moving Average.

    Args:
        prices: List of prices (oldest to newest)
        period: EMA period

    Returns:
        EMA value or None if insufficient data
    """
    if len(prices) < period:
        return None

    # Use only the last 'period * 3' prices for efficiency
    prices = prices[-(period * 3):]

    # Calculate SMA for initial EMA seed
    sma = sum(prices[:period]) / period

    # EMA multiplier
    multiplier = 2 / (period + 1)

    # Calculate EMA
    ema = sma
    for price in prices[period:]:
        ema = (price - ema) * multiplier + ema

    return ema


def calculate_ema_series(prices: List[float], period: int) -> List[float]:
    """
    Calculate EMA series for all prices.

    Args:
        prices: List of prices (oldest to newest)
        period: EMA period

    Returns:
        List of EMA values (same length as input, with None for initial values)
    """
    if len(prices) < period:
        return []

    result = []
    multiplier = 2 / (period + 1)

    # Initial SMA
    sma = sum(prices[:period]) / period
    result.extend([None] * (period - 1))
    result.append(sma)

    # Calculate EMA for remaining prices
    ema = sma
    for price in prices[period:]:
        ema = (price - ema) * multiplier + ema
        result.append(ema)

    return result


def build_hourly_candles(
    candles_1m: Tuple[Candle, ...],
    timeframe_minutes: int = 60
) -> List[Dict[str, Any]]:
    """
    Build higher timeframe candles from 1-minute candles.

    Args:
        candles_1m: Tuple of 1-minute Candle objects
        timeframe_minutes: Target timeframe in minutes (default: 60 for 1H)

    Returns:
        List of candle dicts with timestamp, open, high, low, close, volume
    """
    if not candles_1m:
        return []

    hourly_candles = []
    current_candle = None
    current_hour_key = None

    for candle in candles_1m:
        # Calculate the hour boundary for this candle
        # Floor to the nearest timeframe boundary
        minute_floor = (candle.timestamp.minute // timeframe_minutes) * timeframe_minutes
        hour_key = candle.timestamp.replace(
            minute=minute_floor,
            second=0,
            microsecond=0
        )

        if current_hour_key is None or hour_key != current_hour_key:
            # Save previous candle if exists
            if current_candle is not None:
                hourly_candles.append(current_candle)

            # Start new candle
            current_candle = {
                'timestamp': hour_key,
                'open': candle.open,
                'high': candle.high,
                'low': candle.low,
                'close': candle.close,
                'volume': candle.volume
            }
            current_hour_key = hour_key
        else:
            # Update existing candle
            current_candle['high'] = max(current_candle['high'], candle.high)
            current_candle['low'] = min(current_candle['low'], candle.low)
            current_candle['close'] = candle.close
            current_candle['volume'] += candle.volume

    # Add the last candle if it exists
    if current_candle is not None:
        hourly_candles.append(current_candle)

    return hourly_candles


def calculate_atr(candles: List[Dict[str, Any]], period: int = 14) -> Optional[float]:
    """
    Calculate Average True Range from candle dicts.

    Args:
        candles: List of candle dicts with high, low, close
        period: ATR period

    Returns:
        ATR value or None if insufficient data
    """
    if len(candles) < period + 1:
        return None

    true_ranges = []
    for i in range(1, len(candles)):
        high = candles[i]['high']
        low = candles[i]['low']
        prev_close = candles[i - 1]['close']

        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )
        true_ranges.append(tr)

    # Use simple average for ATR
    if len(true_ranges) < period:
        return None

    return sum(true_ranges[-period:]) / period


def get_candle_position(
    candle: Dict[str, Any],
    ema: float,
    buffer_pct: float,
    use_open: bool = True,
    strict_mode: bool = False
) -> str:
    """
    Determine if candle is above or below EMA with buffer.

    Args:
        candle: Candle dict
        ema: EMA value
        buffer_pct: Buffer percentage (e.g., 0.1 for 0.1%)
        use_open: Use open price if True, close if False (ignored if strict_mode=True)
        strict_mode: If True, require ENTIRE candle (including wicks) to be above/below EMA

    Returns:
        'above', 'below', or 'neutral' if within buffer or crossing EMA
    """
    buffer = ema * (buffer_pct / 100)

    if strict_mode:
        # STRICT MODE: Entire candle must be above/below EMA
        # For 'above': candle low must be above EMA (whole candle above)
        # For 'below': candle high must be below EMA (whole candle below)
        if candle['low'] > ema + buffer:
            return 'above'
        elif candle['high'] < ema - buffer:
            return 'below'
        else:
            return 'neutral'  # Candle crosses or touches EMA
    else:
        # LEGACY MODE: Only check one price point
        price = candle['open'] if use_open else candle['close']
        if price > ema + buffer:
            return 'above'
        elif price < ema - buffer:
            return 'below'
        else:
            return 'neutral'


def get_candle_clearance(
    candle: Dict[str, Any],
    ema: float,
    position: str
) -> float:
    """
    Calculate how far the candle is from the EMA as a percentage.

    For 'above' positions: returns (candle_low - ema) / ema * 100
    For 'below' positions: returns (ema - candle_high) / ema * 100

    Args:
        candle: Candle dict with high, low
        ema: EMA value
        position: 'above' or 'below'

    Returns:
        Clearance percentage (positive = clear of EMA, negative = crossing EMA)
    """
    if position == 'above':
        return (candle['low'] - ema) / ema * 100
    elif position == 'below':
        return (ema - candle['high']) / ema * 100
    else:
        return 0.0


def check_consecutive_positions(
    candles: List[Dict[str, Any]],
    ema_values: List[float],
    n_consecutive: int,
    buffer_pct: float,
    use_open: bool = True,
    strict_mode: bool = False
) -> Tuple[str, int]:
    """
    Check for consecutive candles on same side of EMA.

    Checks the last N candles in the provided array for consistent positioning
    relative to the EMA (all above or all below).

    Args:
        candles: List of hourly candle dicts (should exclude current candle)
        ema_values: List of EMA values (same length as candles)
        n_consecutive: Required consecutive candles
        buffer_pct: Buffer percentage for above/below determination
        use_open: Use open price if True, close if False (ignored if strict_mode=True)
        strict_mode: If True, require entire candle to be above/below EMA

    Returns:
        Tuple of (position: 'above'/'below'/'mixed'/'neutral', count: int)
        - 'above'/'below': All checked candles are consistently on one side
        - 'mixed': Candles are on different sides or insufficient data
        - 'neutral': Most recent candle is within the buffer zone or crosses EMA
    """
    if len(candles) < n_consecutive:
        return ('mixed', 0)

    if len(ema_values) < len(candles):
        return ('mixed', 0)

    # Check the last N candles (the most recent ones in the array)
    # range(-n_consecutive, 0) gives us [-3, -2, -1] for n=3
    positions = []
    for i in range(-n_consecutive, 0):
        candle_idx = i  # Negative index into candles array
        # EMA values array should be same length as candles, so use same index
        ema_idx = i

        candle = candles[candle_idx]
        ema = ema_values[ema_idx]

        if ema is None:
            return ('mixed', 0)

        pos = get_candle_position(candle, ema, buffer_pct, use_open, strict_mode)
        positions.append(pos)

    # Check if all positions are the same (and not neutral)
    if not positions:
        return ('mixed', 0)

    # The most recent position determines the baseline
    last_pos = positions[-1]
    if last_pos == 'neutral':
        return ('neutral', 0)

    # Count consecutive same positions from most recent going backwards
    count = 0
    for pos in reversed(positions):
        if pos == last_pos:
            count += 1
        else:
            break

    if count >= n_consecutive:
        return (last_pos, count)
    else:
        return ('mixed', count)
