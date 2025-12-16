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
    use_open: bool = True
) -> str:
    """
    Determine if candle is above or below EMA with buffer.

    Args:
        candle: Candle dict
        ema: EMA value
        buffer_pct: Buffer percentage (e.g., 0.1 for 0.1%)
        use_open: Use open price if True, close if False

    Returns:
        'above', 'below', or 'neutral' if within buffer
    """
    price = candle['open'] if use_open else candle['close']
    buffer = ema * (buffer_pct / 100)

    if price > ema + buffer:
        return 'above'
    elif price < ema - buffer:
        return 'below'
    else:
        return 'neutral'


def check_consecutive_positions(
    candles: List[Dict[str, Any]],
    ema_values: List[float],
    n_consecutive: int,
    buffer_pct: float,
    use_open: bool = True
) -> Tuple[str, int]:
    """
    Check for consecutive candles opening on same side of EMA.

    Args:
        candles: List of hourly candle dicts
        ema_values: List of EMA values
        n_consecutive: Required consecutive candles
        buffer_pct: Buffer percentage
        use_open: Use open price

    Returns:
        Tuple of (position: 'above'/'below'/'mixed', count: int)
    """
    if len(candles) < n_consecutive or len(ema_values) < len(candles):
        return ('mixed', 0)

    # Check the last N candles before current (excluding current)
    positions = []
    for i in range(-n_consecutive - 1, -1):
        if i + len(candles) < 0:
            continue
        candle = candles[i]
        ema = ema_values[i + len(ema_values) - len(candles)]
        if ema is None:
            return ('mixed', 0)
        pos = get_candle_position(candle, ema, buffer_pct, use_open)
        positions.append(pos)

    # Check if all positions are the same (and not neutral)
    if not positions:
        return ('mixed', 0)

    # Count consecutive same positions from most recent
    last_pos = positions[-1]
    if last_pos == 'neutral':
        return ('neutral', 0)

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
