"""
Oscillator Calculations

Contains RSI, ADX, and MACD implementations.

Source implementations:
- RSI: momentum_scalping/indicators.py:70-162, grid_rsi_reversion/indicators.py:12-54
- ADX: mean_reversion/indicators.py:204-310, momentum_scalping/indicators.py:654-760
- MACD: momentum_scalping/indicators.py:165-310

Note: RSI uses Wilder's smoothing method (industry standard) per user decision.
"""
from typing import List, Dict, Any, Optional

from ._types import PriceInput, ADXResult, extract_closes, extract_hlc
from .moving_averages import calculate_ema_series


def calculate_rsi(data: PriceInput, period: int = 14) -> Optional[float]:
    """
    Calculate Relative Strength Index using Wilder's smoothing method.

    Formula:
        RS = Average Gain / Average Loss (with Wilder's smoothing)
        RSI = 100 - (100 / (1 + RS))

    Wilder's smoothing:
        avg_gain = (prev_avg_gain * (period - 1) + current_gain) / period
        avg_loss = (prev_avg_loss * (period - 1) + current_loss) / period

    Args:
        data: Either List/Tuple of Candle objects or List[float] prices
        period: RSI period (default 14, scalping often uses 7)

    Returns:
        RSI value (0-100) or None if insufficient data
    """
    closes = extract_closes(data)

    if len(closes) < period + 1:
        return None

    # Convert to float to handle Decimal types from database
    closes = [float(c) for c in closes]

    # Calculate price changes
    changes = [closes[i] - closes[i - 1] for i in range(1, len(closes))]

    # Initial average gain/loss (first 'period' changes)
    gains = [max(0, change) for change in changes[:period]]
    losses = [max(0, -change) for change in changes[:period]]

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    # Wilder's smoothing for remaining periods
    for change in changes[period:]:
        gain = max(0, change)
        loss = max(0, -change)

        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))

    return rsi


def calculate_rsi_series(data: PriceInput, period: int = 14) -> List[float]:
    """
    Calculate RSI series for the entire price history.

    Args:
        data: Either List/Tuple of Candle objects or List[float] prices
        period: RSI period

    Returns:
        List of RSI values
    """
    closes = extract_closes(data)

    if len(closes) < period + 1:
        return []

    # Convert to float to handle Decimal types from database
    closes = [float(c) for c in closes]

    # Calculate price changes
    changes = [closes[i] - closes[i - 1] for i in range(1, len(closes))]

    # Initial average gain/loss
    gains = [max(0, change) for change in changes[:period]]
    losses = [max(0, -change) for change in changes[:period]]

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    rsi_series = []

    # First RSI value
    if avg_loss == 0:
        rsi_series.append(100.0 if avg_gain > 0 else 50.0)
    else:
        rs = avg_gain / avg_loss
        rsi_series.append(100.0 - (100.0 / (1.0 + rs)))

    # Calculate RSI for remaining periods
    for change in changes[period:]:
        gain = max(0, change)
        loss = max(0, -change)

        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

        if avg_loss == 0:
            rsi_series.append(100.0 if avg_gain > 0 else 50.0)
        else:
            rs = avg_gain / avg_loss
            rsi_series.append(100.0 - (100.0 / (1.0 + rs)))

    return rsi_series


def calculate_adx(data: PriceInput, period: int = 14) -> Optional[float]:
    """
    Calculate Average Directional Index for trend strength.

    ADX measures trend strength (not direction):
    - ADX < 20: Weak trend / ranging
    - ADX 20-25: Trend developing
    - ADX 25-50: Strong trend
    - ADX > 50: Very strong trend

    Uses Wilder's smoothing method.

    Args:
        data: Must be Candle data with high, low, close attributes
        period: ADX period (default 14)

    Returns:
        ADX value (0-100) or None if insufficient data
    """
    highs, lows, closes = extract_hlc(data)

    if len(closes) < period * 2:
        return None

    # Convert to float to handle Decimal types from database
    highs = [float(h) for h in highs]
    lows = [float(l) for l in lows]
    closes = [float(c) for c in closes]

    # Calculate True Range, +DM, -DM
    true_ranges = []
    plus_dm = []
    minus_dm = []

    for i in range(1, len(closes)):
        high = highs[i]
        low = lows[i]
        prev_high = highs[i - 1]
        prev_low = lows[i - 1]
        prev_close = closes[i - 1]

        # True Range
        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )
        true_ranges.append(tr)

        # Directional Movement
        up_move = high - prev_high
        down_move = prev_low - low

        if up_move > down_move and up_move > 0:
            plus_dm.append(up_move)
        else:
            plus_dm.append(0)

        if down_move > up_move and down_move > 0:
            minus_dm.append(down_move)
        else:
            minus_dm.append(0)

    if len(true_ranges) < period:
        return None

    # Wilder's smoothing
    def wilder_smooth(values: List[float], period: int) -> List[float]:
        if len(values) < period:
            return []
        smoothed = [sum(values[:period]) / period]
        for i in range(period, len(values)):
            smoothed.append((smoothed[-1] * (period - 1) + values[i]) / period)
        return smoothed

    atr = wilder_smooth(true_ranges, period)
    smooth_plus_dm = wilder_smooth(plus_dm, period)
    smooth_minus_dm = wilder_smooth(minus_dm, period)

    if not atr or not smooth_plus_dm or not smooth_minus_dm:
        return None

    # Calculate +DI and -DI, then DX
    dx_values = []
    for i in range(len(atr)):
        if atr[i] == 0:
            dx_values.append(0)
        else:
            plus_di = 100 * smooth_plus_dm[i] / atr[i]
            minus_di = 100 * smooth_minus_dm[i] / atr[i]

            di_sum = plus_di + minus_di
            if di_sum == 0:
                dx_values.append(0)
            else:
                dx_values.append(100 * abs(plus_di - minus_di) / di_sum)

    if len(dx_values) < period:
        return None

    # Calculate ADX (smoothed DX)
    adx_values = wilder_smooth(dx_values, period)

    if not adx_values:
        return None

    return adx_values[-1]


def calculate_adx_with_di(data: PriceInput, period: int = 14) -> Optional[ADXResult]:
    """
    Calculate Average Directional Index with Directional Indicators (+DI/-DI).

    This enhanced version returns the full ADX analysis including:
    - ADX value (trend strength, 0-100)
    - +DI (positive directional indicator)
    - -DI (negative directional indicator)
    - Trend strength classification

    ADX interpretation:
    - ADX < 15: Absent trend (ABSENT)
    - ADX 15-20: Weak trend (WEAK)
    - ADX 20-25: Emerging trend (EMERGING)
    - ADX 25-40: Strong trend (STRONG)
    - ADX > 40: Very strong trend (VERY_STRONG)

    Directional interpretation:
    - +DI > -DI: Bullish trend direction
    - -DI > +DI: Bearish trend direction

    Args:
        data: Must be Candle data with high, low, close attributes
        period: ADX period (default 14)

    Returns:
        ADXResult with (adx, plus_di, minus_di, trend_strength) or None if insufficient data

    Example:
        >>> result = calculate_adx_with_di(candles)
        >>> if result and result.adx > 25 and result.plus_di > result.minus_di:
        ...     print("Strong bullish trend")
    """
    highs, lows, closes = extract_hlc(data)

    if len(closes) < period * 2:
        return None

    # Convert to float to handle Decimal types from database
    highs = [float(h) for h in highs]
    lows = [float(l) for l in lows]
    closes = [float(c) for c in closes]

    # Calculate True Range, +DM, -DM
    true_ranges: List[float] = []
    plus_dm: List[float] = []
    minus_dm: List[float] = []

    for i in range(1, len(closes)):
        high = highs[i]
        low = lows[i]
        prev_high = highs[i - 1]
        prev_low = lows[i - 1]
        prev_close = closes[i - 1]

        # True Range
        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )
        true_ranges.append(tr)

        # Directional Movement
        up_move = high - prev_high
        down_move = prev_low - low

        if up_move > down_move and up_move > 0:
            plus_dm.append(up_move)
        else:
            plus_dm.append(0)

        if down_move > up_move and down_move > 0:
            minus_dm.append(down_move)
        else:
            minus_dm.append(0)

    if len(true_ranges) < period:
        return None

    # Wilder's smoothing
    def wilder_smooth(values: List[float], period: int) -> List[float]:
        if len(values) < period:
            return []
        smoothed = [sum(values[:period]) / period]
        for i in range(period, len(values)):
            smoothed.append((smoothed[-1] * (period - 1) + values[i]) / period)
        return smoothed

    atr = wilder_smooth(true_ranges, period)
    smooth_plus_dm = wilder_smooth(plus_dm, period)
    smooth_minus_dm = wilder_smooth(minus_dm, period)

    if not atr or not smooth_plus_dm or not smooth_minus_dm:
        return None

    # Calculate +DI, -DI, and DX series
    plus_di_series: List[float] = []
    minus_di_series: List[float] = []
    dx_values: List[float] = []

    for i in range(len(atr)):
        if atr[i] == 0:
            plus_di_series.append(0)
            minus_di_series.append(0)
            dx_values.append(0)
        else:
            plus_di = 100 * smooth_plus_dm[i] / atr[i]
            minus_di = 100 * smooth_minus_dm[i] / atr[i]
            plus_di_series.append(plus_di)
            minus_di_series.append(minus_di)

            di_sum = plus_di + minus_di
            if di_sum == 0:
                dx_values.append(0)
            else:
                dx_values.append(100 * abs(plus_di - minus_di) / di_sum)

    if len(dx_values) < period:
        return None

    # Calculate ADX (smoothed DX)
    adx_values = wilder_smooth(dx_values, period)

    if not adx_values:
        return None

    adx = adx_values[-1]
    plus_di = plus_di_series[-1]
    minus_di = minus_di_series[-1]

    # Classify trend strength
    if adx < 15:
        trend_strength = 'ABSENT'
    elif adx < 20:
        trend_strength = 'WEAK'
    elif adx < 25:
        trend_strength = 'EMERGING'
    elif adx < 40:
        trend_strength = 'STRONG'
    else:
        trend_strength = 'VERY_STRONG'

    return ADXResult(
        adx=adx,
        plus_di=plus_di,
        minus_di=minus_di,
        trend_strength=trend_strength
    )


def calculate_macd(
    data: PriceInput,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> Dict[str, Optional[float]]:
    """
    Calculate MACD (Moving Average Convergence Divergence).

    Formula:
        MACD Line = EMA(fast) - EMA(slow)
        Signal Line = EMA(MACD Line, signal_period)
        Histogram = MACD Line - Signal Line

    Default settings (12, 26, 9) are standard.
    Scalping often uses (6, 13, 5) for faster signals.

    Args:
        data: Either List/Tuple of Candle objects or List[float] prices
        fast_period: Fast EMA period (default 12)
        slow_period: Slow EMA period (default 26)
        signal_period: Signal line period (default 9)

    Returns:
        Dict with 'macd', 'signal', 'histogram' values
    """
    closes = extract_closes(data)
    result = {'macd': None, 'signal': None, 'histogram': None}

    if len(closes) < slow_period + signal_period:
        return result

    # Calculate fast and slow EMAs
    fast_ema_series = calculate_ema_series(closes, fast_period)
    slow_ema_series = calculate_ema_series(closes, slow_period)

    if not fast_ema_series or not slow_ema_series:
        return result

    # Align series (slow EMA starts later)
    offset = slow_period - fast_period
    if offset < 0 or offset >= len(fast_ema_series):
        return result

    aligned_fast = fast_ema_series[offset:]
    macd_line = [f - s for f, s in zip(aligned_fast, slow_ema_series)]

    if len(macd_line) < signal_period:
        return result

    # Calculate signal line (EMA of MACD line)
    signal_ema_series = calculate_ema_series(macd_line, signal_period)

    if not signal_ema_series:
        return result

    # Get current values
    result['macd'] = macd_line[-1]
    result['signal'] = signal_ema_series[-1]
    result['histogram'] = result['macd'] - result['signal']

    return result


def calculate_macd_with_history(
    data: PriceInput,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    history_length: int = 2
) -> Dict[str, Any]:
    """
    Calculate MACD with recent history for crossover detection.

    Args:
        data: Either List/Tuple of Candle objects or List[float] prices
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal line period
        history_length: Number of historical values to return

    Returns:
        Dict with current values, history, and crossover flags
    """
    closes = extract_closes(data)
    result = {
        'macd': None,
        'signal': None,
        'histogram': None,
        'macd_history': [],
        'signal_history': [],
        'histogram_history': [],
        'bullish_crossover': False,
        'bearish_crossover': False,
    }

    if len(closes) < slow_period + signal_period + history_length:
        return result

    # Calculate full series
    fast_ema_series = calculate_ema_series(closes, fast_period)
    slow_ema_series = calculate_ema_series(closes, slow_period)

    if not fast_ema_series or not slow_ema_series:
        return result

    offset = slow_period - fast_period
    if offset < 0 or offset >= len(fast_ema_series):
        return result

    aligned_fast = fast_ema_series[offset:]
    macd_line = [f - s for f, s in zip(aligned_fast, slow_ema_series)]

    if len(macd_line) < signal_period + history_length:
        return result

    signal_ema_series = calculate_ema_series(macd_line, signal_period)

    if not signal_ema_series or len(signal_ema_series) < history_length:
        return result

    # Get current and historical values
    result['macd'] = macd_line[-1]
    result['signal'] = signal_ema_series[-1]
    result['histogram'] = result['macd'] - result['signal']

    # History (last N values)
    hist_offset = signal_period - 1
    macd_for_history = macd_line[hist_offset:]
    result['macd_history'] = macd_for_history[-history_length:]
    result['signal_history'] = signal_ema_series[-history_length:]
    result['histogram_history'] = [
        m - s for m, s in zip(result['macd_history'], result['signal_history'])
    ]

    # Detect crossovers
    if len(result['histogram_history']) >= 2:
        prev_hist = result['histogram_history'][-2]
        curr_hist = result['histogram_history'][-1]

        # Bullish crossover: histogram goes from negative to positive
        if prev_hist < 0 and curr_hist >= 0:
            result['bullish_crossover'] = True
        # Bearish crossover: histogram goes from positive to negative
        elif prev_hist > 0 and curr_hist <= 0:
            result['bearish_crossover'] = True

    return result
