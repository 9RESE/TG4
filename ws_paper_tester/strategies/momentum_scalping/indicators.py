"""
Momentum Scalping Strategy - Indicator Calculations

Contains functions for calculating EMA, RSI, MACD, and volume metrics.
Based on research from master-plan-v1.0.md:
- RSI period 7 for fast momentum detection
- MACD settings (6, 13, 5) optimized for 1-minute scalping
- EMA 8/21/50 ribbon for trend detection
"""
from typing import List, Tuple, Dict, Any, Optional


def calculate_ema(closes: List[float], period: int) -> Optional[float]:
    """
    Calculate Exponential Moving Average.

    Formula:
    Multiplier = 2 / (Period + 1)
    EMA_today = (Price_today * Multiplier) + (EMA_yesterday * (1 - Multiplier))

    Args:
        closes: List of closing prices (oldest first)
        period: EMA period

    Returns:
        EMA value or None if insufficient data
    """
    if len(closes) < period:
        return None

    # Initialize EMA with SMA for the first 'period' values
    sma = sum(closes[:period]) / period
    multiplier = 2.0 / (period + 1)

    ema = sma
    for price in closes[period:]:
        ema = (price * multiplier) + (ema * (1 - multiplier))

    return ema


def calculate_ema_series(closes: List[float], period: int) -> List[float]:
    """
    Calculate EMA series for the entire price history.

    Args:
        closes: List of closing prices (oldest first)
        period: EMA period

    Returns:
        List of EMA values (same length as input after period-1 values)
    """
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


def calculate_rsi(closes: List[float], period: int = 7) -> Optional[float]:
    """
    Calculate Relative Strength Index.

    Formula:
    RS = Average Gain / Average Loss (over N periods)
    RSI = 100 - (100 / (1 + RS))

    Args:
        closes: List of closing prices (oldest first)
        period: RSI period (default 7 for scalping)

    Returns:
        RSI value (0-100) or None if insufficient data
    """
    if len(closes) < period + 1:
        return None

    # Calculate price changes
    changes = [closes[i] - closes[i - 1] for i in range(1, len(closes))]

    # Initial average gain/loss (first 'period' changes)
    gains = [max(0, change) for change in changes[:period]]
    losses = [max(0, -change) for change in changes[:period]]

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    # Smoothed RSI calculation for remaining periods (Wilder's smoothing)
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


def calculate_rsi_series(closes: List[float], period: int = 7) -> List[float]:
    """
    Calculate RSI series for the entire price history.

    Args:
        closes: List of closing prices (oldest first)
        period: RSI period

    Returns:
        List of RSI values
    """
    if len(closes) < period + 1:
        return []

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


def calculate_macd(
    closes: List[float],
    fast_period: int = 6,
    slow_period: int = 13,
    signal_period: int = 5
) -> Dict[str, Optional[float]]:
    """
    Calculate MACD (Moving Average Convergence Divergence).

    Optimized settings for 1-minute scalping: (6, 13, 5)
    Standard settings: (12, 26, 9)

    Formula:
    MACD Line = EMA(fast) - EMA(slow)
    Signal Line = EMA(MACD Line, signal_period)
    Histogram = MACD Line - Signal Line

    Args:
        closes: List of closing prices (oldest first)
        fast_period: Fast EMA period (default 6)
        slow_period: Slow EMA period (default 13)
        signal_period: Signal line period (default 5)

    Returns:
        Dict with 'macd', 'signal', 'histogram' values
    """
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
    closes: List[float],
    fast_period: int = 6,
    slow_period: int = 13,
    signal_period: int = 5,
    history_length: int = 2
) -> Dict[str, Any]:
    """
    Calculate MACD with recent history for crossover detection.

    Args:
        closes: List of closing prices
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal line period
        history_length: Number of historical values to return

    Returns:
        Dict with current values and history for crossover detection
    """
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


def calculate_volume_ratio(
    volumes: List[float],
    lookback: int = 20
) -> float:
    """
    Calculate current volume ratio vs rolling average.

    Formula:
    Volume Ratio = Current Volume / SMA(Volume, lookback)

    Args:
        volumes: List of volume values (oldest first)
        lookback: Lookback period for average

    Returns:
        Volume ratio (1.0 = average, >1.0 = above average)
    """
    if len(volumes) < lookback:
        return 1.0

    # Calculate rolling average
    avg_volume = sum(volumes[-lookback:]) / lookback

    if avg_volume <= 0:
        return 1.0

    # Current volume is the last value
    current_volume = volumes[-1]

    return current_volume / avg_volume


def calculate_volume_spike(
    volumes: List[float],
    lookback: int = 20,
    recent_count: int = 3
) -> float:
    """
    Calculate volume spike ratio for recent candles.

    Args:
        volumes: List of volume values
        lookback: Lookback period for average
        recent_count: Number of recent candles to check

    Returns:
        Volume spike ratio
    """
    if len(volumes) < lookback + recent_count:
        return 1.0

    # Rolling average excluding recent candles
    avg_volume = sum(volumes[-(lookback + recent_count):-recent_count]) / lookback

    if avg_volume <= 0:
        return 1.0

    # Average of recent volumes
    recent_avg = sum(volumes[-recent_count:]) / recent_count

    return recent_avg / avg_volume


def calculate_volatility(candles, lookback: int = 20) -> float:
    """
    Calculate price volatility from candle closes.

    Returns volatility as a percentage (std dev of returns * 100).

    Args:
        candles: Tuple of candle objects with .close attribute
        lookback: Number of candles for calculation

    Returns:
        Volatility percentage
    """
    if len(candles) < lookback + 1:
        return 0.0

    closes = [c.close for c in candles[-(lookback + 1):]]
    if len(closes) < 2:
        return 0.0

    returns = [(closes[i] - closes[i - 1]) / closes[i - 1]
               for i in range(1, len(closes)) if closes[i - 1] != 0]

    if not returns:
        return 0.0

    mean_return = sum(returns) / len(returns)
    variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
    return (variance ** 0.5) * 100


def calculate_atr(candles, period: int = 14) -> Optional[float]:
    """
    Calculate Average True Range for volatility measurement.

    Args:
        candles: Tuple of candle objects
        period: ATR period

    Returns:
        ATR value or None
    """
    if len(candles) < period + 1:
        return None

    true_ranges = []
    for i in range(1, len(candles)):
        high = candles[i].high
        low = candles[i].low
        prev_close = candles[i - 1].close

        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )
        true_ranges.append(tr)

    if len(true_ranges) < period:
        return None

    # Simple average of recent true ranges
    return sum(true_ranges[-period:]) / period


def check_ema_alignment(
    price: float,
    ema_fast: Optional[float],
    ema_slow: Optional[float],
    ema_filter: Optional[float]
) -> Dict[str, Any]:
    """
    Check EMA ribbon alignment for trend confirmation.

    Args:
        price: Current price
        ema_fast: Fast EMA (8)
        ema_slow: Slow EMA (21)
        ema_filter: Filter EMA (50)

    Returns:
        Dict with alignment status and direction
    """
    result = {
        'bullish_aligned': False,
        'bearish_aligned': False,
        'trend_direction': 'neutral',
        'price_above_filter': None,
        'emas_bullish_order': False,
        'emas_bearish_order': False,
    }

    if None in (ema_fast, ema_slow, ema_filter):
        return result

    # Check if price is above/below the filter EMA
    result['price_above_filter'] = price > ema_filter

    # Check EMA order (bullish: fast > slow > filter)
    result['emas_bullish_order'] = ema_fast > ema_slow > ema_filter
    result['emas_bearish_order'] = ema_fast < ema_slow < ema_filter

    # Bullish alignment: price above filter, EMAs in bullish order
    if price > ema_filter and ema_fast > ema_slow:
        result['bullish_aligned'] = True
        result['trend_direction'] = 'bullish'

    # Bearish alignment: price below filter, EMAs in bearish order
    elif price < ema_filter and ema_fast < ema_slow:
        result['bearish_aligned'] = True
        result['trend_direction'] = 'bearish'

    return result


def check_momentum_signal(
    rsi: Optional[float],
    prev_rsi: Optional[float],
    macd_result: Dict[str, Any],
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Check for momentum entry signals.

    Entry Logic from research:
    - Long: RSI crosses above 30 after being oversold, or momentum above 50 rising
    - Short: RSI crosses below 70 after being overbought, or momentum below 50 falling
    - MACD crossover adds confirmation

    Args:
        rsi: Current RSI value
        prev_rsi: Previous RSI value
        macd_result: MACD calculation result with crossover info
        config: Strategy configuration

    Returns:
        Dict with signal type and confidence
    """
    result = {
        'long_signal': False,
        'short_signal': False,
        'signal_strength': 0.0,
        'rsi_signal': False,
        'macd_signal': False,
        'reasons': [],
    }

    if rsi is None:
        return result

    overbought = config.get('rsi_overbought', 70)
    oversold = config.get('rsi_oversold', 30)

    # RSI-based signals
    if prev_rsi is not None:
        # Long signal: RSI crosses above oversold or rising from mid-range
        if prev_rsi < oversold and rsi >= oversold:
            result['long_signal'] = True
            result['rsi_signal'] = True
            result['signal_strength'] += 0.5
            result['reasons'].append(f"RSI crossed above {oversold}")
        elif 40 < rsi < 60 and rsi > prev_rsi:
            # Momentum continuation in mid-range
            result['long_signal'] = True
            result['rsi_signal'] = True
            result['signal_strength'] += 0.25
            result['reasons'].append("RSI momentum rising in mid-range")

        # Short signal: RSI crosses below overbought or falling from mid-range
        if prev_rsi > overbought and rsi <= overbought:
            result['short_signal'] = True
            result['rsi_signal'] = True
            result['signal_strength'] += 0.5
            result['reasons'].append(f"RSI crossed below {overbought}")
        elif 40 < rsi < 60 and rsi < prev_rsi:
            # Momentum continuation in mid-range
            result['short_signal'] = True
            result['rsi_signal'] = True
            result['signal_strength'] += 0.25
            result['reasons'].append("RSI momentum falling in mid-range")

    # MACD confirmation
    if macd_result.get('bullish_crossover'):
        result['macd_signal'] = True
        if result['long_signal']:
            result['signal_strength'] += 0.25
            result['reasons'].append("MACD bullish crossover confirms")
        else:
            result['long_signal'] = True
            result['signal_strength'] += 0.4
            result['reasons'].append("MACD bullish crossover")

    if macd_result.get('bearish_crossover'):
        result['macd_signal'] = True
        if result['short_signal']:
            result['signal_strength'] += 0.25
            result['reasons'].append("MACD bearish crossover confirms")
        else:
            result['short_signal'] = True
            result['signal_strength'] += 0.4
            result['reasons'].append("MACD bearish crossover")

    # Cap strength at 1.0
    result['signal_strength'] = min(1.0, result['signal_strength'])

    return result


# =============================================================================
# Correlation Calculation (REC-001 v2.0.0)
# =============================================================================
def calculate_correlation(
    candles_a: List,
    candles_b: List,
    lookback: int = 50
) -> Optional[float]:
    """
    Calculate rolling Pearson correlation between two assets' price movements.

    REC-001 (v2.0.0): XRP-BTC correlation has declined from ~0.85 to ~0.40-0.67.
    Momentum signals on XRP/BTC are unreliable when correlation is low.

    Args:
        candles_a: First asset candles (e.g., XRP/USDT)
        candles_b: Second asset candles (e.g., BTC/USDT)
        lookback: Number of candles for correlation calculation

    Returns:
        Correlation coefficient (-1 to +1), None if insufficient data
    """
    if len(candles_a) < lookback + 1 or len(candles_b) < lookback + 1:
        return None

    # Get closes for correlation calculation
    closes_a = [c.close for c in candles_a[-(lookback + 1):]]
    closes_b = [c.close for c in candles_b[-(lookback + 1):]]

    if len(closes_a) != len(closes_b):
        return None

    # Calculate returns
    returns_a = [(closes_a[i] - closes_a[i-1]) / closes_a[i-1]
                 for i in range(1, len(closes_a)) if closes_a[i-1] != 0]
    returns_b = [(closes_b[i] - closes_b[i-1]) / closes_b[i-1]
                 for i in range(1, len(closes_b)) if closes_b[i-1] != 0]

    if len(returns_a) < 2 or len(returns_b) < 2:
        return None

    # Ensure same length
    n = min(len(returns_a), len(returns_b))
    returns_a = returns_a[-n:]
    returns_b = returns_b[-n:]

    # Calculate Pearson correlation
    mean_a = sum(returns_a) / n
    mean_b = sum(returns_b) / n

    # Covariance
    covariance = sum((returns_a[i] - mean_a) * (returns_b[i] - mean_b)
                     for i in range(n)) / n

    # Standard deviations
    std_a = (sum((r - mean_a) ** 2 for r in returns_a) / n) ** 0.5
    std_b = (sum((r - mean_b) ** 2 for r in returns_b) / n) ** 0.5

    if std_a == 0 or std_b == 0:
        return None

    correlation = covariance / (std_a * std_b)

    # Clamp to valid range
    return max(-1.0, min(1.0, correlation))


# =============================================================================
# ADX Calculation (REC-003 v2.0.0)
# =============================================================================
def calculate_adx(candles, period: int = 14) -> Optional[float]:
    """
    Calculate Average Directional Index (ADX) for trend strength measurement.

    REC-003 (v2.0.0): BTC exhibits strong trending behavior at price extremes.
    ADX > 25 indicates strong trend where momentum scalping may fail.

    The ADX measures trend strength (not direction):
    - ADX < 20: Weak trend / ranging
    - ADX 20-25: Trend developing
    - ADX 25-50: Strong trend
    - ADX > 50: Very strong trend

    Args:
        candles: Tuple/list of candle objects with high, low, close
        period: ADX period (default 14)

    Returns:
        ADX value (0-100) or None if insufficient data
    """
    if len(candles) < period * 2:
        return None

    # Calculate True Range, +DM, -DM
    true_ranges = []
    plus_dm = []
    minus_dm = []

    for i in range(1, len(candles)):
        high = candles[i].high
        low = candles[i].low
        prev_high = candles[i - 1].high
        prev_low = candles[i - 1].low
        prev_close = candles[i - 1].close

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

    # Wilder's smoothing for ATR, +DI, -DI
    def wilder_smooth(values: List[float], period: int) -> List[float]:
        """Apply Wilder's smoothing method."""
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

    # Calculate +DI and -DI
    plus_di = []
    minus_di = []
    dx_values = []

    for i in range(len(atr)):
        if atr[i] == 0:
            plus_di.append(0)
            minus_di.append(0)
        else:
            plus_di.append(100 * smooth_plus_dm[i] / atr[i])
            minus_di.append(100 * smooth_minus_dm[i] / atr[i])

        # Calculate DX
        di_sum = plus_di[-1] + minus_di[-1]
        if di_sum == 0:
            dx_values.append(0)
        else:
            dx_values.append(100 * abs(plus_di[-1] - minus_di[-1]) / di_sum)

    if len(dx_values) < period:
        return None

    # Calculate ADX (smoothed DX)
    adx_values = wilder_smooth(dx_values, period)

    if not adx_values:
        return None

    return adx_values[-1]


def check_5m_trend_alignment(
    candles_5m,
    price: float,
    ema_period: int = 50
) -> Dict[str, Any]:
    """
    Check if 5m timeframe confirms 1m trend direction.

    REC-002 (v2.0.0): Multi-timeframe confirmation reduces false signals by ~30%.
    Entry on 1m should align with 5m trend direction.

    Args:
        candles_5m: 5-minute candles
        price: Current price
        ema_period: EMA period for 5m trend filter

    Returns:
        Dict with alignment status
    """
    result = {
        '5m_ema': None,
        '5m_trend': 'neutral',
        'bullish_aligned': False,
        'bearish_aligned': False,
    }

    if len(candles_5m) < ema_period:
        return result

    closes = [c.close for c in candles_5m]
    ema_5m = calculate_ema(closes, ema_period)

    if ema_5m is None:
        return result

    result['5m_ema'] = ema_5m

    if price > ema_5m:
        result['5m_trend'] = 'bullish'
        result['bullish_aligned'] = True
    elif price < ema_5m:
        result['5m_trend'] = 'bearish'
        result['bearish_aligned'] = True

    return result
