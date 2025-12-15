"""
Grid RSI Reversion Strategy - Indicator Calculations

Contains functions for calculating RSI, ATR, and adaptive RSI zones.
Implements the RSI confidence calculation from legacy code.
"""
from typing import Dict, Any, List, Optional, Tuple

from .config import RSIZone, get_symbol_config


def calculate_rsi(closes: List[float], period: int = 14) -> Optional[float]:
    """
    Calculate Relative Strength Index using Wilder's smoothing method.

    Formula:
    RS = Average Gain / Average Loss (over N periods)
    RSI = 100 - (100 / (1 + RS))

    Args:
        closes: List of closing prices (oldest first)
        period: RSI period

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


def calculate_atr(candles, period: int = 14) -> Optional[float]:
    """
    Calculate Average True Range.

    Args:
        candles: Tuple/list of candle objects with high, low, close
        period: ATR period

    Returns:
        ATR value or None if insufficient data
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


def calculate_adx(candles, period: int = 14) -> Optional[float]:
    """
    Calculate Average Directional Index for trend strength.

    ADX measures trend strength (not direction):
    - ADX < 20: Weak trend / ranging
    - ADX 20-25: Trend developing
    - ADX 25-50: Strong trend
    - ADX > 50: Very strong trend

    Args:
        candles: Tuple/list of candle objects
        period: ADX period

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

    # Calculate +DI and -DI
    dx_values = []
    for i in range(len(atr)):
        if atr[i] == 0:
            plus_di = 0
            minus_di = 0
        else:
            plus_di = 100 * smooth_plus_dm[i] / atr[i]
            minus_di = 100 * smooth_minus_dm[i] / atr[i]

        # Calculate DX
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


def get_adaptive_rsi_zones(
    current_atr: Optional[float],
    current_price: float,
    config: Dict[str, Any],
    symbol: str
) -> Tuple[float, float]:
    """
    Get RSI zones adjusted by volatility.

    During volatile markets, RSI thresholds are expanded to reduce
    false signals from sustained overbought/oversold conditions.

    Args:
        current_atr: Current ATR value
        current_price: Current market price
        config: Strategy configuration
        symbol: Trading symbol

    Returns:
        Tuple of (oversold_threshold, overbought_threshold)
    """
    base_oversold = get_symbol_config(symbol, config, 'rsi_oversold')
    base_overbought = get_symbol_config(symbol, config, 'rsi_overbought')

    if not config.get('use_adaptive_rsi', True):
        return base_oversold, base_overbought

    if current_atr is None or current_atr <= 0 or current_price <= 0:
        return base_oversold, base_overbought

    # Calculate ATR as percentage of price
    atr_pct = (current_atr / current_price) * 100
    zone_expansion_limit = config.get('rsi_zone_expansion', 5)

    # Expand zones based on volatility
    expansion = min(zone_expansion_limit, atr_pct * 2)

    adaptive_oversold = max(15, base_oversold - expansion)
    adaptive_overbought = min(85, base_overbought + expansion)

    return adaptive_oversold, adaptive_overbought


def classify_rsi_zone(
    rsi: float,
    oversold: float,
    overbought: float
) -> RSIZone:
    """
    Classify RSI into zone.

    Args:
        rsi: Current RSI value
        oversold: Oversold threshold
        overbought: Overbought threshold

    Returns:
        RSIZone enum value
    """
    if rsi < oversold:
        return RSIZone.OVERSOLD
    elif rsi > overbought:
        return RSIZone.OVERBOUGHT
    else:
        return RSIZone.NEUTRAL


def calculate_rsi_confidence(
    side: str,
    rsi: float,
    oversold: float,
    overbought: float,
    config: Dict[str, Any]
) -> Tuple[float, str]:
    """
    Calculate confidence based on RSI position.

    From legacy RSIMeanReversionGrid implementation:
    - RSI in extreme zone: 0.7-1.0 confidence
    - RSI approaching zone: 0.5-0.8 confidence
    - RSI neutral: 0.2-0.4 confidence

    Args:
        side: 'buy' or 'sell'
        rsi: Current RSI value
        oversold: Oversold threshold
        overbought: Overbought threshold
        config: Strategy configuration

    Returns:
        Tuple of (confidence value 0-1, confidence reason)
    """
    if side == 'buy':
        if rsi < oversold:
            # Deep oversold - highest confidence
            boost = min(0.4, (oversold - rsi) / 50)
            confidence = min(1.0, 0.7 + boost)
            reason = f"deep_oversold (RSI={rsi:.1f}<{oversold})"
        elif rsi < oversold + 15:
            # Approaching oversold - moderate confidence
            boost = (oversold + 15 - rsi) / 30 * 0.3
            confidence = min(1.0, 0.5 + boost)
            reason = f"approaching_oversold (RSI={rsi:.1f})"
        else:
            # Neutral - low confidence
            confidence = max(0.2, 0.5 - 0.1)
            reason = f"neutral_rsi (RSI={rsi:.1f})"
    else:  # sell
        if rsi > overbought:
            # Deep overbought - highest confidence
            boost = min(0.4, (rsi - overbought) / 50)
            confidence = min(1.0, 0.7 + boost)
            reason = f"deep_overbought (RSI={rsi:.1f}>{overbought})"
        elif rsi > overbought - 15:
            # Approaching overbought - moderate confidence
            boost = (rsi - (overbought - 15)) / 30 * 0.3
            confidence = min(1.0, 0.5 + boost)
            reason = f"approaching_overbought (RSI={rsi:.1f})"
        else:
            # Neutral - low confidence
            confidence = max(0.2, 0.5 - 0.1)
            reason = f"neutral_rsi (RSI={rsi:.1f})"

    return confidence, reason


def calculate_position_size_multiplier(
    rsi: float,
    oversold: float,
    overbought: float,
    config: Dict[str, Any]
) -> float:
    """
    Calculate position size multiplier based on RSI.

    From legacy code: RSI extreme positions warrant larger sizes
    as mean reversion probability is higher.

    Args:
        rsi: Current RSI value
        oversold: Oversold threshold
        overbought: Overbought threshold
        config: Strategy configuration

    Returns:
        Position size multiplier (1.0 = base size)
    """
    extreme_multiplier = config.get('rsi_extreme_multiplier', 1.3)

    if rsi < oversold:
        # Oversold - larger buy sizes
        return extreme_multiplier
    elif rsi < oversold + 10:
        # Approaching oversold
        return 1.0 + (extreme_multiplier - 1.0) * 0.5
    elif rsi > overbought:
        # Overbought - larger sell sizes
        return extreme_multiplier
    elif rsi > overbought - 10:
        # Approaching overbought
        return 1.0 + (extreme_multiplier - 1.0) * 0.5
    else:
        return 1.0


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
