"""
Ratio Trading Strategy - Indicators Module

Technical indicator calculations: Bollinger Bands, RSI, volatility,
trend detection, correlation, trailing stops, position decay.
"""
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple


def calculate_bollinger_bands(
    prices: List[float],
    lookback: int = 20,
    num_std: float = 2.0
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Calculate Bollinger Bands.

    Returns:
        (sma, upper_band, lower_band, std_dev)
    """
    if len(prices) < lookback:
        return None, None, None, None

    recent = prices[-lookback:]

    # Simple Moving Average
    sma = sum(recent) / len(recent)

    # Standard Deviation
    variance = sum((p - sma) ** 2 for p in recent) / len(recent)
    std_dev = variance ** 0.5

    # Bands
    upper = sma + (num_std * std_dev)
    lower = sma - (num_std * std_dev)

    return sma, upper, lower, std_dev


def calculate_z_score(price: float, sma: float, std_dev: float) -> float:
    """Calculate z-score (number of std devs from mean)."""
    if std_dev == 0:
        return 0.0
    return (price - sma) / std_dev


def calculate_volatility(prices: List[float], lookback: int = 20) -> float:
    """
    Calculate price volatility from price history.

    Returns volatility as a percentage (std dev of returns * 100).
    """
    if len(prices) < lookback + 1:
        return 0.0

    recent = prices[-(lookback + 1):]
    if len(recent) < 2:
        return 0.0

    returns = [(recent[i] - recent[i - 1]) / recent[i - 1]
               for i in range(1, len(recent)) if recent[i - 1] != 0]

    if not returns:
        return 0.0

    mean_return = sum(returns) / len(returns)
    variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
    return (variance ** 0.5) * 100


def calculate_rsi(prices: List[float], period: int = 14) -> float:
    """
    Calculate RSI indicator from price history.

    REC-014: RSI confirmation for signal quality.

    Returns:
        RSI value (0-100), 50.0 if insufficient data
    """
    if len(prices) < period + 1:
        return 50.0  # Neutral

    gains = []
    losses = []

    start_idx = max(1, len(prices) - period)
    for i in range(start_idx, len(prices)):
        change = prices[i] - prices[i - 1]
        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(change))

    if not gains:
        return 50.0  # Neutral if no data

    avg_gain = sum(gains) / len(gains)
    avg_loss = sum(losses) / len(losses)

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def detect_trend_strength(
    prices: List[float],
    lookback: int = 10,
    threshold: float = 0.7
) -> Tuple[bool, str, float]:
    """
    Detect if there's a strong trend in recent price action.

    REC-015: Trend detection warning system.

    Returns:
        (is_strong_trend, direction, strength)
        - is_strong_trend: True if trend is strong enough to warn
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
    lowest_price: float,
    side: str,
    activation_pct: float,
    trail_distance_pct: float
) -> Optional[float]:
    """
    Calculate trailing stop price.

    From mean reversion patterns.

    Returns:
        Trailing stop price if activated, None otherwise
    """
    if side == 'long':
        profit_pct = (highest_price - entry_price) / entry_price * 100
        if profit_pct >= activation_pct:
            return highest_price * (1 - trail_distance_pct / 100)
    elif side == 'short':
        profit_pct = (entry_price - lowest_price) / entry_price * 100
        if profit_pct >= activation_pct:
            return lowest_price * (1 + trail_distance_pct / 100)
    return None


def check_position_decay(
    entry_time: datetime,
    current_time: datetime,
    decay_minutes: float
) -> Tuple[bool, float]:
    """
    Check if position has decayed (exceeded time threshold).

    From mean reversion patterns.

    Returns:
        (is_decayed, minutes_held)
    """
    if entry_time is None:
        return False, 0.0

    minutes_held = (current_time - entry_time).total_seconds() / 60
    is_decayed = minutes_held >= decay_minutes

    return is_decayed, minutes_held


def calculate_rolling_correlation(
    prices_a: List[float],
    prices_b: List[float],
    lookback: int = 20
) -> float:
    """
    Calculate rolling Pearson correlation between two price series.

    REC-021: Rolling correlation monitoring.

    Args:
        prices_a: First price series (e.g., XRP prices in BTC)
        prices_b: Second price series (e.g., BTC prices in USD)
        lookback: Number of periods for correlation calculation

    Returns:
        Correlation coefficient (-1 to 1), 0.0 if insufficient data
    """
    if len(prices_a) < lookback or len(prices_b) < lookback:
        return 0.0

    # Use most recent lookback periods
    a = prices_a[-lookback:]
    b = prices_b[-lookback:]

    if len(a) != len(b):
        # Align lengths
        min_len = min(len(a), len(b))
        a = a[-min_len:]
        b = b[-min_len:]

    if len(a) < 3:
        return 0.0

    # Calculate means
    mean_a = sum(a) / len(a)
    mean_b = sum(b) / len(b)

    # Calculate covariance and standard deviations
    covariance = sum((a[i] - mean_a) * (b[i] - mean_b) for i in range(len(a))) / len(a)
    variance_a = sum((x - mean_a) ** 2 for x in a) / len(a)
    variance_b = sum((x - mean_b) ** 2 for x in b) / len(b)

    std_a = variance_a ** 0.5
    std_b = variance_b ** 0.5

    if std_a == 0 or std_b == 0:
        return 0.0

    correlation = covariance / (std_a * std_b)

    # Clamp to [-1, 1] to handle floating point errors
    return max(-1.0, min(1.0, correlation))


def calculate_correlation_trend(
    correlation_history: List[float],
    lookback: int = 10
) -> Tuple[float, bool, str]:
    """
    Calculate correlation trend (slope) to detect deteriorating relationship.

    REC-037: Correlation trend monitoring for proactive protection.

    Args:
        correlation_history: Historical correlation values
        lookback: Number of periods for trend calculation

    Returns:
        (slope, is_declining, trend_direction)
        - slope: Linear regression slope of correlation (-1 to 1 per period)
        - is_declining: True if slope is significantly negative
        - trend_direction: 'declining', 'stable', or 'improving'
    """
    if len(correlation_history) < lookback:
        return 0.0, False, 'stable'

    recent = correlation_history[-lookback:]
    n = len(recent)

    # Simple linear regression: y = mx + b
    x_mean = (n - 1) / 2
    y_mean = sum(recent) / n

    numerator = sum((i - x_mean) * (recent[i] - y_mean) for i in range(n))
    denominator = sum((i - x_mean) ** 2 for i in range(n))

    if denominator == 0:
        return 0.0, False, 'stable'

    slope = numerator / denominator

    # Classify trend direction
    if slope < -0.01:
        trend_direction = 'declining'
        is_declining = True
    elif slope > 0.01:
        trend_direction = 'improving'
        is_declining = False
    else:
        trend_direction = 'stable'
        is_declining = False

    return slope, is_declining, trend_direction


def get_btc_price_usd(data, config: Dict[str, Any]) -> float:
    """
    Get BTC/USD price from market data or fallback.

    REC-018: Dynamic BTC price for USD conversion.

    Args:
        data: Market data snapshot
        config: Strategy configuration

    Returns:
        BTC price in USD
    """
    btc_symbols = config.get('btc_price_symbols', ['BTC/USDT', 'BTC/USD'])
    fallback = config.get('btc_price_fallback', 100000.0)

    for symbol in btc_symbols:
        price = data.prices.get(symbol)
        if price and price > 0:
            return price

    return fallback


def convert_usd_to_xrp(
    usd_amount: float,
    price_btc_per_xrp: float,
    btc_price_usd: float
) -> float:
    """
    Convert USD amount to XRP for ratio trading.

    REC-018: Now uses dynamic BTC price instead of hardcoded fallback.

    For XRP/BTC pair, we need to convert through BTC:
    USD -> BTC -> XRP

    Args:
        usd_amount: Amount in USD to convert
        price_btc_per_xrp: Current XRP/BTC price (BTC per XRP)
        btc_price_usd: Current BTC/USD price

    Returns:
        Equivalent XRP amount
    """
    if btc_price_usd <= 0 or price_btc_per_xrp <= 0:
        return 0.0

    btc_amount = usd_amount / btc_price_usd
    xrp_amount = btc_amount / price_btc_per_xrp

    return xrp_amount
