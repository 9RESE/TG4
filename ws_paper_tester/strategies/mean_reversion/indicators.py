"""
Mean Reversion Strategy - Indicator Calculations

Contains pure functions for calculating technical indicators:
- Simple Moving Average (SMA)
- Relative Strength Index (RSI)
- Bollinger Bands
- Volatility (standard deviation of returns)
"""
from typing import List, Tuple, Optional


def calculate_sma(candles: List, period: int) -> float:
    """Calculate simple moving average."""
    if len(candles) < period:
        return 0.0
    closes = [c.close for c in candles[-period:]]
    return sum(closes) / len(closes)


def calculate_rsi(candles: List, period: int = 14) -> float:
    """
    Calculate RSI indicator.

    LOW-007: Fixed edge case where index could go negative.
    """
    if len(candles) < period + 1:
        return 50.0  # Neutral

    gains = []
    losses = []

    # LOW-007: Ensure we don't access negative indices
    start_idx = max(1, len(candles) - period)
    for i in range(start_idx, len(candles)):
        change = candles[i].close - candles[i-1].close
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


def calculate_bollinger_bands(
    candles: List,
    period: int = 20,
    std_dev: float = 2.0
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Calculate Bollinger Bands."""
    if len(candles) < period:
        return None, None, None

    closes = [c.close for c in candles[-period:]]
    sma = sum(closes) / len(closes)

    variance = sum((c - sma) ** 2 for c in closes) / len(closes)
    std = variance ** 0.5

    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)

    return lower, sma, upper


def calculate_volatility(candles: List, lookback: int = 20) -> float:
    """
    Calculate price volatility from candle closes.

    Returns volatility as a percentage (std dev of returns * 100).
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


def calculate_trend_slope(candles: List, period: int = 50) -> float:
    """
    Calculate the slope of price trend over given period.

    Returns slope as percentage change per candle.
    Positive = uptrend, Negative = downtrend, Near zero = ranging.
    """
    if len(candles) < period:
        return 0.0

    closes = [c.close for c in candles[-period:]]
    if len(closes) < 2:
        return 0.0

    # Calculate linear regression slope
    n = len(closes)
    sum_x = sum(range(n))
    sum_y = sum(closes)
    sum_xy = sum(i * closes[i] for i in range(n))
    sum_x2 = sum(i * i for i in range(n))

    denominator = n * sum_x2 - sum_x * sum_x
    if denominator == 0:
        return 0.0

    slope = (n * sum_xy - sum_x * sum_y) / denominator

    # Convert to percentage of average price
    avg_price = sum_y / n if n > 0 else 1.0
    slope_pct = (slope / avg_price) * 100 if avg_price > 0 else 0.0

    return slope_pct


def calculate_correlation(
    xrp_candles: List,
    btc_candles: List,
    lookback: int = 50
) -> Optional[float]:
    """
    Calculate rolling Pearson correlation between XRP and BTC price movements.

    REC-005 (v4.0.0): Added for XRP/BTC ratio trading analysis.
    Research shows XRP correlation with BTC declining (24.86% over 90 days),
    which affects ratio mean reversion timing.

    Args:
        xrp_candles: XRP/USDT candles
        btc_candles: BTC/USDT candles
        lookback: Number of candles for correlation

    Returns:
        Correlation coefficient (-1 to +1), None if insufficient data
    """
    if len(xrp_candles) < lookback + 1 or len(btc_candles) < lookback + 1:
        return None

    # Get returns for correlation calculation
    xrp_closes = [c.close for c in xrp_candles[-(lookback + 1):]]
    btc_closes = [c.close for c in btc_candles[-(lookback + 1):]]

    if len(xrp_closes) != len(btc_closes):
        return None

    # Calculate returns
    xrp_returns = [(xrp_closes[i] - xrp_closes[i-1]) / xrp_closes[i-1]
                   for i in range(1, len(xrp_closes)) if xrp_closes[i-1] != 0]
    btc_returns = [(btc_closes[i] - btc_closes[i-1]) / btc_closes[i-1]
                   for i in range(1, len(btc_closes)) if btc_closes[i-1] != 0]

    if len(xrp_returns) < 2 or len(btc_returns) < 2:
        return None

    # Ensure same length
    n = min(len(xrp_returns), len(btc_returns))
    xrp_returns = xrp_returns[-n:]
    btc_returns = btc_returns[-n:]

    # Calculate Pearson correlation
    mean_xrp = sum(xrp_returns) / n
    mean_btc = sum(btc_returns) / n

    # Covariance
    covariance = sum((xrp_returns[i] - mean_xrp) * (btc_returns[i] - mean_btc)
                     for i in range(n)) / n

    # Standard deviations
    std_xrp = (sum((r - mean_xrp) ** 2 for r in xrp_returns) / n) ** 0.5
    std_btc = (sum((r - mean_btc) ** 2 for r in btc_returns) / n) ** 0.5

    if std_xrp == 0 or std_btc == 0:
        return None

    correlation = covariance / (std_xrp * std_btc)

    # Clamp to valid range
    return max(-1.0, min(1.0, correlation))
