"""
Correlation Calculations

Contains rolling Pearson correlation for cross-asset analysis.

Source implementations:
- grid_rsi_reversion/indicators.py:473-530
- mean_reversion/indicators.py:139-201
- wavetrend/indicators.py:396-462
- whale_sentiment/indicators.py:584-648
- ratio_trading/indicators.py:205-290
"""
from typing import List, Optional, Tuple

from ._types import PriceInput, CorrelationTrendResult, extract_closes


def calculate_rolling_correlation(
    prices_a: PriceInput,
    prices_b: PriceInput,
    window: int = 20
) -> Optional[float]:
    """
    Calculate Pearson correlation on price returns.

    Uses returns (percentage changes) rather than raw prices to avoid
    spurious correlation from trending prices.

    Formula: r = Cov(returns_a, returns_b) / (StdDev(returns_a) * StdDev(returns_b))

    Args:
        prices_a: First asset prices (oldest first)
        prices_b: Second asset prices (oldest first)
        window: Number of periods for correlation calculation

    Returns:
        Pearson correlation coefficient (-1 to +1) or None if insufficient data
    """
    closes_a = extract_closes(prices_a)
    closes_b = extract_closes(prices_b)

    if len(closes_a) < window + 1 or len(closes_b) < window + 1:
        return None

    # Use last window+1 prices to calculate window returns
    a = closes_a[-(window + 1):]
    b = closes_b[-(window + 1):]

    # Calculate returns - skip pairs where either denominator is zero
    # to maintain alignment between the two return series
    returns_a = []
    returns_b = []
    for i in range(1, min(len(a), len(b))):
        if a[i-1] != 0 and b[i-1] != 0:
            returns_a.append((a[i] - a[i-1]) / a[i-1])
            returns_b.append((b[i] - b[i-1]) / b[i-1])

    if len(returns_a) < 2:
        return None

    n = len(returns_a)

    # Calculate means
    mean_a = sum(returns_a) / n
    mean_b = sum(returns_b) / n

    # Calculate covariance and standard deviations
    covariance = sum((returns_a[i] - mean_a) * (returns_b[i] - mean_b)
                     for i in range(n)) / n
    variance_a = sum((r - mean_a) ** 2 for r in returns_a) / n
    variance_b = sum((r - mean_b) ** 2 for r in returns_b) / n

    std_a = variance_a ** 0.5
    std_b = variance_b ** 0.5

    if std_a == 0 or std_b == 0:
        return None

    correlation = covariance / (std_a * std_b)

    # Clamp to valid range to handle floating point errors
    return max(-1.0, min(1.0, correlation))


def calculate_correlation_trend(
    correlation_history: List[float],
    lookback: int = 10
) -> CorrelationTrendResult:
    """
    Calculate correlation trend (slope) to detect deteriorating relationships.

    Uses simple linear regression on correlation history to determine
    if correlation is declining, stable, or improving.

    Args:
        correlation_history: Historical correlation values (oldest first)
        lookback: Number of periods for trend calculation

    Returns:
        CorrelationTrendResult(slope, is_declining, direction)
        - slope: Linear regression slope (-1 to 1 per period)
        - is_declining: True if slope is significantly negative
        - direction: 'declining', 'stable', or 'improving'
    """
    if len(correlation_history) < lookback:
        return CorrelationTrendResult(slope=0.0, is_declining=False, direction='stable')

    recent = correlation_history[-lookback:]
    n = len(recent)

    # Simple linear regression: y = mx + b
    x_mean = (n - 1) / 2
    y_mean = sum(recent) / n

    numerator = sum((i - x_mean) * (recent[i] - y_mean) for i in range(n))
    denominator = sum((i - x_mean) ** 2 for i in range(n))

    if denominator == 0:
        return CorrelationTrendResult(slope=0.0, is_declining=False, direction='stable')

    slope = numerator / denominator

    # Classify trend direction
    if slope < -0.01:
        direction = 'declining'
        is_declining = True
    elif slope > 0.01:
        direction = 'improving'
        is_declining = False
    else:
        direction = 'stable'
        is_declining = False

    return CorrelationTrendResult(slope=slope, is_declining=is_declining, direction=direction)
