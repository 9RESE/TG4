"""
Volatility Calculations

Contains volatility percentage, ATR, Bollinger Bands, and volatility regime.

Source implementations:
- Volatility: All 6 implementations are IDENTICAL
- ATR: momentum_scalping/indicators.py:407-438, whale_sentiment/indicators.py:90-146
- Bollinger: mean_reversion/indicators.py:60-78, ratio_trading/indicators.py:11-38
"""
from typing import List, Dict, Any, Optional, Tuple, Union

from ._types import PriceInput, BollingerResult, ATRResult, extract_closes, extract_hlc


def calculate_volatility(data: PriceInput, lookback: int = 20) -> float:
    """
    Calculate price volatility from price data.

    Volatility = Standard Deviation of Returns * 100 (as percentage).

    This is the most duplicated function across strategies - all 6
    implementations are identical.

    Args:
        data: Either List/Tuple of Candle objects or List[float] prices
        lookback: Number of periods for calculation

    Returns:
        Volatility as percentage (std dev of returns * 100)
        Returns 0.0 if insufficient data
    """
    closes = extract_closes(data)

    if len(closes) < lookback + 1:
        return 0.0

    closes = closes[-(lookback + 1):]
    if len(closes) < 2:
        return 0.0

    # Convert to float to handle Decimal types from database
    closes = [float(c) for c in closes]

    # Calculate returns
    returns = [(closes[i] - closes[i - 1]) / closes[i - 1]
               for i in range(1, len(closes)) if closes[i - 1] != 0]

    if not returns:
        return 0.0

    # Calculate standard deviation using population variance (N, not N-1)
    # This is consistent with original strategy implementations and is appropriate
    # for describing the volatility of the observed data rather than estimating
    # the volatility of a larger population from which these returns were sampled.
    mean_return = sum(returns) / len(returns)
    variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)

    return (variance ** 0.5) * 100


def calculate_atr(
    data: PriceInput,
    period: int = 14,
    rich_output: bool = False
) -> Union[Optional[float], ATRResult]:
    """
    Calculate Average True Range.

    True Range = max(high-low, |high-prev_close|, |low-prev_close|)
    ATR = Wilder's smoothed average of True Range

    Args:
        data: Must be Candle data with high, low, close attributes
        period: ATR period (default 14)
        rich_output: If True, returns ATRResult with full details

    Returns:
        If rich_output=False: ATR value or None if insufficient data
        If rich_output=True: ATRResult NamedTuple with atr, atr_pct, tr_series
    """
    highs, lows, closes = extract_hlc(data)

    if len(closes) < period + 1:
        if rich_output:
            return ATRResult(atr=None, atr_pct=None, tr_series=[])
        return None

    # Convert to float to handle Decimal types from database
    highs = [float(h) for h in highs]
    lows = [float(l) for l in lows]
    closes = [float(c) for c in closes]

    # Calculate True Range series
    tr_series = []
    for i in range(1, len(closes)):
        high = highs[i]
        low = lows[i]
        prev_close = closes[i - 1]

        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )
        tr_series.append(tr)

    if len(tr_series) < period:
        if rich_output:
            return ATRResult(atr=None, atr_pct=None, tr_series=tr_series)
        return None

    # Calculate ATR using Wilder's smoothing
    atr = sum(tr_series[:period]) / period

    for i in range(period, len(tr_series)):
        atr = (atr * (period - 1) + tr_series[i]) / period

    if rich_output:
        # Calculate ATR as percentage of current price
        current_price = closes[-1]
        atr_pct = (atr / current_price) * 100 if current_price > 0 else None
        return ATRResult(atr=atr, atr_pct=atr_pct, tr_series=tr_series)

    return atr


def calculate_bollinger_bands(
    data: PriceInput,
    period: int = 20,
    num_std: float = 2.0
) -> BollingerResult:
    """
    Calculate Bollinger Bands.

    Upper Band = SMA + (num_std * Standard Deviation)
    Lower Band = SMA - (num_std * Standard Deviation)

    Args:
        data: Either List/Tuple of Candle objects or List[float] prices
        period: SMA period (default 20)
        num_std: Number of standard deviations (default 2.0)

    Returns:
        BollingerResult(sma, upper, lower, std_dev)
        All values are None if insufficient data
    """
    closes = extract_closes(data)

    if len(closes) < period:
        return BollingerResult(sma=None, upper=None, lower=None, std_dev=None)

    # Convert to float to handle Decimal types from database
    recent = [float(c) for c in closes[-period:]]

    # Simple Moving Average
    sma = sum(recent) / len(recent)

    # Standard Deviation
    variance = sum((p - sma) ** 2 for p in recent) / len(recent)
    std_dev = variance ** 0.5

    # Bands
    upper = sma + (num_std * std_dev)
    lower = sma - (num_std * std_dev)

    return BollingerResult(sma=sma, upper=upper, lower=lower, std_dev=std_dev)


def calculate_z_score(price: float, sma: float, std_dev: float) -> float:
    """
    Calculate z-score (number of standard deviations from mean).

    Z-Score = (Price - SMA) / Standard Deviation

    Args:
        price: Current price
        sma: Simple Moving Average
        std_dev: Standard Deviation

    Returns:
        Z-score value (0 if std_dev is 0)
    """
    if std_dev == 0:
        return 0.0
    return (price - sma) / std_dev


def get_volatility_regime(
    volatility_pct: float,
    config: Dict[str, Any]
) -> Tuple[str, float, float]:
    """
    Classify volatility into regime.

    Regimes:
    - LOW: < 0.3% - Tighter thresholds, normal size
    - MEDIUM: 0.3% - 0.8% - Baseline
    - HIGH: 0.8% - 1.5% - Wider thresholds, reduced size
    - EXTREME: > 1.5% - PAUSE TRADING (size=0)

    Args:
        volatility_pct: Current price volatility as percentage
        config: Strategy configuration with regime thresholds

    Returns:
        Tuple of (regime_name, threshold_multiplier, size_multiplier)
    """
    low_thresh = config.get('regime_low_threshold', 0.3)
    med_thresh = config.get('regime_medium_threshold', 0.8)
    high_thresh = config.get('regime_high_threshold', 1.5)

    low_mult = config.get('regime_low_threshold_mult', 0.9)
    high_mult = config.get('regime_high_threshold_mult', 1.3)
    high_size = config.get('regime_high_size_mult', 0.7)

    if volatility_pct < low_thresh:
        return "LOW", low_mult, 1.0
    elif volatility_pct < med_thresh:
        return "MEDIUM", 1.0, 1.0
    elif volatility_pct < high_thresh:
        return "HIGH", high_mult, high_size
    else:
        return "EXTREME", 2.0, 0.0  # EXTREME = pause (size=0)
