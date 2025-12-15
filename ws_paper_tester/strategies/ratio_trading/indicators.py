"""
Ratio Trading Strategy - Indicators Module

Contains strategy-specific indicator functions for ratio trading analysis.
Common indicators are imported from the centralized ws_tester.indicators library.

Technical indicators: Bollinger Bands, RSI, volatility, trend detection,
correlation, trailing stops, position decay.
"""
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

# Import common indicators from centralized library
from ws_tester.indicators import (
    calculate_bollinger_bands as _calculate_bollinger_bands_lib,
    calculate_z_score,
    calculate_volatility,
    calculate_rsi,
    detect_trend_strength,
    calculate_trailing_stop,
    calculate_rolling_correlation,
    calculate_correlation_trend,
    BollingerResult,
)


def calculate_bollinger_bands(
    prices: List[float],
    lookback: int = 20,
    num_std: float = 2.0
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Calculate Bollinger Bands.

    Thin wrapper around ws_tester.indicators.calculate_bollinger_bands
    that returns (sma, upper, lower, std_dev) tuple for backward compatibility.

    Returns:
        (sma, upper_band, lower_band, std_dev)
    """
    result = _calculate_bollinger_bands_lib(prices, lookback, num_std)
    return result.sma, result.upper, result.lower, result.std_dev


# Re-export for backward compatibility
__all__ = [
    # From centralized library (re-exported)
    'calculate_bollinger_bands',
    'calculate_z_score',
    'calculate_volatility',
    'calculate_rsi',
    'detect_trend_strength',
    'calculate_trailing_stop',
    'calculate_rolling_correlation',
    'calculate_correlation_trend',
    # Strategy-specific functions
    'check_position_decay',
    'get_btc_price_usd',
    'convert_usd_to_xrp',
]


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
