"""
Phase 11: Mean Reversion Strategy - VWAP/RSI Filter for Shorts
Provides VWAP calculation and deviation detection for short entry/exit.
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


def calculate_vwap(data: dict, symbol: str = 'XRP/USDT', period: int = 20) -> float:
    """
    Calculate Volume Weighted Average Price (VWAP).

    Args:
        data: Dictionary of DataFrames keyed by symbol
        symbol: Trading pair symbol
        period: Number of candles to use for VWAP

    Returns:
        float: VWAP value, or 0.0 if insufficient data
    """
    if symbol not in data:
        return 0.0

    df = data[symbol]
    if len(df) < period:
        return 0.0

    # Use last 'period' candles
    recent = df.tail(period)

    # VWAP = Sum(Price * Volume) / Sum(Volume)
    # Use typical price: (High + Low + Close) / 3
    typical_price = (recent['high'] + recent['low'] + recent['close']) / 3
    volume = recent['volume']

    vwap = (typical_price * volume).sum() / volume.sum() if volume.sum() > 0 else 0.0

    return vwap


def is_above_vwap(data: dict, symbol: str = 'XRP/USDT',
                  period: int = 20, threshold: float = 0.0) -> bool:
    """
    Check if current price is above VWAP (short signal).

    Args:
        data: Dictionary of DataFrames keyed by symbol
        symbol: Trading pair symbol
        period: VWAP period
        threshold: Minimum % above VWAP to return True (0.0 = any amount above)

    Returns:
        bool: True if current price is above VWAP by threshold
    """
    if symbol not in data:
        return False

    df = data[symbol]
    if len(df) < period:
        return False

    current_price = df['close'].iloc[-1]
    vwap = calculate_vwap(data, symbol, period)

    if vwap <= 0:
        return False

    deviation_pct = (current_price - vwap) / vwap

    return deviation_pct > threshold


def get_vwap_deviation(data: dict, symbol: str = 'XRP/USDT',
                       period: int = 20) -> Tuple[float, float]:
    """
    Get current price deviation from VWAP.

    Args:
        data: Dictionary of DataFrames keyed by symbol
        symbol: Trading pair symbol
        period: VWAP period

    Returns:
        tuple: (deviation_pct, vwap_value)
    """
    if symbol not in data:
        return 0.0, 0.0

    df = data[symbol]
    if len(df) < period:
        return 0.0, 0.0

    current_price = df['close'].iloc[-1]
    vwap = calculate_vwap(data, symbol, period)

    if vwap <= 0:
        return 0.0, 0.0

    deviation_pct = (current_price - vwap) / vwap

    return deviation_pct, vwap


def generate_mean_reversion_signal(data: dict, symbol: str = 'XRP/USDT') -> Dict:
    """
    Generate mean reversion trading signal based on VWAP and RSI.

    Args:
        data: Dictionary of DataFrames keyed by symbol
        symbol: Trading pair symbol

    Returns:
        dict: Signal details including action, confidence, deviation
    """
    if symbol not in data:
        return {
            'signal': 'hold',
            'confidence': 0.0,
            'deviation': 0.0,
            'vwap': 0.0,
            'above_vwap': False
        }

    df = data[symbol]
    if len(df) < 30:
        return {
            'signal': 'hold',
            'confidence': 0.0,
            'deviation': 0.0,
            'vwap': 0.0,
            'above_vwap': False
        }

    current_price = df['close'].iloc[-1]
    deviation_pct, vwap = get_vwap_deviation(data, symbol, period=20)

    # Calculate RSI
    close = df['close'].values
    deltas = np.diff(close[-15:])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains) if len(gains) > 0 else 0
    avg_loss = np.mean(losses) if len(losses) > 0 else 0.001
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Determine signal
    signal = 'hold'
    confidence = 0.0

    # Short signal: Price above VWAP + RSI > 65 (overbought)
    if deviation_pct > 0.02 and rsi > 65:  # >2% above VWAP + overbought
        signal = 'short'
        # Confidence scales with deviation and RSI extremity
        confidence = min(0.5 + (deviation_pct * 5) + ((rsi - 65) / 70), 1.0)

    # Long signal: Price below VWAP + RSI < 35 (oversold)
    elif deviation_pct < -0.02 and rsi < 35:  # >2% below VWAP + oversold
        signal = 'long'
        confidence = min(0.5 + (abs(deviation_pct) * 5) + ((35 - rsi) / 70), 1.0)

    return {
        'signal': signal,
        'confidence': confidence,
        'deviation': deviation_pct,
        'vwap': vwap,
        'above_vwap': deviation_pct > 0,
        'current_price': current_price,
        'rsi': rsi
    }


def calculate_bands(data: dict, symbol: str = 'XRP/USDT',
                    period: int = 20, std_mult: float = 2.0) -> Dict:
    """
    Calculate VWAP bands (similar to Bollinger Bands but around VWAP).

    Args:
        data: Dictionary of DataFrames keyed by symbol
        symbol: Trading pair symbol
        period: Period for calculations
        std_mult: Standard deviation multiplier for bands

    Returns:
        dict: VWAP and upper/lower bands
    """
    if symbol not in data:
        return {'vwap': 0.0, 'upper': 0.0, 'lower': 0.0}

    df = data[symbol]
    if len(df) < period:
        return {'vwap': 0.0, 'upper': 0.0, 'lower': 0.0}

    vwap = calculate_vwap(data, symbol, period)

    # Calculate standard deviation of typical prices
    recent = df.tail(period)
    typical_price = (recent['high'] + recent['low'] + recent['close']) / 3
    std_dev = typical_price.std()

    return {
        'vwap': vwap,
        'upper': vwap + (std_mult * std_dev),
        'lower': vwap - (std_mult * std_dev),
        'std_dev': std_dev
    }
