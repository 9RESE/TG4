"""
Momentum Scalping Strategy - Indicator Calculations

Contains strategy-specific indicator functions for momentum scalping analysis.
Common indicators are imported from the centralized ws_tester.indicators library.

Based on research from master-plan-v1.0.md:
- RSI period 7 for fast momentum detection
- MACD settings (6, 13, 5) optimized for 1-minute scalping
- EMA 8/21/50 ribbon for trend detection
"""
from typing import List, Tuple, Dict, Any, Optional

# Import common indicators from centralized library
from ws_tester.indicators import (
    calculate_ema,
    calculate_ema_series,
    calculate_rsi,
    calculate_rsi_series,
    calculate_macd,
    calculate_macd_with_history,
    calculate_volume_ratio,
    calculate_volume_spike,
    calculate_volatility,
    calculate_atr,
    calculate_adx,
    calculate_rolling_correlation,
)


# Re-export for backward compatibility
__all__ = [
    # From centralized library (re-exported)
    'calculate_ema',
    'calculate_ema_series',
    'calculate_rsi',
    'calculate_rsi_series',
    'calculate_macd',
    'calculate_macd_with_history',
    'calculate_volume_ratio',
    'calculate_volume_spike',
    'calculate_volatility',
    'calculate_atr',
    'calculate_adx',
    'calculate_rolling_correlation',
    # Strategy-specific functions
    'check_ema_alignment',
    'check_momentum_signal',
    'calculate_correlation',
    'check_5m_trend_alignment',
]


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
