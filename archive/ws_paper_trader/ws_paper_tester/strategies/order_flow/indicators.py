"""
Order Flow Strategy - Indicator Calculations

Contains strategy-specific indicator functions for order flow analysis.
Common indicators are imported from the centralized ws_tester.indicators library.
"""
from typing import Dict, Any, Tuple

# Import common indicators from centralized library
from ws_tester.indicators import (
    calculate_volatility,
    calculate_micro_price,
    calculate_vpin,
)

# Re-export for backward compatibility with existing strategy code
__all__ = [
    'calculate_volatility',
    'calculate_micro_price',
    'calculate_vpin',
    'check_volume_anomaly',
]


def check_volume_anomaly(
    trades: Tuple,
    config: Dict[str, Any],
    current_price: float = 0.0,
    previous_price: float = 0.0
) -> Dict[str, Any]:
    """
    REC-005 (v5.0.0): Detect potential wash trading patterns.

    Implements three indicators:
    1. Volume consistency vs rolling average - detects abnormal volume levels
    2. Repetitive exact-size trades - detects potential wash trading patterns
    3. Volume spike without corresponding price movement - detects fake volume

    Args:
        trades: Tuple of recent trades
        config: Strategy configuration dict
        current_price: Current price for volume-price divergence check
        previous_price: Previous price for volume-price divergence check

    Returns:
        Dict with:
            - anomaly_detected: bool - whether any anomaly was detected
            - anomaly_types: list - types of anomalies detected
            - confidence_score: float - confidence of anomaly detection (0-1)
            - details: dict - detailed information about each check
    """
    result = {
        'anomaly_detected': False,
        'anomaly_types': [],
        'confidence_score': 0.0,
        'details': {
            'volume_consistency': {'checked': False, 'anomaly': False},
            'repetitive_trades': {'checked': False, 'anomaly': False},
            'volume_price_divergence': {'checked': False, 'anomaly': False},
        }
    }

    lookback = config.get('volume_anomaly_lookback_trades', 100)

    if len(trades) < lookback:
        return result

    recent_trades = trades[-lookback:]

    # =========================================================================
    # 1. Volume Consistency Check
    # Flag if current volume is abnormally different from rolling average
    # =========================================================================
    low_ratio = config.get('volume_anomaly_low_ratio', 0.2)
    high_ratio = config.get('volume_anomaly_high_ratio', 5.0)

    # Calculate rolling average volume per trade
    total_volume = sum(t.size for t in recent_trades)
    avg_volume_per_trade = total_volume / len(recent_trades) if recent_trades else 0

    # Check last 10 trades against rolling average
    last_10_volume = sum(t.size for t in recent_trades[-10:])
    expected_10_volume = avg_volume_per_trade * 10

    volume_ratio = last_10_volume / expected_10_volume if expected_10_volume > 0 else 1.0

    result['details']['volume_consistency'] = {
        'checked': True,
        'anomaly': False,
        'volume_ratio': round(volume_ratio, 4),
        'avg_volume_per_trade': round(avg_volume_per_trade, 6),
        'last_10_volume': round(last_10_volume, 6),
    }

    if volume_ratio < low_ratio:
        result['details']['volume_consistency']['anomaly'] = True
        result['details']['volume_consistency']['reason'] = 'suspiciously_low_volume'
        result['anomaly_types'].append('low_volume')
    elif volume_ratio > high_ratio:
        result['details']['volume_consistency']['anomaly'] = True
        result['details']['volume_consistency']['reason'] = 'suspiciously_high_volume'
        result['anomaly_types'].append('high_volume')

    # =========================================================================
    # 2. Repetitive Trade Detection
    # Flag if too many trades have identical sizes (suspicious pattern)
    # =========================================================================
    repetitive_threshold = config.get('volume_anomaly_repetitive_threshold', 0.4)
    tolerance = config.get('volume_anomaly_repetitive_tolerance', 0.001)

    # Group trades by size with tolerance
    size_groups = {}
    for trade in recent_trades:
        matched = False
        for key_size in size_groups:
            if abs(trade.size - key_size) / key_size <= tolerance if key_size > 0 else trade.size == 0:
                size_groups[key_size] += 1
                matched = True
                break
        if not matched:
            size_groups[trade.size] = 1

    # Find the most common size group
    if size_groups:
        max_count = max(size_groups.values())
        repetitive_ratio = max_count / len(recent_trades)
    else:
        repetitive_ratio = 0.0

    result['details']['repetitive_trades'] = {
        'checked': True,
        'anomaly': False,
        'repetitive_ratio': round(repetitive_ratio, 4),
        'unique_sizes': len(size_groups),
        'max_same_size_count': max_count if size_groups else 0,
    }

    if repetitive_ratio > repetitive_threshold:
        result['details']['repetitive_trades']['anomaly'] = True
        result['details']['repetitive_trades']['reason'] = 'high_repetitive_trades'
        result['anomaly_types'].append('repetitive_trades')

    # =========================================================================
    # 3. Volume-Price Divergence
    # Flag volume spike without corresponding price movement (fake volume)
    # =========================================================================
    price_move_threshold = config.get('volume_anomaly_price_move_threshold', 0.001)
    volume_spike_threshold = config.get('volume_anomaly_volume_spike_threshold', 3.0)

    result['details']['volume_price_divergence'] = {
        'checked': False,
        'anomaly': False,
    }

    if current_price > 0 and previous_price > 0 and volume_ratio >= volume_spike_threshold:
        price_change = abs(current_price - previous_price) / previous_price

        result['details']['volume_price_divergence'] = {
            'checked': True,
            'anomaly': False,
            'price_change_pct': round(price_change * 100, 4),
            'volume_spike_ratio': round(volume_ratio, 4),
        }

        # Volume spike but no significant price movement
        if price_change < price_move_threshold:
            result['details']['volume_price_divergence']['anomaly'] = True
            result['details']['volume_price_divergence']['reason'] = 'volume_spike_no_price_move'
            result['anomaly_types'].append('volume_price_divergence')

    # =========================================================================
    # Calculate overall confidence score
    # =========================================================================
    anomaly_count = len(result['anomaly_types'])

    if anomaly_count == 0:
        result['confidence_score'] = 0.0
    elif anomaly_count == 1:
        result['confidence_score'] = 0.5
    elif anomaly_count == 2:
        result['confidence_score'] = 0.75
    else:
        result['confidence_score'] = 0.95

    result['anomaly_detected'] = anomaly_count > 0

    return result
