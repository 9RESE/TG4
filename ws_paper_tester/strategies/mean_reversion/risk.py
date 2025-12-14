"""
Mean Reversion Strategy - Risk Management

Contains functions for:
- Circuit breaker protection
- Trade flow confirmation
- Trend filtering
- XRP/BTC correlation monitoring
- Fee profitability checks
- Trailing stops
- Position decay
- ADX trend strength filter (v4.3.0)
"""
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

from .indicators import calculate_trend_slope, calculate_correlation, calculate_adx


# =============================================================================
# Circuit Breaker - REC-005
# =============================================================================
def check_circuit_breaker(
    state: Dict[str, Any],
    current_time: datetime,
    max_losses: int,
    cooldown_minutes: float
) -> bool:
    """Check if circuit breaker is active."""
    consecutive_losses = state.get('consecutive_losses', 0)

    if consecutive_losses < max_losses:
        return False

    breaker_time = state.get('circuit_breaker_time')
    if breaker_time is None:
        return False

    elapsed_minutes = (current_time - breaker_time).total_seconds() / 60

    if elapsed_minutes >= cooldown_minutes:
        state['consecutive_losses'] = 0
        state['circuit_breaker_time'] = None
        return False

    return True


# =============================================================================
# Trade Flow Confirmation - REC-008
# =============================================================================
def is_trade_flow_aligned(
    data,  # DataSnapshot
    symbol: str,
    direction: str,
    threshold: float,
    n_trades: int = 50
) -> bool:
    """Check if trade flow confirms the signal direction."""
    trade_flow = data.get_trade_imbalance(symbol, n_trades)

    if direction == 'buy':
        return trade_flow > threshold
    elif direction == 'sell':
        return trade_flow < -threshold

    return True


# =============================================================================
# Trend Filter - REC-004 (v3.0.0)
# =============================================================================
def is_trending(
    candles: List,
    config: Dict[str, Any],
    state: Dict[str, Any],
    symbol: str
) -> Tuple[bool, float, int]:
    """
    Check if market is currently trending (unsuitable for mean reversion).

    REC-003 (v4.0): Added confirmation period to reduce false positives
    in choppy markets. Only consider trending if slope exceeds threshold
    for N consecutive evaluations.

    Returns:
        Tuple of (is_confirmed_trending, slope_pct, consecutive_trending_count)
    """
    period = config.get('trend_sma_period', 50)
    threshold = config.get('trend_slope_threshold', 0.05)
    confirmation_periods = config.get('trend_confirmation_periods', 3)

    slope_pct = calculate_trend_slope(candles, period)

    # Check if slope exceeds threshold
    is_slope_trending = abs(slope_pct) > threshold

    # Initialize trend confirmation tracking per symbol
    if 'trend_confirmation_counts' not in state:
        state['trend_confirmation_counts'] = {}

    # Update consecutive trending count
    if is_slope_trending:
        state['trend_confirmation_counts'][symbol] = \
            state['trend_confirmation_counts'].get(symbol, 0) + 1
    else:
        state['trend_confirmation_counts'][symbol] = 0

    consecutive_count = state['trend_confirmation_counts'].get(symbol, 0)

    # Only confirm trending if threshold exceeded for N consecutive periods
    is_confirmed_trending = consecutive_count >= confirmation_periods

    return is_confirmed_trending, slope_pct, consecutive_count


# =============================================================================
# XRP/BTC Correlation Monitoring - REC-005 (v4.0.0)
# =============================================================================
def get_xrp_btc_correlation(
    data,  # DataSnapshot
    config: Dict[str, Any],
    state: Dict[str, Any]
) -> Optional[float]:
    """
    Get XRP/BTC correlation and update tracking.

    REC-001 (v4.2.0): Enhanced with pause threshold for automatic XRP/BTC
    trading suspension when correlation drops critically low.

    Returns correlation coefficient or None if unavailable.
    """
    if not config.get('use_correlation_monitoring', True):
        return None

    xrp_candles = data.candles_5m.get('XRP/USDT', ())
    btc_candles = data.candles_5m.get('BTC/USDT', ())
    lookback = config.get('correlation_lookback', 50)

    correlation = calculate_correlation(
        list(xrp_candles), list(btc_candles), lookback
    )

    if correlation is not None:
        # Track correlation history
        if 'correlation_history' not in state:
            state['correlation_history'] = []
        state['correlation_history'].append(correlation)
        # Keep bounded
        if len(state['correlation_history']) > 100:
            state['correlation_history'] = state['correlation_history'][-100:]

        # Store latest
        state['xrp_btc_correlation'] = correlation

        # REC-001 (v4.2.0): Check for low correlation warning (0.4 threshold)
        warn_threshold = config.get('correlation_warn_threshold', 0.4)
        if correlation < warn_threshold and not state.get('_correlation_warned', False):
            state['_correlation_warned'] = True

        # REC-001 (v4.2.0): Check for critical correlation pause (0.25 threshold)
        pause_threshold = config.get('correlation_pause_threshold', 0.25)
        pause_enabled = config.get('correlation_pause_enabled', True)
        state['correlation_below_pause_threshold'] = (
            pause_enabled and correlation < pause_threshold
        )

    return correlation


def should_pause_for_low_correlation(
    symbol: str,
    state: Dict[str, Any],
    config: Dict[str, Any]
) -> bool:
    """
    Check if XRP/BTC trading should pause due to low correlation.

    REC-001 (v4.2.0): With XRP-BTC correlation at ~40% (down from ~80%),
    pause ratio trading when correlation drops below critical threshold.

    Args:
        symbol: Trading symbol
        state: Strategy state
        config: Strategy configuration

    Returns:
        True if trading should be paused for this symbol
    """
    # Only applies to XRP/BTC ratio trading
    if symbol != 'XRP/BTC':
        return False

    if not config.get('correlation_pause_enabled', True):
        return False

    return state.get('correlation_below_pause_threshold', False)


# =============================================================================
# Fee Profitability Checks - REC-002 (v4.1.0)
# =============================================================================
def check_fee_profitability(
    expected_profit_pct: float,
    config: Dict[str, Any]
) -> Tuple[bool, float]:
    """
    Check if trade is profitable after round-trip fees.

    REC-002 (v4.1.0): Guide v2.0 Section 23 compliance.
    With typical 0.1% maker/taker fees, round-trip cost is ~0.2%.

    Args:
        expected_profit_pct: Expected take profit percentage
        config: Strategy configuration

    Returns:
        Tuple of (is_profitable, net_profit_pct)
    """
    if not config.get('check_fee_profitability', True):
        return True, expected_profit_pct

    fee_rate = config.get('estimated_fee_rate', 0.001)
    min_net_profit = config.get('min_net_profit_pct', 0.05)

    # Round-trip fees: entry + exit
    round_trip_fee_pct = fee_rate * 2 * 100  # Convert to percentage
    net_profit_pct = expected_profit_pct - round_trip_fee_pct

    return net_profit_pct >= min_net_profit, net_profit_pct


# =============================================================================
# Trailing Stops - REC-006 (v3.0.0)
# =============================================================================
def calculate_trailing_stop(
    entry_price: float,
    highest_price: float,
    lowest_price: float,
    side: str,
    activation_pct: float,
    trail_distance_pct: float,
) -> Optional[float]:
    """
    Calculate trailing stop price if activated.

    Args:
        entry_price: Original entry price
        highest_price: Highest price since entry (for longs)
        lowest_price: Lowest price since entry (for shorts)
        side: 'long' or 'short'
        activation_pct: Profit % to activate trailing stop
        trail_distance_pct: Distance % to trail from extreme

    Returns:
        Trailing stop price if activated, None otherwise
    """
    if entry_price <= 0:
        return None

    if side == 'long':
        profit_pct = ((highest_price - entry_price) / entry_price) * 100
        if profit_pct >= activation_pct:
            return highest_price * (1 - trail_distance_pct / 100)
    elif side == 'short':
        profit_pct = ((entry_price - lowest_price) / entry_price) * 100
        if profit_pct >= activation_pct:
            return lowest_price * (1 + trail_distance_pct / 100)

    return None


def update_position_extremes(
    state: Dict[str, Any],
    symbol: str,
    current_price: float
) -> None:
    """Update highest/lowest prices for trailing stop calculation."""
    if 'position_entries' not in state:
        return

    entry = state['position_entries'].get(symbol)
    if not entry:
        return

    # Initialize extremes if not present
    if 'highest_price' not in entry:
        entry['highest_price'] = entry.get('entry_price', current_price)
    if 'lowest_price' not in entry:
        entry['lowest_price'] = entry.get('entry_price', current_price)

    # Update extremes
    entry['highest_price'] = max(entry['highest_price'], current_price)
    entry['lowest_price'] = min(entry['lowest_price'], current_price)


# =============================================================================
# Position Decay - REC-007 (v3.0.0)
# =============================================================================
def get_decayed_take_profit(
    entry_price: float,
    original_tp_pct: float,
    entry_time: datetime,
    current_time: datetime,
    side: str,
    config: Dict[str, Any]
) -> Tuple[float, float]:
    """
    Calculate decayed take profit based on position age.

    Mean reversion assumes timely return to mean. If price doesn't revert
    within expected timeframe, reduce TP to exit sooner.

    Args:
        entry_price: Original entry price
        original_tp_pct: Original take profit percentage
        entry_time: Time of entry
        current_time: Current time
        side: 'long' or 'short'
        config: Strategy configuration

    Returns:
        Tuple of (decayed_tp_price, decay_multiplier)
    """
    decay_start = config.get('decay_start_minutes', 3.0)
    decay_interval = config.get('decay_interval_minutes', 1.0)
    decay_multipliers = config.get('decay_multipliers', [1.0, 0.75, 0.5, 0.25])

    # Calculate position age in minutes
    if entry_time is None:
        return entry_price, 1.0

    age_minutes = (current_time - entry_time).total_seconds() / 60

    # No decay before start time
    if age_minutes < decay_start:
        multiplier = 1.0
    else:
        # Calculate decay stage
        decay_stage = int((age_minutes - decay_start) / decay_interval)
        decay_stage = min(decay_stage, len(decay_multipliers) - 1)
        multiplier = decay_multipliers[decay_stage]

    # Calculate decayed TP
    effective_tp_pct = original_tp_pct * multiplier

    if side == 'long':
        decayed_tp = entry_price * (1 + effective_tp_pct / 100)
    else:  # short
        decayed_tp = entry_price * (1 - effective_tp_pct / 100)

    return decayed_tp, multiplier


# =============================================================================
# ADX Trend Strength Filter - REC-003 (v4.3.0)
# =============================================================================
def check_adx_strong_trend(
    candles: List,
    symbol: str,
    config: Dict[str, Any],
    state: Dict[str, Any]
) -> Tuple[bool, Optional[float]]:
    """
    Check if ADX indicates a strong trend (unsuitable for mean reversion).

    REC-003 (v4.3.0): Deep Review v8.0 HIGH-001 finding.
    Research shows BTC exhibits stronger trending behavior than mean reversion.
    "BTC tends to trend when it is at its maximum and bounce back when at the minimum."

    Only applies to BTC/USDT by default. ADX > 25 indicates strong trend.

    Args:
        candles: List of candles for ADX calculation
        symbol: Trading symbol
        config: Strategy configuration
        state: Strategy state

    Returns:
        Tuple of (is_strong_trend, adx_value)
    """
    # Only apply to BTC/USDT unless explicitly configured otherwise
    if symbol != 'BTC/USDT' and not config.get('adx_filter_all_symbols', False):
        return False, None

    if not config.get('use_adx_filter', True):
        return False, None

    period = config.get('adx_period', 14)
    threshold = config.get('adx_strong_trend_threshold', 25)

    adx = calculate_adx(candles, period)

    if adx is None:
        return False, None

    # Store ADX in state for monitoring
    if 'adx_by_symbol' not in state:
        state['adx_by_symbol'] = {}
    state['adx_by_symbol'][symbol] = adx

    is_strong_trend = adx > threshold

    return is_strong_trend, adx
