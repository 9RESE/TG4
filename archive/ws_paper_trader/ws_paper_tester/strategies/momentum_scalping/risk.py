"""
Momentum Scalping Strategy - Risk Management

Contains fee profitability checks, position limit checks, correlation management,
circuit breaker logic, and trend strength filters.

v2.0.0 REC-001: Added XRP/BTC correlation monitoring with pause thresholds
v2.0.0 REC-003: Added ADX strong trend filter for BTC/USDT
"""
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, List

from ws_tester.types import DataSnapshot

from .indicators import calculate_correlation, calculate_adx


def check_fee_profitability(
    expected_move_pct: float,
    fee_rate: float,
    min_profit_pct: float
) -> Tuple[bool, float]:
    """
    Check if trade is profitable after fees.

    Transaction cost erosion is a critical risk for momentum scalping:
    - 0.1% fee × 2 = 0.2% per round trip
    - Minimum TP target should be 0.5% for 0.3% net profit

    Args:
        expected_move_pct: Expected price move percentage (e.g., 0.8 for 0.8%)
        fee_rate: Fee rate per trade (e.g., 0.001 for 0.1%)
        min_profit_pct: Minimum acceptable profit percentage after fees

    Returns:
        Tuple of (is_profitable, net_profit_pct)
    """
    # Round trip fee in percentage terms
    round_trip_fee_pct = fee_rate * 2 * 100

    # Net profit after fees
    net_profit_pct = expected_move_pct - round_trip_fee_pct

    is_profitable = net_profit_pct >= min_profit_pct

    return is_profitable, net_profit_pct


def check_circuit_breaker(
    state: Dict[str, Any],
    current_time: datetime,
    max_losses: int,
    cooldown_minutes: float
) -> bool:
    """
    Check if circuit breaker is active.

    Circuit breaker triggers after max_consecutive_losses and pauses
    trading for cooldown_minutes.

    From research:
    "Professional scalpers implement hard rules: maximum loss per trade
    (typically 0.1% of capital), daily loss limits (usually 2% of capital)."

    Args:
        state: Strategy state dict
        current_time: Current timestamp
        max_losses: Maximum consecutive losses before circuit breaker
        cooldown_minutes: Cooldown duration in minutes

    Returns:
        True if circuit breaker is active and trading should be paused
    """
    consecutive_losses = state.get('consecutive_losses', 0)

    # Not enough losses to trigger
    if consecutive_losses < max_losses:
        return False

    # Check if we're in cooldown period
    breaker_time = state.get('circuit_breaker_time')
    if breaker_time is None:
        return False

    elapsed_minutes = (current_time - breaker_time).total_seconds() / 60

    # Cooldown period expired - reset and allow trading
    if elapsed_minutes >= cooldown_minutes:
        state['consecutive_losses'] = 0
        state['circuit_breaker_time'] = None
        return False

    # Still in cooldown
    return True


def check_correlation_exposure(
    state: Dict[str, Any],
    symbol: str,
    direction: str,
    size: float,
    config: Dict[str, Any]
) -> Tuple[bool, float]:
    """
    Check and adjust for cross-pair correlation.

    From research (XRP-BTC correlation: 0.84):
    - Simultaneous signals on XRP/USDT and BTC/USDT are correlated
    - Need total exposure limits across all pairs
    - Reduce size if multiple pairs have same direction

    Args:
        state: Strategy state dict
        symbol: Symbol being traded
        direction: Trade direction ('buy' or 'short')
        size: Requested position size
        config: Strategy configuration

    Returns:
        Tuple of (can_trade, adjusted_size)
    """
    if not config.get('use_correlation_management', True):
        return True, size

    max_long = config.get('max_total_long_exposure', 100.0)
    max_short = config.get('max_total_short_exposure', 100.0)
    same_dir_mult = config.get('same_direction_size_mult', 0.75)

    position_by_symbol = state.get('position_by_symbol', {})
    position_entries = state.get('position_entries', {})

    # Calculate current exposure
    total_long = 0.0
    total_short = 0.0
    other_symbols_same_direction = False

    for sym, pos_data in position_entries.items():
        if sym == symbol:
            continue

        pos_size = position_by_symbol.get(sym, 0)
        pos_side = pos_data.get('side', '')

        if pos_side == 'long':
            total_long += pos_size
            if direction == 'buy':
                other_symbols_same_direction = True
        elif pos_side == 'short':
            total_short += pos_size
            if direction == 'short':
                other_symbols_same_direction = True

    # Reduce size if other symbols have same direction
    adjusted_size = size
    if other_symbols_same_direction:
        adjusted_size = size * same_dir_mult

    # Check total exposure limits
    if direction == 'buy':
        if total_long + adjusted_size > max_long:
            available = max(0, max_long - total_long)
            adjusted_size = min(adjusted_size, available)
    elif direction == 'short':
        if total_short + adjusted_size > max_short:
            available = max(0, max_short - total_short)
            adjusted_size = min(adjusted_size, available)

    min_trade = config.get('min_trade_size_usd', 5.0)
    can_trade = adjusted_size >= min_trade

    return can_trade, adjusted_size


def check_position_limits(
    state: Dict[str, Any],
    symbol: str,
    requested_size: float,
    config: Dict[str, Any]
) -> Tuple[bool, float, str]:
    """
    Check position limits and calculate available size.

    Enforces both total and per-symbol position limits.

    Args:
        state: Strategy state dict
        symbol: Symbol being traded
        requested_size: Requested position size
        config: Strategy configuration

    Returns:
        Tuple of (can_trade, available_size, limit_reason)
    """
    current_position = state.get('position_size', 0)
    current_position_symbol = state.get('position_by_symbol', {}).get(symbol, 0)

    max_position = config.get('max_position_usd', 75.0)
    max_position_symbol = config.get('max_position_per_symbol_usd', 50.0)
    min_trade = config.get('min_trade_size_usd', 5.0)

    # Check total position limit
    if current_position >= max_position:
        return False, 0.0, 'total_limit'

    # Check per-symbol position limit
    if current_position_symbol >= max_position_symbol:
        return False, 0.0, 'symbol_limit'

    # Calculate available size respecting both limits
    available_total = max_position - current_position
    available_symbol = max_position_symbol - current_position_symbol
    available = min(available_total, available_symbol)

    actual_size = min(requested_size, available)

    if actual_size < min_trade:
        return False, actual_size, 'min_trade'

    return True, actual_size, 'ok'


def is_volume_confirmed(
    volume_ratio: float,
    threshold: float
) -> bool:
    """
    Check if volume confirms the momentum signal.

    From research:
    "Breakouts accompanied by an increasing volume are more reliable since
    false breakouts are also common."

    Args:
        volume_ratio: Current volume ratio vs average
        threshold: Volume spike threshold (e.g., 1.5)

    Returns:
        True if volume confirms the signal
    """
    return volume_ratio >= threshold


def calculate_position_age(
    state: Dict[str, Any],
    symbol: str,
    current_time: datetime
) -> float:
    """
    Calculate how long a position has been held.

    Used for time-based exits (momentum scalping targets 1-3 minute holds).

    Args:
        state: Strategy state dict
        symbol: Symbol to check
        current_time: Current timestamp

    Returns:
        Position age in seconds, or 0 if no position
    """
    pos_entry = state.get('position_entries', {}).get(symbol)
    if not pos_entry:
        return 0.0

    entry_time = pos_entry.get('entry_time')
    if not entry_time:
        return 0.0

    return (current_time - entry_time).total_seconds()


def calculate_position_pnl(
    state: Dict[str, Any],
    symbol: str,
    current_price: float
) -> Tuple[float, float]:
    """
    Calculate current position PnL.

    Args:
        state: Strategy state dict
        symbol: Symbol to check
        current_price: Current market price

    Returns:
        Tuple of (pnl_pct, pnl_usd) or (0, 0) if no position
    """
    pos_entry = state.get('position_entries', {}).get(symbol)
    if not pos_entry:
        return 0.0, 0.0

    entry_price = pos_entry.get('entry_price', 0)
    if entry_price <= 0:
        return 0.0, 0.0

    position_size = state.get('position_by_symbol', {}).get(symbol, 0)

    if pos_entry['side'] == 'long':
        pnl_pct = (current_price - entry_price) / entry_price * 100
    else:  # short
        pnl_pct = (entry_price - current_price) / entry_price * 100

    # Approximate USD PnL (position is in USD notional)
    pnl_usd = position_size * (pnl_pct / 100)

    return pnl_pct, pnl_usd


# =============================================================================
# XRP/BTC Correlation Monitoring (REC-001 v2.0.0)
# =============================================================================
def get_xrp_btc_correlation(
    data: DataSnapshot,
    config: Dict[str, Any],
    state: Dict[str, Any]
) -> Optional[float]:
    """
    Get XRP/BTC correlation and update tracking.

    REC-001 (v2.0.0): XRP-BTC correlation has declined from ~0.85 to ~0.40-0.67.
    Momentum signals on XRP/BTC are unreliable when correlation is low.

    Args:
        data: Market data snapshot
        config: Strategy configuration
        state: Strategy state

    Returns:
        Correlation coefficient or None if unavailable
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

        # Check for low correlation warning
        warn_threshold = config.get('correlation_warn_threshold', 0.55)
        if correlation < warn_threshold and not state.get('_correlation_warned', False):
            state['_correlation_warned'] = True

        # Check for critical correlation pause
        pause_threshold = config.get('correlation_pause_threshold', 0.50)
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

    REC-001 (v2.0.0): With XRP-BTC correlation at ~40-67% (down from ~85%),
    pause XRP/BTC trading when correlation drops below critical threshold.

    Args:
        symbol: Trading symbol
        state: Strategy state
        config: Strategy configuration

    Returns:
        True if trading should be paused for this symbol
    """
    # Only applies to XRP/BTC
    if symbol != 'XRP/BTC':
        return False

    if not config.get('correlation_pause_enabled', True):
        return False

    return state.get('correlation_below_pause_threshold', False)


# =============================================================================
# ADX Strong Trend Filter (REC-003 v2.0.0)
# =============================================================================
def check_adx_strong_trend(
    candles: List,
    symbol: str,
    config: Dict[str, Any],
    state: Dict[str, Any]
) -> Tuple[bool, Optional[float]]:
    """
    Check if ADX indicates a strong trend (unsuitable for momentum scalping).

    REC-003 (v2.0.0): BTC exhibits strong trending behavior at price extremes.
    Research: "BTC tends to trend when it is at its maximum and bounce back
    when at the minimum."

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
    if config.get('adx_filter_btc_only', True) and symbol != 'BTC/USDT':
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
