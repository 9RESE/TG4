"""
Grid RSI Reversion Strategy - Risk Management

Contains risk checks for:
- Position accumulation limits
- Maximum position size
- Correlation exposure
- Circuit breaker
- Trend filter
"""
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

from .config import get_symbol_config
from .grid import count_filled_levels


def check_accumulation_limit(
    state: Dict[str, Any],
    symbol: str,
    config: Dict[str, Any]
) -> Tuple[bool, int, int]:
    """
    Check if accumulation limit allows new entry.

    Grid strategies can accumulate positions as price moves against them.
    This limit prevents excessive accumulation in one direction.

    Args:
        state: Strategy state dict
        symbol: Trading symbol
        config: Strategy configuration

    Returns:
        Tuple of (can_enter, filled_count, max_allowed)
    """
    max_accumulation = get_symbol_config(symbol, config, 'max_accumulation_levels')

    grid_levels = state.get('grid_levels', {}).get(symbol, [])
    if not grid_levels:
        return True, 0, max_accumulation

    # Count filled buy levels (accumulation)
    filled_buys = count_filled_levels(grid_levels, 'buy')

    can_enter = filled_buys < max_accumulation
    return can_enter, filled_buys, max_accumulation


def check_position_limits(
    state: Dict[str, Any],
    symbol: str,
    requested_size: float,
    config: Dict[str, Any]
) -> Tuple[bool, float, str]:
    """
    Check if position limits allow the trade.

    Args:
        state: Strategy state dict
        symbol: Trading symbol
        requested_size: Requested trade size in USD
        config: Strategy configuration

    Returns:
        Tuple of (can_trade, available_size, limit_reason)
    """
    # Per-symbol limit
    max_per_symbol = get_symbol_config(symbol, config, 'max_position_usd')
    current_symbol_position = state.get('position_by_symbol', {}).get(symbol, 0)

    available_symbol = max_per_symbol - current_symbol_position
    if available_symbol <= 0:
        return False, 0, 'symbol_limit'

    # Total exposure limit
    max_total = config.get('max_total_long_exposure', 150.0)
    total_position = sum(state.get('position_by_symbol', {}).values())

    available_total = max_total - total_position
    if available_total <= 0:
        return False, 0, 'total_limit'

    # Use the smaller of available limits
    available = min(available_symbol, available_total, requested_size)

    # Check minimum size
    min_size = config.get('min_trade_size_usd', 5.0)
    if available < min_size:
        return False, 0, 'min_size'

    return True, available, 'ok'


def check_correlation_exposure(
    state: Dict[str, Any],
    symbol: str,
    side: str,
    size: float,
    config: Dict[str, Any],
    correlations: Dict[str, float] = None
) -> Tuple[bool, float, str]:
    """
    Check and adjust for correlation exposure across pairs.

    REC-005: Enhanced with real correlation monitoring.

    When multiple correlated pairs are in the same direction,
    reduce position size to limit correlation risk.

    Args:
        state: Strategy state dict
        symbol: Trading symbol
        side: 'buy' or 'sell'
        size: Proposed trade size
        config: Strategy configuration
        correlations: Dict of symbol pairs to correlation values (REC-005)

    Returns:
        Tuple of (can_enter, adjusted_size, reason)
    """
    if not config.get('use_correlation_management', True):
        return True, size, 'correlation_disabled'

    position_by_symbol = state.get('position_by_symbol', {})
    correlation_block_threshold = config.get('correlation_block_threshold', 0.85)
    use_real_correlation = config.get('use_real_correlation', True)

    # REC-005: Check real correlation if available
    if use_real_correlation and correlations:
        for other_symbol, other_pos in position_by_symbol.items():
            if other_symbol == symbol or other_pos <= 0:
                continue

            # Get correlation between this symbol and other
            pair_key = f"{symbol}_{other_symbol}"
            reverse_key = f"{other_symbol}_{symbol}"
            correlation = correlations.get(pair_key) or correlations.get(reverse_key)

            if correlation is not None and correlation > correlation_block_threshold:
                # Block entry due to high correlation with existing position
                return False, 0, f'high_correlation_{other_symbol}_{correlation:.2f}'

    # Original position-based correlation check
    same_direction_count = 0
    for sym, pos in position_by_symbol.items():
        if sym != symbol and pos > 0:
            same_direction_count += 1

    if same_direction_count > 0:
        # Reduce size for correlated exposure
        mult = config.get('same_direction_size_mult', 0.75)
        adjusted_size = size * (mult ** same_direction_count)
        min_size = config.get('min_trade_size_usd', 5.0)
        if adjusted_size < min_size:
            return False, 0, 'correlation_size_too_small'
        return True, adjusted_size, f'size_reduced_{same_direction_count}_positions'

    return True, size, 'ok'


def check_circuit_breaker(
    state: Dict[str, Any],
    current_time: datetime,
    max_losses: int,
    cooldown_min: int
) -> bool:
    """
    Check if circuit breaker is active.

    Args:
        state: Strategy state dict
        current_time: Current timestamp
        max_losses: Max consecutive losses to trigger
        cooldown_min: Cooldown period in minutes

    Returns:
        True if trading should be blocked
    """
    consecutive_losses = state.get('consecutive_losses', 0)

    if consecutive_losses < max_losses:
        return False

    circuit_breaker_time = state.get('circuit_breaker_time')
    if circuit_breaker_time is None:
        return False

    elapsed = (current_time - circuit_breaker_time).total_seconds() / 60

    if elapsed >= cooldown_min:
        # Reset circuit breaker
        state['consecutive_losses'] = 0
        state['circuit_breaker_time'] = None
        return False

    return True


def check_trend_filter(
    adx: Optional[float],
    config: Dict[str, Any]
) -> Tuple[bool, Optional[float]]:
    """
    Check if trend filter blocks trading.

    Grid strategies perform poorly in strong trends.
    ADX > threshold indicates a strong trend.

    Args:
        adx: Current ADX value
        config: Strategy configuration

    Returns:
        Tuple of (is_trending, adx_value)
    """
    if not config.get('use_trend_filter', True):
        return False, adx

    if adx is None:
        return False, None

    threshold = config.get('adx_threshold', 30)
    is_trending = adx > threshold

    return is_trending, adx


def check_max_drawdown(
    state: Dict[str, Any],
    config: Dict[str, Any]
) -> Tuple[bool, float]:
    """
    Check if max drawdown has been exceeded.

    Args:
        state: Strategy state dict
        config: Strategy configuration

    Returns:
        Tuple of (is_exceeded, current_drawdown_pct)
    """
    max_drawdown_pct = config.get('max_drawdown_pct', 10.0)

    # Calculate total unrealized P&L
    total_pnl = sum(state.get('pnl_by_symbol', {}).values())
    total_position = sum(state.get('position_by_symbol', {}).values())

    if total_position <= 0:
        return False, 0.0

    # Simple drawdown calculation from peak
    peak_value = state.get('peak_value', total_position)
    current_value = total_position + total_pnl

    if current_value > peak_value:
        state['peak_value'] = current_value
        return False, 0.0

    drawdown_pct = (peak_value - current_value) / peak_value * 100

    return drawdown_pct >= max_drawdown_pct, drawdown_pct


def calculate_position_pnl(
    state: Dict[str, Any],
    symbol: str,
    current_price: float
) -> Tuple[float, float]:
    """
    Calculate unrealized P&L for a position.

    Args:
        state: Strategy state dict
        symbol: Trading symbol
        current_price: Current market price

    Returns:
        Tuple of (pnl_pct, pnl_usd)
    """
    position_entries = state.get('position_entries', {})
    pos = position_entries.get(symbol)

    if not pos:
        return 0.0, 0.0

    entry_price = pos.get('entry_price', 0)
    if entry_price <= 0:
        return 0.0, 0.0

    position_size = state.get('position_by_symbol', {}).get(symbol, 0)
    if position_size <= 0:
        return 0.0, 0.0

    # Grid strategy is long-only
    pnl_pct = (current_price - entry_price) / entry_price * 100

    # Estimate position size in base asset
    base_size = position_size / entry_price
    pnl_usd = (current_price - entry_price) * base_size

    return pnl_pct, pnl_usd


def calculate_position_age(
    state: Dict[str, Any],
    symbol: str,
    current_time: datetime
) -> float:
    """
    Calculate position age in seconds.

    Args:
        state: Strategy state dict
        symbol: Trading symbol
        current_time: Current timestamp

    Returns:
        Position age in seconds (0 if no position)
    """
    position_entries = state.get('position_entries', {})
    pos = position_entries.get(symbol)

    if not pos:
        return 0.0

    entry_time = pos.get('entry_time')
    if not entry_time:
        return 0.0

    return (current_time - entry_time).total_seconds()


def check_all_risk_limits(
    state: Dict[str, Any],
    symbol: str,
    requested_size: float,
    config: Dict[str, Any],
    adx: Optional[float] = None,
    current_time: Optional[datetime] = None,
    correlations: Dict[str, float] = None
) -> Tuple[bool, float, str]:
    """
    Check all risk limits for entry.

    REC-005: Enhanced with real correlation data.

    Args:
        state: Strategy state dict
        symbol: Trading symbol
        requested_size: Requested trade size
        config: Strategy configuration
        adx: Current ADX value
        current_time: Current timestamp
        correlations: Dict of symbol pair correlations (REC-005)

    Returns:
        Tuple of (can_trade, available_size, block_reason)
    """
    # Circuit breaker check
    if current_time and config.get('use_circuit_breaker', True):
        max_losses = config.get('max_consecutive_losses', 3)
        cooldown_min = config.get('circuit_breaker_minutes', 15)
        if check_circuit_breaker(state, current_time, max_losses, cooldown_min):
            return False, 0, 'circuit_breaker'

    # Trend filter check
    is_trending, _ = check_trend_filter(adx, config)
    if is_trending:
        return False, 0, 'trend_filter'

    # Max drawdown check
    drawdown_exceeded, _ = check_max_drawdown(state, config)
    if drawdown_exceeded:
        return False, 0, 'max_drawdown'

    # Accumulation limit check
    can_accumulate, _, _ = check_accumulation_limit(state, symbol, config)
    if not can_accumulate:
        return False, 0, 'accumulation_limit'

    # Position limits check
    can_trade, available, limit_reason = check_position_limits(
        state, symbol, requested_size, config
    )
    if not can_trade:
        return False, 0, limit_reason

    # Correlation exposure check (REC-005: now with real correlations)
    can_enter, adjusted_size, corr_reason = check_correlation_exposure(
        state, symbol, 'buy', available, config, correlations
    )
    if not can_enter:
        return False, 0, f'correlation_limit_{corr_reason}'

    return True, adjusted_size, 'ok'
