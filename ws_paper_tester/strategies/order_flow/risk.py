"""
Order Flow Strategy - Risk Management

Contains fee profitability checks, trailing stop calculations, correlation management,
circuit breaker logic, and trade flow confirmation.
"""
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List

from ws_tester.types import DataSnapshot


def check_fee_profitability(
    expected_move_pct: float,
    fee_rate: float,
    min_profit_pct: float
) -> Tuple[bool, float]:
    """Check if trade is profitable after fees."""
    round_trip_fee_pct = fee_rate * 2 * 100  # Convert to percentage
    net_profit_pct = expected_move_pct - round_trip_fee_pct
    return net_profit_pct >= min_profit_pct, net_profit_pct


def calculate_trailing_stop(
    entry_price: float,
    highest_price: float,
    side: str,
    activation_pct: float,
    trail_distance_pct: float
) -> Optional[float]:
    """Calculate trailing stop level."""
    if side == 'long':
        profit_pct = (highest_price - entry_price) / entry_price * 100
        if profit_pct >= activation_pct:
            return highest_price * (1 - trail_distance_pct / 100)
    elif side == 'short':
        profit_pct = (entry_price - highest_price) / entry_price * 100
        if profit_pct >= activation_pct:
            return highest_price * (1 + trail_distance_pct / 100)
    return None


def get_progressive_decay_multiplier(
    position_age_seconds: float,
    decay_stages: List[Tuple[int, float]]
) -> float:
    """
    Get progressive TP multiplier based on position age.

    Returns:
        TP multiplier (1.0 if no decay, lower values for older positions)
    """
    sorted_stages = sorted(decay_stages, key=lambda x: x[0])

    multiplier = 1.0
    for age_threshold, tp_mult in sorted_stages:
        if position_age_seconds >= age_threshold:
            multiplier = tp_mult
        else:
            break

    return multiplier


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


def is_trade_flow_aligned(
    data: DataSnapshot,
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


def check_correlation_exposure(
    state: Dict[str, Any],
    symbol: str,
    direction: str,
    size: float,
    config: Dict[str, Any]
) -> Tuple[bool, float]:
    """Check and adjust for cross-pair correlation."""
    if not config.get('use_correlation_management', True):
        return True, size

    max_long = config.get('max_total_long_exposure', 150.0)
    max_short = config.get('max_total_short_exposure', 150.0)
    same_dir_mult = config.get('same_direction_size_mult', 0.75)

    position_by_symbol = state.get('position_by_symbol', {})
    position_entries = state.get('position_entries', {})

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
            if direction in ('sell', 'short'):
                other_symbols_same_direction = True

    adjusted_size = size

    if other_symbols_same_direction:
        adjusted_size = size * same_dir_mult

    if direction == 'buy':
        if total_long + adjusted_size > max_long:
            available = max(0, max_long - total_long)
            adjusted_size = min(adjusted_size, available)
    elif direction in ('sell', 'short'):
        if total_short + adjusted_size > max_short:
            available = max(0, max_short - total_short)
            adjusted_size = min(adjusted_size, available)

    can_trade = adjusted_size >= config.get('min_trade_size_usd', 5.0)

    return can_trade, adjusted_size
