"""
Ratio Trading Strategy - Risk Management Module

Circuit breaker, spread monitoring, and trade flow confirmation.
"""
from datetime import datetime
from typing import Dict, Any, Tuple


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
    data,
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


def check_spread(
    data,
    symbol: str,
    max_spread_pct: float,
    take_profit_pct: float,
    min_profitability_mult: float
) -> Tuple[bool, float]:
    """
    Check if spread is acceptable for trading.

    Returns:
        (is_acceptable, current_spread_pct)
    """
    ob = data.orderbooks.get(symbol)
    if not ob or not ob.spread_pct:
        return True, 0.0  # Allow trading if no orderbook data

    spread_pct = ob.spread_pct

    # Check against max spread
    if spread_pct > max_spread_pct:
        return False, spread_pct

    # Check profitability: TP must exceed spread by multiplier
    if take_profit_pct < spread_pct * min_profitability_mult:
        return False, spread_pct

    return True, spread_pct


def calculate_position_size(
    config: Dict[str, Any],
    current_position_usd: float,
    regime_size_mult: float
) -> Tuple[float, float]:
    """
    Calculate actual position size in USD.

    Returns:
        (actual_size_usd, available_usd)
    """
    max_position = config.get('max_position_usd', 50.0)
    base_size = config.get('position_size_usd', 15.0)
    min_trade_size = config.get('min_trade_size_usd', 5.0)

    available = max_position - current_position_usd
    adjusted_size = base_size * regime_size_mult
    actual_size = min(adjusted_size, available)

    if actual_size < min_trade_size:
        actual_size = 0.0

    return actual_size, available
