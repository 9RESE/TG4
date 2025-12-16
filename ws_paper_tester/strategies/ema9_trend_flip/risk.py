"""
EMA-9 Trend Flip Strategy - Risk Management

Position sizing and risk management functions.
"""
from typing import Dict, Any, Tuple, Optional

from ws_tester.types import Signal


def check_position_limits(
    state: Dict[str, Any],
    config: Dict[str, Any]
) -> Tuple[bool, float]:
    """
    Check if position limits allow a new trade.

    Args:
        state: Strategy state
        config: Strategy configuration

    Returns:
        Tuple of (can_trade: bool, available_size: float)
    """
    current_position = state.get('position', 0)
    max_position = config.get('max_position_usd', 100.0)
    min_trade_size = config.get('min_trade_size_usd', 10.0)

    available = max_position - current_position

    if available < min_trade_size:
        return (False, 0.0)

    return (True, available)


def calculate_entry_stops(
    price: float,
    direction: str,
    config: Dict[str, Any],
    atr: Optional[float] = None
) -> Tuple[float, float]:
    """
    Calculate stop loss and take profit levels for an entry.

    Args:
        price: Entry price
        direction: 'long' or 'short'
        config: Strategy configuration
        atr: Average True Range value (optional, for ATR-based stops)

    Returns:
        Tuple of (stop_loss, take_profit)
    """
    if config.get('use_atr_stops', False) and atr:
        atr_sl_mult = config.get('atr_stop_mult', 1.5)
        atr_tp_mult = config.get('atr_tp_mult', 3.0)

        if direction == 'long':
            stop_loss = price - (atr * atr_sl_mult)
            take_profit = price + (atr * atr_tp_mult)
        else:  # short
            stop_loss = price + (atr * atr_sl_mult)
            take_profit = price - (atr * atr_tp_mult)
    else:
        sl_pct = config.get('stop_loss_pct', 1.0)
        tp_pct = config.get('take_profit_pct', 2.0)

        if direction == 'long':
            stop_loss = price * (1 - sl_pct / 100)
            take_profit = price * (1 + tp_pct / 100)
        else:  # short
            stop_loss = price * (1 + sl_pct / 100)
            take_profit = price * (1 - tp_pct / 100)

    return (stop_loss, take_profit)


def create_entry_signal(
    symbol: str,
    direction: str,
    price: float,
    ema: float,
    config: Dict[str, Any],
    state: Dict[str, Any],
    atr: Optional[float],
    prev_count: int
) -> Optional[Signal]:
    """
    Create entry signal with appropriate stop loss and take profit.

    Args:
        symbol: Trading symbol
        direction: 'long' or 'short'
        price: Entry price
        ema: Current EMA value
        config: Strategy configuration
        state: Strategy state
        atr: Average True Range value
        prev_count: Previous consecutive candle count

    Returns:
        Signal object or None if position limits exceeded
    """
    position_size = config.get('position_size_usd', 50.0)

    # Check position limits
    can_trade, available = check_position_limits(state, config)
    if not can_trade:
        return None

    actual_size = min(position_size, available)

    # Calculate stops
    stop_loss, take_profit = calculate_entry_stops(price, direction, config, atr)

    # Create signal
    if direction == 'long':
        return Signal(
            action='buy',
            symbol=symbol,
            size=actual_size,
            price=price,
            reason=f"EMA9: Long flip ({prev_count} candles below, now above EMA={ema:.2f})",
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata={
                'entry_type': 'ema_flip_long',
                'ema_9': ema,
                'consecutive_candles': prev_count,
            }
        )
    else:  # short
        return Signal(
            action='short',
            symbol=symbol,
            size=actual_size,
            price=price,
            reason=f"EMA9: Short flip ({prev_count} candles above, now below EMA={ema:.2f})",
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata={
                'entry_type': 'ema_flip_short',
                'ema_9': ema,
                'consecutive_candles': prev_count,
            }
        )


def track_rejection(
    state: Dict[str, Any],
    reason: 'RejectionReason',
    symbol: str = None
) -> None:
    """
    Track signal rejection for analysis.

    Args:
        state: Strategy state
        reason: RejectionReason enum value
        symbol: Optional symbol for per-symbol tracking
    """
    if 'rejection_counts' not in state:
        state['rejection_counts'] = {}

    reason_key = reason.value
    state['rejection_counts'][reason_key] = state['rejection_counts'].get(reason_key, 0) + 1
