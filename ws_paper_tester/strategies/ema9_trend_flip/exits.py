"""
EMA-9 Trend Flip Strategy - Exit Logic

Exit condition checks for open positions.
"""
from datetime import datetime
from typing import Dict, Any, Optional

from ws_tester.types import Signal


def check_ema_flip_exit(
    position_side: str,
    current_position: str,
    current_ema: float
) -> Optional[str]:
    """
    Check if EMA has flipped to opposite side.

    Args:
        position_side: Current position ('long' or 'short')
        current_position: Current candle position relative to EMA
        current_ema: Current EMA value

    Returns:
        Exit reason string if flip detected, None otherwise
    """
    if position_side == 'long' and current_position == 'below':
        return f"EMA flip to below (EMA={current_ema:.2f})"
    elif position_side == 'short' and current_position == 'above':
        return f"EMA flip to above (EMA={current_ema:.2f})"
    return None


def check_max_hold_time_exit(
    entry_time: datetime,
    current_time: datetime,
    max_hold_hours: float
) -> Optional[str]:
    """
    Check if position has exceeded maximum hold time.

    Args:
        entry_time: Position entry timestamp
        current_time: Current timestamp
        max_hold_hours: Maximum hold time in hours

    Returns:
        Exit reason string if exceeded, None otherwise
    """
    if entry_time is None:
        return None

    hold_duration = (current_time - entry_time).total_seconds() / 3600
    if hold_duration >= max_hold_hours:
        return f"Max hold time ({max_hold_hours}h) reached"

    return None


def check_exit_conditions(
    state: Dict[str, Any],
    symbol: str,
    current_price: float,
    current_ema: float,
    current_position: str,
    current_time: datetime,
    config: Dict[str, Any],
    atr: Optional[float] = None
) -> Optional[Signal]:
    """
    Check all exit conditions for an open position.

    Args:
        state: Strategy state
        symbol: Trading symbol
        current_price: Current market price
        current_ema: Current EMA value
        current_position: Current candle position ('above', 'below', 'neutral')
        current_time: Current timestamp
        config: Strategy configuration
        atr: Average True Range value (optional)

    Returns:
        Signal if exit condition met, None otherwise
    """
    position_side = state.get('position_side')
    entry_price = state.get('entry_price', 0)
    position_value = state.get('position', 0)

    if not position_side or position_value <= 0:
        return None

    exit_reason = None

    # Check EMA flip exit
    if config.get('exit_on_flip', True):
        exit_reason = check_ema_flip_exit(position_side, current_position, current_ema)

    # Check max hold time
    if not exit_reason and state.get('entry_time'):
        max_hold_hours = config.get('max_hold_hours', 72)
        exit_reason = check_max_hold_time_exit(
            state['entry_time'],
            current_time,
            max_hold_hours
        )

    # Generate exit signal if needed
    if exit_reason:
        if position_side == 'long':
            pnl_pct = (current_price - entry_price) / entry_price * 100
            return Signal(
                action='sell',
                symbol=symbol,
                size=position_value,
                price=current_price,
                reason=f"EMA9: Exit long - {exit_reason}, P&L={pnl_pct:.2f}%",
                metadata={
                    'exit_type': 'ema_flip' if 'flip' in exit_reason.lower() else 'time_exit',
                    'entry_price': entry_price,
                    'pnl_pct': pnl_pct,
                }
            )
        else:  # short
            pnl_pct = (entry_price - current_price) / entry_price * 100
            return Signal(
                action='cover',
                symbol=symbol,
                size=position_value,
                price=current_price,
                reason=f"EMA9: Exit short - {exit_reason}, P&L={pnl_pct:.2f}%",
                metadata={
                    'exit_type': 'ema_flip' if 'flip' in exit_reason.lower() else 'time_exit',
                    'entry_price': entry_price,
                    'pnl_pct': pnl_pct,
                }
            )

    return None
