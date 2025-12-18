"""
EMA-9 Trend Flip Strategy - Exit Logic

Exit condition checks for open positions.

Strategy Philosophy:
- The EMA flip IS the profit exit (no fixed take_profit_pct)
- Stop loss is for PROTECTION ONLY (catastrophic moves before flip)
- Exit on flip is ALWAYS enabled - it's the core strategy
"""
from datetime import datetime
from typing import Dict, Any, Optional

from ws_tester.types import Signal


def check_ema_flip_exit(
    position_side: str,
    current_position: str,
    current_ema: float,
    exit_confirmation_count: int = 0,
    required_confirmations: int = 1
) -> Optional[str]:
    """
    Check if EMA has flipped to opposite side with optional confirmation.

    This is the PRIMARY exit mechanism for the EMA-9 trend flip strategy.
    When price flips back to the original side, the trend has reversed.

    Args:
        position_side: Current position ('long' or 'short')
        current_position: Current candle position relative to EMA ('above', 'below', 'neutral')
        current_ema: Current EMA value
        exit_confirmation_count: Number of candles already confirmed on opposite side
        required_confirmations: Number of confirmations needed before exit (default 1 = immediate)

    Returns:
        Exit reason string if flip confirmed, None otherwise
    """
    # Check if current candle is on opposite side
    flip_detected = False
    if position_side == 'long' and current_position == 'below':
        flip_detected = True
    elif position_side == 'short' and current_position == 'above':
        flip_detected = True

    if not flip_detected:
        return None

    # Check if we have enough confirmations
    total_confirmations = exit_confirmation_count + 1  # +1 for current candle

    if total_confirmations >= required_confirmations:
        return f"EMA flip to {current_position} (EMA={current_ema:.2f}, confirmed={total_confirmations})"

    # Not enough confirmations yet
    return None


def track_exit_confirmation(
    state: dict,
    position_side: str,
    current_position: str
) -> int:
    """
    Track consecutive candles on opposite side for exit confirmation.

    Args:
        state: Strategy state dict
        position_side: Current position ('long' or 'short')
        current_position: Current candle position ('above', 'below', 'neutral')

    Returns:
        Number of consecutive candles on opposite side
    """
    # Initialize tracking if not present
    if 'exit_confirmation_count' not in state:
        state['exit_confirmation_count'] = 0
        state['exit_confirmation_side'] = None

    # Determine if current candle is on opposite side
    opposite_side = False
    if position_side == 'long' and current_position == 'below':
        opposite_side = True
        expected_side = 'below'
    elif position_side == 'short' and current_position == 'above':
        opposite_side = True
        expected_side = 'above'
    else:
        # Not on opposite side - reset counter
        state['exit_confirmation_count'] = 0
        state['exit_confirmation_side'] = None
        return 0

    # Check if continuing same opposite side
    if state['exit_confirmation_side'] == expected_side:
        state['exit_confirmation_count'] += 1
    else:
        # New opposite side detection
        state['exit_confirmation_count'] = 1
        state['exit_confirmation_side'] = expected_side

    return state['exit_confirmation_count']


def check_stop_loss(
    position_side: str,
    entry_price: float,
    current_price: float,
    stop_loss_pct: float,
    use_atr_stops: bool = False,
    atr: Optional[float] = None,
    atr_stop_mult: float = 2.0
) -> Optional[str]:
    """
    Check if stop loss has been hit (PROTECTION ONLY).

    This is a safety mechanism for catastrophic price moves that occur
    BEFORE the EMA flip exit can trigger. The stop should be wide enough
    to not interfere with normal flip-based exits.

    Args:
        position_side: 'long' or 'short'
        entry_price: Entry price
        current_price: Current market price
        stop_loss_pct: Stop loss percentage (should be wide, e.g., 2.5%)
        use_atr_stops: Use ATR-based stops (recommended)
        atr: Average True Range value
        atr_stop_mult: ATR multiplier for stop loss (default 2.0)

    Returns:
        Exit reason string if stop hit, None otherwise
    """
    if entry_price <= 0:
        return None

    # Calculate stop loss price
    if use_atr_stops and atr and atr > 0:
        # ATR-based stop (dynamic, adjusts to volatility)
        if position_side == 'long':
            stop_price = entry_price - (atr * atr_stop_mult)
        else:  # short
            stop_price = entry_price + (atr * atr_stop_mult)
        stop_desc = f"{atr_stop_mult}x ATR"
    else:
        # Percentage-based stop
        if position_side == 'long':
            stop_price = entry_price * (1 - stop_loss_pct / 100)
        else:  # short
            stop_price = entry_price * (1 + stop_loss_pct / 100)
        stop_desc = f"{stop_loss_pct}%"

    # Check stop loss
    if position_side == 'long':
        if current_price <= stop_price:
            pnl_pct = (current_price - entry_price) / entry_price * 100
            return f"Stop loss hit ({stop_desc}), P&L={pnl_pct:.2f}%"
    else:  # short
        if current_price >= stop_price:
            pnl_pct = (entry_price - current_price) / entry_price * 100
            return f"Stop loss hit ({stop_desc}), P&L={pnl_pct:.2f}%"

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

    Priority order (IMPORTANT - this IS the strategy):
    1. EMA flip exit (PRIMARY - the core strategy, trend reversal) with confirmation
    2. Stop loss (PROTECTION ONLY - for catastrophic moves)

    The EMA flip IS the profit exit. There is no take_profit_pct.
    Hold until flip occurs - no time limit.

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
    exit_type = None

    # ==========================================================================
    # 1. PRIMARY EXIT: EMA Flip (The Core Strategy) with Confirmation
    # ==========================================================================
    # This is the main exit mechanism. When the candle closes on the opposite
    # side of the EMA, the trend has flipped and we exit.
    # NEW: Require confirmation candles before exiting to avoid whipsaw exits.

    # Get exit confirmation settings
    exit_confirmation_candles = config.get('exit_confirmation_candles', 1)

    # Track consecutive candles on opposite side
    exit_confirmation_count = track_exit_confirmation(
        state, position_side, current_position
    )

    exit_reason = check_ema_flip_exit(
        position_side,
        current_position,
        current_ema,
        exit_confirmation_count=exit_confirmation_count - 1 if exit_confirmation_count > 0 else 0,
        required_confirmations=exit_confirmation_candles
    )
    if exit_reason:
        exit_type = 'ema_flip'
        # Reset confirmation counter on exit
        state['exit_confirmation_count'] = 0
        state['exit_confirmation_side'] = None

    # ==========================================================================
    # 2. PROTECTION: Stop Loss (Catastrophic Protection Only)
    # ==========================================================================
    # Only checked if no flip exit. This is for violent price moves that
    # happen BEFORE the flip can occur. Should be wide (2.5% or 2x ATR).
    if not exit_reason:
        stop_loss_pct = config.get('stop_loss_pct', 2.5)
        use_atr_stops = config.get('use_atr_stops', True)
        atr_stop_mult = config.get('atr_stop_mult', 2.0)

        exit_reason = check_stop_loss(
            position_side=position_side,
            entry_price=entry_price,
            current_price=current_price,
            stop_loss_pct=stop_loss_pct,
            use_atr_stops=use_atr_stops,
            atr=atr,
            atr_stop_mult=atr_stop_mult
        )
        if exit_reason:
            exit_type = 'stop_loss'

    # ==========================================================================
    # Generate Exit Signal
    # ==========================================================================
    if exit_reason:
        if position_side == 'long':
            pnl_pct = (current_price - entry_price) / entry_price * 100
            return Signal(
                action='sell',
                symbol=symbol,
                size=position_value,
                price=current_price,
                reason=f"EMA9: Exit long - {exit_reason}",
                metadata={
                    'exit_type': exit_type,
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
                reason=f"EMA9: Exit short - {exit_reason}",
                metadata={
                    'exit_type': exit_type,
                    'entry_price': entry_price,
                    'pnl_pct': pnl_pct,
                }
            )

    return None
