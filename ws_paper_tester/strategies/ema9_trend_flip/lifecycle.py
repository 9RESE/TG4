"""
EMA-9 Trend Flip Strategy - Lifecycle

Strategy lifecycle callbacks: on_start, on_fill, on_stop.
"""
from datetime import datetime
from typing import Dict, Any


def initialize_state(state: Dict[str, Any]) -> None:
    """Initialize strategy state."""
    state['initialized'] = True
    state['position'] = 0.0
    state['position_side'] = None  # 'long' or 'short'
    state['entry_price'] = 0.0
    state['entry_time'] = None
    state['trade_count'] = 0
    state['win_count'] = 0
    state['loss_count'] = 0
    state['total_pnl'] = 0.0
    state['last_signal_time'] = None
    state['last_trade_was_loss'] = False
    state['prev_trend'] = None
    state['consecutive_count'] = 0
    state['hourly_candles'] = []
    state['ema_values'] = []
    state['indicators'] = {}
    state['rejection_counts'] = {}


def on_start(config: Dict[str, Any], state: Dict[str, Any]) -> None:
    """
    Initialize strategy state.

    Args:
        config: Strategy configuration
        state: Strategy state dict to initialize
    """
    initialize_state(state)


def on_fill(fill: Dict[str, Any], state: Dict[str, Any]) -> None:
    """
    Update state after trade execution.

    Args:
        fill: Fill information dict with side, value, price, pnl
        state: Strategy state dict to update
    """
    side = fill.get('side', '')
    value = fill.get('value', 0)
    price = fill.get('price', 0)
    pnl = fill.get('pnl', 0)

    if side == 'buy':
        if state.get('position_side') == 'short':
            # Closing short
            state['position'] = 0.0
            state['position_side'] = None
            state['entry_price'] = 0.0
            state['entry_time'] = None
            state['total_pnl'] += pnl
            state['trade_count'] += 1
            if pnl > 0:
                state['win_count'] += 1
                state['last_trade_was_loss'] = False
            else:
                state['loss_count'] += 1
                state['last_trade_was_loss'] = True
        else:
            # Opening long
            state['position'] = value
            state['position_side'] = 'long'
            state['entry_price'] = price
            state['entry_time'] = datetime.now()

    elif side == 'sell':
        if state.get('position_side') == 'long':
            # Closing long
            state['position'] = 0.0
            state['position_side'] = None
            state['entry_price'] = 0.0
            state['entry_time'] = None
            state['total_pnl'] += pnl
            state['trade_count'] += 1
            if pnl > 0:
                state['win_count'] += 1
                state['last_trade_was_loss'] = False
            else:
                state['loss_count'] += 1
                state['last_trade_was_loss'] = True
        else:
            # Opening short
            state['position'] = value
            state['position_side'] = 'short'
            state['entry_price'] = price
            state['entry_time'] = datetime.now()


def on_stop(state: Dict[str, Any]) -> None:
    """
    Cleanup when strategy stops.

    Args:
        state: Strategy state dict
    """
    pass
