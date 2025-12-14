"""
Order Flow Strategy - Exit Signal Checks

Contains trailing stop exit and position decay exit logic.
"""
from datetime import datetime
from typing import Dict, Any, Optional

from ws_tester.types import DataSnapshot, Signal, OrderbookSnapshot

from .config import get_symbol_config
from .risk import calculate_trailing_stop, get_progressive_decay_multiplier


def check_trailing_stop_exit(
    data: DataSnapshot,
    config: Dict[str, Any],
    state: Dict[str, Any],
    symbol: str,
    current_price: float,
    ob: Optional[OrderbookSnapshot]
) -> Optional[Signal]:
    """Check if trailing stop should trigger exit."""
    if not config.get('use_trailing_stop', False):
        return None

    pos_entry = state.get('position_entries', {}).get(symbol)
    if not pos_entry:
        return None

    trailing_activation = config.get('trailing_stop_activation', 0.3)
    trailing_distance = config.get('trailing_stop_distance', 0.2)

    if pos_entry['side'] == 'long':
        pos_entry['highest_price'] = max(pos_entry.get('highest_price', current_price), current_price)
        tracking_price = pos_entry['highest_price']
    else:
        pos_entry['lowest_price'] = min(pos_entry.get('lowest_price', current_price), current_price)
        tracking_price = pos_entry['lowest_price']

    trailing_stop_price = calculate_trailing_stop(
        entry_price=pos_entry['entry_price'],
        highest_price=tracking_price,
        side=pos_entry['side'],
        activation_pct=trailing_activation,
        trail_distance_pct=trailing_distance
    )

    if trailing_stop_price is None:
        return None

    exit_price = current_price
    if ob:
        exit_price = ob.best_bid if pos_entry['side'] == 'long' else ob.best_ask

    if pos_entry['side'] == 'long' and current_price <= trailing_stop_price:
        # REC-003 (v4.2.0): Use per-symbol position size for multi-symbol accuracy
        close_size = state.get('position_by_symbol', {}).get(symbol, 0)
        if close_size > 0:
            return Signal(
                action='sell',
                symbol=symbol,
                size=close_size,
                price=exit_price,
                reason=f"OF: Trailing stop (entry={pos_entry['entry_price']:.6f}, high={pos_entry['highest_price']:.6f})",
                metadata={'trailing_stop': True},
            )

    elif pos_entry['side'] == 'short' and current_price >= trailing_stop_price:
        # REC-003 (v4.2.0): Use per-symbol position size for multi-symbol accuracy
        close_size = state.get('position_by_symbol', {}).get(symbol, 0)
        if close_size > 0:
            return Signal(
                action='cover',
                symbol=symbol,
                size=close_size,
                price=exit_price,
                reason=f"OF: Trailing stop (entry={pos_entry['entry_price']:.6f}, low={pos_entry['lowest_price']:.6f})",
                metadata={'trailing_stop': True},
            )

    return None


def check_position_decay_exit(
    data: DataSnapshot,
    config: Dict[str, Any],
    state: Dict[str, Any],
    symbol: str,
    current_price: float,
    ob: Optional[OrderbookSnapshot],
    current_time: datetime
) -> Optional[Signal]:
    """
    REC-004: Check if stale position should be closed with progressive TP.

    Enhanced to allow closing at any profit > fees during intermediate decay stages.
    REC-003 (v4.2.0): Uses per-symbol position size for accurate multi-symbol behavior.
    """
    if not config.get('use_position_decay', True):
        return None

    pos_entry = state.get('position_entries', {}).get(symbol)
    if not pos_entry:
        return None

    entry_time = pos_entry.get('entry_time')
    if not entry_time:
        return None

    age_seconds = (current_time - entry_time).total_seconds()

    decay_stages = config.get('position_decay_stages', [
        (180, 0.90), (240, 0.75), (300, 0.50), (360, 0.0)
    ])
    tp_mult = get_progressive_decay_multiplier(age_seconds, decay_stages)

    # If multiplier is 1.0, no decay yet
    if tp_mult >= 1.0:
        return None

    entry_price = pos_entry['entry_price']
    tp_pct = get_symbol_config(symbol, config, 'take_profit_pct')
    adjusted_tp_pct = tp_pct * tp_mult

    exit_price = current_price
    if ob:
        exit_price = ob.best_bid if pos_entry['side'] == 'long' else ob.best_ask

    # REC-004: Enhanced close-at-profit-after-fees for intermediate stages
    use_early_close = config.get('decay_close_at_profit_after_fees', True)
    fee_rate = config.get('fee_rate', 0.001)
    min_profit_after_fees = config.get('decay_min_profit_after_fees_pct', 0.05)
    round_trip_fee_pct = fee_rate * 2 * 100

    if pos_entry['side'] == 'long':
        profit_pct = (current_price - entry_price) / entry_price * 100
        net_profit_pct = profit_pct - round_trip_fee_pct

        # For tp_mult=0, exit at any profit
        if tp_mult == 0 and profit_pct > 0:
            # REC-003 (v4.2.0): Use per-symbol position size
            close_size = state.get('position_by_symbol', {}).get(symbol, 0)
            if close_size > 0:
                return Signal(
                    action='sell',
                    symbol=symbol,
                    size=close_size,
                    price=exit_price,
                    reason=f"OF: Decay exit (age={age_seconds:.0f}s, profit={profit_pct:.2f}%)",
                    metadata={'position_decay': True, 'decay_stage': 'any_profit'},
                )

        # REC-004: Close at profit after fees during intermediate stages
        elif use_early_close and tp_mult < 1.0 and net_profit_pct >= min_profit_after_fees:
            # REC-003 (v4.2.0): Use per-symbol position size
            close_size = state.get('position_by_symbol', {}).get(symbol, 0)
            if close_size > 0:
                return Signal(
                    action='sell',
                    symbol=symbol,
                    size=close_size,
                    price=exit_price,
                    reason=f"OF: Decay early exit (age={age_seconds:.0f}s, net_profit={net_profit_pct:.2f}%)",
                    metadata={'position_decay': True, 'decay_stage': 'profit_after_fees', 'decay_mult': tp_mult},
                )

        # Standard decay exit at adjusted TP
        elif profit_pct >= adjusted_tp_pct > 0:
            # REC-003 (v4.2.0): Use per-symbol position size
            close_size = state.get('position_by_symbol', {}).get(symbol, 0)
            if close_size > 0:
                return Signal(
                    action='sell',
                    symbol=symbol,
                    size=close_size,
                    price=exit_price,
                    reason=f"OF: Decay exit (age={age_seconds:.0f}s, profit={profit_pct:.2f}%, target={adjusted_tp_pct:.2f}%)",
                    metadata={'position_decay': True, 'decay_mult': tp_mult},
                )

    elif pos_entry['side'] == 'short':
        profit_pct = (entry_price - current_price) / entry_price * 100
        net_profit_pct = profit_pct - round_trip_fee_pct

        if tp_mult == 0 and profit_pct > 0:
            # REC-003 (v4.2.0): Use per-symbol position size
            close_size = state.get('position_by_symbol', {}).get(symbol, 0)
            if close_size > 0:
                return Signal(
                    action='cover',
                    symbol=symbol,
                    size=close_size,
                    price=exit_price,
                    reason=f"OF: Decay exit (age={age_seconds:.0f}s, profit={profit_pct:.2f}%)",
                    metadata={'position_decay': True, 'decay_stage': 'any_profit'},
                )

        elif use_early_close and tp_mult < 1.0 and net_profit_pct >= min_profit_after_fees:
            # REC-003 (v4.2.0): Use per-symbol position size
            close_size = state.get('position_by_symbol', {}).get(symbol, 0)
            if close_size > 0:
                return Signal(
                    action='cover',
                    symbol=symbol,
                    size=close_size,
                    price=exit_price,
                    reason=f"OF: Decay early exit (age={age_seconds:.0f}s, net_profit={net_profit_pct:.2f}%)",
                    metadata={'position_decay': True, 'decay_stage': 'profit_after_fees', 'decay_mult': tp_mult},
                )

        elif profit_pct >= adjusted_tp_pct > 0:
            # REC-003 (v4.2.0): Use per-symbol position size
            close_size = state.get('position_by_symbol', {}).get(symbol, 0)
            if close_size > 0:
                return Signal(
                    action='cover',
                    symbol=symbol,
                    size=close_size,
                    price=exit_price,
                    reason=f"OF: Decay exit (age={age_seconds:.0f}s, profit={profit_pct:.2f}%, target={adjusted_tp_pct:.2f}%)",
                    metadata={'position_decay': True, 'decay_mult': tp_mult},
                )

    return None
