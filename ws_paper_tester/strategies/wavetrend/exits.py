"""
WaveTrend Oscillator Strategy - Exit Signal Logic

Contains exit checks for:
- WaveTrend crossover reversal (primary exit)
- Extreme zone profit taking
- Stop loss
- Take profit
"""
from typing import Dict, Any, Optional

from ws_tester.types import Signal

from .config import get_symbol_config, WaveTrendZone, CrossoverType
from .risk import calculate_position_pnl
from .indicators import is_extreme_zone


def check_crossover_exit(
    state: Dict[str, Any],
    symbol: str,
    current_price: float,
    crossover: CrossoverType,
    current_zone: WaveTrendZone,
    wt1: float
) -> Optional[Signal]:
    """
    Check if WaveTrend crossover reversal should trigger exit.

    Exit Logic:
    - Long position + bearish crossover = exit
    - Short position + bullish crossover = exit

    Args:
        state: Strategy state dict
        symbol: Symbol to check
        current_price: Current market price
        crossover: Current crossover type
        current_zone: Current WaveTrend zone
        wt1: Current WT1 value

    Returns:
        Exit Signal or None
    """
    pos_entry = state.get('position_entries', {}).get(symbol)
    if not pos_entry:
        return None

    position_size = state.get('position_by_symbol', {}).get(symbol, 0)
    if position_size <= 0:
        return None

    entry_price = pos_entry.get('entry_price', 0)
    pnl_pct, _ = calculate_position_pnl(state, symbol, current_price)

    # Exit long on bearish crossover
    if pos_entry['side'] == 'long' and crossover == CrossoverType.BEARISH:
        zone_str = current_zone.name.lower()
        return Signal(
            action='sell',
            symbol=symbol,
            size=position_size,
            price=current_price,
            reason=f"WT: Bearish crossover exit (WT1={wt1:.0f}, zone={zone_str}, pnl={pnl_pct:.2f}%)",
            metadata={
                'exit_type': 'crossover_reversal',
                'wt1': wt1,
                'zone': zone_str,
            },
        )

    # Exit short on bullish crossover
    if pos_entry['side'] == 'short' and crossover == CrossoverType.BULLISH:
        zone_str = current_zone.name.lower()
        return Signal(
            action='cover',
            symbol=symbol,
            size=position_size,
            price=current_price,
            reason=f"WT: Bullish crossover exit (WT1={wt1:.0f}, zone={zone_str}, pnl={pnl_pct:.2f}%)",
            metadata={
                'exit_type': 'crossover_reversal',
                'wt1': wt1,
                'zone': zone_str,
            },
        )

    return None


def check_extreme_zone_exit(
    state: Dict[str, Any],
    symbol: str,
    current_price: float,
    current_zone: WaveTrendZone,
    wt1: float
) -> Optional[Signal]:
    """
    Check if extreme zone profit taking should trigger exit.

    Exit Logic:
    - Long position + extreme overbought = take profit
    - Short position + extreme oversold = take profit

    Args:
        state: Strategy state dict
        symbol: Symbol to check
        current_price: Current market price
        current_zone: Current WaveTrend zone
        wt1: Current WT1 value

    Returns:
        Exit Signal or None
    """
    pos_entry = state.get('position_entries', {}).get(symbol)
    if not pos_entry:
        return None

    position_size = state.get('position_by_symbol', {}).get(symbol, 0)
    if position_size <= 0:
        return None

    pnl_pct, _ = calculate_position_pnl(state, symbol, current_price)

    # Exit long at extreme overbought
    if pos_entry['side'] == 'long' and current_zone == WaveTrendZone.EXTREME_OVERBOUGHT:
        return Signal(
            action='sell',
            symbol=symbol,
            size=position_size,
            price=current_price,
            reason=f"WT: Extreme overbought exit (WT1={wt1:.0f}, pnl={pnl_pct:.2f}%)",
            metadata={
                'exit_type': 'extreme_zone_profit_take',
                'wt1': wt1,
                'zone': 'extreme_overbought',
            },
        )

    # Exit short at extreme oversold
    if pos_entry['side'] == 'short' and current_zone == WaveTrendZone.EXTREME_OVERSOLD:
        return Signal(
            action='cover',
            symbol=symbol,
            size=position_size,
            price=current_price,
            reason=f"WT: Extreme oversold exit (WT1={wt1:.0f}, pnl={pnl_pct:.2f}%)",
            metadata={
                'exit_type': 'extreme_zone_profit_take',
                'wt1': wt1,
                'zone': 'extreme_oversold',
            },
        )

    return None


def check_stop_loss_exit(
    state: Dict[str, Any],
    symbol: str,
    current_price: float,
    config: Dict[str, Any]
) -> Optional[Signal]:
    """
    Check if stop loss should trigger exit.

    Args:
        state: Strategy state dict
        symbol: Symbol to check
        current_price: Current market price
        config: Strategy configuration

    Returns:
        Exit Signal or None
    """
    pos_entry = state.get('position_entries', {}).get(symbol)
    if not pos_entry:
        return None

    entry_price = pos_entry.get('entry_price', 0)
    if entry_price <= 0:
        return None

    sl_pct = get_symbol_config(symbol, config, 'stop_loss_pct')
    position_size = state.get('position_by_symbol', {}).get(symbol, 0)

    if position_size <= 0:
        return None

    # Calculate current P&L
    if pos_entry['side'] == 'long':
        pnl_pct = (current_price - entry_price) / entry_price * 100
        if pnl_pct <= -sl_pct:
            return Signal(
                action='sell',
                symbol=symbol,
                size=position_size,
                price=current_price,
                reason=f"WT: Stop loss (entry={entry_price:.6f}, pnl={pnl_pct:.2f}%)",
                metadata={'exit_type': 'stop_loss'},
            )
    else:  # short
        pnl_pct = (entry_price - current_price) / entry_price * 100
        if pnl_pct <= -sl_pct:
            return Signal(
                action='cover',
                symbol=symbol,
                size=position_size,
                price=current_price,
                reason=f"WT: Stop loss (entry={entry_price:.6f}, pnl={pnl_pct:.2f}%)",
                metadata={'exit_type': 'stop_loss'},
            )

    return None


def check_take_profit_exit(
    state: Dict[str, Any],
    symbol: str,
    current_price: float,
    config: Dict[str, Any]
) -> Optional[Signal]:
    """
    Check if take profit should trigger exit.

    Args:
        state: Strategy state dict
        symbol: Symbol to check
        current_price: Current market price
        config: Strategy configuration

    Returns:
        Exit Signal or None
    """
    pos_entry = state.get('position_entries', {}).get(symbol)
    if not pos_entry:
        return None

    entry_price = pos_entry.get('entry_price', 0)
    if entry_price <= 0:
        return None

    tp_pct = get_symbol_config(symbol, config, 'take_profit_pct')
    position_size = state.get('position_by_symbol', {}).get(symbol, 0)

    if position_size <= 0:
        return None

    # Calculate current P&L
    if pos_entry['side'] == 'long':
        pnl_pct = (current_price - entry_price) / entry_price * 100
        if pnl_pct >= tp_pct:
            return Signal(
                action='sell',
                symbol=symbol,
                size=position_size,
                price=current_price,
                reason=f"WT: Take profit (entry={entry_price:.6f}, pnl={pnl_pct:.2f}%)",
                metadata={'exit_type': 'take_profit'},
            )
    else:  # short
        pnl_pct = (entry_price - current_price) / entry_price * 100
        if pnl_pct >= tp_pct:
            return Signal(
                action='cover',
                symbol=symbol,
                size=position_size,
                price=current_price,
                reason=f"WT: Take profit (entry={entry_price:.6f}, pnl={pnl_pct:.2f}%)",
                metadata={'exit_type': 'take_profit'},
            )

    return None


def check_all_exits(
    state: Dict[str, Any],
    symbol: str,
    current_price: float,
    crossover: CrossoverType,
    current_zone: WaveTrendZone,
    wt1: float,
    config: Dict[str, Any]
) -> Optional[Signal]:
    """
    Check all exit conditions in priority order.

    Priority order:
    1. Stop loss (highest priority - capital preservation)
    2. Take profit
    3. Crossover reversal (primary WaveTrend exit)
    4. Extreme zone profit taking

    Args:
        state: Strategy state dict
        symbol: Symbol to check
        current_price: Current market price
        crossover: Current crossover type
        current_zone: Current WaveTrend zone
        wt1: Current WT1 value
        config: Strategy configuration

    Returns:
        Exit Signal or None
    """
    # 1. Stop loss - highest priority
    signal = check_stop_loss_exit(state, symbol, current_price, config)
    if signal:
        return signal

    # 2. Take profit
    signal = check_take_profit_exit(state, symbol, current_price, config)
    if signal:
        return signal

    # 3. Crossover reversal exit
    signal = check_crossover_exit(
        state, symbol, current_price, crossover, current_zone, wt1
    )
    if signal:
        return signal

    # 4. Extreme zone profit taking
    signal = check_extreme_zone_exit(
        state, symbol, current_price, current_zone, wt1
    )
    if signal:
        return signal

    return None
