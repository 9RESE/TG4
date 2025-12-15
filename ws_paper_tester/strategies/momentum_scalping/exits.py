"""
Momentum Scalping Strategy - Exit Signal Logic

Contains exit checks for:
- Take profit
- Stop loss
- Time-based exits (max hold)
- Momentum exhaustion exits (RSI extreme)
"""
from datetime import datetime
from typing import Dict, Any, Optional

from ws_tester.types import Signal, OrderbookSnapshot

from .config import get_symbol_config
from .risk import calculate_position_age, calculate_position_pnl


def check_take_profit_exit(
    state: Dict[str, Any],
    symbol: str,
    current_price: float,
    config: Dict[str, Any]
) -> Optional[Signal]:
    """
    Check if take profit should trigger exit.

    From research:
    - Fixed Target: 0.5% - 1.0% profit is primary exit method

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
                reason=f"MS: Take profit (entry={entry_price:.6f}, pnl={pnl_pct:.2f}%)",
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
                reason=f"MS: Take profit (entry={entry_price:.6f}, pnl={pnl_pct:.2f}%)",
                metadata={'exit_type': 'take_profit'},
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

    From research:
    - Fixed Stop: 0.3% - 0.5% loss for primary protection

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
                reason=f"MS: Stop loss (entry={entry_price:.6f}, pnl={pnl_pct:.2f}%)",
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
                reason=f"MS: Stop loss (entry={entry_price:.6f}, pnl={pnl_pct:.2f}%)",
                metadata={'exit_type': 'stop_loss'},
            )

    return None


def check_time_based_exit(
    state: Dict[str, Any],
    symbol: str,
    current_price: float,
    current_time: datetime,
    config: Dict[str, Any]
) -> Optional[Signal]:
    """
    Check if time-based exit should trigger.

    From research:
    - Time Stop: Position age > 3-5 minutes to prevent stagnation
    - Momentum scalping targets quick moves, stagnant positions tie up capital

    Args:
        state: Strategy state dict
        symbol: Symbol to check
        current_price: Current market price
        current_time: Current timestamp
        config: Strategy configuration

    Returns:
        Exit Signal or None
    """
    pos_entry = state.get('position_entries', {}).get(symbol)
    if not pos_entry:
        return None

    position_size = state.get('position_by_symbol', {}).get(symbol, 0)
    if position_size <= 0:
        return None

    max_hold_seconds = config.get('max_hold_seconds', 180)
    position_age = calculate_position_age(state, symbol, current_time)

    if position_age <= max_hold_seconds:
        return None

    entry_price = pos_entry.get('entry_price', 0)
    pnl_pct, _ = calculate_position_pnl(state, symbol, current_price)

    if pos_entry['side'] == 'long':
        return Signal(
            action='sell',
            symbol=symbol,
            size=position_size,
            price=current_price,
            reason=f"MS: Time exit (age={position_age:.0f}s, pnl={pnl_pct:.2f}%)",
            metadata={'exit_type': 'time_exit', 'position_age': position_age},
        )
    else:  # short
        return Signal(
            action='cover',
            symbol=symbol,
            size=position_size,
            price=current_price,
            reason=f"MS: Time exit (age={position_age:.0f}s, pnl={pnl_pct:.2f}%)",
            metadata={'exit_type': 'time_exit', 'position_age': position_age},
        )


def check_momentum_exhaustion_exit(
    state: Dict[str, Any],
    symbol: str,
    current_price: float,
    rsi: Optional[float],
    config: Dict[str, Any]
) -> Optional[Signal]:
    """
    Check if momentum exhaustion should trigger exit.

    From research:
    - RSI reverses sharply from entry direction
    - Long position with RSI > 70: momentum exhausted
    - Short position with RSI < 30: momentum exhausted

    Args:
        state: Strategy state dict
        symbol: Symbol to check
        current_price: Current market price
        rsi: Current RSI value
        config: Strategy configuration

    Returns:
        Exit Signal or None
    """
    if rsi is None:
        return None

    pos_entry = state.get('position_entries', {}).get(symbol)
    if not pos_entry:
        return None

    position_size = state.get('position_by_symbol', {}).get(symbol, 0)
    if position_size <= 0:
        return None

    overbought = config.get('rsi_overbought', 70)
    oversold = config.get('rsi_oversold', 30)

    entry_price = pos_entry.get('entry_price', 0)
    pnl_pct, _ = calculate_position_pnl(state, symbol, current_price)

    # Only exit on momentum exhaustion if position is in profit
    # Don't exit at a loss just because RSI is extreme
    if pnl_pct <= 0:
        return None

    if pos_entry['side'] == 'long' and rsi > overbought:
        return Signal(
            action='sell',
            symbol=symbol,
            size=position_size,
            price=current_price,
            reason=f"MS: Momentum exhaustion (RSI={rsi:.1f}, pnl={pnl_pct:.2f}%)",
            metadata={'exit_type': 'momentum_exhaustion', 'rsi': rsi},
        )

    elif pos_entry['side'] == 'short' and rsi < oversold:
        return Signal(
            action='cover',
            symbol=symbol,
            size=position_size,
            price=current_price,
            reason=f"MS: Momentum exhaustion (RSI={rsi:.1f}, pnl={pnl_pct:.2f}%)",
            metadata={'exit_type': 'momentum_exhaustion', 'rsi': rsi},
        )

    return None


def check_ema_cross_exit(
    state: Dict[str, Any],
    symbol: str,
    current_price: float,
    ema_fast: Optional[float],
    config: Dict[str, Any]
) -> Optional[Signal]:
    """
    Check if EMA cross should trigger exit.

    From research:
    - EMA Cross: Price crosses against 8 EMA as trailing protection

    Args:
        state: Strategy state dict
        symbol: Symbol to check
        current_price: Current market price
        ema_fast: Fast EMA value (8-period)
        config: Strategy configuration

    Returns:
        Exit Signal or None
    """
    if ema_fast is None:
        return None

    pos_entry = state.get('position_entries', {}).get(symbol)
    if not pos_entry:
        return None

    position_size = state.get('position_by_symbol', {}).get(symbol, 0)
    if position_size <= 0:
        return None

    pnl_pct, _ = calculate_position_pnl(state, symbol, current_price)

    # Only use EMA cross exit if in profit - protects gains
    if pnl_pct <= 0:
        return None

    if pos_entry['side'] == 'long' and current_price < ema_fast:
        return Signal(
            action='sell',
            symbol=symbol,
            size=position_size,
            price=current_price,
            reason=f"MS: EMA cross exit (price < EMA8, pnl={pnl_pct:.2f}%)",
            metadata={'exit_type': 'ema_cross', 'ema_fast': ema_fast},
        )

    elif pos_entry['side'] == 'short' and current_price > ema_fast:
        return Signal(
            action='cover',
            symbol=symbol,
            size=position_size,
            price=current_price,
            reason=f"MS: EMA cross exit (price > EMA8, pnl={pnl_pct:.2f}%)",
            metadata={'exit_type': 'ema_cross', 'ema_fast': ema_fast},
        )

    return None


def check_all_exits(
    state: Dict[str, Any],
    symbol: str,
    current_price: float,
    current_time: datetime,
    rsi: Optional[float],
    ema_fast: Optional[float],
    config: Dict[str, Any]
) -> Optional[Signal]:
    """
    Check all exit conditions in priority order.

    Priority order:
    1. Stop loss (highest priority - capital preservation)
    2. Take profit
    3. Momentum exhaustion (RSI extreme with profit)
    4. Time-based exit (stagnant position)
    5. EMA cross exit (trend reversal with profit)

    Args:
        state: Strategy state dict
        symbol: Symbol to check
        current_price: Current market price
        current_time: Current timestamp
        rsi: Current RSI value
        ema_fast: Fast EMA value
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

    # 3. Momentum exhaustion
    signal = check_momentum_exhaustion_exit(state, symbol, current_price, rsi, config)
    if signal:
        return signal

    # 4. Time-based exit
    signal = check_time_based_exit(state, symbol, current_price, current_time, config)
    if signal:
        return signal

    # 5. EMA cross exit (optional - can be disabled by not passing ema_fast)
    if ema_fast is not None:
        signal = check_ema_cross_exit(state, symbol, current_price, ema_fast, config)
        if signal:
            return signal

    return None
