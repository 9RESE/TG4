"""
Grid RSI Reversion Strategy - Exit Signal Logic

Contains exit checks for:
- Grid cycle completion (matched sell)
- Stop loss (below lowest grid)
- Max drawdown
- Stale position timeout
"""
from datetime import datetime
from typing import Dict, Any, Optional

from ws_tester.types import Signal

from .config import get_symbol_config
from .risk import calculate_position_pnl, calculate_position_age


def check_grid_stop_loss(
    state: Dict[str, Any],
    symbol: str,
    current_price: float,
    config: Dict[str, Any]
) -> Optional[Signal]:
    """
    Check if grid-wide stop loss should trigger.

    Stop loss is set below the lowest grid level.

    Args:
        state: Strategy state dict
        symbol: Trading symbol
        current_price: Current market price
        config: Strategy configuration

    Returns:
        Exit Signal or None
    """
    position_size = state.get('position_by_symbol', {}).get(symbol, 0)
    if position_size <= 0:
        return None

    grid_metadata = state.get('grid_metadata', {}).get(symbol, {})
    lower_price = grid_metadata.get('lower_price')
    if not lower_price:
        return None

    # Stop loss below lowest grid level
    stop_loss_pct = config.get('stop_loss_pct', 3.0)
    stop_price = lower_price * (1 - stop_loss_pct / 100)

    if current_price <= stop_price:
        pnl_pct, _ = calculate_position_pnl(state, symbol, current_price)

        return Signal(
            action='sell',
            symbol=symbol,
            size=position_size,
            price=current_price,
            reason=f"Grid SL: price={current_price:.6f} < stop={stop_price:.6f}, pnl={pnl_pct:.2f}%",
            metadata={
                'exit_type': 'grid_stop_loss',
                'size_unit': 'usd',  # position_size is stored in USD
                'stop_price': stop_price,
                'lower_grid': lower_price,
            },
        )

    return None


def check_max_drawdown_exit(
    state: Dict[str, Any],
    symbol: str,
    current_price: float,
    config: Dict[str, Any]
) -> Optional[Signal]:
    """
    Check if max drawdown should trigger exit.

    Args:
        state: Strategy state dict
        symbol: Trading symbol
        current_price: Current market price
        config: Strategy configuration

    Returns:
        Exit Signal or None
    """
    position_size = state.get('position_by_symbol', {}).get(symbol, 0)
    if position_size <= 0:
        return None

    max_drawdown_pct = config.get('max_drawdown_pct', 10.0)

    pnl_pct, _ = calculate_position_pnl(state, symbol, current_price)

    # Drawdown is negative P&L
    if pnl_pct <= -max_drawdown_pct:
        return Signal(
            action='sell',
            symbol=symbol,
            size=position_size,
            price=current_price,
            reason=f"Grid DD: pnl={pnl_pct:.2f}% <= -{max_drawdown_pct}%",
            metadata={
                'exit_type': 'max_drawdown',
                'size_unit': 'usd',  # position_size is stored in USD
                'drawdown_pct': abs(pnl_pct),
            },
        )

    return None


def check_stale_position_exit(
    state: Dict[str, Any],
    symbol: str,
    current_price: float,
    current_time: datetime,
    config: Dict[str, Any]
) -> Optional[Signal]:
    """
    Check if stale position should be closed.

    Grid positions that haven't completed cycles for extended time
    may indicate the grid has broken out of range.

    Args:
        state: Strategy state dict
        symbol: Trading symbol
        current_price: Current market price
        current_time: Current timestamp
        config: Strategy configuration

    Returns:
        Exit Signal or None
    """
    position_size = state.get('position_by_symbol', {}).get(symbol, 0)
    if position_size <= 0:
        return None

    # Get oldest unfilled position time
    grid_metadata = state.get('grid_metadata', {}).get(symbol, {})
    last_recenter = grid_metadata.get('last_recenter_time')

    if not last_recenter:
        return None

    # Check if too long since recenter with no cycle completions
    elapsed = (current_time - last_recenter).total_seconds()
    stale_timeout = config.get('stale_position_timeout', 7200)  # 2 hours default

    cycles_since_recenter = (
        grid_metadata.get('cycles_completed', 0) -
        grid_metadata.get('cycles_at_last_recenter', 0)
    )

    if elapsed > stale_timeout and cycles_since_recenter == 0:
        pnl_pct, _ = calculate_position_pnl(state, symbol, current_price)

        return Signal(
            action='sell',
            symbol=symbol,
            size=position_size,
            price=current_price,
            reason=f"Grid stale: no cycles in {elapsed/3600:.1f}h, pnl={pnl_pct:.2f}%",
            metadata={
                'exit_type': 'stale_position',
                'size_unit': 'usd',  # position_size is stored in USD
                'elapsed_hours': elapsed / 3600,
            },
        )

    return None


def check_grid_cycle_sell(
    state: Dict[str, Any],
    symbol: str,
    current_price: float,
    config: Dict[str, Any]
) -> Optional[Signal]:
    """
    Check if a grid sell level is triggered.

    This is the normal profit-taking mechanism for grid trading:
    when price rises to a sell level, sell the corresponding buy position.

    Args:
        state: Strategy state dict
        symbol: Trading symbol
        current_price: Current market price
        config: Strategy configuration

    Returns:
        Sell Signal or None
    """
    grid_levels = state.get('grid_levels', {}).get(symbol, [])
    if not grid_levels:
        return None

    tolerance_pct = config.get('slippage_tolerance_pct', 0.5)

    # Find unfilled sell levels that have a filled matching buy
    for level in grid_levels:
        if level['side'] != 'sell' or level['filled']:
            continue

        # Check if price is at this sell level
        tolerance = level['price'] * (tolerance_pct / 100)
        if current_price < level['price'] - tolerance:
            continue

        # Check if matching buy is filled
        matched_id = level.get('matched_order_id')
        if not matched_id:
            continue

        # Find the matching buy level
        matching_buy = None
        for buy_level in grid_levels:
            if buy_level['order_id'] == matched_id and buy_level['filled']:
                matching_buy = buy_level
                break

        if matching_buy:
            # Calculate cycle profit
            buy_price = matching_buy.get('fill_price', matching_buy['price'])
            profit_pct = (current_price - buy_price) / buy_price * 100

            return Signal(
                action='sell',
                symbol=symbol,
                size=level['size'],
                price=current_price,
                reason=f"Grid cycle: sell@{level['price']:.6f}, profit={profit_pct:.2f}%",
                stop_loss=None,
                take_profit=None,
                metadata={
                    'exit_type': 'grid_cycle',
                    'grid_order_id': level['order_id'],
                    'size_unit': 'usd',  # level['size'] is position_size_usd from config
                    'matched_buy_id': matched_id,
                    'buy_price': buy_price,
                    'sell_price': current_price,
                    'cycle_profit_pct': profit_pct,
                },
            )

    return None


def check_all_exits(
    state: Dict[str, Any],
    symbol: str,
    current_price: float,
    current_time: datetime,
    config: Dict[str, Any]
) -> Optional[Signal]:
    """
    Check all exit conditions in priority order.

    Priority order:
    1. Stop loss (capital preservation)
    2. Max drawdown
    3. Grid cycle sells (normal profit taking)
    4. Stale position timeout

    Args:
        state: Strategy state dict
        symbol: Trading symbol
        current_price: Current market price
        current_time: Current timestamp
        config: Strategy configuration

    Returns:
        Exit Signal or None
    """
    # 1. Stop loss - highest priority
    signal = check_grid_stop_loss(state, symbol, current_price, config)
    if signal:
        return signal

    # 2. Max drawdown
    signal = check_max_drawdown_exit(state, symbol, current_price, config)
    if signal:
        return signal

    # 3. Grid cycle sells (normal profit taking)
    signal = check_grid_cycle_sell(state, symbol, current_price, config)
    if signal:
        return signal

    # 4. Stale position timeout
    signal = check_stale_position_exit(
        state, symbol, current_price, current_time, config
    )
    if signal:
        return signal

    return None
