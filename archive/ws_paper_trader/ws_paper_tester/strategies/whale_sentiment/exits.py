"""
Whale Sentiment Strategy - Exit Signal Logic

Contains exit checks for:
- Sentiment reversal exit (primary - sentiment shifts opposite)
- Stop loss
- Take profit
- Trailing stop (optional)
"""
from typing import Dict, Any, Optional

from ws_tester.types import Signal

from .config import get_symbol_config, SentimentZone
from .risk import calculate_position_pnl, update_position_extremes
from .indicators import is_fear_zone, is_greed_zone


def check_sentiment_reversal_exit(
    state: Dict[str, Any],
    symbol: str,
    current_price: float,
    sentiment_zone: SentimentZone,
    config: Dict[str, Any]
) -> Optional[Signal]:
    """
    Check if sentiment reversal should trigger exit.

    Exit Logic (Contrarian Mode):
    - Long position (entered in fear) + now in greed = exit
    - Short position (entered in greed) + now in fear = exit

    Args:
        state: Strategy state dict
        symbol: Symbol to check
        current_price: Current market price
        sentiment_zone: Current sentiment zone
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

    pnl_pct, _ = calculate_position_pnl(state, symbol, current_price)

    # Exit long when sentiment shifts to greed (opposite of entry condition)
    if pos_entry['side'] == 'long' and is_greed_zone(sentiment_zone):
        zone_str = sentiment_zone.name.lower()
        return Signal(
            action='sell',
            symbol=symbol,
            size=position_size,
            price=current_price,
            reason=f"WS: Sentiment reversal exit - {zone_str} (pnl={pnl_pct:.2f}%)",
            metadata={
                'exit_type': 'sentiment_reversal',
                'sentiment': zone_str,
            },
        )

    # Exit short when sentiment shifts to fear (opposite of entry condition)
    if pos_entry['side'] == 'short' and is_fear_zone(sentiment_zone):
        zone_str = sentiment_zone.name.lower()
        return Signal(
            action='cover',
            symbol=symbol,
            size=position_size,
            price=current_price,
            reason=f"WS: Sentiment reversal exit - {zone_str} (pnl={pnl_pct:.2f}%)",
            metadata={
                'exit_type': 'sentiment_reversal',
                'sentiment': zone_str,
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

    Wider stops for contrarian strategy (counter-trend entries).

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
    if sl_pct is None:
        sl_pct = config.get('stop_loss_pct', 2.5)

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
                reason=f"WS: Stop loss (entry={entry_price:.6f}, pnl={pnl_pct:.2f}%)",
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
                reason=f"WS: Stop loss (entry={entry_price:.6f}, pnl={pnl_pct:.2f}%)",
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
    if tp_pct is None:
        tp_pct = config.get('take_profit_pct', 5.0)

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
                reason=f"WS: Take profit (entry={entry_price:.6f}, pnl={pnl_pct:.2f}%)",
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
                reason=f"WS: Take profit (entry={entry_price:.6f}, pnl={pnl_pct:.2f}%)",
                metadata={'exit_type': 'take_profit'},
            )

    return None


def check_trailing_stop_exit(
    state: Dict[str, Any],
    symbol: str,
    current_price: float,
    config: Dict[str, Any]
) -> Optional[Signal]:
    """
    Check if trailing stop should trigger exit.

    Activates after position reaches activation_pct of take profit.
    Trails at distance_pct from highest (long) or lowest (short) price.

    Args:
        state: Strategy state dict
        symbol: Symbol to check
        current_price: Current market price
        config: Strategy configuration

    Returns:
        Exit Signal or None
    """
    if not config.get('use_trailing_stop', False):
        return None

    pos_entry = state.get('position_entries', {}).get(symbol)
    if not pos_entry:
        return None

    entry_price = pos_entry.get('entry_price', 0)
    if entry_price <= 0:
        return None

    position_size = state.get('position_by_symbol', {}).get(symbol, 0)
    if position_size <= 0:
        return None

    # Get trailing stop parameters
    tp_pct = get_symbol_config(symbol, config, 'take_profit_pct') or config.get('take_profit_pct', 5.0)
    activation_pct = config.get('trailing_stop_activation_pct', 50.0) / 100  # Convert to decimal
    trail_distance = config.get('trailing_stop_distance_pct', 1.0)

    # Update position extremes
    update_position_extremes(state, symbol, current_price)

    if pos_entry['side'] == 'long':
        pnl_pct = (current_price - entry_price) / entry_price * 100
        activation_threshold = tp_pct * activation_pct

        # Check if activation threshold reached
        if pnl_pct >= activation_threshold:
            highest_price = pos_entry.get('highest_price', entry_price)
            # Trailing stop triggers if price drops trail_distance% from high
            trail_trigger = highest_price * (1 - trail_distance / 100)

            if current_price <= trail_trigger:
                return Signal(
                    action='sell',
                    symbol=symbol,
                    size=position_size,
                    price=current_price,
                    reason=f"WS: Trailing stop (high={highest_price:.6f}, pnl={pnl_pct:.2f}%)",
                    metadata={
                        'exit_type': 'trailing_stop',
                        'highest_price': highest_price,
                    },
                )

    else:  # short
        pnl_pct = (entry_price - current_price) / entry_price * 100
        activation_threshold = tp_pct * activation_pct

        # Check if activation threshold reached
        if pnl_pct >= activation_threshold:
            lowest_price = pos_entry.get('lowest_price', entry_price)
            # Trailing stop triggers if price rises trail_distance% from low
            trail_trigger = lowest_price * (1 + trail_distance / 100)

            if current_price >= trail_trigger:
                return Signal(
                    action='cover',
                    symbol=symbol,
                    size=position_size,
                    price=current_price,
                    reason=f"WS: Trailing stop (low={lowest_price:.6f}, pnl={pnl_pct:.2f}%)",
                    metadata={
                        'exit_type': 'trailing_stop',
                        'lowest_price': lowest_price,
                    },
                )

    return None


def check_all_exits(
    state: Dict[str, Any],
    symbol: str,
    current_price: float,
    sentiment_zone: SentimentZone,
    config: Dict[str, Any]
) -> Optional[Signal]:
    """
    Check all exit conditions in priority order.

    Priority order:
    1. Stop loss (highest priority - capital preservation)
    2. Take profit
    3. Trailing stop (if enabled)
    4. Sentiment reversal exit (primary whale sentiment exit)

    Args:
        state: Strategy state dict
        symbol: Symbol to check
        current_price: Current market price
        sentiment_zone: Current sentiment zone
        config: Strategy configuration

    Returns:
        Exit Signal or None
    """
    # Update position extremes for trailing stop
    update_position_extremes(state, symbol, current_price)

    # 1. Stop loss - highest priority
    signal = check_stop_loss_exit(state, symbol, current_price, config)
    if signal:
        return signal

    # 2. Take profit
    signal = check_take_profit_exit(state, symbol, current_price, config)
    if signal:
        return signal

    # 3. Trailing stop (if enabled)
    signal = check_trailing_stop_exit(state, symbol, current_price, config)
    if signal:
        return signal

    # 4. Sentiment reversal exit
    signal = check_sentiment_reversal_exit(
        state, symbol, current_price, sentiment_zone, config
    )
    if signal:
        return signal

    return None
