"""
Market Making Strategy - Signal Generation

Main signal generation logic and helper functions for building signals.
"""
from datetime import datetime
from typing import Dict, Any, Optional

from ws_tester.types import DataSnapshot, Signal, OrderbookSnapshot

from .config import SYMBOLS, get_symbol_config, is_xrp_btc
from .calculations import (
    calculate_micro_price,
    calculate_optimal_spread,
    calculate_reservation_price,
    calculate_trailing_stop,
    calculate_volatility,
    get_trade_flow_imbalance,
    check_fee_profitability,
    check_position_decay,
    get_xrp_usdt_price,
    calculate_effective_thresholds,
)


def build_entry_signal(
    symbol: str,
    action: str,
    size: float,
    entry_price: float,
    reason: str,
    sl_pct: float,
    tp_pct: float,
    is_cross_pair: bool,
    xrp_usdt_price: float
) -> Signal:
    """
    Build a trading signal with proper stop/TP levels (MM-010 refactor).

    Returns:
        Constructed Signal object
    """
    # Calculate stop and take profit based on action direction
    if action in ('buy', 'cover'):
        stop_loss = entry_price * (1 - sl_pct / 100)
        take_profit = entry_price * (1 + tp_pct / 100)
    else:  # sell, short
        stop_loss = entry_price * (1 + sl_pct / 100)
        take_profit = entry_price * (1 - tp_pct / 100)

    # Calculate XRP size for cross-pair
    metadata = None
    if is_cross_pair:
        xrp_size = size / xrp_usdt_price
        metadata = {'xrp_size': xrp_size}

    return Signal(
        action=action,
        symbol=symbol,
        size=size,
        price=entry_price,
        reason=reason,
        stop_loss=stop_loss,
        take_profit=take_profit,
        metadata=metadata,
    )


def check_trailing_stop_exit(
    data: DataSnapshot,
    config: Dict[str, Any],
    state: Dict[str, Any],
    symbol: str,
    price: float,
    ob: OrderbookSnapshot,
    inventory: float,
    is_cross_pair: bool
) -> Optional[Signal]:
    """
    Check if trailing stop should trigger exit (MM-010 refactor).

    Returns:
        Signal if trailing stop triggered, None otherwise
    """
    use_trailing = config.get('use_trailing_stop', False)
    if not use_trailing:
        return None

    if 'position_entries' not in state:
        return None

    pos_entry = state['position_entries'].get(symbol)
    if not pos_entry:
        return None

    trailing_activation = config.get('trailing_stop_activation', 0.2)
    trailing_distance = config.get('trailing_stop_distance', 0.15)

    # Update highest/lowest price for tracking
    if pos_entry['side'] == 'long':
        pos_entry['highest_price'] = max(pos_entry['highest_price'], price)
        tracking_price = pos_entry['highest_price']
    else:
        pos_entry['lowest_price'] = min(pos_entry['lowest_price'], price)
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

    xrp_usdt_price = get_xrp_usdt_price(data, config)

    # Check if trailing stop is triggered
    if pos_entry['side'] == 'long' and price <= trailing_stop_price:
        close_size = abs(inventory)
        if is_cross_pair:
            close_size = close_size * xrp_usdt_price

        return Signal(
            action='sell',
            symbol=symbol,
            size=close_size,
            price=ob.best_bid,
            reason=f"MM: Trailing stop hit (entry={pos_entry['entry_price']:.6f}, high={pos_entry['highest_price']:.6f}, trail={trailing_stop_price:.6f})",
            metadata={'trailing_stop': True, 'xrp_size': close_size / xrp_usdt_price if is_cross_pair else None},
        )

    elif pos_entry['side'] == 'short' and price >= trailing_stop_price:
        close_size = abs(inventory)
        if is_cross_pair:
            close_size = close_size * xrp_usdt_price

        return Signal(
            action='cover',
            symbol=symbol,
            size=close_size,
            price=ob.best_ask,
            reason=f"MM: Trailing stop hit (entry={pos_entry['entry_price']:.6f}, low={pos_entry['lowest_price']:.6f}, trail={trailing_stop_price:.6f})",
            metadata={'trailing_stop': True, 'xrp_size': close_size / xrp_usdt_price if is_cross_pair else None},
        )

    return None


def check_position_decay_exit(
    data: DataSnapshot,
    config: Dict[str, Any],
    state: Dict[str, Any],
    symbol: str,
    price: float,
    ob: OrderbookSnapshot,
    inventory: float,
    is_cross_pair: bool,
    current_time: datetime
) -> Optional[Signal]:
    """
    Check if stale position should be closed with reduced TP (MM-E04).

    Returns:
        Signal if stale position should exit, None otherwise
    """
    use_decay = config.get('use_position_decay', True)
    if not use_decay:
        return None

    if 'position_entries' not in state:
        return None

    pos_entry = state['position_entries'].get(symbol)
    if not pos_entry:
        return None

    max_age = config.get('max_position_age_seconds', 300)
    tp_mult = config.get('position_decay_tp_multiplier', 0.5)

    is_stale, adjusted_mult = check_position_decay(pos_entry, current_time, max_age, tp_mult)

    if not is_stale:
        return None

    # Check if we're in profit and should exit with reduced TP
    entry_price = pos_entry['entry_price']
    tp_pct = get_symbol_config(symbol, config, 'take_profit_pct')
    adjusted_tp_pct = tp_pct * adjusted_mult

    xrp_usdt_price = get_xrp_usdt_price(data, config)

    if pos_entry['side'] == 'long':
        profit_pct = (price - entry_price) / entry_price * 100
        if profit_pct >= adjusted_tp_pct:
            close_size = abs(inventory)
            if is_cross_pair:
                close_size = close_size * xrp_usdt_price

            return Signal(
                action='sell',
                symbol=symbol,
                size=close_size,
                price=ob.best_bid,
                reason=f"MM: Stale position exit (age>{max_age}s, profit={profit_pct:.2f}%, adj_tp={adjusted_tp_pct:.2f}%)",
                metadata={'position_decay': True, 'xrp_size': close_size / xrp_usdt_price if is_cross_pair else None},
            )

    elif pos_entry['side'] == 'short':
        profit_pct = (entry_price - price) / entry_price * 100
        if profit_pct >= adjusted_tp_pct:
            close_size = abs(inventory)
            if is_cross_pair:
                close_size = close_size * xrp_usdt_price

            return Signal(
                action='cover',
                symbol=symbol,
                size=close_size,
                price=ob.best_ask,
                reason=f"MM: Stale position exit (age>{max_age}s, profit={profit_pct:.2f}%, adj_tp={adjusted_tp_pct:.2f}%)",
                metadata={'position_decay': True, 'xrp_size': close_size / xrp_usdt_price if is_cross_pair else None},
            )

    return None


def _evaluate_symbol(
    data: DataSnapshot,
    config: Dict[str, Any],
    state: Dict[str, Any],
    symbol: str,
    current_time: datetime
) -> Optional[Signal]:
    """Evaluate a single symbol for market making opportunity."""
    # Get orderbook
    ob = data.orderbooks.get(symbol)
    if not ob or not ob.best_bid or not ob.best_ask:
        return None

    price = data.prices.get(symbol, 0)
    if not price:
        return None

    is_cross_pair = is_xrp_btc(symbol)
    xrp_usdt_price = get_xrp_usdt_price(data, config)

    # MM-E01: Calculate micro-price for better price discovery
    use_micro = config.get('use_micro_price', True)
    if use_micro:
        micro_price = calculate_micro_price(ob)
    else:
        micro_price = ob.mid

    # Calculate spread
    spread_pct = ob.spread_pct

    # Get symbol-specific config
    tp_pct = get_symbol_config(symbol, config, 'take_profit_pct')
    sl_pct = get_symbol_config(symbol, config, 'stop_loss_pct')

    # MM-002: Calculate volatility
    candles = data.candles_1m.get(symbol, ())
    volatility = calculate_volatility(candles, config.get('volatility_lookback', 20))

    # Calculate effective thresholds (MM-010 refactor)
    effective_min_spread, effective_threshold, vol_multiplier = calculate_effective_thresholds(
        config, symbol, volatility
    )

    # MM-E02: Calculate optimal spread if enabled
    use_optimal = config.get('use_optimal_spread', False)
    optimal_spread = 0.0
    if use_optimal and volatility > 0:
        gamma = config.get('gamma', 0.1)
        kappa = config.get('kappa', 1.5)
        optimal_spread = calculate_optimal_spread(volatility, gamma, kappa)
        # Use maximum of configured min spread and A-S optimal spread
        effective_min_spread = max(effective_min_spread, optimal_spread)

    # MM-007: Get trade flow imbalance for confirmation
    trade_flow = get_trade_flow_imbalance(data, symbol, 50)
    trade_flow_threshold = config.get('trade_flow_threshold', 0.15)
    use_trade_flow = config.get('use_trade_flow', True)

    # Get inventory (different units for XRP/BTC)
    if 'inventory_by_symbol' not in state:
        state['inventory_by_symbol'] = {}

    if is_cross_pair:
        inventory = state['inventory_by_symbol'].get(symbol, 0)  # in XRP
        max_inventory = get_symbol_config(symbol, config, 'max_inventory_xrp') or 150
        base_size_xrp = get_symbol_config(symbol, config, 'position_size_xrp') or 25
        base_size = base_size_xrp * xrp_usdt_price
    else:
        inventory = state['inventory_by_symbol'].get(symbol, 0)  # in USD
        max_inventory = get_symbol_config(symbol, config, 'max_inventory') or 100
        base_size = get_symbol_config(symbol, config, 'position_size_usd') or 20

    # v1.4.0: Calculate reservation price if enabled
    use_reservation = config.get('use_reservation_price', False)
    gamma = config.get('gamma', 0.1)
    reservation_price = micro_price  # Use micro-price as base

    if use_reservation and volatility > 0:
        reservation_price = calculate_reservation_price(
            mid_price=micro_price,
            inventory=inventory,
            max_inventory=max_inventory,
            gamma=gamma,
            volatility_pct=volatility
        )

    # MM-E03: Check fee profitability
    use_fee_check = config.get('use_fee_check', True)
    fee_rate = config.get('fee_rate', 0.001)
    min_profit_pct = config.get('min_profit_after_fees_pct', 0.05)

    is_fee_profitable = True
    expected_profit = spread_pct / 2
    if use_fee_check:
        is_fee_profitable, expected_profit = check_fee_profitability(
            spread_pct, fee_rate, min_profit_pct
        )

    # Check trailing stop exit (refactored)
    trailing_signal = check_trailing_stop_exit(
        data, config, state, symbol, price, ob, inventory, is_cross_pair
    )
    if trailing_signal:
        return trailing_signal

    # Check position decay exit (MM-E04)
    decay_signal = check_position_decay_exit(
        data, config, state, symbol, price, ob, inventory, is_cross_pair, current_time
    )
    if decay_signal:
        return decay_signal

    # Get trailing stop price for logging
    trailing_stop_price = None
    if config.get('use_trailing_stop', False) and 'position_entries' in state:
        pos_entry = state['position_entries'].get(symbol)
        if pos_entry:
            tracking_price = pos_entry.get('highest_price' if pos_entry['side'] == 'long' else 'lowest_price', price)
            trailing_stop_price = calculate_trailing_stop(
                entry_price=pos_entry['entry_price'],
                highest_price=tracking_price,
                side=pos_entry['side'],
                activation_pct=config.get('trailing_stop_activation', 0.2),
                trail_distance_pct=config.get('trailing_stop_distance', 0.15)
            )

    # MM-008: Enhanced indicator logging
    state['indicators'] = {
        'symbol': symbol,
        'spread_pct': round(spread_pct, 4),
        'min_spread_pct': round(get_symbol_config(symbol, config, 'min_spread_pct'), 4),
        'effective_min_spread': round(effective_min_spread, 4),
        'best_bid': round(ob.best_bid, 8),
        'best_ask': round(ob.best_ask, 8),
        'mid': round(ob.mid, 8),
        'micro_price': round(micro_price, 8),  # v1.5.0
        'inventory': round(inventory, 4),
        'max_inventory': max_inventory,
        'imbalance': round(ob.imbalance, 4),
        'effective_threshold': round(effective_threshold, 4),
        'is_cross_pair': is_cross_pair,
        # Volatility metrics
        'volatility_pct': round(volatility, 4),
        'vol_multiplier': round(vol_multiplier, 2),
        'optimal_spread': round(optimal_spread, 4) if use_optimal else None,  # v1.5.0
        # Trade flow
        'trade_flow': round(trade_flow, 4),
        'trade_flow_aligned': False,
        # Fee profitability (v1.5.0)
        'is_fee_profitable': is_fee_profitable,
        'expected_profit_pct': round(expected_profit, 4),
        # Reservation price and trailing stop
        'reservation_price': round(reservation_price, 8) if use_reservation else None,
        'trailing_stop_price': round(trailing_stop_price, 8) if trailing_stop_price else None,
        # Per-pair metrics
        'pnl_symbol': state.get('pnl_by_symbol', {}).get(symbol, 0),
        'trades_symbol': state.get('trades_by_symbol', {}).get(symbol, 0),
    }

    # Check minimum spread (with volatility adjustment)
    if spread_pct < effective_min_spread:
        return None

    # MM-E03: Skip if not profitable after fees
    if use_fee_check and not is_fee_profitable:
        return None

    # Calculate position size with inventory skew
    skew_factor = 1.0 - abs(inventory / max_inventory) * config.get('inventory_skew', 0.5)
    position_size = base_size * max(skew_factor, 0.1)

    # Minimum trade size check (in USD)
    min_size = 5.0
    if position_size < min_size:
        return None

    # Decide action based on orderbook imbalance and inventory
    imbalance = ob.imbalance

    # MM-007: Trade flow alignment check
    def is_trade_flow_aligned(direction: str) -> bool:
        if not use_trade_flow:
            return True
        if direction == 'buy':
            return trade_flow > trade_flow_threshold
        elif direction == 'sell':
            return trade_flow < -trade_flow_threshold
        return True

    # If we're long and see selling pressure, reduce position
    if inventory > 0 and imbalance < -0.2:
        close_size = min(position_size, abs(inventory))
        if is_cross_pair:
            close_size = close_size * xrp_usdt_price
        if close_size >= min_size:
            entry_price = ob.best_bid
            state['indicators']['trade_flow_aligned'] = is_trade_flow_aligned('sell')
            return build_entry_signal(
                symbol=symbol,
                action='sell',
                size=close_size,
                entry_price=entry_price,
                reason=f"MM: Reduce long on sell pressure (imbal={imbalance:.2f})",
                sl_pct=sl_pct,
                tp_pct=tp_pct,
                is_cross_pair=is_cross_pair,
                xrp_usdt_price=xrp_usdt_price,
            )

    # If we're short and see buying pressure, cover
    if inventory < 0 and imbalance > 0.2:
        cover_size = min(position_size, abs(inventory))
        if is_cross_pair:
            cover_size = cover_size * xrp_usdt_price
        if cover_size >= min_size:
            entry_price = ob.best_ask
            state['indicators']['trade_flow_aligned'] = is_trade_flow_aligned('buy')
            return build_entry_signal(
                symbol=symbol,
                action='buy',
                size=cover_size,
                entry_price=entry_price,
                reason=f"MM: Reduce short on buy pressure (imbal={imbalance:.2f})",
                sl_pct=sl_pct,
                tp_pct=tp_pct,
                is_cross_pair=is_cross_pair,
                xrp_usdt_price=xrp_usdt_price,
            )

    # If inventory allows, trade based on spread capture
    if inventory < max_inventory and imbalance > effective_threshold:
        if not is_trade_flow_aligned('buy'):
            state['indicators']['trade_flow_aligned'] = False
            return None

        state['indicators']['trade_flow_aligned'] = True
        entry_price = ob.best_ask

        return build_entry_signal(
            symbol=symbol,
            action='buy',
            size=position_size,
            entry_price=entry_price,
            reason=f"MM: Spread capture buy (spread={spread_pct:.3f}%, imbal={imbalance:.2f}, vol={volatility:.2f}%, profit={expected_profit:.3f}%)",
            sl_pct=sl_pct,
            tp_pct=tp_pct,
            is_cross_pair=is_cross_pair,
            xrp_usdt_price=xrp_usdt_price,
        )

    if inventory > -max_inventory and imbalance < -effective_threshold:
        if not is_trade_flow_aligned('sell'):
            state['indicators']['trade_flow_aligned'] = False
            return None

        state['indicators']['trade_flow_aligned'] = True

        if inventory > 0:
            # We have a long position - sell to reduce
            close_size = min(position_size, inventory)
            if is_cross_pair:
                close_size_usd = close_size * xrp_usdt_price
            else:
                close_size_usd = close_size

            if close_size_usd >= min_size:
                entry_price = ob.best_bid
                return build_entry_signal(
                    symbol=symbol,
                    action='sell',
                    size=close_size_usd,
                    entry_price=entry_price,
                    reason=f"MM: Reduce long on imbalance (spread={spread_pct:.3f}%, vol={volatility:.2f}%)",
                    sl_pct=sl_pct,
                    tp_pct=tp_pct,
                    is_cross_pair=is_cross_pair,
                    xrp_usdt_price=xrp_usdt_price,
                )
        else:
            # We're flat or short
            if is_cross_pair:
                # For XRP/BTC, we don't short - just sell to accumulate BTC
                entry_price = ob.best_bid
                return build_entry_signal(
                    symbol=symbol,
                    action='sell',
                    size=position_size,
                    entry_price=entry_price,
                    reason=f"MM: Sell XRP for BTC (spread={spread_pct:.3f}%, imbal={imbalance:.2f})",
                    sl_pct=sl_pct,
                    tp_pct=tp_pct,
                    is_cross_pair=is_cross_pair,
                    xrp_usdt_price=xrp_usdt_price,
                )
            else:
                # For USD pairs, open short position
                entry_price = ob.best_bid
                return build_entry_signal(
                    symbol=symbol,
                    action='short',
                    size=position_size,
                    entry_price=entry_price,
                    reason=f"MM: Spread capture short (spread={spread_pct:.3f}%, vol={volatility:.2f}%, profit={expected_profit:.3f}%)",
                    sl_pct=sl_pct,
                    tp_pct=tp_pct,
                    is_cross_pair=is_cross_pair,
                    xrp_usdt_price=xrp_usdt_price,
                )

    return None


def generate_signal(
    data: DataSnapshot,
    config: Dict[str, Any],
    state: Dict[str, Any]
) -> Optional[Signal]:
    """
    Generate market making signal.

    Strategy:
    - Trade when spread is wide enough to capture
    - Skew quotes based on inventory
    - Use stop-loss and take-profit
    - For XRP/BTC: accumulate both assets through spread capture
    - MM-002: Adjust thresholds based on volatility
    - MM-003: Respect cooldown between signals
    - MM-007: Confirm with trade flow
    - MM-E03: Check fee profitability before entry
    - MM-E01: Use micro-price for better price discovery
    - MM-E04: Handle stale positions with decay

    Args:
        data: Immutable market data snapshot
        config: Strategy configuration (CONFIG + overrides)
        state: Mutable strategy state dict

    Returns:
        Signal if trading opportunity found, None otherwise
    """
    # Lazy initialization
    if 'initialized' not in state:
        state['initialized'] = True
        state['inventory'] = 0
        state['inventory_by_symbol'] = {}
        state['xrp_accumulated'] = 0.0
        state['btc_accumulated'] = 0.0
        state['last_signal_time'] = None
        state['indicators'] = {}

    current_time = data.timestamp

    # MM-003: Global cooldown check
    if state.get('last_signal_time') is not None:
        elapsed = (current_time - state['last_signal_time']).total_seconds()
        global_cooldown = config.get('cooldown_seconds', 5.0)
        if elapsed < global_cooldown:
            return None

    # Iterate over configured symbols
    for symbol in SYMBOLS:
        signal = _evaluate_symbol(data, config, state, symbol, current_time)
        if signal is not None:
            state['last_signal_time'] = current_time
            return signal

    return None
