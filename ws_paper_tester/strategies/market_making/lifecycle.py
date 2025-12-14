"""
Market Making Strategy - Lifecycle Callbacks

Handles on_start, on_fill, and on_stop lifecycle events.

Version History:
v2.1.0 (2025-12-14) - Deep Review v3.0 Implementation:
- No lifecycle changes in this version (changes in signals.py and config.py)

v2.0.0 additions:
- Circuit breaker tracking in on_fill (MM-C01)
- Rejection counts logging in on_stop (MM-M01)
"""
from datetime import datetime
from typing import Dict, Any

from .config import validate_config, is_xrp_btc
from .calculations import update_circuit_breaker_on_fill


def on_start(config: Dict[str, Any], state: Dict[str, Any]) -> None:
    """
    Initialize state and validate configuration.

    v1.4.0: Added config validation and trailing stop tracking.
    v1.5.0: Enhanced with all v1.4 review recommendations.
    v2.0.0: Added circuit breaker and rejection tracking initialization.
    """
    # Validate configuration
    errors = validate_config(config)
    if errors:
        for error in errors:
            print(f"[market_making] Config warning: {error}")
        state['config_warnings'] = errors
    state['config_validated'] = True

    # Core state
    state['initialized'] = True
    state['inventory'] = 0
    state['inventory_by_symbol'] = {}
    state['xrp_accumulated'] = 0.0
    state['btc_accumulated'] = 0.0
    state['last_signal_time'] = None
    state['indicators'] = {}

    # Position tracking for trailing stops and decay
    state['position_entries'] = {}

    # Per-pair metrics
    state['pnl_by_symbol'] = {}
    state['trades_by_symbol'] = {}

    # v2.0.0 MM-C01: Circuit breaker state
    state['consecutive_losses'] = 0
    state['circuit_breaker_triggered_time'] = None
    state['circuit_breaker_trigger_count'] = 0

    # v2.0.0 MM-M01: Signal rejection tracking
    state['rejection_counts'] = {}

    # v2.0.0 MM-H02: Trend tracking
    state['trend_consecutive'] = {}


def on_fill(fill: Dict[str, Any], state: Dict[str, Any], config: Dict[str, Any] = None) -> None:
    """
    Update inventory on fill.

    MM-005: Fixed unit handling - use value field from executor for USD pairs.
    v1.4.0: Added position tracking for trailing stops and per-pair metrics.
    v1.5.0: Added entry_time tracking for position decay (MM-E04).
    v2.0.0: Added circuit breaker tracking (MM-C01).

    For XRP/BTC:
    - Buy: +XRP inventory, track BTC spent
    - Sell: -XRP inventory, track BTC received
    """
    symbol = fill.get('symbol', 'XRP/USDT')
    side = fill.get('side', '')
    size = fill.get('size', 0)
    price = fill.get('price', 0)
    pnl = fill.get('pnl', 0)
    timestamp = fill.get('timestamp', datetime.now())
    value = fill.get('value', size * price)

    # v2.0.0 MM-C01: Update circuit breaker on fill with PnL
    if config is not None and pnl != 0:
        triggered = update_circuit_breaker_on_fill(state, config, pnl, timestamp)
        if triggered:
            print(f"[market_making] Circuit breaker triggered after {state.get('consecutive_losses', 0)} consecutive losses")

    # Initialize inventory tracking by symbol
    if 'inventory_by_symbol' not in state:
        state['inventory_by_symbol'] = {}

    current_inventory = state['inventory_by_symbol'].get(symbol, 0)
    is_cross_pair = is_xrp_btc(symbol)

    if is_cross_pair:
        xrp_amount = size
        btc_amount = size * price

        if side == 'buy':
            state['inventory_by_symbol'][symbol] = current_inventory + xrp_amount
            state['xrp_accumulated'] = state.get('xrp_accumulated', 0) + xrp_amount
        elif side == 'sell':
            state['inventory_by_symbol'][symbol] = current_inventory - xrp_amount
            state['btc_accumulated'] = state.get('btc_accumulated', 0) + btc_amount
    else:
        size_usd = value

        if side == 'buy':
            state['inventory_by_symbol'][symbol] = current_inventory + size_usd
        elif side == 'sell':
            state['inventory_by_symbol'][symbol] = current_inventory - size_usd
        elif side == 'short':
            state['inventory_by_symbol'][symbol] = current_inventory - size_usd
        elif side == 'cover':
            state['inventory_by_symbol'][symbol] = current_inventory + size_usd

    # Keep legacy inventory for backward compatibility
    state['inventory'] = sum(state['inventory_by_symbol'].values())
    state['last_fill'] = fill

    # Track position entries for trailing stops and decay (MM-E04)
    if 'position_entries' not in state:
        state['position_entries'] = {}

    if side == 'buy':
        if symbol not in state['position_entries']:
            state['position_entries'][symbol] = {
                'entry_price': price,
                'highest_price': price,
                'lowest_price': price,
                'side': 'long',
                'entry_time': timestamp,  # v1.5.0: Track entry time for decay
            }
        else:
            pos = state['position_entries'][symbol]
            pos['highest_price'] = max(pos['highest_price'], price)
    elif side == 'sell':
        if symbol in state['position_entries']:
            del state['position_entries'][symbol]
    elif side == 'short':
        if symbol not in state['position_entries']:
            state['position_entries'][symbol] = {
                'entry_price': price,
                'highest_price': price,
                'lowest_price': price,
                'side': 'short',
                'entry_time': timestamp,
            }
        else:
            pos = state['position_entries'][symbol]
            pos['lowest_price'] = min(pos['lowest_price'], price)
    elif side == 'cover':
        if symbol in state['position_entries']:
            del state['position_entries'][symbol]

    # Track per-pair metrics
    if 'pnl_by_symbol' not in state:
        state['pnl_by_symbol'] = {}
    if 'trades_by_symbol' not in state:
        state['trades_by_symbol'] = {}

    if pnl != 0:
        state['pnl_by_symbol'][symbol] = state['pnl_by_symbol'].get(symbol, 0) + pnl
    state['trades_by_symbol'][symbol] = state['trades_by_symbol'].get(symbol, 0) + 1


def on_stop(state: Dict[str, Any]) -> None:
    """
    Called when strategy stops.

    v1.4.0: Enhanced with per-pair metrics.
    v1.5.0: Added position decay stats.
    v2.0.0: Added rejection counts and circuit breaker stats logging (MM-M01, MM-C01).
    """
    # v2.0.0 MM-M01: Log rejection counts
    rejection_counts = state.get('rejection_counts', {})
    if rejection_counts:
        print("[market_making] Signal rejection counts:")
        for reason, count in sorted(rejection_counts.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")

    # v2.0.0 MM-C01: Log circuit breaker stats
    cb_triggers = state.get('circuit_breaker_trigger_count', 0)
    if cb_triggers > 0:
        print(f"[market_making] Circuit breaker triggered {cb_triggers} time(s)")

    state['final_summary'] = {
        'inventory_by_symbol': state.get('inventory_by_symbol', {}),
        'xrp_accumulated': state.get('xrp_accumulated', 0),
        'btc_accumulated': state.get('btc_accumulated', 0),
        'pnl_by_symbol': state.get('pnl_by_symbol', {}),
        'trades_by_symbol': state.get('trades_by_symbol', {}),
        'config_warnings': state.get('config_warnings', []),
        'position_entries': state.get('position_entries', {}),  # v1.5.0
        # v2.0.0: New stats
        'rejection_counts': rejection_counts,
        'circuit_breaker_trigger_count': cb_triggers,
        'consecutive_losses_at_end': state.get('consecutive_losses', 0),
    }
