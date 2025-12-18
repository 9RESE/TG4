"""
Order Flow Strategy - Lifecycle Callbacks

Contains on_start, on_fill, and on_stop callbacks.
"""
from datetime import datetime
from typing import Dict, Any

from .config import STRATEGY_VERSION
from .validation import validate_config
from .signal import initialize_state


def on_start(config: Dict[str, Any], state: Dict[str, Any]) -> None:
    """Initialize strategy state on startup."""
    # REC-002: Validate configuration
    errors = validate_config(config)
    if errors:
        for error in errors:
            print(f"[order_flow] Config warning: {error}")
        state['config_warnings'] = errors
    state['config_validated'] = True

    initialize_state(state)

    # REC-002 (v4.2.0): Store circuit breaker config in state for on_fill access
    state['max_consecutive_losses'] = config.get('max_consecutive_losses', 3)

    print(f"[order_flow] v{STRATEGY_VERSION} started")
    print(f"[order_flow] Features: VPIN={config.get('use_vpin', True)}, "
          f"Regimes={config.get('use_volatility_regimes', True)}, "
          f"Sessions={config.get('use_session_awareness', True)}, "
          f"Correlation={config.get('use_correlation_management', True)}, "
          f"RejectionTracking={config.get('track_rejections', True)}")


def on_fill(fill: Dict[str, Any], state: Dict[str, Any]) -> None:
    """Track fills and update position state."""
    if 'fills' not in state:
        state['fills'] = []
    state['fills'].append(fill)
    state['fills'] = state['fills'][-50:]

    side = fill.get('side', '')
    value = fill.get('value', fill.get('size', 0) * fill.get('price', 0))
    symbol = fill.get('symbol', 'XRP/USDT')
    price = fill.get('price', 0)
    pnl = fill.get('pnl', 0)
    timestamp = fill.get('timestamp', datetime.now())

    for key in ['pnl_by_symbol', 'trades_by_symbol', 'wins_by_symbol', 'losses_by_symbol']:
        if key not in state:
            state[key] = {}

    if pnl != 0:
        state['pnl_by_symbol'][symbol] = state['pnl_by_symbol'].get(symbol, 0) + pnl

        if pnl > 0:
            state['wins_by_symbol'][symbol] = state['wins_by_symbol'].get(symbol, 0) + 1
            state['consecutive_losses'] = 0
        else:
            state['losses_by_symbol'][symbol] = state['losses_by_symbol'].get(symbol, 0) + 1
            state['consecutive_losses'] = state.get('consecutive_losses', 0) + 1

            # REC-002 (v4.2.0): Use config value from state instead of hardcoded
            max_losses = state.get('max_consecutive_losses', 3)
            if state['consecutive_losses'] >= max_losses:
                state['circuit_breaker_time'] = timestamp

    state['trades_by_symbol'][symbol] = state['trades_by_symbol'].get(symbol, 0) + 1

    if 'position_entries' not in state:
        state['position_entries'] = {}
    if 'position_by_symbol' not in state:
        state['position_by_symbol'] = {}

    if side == 'buy':
        state['position_side'] = 'long'
        state['position_size'] = state.get('position_size', 0) + value
        state['position_by_symbol'][symbol] = state['position_by_symbol'].get(symbol, 0) + value

        if symbol not in state['position_entries']:
            state['position_entries'][symbol] = {
                'entry_price': price,
                'entry_time': timestamp,
                'highest_price': price,
                'lowest_price': price,
                'side': 'long',
            }
        else:
            pos = state['position_entries'][symbol]
            pos['highest_price'] = max(pos.get('highest_price', price), price)

    elif side == 'sell':
        state['position_size'] = max(0, state.get('position_size', 0) - value)
        state['position_by_symbol'][symbol] = max(0, state['position_by_symbol'].get(symbol, 0) - value)

        if state['position_size'] < 0.01:
            state['position_side'] = None
            state['position_size'] = 0.0

        if state['position_by_symbol'].get(symbol, 0) < 0.01:
            if symbol in state['position_entries']:
                del state['position_entries'][symbol]
            state['position_by_symbol'][symbol] = 0.0

    elif side == 'short':
        state['position_side'] = 'short'
        state['position_size'] = state.get('position_size', 0) + value
        state['position_by_symbol'][symbol] = state['position_by_symbol'].get(symbol, 0) + value

        if symbol not in state['position_entries']:
            state['position_entries'][symbol] = {
                'entry_price': price,
                'entry_time': timestamp,
                'highest_price': price,
                'lowest_price': price,
                'side': 'short',
            }
        else:
            pos = state['position_entries'][symbol]
            pos['lowest_price'] = min(pos.get('lowest_price', price), price)

    elif side == 'cover':
        state['position_size'] = max(0, state.get('position_size', 0) - value)
        state['position_by_symbol'][symbol] = max(0, state['position_by_symbol'].get(symbol, 0) - value)

        if state['position_size'] < 0.01:
            state['position_side'] = None
            state['position_size'] = 0.0

        if state['position_by_symbol'].get(symbol, 0) < 0.01:
            if symbol in state['position_entries']:
                del state['position_entries'][symbol]
            state['position_by_symbol'][symbol] = 0.0


def on_stop(state: Dict[str, Any]) -> None:
    """Called when strategy stops."""
    final_position = state.get('position_size', 0)
    final_side = state.get('position_side')
    total_fills = len(state.get('fills', []))

    total_pnl = sum(state.get('pnl_by_symbol', {}).values())
    total_trades = sum(state.get('trades_by_symbol', {}).values())
    total_wins = sum(state.get('wins_by_symbol', {}).values())
    total_losses = sum(state.get('losses_by_symbol', {}).values())

    win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0

    # REC-001: Include rejection statistics in summary
    rejection_counts = state.get('rejection_counts', {})
    total_rejections = sum(rejection_counts.values())

    state['indicators'] = {}

    state['final_summary'] = {
        'position_side': final_side,
        'position_size': final_position,
        'total_fills': total_fills,
        'pnl_by_symbol': state.get('pnl_by_symbol', {}),
        'trades_by_symbol': state.get('trades_by_symbol', {}),
        'wins_by_symbol': state.get('wins_by_symbol', {}),
        'losses_by_symbol': state.get('losses_by_symbol', {}),
        'total_pnl': total_pnl,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'config_warnings': state.get('config_warnings', []),
        # REC-001: Rejection statistics
        'rejection_counts': rejection_counts,
        'rejection_counts_by_symbol': state.get('rejection_counts_by_symbol', {}),
        'total_rejections': total_rejections,
    }

    print(f"[order_flow] Stopped. PnL: ${total_pnl:.2f}, Trades: {total_trades}, Win Rate: {win_rate:.1f}%")

    # REC-001: Print rejection summary
    if rejection_counts:
        print(f"[order_flow] Signal rejections ({total_rejections} total):")
        for reason, count in sorted(rejection_counts.items(), key=lambda x: -x[1])[:5]:
            print(f"[order_flow]   - {reason}: {count}")
