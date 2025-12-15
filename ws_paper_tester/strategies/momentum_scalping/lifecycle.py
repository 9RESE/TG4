"""
Momentum Scalping Strategy - Lifecycle Callbacks

Contains on_start, on_fill, and on_stop callbacks for strategy lifecycle management.
"""
from datetime import datetime
from typing import Dict, Any

from .config import STRATEGY_VERSION
from .validation import validate_config


def initialize_state(state: Dict[str, Any]) -> None:
    """Initialize strategy state with all required fields."""
    state['initialized'] = True
    state['last_signal_time'] = None
    state['last_candle_count'] = {}
    state['position_side'] = None
    state['position_size'] = 0.0
    state['position_by_symbol'] = {}
    state['fills'] = []
    state['indicators'] = {}
    state['pnl_by_symbol'] = {}
    state['trades_by_symbol'] = {}
    state['wins_by_symbol'] = {}
    state['losses_by_symbol'] = {}
    state['position_entries'] = {}
    state['consecutive_losses'] = 0
    state['circuit_breaker_time'] = None
    # RSI history for crossover detection
    state['prev_rsi'] = {}
    # MACD history
    state['prev_macd'] = {}
    # Rejection tracking
    state['rejection_counts'] = {}
    state['rejection_counts_by_symbol'] = {}


def on_start(config: Dict[str, Any], state: Dict[str, Any]) -> None:
    """
    Initialize strategy state on startup.

    Called once when the strategy is loaded.
    """
    # Validate configuration
    errors = validate_config(config)
    if errors:
        for error in errors:
            print(f"[momentum_scalping] Config warning: {error}")
        state['config_warnings'] = errors
    state['config_validated'] = True

    # Initialize state
    initialize_state(state)

    # Store circuit breaker config for on_fill access
    state['max_consecutive_losses'] = config.get('max_consecutive_losses', 3)

    print(f"[momentum_scalping] v{STRATEGY_VERSION} started")
    print(f"[momentum_scalping] Features: "
          f"Regimes={config.get('use_volatility_regimes', True)}, "
          f"Sessions={config.get('use_session_awareness', True)}, "
          f"Correlation={config.get('use_correlation_management', True)}, "
          f"5m_Filter={config.get('use_5m_trend_filter', True)}, "
          f"MACD_Confirm={config.get('use_macd_confirmation', True)}")
    # v2.0.0 features
    print(f"[momentum_scalping] v2.0 Features: "
          f"CorrelationPause={config.get('correlation_pause_enabled', True)}, "
          f"ADXFilter={config.get('use_adx_filter', True)}, "
          f"RegimeRSI={config.get('regime_high_rsi_overbought', 75)}/{config.get('regime_high_rsi_oversold', 25)}")
    if config.get('correlation_pause_enabled', True):
        print(f"[momentum_scalping] Correlation: warn<{config.get('correlation_warn_threshold', 0.55)}, "
              f"pause<{config.get('correlation_pause_threshold', 0.50)}")
    if config.get('use_adx_filter', True):
        print(f"[momentum_scalping] ADX: threshold>{config.get('adx_strong_trend_threshold', 25)} "
              f"(BTC only={config.get('adx_filter_btc_only', True)})")


def on_fill(fill: Dict[str, Any], state: Dict[str, Any]) -> None:
    """
    Track fills and update position state.

    Called after each trade execution.

    Args:
        fill: Fill information dict with side, value, price, pnl, etc.
        state: Strategy state dict to update
    """
    # Store fill in history (keep last 50)
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

    # Initialize tracking dicts if needed
    for key in ['pnl_by_symbol', 'trades_by_symbol', 'wins_by_symbol', 'losses_by_symbol']:
        if key not in state:
            state[key] = {}

    # Track PnL and wins/losses
    if pnl != 0:
        state['pnl_by_symbol'][symbol] = state['pnl_by_symbol'].get(symbol, 0) + pnl

        if pnl > 0:
            state['wins_by_symbol'][symbol] = state['wins_by_symbol'].get(symbol, 0) + 1
            # Reset consecutive losses on win
            state['consecutive_losses'] = 0
        else:
            state['losses_by_symbol'][symbol] = state['losses_by_symbol'].get(symbol, 0) + 1
            state['consecutive_losses'] = state.get('consecutive_losses', 0) + 1

            # Check circuit breaker
            max_losses = state.get('max_consecutive_losses', 3)
            if state['consecutive_losses'] >= max_losses:
                state['circuit_breaker_time'] = timestamp

    # Track trade count
    state['trades_by_symbol'][symbol] = state['trades_by_symbol'].get(symbol, 0) + 1

    # Initialize position tracking dicts
    if 'position_entries' not in state:
        state['position_entries'] = {}
    if 'position_by_symbol' not in state:
        state['position_by_symbol'] = {}

    # Update position based on fill side
    if side == 'buy':
        state['position_side'] = 'long'
        state['position_size'] = state.get('position_size', 0) + value
        state['position_by_symbol'][symbol] = state['position_by_symbol'].get(symbol, 0) + value

        # Track entry for this symbol
        if symbol not in state['position_entries']:
            state['position_entries'][symbol] = {
                'entry_price': price,
                'entry_time': timestamp,
                'highest_price': price,
                'lowest_price': price,
                'side': 'long',
            }
        else:
            # Update highest price for trailing stop
            pos = state['position_entries'][symbol]
            pos['highest_price'] = max(pos.get('highest_price', price), price)

    elif side == 'sell':
        # Closing long position
        state['position_size'] = max(0, state.get('position_size', 0) - value)
        state['position_by_symbol'][symbol] = max(0, state['position_by_symbol'].get(symbol, 0) - value)

        # Clear position if fully closed
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

        # Track entry for this symbol
        if symbol not in state['position_entries']:
            state['position_entries'][symbol] = {
                'entry_price': price,
                'entry_time': timestamp,
                'highest_price': price,
                'lowest_price': price,
                'side': 'short',
            }
        else:
            # Update lowest price for trailing stop
            pos = state['position_entries'][symbol]
            pos['lowest_price'] = min(pos.get('lowest_price', price), price)

    elif side == 'cover':
        # Closing short position
        state['position_size'] = max(0, state.get('position_size', 0) - value)
        state['position_by_symbol'][symbol] = max(0, state['position_by_symbol'].get(symbol, 0) - value)

        # Clear position if fully closed
        if state['position_size'] < 0.01:
            state['position_side'] = None
            state['position_size'] = 0.0

        if state['position_by_symbol'].get(symbol, 0) < 0.01:
            if symbol in state['position_entries']:
                del state['position_entries'][symbol]
            state['position_by_symbol'][symbol] = 0.0


def on_stop(state: Dict[str, Any]) -> None:
    """
    Called when strategy stops. Log summary statistics.
    """
    final_position = state.get('position_size', 0)
    final_side = state.get('position_side')
    total_fills = len(state.get('fills', []))

    total_pnl = sum(state.get('pnl_by_symbol', {}).values())
    total_trades = sum(state.get('trades_by_symbol', {}).values())
    total_wins = sum(state.get('wins_by_symbol', {}).values())
    total_losses = sum(state.get('losses_by_symbol', {}).values())

    win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0

    # Rejection statistics
    rejection_counts = state.get('rejection_counts', {})
    total_rejections = sum(rejection_counts.values())

    # Clear indicators to save memory
    state['indicators'] = {}

    # Store final summary
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
        'rejection_counts': rejection_counts,
        'rejection_counts_by_symbol': state.get('rejection_counts_by_symbol', {}),
        'total_rejections': total_rejections,
    }

    print(f"[momentum_scalping] Stopped. PnL: ${total_pnl:.2f}, "
          f"Trades: {total_trades}, Win Rate: {win_rate:.1f}%")

    # Print rejection summary
    if rejection_counts:
        print(f"[momentum_scalping] Signal rejections ({total_rejections} total):")
        for reason, count in sorted(rejection_counts.items(), key=lambda x: -x[1])[:5]:
            print(f"[momentum_scalping]   - {reason}: {count}")
