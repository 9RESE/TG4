"""
Whale Sentiment Strategy - Lifecycle Callbacks

Contains on_start, on_fill, and on_stop callbacks for strategy lifecycle management.
"""
import logging
from datetime import datetime
from typing import Dict, Any

from .config import STRATEGY_NAME, STRATEGY_VERSION, SYMBOL_CONFIGS, CONFIG
from .validation import validate_config, validate_symbol_configs

# Configure structured logger
logger = logging.getLogger(STRATEGY_NAME)


def initialize_state(state: Dict[str, Any]) -> None:
    """Initialize strategy state with all required fields."""
    state['initialized'] = True
    state['last_signal_time'] = None
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

    # Whale sentiment specific state
    state['prev_rsi'] = {}
    state['prev_volume_ratio'] = {}
    state['prev_sentiment'] = {}
    state['volume_baseline'] = {}

    # Rejection tracking
    state['rejection_counts'] = {}
    state['rejection_counts_by_symbol'] = {}


def on_start(config: Dict[str, Any], state: Dict[str, Any]) -> None:
    """
    Initialize strategy state on startup.

    Called once when the strategy is loaded.

    Args:
        config: Strategy configuration
        state: Strategy state dict to initialize
    """
    # Validate configuration - REC-006: Include SYMBOL_CONFIGS validation
    errors = validate_config(config)
    symbol_errors = validate_symbol_configs(SYMBOL_CONFIGS, CONFIG)
    all_errors = errors + symbol_errors

    if all_errors:
        blocking_errors = [e for e in all_errors if e.startswith('BLOCKING:')]
        warnings = [e for e in all_errors if not e.startswith('BLOCKING:')]

        for warning in warnings:
            logger.warning("Config warning: %s", warning)
        for error in blocking_errors:
            logger.error("Config error: %s", error)

        state['config_warnings'] = warnings
        state['config_errors'] = blocking_errors

        # If blocking errors exist, strategy should not trade
        if blocking_errors:
            state['config_valid'] = False
            logger.error("Strategy has blocking config errors - trading disabled")
        else:
            state['config_valid'] = True
    else:
        state['config_valid'] = True

    state['config_validated'] = True

    # Initialize state
    initialize_state(state)

    # Store circuit breaker config for on_fill access
    state['max_consecutive_losses'] = config.get('max_consecutive_losses', 2)

    # Log startup
    logger.info(
        "Whale Sentiment v%s started",
        STRATEGY_VERSION,
        extra={
            'version': STRATEGY_VERSION,
            'features': {
                'contrarian_mode': config.get('contrarian_mode', True),
                'trade_flow': config.get('use_trade_flow_confirmation', True),
                'sessions': config.get('use_session_awareness', True),
                'correlation': config.get('use_correlation_management', True),
            }
        }
    )

    # Log whale sentiment specific settings
    logger.info(
        "Whale Sentiment settings",
        extra={
            'volume_spike_mult': config.get('volume_spike_mult', 2.0),
            'volume_window': config.get('volume_window', 288),
            'rsi_period': config.get('rsi_period', 14),
            'rsi_fear': config.get('rsi_extreme_fear', 25),
            'rsi_greed': config.get('rsi_extreme_greed', 75),
            'contrarian_mode': config.get('contrarian_mode', True),
        }
    )


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

            # Check circuit breaker - stricter for contrarian strategy
            max_losses = state.get('max_consecutive_losses', 2)
            if state['consecutive_losses'] >= max_losses:
                state['circuit_breaker_time'] = timestamp
                logger.warning(
                    "Circuit breaker triggered after %d consecutive losses",
                    state['consecutive_losses'],
                    extra={'symbol': symbol, 'pnl': pnl}
                )

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

    Args:
        state: Strategy state dict
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

    # Log shutdown summary
    logger.info(
        "Whale Sentiment strategy stopped",
        extra={
            'pnl': total_pnl,
            'trades': total_trades,
            'win_rate': win_rate,
            'wins': total_wins,
            'losses': total_losses,
            'final_position': final_position,
            'final_side': final_side,
        }
    )

    # Log rejection summary
    if rejection_counts:
        logger.info(
            "Signal rejections summary",
            extra={
                'total_rejections': total_rejections,
                'top_rejections': dict(sorted(rejection_counts.items(), key=lambda x: -x[1])[:5]),
            }
        )
