"""
EMA-9 Trend Flip Strategy - Lifecycle Callbacks

Strategy lifecycle callbacks: on_start, on_fill, on_stop.
Implements complete position tracking and structured logging.

Code Review Fixes Applied:
- Issue #1: Complete on_fill() handler with per-symbol tracking
- Issue #2: Config validation in on_start()
- Issue #3: Structured logging using Python logging module
- Issue #4: All required state fields for multi-symbol support
- Issue #6: Circuit breaker integration
- Issue #7: on_stop() with summary logging
"""
import logging
from datetime import datetime
from typing import Dict, Any, List

from .config import STRATEGY_NAME, STRATEGY_VERSION, CONFIG


# Configure structured logger
logger = logging.getLogger(STRATEGY_NAME)


def validate_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate strategy configuration.

    Args:
        config: Strategy configuration dict

    Returns:
        List of warning/error messages (empty if valid)
    """
    errors = []

    # Required fields
    required_fields = [
        'ema_period', 'consecutive_candles', 'position_size_usd',
        'max_position_usd', 'stop_loss_pct', 'take_profit_pct'
    ]

    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required config: {field}")

    # Validate ranges
    if config.get('ema_period', 9) < 2:
        errors.append("ema_period must be >= 2")

    if config.get('consecutive_candles', 3) < 1:
        errors.append("consecutive_candles must be >= 1")

    if config.get('position_size_usd', 0) <= 0:
        errors.append("position_size_usd must be > 0")

    if config.get('max_position_usd', 0) <= 0:
        errors.append("max_position_usd must be > 0")

    if config.get('stop_loss_pct', 0) <= 0:
        errors.append("stop_loss_pct must be > 0")

    if config.get('take_profit_pct', 0) <= 0:
        errors.append("take_profit_pct must be > 0")

    # Validate R:R ratio warning
    sl = config.get('stop_loss_pct', 1.0)
    tp = config.get('take_profit_pct', 2.0)
    if tp < sl:
        errors.append(f"Warning: R:R ratio < 1:1 (TP={tp}%, SL={sl}%)")

    # Validate buffer percentage
    buffer = config.get('buffer_pct', 0.1)
    if buffer < 0 or buffer > 5:
        errors.append(f"buffer_pct should be between 0 and 5, got {buffer}")

    # Validate timeframe
    timeframe = config.get('candle_timeframe_minutes', 60)
    if timeframe not in [5, 15, 30, 60, 240, 1440]:
        errors.append(f"Unusual candle_timeframe_minutes: {timeframe}")

    return errors


def initialize_state(state: Dict[str, Any]) -> None:
    """
    Initialize strategy state with all required fields.

    Implements complete state structure matching production strategies.
    """
    state['initialized'] = True

    # === Position Tracking (Global) ===
    state['position'] = 0.0
    state['position_side'] = None  # 'long' or 'short' or None
    state['entry_price'] = 0.0
    state['entry_time'] = None

    # === Position Tracking (Per-Symbol) ===
    state['position_by_symbol'] = {}  # symbol -> USD value
    state['position_entries'] = {}    # symbol -> entry details dict

    # === Trade Statistics (Global) ===
    state['trade_count'] = 0
    state['win_count'] = 0
    state['loss_count'] = 0
    state['total_pnl'] = 0.0

    # === Trade Statistics (Per-Symbol) ===
    state['pnl_by_symbol'] = {}
    state['trades_by_symbol'] = {}
    state['wins_by_symbol'] = {}
    state['losses_by_symbol'] = {}

    # === Signal Tracking ===
    state['last_signal_time'] = None
    state['last_trade_was_loss'] = False

    # === Circuit Breaker ===
    state['consecutive_losses'] = 0
    state['circuit_breaker_time'] = None
    state['max_consecutive_losses'] = 3  # Will be updated from config

    # === EMA-9 Specific ===
    state['prev_trend'] = None
    state['consecutive_count'] = 0
    state['hourly_candles'] = []
    state['ema_values'] = []

    # === Logging/Debugging ===
    state['indicators'] = {}
    state['rejection_counts'] = {}
    state['rejection_counts_by_symbol'] = {}

    # === Fill History (bounded) ===
    state['fills'] = []
    state['_max_fills'] = 100


def on_start(config: Dict[str, Any], state: Dict[str, Any]) -> None:
    """
    Initialize strategy state on startup.

    Called once when the strategy is loaded.
    Validates configuration and logs startup info.

    Args:
        config: Strategy configuration
        state: Strategy state dict to initialize
    """
    # Validate configuration
    errors = validate_config(config)
    if errors:
        for error in errors:
            logger.warning(f"Config issue: {error}")
        state['config_warnings'] = errors
    state['config_validated'] = True

    # Initialize state
    initialize_state(state)

    # Store circuit breaker config for on_fill access
    state['max_consecutive_losses'] = config.get('max_consecutive_losses', 3)

    # Log startup with structured data
    logger.info(f"EMA-9 Trend Flip v{STRATEGY_VERSION} started", extra={
        'version': STRATEGY_VERSION,
        'config': {
            'ema_period': config.get('ema_period', 9),
            'consecutive_candles': config.get('consecutive_candles', 3),
            'timeframe_minutes': config.get('candle_timeframe_minutes', 60),
            'buffer_pct': config.get('buffer_pct', 0.1),
            'position_size_usd': config.get('position_size_usd', 50.0),
            'max_position_usd': config.get('max_position_usd', 100.0),
            'stop_loss_pct': config.get('stop_loss_pct', 1.0),
            'take_profit_pct': config.get('take_profit_pct', 2.0),
            'use_atr_stops': config.get('use_atr_stops', False),
            'exit_on_flip': config.get('exit_on_flip', True),
            'max_hold_hours': config.get('max_hold_hours', 72),
        }
    })

    # Log feature status
    logger.info("EMA-9 features", extra={
        'exit_on_flip': config.get('exit_on_flip', True),
        'use_atr_stops': config.get('use_atr_stops', False),
        'track_rejections': config.get('track_rejections', True),
        'cooldown_minutes': config.get('cooldown_minutes', 30),
        'cooldown_after_loss_minutes': config.get('cooldown_after_loss_minutes', 60),
    })


def on_fill(fill: Dict[str, Any], state: Dict[str, Any]) -> None:
    """
    Track fills and update position state.

    Called after each trade execution by the PaperExecutor.

    Args:
        fill: Fill information dict with:
            - side: 'buy' | 'sell' | 'short' | 'cover'
            - value: USD value of the fill
            - price: Execution price
            - pnl: Realized P&L (0 for entries, non-zero for exits)
            - symbol: Trading pair (e.g., 'BTC/USDT')
            - timestamp: Fill timestamp
            - trigger: 'strategy' | 'stop_loss' | 'take_profit' | 'margin_call'
        state: Strategy state dict to update
    """
    # Extract fill data with safe defaults
    side = fill.get('side', '')
    value = fill.get('value', fill.get('size', 0) * fill.get('price', 0))
    symbol = fill.get('symbol', 'BTC/USDT')
    price = fill.get('price', 0)
    pnl = fill.get('pnl', 0)
    timestamp = fill.get('timestamp', datetime.now())
    trigger = fill.get('trigger', 'strategy')

    # === Store fill in bounded history ===
    if 'fills' not in state:
        state['fills'] = []
    state['fills'].append(fill)
    max_fills = state.get('_max_fills', 100)
    if len(state['fills']) > max_fills:
        state['fills'] = state['fills'][-max_fills:]

    # === Initialize tracking dicts if needed ===
    for key in ['pnl_by_symbol', 'trades_by_symbol', 'wins_by_symbol',
                'losses_by_symbol', 'position_by_symbol', 'position_entries']:
        if key not in state:
            state[key] = {}

    # === Track P&L and wins/losses ===
    if pnl != 0:
        state['pnl_by_symbol'][symbol] = state['pnl_by_symbol'].get(symbol, 0) + pnl
        state['total_pnl'] += pnl

        if pnl > 0:
            state['wins_by_symbol'][symbol] = state['wins_by_symbol'].get(symbol, 0) + 1
            state['win_count'] += 1
            state['consecutive_losses'] = 0
            state['last_trade_was_loss'] = False
        else:
            state['losses_by_symbol'][symbol] = state['losses_by_symbol'].get(symbol, 0) + 1
            state['loss_count'] += 1
            state['consecutive_losses'] = state.get('consecutive_losses', 0) + 1
            state['last_trade_was_loss'] = True

            # Check circuit breaker
            max_losses = state.get('max_consecutive_losses', 3)
            if state['consecutive_losses'] >= max_losses:
                state['circuit_breaker_time'] = timestamp
                logger.warning(f"Circuit breaker triggered after {max_losses} consecutive losses")

        # Track trade count
        state['trades_by_symbol'][symbol] = state['trades_by_symbol'].get(symbol, 0) + 1
        state['trade_count'] += 1

    # === Update position based on fill side ===
    if side == 'buy':
        # Opening or adding to long position
        if state.get('position_side') == 'short':
            # This is a cover (closing short) - shouldn't happen with 'buy' side
            # but handle gracefully
            _close_position(state, symbol, value, pnl)
        else:
            # Opening/adding long
            state['position_side'] = 'long'
            state['position'] = state.get('position', 0) + value
            state['position_by_symbol'][symbol] = state['position_by_symbol'].get(symbol, 0) + value

            # Track entry for this symbol
            if symbol not in state['position_entries'] or state['position_entries'][symbol].get('side') != 'long':
                state['position_entries'][symbol] = {
                    'entry_price': price,
                    'entry_time': timestamp,
                    'highest_price': price,
                    'lowest_price': price,
                    'side': 'long',
                    'trigger': trigger,
                }
                state['entry_price'] = price
                state['entry_time'] = timestamp
            else:
                # Update highest price for trailing stop potential
                pos = state['position_entries'][symbol]
                pos['highest_price'] = max(pos.get('highest_price', price), price)

    elif side == 'sell':
        # Closing long position
        _close_position(state, symbol, value, pnl)

    elif side == 'short':
        # Opening or adding to short position
        if state.get('position_side') == 'long':
            # This shouldn't happen - close long first
            _close_position(state, symbol, value, pnl)
        else:
            # Opening/adding short
            state['position_side'] = 'short'
            state['position'] = state.get('position', 0) + value
            state['position_by_symbol'][symbol] = state['position_by_symbol'].get(symbol, 0) + value

            # Track entry for this symbol
            if symbol not in state['position_entries'] or state['position_entries'][symbol].get('side') != 'short':
                state['position_entries'][symbol] = {
                    'entry_price': price,
                    'entry_time': timestamp,
                    'highest_price': price,
                    'lowest_price': price,
                    'side': 'short',
                    'trigger': trigger,
                }
                state['entry_price'] = price
                state['entry_time'] = timestamp
            else:
                # Update lowest price for trailing stop potential
                pos = state['position_entries'][symbol]
                pos['lowest_price'] = min(pos.get('lowest_price', price), price)

    elif side == 'cover':
        # Closing short position
        _close_position(state, symbol, value, pnl)

    # Log fill
    logger.debug(f"Fill processed: {side} {symbol}", extra={
        'side': side,
        'symbol': symbol,
        'value': value,
        'price': price,
        'pnl': pnl,
        'trigger': trigger,
        'position_after': state.get('position', 0),
        'position_side': state.get('position_side'),
    })


def _close_position(state: Dict[str, Any], symbol: str, value: float, pnl: float) -> None:
    """
    Helper to close/reduce a position.

    Args:
        state: Strategy state dict
        symbol: Trading pair
        value: USD value being closed
        pnl: Realized P&L
    """
    state['position'] = max(0, state.get('position', 0) - value)
    state['position_by_symbol'][symbol] = max(0, state['position_by_symbol'].get(symbol, 0) - value)

    # Clear position if fully closed
    if state['position'] < 0.01:
        state['position_side'] = None
        state['position'] = 0.0
        state['entry_price'] = 0.0
        state['entry_time'] = None

    if state['position_by_symbol'].get(symbol, 0) < 0.01:
        if symbol in state.get('position_entries', {}):
            del state['position_entries'][symbol]
        state['position_by_symbol'][symbol] = 0.0


def on_stop(state: Dict[str, Any]) -> None:
    """
    Called when strategy stops. Log summary statistics.

    Args:
        state: Strategy state dict
    """
    # Calculate summary statistics
    final_position = state.get('position', 0)
    final_side = state.get('position_side')
    total_fills = len(state.get('fills', []))

    total_pnl = state.get('total_pnl', 0)
    total_trades = state.get('trade_count', 0)
    total_wins = state.get('win_count', 0)
    total_losses = state.get('loss_count', 0)

    win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0

    # Rejection statistics
    rejection_counts = state.get('rejection_counts', {})
    total_rejections = sum(rejection_counts.values())

    # Store final summary
    state['final_summary'] = {
        'version': STRATEGY_VERSION,
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
        'total_rejections': total_rejections,
    }

    # Log shutdown summary
    logger.info(f"EMA-9 Trend Flip v{STRATEGY_VERSION} stopped", extra={
        'pnl': round(total_pnl, 4),
        'trades': total_trades,
        'win_rate': round(win_rate, 2),
        'wins': total_wins,
        'losses': total_losses,
        'final_position': round(final_position, 2),
        'final_side': final_side,
    })

    # Log per-symbol summary
    for symbol in state.get('pnl_by_symbol', {}):
        symbol_pnl = state['pnl_by_symbol'].get(symbol, 0)
        symbol_trades = state['trades_by_symbol'].get(symbol, 0)
        symbol_wins = state['wins_by_symbol'].get(symbol, 0)
        symbol_wr = (symbol_wins / symbol_trades * 100) if symbol_trades > 0 else 0

        logger.info(f"Symbol summary: {symbol}", extra={
            'symbol': symbol,
            'pnl': round(symbol_pnl, 4),
            'trades': symbol_trades,
            'win_rate': round(symbol_wr, 2),
        })

    # Log rejection summary
    if rejection_counts:
        logger.info("Signal rejections summary", extra={
            'total_rejections': total_rejections,
            'top_rejections': dict(sorted(rejection_counts.items(), key=lambda x: -x[1])[:5]),
        })

    # Clear indicators to save memory
    state['indicators'] = {}
