"""
Grid RSI Reversion Strategy - Lifecycle Callbacks

Contains on_start, on_fill, and on_stop callbacks for strategy lifecycle management.
Handles grid state initialization and position tracking.
"""
import logging
from datetime import datetime
from typing import Dict, Any

from .config import STRATEGY_NAME, STRATEGY_VERSION, SYMBOLS
from .validation import validate_config
from .grid import (
    setup_grid_levels, mark_level_filled, check_cycle_completion,
    calculate_grid_stats, should_recenter_grid, recenter_grid
)

# Configure logger
logger = logging.getLogger(STRATEGY_NAME)


def initialize_state(state: Dict[str, Any]) -> None:
    """
    Initialize strategy state with all required fields.

    Args:
        state: Strategy state dict to initialize
    """
    state['initialized'] = True
    state['last_signal_time'] = None
    state['position_side'] = None
    state['position_size'] = 0.0
    state['position_by_symbol'] = {}
    state['position_entries'] = {}
    state['fills'] = []
    state['indicators'] = {}
    state['pnl_by_symbol'] = {}
    state['trades_by_symbol'] = {}
    state['wins_by_symbol'] = {}
    state['losses_by_symbol'] = {}
    state['consecutive_losses'] = 0
    state['circuit_breaker_time'] = None

    # Grid-specific state
    state['grid_levels'] = {}           # {symbol: [level_dicts]}
    state['grid_metadata'] = {}         # {symbol: metadata_dict}
    state['grids_initialized'] = {}     # {symbol: bool}

    # Rejection tracking
    state['rejection_counts'] = {}
    state['rejection_counts_by_symbol'] = {}


def initialize_grid_for_symbol(
    symbol: str,
    center_price: float,
    config: Dict[str, Any],
    state: Dict[str, Any],
    atr: float = None
) -> None:
    """
    Initialize grid levels for a specific symbol.

    Args:
        symbol: Trading symbol
        center_price: Center price for the grid
        config: Strategy configuration
        state: Strategy state dict
        atr: Optional ATR for dynamic spacing
    """
    grid_levels, metadata = setup_grid_levels(symbol, center_price, config, atr)

    state['grid_levels'][symbol] = grid_levels
    state['grid_metadata'][symbol] = metadata
    state['grids_initialized'][symbol] = True

    logger.info(f"Grid initialized for {symbol}", extra={
        'symbol': symbol,
        'center_price': center_price,
        'num_buy_levels': metadata['num_buy_levels'],
        'num_sell_levels': metadata['num_sell_levels'],
        'grid_spacing_pct': metadata['grid_spacing_pct'],
        'upper_price': metadata['upper_price'],
        'lower_price': metadata['lower_price'],
    })


def on_start(config: Dict[str, Any], state: Dict[str, Any]) -> None:
    """
    Initialize strategy state on startup.

    Called once when the strategy is loaded.

    Args:
        config: Strategy configuration
        state: Strategy state dict
    """
    # Validate configuration
    errors = validate_config(config)
    if errors:
        for error in errors:
            logger.warning(f"Config warning: {error}")
        state['config_warnings'] = errors
    state['config_validated'] = True

    # Initialize state
    initialize_state(state)

    # Store config for lifecycle access
    state['max_consecutive_losses'] = config.get('max_consecutive_losses', 3)

    logger.info(f"{STRATEGY_NAME} v{STRATEGY_VERSION} started", extra={
        'version': STRATEGY_VERSION,
        'symbols': SYMBOLS,
        'features': {
            'use_adaptive_rsi': config.get('use_adaptive_rsi', True),
            'use_trend_filter': config.get('use_trend_filter', True),
            'use_volatility_regimes': config.get('use_volatility_regimes', True),
            'use_atr_spacing': config.get('use_atr_spacing', True),
        }
    })

    # Log per-symbol configuration
    for symbol in SYMBOLS:
        from .config import SYMBOL_CONFIGS
        sym_config = SYMBOL_CONFIGS.get(symbol, {})
        logger.info(f"Symbol config: {symbol}", extra={
            'grid_type': sym_config.get('grid_type', 'geometric'),
            'num_grids': sym_config.get('num_grids', config.get('num_grids', 15)),
            'grid_spacing_pct': sym_config.get('grid_spacing_pct', config.get('grid_spacing_pct', 1.5)),
            'position_size_usd': sym_config.get('position_size_usd', config.get('position_size_usd', 20.0)),
        })


def on_fill(fill: Dict[str, Any], state: Dict[str, Any]) -> None:
    """
    Handle order fill - update grid state and position tracking.

    Called after each trade execution.

    Args:
        fill: Fill information dict with side, value, price, pnl, etc.
        state: Strategy state dict to update
    """
    # Store fill in history (keep last 100)
    if 'fills' not in state:
        state['fills'] = []
    state['fills'].append(fill)
    state['fills'] = state['fills'][-100:]

    side = fill.get('side', '')
    value = fill.get('value', fill.get('size', 0) * fill.get('price', 0))
    symbol = fill.get('symbol', 'XRP/USDT')
    price = fill.get('price', 0)
    pnl = fill.get('pnl', 0)
    timestamp = fill.get('timestamp', datetime.now())

    # Get grid order ID from fill metadata
    grid_order_id = fill.get('metadata', {}).get('grid_order_id')

    # Initialize tracking dicts
    for key in ['pnl_by_symbol', 'trades_by_symbol', 'wins_by_symbol', 'losses_by_symbol']:
        if key not in state:
            state[key] = {}

    # Track PnL and wins/losses
    if pnl != 0:
        state['pnl_by_symbol'][symbol] = state['pnl_by_symbol'].get(symbol, 0) + pnl

        if pnl > 0:
            state['wins_by_symbol'][symbol] = state['wins_by_symbol'].get(symbol, 0) + 1
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

    # Initialize position tracking
    if 'position_entries' not in state:
        state['position_entries'] = {}
    if 'position_by_symbol' not in state:
        state['position_by_symbol'] = {}

    # Update grid level state
    if grid_order_id and symbol in state.get('grid_levels', {}):
        grid_levels = state['grid_levels'][symbol]
        mark_level_filled(grid_levels, grid_order_id, price, timestamp)

        # Check for cycle completion
        matching = check_cycle_completion(grid_levels, grid_order_id)
        if matching:
            # Increment cycle count
            metadata = state.get('grid_metadata', {}).get(symbol, {})
            metadata['cycles_completed'] = metadata.get('cycles_completed', 0) + 1

            logger.info(f"Grid cycle completed: {symbol}", extra={
                'order_id': grid_order_id,
                'matched_id': matching.get('order_id'),
                'cycles_completed': metadata['cycles_completed'],
            })

    # Update position based on fill side
    if side == 'buy':
        state['position_side'] = 'long'
        state['position_size'] = state.get('position_size', 0) + value
        state['position_by_symbol'][symbol] = state['position_by_symbol'].get(symbol, 0) + value

        # Track entry
        if symbol not in state['position_entries']:
            state['position_entries'][symbol] = {
                'entry_price': price,
                'entry_time': timestamp,
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


def on_stop(state: Dict[str, Any]) -> None:
    """
    Called when strategy stops. Log summary statistics.

    Args:
        state: Strategy state dict
    """
    total_pnl = sum(state.get('pnl_by_symbol', {}).values())
    total_trades = sum(state.get('trades_by_symbol', {}).values())
    total_wins = sum(state.get('wins_by_symbol', {}).values())
    total_losses = sum(state.get('losses_by_symbol', {}).values())

    win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0

    # Grid statistics
    total_cycles = 0
    for symbol in SYMBOLS:
        metadata = state.get('grid_metadata', {}).get(symbol, {})
        total_cycles += metadata.get('cycles_completed', 0)

    # Rejection statistics
    rejection_counts = state.get('rejection_counts', {})
    total_rejections = sum(rejection_counts.values())

    # Clear indicators to save memory
    state['indicators'] = {}

    # Store final summary
    state['final_summary'] = {
        'strategy': STRATEGY_NAME,
        'version': STRATEGY_VERSION,
        'total_pnl': total_pnl,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'total_wins': total_wins,
        'total_losses': total_losses,
        'total_cycles': total_cycles,
        'pnl_by_symbol': state.get('pnl_by_symbol', {}),
        'trades_by_symbol': state.get('trades_by_symbol', {}),
        'rejection_counts': rejection_counts,
        'total_rejections': total_rejections,
    }

    logger.info(f"{STRATEGY_NAME} stopped", extra={
        'pnl': total_pnl,
        'trades': total_trades,
        'win_rate': win_rate,
        'wins': total_wins,
        'losses': total_losses,
        'cycles': total_cycles,
    })

    # Log grid statistics per symbol
    for symbol in SYMBOLS:
        grid_levels = state.get('grid_levels', {}).get(symbol, [])
        metadata = state.get('grid_metadata', {}).get(symbol, {})

        if grid_levels:
            stats = calculate_grid_stats(grid_levels, metadata)
            logger.info(f"Grid stats: {symbol}", extra={
                'cycles_completed': stats['cycles_completed'],
                'filled_buys': stats['filled_buys'],
                'filled_sells': stats['filled_sells'],
                'unfilled_buys': stats['unfilled_buys'],
                'unfilled_sells': stats['unfilled_sells'],
            })

    # Log rejection summary
    if rejection_counts:
        logger.info("Signal rejections summary", extra={
            'total_rejections': total_rejections,
            'top_rejections': dict(sorted(rejection_counts.items(), key=lambda x: -x[1])[:5]),
        })
