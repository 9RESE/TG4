"""
Momentum Scalping Strategy - Lifecycle Callbacks

Contains on_start, on_fill, and on_stop callbacks for strategy lifecycle management.

REC-010 (v2.1.0): Structured logging using Python logging module.
REC-012/REC-013 (v2.1.0): Monitoring integration for correlation and sentiment tracking.
"""
import logging
from datetime import datetime
from typing import Dict, Any

from .config import STRATEGY_NAME, STRATEGY_VERSION
from .validation import validate_config
# REC-012/REC-013 (v2.1.0): Monitoring imports
from .monitoring import MonitoringManager, get_or_create_monitoring_manager

# REC-010 (v2.1.0): Configure structured logger
logger = logging.getLogger(STRATEGY_NAME)


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
    REC-010 (v2.1.0): Uses structured logging.
    """
    # Validate configuration
    errors = validate_config(config)
    if errors:
        for error in errors:
            # REC-010: Use structured logging instead of print
            logger.warning("Config warning", extra={'warning': error})
        state['config_warnings'] = errors
    state['config_validated'] = True

    # Initialize state
    initialize_state(state)

    # Store circuit breaker config for on_fill access
    state['max_consecutive_losses'] = config.get('max_consecutive_losses', 3)

    # REC-010 (v2.1.0): Structured logging for startup
    logger.info(f"v{STRATEGY_VERSION} started", extra={
        'version': STRATEGY_VERSION,
        'features': {
            'regimes': config.get('use_volatility_regimes', True),
            'sessions': config.get('use_session_awareness', True),
            'correlation': config.get('use_correlation_management', True),
            '5m_filter': config.get('use_5m_trend_filter', True),
            'macd_confirm': config.get('use_macd_confirmation', True),
        }
    })

    # v2.0.0 features
    logger.info("v2.0 features enabled", extra={
        'correlation_pause': config.get('correlation_pause_enabled', True),
        'adx_filter': config.get('use_adx_filter', True),
        'regime_rsi_overbought': config.get('regime_high_rsi_overbought', 75),
        'regime_rsi_oversold': config.get('regime_high_rsi_oversold', 25),
    })

    # v2.1.0 features
    logger.info("v2.1 features enabled", extra={
        'trailing_stop': config.get('use_trailing_stop', True),
        'trail_atr_mult': config.get('trail_atr_mult', 1.5),
        'trade_flow_confirm': config.get('use_trade_flow_confirmation', True),
        'imbalance_threshold': config.get('trade_imbalance_threshold', 0.1),
        'breakeven_exit': config.get('exit_breakeven_on_momentum_exhaustion', False),
    })

    if config.get('correlation_pause_enabled', True):
        logger.info("Correlation monitoring active", extra={
            'warn_threshold': config.get('correlation_warn_threshold', 0.60),
            'pause_threshold': config.get('correlation_pause_threshold', 0.60),
            'lookback': config.get('correlation_lookback', 100),
        })

    if config.get('use_adx_filter', True):
        logger.info("ADX filter active", extra={
            'threshold': config.get('adx_strong_trend_threshold', 30),
            'btc_only': config.get('adx_filter_btc_only', True),
        })

    # REC-012/REC-013 (v2.1.0): Initialize monitoring manager
    monitoring_manager = get_or_create_monitoring_manager(state)
    if monitoring_manager.needs_weekly_review():
        logger.info("Weekly correlation review due - run manager.generate_weekly_report()")

    logger.info("REC-012/REC-013 monitoring active", extra={
        'correlation_records': len(monitoring_manager.state.correlation_history),
        'sentiment_records': len(monitoring_manager.state.sentiment_history),
        'sessions_tracked': monitoring_manager.state.total_session_count,
    })


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

    REC-010 (v2.1.0): Uses structured logging.
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

    # REC-010 (v2.1.0): Structured logging for shutdown
    logger.info("Strategy stopped", extra={
        'pnl': total_pnl,
        'trades': total_trades,
        'win_rate': win_rate,
        'wins': total_wins,
        'losses': total_losses,
        'final_position': final_position,
        'final_side': final_side,
    })

    # Log rejection summary
    if rejection_counts:
        logger.info("Signal rejections summary", extra={
            'total_rejections': total_rejections,
            'top_rejections': dict(sorted(rejection_counts.items(), key=lambda x: -x[1])[:5]),
        })

    # REC-012/REC-013 (v2.1.0): Save monitoring state and generate report if due
    if '_monitoring_manager' in state:
        monitoring_manager = state['_monitoring_manager']

        # Generate weekly correlation report if due
        if monitoring_manager.needs_weekly_review():
            report = monitoring_manager.correlation_monitor.generate_weekly_report()
            logger.info("Weekly correlation report generated on shutdown", extra={
                'correlation_mean': report.get('correlation_stats', {}).get('mean'),
                'pause_rate': report.get('pause_stats', {}).get('pause_rate'),
                'recommendation': report.get('recommendation'),
            })

        # Save monitoring state
        monitoring_manager.save_state()

        # Log monitoring summary
        summary = monitoring_manager.get_monitoring_summary()
        logger.info("Monitoring summary on shutdown", extra={
            'correlation_records': summary['state_summary']['correlation_records'],
            'sentiment_records': summary['state_summary']['sentiment_records'],
            'total_sessions': summary['state_summary']['total_sessions_tracked'],
            'consecutive_low_days': summary['correlation']['consecutive_low_days'],
        })
