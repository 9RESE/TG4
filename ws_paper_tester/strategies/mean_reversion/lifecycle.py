"""
Mean Reversion Strategy - Lifecycle Callbacks

Contains optional lifecycle callbacks:
- on_start: Initialize strategy on startup
- on_fill: Update position tracking on fill
- on_stop: Cleanup and summary on stop
"""
from datetime import datetime
from typing import Dict, Any

from .config import STRATEGY_VERSION, SYMBOLS, validate_config
from .signals import initialize_state


def on_start(config: Dict[str, Any], state: Dict[str, Any]) -> None:
    """Initialize strategy state on startup."""
    # REC-007: Validate configuration
    errors = validate_config(config)
    if errors:
        for error in errors:
            print(f"[mean_reversion] Config warning: {error}")
        state['config_warnings'] = errors
    state['config_validated'] = True

    # REC-002 (v3.0.0): Store config values for use in on_fill
    state['config_max_consecutive_losses'] = config.get('max_consecutive_losses', 3)

    initialize_state(state)

    print(f"[mean_reversion] v{STRATEGY_VERSION} started")
    print(f"[mean_reversion] Symbols: {SYMBOLS}")
    print(f"[mean_reversion] Features: VolatilityRegimes={config.get('use_volatility_regimes', True)}, "
          f"CircuitBreaker={config.get('use_circuit_breaker', True)}, "
          f"TradeFlowConfirm={config.get('use_trade_flow_confirmation', True)}, "
          f"TrendFilter={config.get('use_trend_filter', True)}, "
          f"TrailingStop={config.get('use_trailing_stop', False)}, "  # v4.0: default False per REC-001
          f"PositionDecay={config.get('use_position_decay', True)}, "
          f"CorrelationMonitor={config.get('use_correlation_monitoring', True)}, "  # v4.0: REC-005
          f"FeeCheck={config.get('check_fee_profitability', True)}")  # v4.1: REC-002
    # v4.1.0: Log key parameter changes
    fee_rate = config.get('estimated_fee_rate', 0.001)
    min_net = config.get('min_net_profit_pct', 0.05)
    print(f"[mean_reversion] v4.1 Params: DecayStart={config.get('decay_start_minutes', 15.0)}min, "
          f"TrendConfirm={config.get('trend_confirmation_periods', 3)} periods, "
          f"FeeRate={fee_rate*100:.2f}%/side, MinNet={min_net}%")
    # v4.2.0: Log correlation pause parameters (REC-001)
    corr_warn = config.get('correlation_warn_threshold', 0.4)
    corr_pause = config.get('correlation_pause_threshold', 0.25)
    corr_pause_enabled = config.get('correlation_pause_enabled', True)
    print(f"[mean_reversion] v4.2 Params: CorrelationWarn={corr_warn}, "
          f"CorrelationPause={corr_pause}, PauseEnabled={corr_pause_enabled}")


def on_fill(fill: Dict[str, Any], state: Dict[str, Any]) -> None:
    """
    Update position tracking and per-pair metrics.

    REC-006: Per-pair PnL tracking
    REC-005: Circuit breaker consecutive loss tracking
    """
    side = fill.get('side', '')
    symbol = fill.get('symbol', 'XRP/USDT')
    price = fill.get('price', 0)
    size = fill.get('size', 0)
    pnl = fill.get('pnl', 0)
    timestamp = fill.get('timestamp', datetime.now())

    value = fill.get('value', size * price)

    # Initialize tracking dicts
    for key in ['pnl_by_symbol', 'trades_by_symbol', 'wins_by_symbol',
                'losses_by_symbol', 'position_by_symbol', 'position_entries']:
        if key not in state:
            state[key] = {}

    # REC-006: Per-pair PnL tracking
    if pnl != 0:
        state['pnl_by_symbol'][symbol] = state['pnl_by_symbol'].get(symbol, 0) + pnl

        # REC-005: Circuit breaker tracking
        if pnl > 0:
            state['wins_by_symbol'][symbol] = state['wins_by_symbol'].get(symbol, 0) + 1
            state['consecutive_losses'] = 0
        else:
            state['losses_by_symbol'][symbol] = state['losses_by_symbol'].get(symbol, 0) + 1
            state['consecutive_losses'] = state.get('consecutive_losses', 0) + 1

            # REC-002 (v3.0.0): Use config value instead of hardcoded 3
            max_losses = state.get('config_max_consecutive_losses', 3)
            if state['consecutive_losses'] >= max_losses:
                state['circuit_breaker_time'] = timestamp

    state['trades_by_symbol'][symbol] = state['trades_by_symbol'].get(symbol, 0) + 1

    # Position tracking
    current_position = state['position_by_symbol'].get(symbol, 0)

    if side == 'buy':
        state['position_by_symbol'][symbol] = current_position + value
        if symbol not in state['position_entries']:
            state['position_entries'][symbol] = {
                'entry_price': price,
                'entry_time': timestamp,
                'side': 'long',
            }
    elif side == 'sell':
        state['position_by_symbol'][symbol] = max(0, current_position - value)
        if state['position_by_symbol'][symbol] < 0.01:
            state['position_by_symbol'][symbol] = 0.0
            if symbol in state['position_entries']:
                del state['position_entries'][symbol]
    elif side == 'short':
        state['position_by_symbol'][symbol] = current_position - value
        if symbol not in state['position_entries']:
            state['position_entries'][symbol] = {
                'entry_price': price,
                'entry_time': timestamp,
                'side': 'short',
            }
    elif side == 'cover':
        state['position_by_symbol'][symbol] = min(0, current_position + value)
        if abs(state['position_by_symbol'][symbol]) < 0.01:
            state['position_by_symbol'][symbol] = 0.0
            if symbol in state['position_entries']:
                del state['position_entries'][symbol]

    # Update aggregate position
    state['position'] = sum(state['position_by_symbol'].values())
    state['last_fill'] = fill


def on_stop(state: Dict[str, Any]) -> None:
    """
    Called when strategy stops.

    Finding #6: Added on_stop() with summary logging.
    """
    total_pnl = sum(state.get('pnl_by_symbol', {}).values())
    total_trades = sum(state.get('trades_by_symbol', {}).values())
    total_wins = sum(state.get('wins_by_symbol', {}).values())
    total_losses = sum(state.get('losses_by_symbol', {}).values())

    win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0

    rejection_counts = state.get('rejection_counts', {})
    total_rejections = sum(rejection_counts.values())

    state['indicators'] = {}

    state['final_summary'] = {
        'position_by_symbol': state.get('position_by_symbol', {}),
        'pnl_by_symbol': state.get('pnl_by_symbol', {}),
        'trades_by_symbol': state.get('trades_by_symbol', {}),
        'wins_by_symbol': state.get('wins_by_symbol', {}),
        'losses_by_symbol': state.get('losses_by_symbol', {}),
        'total_pnl': total_pnl,
        'total_trades': total_trades,
        'total_wins': total_wins,
        'total_losses': total_losses,
        'win_rate': win_rate,
        'config_warnings': state.get('config_warnings', []),
        'rejection_counts': rejection_counts,
        'rejection_counts_by_symbol': state.get('rejection_counts_by_symbol', {}),
        'total_rejections': total_rejections,
    }

    print(f"[mean_reversion] Stopped. PnL: ${total_pnl:.2f}, Trades: {total_trades}, Win Rate: {win_rate:.1f}%")

    # Print per-symbol summary
    for symbol in SYMBOLS:
        sym_pnl = state.get('pnl_by_symbol', {}).get(symbol, 0)
        sym_trades = state.get('trades_by_symbol', {}).get(symbol, 0)
        sym_wins = state.get('wins_by_symbol', {}).get(symbol, 0)
        sym_wr = (sym_wins / sym_trades * 100) if sym_trades > 0 else 0
        print(f"[mean_reversion]   {symbol}: PnL=${sym_pnl:.2f}, Trades={sym_trades}, WR={sym_wr:.1f}%")

    # Print rejection summary
    if rejection_counts:
        print(f"[mean_reversion] Signal rejections ({total_rejections} total):")
        for reason, count in sorted(rejection_counts.items(), key=lambda x: -x[1])[:5]:
            print(f"[mean_reversion]   - {reason}: {count}")
