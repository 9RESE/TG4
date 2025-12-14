"""
Ratio Trading Strategy - Lifecycle Module

Strategy lifecycle callbacks: on_start, on_fill, on_stop.
"""
from datetime import datetime
from typing import Dict, Any

from .config import STRATEGY_VERSION, SYMBOLS, validate_config
from .tracking import initialize_state
from .indicators import convert_usd_to_xrp


def on_start(config: Dict[str, Any], state: Dict[str, Any]) -> None:
    """Initialize strategy state on startup."""
    # REC-007: Validate configuration
    errors = validate_config(config)
    if errors:
        for error in errors:
            print(f"[ratio_trading] Config warning: {error}")
        state['config_warnings'] = errors
    state['config_validated'] = True

    initialize_state(state)

    # Store config values in state for use in on_fill (fixes hardcoded max_losses)
    state['max_consecutive_losses'] = config.get('max_consecutive_losses', 3)

    print(f"[ratio_trading] v{STRATEGY_VERSION} started")
    print(f"[ratio_trading] Symbol: {SYMBOLS[0]} (ratio pair)")
    print(f"[ratio_trading] Entry threshold: {config.get('entry_threshold', 1.5)} std (REC-013)")
    print(f"[ratio_trading] Core Features: VolatilityRegimes={config.get('use_volatility_regimes', True)}, "
          f"CircuitBreaker={config.get('use_circuit_breaker', True)}, "
          f"SpreadFilter={config.get('use_spread_filter', True)}")
    print(f"[ratio_trading] v2.1 Features: RSI={config.get('use_rsi_confirmation', True)}, "
          f"TrendFilter={config.get('use_trend_filter', True)}, "
          f"TrailingStop={config.get('use_trailing_stop', True)}, "
          f"PositionDecay={config.get('use_position_decay', True)}")
    print(f"[ratio_trading] v3.0 Features: CorrelationMonitoring={config.get('use_correlation_monitoring', True)}, "
          f"DynamicBTCPrice=True, SeparateExitTracking=True")
    print(f"[ratio_trading] v4.0 Features: CorrelationPauseEnabled={config.get('correlation_pause_enabled', True)}, "
          f"RaisedThresholds=True (research-validated)")
    # REC-036: Crypto Bollinger Bands
    bollinger_std = config.get('bollinger_std_crypto', 2.5) if config.get('use_crypto_bollinger_std', False) else config.get('bollinger_std', 2.0)
    print(f"[ratio_trading] v4.1 Features: CryptoBollingerStd={config.get('use_crypto_bollinger_std', False)} "
          f"(std={bollinger_std})")
    # REC-050: Fee profitability check
    fee_rate = config.get('estimated_fee_rate', 0.0026)
    print(f"[ratio_trading] v4.3 Features: FeeProfitabilityCheck={config.get('use_fee_profitability_check', True)}, "
          f"FeeRate={fee_rate*100:.2f}%, DecayMinutes={config.get('position_decay_minutes', 10)}")
    if config.get('use_correlation_monitoring', True):
        print(f"[ratio_trading] Correlation (v4.3.0): warn<{config.get('correlation_warning_threshold', 0.7)}, "
              f"pause<{config.get('correlation_pause_threshold', 0.4)} "
              f"(pause_enabled={config.get('correlation_pause_enabled', True)})")
    print(f"[ratio_trading] Position sizing: {config.get('position_size_usd', 15.0)} USD, "
          f"Max: {config.get('max_position_usd', 50.0)} USD")
    print(f"[ratio_trading] R:R ratio: {config.get('take_profit_pct', 0.6)}/{config.get('stop_loss_pct', 0.6)} "
          f"({config.get('take_profit_pct', 0.6)/config.get('stop_loss_pct', 0.6):.2f}:1)")


def on_fill(fill: Dict[str, Any], state: Dict[str, Any]) -> None:
    """
    Track fills and update position/accumulation state.

    For XRP/BTC:
    - Buy: Spent BTC to get XRP -> +XRP position
    - Sell: Sold XRP to get BTC -> -XRP position, +BTC accumulated

    REC-006: Per-pair PnL tracking
    REC-005: Circuit breaker consecutive loss tracking
    REC-016: Enhanced accumulation metrics
    REC-018: Dynamic BTC price for USD conversion
    """
    side = fill.get('side', '')
    symbol = fill.get('symbol', SYMBOLS[0])
    size = fill.get('size', 0)  # Size in USD now - REC-002
    price = fill.get('price', 0)  # Price in BTC per XRP
    pnl = fill.get('pnl', 0)
    timestamp = fill.get('timestamp', datetime.now())

    value = fill.get('value', size)  # USD value

    # REC-018: Get BTC price from state (set in generate_signal) or fallback
    btc_price_usd = state.get('last_btc_price_usd', 100000.0)

    # Convert USD to approximate XRP for tracking
    if price > 0:
        xrp_amount = convert_usd_to_xrp(value, price, btc_price_usd)
    else:
        xrp_amount = 0

    btc_value = xrp_amount * price

    # Initialize tracking dicts
    for key in ['pnl_by_symbol', 'trades_by_symbol', 'wins_by_symbol',
                'losses_by_symbol', 'position_entries', 'highest_price_since_entry',
                'lowest_price_since_entry']:
        if key not in state:
            state[key] = {}

    # REC-006: Per-pair PnL tracking
    if pnl != 0:
        state['pnl_by_symbol'][symbol] = state['pnl_by_symbol'].get(symbol, 0) + pnl

        # REC-005: Circuit breaker tracking (fixed: use state config instead of hardcoded)
        if pnl > 0:
            state['wins_by_symbol'][symbol] = state['wins_by_symbol'].get(symbol, 0) + 1
            state['consecutive_losses'] = 0
        else:
            state['losses_by_symbol'][symbol] = state['losses_by_symbol'].get(symbol, 0) + 1
            state['consecutive_losses'] = state.get('consecutive_losses', 0) + 1

            # Use config value from state (set in on_start) instead of hardcoded
            max_losses = state.get('max_consecutive_losses', 3)
            if state['consecutive_losses'] >= max_losses:
                state['circuit_breaker_time'] = timestamp

    state['trades_by_symbol'][symbol] = state['trades_by_symbol'].get(symbol, 0) + 1

    # Position tracking (USD and XRP)
    if side == 'buy':
        # Bought XRP with BTC
        state['position_usd'] = state.get('position_usd', 0) + value
        state['position_xrp'] = state.get('position_xrp', 0) + xrp_amount
        state['xrp_accumulated'] = state.get('xrp_accumulated', 0) + xrp_amount

        # REC-016: Track USD value at time of acquisition
        state['xrp_accumulated_value_usd'] = state.get('xrp_accumulated_value_usd', 0) + value
        state['total_trades_xrp_bought'] = state.get('total_trades_xrp_bought', 0) + 1

        if symbol not in state['position_entries']:
            state['position_entries'][symbol] = {
                'entry_price': price,
                'entry_time': timestamp,
                'side': 'long',
            }
            # Initialize trailing stop tracking
            state['highest_price_since_entry'][symbol] = price
            state['lowest_price_since_entry'][symbol] = price

    elif side == 'sell':
        # Sold XRP for BTC
        state['position_usd'] = max(0, state.get('position_usd', 0) - value)
        state['position_xrp'] = max(0, state.get('position_xrp', 0) - xrp_amount)
        state['btc_accumulated'] = state.get('btc_accumulated', 0) + btc_value

        # REC-016: Track USD value at time of BTC acquisition
        state['btc_accumulated_value_usd'] = state.get('btc_accumulated_value_usd', 0) + value
        state['total_trades_btc_bought'] = state.get('total_trades_btc_bought', 0) + 1

        if state['position_usd'] < 0.01:
            state['position_usd'] = 0.0
            state['position_xrp'] = 0.0
            if symbol in state['position_entries']:
                del state['position_entries'][symbol]
            # Clean up trailing stop tracking
            if symbol in state.get('highest_price_since_entry', {}):
                del state['highest_price_since_entry'][symbol]
            if symbol in state.get('lowest_price_since_entry', {}):
                del state['lowest_price_since_entry'][symbol]

    # Track fill history
    if 'fills' not in state:
        state['fills'] = []
    state['fills'].append(fill)
    state['fills'] = state['fills'][-20:]  # Keep last 20

    state['last_fill'] = fill


def on_stop(state: Dict[str, Any]) -> None:
    """
    Called when strategy stops.

    Logs comprehensive summary of trading performance.
    REC-016: Enhanced accumulation metrics in summary.
    REC-020: Exit tracking statistics.
    REC-021: Correlation monitoring summary.
    """
    symbol = SYMBOLS[0]

    total_pnl = sum(state.get('pnl_by_symbol', {}).values())
    total_trades = sum(state.get('trades_by_symbol', {}).values())
    total_wins = sum(state.get('wins_by_symbol', {}).values())
    total_losses = sum(state.get('losses_by_symbol', {}).values())

    win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0

    rejection_counts = state.get('rejection_counts', {})
    total_rejections = sum(rejection_counts.values())

    # REC-020: Exit tracking statistics
    exit_counts = state.get('exit_counts', {})
    exit_pnl_by_reason = state.get('exit_pnl_by_reason', {})
    total_exits = sum(exit_counts.values())

    # REC-021: Correlation monitoring statistics
    correlation_warnings = state.get('correlation_warnings', 0)
    correlation_history = state.get('correlation_history', [])
    avg_correlation = sum(correlation_history) / len(correlation_history) if correlation_history else 0

    # REC-016: Enhanced accumulation metrics
    xrp_acc = state.get('xrp_accumulated', 0)
    btc_acc = state.get('btc_accumulated', 0)
    xrp_value_usd = state.get('xrp_accumulated_value_usd', 0)
    btc_value_usd = state.get('btc_accumulated_value_usd', 0)
    xrp_trades = state.get('total_trades_xrp_bought', 0)
    btc_trades = state.get('total_trades_btc_bought', 0)

    state['indicators'] = {}

    state['final_summary'] = {
        'symbol': symbol,
        'position_usd': state.get('position_usd', 0),
        'position_xrp': state.get('position_xrp', 0),
        'xrp_accumulated': xrp_acc,
        'btc_accumulated': btc_acc,
        # REC-016: Enhanced metrics
        'xrp_accumulated_value_usd': xrp_value_usd,
        'btc_accumulated_value_usd': btc_value_usd,
        'total_trades_xrp_bought': xrp_trades,
        'total_trades_btc_bought': btc_trades,
        'avg_xrp_buy_value_usd': xrp_value_usd / xrp_trades if xrp_trades > 0 else 0,
        'avg_btc_buy_value_usd': btc_value_usd / btc_trades if btc_trades > 0 else 0,
        'pnl_by_symbol': state.get('pnl_by_symbol', {}),
        'trades_by_symbol': state.get('trades_by_symbol', {}),
        'wins_by_symbol': state.get('wins_by_symbol', {}),
        'losses_by_symbol': state.get('losses_by_symbol', {}),
        'total_pnl': total_pnl,
        'total_trades': total_trades,
        'total_wins': total_wins,
        'total_losses': total_losses,
        'win_rate': win_rate,
        'trade_count': state.get('trade_count', 0),
        'total_fills': len(state.get('fills', [])),
        'config_warnings': state.get('config_warnings', []),
        'rejection_counts': rejection_counts,
        'rejection_counts_by_symbol': state.get('rejection_counts_by_symbol', {}),
        'total_rejections': total_rejections,
        # REC-020: Exit tracking
        'exit_counts': exit_counts,
        'exit_counts_by_symbol': state.get('exit_counts_by_symbol', {}),
        'exit_pnl_by_reason': exit_pnl_by_reason,
        'total_exits': total_exits,
        # REC-021: Correlation monitoring
        'correlation_warnings': correlation_warnings,
        'avg_correlation': avg_correlation,
        'last_btc_price_usd': state.get('last_btc_price_usd'),
    }

    print(f"[ratio_trading] Stopped. PnL: ${total_pnl:.4f}, Trades: {total_trades}, Win Rate: {win_rate:.1f}%")

    # Print symbol summary
    sym_pnl = state.get('pnl_by_symbol', {}).get(symbol, 0)
    sym_trades = state.get('trades_by_symbol', {}).get(symbol, 0)
    sym_wins = state.get('wins_by_symbol', {}).get(symbol, 0)
    sym_losses = state.get('losses_by_symbol', {}).get(symbol, 0)
    sym_wr = (sym_wins / sym_trades * 100) if sym_trades > 0 else 0
    print(f"[ratio_trading]   {symbol}: PnL=${sym_pnl:.6f}, Trades={sym_trades}, WR={sym_wr:.1f}%")

    # Print accumulation summary (unique to ratio trading) - REC-016: Enhanced
    print(f"[ratio_trading]   Accumulated: XRP={xrp_acc:.4f} (${xrp_value_usd:.2f} cost, {xrp_trades} trades)")
    print(f"[ratio_trading]   Accumulated: BTC={btc_acc:.8f} (${btc_value_usd:.2f} value, {btc_trades} trades)")

    # REC-020: Print exit tracking summary
    if exit_counts:
        print(f"[ratio_trading] Intentional exits ({total_exits} total):")
        for reason, count in sorted(exit_counts.items(), key=lambda x: -x[1]):
            pnl_for_reason = exit_pnl_by_reason.get(reason, 0)
            print(f"[ratio_trading]   - {reason}: {count} (PnL: ${pnl_for_reason:.4f})")

    # REC-021: Print correlation summary
    if correlation_history:
        print(f"[ratio_trading] Correlation: avg={avg_correlation:.4f}, warnings={correlation_warnings}")

    # Print rejection summary
    if rejection_counts:
        print(f"[ratio_trading] Signal rejections ({total_rejections} total):")
        for reason, count in sorted(rejection_counts.items(), key=lambda x: -x[1])[:5]:
            print(f"[ratio_trading]   - {reason}: {count}")
