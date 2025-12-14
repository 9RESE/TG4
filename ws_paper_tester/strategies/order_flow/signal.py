"""
Order Flow Strategy - Signal Generation

Contains the main generate_signal function and symbol evaluation logic.
"""
from datetime import datetime
from typing import Dict, Any, Optional

from ws_tester.types import DataSnapshot, Signal

from .config import (
    SYMBOLS, VolatilityRegime, TradingSession, RejectionReason,
    get_symbol_config
)
from .indicators import calculate_volatility, calculate_micro_price, calculate_vpin
from .regimes import (
    classify_volatility_regime, get_regime_adjustments,
    classify_trading_session, get_session_adjustments
)
from .risk import (
    check_fee_profitability, calculate_trailing_stop,
    check_circuit_breaker, is_trade_flow_aligned, check_correlation_exposure
)
from .exits import check_trailing_stop_exit, check_position_decay_exit


def track_rejection(
    state: Dict[str, Any],
    reason: RejectionReason,
    symbol: str = None
) -> None:
    """
    REC-001: Track signal rejection for analysis.

    Args:
        state: Strategy state dict
        reason: RejectionReason enum value
        symbol: Optional symbol for per-symbol tracking
    """
    if 'rejection_counts' not in state:
        state['rejection_counts'] = {}
    if 'rejection_counts_by_symbol' not in state:
        state['rejection_counts_by_symbol'] = {}

    reason_key = reason.value
    state['rejection_counts'][reason_key] = state['rejection_counts'].get(reason_key, 0) + 1

    if symbol:
        if symbol not in state['rejection_counts_by_symbol']:
            state['rejection_counts_by_symbol'][symbol] = {}
        state['rejection_counts_by_symbol'][symbol][reason_key] = \
            state['rejection_counts_by_symbol'][symbol].get(reason_key, 0) + 1


def build_base_indicators(
    symbol: str,
    trade_count: int,
    status: str,
    state: Dict[str, Any]
) -> Dict[str, Any]:
    """Build base indicators dict for early returns."""
    return {
        'symbol': symbol,
        'trade_count': trade_count,
        'status': status,
        'position_side': state.get('position_side'),
        'position_size': state.get('position_size', 0),
        'consecutive_losses': state.get('consecutive_losses', 0),
        'pnl_symbol': state.get('pnl_by_symbol', {}).get(symbol, 0),
        'trades_symbol': state.get('trades_by_symbol', {}).get(symbol, 0),
    }


def initialize_state(state: Dict[str, Any]) -> None:
    """Initialize strategy state."""
    state['initialized'] = True
    state['last_signal_idx'] = 0
    state['total_trades_seen'] = 0
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
    # REC-001: Rejection tracking
    state['rejection_counts'] = {}
    state['rejection_counts_by_symbol'] = {}


def generate_signal(
    data: DataSnapshot,
    config: Dict[str, Any],
    state: Dict[str, Any]
) -> Optional[Signal]:
    """
    Generate order flow signal based on trade tape analysis.

    Args:
        data: Immutable market data snapshot
        config: Strategy configuration (CONFIG + overrides)
        state: Mutable strategy state dict

    Returns:
        Signal if trading opportunity found, None otherwise
    """
    if 'initialized' not in state:
        initialize_state(state)

    current_time = data.timestamp
    track_rejections = config.get('track_rejections', True)

    # Circuit breaker check
    if config.get('use_circuit_breaker', True):
        max_losses = config.get('max_consecutive_losses', 3)
        cooldown_min = config.get('circuit_breaker_minutes', 15)

        if check_circuit_breaker(state, current_time, max_losses, cooldown_min):
            state['indicators'] = build_base_indicators(
                symbol='N/A', trade_count=0, status='circuit_breaker', state=state
            )
            state['indicators']['circuit_breaker_active'] = True
            if track_rejections:
                track_rejection(state, RejectionReason.CIRCUIT_BREAKER)
            return None

    # Time-based cooldown check
    if state['last_signal_time'] is not None:
        elapsed = (current_time - state['last_signal_time']).total_seconds()
        if elapsed < config.get('cooldown_seconds', 5.0):
            state['indicators'] = build_base_indicators(
                symbol='N/A', trade_count=0, status='cooldown', state=state
            )
            state['indicators']['cooldown_remaining'] = config.get('cooldown_seconds', 5.0) - elapsed
            if track_rejections:
                track_rejection(state, RejectionReason.TIME_COOLDOWN)
            return None

    # Evaluate each symbol
    for symbol in SYMBOLS:
        signal = _evaluate_symbol(data, config, state, symbol, current_time)
        if signal:
            return signal

    return None


def _evaluate_symbol(
    data: DataSnapshot,
    config: Dict[str, Any],
    state: Dict[str, Any],
    symbol: str,
    current_time: datetime
) -> Optional[Signal]:
    """Evaluate order flow for a specific symbol."""
    trades = data.trades.get(symbol, ())
    base_lookback = config.get('lookback_trades', 50)
    track_rejections = config.get('track_rejections', True)

    # Not enough trades yet
    if len(trades) < base_lookback:
        state['indicators'] = build_base_indicators(
            symbol=symbol, trade_count=len(trades), status='warming_up', state=state
        )
        state['indicators']['required_trades'] = base_lookback
        if track_rejections:
            track_rejection(state, RejectionReason.WARMING_UP, symbol)
        return None

    # Calculate volatility for regime classification
    candles = data.candles_1m.get(symbol, ())
    volatility = calculate_volatility(candles, config.get('volatility_lookback', 20))

    # Classify volatility regime
    regime = VolatilityRegime.MEDIUM
    regime_adjustments = {'threshold_mult': 1.0, 'size_mult': 1.0, 'pause_trading': False}

    if config.get('use_volatility_regimes', True):
        regime = classify_volatility_regime(volatility, config)
        regime_adjustments = get_regime_adjustments(regime, config)

        if regime_adjustments['pause_trading']:
            state['indicators'] = build_base_indicators(
                symbol=symbol, trade_count=len(trades), status='regime_pause', state=state
            )
            state['indicators']['volatility_regime'] = regime.name
            state['indicators']['volatility_pct'] = round(volatility, 4)
            if track_rejections:
                track_rejection(state, RejectionReason.REGIME_PAUSE, symbol)
            return None

    # REC-003: Get session adjustments with configurable boundaries
    session = TradingSession.EUROPE
    session_adjustments = {'threshold_mult': 1.0, 'size_mult': 1.0}

    if config.get('use_session_awareness', True):
        session = classify_trading_session(current_time, config)
        session_adjustments = get_session_adjustments(session, config)

    # Adjust lookback based on regime
    lookback = base_lookback
    if regime == VolatilityRegime.HIGH:
        lookback = int(base_lookback * 0.75)
    elif regime == VolatilityRegime.EXTREME:
        lookback = int(base_lookback * 0.5)
    elif regime == VolatilityRegime.LOW:
        lookback = int(base_lookback * 1.25)

    lookback = max(20, min(lookback, len(trades)))

    recent_trades = trades[-lookback:]

    # Trade-based cooldown check
    cooldown_trades = get_symbol_config(symbol, config, 'cooldown_trades') or config.get('cooldown_trades', 10)
    state['total_trades_seen'] = len(trades)
    trades_since_signal = state['total_trades_seen'] - state['last_signal_idx']

    if trades_since_signal < cooldown_trades:
        state['indicators'] = build_base_indicators(
            symbol=symbol, trade_count=len(trades), status='trade_cooldown', state=state
        )
        state['indicators']['trades_since_signal'] = trades_since_signal
        if track_rejections:
            track_rejection(state, RejectionReason.TRADE_COOLDOWN, symbol)
        return None

    # Calculate VPIN
    vpin_value = 0.0
    vpin_pause = False
    if config.get('use_vpin', True):
        vpin_lookback = config.get('vpin_lookback_trades', 200)
        vpin_trades = trades[-vpin_lookback:] if len(trades) >= vpin_lookback else trades
        vpin_value = calculate_vpin(vpin_trades, config.get('vpin_bucket_count', 50))

        if vpin_value >= config.get('vpin_high_threshold', 0.7):
            if config.get('vpin_pause_on_high', True):
                vpin_pause = True

    if vpin_pause:
        state['indicators'] = build_base_indicators(
            symbol=symbol, trade_count=len(trades), status='vpin_pause', state=state
        )
        state['indicators']['vpin'] = round(vpin_value, 4)
        state['indicators']['vpin_threshold'] = config.get('vpin_high_threshold', 0.7)
        if track_rejections:
            track_rejection(state, RejectionReason.VPIN_PAUSE, symbol)
        return None

    # Calculate buy/sell imbalance
    buy_volume = sum(t.size for t in recent_trades if t.side == 'buy')
    sell_volume = sum(t.size for t in recent_trades if t.side == 'sell')
    total_volume = buy_volume + sell_volume

    if total_volume == 0:
        state['indicators'] = build_base_indicators(
            symbol=symbol, trade_count=len(trades), status='no_volume', state=state
        )
        if track_rejections:
            track_rejection(state, RejectionReason.NO_VOLUME, symbol)
        return None

    imbalance = (buy_volume - sell_volume) / total_volume

    # Calculate volume spike
    avg_trade_size = total_volume / len(recent_trades)
    last_5_trades = recent_trades[-5:]
    last_5_volume = sum(t.size for t in last_5_trades)
    expected_5_volume = avg_trade_size * 5
    volume_spike = last_5_volume / expected_5_volume if expected_5_volume > 0 else 1.0

    # VWAP and price
    vwap = data.get_vwap(symbol, lookback)
    current_price = data.prices.get(symbol, 0)

    if not current_price or not vwap:
        state['indicators'] = build_base_indicators(
            symbol=symbol, trade_count=len(trades), status='no_price_or_vwap', state=state
        )
        if track_rejections:
            track_rejection(state, RejectionReason.NO_PRICE_DATA, symbol)
        return None

    price_vs_vwap = (current_price - vwap) / vwap

    # Micro-price
    # REC-005 (v4.2.0): Track micro-price fallback for debugging
    ob = data.orderbooks.get(symbol)
    micro_price = current_price
    micro_price_fallback = True  # Assume fallback until proven otherwise
    if config.get('use_micro_price', True) and ob:
        micro_price = calculate_micro_price(ob)
        micro_price_fallback = False

    # Get symbol-specific config
    volume_spike_mult = get_symbol_config(symbol, config, 'volume_spike_mult') or config.get('volume_spike_mult', 2.0)
    base_position_size = get_symbol_config(symbol, config, 'position_size_usd')
    tp_pct = get_symbol_config(symbol, config, 'take_profit_pct')
    sl_pct = get_symbol_config(symbol, config, 'stop_loss_pct')

    # Asymmetric thresholds
    use_asymmetric = config.get('use_asymmetric_thresholds', True)
    if use_asymmetric:
        base_buy_threshold = get_symbol_config(symbol, config, 'buy_imbalance_threshold') or 0.30
        base_sell_threshold = get_symbol_config(symbol, config, 'sell_imbalance_threshold') or 0.25
    else:
        base_buy_threshold = get_symbol_config(symbol, config, 'imbalance_threshold') or 0.30
        base_sell_threshold = base_buy_threshold

    # Apply regime and session adjustments
    combined_threshold_mult = regime_adjustments['threshold_mult'] * session_adjustments['threshold_mult']
    effective_buy_threshold = base_buy_threshold * combined_threshold_mult
    effective_sell_threshold = base_sell_threshold * combined_threshold_mult

    combined_size_mult = regime_adjustments['size_mult'] * session_adjustments['size_mult']
    adjusted_position_size = base_position_size * combined_size_mult

    # Trade flow confirmation
    use_trade_flow = config.get('use_trade_flow_confirmation', True)
    trade_flow_threshold = config.get('trade_flow_threshold', 0.15)
    trade_flow = data.get_trade_imbalance(symbol, lookback)

    # Fee profitability check
    use_fee_check = config.get('use_fee_check', True)
    fee_rate = config.get('fee_rate', 0.001)
    min_profit_pct = config.get('min_profit_after_fees_pct', 0.05)

    is_fee_profitable = True
    expected_profit = tp_pct
    if use_fee_check:
        is_fee_profitable, expected_profit = check_fee_profitability(tp_pct, fee_rate, min_profit_pct)

    # Position limits
    # REC-006 (v4.2.0): Support both total and per-symbol position limits
    current_position = state.get('position_size', 0)
    current_position_symbol = state.get('position_by_symbol', {}).get(symbol, 0)
    max_position = config.get('max_position_usd', 100.0)
    max_position_symbol = config.get('max_position_per_symbol_usd', max_position)  # Default to total if not set
    min_trade = config.get('min_trade_size_usd', 5.0)

    # Check trailing stop exit
    trailing_signal = check_trailing_stop_exit(data, config, state, symbol, current_price, ob)
    if trailing_signal:
        return trailing_signal

    # Check position decay exit
    decay_signal = check_position_decay_exit(data, config, state, symbol, current_price, ob, current_time)
    if decay_signal:
        return decay_signal

    # Get trailing stop price for logging
    trailing_stop_price = None
    if config.get('use_trailing_stop', False):
        pos_entry = state.get('position_entries', {}).get(symbol)
        if pos_entry:
            tracking_price = pos_entry.get(
                'highest_price' if pos_entry['side'] == 'long' else 'lowest_price',
                current_price
            )
            trailing_stop_price = calculate_trailing_stop(
                entry_price=pos_entry['entry_price'],
                highest_price=tracking_price,
                side=pos_entry['side'],
                activation_pct=config.get('trailing_stop_activation', 0.3),
                trail_distance_pct=config.get('trailing_stop_distance', 0.2)
            )

    # Build comprehensive indicators
    state['indicators'] = {
        'symbol': symbol,
        'status': 'active',
        'trade_count': len(trades),
        'imbalance': round(imbalance, 4),
        'buy_volume': round(buy_volume, 2),
        'sell_volume': round(sell_volume, 2),
        'volume_spike': round(volume_spike, 2),
        'volume_spike_threshold': round(volume_spike_mult, 2),
        'vwap': round(vwap, 6),
        'price': round(current_price, 6),
        'micro_price': round(micro_price, 6),
        'micro_price_fallback': micro_price_fallback,  # REC-005 (v4.2.0): Log fallback status
        'price_vs_vwap': round(price_vs_vwap, 6),
        'volatility_pct': round(volatility, 4),
        'volatility_regime': regime.name,
        'regime_threshold_mult': round(regime_adjustments['threshold_mult'], 2),
        'regime_size_mult': round(regime_adjustments['size_mult'], 2),
        'trading_session': session.name,
        'session_threshold_mult': round(session_adjustments['threshold_mult'], 2),
        'session_size_mult': round(session_adjustments['size_mult'], 2),
        'combined_threshold_mult': round(combined_threshold_mult, 2),
        'combined_size_mult': round(combined_size_mult, 2),
        'base_buy_threshold': round(base_buy_threshold, 4),
        'base_sell_threshold': round(base_sell_threshold, 4),
        'effective_buy_threshold': round(effective_buy_threshold, 4),
        'effective_sell_threshold': round(effective_sell_threshold, 4),
        'adjusted_lookback': lookback,
        'vpin': round(vpin_value, 4),
        'vpin_threshold': config.get('vpin_high_threshold', 0.7),
        'trade_flow': round(trade_flow, 4),
        'trade_flow_threshold': round(trade_flow_threshold, 4),
        'use_trade_flow': use_trade_flow,
        'is_fee_profitable': is_fee_profitable,
        'expected_profit_pct': round(expected_profit, 4),
        'position_side': state.get('position_side'),
        'position_size': round(current_position, 2),
        'position_size_symbol': round(current_position_symbol, 2),  # REC-006 (v4.2.0)
        'max_position': max_position,
        'max_position_symbol': max_position_symbol,  # REC-006 (v4.2.0)
        'adjusted_position_size': round(adjusted_position_size, 2),
        'trailing_stop_price': round(trailing_stop_price, 6) if trailing_stop_price else None,
        'pnl_symbol': round(state.get('pnl_by_symbol', {}).get(symbol, 0), 4),
        'trades_symbol': state.get('trades_by_symbol', {}).get(symbol, 0),
        'consecutive_losses': state.get('consecutive_losses', 0),
    }

    # REC-006 (v4.2.0): Check both total and per-symbol position limits
    if current_position >= max_position:
        state['indicators']['status'] = 'max_position_reached'
        state['indicators']['max_position_reason'] = 'total'
        if track_rejections:
            track_rejection(state, RejectionReason.MAX_POSITION, symbol)
        return None

    if current_position_symbol >= max_position_symbol:
        state['indicators']['status'] = 'max_position_reached'
        state['indicators']['max_position_reason'] = 'per_symbol'
        if track_rejections:
            track_rejection(state, RejectionReason.MAX_POSITION, symbol)
        return None

    # REC-006 (v4.2.0): Respect both limits when calculating available size
    available_total = max_position - current_position
    available_symbol = max_position_symbol - current_position_symbol
    available = min(available_total, available_symbol)
    actual_size = min(adjusted_position_size, available)
    if actual_size < min_trade:
        state['indicators']['status'] = 'insufficient_size'
        if track_rejections:
            track_rejection(state, RejectionReason.INSUFFICIENT_SIZE, symbol)
        return None

    if use_fee_check and not is_fee_profitable:
        state['indicators']['status'] = 'not_fee_profitable'
        if track_rejections:
            track_rejection(state, RejectionReason.NOT_FEE_PROFITABLE, symbol)
        return None

    signal = None

    # Strong buy pressure with volume spike
    if imbalance > effective_buy_threshold and volume_spike > volume_spike_mult:
        if use_trade_flow and not is_trade_flow_aligned(data, symbol, 'buy', trade_flow_threshold, lookback):
            state['indicators']['status'] = 'trade_flow_not_aligned'
            state['indicators']['trade_flow_aligned'] = False
            if track_rejections:
                track_rejection(state, RejectionReason.TRADE_FLOW_NOT_ALIGNED, symbol)
            return None

        state['indicators']['trade_flow_aligned'] = True

        can_trade, adjusted_size = check_correlation_exposure(state, symbol, 'buy', actual_size, config)
        if not can_trade:
            state['indicators']['status'] = 'correlation_limit'
            if track_rejections:
                track_rejection(state, RejectionReason.CORRELATION_LIMIT, symbol)
            return None

        signal = Signal(
            action='buy',
            symbol=symbol,
            size=adjusted_size,
            price=current_price,
            reason=f"OF: Buy (imbal={imbalance:.2f}, vol={volume_spike:.1f}x, regime={regime.name}, session={session.name})",
            stop_loss=current_price * (1 - sl_pct / 100),
            take_profit=current_price * (1 + tp_pct / 100),
        )

    # Strong sell pressure with volume spike
    elif imbalance < -effective_sell_threshold and volume_spike > volume_spike_mult:
        if use_trade_flow and not is_trade_flow_aligned(data, symbol, 'sell', trade_flow_threshold, lookback):
            state['indicators']['status'] = 'trade_flow_not_aligned'
            state['indicators']['trade_flow_aligned'] = False
            if track_rejections:
                track_rejection(state, RejectionReason.TRADE_FLOW_NOT_ALIGNED, symbol)
            return None

        state['indicators']['trade_flow_aligned'] = True

        has_long = state.get('position_side') == 'long' and state.get('position_size', 0) > 0

        if has_long:
            close_size = min(actual_size, state.get('position_size', actual_size))
            signal = Signal(
                action='sell',
                symbol=symbol,
                size=close_size,
                price=current_price,
                reason=f"OF: Close long (imbal={imbalance:.2f}, vol={volume_spike:.1f}x)",
                stop_loss=current_price * (1 - sl_pct / 100),
                take_profit=current_price * (1 + tp_pct / 100),
            )
        else:
            can_trade, adjusted_size = check_correlation_exposure(state, symbol, 'short', actual_size, config)
            if not can_trade:
                state['indicators']['status'] = 'correlation_limit'
                if track_rejections:
                    track_rejection(state, RejectionReason.CORRELATION_LIMIT, symbol)
                return None

            signal = Signal(
                action='short',
                symbol=symbol,
                size=adjusted_size,
                price=current_price,
                reason=f"OF: Short (imbal={imbalance:.2f}, vol={volume_spike:.1f}x, regime={regime.name}, session={session.name})",
                stop_loss=current_price * (1 + sl_pct / 100),
                take_profit=current_price * (1 - tp_pct / 100),
            )

    # VWAP mean reversion opportunities
    # REC-004 (v4.2.0): Added trade flow confirmation for VWAP reversion signals
    vwap_threshold_mult = config.get('vwap_reversion_threshold_mult', 0.7)
    vwap_size_mult = config.get('vwap_reversion_size_mult', 0.75)
    vwap_deviation = config.get('vwap_deviation_threshold', 0.001)

    if signal is None and (imbalance > effective_buy_threshold * vwap_threshold_mult and
          price_vs_vwap < -vwap_deviation):
        # REC-004 (v4.2.0): Apply trade flow confirmation to VWAP reversion
        if use_trade_flow and not is_trade_flow_aligned(data, symbol, 'buy', trade_flow_threshold, lookback):
            state['indicators']['vwap_reversion_rejected'] = 'trade_flow_not_aligned'
        else:
            reduced_size = actual_size * vwap_size_mult
            can_trade, adjusted_size = check_correlation_exposure(state, symbol, 'buy', reduced_size, config)

            if can_trade and adjusted_size >= min_trade:
                signal = Signal(
                    action='buy',
                    symbol=symbol,
                    size=adjusted_size,
                    price=current_price,
                    reason=f"OF: Buy below VWAP (imbal={imbalance:.2f}, dev={price_vs_vwap:.4f})",
                    stop_loss=current_price * (1 - sl_pct / 100),
                    take_profit=vwap,
                )

    if signal is None and (imbalance < -effective_sell_threshold * vwap_threshold_mult and
          price_vs_vwap > vwap_deviation):
        has_long = state.get('position_side') == 'long' and state.get('position_size', 0) > 0
        reduced_size = actual_size * vwap_size_mult

        # REC-004 (v4.2.0): Only check trade flow for new shorts, not for closing longs
        if has_long and reduced_size >= min_trade:
            close_size = min(reduced_size, state.get('position_size', reduced_size))
            signal = Signal(
                action='sell',
                symbol=symbol,
                size=close_size,
                price=current_price,
                reason=f"OF: Close long above VWAP (imbal={imbalance:.2f}, dev={price_vs_vwap:.4f})",
            )

    if signal:
        state['last_signal_idx'] = state['total_trades_seen']
        state['last_signal_time'] = current_time
        state['indicators']['status'] = 'signal_generated'
        return signal

    state['indicators']['status'] = 'no_signal'
    if track_rejections:
        track_rejection(state, RejectionReason.NO_SIGNAL_CONDITIONS, symbol)
    return None
