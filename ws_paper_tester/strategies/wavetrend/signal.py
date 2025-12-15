"""
WaveTrend Oscillator Strategy - Signal Generation

Contains the main generate_signal function and symbol evaluation logic.
Based on research from master-plan-v1.0.md.

Signal Generation Flow:
1. Initialize state (on first call)
2. Check circuit breaker / cooldowns
3. Calculate WaveTrend indicators (WT1, WT2)
4. Classify current zone (OB/OS/neutral/extreme)
5. Check existing position exits first:
   - Crossover reversal exit
   - Extreme zone profit taking
   - Stop loss / Take profit
6. Check entry conditions:
   - Crossover detection
   - Zone requirement (if require_zone_exit)
   - Divergence bonus
7. Check risk limits:
   - Position limits
   - Correlation exposure
   - Fee profitability
8. Generate Signal or None
"""
from datetime import datetime
from typing import Dict, Any, Optional

from ws_tester.types import DataSnapshot, Signal

from .config import (
    SYMBOLS, RejectionReason, CrossoverType, DivergenceType,
    get_symbol_config
)
from .indicators import (
    calculate_wavetrend, classify_zone, detect_crossover, detect_divergence,
    calculate_confidence, get_zone_string, is_in_oversold_zone, is_in_overbought_zone,
    check_trade_flow_confirmation
)
from .regimes import (
    classify_trading_session, get_session_adjustments
)
from .risk import (
    check_fee_profitability, check_circuit_breaker,
    check_correlation_exposure, check_position_limits,
    check_real_correlation
)
from .exits import check_all_exits
from .lifecycle import initialize_state


def track_rejection(
    state: Dict[str, Any],
    reason: RejectionReason,
    symbol: str = None
) -> None:
    """
    Track signal rejection for analysis.

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
    candle_count: int,
    status: str,
    state: Dict[str, Any]
) -> Dict[str, Any]:
    """Build base indicators dict for early returns."""
    return {
        'symbol': symbol,
        'candle_count': candle_count,
        'status': status,
        'position_side': state.get('position_side'),
        'position_size': state.get('position_size', 0),
        'consecutive_losses': state.get('consecutive_losses', 0),
        'pnl_symbol': state.get('pnl_by_symbol', {}).get(symbol, 0),
        'trades_symbol': state.get('trades_by_symbol', {}).get(symbol, 0),
    }


def generate_signal(
    data: DataSnapshot,
    config: Dict[str, Any],
    state: Dict[str, Any]
) -> Optional[Signal]:
    """
    Generate WaveTrend signal based on WT1/WT2 crossover.

    Called every tick (default: 100ms).

    Args:
        data: Immutable market data snapshot
        config: Strategy configuration (CONFIG + overrides)
        state: Mutable strategy state dict

    Returns:
        Signal if trading opportunity found, None otherwise
    """
    # Initialize state on first call
    if 'initialized' not in state:
        initialize_state(state)

    current_time = data.timestamp
    track_rejections = config.get('track_rejections', True)

    # ==========================================================================
    # REC-006: Block trading if configuration is invalid
    # ==========================================================================
    if not state.get('config_valid', True):
        state['indicators'] = build_base_indicators(
            symbol='N/A', candle_count=0, status='config_invalid', state=state
        )
        state['indicators']['config_errors'] = state.get('config_errors', [])
        return None

    # ==========================================================================
    # Circuit Breaker Check
    # ==========================================================================
    if config.get('use_circuit_breaker', True):
        max_losses = config.get('max_consecutive_losses', 3)
        cooldown_min = config.get('circuit_breaker_minutes', 30)

        if check_circuit_breaker(state, current_time, max_losses, cooldown_min):
            state['indicators'] = build_base_indicators(
                symbol='N/A', candle_count=0, status='circuit_breaker', state=state
            )
            state['indicators']['circuit_breaker_active'] = True
            if track_rejections:
                track_rejection(state, RejectionReason.CIRCUIT_BREAKER)
            return None

    # ==========================================================================
    # Time-Based Cooldown Check
    # ==========================================================================
    if state['last_signal_time'] is not None:
        elapsed = (current_time - state['last_signal_time']).total_seconds()
        if elapsed < config.get('cooldown_seconds', 60.0):
            state['indicators'] = build_base_indicators(
                symbol='N/A', candle_count=0, status='cooldown', state=state
            )
            state['indicators']['cooldown_remaining'] = config.get('cooldown_seconds', 60.0) - elapsed
            if track_rejections:
                track_rejection(state, RejectionReason.TIME_COOLDOWN)
            return None

    # ==========================================================================
    # Evaluate Each Symbol
    # ==========================================================================
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
    """
    Evaluate WaveTrend signals for a specific symbol.

    Args:
        data: Market data snapshot
        config: Strategy configuration
        state: Strategy state
        symbol: Symbol to evaluate
        current_time: Current timestamp

    Returns:
        Signal if entry/exit opportunity found, None otherwise
    """
    track_rejections = config.get('track_rejections', True)

    # ==========================================================================
    # Get Candle Data
    # WaveTrend is designed for hourly timeframes, but we use 5m candles
    # as our longer timeframe data available in ws_paper_tester
    # ==========================================================================
    candles_5m = data.candles_5m.get(symbol, ())
    candles_1m = data.candles_1m.get(symbol, ())

    # Use 5m candles for WaveTrend calculation (closer to hourly behavior)
    # Fall back to 1m if 5m not available
    candles = candles_5m if len(candles_5m) > 0 else candles_1m

    # Minimum candles required
    min_candles = config.get('min_candle_buffer', 50)
    if len(candles) < min_candles:
        state['indicators'] = build_base_indicators(
            symbol=symbol, candle_count=len(candles), status='warming_up', state=state
        )
        state['indicators']['required_candles'] = min_candles
        if track_rejections:
            track_rejection(state, RejectionReason.INSUFFICIENT_CANDLES, symbol)
        return None

    # ==========================================================================
    # Session Awareness
    # ==========================================================================
    session = classify_trading_session(current_time, config)
    session_adjustments = get_session_adjustments(session, config)

    # ==========================================================================
    # Calculate WaveTrend Indicators
    # ==========================================================================
    channel_length = config.get('wt_channel_length', 10)
    average_length = config.get('wt_average_length', 21)
    ma_length = config.get('wt_ma_length', 4)

    wt_result = calculate_wavetrend(
        candles, channel_length, average_length, ma_length
    )

    wt1 = wt_result['wt1']
    wt2 = wt_result['wt2']
    prev_wt1 = wt_result['prev_wt1']
    prev_wt2 = wt_result['prev_wt2']
    wt1_series = wt_result['wt1_series']

    if wt1 is None or wt2 is None:
        state['indicators'] = build_base_indicators(
            symbol=symbol, candle_count=len(candles), status='wt_calc_failed', state=state
        )
        if track_rejections:
            track_rejection(state, RejectionReason.WARMING_UP, symbol)
        return None

    # ==========================================================================
    # Zone Classification
    # ==========================================================================
    # Get symbol-specific thresholds
    symbol_config = {
        'wt_overbought': get_symbol_config(symbol, config, 'wt_overbought'),
        'wt_oversold': get_symbol_config(symbol, config, 'wt_oversold'),
        'wt_extreme_overbought': get_symbol_config(symbol, config, 'wt_extreme_overbought'),
        'wt_extreme_oversold': get_symbol_config(symbol, config, 'wt_extreme_oversold'),
    }
    # Fill in defaults for None values
    for key in symbol_config:
        if symbol_config[key] is None:
            symbol_config[key] = config.get(key)

    current_zone = classify_zone(wt1, symbol_config)

    # Get previous zone from state
    prev_zone_str = state.get('prev_zone', {}).get(symbol, 'neutral')
    from .config import WaveTrendZone
    prev_zone_map = {
        'extreme_overbought': WaveTrendZone.EXTREME_OVERBOUGHT,
        'overbought': WaveTrendZone.OVERBOUGHT,
        'neutral': WaveTrendZone.NEUTRAL,
        'oversold': WaveTrendZone.OVERSOLD,
        'extreme_oversold': WaveTrendZone.EXTREME_OVERSOLD,
    }
    prev_zone = prev_zone_map.get(prev_zone_str, WaveTrendZone.NEUTRAL)

    # Update state with current zone
    state.setdefault('prev_zone', {})[symbol] = get_zone_string(current_zone)
    state.setdefault('prev_wt1', {})[symbol] = wt1
    state.setdefault('prev_wt2', {})[symbol] = wt2

    # ==========================================================================
    # Crossover Detection
    # ==========================================================================
    crossover = detect_crossover(wt1, wt2, prev_wt1, prev_wt2)

    # ==========================================================================
    # Divergence Detection
    # ==========================================================================
    divergence = DivergenceType.NONE
    if config.get('use_divergence', True) and len(wt1_series) > 0:
        closes = [c.close for c in candles]
        lookback = config.get('divergence_lookback', 14)
        divergence = detect_divergence(
            closes, wt1_series, lookback,
            symbol_config['wt_oversold'], symbol_config['wt_overbought'], wt1
        )

    # ==========================================================================
    # Get Current Price
    # ==========================================================================
    closes = [c.close for c in candles]
    current_price = data.prices.get(symbol, closes[-1])

    # ==========================================================================
    # Get Symbol-Specific Config
    # ==========================================================================
    base_position_size = get_symbol_config(symbol, config, 'position_size_usd') or config.get('position_size_usd', 25.0)
    tp_pct = get_symbol_config(symbol, config, 'take_profit_pct') or config.get('take_profit_pct', 3.0)
    sl_pct = get_symbol_config(symbol, config, 'stop_loss_pct') or config.get('stop_loss_pct', 1.5)
    short_mult = config.get('short_size_multiplier', 0.8)

    # Apply session adjustment
    adjusted_position_size = base_position_size * session_adjustments['size_mult']

    # Fee profitability check
    use_fee_check = config.get('use_fee_check', True)
    fee_rate = config.get('fee_rate', 0.001)
    min_profit_pct = config.get('min_profit_after_fees_pct', 0.1)
    is_fee_profitable, expected_profit = check_fee_profitability(tp_pct, fee_rate, min_profit_pct)

    # ==========================================================================
    # Build Comprehensive Indicators
    # ==========================================================================
    state['indicators'] = {
        'symbol': symbol,
        'status': 'active',
        'candle_count': len(candles),
        'price': round(current_price, 6),
        # WaveTrend
        'wt1': round(wt1, 2),
        'wt2': round(wt2, 2),
        'wt_diff': round(wt1 - wt2, 2),
        'prev_wt1': round(prev_wt1, 2) if prev_wt1 else None,
        'prev_wt2': round(prev_wt2, 2) if prev_wt2 else None,
        # Zone
        'zone': get_zone_string(current_zone),
        'prev_zone': get_zone_string(prev_zone),
        'in_oversold': is_in_oversold_zone(current_zone),
        'in_overbought': is_in_overbought_zone(current_zone),
        # Crossover
        'crossover': crossover.name.lower() if crossover != CrossoverType.NONE else None,
        # Divergence
        'divergence': divergence.name.lower() if divergence != DivergenceType.NONE else None,
        # Session
        'trading_session': session.name,
        'session_size_mult': round(session_adjustments['size_mult'], 2),
        # Risk
        'is_fee_profitable': is_fee_profitable,
        'expected_profit_pct': round(expected_profit, 4),
        'adjusted_position_size': round(adjusted_position_size, 2),
        # Position
        'position_side': state.get('position_side'),
        'position_size': round(state.get('position_size', 0), 2),
        'position_size_symbol': round(state.get('position_by_symbol', {}).get(symbol, 0), 2),
        'consecutive_losses': state.get('consecutive_losses', 0),
        'pnl_symbol': round(state.get('pnl_by_symbol', {}).get(symbol, 0), 4),
    }

    # ==========================================================================
    # Check Exits First (For Existing Positions)
    # ==========================================================================
    exit_signal = check_all_exits(
        state=state,
        symbol=symbol,
        current_price=current_price,
        crossover=crossover,
        current_zone=current_zone,
        wt1=wt1,
        config=config
    )

    if exit_signal:
        state['last_signal_time'] = current_time
        state['indicators']['status'] = 'exit_signal'
        return exit_signal

    # ==========================================================================
    # Check If We Already Have Position In This Symbol
    # ==========================================================================
    current_position_symbol = state.get('position_by_symbol', {}).get(symbol, 0)
    if current_position_symbol > 0:
        state['indicators']['status'] = 'existing_position'
        if track_rejections:
            track_rejection(state, RejectionReason.EXISTING_POSITION, symbol)
        return None

    # ==========================================================================
    # No Crossover - No Entry
    # ==========================================================================
    if crossover == CrossoverType.NONE:
        state['indicators']['status'] = 'no_crossover'
        if track_rejections:
            track_rejection(state, RejectionReason.NO_CROSSOVER, symbol)
        return None

    # ==========================================================================
    # Position Limits Check
    # ==========================================================================
    can_trade, available_size, limit_reason = check_position_limits(
        state, symbol, adjusted_position_size, config
    )

    if not can_trade:
        state['indicators']['status'] = f'position_limit_{limit_reason}'
        if track_rejections:
            track_rejection(state, RejectionReason.MAX_POSITION, symbol)
        return None

    # ==========================================================================
    # Fee Profitability Check
    # ==========================================================================
    if use_fee_check and not is_fee_profitable:
        state['indicators']['status'] = 'not_fee_profitable'
        if track_rejections:
            track_rejection(state, RejectionReason.NOT_FEE_PROFITABLE, symbol)
        return None

    # ==========================================================================
    # Entry Signal Evaluation
    # ==========================================================================
    signal = None

    # ---------------------------------------------------------------------
    # Long Entry Conditions (Bullish Crossover)
    # ---------------------------------------------------------------------
    if crossover == CrossoverType.BULLISH:
        # Zone requirement check
        require_zone = config.get('require_zone_exit', True)

        # For bullish entry, we want crossover from/in oversold zone
        zone_confirmed = is_in_oversold_zone(prev_zone) or is_in_oversold_zone(current_zone)

        if require_zone and not zone_confirmed:
            state['indicators']['status'] = 'zone_not_confirmed_long'
            if track_rejections:
                track_rejection(state, RejectionReason.ZONE_NOT_CONFIRMED, symbol)
        else:
            # REC-001: Check trade flow confirmation
            use_trade_flow = config.get('use_trade_flow_confirmation', True)
            flow_confirmed = True
            flow_data = {}

            if use_trade_flow:
                trades = data.trades.get(symbol, ())
                flow_threshold = config.get('trade_flow_threshold', 0.10)
                flow_lookback = config.get('trade_flow_lookback', 50)
                flow_confirmed, flow_data = check_trade_flow_confirmation(
                    trades, 'buy', flow_threshold, flow_lookback
                )
                # Add flow data to indicators
                state['indicators']['trade_flow_imbalance'] = round(flow_data.get('imbalance', 0), 3)
                state['indicators']['trade_flow_confirms'] = flow_confirmed

            if not flow_confirmed:
                state['indicators']['status'] = 'trade_flow_against'
                if track_rejections:
                    track_rejection(state, RejectionReason.TRADE_FLOW_AGAINST, symbol)
            else:
                # REC-002: Check real-time correlation with existing positions
                candles_by_symbol = {
                    sym: data.candles_5m.get(sym, data.candles_1m.get(sym, ()))
                    for sym in SYMBOLS
                }
                corr_allowed, corr_adj, corr_info = check_real_correlation(
                    state, symbol, 'buy', candles_by_symbol, config
                )
                state['indicators']['real_correlation'] = corr_info.get('correlations', {})
                state['indicators']['correlation_blocked'] = corr_info.get('blocked', False)

                if not corr_allowed:
                    state['indicators']['status'] = 'real_correlation_blocked'
                    if track_rejections:
                        track_rejection(state, RejectionReason.CORRELATION_LIMIT, symbol)
                else:
                    # Calculate confidence
                    confidence, reasons = calculate_confidence(
                        crossover, current_zone, prev_zone, divergence, config
                    )

                    # Check correlation limits and apply REC-002 adjustment
                    adjusted_size = available_size * corr_adj
                    can_enter, final_size = check_correlation_exposure(
                        state, symbol, 'buy', adjusted_size, config
                    )

                    if not can_enter:
                        state['indicators']['status'] = 'correlation_limit'
                        if track_rejections:
                            track_rejection(state, RejectionReason.CORRELATION_LIMIT, symbol)
                    else:
                        reason_str = ', '.join(reasons[:3])
                        signal = Signal(
                            action='buy',
                            symbol=symbol,
                            size=final_size,
                            price=current_price,
                            reason=f"WT: Long ({reason_str}, WT1={wt1:.0f})",
                            stop_loss=current_price * (1 - sl_pct / 100),
                            take_profit=current_price * (1 + tp_pct / 100),
                            metadata={
                                'entry_type': 'wavetrend_long',
                                'wt1': wt1,
                                'wt2': wt2,
                                'zone': get_zone_string(current_zone),
                                'confidence': confidence,
                                'divergence': divergence.name.lower() if divergence != DivergenceType.NONE else None,
                                'trade_flow_imbalance': flow_data.get('imbalance'),
                                'correlation_adjustment': corr_adj,
                            }
                        )

    # ---------------------------------------------------------------------
    # Short Entry Conditions (Bearish Crossover)
    # ---------------------------------------------------------------------
    elif crossover == CrossoverType.BEARISH:
        # Zone requirement check
        require_zone = config.get('require_zone_exit', True)

        # For bearish entry, we want crossover from/in overbought zone
        zone_confirmed = is_in_overbought_zone(prev_zone) or is_in_overbought_zone(current_zone)

        if require_zone and not zone_confirmed:
            state['indicators']['status'] = 'zone_not_confirmed_short'
            if track_rejections:
                track_rejection(state, RejectionReason.ZONE_NOT_CONFIRMED, symbol)
        else:
            # REC-001: Check trade flow confirmation
            use_trade_flow = config.get('use_trade_flow_confirmation', True)
            flow_confirmed = True
            flow_data = {}

            if use_trade_flow:
                trades = data.trades.get(symbol, ())
                flow_threshold = config.get('trade_flow_threshold', 0.10)
                flow_lookback = config.get('trade_flow_lookback', 50)
                flow_confirmed, flow_data = check_trade_flow_confirmation(
                    trades, 'short', flow_threshold, flow_lookback
                )
                # Add flow data to indicators
                state['indicators']['trade_flow_imbalance'] = round(flow_data.get('imbalance', 0), 3)
                state['indicators']['trade_flow_confirms'] = flow_confirmed

            if not flow_confirmed:
                state['indicators']['status'] = 'trade_flow_against'
                if track_rejections:
                    track_rejection(state, RejectionReason.TRADE_FLOW_AGAINST, symbol)
            else:
                # REC-002: Check real-time correlation with existing positions
                candles_by_symbol = {
                    sym: data.candles_5m.get(sym, data.candles_1m.get(sym, ()))
                    for sym in SYMBOLS
                }
                corr_allowed, corr_adj, corr_info = check_real_correlation(
                    state, symbol, 'short', candles_by_symbol, config
                )
                state['indicators']['real_correlation'] = corr_info.get('correlations', {})
                state['indicators']['correlation_blocked'] = corr_info.get('blocked', False)

                if not corr_allowed:
                    state['indicators']['status'] = 'real_correlation_blocked'
                    if track_rejections:
                        track_rejection(state, RejectionReason.CORRELATION_LIMIT, symbol)
                else:
                    # Calculate confidence
                    confidence, reasons = calculate_confidence(
                        crossover, current_zone, prev_zone, divergence, config
                    )

                    # Apply short size multiplier and REC-002 correlation adjustment
                    short_size = available_size * short_mult * corr_adj

                    # Check correlation limits
                    can_enter, final_size = check_correlation_exposure(
                        state, symbol, 'short', short_size, config
                    )

                    if not can_enter:
                        state['indicators']['status'] = 'correlation_limit'
                        if track_rejections:
                            track_rejection(state, RejectionReason.CORRELATION_LIMIT, symbol)
                    else:
                        reason_str = ', '.join(reasons[:3])
                        signal = Signal(
                            action='short',
                            symbol=symbol,
                            size=final_size,
                            price=current_price,
                            reason=f"WT: Short ({reason_str}, WT1={wt1:.0f})",
                            stop_loss=current_price * (1 + sl_pct / 100),
                            take_profit=current_price * (1 - tp_pct / 100),
                            metadata={
                                'entry_type': 'wavetrend_short',
                                'wt1': wt1,
                                'wt2': wt2,
                                'zone': get_zone_string(current_zone),
                                'confidence': confidence,
                                'divergence': divergence.name.lower() if divergence != DivergenceType.NONE else None,
                                'trade_flow_imbalance': flow_data.get('imbalance'),
                                'correlation_adjustment': corr_adj,
                            }
                        )

    # ==========================================================================
    # Return Signal or None
    # ==========================================================================
    if signal:
        state['last_signal_time'] = current_time
        state['indicators']['status'] = 'signal_generated'
        return signal

    if state['indicators'].get('status') == 'active':
        state['indicators']['status'] = 'no_signal'
        if track_rejections:
            track_rejection(state, RejectionReason.NO_SIGNAL_CONDITIONS, symbol)

    return None
