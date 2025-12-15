"""
Momentum Scalping Strategy - Signal Generation

Contains the main generate_signal function and symbol evaluation logic.
Based on research from master-plan-v1.0.md.

Signal Generation Flow:
1. Initialize state (on first call)
2. Check circuit breaker / cooldowns
3. Check volatility regime
4. Calculate indicators (RSI, MACD, EMA)
5. Check existing position exits first
6. Check entry conditions:
   - Trend alignment (EMA filter)
   - Momentum signal (RSI/MACD)
   - Volume confirmation
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
    SYMBOLS, VolatilityRegime, TradingSession, RejectionReason,
    get_symbol_config
)
from .indicators import (
    calculate_ema, calculate_rsi, calculate_rsi_series,
    calculate_macd_with_history, calculate_volume_ratio,
    calculate_volatility, check_ema_alignment, check_momentum_signal,
    # REC-002 (v2.0.0): 5m trend filter
    check_5m_trend_alignment,
    # REC-005 (v2.1.0): ATR for trailing stops
    calculate_atr,
)
from .regimes import (
    classify_volatility_regime, get_regime_adjustments,
    classify_trading_session, get_session_adjustments
)
from .risk import (
    check_fee_profitability, check_circuit_breaker,
    check_correlation_exposure, check_position_limits,
    is_volume_confirmed,
    # REC-001 (v2.0.0): Correlation monitoring
    get_xrp_btc_correlation, should_pause_for_low_correlation,
    # REC-003 (v2.0.0): ADX filter
    check_adx_strong_trend,
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
    Generate momentum scalping signal based on RSI, MACD, and EMA indicators.

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
    # Circuit Breaker Check
    # ==========================================================================
    if config.get('use_circuit_breaker', True):
        max_losses = config.get('max_consecutive_losses', 3)
        cooldown_min = config.get('circuit_breaker_minutes', 10)

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
        if elapsed < config.get('cooldown_seconds', 30.0):
            state['indicators'] = build_base_indicators(
                symbol='N/A', candle_count=0, status='cooldown', state=state
            )
            state['indicators']['cooldown_remaining'] = config.get('cooldown_seconds', 30.0) - elapsed
            if track_rejections:
                track_rejection(state, RejectionReason.TIME_COOLDOWN)
            return None

    # ==========================================================================
    # REC-001 (v2.0.0): XRP/BTC Correlation Monitoring
    # ==========================================================================
    if config.get('use_correlation_monitoring', True):
        get_xrp_btc_correlation(data, config, state)

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
    Evaluate momentum signals for a specific symbol.

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
    # ==========================================================================
    candles_1m = data.candles_1m.get(symbol, ())
    candles_5m = data.candles_5m.get(symbol, ())

    # Minimum candles required for indicators
    ema_filter_period = config.get('ema_filter_period', 50)
    rsi_period = config.get('rsi_period', 7)
    min_candles = max(ema_filter_period + 1, rsi_period + 2)

    if len(candles_1m) < min_candles:
        state['indicators'] = build_base_indicators(
            symbol=symbol, candle_count=len(candles_1m), status='warming_up', state=state
        )
        state['indicators']['required_candles'] = min_candles
        if track_rejections:
            track_rejection(state, RejectionReason.WARMING_UP, symbol)
        return None

    # ==========================================================================
    # Calculate Volatility and Regime
    # ==========================================================================
    volatility = calculate_volatility(candles_1m, config.get('volatility_lookback', 20))

    regime = VolatilityRegime.MEDIUM
    regime_adjustments = {'threshold_mult': 1.0, 'size_mult': 1.0, 'pause_trading': False}

    if config.get('use_volatility_regimes', True):
        regime = classify_volatility_regime(volatility, config)
        regime_adjustments = get_regime_adjustments(regime, config)

        if regime_adjustments['pause_trading']:
            state['indicators'] = build_base_indicators(
                symbol=symbol, candle_count=len(candles_1m), status='regime_pause', state=state
            )
            state['indicators']['volatility_regime'] = regime.name
            state['indicators']['volatility_pct'] = round(volatility, 4)
            if track_rejections:
                track_rejection(state, RejectionReason.REGIME_PAUSE, symbol)
            return None

    # ==========================================================================
    # Session Awareness
    # ==========================================================================
    session = TradingSession.EUROPE
    session_adjustments = {'threshold_mult': 1.0, 'size_mult': 1.0}

    if config.get('use_session_awareness', True):
        session = classify_trading_session(current_time, config)
        session_adjustments = get_session_adjustments(session, config)

    # Combined adjustments
    combined_size_mult = regime_adjustments['size_mult'] * session_adjustments['size_mult']

    # ==========================================================================
    # Calculate Indicators
    # ==========================================================================
    closes = [c.close for c in candles_1m]
    volumes = [c.volume for c in candles_1m]
    current_price = data.prices.get(symbol, closes[-1])

    # EMAs
    ema_fast_period = get_symbol_config(symbol, config, 'ema_fast_period') or config.get('ema_fast_period', 8)
    ema_slow_period = get_symbol_config(symbol, config, 'ema_slow_period') or config.get('ema_slow_period', 21)
    ema_filter_period = get_symbol_config(symbol, config, 'ema_filter_period') or config.get('ema_filter_period', 50)

    ema_fast = calculate_ema(closes, ema_fast_period)
    ema_slow = calculate_ema(closes, ema_slow_period)
    ema_filter = calculate_ema(closes, ema_filter_period)

    # EMA alignment check
    ema_alignment = check_ema_alignment(current_price, ema_fast, ema_slow, ema_filter)

    # RSI
    rsi_period = get_symbol_config(symbol, config, 'rsi_period') or config.get('rsi_period', 7)
    rsi = calculate_rsi(closes, rsi_period)

    # Get previous RSI for crossover detection
    prev_rsi = state.get('prev_rsi', {}).get(symbol)
    if rsi is not None:
        state.setdefault('prev_rsi', {})[symbol] = rsi

    # MACD
    macd_fast = config.get('macd_fast', 6)
    macd_slow = config.get('macd_slow', 13)
    macd_signal = config.get('macd_signal', 5)
    macd_result = calculate_macd_with_history(closes, macd_fast, macd_slow, macd_signal)

    # Volume
    volume_lookback = config.get('volume_lookback', 20)
    volume_ratio = calculate_volume_ratio(volumes, volume_lookback)
    volume_threshold = get_symbol_config(symbol, config, 'volume_spike_threshold') or config.get('volume_spike_threshold', 1.5)

    # REC-005 (v2.1.0): Calculate ATR for trailing stops
    atr = calculate_atr(candles_1m, period=14)

    # REC-007 (v2.1.0): Calculate trade imbalance for flow confirmation
    trade_imbalance = None
    if config.get('use_trade_flow_confirmation', True):
        trade_imbalance = data.get_trade_imbalance(symbol, n_trades=50)
        state['indicators']['trade_imbalance'] = round(trade_imbalance, 3) if trade_imbalance else None

    # ==========================================================================
    # REC-004 (v2.0.0): Regime-Based RSI Adjustment
    # Widen RSI bands during HIGH volatility regime to account for crypto's
    # tendency to sustain overbought conditions longer
    # ==========================================================================
    effective_config = config.copy()
    if regime == VolatilityRegime.HIGH or regime == VolatilityRegime.EXTREME:
        effective_config['rsi_overbought'] = config.get('regime_high_rsi_overbought', 75)
        effective_config['rsi_oversold'] = config.get('regime_high_rsi_oversold', 25)
        state['indicators']['rsi_adjusted_for_regime'] = True
        state['indicators']['regime_rsi_overbought'] = effective_config['rsi_overbought']
        state['indicators']['regime_rsi_oversold'] = effective_config['rsi_oversold']
    else:
        state['indicators']['rsi_adjusted_for_regime'] = False

    # Momentum signal check (using effective config with regime adjustments)
    momentum_signal = check_momentum_signal(rsi, prev_rsi, macd_result, effective_config)

    # ==========================================================================
    # Get Symbol-Specific Config
    # ==========================================================================
    base_position_size = get_symbol_config(symbol, config, 'position_size_usd')
    tp_pct = get_symbol_config(symbol, config, 'take_profit_pct')
    sl_pct = get_symbol_config(symbol, config, 'stop_loss_pct')
    adjusted_position_size = base_position_size * combined_size_mult

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
        'candle_count': len(candles_1m),
        'price': round(current_price, 6),
        # EMAs
        'ema_fast': round(ema_fast, 6) if ema_fast else None,
        'ema_slow': round(ema_slow, 6) if ema_slow else None,
        'ema_filter': round(ema_filter, 6) if ema_filter else None,
        'trend_direction': ema_alignment['trend_direction'],
        'bullish_aligned': ema_alignment['bullish_aligned'],
        'bearish_aligned': ema_alignment['bearish_aligned'],
        # RSI
        'rsi': round(rsi, 2) if rsi else None,
        'prev_rsi': round(prev_rsi, 2) if prev_rsi else None,
        # MACD
        'macd': round(macd_result['macd'], 6) if macd_result['macd'] else None,
        'macd_signal': round(macd_result['signal'], 6) if macd_result['signal'] else None,
        'macd_histogram': round(macd_result['histogram'], 6) if macd_result['histogram'] else None,
        'macd_bullish_crossover': macd_result.get('bullish_crossover', False),
        'macd_bearish_crossover': macd_result.get('bearish_crossover', False),
        # Volume
        'volume_ratio': round(volume_ratio, 2),
        'volume_threshold': volume_threshold,
        'volume_confirmed': volume_ratio >= volume_threshold,
        # Momentum
        'long_signal': momentum_signal['long_signal'],
        'short_signal': momentum_signal['short_signal'],
        'signal_strength': round(momentum_signal['signal_strength'], 2),
        'momentum_reasons': momentum_signal['reasons'],
        # Regime & Session
        'volatility_pct': round(volatility, 4),
        'volatility_regime': regime.name,
        'trading_session': session.name,
        'combined_size_mult': round(combined_size_mult, 2),
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
        # REC-005 (v2.1.0): ATR for trailing stops
        'atr': round(atr, 8) if atr else None,
    }

    # ==========================================================================
    # Check Exits First (For Existing Positions)
    # REC-005 (v2.1.0): Pass ATR for trailing stop check
    # ==========================================================================
    exit_signal = check_all_exits(
        state=state,
        symbol=symbol,
        current_price=current_price,
        current_time=current_time,
        rsi=rsi,
        ema_fast=ema_fast,
        config=config,
        atr=atr
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
    # REC-001 (v2.0.0): XRP/BTC Correlation Pause Check
    # Skip XRP/BTC entries when correlation drops below threshold
    # ==========================================================================
    if should_pause_for_low_correlation(symbol, state, config):
        state['indicators']['status'] = 'correlation_breakdown'
        state['indicators']['xrp_btc_correlation'] = state.get('xrp_btc_correlation')
        state['indicators']['correlation_pause_threshold'] = config.get('correlation_pause_threshold', 0.50)
        if track_rejections:
            track_rejection(state, RejectionReason.CORRELATION_BREAKDOWN, symbol)
        return None

    # ==========================================================================
    # REC-003 (v2.0.0): ADX Strong Trend Filter for BTC/USDT
    # Skip BTC/USDT entries when ADX indicates strong trending market
    # ==========================================================================
    if config.get('use_adx_filter', True):
        is_strong_trend, adx_value = check_adx_strong_trend(
            list(candles_1m), symbol, config, state
        )
        if adx_value is not None:
            state['indicators']['adx'] = round(adx_value, 2)
            state['indicators']['adx_threshold'] = config.get('adx_strong_trend_threshold', 25)
        if is_strong_trend:
            state['indicators']['status'] = 'adx_strong_trend'
            if track_rejections:
                track_rejection(state, RejectionReason.ADX_STRONG_TREND, symbol)
            return None

    # ==========================================================================
    # REC-002 (v2.0.0): 5m Trend Filter Check
    # Skip entries that don't align with 5m trend direction
    # ==========================================================================
    trend_5m_data = None
    if config.get('use_5m_trend_filter', True) and len(candles_5m) > 0:
        ema_5m_period = config.get('5m_ema_period', 50)
        trend_5m_data = check_5m_trend_alignment(
            candles_5m, current_price, ema_5m_period
        )
        state['indicators']['5m_ema'] = round(trend_5m_data['5m_ema'], 6) if trend_5m_data['5m_ema'] else None
        state['indicators']['5m_trend'] = trend_5m_data['5m_trend']
        state['indicators']['5m_bullish_aligned'] = trend_5m_data['bullish_aligned']
        state['indicators']['5m_bearish_aligned'] = trend_5m_data['bearish_aligned']

    # ==========================================================================
    # Entry Signal Evaluation
    # ==========================================================================
    signal = None

    # ---------------------------------------------------------------------
    # Long Entry Conditions
    # ---------------------------------------------------------------------
    if momentum_signal['long_signal'] and momentum_signal['signal_strength'] > 0:
        # Check trend alignment (1m EMA)
        require_ema_alignment = config.get('require_ema_alignment', True)
        if require_ema_alignment and not ema_alignment['bullish_aligned']:
            state['indicators']['status'] = 'trend_not_aligned_long'
            if track_rejections:
                track_rejection(state, RejectionReason.TREND_NOT_ALIGNED, symbol)
        # REC-002 (v2.0.0): Check 5m trend alignment
        elif config.get('use_5m_trend_filter', True) and trend_5m_data and not trend_5m_data['bullish_aligned']:
            state['indicators']['status'] = 'timeframe_misalignment_long'
            if track_rejections:
                track_rejection(state, RejectionReason.TIMEFRAME_MISALIGNMENT, symbol)
        else:
            # Check volume confirmation
            require_volume = config.get('require_volume_confirmation', True)
            if require_volume and volume_ratio < volume_threshold:
                state['indicators']['status'] = 'volume_not_confirmed_long'
                if track_rejections:
                    track_rejection(state, RejectionReason.VOLUME_NOT_CONFIRMED, symbol)
            # REC-007 (v2.1.0): Trade flow confirmation - require positive imbalance for longs
            elif config.get('use_trade_flow_confirmation', True):
                imbalance_threshold = config.get('trade_imbalance_threshold', 0.1)
                if trade_imbalance is not None and trade_imbalance < imbalance_threshold:
                    state['indicators']['status'] = 'trade_flow_misalignment_long'
                    if track_rejections:
                        track_rejection(state, RejectionReason.TRADE_FLOW_MISALIGNMENT, symbol)
                else:
                    # Check correlation limits
                    can_enter, final_size = check_correlation_exposure(
                        state, symbol, 'buy', available_size, config
                    )

                    if not can_enter:
                        state['indicators']['status'] = 'correlation_limit'
                        if track_rejections:
                            track_rejection(state, RejectionReason.CORRELATION_LIMIT, symbol)
                    else:
                        reasons = ', '.join(momentum_signal['reasons'][:2])
                        signal = Signal(
                            action='buy',
                            symbol=symbol,
                            size=final_size,
                            price=current_price,
                            reason=f"MS: Long ({reasons}, vol={volume_ratio:.1f}x, {regime.name})",
                            stop_loss=current_price * (1 - sl_pct / 100),
                            take_profit=current_price * (1 + tp_pct / 100),
                            metadata={
                                'entry_type': 'momentum_long',
                                'rsi': rsi,
                                'signal_strength': momentum_signal['signal_strength'],
                            }
                        )
            else:
                # Trade flow confirmation disabled - check correlation limits directly
                can_enter, final_size = check_correlation_exposure(
                    state, symbol, 'buy', available_size, config
                )

                if not can_enter:
                    state['indicators']['status'] = 'correlation_limit'
                    if track_rejections:
                        track_rejection(state, RejectionReason.CORRELATION_LIMIT, symbol)
                else:
                    reasons = ', '.join(momentum_signal['reasons'][:2])
                    signal = Signal(
                        action='buy',
                        symbol=symbol,
                        size=final_size,
                        price=current_price,
                        reason=f"MS: Long ({reasons}, vol={volume_ratio:.1f}x, {regime.name})",
                        stop_loss=current_price * (1 - sl_pct / 100),
                        take_profit=current_price * (1 + tp_pct / 100),
                        metadata={
                            'entry_type': 'momentum_long',
                            'rsi': rsi,
                            'signal_strength': momentum_signal['signal_strength'],
                        }
                    )

    # ---------------------------------------------------------------------
    # Short Entry Conditions
    # ---------------------------------------------------------------------
    elif momentum_signal['short_signal'] and momentum_signal['signal_strength'] > 0:
        # Check trend alignment (1m EMA)
        require_ema_alignment = config.get('require_ema_alignment', True)
        if require_ema_alignment and not ema_alignment['bearish_aligned']:
            state['indicators']['status'] = 'trend_not_aligned_short'
            if track_rejections:
                track_rejection(state, RejectionReason.TREND_NOT_ALIGNED, symbol)
        # REC-002 (v2.0.0): Check 5m trend alignment
        elif config.get('use_5m_trend_filter', True) and trend_5m_data and not trend_5m_data['bearish_aligned']:
            state['indicators']['status'] = 'timeframe_misalignment_short'
            if track_rejections:
                track_rejection(state, RejectionReason.TIMEFRAME_MISALIGNMENT, symbol)
        else:
            # Check volume confirmation
            require_volume = config.get('require_volume_confirmation', True)
            if require_volume and volume_ratio < volume_threshold:
                state['indicators']['status'] = 'volume_not_confirmed_short'
                if track_rejections:
                    track_rejection(state, RejectionReason.VOLUME_NOT_CONFIRMED, symbol)
            # REC-007 (v2.1.0): Trade flow confirmation - require negative imbalance for shorts
            elif config.get('use_trade_flow_confirmation', True):
                imbalance_threshold = config.get('trade_imbalance_threshold', 0.1)
                if trade_imbalance is not None and trade_imbalance > -imbalance_threshold:
                    state['indicators']['status'] = 'trade_flow_misalignment_short'
                    if track_rejections:
                        track_rejection(state, RejectionReason.TRADE_FLOW_MISALIGNMENT, symbol)
                else:
                    # Check correlation limits
                    can_enter, final_size = check_correlation_exposure(
                        state, symbol, 'short', available_size, config
                    )

                    if not can_enter:
                        state['indicators']['status'] = 'correlation_limit'
                        if track_rejections:
                            track_rejection(state, RejectionReason.CORRELATION_LIMIT, symbol)
                    else:
                        reasons = ', '.join(momentum_signal['reasons'][:2])
                        signal = Signal(
                            action='short',
                            symbol=symbol,
                            size=final_size,
                            price=current_price,
                            reason=f"MS: Short ({reasons}, vol={volume_ratio:.1f}x, {regime.name})",
                            stop_loss=current_price * (1 + sl_pct / 100),
                            take_profit=current_price * (1 - tp_pct / 100),
                            metadata={
                                'entry_type': 'momentum_short',
                                'rsi': rsi,
                                'signal_strength': momentum_signal['signal_strength'],
                            }
                        )
            else:
                # Trade flow confirmation disabled - check correlation limits directly
                can_enter, final_size = check_correlation_exposure(
                    state, symbol, 'short', available_size, config
                )

                if not can_enter:
                    state['indicators']['status'] = 'correlation_limit'
                    if track_rejections:
                        track_rejection(state, RejectionReason.CORRELATION_LIMIT, symbol)
                else:
                    reasons = ', '.join(momentum_signal['reasons'][:2])
                    signal = Signal(
                        action='short',
                        symbol=symbol,
                        size=final_size,
                        price=current_price,
                        reason=f"MS: Short ({reasons}, vol={volume_ratio:.1f}x, {regime.name})",
                        stop_loss=current_price * (1 + sl_pct / 100),
                        take_profit=current_price * (1 - tp_pct / 100),
                        metadata={
                            'entry_type': 'momentum_short',
                            'rsi': rsi,
                            'signal_strength': momentum_signal['signal_strength'],
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
