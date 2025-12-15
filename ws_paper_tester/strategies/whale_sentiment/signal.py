"""
Whale Sentiment Strategy - Signal Generation

Contains the main generate_signal function and symbol evaluation logic.
Based on research from master-plan-v1.0.md.

Signal Generation Flow:
1. Initialize state (on first call)
2. Check circuit breaker / cooldowns
3. Calculate indicators:
   - Volume spike detection (whale proxy)
   - Fear/greed price deviation (PRIMARY sentiment per REC-021)
   - ATR for volatility regime classification (REC-023)
   - Trade flow analysis
   NOTE: RSI removed in v1.3.0 per REC-021 (academic evidence)
4. Classify sentiment zone (using price deviation only)
5. Check existing position exits first
6. Check entry conditions:
   - Volume spike present (or moderate sentiment)
   - Sentiment zone aligned with entry direction
   - Whale signal alignment (contrarian mode)
7. Check risk limits
8. Generate Signal or None
"""
from datetime import datetime
from typing import Dict, Any, Optional

from ws_tester.types import DataSnapshot, Signal

from .config import (
    SYMBOLS, RejectionReason, SentimentZone, WhaleSignal,
    get_symbol_config
)
from .indicators import (
    detect_volume_spike, classify_whale_signal,
    calculate_fear_greed_proxy, classify_sentiment_zone,
    get_sentiment_string, is_fear_zone, is_greed_zone,
    detect_rsi_divergence, check_trade_flow_confirmation,
    calculate_composite_confidence, validate_volume_spike,
    calculate_atr  # REC-023: Volatility regime
)
from .regimes import (
    classify_trading_session, get_session_adjustments,
    get_sentiment_regime_adjustments, is_contrarian_opportunity,
    should_reduce_size_for_sentiment,
    # REC-023, REC-025, REC-027: New regime functions
    classify_volatility_regime, get_volatility_adjustments,
    check_extended_fear_period, calculate_dynamic_confidence_threshold
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
    Generate Whale Sentiment signal based on volume spikes and sentiment.

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
    # Circuit Breaker Check (Stricter for contrarian strategy)
    # ==========================================================================
    if config.get('use_circuit_breaker', True):
        max_losses = config.get('max_consecutive_losses', 2)  # Stricter default
        cooldown_min = config.get('circuit_breaker_minutes', 45)

        if check_circuit_breaker(state, current_time, max_losses, cooldown_min):
            # REC-005: Enhanced indicator logging for circuit breaker path
            state['indicators'] = build_base_indicators(
                symbol='N/A', candle_count=0, status='circuit_breaker', state=state
            )
            state['indicators']['circuit_breaker_active'] = True
            state['indicators']['max_consecutive_losses'] = max_losses
            state['indicators']['cooldown_minutes'] = cooldown_min
            breaker_time = state.get('circuit_breaker_time')
            if breaker_time:
                elapsed = (current_time - breaker_time).total_seconds() / 60
                state['indicators']['cooldown_elapsed_min'] = round(elapsed, 1)
                state['indicators']['cooldown_remaining_min'] = round(cooldown_min - elapsed, 1)
            if track_rejections:
                track_rejection(state, RejectionReason.CIRCUIT_BREAKER)
            return None

    # ==========================================================================
    # Time-Based Cooldown Check
    # ==========================================================================
    if state['last_signal_time'] is not None:
        elapsed = (current_time - state['last_signal_time']).total_seconds()
        cooldown_secs = config.get('cooldown_seconds', 120.0)
        if elapsed < cooldown_secs:
            # REC-005: Enhanced indicator logging for cooldown path
            state['indicators'] = build_base_indicators(
                symbol='N/A', candle_count=0, status='cooldown', state=state
            )
            state['indicators']['cooldown_seconds'] = cooldown_secs
            state['indicators']['cooldown_elapsed'] = round(elapsed, 1)
            state['indicators']['cooldown_remaining'] = round(cooldown_secs - elapsed, 1)
            state['indicators']['last_signal_time'] = state['last_signal_time'].isoformat()
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
    Evaluate whale sentiment signals for a specific symbol.

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
    candles_5m = data.candles_5m.get(symbol, ())
    candles_1m = data.candles_1m.get(symbol, ())

    # Use 5m candles for calculations (better signal quality)
    # Fall back to 1m if 5m not available
    candles = candles_5m if len(candles_5m) > 0 else candles_1m

    # Minimum candles required (25h warmup at 5m)
    min_candles = config.get('min_candle_buffer', 300)
    if len(candles) < min_candles:
        # REC-012: Add warmup progress indicator
        warmup_pct = (len(candles) / min_candles) * 100
        warmup_remaining = min_candles - len(candles)
        warmup_eta_minutes = warmup_remaining * 5  # 5m candles
        warmup_eta_hours = warmup_eta_minutes / 60

        state['indicators'] = build_base_indicators(
            symbol=symbol, candle_count=len(candles), status='warming_up', state=state
        )
        state['indicators']['required_candles'] = min_candles
        # REC-012: Warmup progress fields
        state['indicators']['warmup_pct'] = round(warmup_pct, 1)
        state['indicators']['warmup_remaining_candles'] = warmup_remaining
        state['indicators']['warmup_eta_minutes'] = warmup_eta_minutes
        state['indicators']['warmup_eta_hours'] = round(warmup_eta_hours, 1)
        if track_rejections:
            track_rejection(state, RejectionReason.INSUFFICIENT_CANDLES, symbol)
        return None

    # ==========================================================================
    # Session Awareness
    # ==========================================================================
    session = classify_trading_session(current_time, config)
    session_adjustments = get_session_adjustments(session, config)

    # ==========================================================================
    # Calculate Indicators
    # REC-021: RSI completely removed from strategy (v1.3.0)
    # ==========================================================================
    # Volume spike detection (whale proxy) - PRIMARY signal
    volume_window = get_symbol_config(symbol, config, 'volume_window') or config.get('volume_window', 288)
    spike_mult = get_symbol_config(symbol, config, 'volume_spike_mult') or config.get('volume_spike_mult', 2.0)
    volume_spike = detect_volume_spike(candles, volume_window, spike_mult)

    # Classify whale signal
    price_move_threshold = config.get('volume_spike_price_move_pct', 0.1)
    whale_signal = classify_whale_signal(volume_spike, candles, price_move_threshold)

    # Fear/greed price deviation - PRIMARY sentiment signal (per REC-021)
    price_lookback = config.get('price_lookback', 48)
    fear_dev = get_symbol_config(symbol, config, 'fear_deviation_pct') or config.get('fear_deviation_pct', -5.0)
    greed_dev = get_symbol_config(symbol, config, 'greed_deviation_pct') or config.get('greed_deviation_pct', 5.0)
    fear_greed = calculate_fear_greed_proxy(candles, price_lookback, fear_dev, greed_dev)

    # Sentiment zone classification - REC-021: Now uses ONLY price deviation
    sentiment_zone = classify_sentiment_zone(fear_greed, config)

    # REC-023: Calculate ATR for volatility regime
    atr_result = calculate_atr(candles, 14)
    atr_pct = atr_result.get('atr_pct')
    volatility_regime = classify_volatility_regime(atr_pct, config)
    volatility_adjustments = get_volatility_adjustments(volatility_regime, config)

    # REC-025: Check extended fear period
    extended_fear = check_extended_fear_period(state, sentiment_zone, current_time, config)

    # REC-021: RSI divergence deprecated - always returns 'none'
    divergence = detect_rsi_divergence([], [], 14)

    # ==========================================================================
    # Get Current Price
    # ==========================================================================
    current_price = data.prices.get(symbol, candles[-1].close if candles else 0)

    if current_price <= 0:
        state['indicators'] = build_base_indicators(
            symbol=symbol, candle_count=len(candles), status='no_price', state=state
        )
        if track_rejections:
            track_rejection(state, RejectionReason.NO_PRICE_DATA, symbol)
        return None

    # ==========================================================================
    # Validate Volume Spike (False Positive Filter)
    # ==========================================================================
    ob = data.orderbooks.get(symbol)
    spread_pct = ob.spread_pct if ob else 0.0
    trades = data.trades.get(symbol, ())
    trade_count = len(trades[-50:]) if trades else 0
    price_change_pct = (candles[-1].close - candles[-2].close) / candles[-2].close * 100 if len(candles) >= 2 else 0

    is_valid_spike, spike_rejection = validate_volume_spike(
        volume_spike.get('volume_ratio', 0),
        price_change_pct,
        spread_pct,
        trade_count,
        config
    )

    # ==========================================================================
    # Get Symbol-Specific Config
    # ==========================================================================
    base_position_size = get_symbol_config(symbol, config, 'position_size_usd') or config.get('position_size_usd', 25.0)
    tp_pct = get_symbol_config(symbol, config, 'take_profit_pct') or config.get('take_profit_pct', 5.0)
    sl_pct = get_symbol_config(symbol, config, 'stop_loss_pct') or config.get('stop_loss_pct', 2.5)
    short_mult = config.get('short_size_multiplier', 0.75)

    # Apply session adjustment
    adjusted_position_size = base_position_size * session_adjustments['size_mult']

    # Fee profitability check
    use_fee_check = config.get('use_fee_check', True)
    fee_rate = config.get('fee_rate', 0.001)
    min_profit_pct = config.get('min_profit_after_fees_pct', 0.1)
    is_fee_profitable, expected_profit = check_fee_profitability(tp_pct, fee_rate, min_profit_pct)

    # ==========================================================================
    # Build Comprehensive Indicators
    # REC-021: RSI removed; REC-023: ATR added for volatility regime
    # ==========================================================================
    state['indicators'] = {
        'symbol': symbol,
        'status': 'active',
        'candle_count': len(candles),
        'price': round(current_price, 6),
        # Volume Spike (PRIMARY signal)
        'volume_ratio': round(volume_spike.get('volume_ratio', 0), 2),
        'has_volume_spike': volume_spike.get('has_spike', False),
        'volume_spike_valid': is_valid_spike,
        'whale_signal': whale_signal.name.lower(),
        # Fear/Greed Price Deviation (PRIMARY sentiment per REC-021)
        'from_high_pct': round(fear_greed.get('from_high_pct', 0), 2),
        'from_low_pct': round(fear_greed.get('from_low_pct', 0), 2),
        # Sentiment
        'sentiment_zone': get_sentiment_string(sentiment_zone),
        'is_fear': is_fear_zone(sentiment_zone),
        'is_greed': is_greed_zone(sentiment_zone),
        # REC-023: Volatility Regime
        # REC-031: Added should_pause indicator
        'atr_pct': round(atr_pct, 3) if atr_pct else None,
        'volatility_regime': volatility_regime,
        'volatility_size_mult': round(volatility_adjustments['size_mult'], 2),
        'volatility_should_pause': volatility_adjustments.get('should_pause', False),
        # REC-025: Extended Fear Period
        'extended_fear_active': extended_fear['is_extended'],
        'hours_in_extreme': round(extended_fear['hours_in_extreme'], 1),
        'extended_fear_paused': extended_fear['should_pause'],
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
        sentiment_zone=sentiment_zone,
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
    # Sentiment Regime Check
    # ==========================================================================
    sentiment_adjustments = get_sentiment_regime_adjustments(sentiment_zone, config)
    if not sentiment_adjustments['allow_entry']:
        state['indicators']['status'] = 'neutral_sentiment'
        if track_rejections:
            track_rejection(state, RejectionReason.NEUTRAL_SENTIMENT, symbol)
        return None

    # ==========================================================================
    # REC-031: Extreme Volatility Check
    # ==========================================================================
    if volatility_adjustments.get('should_pause', False):
        state['indicators']['status'] = 'extreme_volatility_paused'
        state['indicators']['volatility_pause_reason'] = f"ATR {atr_pct:.2f}% exceeds extreme threshold"
        if track_rejections:
            track_rejection(state, RejectionReason.EXTREME_VOLATILITY, symbol)
        return None

    # ==========================================================================
    # REC-025: Extended Fear Period Check
    # ==========================================================================
    if extended_fear['should_pause']:
        state['indicators']['status'] = 'extended_fear_paused'
        state['indicators']['extended_fear_hours'] = round(extended_fear['hours_in_extreme'], 1)
        # Don't track as rejection - this is protective behavior
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
    is_contrarian = config.get('contrarian_mode', True)
    signal = None

    # Check for contrarian opportunity
    opportunity = is_contrarian_opportunity(sentiment_zone, whale_signal.name.lower(), config)

    if not opportunity['has_opportunity']:
        # No opportunity - check alignment
        if opportunity['alignment'] == 'conflicting':
            state['indicators']['status'] = 'whale_signal_mismatch'
            if track_rejections:
                track_rejection(state, RejectionReason.WHALE_SIGNAL_MISMATCH, symbol)
        else:
            state['indicators']['status'] = 'no_signal_conditions'
            if track_rejections:
                track_rejection(state, RejectionReason.NO_SIGNAL_CONDITIONS, symbol)
        return None

    direction = opportunity['direction']

    # ==========================================================================
    # Volume Spike Validation (for strong signals)
    # ==========================================================================
    if volume_spike.get('has_spike', False) and not is_valid_spike:
        state['indicators']['status'] = 'volume_false_positive'
        state['indicators']['spike_rejection_reason'] = spike_rejection
        if track_rejections:
            track_rejection(state, RejectionReason.VOLUME_FALSE_POSITIVE, symbol)
        return None

    # ==========================================================================
    # Trade Flow Confirmation
    # ==========================================================================
    use_trade_flow = config.get('use_trade_flow_confirmation', True)
    flow_confirmed = True
    flow_data = {}

    if use_trade_flow:
        flow_threshold = config.get('trade_flow_threshold', 0.10)
        flow_lookback = config.get('trade_flow_lookback', 50)
        flow_confirmed, flow_data = check_trade_flow_confirmation(
            trades, direction, flow_threshold, flow_lookback
        )
        # Add flow data to indicators
        state['indicators']['trade_flow_imbalance'] = round(flow_data.get('imbalance', 0), 3)
        state['indicators']['trade_flow_confirms'] = flow_confirmed
        # REC-018: Add trade flow expected indicator for clarity
        # In contrarian mode, opposite flow is actually expected (buy during selling pressure)
        if is_contrarian:
            state['indicators']['trade_flow_expected'] = 'negative' if direction == 'buy' else 'positive'
            state['indicators']['trade_flow_mode'] = 'contrarian'
        else:
            state['indicators']['trade_flow_expected'] = 'positive' if direction == 'buy' else 'negative'
            state['indicators']['trade_flow_mode'] = 'momentum'

    if not flow_confirmed:
        state['indicators']['status'] = 'trade_flow_against'
        if track_rejections:
            track_rejection(state, RejectionReason.TRADE_FLOW_AGAINST, symbol)
        return None

    # ==========================================================================
    # Real Correlation Check
    # ==========================================================================
    candles_by_symbol = {
        sym: data.candles_5m.get(sym, data.candles_1m.get(sym, ()))
        for sym in SYMBOLS
    }
    corr_allowed, corr_adj, corr_info = check_real_correlation(
        state, symbol, direction, candles_by_symbol, config
    )
    state['indicators']['real_correlation'] = corr_info.get('correlations', {})
    state['indicators']['correlation_blocked'] = corr_info.get('blocked', False)

    if not corr_allowed:
        state['indicators']['status'] = 'real_correlation_blocked'
        if track_rejections:
            track_rejection(state, RejectionReason.CORRELATION_LIMIT, symbol)
        return None

    # ==========================================================================
    # Calculate Composite Confidence
    # ==========================================================================
    confidence, reasons = calculate_composite_confidence(
        whale_signal=whale_signal,
        sentiment_zone=sentiment_zone,
        volume_ratio=volume_spike.get('volume_ratio', 0),
        trade_flow_imbalance=flow_data.get('imbalance', 0),
        divergence=divergence,
        is_contrarian=is_contrarian,
        direction=direction,
        config=config
    )

    state['indicators']['confidence'] = round(confidence, 3)
    state['indicators']['confidence_reasons'] = reasons[:3]

    # REC-027: Dynamic confidence threshold
    base_min_confidence = config.get('min_confidence', 0.50)
    min_confidence = calculate_dynamic_confidence_threshold(
        base_min_confidence, sentiment_zone, volatility_regime, config
    )
    state['indicators']['min_confidence'] = round(min_confidence, 3)
    state['indicators']['confidence_margin'] = round(confidence - min_confidence, 3)

    if confidence < min_confidence:
        state['indicators']['status'] = 'insufficient_confidence'
        if track_rejections:
            track_rejection(state, RejectionReason.INSUFFICIENT_CONFIDENCE, symbol)
        return None

    # ==========================================================================
    # Apply Size Adjustments
    # ==========================================================================
    # Sentiment-based size adjustment
    sentiment_size_mult = should_reduce_size_for_sentiment(sentiment_zone, direction, config)

    # REC-023: Volatility regime adjustment
    volatility_size_mult = volatility_adjustments['size_mult']

    # REC-025: Extended fear period adjustment
    extended_fear_size_mult = extended_fear['size_mult']

    # Correlation adjustment plus all regime adjustments
    adjusted_size = (available_size * corr_adj * sentiment_size_mult *
                     volatility_size_mult * extended_fear_size_mult)

    # Short size multiplier
    if direction == 'short':
        adjusted_size *= short_mult

    # Check correlation exposure limits
    can_enter, final_size = check_correlation_exposure(
        state, symbol, direction, adjusted_size, config
    )

    if not can_enter:
        state['indicators']['status'] = 'correlation_limit'
        if track_rejections:
            track_rejection(state, RejectionReason.CORRELATION_LIMIT, symbol)
        return None

    # ==========================================================================
    # Generate Entry Signal
    # ==========================================================================
    reason_str = ', '.join(reasons[:3])

    if direction == 'buy':
        signal = Signal(
            action='buy',
            symbol=symbol,
            size=final_size,
            price=current_price,
            reason=f"WS: Long ({reason_str})",
            stop_loss=current_price * (1 - sl_pct / 100),
            take_profit=current_price * (1 + tp_pct / 100),
            metadata={
                'entry_type': 'whale_sentiment_long',
                'sentiment_zone': get_sentiment_string(sentiment_zone),
                'whale_signal': whale_signal.name.lower(),
                'volume_ratio': volume_spike.get('volume_ratio'),
                'confidence': confidence,
                'trade_flow_imbalance': flow_data.get('imbalance'),
                'correlation_adjustment': corr_adj,
                'atr_pct': atr_pct,  # REC-023
                'volatility_regime': volatility_regime,  # REC-030: Fixed undefined function reference
            }
        )
    else:  # short
        signal = Signal(
            action='short',
            symbol=symbol,
            size=final_size,
            price=current_price,
            reason=f"WS: Short ({reason_str})",
            stop_loss=current_price * (1 + sl_pct / 100),
            take_profit=current_price * (1 - tp_pct / 100),
            metadata={
                'entry_type': 'whale_sentiment_short',
                'sentiment_zone': get_sentiment_string(sentiment_zone),
                'whale_signal': whale_signal.name.lower(),
                'volume_ratio': volume_spike.get('volume_ratio'),
                'confidence': confidence,
                'trade_flow_imbalance': flow_data.get('imbalance'),
                'correlation_adjustment': corr_adj,
                'atr_pct': atr_pct,  # REC-023
                'volatility_regime': volatility_regime,  # REC-030: Fixed undefined function reference
            }
        )

    # ==========================================================================
    # Return Signal
    # ==========================================================================
    if signal:
        state['last_signal_time'] = current_time
        state['indicators']['status'] = 'signal_generated'
        return signal

    state['indicators']['status'] = 'no_signal'
    return None
