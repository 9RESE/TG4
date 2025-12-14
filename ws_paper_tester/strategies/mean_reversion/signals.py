"""
Mean Reversion Strategy - Signal Generation

Contains the main signal generation logic and helper functions.
"""
from datetime import datetime
from typing import Dict, Any, Optional

from .config import (
    SYMBOLS, RejectionReason, VolatilityRegime,
    get_symbol_config,
)
from .indicators import (
    calculate_sma, calculate_rsi, calculate_bollinger_bands,
    calculate_volatility,
)
from .regimes import classify_volatility_regime, get_regime_adjustments
from .risk import (
    check_circuit_breaker, is_trade_flow_aligned, is_trending,
    get_xrp_btc_correlation, should_pause_for_low_correlation,
    check_fee_profitability, calculate_trailing_stop,
    update_position_extremes, get_decayed_take_profit,
    check_adx_strong_trend,  # REC-003 (v4.3.0)
)


# =============================================================================
# State Initialization
# =============================================================================
def initialize_state(state: Dict[str, Any]) -> None:
    """Initialize strategy state."""
    state['initialized'] = True
    state['last_signal_time'] = None
    state['position'] = 0.0
    state['position_by_symbol'] = {}
    state['indicators'] = {}

    # Per-pair tracking - REC-006
    state['pnl_by_symbol'] = {}
    state['trades_by_symbol'] = {}
    state['wins_by_symbol'] = {}
    state['losses_by_symbol'] = {}

    # Circuit breaker - REC-005
    state['consecutive_losses'] = 0
    state['circuit_breaker_time'] = None

    # Rejection tracking
    state['rejection_counts'] = {}
    state['rejection_counts_by_symbol'] = {}

    # Entry tracking
    state['position_entries'] = {}


# =============================================================================
# Rejection Tracking
# =============================================================================
def track_rejection(
    state: Dict[str, Any],
    reason: RejectionReason,
    symbol: str = None
) -> None:
    """Track signal rejection for analysis."""
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
    status: str,
    state: Dict[str, Any]
) -> Dict[str, Any]:
    """Build base indicators dict for early returns."""
    return {
        'symbol': symbol,
        'status': status,
        'position': state.get('position_by_symbol', {}).get(symbol, 0),
        'consecutive_losses': state.get('consecutive_losses', 0),
        'pnl_symbol': state.get('pnl_by_symbol', {}).get(symbol, 0),
        'trades_symbol': state.get('trades_by_symbol', {}).get(symbol, 0),
    }


# =============================================================================
# Main Signal Generation Function
# =============================================================================
def generate_signal(
    data,  # DataSnapshot
    config: Dict[str, Any],
    state: Dict[str, Any]
) -> Optional[Any]:  # Optional[Signal]
    """
    Generate mean reversion signal based on price deviation from moving average.

    Strategy:
    - Calculate SMA and deviation
    - Use RSI and Bollinger Bands for confirmation
    - Trade when price deviates significantly from mean
    - Use volatility regimes to adjust thresholds
    - Apply circuit breaker and cooldown mechanisms

    Args:
        data: Immutable market data snapshot
        config: Strategy configuration (CONFIG + overrides)
        state: Mutable strategy state dict

    Returns:
        Signal if trading opportunity found, None otherwise
    """
    from ws_tester.types import Signal

    if 'initialized' not in state:
        initialize_state(state)

    current_time = data.timestamp
    track_rejections = config.get('track_rejections', True)

    # REC-005: Circuit breaker check
    if config.get('use_circuit_breaker', True):
        max_losses = config.get('max_consecutive_losses', 3)
        cooldown_min = config.get('circuit_breaker_minutes', 15)

        if check_circuit_breaker(state, current_time, max_losses, cooldown_min):
            state['indicators'] = build_base_indicators(
                symbol='N/A', status='circuit_breaker', state=state
            )
            state['indicators']['circuit_breaker_active'] = True
            if track_rejections:
                track_rejection(state, RejectionReason.CIRCUIT_BREAKER)
            return None

    # REC-003: Time-based cooldown check
    if state['last_signal_time'] is not None:
        elapsed = (current_time - state['last_signal_time']).total_seconds()
        global_cooldown = config.get('cooldown_seconds', 10.0)
        if elapsed < global_cooldown:
            state['indicators'] = build_base_indicators(
                symbol='N/A', status='cooldown', state=state
            )
            state['indicators']['cooldown_remaining'] = global_cooldown - elapsed
            if track_rejections:
                track_rejection(state, RejectionReason.TIME_COOLDOWN)
            return None

    # Evaluate each symbol
    for symbol in SYMBOLS:
        signal = _evaluate_symbol(data, config, state, symbol, current_time, Signal)
        if signal:
            state['last_signal_time'] = current_time
            return signal

    return None


def _evaluate_symbol(
    data,  # DataSnapshot
    config: Dict[str, Any],
    state: Dict[str, Any],
    symbol: str,
    current_time: datetime,
    Signal
) -> Optional[Any]:
    """
    Evaluate a single symbol for mean reversion opportunity.

    Refactored in v3.0.0 to integrate:
    - REC-004: Trend filter
    - REC-006: Trailing stops
    - REC-007: Position decay
    - Finding #4: Extracted signal logic into helper functions
    """
    track_rejections = config.get('track_rejections', True)

    # Get candles
    candles_5m = data.candles_5m.get(symbol, ())
    lookback = config.get('lookback_candles', 20)

    if len(candles_5m) < lookback:
        state['indicators'] = build_base_indicators(
            symbol=symbol, status='warming_up', state=state
        )
        state['indicators']['candles_available'] = len(candles_5m)
        state['indicators']['candles_required'] = lookback
        if track_rejections:
            track_rejection(state, RejectionReason.WARMING_UP, symbol)
        return None

    current_price = data.prices.get(symbol, 0)
    if not current_price:
        state['indicators'] = build_base_indicators(
            symbol=symbol, status='no_price', state=state
        )
        if track_rejections:
            track_rejection(state, RejectionReason.NO_PRICE_DATA, symbol)
        return None

    # Calculate indicators
    candles_list = list(candles_5m)
    sma = calculate_sma(candles_list, lookback)
    rsi = calculate_rsi(candles_list, config.get('rsi_period', 14))

    bb_period = config.get('bb_period', 20)
    bb_std = config.get('bb_std_dev', 2.0)
    bb_lower, bb_mid, bb_upper = calculate_bollinger_bands(candles_list, bb_period, bb_std)

    if not sma or not bb_lower:
        return None

    # Calculate volatility and classify regime
    candles_1m = data.candles_1m.get(symbol, candles_5m)
    volatility = calculate_volatility(list(candles_1m), config.get('volatility_lookback', 20))

    regime = VolatilityRegime.MEDIUM
    regime_adjustments = {'threshold_mult': 1.0, 'size_mult': 1.0, 'pause_trading': False}

    if config.get('use_volatility_regimes', True):
        regime = classify_volatility_regime(volatility, config)
        regime_adjustments = get_regime_adjustments(regime, config)

        if regime_adjustments['pause_trading']:
            state['indicators'] = build_base_indicators(
                symbol=symbol, status='regime_pause', state=state
            )
            state['indicators']['volatility_regime'] = regime.name
            state['indicators']['volatility_pct'] = round(volatility, 4)
            if track_rejections:
                track_rejection(state, RejectionReason.REGIME_PAUSE, symbol)
            return None

    # REC-004 (v3.0.0): Trend filter check with confirmation (REC-003 v4.0)
    trend_slope = 0.0
    is_market_trending = False
    trend_confirmation_count = 0
    if config.get('use_trend_filter', True):
        is_market_trending, trend_slope, trend_confirmation_count = is_trending(
            candles_list, config, state, symbol
        )

    # Calculate VWAP
    vwap = data.get_vwap(symbol, config.get('vwap_lookback', 50))

    # Calculate deviation from SMA
    deviation_pct = ((current_price - sma) / sma) * 100

    # Get symbol-specific config
    base_deviation_threshold = get_symbol_config(symbol, config, 'deviation_threshold')
    rsi_oversold = get_symbol_config(symbol, config, 'rsi_oversold')
    rsi_overbought = get_symbol_config(symbol, config, 'rsi_overbought')
    base_position_size = get_symbol_config(symbol, config, 'position_size_usd')
    max_position = get_symbol_config(symbol, config, 'max_position')
    tp_pct = get_symbol_config(symbol, config, 'take_profit_pct')
    sl_pct = get_symbol_config(symbol, config, 'stop_loss_pct')

    # Apply regime adjustments
    effective_deviation_threshold = base_deviation_threshold * regime_adjustments['threshold_mult']
    adjusted_position_size = base_position_size * regime_adjustments['size_mult']

    # Get trade flow for confirmation
    use_trade_flow = config.get('use_trade_flow_confirmation', True)
    trade_flow_threshold = config.get('trade_flow_threshold', 0.10)
    trade_flow = data.get_trade_imbalance(symbol, 50)

    # Get current position for this symbol (moved up for use in filters)
    current_position = state.get('position_by_symbol', {}).get(symbol, 0)

    # REC-005 (v4.0.0): XRP/BTC correlation monitoring
    xrp_btc_correlation = None
    if symbol == 'XRP/BTC' and config.get('use_correlation_monitoring', True):
        xrp_btc_correlation = get_xrp_btc_correlation(data, config, state)

        # REC-001/002 (v4.3.0): Check correlation pause threshold for XRP/BTC
        # Raised from 0.25 to 0.5 per Deep Review v8.0 CRITICAL-001
        if should_pause_for_low_correlation(symbol, state, config):
            state['indicators'] = build_base_indicators(
                symbol=symbol, status='low_correlation_pause', state=state
            )
            state['indicators']['xrp_btc_correlation'] = round(xrp_btc_correlation, 4) if xrp_btc_correlation else None
            state['indicators']['correlation_pause_threshold'] = config.get('correlation_pause_threshold', 0.5)
            if track_rejections:
                track_rejection(state, RejectionReason.LOW_CORRELATION, symbol)
            return None

    # ==========================================================================
    # REC-003 (v4.3.0): ADX filter for BTC - Deep Review v8.0 HIGH-001
    # Research: BTC exhibits stronger trending behavior than mean reversion
    # Pause BTC entries when ADX > 25 indicates strong trend
    # ==========================================================================
    is_adx_strong_trend = False
    adx_value = None
    if config.get('use_adx_filter', True):
        is_adx_strong_trend, adx_value = check_adx_strong_trend(
            candles_list, symbol, config, state
        )

        if is_adx_strong_trend and current_position == 0:
            state['indicators'] = build_base_indicators(
                symbol=symbol, status='strong_trend_adx', state=state
            )
            state['indicators']['adx'] = round(adx_value, 2) if adx_value else None
            state['indicators']['adx_threshold'] = config.get('adx_strong_trend_threshold', 25)
            if track_rejections:
                track_rejection(state, RejectionReason.STRONG_TREND_ADX, symbol)
            return None

    # REC-006 (v3.0.0): Update position extremes for trailing stop
    update_position_extremes(state, symbol, current_price)

    # Store indicators (including v3.0.0 additions)
    state['indicators'] = {
        'symbol': symbol,
        'status': 'active',
        'sma': round(sma, 8),
        'rsi': round(rsi, 2),
        'deviation_pct': round(deviation_pct, 4),
        'bb_lower': round(bb_lower, 8),
        'bb_mid': round(bb_mid, 8),
        'bb_upper': round(bb_upper, 8),
        'vwap': round(vwap, 8) if vwap else None,
        'price': round(current_price, 8),
        'position': round(current_position, 2),
        'max_position': max_position,
        'volatility_pct': round(volatility, 4),
        'volatility_regime': regime.name,
        'regime_threshold_mult': round(regime_adjustments['threshold_mult'], 2),
        'regime_size_mult': round(regime_adjustments['size_mult'], 2),
        'base_deviation_threshold': round(base_deviation_threshold, 4),
        'effective_deviation_threshold': round(effective_deviation_threshold, 4),
        'trade_flow': round(trade_flow, 4),
        'trade_flow_threshold': round(trade_flow_threshold, 4),
        'consecutive_losses': state.get('consecutive_losses', 0),
        'pnl_symbol': round(state.get('pnl_by_symbol', {}).get(symbol, 0), 4),
        'trades_symbol': state.get('trades_by_symbol', {}).get(symbol, 0),
        # v3.0.0 indicators
        'trend_slope': round(trend_slope, 4),
        'is_trending': is_market_trending,
        'use_trend_filter': config.get('use_trend_filter', True),
        # v4.0.0 indicators - REC-003 trend confirmation
        'trend_confirmation_count': trend_confirmation_count,
        'trend_confirmation_required': config.get('trend_confirmation_periods', 3),
        # v4.0.0 indicators - REC-005 XRP/BTC correlation (only for XRP/BTC)
        'xrp_btc_correlation': round(xrp_btc_correlation, 4) if xrp_btc_correlation else None,
        # v4.2.0 indicators - REC-001 correlation pause thresholds
        'correlation_warn_threshold': config.get('correlation_warn_threshold', 0.55),
        'correlation_pause_threshold': config.get('correlation_pause_threshold', 0.5),
        'correlation_pause_enabled': config.get('correlation_pause_enabled', True),
        # v4.3.0 indicators - REC-003 ADX filter
        'adx': round(adx_value, 2) if adx_value else None,
        'adx_threshold': config.get('adx_strong_trend_threshold', 25),
        'use_adx_filter': config.get('use_adx_filter', True),
        'is_adx_strong_trend': is_adx_strong_trend,
    }

    # ==========================================================================
    # REC-006 (v3.0.0): Check for trailing stop exit on existing positions
    # ==========================================================================
    if current_position != 0 and config.get('use_trailing_stop', True):
        trailing_signal = _check_trailing_stop_exit(
            state, symbol, current_price, current_position, config, Signal
        )
        if trailing_signal:
            state['indicators']['status'] = 'trailing_stop_exit'
            return trailing_signal

    # ==========================================================================
    # REC-007 (v3.0.0): Check for position decay exit on stale positions
    # ==========================================================================
    if current_position != 0 and config.get('use_position_decay', True):
        decay_signal = _check_position_decay_exit(
            state, symbol, current_price, current_position, current_time, config, Signal
        )
        if decay_signal:
            state['indicators']['status'] = 'position_decay_exit'
            return decay_signal

    # ==========================================================================
    # REC-004 (v3.0.0): Skip new entries in trending markets
    # ==========================================================================
    if is_market_trending and current_position == 0:
        state['indicators']['status'] = 'trending_market'
        if track_rejections:
            track_rejection(state, RejectionReason.TRENDING_MARKET, symbol)
        return None

    # Position limit check
    if current_position >= max_position:
        state['indicators']['status'] = 'max_position_reached'
        if track_rejections:
            track_rejection(state, RejectionReason.MAX_POSITION, symbol)
        return None

    available = max_position - current_position
    actual_size = min(adjusted_position_size, available)

    min_trade_size = config.get('min_trade_size_usd', 5.0)
    if actual_size < min_trade_size:
        state['indicators']['status'] = 'insufficient_size'
        if track_rejections:
            track_rejection(state, RejectionReason.INSUFFICIENT_SIZE, symbol)
        return None

    # ==========================================================================
    # Signal Logic: Use extracted helper functions (Finding #4)
    # ==========================================================================
    signal = _generate_entry_signal(
        data=data,
        config=config,
        state=state,
        symbol=symbol,
        current_price=current_price,
        deviation_pct=deviation_pct,
        effective_deviation_threshold=effective_deviation_threshold,
        rsi=rsi,
        rsi_oversold=rsi_oversold,
        rsi_overbought=rsi_overbought,
        bb_lower=bb_lower,
        bb_upper=bb_upper,
        current_position=current_position,
        max_position=max_position,
        actual_size=actual_size,
        min_trade_size=min_trade_size,
        tp_pct=tp_pct,
        sl_pct=sl_pct,
        vwap=vwap,
        regime=regime,
        use_trade_flow=use_trade_flow,
        trade_flow_threshold=trade_flow_threshold,
        track_rejections=track_rejections,
        Signal=Signal,
    )

    if signal:
        state['indicators']['status'] = 'signal_generated'
        return signal

    state['indicators']['status'] = 'no_signal'
    if track_rejections:
        track_rejection(state, RejectionReason.NO_SIGNAL_CONDITIONS, symbol)
    return None


# =============================================================================
# Signal Generation Helpers (Finding #4 - v3.0.0)
# =============================================================================
def _check_trailing_stop_exit(
    state: Dict[str, Any],
    symbol: str,
    current_price: float,
    current_position: float,
    config: Dict[str, Any],
    Signal
) -> Optional[Any]:
    """Check if trailing stop should trigger exit."""
    entry = state.get('position_entries', {}).get(symbol)
    if not entry:
        return None

    entry_price = entry.get('entry_price', 0)
    highest_price = entry.get('highest_price', entry_price)
    lowest_price = entry.get('lowest_price', entry_price)
    side = entry.get('side', 'long')

    activation_pct = config.get('trailing_activation_pct', 0.3)
    trail_distance_pct = config.get('trailing_distance_pct', 0.2)

    trailing_stop = calculate_trailing_stop(
        entry_price, highest_price, lowest_price, side,
        activation_pct, trail_distance_pct
    )

    if trailing_stop is None:
        return None

    # Check if trailing stop is hit
    if side == 'long' and current_price <= trailing_stop:
        return Signal(
            action='sell',
            symbol=symbol,
            size=abs(current_position),
            price=current_price,
            reason=f"MR: Trailing stop (high={highest_price:.4f}, stop={trailing_stop:.4f})",
        )
    elif side == 'short' and current_price >= trailing_stop:
        return Signal(
            action='cover',
            symbol=symbol,
            size=abs(current_position),
            price=current_price,
            reason=f"MR: Trailing stop (low={lowest_price:.4f}, stop={trailing_stop:.4f})",
        )

    return None


def _check_position_decay_exit(
    state: Dict[str, Any],
    symbol: str,
    current_price: float,
    current_position: float,
    current_time: datetime,
    config: Dict[str, Any],
    Signal
) -> Optional[Any]:
    """Check if position decay should trigger exit."""
    entry = state.get('position_entries', {}).get(symbol)
    if not entry:
        return None

    entry_price = entry.get('entry_price', 0)
    entry_time = entry.get('entry_time')
    side = entry.get('side', 'long')

    if not entry_time or not entry_price:
        return None

    tp_pct = get_symbol_config(symbol, config, 'take_profit_pct')
    decayed_tp, decay_mult = get_decayed_take_profit(
        entry_price, tp_pct, entry_time, current_time, side, config
    )

    # Store decay info in indicators
    state['indicators']['decay_multiplier'] = round(decay_mult, 2)
    state['indicators']['decayed_tp'] = round(decayed_tp, 8)

    # Check if decayed TP is reached
    if side == 'long' and current_price >= decayed_tp and decay_mult < 1.0:
        profit_pct = ((current_price - entry_price) / entry_price) * 100
        return Signal(
            action='sell',
            symbol=symbol,
            size=abs(current_position),
            price=current_price,
            reason=f"MR: Position decay exit (decay={decay_mult:.0%}, profit={profit_pct:.2f}%)",
        )
    elif side == 'short' and current_price <= decayed_tp and decay_mult < 1.0:
        profit_pct = ((entry_price - current_price) / entry_price) * 100
        return Signal(
            action='cover',
            symbol=symbol,
            size=abs(current_position),
            price=current_price,
            reason=f"MR: Position decay exit (decay={decay_mult:.0%}, profit={profit_pct:.2f}%)",
        )

    return None


def _generate_entry_signal(
    data,  # DataSnapshot
    config: Dict[str, Any],
    state: Dict[str, Any],
    symbol: str,
    current_price: float,
    deviation_pct: float,
    effective_deviation_threshold: float,
    rsi: float,
    rsi_oversold: float,
    rsi_overbought: float,
    bb_lower: float,
    bb_upper: float,
    current_position: float,
    max_position: float,
    actual_size: float,
    min_trade_size: float,
    tp_pct: float,
    sl_pct: float,
    vwap: Optional[float],
    regime: VolatilityRegime,
    use_trade_flow: bool,
    trade_flow_threshold: float,
    track_rejections: bool,
    Signal
) -> Optional[Any]:
    """
    Generate entry signal based on mean reversion conditions.

    Finding #4: Extracted from _evaluate_symbol to reduce complexity.
    REC-002 (v4.1.0): Added fee profitability check before signal generation.
    """
    signal = None

    # ==========================================================================
    # REC-002 (v4.1.0): Fee Profitability Check - Guide v2.0 Section 23
    # Validate that expected profit exceeds round-trip fees before signal
    # ==========================================================================
    is_fee_profitable, net_profit_pct = check_fee_profitability(tp_pct, config)
    if not is_fee_profitable:
        state['indicators']['status'] = 'fee_unprofitable'
        state['indicators']['expected_tp_pct'] = round(tp_pct, 4)
        state['indicators']['net_profit_pct'] = round(net_profit_pct, 4)
        if track_rejections:
            track_rejection(state, RejectionReason.FEE_UNPROFITABLE, symbol)
        return None

    # Store fee info in indicators for monitoring
    state['indicators']['net_profit_pct'] = round(net_profit_pct, 4)

    # ==========================================================================
    # Signal Logic: Oversold - Buy Signal
    # ==========================================================================
    if (deviation_pct < -effective_deviation_threshold and
        rsi < rsi_oversold and
        current_position < max_position):

        # Extra confirmation: price near or below lower BB
        if current_price <= bb_lower * 1.005:
            # Trade flow confirmation
            if use_trade_flow:
                if not is_trade_flow_aligned(data, symbol, 'buy', trade_flow_threshold):
                    state['indicators']['status'] = 'trade_flow_not_aligned'
                    state['indicators']['trade_flow_aligned'] = False
                    if track_rejections:
                        track_rejection(state, RejectionReason.TRADE_FLOW_NOT_ALIGNED, symbol)
                    return None

            state['indicators']['trade_flow_aligned'] = True

            signal = Signal(
                action='buy',
                symbol=symbol,
                size=actual_size,
                price=current_price,
                reason=f"MR: Oversold (dev={deviation_pct:.2f}%, RSI={rsi:.1f}, regime={regime.name})",
                stop_loss=current_price * (1 - sl_pct / 100),
                take_profit=current_price * (1 + tp_pct / 100),
            )
            return signal

    # ==========================================================================
    # Signal Logic: Overbought - Sell/Short Signal
    # ==========================================================================
    if (deviation_pct > effective_deviation_threshold and
        rsi > rsi_overbought and
        current_position > -max_position):

        # Extra confirmation: price near or above upper BB
        if current_price >= bb_upper * 0.995:
            # Trade flow confirmation
            if use_trade_flow:
                if not is_trade_flow_aligned(data, symbol, 'sell', trade_flow_threshold):
                    state['indicators']['status'] = 'trade_flow_not_aligned'
                    state['indicators']['trade_flow_aligned'] = False
                    if track_rejections:
                        track_rejection(state, RejectionReason.TRADE_FLOW_NOT_ALIGNED, symbol)
                    return None

            state['indicators']['trade_flow_aligned'] = True

            if current_position > 0:
                # We have a long position - sell to reduce/close
                close_size = min(actual_size, current_position)
                signal = Signal(
                    action='sell',
                    symbol=symbol,
                    size=close_size,
                    price=current_price,
                    reason=f"MR: Close long - overbought (dev={deviation_pct:.2f}%, RSI={rsi:.1f})",
                )
            else:
                # We're flat or short - open/add to short position
                signal = Signal(
                    action='short',
                    symbol=symbol,
                    size=actual_size,
                    price=current_price,
                    reason=f"MR: Short - overbought (dev={deviation_pct:.2f}%, RSI={rsi:.1f}, regime={regime.name})",
                    stop_loss=current_price * (1 + sl_pct / 100),
                    take_profit=current_price * (1 - tp_pct / 100),
                )
            return signal

    # ==========================================================================
    # Signal Logic: VWAP Reversion
    # ==========================================================================
    if vwap:
        vwap_deviation = ((current_price - vwap) / vwap) * 100
        vwap_threshold = config.get('vwap_deviation_threshold', 0.3)
        vwap_size_mult = config.get('vwap_size_multiplier', 0.5)

        # Price significantly below VWAP with neutral RSI
        if (vwap_deviation < -vwap_threshold and
            40 < rsi < 60 and
            current_position < max_position):

            vwap_size = actual_size * vwap_size_mult
            if vwap_size >= min_trade_size:
                signal = Signal(
                    action='buy',
                    symbol=symbol,
                    size=vwap_size,
                    price=current_price,
                    reason=f"MR: VWAP reversion (vwap_dev={vwap_deviation:.2f}%)",
                    stop_loss=current_price * (1 - sl_pct / 100),
                    take_profit=vwap,
                )
                return signal

    return None
