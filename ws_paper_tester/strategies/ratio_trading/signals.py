"""
Ratio Trading Strategy - Signal Generation Module

Main generate_signal function and signal generation helpers.
"""
from typing import Dict, Any, Optional

from .config import SYMBOLS
from .enums import VolatilityRegime, RejectionReason, ExitReason
from .indicators import (
    calculate_bollinger_bands,
    calculate_z_score,
    calculate_volatility,
    calculate_rsi,
    detect_trend_strength,
    calculate_trailing_stop,
    check_position_decay,
    calculate_rolling_correlation,
    calculate_correlation_trend,
    get_btc_price_usd,
)
from .regimes import classify_volatility_regime, get_regime_adjustments
from .risk import (
    check_circuit_breaker,
    is_trade_flow_aligned,
    check_spread,
    calculate_position_size,
    check_fee_profitability,
)
from .tracking import (
    track_rejection,
    track_exit,
    build_base_indicators,
    initialize_state,
    update_price_history,
)


def generate_buy_signal(
    symbol: str,
    price: float,
    size_usd: float,
    z_score: float,
    entry_threshold: float,
    sl_pct: float,
    tp_pct: float,
    regime_name: str,
    Signal
):
    """Generate a buy signal."""
    return Signal(
        action='buy',
        symbol=symbol,
        size=size_usd,
        price=price,
        reason=f"RT: Buy XRP (z={z_score:.2f}, below {-entry_threshold:.1f}σ, regime={regime_name})",
        stop_loss=price * (1 - sl_pct / 100),
        take_profit=price * (1 + tp_pct / 100),
        metadata={
            'strategy': 'ratio_trading',
            'signal_type': 'entry',
            'z_score': round(z_score, 3),
            'regime': regime_name,
        }
    )


def generate_sell_signal(
    symbol: str,
    price: float,
    size_usd: float,
    z_score: float,
    entry_threshold: float,
    sl_pct: float,
    tp_pct: float,
    regime_name: str,
    Signal
):
    """Generate a sell signal."""
    return Signal(
        action='sell',
        symbol=symbol,
        size=size_usd,
        price=price,
        reason=f"RT: Sell XRP (z={z_score:.2f}, above {entry_threshold:.1f}σ, regime={regime_name})",
        stop_loss=price * (1 + sl_pct / 100),
        take_profit=price * (1 - tp_pct / 100),
        metadata={
            'strategy': 'ratio_trading',
            'signal_type': 'entry',
            'z_score': round(z_score, 3),
            'regime': regime_name,
        }
    )


def generate_exit_signal(
    symbol: str,
    price: float,
    size_usd: float,
    z_score: float,
    exit_threshold: float,
    Signal
):
    """Generate an exit/take profit signal."""
    return Signal(
        action='sell',
        symbol=symbol,
        size=size_usd,
        price=price,
        reason=f"RT: Take profit (z={z_score:.2f}, near mean, |z|<{exit_threshold:.1f})",
        metadata={
            'strategy': 'ratio_trading',
            'signal_type': 'exit',
            'z_score': round(z_score, 3),
        }
    )


def generate_signal(
    data,
    config: Dict[str, Any],
    state: Dict[str, Any]
) -> Optional[Any]:
    """
    Generate ratio trading signal based on mean reversion.

    Strategy:
    - Track XRP/BTC price history
    - Calculate Bollinger Bands
    - Buy XRP when ratio below lower band (XRP cheap)
    - Sell XRP when ratio above upper band (XRP expensive)
    - Goal: Accumulate both XRP and BTC over time

    Args:
        data: Immutable market data snapshot
        config: Strategy configuration (CONFIG + overrides)
        state: Mutable strategy state dict

    Returns:
        Signal if trading opportunity found, None otherwise
    """
    from ws_tester.types import Signal

    # Lazy initialization
    if 'initialized' not in state:
        initialize_state(state)

    symbol = SYMBOLS[0]  # XRP/BTC
    current_time = data.timestamp
    track_rejections = config.get('track_rejections', True)

    # REC-005: Circuit breaker check
    if config.get('use_circuit_breaker', True):
        max_losses = config.get('max_consecutive_losses', 3)
        cooldown_min = config.get('circuit_breaker_minutes', 15)

        if check_circuit_breaker(state, current_time, max_losses, cooldown_min):
            state['indicators'] = build_base_indicators(
                symbol=symbol, status='circuit_breaker', state=state
            )
            state['indicators']['circuit_breaker_active'] = True
            if track_rejections:
                track_rejection(state, RejectionReason.CIRCUIT_BREAKER, symbol)
            return None

    # Time-based cooldown
    if state['last_signal_time'] is not None:
        elapsed = (current_time - state['last_signal_time']).total_seconds()
        cooldown = config.get('cooldown_seconds', 30.0)
        if elapsed < cooldown:
            state['indicators'] = build_base_indicators(
                symbol=symbol, status='cooldown', state=state
            )
            state['indicators']['cooldown_remaining'] = round(cooldown - elapsed, 1)
            if track_rejections:
                track_rejection(state, RejectionReason.TIME_COOLDOWN, symbol)
            return None

    # Get current price
    price = data.prices.get(symbol)
    if not price:
        state['indicators'] = build_base_indicators(
            symbol=symbol, status='no_price', state=state
        )
        if track_rejections:
            track_rejection(state, RejectionReason.NO_PRICE_DATA, symbol)
        return None

    # Update price history
    price_history = update_price_history(data, state, symbol, price)

    # REC-018: Get dynamic BTC price for USD conversion
    btc_price_usd = get_btc_price_usd(data, config)
    state['last_btc_price_usd'] = btc_price_usd

    # REC-021: Update BTC price history for correlation monitoring
    if config.get('use_correlation_monitoring', True):
        if 'btc_price_history' not in state:
            state['btc_price_history'] = []
        state['btc_price_history'].append(btc_price_usd)
        state['btc_price_history'] = state['btc_price_history'][-50:]  # Keep last 50

    # Check minimum candles
    min_candles = config.get('min_candles', 10)
    if len(price_history) < min_candles:
        state['indicators'] = build_base_indicators(
            symbol=symbol, status='warming_up', state=state, price=price
        )
        state['indicators']['candles_available'] = len(price_history)
        state['indicators']['candles_required'] = min_candles
        if track_rejections:
            track_rejection(state, RejectionReason.WARMING_UP, symbol)
        return None

    # REC-008: Spread monitoring
    tp_pct = config.get('take_profit_pct', 0.6)
    if config.get('use_spread_filter', True):
        max_spread = config.get('max_spread_pct', 0.10)
        min_profit_mult = config.get('min_profitability_mult', 0.5)

        spread_ok, current_spread = check_spread(data, symbol, max_spread, tp_pct, min_profit_mult)
        if not spread_ok:
            state['indicators'] = build_base_indicators(
                symbol=symbol, status='spread_too_wide', state=state, price=price
            )
            state['indicators']['current_spread_pct'] = round(current_spread, 4)
            state['indicators']['max_spread_pct'] = max_spread
            if track_rejections:
                track_rejection(state, RejectionReason.SPREAD_TOO_WIDE, symbol)
            return None
    else:
        current_spread = 0.0

    # REC-050: Explicit fee profitability check (v4.3.0)
    use_fee_check = config.get('use_fee_profitability_check', True)
    fee_rate = config.get('estimated_fee_rate', 0.0026)
    min_net_profit = config.get('min_net_profit_pct', 0.10)
    fee_profitable = True
    net_profit_pct = tp_pct

    if use_fee_check:
        fee_profitable, net_profit_pct = check_fee_profitability(
            tp_pct, fee_rate, min_net_profit
        )
        if not fee_profitable:
            state['indicators'] = build_base_indicators(
                symbol=symbol, status='fee_not_profitable', state=state, price=price
            )
            state['indicators']['take_profit_pct'] = round(tp_pct, 4)
            state['indicators']['round_trip_fee_pct'] = round(fee_rate * 2 * 100, 4)
            state['indicators']['net_profit_pct'] = round(net_profit_pct, 4)
            state['indicators']['min_net_profit_pct'] = round(min_net_profit, 4)
            if track_rejections:
                track_rejection(state, RejectionReason.FEE_NOT_PROFITABLE, symbol)
            return None

    # Calculate Bollinger Bands
    # REC-036: Optionally use wider bands for crypto volatility
    lookback = config.get('lookback_periods', 20)
    if config.get('use_crypto_bollinger_std', False):
        num_std = config.get('bollinger_std_crypto', 2.5)
    else:
        num_std = config.get('bollinger_std', 2.0)

    sma, upper, lower, std_dev = calculate_bollinger_bands(
        price_history,
        lookback,
        num_std
    )

    if sma is None:
        return None

    # Calculate z-score
    z_score = calculate_z_score(price, sma, std_dev)

    # REC-004: Calculate volatility and classify regime
    volatility = calculate_volatility(price_history, config.get('volatility_lookback', 20))
    regime = VolatilityRegime.MEDIUM
    regime_adjustments = {'threshold_mult': 1.0, 'size_mult': 1.0, 'pause_trading': False}

    if config.get('use_volatility_regimes', True):
        regime = classify_volatility_regime(volatility, config)
        regime_adjustments = get_regime_adjustments(regime, config)

        if regime_adjustments['pause_trading']:
            state['indicators'] = build_base_indicators(
                symbol=symbol, status='regime_pause', state=state, price=price
            )
            state['indicators']['volatility_regime'] = regime.name
            state['indicators']['volatility_pct'] = round(volatility, 4)
            if track_rejections:
                track_rejection(state, RejectionReason.REGIME_PAUSE, symbol)
            return None

    # REC-021: Correlation monitoring
    correlation = 0.0
    use_correlation = config.get('use_correlation_monitoring', True)
    if use_correlation:
        corr_lookback = config.get('correlation_lookback', 20)
        btc_history = state.get('btc_price_history', [])

        if len(price_history) >= corr_lookback and len(btc_history) >= corr_lookback:
            correlation = calculate_rolling_correlation(
                price_history, btc_history, corr_lookback
            )

            # Store correlation history
            if 'correlation_history' not in state:
                state['correlation_history'] = []
            state['correlation_history'].append(correlation)
            state['correlation_history'] = state['correlation_history'][-20:]  # Keep last 20

            # Check correlation thresholds
            warn_threshold = config.get('correlation_warning_threshold', 0.5)
            pause_threshold = config.get('correlation_pause_threshold', 0.3)
            pause_enabled = config.get('correlation_pause_enabled', False)

            # Warning if correlation is low
            if correlation < warn_threshold:
                state['correlation_warnings'] = state.get('correlation_warnings', 0) + 1

            # Pause trading if enabled and correlation is very low
            if pause_enabled and correlation < pause_threshold and correlation != 0.0:
                state['indicators'] = build_base_indicators(
                    symbol=symbol, status='correlation_pause', state=state, price=price
                )
                state['indicators']['correlation'] = round(correlation, 4)
                state['indicators']['correlation_threshold'] = pause_threshold
                if track_rejections:
                    track_rejection(state, RejectionReason.CORRELATION_TOO_LOW, symbol)
                return None

            # REC-037: Correlation trend detection
            use_trend_detection = config.get('use_correlation_trend_detection', True)
            if use_trend_detection and len(state.get('correlation_history', [])) >= 5:
                trend_lookback = config.get('correlation_trend_lookback', 10)
                trend_threshold = config.get('correlation_trend_threshold', -0.02)
                trend_level = config.get('correlation_trend_level', 0.7)
                trend_pause_enabled = config.get('correlation_trend_pause_enabled', False)

                corr_slope, is_declining, corr_trend_direction = calculate_correlation_trend(
                    state.get('correlation_history', []), trend_lookback
                )

                # Store trend info in state
                state['correlation_slope'] = corr_slope
                state['correlation_trend_direction'] = corr_trend_direction

                # Track declining trend warnings
                if is_declining and corr_slope < trend_threshold and correlation < trend_level:
                    state['correlation_trend_warnings'] = state.get('correlation_trend_warnings', 0) + 1

                    # Optional: Pause on declining trend (conservative mode)
                    if trend_pause_enabled:
                        state['indicators'] = build_base_indicators(
                            symbol=symbol, status='correlation_trend_pause', state=state, price=price
                        )
                        state['indicators']['correlation'] = round(correlation, 4)
                        state['indicators']['correlation_slope'] = round(corr_slope, 6)
                        state['indicators']['correlation_trend'] = corr_trend_direction
                        if track_rejections:
                            track_rejection(state, RejectionReason.CORRELATION_DECLINING, symbol)
                        return None

    # Entry/exit thresholds with regime adjustment
    base_entry_threshold = config.get('entry_threshold', 1.5)  # REC-013: Higher default
    effective_entry_threshold = base_entry_threshold * regime_adjustments['threshold_mult']
    exit_threshold = config.get('exit_threshold', 0.5)

    # REC-002: Position sizing in USD
    current_position_usd = state.get('position_usd', 0)
    actual_size_usd, available_usd = calculate_position_size(
        config, current_position_usd, regime_adjustments['size_mult']
    )

    # Calculate band widths for indicators
    band_width = (upper - lower) / sma * 100 if sma else 0  # As percentage

    # Risk management percentages - REC-003
    sl_pct = config.get('stop_loss_pct', 0.6)

    # REC-014: Calculate RSI if enabled
    use_rsi = config.get('use_rsi_confirmation', True)  # Default matches CONFIG
    rsi_period = config.get('rsi_period', 14)
    rsi_oversold = config.get('rsi_oversold', 35)
    rsi_overbought = config.get('rsi_overbought', 65)
    rsi = calculate_rsi(price_history, rsi_period) if use_rsi else 50.0

    # REC-015: Detect trend strength if enabled
    use_trend_filter = config.get('use_trend_filter', True)  # Default matches CONFIG
    trend_lookback = config.get('trend_lookback', 10)
    trend_threshold = config.get('trend_strength_threshold', 0.7)
    is_strong_trend, trend_direction, trend_strength = detect_trend_strength(
        price_history, trend_lookback, trend_threshold
    ) if use_trend_filter else (False, 'neutral', 0.0)

    # Trailing stop and position decay config
    use_trailing = config.get('use_trailing_stop', False)
    trailing_activation = config.get('trailing_activation_pct', 0.3)
    trailing_distance = config.get('trailing_distance_pct', 0.2)
    use_decay = config.get('use_position_decay', False)
    decay_minutes = config.get('position_decay_minutes', 5)

    # Update highest/lowest price tracking for trailing stops
    if symbol in state.get('position_entries', {}):
        if symbol not in state.get('highest_price_since_entry', {}):
            state['highest_price_since_entry'][symbol] = price
        if symbol not in state.get('lowest_price_since_entry', {}):
            state['lowest_price_since_entry'][symbol] = price
        state['highest_price_since_entry'][symbol] = max(
            state['highest_price_since_entry'].get(symbol, price), price
        )
        state['lowest_price_since_entry'][symbol] = min(
            state['lowest_price_since_entry'].get(symbol, price), price
        )

    # Store comprehensive indicators
    state['indicators'] = {
        'symbol': symbol,
        'status': 'active',
        'price': round(price, 8),
        'sma': round(sma, 8),
        'upper_band': round(upper, 8),
        'lower_band': round(lower, 8),
        'std_dev': round(std_dev, 10),
        'z_score': round(z_score, 3),
        'band_width_pct': round(band_width, 4),
        'bollinger_std': num_std,  # REC-036: Track which std dev is being used
        'position_usd': round(current_position_usd, 4),
        'position_xrp': round(state.get('position_xrp', 0), 4),
        'max_position_usd': config.get('max_position_usd', 50.0),
        'available_usd': round(available_usd, 4),
        'actual_size_usd': round(actual_size_usd, 4),
        'xrp_accumulated': round(state.get('xrp_accumulated', 0), 4),
        'btc_accumulated': round(state.get('btc_accumulated', 0), 8),
        # REC-016: Enhanced accumulation metrics
        'xrp_accumulated_value_usd': round(state.get('xrp_accumulated_value_usd', 0), 4),
        'btc_accumulated_value_usd': round(state.get('btc_accumulated_value_usd', 0), 4),
        'volatility_pct': round(volatility, 4),
        'volatility_regime': regime.name,
        'regime_threshold_mult': round(regime_adjustments['threshold_mult'], 2),
        'regime_size_mult': round(regime_adjustments['size_mult'], 2),
        'base_entry_threshold': round(base_entry_threshold, 2),
        'effective_entry_threshold': round(effective_entry_threshold, 2),
        'exit_threshold': round(exit_threshold, 2),
        'current_spread_pct': round(current_spread, 4),
        'consecutive_losses': state.get('consecutive_losses', 0),
        'pnl_symbol': round(state.get('pnl_by_symbol', {}).get(symbol, 0), 8),
        'trades_symbol': state.get('trades_by_symbol', {}).get(symbol, 0),
        # REC-014: RSI confirmation
        'rsi': round(rsi, 2),
        'use_rsi_confirmation': use_rsi,
        # REC-015: Trend detection
        'is_strong_trend': is_strong_trend,
        'trend_direction': trend_direction,
        'trend_strength': round(trend_strength, 2),
        'use_trend_filter': use_trend_filter,
        # REC-018: Dynamic BTC price
        'btc_price_usd': round(btc_price_usd, 2),
        # REC-021: Correlation monitoring
        'correlation': round(correlation, 4),
        'use_correlation_monitoring': use_correlation,
        'correlation_warnings': state.get('correlation_warnings', 0),
        # REC-037: Correlation trend detection
        'correlation_slope': round(state.get('correlation_slope', 0.0), 6),
        'correlation_trend': state.get('correlation_trend_direction', 'stable'),
        'correlation_trend_warnings': state.get('correlation_trend_warnings', 0),
        # REC-050: Fee profitability check (v4.3.0)
        'use_fee_profitability_check': use_fee_check,
        'estimated_fee_rate': round(fee_rate, 6),
        'net_profit_pct': round(net_profit_pct, 4),
    }

    # Position limit check
    max_position = config.get('max_position_usd', 50.0)
    if current_position_usd >= max_position:
        state['indicators']['status'] = 'max_position_reached'
        if track_rejections:
            track_rejection(state, RejectionReason.MAX_POSITION, symbol)
        return None

    if actual_size_usd <= 0:
        state['indicators']['status'] = 'insufficient_size'
        if track_rejections:
            track_rejection(state, RejectionReason.INSUFFICIENT_SIZE, symbol)
        return None

    # REC-010: Trade flow confirmation (optional)
    use_trade_flow = config.get('use_trade_flow_confirmation', False)
    trade_flow_threshold = config.get('trade_flow_threshold', 0.10)
    trade_flow = data.get_trade_imbalance(symbol, 50)
    state['indicators']['trade_flow'] = round(trade_flow, 4)
    state['indicators']['use_trade_flow'] = use_trade_flow

    signal = None

    # ==========================================================================
    # Check trailing stop first if position exists
    # ==========================================================================
    if current_position_usd > 0 and use_trailing:
        entry_info = state.get('position_entries', {}).get(symbol, {})
        entry_price = entry_info.get('entry_price', price)
        highest = state.get('highest_price_since_entry', {}).get(symbol, price)
        lowest = state.get('lowest_price_since_entry', {}).get(symbol, price)

        trailing_stop_price = calculate_trailing_stop(
            entry_price, highest, lowest, 'long',
            trailing_activation, trailing_distance
        )

        if trailing_stop_price and price <= trailing_stop_price:
            min_trade_size = config.get('min_trade_size_usd', 5.0)
            exit_size = min(actual_size_usd, current_position_usd)
            if exit_size >= min_trade_size:
                signal = Signal(
                    action='sell',
                    symbol=symbol,
                    size=exit_size,
                    price=price,
                    reason=f"RT: Trailing stop hit (entry={entry_price:.8f}, high={highest:.8f}, stop={trailing_stop_price:.8f})",
                    metadata={
                        'strategy': 'ratio_trading',
                        'signal_type': 'trailing_stop',
                        'trailing_stop_price': round(trailing_stop_price, 8),
                        'exit_reason': ExitReason.TRAILING_STOP.value,  # REC-020
                    }
                )
                state['indicators']['status'] = 'trailing_stop_triggered'
                # REC-020: Track as exit, not rejection
                track_exit(state, ExitReason.TRAILING_STOP, symbol)
                state['last_signal_time'] = current_time
                state['trade_count'] += 1
                return signal

    # ==========================================================================
    # Check position decay if position exists
    # ==========================================================================
    if current_position_usd > 0 and use_decay:
        entry_info = state.get('position_entries', {}).get(symbol, {})
        entry_time = entry_info.get('entry_time')

        is_decayed, minutes_held = check_position_decay(entry_time, current_time, decay_minutes)
        state['indicators']['position_minutes_held'] = round(minutes_held, 1)
        state['indicators']['position_decayed'] = is_decayed

        if is_decayed and abs(z_score) < exit_threshold * 1.5:  # Close if somewhat near mean
            min_trade_size = config.get('min_trade_size_usd', 5.0)
            exit_size = min(actual_size_usd, current_position_usd * 0.5)  # Partial exit

            if exit_size >= min_trade_size:
                signal = Signal(
                    action='sell',
                    symbol=symbol,
                    size=exit_size,
                    price=price,
                    reason=f"RT: Position decay exit (held {minutes_held:.1f}min, z={z_score:.2f})",
                    metadata={
                        'strategy': 'ratio_trading',
                        'signal_type': 'position_decay',
                        'minutes_held': round(minutes_held, 1),
                        'z_score': round(z_score, 3),
                        'exit_reason': ExitReason.POSITION_DECAY.value,  # REC-020
                    }
                )
                state['indicators']['status'] = 'position_decay_exit'
                # REC-020: Track as exit, not rejection
                track_exit(state, ExitReason.POSITION_DECAY, symbol)
                state['last_signal_time'] = current_time
                state['trade_count'] += 1
                return signal

    # ==========================================================================
    # BUY Signal: Price below lower band (XRP cheap vs BTC)
    # Action: Spend BTC to buy XRP
    # ==========================================================================
    if z_score < -effective_entry_threshold:
        # REC-015: Trend filter check - don't buy into strong downtrend
        if use_trend_filter and is_strong_trend and trend_direction == 'down':
            state['indicators']['status'] = 'strong_trend_detected'
            state['indicators']['trend_warning'] = 'Strong downtrend - buy signal blocked'
            if track_rejections:
                track_rejection(state, RejectionReason.STRONG_TREND_DETECTED, symbol)
            # Don't return None - just skip buy signal, check other conditions

        # REC-014: RSI confirmation check - buy only if oversold
        elif use_rsi and rsi > rsi_oversold:
            state['indicators']['status'] = 'rsi_not_confirmed'
            state['indicators']['rsi_required'] = f'RSI {rsi:.1f} > {rsi_oversold} (not oversold)'
            if track_rejections:
                track_rejection(state, RejectionReason.RSI_NOT_CONFIRMED, symbol)
            # Don't return None - just skip buy signal

        else:
            # Trade flow confirmation if enabled
            if use_trade_flow:
                if not is_trade_flow_aligned(data, symbol, 'buy', trade_flow_threshold):
                    state['indicators']['status'] = 'trade_flow_not_aligned'
                    state['indicators']['trade_flow_aligned'] = False
                    if track_rejections:
                        track_rejection(state, RejectionReason.TRADE_FLOW_NOT_ALIGNED, symbol)
                    return None

            state['indicators']['trade_flow_aligned'] = True
            signal = generate_buy_signal(
                symbol, price, actual_size_usd, z_score,
                effective_entry_threshold, sl_pct, tp_pct, regime.name, Signal
            )

    # ==========================================================================
    # SELL Signal: Price above upper band (XRP expensive vs BTC)
    # Action: Sell XRP to get BTC
    # ==========================================================================
    elif z_score > effective_entry_threshold:
        # REC-015: Trend filter check - don't sell into strong uptrend
        if use_trend_filter and is_strong_trend and trend_direction == 'up':
            state['indicators']['status'] = 'strong_trend_detected'
            state['indicators']['trend_warning'] = 'Strong uptrend - sell signal blocked'
            if track_rejections:
                track_rejection(state, RejectionReason.STRONG_TREND_DETECTED, symbol)
            # Don't return None - just skip sell signal

        # REC-014: RSI confirmation check - sell only if overbought
        elif use_rsi and rsi < rsi_overbought:
            state['indicators']['status'] = 'rsi_not_confirmed'
            state['indicators']['rsi_required'] = f'RSI {rsi:.1f} < {rsi_overbought} (not overbought)'
            if track_rejections:
                track_rejection(state, RejectionReason.RSI_NOT_CONFIRMED, symbol)
            # Don't return None - just skip sell signal

        else:
            # Trade flow confirmation if enabled
            if use_trade_flow:
                if not is_trade_flow_aligned(data, symbol, 'sell', trade_flow_threshold):
                    state['indicators']['status'] = 'trade_flow_not_aligned'
                    state['indicators']['trade_flow_aligned'] = False
                    if track_rejections:
                        track_rejection(state, RejectionReason.TRADE_FLOW_NOT_ALIGNED, symbol)
                    return None

            state['indicators']['trade_flow_aligned'] = True

            if current_position_usd > 0:
                # Sell from our position
                sell_size = min(actual_size_usd, current_position_usd)
                signal = generate_sell_signal(
                    symbol, price, sell_size, z_score,
                    effective_entry_threshold, sl_pct, tp_pct, regime.name, Signal
                )
            else:
                # No position but still signal for accumulating BTC
                # Sell from "starting XRP holdings" concept
                signal = generate_sell_signal(
                    symbol, price, actual_size_usd, z_score,
                    effective_entry_threshold, sl_pct, tp_pct, regime.name, Signal
                )

    # ==========================================================================
    # EXIT/TAKE PROFIT: Position exists and price reverted toward mean
    # ==========================================================================
    elif current_position_usd > 0 and abs(z_score) < exit_threshold:
        # Partial exit for take profit
        min_trade_size = config.get('min_trade_size_usd', 5.0)
        exit_size = min(actual_size_usd, current_position_usd * 0.5)  # Partial exit

        if exit_size >= min_trade_size:
            signal = generate_exit_signal(
                symbol, price, exit_size, z_score, exit_threshold, Signal
            )

    if signal:
        state['last_signal_time'] = current_time
        state['trade_count'] += 1
        state['indicators']['status'] = 'signal_generated'
        return signal

    state['indicators']['status'] = 'no_signal'
    if track_rejections:
        track_rejection(state, RejectionReason.NO_SIGNAL_CONDITIONS, symbol)
    return None
