"""
Grid RSI Reversion Strategy - Signal Generation

Contains the main generate_signal function and symbol evaluation logic.
Based on research from master-plan-v1.0.md.

Signal Generation Flow:
1. Initialize state + grid levels (on first call or per symbol)
2. Check trend filter (ADX < threshold)
3. Check volatility regime
4. Calculate indicators (RSI, ATR)
5. Check existing position exits first:
   - Grid stop loss
   - Max drawdown
   - Cycle completion (matched sell)
6. Check grid entry conditions:
   - Price at/near grid level
   - Grid level not filled
   - RSI zone (confidence modifier)
7. Check risk limits:
   - Max accumulation
   - Position limits
   - Correlation exposure
8. Generate Signal or None
"""
from datetime import datetime
from typing import Dict, Any, Optional

from ws_tester.types import DataSnapshot, Signal

from .config import (
    SYMBOLS, VolatilityRegime, RejectionReason,
    get_symbol_config
)
from .indicators import (
    calculate_rsi, calculate_atr, calculate_adx, calculate_volatility,
    get_adaptive_rsi_zones, calculate_rsi_confidence,
    calculate_position_size_multiplier, calculate_grid_rr_ratio,
    # REC-003: Trade flow confirmation
    calculate_trade_flow, calculate_volume_ratio, check_trade_flow_confirmation,
    # REC-005: Correlation monitoring
    calculate_rolling_correlation,
    # REC-006: Liquidity validation
    check_liquidity_threshold
)
from .regimes import (
    classify_volatility_regime, get_regime_adjustments,
    classify_trading_session, get_session_adjustments
)
from .risk import (
    check_accumulation_limit, check_trend_filter,
    check_all_risk_limits
)
from .grid import (
    check_price_at_grid_level, calculate_grid_stats,
    should_recenter_grid, recenter_grid
)
from .exits import check_all_exits
from .lifecycle import initialize_state, initialize_grid_for_symbol


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
    state: Dict[str, Any],
    price: float = None,
    rsi: float = None,
    atr: float = None,
    adx: float = None,
    volatility: float = None,
    regime: str = None,
) -> Dict[str, Any]:
    """
    Build base indicators dict for early returns.

    REC-002: Ensures comprehensive indicator logging on all code paths,
    including early exits, for complete analysis capability.

    Args:
        symbol: Trading symbol
        candle_count: Number of available candles
        status: Current signal status/rejection reason
        state: Strategy state dict
        price: Current price (optional)
        rsi: Current RSI value (optional)
        atr: Current ATR value (optional)
        adx: Current ADX value (optional)
        volatility: Current volatility percentage (optional)
        regime: Current volatility regime (optional)

    Returns:
        Dict containing all indicator data for logging
    """
    indicators = {
        'symbol': symbol,
        'candle_count': candle_count,
        'status': status,
        # Position tracking
        'position_side': state.get('position_side'),
        'position_size': state.get('position_size', 0),
        'position_size_symbol': round(state.get('position_by_symbol', {}).get(symbol, 0), 2),
        'consecutive_losses': state.get('consecutive_losses', 0),
        'pnl_symbol': round(state.get('pnl_by_symbol', {}).get(symbol, 0), 4),
        'trades_symbol': state.get('trades_by_symbol', {}).get(symbol, 0),
    }

    # Add optional indicator values if available (REC-002)
    if price is not None:
        indicators['price'] = round(price, 6)
    if rsi is not None:
        indicators['rsi'] = round(rsi, 2)
    if atr is not None:
        indicators['atr'] = round(atr, 8)
    if adx is not None:
        indicators['adx'] = round(adx, 2)
    if volatility is not None:
        indicators['volatility_pct'] = round(volatility, 4)
    if regime is not None:
        indicators['volatility_regime'] = regime

    # Grid stats if available
    grid_metadata = state.get('grid_metadata', {}).get(symbol, {})
    if grid_metadata:
        indicators['grid_center'] = grid_metadata.get('center_price')
        indicators['grid_upper'] = grid_metadata.get('upper_price')
        indicators['grid_lower'] = grid_metadata.get('lower_price')
        indicators['cycles_completed'] = grid_metadata.get('cycles_completed', 0)

    return indicators


def generate_signal(
    data: DataSnapshot,
    config: Dict[str, Any],
    state: Dict[str, Any]
) -> Optional[Signal]:
    """
    Generate grid RSI reversion signal.

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
    # Cooldown Check
    # ==========================================================================
    if state['last_signal_time'] is not None:
        elapsed = (current_time - state['last_signal_time']).total_seconds()
        cooldown = config.get('cooldown_seconds', 60.0)
        if elapsed < cooldown:
            state['indicators'] = build_base_indicators(
                symbol='N/A', candle_count=0, status='cooldown', state=state
            )
            state['indicators']['cooldown_remaining'] = cooldown - elapsed
            if track_rejections:
                track_rejection(state, RejectionReason.COOLDOWN)
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
    Evaluate grid signals for a specific symbol.

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
    # Use 5m candles as primary timeframe for grid strategy
    candles_5m = data.candles_5m.get(symbol, ())
    candles_1m = data.candles_1m.get(symbol, ())

    # Minimum candles required for indicators
    rsi_period = config.get('rsi_period', 14)
    atr_period = config.get('atr_period', 14)
    adx_period = config.get('adx_period', 14)
    min_candles = max(rsi_period, atr_period, adx_period) + 5

    if len(candles_5m) < min_candles:
        # REC-002: Log available price even during warmup
        price = data.prices.get(symbol)
        state['indicators'] = build_base_indicators(
            symbol=symbol, candle_count=len(candles_5m), status='warming_up', state=state,
            price=price
        )
        state['indicators']['required_candles'] = min_candles
        if track_rejections:
            track_rejection(state, RejectionReason.WARMING_UP, symbol)
        return None

    # ==========================================================================
    # Calculate Indicators
    # ==========================================================================
    closes_5m = [c.close for c in candles_5m]
    current_price = data.prices.get(symbol, closes_5m[-1])

    # RSI
    rsi = calculate_rsi(closes_5m, rsi_period)

    # ATR
    atr = calculate_atr(candles_5m, atr_period)

    # ADX for trend filter
    adx = calculate_adx(candles_5m, adx_period)

    # Volatility
    volatility = calculate_volatility(candles_5m, lookback=20)

    # ==========================================================================
    # Initialize Grid if Not Done
    # ==========================================================================
    if not state.get('grids_initialized', {}).get(symbol, False):
        initialize_grid_for_symbol(symbol, current_price, config, state, atr)

    grid_levels = state.get('grid_levels', {}).get(symbol, [])
    grid_metadata = state.get('grid_metadata', {}).get(symbol, {})

    # ==========================================================================
    # Liquidity Validation (REC-006)
    # ==========================================================================
    min_volume_usd = get_symbol_config(symbol, config, 'min_volume_usd')
    if min_volume_usd and min_volume_usd > 0:
        # Get 24h volume from data if available
        volume_24h = data.volumes.get(symbol, 0) if hasattr(data, 'volumes') else 0

        liquidity_ok, liquidity_reason = check_liquidity_threshold(volume_24h, min_volume_usd)
        state['indicators']['volume_24h'] = volume_24h
        state['indicators']['min_volume_required'] = min_volume_usd
        state['indicators']['liquidity_reason'] = liquidity_reason

        if not liquidity_ok:
            state['indicators'] = build_base_indicators(
                symbol=symbol, candle_count=len(candles_5m), status='low_liquidity', state=state,
                price=current_price, rsi=rsi, atr=atr, adx=adx, volatility=volatility
            )
            state['indicators']['liquidity_reason'] = liquidity_reason
            if track_rejections:
                track_rejection(state, RejectionReason.LOW_LIQUIDITY, symbol)
            return None

    if not grid_levels:
        # REC-002: Include calculated indicators even when grid not initialized
        state['indicators'] = build_base_indicators(
            symbol=symbol, candle_count=len(candles_5m), status='no_grid', state=state,
            price=current_price, rsi=rsi, atr=atr, adx=adx, volatility=volatility
        )
        if track_rejections:
            track_rejection(state, RejectionReason.NO_GRID_LEVELS, symbol)
        return None

    # ==========================================================================
    # Volatility Regime
    # ==========================================================================
    regime = VolatilityRegime.MEDIUM
    regime_adjustments = {'grid_spacing_mult': 1.0, 'size_mult': 1.0, 'pause_trading': False}

    if config.get('use_volatility_regimes', True):
        regime = classify_volatility_regime(volatility, config)
        regime_adjustments = get_regime_adjustments(regime, config)

        if regime_adjustments['pause_trading']:
            # REC-002: Include all available indicators on regime pause
            state['indicators'] = build_base_indicators(
                symbol=symbol, candle_count=len(candles_5m), status='regime_pause', state=state,
                price=current_price, rsi=rsi, atr=atr, adx=adx, volatility=volatility,
                regime=regime.name
            )
            if track_rejections:
                track_rejection(state, RejectionReason.REGIME_PAUSE, symbol)
            return None

    # ==========================================================================
    # Trend Filter
    # ==========================================================================
    is_trending, adx_value = check_trend_filter(adx, config)
    if is_trending:
        # REC-002: Include all indicators on trend filter rejection
        state['indicators'] = build_base_indicators(
            symbol=symbol, candle_count=len(candles_5m), status='trend_filter', state=state,
            price=current_price, rsi=rsi, atr=atr, adx=adx_value, volatility=volatility,
            regime=regime.name
        )
        state['indicators']['adx_threshold'] = config.get('adx_threshold', 30)
        if track_rejections:
            track_rejection(state, RejectionReason.TREND_FILTER, symbol)
        return None

    # ==========================================================================
    # Session Awareness
    # ==========================================================================
    session = classify_trading_session(current_time, config)
    session_adjustments = get_session_adjustments(session, config)
    combined_size_mult = regime_adjustments['size_mult'] * session_adjustments['size_mult']

    # ==========================================================================
    # Adaptive RSI Zones
    # ==========================================================================
    oversold, overbought = get_adaptive_rsi_zones(atr, current_price, config, symbol)

    # ==========================================================================
    # Build Indicators
    # ==========================================================================
    grid_stats = calculate_grid_stats(grid_levels, grid_metadata)

    state['indicators'] = {
        'symbol': symbol,
        'status': 'active',
        'candle_count': len(candles_5m),
        'price': round(current_price, 6),
        # RSI
        'rsi': round(rsi, 2) if rsi else None,
        'rsi_oversold': oversold,
        'rsi_overbought': overbought,
        # ATR
        'atr': round(atr, 8) if atr else None,
        # ADX
        'adx': round(adx, 2) if adx else None,
        # Regime & Session
        'volatility_pct': round(volatility, 4),
        'volatility_regime': regime.name,
        'trading_session': session.name,
        'combined_size_mult': round(combined_size_mult, 2),
        # Grid stats
        'grid_center': grid_metadata.get('center_price'),
        'grid_upper': grid_metadata.get('upper_price'),
        'grid_lower': grid_metadata.get('lower_price'),
        'filled_buys': grid_stats['filled_buys'],
        'filled_sells': grid_stats['filled_sells'],
        'cycles_completed': grid_stats['cycles_completed'],
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
    exit_signal = check_all_exits(state, symbol, current_price, current_time, config)
    if exit_signal:
        state['last_signal_time'] = current_time
        state['indicators']['status'] = 'exit_signal'
        return exit_signal

    # ==========================================================================
    # Check Grid Recentering (REC-008)
    # ==========================================================================
    should_recenter, recenter_reason = should_recenter_grid(grid_metadata, current_time, config, adx)
    state['indicators']['recenter_check_result'] = recenter_reason

    if should_recenter:
        new_levels, new_metadata = recenter_grid(
            symbol, current_price, config, grid_metadata, atr
        )
        state['grid_levels'][symbol] = new_levels
        state['grid_metadata'][symbol] = new_metadata
        grid_levels = new_levels
        grid_metadata = new_metadata
        state['indicators']['status'] = 'grid_recentered'

    # ==========================================================================
    # Calculate Correlations (REC-005)
    # ==========================================================================
    correlations = {}
    if config.get('use_real_correlation', True):
        # Store price history in state for correlation calculation
        if 'price_history' not in state:
            state['price_history'] = {}

        # Update price history for current symbol
        if symbol not in state['price_history']:
            state['price_history'][symbol] = []
        state['price_history'][symbol].append(current_price)
        # Keep bounded history
        max_history = config.get('correlation_lookback', 20) + 10
        state['price_history'][symbol] = state['price_history'][symbol][-max_history:]

        # Calculate correlations with other symbols
        lookback = config.get('correlation_lookback', 20)
        for other_symbol in SYMBOLS:
            if other_symbol == symbol:
                continue
            other_prices = state['price_history'].get(other_symbol, [])
            if len(other_prices) >= lookback and len(state['price_history'][symbol]) >= lookback:
                corr = calculate_rolling_correlation(
                    state['price_history'][symbol],
                    other_prices,
                    lookback
                )
                if corr is not None:
                    pair_key = f"{symbol}_{other_symbol}"
                    correlations[pair_key] = corr

        # Log correlations in indicators
        if correlations:
            state['indicators']['correlations'] = {k: round(v, 3) for k, v in correlations.items()}

    # ==========================================================================
    # Check Risk Limits
    # ==========================================================================
    base_size = get_symbol_config(symbol, config, 'position_size_usd')
    adjusted_size = base_size * combined_size_mult

    # REC-005: Pass correlations to risk check
    can_trade, available_size, block_reason = check_all_risk_limits(
        state, symbol, adjusted_size, config, adx, current_time, correlations
    )

    if not can_trade:
        # REC-002: Include detailed rejection reason and available size info
        state['indicators']['status'] = f'risk_block_{block_reason}'
        state['indicators']['rejection_reason'] = block_reason
        state['indicators']['requested_size'] = round(adjusted_size, 2)
        state['indicators']['available_size'] = round(available_size, 2)
        if track_rejections:
            reason_map = {
                'circuit_breaker': RejectionReason.CIRCUIT_BREAKER,
                'trend_filter': RejectionReason.TREND_FILTER,
                'max_drawdown': RejectionReason.MAX_POSITION,
                'accumulation_limit': RejectionReason.MAX_ACCUMULATION,
                'symbol_limit': RejectionReason.MAX_POSITION,
                'total_limit': RejectionReason.MAX_POSITION,
                'min_size': RejectionReason.MAX_POSITION,
                'correlation_limit': RejectionReason.CORRELATION_LIMIT,
            }
            track_rejection(state, reason_map.get(block_reason, RejectionReason.NO_SIGNAL_CONDITIONS), symbol)
        return None

    # ==========================================================================
    # Trade Flow Confirmation (REC-003)
    # ==========================================================================
    trade_flow_ok = True
    flow_imbalance = 0.0
    volume_ratio = 1.0

    if config.get('use_trade_flow_confirmation', True):
        trades = data.trades.get(symbol, ())

        # Calculate trade flow metrics
        buy_vol, sell_vol, flow_imbalance = calculate_trade_flow(trades, lookback=50)
        volume_ratio = calculate_volume_ratio(candles_5m, lookback=20)

        # Log trade flow metrics in indicators
        state['indicators']['trade_flow_buy_vol'] = round(buy_vol, 2)
        state['indicators']['trade_flow_sell_vol'] = round(sell_vol, 2)
        state['indicators']['trade_flow_imbalance'] = round(flow_imbalance, 3)
        state['indicators']['volume_ratio'] = round(volume_ratio, 2)

        # Check volume ratio
        min_volume_ratio = config.get('min_volume_ratio', 0.8)
        if volume_ratio < min_volume_ratio:
            state['indicators']['status'] = 'low_volume'
            state['indicators']['rejection_reason'] = f'volume_ratio={volume_ratio:.2f}<{min_volume_ratio}'
            if track_rejections:
                track_rejection(state, RejectionReason.LOW_VOLUME, symbol)
            return None

    # ==========================================================================
    # Check Grid Buy Entry
    # ==========================================================================
    if rsi is None:
        # REC-002: Update status for RSI not available
        state['indicators']['status'] = 'no_rsi'
        state['indicators']['rejection_reason'] = 'rsi_calculation_failed'
        if track_rejections:
            track_rejection(state, RejectionReason.WARMING_UP, symbol)
        return None

    # Check if price is at a buy grid level
    buy_level = check_price_at_grid_level(grid_levels, current_price, 'buy', config)

    if buy_level:
        # REC-003: Check trade flow confirmation for buy signals
        if config.get('use_trade_flow_confirmation', True):
            flow_threshold = config.get('flow_confirmation_threshold', 0.2)
            flow_confirmed, flow_reason = check_trade_flow_confirmation(
                flow_imbalance, 'buy', flow_threshold
            )
            if not flow_confirmed:
                state['indicators']['status'] = 'flow_rejected'
                state['indicators']['rejection_reason'] = flow_reason
                if track_rejections:
                    track_rejection(state, RejectionReason.FLOW_AGAINST_TRADE, symbol)
                return None

        # Calculate RSI confidence
        confidence, confidence_reason = calculate_rsi_confidence(
            'buy', rsi, oversold, overbought, config
        )

        # Apply RSI-based size multiplier
        rsi_size_mult = calculate_position_size_multiplier(rsi, oversold, overbought, config)
        final_size = min(available_size, adjusted_size * rsi_size_mult)

        # Calculate stop loss (below lowest grid level)
        lower_price = grid_metadata.get('lower_price', current_price * 0.9)
        stop_loss_pct = get_symbol_config(symbol, config, 'stop_loss_pct')
        stop_loss = lower_price * (1 - stop_loss_pct / 100)

        # REC-007: Calculate and log R:R ratio
        grid_spacing_pct = get_symbol_config(symbol, config, 'grid_spacing_pct')
        filled_buys = grid_stats['filled_buys'] + 1  # Including this entry
        rr_ratio, rr_desc = calculate_grid_rr_ratio(
            grid_spacing_pct, stop_loss_pct, filled_buys
        )

        signal = Signal(
            action='buy',
            symbol=symbol,
            size=final_size,
            price=current_price,
            reason=f"Grid buy @ ${buy_level['price']:.4f}, RSI={rsi:.1f} ({confidence_reason})",
            stop_loss=stop_loss,
            take_profit=None,  # Grid uses cycle completion, not fixed TP
            metadata={
                'entry_type': 'grid_buy',
                'grid_order_id': buy_level['order_id'],
                'grid_level_price': buy_level['price'],
                'rsi': rsi,
                'rsi_zone': 'oversold' if rsi < oversold else 'neutral',
                'confidence': confidence,
                # REC-007: Include R:R metrics
                'rr_ratio': round(rr_ratio, 2),
                'rr_description': rr_desc,
                'grid_spacing_pct': grid_spacing_pct,
                'stop_loss_pct': stop_loss_pct,
                'accumulation_level': filled_buys,
                # REC-003: Include trade flow metrics
                'flow_imbalance': round(flow_imbalance, 3),
                'volume_ratio': round(volume_ratio, 2),
            }
        )

        state['last_signal_time'] = current_time
        state['indicators']['status'] = 'signal_generated'
        return signal

    # ==========================================================================
    # No Signal Conditions Met
    # ==========================================================================
    state['indicators']['status'] = 'no_signal'
    if track_rejections:
        track_rejection(state, RejectionReason.PRICE_NOT_AT_LEVEL, symbol)

    return None
