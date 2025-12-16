"""
EMA-9 Trend Flip Strategy - Signal Generation

Main signal generation logic.

Code Review Fixes Applied:
- Issue #5: Improved indicator logging consistency
- Issue #6: Circuit breaker integration
- Issue #8: Database warmup integration
"""
import logging
from datetime import datetime
from typing import Dict, Any, Optional

from ws_tester.types import DataSnapshot, Signal

from .config import STRATEGY_NAME, SYMBOLS, RejectionReason
from .indicators import (
    calculate_ema_series, build_hourly_candles, calculate_atr,
    get_candle_position, check_consecutive_positions
)
from .exits import check_exit_conditions
from .risk import create_entry_signal, track_rejection
from .lifecycle import initialize_state

# Configure logger
logger = logging.getLogger(STRATEGY_NAME)


def check_circuit_breaker(
    state: Dict[str, Any],
    current_time: datetime,
    config: Dict[str, Any]
) -> bool:
    """
    Check if circuit breaker is active.

    Args:
        state: Strategy state
        current_time: Current timestamp
        config: Strategy configuration

    Returns:
        True if circuit breaker is active (should not trade), False otherwise
    """
    if not config.get('use_circuit_breaker', True):
        return False

    circuit_breaker_time = state.get('circuit_breaker_time')
    if circuit_breaker_time is None:
        return False

    cooldown_min = config.get('circuit_breaker_minutes', 30)
    elapsed_min = (current_time - circuit_breaker_time).total_seconds() / 60

    if elapsed_min < cooldown_min:
        return True

    # Circuit breaker expired, reset
    state['circuit_breaker_time'] = None
    state['consecutive_losses'] = 0
    return False


def build_indicators(
    symbol: str,
    status: str,
    state: Dict[str, Any],
    candle_count: int = 0,
    **extra
) -> Dict[str, Any]:
    """
    Build indicators dict for logging.

    Provides consistent structure for all indicator logging paths.

    Args:
        symbol: Trading symbol
        status: Current status string
        state: Strategy state
        candle_count: Number of candles available
        **extra: Additional indicator values

    Returns:
        Dict with all indicator values for logging
    """
    indicators = {
        'symbol': symbol,
        'status': status,
        'candle_count': candle_count,
        # Position tracking
        'position_side': state.get('position_side'),
        'position_size': round(state.get('position', 0), 2),
        'entry_price': round(state.get('entry_price', 0), 2),
        # Trade stats
        'trade_count': state.get('trade_count', 0),
        'win_count': state.get('win_count', 0),
        'loss_count': state.get('loss_count', 0),
        'total_pnl': round(state.get('total_pnl', 0), 4),
        # Circuit breaker
        'consecutive_losses': state.get('consecutive_losses', 0),
        'circuit_breaker_active': state.get('circuit_breaker_time') is not None,
    }
    indicators.update(extra)
    return indicators


def generate_signal(
    data: DataSnapshot,
    config: Dict[str, Any],
    state: Dict[str, Any]
) -> Optional[Signal]:
    """
    Generate EMA-9 Trend Flip signal.

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
    # Circuit Breaker Check (Issue #6)
    # ==========================================================================
    if check_circuit_breaker(state, current_time, config):
        cooldown_min = config.get('circuit_breaker_minutes', 30)
        elapsed = 0
        if state.get('circuit_breaker_time'):
            elapsed = (current_time - state['circuit_breaker_time']).total_seconds() / 60
        remaining = cooldown_min - elapsed

        state['indicators'] = build_indicators(
            symbol='N/A',
            status='circuit_breaker',
            state=state,
            candle_count=0,
            circuit_breaker_remaining_min=round(remaining, 1),
            max_consecutive_losses=config.get('max_consecutive_losses', 3),
        )
        if track_rejections:
            track_rejection(state, RejectionReason.TIME_COOLDOWN, None)
        return None

    # Process each symbol
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
    Evaluate EMA-9 flip signals for a specific symbol.

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
    # Get Price Data
    # ==========================================================================
    current_price = data.prices.get(symbol)
    if not current_price:
        state['indicators'] = build_indicators(
            symbol=symbol,
            status='no_price_data',
            state=state
        )
        if track_rejections:
            track_rejection(state, RejectionReason.NO_PRICE_DATA, symbol)
        return None

    # ==========================================================================
    # Get Candle Data
    # ==========================================================================
    timeframe_minutes = config.get('candle_timeframe_minutes', 60)

    # Use pre-aggregated hourly candles if available (PERF: ~60x faster)
    # Fall back to building from 1m candles for live trading or if not available
    if timeframe_minutes == 60 and hasattr(data, 'candles_1h') and data.candles_1h.get(symbol):
        # Use pre-aggregated hourly candles from database
        raw_candles = data.candles_1h.get(symbol, ())
        hourly_candles = [
            {'timestamp': c.timestamp, 'open': c.open, 'high': c.high,
             'low': c.low, 'close': c.close, 'volume': c.volume}
            for c in raw_candles
        ]
    else:
        # Build hourly candles from 1m data (legacy path for live trading)
        candles_1m = data.candles_1m.get(symbol, ())
        if not candles_1m:
            state['indicators'] = build_indicators(
                symbol=symbol,
                status='no_candle_data',
                state=state
            )
            return None
        hourly_candles = build_hourly_candles(candles_1m, timeframe_minutes)

    min_candles = config.get('min_candles_required', 15)
    if len(hourly_candles) < min_candles:
        state['indicators'] = build_indicators(
            symbol=symbol,
            status='warming_up',
            state=state,
            hourly_candle_count=len(hourly_candles),
            required_candles=min_candles
        )
        if track_rejections:
            track_rejection(state, RejectionReason.WARMING_UP, symbol)
        return None

    # ==========================================================================
    # Calculate EMA on Open Prices
    # ==========================================================================
    ema_period = config.get('ema_period', 9)
    use_open = config.get('use_open_price', True)

    if use_open:
        prices = [c['open'] for c in hourly_candles]
    else:
        prices = [c['close'] for c in hourly_candles]

    ema_values = calculate_ema_series(prices, ema_period)
    current_ema = ema_values[-1] if ema_values and ema_values[-1] is not None else None

    if current_ema is None:
        state['indicators'] = build_indicators(
            symbol=symbol,
            status='insufficient_ema_data',
            state=state
        )
        if track_rejections:
            track_rejection(state, RejectionReason.INSUFFICIENT_CANDLES, symbol)
        return None

    # ==========================================================================
    # Calculate ATR for Dynamic Stops
    # ==========================================================================
    atr = calculate_atr(hourly_candles, period=14)

    # ==========================================================================
    # Check Current Candle Position
    # ==========================================================================
    buffer_pct = config.get('buffer_pct', 0.0)
    strict_candle_mode = config.get('strict_candle_mode', True)  # REQUIRED: Use whole candle check

    latest_candle = hourly_candles[-1]
    current_position = get_candle_position(
        latest_candle, current_ema, buffer_pct, use_open, strict_candle_mode
    )

    # ==========================================================================
    # Check Previous Consecutive Positions
    # ==========================================================================
    n_consecutive = config.get('consecutive_candles', 3)
    prev_position, prev_count = check_consecutive_positions(
        hourly_candles[:-1],  # Exclude current candle
        ema_values[:-1],
        n_consecutive,
        buffer_pct,
        use_open,
        strict_candle_mode  # NEW: Use strict mode for previous candles too
    )

    # ==========================================================================
    # Build Comprehensive Indicators
    # ==========================================================================
    state['indicators'] = {
        'symbol': symbol,
        'status': 'active',
        'price': round(current_price, 2),
        'ema_9': round(current_ema, 2) if current_ema else None,
        'ema_period': ema_period,
        'timeframe_minutes': timeframe_minutes,
        'hourly_candle_count': len(hourly_candles),
        'current_position': current_position,
        'prev_position': prev_position,
        'prev_consecutive_count': prev_count,
        'buffer_pct': buffer_pct,
        'strict_candle_mode': strict_candle_mode,
        'atr': round(atr, 4) if atr else None,
        # Position info
        'position_side': state.get('position_side'),
        'position_size': round(state.get('position', 0), 2),
        'entry_price': round(state.get('entry_price', 0), 2),
        # Trade stats
        'trade_count': state.get('trade_count', 0),
        'win_count': state.get('win_count', 0),
        'loss_count': state.get('loss_count', 0),
        'total_pnl': round(state.get('total_pnl', 0), 4),
    }

    # ==========================================================================
    # Time-Based Cooldown Check
    # ==========================================================================
    if state.get('last_signal_time'):
        cooldown_min = config.get('cooldown_minutes', 30)
        if state.get('last_trade_was_loss'):
            cooldown_min = config.get('cooldown_after_loss_minutes', 60)

        elapsed_min = (current_time - state['last_signal_time']).total_seconds() / 60
        if elapsed_min < cooldown_min:
            state['indicators']['status'] = 'cooldown'
            state['indicators']['cooldown_remaining_min'] = round(cooldown_min - elapsed_min, 1)
            if track_rejections:
                track_rejection(state, RejectionReason.TIME_COOLDOWN, symbol)
            return None

    # ==========================================================================
    # Check Exit Conditions First (If In Position)
    # ==========================================================================
    if state.get('position_side'):
        exit_signal = check_exit_conditions(
            state=state,
            symbol=symbol,
            current_price=current_price,
            current_ema=current_ema,
            current_position=current_position,
            current_time=current_time,
            config=config,
            atr=atr
        )
        if exit_signal:
            state['last_signal_time'] = current_time
            state['indicators']['status'] = 'exit_signal'
            return exit_signal

        # If in position but no exit signal, skip entry logic
        state['indicators']['status'] = 'in_position'
        return None

    # ==========================================================================
    # Check Entry Conditions (EMA Flip Detection)
    # ==========================================================================
    # A "flip" occurs when:
    # 1. Previous N candles were consistently on one side (above or below)
    # 2. Current candle opens on the opposite side (with buffer)

    signal = None

    # Debug logging for flip detection (only log occasionally to avoid spam)
    if config.get('debug_signals', False):
        logger.debug(
            f"Flip check: prev_pos={prev_position}({prev_count}/{n_consecutive}), "
            f"curr_pos={current_position}, price={current_price:.2f}, ema={current_ema:.2f}"
        )

    # Check for LONG entry (flip from below to above)
    if prev_position == 'below' and prev_count >= n_consecutive:
        if current_position == 'above':
            # FLIP DETECTED: Enter Long
            # strict_candle_mode already ensures quality entry (whole candle above EMA)
            logger.info(
                f"LONG flip detected: {symbol} - {prev_count} candles below EMA, "
                f"current above. Price={current_price:.2f}, EMA={current_ema:.2f}"
            )
            signal = create_entry_signal(
                symbol=symbol,
                direction='long',
                price=current_price,
                ema=current_ema,
                config=config,
                state=state,
                atr=atr,
                prev_count=prev_count
            )

    # Check for SHORT entry (flip from above to below)
    elif prev_position == 'above' and prev_count >= n_consecutive:
        if current_position == 'below':
            # FLIP DETECTED: Enter Short
            # strict_candle_mode already ensures quality entry (whole candle below EMA)
            logger.info(
                f"SHORT flip detected: {symbol} - {prev_count} candles above EMA, "
                f"current below. Price={current_price:.2f}, EMA={current_ema:.2f}"
            )
            signal = create_entry_signal(
                symbol=symbol,
                direction='short',
                price=current_price,
                ema=current_ema,
                config=config,
                state=state,
                atr=atr,
                prev_count=prev_count
            )

    # ==========================================================================
    # Return Signal or None
    # ==========================================================================
    if signal:
        state['last_signal_time'] = current_time
        state['indicators']['status'] = 'signal_generated'
        state['indicators']['signal_action'] = signal.action
        return signal

    # No signal conditions met
    if state['indicators'].get('status') == 'active':
        state['indicators']['status'] = 'no_flip_signal'
        if track_rejections:
            track_rejection(state, RejectionReason.NO_FLIP_SIGNAL, symbol)

    return None
