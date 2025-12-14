"""
Ratio Trading Strategy - Tracking Module

Signal rejection tracking, exit tracking, state initialization,
and helper functions for building indicator dictionaries.
"""
from typing import Dict, Any

from .enums import RejectionReason, ExitReason


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


def track_exit(
    state: Dict[str, Any],
    reason: ExitReason,
    symbol: str = None,
    pnl: float = 0.0
) -> None:
    """
    Track intentional exit for analysis.

    REC-020: Separate exit tracking from rejection tracking.
    """
    if 'exit_counts' not in state:
        state['exit_counts'] = {}
    if 'exit_counts_by_symbol' not in state:
        state['exit_counts_by_symbol'] = {}
    if 'exit_pnl_by_reason' not in state:
        state['exit_pnl_by_reason'] = {}

    reason_key = reason.value
    state['exit_counts'][reason_key] = state['exit_counts'].get(reason_key, 0) + 1

    # Track P&L by exit reason
    state['exit_pnl_by_reason'][reason_key] = state['exit_pnl_by_reason'].get(reason_key, 0) + pnl

    if symbol:
        if symbol not in state['exit_counts_by_symbol']:
            state['exit_counts_by_symbol'][symbol] = {}
        state['exit_counts_by_symbol'][symbol][reason_key] = \
            state['exit_counts_by_symbol'][symbol].get(reason_key, 0) + 1


def build_base_indicators(
    symbol: str,
    status: str,
    state: Dict[str, Any],
    price: float = None
) -> Dict[str, Any]:
    """Build base indicators dict for early returns."""
    return {
        'symbol': symbol,
        'status': status,
        'price': round(price, 8) if price else None,
        'position_usd': round(state.get('position_usd', 0), 4),
        'position_xrp': round(state.get('position_xrp', 0), 4),
        'xrp_accumulated': round(state.get('xrp_accumulated', 0), 4),
        'btc_accumulated': round(state.get('btc_accumulated', 0), 8),
        'consecutive_losses': state.get('consecutive_losses', 0),
        'pnl_symbol': round(state.get('pnl_by_symbol', {}).get(symbol, 0), 8),
        'trades_symbol': state.get('trades_by_symbol', {}).get(symbol, 0),
    }


def initialize_state(state: Dict[str, Any]) -> None:
    """Initialize strategy state."""
    state['initialized'] = True
    state['price_history'] = []
    state['last_signal_time'] = None
    state['trade_count'] = 0
    state['indicators'] = {}

    # Position tracking in USD - REC-002
    state['position_usd'] = 0.0
    state['position_xrp'] = 0.0  # Keep XRP tracking for ratio pair logic

    # Dual-asset accumulation (unique to ratio trading)
    state['xrp_accumulated'] = 0.0
    state['btc_accumulated'] = 0.0

    # REC-016: Enhanced accumulation metrics
    state['xrp_accumulated_value_usd'] = 0.0  # USD value at time of acquisition
    state['btc_accumulated_value_usd'] = 0.0  # USD value at time of acquisition
    state['total_trades_xrp_bought'] = 0
    state['total_trades_btc_bought'] = 0

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

    # REC-020: Exit tracking (separate from rejections)
    state['exit_counts'] = {}
    state['exit_counts_by_symbol'] = {}
    state['exit_pnl_by_reason'] = {}

    # REC-021: Correlation monitoring
    state['btc_price_history'] = []  # BTC/USD price history for correlation
    state['correlation_history'] = []  # Rolling correlation values
    state['correlation_warnings'] = 0  # Count of low correlation warnings
    state['last_btc_price_usd'] = None  # Last BTC price used for conversion

    # REC-037: Correlation trend detection
    state['correlation_slope'] = 0.0  # Current correlation slope
    state['correlation_trend_direction'] = 'stable'  # 'declining', 'stable', 'improving'
    state['correlation_trend_warnings'] = 0  # Count of declining trend warnings

    # Entry tracking with trailing stop support
    state['position_entries'] = {}
    state['highest_price_since_entry'] = {}
    state['lowest_price_since_entry'] = {}

    # Fill history
    state['fills'] = []


def update_price_history(
    data,
    state: Dict[str, Any],
    symbol: str,
    current_price: float,
    max_history: int = 50
) -> list:
    """
    Update price history from candles or current price.

    Returns the updated price history list.
    """
    candles = data.candles_1m.get(symbol, ())

    if candles:
        # Use candle closes for history
        closes = [c.close for c in candles]
        state['price_history'] = closes[-max_history:]
    else:
        # Fall back to current price
        state['price_history'].append(current_price)
        state['price_history'] = state['price_history'][-max_history:]

    return state['price_history']
