"""
Whale Sentiment Strategy - Risk Management

Contains fee profitability checks, position limit checks, correlation management,
and circuit breaker logic.

Note: Contrarian strategy has stricter circuit breaker (2 consecutive losses)
due to higher risk of consecutive losses during trending markets.
"""
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

from .indicators import calculate_rolling_correlation


def check_fee_profitability(
    expected_move_pct: float,
    fee_rate: float,
    min_profit_pct: float
) -> Tuple[bool, float]:
    """
    Check if trade is profitable after fees.

    Transaction cost erosion is a critical risk:
    - 0.1% fee x 2 = 0.2% per round trip
    - Minimum TP target should account for fees

    Args:
        expected_move_pct: Expected price move percentage (e.g., 5.0 for 5.0%)
        fee_rate: Fee rate per trade (e.g., 0.001 for 0.1%)
        min_profit_pct: Minimum acceptable profit percentage after fees

    Returns:
        Tuple of (is_profitable, net_profit_pct)
    """
    # Round trip fee in percentage terms
    round_trip_fee_pct = fee_rate * 2 * 100

    # Net profit after fees
    net_profit_pct = expected_move_pct - round_trip_fee_pct

    is_profitable = net_profit_pct >= min_profit_pct

    return is_profitable, net_profit_pct


def check_circuit_breaker(
    state: Dict[str, Any],
    current_time: datetime,
    max_losses: int,
    cooldown_minutes: float
) -> bool:
    """
    Check if circuit breaker is active.

    Contrarian strategies are prone to consecutive losses during trending
    markets. Circuit breaker triggers after max_consecutive_losses and pauses
    trading for cooldown_minutes.

    Default for whale sentiment: 2 losses, 45 min cooldown (stricter than most)

    Args:
        state: Strategy state dict
        current_time: Current timestamp
        max_losses: Maximum consecutive losses before circuit breaker
        cooldown_minutes: Cooldown duration in minutes

    Returns:
        True if circuit breaker is active and trading should be paused
    """
    consecutive_losses = state.get('consecutive_losses', 0)

    # Not enough losses to trigger
    if consecutive_losses < max_losses:
        return False

    # Check if we're in cooldown period
    breaker_time = state.get('circuit_breaker_time')
    if breaker_time is None:
        return False

    elapsed_minutes = (current_time - breaker_time).total_seconds() / 60

    # Cooldown period expired - reset and allow trading
    if elapsed_minutes >= cooldown_minutes:
        state['consecutive_losses'] = 0
        state['circuit_breaker_time'] = None
        return False

    # Still in cooldown
    return True


def calculate_pair_correlation(
    candles_a,
    candles_b,
    window: int = 20
) -> Optional[float]:
    """
    Calculate rolling correlation between two price series.

    Args:
        candles_a: Candles for symbol A
        candles_b: Candles for symbol B
        window: Number of candles for correlation calculation

    Returns:
        Correlation coefficient (-1 to +1) or None if insufficient data
    """
    if not candles_a or not candles_b:
        return None

    if len(candles_a) < window + 1 or len(candles_b) < window + 1:
        return None

    # Extract closing prices
    prices_a = [c.close for c in candles_a]
    prices_b = [c.close for c in candles_b]

    return calculate_rolling_correlation(prices_a, prices_b, window)


def check_real_correlation(
    state: Dict[str, Any],
    symbol: str,
    direction: str,
    candles_by_symbol: Dict[str, tuple],
    config: Dict[str, Any]
) -> Tuple[bool, float, Dict[str, Any]]:
    """
    Check if real-time correlation blocks trade entry.

    Args:
        state: Strategy state dict
        symbol: Symbol being traded
        direction: Trade direction ('buy' or 'short')
        candles_by_symbol: Dict of candles by symbol
        config: Strategy configuration

    Returns:
        Tuple of (can_trade, correlation_adjustment, correlation_info)
    """
    correlation_info = {
        'correlations': {},
        'high_correlation_pairs': [],
        'adjustment_factor': 1.0,
        'blocked': False,
    }

    if not config.get('use_real_correlation', True):
        return True, 1.0, correlation_info

    window = config.get('correlation_window', 20)
    block_threshold = config.get('correlation_block_threshold', 0.85)

    # Get existing positions
    position_entries = state.get('position_entries', {})

    # Check correlation with each existing position
    for existing_symbol, pos_data in position_entries.items():
        if existing_symbol == symbol:
            continue

        existing_side = pos_data.get('side', '')

        # Get candles for both symbols
        candles_a = candles_by_symbol.get(symbol, ())
        candles_b = candles_by_symbol.get(existing_symbol, ())

        correlation = calculate_pair_correlation(candles_a, candles_b, window)

        if correlation is not None:
            correlation_info['correlations'][f"{symbol}/{existing_symbol}"] = round(correlation, 3)

            # Check if highly correlated in same direction
            same_direction = (
                (direction == 'buy' and existing_side == 'long') or
                (direction == 'short' and existing_side == 'short')
            )

            if abs(correlation) >= block_threshold and same_direction:
                correlation_info['high_correlation_pairs'].append({
                    'pair': f"{symbol}/{existing_symbol}",
                    'correlation': round(correlation, 3),
                    'same_direction': True,
                })
                correlation_info['blocked'] = True

            # Reduce size based on correlation strength
            if same_direction and correlation > 0.5:
                # Linear reduction: 50% correlation = 100% size, 100% correlation = 50% size
                reduction = 1.0 - (correlation - 0.5) * (0.5 / 0.5)
                correlation_info['adjustment_factor'] = min(
                    correlation_info['adjustment_factor'],
                    max(0.5, reduction)
                )

    return not correlation_info['blocked'], correlation_info['adjustment_factor'], correlation_info


def check_correlation_exposure(
    state: Dict[str, Any],
    symbol: str,
    direction: str,
    size: float,
    config: Dict[str, Any]
) -> Tuple[bool, float]:
    """
    Check and adjust for cross-pair correlation.

    XRP-BTC correlation (~0.84) means simultaneous signals on XRP/USDT
    and BTC/USDT are correlated. Need total exposure limits.

    Note: Short exposure limit is lower due to squeeze risk.

    Args:
        state: Strategy state dict
        symbol: Symbol being traded
        direction: Trade direction ('buy' or 'short')
        size: Requested position size
        config: Strategy configuration

    Returns:
        Tuple of (can_trade, adjusted_size)
    """
    if not config.get('use_correlation_management', True):
        return True, size

    max_long = config.get('max_total_long_exposure', 100.0)
    max_short = config.get('max_total_short_exposure', 75.0)  # Lower for shorts
    same_dir_mult = config.get('same_direction_size_mult', 0.75)

    position_by_symbol = state.get('position_by_symbol', {})
    position_entries = state.get('position_entries', {})

    # Calculate current exposure
    total_long = 0.0
    total_short = 0.0
    other_symbols_same_direction = False

    for sym, pos_data in position_entries.items():
        if sym == symbol:
            continue

        pos_size = position_by_symbol.get(sym, 0)
        pos_side = pos_data.get('side', '')

        if pos_side == 'long':
            total_long += pos_size
            if direction == 'buy':
                other_symbols_same_direction = True
        elif pos_side == 'short':
            total_short += pos_size
            if direction == 'short':
                other_symbols_same_direction = True

    # Reduce size if other symbols have same direction
    adjusted_size = size
    if other_symbols_same_direction:
        adjusted_size = size * same_dir_mult

    # Check total exposure limits
    if direction == 'buy':
        if total_long + adjusted_size > max_long:
            available = max(0, max_long - total_long)
            adjusted_size = min(adjusted_size, available)
    elif direction == 'short':
        if total_short + adjusted_size > max_short:
            available = max(0, max_short - total_short)
            adjusted_size = min(adjusted_size, available)

    min_trade = config.get('min_trade_size_usd', 5.0)
    can_trade = adjusted_size >= min_trade

    return can_trade, adjusted_size


def check_position_limits(
    state: Dict[str, Any],
    symbol: str,
    requested_size: float,
    config: Dict[str, Any]
) -> Tuple[bool, float, str]:
    """
    Check position limits and calculate available size.

    Enforces both total and per-symbol position limits.

    Args:
        state: Strategy state dict
        symbol: Symbol being traded
        requested_size: Requested position size
        config: Strategy configuration

    Returns:
        Tuple of (can_trade, available_size, limit_reason)
    """
    current_position = state.get('position_size', 0)
    current_position_symbol = state.get('position_by_symbol', {}).get(symbol, 0)

    max_position = config.get('max_position_usd', 150.0)
    max_position_symbol = config.get('max_position_per_symbol_usd', 75.0)
    min_trade = config.get('min_trade_size_usd', 5.0)

    # Check total position limit
    if current_position >= max_position:
        return False, 0.0, 'total_limit'

    # Check per-symbol position limit
    if current_position_symbol >= max_position_symbol:
        return False, 0.0, 'symbol_limit'

    # Calculate available size respecting both limits
    available_total = max_position - current_position
    available_symbol = max_position_symbol - current_position_symbol
    available = min(available_total, available_symbol)

    actual_size = min(requested_size, available)

    if actual_size < min_trade:
        return False, actual_size, 'min_trade'

    return True, actual_size, 'ok'


def calculate_position_age(
    state: Dict[str, Any],
    symbol: str,
    current_time: datetime
) -> float:
    """
    Calculate how long a position has been held.

    Args:
        state: Strategy state dict
        symbol: Symbol to check
        current_time: Current timestamp

    Returns:
        Position age in seconds, or 0 if no position
    """
    pos_entry = state.get('position_entries', {}).get(symbol)
    if not pos_entry:
        return 0.0

    entry_time = pos_entry.get('entry_time')
    if not entry_time:
        return 0.0

    return (current_time - entry_time).total_seconds()


def calculate_position_pnl(
    state: Dict[str, Any],
    symbol: str,
    current_price: float
) -> Tuple[float, float]:
    """
    Calculate current position PnL.

    Args:
        state: Strategy state dict
        symbol: Symbol to check
        current_price: Current market price

    Returns:
        Tuple of (pnl_pct, pnl_usd) or (0, 0) if no position
    """
    pos_entry = state.get('position_entries', {}).get(symbol)
    if not pos_entry:
        return 0.0, 0.0

    entry_price = pos_entry.get('entry_price', 0)
    if entry_price <= 0:
        return 0.0, 0.0

    position_size = state.get('position_by_symbol', {}).get(symbol, 0)

    if pos_entry['side'] == 'long':
        pnl_pct = (current_price - entry_price) / entry_price * 100
    else:  # short
        pnl_pct = (entry_price - current_price) / entry_price * 100

    # Approximate USD PnL (position is in USD notional)
    pnl_usd = position_size * (pnl_pct / 100)

    return pnl_pct, pnl_usd


def update_position_extremes(
    state: Dict[str, Any],
    symbol: str,
    current_price: float
) -> None:
    """
    Update highest/lowest price for trailing stop calculation.

    Args:
        state: Strategy state dict
        symbol: Symbol to update
        current_price: Current market price
    """
    pos_entry = state.get('position_entries', {}).get(symbol)
    if not pos_entry:
        return

    # Update highest price (for long trailing stop)
    if current_price > pos_entry.get('highest_price', 0):
        pos_entry['highest_price'] = current_price

    # Update lowest price (for short trailing stop)
    if current_price < pos_entry.get('lowest_price', float('inf')):
        pos_entry['lowest_price'] = current_price
