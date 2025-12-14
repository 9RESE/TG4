"""
Market Making Strategy - Calculations

Pure calculation functions for market making signals.
All functions are stateless and side-effect free.
"""
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import math

from ws_tester.types import DataSnapshot, OrderbookSnapshot


def calculate_micro_price(ob: OrderbookSnapshot) -> float:
    """
    Calculate volume-weighted micro-price (MM-E01).

    Micro-price provides better price discovery than simple mid-price
    by weighting by order sizes at best bid/ask.

    Formula: micro = (best_bid * ask_size + best_ask * bid_size) / (bid_size + ask_size)
    """
    if not ob.bids or not ob.asks:
        return ob.mid

    best_bid, bid_size = ob.bids[0]
    best_ask, ask_size = ob.asks[0]

    total_size = bid_size + ask_size
    if total_size <= 0:
        return ob.mid

    micro_price = (best_bid * ask_size + best_ask * bid_size) / total_size
    return micro_price


def calculate_optimal_spread(
    volatility_pct: float,
    gamma: float,
    kappa: float,
    time_horizon: float = 1.0
) -> float:
    """
    Calculate Avellaneda-Stoikov optimal spread (MM-E02).

    Formula: optimal_spread = gamma * sigma^2 * T + (2/gamma) * ln(1 + gamma/kappa)

    Args:
        volatility_pct: Price volatility as percentage
        gamma: Risk aversion parameter
        kappa: Market liquidity parameter
        time_horizon: Time horizon (normalized, default 1.0)

    Returns:
        Optimal spread as percentage
    """
    if gamma <= 0 or kappa <= 0:
        return 0.0

    sigma = volatility_pct / 100  # Convert to decimal
    sigma_sq = sigma ** 2

    # A-S optimal spread formula
    inventory_term = gamma * sigma_sq * time_horizon * 100  # Convert back to %
    liquidity_term = (2 / gamma) * math.log(1 + gamma / kappa) * 100

    return inventory_term + liquidity_term


def calculate_reservation_price(
    mid_price: float,
    inventory: float,
    max_inventory: float,
    gamma: float,
    volatility_pct: float
) -> float:
    """
    Calculate Avellaneda-Stoikov reservation price.

    The reservation price adjusts the mid price based on inventory risk:
    r = s - q * gamma * sigma^2

    Where:
    - s: mid price
    - q: normalized inventory (-1 to 1)
    - gamma: risk aversion parameter
    - sigma^2: variance of price (volatility squared)

    Returns:
        Adjusted reservation price
    """
    if max_inventory <= 0:
        return mid_price

    # Normalize inventory to -1 to 1 range
    q = inventory / max_inventory

    # Convert volatility percentage to decimal variance
    sigma_sq = (volatility_pct / 100) ** 2

    # Calculate reservation price
    # Positive inventory (long) -> lower reservation price (favor selling)
    # Negative inventory (short) -> higher reservation price (favor buying)
    reservation = mid_price * (1 - q * gamma * sigma_sq * 100)

    return reservation


def calculate_trailing_stop(
    entry_price: float,
    highest_price: float,
    side: str,
    activation_pct: float,
    trail_distance_pct: float
) -> Optional[float]:
    """
    Calculate trailing stop level.

    Args:
        entry_price: Original entry price
        highest_price: Highest price since entry (for longs) or lowest (for shorts)
        side: 'long' or 'short'
        activation_pct: Minimum profit % to activate trailing
        trail_distance_pct: Distance from high/low to trail

    Returns:
        Trailing stop price or None if not activated
    """
    if side == 'long':
        # Long: profit when price increases
        profit_pct = (highest_price - entry_price) / entry_price * 100
        if profit_pct >= activation_pct:
            return highest_price * (1 - trail_distance_pct / 100)
    elif side == 'short':
        # Short: profit when price decreases (highest_price is actually lowest)
        profit_pct = (entry_price - highest_price) / entry_price * 100
        if profit_pct >= activation_pct:
            return highest_price * (1 + trail_distance_pct / 100)

    return None


def calculate_volatility(candles, lookback: int = 20) -> float:
    """
    Calculate price volatility from candle closes.

    Returns volatility as a percentage (std dev of returns * 100).
    """
    if not candles or len(candles) < lookback + 1:
        return 0.0

    closes = [c.close for c in candles[-(lookback + 1):]]
    if len(closes) < 2:
        return 0.0

    # Calculate returns
    returns = [(closes[i] - closes[i - 1]) / closes[i - 1]
               for i in range(1, len(closes)) if closes[i - 1] != 0]

    if not returns:
        return 0.0

    # Calculate standard deviation of returns
    mean_return = sum(returns) / len(returns)
    variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
    std_dev = variance ** 0.5

    return std_dev * 100  # Return as percentage


def get_trade_flow_imbalance(data: DataSnapshot, symbol: str, n_trades: int = 50) -> float:
    """Get trade flow imbalance from recent trades."""
    return data.get_trade_imbalance(symbol, n_trades)


def check_fee_profitability(
    spread_pct: float,
    fee_rate: float,
    min_profit_pct: float
) -> Tuple[bool, float]:
    """
    Check if trade is profitable after fees (MM-E03).

    Args:
        spread_pct: Current spread as percentage
        fee_rate: Fee per trade (e.g., 0.001 for 0.1%)
        min_profit_pct: Minimum required profit after fees

    Returns:
        Tuple of (is_profitable, expected_profit_pct)
    """
    # Round-trip fees (entry + exit)
    round_trip_fee_pct = fee_rate * 2 * 100  # Convert to percentage

    # Expected profit from spread capture (we capture half the spread)
    expected_capture = spread_pct / 2

    # Net profit after fees
    net_profit_pct = expected_capture - round_trip_fee_pct

    is_profitable = net_profit_pct >= min_profit_pct

    return is_profitable, net_profit_pct


def check_position_decay(
    position_entry: Dict[str, Any],
    current_time: datetime,
    max_age_seconds: float,
    tp_multiplier: float
) -> Tuple[bool, float]:
    """
    Check if position is stale and should have adjusted TP (MM-E04).

    Args:
        position_entry: Position entry data with 'entry_time'
        current_time: Current timestamp
        max_age_seconds: Maximum age before decay kicks in
        tp_multiplier: Multiplier to reduce TP (e.g., 0.5 = 50% of original)

    Returns:
        Tuple of (is_stale, adjusted_tp_multiplier)
    """
    entry_time = position_entry.get('entry_time')
    if not entry_time:
        return False, 1.0

    age_seconds = (current_time - entry_time).total_seconds()

    if age_seconds > max_age_seconds:
        # Position is stale - reduce TP requirement
        return True, tp_multiplier

    return False, 1.0


def get_xrp_usdt_price(data: DataSnapshot, config: Dict[str, Any]) -> float:
    """
    Get XRP/USDT price with configurable fallback (MM-011).

    Args:
        data: Market data snapshot
        config: Strategy configuration

    Returns:
        XRP/USDT price
    """
    price = data.prices.get('XRP/USDT')
    if price and price > 0:
        return price

    # Use configurable fallback instead of hardcoded value
    return config.get('fallback_xrp_usdt', 2.50)


def calculate_effective_thresholds(
    config: Dict[str, Any],
    symbol: str,
    volatility: float
) -> Tuple[float, float, float]:
    """
    Calculate volatility-adjusted thresholds (MM-010 refactor).

    Returns:
        Tuple of (effective_min_spread, effective_imbalance_threshold, vol_multiplier)
    """
    from .config import get_symbol_config, SYMBOL_CONFIGS

    min_spread = get_symbol_config(symbol, config, 'min_spread_pct')
    imbalance_threshold = get_symbol_config(symbol, config, 'imbalance_threshold')
    base_vol = config.get('base_volatility_pct', 0.5)

    vol_multiplier = 1.0
    if base_vol > 0 and volatility > 0:
        vol_ratio = volatility / base_vol
        if vol_ratio > 1.0:
            vol_multiplier = min(vol_ratio, config.get('volatility_threshold_mult', 1.5))

    effective_threshold = imbalance_threshold * vol_multiplier
    effective_min_spread = min_spread * vol_multiplier

    return effective_min_spread, effective_threshold, vol_multiplier
