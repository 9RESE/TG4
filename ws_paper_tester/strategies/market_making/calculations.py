"""
Market Making Strategy - Calculations

Pure calculation functions for market making signals.
All functions are stateless and side-effect free.

Version History:
v2.2.0 (2025-12-14) - Session Awareness & Correlation Monitoring:
- get_trading_session: Classify current hour into trading session (REC-002)
- get_session_multipliers: Get threshold/size multipliers for session (REC-002)
- calculate_rolling_correlation: Pearson correlation for price series (REC-003)
- check_correlation_pause: Determine if XRP/BTC should pause (REC-003)

v2.0.0 additions:
- get_volatility_regime: Volatility regime classification (MM-H01)
- calculate_trend_slope: Linear regression slope for trending detection (MM-H02)
- check_circuit_breaker: Circuit breaker status check (MM-C01)
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


def get_volatility_regime(
    volatility_pct: float,
    config: Dict[str, Any]
) -> Tuple[str, float, float]:
    """
    Classify volatility into regime (MM-H01, Guide v2.0 Section 15).

    Regimes:
    - LOW: < 0.3% - Tighter thresholds, normal size
    - MEDIUM: 0.3% - 0.8% - Baseline
    - HIGH: 0.8% - 1.5% - Wider thresholds, reduced size
    - EXTREME: > 1.5% - PAUSE TRADING

    Args:
        volatility_pct: Current price volatility as percentage
        config: Strategy configuration

    Returns:
        Tuple of (regime_name, threshold_mult, size_mult)
    """
    low_thresh = config.get('regime_low_threshold', 0.3)
    med_thresh = config.get('regime_medium_threshold', 0.8)
    high_thresh = config.get('regime_high_threshold', 1.5)

    low_mult = config.get('regime_low_threshold_mult', 0.9)
    high_mult = config.get('regime_high_threshold_mult', 1.3)
    high_size = config.get('regime_high_size_mult', 0.7)

    if volatility_pct < low_thresh:
        return "LOW", low_mult, 1.0
    elif volatility_pct < med_thresh:
        return "MEDIUM", 1.0, 1.0
    elif volatility_pct < high_thresh:
        return "HIGH", high_mult, high_size
    else:
        return "EXTREME", 2.0, 0.0  # EXTREME = pause (size=0)


def calculate_trend_slope(
    candles,
    lookback: int = 20
) -> Tuple[float, bool]:
    """
    Calculate price trend using linear regression slope (MM-H02).

    Uses simple linear regression on closing prices to determine trend direction
    and strength.

    Args:
        candles: List of candle data
        lookback: Number of candles to analyze

    Returns:
        Tuple of (slope_pct, is_trending)
        - slope_pct: Price change per candle as percentage
        - is_trending: True if absolute slope > threshold
    """
    if not candles or len(candles) < lookback:
        return 0.0, False

    closes = [c.close for c in candles[-lookback:]]
    if len(closes) < 2:
        return 0.0, False

    # Simple linear regression: y = mx + b
    n = len(closes)
    x_mean = (n - 1) / 2
    y_mean = sum(closes) / n

    # Calculate slope
    numerator = sum((i - x_mean) * (closes[i] - y_mean) for i in range(n))
    denominator = sum((i - x_mean) ** 2 for i in range(n))

    if denominator == 0:
        return 0.0, False

    slope = numerator / denominator

    # Convert to percentage change per candle
    if y_mean != 0:
        slope_pct = (slope / y_mean) * 100
    else:
        slope_pct = 0.0

    return slope_pct, True


def check_circuit_breaker(
    state: Dict[str, Any],
    config: Dict[str, Any],
    current_time: datetime
) -> Tuple[bool, Optional[float]]:
    """
    Check if circuit breaker is active (MM-C01, Guide v2.0 Section 16).

    Circuit breaker triggers after consecutive losses and pauses trading
    for a cooldown period.

    Args:
        state: Strategy state dict
        config: Strategy configuration
        current_time: Current timestamp

    Returns:
        Tuple of (is_triggered, seconds_remaining)
        - is_triggered: True if circuit breaker is active
        - seconds_remaining: Seconds until reset (None if not triggered)
    """
    if not config.get('use_circuit_breaker', True):
        return False, None

    cb_time = state.get('circuit_breaker_triggered_time')
    if cb_time is None:
        return False, None

    cooldown_minutes = config.get('circuit_breaker_cooldown_minutes', 15)
    cooldown_seconds = cooldown_minutes * 60

    elapsed = (current_time - cb_time).total_seconds()
    if elapsed < cooldown_seconds:
        remaining = cooldown_seconds - elapsed
        return True, remaining
    else:
        # Cooldown expired - reset circuit breaker
        state['circuit_breaker_triggered_time'] = None
        state['consecutive_losses'] = 0
        return False, None


def update_circuit_breaker_on_fill(
    state: Dict[str, Any],
    config: Dict[str, Any],
    pnl: float,
    timestamp: datetime
) -> bool:
    """
    Update circuit breaker state on fill (MM-C01).

    Args:
        state: Strategy state dict
        config: Strategy configuration
        pnl: Profit/loss from the fill
        timestamp: Fill timestamp

    Returns:
        True if circuit breaker was just triggered
    """
    if not config.get('use_circuit_breaker', True):
        return False

    if 'consecutive_losses' not in state:
        state['consecutive_losses'] = 0

    if pnl < 0:
        # Loss - increment counter
        state['consecutive_losses'] = state.get('consecutive_losses', 0) + 1

        max_losses = config.get('max_consecutive_losses', 3)
        if state['consecutive_losses'] >= max_losses:
            # Trigger circuit breaker
            state['circuit_breaker_triggered_time'] = timestamp
            state['circuit_breaker_trigger_count'] = state.get('circuit_breaker_trigger_count', 0) + 1
            return True
    else:
        # Win - reset counter
        state['consecutive_losses'] = 0

    return False


# =============================================================================
# v2.2.0: Session Awareness (REC-002, Guide v2.0 Section 20)
# =============================================================================

def get_trading_session(hour_utc: int) -> str:
    """
    Classify current UTC hour into trading session (REC-002).

    Trading sessions based on global forex market activity:
    - ASIA: 00:00-08:00 UTC - Lower liquidity, wider spreads
    - EUROPE: 08:00-14:00 UTC - Increasing volume
    - US_EUROPE_OVERLAP: 14:00-17:00 UTC - Highest activity
    - US: 17:00-22:00 UTC - High volume, often directional
    - OFF_HOURS: 22:00-00:00 UTC - Low liquidity

    Args:
        hour_utc: Current hour in UTC (0-23)

    Returns:
        Session name: 'ASIA', 'EUROPE', 'US_EUROPE_OVERLAP', 'US', or 'OFF_HOURS'
    """
    if 0 <= hour_utc < 8:
        return "ASIA"
    elif 8 <= hour_utc < 14:
        return "EUROPE"
    elif 14 <= hour_utc < 17:
        return "US_EUROPE_OVERLAP"
    elif 17 <= hour_utc < 22:
        return "US"
    else:
        return "OFF_HOURS"


def get_session_multipliers(
    hour_utc: int,
    config: Dict[str, Any]
) -> Tuple[float, float, str]:
    """
    Get threshold and size multipliers for current trading session (REC-002).

    Session adjustments optimize trading for market conditions:
    - Asia: Conservative (wider thresholds, smaller size)
    - Europe/US: Baseline
    - Overlap: Aggressive (tighter thresholds, larger size)
    - Off-hours: Very conservative

    Args:
        hour_utc: Current hour in UTC (0-23)
        config: Strategy configuration

    Returns:
        Tuple of (threshold_mult, size_mult, session_name)
    """
    if not config.get('use_session_awareness', True):
        return 1.0, 1.0, "DISABLED"

    session = get_trading_session(hour_utc)

    if session == "ASIA":
        return (
            config.get('session_asia_threshold_mult', 1.2),
            config.get('session_asia_size_mult', 0.8),
            session
        )
    elif session == "US_EUROPE_OVERLAP":
        return (
            config.get('session_overlap_threshold_mult', 0.85),
            config.get('session_overlap_size_mult', 1.1),
            session
        )
    elif session == "OFF_HOURS":
        return (
            config.get('session_off_hours_threshold_mult', 1.3),
            config.get('session_off_hours_size_mult', 0.6),
            session
        )
    else:
        # EUROPE, US - baseline
        return 1.0, 1.0, session


# =============================================================================
# v2.2.0: Correlation Monitoring (REC-003, Guide v2.0 Section 24)
# =============================================================================

def calculate_rolling_correlation(
    prices_a: list,
    prices_b: list,
    window: int = 20
) -> Optional[float]:
    """
    Calculate Pearson correlation coefficient between two price series (REC-003).

    Used for XRP/BTC correlation monitoring to detect when the
    dual-accumulation strategy may underperform.

    Formula: r = Cov(A,B) / (StdDev(A) * StdDev(B))

    Args:
        prices_a: First price series (e.g., XRP/USDT closes)
        prices_b: Second price series (e.g., BTC/USDT closes)
        window: Lookback window for correlation calculation

    Returns:
        Correlation coefficient (-1.0 to 1.0) or None if insufficient data
    """
    if len(prices_a) < window or len(prices_b) < window:
        return None

    # Use most recent 'window' prices
    a = prices_a[-window:]
    b = prices_b[-window:]

    # Calculate means
    mean_a = sum(a) / len(a)
    mean_b = sum(b) / len(b)

    # Calculate covariance and standard deviations
    covariance = sum((a[i] - mean_a) * (b[i] - mean_b) for i in range(window)) / window
    var_a = sum((x - mean_a) ** 2 for x in a) / window
    var_b = sum((x - mean_b) ** 2 for x in b) / window

    std_a = var_a ** 0.5
    std_b = var_b ** 0.5

    # Avoid division by zero
    if std_a == 0 or std_b == 0:
        return None

    correlation = covariance / (std_a * std_b)

    # Clamp to [-1, 1] to handle floating point errors
    return max(-1.0, min(1.0, correlation))


def check_correlation_pause(
    correlation: Optional[float],
    config: Dict[str, Any]
) -> Tuple[bool, bool, Optional[float]]:
    """
    Check if XRP/BTC trading should pause due to low correlation (REC-003).

    Correlation breakdown detection:
    - Below warning threshold: Log warning, continue trading
    - Below pause threshold: Pause XRP/BTC trading

    Args:
        correlation: Current XRP-BTC correlation (-1.0 to 1.0) or None
        config: Strategy configuration

    Returns:
        Tuple of (should_pause, should_warn, correlation_value)
    """
    if not config.get('use_correlation_monitoring', True):
        return False, False, correlation

    if correlation is None:
        # Insufficient data - don't pause, but flag as unknown
        return False, False, None

    warning_threshold = config.get('correlation_warning_threshold', 0.6)
    pause_threshold = config.get('correlation_pause_threshold', 0.5)

    should_warn = correlation < warning_threshold
    should_pause = correlation < pause_threshold

    return should_pause, should_warn, correlation


def get_correlation_prices(
    data: 'DataSnapshot',
    lookback: int = 20
) -> Tuple[list, list]:
    """
    Extract price series for XRP and BTC from candle data (REC-003).

    Args:
        data: Market data snapshot
        lookback: Number of candles to extract

    Returns:
        Tuple of (xrp_prices, btc_prices) as lists of closing prices
    """
    xrp_candles = data.candles_1m.get('XRP/USDT', ())
    btc_candles = data.candles_1m.get('BTC/USDT', ())

    xrp_prices = [c.close for c in xrp_candles[-(lookback + 1):]] if xrp_candles else []
    btc_prices = [c.close for c in btc_candles[-(lookback + 1):]] if btc_candles else []

    return xrp_prices, btc_prices
