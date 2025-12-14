"""
Mean Reversion Strategy v4.0.0

Trades price deviations from moving average and VWAP.
Enhanced with volatility regimes, circuit breaker, multi-symbol support,
trend filtering, trailing stops, position decay, and comprehensive risk management.

Version History:
- 1.0.0: Initial implementation
- 1.0.1: Fixed RSI edge case (LOW-007)
- 2.0.0: Major refactor per mean-reversion-strategy-review-v1.0.md
         - REC-001: Fixed R:R ratio to 1:1 (0.5%/0.5%)
         - REC-002: Added multi-symbol support (XRP/USDT, BTC/USDT)
         - REC-003: Added cooldown mechanisms
         - REC-004: Added volatility regime classification
         - REC-005: Added circuit breaker protection
         - REC-006: Added per-pair PnL tracking
         - REC-007: Added configuration validation
         - REC-008: Added trade flow confirmation
         - Finding #6: Added on_stop() callback
         - Code cleanup and optimization
- 3.0.0: Major enhancement per mean-reversion-strategy-review-v3.1.md
         - REC-001: Added XRP/BTC ratio trading pair
         - REC-002: Fixed hardcoded max_losses in on_fill
         - REC-003: Added wider stop-loss option research support
         - REC-004: Added optional trend filter
         - REC-006: Added trailing stops
         - REC-007: Added position decay
         - Finding #4: Refactored _evaluate_symbol for lower complexity
- 4.0.0: Optimization per mean-reversion-deep-review-v4.0.md
         - REC-001: Trailing stops disabled by default (research: fixed TP better for MR)
         - REC-002: Extended position decay timing (15 min start, 5 min intervals)
         - REC-003: Added trend confirmation period (reduce false positives in choppy markets)
         - REC-005: Added XRP/BTC correlation monitoring (rolling Pearson coefficient)
         - Gentler decay multipliers: [1.0, 0.85, 0.7, 0.5]
         - Wider trailing distance: 0.3% (if enabled)
         - Higher trailing activation: 0.4% (if enabled)
"""
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum, auto

try:
    from ws_tester.types import DataSnapshot, Signal, Candle
except ImportError:
    DataSnapshot = None
    Signal = None
    Candle = None


# =============================================================================
# REQUIRED: Strategy Metadata
# =============================================================================
STRATEGY_NAME = "mean_reversion"
STRATEGY_VERSION = "4.0.0"
SYMBOLS = ["XRP/USDT", "BTC/USDT", "XRP/BTC"]


# =============================================================================
# Enums for Type Safety
# =============================================================================
class VolatilityRegime(Enum):
    """Volatility regime classification."""
    LOW = auto()       # volatility < low_threshold
    MEDIUM = auto()    # low_threshold - medium_threshold
    HIGH = auto()      # medium_threshold - high_threshold
    EXTREME = auto()   # > high_threshold


class RejectionReason(Enum):
    """Signal rejection reasons for tracking."""
    CIRCUIT_BREAKER = "circuit_breaker"
    TIME_COOLDOWN = "time_cooldown"
    WARMING_UP = "warming_up"
    REGIME_PAUSE = "regime_pause"
    NO_PRICE_DATA = "no_price_data"
    MAX_POSITION = "max_position"
    INSUFFICIENT_SIZE = "insufficient_size"
    TRADE_FLOW_NOT_ALIGNED = "trade_flow_not_aligned"
    TRENDING_MARKET = "trending_market"  # REC-004 (v3.0.0)
    NO_SIGNAL_CONDITIONS = "no_signal_conditions"


# =============================================================================
# REQUIRED: Default Configuration
# =============================================================================
CONFIG = {
    # ==========================================================================
    # Core Mean Reversion Parameters
    # ==========================================================================
    'lookback_candles': 20,           # Candles for MA calculation
    'deviation_threshold': 0.5,       # % deviation to trigger (0.5%)
    'bb_period': 20,                  # Bollinger Bands period
    'bb_std_dev': 2.0,                # Bollinger Bands standard deviations
    'rsi_period': 14,                 # RSI calculation period

    # ==========================================================================
    # Position Sizing
    # ==========================================================================
    'position_size_usd': 20.0,        # Size per trade in USD
    'max_position': 50.0,             # Max position size in USD
    'min_trade_size_usd': 5.0,        # Minimum USD per trade

    # ==========================================================================
    # RSI Thresholds
    # ==========================================================================
    'rsi_oversold': 35,               # RSI oversold level
    'rsi_overbought': 65,             # RSI overbought level

    # ==========================================================================
    # Risk Management - REC-001: Fixed to 1:1 R:R
    # ==========================================================================
    'take_profit_pct': 0.5,           # Take profit at 0.5%
    'stop_loss_pct': 0.5,             # Stop loss at 0.5%

    # ==========================================================================
    # Cooldown Mechanisms - REC-003
    # ==========================================================================
    'cooldown_seconds': 10.0,         # Min time between signals

    # ==========================================================================
    # Volatility Parameters - REC-004
    # ==========================================================================
    'use_volatility_regimes': True,   # Enable regime-based adjustments
    'base_volatility_pct': 0.5,       # Baseline volatility for scaling
    'volatility_lookback': 20,        # Candles for volatility calculation
    'regime_low_threshold': 0.3,      # Below = LOW regime
    'regime_medium_threshold': 0.8,   # Below = MEDIUM regime
    'regime_high_threshold': 1.5,     # Below = HIGH regime, above = EXTREME
    'regime_extreme_pause': True,     # Pause trading in EXTREME regime

    # ==========================================================================
    # Circuit Breaker - REC-005
    # ==========================================================================
    'use_circuit_breaker': True,      # Enable consecutive loss circuit breaker
    'max_consecutive_losses': 3,      # Max losses before cooldown
    'circuit_breaker_minutes': 15,    # Cooldown after max losses

    # ==========================================================================
    # Trade Flow Confirmation - REC-008
    # ==========================================================================
    'use_trade_flow_confirmation': True,
    'trade_flow_threshold': 0.10,     # Minimum trade flow alignment

    # ==========================================================================
    # VWAP Parameters
    # ==========================================================================
    'vwap_lookback': 50,              # Trades for VWAP calculation
    'vwap_deviation_threshold': 0.3,  # % deviation for VWAP signal
    'vwap_size_multiplier': 0.5,      # Position size multiplier for VWAP signals

    # ==========================================================================
    # Trend Filter - REC-004 (v3.0.0), Updated per v4.0 Review REC-003
    # Added confirmation period to reduce false positives in choppy markets
    # ==========================================================================
    'use_trend_filter': True,         # Enable trend filtering
    'trend_sma_period': 50,           # Lookback for trend SMA
    'trend_slope_threshold': 0.05,    # Min slope % to consider trending
    'trend_confirmation_periods': 3,  # Consecutive trending evals before rejection (REC-003)

    # ==========================================================================
    # Trailing Stops - REC-006 (v3.0.0), Updated per v4.0 Review REC-001
    # Research: Fixed TP better for mean reversion than trailing stops
    # Trailing stops designed for trend-following; may exit prematurely during MR
    # ==========================================================================
    'use_trailing_stop': False,       # Disabled by default (per REC-001 research)
    'trailing_activation_pct': 0.4,   # Activate at 0.4% profit (increased for MR)
    'trailing_distance_pct': 0.3,     # Trail 0.3% from high/low (wider per REC-001)

    # ==========================================================================
    # Position Decay - REC-007 (v3.0.0), Updated per v4.0 Review REC-002
    # Research: Crypto mean reversion needs multi-candle periods for completion
    # ==========================================================================
    'use_position_decay': True,       # Enable time-based TP reduction
    'decay_start_minutes': 15.0,      # Start reducing TP after 15 min (was 3, per REC-002)
    'decay_interval_minutes': 5.0,    # Reduce TP every 5 min (was 1, per REC-002)
    'decay_multipliers': [1.0, 0.85, 0.7, 0.5],  # Gentler reduction (per REC-002)

    # ==========================================================================
    # XRP/BTC Correlation Monitoring - REC-005 (v4.0.0)
    # Research: XRP correlation with BTC declining (24.86% over 90 days)
    # Monitor for adaptive ratio trading decisions
    # ==========================================================================
    'use_correlation_monitoring': True,  # Enable XRP/BTC correlation tracking
    'correlation_lookback': 50,          # Candles for correlation calculation
    'correlation_warn_threshold': 0.5,   # Log warning if correlation drops below

    # ==========================================================================
    # Rejection Tracking
    # ==========================================================================
    'track_rejections': True,         # Enable rejection tracking
}

# =============================================================================
# Per-Symbol Configurations - REC-002
# =============================================================================
SYMBOL_CONFIGS = {
    'XRP/USDT': {
        'deviation_threshold': 0.5,   # % deviation threshold
        'rsi_oversold': 35,
        'rsi_overbought': 65,
        'position_size_usd': 20.0,
        'max_position': 50.0,
        'take_profit_pct': 0.5,       # 1:1 R:R
        'stop_loss_pct': 0.5,
        'cooldown_seconds': 10.0,
    },
    'BTC/USDT': {
        'deviation_threshold': 0.3,   # Tighter for lower volatility BTC
        'rsi_oversold': 30,           # More aggressive for efficient market
        'rsi_overbought': 70,
        'position_size_usd': 50.0,    # Larger size for BTC liquidity
        'max_position': 150.0,
        'take_profit_pct': 0.4,       # Tighter for BTC
        'stop_loss_pct': 0.4,
        'cooldown_seconds': 5.0,      # Faster for liquid BTC
    },
    # REC-001 (v3.0.0): XRP/BTC ratio trading pair
    'XRP/BTC': {
        'deviation_threshold': 1.0,   # Wider for ratio volatility (1.55x XRP vs BTC)
        'rsi_oversold': 35,           # Conservative for ratio trading
        'rsi_overbought': 65,
        'position_size_usd': 15.0,    # Lower for less liquidity
        'max_position': 40.0,         # Conservative limit
        'take_profit_pct': 0.8,       # Account for wider spreads, 1:1 R:R
        'stop_loss_pct': 0.8,
        'cooldown_seconds': 20.0,     # Slower for ratio trades
    },
}


# =============================================================================
# Section 1: Configuration and Validation - REC-007
# =============================================================================
def _get_symbol_config(symbol: str, config: Dict[str, Any], key: str) -> Any:
    """Get symbol-specific config or fall back to global config."""
    symbol_cfg = SYMBOL_CONFIGS.get(symbol, {})
    return symbol_cfg.get(key, config.get(key))


def _validate_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate configuration on startup.

    Returns:
        List of error/warning messages (empty if valid)
    """
    errors = []

    # Required positive values
    required_positive = [
        'position_size_usd', 'max_position', 'stop_loss_pct',
        'take_profit_pct', 'lookback_candles', 'cooldown_seconds',
    ]

    for key in required_positive:
        val = config.get(key)
        if val is None:
            errors.append(f"Missing required config: {key}")
        elif not isinstance(val, (int, float)):
            errors.append(f"{key} must be numeric, got {type(val).__name__}")
        elif val <= 0:
            errors.append(f"{key} must be positive, got {val}")

    # Bounds checks
    deviation = config.get('deviation_threshold', 0.5)
    if deviation < 0.1 or deviation > 2.0:
        errors.append(f"deviation_threshold should be 0.1-2.0, got {deviation}")

    rsi_oversold = config.get('rsi_oversold', 35)
    rsi_overbought = config.get('rsi_overbought', 65)
    if rsi_oversold >= rsi_overbought:
        errors.append(f"rsi_oversold ({rsi_oversold}) must be < rsi_overbought ({rsi_overbought})")
    if rsi_oversold < 10 or rsi_oversold > 50:
        errors.append(f"rsi_oversold should be 10-50, got {rsi_oversold}")
    if rsi_overbought < 50 or rsi_overbought > 90:
        errors.append(f"rsi_overbought should be 50-90, got {rsi_overbought}")

    # R:R ratio check
    sl = config.get('stop_loss_pct', 0.5)
    tp = config.get('take_profit_pct', 0.5)
    if sl > 0 and tp > 0:
        rr_ratio = tp / sl
        if rr_ratio < 1.0:
            errors.append(f"Warning: R:R ratio ({rr_ratio:.2f}:1) < 1:1")
        elif rr_ratio < 1.5:
            errors.append(f"Info: R:R ratio {rr_ratio:.2f}:1 acceptable")

    # Validate SYMBOL_CONFIGS
    symbol_positive_keys = ['stop_loss_pct', 'take_profit_pct', 'position_size_usd']
    for symbol, sym_cfg in SYMBOL_CONFIGS.items():
        for key in symbol_positive_keys:
            if key in sym_cfg:
                val = sym_cfg[key]
                if not isinstance(val, (int, float)):
                    errors.append(f"{symbol}.{key} must be numeric")
                elif val <= 0:
                    errors.append(f"{symbol}.{key} must be positive, got {val}")

        # Per-symbol R:R check
        sym_sl = sym_cfg.get('stop_loss_pct')
        sym_tp = sym_cfg.get('take_profit_pct')
        if sym_sl and sym_tp and sym_sl > 0 and sym_tp > 0:
            rr = sym_tp / sym_sl
            if rr < 1.0:
                errors.append(f"Warning: {symbol} R:R ratio ({rr:.2f}:1) < 1:1")

    return errors


# =============================================================================
# Section 2: Indicator Calculations
# =============================================================================
def _calculate_sma(candles: List, period: int) -> float:
    """Calculate simple moving average."""
    if len(candles) < period:
        return 0.0
    closes = [c.close for c in candles[-period:]]
    return sum(closes) / len(closes)


def _calculate_rsi(candles: List, period: int = 14) -> float:
    """
    Calculate RSI indicator.

    LOW-007: Fixed edge case where index could go negative.
    """
    if len(candles) < period + 1:
        return 50.0  # Neutral

    gains = []
    losses = []

    # LOW-007: Ensure we don't access negative indices
    start_idx = max(1, len(candles) - period)
    for i in range(start_idx, len(candles)):
        change = candles[i].close - candles[i-1].close
        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(change))

    if not gains:
        return 50.0  # Neutral if no data

    avg_gain = sum(gains) / len(gains)
    avg_loss = sum(losses) / len(losses)

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def _calculate_bollinger_bands(
    candles: List,
    period: int = 20,
    std_dev: float = 2.0
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Calculate Bollinger Bands."""
    if len(candles) < period:
        return None, None, None

    closes = [c.close for c in candles[-period:]]
    sma = sum(closes) / len(closes)

    variance = sum((c - sma) ** 2 for c in closes) / len(closes)
    std = variance ** 0.5

    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)

    return lower, sma, upper


def _calculate_volatility(candles: List, lookback: int = 20) -> float:
    """
    Calculate price volatility from candle closes.

    Returns volatility as a percentage (std dev of returns * 100).
    """
    if len(candles) < lookback + 1:
        return 0.0

    closes = [c.close for c in candles[-(lookback + 1):]]
    if len(closes) < 2:
        return 0.0

    returns = [(closes[i] - closes[i - 1]) / closes[i - 1]
               for i in range(1, len(closes)) if closes[i - 1] != 0]

    if not returns:
        return 0.0

    mean_return = sum(returns) / len(returns)
    variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
    return (variance ** 0.5) * 100


# =============================================================================
# Section 3: Volatility Regime Classification - REC-004
# =============================================================================
def _classify_volatility_regime(
    volatility_pct: float,
    config: Dict[str, Any]
) -> VolatilityRegime:
    """Classify current volatility into a regime."""
    low = config.get('regime_low_threshold', 0.3)
    medium = config.get('regime_medium_threshold', 0.8)
    high = config.get('regime_high_threshold', 1.5)

    if volatility_pct < low:
        return VolatilityRegime.LOW
    elif volatility_pct < medium:
        return VolatilityRegime.MEDIUM
    elif volatility_pct < high:
        return VolatilityRegime.HIGH
    else:
        return VolatilityRegime.EXTREME


def _get_regime_adjustments(
    regime: VolatilityRegime,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Get threshold and size adjustments based on volatility regime."""
    adjustments = {
        'threshold_mult': 1.0,
        'size_mult': 1.0,
        'pause_trading': False,
    }

    if regime == VolatilityRegime.LOW:
        # Tighter thresholds in low volatility
        adjustments['threshold_mult'] = 0.8
        adjustments['size_mult'] = 1.0
    elif regime == VolatilityRegime.MEDIUM:
        adjustments['threshold_mult'] = 1.0
        adjustments['size_mult'] = 1.0
    elif regime == VolatilityRegime.HIGH:
        # Wider thresholds, smaller sizes in high volatility
        adjustments['threshold_mult'] = 1.3
        adjustments['size_mult'] = 0.8
    elif regime == VolatilityRegime.EXTREME:
        adjustments['threshold_mult'] = 1.5
        adjustments['size_mult'] = 0.5
        adjustments['pause_trading'] = config.get('regime_extreme_pause', True)

    return adjustments


# =============================================================================
# Section 4: Risk Management - REC-005
# =============================================================================
def _check_circuit_breaker(
    state: Dict[str, Any],
    current_time: datetime,
    max_losses: int,
    cooldown_minutes: float
) -> bool:
    """Check if circuit breaker is active."""
    consecutive_losses = state.get('consecutive_losses', 0)

    if consecutive_losses < max_losses:
        return False

    breaker_time = state.get('circuit_breaker_time')
    if breaker_time is None:
        return False

    elapsed_minutes = (current_time - breaker_time).total_seconds() / 60

    if elapsed_minutes >= cooldown_minutes:
        state['consecutive_losses'] = 0
        state['circuit_breaker_time'] = None
        return False

    return True


def _is_trade_flow_aligned(
    data: DataSnapshot,
    symbol: str,
    direction: str,
    threshold: float,
    n_trades: int = 50
) -> bool:
    """Check if trade flow confirms the signal direction."""
    trade_flow = data.get_trade_imbalance(symbol, n_trades)

    if direction == 'buy':
        return trade_flow > threshold
    elif direction == 'sell':
        return trade_flow < -threshold

    return True


# =============================================================================
# Section 4b: Trend Filter - REC-004 (v3.0.0)
# =============================================================================
def _calculate_trend_slope(candles: List, period: int = 50) -> float:
    """
    Calculate the slope of price trend over given period.

    Returns slope as percentage change per candle.
    Positive = uptrend, Negative = downtrend, Near zero = ranging.
    """
    if len(candles) < period:
        return 0.0

    closes = [c.close for c in candles[-period:]]
    if len(closes) < 2:
        return 0.0

    # Calculate linear regression slope
    n = len(closes)
    sum_x = sum(range(n))
    sum_y = sum(closes)
    sum_xy = sum(i * closes[i] for i in range(n))
    sum_x2 = sum(i * i for i in range(n))

    denominator = n * sum_x2 - sum_x * sum_x
    if denominator == 0:
        return 0.0

    slope = (n * sum_xy - sum_x * sum_y) / denominator

    # Convert to percentage of average price
    avg_price = sum_y / n if n > 0 else 1.0
    slope_pct = (slope / avg_price) * 100 if avg_price > 0 else 0.0

    return slope_pct


def _is_trending(
    candles: List,
    config: Dict[str, Any],
    state: Dict[str, Any],
    symbol: str
) -> Tuple[bool, float, int]:
    """
    Check if market is currently trending (unsuitable for mean reversion).

    REC-003 (v4.0): Added confirmation period to reduce false positives
    in choppy markets. Only consider trending if slope exceeds threshold
    for N consecutive evaluations.

    Returns:
        Tuple of (is_confirmed_trending, slope_pct, consecutive_trending_count)
    """
    period = config.get('trend_sma_period', 50)
    threshold = config.get('trend_slope_threshold', 0.05)
    confirmation_periods = config.get('trend_confirmation_periods', 3)

    slope_pct = _calculate_trend_slope(candles, period)

    # Check if slope exceeds threshold
    is_slope_trending = abs(slope_pct) > threshold

    # Initialize trend confirmation tracking per symbol
    if 'trend_confirmation_counts' not in state:
        state['trend_confirmation_counts'] = {}

    # Update consecutive trending count
    if is_slope_trending:
        state['trend_confirmation_counts'][symbol] = \
            state['trend_confirmation_counts'].get(symbol, 0) + 1
    else:
        state['trend_confirmation_counts'][symbol] = 0

    consecutive_count = state['trend_confirmation_counts'].get(symbol, 0)

    # Only confirm trending if threshold exceeded for N consecutive periods
    is_confirmed_trending = consecutive_count >= confirmation_periods

    return is_confirmed_trending, slope_pct, consecutive_count


# =============================================================================
# Section 4b.2: XRP/BTC Correlation Monitoring - REC-005 (v4.0.0)
# =============================================================================
def _calculate_correlation(
    xrp_candles: List,
    btc_candles: List,
    lookback: int = 50
) -> Optional[float]:
    """
    Calculate rolling Pearson correlation between XRP and BTC price movements.

    REC-005 (v4.0.0): Added for XRP/BTC ratio trading analysis.
    Research shows XRP correlation with BTC declining (24.86% over 90 days),
    which affects ratio mean reversion timing.

    Args:
        xrp_candles: XRP/USDT candles
        btc_candles: BTC/USDT candles
        lookback: Number of candles for correlation

    Returns:
        Correlation coefficient (-1 to +1), None if insufficient data
    """
    if len(xrp_candles) < lookback + 1 or len(btc_candles) < lookback + 1:
        return None

    # Get returns for correlation calculation
    xrp_closes = [c.close for c in xrp_candles[-(lookback + 1):]]
    btc_closes = [c.close for c in btc_candles[-(lookback + 1):]]

    if len(xrp_closes) != len(btc_closes):
        return None

    # Calculate returns
    xrp_returns = [(xrp_closes[i] - xrp_closes[i-1]) / xrp_closes[i-1]
                   for i in range(1, len(xrp_closes)) if xrp_closes[i-1] != 0]
    btc_returns = [(btc_closes[i] - btc_closes[i-1]) / btc_closes[i-1]
                   for i in range(1, len(btc_closes)) if btc_closes[i-1] != 0]

    if len(xrp_returns) < 2 or len(btc_returns) < 2:
        return None

    # Ensure same length
    n = min(len(xrp_returns), len(btc_returns))
    xrp_returns = xrp_returns[-n:]
    btc_returns = btc_returns[-n:]

    # Calculate Pearson correlation
    mean_xrp = sum(xrp_returns) / n
    mean_btc = sum(btc_returns) / n

    # Covariance
    covariance = sum((xrp_returns[i] - mean_xrp) * (btc_returns[i] - mean_btc)
                     for i in range(n)) / n

    # Standard deviations
    std_xrp = (sum((r - mean_xrp) ** 2 for r in xrp_returns) / n) ** 0.5
    std_btc = (sum((r - mean_btc) ** 2 for r in btc_returns) / n) ** 0.5

    if std_xrp == 0 or std_btc == 0:
        return None

    correlation = covariance / (std_xrp * std_btc)

    # Clamp to valid range
    return max(-1.0, min(1.0, correlation))


def _get_xrp_btc_correlation(
    data: DataSnapshot,
    config: Dict[str, Any],
    state: Dict[str, Any]
) -> Optional[float]:
    """
    Get XRP/BTC correlation and update tracking.

    Returns correlation coefficient or None if unavailable.
    """
    if not config.get('use_correlation_monitoring', True):
        return None

    xrp_candles = data.candles_5m.get('XRP/USDT', ())
    btc_candles = data.candles_5m.get('BTC/USDT', ())
    lookback = config.get('correlation_lookback', 50)

    correlation = _calculate_correlation(
        list(xrp_candles), list(btc_candles), lookback
    )

    if correlation is not None:
        # Track correlation history
        if 'correlation_history' not in state:
            state['correlation_history'] = []
        state['correlation_history'].append(correlation)
        # Keep bounded
        if len(state['correlation_history']) > 100:
            state['correlation_history'] = state['correlation_history'][-100:]

        # Store latest
        state['xrp_btc_correlation'] = correlation

        # Check for low correlation warning
        warn_threshold = config.get('correlation_warn_threshold', 0.5)
        if correlation < warn_threshold and not state.get('_correlation_warned', False):
            state['_correlation_warned'] = True

    return correlation


# =============================================================================
# Section 4c: Trailing Stops - REC-006 (v3.0.0)
# =============================================================================
def _calculate_trailing_stop(
    entry_price: float,
    highest_price: float,
    lowest_price: float,
    side: str,
    activation_pct: float,
    trail_distance_pct: float,
) -> Optional[float]:
    """
    Calculate trailing stop price if activated.

    Args:
        entry_price: Original entry price
        highest_price: Highest price since entry (for longs)
        lowest_price: Lowest price since entry (for shorts)
        side: 'long' or 'short'
        activation_pct: Profit % to activate trailing stop
        trail_distance_pct: Distance % to trail from extreme

    Returns:
        Trailing stop price if activated, None otherwise
    """
    if entry_price <= 0:
        return None

    if side == 'long':
        profit_pct = ((highest_price - entry_price) / entry_price) * 100
        if profit_pct >= activation_pct:
            return highest_price * (1 - trail_distance_pct / 100)
    elif side == 'short':
        profit_pct = ((entry_price - lowest_price) / entry_price) * 100
        if profit_pct >= activation_pct:
            return lowest_price * (1 + trail_distance_pct / 100)

    return None


def _update_position_extremes(
    state: Dict[str, Any],
    symbol: str,
    current_price: float
) -> None:
    """Update highest/lowest prices for trailing stop calculation."""
    if 'position_entries' not in state:
        return

    entry = state['position_entries'].get(symbol)
    if not entry:
        return

    # Initialize extremes if not present
    if 'highest_price' not in entry:
        entry['highest_price'] = entry.get('entry_price', current_price)
    if 'lowest_price' not in entry:
        entry['lowest_price'] = entry.get('entry_price', current_price)

    # Update extremes
    entry['highest_price'] = max(entry['highest_price'], current_price)
    entry['lowest_price'] = min(entry['lowest_price'], current_price)


# =============================================================================
# Section 4d: Position Decay - REC-007 (v3.0.0)
# =============================================================================
def _get_decayed_take_profit(
    entry_price: float,
    original_tp_pct: float,
    entry_time: datetime,
    current_time: datetime,
    side: str,
    config: Dict[str, Any]
) -> Tuple[float, float]:
    """
    Calculate decayed take profit based on position age.

    Mean reversion assumes timely return to mean. If price doesn't revert
    within expected timeframe, reduce TP to exit sooner.

    Args:
        entry_price: Original entry price
        original_tp_pct: Original take profit percentage
        entry_time: Time of entry
        current_time: Current time
        side: 'long' or 'short'
        config: Strategy configuration

    Returns:
        Tuple of (decayed_tp_price, decay_multiplier)
    """
    decay_start = config.get('decay_start_minutes', 3.0)
    decay_interval = config.get('decay_interval_minutes', 1.0)
    decay_multipliers = config.get('decay_multipliers', [1.0, 0.75, 0.5, 0.25])

    # Calculate position age in minutes
    if entry_time is None:
        return entry_price, 1.0

    age_minutes = (current_time - entry_time).total_seconds() / 60

    # No decay before start time
    if age_minutes < decay_start:
        multiplier = 1.0
    else:
        # Calculate decay stage
        decay_stage = int((age_minutes - decay_start) / decay_interval)
        decay_stage = min(decay_stage, len(decay_multipliers) - 1)
        multiplier = decay_multipliers[decay_stage]

    # Calculate decayed TP
    effective_tp_pct = original_tp_pct * multiplier

    if side == 'long':
        decayed_tp = entry_price * (1 + effective_tp_pct / 100)
    else:  # short
        decayed_tp = entry_price * (1 - effective_tp_pct / 100)

    return decayed_tp, multiplier


# =============================================================================
# Section 5: Signal Rejection Tracking
# =============================================================================
def _track_rejection(
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


def _build_base_indicators(
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
# Section 6: State Initialization
# =============================================================================
def _initialize_state(state: Dict[str, Any]) -> None:
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
# REQUIRED: Main Signal Generation Function
# =============================================================================
def generate_signal(
    data: DataSnapshot,
    config: Dict[str, Any],
    state: Dict[str, Any]
) -> Optional[Signal]:
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
        _initialize_state(state)

    current_time = data.timestamp
    track_rejections = config.get('track_rejections', True)

    # REC-005: Circuit breaker check
    if config.get('use_circuit_breaker', True):
        max_losses = config.get('max_consecutive_losses', 3)
        cooldown_min = config.get('circuit_breaker_minutes', 15)

        if _check_circuit_breaker(state, current_time, max_losses, cooldown_min):
            state['indicators'] = _build_base_indicators(
                symbol='N/A', status='circuit_breaker', state=state
            )
            state['indicators']['circuit_breaker_active'] = True
            if track_rejections:
                _track_rejection(state, RejectionReason.CIRCUIT_BREAKER)
            return None

    # REC-003: Time-based cooldown check
    if state['last_signal_time'] is not None:
        elapsed = (current_time - state['last_signal_time']).total_seconds()
        global_cooldown = config.get('cooldown_seconds', 10.0)
        if elapsed < global_cooldown:
            state['indicators'] = _build_base_indicators(
                symbol='N/A', status='cooldown', state=state
            )
            state['indicators']['cooldown_remaining'] = global_cooldown - elapsed
            if track_rejections:
                _track_rejection(state, RejectionReason.TIME_COOLDOWN)
            return None

    # Evaluate each symbol
    for symbol in SYMBOLS:
        signal = _evaluate_symbol(data, config, state, symbol, current_time, Signal)
        if signal:
            state['last_signal_time'] = current_time
            return signal

    return None


def _evaluate_symbol(
    data: DataSnapshot,
    config: Dict[str, Any],
    state: Dict[str, Any],
    symbol: str,
    current_time: datetime,
    Signal
) -> Optional[Signal]:
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
        state['indicators'] = _build_base_indicators(
            symbol=symbol, status='warming_up', state=state
        )
        state['indicators']['candles_available'] = len(candles_5m)
        state['indicators']['candles_required'] = lookback
        if track_rejections:
            _track_rejection(state, RejectionReason.WARMING_UP, symbol)
        return None

    current_price = data.prices.get(symbol, 0)
    if not current_price:
        state['indicators'] = _build_base_indicators(
            symbol=symbol, status='no_price', state=state
        )
        if track_rejections:
            _track_rejection(state, RejectionReason.NO_PRICE_DATA, symbol)
        return None

    # Calculate indicators
    candles_list = list(candles_5m)
    sma = _calculate_sma(candles_list, lookback)
    rsi = _calculate_rsi(candles_list, config.get('rsi_period', 14))

    bb_period = config.get('bb_period', 20)
    bb_std = config.get('bb_std_dev', 2.0)
    bb_lower, bb_mid, bb_upper = _calculate_bollinger_bands(candles_list, bb_period, bb_std)

    if not sma or not bb_lower:
        return None

    # Calculate volatility and classify regime
    candles_1m = data.candles_1m.get(symbol, candles_5m)
    volatility = _calculate_volatility(list(candles_1m), config.get('volatility_lookback', 20))

    regime = VolatilityRegime.MEDIUM
    regime_adjustments = {'threshold_mult': 1.0, 'size_mult': 1.0, 'pause_trading': False}

    if config.get('use_volatility_regimes', True):
        regime = _classify_volatility_regime(volatility, config)
        regime_adjustments = _get_regime_adjustments(regime, config)

        if regime_adjustments['pause_trading']:
            state['indicators'] = _build_base_indicators(
                symbol=symbol, status='regime_pause', state=state
            )
            state['indicators']['volatility_regime'] = regime.name
            state['indicators']['volatility_pct'] = round(volatility, 4)
            if track_rejections:
                _track_rejection(state, RejectionReason.REGIME_PAUSE, symbol)
            return None

    # REC-004 (v3.0.0): Trend filter check with confirmation (REC-003 v4.0)
    trend_slope = 0.0
    is_market_trending = False
    trend_confirmation_count = 0
    if config.get('use_trend_filter', True):
        is_market_trending, trend_slope, trend_confirmation_count = _is_trending(
            candles_list, config, state, symbol
        )

    # Calculate VWAP
    vwap = data.get_vwap(symbol, config.get('vwap_lookback', 50))

    # Calculate deviation from SMA
    deviation_pct = ((current_price - sma) / sma) * 100

    # Get symbol-specific config
    base_deviation_threshold = _get_symbol_config(symbol, config, 'deviation_threshold')
    rsi_oversold = _get_symbol_config(symbol, config, 'rsi_oversold')
    rsi_overbought = _get_symbol_config(symbol, config, 'rsi_overbought')
    base_position_size = _get_symbol_config(symbol, config, 'position_size_usd')
    max_position = _get_symbol_config(symbol, config, 'max_position')
    tp_pct = _get_symbol_config(symbol, config, 'take_profit_pct')
    sl_pct = _get_symbol_config(symbol, config, 'stop_loss_pct')

    # Apply regime adjustments
    effective_deviation_threshold = base_deviation_threshold * regime_adjustments['threshold_mult']
    adjusted_position_size = base_position_size * regime_adjustments['size_mult']

    # Get trade flow for confirmation
    use_trade_flow = config.get('use_trade_flow_confirmation', True)
    trade_flow_threshold = config.get('trade_flow_threshold', 0.10)
    trade_flow = data.get_trade_imbalance(symbol, 50)

    # REC-005 (v4.0.0): XRP/BTC correlation monitoring
    xrp_btc_correlation = None
    if symbol == 'XRP/BTC' and config.get('use_correlation_monitoring', True):
        xrp_btc_correlation = _get_xrp_btc_correlation(data, config, state)

    # Get current position for this symbol
    current_position = state.get('position_by_symbol', {}).get(symbol, 0)

    # REC-006 (v3.0.0): Update position extremes for trailing stop
    _update_position_extremes(state, symbol, current_price)

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
            _track_rejection(state, RejectionReason.TRENDING_MARKET, symbol)
        return None

    # Position limit check
    if current_position >= max_position:
        state['indicators']['status'] = 'max_position_reached'
        if track_rejections:
            _track_rejection(state, RejectionReason.MAX_POSITION, symbol)
        return None

    available = max_position - current_position
    actual_size = min(adjusted_position_size, available)

    min_trade_size = config.get('min_trade_size_usd', 5.0)
    if actual_size < min_trade_size:
        state['indicators']['status'] = 'insufficient_size'
        if track_rejections:
            _track_rejection(state, RejectionReason.INSUFFICIENT_SIZE, symbol)
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
        _track_rejection(state, RejectionReason.NO_SIGNAL_CONDITIONS, symbol)
    return None


# =============================================================================
# Section 7b: Signal Generation Helpers (Finding #4 - v3.0.0)
# =============================================================================
def _check_trailing_stop_exit(
    state: Dict[str, Any],
    symbol: str,
    current_price: float,
    current_position: float,
    config: Dict[str, Any],
    Signal
) -> Optional[Signal]:
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

    trailing_stop = _calculate_trailing_stop(
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
) -> Optional[Signal]:
    """Check if position decay should trigger exit."""
    entry = state.get('position_entries', {}).get(symbol)
    if not entry:
        return None

    entry_price = entry.get('entry_price', 0)
    entry_time = entry.get('entry_time')
    side = entry.get('side', 'long')

    if not entry_time or not entry_price:
        return None

    tp_pct = _get_symbol_config(symbol, config, 'take_profit_pct')
    decayed_tp, decay_mult = _get_decayed_take_profit(
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
    data: DataSnapshot,
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
) -> Optional[Signal]:
    """
    Generate entry signal based on mean reversion conditions.

    Finding #4: Extracted from _evaluate_symbol to reduce complexity.
    """
    signal = None

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
                if not _is_trade_flow_aligned(data, symbol, 'buy', trade_flow_threshold):
                    state['indicators']['status'] = 'trade_flow_not_aligned'
                    state['indicators']['trade_flow_aligned'] = False
                    if track_rejections:
                        _track_rejection(state, RejectionReason.TRADE_FLOW_NOT_ALIGNED, symbol)
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
                if not _is_trade_flow_aligned(data, symbol, 'sell', trade_flow_threshold):
                    state['indicators']['status'] = 'trade_flow_not_aligned'
                    state['indicators']['trade_flow_aligned'] = False
                    if track_rejections:
                        _track_rejection(state, RejectionReason.TRADE_FLOW_NOT_ALIGNED, symbol)
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


# =============================================================================
# OPTIONAL: Lifecycle Callbacks
# =============================================================================
def on_start(config: Dict[str, Any], state: Dict[str, Any]) -> None:
    """Initialize strategy state on startup."""
    # REC-007: Validate configuration
    errors = _validate_config(config)
    if errors:
        for error in errors:
            print(f"[mean_reversion] Config warning: {error}")
        state['config_warnings'] = errors
    state['config_validated'] = True

    # REC-002 (v3.0.0): Store config values for use in on_fill
    state['config_max_consecutive_losses'] = config.get('max_consecutive_losses', 3)

    _initialize_state(state)

    print(f"[mean_reversion] v{STRATEGY_VERSION} started")
    print(f"[mean_reversion] Symbols: {SYMBOLS}")
    print(f"[mean_reversion] Features: VolatilityRegimes={config.get('use_volatility_regimes', True)}, "
          f"CircuitBreaker={config.get('use_circuit_breaker', True)}, "
          f"TradeFlowConfirm={config.get('use_trade_flow_confirmation', True)}, "
          f"TrendFilter={config.get('use_trend_filter', True)}, "
          f"TrailingStop={config.get('use_trailing_stop', False)}, "  # v4.0: default False per REC-001
          f"PositionDecay={config.get('use_position_decay', True)}, "
          f"CorrelationMonitor={config.get('use_correlation_monitoring', True)}")  # v4.0: REC-005
    # v4.0.0: Log key parameter changes
    print(f"[mean_reversion] v4.0 Params: DecayStart={config.get('decay_start_minutes', 15.0)}min, "
          f"TrendConfirm={config.get('trend_confirmation_periods', 3)} periods")


def on_fill(fill: Dict[str, Any], state: Dict[str, Any]) -> None:
    """
    Update position tracking and per-pair metrics.

    REC-006: Per-pair PnL tracking
    REC-005: Circuit breaker consecutive loss tracking
    """
    side = fill.get('side', '')
    symbol = fill.get('symbol', 'XRP/USDT')
    price = fill.get('price', 0)
    size = fill.get('size', 0)
    pnl = fill.get('pnl', 0)
    timestamp = fill.get('timestamp', datetime.now())

    value = fill.get('value', size * price)

    # Initialize tracking dicts
    for key in ['pnl_by_symbol', 'trades_by_symbol', 'wins_by_symbol',
                'losses_by_symbol', 'position_by_symbol', 'position_entries']:
        if key not in state:
            state[key] = {}

    # REC-006: Per-pair PnL tracking
    if pnl != 0:
        state['pnl_by_symbol'][symbol] = state['pnl_by_symbol'].get(symbol, 0) + pnl

        # REC-005: Circuit breaker tracking
        if pnl > 0:
            state['wins_by_symbol'][symbol] = state['wins_by_symbol'].get(symbol, 0) + 1
            state['consecutive_losses'] = 0
        else:
            state['losses_by_symbol'][symbol] = state['losses_by_symbol'].get(symbol, 0) + 1
            state['consecutive_losses'] = state.get('consecutive_losses', 0) + 1

            # REC-002 (v3.0.0): Use config value instead of hardcoded 3
            max_losses = state.get('config_max_consecutive_losses', 3)
            if state['consecutive_losses'] >= max_losses:
                state['circuit_breaker_time'] = timestamp

    state['trades_by_symbol'][symbol] = state['trades_by_symbol'].get(symbol, 0) + 1

    # Position tracking
    current_position = state['position_by_symbol'].get(symbol, 0)

    if side == 'buy':
        state['position_by_symbol'][symbol] = current_position + value
        if symbol not in state['position_entries']:
            state['position_entries'][symbol] = {
                'entry_price': price,
                'entry_time': timestamp,
                'side': 'long',
            }
    elif side == 'sell':
        state['position_by_symbol'][symbol] = max(0, current_position - value)
        if state['position_by_symbol'][symbol] < 0.01:
            state['position_by_symbol'][symbol] = 0.0
            if symbol in state['position_entries']:
                del state['position_entries'][symbol]
    elif side == 'short':
        state['position_by_symbol'][symbol] = current_position - value
        if symbol not in state['position_entries']:
            state['position_entries'][symbol] = {
                'entry_price': price,
                'entry_time': timestamp,
                'side': 'short',
            }
    elif side == 'cover':
        state['position_by_symbol'][symbol] = min(0, current_position + value)
        if abs(state['position_by_symbol'][symbol]) < 0.01:
            state['position_by_symbol'][symbol] = 0.0
            if symbol in state['position_entries']:
                del state['position_entries'][symbol]

    # Update aggregate position
    state['position'] = sum(state['position_by_symbol'].values())
    state['last_fill'] = fill


def on_stop(state: Dict[str, Any]) -> None:
    """
    Called when strategy stops.

    Finding #6: Added on_stop() with summary logging.
    """
    total_pnl = sum(state.get('pnl_by_symbol', {}).values())
    total_trades = sum(state.get('trades_by_symbol', {}).values())
    total_wins = sum(state.get('wins_by_symbol', {}).values())
    total_losses = sum(state.get('losses_by_symbol', {}).values())

    win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0

    rejection_counts = state.get('rejection_counts', {})
    total_rejections = sum(rejection_counts.values())

    state['indicators'] = {}

    state['final_summary'] = {
        'position_by_symbol': state.get('position_by_symbol', {}),
        'pnl_by_symbol': state.get('pnl_by_symbol', {}),
        'trades_by_symbol': state.get('trades_by_symbol', {}),
        'wins_by_symbol': state.get('wins_by_symbol', {}),
        'losses_by_symbol': state.get('losses_by_symbol', {}),
        'total_pnl': total_pnl,
        'total_trades': total_trades,
        'total_wins': total_wins,
        'total_losses': total_losses,
        'win_rate': win_rate,
        'config_warnings': state.get('config_warnings', []),
        'rejection_counts': rejection_counts,
        'rejection_counts_by_symbol': state.get('rejection_counts_by_symbol', {}),
        'total_rejections': total_rejections,
    }

    print(f"[mean_reversion] Stopped. PnL: ${total_pnl:.2f}, Trades: {total_trades}, Win Rate: {win_rate:.1f}%")

    # Print per-symbol summary
    for symbol in SYMBOLS:
        sym_pnl = state.get('pnl_by_symbol', {}).get(symbol, 0)
        sym_trades = state.get('trades_by_symbol', {}).get(symbol, 0)
        sym_wins = state.get('wins_by_symbol', {}).get(symbol, 0)
        sym_losses = state.get('losses_by_symbol', {}).get(symbol, 0)
        sym_wr = (sym_wins / sym_trades * 100) if sym_trades > 0 else 0
        print(f"[mean_reversion]   {symbol}: PnL=${sym_pnl:.2f}, Trades={sym_trades}, WR={sym_wr:.1f}%")

    # Print rejection summary
    if rejection_counts:
        print(f"[mean_reversion] Signal rejections ({total_rejections} total):")
        for reason, count in sorted(rejection_counts.items(), key=lambda x: -x[1])[:5]:
            print(f"[mean_reversion]   - {reason}: {count}")
