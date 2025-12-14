"""
Ratio Trading Strategy v4.1.0

Mean reversion strategy for XRP/BTC pair accumulation.
Trades the XRP/BTC ratio to grow holdings of both assets.

Strategy Logic:
- Calculate moving average of XRP/BTC ratio
- Use Bollinger Bands for entry/exit zones
- Buy when ratio is below lower band (XRP cheap vs BTC)
- Sell when ratio is above upper band (XRP expensive vs BTC)
- Rebalance to maintain balanced holdings

IMPORTANT: This strategy is designed ONLY for crypto-to-crypto ratio pairs
(XRP/BTC). It is NOT suitable for USDT-denominated pairs. For USDT pairs,
use the mean_reversion.py strategy instead.

WARNING - Trend Continuation Risk:
Bollinger Band touches can signal trend CONTINUATION rather than reversal.
Price exceeding the bands may indicate strong momentum, not necessarily a
mean reversion opportunity. The volatility regime system helps mitigate this
by pausing in EXTREME conditions and widening thresholds in HIGH volatility.

CRITICAL WARNING - XRP/BTC Correlation Crisis (December 2025):
XRP/BTC correlation has declined to ~0.40-0.54, representing a ~37-53% drop from
historical norms (~0.85). This fundamentally challenges pairs trading viability.
The strategy now enables correlation_pause_enabled by default (v4.0.0) which will
auto-pause trading when correlation drops below 0.4. Monitor correlation closely
and consider the following options:

- CONSERVATIVE: Pause XRP/BTC trading until correlation stabilizes above 0.6
- MODERATE: Use v4.0.0+ correlation protection (enabled by default)
- AGGRESSIVE: Lower correlation_pause_threshold to 0.3 (more trading, higher risk)

ALTERNATIVE PAIRS (REC-033):
If XRP/BTC correlation remains low, consider evaluating alternative pairs:
- ETH/BTC: Stronger historical cointegration (~0.80 correlation), higher liquidity
- LTC/BTC: Classical pairs candidate (~0.80 correlation)
- BCH/BTC: Bitcoin fork relationship (~0.75 correlation)
These would require strategy scope expansion (future enhancement).

FUTURE ENHANCEMENTS:
- REC-034: Generalized Hurst Exponent (GHE) for mean-reversion validation
  H < 0.5 = mean-reverting (good), H >= 0.5 = trending (pause)
- REC-035: ADF Cointegration Test for formal cointegration validation
  Currently uses correlation as proxy; formal testing would be more robust

Version History:
- 1.0.0: Initial implementation
         - Mean reversion with Bollinger Bands
         - Dual-asset accumulation tracking
         - Research-based config from Kraken data
- 2.0.0: Major refactor per ratio-trading-strategy-review-v1.0.md
         - REC-002: Converted to USD-based position sizing
         - REC-003: Fixed R:R ratio to 1:1 (0.6%/0.6%)
         - REC-004: Added volatility regime classification
         - REC-005: Added circuit breaker protection
         - REC-006: Added per-pair PnL tracking
         - REC-007: Added configuration validation
         - REC-008: Added spread monitoring
         - REC-010: Added trade flow confirmation
         - Refactored generate_signal into smaller functions
         - Fixed take profit to use price-based percentage
         - Added rejection tracking
         - Added comprehensive on_stop() summary
- 2.1.0: Enhancement refactor per ratio-trading-strategy-review-v2.0.md
         - REC-013: Higher entry threshold (1.0 -> 1.5 std)
         - REC-014: Optional RSI confirmation filter
         - REC-015: Trend detection warning system
         - REC-016: Enhanced accumulation metrics
         - REC-017: Documentation updates (trend risk warning)
         - Added trailing stops (from mean reversion patterns)
         - Added position decay for stale positions
         - Fixed hardcoded max_losses in on_fill
- 3.0.0: Review recommendations per ratio-trading-strategy-review-v3.1.md
         - REC-018: Dynamic BTC price for USD conversion
         - REC-019: Fixed on_start print statement (already correct)
         - REC-020: Separate exit tracking from rejection tracking
         - REC-021: Rolling correlation monitoring with pause option
         - REC-022: Hedge ratio calculation (optional, future enhancement)
         - Added ExitReason enum for intentional exit tracking
         - Added correlation warning system
         - Improved accumulation metrics with real-time USD values
- 4.0.0: Deep review recommendations per ratio-trading-strategy-review-v4.0.md
         - REC-023: Enable correlation_pause_enabled by default (HIGH priority)
         - REC-024: Raised correlation thresholds for earlier warning/pause
           - correlation_warning_threshold: 0.5 → 0.6
           - correlation_pause_threshold: 0.3 → 0.4
         - Research-validated: XRP/BTC correlation declining (~24.86% over 90 days)
         - Strategy parameters confirmed aligned with academic research
- 4.1.0: Deep review v6.0 recommendations
         - REC-033: Document alternative pairs consideration (ETH/BTC, LTC/BTC)
           - Strategic recommendation for when XRP/BTC correlation remains low
           - Added prominent warning about correlation crisis in docstring
         - REC-034: Document GHE validation as future enhancement
         - REC-035: Document ADF cointegration test as future enhancement
         - REC-036: Add optional wider Bollinger Bands for crypto volatility
           - New config: bollinger_std_crypto (2.5 std), use_crypto_bollinger_std
           - Research suggests 2.5-3.0 std more appropriate for crypto volatility
           - Disabled by default; current mitigations (trend filter, RSI, volatility
             regimes) may make this unnecessary per review assessment
         - Compliance: Maintained 100% with Guide v2.0
         - Status: Production ready with correlation monitoring critical
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
STRATEGY_NAME = "ratio_trading"
STRATEGY_VERSION = "4.1.0"
SYMBOLS = ["XRP/BTC"]


# =============================================================================
# Enums for Type Safety - REC-004
# =============================================================================
class VolatilityRegime(Enum):
    """Volatility regime classification for ratio trading."""
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
    SPREAD_TOO_WIDE = "spread_too_wide"
    RSI_NOT_CONFIRMED = "rsi_not_confirmed"  # REC-014
    STRONG_TREND_DETECTED = "strong_trend_detected"  # REC-015
    NO_SIGNAL_CONDITIONS = "no_signal_conditions"
    CORRELATION_TOO_LOW = "correlation_too_low"  # REC-021


class ExitReason(Enum):
    """
    Intentional exit reasons for tracking (separate from rejections).

    REC-020: Exit tracking should be separate from rejection tracking.
    """
    TRAILING_STOP = "trailing_stop"
    POSITION_DECAY = "position_decay"
    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"
    MEAN_REVERSION = "mean_reversion"  # Z-score returned to exit threshold
    CORRELATION_EXIT = "correlation_exit"  # Exit due to low correlation


# =============================================================================
# REQUIRED: Default Configuration
# =============================================================================
CONFIG = {
    # ==========================================================================
    # Core Ratio Trading Parameters - REC-013: Higher entry threshold
    # ==========================================================================
    'lookback_periods': 20,           # Periods for moving average
    'bollinger_std': 2.0,             # Standard deviations for bands
    'entry_threshold': 1.5,           # Entry at N std devs from mean (was 1.0)
    'exit_threshold': 0.5,            # Exit at N std devs (closer to mean)

    # ==========================================================================
    # REC-036: Dynamic Bollinger Settings for Crypto Volatility
    # Research suggests 2.5-3.0 std more appropriate for volatile crypto markets
    # to avoid false signals. However, current mitigations (trend filter, RSI,
    # volatility regimes) may make this unnecessary per review assessment.
    # ==========================================================================
    'use_crypto_bollinger_std': False,  # Enable wider bands for crypto volatility
    'bollinger_std_crypto': 2.5,        # Wider std dev when enabled (2.5-3.0 range)

    # ==========================================================================
    # Position Sizing - REC-002: USD-based sizing
    # ==========================================================================
    'position_size_usd': 15.0,        # Base size per trade in USD
    'max_position_usd': 50.0,         # Maximum position exposure in USD
    'min_trade_size_usd': 5.0,        # Minimum USD per trade

    # ==========================================================================
    # Risk Management - REC-003: Fixed to 1:1 R:R
    # ==========================================================================
    'stop_loss_pct': 0.6,             # Stop loss percentage
    'take_profit_pct': 0.6,           # Take profit percentage (1:1 R:R)

    # ==========================================================================
    # Cooldown Mechanisms
    # ==========================================================================
    'cooldown_seconds': 30.0,         # Minimum time between trades
    'min_candles': 10,                # Minimum candles before trading

    # ==========================================================================
    # Volatility Parameters - REC-004
    # ==========================================================================
    'use_volatility_regimes': True,   # Enable regime-based adjustments
    'volatility_lookback': 20,        # Candles for volatility calculation
    'regime_low_threshold': 0.2,      # Below = LOW regime (tighter for ratio)
    'regime_medium_threshold': 0.5,   # Below = MEDIUM regime
    'regime_high_threshold': 1.0,     # Below = HIGH regime, above = EXTREME
    'regime_extreme_pause': True,     # Pause trading in EXTREME regime

    # ==========================================================================
    # Circuit Breaker - REC-005
    # ==========================================================================
    'use_circuit_breaker': True,      # Enable consecutive loss circuit breaker
    'max_consecutive_losses': 3,      # Max losses before cooldown
    'circuit_breaker_minutes': 15,    # Cooldown after max losses

    # ==========================================================================
    # Spread Monitoring - REC-008
    # ==========================================================================
    'use_spread_filter': True,        # Enable spread filtering
    'max_spread_pct': 0.10,           # Max spread % (XRP/BTC has wider spreads)
    'min_profitability_mult': 0.5,    # TP must exceed spread * this multiplier

    # ==========================================================================
    # Trade Flow Confirmation - REC-010
    # ==========================================================================
    'use_trade_flow_confirmation': False,  # Disabled by default for ratio pairs
    'trade_flow_threshold': 0.10,          # Minimum trade flow alignment

    # ==========================================================================
    # RSI Confirmation Filter - REC-014
    # ==========================================================================
    'use_rsi_confirmation': True,     # Enable RSI filter
    'rsi_period': 14,                 # RSI calculation period
    'rsi_oversold': 35,               # RSI oversold level for buy confirmation
    'rsi_overbought': 65,             # RSI overbought level for sell confirmation

    # ==========================================================================
    # Trend Detection Warning - REC-015
    # ==========================================================================
    'use_trend_filter': True,         # Enable trend filtering
    'trend_lookback': 10,             # Candles to check for trend
    'trend_strength_threshold': 0.7,  # % of candles in same direction = strong trend

    # ==========================================================================
    # Trailing Stops (from mean reversion patterns)
    # ==========================================================================
    'use_trailing_stop': True,        # Enable trailing stops
    'trailing_activation_pct': 0.3,   # Activate at 0.3% profit
    'trailing_distance_pct': 0.2,     # Trail 0.2% from high/low

    # ==========================================================================
    # Position Decay (from mean reversion patterns)
    # ==========================================================================
    'use_position_decay': True,       # Enable position decay
    'position_decay_minutes': 5,      # Start decay after 5 minutes
    'position_decay_tp_mult': 0.5,    # Reduce TP target to 50% after decay

    # ==========================================================================
    # Rejection Tracking
    # ==========================================================================
    'track_rejections': True,         # Enable rejection tracking

    # ==========================================================================
    # Correlation Monitoring - REC-021
    # ==========================================================================
    'use_correlation_monitoring': True,   # Enable correlation monitoring
    'correlation_lookback': 20,           # Periods for correlation calculation
    'correlation_warning_threshold': 0.6, # REC-024: Warn if correlation below this (raised from 0.5)
    'correlation_pause_threshold': 0.4,   # REC-024: Pause trading if below this (raised from 0.3)
    'correlation_pause_enabled': True,    # REC-023: Enabled by default for declining XRP/BTC correlation

    # ==========================================================================
    # Dynamic BTC Price - REC-018
    # ==========================================================================
    'btc_price_fallback': 100000.0,   # Fallback BTC/USD price if unavailable
    'btc_price_symbols': ['BTC/USDT', 'BTC/USD'],  # Symbols to check for BTC price

    # ==========================================================================
    # Hedge Ratio - REC-022 (Future Enhancement)
    # ==========================================================================
    'use_hedge_ratio': False,         # Enable hedge ratio optimization
    'hedge_ratio_lookback': 50,       # Periods for hedge ratio calculation

    # ==========================================================================
    # Rebalancing (for future implementation)
    # ==========================================================================
    'rebalance_threshold': 0.3,       # Rebalance when holdings differ by 30%
}


# =============================================================================
# Section 1: Configuration Validation - REC-007
# =============================================================================
def _validate_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate configuration on startup.

    Returns:
        List of error/warning messages (empty if valid)
    """
    errors = []

    # Required positive values
    required_positive = [
        'position_size_usd', 'max_position_usd', 'stop_loss_pct',
        'take_profit_pct', 'lookback_periods', 'cooldown_seconds',
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
    entry_threshold = config.get('entry_threshold', 1.0)
    if entry_threshold < 0.5 or entry_threshold > 3.0:
        errors.append(f"entry_threshold should be 0.5-3.0, got {entry_threshold}")

    exit_threshold = config.get('exit_threshold', 0.5)
    if exit_threshold < 0.1 or exit_threshold > 2.0:
        errors.append(f"exit_threshold should be 0.1-2.0, got {exit_threshold}")

    if exit_threshold >= entry_threshold:
        errors.append(f"exit_threshold ({exit_threshold}) should be < entry_threshold ({entry_threshold})")

    # R:R ratio check
    sl = config.get('stop_loss_pct', 0.6)
    tp = config.get('take_profit_pct', 0.6)
    if sl > 0 and tp > 0:
        rr_ratio = tp / sl
        if rr_ratio < 1.0:
            errors.append(f"Warning: R:R ratio ({rr_ratio:.2f}:1) < 1:1 - unfavorable")
        elif rr_ratio >= 1.0:
            # Info message for acceptable R:R
            pass

    # Spread filter check
    max_spread = config.get('max_spread_pct', 0.10)
    if max_spread > tp * 0.5:
        errors.append(f"Warning: max_spread_pct ({max_spread}%) > 50% of take_profit_pct ({tp}%)")

    return errors


# =============================================================================
# Section 2: Indicator Calculations
# =============================================================================
def _calculate_bollinger_bands(
    prices: List[float],
    lookback: int = 20,
    num_std: float = 2.0
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Calculate Bollinger Bands.

    Returns:
        (sma, upper_band, lower_band, std_dev)
    """
    if len(prices) < lookback:
        return None, None, None, None

    recent = prices[-lookback:]

    # Simple Moving Average
    sma = sum(recent) / len(recent)

    # Standard Deviation
    variance = sum((p - sma) ** 2 for p in recent) / len(recent)
    std_dev = variance ** 0.5

    # Bands
    upper = sma + (num_std * std_dev)
    lower = sma - (num_std * std_dev)

    return sma, upper, lower, std_dev


def _calculate_z_score(price: float, sma: float, std_dev: float) -> float:
    """Calculate z-score (number of std devs from mean)."""
    if std_dev == 0:
        return 0.0
    return (price - sma) / std_dev


def _calculate_volatility(prices: List[float], lookback: int = 20) -> float:
    """
    Calculate price volatility from price history.

    Returns volatility as a percentage (std dev of returns * 100).
    """
    if len(prices) < lookback + 1:
        return 0.0

    recent = prices[-(lookback + 1):]
    if len(recent) < 2:
        return 0.0

    returns = [(recent[i] - recent[i - 1]) / recent[i - 1]
               for i in range(1, len(recent)) if recent[i - 1] != 0]

    if not returns:
        return 0.0

    mean_return = sum(returns) / len(returns)
    variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
    return (variance ** 0.5) * 100


def _calculate_rsi(prices: List[float], period: int = 14) -> float:
    """
    Calculate RSI indicator from price history.

    REC-014: RSI confirmation for signal quality.

    Returns:
        RSI value (0-100), 50.0 if insufficient data
    """
    if len(prices) < period + 1:
        return 50.0  # Neutral

    gains = []
    losses = []

    start_idx = max(1, len(prices) - period)
    for i in range(start_idx, len(prices)):
        change = prices[i] - prices[i - 1]
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


def _detect_trend_strength(
    prices: List[float],
    lookback: int = 10,
    threshold: float = 0.7
) -> Tuple[bool, str, float]:
    """
    Detect if there's a strong trend in recent price action.

    REC-015: Trend detection warning system.

    Returns:
        (is_strong_trend, direction, strength)
        - is_strong_trend: True if trend is strong enough to warn
        - direction: 'up', 'down', or 'neutral'
        - strength: 0.0 to 1.0 (% of candles in same direction)
    """
    if len(prices) < lookback + 1:
        return False, 'neutral', 0.0

    recent = prices[-(lookback + 1):]
    up_moves = 0
    down_moves = 0

    for i in range(1, len(recent)):
        if recent[i] > recent[i - 1]:
            up_moves += 1
        elif recent[i] < recent[i - 1]:
            down_moves += 1

    total_moves = up_moves + down_moves
    if total_moves == 0:
        return False, 'neutral', 0.0

    up_strength = up_moves / total_moves
    down_strength = down_moves / total_moves

    if up_strength >= threshold:
        return True, 'up', up_strength
    elif down_strength >= threshold:
        return True, 'down', down_strength

    return False, 'neutral', max(up_strength, down_strength)


def _calculate_trailing_stop(
    entry_price: float,
    highest_price: float,
    lowest_price: float,
    side: str,
    activation_pct: float,
    trail_distance_pct: float
) -> Optional[float]:
    """
    Calculate trailing stop price.

    From mean reversion patterns.

    Returns:
        Trailing stop price if activated, None otherwise
    """
    if side == 'long':
        profit_pct = (highest_price - entry_price) / entry_price * 100
        if profit_pct >= activation_pct:
            return highest_price * (1 - trail_distance_pct / 100)
    elif side == 'short':
        profit_pct = (entry_price - lowest_price) / entry_price * 100
        if profit_pct >= activation_pct:
            return lowest_price * (1 + trail_distance_pct / 100)
    return None


def _check_position_decay(
    entry_time: datetime,
    current_time: datetime,
    decay_minutes: float
) -> Tuple[bool, float]:
    """
    Check if position has decayed (exceeded time threshold).

    From mean reversion patterns.

    Returns:
        (is_decayed, minutes_held)
    """
    if entry_time is None:
        return False, 0.0

    minutes_held = (current_time - entry_time).total_seconds() / 60
    is_decayed = minutes_held >= decay_minutes

    return is_decayed, minutes_held


def _calculate_rolling_correlation(
    prices_a: List[float],
    prices_b: List[float],
    lookback: int = 20
) -> float:
    """
    Calculate rolling Pearson correlation between two price series.

    REC-021: Rolling correlation monitoring.

    Args:
        prices_a: First price series (e.g., XRP prices in BTC)
        prices_b: Second price series (e.g., BTC prices in USD)
        lookback: Number of periods for correlation calculation

    Returns:
        Correlation coefficient (-1 to 1), 0.0 if insufficient data
    """
    if len(prices_a) < lookback or len(prices_b) < lookback:
        return 0.0

    # Use most recent lookback periods
    a = prices_a[-lookback:]
    b = prices_b[-lookback:]

    if len(a) != len(b):
        # Align lengths
        min_len = min(len(a), len(b))
        a = a[-min_len:]
        b = b[-min_len:]

    if len(a) < 3:
        return 0.0

    # Calculate means
    mean_a = sum(a) / len(a)
    mean_b = sum(b) / len(b)

    # Calculate covariance and standard deviations
    covariance = sum((a[i] - mean_a) * (b[i] - mean_b) for i in range(len(a))) / len(a)
    variance_a = sum((x - mean_a) ** 2 for x in a) / len(a)
    variance_b = sum((x - mean_b) ** 2 for x in b) / len(b)

    std_a = variance_a ** 0.5
    std_b = variance_b ** 0.5

    if std_a == 0 or std_b == 0:
        return 0.0

    correlation = covariance / (std_a * std_b)

    # Clamp to [-1, 1] to handle floating point errors
    return max(-1.0, min(1.0, correlation))


def _get_btc_price_usd(
    data: DataSnapshot,
    config: Dict[str, Any]
) -> float:
    """
    Get BTC/USD price from market data or fallback.

    REC-018: Dynamic BTC price for USD conversion.

    Args:
        data: Market data snapshot
        config: Strategy configuration

    Returns:
        BTC price in USD
    """
    btc_symbols = config.get('btc_price_symbols', ['BTC/USDT', 'BTC/USD'])
    fallback = config.get('btc_price_fallback', 100000.0)

    for symbol in btc_symbols:
        price = data.prices.get(symbol)
        if price and price > 0:
            return price

    return fallback


# =============================================================================
# Section 3: Volatility Regime Classification - REC-004
# =============================================================================
def _classify_volatility_regime(
    volatility_pct: float,
    config: Dict[str, Any]
) -> VolatilityRegime:
    """Classify current volatility into a regime."""
    low = config.get('regime_low_threshold', 0.2)
    medium = config.get('regime_medium_threshold', 0.5)
    high = config.get('regime_high_threshold', 1.0)

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
        # Tighter entry in low volatility
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


def _check_spread(
    data: DataSnapshot,
    symbol: str,
    max_spread_pct: float,
    take_profit_pct: float,
    min_profitability_mult: float
) -> Tuple[bool, float]:
    """
    Check if spread is acceptable for trading.

    Returns:
        (is_acceptable, current_spread_pct)
    """
    ob = data.orderbooks.get(symbol)
    if not ob or not ob.spread_pct:
        return True, 0.0  # Allow trading if no orderbook data

    spread_pct = ob.spread_pct

    # Check against max spread
    if spread_pct > max_spread_pct:
        return False, spread_pct

    # Check profitability: TP must exceed spread by multiplier
    if take_profit_pct < spread_pct * min_profitability_mult:
        return False, spread_pct

    return True, spread_pct


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


def _track_exit(
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


def _build_base_indicators(
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


# =============================================================================
# Section 6: State Initialization
# =============================================================================
def _initialize_state(state: Dict[str, Any]) -> None:
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

    # Entry tracking with trailing stop support
    state['position_entries'] = {}
    state['highest_price_since_entry'] = {}
    state['lowest_price_since_entry'] = {}

    # Fill history
    state['fills'] = []


# =============================================================================
# Section 7: Price History Management
# =============================================================================
def _update_price_history(
    data: DataSnapshot,
    state: Dict[str, Any],
    symbol: str,
    current_price: float,
    max_history: int = 50
) -> List[float]:
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


# =============================================================================
# Section 8: Signal Generation Helpers
# =============================================================================
def _calculate_position_size(
    config: Dict[str, Any],
    current_position_usd: float,
    regime_size_mult: float
) -> Tuple[float, float]:
    """
    Calculate actual position size in USD.

    Returns:
        (actual_size_usd, available_usd)
    """
    max_position = config.get('max_position_usd', 50.0)
    base_size = config.get('position_size_usd', 15.0)
    min_trade_size = config.get('min_trade_size_usd', 5.0)

    available = max_position - current_position_usd
    adjusted_size = base_size * regime_size_mult
    actual_size = min(adjusted_size, available)

    if actual_size < min_trade_size:
        actual_size = 0.0

    return actual_size, available


def _convert_usd_to_xrp(
    usd_amount: float,
    price_btc_per_xrp: float,
    btc_price_usd: float
) -> float:
    """
    Convert USD amount to XRP for ratio trading.

    REC-018: Now uses dynamic BTC price instead of hardcoded fallback.

    For XRP/BTC pair, we need to convert through BTC:
    USD -> BTC -> XRP

    Args:
        usd_amount: Amount in USD to convert
        price_btc_per_xrp: Current XRP/BTC price (BTC per XRP)
        btc_price_usd: Current BTC/USD price

    Returns:
        Equivalent XRP amount
    """
    if btc_price_usd <= 0 or price_btc_per_xrp <= 0:
        return 0.0

    btc_amount = usd_amount / btc_price_usd
    xrp_amount = btc_amount / price_btc_per_xrp

    return xrp_amount


def _generate_buy_signal(
    symbol: str,
    price: float,
    size_usd: float,
    z_score: float,
    entry_threshold: float,
    sl_pct: float,
    tp_pct: float,
    regime_name: str,
    Signal
) -> Signal:
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


def _generate_sell_signal(
    symbol: str,
    price: float,
    size_usd: float,
    z_score: float,
    entry_threshold: float,
    sl_pct: float,
    tp_pct: float,
    regime_name: str,
    Signal
) -> Signal:
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


def _generate_exit_signal(
    symbol: str,
    price: float,
    size_usd: float,
    z_score: float,
    exit_threshold: float,
    Signal
) -> Signal:
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


# =============================================================================
# REQUIRED: Main Signal Generation Function
# =============================================================================
def generate_signal(
    data: DataSnapshot,
    config: Dict[str, Any],
    state: Dict[str, Any]
) -> Optional[Signal]:
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
        _initialize_state(state)

    symbol = SYMBOLS[0]  # XRP/BTC
    current_time = data.timestamp
    track_rejections = config.get('track_rejections', True)

    # REC-005: Circuit breaker check
    if config.get('use_circuit_breaker', True):
        max_losses = config.get('max_consecutive_losses', 3)
        cooldown_min = config.get('circuit_breaker_minutes', 15)

        if _check_circuit_breaker(state, current_time, max_losses, cooldown_min):
            state['indicators'] = _build_base_indicators(
                symbol=symbol, status='circuit_breaker', state=state
            )
            state['indicators']['circuit_breaker_active'] = True
            if track_rejections:
                _track_rejection(state, RejectionReason.CIRCUIT_BREAKER, symbol)
            return None

    # Time-based cooldown
    if state['last_signal_time'] is not None:
        elapsed = (current_time - state['last_signal_time']).total_seconds()
        cooldown = config.get('cooldown_seconds', 30.0)
        if elapsed < cooldown:
            state['indicators'] = _build_base_indicators(
                symbol=symbol, status='cooldown', state=state
            )
            state['indicators']['cooldown_remaining'] = round(cooldown - elapsed, 1)
            if track_rejections:
                _track_rejection(state, RejectionReason.TIME_COOLDOWN, symbol)
            return None

    # Get current price
    price = data.prices.get(symbol)
    if not price:
        state['indicators'] = _build_base_indicators(
            symbol=symbol, status='no_price', state=state
        )
        if track_rejections:
            _track_rejection(state, RejectionReason.NO_PRICE_DATA, symbol)
        return None

    # Update price history
    price_history = _update_price_history(data, state, symbol, price)

    # REC-018: Get dynamic BTC price for USD conversion
    btc_price_usd = _get_btc_price_usd(data, config)
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
        state['indicators'] = _build_base_indicators(
            symbol=symbol, status='warming_up', state=state, price=price
        )
        state['indicators']['candles_available'] = len(price_history)
        state['indicators']['candles_required'] = min_candles
        if track_rejections:
            _track_rejection(state, RejectionReason.WARMING_UP, symbol)
        return None

    # REC-008: Spread monitoring
    tp_pct = config.get('take_profit_pct', 0.6)
    if config.get('use_spread_filter', True):
        max_spread = config.get('max_spread_pct', 0.10)
        min_profit_mult = config.get('min_profitability_mult', 0.5)

        spread_ok, current_spread = _check_spread(data, symbol, max_spread, tp_pct, min_profit_mult)
        if not spread_ok:
            state['indicators'] = _build_base_indicators(
                symbol=symbol, status='spread_too_wide', state=state, price=price
            )
            state['indicators']['current_spread_pct'] = round(current_spread, 4)
            state['indicators']['max_spread_pct'] = max_spread
            if track_rejections:
                _track_rejection(state, RejectionReason.SPREAD_TOO_WIDE, symbol)
            return None
    else:
        current_spread = 0.0

    # Calculate Bollinger Bands
    # REC-036: Optionally use wider bands for crypto volatility
    lookback = config.get('lookback_periods', 20)
    if config.get('use_crypto_bollinger_std', False):
        num_std = config.get('bollinger_std_crypto', 2.5)
    else:
        num_std = config.get('bollinger_std', 2.0)

    sma, upper, lower, std_dev = _calculate_bollinger_bands(
        price_history,
        lookback,
        num_std
    )

    if sma is None:
        return None

    # Calculate z-score
    z_score = _calculate_z_score(price, sma, std_dev)

    # REC-004: Calculate volatility and classify regime
    volatility = _calculate_volatility(price_history, config.get('volatility_lookback', 20))
    regime = VolatilityRegime.MEDIUM
    regime_adjustments = {'threshold_mult': 1.0, 'size_mult': 1.0, 'pause_trading': False}

    if config.get('use_volatility_regimes', True):
        regime = _classify_volatility_regime(volatility, config)
        regime_adjustments = _get_regime_adjustments(regime, config)

        if regime_adjustments['pause_trading']:
            state['indicators'] = _build_base_indicators(
                symbol=symbol, status='regime_pause', state=state, price=price
            )
            state['indicators']['volatility_regime'] = regime.name
            state['indicators']['volatility_pct'] = round(volatility, 4)
            if track_rejections:
                _track_rejection(state, RejectionReason.REGIME_PAUSE, symbol)
            return None

    # REC-021: Correlation monitoring
    correlation = 0.0
    use_correlation = config.get('use_correlation_monitoring', True)
    if use_correlation:
        corr_lookback = config.get('correlation_lookback', 20)
        btc_history = state.get('btc_price_history', [])

        if len(price_history) >= corr_lookback and len(btc_history) >= corr_lookback:
            correlation = _calculate_rolling_correlation(
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
                state['indicators'] = _build_base_indicators(
                    symbol=symbol, status='correlation_pause', state=state, price=price
                )
                state['indicators']['correlation'] = round(correlation, 4)
                state['indicators']['correlation_threshold'] = pause_threshold
                if track_rejections:
                    _track_rejection(state, RejectionReason.CORRELATION_TOO_LOW, symbol)
                return None

    # Entry/exit thresholds with regime adjustment
    base_entry_threshold = config.get('entry_threshold', 1.5)  # REC-013: Higher default
    effective_entry_threshold = base_entry_threshold * regime_adjustments['threshold_mult']
    exit_threshold = config.get('exit_threshold', 0.5)

    # REC-002: Position sizing in USD
    current_position_usd = state.get('position_usd', 0)
    actual_size_usd, available_usd = _calculate_position_size(
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
    rsi = _calculate_rsi(price_history, rsi_period) if use_rsi else 50.0

    # REC-015: Detect trend strength if enabled
    use_trend_filter = config.get('use_trend_filter', True)  # Default matches CONFIG
    trend_lookback = config.get('trend_lookback', 10)
    trend_threshold = config.get('trend_strength_threshold', 0.7)
    is_strong_trend, trend_direction, trend_strength = _detect_trend_strength(
        price_history, trend_lookback, trend_threshold
    ) if use_trend_filter else (False, 'neutral', 0.0)

    # Trailing stop and position decay config
    use_trailing = config.get('use_trailing_stop', False)
    trailing_activation = config.get('trailing_activation_pct', 0.3)
    trailing_distance = config.get('trailing_distance_pct', 0.2)
    use_decay = config.get('use_position_decay', False)
    decay_minutes = config.get('position_decay_minutes', 5)
    decay_tp_mult = config.get('position_decay_tp_mult', 0.5)

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
    }

    # Position limit check
    max_position = config.get('max_position_usd', 50.0)
    if current_position_usd >= max_position:
        state['indicators']['status'] = 'max_position_reached'
        if track_rejections:
            _track_rejection(state, RejectionReason.MAX_POSITION, symbol)
        return None

    if actual_size_usd <= 0:
        state['indicators']['status'] = 'insufficient_size'
        if track_rejections:
            _track_rejection(state, RejectionReason.INSUFFICIENT_SIZE, symbol)
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

        trailing_stop_price = _calculate_trailing_stop(
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
                _track_exit(state, ExitReason.TRAILING_STOP, symbol)
                state['last_signal_time'] = current_time
                state['trade_count'] += 1
                return signal

    # ==========================================================================
    # Check position decay if position exists
    # ==========================================================================
    if current_position_usd > 0 and use_decay:
        entry_info = state.get('position_entries', {}).get(symbol, {})
        entry_time = entry_info.get('entry_time')

        is_decayed, minutes_held = _check_position_decay(entry_time, current_time, decay_minutes)
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
                _track_exit(state, ExitReason.POSITION_DECAY, symbol)
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
                _track_rejection(state, RejectionReason.STRONG_TREND_DETECTED, symbol)
            # Don't return None - just skip buy signal, check other conditions

        # REC-014: RSI confirmation check - buy only if oversold
        elif use_rsi and rsi > rsi_oversold:
            state['indicators']['status'] = 'rsi_not_confirmed'
            state['indicators']['rsi_required'] = f'RSI {rsi:.1f} > {rsi_oversold} (not oversold)'
            if track_rejections:
                _track_rejection(state, RejectionReason.RSI_NOT_CONFIRMED, symbol)
            # Don't return None - just skip buy signal

        else:
            # Trade flow confirmation if enabled
            if use_trade_flow:
                if not _is_trade_flow_aligned(data, symbol, 'buy', trade_flow_threshold):
                    state['indicators']['status'] = 'trade_flow_not_aligned'
                    state['indicators']['trade_flow_aligned'] = False
                    if track_rejections:
                        _track_rejection(state, RejectionReason.TRADE_FLOW_NOT_ALIGNED, symbol)
                    return None

            state['indicators']['trade_flow_aligned'] = True
            signal = _generate_buy_signal(
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
                _track_rejection(state, RejectionReason.STRONG_TREND_DETECTED, symbol)
            # Don't return None - just skip sell signal

        # REC-014: RSI confirmation check - sell only if overbought
        elif use_rsi and rsi < rsi_overbought:
            state['indicators']['status'] = 'rsi_not_confirmed'
            state['indicators']['rsi_required'] = f'RSI {rsi:.1f} < {rsi_overbought} (not overbought)'
            if track_rejections:
                _track_rejection(state, RejectionReason.RSI_NOT_CONFIRMED, symbol)
            # Don't return None - just skip sell signal

        else:
            # Trade flow confirmation if enabled
            if use_trade_flow:
                if not _is_trade_flow_aligned(data, symbol, 'sell', trade_flow_threshold):
                    state['indicators']['status'] = 'trade_flow_not_aligned'
                    state['indicators']['trade_flow_aligned'] = False
                    if track_rejections:
                        _track_rejection(state, RejectionReason.TRADE_FLOW_NOT_ALIGNED, symbol)
                    return None

            state['indicators']['trade_flow_aligned'] = True

            if current_position_usd > 0:
                # Sell from our position
                sell_size = min(actual_size_usd, current_position_usd)
                signal = _generate_sell_signal(
                    symbol, price, sell_size, z_score,
                    effective_entry_threshold, sl_pct, tp_pct, regime.name, Signal
                )
            else:
                # No position but still signal for accumulating BTC
                # Sell from "starting XRP holdings" concept
                signal = _generate_sell_signal(
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
            signal = _generate_exit_signal(
                symbol, price, exit_size, z_score, exit_threshold, Signal
            )

    if signal:
        state['last_signal_time'] = current_time
        state['trade_count'] += 1
        state['indicators']['status'] = 'signal_generated'
        return signal

    state['indicators']['status'] = 'no_signal'
    if track_rejections:
        _track_rejection(state, RejectionReason.NO_SIGNAL_CONDITIONS, symbol)
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
            print(f"[ratio_trading] Config warning: {error}")
        state['config_warnings'] = errors
    state['config_validated'] = True

    _initialize_state(state)

    # Store config values in state for use in on_fill (fixes hardcoded max_losses)
    state['max_consecutive_losses'] = config.get('max_consecutive_losses', 3)

    print(f"[ratio_trading] v{STRATEGY_VERSION} started")
    print(f"[ratio_trading] Symbol: {SYMBOLS[0]} (ratio pair)")
    print(f"[ratio_trading] Entry threshold: {config.get('entry_threshold', 1.5)} std (REC-013)")
    print(f"[ratio_trading] Core Features: VolatilityRegimes={config.get('use_volatility_regimes', True)}, "
          f"CircuitBreaker={config.get('use_circuit_breaker', True)}, "
          f"SpreadFilter={config.get('use_spread_filter', True)}")
    print(f"[ratio_trading] v2.1 Features: RSI={config.get('use_rsi_confirmation', True)}, "
          f"TrendFilter={config.get('use_trend_filter', True)}, "
          f"TrailingStop={config.get('use_trailing_stop', True)}, "
          f"PositionDecay={config.get('use_position_decay', True)}")
    print(f"[ratio_trading] v3.0 Features: CorrelationMonitoring={config.get('use_correlation_monitoring', True)}, "
          f"DynamicBTCPrice=True, SeparateExitTracking=True")
    print(f"[ratio_trading] v4.0 Features: CorrelationPauseEnabled={config.get('correlation_pause_enabled', True)}, "
          f"RaisedThresholds=True (research-validated)")
    # REC-036: Crypto Bollinger Bands
    bollinger_std = config.get('bollinger_std_crypto', 2.5) if config.get('use_crypto_bollinger_std', False) else config.get('bollinger_std', 2.0)
    print(f"[ratio_trading] v4.1 Features: CryptoBollingerStd={config.get('use_crypto_bollinger_std', False)} "
          f"(std={bollinger_std})")
    if config.get('use_correlation_monitoring', True):
        print(f"[ratio_trading] Correlation (REC-023/024): warn<{config.get('correlation_warning_threshold', 0.6)}, "
              f"pause<{config.get('correlation_pause_threshold', 0.4)} "
              f"(pause_enabled={config.get('correlation_pause_enabled', True)})")
    print(f"[ratio_trading] Position sizing: {config.get('position_size_usd', 15.0)} USD, "
          f"Max: {config.get('max_position_usd', 50.0)} USD")
    print(f"[ratio_trading] R:R ratio: {config.get('take_profit_pct', 0.6)}/{config.get('stop_loss_pct', 0.6)} "
          f"({config.get('take_profit_pct', 0.6)/config.get('stop_loss_pct', 0.6):.2f}:1)")


def on_fill(fill: Dict[str, Any], state: Dict[str, Any]) -> None:
    """
    Track fills and update position/accumulation state.

    For XRP/BTC:
    - Buy: Spent BTC to get XRP → +XRP position
    - Sell: Sold XRP to get BTC → -XRP position, +BTC accumulated

    REC-006: Per-pair PnL tracking
    REC-005: Circuit breaker consecutive loss tracking
    REC-016: Enhanced accumulation metrics
    REC-018: Dynamic BTC price for USD conversion
    """
    side = fill.get('side', '')
    symbol = fill.get('symbol', SYMBOLS[0])
    size = fill.get('size', 0)  # Size in USD now - REC-002
    price = fill.get('price', 0)  # Price in BTC per XRP
    pnl = fill.get('pnl', 0)
    timestamp = fill.get('timestamp', datetime.now())

    value = fill.get('value', size)  # USD value

    # REC-018: Get BTC price from state (set in generate_signal) or fallback
    btc_price_usd = state.get('last_btc_price_usd', 100000.0)

    # Convert USD to approximate XRP for tracking
    if price > 0:
        xrp_amount = _convert_usd_to_xrp(value, price, btc_price_usd)
    else:
        xrp_amount = 0

    btc_value = xrp_amount * price

    # Initialize tracking dicts
    for key in ['pnl_by_symbol', 'trades_by_symbol', 'wins_by_symbol',
                'losses_by_symbol', 'position_entries', 'highest_price_since_entry',
                'lowest_price_since_entry']:
        if key not in state:
            state[key] = {}

    # REC-006: Per-pair PnL tracking
    if pnl != 0:
        state['pnl_by_symbol'][symbol] = state['pnl_by_symbol'].get(symbol, 0) + pnl

        # REC-005: Circuit breaker tracking (fixed: use state config instead of hardcoded)
        if pnl > 0:
            state['wins_by_symbol'][symbol] = state['wins_by_symbol'].get(symbol, 0) + 1
            state['consecutive_losses'] = 0
        else:
            state['losses_by_symbol'][symbol] = state['losses_by_symbol'].get(symbol, 0) + 1
            state['consecutive_losses'] = state.get('consecutive_losses', 0) + 1

            # Use config value from state (set in on_start) instead of hardcoded
            max_losses = state.get('max_consecutive_losses', 3)
            if state['consecutive_losses'] >= max_losses:
                state['circuit_breaker_time'] = timestamp

    state['trades_by_symbol'][symbol] = state['trades_by_symbol'].get(symbol, 0) + 1

    # Position tracking (USD and XRP)
    if side == 'buy':
        # Bought XRP with BTC
        state['position_usd'] = state.get('position_usd', 0) + value
        state['position_xrp'] = state.get('position_xrp', 0) + xrp_amount
        state['xrp_accumulated'] = state.get('xrp_accumulated', 0) + xrp_amount

        # REC-016: Track USD value at time of acquisition
        state['xrp_accumulated_value_usd'] = state.get('xrp_accumulated_value_usd', 0) + value
        state['total_trades_xrp_bought'] = state.get('total_trades_xrp_bought', 0) + 1

        if symbol not in state['position_entries']:
            state['position_entries'][symbol] = {
                'entry_price': price,
                'entry_time': timestamp,
                'side': 'long',
            }
            # Initialize trailing stop tracking
            state['highest_price_since_entry'][symbol] = price
            state['lowest_price_since_entry'][symbol] = price

    elif side == 'sell':
        # Sold XRP for BTC
        state['position_usd'] = max(0, state.get('position_usd', 0) - value)
        state['position_xrp'] = max(0, state.get('position_xrp', 0) - xrp_amount)
        state['btc_accumulated'] = state.get('btc_accumulated', 0) + btc_value

        # REC-016: Track USD value at time of BTC acquisition
        state['btc_accumulated_value_usd'] = state.get('btc_accumulated_value_usd', 0) + value
        state['total_trades_btc_bought'] = state.get('total_trades_btc_bought', 0) + 1

        if state['position_usd'] < 0.01:
            state['position_usd'] = 0.0
            state['position_xrp'] = 0.0
            if symbol in state['position_entries']:
                del state['position_entries'][symbol]
            # Clean up trailing stop tracking
            if symbol in state.get('highest_price_since_entry', {}):
                del state['highest_price_since_entry'][symbol]
            if symbol in state.get('lowest_price_since_entry', {}):
                del state['lowest_price_since_entry'][symbol]

    # Track fill history
    if 'fills' not in state:
        state['fills'] = []
    state['fills'].append(fill)
    state['fills'] = state['fills'][-20:]  # Keep last 20

    state['last_fill'] = fill


def on_stop(state: Dict[str, Any]) -> None:
    """
    Called when strategy stops.

    Logs comprehensive summary of trading performance.
    REC-016: Enhanced accumulation metrics in summary.
    REC-020: Exit tracking statistics.
    REC-021: Correlation monitoring summary.
    """
    symbol = SYMBOLS[0]

    total_pnl = sum(state.get('pnl_by_symbol', {}).values())
    total_trades = sum(state.get('trades_by_symbol', {}).values())
    total_wins = sum(state.get('wins_by_symbol', {}).values())
    total_losses = sum(state.get('losses_by_symbol', {}).values())

    win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0

    rejection_counts = state.get('rejection_counts', {})
    total_rejections = sum(rejection_counts.values())

    # REC-020: Exit tracking statistics
    exit_counts = state.get('exit_counts', {})
    exit_pnl_by_reason = state.get('exit_pnl_by_reason', {})
    total_exits = sum(exit_counts.values())

    # REC-021: Correlation monitoring statistics
    correlation_warnings = state.get('correlation_warnings', 0)
    correlation_history = state.get('correlation_history', [])
    avg_correlation = sum(correlation_history) / len(correlation_history) if correlation_history else 0

    # REC-016: Enhanced accumulation metrics
    xrp_acc = state.get('xrp_accumulated', 0)
    btc_acc = state.get('btc_accumulated', 0)
    xrp_value_usd = state.get('xrp_accumulated_value_usd', 0)
    btc_value_usd = state.get('btc_accumulated_value_usd', 0)
    xrp_trades = state.get('total_trades_xrp_bought', 0)
    btc_trades = state.get('total_trades_btc_bought', 0)

    state['indicators'] = {}

    state['final_summary'] = {
        'symbol': symbol,
        'position_usd': state.get('position_usd', 0),
        'position_xrp': state.get('position_xrp', 0),
        'xrp_accumulated': xrp_acc,
        'btc_accumulated': btc_acc,
        # REC-016: Enhanced metrics
        'xrp_accumulated_value_usd': xrp_value_usd,
        'btc_accumulated_value_usd': btc_value_usd,
        'total_trades_xrp_bought': xrp_trades,
        'total_trades_btc_bought': btc_trades,
        'avg_xrp_buy_value_usd': xrp_value_usd / xrp_trades if xrp_trades > 0 else 0,
        'avg_btc_buy_value_usd': btc_value_usd / btc_trades if btc_trades > 0 else 0,
        'pnl_by_symbol': state.get('pnl_by_symbol', {}),
        'trades_by_symbol': state.get('trades_by_symbol', {}),
        'wins_by_symbol': state.get('wins_by_symbol', {}),
        'losses_by_symbol': state.get('losses_by_symbol', {}),
        'total_pnl': total_pnl,
        'total_trades': total_trades,
        'total_wins': total_wins,
        'total_losses': total_losses,
        'win_rate': win_rate,
        'trade_count': state.get('trade_count', 0),
        'total_fills': len(state.get('fills', [])),
        'config_warnings': state.get('config_warnings', []),
        'rejection_counts': rejection_counts,
        'rejection_counts_by_symbol': state.get('rejection_counts_by_symbol', {}),
        'total_rejections': total_rejections,
        # REC-020: Exit tracking
        'exit_counts': exit_counts,
        'exit_counts_by_symbol': state.get('exit_counts_by_symbol', {}),
        'exit_pnl_by_reason': exit_pnl_by_reason,
        'total_exits': total_exits,
        # REC-021: Correlation monitoring
        'correlation_warnings': correlation_warnings,
        'avg_correlation': avg_correlation,
        'last_btc_price_usd': state.get('last_btc_price_usd'),
    }

    print(f"[ratio_trading] Stopped. PnL: ${total_pnl:.4f}, Trades: {total_trades}, Win Rate: {win_rate:.1f}%")

    # Print symbol summary
    sym_pnl = state.get('pnl_by_symbol', {}).get(symbol, 0)
    sym_trades = state.get('trades_by_symbol', {}).get(symbol, 0)
    sym_wins = state.get('wins_by_symbol', {}).get(symbol, 0)
    sym_losses = state.get('losses_by_symbol', {}).get(symbol, 0)
    sym_wr = (sym_wins / sym_trades * 100) if sym_trades > 0 else 0
    print(f"[ratio_trading]   {symbol}: PnL=${sym_pnl:.6f}, Trades={sym_trades}, WR={sym_wr:.1f}%")

    # Print accumulation summary (unique to ratio trading) - REC-016: Enhanced
    print(f"[ratio_trading]   Accumulated: XRP={xrp_acc:.4f} (${xrp_value_usd:.2f} cost, {xrp_trades} trades)")
    print(f"[ratio_trading]   Accumulated: BTC={btc_acc:.8f} (${btc_value_usd:.2f} value, {btc_trades} trades)")

    # REC-020: Print exit tracking summary
    if exit_counts:
        print(f"[ratio_trading] Intentional exits ({total_exits} total):")
        for reason, count in sorted(exit_counts.items(), key=lambda x: -x[1]):
            pnl_for_reason = exit_pnl_by_reason.get(reason, 0)
            print(f"[ratio_trading]   - {reason}: {count} (PnL: ${pnl_for_reason:.4f})")

    # REC-021: Print correlation summary
    if correlation_history:
        print(f"[ratio_trading] Correlation: avg={avg_correlation:.4f}, warnings={correlation_warnings}")

    # Print rejection summary
    if rejection_counts:
        print(f"[ratio_trading] Signal rejections ({total_rejections} total):")
        for reason, count in sorted(rejection_counts.items(), key=lambda x: -x[1])[:5]:
            print(f"[ratio_trading]   - {reason}: {count}")
