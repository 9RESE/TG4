"""
Order Flow Strategy v4.1.0

Trades based on trade tape analysis and buy/sell imbalance.
Enhanced with VPIN, volatility regimes, session awareness, and advanced risk management.

Version History:
- 1.0.0: Initial implementation
- 2.0.0: Added volatility adjustment, dynamic thresholds
- 3.0.0: Added per-pair PnL, trade flow confirmation, fee check, circuit breaker
- 3.1.0: Fixed asymmetric thresholds, config validation
- 4.0.0: Major refactor per order-flow-strategy-review-v3.1.md
         - REC-001: VPIN (Volume-Synchronized Probability of Informed Trading)
         - REC-002: Volatility regime classification (LOW/MEDIUM/HIGH/EXTREME)
         - REC-003: Time-of-day session awareness (Asia/Europe/US/Overlap)
         - REC-004: Progressive position decay (gradual TP reduction)
         - REC-005: Cross-pair correlation management
- 4.1.0: Improvements per order-flow-strategy-review-v4.0.md
         - REC-001: Signal rejection logging and statistics
         - REC-002: Configuration override validation with type checking
         - REC-003: Configurable session boundaries (DST-aware)
         - REC-004: Enhanced position decay with close-at-profit-after-fees option
         - Finding #1: Improved VPIN bucket overflow logic
         - Finding #5: Better position decay exit at intermediate stages
"""
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum, auto

from ws_tester.types import DataSnapshot, Signal, OrderbookSnapshot


# =============================================================================
# REQUIRED: Strategy Metadata
# =============================================================================
STRATEGY_NAME = "order_flow"
STRATEGY_VERSION = "4.1.0"
SYMBOLS = ["XRP/USDT", "BTC/USDT"]


# =============================================================================
# Enums for Type Safety
# =============================================================================
class VolatilityRegime(Enum):
    """Volatility regime classification."""
    LOW = auto()       # volatility < low_threshold
    MEDIUM = auto()    # low_threshold - medium_threshold
    HIGH = auto()      # medium_threshold - high_threshold
    EXTREME = auto()   # > high_threshold


class TradingSession(Enum):
    """Trading session classification."""
    ASIA = auto()
    EUROPE = auto()
    US = auto()
    US_EUROPE_OVERLAP = auto()


class RejectionReason(Enum):
    """Signal rejection reasons for tracking."""
    CIRCUIT_BREAKER = "circuit_breaker"
    TIME_COOLDOWN = "time_cooldown"
    TRADE_COOLDOWN = "trade_cooldown"
    WARMING_UP = "warming_up"
    REGIME_PAUSE = "regime_pause"
    VPIN_PAUSE = "vpin_pause"
    NO_VOLUME = "no_volume"
    NO_PRICE_DATA = "no_price_data"
    MAX_POSITION = "max_position"
    INSUFFICIENT_SIZE = "insufficient_size"
    NOT_FEE_PROFITABLE = "not_fee_profitable"
    TRADE_FLOW_NOT_ALIGNED = "trade_flow_not_aligned"
    CORRELATION_LIMIT = "correlation_limit"
    NO_SIGNAL_CONDITIONS = "no_signal_conditions"


# =============================================================================
# REQUIRED: Default Configuration
# =============================================================================
CONFIG = {
    # ==========================================================================
    # Core Order Flow Parameters
    # ==========================================================================
    'imbalance_threshold': 0.30,        # Default threshold (fallback)
    'buy_imbalance_threshold': 0.30,    # Threshold for buy signals
    'sell_imbalance_threshold': 0.25,   # Lower for sell (selling pressure more impactful)
    'use_asymmetric_thresholds': True,  # Enable asymmetric buy/sell thresholds
    'volume_spike_mult': 2.0,           # Volume spike multiplier
    'lookback_trades': 50,              # Base number of trades to analyze

    # ==========================================================================
    # Position Sizing
    # ==========================================================================
    'position_size_usd': 25.0,          # Size per trade in USD
    'max_position_usd': 100.0,          # Maximum position exposure per pair
    'min_trade_size_usd': 5.0,          # Minimum USD per trade

    # ==========================================================================
    # Risk Management - 2:1 R:R ratio
    # ==========================================================================
    'take_profit_pct': 1.0,             # Take profit at 1.0%
    'stop_loss_pct': 0.5,               # Stop loss at 0.5%

    # ==========================================================================
    # Cooldown Mechanisms
    # ==========================================================================
    'cooldown_trades': 10,              # Min trades between signals
    'cooldown_seconds': 5.0,            # Min time between signals

    # ==========================================================================
    # Volatility Parameters
    # ==========================================================================
    'base_volatility_pct': 0.5,         # Baseline volatility for scaling
    'volatility_lookback': 20,          # Candles for volatility calculation
    'volatility_threshold_mult': 1.5,   # Max threshold multiplier

    # ==========================================================================
    # VPIN (Volume-Synchronized Probability of Informed Trading)
    # ==========================================================================
    'use_vpin': True,                   # Enable VPIN calculation
    'vpin_bucket_count': 50,            # Number of volume buckets
    'vpin_high_threshold': 0.7,         # High VPIN = potential informed trading
    'vpin_pause_on_high': True,         # Pause trading when VPIN > threshold
    'vpin_lookback_trades': 200,        # Trades for VPIN calculation

    # ==========================================================================
    # Volatility Regime Classification
    # ==========================================================================
    'use_volatility_regimes': True,     # Enable regime-based adjustments
    'regime_low_threshold': 0.3,        # Below = LOW regime
    'regime_medium_threshold': 0.8,     # Below = MEDIUM regime
    'regime_high_threshold': 1.5,       # Below = HIGH regime, above = EXTREME
    'regime_extreme_reduce_size': 0.5,  # Position size multiplier in EXTREME
    'regime_extreme_pause': False,      # Pause trading in EXTREME (conservative)

    # ==========================================================================
    # Time-of-Day Session Awareness (REC-003: Configurable Boundaries)
    # ==========================================================================
    'use_session_awareness': True,      # Enable session-based adjustments
    # Session boundaries in UTC (configurable for DST adjustments)
    'session_boundaries': {
        'asia_start': 0,                # 00:00 UTC
        'asia_end': 8,                  # 08:00 UTC
        'europe_start': 8,              # 08:00 UTC
        'europe_end': 14,               # 14:00 UTC
        'overlap_start': 14,            # 14:00 UTC (US/Europe overlap)
        'overlap_end': 17,              # 17:00 UTC
        'us_start': 17,                 # 17:00 UTC
        'us_end': 21,                   # 21:00 UTC
    },
    'session_threshold_multipliers': {
        'ASIA': 1.2,                    # Wider thresholds (lower volume)
        'EUROPE': 1.0,                  # Standard thresholds
        'US': 1.0,                      # Standard thresholds
        'US_EUROPE_OVERLAP': 0.85,      # Tighter thresholds (peak liquidity)
    },
    'session_size_multipliers': {
        'ASIA': 0.8,                    # Smaller sizes (lower liquidity)
        'EUROPE': 1.0,                  # Standard sizes
        'US': 1.0,                      # Standard sizes
        'US_EUROPE_OVERLAP': 1.1,       # Larger sizes (peak liquidity)
    },

    # ==========================================================================
    # Progressive Position Decay (REC-004: Enhanced with profit-after-fees)
    # ==========================================================================
    'use_position_decay': True,         # Enable time-based position decay
    'position_decay_stages': [
        # (age_seconds, tp_multiplier)
        (180, 0.90),                    # 3 min: 90% of original TP
        (240, 0.75),                    # 4 min: 75% of original TP
        (300, 0.50),                    # 5 min: 50% of original TP
        (360, 0.0),                     # 6+ min: Close at any profit
    ],
    # REC-004: Allow closing at any profit > fees during intermediate stages
    'decay_close_at_profit_after_fees': True,
    'decay_min_profit_after_fees_pct': 0.05,  # Minimum profit after fees for early close

    # ==========================================================================
    # Cross-Pair Correlation Management
    # ==========================================================================
    'use_correlation_management': True,  # Enable cross-pair exposure limits
    'max_total_long_exposure': 150.0,   # Max total long USD exposure
    'max_total_short_exposure': 150.0,  # Max total short USD exposure
    'same_direction_size_mult': 0.75,   # Reduce size if both pairs same direction

    # ==========================================================================
    # VWAP Parameters
    # ==========================================================================
    'vwap_deviation_threshold': 0.001,  # Min deviation from VWAP for reversion
    'vwap_reversion_size_mult': 0.75,   # Position size multiplier for VWAP reversion
    'vwap_reversion_threshold_mult': 0.7,  # Threshold multiplier for VWAP reversion

    # ==========================================================================
    # Trade Flow Confirmation
    # ==========================================================================
    'use_trade_flow_confirmation': True,
    'trade_flow_threshold': 0.15,       # Minimum trade flow alignment

    # ==========================================================================
    # Fee Profitability
    # ==========================================================================
    'fee_rate': 0.001,                  # 0.1% per trade
    'min_profit_after_fees_pct': 0.05,  # Minimum profit after fees
    'use_fee_check': True,              # Enable fee profitability check

    # ==========================================================================
    # Micro-Price
    # ==========================================================================
    'use_micro_price': True,            # Use volume-weighted micro-price

    # ==========================================================================
    # Trailing Stops
    # ==========================================================================
    'use_trailing_stop': False,         # Enable trailing stops
    'trailing_stop_activation': 0.3,    # Activate after 0.3% profit
    'trailing_stop_distance': 0.2,      # Trail at 0.2% from high

    # ==========================================================================
    # Circuit Breaker
    # ==========================================================================
    'use_circuit_breaker': True,        # Enable consecutive loss circuit breaker
    'max_consecutive_losses': 3,        # Max losses before cooldown
    'circuit_breaker_minutes': 15,      # Cooldown after max losses

    # ==========================================================================
    # REC-001: Signal Rejection Logging
    # ==========================================================================
    'track_rejections': True,           # Enable rejection tracking
}

# Per-symbol configurations
SYMBOL_CONFIGS = {
    'XRP/USDT': {
        'buy_imbalance_threshold': 0.30,
        'sell_imbalance_threshold': 0.25,
        'imbalance_threshold': 0.30,
        'position_size_usd': 25.0,
        'volume_spike_mult': 2.0,
        'take_profit_pct': 1.0,
        'stop_loss_pct': 0.5,
    },
    'BTC/USDT': {
        'buy_imbalance_threshold': 0.25,
        'sell_imbalance_threshold': 0.20,
        'imbalance_threshold': 0.25,
        'position_size_usd': 50.0,
        'volume_spike_mult': 1.8,
        'take_profit_pct': 0.8,
        'stop_loss_pct': 0.4,
    },
}


# =============================================================================
# Section 1: Configuration and Validation
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
        'position_size_usd', 'max_position_usd', 'stop_loss_pct',
        'take_profit_pct', 'lookback_trades', 'cooldown_seconds',
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
    imbalance = config.get('imbalance_threshold', 0.3)
    if imbalance < 0.1 or imbalance > 0.8:
        errors.append(f"imbalance_threshold should be 0.1-0.8, got {imbalance}")

    fee_rate = config.get('fee_rate', 0.001)
    if fee_rate < 0 or fee_rate > 0.01:
        errors.append(f"fee_rate should be 0-0.01, got {fee_rate}")

    # R:R ratio warning
    sl = config.get('stop_loss_pct', 0.5)
    tp = config.get('take_profit_pct', 1.0)
    if sl > 0 and tp > 0:
        rr_ratio = tp / sl
        if rr_ratio < 1.0:
            errors.append(f"Warning: R:R ratio ({rr_ratio:.2f}:1) < 1:1")
        elif rr_ratio < 1.5:
            errors.append(f"Info: R:R ratio {rr_ratio:.2f}:1 acceptable but consider 2:1+")

    # VPIN bounds
    vpin_threshold = config.get('vpin_high_threshold', 0.7)
    if vpin_threshold < 0.5 or vpin_threshold > 0.9:
        errors.append(f"vpin_high_threshold should be 0.5-0.9, got {vpin_threshold}")

    # REC-002: Validate session boundaries
    session_bounds = config.get('session_boundaries', {})
    for key in ['asia_start', 'asia_end', 'europe_start', 'europe_end',
                'overlap_start', 'overlap_end', 'us_start', 'us_end']:
        if key in session_bounds:
            val = session_bounds[key]
            if not isinstance(val, (int, float)) or val < 0 or val > 24:
                errors.append(f"session_boundaries.{key} must be 0-24, got {val}")

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


def _validate_config_overrides(config: Dict[str, Any], overrides: Dict[str, Any]) -> List[str]:
    """
    REC-002: Validate configuration overrides match expected types.

    Returns:
        List of error messages for invalid overrides
    """
    errors = []

    # Type expectations for key parameters
    type_checks = {
        'position_size_usd': (int, float),
        'max_position_usd': (int, float),
        'stop_loss_pct': (int, float),
        'take_profit_pct': (int, float),
        'lookback_trades': (int,),
        'cooldown_trades': (int,),
        'cooldown_seconds': (int, float),
        'imbalance_threshold': (int, float),
        'buy_imbalance_threshold': (int, float),
        'sell_imbalance_threshold': (int, float),
        'volume_spike_mult': (int, float),
        'fee_rate': (int, float),
        'vpin_high_threshold': (int, float),
        'vpin_bucket_count': (int,),
        'use_vpin': (bool,),
        'use_volatility_regimes': (bool,),
        'use_session_awareness': (bool,),
        'use_correlation_management': (bool,),
        'use_trailing_stop': (bool,),
        'use_circuit_breaker': (bool,),
        'use_fee_check': (bool,),
        'use_trade_flow_confirmation': (bool,),
        'use_position_decay': (bool,),
        'use_asymmetric_thresholds': (bool,),
    }

    for key, value in overrides.items():
        if key in type_checks:
            expected_types = type_checks[key]
            if not isinstance(value, expected_types):
                type_names = '/'.join(t.__name__ for t in expected_types)
                errors.append(f"Override {key}: expected {type_names}, got {type(value).__name__}")

    return errors


# =============================================================================
# Section 2: Indicator Calculations
# =============================================================================
def _calculate_volatility(candles, lookback: int = 20) -> float:
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


def _calculate_micro_price(ob: OrderbookSnapshot) -> float:
    """
    Calculate volume-weighted micro-price.

    Micro-price = (best_bid * ask_size + best_ask * bid_size) / (bid_size + ask_size)
    """
    if not ob or not ob.bids or not ob.asks:
        return 0.0

    best_bid, bid_size = ob.bids[0]
    best_ask, ask_size = ob.asks[0]

    total_size = bid_size + ask_size
    if total_size <= 0:
        return ob.mid if ob else 0.0

    return (best_bid * ask_size + best_ask * bid_size) / total_size


def _calculate_vpin(trades: Tuple, bucket_count: int = 50) -> float:
    """
    Calculate VPIN (Volume-Synchronized Probability of Informed Trading).

    Finding #1 Fix: Improved bucket overflow logic that properly distributes
    volume across bucket boundaries based on cumulative volume, not last trade side.

    Formula:
    - Divide trades into equal-volume buckets
    - For each bucket, calculate |buy_volume - sell_volume| / total_volume
    - VPIN = average of bucket imbalances

    Args:
        trades: Tuple of recent trades
        bucket_count: Number of volume buckets to use

    Returns:
        VPIN value between 0 and 1
    """
    if len(trades) < bucket_count:
        return 0.0

    # Calculate total volume and bucket size
    total_volume = sum(t.size for t in trades)
    if total_volume <= 0:
        return 0.0

    bucket_volume = total_volume / bucket_count

    # Build volume buckets with improved overflow handling
    buckets = []
    current_bucket_buy = 0.0
    current_bucket_sell = 0.0
    cumulative_volume = 0.0
    bucket_boundary = bucket_volume

    for trade in trades:
        trade_volume = trade.size
        trade_buy = trade_volume if trade.side == 'buy' else 0.0
        trade_sell = trade_volume if trade.side == 'sell' else 0.0

        # Handle trade that may span multiple buckets
        remaining_buy = trade_buy
        remaining_sell = trade_sell
        remaining_volume = trade_volume

        while remaining_volume > 0 and len(buckets) < bucket_count:
            # How much volume fits in current bucket
            space_in_bucket = bucket_boundary - cumulative_volume
            volume_for_bucket = min(remaining_volume, space_in_bucket)

            if remaining_volume > 0:
                # Proportionally split buy/sell based on trade composition
                proportion = volume_for_bucket / remaining_volume
                buy_portion = remaining_buy * proportion
                sell_portion = remaining_sell * proportion

                current_bucket_buy += buy_portion
                current_bucket_sell += sell_portion
                cumulative_volume += volume_for_bucket

                remaining_buy -= buy_portion
                remaining_sell -= sell_portion
                remaining_volume -= volume_for_bucket

            # Check if bucket is complete
            if cumulative_volume >= bucket_boundary - 1e-10:  # Small tolerance for float comparison
                bucket_total = current_bucket_buy + current_bucket_sell
                if bucket_total > 0:
                    bucket_imbalance = abs(current_bucket_buy - current_bucket_sell) / bucket_total
                    buckets.append(bucket_imbalance)

                # Reset for next bucket
                current_bucket_buy = 0.0
                current_bucket_sell = 0.0
                bucket_boundary += bucket_volume

    # Handle final partial bucket if it has meaningful volume
    if current_bucket_buy + current_bucket_sell > bucket_volume * 0.5:
        bucket_total = current_bucket_buy + current_bucket_sell
        if bucket_total > 0:
            bucket_imbalance = abs(current_bucket_buy - current_bucket_sell) / bucket_total
            buckets.append(bucket_imbalance)

    if not buckets:
        return 0.0

    # VPIN is the average bucket imbalance
    return sum(buckets) / len(buckets)


# =============================================================================
# Section 3: Regime and Session Classification
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
) -> Dict[str, float]:
    """Get threshold and size adjustments based on volatility regime."""
    adjustments = {
        'threshold_mult': 1.0,
        'size_mult': 1.0,
        'pause_trading': False,
    }

    if regime == VolatilityRegime.LOW:
        adjustments['threshold_mult'] = 0.9
        adjustments['size_mult'] = 1.0
    elif regime == VolatilityRegime.MEDIUM:
        adjustments['threshold_mult'] = 1.0
        adjustments['size_mult'] = 1.0
    elif regime == VolatilityRegime.HIGH:
        adjustments['threshold_mult'] = 1.3
        adjustments['size_mult'] = 0.8
    elif regime == VolatilityRegime.EXTREME:
        adjustments['threshold_mult'] = config.get('volatility_threshold_mult', 1.5)
        adjustments['size_mult'] = config.get('regime_extreme_reduce_size', 0.5)
        adjustments['pause_trading'] = config.get('regime_extreme_pause', False)

    return adjustments


def _classify_trading_session(
    timestamp: datetime,
    config: Dict[str, Any]
) -> TradingSession:
    """
    REC-003: Classify current time into a trading session using configurable boundaries.

    Session boundaries are configurable in CONFIG['session_boundaries'] to allow
    adjustment for daylight saving time changes.

    Args:
        timestamp: Current timestamp (assumed UTC)
        config: Strategy configuration with session_boundaries

    Returns:
        TradingSession enum value
    """
    hour = timestamp.hour

    # Get configurable boundaries with defaults
    bounds = config.get('session_boundaries', {})
    overlap_start = bounds.get('overlap_start', 14)
    overlap_end = bounds.get('overlap_end', 17)
    europe_start = bounds.get('europe_start', 8)
    europe_end = bounds.get('europe_end', 14)
    us_start = bounds.get('us_start', 17)
    us_end = bounds.get('us_end', 21)

    if overlap_start <= hour < overlap_end:
        return TradingSession.US_EUROPE_OVERLAP
    elif europe_start <= hour < europe_end:
        return TradingSession.EUROPE
    elif us_start <= hour < us_end:
        return TradingSession.US
    else:
        return TradingSession.ASIA


def _get_session_adjustments(
    session: TradingSession,
    config: Dict[str, Any]
) -> Dict[str, float]:
    """Get threshold and size adjustments based on trading session."""
    threshold_mults = config.get('session_threshold_multipliers', {})
    size_mults = config.get('session_size_multipliers', {})

    session_name = session.name

    return {
        'threshold_mult': threshold_mults.get(session_name, 1.0),
        'size_mult': size_mults.get(session_name, 1.0),
    }


# =============================================================================
# Section 4: Risk Management
# =============================================================================
def _check_fee_profitability(
    expected_move_pct: float,
    fee_rate: float,
    min_profit_pct: float
) -> Tuple[bool, float]:
    """Check if trade is profitable after fees."""
    round_trip_fee_pct = fee_rate * 2 * 100  # Convert to percentage
    net_profit_pct = expected_move_pct - round_trip_fee_pct
    return net_profit_pct >= min_profit_pct, net_profit_pct


def _calculate_trailing_stop(
    entry_price: float,
    highest_price: float,
    side: str,
    activation_pct: float,
    trail_distance_pct: float
) -> Optional[float]:
    """Calculate trailing stop level."""
    if side == 'long':
        profit_pct = (highest_price - entry_price) / entry_price * 100
        if profit_pct >= activation_pct:
            return highest_price * (1 - trail_distance_pct / 100)
    elif side == 'short':
        profit_pct = (entry_price - highest_price) / entry_price * 100
        if profit_pct >= activation_pct:
            return highest_price * (1 + trail_distance_pct / 100)
    return None


def _get_progressive_decay_multiplier(
    position_age_seconds: float,
    decay_stages: List[Tuple[int, float]]
) -> float:
    """
    Get progressive TP multiplier based on position age.

    Returns:
        TP multiplier (1.0 if no decay, lower values for older positions)
    """
    sorted_stages = sorted(decay_stages, key=lambda x: x[0])

    multiplier = 1.0
    for age_threshold, tp_mult in sorted_stages:
        if position_age_seconds >= age_threshold:
            multiplier = tp_mult
        else:
            break

    return multiplier


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


def _check_correlation_exposure(
    state: Dict[str, Any],
    symbol: str,
    direction: str,
    size: float,
    config: Dict[str, Any]
) -> Tuple[bool, float]:
    """Check and adjust for cross-pair correlation."""
    if not config.get('use_correlation_management', True):
        return True, size

    max_long = config.get('max_total_long_exposure', 150.0)
    max_short = config.get('max_total_short_exposure', 150.0)
    same_dir_mult = config.get('same_direction_size_mult', 0.75)

    position_by_symbol = state.get('position_by_symbol', {})
    position_entries = state.get('position_entries', {})

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
            if direction in ('sell', 'short'):
                other_symbols_same_direction = True

    adjusted_size = size

    if other_symbols_same_direction:
        adjusted_size = size * same_dir_mult

    if direction == 'buy':
        if total_long + adjusted_size > max_long:
            available = max(0, max_long - total_long)
            adjusted_size = min(adjusted_size, available)
    elif direction in ('sell', 'short'):
        if total_short + adjusted_size > max_short:
            available = max(0, max_short - total_short)
            adjusted_size = min(adjusted_size, available)

    can_trade = adjusted_size >= config.get('min_trade_size_usd', 5.0)

    return can_trade, adjusted_size


# =============================================================================
# Section 5: Signal Rejection Tracking (REC-001)
# =============================================================================
def _track_rejection(
    state: Dict[str, Any],
    reason: RejectionReason,
    symbol: str = None
) -> None:
    """
    REC-001: Track signal rejection for analysis.

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


def _build_base_indicators(
    symbol: str,
    trade_count: int,
    status: str,
    state: Dict[str, Any]
) -> Dict[str, Any]:
    """Build base indicators dict for early returns."""
    return {
        'symbol': symbol,
        'trade_count': trade_count,
        'status': status,
        'position_side': state.get('position_side'),
        'position_size': state.get('position_size', 0),
        'consecutive_losses': state.get('consecutive_losses', 0),
        'pnl_symbol': state.get('pnl_by_symbol', {}).get(symbol, 0),
        'trades_symbol': state.get('trades_by_symbol', {}).get(symbol, 0),
    }


# =============================================================================
# Section 6: Exit Signal Checks
# =============================================================================
def _check_trailing_stop_exit(
    data: DataSnapshot,
    config: Dict[str, Any],
    state: Dict[str, Any],
    symbol: str,
    current_price: float,
    ob: Optional[OrderbookSnapshot]
) -> Optional[Signal]:
    """Check if trailing stop should trigger exit."""
    if not config.get('use_trailing_stop', False):
        return None

    pos_entry = state.get('position_entries', {}).get(symbol)
    if not pos_entry:
        return None

    trailing_activation = config.get('trailing_stop_activation', 0.3)
    trailing_distance = config.get('trailing_stop_distance', 0.2)

    if pos_entry['side'] == 'long':
        pos_entry['highest_price'] = max(pos_entry.get('highest_price', current_price), current_price)
        tracking_price = pos_entry['highest_price']
    else:
        pos_entry['lowest_price'] = min(pos_entry.get('lowest_price', current_price), current_price)
        tracking_price = pos_entry['lowest_price']

    trailing_stop_price = _calculate_trailing_stop(
        entry_price=pos_entry['entry_price'],
        highest_price=tracking_price,
        side=pos_entry['side'],
        activation_pct=trailing_activation,
        trail_distance_pct=trailing_distance
    )

    if trailing_stop_price is None:
        return None

    exit_price = current_price
    if ob:
        exit_price = ob.best_bid if pos_entry['side'] == 'long' else ob.best_ask

    if pos_entry['side'] == 'long' and current_price <= trailing_stop_price:
        close_size = state.get('position_size', 0)
        if close_size > 0:
            return Signal(
                action='sell',
                symbol=symbol,
                size=close_size,
                price=exit_price,
                reason=f"OF: Trailing stop (entry={pos_entry['entry_price']:.6f}, high={pos_entry['highest_price']:.6f})",
                metadata={'trailing_stop': True},
            )

    elif pos_entry['side'] == 'short' and current_price >= trailing_stop_price:
        close_size = state.get('position_size', 0)
        if close_size > 0:
            return Signal(
                action='cover',
                symbol=symbol,
                size=close_size,
                price=exit_price,
                reason=f"OF: Trailing stop (entry={pos_entry['entry_price']:.6f}, low={pos_entry['lowest_price']:.6f})",
                metadata={'trailing_stop': True},
            )

    return None


def _check_position_decay_exit(
    data: DataSnapshot,
    config: Dict[str, Any],
    state: Dict[str, Any],
    symbol: str,
    current_price: float,
    ob: Optional[OrderbookSnapshot],
    current_time: datetime
) -> Optional[Signal]:
    """
    REC-004: Check if stale position should be closed with progressive TP.

    Enhanced to allow closing at any profit > fees during intermediate decay stages.
    """
    if not config.get('use_position_decay', True):
        return None

    pos_entry = state.get('position_entries', {}).get(symbol)
    if not pos_entry:
        return None

    entry_time = pos_entry.get('entry_time')
    if not entry_time:
        return None

    age_seconds = (current_time - entry_time).total_seconds()

    decay_stages = config.get('position_decay_stages', [
        (180, 0.90), (240, 0.75), (300, 0.50), (360, 0.0)
    ])
    tp_mult = _get_progressive_decay_multiplier(age_seconds, decay_stages)

    # If multiplier is 1.0, no decay yet
    if tp_mult >= 1.0:
        return None

    entry_price = pos_entry['entry_price']
    tp_pct = _get_symbol_config(symbol, config, 'take_profit_pct')
    adjusted_tp_pct = tp_pct * tp_mult

    exit_price = current_price
    if ob:
        exit_price = ob.best_bid if pos_entry['side'] == 'long' else ob.best_ask

    # REC-004: Enhanced close-at-profit-after-fees for intermediate stages
    use_early_close = config.get('decay_close_at_profit_after_fees', True)
    fee_rate = config.get('fee_rate', 0.001)
    min_profit_after_fees = config.get('decay_min_profit_after_fees_pct', 0.05)
    round_trip_fee_pct = fee_rate * 2 * 100

    if pos_entry['side'] == 'long':
        profit_pct = (current_price - entry_price) / entry_price * 100
        net_profit_pct = profit_pct - round_trip_fee_pct

        # For tp_mult=0, exit at any profit
        if tp_mult == 0 and profit_pct > 0:
            close_size = state.get('position_size', 0)
            if close_size > 0:
                return Signal(
                    action='sell',
                    symbol=symbol,
                    size=close_size,
                    price=exit_price,
                    reason=f"OF: Decay exit (age={age_seconds:.0f}s, profit={profit_pct:.2f}%)",
                    metadata={'position_decay': True, 'decay_stage': 'any_profit'},
                )

        # REC-004: Close at profit after fees during intermediate stages
        elif use_early_close and tp_mult < 1.0 and net_profit_pct >= min_profit_after_fees:
            close_size = state.get('position_size', 0)
            if close_size > 0:
                return Signal(
                    action='sell',
                    symbol=symbol,
                    size=close_size,
                    price=exit_price,
                    reason=f"OF: Decay early exit (age={age_seconds:.0f}s, net_profit={net_profit_pct:.2f}%)",
                    metadata={'position_decay': True, 'decay_stage': 'profit_after_fees', 'decay_mult': tp_mult},
                )

        # Standard decay exit at adjusted TP
        elif profit_pct >= adjusted_tp_pct > 0:
            close_size = state.get('position_size', 0)
            if close_size > 0:
                return Signal(
                    action='sell',
                    symbol=symbol,
                    size=close_size,
                    price=exit_price,
                    reason=f"OF: Decay exit (age={age_seconds:.0f}s, profit={profit_pct:.2f}%, target={adjusted_tp_pct:.2f}%)",
                    metadata={'position_decay': True, 'decay_mult': tp_mult},
                )

    elif pos_entry['side'] == 'short':
        profit_pct = (entry_price - current_price) / entry_price * 100
        net_profit_pct = profit_pct - round_trip_fee_pct

        if tp_mult == 0 and profit_pct > 0:
            close_size = state.get('position_size', 0)
            if close_size > 0:
                return Signal(
                    action='cover',
                    symbol=symbol,
                    size=close_size,
                    price=exit_price,
                    reason=f"OF: Decay exit (age={age_seconds:.0f}s, profit={profit_pct:.2f}%)",
                    metadata={'position_decay': True, 'decay_stage': 'any_profit'},
                )

        elif use_early_close and tp_mult < 1.0 and net_profit_pct >= min_profit_after_fees:
            close_size = state.get('position_size', 0)
            if close_size > 0:
                return Signal(
                    action='cover',
                    symbol=symbol,
                    size=close_size,
                    price=exit_price,
                    reason=f"OF: Decay early exit (age={age_seconds:.0f}s, net_profit={net_profit_pct:.2f}%)",
                    metadata={'position_decay': True, 'decay_stage': 'profit_after_fees', 'decay_mult': tp_mult},
                )

        elif profit_pct >= adjusted_tp_pct > 0:
            close_size = state.get('position_size', 0)
            if close_size > 0:
                return Signal(
                    action='cover',
                    symbol=symbol,
                    size=close_size,
                    price=exit_price,
                    reason=f"OF: Decay exit (age={age_seconds:.0f}s, profit={profit_pct:.2f}%, target={adjusted_tp_pct:.2f}%)",
                    metadata={'position_decay': True, 'decay_mult': tp_mult},
                )

    return None


# =============================================================================
# REQUIRED: Main Signal Generation Function
# =============================================================================
def generate_signal(
    data: DataSnapshot,
    config: Dict[str, Any],
    state: Dict[str, Any]
) -> Optional[Signal]:
    """
    Generate order flow signal based on trade tape analysis.

    Args:
        data: Immutable market data snapshot
        config: Strategy configuration (CONFIG + overrides)
        state: Mutable strategy state dict

    Returns:
        Signal if trading opportunity found, None otherwise
    """
    if 'initialized' not in state:
        _initialize_state(state)

    current_time = data.timestamp
    track_rejections = config.get('track_rejections', True)

    # Circuit breaker check
    if config.get('use_circuit_breaker', True):
        max_losses = config.get('max_consecutive_losses', 3)
        cooldown_min = config.get('circuit_breaker_minutes', 15)

        if _check_circuit_breaker(state, current_time, max_losses, cooldown_min):
            state['indicators'] = _build_base_indicators(
                symbol='N/A', trade_count=0, status='circuit_breaker', state=state
            )
            state['indicators']['circuit_breaker_active'] = True
            if track_rejections:
                _track_rejection(state, RejectionReason.CIRCUIT_BREAKER)
            return None

    # Time-based cooldown check
    if state['last_signal_time'] is not None:
        elapsed = (current_time - state['last_signal_time']).total_seconds()
        if elapsed < config.get('cooldown_seconds', 5.0):
            state['indicators'] = _build_base_indicators(
                symbol='N/A', trade_count=0, status='cooldown', state=state
            )
            state['indicators']['cooldown_remaining'] = config.get('cooldown_seconds', 5.0) - elapsed
            if track_rejections:
                _track_rejection(state, RejectionReason.TIME_COOLDOWN)
            return None

    # Evaluate each symbol
    for symbol in SYMBOLS:
        signal = _evaluate_symbol(data, config, state, symbol, current_time)
        if signal:
            return signal

    return None


def _initialize_state(state: Dict[str, Any]) -> None:
    """Initialize strategy state."""
    state['initialized'] = True
    state['last_signal_idx'] = 0
    state['total_trades_seen'] = 0
    state['last_signal_time'] = None
    state['position_side'] = None
    state['position_size'] = 0.0
    state['position_by_symbol'] = {}
    state['fills'] = []
    state['indicators'] = {}
    state['pnl_by_symbol'] = {}
    state['trades_by_symbol'] = {}
    state['wins_by_symbol'] = {}
    state['losses_by_symbol'] = {}
    state['position_entries'] = {}
    state['consecutive_losses'] = 0
    state['circuit_breaker_time'] = None
    # REC-001: Rejection tracking
    state['rejection_counts'] = {}
    state['rejection_counts_by_symbol'] = {}


def _evaluate_symbol(
    data: DataSnapshot,
    config: Dict[str, Any],
    state: Dict[str, Any],
    symbol: str,
    current_time: datetime
) -> Optional[Signal]:
    """Evaluate order flow for a specific symbol."""
    trades = data.trades.get(symbol, ())
    base_lookback = config.get('lookback_trades', 50)
    track_rejections = config.get('track_rejections', True)

    # Not enough trades yet
    if len(trades) < base_lookback:
        state['indicators'] = _build_base_indicators(
            symbol=symbol, trade_count=len(trades), status='warming_up', state=state
        )
        state['indicators']['required_trades'] = base_lookback
        if track_rejections:
            _track_rejection(state, RejectionReason.WARMING_UP, symbol)
        return None

    # Calculate volatility for regime classification
    candles = data.candles_1m.get(symbol, ())
    volatility = _calculate_volatility(candles, config.get('volatility_lookback', 20))

    # Classify volatility regime
    regime = VolatilityRegime.MEDIUM
    regime_adjustments = {'threshold_mult': 1.0, 'size_mult': 1.0, 'pause_trading': False}

    if config.get('use_volatility_regimes', True):
        regime = _classify_volatility_regime(volatility, config)
        regime_adjustments = _get_regime_adjustments(regime, config)

        if regime_adjustments['pause_trading']:
            state['indicators'] = _build_base_indicators(
                symbol=symbol, trade_count=len(trades), status='regime_pause', state=state
            )
            state['indicators']['volatility_regime'] = regime.name
            state['indicators']['volatility_pct'] = round(volatility, 4)
            if track_rejections:
                _track_rejection(state, RejectionReason.REGIME_PAUSE, symbol)
            return None

    # REC-003: Get session adjustments with configurable boundaries
    session = TradingSession.EUROPE
    session_adjustments = {'threshold_mult': 1.0, 'size_mult': 1.0}

    if config.get('use_session_awareness', True):
        session = _classify_trading_session(current_time, config)
        session_adjustments = _get_session_adjustments(session, config)

    # Adjust lookback based on regime
    lookback = base_lookback
    if regime == VolatilityRegime.HIGH:
        lookback = int(base_lookback * 0.75)
    elif regime == VolatilityRegime.EXTREME:
        lookback = int(base_lookback * 0.5)
    elif regime == VolatilityRegime.LOW:
        lookback = int(base_lookback * 1.25)

    lookback = max(20, min(lookback, len(trades)))

    recent_trades = trades[-lookback:]

    # Trade-based cooldown check
    cooldown_trades = _get_symbol_config(symbol, config, 'cooldown_trades') or config.get('cooldown_trades', 10)
    state['total_trades_seen'] = len(trades)
    trades_since_signal = state['total_trades_seen'] - state['last_signal_idx']

    if trades_since_signal < cooldown_trades:
        state['indicators'] = _build_base_indicators(
            symbol=symbol, trade_count=len(trades), status='trade_cooldown', state=state
        )
        state['indicators']['trades_since_signal'] = trades_since_signal
        if track_rejections:
            _track_rejection(state, RejectionReason.TRADE_COOLDOWN, symbol)
        return None

    # Calculate VPIN
    vpin_value = 0.0
    vpin_pause = False
    if config.get('use_vpin', True):
        vpin_lookback = config.get('vpin_lookback_trades', 200)
        vpin_trades = trades[-vpin_lookback:] if len(trades) >= vpin_lookback else trades
        vpin_value = _calculate_vpin(vpin_trades, config.get('vpin_bucket_count', 50))

        if vpin_value >= config.get('vpin_high_threshold', 0.7):
            if config.get('vpin_pause_on_high', True):
                vpin_pause = True

    if vpin_pause:
        state['indicators'] = _build_base_indicators(
            symbol=symbol, trade_count=len(trades), status='vpin_pause', state=state
        )
        state['indicators']['vpin'] = round(vpin_value, 4)
        state['indicators']['vpin_threshold'] = config.get('vpin_high_threshold', 0.7)
        if track_rejections:
            _track_rejection(state, RejectionReason.VPIN_PAUSE, symbol)
        return None

    # Calculate buy/sell imbalance
    buy_volume = sum(t.size for t in recent_trades if t.side == 'buy')
    sell_volume = sum(t.size for t in recent_trades if t.side == 'sell')
    total_volume = buy_volume + sell_volume

    if total_volume == 0:
        state['indicators'] = _build_base_indicators(
            symbol=symbol, trade_count=len(trades), status='no_volume', state=state
        )
        if track_rejections:
            _track_rejection(state, RejectionReason.NO_VOLUME, symbol)
        return None

    imbalance = (buy_volume - sell_volume) / total_volume

    # Calculate volume spike
    avg_trade_size = total_volume / len(recent_trades)
    last_5_trades = recent_trades[-5:]
    last_5_volume = sum(t.size for t in last_5_trades)
    expected_5_volume = avg_trade_size * 5
    volume_spike = last_5_volume / expected_5_volume if expected_5_volume > 0 else 1.0

    # VWAP and price
    vwap = data.get_vwap(symbol, lookback)
    current_price = data.prices.get(symbol, 0)

    if not current_price or not vwap:
        state['indicators'] = _build_base_indicators(
            symbol=symbol, trade_count=len(trades), status='no_price_or_vwap', state=state
        )
        if track_rejections:
            _track_rejection(state, RejectionReason.NO_PRICE_DATA, symbol)
        return None

    price_vs_vwap = (current_price - vwap) / vwap

    # Micro-price
    ob = data.orderbooks.get(symbol)
    micro_price = current_price
    if config.get('use_micro_price', True) and ob:
        micro_price = _calculate_micro_price(ob)

    # Get symbol-specific config
    volume_spike_mult = _get_symbol_config(symbol, config, 'volume_spike_mult') or config.get('volume_spike_mult', 2.0)
    base_position_size = _get_symbol_config(symbol, config, 'position_size_usd')
    tp_pct = _get_symbol_config(symbol, config, 'take_profit_pct')
    sl_pct = _get_symbol_config(symbol, config, 'stop_loss_pct')

    # Asymmetric thresholds
    use_asymmetric = config.get('use_asymmetric_thresholds', True)
    if use_asymmetric:
        base_buy_threshold = _get_symbol_config(symbol, config, 'buy_imbalance_threshold') or 0.30
        base_sell_threshold = _get_symbol_config(symbol, config, 'sell_imbalance_threshold') or 0.25
    else:
        base_buy_threshold = _get_symbol_config(symbol, config, 'imbalance_threshold') or 0.30
        base_sell_threshold = base_buy_threshold

    # Apply regime and session adjustments
    combined_threshold_mult = regime_adjustments['threshold_mult'] * session_adjustments['threshold_mult']
    effective_buy_threshold = base_buy_threshold * combined_threshold_mult
    effective_sell_threshold = base_sell_threshold * combined_threshold_mult

    combined_size_mult = regime_adjustments['size_mult'] * session_adjustments['size_mult']
    adjusted_position_size = base_position_size * combined_size_mult

    # Trade flow confirmation
    use_trade_flow = config.get('use_trade_flow_confirmation', True)
    trade_flow_threshold = config.get('trade_flow_threshold', 0.15)
    trade_flow = data.get_trade_imbalance(symbol, lookback)

    # Fee profitability check
    use_fee_check = config.get('use_fee_check', True)
    fee_rate = config.get('fee_rate', 0.001)
    min_profit_pct = config.get('min_profit_after_fees_pct', 0.05)

    is_fee_profitable = True
    expected_profit = tp_pct
    if use_fee_check:
        is_fee_profitable, expected_profit = _check_fee_profitability(tp_pct, fee_rate, min_profit_pct)

    # Position limits
    current_position = state.get('position_size', 0)
    max_position = config.get('max_position_usd', 100.0)
    min_trade = config.get('min_trade_size_usd', 5.0)

    # Check trailing stop exit
    trailing_signal = _check_trailing_stop_exit(data, config, state, symbol, current_price, ob)
    if trailing_signal:
        return trailing_signal

    # Check position decay exit
    decay_signal = _check_position_decay_exit(data, config, state, symbol, current_price, ob, current_time)
    if decay_signal:
        return decay_signal

    # Get trailing stop price for logging
    trailing_stop_price = None
    if config.get('use_trailing_stop', False):
        pos_entry = state.get('position_entries', {}).get(symbol)
        if pos_entry:
            tracking_price = pos_entry.get(
                'highest_price' if pos_entry['side'] == 'long' else 'lowest_price',
                current_price
            )
            trailing_stop_price = _calculate_trailing_stop(
                entry_price=pos_entry['entry_price'],
                highest_price=tracking_price,
                side=pos_entry['side'],
                activation_pct=config.get('trailing_stop_activation', 0.3),
                trail_distance_pct=config.get('trailing_stop_distance', 0.2)
            )

    # Build comprehensive indicators
    state['indicators'] = {
        'symbol': symbol,
        'status': 'active',
        'trade_count': len(trades),
        'imbalance': round(imbalance, 4),
        'buy_volume': round(buy_volume, 2),
        'sell_volume': round(sell_volume, 2),
        'volume_spike': round(volume_spike, 2),
        'volume_spike_threshold': round(volume_spike_mult, 2),
        'vwap': round(vwap, 6),
        'price': round(current_price, 6),
        'micro_price': round(micro_price, 6),
        'price_vs_vwap': round(price_vs_vwap, 6),
        'volatility_pct': round(volatility, 4),
        'volatility_regime': regime.name,
        'regime_threshold_mult': round(regime_adjustments['threshold_mult'], 2),
        'regime_size_mult': round(regime_adjustments['size_mult'], 2),
        'trading_session': session.name,
        'session_threshold_mult': round(session_adjustments['threshold_mult'], 2),
        'session_size_mult': round(session_adjustments['size_mult'], 2),
        'combined_threshold_mult': round(combined_threshold_mult, 2),
        'combined_size_mult': round(combined_size_mult, 2),
        'base_buy_threshold': round(base_buy_threshold, 4),
        'base_sell_threshold': round(base_sell_threshold, 4),
        'effective_buy_threshold': round(effective_buy_threshold, 4),
        'effective_sell_threshold': round(effective_sell_threshold, 4),
        'adjusted_lookback': lookback,
        'vpin': round(vpin_value, 4),
        'vpin_threshold': config.get('vpin_high_threshold', 0.7),
        'trade_flow': round(trade_flow, 4),
        'trade_flow_threshold': round(trade_flow_threshold, 4),
        'use_trade_flow': use_trade_flow,
        'is_fee_profitable': is_fee_profitable,
        'expected_profit_pct': round(expected_profit, 4),
        'position_side': state.get('position_side'),
        'position_size': round(current_position, 2),
        'max_position': max_position,
        'adjusted_position_size': round(adjusted_position_size, 2),
        'trailing_stop_price': round(trailing_stop_price, 6) if trailing_stop_price else None,
        'pnl_symbol': round(state.get('pnl_by_symbol', {}).get(symbol, 0), 4),
        'trades_symbol': state.get('trades_by_symbol', {}).get(symbol, 0),
        'consecutive_losses': state.get('consecutive_losses', 0),
    }

    if current_position >= max_position:
        state['indicators']['status'] = 'max_position_reached'
        if track_rejections:
            _track_rejection(state, RejectionReason.MAX_POSITION, symbol)
        return None

    available = max_position - current_position
    actual_size = min(adjusted_position_size, available)
    if actual_size < min_trade:
        state['indicators']['status'] = 'insufficient_size'
        if track_rejections:
            _track_rejection(state, RejectionReason.INSUFFICIENT_SIZE, symbol)
        return None

    if use_fee_check and not is_fee_profitable:
        state['indicators']['status'] = 'not_fee_profitable'
        if track_rejections:
            _track_rejection(state, RejectionReason.NOT_FEE_PROFITABLE, symbol)
        return None

    signal = None

    # Strong buy pressure with volume spike
    if imbalance > effective_buy_threshold and volume_spike > volume_spike_mult:
        if use_trade_flow and not _is_trade_flow_aligned(data, symbol, 'buy', trade_flow_threshold, lookback):
            state['indicators']['status'] = 'trade_flow_not_aligned'
            state['indicators']['trade_flow_aligned'] = False
            if track_rejections:
                _track_rejection(state, RejectionReason.TRADE_FLOW_NOT_ALIGNED, symbol)
            return None

        state['indicators']['trade_flow_aligned'] = True

        can_trade, adjusted_size = _check_correlation_exposure(state, symbol, 'buy', actual_size, config)
        if not can_trade:
            state['indicators']['status'] = 'correlation_limit'
            if track_rejections:
                _track_rejection(state, RejectionReason.CORRELATION_LIMIT, symbol)
            return None

        signal = Signal(
            action='buy',
            symbol=symbol,
            size=adjusted_size,
            price=current_price,
            reason=f"OF: Buy (imbal={imbalance:.2f}, vol={volume_spike:.1f}x, regime={regime.name}, session={session.name})",
            stop_loss=current_price * (1 - sl_pct / 100),
            take_profit=current_price * (1 + tp_pct / 100),
        )

    # Strong sell pressure with volume spike
    elif imbalance < -effective_sell_threshold and volume_spike > volume_spike_mult:
        if use_trade_flow and not _is_trade_flow_aligned(data, symbol, 'sell', trade_flow_threshold, lookback):
            state['indicators']['status'] = 'trade_flow_not_aligned'
            state['indicators']['trade_flow_aligned'] = False
            if track_rejections:
                _track_rejection(state, RejectionReason.TRADE_FLOW_NOT_ALIGNED, symbol)
            return None

        state['indicators']['trade_flow_aligned'] = True

        has_long = state.get('position_side') == 'long' and state.get('position_size', 0) > 0

        if has_long:
            close_size = min(actual_size, state.get('position_size', actual_size))
            signal = Signal(
                action='sell',
                symbol=symbol,
                size=close_size,
                price=current_price,
                reason=f"OF: Close long (imbal={imbalance:.2f}, vol={volume_spike:.1f}x)",
                stop_loss=current_price * (1 - sl_pct / 100),
                take_profit=current_price * (1 + tp_pct / 100),
            )
        else:
            can_trade, adjusted_size = _check_correlation_exposure(state, symbol, 'short', actual_size, config)
            if not can_trade:
                state['indicators']['status'] = 'correlation_limit'
                if track_rejections:
                    _track_rejection(state, RejectionReason.CORRELATION_LIMIT, symbol)
                return None

            signal = Signal(
                action='short',
                symbol=symbol,
                size=adjusted_size,
                price=current_price,
                reason=f"OF: Short (imbal={imbalance:.2f}, vol={volume_spike:.1f}x, regime={regime.name}, session={session.name})",
                stop_loss=current_price * (1 + sl_pct / 100),
                take_profit=current_price * (1 - tp_pct / 100),
            )

    # VWAP mean reversion opportunities
    vwap_threshold_mult = config.get('vwap_reversion_threshold_mult', 0.7)
    vwap_size_mult = config.get('vwap_reversion_size_mult', 0.75)
    vwap_deviation = config.get('vwap_deviation_threshold', 0.001)

    if signal is None and (imbalance > effective_buy_threshold * vwap_threshold_mult and
          price_vs_vwap < -vwap_deviation):
        reduced_size = actual_size * vwap_size_mult

        can_trade, adjusted_size = _check_correlation_exposure(state, symbol, 'buy', reduced_size, config)

        if can_trade and adjusted_size >= min_trade:
            signal = Signal(
                action='buy',
                symbol=symbol,
                size=adjusted_size,
                price=current_price,
                reason=f"OF: Buy below VWAP (imbal={imbalance:.2f}, dev={price_vs_vwap:.4f})",
                stop_loss=current_price * (1 - sl_pct / 100),
                take_profit=vwap,
            )

    if signal is None and (imbalance < -effective_sell_threshold * vwap_threshold_mult and
          price_vs_vwap > vwap_deviation):
        has_long = state.get('position_side') == 'long' and state.get('position_size', 0) > 0
        reduced_size = actual_size * vwap_size_mult

        if has_long and reduced_size >= min_trade:
            close_size = min(reduced_size, state.get('position_size', reduced_size))
            signal = Signal(
                action='sell',
                symbol=symbol,
                size=close_size,
                price=current_price,
                reason=f"OF: Close long above VWAP (imbal={imbalance:.2f}, dev={price_vs_vwap:.4f})",
            )

    if signal:
        state['last_signal_idx'] = state['total_trades_seen']
        state['last_signal_time'] = current_time
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
    # REC-002: Validate configuration
    errors = _validate_config(config)
    if errors:
        for error in errors:
            print(f"[order_flow] Config warning: {error}")
        state['config_warnings'] = errors
    state['config_validated'] = True

    _initialize_state(state)

    print(f"[order_flow] v{STRATEGY_VERSION} started")
    print(f"[order_flow] Features: VPIN={config.get('use_vpin', True)}, "
          f"Regimes={config.get('use_volatility_regimes', True)}, "
          f"Sessions={config.get('use_session_awareness', True)}, "
          f"Correlation={config.get('use_correlation_management', True)}, "
          f"RejectionTracking={config.get('track_rejections', True)}")


def on_fill(fill: Dict[str, Any], state: Dict[str, Any]) -> None:
    """Track fills and update position state."""
    if 'fills' not in state:
        state['fills'] = []
    state['fills'].append(fill)
    state['fills'] = state['fills'][-50:]

    side = fill.get('side', '')
    value = fill.get('value', fill.get('size', 0) * fill.get('price', 0))
    symbol = fill.get('symbol', 'XRP/USDT')
    price = fill.get('price', 0)
    pnl = fill.get('pnl', 0)
    timestamp = fill.get('timestamp', datetime.now())

    for key in ['pnl_by_symbol', 'trades_by_symbol', 'wins_by_symbol', 'losses_by_symbol']:
        if key not in state:
            state[key] = {}

    if pnl != 0:
        state['pnl_by_symbol'][symbol] = state['pnl_by_symbol'].get(symbol, 0) + pnl

        if pnl > 0:
            state['wins_by_symbol'][symbol] = state['wins_by_symbol'].get(symbol, 0) + 1
            state['consecutive_losses'] = 0
        else:
            state['losses_by_symbol'][symbol] = state['losses_by_symbol'].get(symbol, 0) + 1
            state['consecutive_losses'] = state.get('consecutive_losses', 0) + 1

            max_losses = 3
            if state['consecutive_losses'] >= max_losses:
                state['circuit_breaker_time'] = timestamp

    state['trades_by_symbol'][symbol] = state['trades_by_symbol'].get(symbol, 0) + 1

    if 'position_entries' not in state:
        state['position_entries'] = {}
    if 'position_by_symbol' not in state:
        state['position_by_symbol'] = {}

    if side == 'buy':
        state['position_side'] = 'long'
        state['position_size'] = state.get('position_size', 0) + value
        state['position_by_symbol'][symbol] = state['position_by_symbol'].get(symbol, 0) + value

        if symbol not in state['position_entries']:
            state['position_entries'][symbol] = {
                'entry_price': price,
                'entry_time': timestamp,
                'highest_price': price,
                'lowest_price': price,
                'side': 'long',
            }
        else:
            pos = state['position_entries'][symbol]
            pos['highest_price'] = max(pos.get('highest_price', price), price)

    elif side == 'sell':
        state['position_size'] = max(0, state.get('position_size', 0) - value)
        state['position_by_symbol'][symbol] = max(0, state['position_by_symbol'].get(symbol, 0) - value)

        if state['position_size'] < 0.01:
            state['position_side'] = None
            state['position_size'] = 0.0

        if state['position_by_symbol'].get(symbol, 0) < 0.01:
            if symbol in state['position_entries']:
                del state['position_entries'][symbol]
            state['position_by_symbol'][symbol] = 0.0

    elif side == 'short':
        state['position_side'] = 'short'
        state['position_size'] = state.get('position_size', 0) + value
        state['position_by_symbol'][symbol] = state['position_by_symbol'].get(symbol, 0) + value

        if symbol not in state['position_entries']:
            state['position_entries'][symbol] = {
                'entry_price': price,
                'entry_time': timestamp,
                'highest_price': price,
                'lowest_price': price,
                'side': 'short',
            }
        else:
            pos = state['position_entries'][symbol]
            pos['lowest_price'] = min(pos.get('lowest_price', price), price)

    elif side == 'cover':
        state['position_size'] = max(0, state.get('position_size', 0) - value)
        state['position_by_symbol'][symbol] = max(0, state['position_by_symbol'].get(symbol, 0) - value)

        if state['position_size'] < 0.01:
            state['position_side'] = None
            state['position_size'] = 0.0

        if state['position_by_symbol'].get(symbol, 0) < 0.01:
            if symbol in state['position_entries']:
                del state['position_entries'][symbol]
            state['position_by_symbol'][symbol] = 0.0


def on_stop(state: Dict[str, Any]) -> None:
    """Called when strategy stops."""
    final_position = state.get('position_size', 0)
    final_side = state.get('position_side')
    total_fills = len(state.get('fills', []))

    total_pnl = sum(state.get('pnl_by_symbol', {}).values())
    total_trades = sum(state.get('trades_by_symbol', {}).values())
    total_wins = sum(state.get('wins_by_symbol', {}).values())
    total_losses = sum(state.get('losses_by_symbol', {}).values())

    win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0

    # REC-001: Include rejection statistics in summary
    rejection_counts = state.get('rejection_counts', {})
    total_rejections = sum(rejection_counts.values())

    state['indicators'] = {}

    state['final_summary'] = {
        'position_side': final_side,
        'position_size': final_position,
        'total_fills': total_fills,
        'pnl_by_symbol': state.get('pnl_by_symbol', {}),
        'trades_by_symbol': state.get('trades_by_symbol', {}),
        'wins_by_symbol': state.get('wins_by_symbol', {}),
        'losses_by_symbol': state.get('losses_by_symbol', {}),
        'total_pnl': total_pnl,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'config_warnings': state.get('config_warnings', []),
        # REC-001: Rejection statistics
        'rejection_counts': rejection_counts,
        'rejection_counts_by_symbol': state.get('rejection_counts_by_symbol', {}),
        'total_rejections': total_rejections,
    }

    print(f"[order_flow] Stopped. PnL: ${total_pnl:.2f}, Trades: {total_trades}, Win Rate: {win_rate:.1f}%")

    # REC-001: Print rejection summary
    if rejection_counts:
        print(f"[order_flow] Signal rejections ({total_rejections} total):")
        for reason, count in sorted(rejection_counts.items(), key=lambda x: -x[1])[:5]:
            print(f"[order_flow]   - {reason}: {count}")
