"""
Order Flow Strategy - Configuration and Enums

Contains strategy metadata, enums for type safety, and default configuration.
"""
from enum import Enum, auto
from typing import Dict, Any


# =============================================================================
# Strategy Metadata
# =============================================================================
STRATEGY_NAME = "order_flow"
STRATEGY_VERSION = "4.4.0"
# REC-003 (v4.4.0): Added XRP/BTC ratio pair
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


class TradingSession(Enum):
    """Trading session classification."""
    ASIA = auto()
    EUROPE = auto()
    US = auto()
    US_EUROPE_OVERLAP = auto()
    # REC-002 (v4.3.0): Add OFF_HOURS session for 21:00-24:00 UTC (post-US thin liquidity)
    OFF_HOURS = auto()


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
# Default Configuration
# =============================================================================
CONFIG: Dict[str, Any] = {
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
    # REC-006 (v4.2.0): Added max_position_per_symbol_usd for per-symbol limits
    # ==========================================================================
    'position_size_usd': 25.0,          # Size per trade in USD
    'max_position_usd': 100.0,          # Maximum TOTAL position exposure across all pairs
    'max_position_per_symbol_usd': 75.0,  # Maximum position exposure PER SYMBOL
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
    # REC-002 (v4.3.0): Added off_hours boundaries for 21:00-24:00 UTC
    'session_boundaries': {
        'asia_start': 0,                # 00:00 UTC
        'asia_end': 8,                  # 08:00 UTC
        'europe_start': 8,              # 08:00 UTC
        'europe_end': 14,               # 14:00 UTC
        'overlap_start': 14,            # 14:00 UTC (US/Europe overlap)
        'overlap_end': 17,              # 17:00 UTC
        'us_start': 17,                 # 17:00 UTC
        'us_end': 21,                   # 21:00 UTC
        'off_hours_start': 21,          # 21:00 UTC (post-US, pre-Asia)
        'off_hours_end': 24,            # 24:00 UTC (midnight)
    },
    'session_threshold_multipliers': {
        'ASIA': 1.2,                    # Wider thresholds (lower volume)
        'EUROPE': 1.0,                  # Standard thresholds
        'US': 1.0,                      # Standard thresholds
        'US_EUROPE_OVERLAP': 0.85,      # Tighter thresholds (peak liquidity)
        # REC-002 (v4.3.0): OFF_HOURS - more conservative (42% below peak liquidity)
        'OFF_HOURS': 1.35,              # Very wide thresholds (thinnest liquidity)
    },
    'session_size_multipliers': {
        'ASIA': 0.8,                    # Smaller sizes (lower liquidity)
        'EUROPE': 1.0,                  # Standard sizes
        'US': 1.0,                      # Standard sizes
        'US_EUROPE_OVERLAP': 1.1,       # Larger sizes (peak liquidity)
        # REC-002 (v4.3.0): OFF_HOURS - smaller sizes for thin liquidity
        'OFF_HOURS': 0.6,               # Smallest sizes (highest risk period)
    },

    # ==========================================================================
    # Progressive Position Decay (REC-004 v4.2.0: Enhanced with profit-after-fees)
    # REC-004 (v4.3.0): Extended decay start time to allow 5 complete 1-minute candles
    # ==========================================================================
    'use_position_decay': True,         # Enable time-based position decay
    'position_decay_stages': [
        # (age_seconds, tp_multiplier)
        # REC-004 (v4.3.0): Delayed start to 5 min for better candle data alignment
        (300, 0.90),                    # 5 min: 90% of original TP
        (360, 0.75),                    # 6 min: 75% of original TP
        (420, 0.50),                    # 7 min: 50% of original TP
        (480, 0.0),                     # 8+ min: Close at any profit
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
    # REC-007 (v4.3.0): Documented design decision for trailing stop default
    # Trailing stops are DISABLED by default for order flow strategies.
    # Rationale: Order flow strategies target quick mean-reversion or momentum
    # moves with fixed profit targets. Trailing stops favor trend-following
    # strategies where moves extend over time. Order flow signals typically
    # resolve within a few minutes - either hitting TP or being exited via
    # position decay. Enable trailing stops only if backtesting shows improved
    # profit factor vs fixed targets.
    # ==========================================================================
    'use_trailing_stop': False,         # Disabled - see rationale above
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


# =============================================================================
# Per-Symbol Configurations
# =============================================================================
SYMBOL_CONFIGS: Dict[str, Dict[str, Any]] = {
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
    # ==========================================================================
    # REC-003 (v4.4.0): XRP/BTC Ratio Pair Configuration
    # ==========================================================================
    # Research findings (December 2025):
    # - Liquidity: ~1,608 BTC/24h (~$160M) - 7-10x less than XRP/USDT
    # - Volatility: 234% daily, XRP is 1.55x more volatile than BTC
    # - Correlation: 0.84 (declining 24.86% over 90 days)
    # - Spread: Wider than USDT pairs due to lower liquidity
    # - Dynamics: Ratio pair behavior with mean reversion potential
    #
    # Configuration rationale:
    # - Higher thresholds: Lower liquidity = more noise in order flow
    # - Smaller position: Lower liquidity = higher slippage risk
    # - Higher volume mult: Need stronger confirmation in thin market
    # - Wider TP/SL: Account for higher volatility while maintaining 2:1 R:R
    # ==========================================================================
    'XRP/BTC': {
        'buy_imbalance_threshold': 0.35,     # Higher: 7-10x lower liquidity than USDT pairs
        'sell_imbalance_threshold': 0.30,    # Higher: requires stronger signal confirmation
        'imbalance_threshold': 0.35,         # Fallback threshold
        'position_size_usd': 15.0,           # Smaller: higher slippage risk in thin market
        'volume_spike_mult': 2.2,            # Higher: need stronger volume confirmation
        'take_profit_pct': 1.5,              # Wider: XRP 1.55x more volatile than BTC
        'stop_loss_pct': 0.75,               # Wider: maintains 2:1 R:R (1.5/0.75)
        'cooldown_trades': 15,               # Higher: fewer quality signals in low liquidity
    },
}


def get_symbol_config(symbol: str, config: Dict[str, Any], key: str) -> Any:
    """Get symbol-specific config or fall back to global config."""
    symbol_cfg = SYMBOL_CONFIGS.get(symbol, {})
    return symbol_cfg.get(key, config.get(key))
