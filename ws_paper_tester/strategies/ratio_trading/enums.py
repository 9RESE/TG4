"""
Ratio Trading Strategy - Enums Module

Type-safe enumerations for volatility regimes, rejection reasons, and exit reasons.
"""
from enum import Enum, auto


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
    CORRELATION_DECLINING = "correlation_declining"  # REC-037
    FEE_NOT_PROFITABLE = "fee_not_profitable"  # REC-050


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
