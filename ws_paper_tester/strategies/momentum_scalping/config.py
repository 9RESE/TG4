"""
Momentum Scalping Strategy - Configuration and Enums

Contains strategy metadata, enums for type safety, and default configuration.
Based on research from master-plan-v1.0.md.
"""
from enum import Enum, auto
from typing import Dict, Any


# =============================================================================
# Strategy Metadata
# =============================================================================
STRATEGY_NAME = "momentum_scalping"
STRATEGY_VERSION = "2.0.0"
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
    OFF_HOURS = auto()


class RejectionReason(Enum):
    """Signal rejection reasons for tracking."""
    CIRCUIT_BREAKER = "circuit_breaker"
    TIME_COOLDOWN = "time_cooldown"
    TRADE_COOLDOWN = "trade_cooldown"
    WARMING_UP = "warming_up"
    REGIME_PAUSE = "regime_pause"
    NO_VOLUME = "no_volume"
    NO_PRICE_DATA = "no_price_data"
    MAX_POSITION = "max_position"
    INSUFFICIENT_SIZE = "insufficient_size"
    NOT_FEE_PROFITABLE = "not_fee_profitable"
    TREND_NOT_ALIGNED = "trend_not_aligned"
    MOMENTUM_NOT_CONFIRMED = "momentum_not_confirmed"
    VOLUME_NOT_CONFIRMED = "volume_not_confirmed"
    CORRELATION_LIMIT = "correlation_limit"
    NO_SIGNAL_CONDITIONS = "no_signal_conditions"
    EXISTING_POSITION = "existing_position"
    # REC-001 (v2.0.0): XRP/BTC correlation breakdown
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    # REC-002 (v2.0.0): 5m timeframe misalignment
    TIMEFRAME_MISALIGNMENT = "timeframe_misalignment"
    # REC-003 (v2.0.0): ADX strong trend filter for BTC
    ADX_STRONG_TREND = "adx_strong_trend"


class MomentumDirection(Enum):
    """Momentum direction classification."""
    BULLISH = auto()
    BEARISH = auto()
    NEUTRAL = auto()


# =============================================================================
# Default Configuration
# =============================================================================
CONFIG: Dict[str, Any] = {
    # ==========================================================================
    # Indicator Settings (Optimized for 1-minute scalping)
    # ==========================================================================
    'ema_fast_period': 8,           # Ultra-fast trend detection
    'ema_slow_period': 21,          # Short-term trend
    'ema_filter_period': 50,        # Trend direction filter
    'rsi_period': 7,                # Fast momentum (scalping optimized)
    'rsi_overbought': 70,           # RSI overbought level
    'rsi_oversold': 30,             # RSI oversold level
    'macd_fast': 6,                 # MACD fast EMA (scalping optimized)
    'macd_slow': 13,                # MACD slow EMA
    'macd_signal': 5,               # MACD signal line
    'use_macd_confirmation': True,  # Use MACD as secondary confirmation

    # ==========================================================================
    # Volume Confirmation
    # ==========================================================================
    'volume_lookback': 20,          # Rolling average volume lookback
    'volume_spike_threshold': 1.5,  # Min volume spike multiplier for entry
    'require_volume_confirmation': True,  # Require volume spike for entries

    # ==========================================================================
    # Position Sizing
    # ==========================================================================
    'position_size_usd': 25.0,          # Size per trade in USD
    'max_position_usd': 75.0,           # Maximum TOTAL position exposure
    'max_position_per_symbol_usd': 50.0,  # Maximum per symbol
    'min_trade_size_usd': 5.0,          # Minimum USD per trade

    # ==========================================================================
    # Risk Management - Target 2:1 R:R ratio
    # ==========================================================================
    'take_profit_pct': 0.8,         # Take profit at 0.8%
    'stop_loss_pct': 0.4,           # Stop loss at 0.4%
    'max_hold_seconds': 180,        # 3 minutes max hold time

    # ==========================================================================
    # Cooldown Mechanisms
    # ==========================================================================
    'cooldown_seconds': 30.0,       # Min time between signals
    'cooldown_trades': 5,           # Min candles between signals

    # ==========================================================================
    # Volatility Regime Classification
    # ==========================================================================
    'use_volatility_regimes': True,
    'regime_low_threshold': 0.2,    # Below = LOW regime
    'regime_medium_threshold': 0.6, # Below = MEDIUM regime
    'regime_high_threshold': 1.2,   # Below = HIGH regime, above = EXTREME
    'regime_extreme_reduce_size': 0.5,  # Position size multiplier in EXTREME
    'regime_extreme_pause': True,   # Pause trading in EXTREME

    # ==========================================================================
    # Session Awareness
    # ==========================================================================
    'use_session_awareness': True,
    'session_boundaries': {
        'asia_start': 0,            # 00:00 UTC
        'asia_end': 8,              # 08:00 UTC
        'europe_start': 8,          # 08:00 UTC
        'europe_end': 14,           # 14:00 UTC
        'overlap_start': 14,        # 14:00 UTC
        'overlap_end': 17,          # 17:00 UTC
        'us_start': 17,             # 17:00 UTC
        'us_end': 21,               # 21:00 UTC
        'off_hours_start': 21,      # 21:00 UTC
        'off_hours_end': 24,        # 24:00 UTC
    },
    'session_threshold_multipliers': {
        'ASIA': 1.2,                # Wider thresholds (lower volume)
        'EUROPE': 1.0,              # Standard thresholds
        'US': 1.0,                  # Standard thresholds
        'US_EUROPE_OVERLAP': 0.9,   # Tighter thresholds (peak liquidity)
        'OFF_HOURS': 1.4,           # Very wide thresholds (thin liquidity)
    },
    'session_size_multipliers': {
        'ASIA': 0.8,                # Smaller sizes
        'EUROPE': 1.0,              # Standard sizes
        'US': 1.0,                  # Standard sizes
        'US_EUROPE_OVERLAP': 1.1,   # Larger sizes
        'OFF_HOURS': 0.5,           # Smallest sizes
    },

    # ==========================================================================
    # Circuit Breaker
    # ==========================================================================
    'use_circuit_breaker': True,
    'max_consecutive_losses': 3,    # Max losses before cooldown
    'circuit_breaker_minutes': 10,  # Cooldown after max losses

    # ==========================================================================
    # Cross-Pair Correlation Management
    # ==========================================================================
    'use_correlation_management': True,
    'max_total_long_exposure': 100.0,   # Max total long USD exposure
    'max_total_short_exposure': 100.0,  # Max total short USD exposure
    'same_direction_size_mult': 0.75,   # Reduce size if both pairs same direction

    # ==========================================================================
    # Fee Profitability
    # ==========================================================================
    'fee_rate': 0.001,              # 0.1% per trade
    'min_profit_after_fees_pct': 0.1,  # Minimum profit after fees
    'use_fee_check': True,

    # ==========================================================================
    # Signal Tracking
    # ==========================================================================
    'track_rejections': True,

    # ==========================================================================
    # Multi-Timeframe Confirmation (REC-002 v2.0.0)
    # ==========================================================================
    'use_5m_trend_filter': True,    # Use 5m timeframe for trend confirmation
    'require_ema_alignment': True,  # Require price alignment with EMAs
    '5m_ema_period': 50,            # REC-002: EMA period for 5m trend filter

    # ==========================================================================
    # XRP/BTC Correlation Monitoring (REC-001 v2.0.0)
    # Deep Review v1.0: XRP-BTC correlation has declined from 0.85 to 0.40-0.67
    # Momentum signals on XRP/BTC are unreliable when correlation is low
    # ==========================================================================
    'use_correlation_monitoring': True,         # Enable correlation monitoring
    'correlation_lookback': 50,                 # Candles for correlation calculation
    'correlation_warn_threshold': 0.55,         # Warn when correlation drops below
    'correlation_pause_threshold': 0.50,        # Pause XRP/BTC when below this
    'correlation_pause_enabled': True,          # Enable auto-pause for XRP/BTC

    # ==========================================================================
    # ADX Trend Strength Filter (REC-003 v2.0.0)
    # Deep Review v1.0: BTC exhibits strong trending behavior at price extremes
    # ADX > 25 indicates strong trend where momentum scalping may fail
    # ==========================================================================
    'use_adx_filter': True,                     # Enable ADX filtering for BTC
    'adx_period': 14,                           # ADX calculation period
    'adx_strong_trend_threshold': 25,           # Strong trend threshold
    'adx_filter_btc_only': True,                # Only apply ADX filter to BTC/USDT

    # ==========================================================================
    # Regime-Based RSI Adjustment (REC-004 v2.0.0)
    # Deep Review v1.0: Crypto can sustain overbought conditions longer
    # Widen RSI bands during HIGH volatility regime
    # ==========================================================================
    'regime_high_rsi_overbought': 75,           # RSI overbought in HIGH regime
    'regime_high_rsi_oversold': 25,             # RSI oversold in HIGH regime
}


# =============================================================================
# Per-Symbol Configurations
# Based on research from master-plan-v1.0.md pair-specific analysis
# =============================================================================
SYMBOL_CONFIGS: Dict[str, Dict[str, Any]] = {
    # ==========================================================================
    # XRP/USDT Configuration
    # Research: High liquidity, 5.1% intraday volatility, 0.15% spread
    # Suitability: HIGH
    # ==========================================================================
    'XRP/USDT': {
        'ema_fast_period': 8,
        'ema_slow_period': 21,
        'ema_filter_period': 50,
        'rsi_period': 7,
        'position_size_usd': 25.0,
        'take_profit_pct': 0.8,     # Account for 0.15% spread
        'stop_loss_pct': 0.4,       # 2:1 R:R ratio
        'volume_spike_threshold': 1.5,
    },

    # ==========================================================================
    # BTC/USDT Configuration
    # Research: Deepest liquidity, lower volatility, institutional dominated
    # Suitability: MEDIUM-HIGH
    # ==========================================================================
    'BTC/USDT': {
        'ema_fast_period': 8,
        'ema_slow_period': 21,
        'ema_filter_period': 50,
        'rsi_period': 9,            # Slightly slower (less noise)
        'position_size_usd': 50.0,  # Higher due to lower volatility
        'take_profit_pct': 0.6,     # Conservative due to efficiency
        'stop_loss_pct': 0.3,       # Tight stops viable with low spread
        'volume_spike_threshold': 1.8,  # Higher threshold (high normal volume)
    },

    # ==========================================================================
    # XRP/BTC Configuration
    # Research: 7-10x lower liquidity, XRP 1.55x more volatile than BTC
    # Suitability: MEDIUM
    # ==========================================================================
    'XRP/BTC': {
        'ema_fast_period': 8,
        'ema_slow_period': 21,
        'ema_filter_period': 50,
        'rsi_period': 9,            # Slower to reduce noise
        'position_size_usd': 15.0,  # Smaller: slippage risk
        'take_profit_pct': 1.2,     # Wider: higher volatility
        'stop_loss_pct': 0.6,       # Wider: maintains 2:1 R:R
        'volume_spike_threshold': 2.0,  # Higher: need strong confirmation
        'cooldown_trades': 15,      # Higher: fewer quality signals
    },
}


def get_symbol_config(symbol: str, config: Dict[str, Any], key: str) -> Any:
    """Get symbol-specific config or fall back to global config."""
    symbol_cfg = SYMBOL_CONFIGS.get(symbol, {})
    return symbol_cfg.get(key, config.get(key))
