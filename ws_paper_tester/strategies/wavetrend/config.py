"""
WaveTrend Oscillator Strategy - Configuration and Enums

Contains strategy metadata, enums for type safety, and default configuration.
Based on research from master-plan-v1.0.md.

The WaveTrend Oscillator (by LazyBear) is a momentum indicator that identifies
overbought/oversold conditions with cleaner signals than RSI. It uses a dual-line
crossover mechanism (WT1/WT2) similar to MACD.

Key Features:
- Dual-line system with crossover signals
- Zone-based signal filtering (overbought/oversold)
- Divergence detection for confirmation
- Works well in volatile crypto markets
"""
from enum import Enum, auto
from typing import Dict, Any


# =============================================================================
# Strategy Metadata
# =============================================================================
STRATEGY_NAME = "wavetrend"
STRATEGY_VERSION = "1.0.0"
SYMBOLS = ["XRP/USDT", "BTC/USDT", "XRP/BTC"]


# =============================================================================
# Enums for Type Safety
# =============================================================================
class WaveTrendZone(Enum):
    """WaveTrend zone classification."""
    EXTREME_OVERBOUGHT = auto()  # WT1 >= extreme_overbought
    OVERBOUGHT = auto()          # WT1 >= overbought
    NEUTRAL = auto()             # Between oversold and overbought
    OVERSOLD = auto()            # WT1 <= oversold
    EXTREME_OVERSOLD = auto()    # WT1 <= extreme_oversold


class CrossoverType(Enum):
    """WaveTrend crossover types."""
    BULLISH = auto()   # WT1 crosses above WT2
    BEARISH = auto()   # WT1 crosses below WT2
    NONE = auto()      # No crossover


class DivergenceType(Enum):
    """Price/WaveTrend divergence types."""
    BULLISH = auto()   # Price lower low, WT higher low
    BEARISH = auto()   # Price higher high, WT lower high
    NONE = auto()      # No divergence


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
    WARMING_UP = "warming_up"
    NO_PRICE_DATA = "no_price_data"
    MAX_POSITION = "max_position"
    INSUFFICIENT_SIZE = "insufficient_size"
    NOT_FEE_PROFITABLE = "not_fee_profitable"
    CORRELATION_LIMIT = "correlation_limit"
    NO_CROSSOVER = "no_crossover"
    ZONE_NOT_CONFIRMED = "zone_not_confirmed"
    WAITING_ZONE_EXIT = "waiting_zone_exit"
    EXISTING_POSITION = "existing_position"
    INSUFFICIENT_CANDLES = "insufficient_candles"
    NO_SIGNAL_CONDITIONS = "no_signal_conditions"


# =============================================================================
# Default Configuration
# =============================================================================
CONFIG: Dict[str, Any] = {
    # ==========================================================================
    # WaveTrend Indicator Settings
    # ==========================================================================
    'wt_channel_length': 10,        # ESA and D calculation period
    'wt_average_length': 21,        # WT1 smoothing period
    'wt_ma_length': 4,              # WT2 signal line smoothing

    # ==========================================================================
    # Zone Thresholds
    # ==========================================================================
    'wt_overbought': 60,            # Standard overbought
    'wt_oversold': -60,             # Standard oversold
    'wt_extreme_overbought': 80,    # Extreme overbought
    'wt_extreme_oversold': -80,     # Extreme oversold

    # ==========================================================================
    # Signal Settings
    # ==========================================================================
    'require_zone_exit': True,      # Wait for zone exit before entry
    'use_divergence': True,         # Include divergence in confidence
    'divergence_lookback': 14,      # Candles for divergence calculation

    # ==========================================================================
    # Position Sizing
    # ==========================================================================
    'position_size_usd': 25.0,          # Base position size in USD
    'max_position_usd': 75.0,           # Maximum TOTAL position exposure
    'max_position_per_symbol_usd': 50.0,  # Maximum per symbol
    'min_trade_size_usd': 5.0,          # Minimum USD per trade
    'short_size_multiplier': 0.8,       # Reduce short position size

    # ==========================================================================
    # Risk Management - Target 2:1 R:R ratio
    # ==========================================================================
    'stop_loss_pct': 1.5,           # Stop loss at 1.5%
    'take_profit_pct': 3.0,         # Take profit at 3.0%

    # ==========================================================================
    # Confidence Caps
    # ==========================================================================
    'max_long_confidence': 0.92,    # Maximum confidence for longs
    'max_short_confidence': 0.88,   # Maximum confidence for shorts

    # ==========================================================================
    # Candle Management
    # For WaveTrend: min_candles = max(channel_length, average_length, divergence_lookback * 2) + 10
    # With defaults: max(10, 21, 28) + 10 = 38 candles minimum
    # ==========================================================================
    'min_candle_buffer': 50,        # Minimum candles required (safety margin)

    # ==========================================================================
    # Cooldown Mechanisms
    # ==========================================================================
    'cooldown_seconds': 60.0,       # Min time between signals (longer for hourly TF)

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
    'session_size_multipliers': {
        'ASIA': 0.8,                # Smaller sizes
        'EUROPE': 1.0,              # Standard sizes
        'US': 1.0,                  # Standard sizes
        'US_EUROPE_OVERLAP': 1.1,   # Larger sizes (peak liquidity)
        'OFF_HOURS': 0.5,           # Smallest sizes (thin liquidity)
    },

    # ==========================================================================
    # Circuit Breaker
    # ==========================================================================
    'use_circuit_breaker': True,
    'max_consecutive_losses': 3,    # Max losses before cooldown
    'circuit_breaker_minutes': 30,  # Longer cooldown for hourly strategy

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
}


# =============================================================================
# Per-Symbol Configurations
# Based on research from master-plan-v1.0.md pair-specific analysis
# =============================================================================
SYMBOL_CONFIGS: Dict[str, Dict[str, Any]] = {
    # ==========================================================================
    # XRP/USDT Configuration
    # Research: 5.1% intraday volatility, good for hourly signals
    # Suitability: HIGH
    # ==========================================================================
    'XRP/USDT': {
        'wt_overbought': 60,
        'wt_oversold': -60,
        'wt_extreme_overbought': 75,    # Slightly lower for faster signals
        'wt_extreme_oversold': -75,     # Slightly higher for faster signals
        'position_size_usd': 25.0,
        'stop_loss_pct': 1.5,           # Wider for hourly timeframe
        'take_profit_pct': 3.0,         # 2:1 R:R ratio
    },

    # ==========================================================================
    # BTC/USDT Configuration
    # Research: Lower % volatility (1.64%), strong trending at extremes
    # Suitability: MEDIUM-HIGH (needs caution in strong trends)
    # ==========================================================================
    'BTC/USDT': {
        'wt_overbought': 65,            # Higher - BTC sustains overbought longer
        'wt_oversold': -65,             # Higher threshold for same reason
        'wt_extreme_overbought': 80,    # Standard extreme
        'wt_extreme_oversold': -80,     # Standard extreme
        'position_size_usd': 50.0,      # Larger due to lower % volatility
        'stop_loss_pct': 1.0,           # Tighter - more predictable moves
        'take_profit_pct': 2.0,         # 2:1 R:R ratio
    },

    # ==========================================================================
    # XRP/BTC Configuration
    # Research: 7-10x lower liquidity, ratio pair dynamics
    # Suitability: MEDIUM (approach cautiously)
    # ==========================================================================
    'XRP/BTC': {
        'wt_overbought': 55,            # Lower - ratio pairs move differently
        'wt_oversold': -55,             # Lower threshold
        'wt_extreme_overbought': 70,    # Lower extreme
        'wt_extreme_oversold': -70,     # Lower extreme
        'position_size_usd': 15.0,      # Smaller - liquidity constraints
        'stop_loss_pct': 2.0,           # Wider - higher volatility
        'take_profit_pct': 4.0,         # Wider targets
        'cooldown_seconds': 120,        # Longer - fewer signals
    },
}


def get_symbol_config(symbol: str, config: Dict[str, Any], key: str) -> Any:
    """
    Get symbol-specific config or fall back to global config.

    Args:
        symbol: Trading symbol (e.g., 'XRP/USDT')
        config: Global configuration dict
        key: Configuration key to look up

    Returns:
        Symbol-specific value if available, otherwise global config value
    """
    symbol_cfg = SYMBOL_CONFIGS.get(symbol, {})
    return symbol_cfg.get(key, config.get(key))
