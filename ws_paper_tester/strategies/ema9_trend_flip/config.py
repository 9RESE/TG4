"""
EMA-9 Trend Flip Strategy - Configuration

Strategy metadata, enums, and configuration.
Based on: ws_paper_tester/docs/development/plans/ema9/ema-9-strategy-analysis.md
"""
from enum import Enum, auto
from typing import Dict, Any


# =============================================================================
# Strategy Metadata
# =============================================================================
STRATEGY_NAME = "ema9_trend_flip"
STRATEGY_VERSION = "1.0.0"
SYMBOLS = ["BTC/USDT"]  # 1H works best for BTC per analysis


# =============================================================================
# Enums for Type Safety
# =============================================================================
class TrendDirection(Enum):
    """Current trend direction based on EMA position."""
    BULLISH = auto()   # Price opening above EMA
    BEARISH = auto()   # Price opening below EMA
    NEUTRAL = auto()   # Not enough data


class RejectionReason(Enum):
    """Signal rejection reasons for tracking."""
    WARMING_UP = "warming_up"
    NO_PRICE_DATA = "no_price_data"
    INSUFFICIENT_CANDLES = "insufficient_candles"
    NO_FLIP_SIGNAL = "no_flip_signal"
    EXISTING_POSITION = "existing_position"
    MAX_POSITION = "max_position"
    TIME_COOLDOWN = "time_cooldown"
    BUFFER_NOT_MET = "buffer_not_met"
    CIRCUIT_BREAKER = "circuit_breaker"  # Issue #6: Circuit breaker active


# =============================================================================
# Default Configuration
# =============================================================================
CONFIG: Dict[str, Any] = {
    # ==========================================================================
    # EMA Settings
    # ==========================================================================
    'ema_period': 9,                    # EMA period (9 is optimal per analysis)
    'consecutive_candles': 3,           # Min consecutive candles on one side before flip
    'buffer_pct': 0.1,                  # Buffer % above/below EMA to reduce whipsaws
    'use_open_price': True,             # Use candle open price (True) or close (False)

    # ==========================================================================
    # Timeframe Settings
    # ==========================================================================
    'candle_timeframe_minutes': 60,     # 1H candles (60 minutes)
    'min_candles_required': 15,         # Minimum 1H candles for EMA calculation

    # ==========================================================================
    # Position Sizing
    # ==========================================================================
    'position_size_usd': 50.0,          # Trade size in USD
    'max_position_usd': 100.0,          # Maximum position exposure
    'min_trade_size_usd': 10.0,         # Minimum trade size

    # ==========================================================================
    # Risk Management - Targeting 2:1 R:R
    # ==========================================================================
    'stop_loss_pct': 1.0,               # Stop loss percentage (larger for 1H)
    'take_profit_pct': 2.0,             # Take profit percentage (2:1 R:R)
    'use_atr_stops': False,             # Use ATR-based stops instead of fixed %
    'atr_stop_mult': 1.5,               # ATR multiplier for stop loss
    'atr_tp_mult': 3.0,                 # ATR multiplier for take profit

    # ==========================================================================
    # Exit Conditions
    # ==========================================================================
    'exit_on_flip': True,               # Exit when EMA flips to opposite side
    'max_hold_hours': 72,               # Maximum hold time in hours (3 days)

    # ==========================================================================
    # Cooldown Mechanisms
    # ==========================================================================
    'cooldown_minutes': 30,             # Min minutes between signals
    'cooldown_after_loss_minutes': 60,  # Extended cooldown after loss

    # ==========================================================================
    # Signal Tracking
    # ==========================================================================
    'track_rejections': True,

    # ==========================================================================
    # Circuit Breaker
    # ==========================================================================
    'use_circuit_breaker': True,
    'max_consecutive_losses': 3,        # Max losses before cooldown
    'circuit_breaker_minutes': 30,      # Cooldown after max losses
}


# =============================================================================
# Per-Symbol Configuration (for future expansion)
# =============================================================================
SYMBOL_CONFIGS: Dict[str, Dict[str, Any]] = {
    'BTC/USDT': {
        'position_size_usd': 50.0,
        'stop_loss_pct': 1.0,
        'take_profit_pct': 2.0,
        'consecutive_candles': 3,
    },
}


def get_symbol_config(symbol: str, config: Dict[str, Any], key: str) -> Any:
    """
    Get configuration value for a symbol, falling back to global config.

    Args:
        symbol: Trading symbol
        config: Global configuration
        key: Configuration key

    Returns:
        Configuration value
    """
    symbol_specific = SYMBOL_CONFIGS.get(symbol, {})
    if key in symbol_specific:
        return symbol_specific[key]
    return config.get(key)
