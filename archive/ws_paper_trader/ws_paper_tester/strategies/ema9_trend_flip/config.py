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
STRATEGY_VERSION = "2.0.1"  # v2.0.1: EMA fix - use CLOSE prices, previous EMA for comparison
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
    'consecutive_candles': 2,           # Min consecutive candles on one side before flip
    'buffer_pct': 0.0,                  # Buffer % above/below EMA (0 = exact crossover)
    # NOTE: EMA calculated on CLOSE prices (industry standard, matches TradingView/Binance)

    # ==========================================================================
    # v2.0: Strict Candle Mode (Whole Candle Check) - REQUIRED
    # ==========================================================================
    # Requires the ENTIRE candle (including wicks) to be above/below EMA
    # This prevents false signals where candles cross the EMA during their timeframe
    # Optimization results: 50% win rate vs 12.6%, 0.48% DD vs 50.44%
    'strict_candle_mode': True,         # REQUIRED: Whole candle above/below EMA

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
    # Risk Management - Stop Loss for Protection Only
    # ==========================================================================
    # NOTE: The EMA flip IS the profit exit. No take_profit_pct needed.
    # Stop loss is for catastrophic protection only (violent moves before flip).
    'stop_loss_pct': 2.5,               # Wide stop loss (protection only)
    'use_atr_stops': True,              # Use ATR-based stops (recommended)
    'atr_stop_mult': 2.0,               # ATR multiplier for stop loss (2x ATR)

    # ==========================================================================
    # Exit Conditions
    # ==========================================================================
    # NOTE: Exit on EMA flip is ALWAYS enabled - it's the core strategy.
    # The flip IS the exit signal. Hold until flip occurs - no time limit.
    'exit_confirmation_candles': 1,     # Candles required to confirm exit (1 = immediate)

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
        'stop_loss_pct': 2.5,           # Wide stop for BTC volatility
        'consecutive_candles': 2,
        'buffer_pct': 0.0,              # No buffer - exact EMA crossover
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
