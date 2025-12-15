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

===============================================================================
REC-010: WARMUP REQUIREMENT WARNING
===============================================================================
This strategy requires 50 candles (5-minute timeframe) before generating signals.

Warmup Calculation:
- min_candle_buffer: 50 candles
- candle_timeframe: 5 minutes (using data.candles_5m)
- Warmup time: 50 * 5 = 250 minutes ≈ 4.2 hours minimum

NO SIGNALS will be generated until warmup is complete. This is by design:
- WaveTrend calculation needs historical data for EMA smoothing
- Divergence detection requires lookback period
- Zone classification needs stable indicator values

If using 1-minute candles as fallback: 50 * 1 = 50 minutes minimum
===============================================================================

Version History:
- 1.0.0: Initial implementation
- 1.1.0: REC-001 Trade Flow Confirmation, REC-002 Real Correlation Monitoring,
         REC-006 Blocking R:R Validation, Documentation updates (REC-008/009/010/011/012)
"""
from enum import Enum, auto
from typing import Dict, Any


# =============================================================================
# Strategy Metadata
# =============================================================================
STRATEGY_NAME = "wavetrend"
STRATEGY_VERSION = "1.1.0"  # REC-001, REC-002, REC-006 + documentation updates
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
    TRADE_FLOW_AGAINST = "trade_flow_against"  # REC-001: Trade flow doesn't confirm signal


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
    # REC-009: Zone Exit Trade-off Documentation
    # require_zone_exit controls signal quality vs frequency:
    # - True (default): Wait for price to exit OB/OS zone before entry
    #   Pros: Higher quality signals, better confirmation
    #   Cons: Fewer signals, may miss some opportunities
    # - False: Enter immediately on crossover in zone
    #   Pros: More signals, earlier entries
    #   Cons: Potentially lower quality, more false signals
    # Research: Zone-filtered signals have ~15-20% higher reliability
    # ==========================================================================
    'require_zone_exit': True,      # Wait for zone exit (higher quality, fewer signals)
    'use_divergence': True,         # Include divergence in confidence calculation
    'divergence_lookback': 14,      # Candles for divergence detection (REC-011: aligned with buffer)

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
    # REC-008: Confidence caps are intentionally asymmetric
    # - Longs have higher cap (0.92) because crypto markets have upward bias
    # - Shorts have lower cap (0.88) due to short squeeze risk and inherently
    #   higher risk of shorting in trending crypto markets
    # Research: Crypto markets historically favor long positions over time
    # ==========================================================================
    'max_long_confidence': 0.92,    # Higher cap for longs (upward market bias)
    'max_short_confidence': 0.88,   # Lower cap for shorts (squeeze risk)

    # ==========================================================================
    # Candle Management
    # REC-011: Divergence lookback alignment with buffer size
    #
    # Minimum candle requirements calculation:
    # - WaveTrend: max(channel_length, average_length) + ma_length + 5
    #              = max(10, 21) + 4 + 5 = 30 candles
    # - Divergence: divergence_lookback * 2 + 5 = 14 * 2 + 5 = 33 candles
    # - Combined minimum: max(30, 33) = 33 candles
    #
    # Current buffer: 50 candles provides:
    # - 17 candles safety margin (50 - 33 = 17)
    # - Allows for stable indicator warmup
    # - Provides full divergence detection window
    #
    # If divergence_lookback is changed, update min_candle_buffer accordingly:
    # Formula: min_candle_buffer >= max(average_length + ma_length + 5, divergence_lookback * 2 + 5)
    # ==========================================================================
    'min_candle_buffer': 50,        # Minimum candles required (includes safety margin)

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
    # REC-002: Real-time correlation monitoring vs estimated values
    # ==========================================================================
    'use_correlation_management': True,
    'max_total_long_exposure': 100.0,   # Max total long USD exposure
    'max_total_short_exposure': 100.0,  # Max total short USD exposure
    'same_direction_size_mult': 0.75,   # Reduce size if both pairs same direction
    'use_real_correlation': True,       # REC-002: Calculate real-time rolling correlation
    'correlation_window': 20,           # REC-002: Candles for correlation calculation
    'correlation_block_threshold': 0.85,  # Block if correlation > this

    # ==========================================================================
    # Fee Profitability
    # ==========================================================================
    'fee_rate': 0.001,              # 0.1% per trade
    'min_profit_after_fees_pct': 0.1,  # Minimum profit after fees
    'use_fee_check': True,

    # ==========================================================================
    # REC-001: Trade Flow Confirmation
    # Validates signals against market microstructure (buy/sell volume imbalance)
    # ==========================================================================
    'use_trade_flow_confirmation': True,
    'trade_flow_threshold': 0.10,       # Min imbalance to confirm signal (10%)
    'trade_flow_lookback': 50,          # Number of recent trades to analyze

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
