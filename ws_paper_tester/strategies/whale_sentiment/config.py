"""
Whale Sentiment Strategy - Configuration and Enums

Contains strategy metadata, enums for type safety, and default configuration.

REC-009: Research Foundation
This strategy is based on internal research documented in the whale_sentiment
feature documentation. Key research sources include:
- "The Moby Dick Effect" (Magner & Sanhueza, 2025): Whale contagion effects
- Philadelphia Federal Reserve (2024): Whale vs retail behavior
- PMC/NIH (2023): RSI ineffectiveness in crypto markets
- QuantifiedStrategies.com (2024): RSI as momentum vs mean reversion
See deep-review-v2.0.md Section 7 for full research references.

The Whale Sentiment Strategy combines institutional activity detection (via volume
spike analysis) with price deviation sentiment indicators to identify contrarian
trading opportunities.

Key Features (v1.2.0):
- Volume spike detection as whale activity proxy (PRIMARY signal - 55% weight)
- Price deviation for sentiment classification (35% weight)
- Trade flow confirmation (10% weight)
- RSI REMOVED from confidence calculation per REC-013 (academic evidence)
- Contrarian mode: buy fear, sell greed
- Candle data persistence for fast restart recovery
- Cross-pair correlation management

===============================================================================
REC-012: WARMUP WITH PERSISTENCE
===============================================================================
Warmup time is significantly reduced with candle persistence (REC-011):
- Fresh start: 310 candles @ 5m = 25.8 hours
- With persistence: Resume from saved data (typically < 30 min gap)

Persistence saves candle data every 5 minutes to:
- data/candles/{symbol}_5m.json

On restart, data is reloaded if:
- File exists and is valid JSON
- Last candle timestamp within max_candle_age_hours (default: 4 hours)

===============================================================================
DEFERRED RECOMMENDATIONS (Future Implementation)
===============================================================================
REC-014 (HIGH, High Effort): Volatility Regime Classification
- Current: Uses sentiment regimes only
- Proposed: Add ATR-based volatility regime detection
- Benefits: Adjust parameters for different market conditions
- Reference: Strategy Development Guide Section 15 (when available)

REC-015 (HIGH, High Effort): Backtest Confidence Weights
- Current: Weights based on theoretical analysis + REC-013 removal
- Proposed: Validate weights with historical backtesting
- Benefits: Empirically optimized confidence calculation
===============================================================================

Version History:
- 1.2.0: Deep Review v2.0 Implementation
         - REC-011: Implemented candle data persistence for fast restarts
         - REC-012: Added warmup progress indicator (pct, ETA)
         - REC-013: REMOVED RSI from confidence calculation (academic evidence)
         - REC-016: Added XRP/BTC re-enablement guard with explicit flag
         - REC-017: Added UTC timezone validation on startup
         - REC-018: Added trade flow expected indicator for clarity
         - REC-019: Volume window now configurable per-symbol
         - REC-020: Extracted magic numbers to config parameters
         - REC-014/REC-015: Documented for future implementation
- 1.1.0: Deep Review v1.0 Implementation
         - REC-001: Recalibrated confidence weights (volume 40%, RSI 15%)
         - REC-005: Enhanced indicator logging on all code paths
         - REC-007: Disabled XRP/BTC by default (liquidity concerns)
         - REC-008: Reduced short size multiplier to 0.5x (squeeze risk)
         - REC-009: Updated research documentation references
         - REC-010: Documented UTC timezone requirement for sessions
         - REC-002/REC-004/REC-006: Documented for future implementation
         - REC-003: Clarified trade flow logic for contrarian mode
- 1.0.0: Initial implementation
         - Volume spike detection as whale proxy
         - RSI sentiment classification
         - Fear/greed price deviation proxy
         - Contrarian mode signal generation
         - Trade flow confirmation
         - Cross-pair correlation management
"""
from enum import Enum, auto
from typing import Dict, Any


# =============================================================================
# Strategy Metadata
# =============================================================================
STRATEGY_NAME = "whale_sentiment"
STRATEGY_VERSION = "1.2.0"  # Deep Review v2.0 Implementation
# REC-007: XRP/BTC disabled by default due to 7-10x lower liquidity than USD pairs.
# REC-016: To re-enable, add to SYMBOLS AND set enable_xrpbtc: true in config.
SYMBOLS = ["XRP/USDT", "BTC/USDT"]
# SYMBOLS_WITH_XRPBTC = ["XRP/USDT", "BTC/USDT", "XRP/BTC"]  # Requires enable_xrpbtc: true


# =============================================================================
# Enums for Type Safety
# =============================================================================
class SentimentZone(Enum):
    """Market sentiment classification based on RSI and price deviation."""
    EXTREME_FEAR = auto()      # RSI < rsi_extreme_fear, large drop from high
    FEAR = auto()              # RSI < rsi_fear, moderate drop
    NEUTRAL = auto()           # RSI between fear and greed thresholds
    GREED = auto()             # RSI > rsi_greed, moderate rise
    EXTREME_GREED = auto()     # RSI > rsi_extreme_greed, large rise from low


class WhaleSignal(Enum):
    """Volume spike classification (whale activity proxy)."""
    ACCUMULATION = auto()      # Volume spike + price up
    DISTRIBUTION = auto()      # Volume spike + price down
    NEUTRAL = auto()           # No significant volume spike


class SignalDirection(Enum):
    """Composite signal direction."""
    BULLISH = auto()
    BEARISH = auto()
    NEUTRAL = auto()


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
    NO_VOLUME_SPIKE = "no_volume_spike"
    NEUTRAL_SENTIMENT = "neutral_sentiment"
    INSUFFICIENT_CONFIDENCE = "insufficient_confidence"
    VOLUME_FALSE_POSITIVE = "volume_false_positive"
    EXISTING_POSITION = "existing_position"
    INSUFFICIENT_CANDLES = "insufficient_candles"
    NO_SIGNAL_CONDITIONS = "no_signal_conditions"
    TRADE_FLOW_AGAINST = "trade_flow_against"
    WHALE_SIGNAL_MISMATCH = "whale_signal_mismatch"
    SENTIMENT_ZONE_MISMATCH = "sentiment_zone_mismatch"


# =============================================================================
# Default Configuration
# =============================================================================
CONFIG: Dict[str, Any] = {
    # ==========================================================================
    # Whale Detection (Volume Spike)
    # Volume spikes 2x+ above average strongly indicate institutional activity
    # ==========================================================================
    'volume_spike_mult': 2.0,           # Volume spike threshold multiplier
    'volume_window': 288,               # 24h in 5m candles (24 * 12)
    'min_spike_trades': 20,             # Minimum trades during spike (false positive filter)
    'max_spread_pct': 0.5,              # Maximum spread during spike (manipulation filter)
    'volume_spike_price_move_pct': 0.1, # Min price move with spike (wash trading filter)

    # ==========================================================================
    # RSI Sentiment Settings
    # Crypto-specific RSI thresholds (adjusted from 30/70 standard)
    # ==========================================================================
    'rsi_period': 14,
    'rsi_extreme_fear': 25,             # Extreme fear (panic selling)
    'rsi_fear': 40,                     # Fear zone
    'rsi_greed': 60,                    # Greed zone
    'rsi_extreme_greed': 75,            # Extreme greed (FOMO buying)

    # ==========================================================================
    # Fear/Greed Price Deviation
    # Price deviation from recent high/low as sentiment proxy
    # ==========================================================================
    'fear_deviation_pct': -5.0,         # -5% from recent high = fear
    'greed_deviation_pct': 5.0,         # +5% from recent low = greed
    'price_lookback': 48,               # 4h window in 5m candles for high/low

    # ==========================================================================
    # Contrarian Mode
    # Default: Buy fear, sell greed (contrarian approach)
    # Set to False for momentum following
    # ==========================================================================
    'contrarian_mode': True,

    # ==========================================================================
    # Composite Confidence Weights
    # REC-013: RSI REMOVED from confidence calculation based on academic evidence
    # showing RSI ineffectiveness in crypto markets (PMC/NIH 2023, QuantifiedStrategies 2024).
    # Weight redistributed to volume spike (primary signal) and price deviation.
    # ==========================================================================
    'weight_volume_spike': 0.55,        # Volume spike contribution (REC-013: increased from 0.40)
    'weight_rsi_sentiment': 0.00,       # REC-013: RSI REMOVED - academically ineffective in crypto
    'weight_price_deviation': 0.35,     # Price deviation contribution (REC-013: increased from 0.20)
    'weight_trade_flow': 0.10,          # Trade flow confirmation (REC-013: reduced from 0.15)
    'weight_divergence': 0.00,          # REC-013: RSI divergence removed with RSI
    'min_confidence': 0.50,             # Minimum confidence (lowered due to fewer components)
    'max_confidence': 0.90,             # Maximum confidence cap
    # REC-020: Extracted magic numbers for volume confidence calculation
    'volume_confidence_base': 0.50,     # Base contribution when volume spike detected
    'volume_confidence_bonus_per_ratio': 0.05,  # Additional per volume ratio above threshold

    # ==========================================================================
    # Position Sizing
    # ==========================================================================
    'position_size_usd': 25.0,          # Base position size in USD
    'max_position_usd': 150.0,          # Maximum TOTAL position exposure
    'max_position_per_symbol_usd': 75.0,  # Maximum per symbol
    'min_trade_size_usd': 5.0,          # Minimum USD per trade
    'short_size_multiplier': 0.50,      # REC-008: Reduced from 0.75 for crypto squeeze risk
    'high_correlation_size_mult': 0.60, # Reduce when correlated

    # ==========================================================================
    # Risk Management - Target 2:1 R:R ratio
    # Wider stops for contrarian entries (counter-trend)
    # ==========================================================================
    'stop_loss_pct': 2.5,               # Stop loss at 2.5%
    'take_profit_pct': 5.0,             # Take profit at 5.0%

    # ==========================================================================
    # Trailing Stop (Optional)
    # Activate after 50% of TP reached
    # ==========================================================================
    'use_trailing_stop': False,
    'trailing_stop_activation_pct': 50.0,  # Activate at 2.5% profit
    'trailing_stop_distance_pct': 1.0,     # Trail 1% below peak

    # ==========================================================================
    # Confidence Caps
    # Asymmetric caps due to crypto market characteristics
    # ==========================================================================
    'max_long_confidence': 0.90,        # Cap for longs
    'max_short_confidence': 0.85,       # Lower cap for shorts (squeeze risk)

    # ==========================================================================
    # Candle Management
    # REC-010: Extended warmup for volume baseline
    # ==========================================================================
    'min_candle_buffer': 310,           # ~26 hours @ 5m candles (volume_window + safety margin)

    # ==========================================================================
    # Cooldown Mechanisms
    # ==========================================================================
    'cooldown_seconds': 120.0,          # Min time between signals (2 minutes)

    # ==========================================================================
    # Session Awareness
    # ==========================================================================
    'use_session_awareness': True,
    # REC-010: Session boundaries are UTC-only. No DST adjustment.
    # All times assume server runs in UTC timezone.
    'session_boundaries': {
        'asia_start': 0,                # 00:00 UTC (Tokyo 09:00 JST)
        'asia_end': 8,                  # 08:00 UTC (Tokyo 17:00 JST)
        'europe_start': 8,              # 08:00 UTC (London 08:00/09:00 GMT/BST)
        'europe_end': 14,               # 14:00 UTC (London 14:00/15:00 GMT/BST)
        'overlap_start': 14,            # 14:00 UTC (NY 09:00/10:00 EST/EDT)
        'overlap_end': 17,              # 17:00 UTC (NY 12:00/13:00 EST/EDT)
        'us_start': 17,                 # 17:00 UTC (NY 12:00/13:00 EST/EDT)
        'us_end': 21,                   # 21:00 UTC (NY 16:00/17:00 EST/EDT)
        'off_hours_start': 21,          # 21:00 UTC
        'off_hours_end': 24,            # 24:00 UTC
    },
    'session_size_multipliers': {
        'ASIA': 0.8,                    # Smaller sizes
        'EUROPE': 1.0,                  # Standard sizes
        'US': 1.0,                      # Standard sizes
        'US_EUROPE_OVERLAP': 1.1,       # Larger sizes (peak liquidity)
        'OFF_HOURS': 0.5,               # Smallest sizes (thin liquidity)
    },

    # ==========================================================================
    # Circuit Breaker
    # Stricter for contrarian strategy (prone to consecutive losses in trends)
    # ==========================================================================
    'use_circuit_breaker': True,
    'max_consecutive_losses': 2,        # Stricter for contrarian
    'circuit_breaker_minutes': 45,      # Longer cooldown

    # ==========================================================================
    # Cross-Pair Correlation Management
    # ==========================================================================
    'use_correlation_management': True,
    'max_total_long_exposure': 100.0,   # Max total long USD exposure
    'max_total_short_exposure': 75.0,   # Lower short exposure (squeeze risk)
    'same_direction_size_mult': 0.75,   # Reduce size if both pairs same direction
    'use_real_correlation': True,       # Calculate real-time rolling correlation
    'correlation_window': 20,           # Candles for correlation calculation
    'correlation_block_threshold': 0.85,  # Block if correlation > this

    # ==========================================================================
    # Fee Profitability
    # ==========================================================================
    'fee_rate': 0.001,                  # 0.1% per trade
    'min_profit_after_fees_pct': 0.1,   # Minimum profit after fees
    'use_fee_check': True,

    # ==========================================================================
    # Trade Flow Confirmation (REC-001)
    # ==========================================================================
    'use_trade_flow_confirmation': True,
    'trade_flow_threshold': 0.10,       # Min imbalance to confirm signal (10%)
    'trade_flow_lookback': 50,          # Number of recent trades to analyze

    # ==========================================================================
    # Signal Tracking
    # ==========================================================================
    'track_rejections': True,

    # ==========================================================================
    # REC-011: Candle Data Persistence
    # Saves candle data to disk for fast restart recovery
    # ==========================================================================
    'use_candle_persistence': True,
    'candle_persistence_dir': 'data/candles',  # Directory for candle files
    'candle_save_interval_candles': 1,         # Save every N new candles
    'max_candle_age_hours': 4.0,               # Max age for reloaded data
    'candle_file_format': '{symbol}_5m.json',  # Filename format

    # ==========================================================================
    # REC-016: XRP/BTC Re-enablement Guard
    # Requires explicit flag to enable XRP/BTC trading
    # ==========================================================================
    'enable_xrpbtc': False,  # Must be explicitly set to True to trade XRP/BTC

    # ==========================================================================
    # REC-017: Timezone Validation
    # Validates server timezone on startup
    # ==========================================================================
    'require_utc_timezone': True,   # Warn if server not in UTC
    'timezone_warning_only': True,  # False = block trading if not UTC
}


# =============================================================================
# Per-Symbol Configurations
# Based on research from master-plan-v1.0.md pair-specific analysis
# =============================================================================
SYMBOL_CONFIGS: Dict[str, Dict[str, Any]] = {
    # ==========================================================================
    # XRP/USDT Configuration
    # Research: 5.1% daily volatility, good for contrarian plays
    # Whale Threshold: 1M+ XRP (~$2M at current prices)
    # Suitability: HIGH
    # ==========================================================================
    'XRP/USDT': {
        'volume_spike_mult': 2.0,       # Standard threshold
        'volume_window': 288,           # REC-019: 24h in 5m candles (per-symbol configurable)
        'rsi_extreme_fear': 25,
        'rsi_extreme_greed': 75,
        'fear_deviation_pct': -5.0,
        'greed_deviation_pct': 5.0,
        'position_size_usd': 25.0,
        'max_position_per_symbol_usd': 75.0,
        'stop_loss_pct': 2.5,           # Wider for contrarian
        'take_profit_pct': 5.0,         # 2:1 R:R ratio
        'cooldown_seconds': 120,        # 2 minutes
    },

    # ==========================================================================
    # BTC/USDT Configuration
    # Research: Lower % volatility (1.64%), institutional dampening
    # Whale Threshold: 100+ BTC (~$8M+ at current prices)
    # Suitability: MEDIUM-HIGH
    # ==========================================================================
    'BTC/USDT': {
        'volume_spike_mult': 2.5,       # Higher threshold (more noise)
        'volume_window': 288,           # REC-019: 24h in 5m candles (per-symbol configurable)
        'rsi_extreme_fear': 22,         # More extreme fear required
        'rsi_extreme_greed': 78,        # More extreme greed required
        'fear_deviation_pct': -7.0,     # Larger deviation required
        'greed_deviation_pct': 7.0,
        'position_size_usd': 50.0,      # Larger due to lower volatility
        'max_position_per_symbol_usd': 100.0,
        'stop_loss_pct': 1.5,           # Tighter - more predictable moves
        'take_profit_pct': 3.0,         # 2:1 R:R ratio
        'cooldown_seconds': 180,        # 3 minutes (less frequent)
    },

    # ==========================================================================
    # XRP/BTC Configuration
    # Research: 7-10x lower liquidity than USD pairs
    # Whale Threshold: 500K+ XRP equivalent
    # Suitability: MEDIUM (approach cautiously)
    # REC-016: Requires enable_xrpbtc: true to activate
    # ==========================================================================
    'XRP/BTC': {
        'volume_spike_mult': 3.0,       # Higher threshold (low liquidity)
        'volume_window': 288,           # REC-019: 24h in 5m candles (per-symbol configurable)
        'rsi_extreme_fear': 28,
        'rsi_extreme_greed': 72,
        'fear_deviation_pct': -8.0,     # Larger moves in ratio pair
        'greed_deviation_pct': 8.0,
        'position_size_usd': 15.0,      # Smaller due to liquidity
        'max_position_per_symbol_usd': 40.0,
        'stop_loss_pct': 3.0,           # Wider due to volatility
        'take_profit_pct': 6.0,         # 2:1 R:R ratio
        'cooldown_seconds': 240,        # 4 minutes
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
