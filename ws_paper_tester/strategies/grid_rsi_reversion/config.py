"""
Grid RSI Reversion Strategy - Configuration and Enums

Contains strategy metadata, enums for type safety, and default configuration.
Based on research from master-plan-v1.0.md.

Strategy Concept:
Grid RSI Reversion combines grid trading mechanics with RSI-based mean reversion
signals. Grid levels provide primary entry signals, while RSI acts as a confidence
modifier to enhance signal quality and position sizing.

Key Differentiators:
- Grid-based entries at predetermined price levels
- RSI confidence modifier (not hard filter)
- Cycle-based position management (buy/sell pairs)
- Multi-position tracking across grid levels
- Geometric grid spacing for crypto volatility
"""
from enum import Enum, auto
from typing import Dict, Any


# =============================================================================
# Strategy Metadata
# =============================================================================
STRATEGY_NAME = "grid_rsi_reversion"
STRATEGY_VERSION = "1.2.0"
SYMBOLS = ["XRP/USDT", "BTC/USDT", "XRP/BTC"]


# =============================================================================
# Enums for Type Safety
# =============================================================================
class GridType(Enum):
    """Grid spacing type."""
    ARITHMETIC = auto()   # Fixed dollar/price spacing
    GEOMETRIC = auto()    # Fixed percentage spacing


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
    WARMING_UP = "warming_up"
    REGIME_PAUSE = "regime_pause"
    TREND_FILTER = "trend_filter"
    MAX_ACCUMULATION = "max_accumulation"
    MAX_POSITION = "max_position"
    NO_GRID_LEVELS = "no_grid_levels"
    GRID_LEVEL_FILLED = "grid_level_filled"
    PRICE_NOT_AT_LEVEL = "price_not_at_level"
    RSI_NEUTRAL = "rsi_neutral"
    COOLDOWN = "cooldown"
    CIRCUIT_BREAKER = "circuit_breaker"
    CORRELATION_LIMIT = "correlation_limit"
    NO_SIGNAL_CONDITIONS = "no_signal_conditions"
    # REC-003: Trade flow rejection reasons
    FLOW_AGAINST_TRADE = "flow_against_trade"
    LOW_VOLUME = "low_volume"
    # REC-006: Liquidity rejection
    LOW_LIQUIDITY = "low_liquidity"


class GridLevelStatus(Enum):
    """Status of a grid level."""
    UNFILLED = auto()    # Order not yet executed
    FILLED = auto()      # Order executed, waiting for matching order
    COMPLETED = auto()   # Full cycle completed (buy + sell)


class RSIZone(Enum):
    """RSI zone classification."""
    OVERSOLD = auto()
    NEUTRAL = auto()
    OVERBOUGHT = auto()


# =============================================================================
# Default Configuration
# =============================================================================
CONFIG: Dict[str, Any] = {
    # ==========================================================================
    # Grid Settings
    # ==========================================================================
    'grid_type': 'geometric',           # 'arithmetic' or 'geometric'
    'num_grids': 15,                    # Grid levels per side
    'grid_spacing_pct': 1.5,            # Spacing between levels (%)
    'range_pct': 7.5,                   # Range from center (±%)
    'recenter_after_cycles': 5,         # Recenter grid after N completed cycles
    'min_recenter_interval': 3600,      # Minimum seconds between recenters (1 hour)
    'slippage_tolerance_pct': 0.5,      # Price tolerance for grid level matching
    # REC-008: Trend check before recentering
    # REC-010: Aligned threshold to match main trend filter (adx_threshold: 30)
    'check_trend_before_recenter': True,  # Check ADX before recentering
    'adx_recenter_threshold': 30,       # Don't recenter if ADX > this (trending)

    # ==========================================================================
    # RSI Settings
    # ==========================================================================
    'rsi_period': 14,                   # RSI calculation period
    'rsi_oversold': 30,                 # RSI oversold threshold
    'rsi_overbought': 70,               # RSI overbought threshold
    'use_adaptive_rsi': True,           # Adjust RSI zones by volatility
    'rsi_zone_expansion': 5,            # Max RSI zone expansion by ATR
    'rsi_mode': 'confidence_only',      # 'confidence_only' or 'filter'
    'rsi_extreme_multiplier': 1.3,      # Size multiplier at RSI extremes

    # ==========================================================================
    # ATR Settings
    # ==========================================================================
    'atr_period': 14,                   # ATR calculation period
    'use_atr_spacing': True,            # Use ATR for dynamic grid spacing
    'atr_multiplier': 0.3,              # ATR multiplier for spacing

    # ==========================================================================
    # Position Sizing
    # ==========================================================================
    'position_size_usd': 20.0,          # Base size per grid level (USD)
    'max_position_usd': 100.0,          # Max total position per symbol
    'max_accumulation_levels': 5,       # Max filled grid levels before pause
    'min_trade_size_usd': 5.0,          # Minimum trade size

    # ==========================================================================
    # Risk Management
    # REC-004: Widened stop-loss from 3% to 8% based on research showing
    # grid strategies need 10-15% stops to avoid premature exits during
    # normal crypto volatility
    # ==========================================================================
    'stop_loss_pct': 8.0,               # Stop loss below lowest grid level (REC-004)
    'max_drawdown_pct': 10.0,           # Max portfolio drawdown
    'use_trend_filter': True,           # Pause in strong trends
    'adx_threshold': 30,                # ADX threshold for trend filter
    'adx_period': 14,                   # ADX calculation period

    # ==========================================================================
    # Fees
    # ==========================================================================
    'fee_rate': 0.001,                  # 0.1% per trade

    # ==========================================================================
    # Cooldowns
    # ==========================================================================
    'cooldown_seconds': 60.0,           # Min time between signals
    'use_circuit_breaker': True,
    'max_consecutive_losses': 3,
    'circuit_breaker_minutes': 15,

    # ==========================================================================
    # Volatility Regime Classification
    # ==========================================================================
    'use_volatility_regimes': True,
    'regime_low_threshold': 0.3,        # Below = LOW regime
    'regime_medium_threshold': 0.8,     # Below = MEDIUM regime
    'regime_high_threshold': 1.5,       # Below = HIGH regime, above = EXTREME
    'regime_extreme_pause': True,       # Pause in EXTREME regime

    # ==========================================================================
    # Cross-Pair Correlation Management (REC-005)
    # ==========================================================================
    'use_correlation_management': True,
    'max_total_long_exposure': 150.0,   # Max total long USD exposure
    'same_direction_size_mult': 0.75,   # Reduce size if multiple pairs same direction
    # REC-005: Real correlation monitoring
    'use_real_correlation': True,       # Calculate actual correlation vs position proxy
    'correlation_block_threshold': 0.85, # Block same-direction when correlation > this
    'correlation_lookback': 20,         # Periods for rolling correlation

    # ==========================================================================
    # Trade Flow Confirmation (REC-003)
    # ==========================================================================
    'use_trade_flow_confirmation': True,   # REC-003: Enable trade flow checks
    'min_volume_ratio': 0.8,               # Minimum volume vs average to trade
    'flow_confirmation_threshold': 0.2,    # Flow imbalance threshold for rejection

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
    # Research: High liquidity, 1.76% daily volatility, 0.15% spread
    # Suitability: HIGH for Grid RSI Reversion
    # REC-004: stop_loss_pct 5% for XRP (moderate volatility)
    # ==========================================================================
    'XRP/USDT': {
        'grid_type': 'geometric',
        'num_grids': 15,
        'grid_spacing_pct': 1.5,
        'range_pct': 7.5,
        'position_size_usd': 25.0,
        'max_position_usd': 100.0,
        'max_accumulation_levels': 5,
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        'stop_loss_pct': 5.0,           # REC-004: Wider than default for crypto
    },

    # ==========================================================================
    # BTC/USDT Configuration
    # Research: Deepest liquidity, 12-18% monthly volatility, institutional dominated
    # Suitability: MEDIUM-HIGH (requires wider range, conservative RSI)
    # REC-004: stop_loss_pct 10% for BTC (higher volatility swings)
    # REC-009: grid_spacing_pct increased from 1.0% to 1.5% for better R:R ratio
    #          Old R:R = 1.0/10.0 = 0.10:1, New R:R = 1.5/10.0 = 0.15:1
    # ==========================================================================
    'BTC/USDT': {
        'grid_type': 'arithmetic',      # Works well for established ranges
        'num_grids': 20,
        'grid_spacing_pct': 1.5,        # REC-009: Wider for better R:R ratio
        'range_pct': 10.0,
        'position_size_usd': 50.0,
        'max_position_usd': 150.0,
        'max_accumulation_levels': 4,
        'rsi_oversold': 35,             # Relaxed (BTC trends)
        'rsi_overbought': 65,           # Relaxed
        'stop_loss_pct': 10.0,          # REC-004: Wider for BTC volatility
    },

    # ==========================================================================
    # XRP/BTC Configuration
    # Research: 7-10x lower liquidity, XRP 1.55x more volatile than BTC
    # Suitability: MEDIUM (requires wider spacing, smaller positions)
    # REC-004: stop_loss_pct 8% for cross-pair ratio volatility
    # REC-006: min_volume_usd for liquidity validation
    # ==========================================================================
    'XRP/BTC': {
        'grid_type': 'geometric',
        'num_grids': 10,                # Fewer levels due to liquidity
        'grid_spacing_pct': 2.5,        # REC-006: Wider to account for slippage
        'range_pct': 10.0,
        'position_size_usd': 10.0,      # REC-006: Smaller due to liquidity
        'max_position_usd': 60.0,
        'max_accumulation_levels': 2,   # REC-006: Very conservative
        'rsi_oversold': 25,             # More aggressive (ratio moves)
        'rsi_overbought': 75,
        'cooldown_seconds': 180.0,      # REC-006: Longer cooldown
        'stop_loss_pct': 8.0,           # REC-004: Wider for ratio volatility
        'min_volume_usd': 100_000_000,  # REC-006: $100M daily minimum
    },
}


def get_symbol_config(symbol: str, config: Dict[str, Any], key: str) -> Any:
    """
    Get symbol-specific config value or fall back to global config.

    Args:
        symbol: Trading symbol (e.g., 'XRP/USDT')
        config: Global configuration dict
        key: Configuration key to retrieve

    Returns:
        Symbol-specific value if exists, else global value
    """
    symbol_cfg = SYMBOL_CONFIGS.get(symbol, {})
    return symbol_cfg.get(key, config.get(key))


def get_grid_type(symbol: str, config: Dict[str, Any]) -> GridType:
    """
    Get the grid type for a symbol.

    Args:
        symbol: Trading symbol
        config: Global configuration dict

    Returns:
        GridType enum value
    """
    grid_type_str = get_symbol_config(symbol, config, 'grid_type')
    if grid_type_str == 'arithmetic':
        return GridType.ARITHMETIC
    return GridType.GEOMETRIC
