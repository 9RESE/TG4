"""
Mean Reversion Strategy - Configuration Module

Contains strategy metadata, enums, configuration constants, and validation.
"""
from typing import Dict, Any, List
from enum import Enum, auto


# =============================================================================
# Strategy Metadata
# =============================================================================
STRATEGY_NAME = "mean_reversion"
STRATEGY_VERSION = "4.3.0"
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
    TRENDING_MARKET = "trending_market"  # REC-004 (v3.0.0)
    NO_SIGNAL_CONDITIONS = "no_signal_conditions"
    FEE_UNPROFITABLE = "fee_unprofitable"  # REC-002 (v4.1.0)
    LOW_CORRELATION = "low_correlation"  # REC-001 (v4.2.0) - XRP/BTC correlation pause
    STRONG_TREND_ADX = "strong_trend_adx"  # REC-003 (v4.3.0) - BTC ADX filter


# =============================================================================
# Default Configuration
# =============================================================================
CONFIG = {
    # ==========================================================================
    # Core Mean Reversion Parameters
    # ==========================================================================
    'lookback_candles': 20,           # Candles for MA calculation
    'deviation_threshold': 0.5,       # % deviation to trigger (0.5%)
    'bb_period': 20,                  # Bollinger Bands period
    'bb_std_dev': 2.0,                # Bollinger Bands standard deviations
    'rsi_period': 14,                 # RSI calculation period

    # ==========================================================================
    # Position Sizing
    # ==========================================================================
    'position_size_usd': 20.0,        # Size per trade in USD
    'max_position': 50.0,             # Max position size in USD
    'min_trade_size_usd': 5.0,        # Minimum USD per trade

    # ==========================================================================
    # RSI Thresholds
    # ==========================================================================
    'rsi_oversold': 35,               # RSI oversold level
    'rsi_overbought': 65,             # RSI overbought level

    # ==========================================================================
    # Risk Management - REC-001: Fixed to 1:1 R:R
    # ==========================================================================
    'take_profit_pct': 0.5,           # Take profit at 0.5%
    'stop_loss_pct': 0.5,             # Stop loss at 0.5%

    # ==========================================================================
    # Cooldown Mechanisms - REC-003
    # ==========================================================================
    'cooldown_seconds': 10.0,         # Min time between signals

    # ==========================================================================
    # Volatility Parameters - REC-004
    # ==========================================================================
    'use_volatility_regimes': True,   # Enable regime-based adjustments
    'base_volatility_pct': 0.5,       # Baseline volatility for scaling
    'volatility_lookback': 20,        # Candles for volatility calculation
    'regime_low_threshold': 0.3,      # Below = LOW regime
    'regime_medium_threshold': 0.8,   # Below = MEDIUM regime
    'regime_high_threshold': 1.5,     # Below = HIGH regime, above = EXTREME
    'regime_extreme_pause': True,     # Pause trading in EXTREME regime

    # ==========================================================================
    # Circuit Breaker - REC-005
    # ==========================================================================
    'use_circuit_breaker': True,      # Enable consecutive loss circuit breaker
    'max_consecutive_losses': 3,      # Max losses before cooldown
    'circuit_breaker_minutes': 15,    # Cooldown after max losses

    # ==========================================================================
    # Trade Flow Confirmation - REC-008
    # ==========================================================================
    'use_trade_flow_confirmation': True,
    'trade_flow_threshold': 0.10,     # Minimum trade flow alignment

    # ==========================================================================
    # VWAP Parameters
    # ==========================================================================
    'vwap_lookback': 50,              # Trades for VWAP calculation
    'vwap_deviation_threshold': 0.3,  # % deviation for VWAP signal
    'vwap_size_multiplier': 0.5,      # Position size multiplier for VWAP signals

    # ==========================================================================
    # Trend Filter - REC-004 (v3.0.0), Updated per v4.0 Review REC-003
    # Added confirmation period to reduce false positives in choppy markets
    # ==========================================================================
    'use_trend_filter': True,         # Enable trend filtering
    'trend_sma_period': 50,           # Lookback for trend SMA
    'trend_slope_threshold': 0.05,    # Min slope % to consider trending
    'trend_confirmation_periods': 3,  # Consecutive trending evals before rejection (REC-003)

    # ==========================================================================
    # Trailing Stops - REC-006 (v3.0.0), Updated per v4.0 Review REC-001
    # Research: Fixed TP better for mean reversion than trailing stops
    # Trailing stops designed for trend-following; may exit prematurely during MR
    # ==========================================================================
    'use_trailing_stop': False,       # Disabled by default (per REC-001 research)
    'trailing_activation_pct': 0.4,   # Activate at 0.4% profit (increased for MR)
    'trailing_distance_pct': 0.3,     # Trail 0.3% from high/low (wider per REC-001)

    # ==========================================================================
    # Position Decay - REC-007 (v3.0.0), Updated per v4.0 Review REC-002
    # Research: Crypto mean reversion needs multi-candle periods for completion
    # ==========================================================================
    'use_position_decay': True,       # Enable time-based TP reduction
    'decay_start_minutes': 15.0,      # Start reducing TP after 15 min (was 3, per REC-002)
    'decay_interval_minutes': 5.0,    # Reduce TP every 5 min (was 1, per REC-002)
    'decay_multipliers': [1.0, 0.85, 0.7, 0.5],  # Gentler reduction (per REC-002)

    # ==========================================================================
    # XRP/BTC Correlation Monitoring - REC-005 (v4.0.0), Updated per v8.0 REC-001/002
    # Research: XRP correlation with BTC declined from ~80% to ~40% (Dec 2025)
    # Decoupling confirmed - XRP trading on "own fundamentals"
    # v4.3.0: Raised pause threshold from 0.25 to 0.5 per deep review v8.0
    # ==========================================================================
    'use_correlation_monitoring': True,  # Enable XRP/BTC correlation tracking
    'correlation_lookback': 50,          # Candles for correlation calculation
    'correlation_warn_threshold': 0.55,  # Raised from 0.4 (REC-001 v4.3.0)
    'correlation_pause_threshold': 0.5,  # Raised from 0.25 (REC-001/002 v4.3.0)
    'correlation_pause_enabled': True,   # Enable correlation-based pause

    # ==========================================================================
    # Fee Profitability Checks - REC-002 (v4.1.0)
    # Guide v2.0 Section 23: Validate net profit after fees before signal
    # ==========================================================================
    'check_fee_profitability': True,  # Enable fee profitability validation
    'estimated_fee_rate': 0.001,      # 0.1% per side (typical maker/taker)
    'min_net_profit_pct': 0.05,       # Minimum net profit after round-trip fees

    # ==========================================================================
    # Rejection Tracking
    # ==========================================================================
    'track_rejections': True,         # Enable rejection tracking

    # ==========================================================================
    # ADX Filter - REC-003 (v4.3.0) - Deep Review v8.0
    # Research: BTC exhibits stronger trending behavior than mean reversion
    # "BTC tends to trend when it is at its maximum and bounce back when at the minimum"
    # Pause BTC/USDT entries when ADX indicates strong trend
    # ==========================================================================
    'use_adx_filter': True,           # Enable ADX trend strength filter
    'adx_period': 14,                 # Standard ADX period
    'adx_strong_trend_threshold': 25, # ADX > 25 indicates strong trend
}

# =============================================================================
# Per-Symbol Configurations - REC-002
# =============================================================================
SYMBOL_CONFIGS = {
    'XRP/USDT': {
        'deviation_threshold': 0.5,   # % deviation threshold
        'rsi_oversold': 35,
        'rsi_overbought': 65,
        'position_size_usd': 20.0,
        'max_position': 50.0,
        'take_profit_pct': 0.5,       # 1:1 R:R
        'stop_loss_pct': 0.5,
        'cooldown_seconds': 10.0,
    },
    # REC-001 (v4.1.0): Reduced position size due to unfavorable BTC market conditions
    # BTC in bearish territory (below all EMAs), Fear & Greed at "Extreme Fear" (23)
    # Academic research (SSRN Oct 2024) shows mean reversion less effective in BTC
    'BTC/USDT': {
        'deviation_threshold': 0.3,   # Tighter for lower volatility BTC
        'rsi_oversold': 30,           # More aggressive for efficient market
        'rsi_overbought': 70,
        'position_size_usd': 25.0,    # Reduced from $50 (REC-001 v4.1.0)
        'max_position': 75.0,         # Reduced from $150 proportionally
        'take_profit_pct': 0.4,       # Tighter for BTC
        'stop_loss_pct': 0.4,
        'cooldown_seconds': 5.0,      # Faster for liquid BTC
    },
    # REC-001 (v3.0.0): XRP/BTC ratio trading pair
    'XRP/BTC': {
        'deviation_threshold': 1.0,   # Wider for ratio volatility (1.55x XRP vs BTC)
        'rsi_oversold': 35,           # Conservative for ratio trading
        'rsi_overbought': 65,
        'position_size_usd': 15.0,    # Lower for less liquidity
        'max_position': 40.0,         # Conservative limit
        'take_profit_pct': 0.8,       # Account for wider spreads, 1:1 R:R
        'stop_loss_pct': 0.8,
        'cooldown_seconds': 20.0,     # Slower for ratio trades
    },
}


# =============================================================================
# Configuration Helpers
# =============================================================================
def get_symbol_config(symbol: str, config: Dict[str, Any], key: str) -> Any:
    """Get symbol-specific config or fall back to global config."""
    symbol_cfg = SYMBOL_CONFIGS.get(symbol, {})
    return symbol_cfg.get(key, config.get(key))


def validate_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate configuration on startup.

    Returns:
        List of error/warning messages (empty if valid)
    """
    errors = []

    # Required positive values
    required_positive = [
        'position_size_usd', 'max_position', 'stop_loss_pct',
        'take_profit_pct', 'lookback_candles', 'cooldown_seconds',
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
    deviation = config.get('deviation_threshold', 0.5)
    if deviation < 0.1 or deviation > 2.0:
        errors.append(f"deviation_threshold should be 0.1-2.0, got {deviation}")

    rsi_oversold = config.get('rsi_oversold', 35)
    rsi_overbought = config.get('rsi_overbought', 65)
    if rsi_oversold >= rsi_overbought:
        errors.append(f"rsi_oversold ({rsi_oversold}) must be < rsi_overbought ({rsi_overbought})")
    if rsi_oversold < 10 or rsi_oversold > 50:
        errors.append(f"rsi_oversold should be 10-50, got {rsi_oversold}")
    if rsi_overbought < 50 or rsi_overbought > 90:
        errors.append(f"rsi_overbought should be 50-90, got {rsi_overbought}")

    # R:R ratio check
    sl = config.get('stop_loss_pct', 0.5)
    tp = config.get('take_profit_pct', 0.5)
    if sl > 0 and tp > 0:
        rr_ratio = tp / sl
        if rr_ratio < 1.0:
            errors.append(f"Warning: R:R ratio ({rr_ratio:.2f}:1) < 1:1")
        elif rr_ratio < 1.5:
            errors.append(f"Info: R:R ratio {rr_ratio:.2f}:1 acceptable")

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
