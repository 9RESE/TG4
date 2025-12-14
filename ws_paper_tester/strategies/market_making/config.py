"""
Market Making Strategy - Configuration

Strategy metadata, default configuration, and per-symbol overrides.

Version History:
v2.2.0 (2025-12-14) - Session Awareness & Correlation Monitoring:
- REC-002: Session awareness with time-of-day adjustments (Guide v2.0 Section 20)
- REC-003: XRP/BTC correlation monitoring with pause thresholds (Guide v2.0 Section 24)

v2.1.0 (2025-12-14) - Deep Review v3.0 Implementation:
- REC-001: Added indicator population on early returns (signals.py)
- REC-004: Raised BTC/USDT min_spread_pct from 0.03% to 0.05%

v2.0.0 Changes (MM-C01, MM-H01, MM-H02, MM-M01):
- Added circuit breaker protection (Guide v2.0 Section 16)
- Added volatility regime classification and EXTREME pause (Guide v2.0 Section 15)
- Added trending market filter (Guide v2.0 research)
- Added signal rejection tracking (Guide v2.0 Section 17)
"""
from typing import Dict, Any, List
from enum import Enum


# =============================================================================
# Strategy Metadata
# =============================================================================
STRATEGY_NAME = "market_making"
STRATEGY_VERSION = "2.2.0"
SYMBOLS = ["XRP/USDT", "BTC/USDT", "XRP/BTC"]


# =============================================================================
# Enums for Volatility Regime and Signal Rejection (v2.0.0)
# =============================================================================
class VolatilityRegime(Enum):
    """Volatility regime classification (MM-H01, Guide v2.0 Section 15)."""
    LOW = "low"           # < 0.3% volatility
    MEDIUM = "medium"     # 0.3% - 0.8%
    HIGH = "high"         # 0.8% - 1.5%
    EXTREME = "extreme"   # > 1.5% - PAUSE TRADING


class RejectionReason(Enum):
    """Signal rejection reasons for tracking (MM-M01, Guide v2.0 Section 17)."""
    NO_ORDERBOOK = "no_orderbook"
    NO_PRICE = "no_price"
    SPREAD_TOO_NARROW = "spread_too_narrow"
    FEE_UNPROFITABLE = "fee_unprofitable"
    TIME_COOLDOWN = "time_cooldown"
    MAX_POSITION = "max_position"
    INSUFFICIENT_SIZE = "insufficient_size"
    TRADE_FLOW_MISALIGNED = "trade_flow_misaligned"
    CIRCUIT_BREAKER = "circuit_breaker"
    EXTREME_VOLATILITY = "extreme_volatility"
    TRENDING_MARKET = "trending_market"
    LOW_CORRELATION = "low_correlation"  # REC-003: v2.2.0


class TradingSession(Enum):
    """Trading session classification (REC-002, Guide v2.0 Section 20)."""
    ASIA = "asia"                    # 00:00-08:00 UTC - Lower liquidity
    EUROPE = "europe"                # 08:00-14:00 UTC - Moderate
    US_EUROPE_OVERLAP = "overlap"    # 14:00-17:00 UTC - Highest activity
    US = "us"                        # 17:00-22:00 UTC - High volume
    OFF_HOURS = "off_hours"          # 22:00-00:00 UTC - Low liquidity


# =============================================================================
# Default Configuration
# =============================================================================
CONFIG = {
    # General settings
    'min_spread_pct': 0.1,        # Minimum spread to trade (0.1%)
    'position_size_usd': 20,      # Size per trade in USD
    'max_inventory': 100,         # Max position size in USD
    'inventory_skew': 0.5,        # Reduce size when inventory builds
    'imbalance_threshold': 0.1,   # Orderbook imbalance to trigger

    # Risk management (MM-009: improved R:R to 1:1)
    'take_profit_pct': 0.5,       # Take profit at 0.5% (was 0.4%)
    'stop_loss_pct': 0.5,         # Stop loss at 0.5%

    # Signal control (MM-003: cooldown)
    'cooldown_seconds': 5.0,      # Minimum time between signals

    # Volatility adjustment (MM-002)
    'base_volatility_pct': 0.5,   # Baseline volatility for scaling
    'volatility_lookback': 20,    # Candles for volatility calculation
    'volatility_threshold_mult': 1.5,  # Max multiplier in high volatility

    # Trade flow confirmation (MM-007)
    'use_trade_flow': True,       # Confirm with trade tape
    'trade_flow_threshold': 0.15, # Minimum trade imbalance alignment

    # v1.4.0: Avellaneda-Stoikov reservation price model (optional)
    'use_reservation_price': False,  # Enable A-S style quote adjustment
    'gamma': 0.1,                    # Risk aversion parameter (0.01-1.0)
    # Higher gamma = more aggressive inventory reduction

    # v1.4.0: Trailing stop support
    'use_trailing_stop': False,      # Enable trailing stops
    'trailing_stop_activation': 0.2, # Activate trailing after 0.2% profit
    'trailing_stop_distance': 0.15,  # Trail at 0.15% from high

    # v1.5.0: Fee-aware profitability (MM-E03)
    'fee_rate': 0.001,               # 0.1% per trade (0.2% round-trip)
    'min_profit_after_fees_pct': 0.05,  # Minimum profit after fees (0.05%)
    'use_fee_check': True,           # Enable fee-aware profitability check

    # v1.5.0: Micro-price (MM-E01)
    'use_micro_price': True,         # Use volume-weighted micro-price

    # v1.5.0: Optimal spread calculation (MM-E02)
    'use_optimal_spread': False,     # Enable A-S optimal spread calculation
    'kappa': 1.5,                    # Market liquidity parameter

    # v1.5.0: Fallback prices (MM-011)
    'fallback_xrp_usdt': 2.50,       # Fallback XRP/USDT price if unavailable

    # v1.5.0: Position decay (MM-E04)
    'use_position_decay': True,      # Enable time-based position decay
    'max_position_age_seconds': 300, # Max age before widening TP (5 minutes)
    'position_decay_tp_multiplier': 0.5,  # Reduce TP by 50% for stale positions

    # v2.0.0: Circuit breaker protection (MM-C01, Guide v2.0 Section 16)
    'use_circuit_breaker': True,         # Enable circuit breaker
    'max_consecutive_losses': 3,         # Trigger after N consecutive losses
    'circuit_breaker_cooldown_minutes': 15,  # Cooldown period after trigger

    # v2.0.0: Volatility regime (MM-H01, Guide v2.0 Section 15)
    'use_volatility_regime': True,       # Enable volatility regime classification
    'regime_low_threshold': 0.3,         # Below this = LOW regime
    'regime_medium_threshold': 0.8,      # Below this = MEDIUM regime
    'regime_high_threshold': 1.5,        # Below this = HIGH, above = EXTREME
    'regime_extreme_pause': True,        # Pause trading in EXTREME
    'regime_high_size_mult': 0.7,        # Reduce size in HIGH regime
    'regime_low_threshold_mult': 0.9,    # Tighter thresholds in LOW
    'regime_high_threshold_mult': 1.3,   # Wider thresholds in HIGH

    # v2.0.0: Trending market filter (MM-H02)
    'use_trend_filter': True,            # Enable trending market filter
    'trend_slope_threshold': 0.05,       # Skip entries if |slope| > this %
    'trend_lookback_candles': 20,        # Candles for trend calculation
    'trend_confirmation_periods': 3,     # Require N consecutive trending signals

    # v2.2.0: Session awareness (REC-002, Guide v2.0 Section 20)
    'use_session_awareness': True,       # Enable session-based adjustments
    'session_asia_threshold_mult': 1.2,  # Wider thresholds during Asia session
    'session_asia_size_mult': 0.8,       # Smaller size during low liquidity
    'session_overlap_threshold_mult': 0.85,  # Tighter thresholds during overlap
    'session_overlap_size_mult': 1.1,    # Slightly larger size during high activity
    'session_off_hours_threshold_mult': 1.3,  # Very wide thresholds during off-hours
    'session_off_hours_size_mult': 0.6,  # Conservative size during off-hours

    # v2.2.0: Correlation monitoring for XRP/BTC (REC-003, Guide v2.0 Section 24)
    'use_correlation_monitoring': True,  # Enable correlation monitoring
    'correlation_warning_threshold': 0.6,  # Warn when correlation drops below
    'correlation_pause_threshold': 0.5,  # Pause XRP/BTC trading when below this
    'correlation_lookback': 20,          # Candles to use for correlation calculation
}

# Per-symbol configurations (MM-009: adjusted R:R ratios)
SYMBOL_CONFIGS = {
    'XRP/USDT': {
        'min_spread_pct': 0.05,       # Tighter spreads on USDT pair
        'position_size_usd': 20,
        'max_inventory': 100,
        'imbalance_threshold': 0.1,
        'take_profit_pct': 0.5,       # MM-009: changed from 0.4 to 0.5 for 1:1
        'stop_loss_pct': 0.5,
        'cooldown_seconds': 5.0,
    },
    'BTC/USDT': {
        # BTC/USDT market making - high liquidity pair
        # - Very tight spreads, high volume
        # - Larger position sizes (BTC trades bigger)
        # REC-004: Raised min_spread_pct from 0.03% to 0.05% for better
        # profitability after 0.2% round-trip fees
        'min_spread_pct': 0.05,       # REC-004: raised from 0.03 for profitability
        'position_size_usd': 50,      # Larger size for BTC
        'max_inventory': 200,         # Higher max inventory
        'imbalance_threshold': 0.08,  # Lower threshold (more liquid)
        'take_profit_pct': 0.35,      # 1:1 R:R
        'stop_loss_pct': 0.35,
        'cooldown_seconds': 3.0,      # Faster for BTC
    },
    'XRP/BTC': {
        # XRP/BTC market making - optimized from Kraken data:
        # - 664 trades/day, 0.0446% spread
        # - Goal: Accumulate both XRP and BTC through spread capture
        'min_spread_pct': 0.03,       # Lower threshold (spread is 0.0446%)
        'position_size_xrp': 25,      # Size in XRP (converted to USD in signal)
        'max_inventory_xrp': 150,     # Max XRP exposure
        'imbalance_threshold': 0.15,  # Slightly higher (less liquid)
        'take_profit_pct': 0.4,       # MM-009: changed from 0.3 to 0.4 for 1:1
        'stop_loss_pct': 0.4,
        'cooldown_seconds': 10.0,     # Slower for cross-pair
    },
}


# =============================================================================
# Configuration Helpers
# =============================================================================
def get_symbol_config(symbol: str, config: Dict[str, Any], key: str) -> Any:
    """Get symbol-specific config or fall back to global config."""
    symbol_cfg = SYMBOL_CONFIGS.get(symbol, {})
    return symbol_cfg.get(key, config.get(key))


def is_xrp_btc(symbol: str) -> bool:
    """Check if this is the XRP/BTC pair."""
    return symbol == 'XRP/BTC'


def validate_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate configuration on startup.

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    # Required positive values
    required_positive = [
        'position_size_usd',
        'max_inventory',
        'stop_loss_pct',
        'take_profit_pct',
        'cooldown_seconds',
    ]

    for key in required_positive:
        val = config.get(key)
        if val is None:
            errors.append(f"Missing required config: {key}")
        elif val <= 0:
            errors.append(f"{key} must be positive, got {val}")

    # Optional values with bounds
    gamma = config.get('gamma', 0.1)
    if gamma < 0.01 or gamma > 1.0:
        errors.append(f"gamma must be between 0.01 and 1.0, got {gamma}")

    inventory_skew = config.get('inventory_skew', 0.5)
    if inventory_skew < 0 or inventory_skew > 1.0:
        errors.append(f"inventory_skew must be between 0 and 1.0, got {inventory_skew}")

    # Warn about risky R:R ratios
    sl = config.get('stop_loss_pct', 0.5)
    tp = config.get('take_profit_pct', 0.5)
    if sl > 0 and tp > 0:
        rr_ratio = tp / sl
        if rr_ratio < 0.5:
            errors.append(f"Warning: Poor R:R ratio ({rr_ratio:.2f}:1), requires {100/(1+rr_ratio):.0f}% win rate")

    # v1.5.0: Validate fee settings
    fee_rate = config.get('fee_rate', 0.001)
    if fee_rate < 0 or fee_rate > 0.01:
        errors.append(f"fee_rate should be between 0 and 0.01, got {fee_rate}")

    # v2.0.0: Validate circuit breaker settings (MM-C01)
    max_losses = config.get('max_consecutive_losses', 3)
    if max_losses < 1 or max_losses > 10:
        errors.append(f"max_consecutive_losses should be between 1 and 10, got {max_losses}")

    cb_cooldown = config.get('circuit_breaker_cooldown_minutes', 15)
    if cb_cooldown < 1 or cb_cooldown > 60:
        errors.append(f"circuit_breaker_cooldown_minutes should be between 1 and 60, got {cb_cooldown}")

    # v2.0.0: Validate volatility regime thresholds (MM-H01)
    low_thresh = config.get('regime_low_threshold', 0.3)
    med_thresh = config.get('regime_medium_threshold', 0.8)
    high_thresh = config.get('regime_high_threshold', 1.5)
    if not (low_thresh < med_thresh < high_thresh):
        errors.append(f"Regime thresholds must be ordered: low < medium < high ({low_thresh} < {med_thresh} < {high_thresh})")

    # v2.0.0: Validate trend filter settings (MM-H02)
    trend_slope = config.get('trend_slope_threshold', 0.05)
    if trend_slope < 0.01 or trend_slope > 0.5:
        errors.append(f"trend_slope_threshold should be between 0.01 and 0.5, got {trend_slope}")

    # v2.2.0: Validate session awareness settings (REC-002)
    session_mults = [
        ('session_asia_threshold_mult', 0.5, 2.0),
        ('session_asia_size_mult', 0.1, 1.5),
        ('session_overlap_threshold_mult', 0.5, 1.5),
        ('session_overlap_size_mult', 0.5, 2.0),
        ('session_off_hours_threshold_mult', 0.5, 2.0),
        ('session_off_hours_size_mult', 0.1, 1.0),
    ]
    for key, min_val, max_val in session_mults:
        val = config.get(key)
        if val is not None and (val < min_val or val > max_val):
            errors.append(f"{key} should be between {min_val} and {max_val}, got {val}")

    # v2.2.0: Validate correlation monitoring settings (REC-003)
    corr_warn = config.get('correlation_warning_threshold', 0.6)
    corr_pause = config.get('correlation_pause_threshold', 0.5)
    if not (0 <= corr_pause < corr_warn <= 1.0):
        errors.append(f"Correlation thresholds must satisfy: 0 <= pause < warning <= 1.0 (pause={corr_pause}, warning={corr_warn})")

    corr_lookback = config.get('correlation_lookback', 20)
    if corr_lookback < 5 or corr_lookback > 100:
        errors.append(f"correlation_lookback should be between 5 and 100, got {corr_lookback}")

    return errors
