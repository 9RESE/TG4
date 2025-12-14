"""
Market Making Strategy - Configuration

Strategy metadata, default configuration, and per-symbol overrides.
"""
from typing import Dict, Any, List


# =============================================================================
# Strategy Metadata
# =============================================================================
STRATEGY_NAME = "market_making"
STRATEGY_VERSION = "1.5.0"
SYMBOLS = ["XRP/USDT", "BTC/USDT", "XRP/BTC"]


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
        'min_spread_pct': 0.03,       # Tighter min spread (more liquid)
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

    return errors
