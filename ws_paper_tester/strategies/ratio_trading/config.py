"""
Ratio Trading Strategy - Configuration Module

Contains strategy metadata, configuration constants, and validation.
"""
from typing import Dict, Any, List


# =============================================================================
# Strategy Metadata
# =============================================================================
STRATEGY_NAME = "ratio_trading"
STRATEGY_VERSION = "4.3.1"
SYMBOLS = ["XRP/BTC"]


# =============================================================================
# Default Configuration
# =============================================================================
CONFIG = {
    # ==========================================================================
    # Core Ratio Trading Parameters - REC-013: Higher entry threshold
    # ==========================================================================
    'lookback_periods': 20,           # Periods for moving average
    'bollinger_std': 2.0,             # Standard deviations for bands
    'entry_threshold': 1.5,           # Entry at N std devs from mean (was 1.0)
    'exit_threshold': 0.5,            # Exit at N std devs (closer to mean)

    # ==========================================================================
    # REC-036: Dynamic Bollinger Settings for Crypto Volatility
    # Research suggests 2.5-3.0 std more appropriate for volatile crypto markets
    # to avoid false signals. However, current mitigations (trend filter, RSI,
    # volatility regimes) may make this unnecessary per review assessment.
    # ==========================================================================
    'use_crypto_bollinger_std': False,  # Enable wider bands for crypto volatility
    'bollinger_std_crypto': 2.5,        # Wider std dev when enabled (2.5-3.0 range)

    # ==========================================================================
    # Position Sizing - REC-002: USD-based sizing
    # ==========================================================================
    'position_size_usd': 15.0,        # Base size per trade in USD
    'max_position_usd': 50.0,         # Maximum position exposure in USD
    'min_trade_size_usd': 5.0,        # Minimum USD per trade

    # ==========================================================================
    # Risk Management - REC-003: Fixed to 1:1 R:R
    # ==========================================================================
    'stop_loss_pct': 0.6,             # Stop loss percentage
    'take_profit_pct': 0.6,           # Take profit percentage (1:1 R:R)

    # ==========================================================================
    # Cooldown Mechanisms
    # ==========================================================================
    'cooldown_seconds': 30.0,         # Minimum time between trades
    'min_candles': 10,                # Minimum candles before trading

    # ==========================================================================
    # Volatility Parameters - REC-004
    # ==========================================================================
    'use_volatility_regimes': True,   # Enable regime-based adjustments
    'volatility_lookback': 20,        # Candles for volatility calculation
    'regime_low_threshold': 0.2,      # Below = LOW regime (tighter for ratio)
    'regime_medium_threshold': 0.5,   # Below = MEDIUM regime
    'regime_high_threshold': 1.0,     # Below = HIGH regime, above = EXTREME
    'regime_extreme_pause': True,     # Pause trading in EXTREME regime

    # ==========================================================================
    # Circuit Breaker - REC-005
    # ==========================================================================
    'use_circuit_breaker': True,      # Enable consecutive loss circuit breaker
    'max_consecutive_losses': 3,      # Max losses before cooldown
    'circuit_breaker_minutes': 15,    # Cooldown after max losses

    # ==========================================================================
    # Spread Monitoring - REC-008
    # ==========================================================================
    'use_spread_filter': True,        # Enable spread filtering
    'max_spread_pct': 0.10,           # Max spread % (XRP/BTC has wider spreads)
    'min_profitability_mult': 0.5,    # TP must exceed spread * this multiplier

    # ==========================================================================
    # Trade Flow Confirmation - REC-010
    # ==========================================================================
    'use_trade_flow_confirmation': False,  # Disabled by default for ratio pairs
    'trade_flow_threshold': 0.10,          # Minimum trade flow alignment

    # ==========================================================================
    # RSI Confirmation Filter - REC-014
    # ==========================================================================
    'use_rsi_confirmation': True,     # Enable RSI filter
    'rsi_period': 14,                 # RSI calculation period
    'rsi_oversold': 35,               # RSI oversold level for buy confirmation
    'rsi_overbought': 65,             # RSI overbought level for sell confirmation

    # ==========================================================================
    # Trend Detection Warning - REC-015
    # ==========================================================================
    'use_trend_filter': True,         # Enable trend filtering
    'trend_lookback': 10,             # Candles to check for trend
    'trend_strength_threshold': 0.7,  # % of candles in same direction = strong trend

    # ==========================================================================
    # Trailing Stops (from mean reversion patterns)
    # ==========================================================================
    'use_trailing_stop': True,        # Enable trailing stops
    'trailing_activation_pct': 0.3,   # Activate at 0.3% profit
    'trailing_distance_pct': 0.2,     # Trail 0.2% from high/low

    # ==========================================================================
    # Position Decay (from mean reversion patterns)
    # v4.3.0: Increased decay_minutes from 5 to 10 per review v9.0 recommendation
    # Research suggests allowing more time for mean reversion in crypto pairs
    # ==========================================================================
    'use_position_decay': True,       # Enable position decay
    'position_decay_minutes': 10,     # Start decay after 10 minutes (was 5, review v9.0)
    'position_decay_tp_mult': 0.5,    # Reduce TP target to 50% after decay

    # ==========================================================================
    # Rejection Tracking
    # ==========================================================================
    'track_rejections': True,         # Enable rejection tracking

    # ==========================================================================
    # Correlation Monitoring - REC-021
    # v4.3.0: Raised warning threshold to 0.7 per review v9.0 recommendation
    # Earlier warning given ongoing XRP structural changes (ETF, regulatory clarity)
    # ==========================================================================
    'use_correlation_monitoring': True,   # Enable correlation monitoring
    'correlation_lookback': 20,           # Periods for correlation calculation
    'correlation_warning_threshold': 0.7, # v4.3.0: Raised to 0.7 (was 0.6) for earlier warning
    'correlation_pause_threshold': 0.4,   # REC-024: Pause trading if below this (raised from 0.3)
    'correlation_pause_enabled': True,    # REC-023: Enabled by default for declining XRP/BTC correlation

    # ==========================================================================
    # Correlation Trend Detection - REC-037
    # ==========================================================================
    'use_correlation_trend_detection': True,   # Enable correlation trend monitoring
    'correlation_trend_lookback': 10,          # Periods for trend calculation
    'correlation_trend_threshold': -0.02,      # Slope threshold for declining trend
    'correlation_trend_level': 0.7,            # Only warn if correlation below this level
    'correlation_trend_pause_enabled': False,  # Optional: pause on declining trend (conservative)

    # ==========================================================================
    # Dynamic BTC Price - REC-018
    # ==========================================================================
    'btc_price_fallback': 100000.0,   # Fallback BTC/USD price if unavailable
    'btc_price_symbols': ['BTC/USDT', 'BTC/USD'],  # Symbols to check for BTC price

    # ==========================================================================
    # Hedge Ratio - REC-022 (Future Enhancement)
    # ==========================================================================
    'use_hedge_ratio': False,         # Enable hedge ratio optimization
    'hedge_ratio_lookback': 50,       # Periods for hedge ratio calculation

    # ==========================================================================
    # Fee Profitability Check - REC-050 (v4.3.0)
    # Explicit fee check to complement spread filter per review v9.0
    # ==========================================================================
    'use_fee_profitability_check': True,  # Enable explicit fee check
    'estimated_fee_rate': 0.0026,         # Kraken XRP/BTC taker fee (0.26%)
    'min_net_profit_pct': 0.10,           # Minimum net profit after round-trip fees

    # ==========================================================================
    # Rebalancing (for future implementation)
    # ==========================================================================
    'rebalance_threshold': 0.3,       # Rebalance when holdings differ by 30%
}


def validate_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate configuration on startup.

    Returns:
        List of error/warning messages (empty if valid)
    """
    errors = []

    # Required positive values
    required_positive = [
        'position_size_usd', 'max_position_usd', 'stop_loss_pct',
        'take_profit_pct', 'lookback_periods', 'cooldown_seconds',
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
    entry_threshold = config.get('entry_threshold', 1.0)
    if entry_threshold < 0.5 or entry_threshold > 3.0:
        errors.append(f"entry_threshold should be 0.5-3.0, got {entry_threshold}")

    exit_threshold = config.get('exit_threshold', 0.5)
    if exit_threshold < 0.1 or exit_threshold > 2.0:
        errors.append(f"exit_threshold should be 0.1-2.0, got {exit_threshold}")

    if exit_threshold >= entry_threshold:
        errors.append(f"exit_threshold ({exit_threshold}) should be < entry_threshold ({entry_threshold})")

    # R:R ratio check
    sl = config.get('stop_loss_pct', 0.6)
    tp = config.get('take_profit_pct', 0.6)
    if sl > 0 and tp > 0:
        rr_ratio = tp / sl
        if rr_ratio < 1.0:
            errors.append(f"Warning: R:R ratio ({rr_ratio:.2f}:1) < 1:1 - unfavorable")
        elif rr_ratio >= 1.0:
            # Info message for acceptable R:R
            pass

    # Spread filter check
    max_spread = config.get('max_spread_pct', 0.10)
    if max_spread > tp * 0.5:
        errors.append(f"Warning: max_spread_pct ({max_spread}%) > 50% of take_profit_pct ({tp}%)")

    return errors
