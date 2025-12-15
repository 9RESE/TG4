"""
WaveTrend Oscillator Strategy - Configuration Validation

Contains validation functions for strategy configuration.
"""
from typing import Dict, Any, List


def validate_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate strategy configuration and return list of warnings/errors.

    Args:
        config: Strategy configuration to validate

    Returns:
        List of warning/error messages (empty if valid)
    """
    errors = []

    # WaveTrend indicator validation
    channel_length = config.get('wt_channel_length', 10)
    average_length = config.get('wt_average_length', 21)
    ma_length = config.get('wt_ma_length', 4)

    if channel_length < 5:
        errors.append(f"wt_channel_length ({channel_length}) is very low, recommend >= 5")
    if average_length < channel_length:
        errors.append(f"wt_average_length ({average_length}) should be >= wt_channel_length ({channel_length})")
    if ma_length < 2:
        errors.append(f"wt_ma_length ({ma_length}) must be >= 2")

    # Zone threshold validation
    overbought = config.get('wt_overbought', 60)
    oversold = config.get('wt_oversold', -60)
    extreme_ob = config.get('wt_extreme_overbought', 80)
    extreme_os = config.get('wt_extreme_oversold', -80)

    if overbought <= 0:
        errors.append(f"wt_overbought ({overbought}) must be positive")
    if oversold >= 0:
        errors.append(f"wt_oversold ({oversold}) must be negative")
    if extreme_ob <= overbought:
        errors.append(f"wt_extreme_overbought ({extreme_ob}) must be > wt_overbought ({overbought})")
    if extreme_os >= oversold:
        errors.append(f"wt_extreme_oversold ({extreme_os}) must be < wt_oversold ({oversold})")

    # Position sizing validation
    position_size = config.get('position_size_usd', 25.0)
    max_position = config.get('max_position_usd', 75.0)
    max_per_symbol = config.get('max_position_per_symbol_usd', 50.0)
    min_trade = config.get('min_trade_size_usd', 5.0)

    if position_size < min_trade:
        errors.append(f"position_size_usd ({position_size}) must be >= min_trade_size_usd ({min_trade})")
    if max_per_symbol > max_position:
        errors.append(f"max_position_per_symbol_usd ({max_per_symbol}) should not exceed max_position_usd ({max_position})")

    # Risk management validation - REC-006: Make R:R validation blocking
    stop_loss = config.get('stop_loss_pct', 1.5)
    take_profit = config.get('take_profit_pct', 3.0)

    if stop_loss <= 0:
        errors.append(f"BLOCKING: stop_loss_pct ({stop_loss}) must be positive")
    if take_profit <= 0:
        errors.append(f"BLOCKING: take_profit_pct ({take_profit}) must be positive")

    # R:R ratio validation - minimum 1:1 required per Strategy Development Guide
    if stop_loss > 0 and take_profit > 0:
        rr_ratio = take_profit / stop_loss
        if rr_ratio < 1.0:
            errors.append(
                f"BLOCKING: R:R ratio {rr_ratio:.2f}:1 below minimum 1:1. "
                f"take_profit_pct ({take_profit}) must be >= stop_loss_pct ({stop_loss})"
            )

    # Fee validation
    fee_rate = config.get('fee_rate', 0.001)
    if fee_rate < 0:
        errors.append(f"fee_rate ({fee_rate}) must be non-negative")
    if fee_rate > 0.01:
        errors.append(f"fee_rate ({fee_rate}) seems high, should typically be < 0.01")

    # Confidence caps validation
    max_long = config.get('max_long_confidence', 0.92)
    max_short = config.get('max_short_confidence', 0.88)

    if max_long > 1.0:
        errors.append(f"max_long_confidence ({max_long}) must be <= 1.0")
    if max_short > 1.0:
        errors.append(f"max_short_confidence ({max_short}) must be <= 1.0")

    # Candle buffer validation
    min_candles = config.get('min_candle_buffer', 50)
    required = max(channel_length, average_length, config.get('divergence_lookback', 14) * 2) + 10

    if min_candles < required:
        errors.append(f"min_candle_buffer ({min_candles}) should be >= {required} based on indicator settings")

    return errors


def validate_symbol_configs(symbol_configs: Dict[str, Dict[str, Any]], global_config: Dict[str, Any]) -> List[str]:
    """
    Validate per-symbol configurations including R:R ratios.

    REC-006: Ensure all symbol-specific R:R ratios meet minimum 1:1 requirement.

    Args:
        symbol_configs: Per-symbol configuration overrides
        global_config: Global configuration for fallback values

    Returns:
        List of warning/error messages (empty if valid)
    """
    errors = []

    for symbol, sym_config in symbol_configs.items():
        # Get R:R parameters with fallback to global
        sl = sym_config.get('stop_loss_pct', global_config.get('stop_loss_pct', 1.5))
        tp = sym_config.get('take_profit_pct', global_config.get('take_profit_pct', 3.0))

        if sl > 0 and tp > 0:
            rr_ratio = tp / sl
            if rr_ratio < 1.0:
                errors.append(
                    f"BLOCKING: {symbol} R:R ratio {rr_ratio:.2f}:1 below minimum 1:1. "
                    f"TP={tp}% / SL={sl}%"
                )
            else:
                # Informational - log achieved R:R
                pass  # All good

    return errors


def validate_config_overrides(overrides: Dict[str, Any]) -> List[str]:
    """
    Validate configuration overrides.

    Args:
        overrides: Configuration overrides to validate

    Returns:
        List of warning/error messages (empty if valid)
    """
    errors = []

    # Check for unknown keys
    known_keys = {
        'wt_channel_length', 'wt_average_length', 'wt_ma_length',
        'wt_overbought', 'wt_oversold', 'wt_extreme_overbought', 'wt_extreme_oversold',
        'require_zone_exit', 'use_divergence', 'divergence_lookback',
        'position_size_usd', 'max_position_usd', 'max_position_per_symbol_usd',
        'min_trade_size_usd', 'short_size_multiplier',
        'stop_loss_pct', 'take_profit_pct',
        'max_long_confidence', 'max_short_confidence',
        'min_candle_buffer', 'cooldown_seconds',
        'use_session_awareness', 'session_boundaries', 'session_size_multipliers',
        'use_circuit_breaker', 'max_consecutive_losses', 'circuit_breaker_minutes',
        'use_correlation_management', 'max_total_long_exposure', 'max_total_short_exposure',
        'same_direction_size_mult',
        'fee_rate', 'min_profit_after_fees_pct', 'use_fee_check',
        'track_rejections',
        # REC-001: Trade flow confirmation
        'use_trade_flow_confirmation', 'trade_flow_threshold', 'trade_flow_lookback',
        # REC-002: Real correlation monitoring
        'use_real_correlation', 'correlation_window', 'correlation_block_threshold',
    }

    for key in overrides:
        if key not in known_keys:
            errors.append(f"Unknown configuration key: {key}")

    return errors
