"""
Whale Sentiment Strategy - Configuration Validation

Contains validation functions for strategy configuration.
REC-006: Includes blocking R:R validation for all configurations.
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

    # RSI validation
    rsi_period = config.get('rsi_period', 14)
    rsi_extreme_fear = config.get('rsi_extreme_fear', 25)
    rsi_fear = config.get('rsi_fear', 40)
    rsi_greed = config.get('rsi_greed', 60)
    rsi_extreme_greed = config.get('rsi_extreme_greed', 75)

    if rsi_period < 5:
        errors.append(f"rsi_period ({rsi_period}) is too low, recommend >= 5")
    if rsi_extreme_fear >= rsi_fear:
        errors.append(f"rsi_extreme_fear ({rsi_extreme_fear}) must be < rsi_fear ({rsi_fear})")
    if rsi_fear >= rsi_greed:
        errors.append(f"rsi_fear ({rsi_fear}) must be < rsi_greed ({rsi_greed})")
    if rsi_greed >= rsi_extreme_greed:
        errors.append(f"rsi_greed ({rsi_greed}) must be < rsi_extreme_greed ({rsi_extreme_greed})")
    if rsi_extreme_fear < 0 or rsi_extreme_greed > 100:
        errors.append("RSI thresholds must be between 0 and 100")

    # Volume spike validation
    volume_spike_mult = config.get('volume_spike_mult', 2.0)
    volume_window = config.get('volume_window', 288)

    if volume_spike_mult < 1.0:
        errors.append(f"volume_spike_mult ({volume_spike_mult}) must be >= 1.0")
    if volume_window < 24:
        errors.append(f"volume_window ({volume_window}) should be >= 24 for reliable baseline")

    # Price deviation validation
    fear_deviation = config.get('fear_deviation_pct', -5.0)
    greed_deviation = config.get('greed_deviation_pct', 5.0)

    if fear_deviation >= 0:
        errors.append(f"fear_deviation_pct ({fear_deviation}) must be negative")
    if greed_deviation <= 0:
        errors.append(f"greed_deviation_pct ({greed_deviation}) must be positive")

    # Position sizing validation
    position_size = config.get('position_size_usd', 25.0)
    max_position = config.get('max_position_usd', 150.0)
    max_per_symbol = config.get('max_position_per_symbol_usd', 75.0)
    min_trade = config.get('min_trade_size_usd', 5.0)

    if position_size < min_trade:
        errors.append(f"position_size_usd ({position_size}) must be >= min_trade_size_usd ({min_trade})")
    if max_per_symbol > max_position:
        errors.append(f"max_position_per_symbol_usd ({max_per_symbol}) should not exceed max_position_usd ({max_position})")

    # Risk management validation - REC-006: Make R:R validation blocking
    stop_loss = config.get('stop_loss_pct', 2.5)
    take_profit = config.get('take_profit_pct', 5.0)

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

    # Confidence validation
    min_confidence = config.get('min_confidence', 0.55)
    max_confidence = config.get('max_confidence', 0.90)

    if min_confidence < 0 or min_confidence > 1:
        errors.append(f"min_confidence ({min_confidence}) must be between 0 and 1")
    if max_confidence < min_confidence:
        errors.append(f"max_confidence ({max_confidence}) must be >= min_confidence ({min_confidence})")
    if max_confidence > 1:
        errors.append(f"max_confidence ({max_confidence}) must be <= 1.0")

    # Fee validation
    fee_rate = config.get('fee_rate', 0.001)
    if fee_rate < 0:
        errors.append(f"fee_rate ({fee_rate}) must be non-negative")
    if fee_rate > 0.01:
        errors.append(f"fee_rate ({fee_rate}) seems high, should typically be < 0.01")

    # Candle buffer validation - needs enough for volume window + RSI
    min_candles = config.get('min_candle_buffer', 300)
    required = max(volume_window, rsi_period * 2, config.get('price_lookback', 48)) + 20

    if min_candles < required:
        errors.append(f"min_candle_buffer ({min_candles}) should be >= {required} based on indicator settings")

    # Weight validation - REC-013: RSI and divergence weights should be 0
    weight_total = (
        config.get('weight_volume_spike', 0.55) +
        config.get('weight_rsi_sentiment', 0.00) +
        config.get('weight_price_deviation', 0.35) +
        config.get('weight_trade_flow', 0.10) +
        config.get('weight_divergence', 0.00)
    )
    if abs(weight_total - 1.0) > 0.01:
        errors.append(f"Confidence weights should sum to 1.0, got {weight_total:.2f}")

    # REC-013: Warn if RSI weight is non-zero (academically ineffective)
    rsi_weight = config.get('weight_rsi_sentiment', 0.00)
    if rsi_weight > 0:
        errors.append(f"WARNING: weight_rsi_sentiment ({rsi_weight}) > 0. RSI is academically ineffective in crypto (REC-013).")

    # Circuit breaker validation
    max_losses = config.get('max_consecutive_losses', 2)
    if max_losses < 1:
        errors.append(f"max_consecutive_losses ({max_losses}) must be >= 1")

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
        sl = sym_config.get('stop_loss_pct', global_config.get('stop_loss_pct', 2.5))
        tp = sym_config.get('take_profit_pct', global_config.get('take_profit_pct', 5.0))

        if sl > 0 and tp > 0:
            rr_ratio = tp / sl
            if rr_ratio < 1.0:
                errors.append(
                    f"BLOCKING: {symbol} R:R ratio {rr_ratio:.2f}:1 below minimum 1:1. "
                    f"TP={tp}% / SL={sl}%"
                )

        # Volume spike multiplier validation
        vol_mult = sym_config.get('volume_spike_mult', global_config.get('volume_spike_mult', 2.0))
        if vol_mult < 1.0:
            errors.append(f"{symbol}: volume_spike_mult ({vol_mult}) must be >= 1.0")

        # RSI threshold validation
        fear = sym_config.get('rsi_extreme_fear', global_config.get('rsi_extreme_fear', 25))
        greed = sym_config.get('rsi_extreme_greed', global_config.get('rsi_extreme_greed', 75))
        if fear >= greed:
            errors.append(f"{symbol}: rsi_extreme_fear ({fear}) must be < rsi_extreme_greed ({greed})")

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

    # Known configuration keys
    known_keys = {
        # Whale Detection
        'volume_spike_mult', 'volume_window', 'min_spike_trades',
        'max_spread_pct', 'volume_spike_price_move_pct',
        # RSI Settings (kept for legacy compatibility, weight should be 0)
        'rsi_period', 'rsi_extreme_fear', 'rsi_fear', 'rsi_greed', 'rsi_extreme_greed',
        # Fear/Greed
        'fear_deviation_pct', 'greed_deviation_pct', 'price_lookback',
        # Mode
        'contrarian_mode',
        # Confidence - REC-013: RSI/divergence weights should be 0
        'weight_volume_spike', 'weight_rsi_sentiment', 'weight_price_deviation',
        'weight_trade_flow', 'weight_divergence', 'min_confidence', 'max_confidence',
        # REC-020: Extracted magic numbers
        'volume_confidence_base', 'volume_confidence_bonus_per_ratio',
        # Position Sizing
        'position_size_usd', 'max_position_usd', 'max_position_per_symbol_usd',
        'min_trade_size_usd', 'short_size_multiplier', 'high_correlation_size_mult',
        # Risk Management
        'stop_loss_pct', 'take_profit_pct',
        'use_trailing_stop', 'trailing_stop_activation_pct', 'trailing_stop_distance_pct',
        # Confidence Caps
        'max_long_confidence', 'max_short_confidence',
        # Candles
        'min_candle_buffer',
        # Cooldown
        'cooldown_seconds',
        # Sessions
        'use_session_awareness', 'session_boundaries', 'session_size_multipliers',
        # Circuit Breaker
        'use_circuit_breaker', 'max_consecutive_losses', 'circuit_breaker_minutes',
        # Correlation
        'use_correlation_management', 'max_total_long_exposure', 'max_total_short_exposure',
        'same_direction_size_mult', 'use_real_correlation', 'correlation_window',
        'correlation_block_threshold',
        # Fees
        'fee_rate', 'min_profit_after_fees_pct', 'use_fee_check',
        # Trade Flow
        'use_trade_flow_confirmation', 'trade_flow_threshold', 'trade_flow_lookback',
        # Tracking
        'track_rejections',
        # REC-011: Candle Persistence
        'use_candle_persistence', 'candle_persistence_dir', 'candle_save_interval_candles',
        'max_candle_age_hours', 'candle_file_format',
        # REC-016: XRP/BTC Guard
        'enable_xrpbtc',
        # REC-017: Timezone Validation
        'require_utc_timezone', 'timezone_warning_only',
    }

    for key in overrides:
        if key not in known_keys:
            errors.append(f"Unknown configuration key: {key}")

    return errors
