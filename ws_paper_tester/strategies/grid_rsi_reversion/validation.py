"""
Grid RSI Reversion Strategy - Configuration Validation

Validates strategy configuration for consistency and safety.
"""
from typing import Dict, Any, List


def validate_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate strategy configuration for consistency and safety.

    Args:
        config: Configuration dict to validate

    Returns:
        List of warning/error messages (empty if valid)
    """
    warnings = []

    # Grid settings validation
    num_grids = config.get('num_grids', 15)
    if num_grids < 5:
        warnings.append(f"num_grids={num_grids} is too low, recommend >= 5")
    if num_grids > 50:
        warnings.append(f"num_grids={num_grids} is high, may spread capital too thin")

    grid_spacing = config.get('grid_spacing_pct', 1.5)
    if grid_spacing < 0.5:
        warnings.append(f"grid_spacing_pct={grid_spacing}% is very tight, may be eaten by fees")
    if grid_spacing > 5.0:
        warnings.append(f"grid_spacing_pct={grid_spacing}% is very wide, may miss opportunities")

    range_pct = config.get('range_pct', 7.5)
    if range_pct < 3.0:
        warnings.append(f"range_pct={range_pct}% is narrow, grid may break out quickly")
    if range_pct > 20.0:
        warnings.append(f"range_pct={range_pct}% is very wide, capital may be tied up")

    # RSI validation
    rsi_period = config.get('rsi_period', 14)
    if rsi_period < 7:
        warnings.append(f"rsi_period={rsi_period} may be too noisy")
    if rsi_period > 21:
        warnings.append(f"rsi_period={rsi_period} may lag significantly")

    rsi_oversold = config.get('rsi_oversold', 30)
    rsi_overbought = config.get('rsi_overbought', 70)
    if rsi_oversold < 15 or rsi_oversold > 40:
        warnings.append(f"rsi_oversold={rsi_oversold} outside typical range (15-40)")
    if rsi_overbought < 60 or rsi_overbought > 85:
        warnings.append(f"rsi_overbought={rsi_overbought} outside typical range (60-85)")
    if rsi_overbought - rsi_oversold < 30:
        warnings.append("RSI bands too narrow, may whipsaw frequently")

    # Position sizing validation
    position_size = config.get('position_size_usd', 20.0)
    max_position = config.get('max_position_usd', 100.0)
    if position_size < 5.0:
        warnings.append(f"position_size_usd={position_size} is very small")
    if position_size > max_position:
        warnings.append(f"position_size_usd ({position_size}) > max_position_usd ({max_position})")

    max_accumulation = config.get('max_accumulation_levels', 5)
    if max_accumulation < 2:
        warnings.append(f"max_accumulation_levels={max_accumulation} limits grid effectiveness")
    if max_accumulation * position_size > max_position:
        warnings.append(
            f"max_accumulation_levels * position_size ({max_accumulation * position_size}) "
            f"> max_position_usd ({max_position})"
        )

    # Risk validation
    # REC-004: Research recommends 10-15% stop-loss for grid strategies
    stop_loss = config.get('stop_loss_pct', 8.0)
    if stop_loss < 5.0:
        warnings.append(
            f"stop_loss_pct={stop_loss}% is tight for grid strategy (REC-004). "
            f"Research recommends 10-15%. Consider 5-10% minimum to reduce premature exits."
        )
    if stop_loss > 15.0:
        warnings.append(f"stop_loss_pct={stop_loss}% allows very large drawdown")

    # Fee check
    fee_rate = config.get('fee_rate', 0.001)
    if grid_spacing < fee_rate * 200 * 2:  # Round trip fees as percentage
        warnings.append(
            f"grid_spacing_pct ({grid_spacing}%) may not cover round-trip fees "
            f"(fee_rate={fee_rate}, ~{fee_rate * 200:.2f}% round trip)"
        )

    # ADX threshold
    adx_threshold = config.get('adx_threshold', 30)
    if adx_threshold < 20:
        warnings.append(f"adx_threshold={adx_threshold} may pause trading too often")
    if adx_threshold > 40:
        warnings.append(f"adx_threshold={adx_threshold} may allow trading in strong trends")

    # REC-007: R:R ratio validation
    # Grid R:R = spacing / stop_loss (simplified for single level)
    rr_ratio = grid_spacing / stop_loss if stop_loss > 0 else 0
    if rr_ratio < 1.0:
        warnings.append(
            f"R:R ratio ({rr_ratio:.2f}:1) is below 1:1 (REC-007). "
            f"spacing={grid_spacing}%, stop_loss={stop_loss}%. "
            f"Need >50% win rate to profit. Consider wider spacing or tighter stop."
        )
    elif rr_ratio < 1.5:
        warnings.append(
            f"R:R ratio ({rr_ratio:.2f}:1) is marginal (REC-007). "
            f"Consider spacing >= 1.5x stop_loss for better risk/reward."
        )

    # Check per-symbol R:R
    from .config import SYMBOL_CONFIGS
    for sym, sym_config in SYMBOL_CONFIGS.items():
        sym_spacing = sym_config.get('grid_spacing_pct', grid_spacing)
        sym_stop = sym_config.get('stop_loss_pct', stop_loss)
        if sym_stop > 0:
            sym_rr = sym_spacing / sym_stop
            if sym_rr < 1.0:
                warnings.append(
                    f"{sym} R:R ratio ({sym_rr:.2f}:1) below 1:1 (REC-007). "
                    f"spacing={sym_spacing}%, stop_loss={sym_stop}%"
                )

    return warnings


def validate_config_overrides(
    overrides: Dict[str, Any],
    base_config: Dict[str, Any]
) -> List[str]:
    """
    Validate configuration overrides against base config.

    Args:
        overrides: User-provided config overrides
        base_config: Base configuration to compare against

    Returns:
        List of warning messages
    """
    warnings = []

    # Check for unknown keys
    known_keys = set(base_config.keys())
    for key in overrides:
        if key not in known_keys:
            warnings.append(f"Unknown config key: {key}")

    # Validate merged config
    merged = {**base_config, **overrides}
    config_warnings = validate_config(merged)
    warnings.extend(config_warnings)

    return warnings


def validate_grid_level(level: Dict[str, Any]) -> List[str]:
    """
    Validate a grid level structure.

    Args:
        level: Grid level dict to validate

    Returns:
        List of validation errors
    """
    errors = []

    required_keys = ['price', 'side', 'size', 'filled', 'order_id']
    for key in required_keys:
        if key not in level:
            errors.append(f"Missing required key: {key}")

    if 'side' in level and level['side'] not in ('buy', 'sell'):
        errors.append(f"Invalid side: {level['side']}")

    if 'price' in level and level['price'] <= 0:
        errors.append(f"Invalid price: {level['price']}")

    if 'size' in level and level['size'] <= 0:
        errors.append(f"Invalid size: {level['size']}")

    return errors
