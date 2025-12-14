"""
Order Flow Strategy - Configuration Validation

Validates configuration on startup and validates runtime overrides.
"""
from typing import Dict, Any, List

from .config import SYMBOL_CONFIGS


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
        'take_profit_pct', 'lookback_trades', 'cooldown_seconds',
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
    imbalance = config.get('imbalance_threshold', 0.3)
    if imbalance < 0.1 or imbalance > 0.8:
        errors.append(f"imbalance_threshold should be 0.1-0.8, got {imbalance}")

    fee_rate = config.get('fee_rate', 0.001)
    if fee_rate < 0 or fee_rate > 0.01:
        errors.append(f"fee_rate should be 0-0.01, got {fee_rate}")

    # R:R ratio warning
    sl = config.get('stop_loss_pct', 0.5)
    tp = config.get('take_profit_pct', 1.0)
    if sl > 0 and tp > 0:
        rr_ratio = tp / sl
        if rr_ratio < 1.0:
            errors.append(f"Warning: R:R ratio ({rr_ratio:.2f}:1) < 1:1")
        elif rr_ratio < 1.5:
            errors.append(f"Info: R:R ratio {rr_ratio:.2f}:1 acceptable but consider 2:1+")

    # VPIN bounds
    vpin_threshold = config.get('vpin_high_threshold', 0.7)
    if vpin_threshold < 0.5 or vpin_threshold > 0.9:
        errors.append(f"vpin_high_threshold should be 0.5-0.9, got {vpin_threshold}")

    # REC-002: Validate session boundaries
    session_bounds = config.get('session_boundaries', {})
    for key in ['asia_start', 'asia_end', 'europe_start', 'europe_end',
                'overlap_start', 'overlap_end', 'us_start', 'us_end']:
        if key in session_bounds:
            val = session_bounds[key]
            if not isinstance(val, (int, float)) or val < 0 or val > 24:
                errors.append(f"session_boundaries.{key} must be 0-24, got {val}")

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


def validate_config_overrides(config: Dict[str, Any], overrides: Dict[str, Any]) -> List[str]:
    """
    REC-002: Validate configuration overrides match expected types.

    Returns:
        List of error messages for invalid overrides
    """
    errors = []

    # Type expectations for key parameters
    type_checks = {
        'position_size_usd': (int, float),
        'max_position_usd': (int, float),
        'stop_loss_pct': (int, float),
        'take_profit_pct': (int, float),
        'lookback_trades': (int,),
        'cooldown_trades': (int,),
        'cooldown_seconds': (int, float),
        'imbalance_threshold': (int, float),
        'buy_imbalance_threshold': (int, float),
        'sell_imbalance_threshold': (int, float),
        'volume_spike_mult': (int, float),
        'fee_rate': (int, float),
        'vpin_high_threshold': (int, float),
        'vpin_bucket_count': (int,),
        'use_vpin': (bool,),
        'use_volatility_regimes': (bool,),
        'use_session_awareness': (bool,),
        'use_correlation_management': (bool,),
        'use_trailing_stop': (bool,),
        'use_circuit_breaker': (bool,),
        'use_fee_check': (bool,),
        'use_trade_flow_confirmation': (bool,),
        'use_position_decay': (bool,),
        'use_asymmetric_thresholds': (bool,),
    }

    for key, value in overrides.items():
        if key in type_checks:
            expected_types = type_checks[key]
            if not isinstance(value, expected_types):
                type_names = '/'.join(t.__name__ for t in expected_types)
                errors.append(f"Override {key}: expected {type_names}, got {type(value).__name__}")

    return errors
