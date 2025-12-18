"""
Momentum Scalping Strategy - Configuration Validation

Validates configuration on startup and validates runtime overrides.
"""
from typing import Dict, Any, List

from .config import SYMBOL_CONFIGS


def validate_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate configuration on startup.

    Args:
        config: Strategy configuration dict

    Returns:
        List of error/warning messages (empty if valid)
    """
    errors = []

    # Required positive values
    required_positive = [
        'position_size_usd', 'max_position_usd', 'stop_loss_pct',
        'take_profit_pct', 'cooldown_seconds', 'max_hold_seconds',
        'ema_fast_period', 'ema_slow_period', 'ema_filter_period',
        'rsi_period', 'macd_fast', 'macd_slow', 'macd_signal',
    ]

    for key in required_positive:
        val = config.get(key)
        if val is None:
            errors.append(f"Missing required config: {key}")
        elif not isinstance(val, (int, float)):
            errors.append(f"{key} must be numeric, got {type(val).__name__}")
        elif val <= 0:
            errors.append(f"{key} must be positive, got {val}")

    # EMA period ordering
    ema_fast = config.get('ema_fast_period', 8)
    ema_slow = config.get('ema_slow_period', 21)
    ema_filter = config.get('ema_filter_period', 50)

    if ema_fast >= ema_slow:
        errors.append(f"ema_fast_period ({ema_fast}) must be < ema_slow_period ({ema_slow})")
    if ema_slow >= ema_filter:
        errors.append(f"ema_slow_period ({ema_slow}) must be < ema_filter_period ({ema_filter})")

    # RSI bounds
    rsi_ob = config.get('rsi_overbought', 70)
    rsi_os = config.get('rsi_oversold', 30)

    if not (50 < rsi_ob <= 100):
        errors.append(f"rsi_overbought should be 50-100, got {rsi_ob}")
    if not (0 <= rsi_os < 50):
        errors.append(f"rsi_oversold should be 0-50, got {rsi_os}")
    if rsi_os >= rsi_ob:
        errors.append(f"rsi_oversold ({rsi_os}) must be < rsi_overbought ({rsi_ob})")

    # MACD period ordering
    macd_fast = config.get('macd_fast', 6)
    macd_slow = config.get('macd_slow', 13)

    if macd_fast >= macd_slow:
        errors.append(f"macd_fast ({macd_fast}) must be < macd_slow ({macd_slow})")

    # Fee rate bounds
    fee_rate = config.get('fee_rate', 0.001)
    if fee_rate < 0 or fee_rate > 0.01:
        errors.append(f"fee_rate should be 0-0.01, got {fee_rate}")

    # R:R ratio warning
    sl = config.get('stop_loss_pct', 0.4)
    tp = config.get('take_profit_pct', 0.8)
    if sl > 0 and tp > 0:
        rr_ratio = tp / sl
        if rr_ratio < 1.0:
            errors.append(f"Warning: R:R ratio ({rr_ratio:.2f}:1) < 1:1 - may not be profitable")
        elif rr_ratio < 1.5:
            errors.append(f"Info: R:R ratio {rr_ratio:.2f}:1 acceptable but consider 2:1+")

    # Volume spike threshold
    vol_spike = config.get('volume_spike_threshold', 1.5)
    if vol_spike < 1.0 or vol_spike > 5.0:
        errors.append(f"volume_spike_threshold should be 1.0-5.0, got {vol_spike}")

    # Max hold time check
    max_hold = config.get('max_hold_seconds', 180)
    if max_hold < 30 or max_hold > 600:
        errors.append(f"max_hold_seconds should be 30-600, got {max_hold}")

    # Session boundaries validation
    session_bounds = config.get('session_boundaries', {})
    for key in ['asia_start', 'asia_end', 'europe_start', 'europe_end',
                'overlap_start', 'overlap_end', 'us_start', 'us_end',
                'off_hours_start', 'off_hours_end']:
        if key in session_bounds:
            val = session_bounds[key]
            if not isinstance(val, (int, float)) or val < 0 or val > 24:
                errors.append(f"session_boundaries.{key} must be 0-24, got {val}")

    # REC-001 (v2.0.0): Correlation threshold validation
    corr_warn = config.get('correlation_warn_threshold', 0.55)
    corr_pause = config.get('correlation_pause_threshold', 0.50)
    if not (0.0 <= corr_warn <= 1.0):
        errors.append(f"correlation_warn_threshold must be 0.0-1.0, got {corr_warn}")
    if not (0.0 <= corr_pause <= 1.0):
        errors.append(f"correlation_pause_threshold must be 0.0-1.0, got {corr_pause}")
    if corr_pause > corr_warn:
        errors.append(f"correlation_pause_threshold ({corr_pause}) should be <= warn ({corr_warn})")

    # REC-003 (v2.0.0): ADX threshold validation
    adx_threshold = config.get('adx_strong_trend_threshold', 25)
    if not (10 <= adx_threshold <= 50):
        errors.append(f"adx_strong_trend_threshold should be 10-50, got {adx_threshold}")

    # REC-004 (v2.0.0): Regime RSI band validation
    regime_rsi_ob = config.get('regime_high_rsi_overbought', 75)
    regime_rsi_os = config.get('regime_high_rsi_oversold', 25)
    if not (50 < regime_rsi_ob <= 100):
        errors.append(f"regime_high_rsi_overbought should be 50-100, got {regime_rsi_ob}")
    if not (0 <= regime_rsi_os < 50):
        errors.append(f"regime_high_rsi_oversold should be 0-50, got {regime_rsi_os}")
    # Regime RSI bands should be wider than default
    default_rsi_ob = config.get('rsi_overbought', 70)
    default_rsi_os = config.get('rsi_oversold', 30)
    if regime_rsi_ob < default_rsi_ob:
        errors.append(f"Warning: regime_high_rsi_overbought ({regime_rsi_ob}) < default ({default_rsi_ob})")
    if regime_rsi_os > default_rsi_os:
        errors.append(f"Warning: regime_high_rsi_oversold ({regime_rsi_os}) > default ({default_rsi_os})")

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

        # Per-symbol EMA ordering
        sym_ema_fast = sym_cfg.get('ema_fast_period')
        sym_ema_slow = sym_cfg.get('ema_slow_period')
        sym_ema_filter = sym_cfg.get('ema_filter_period')
        if sym_ema_fast and sym_ema_slow and sym_ema_fast >= sym_ema_slow:
            errors.append(f"{symbol}: ema_fast_period >= ema_slow_period")

    return errors


def validate_config_overrides(config: Dict[str, Any], overrides: Dict[str, Any]) -> List[str]:
    """
    Validate configuration overrides match expected types.

    Args:
        config: Base configuration
        overrides: Override values to validate

    Returns:
        List of error messages for invalid overrides
    """
    errors = []

    # Type expectations for key parameters
    type_checks = {
        # Numeric parameters
        'position_size_usd': (int, float),
        'max_position_usd': (int, float),
        'max_position_per_symbol_usd': (int, float),
        'stop_loss_pct': (int, float),
        'take_profit_pct': (int, float),
        'max_hold_seconds': (int, float),
        'cooldown_seconds': (int, float),
        'cooldown_trades': (int,),
        'fee_rate': (int, float),
        'volume_spike_threshold': (int, float),
        'volume_lookback': (int,),
        # Indicator parameters
        'ema_fast_period': (int,),
        'ema_slow_period': (int,),
        'ema_filter_period': (int,),
        'rsi_period': (int,),
        'rsi_overbought': (int, float),
        'rsi_oversold': (int, float),
        'macd_fast': (int,),
        'macd_slow': (int,),
        'macd_signal': (int,),
        # Boolean parameters
        'use_volatility_regimes': (bool,),
        'use_session_awareness': (bool,),
        'use_correlation_management': (bool,),
        'use_circuit_breaker': (bool,),
        'use_fee_check': (bool,),
        'use_macd_confirmation': (bool,),
        'use_5m_trend_filter': (bool,),
        'require_volume_confirmation': (bool,),
        'require_ema_alignment': (bool,),
        'track_rejections': (bool,),
        'regime_extreme_pause': (bool,),
        # v2.0.0 REC-001: Correlation monitoring
        'use_correlation_monitoring': (bool,),
        'correlation_lookback': (int,),
        'correlation_warn_threshold': (int, float),
        'correlation_pause_threshold': (int, float),
        'correlation_pause_enabled': (bool,),
        # v2.0.0 REC-002: 5m trend filter
        '5m_ema_period': (int,),
        # v2.0.0 REC-003: ADX filter
        'use_adx_filter': (bool,),
        'adx_period': (int,),
        'adx_strong_trend_threshold': (int, float),
        'adx_filter_btc_only': (bool,),
        # v2.0.0 REC-004: Regime RSI bands
        'regime_high_rsi_overbought': (int, float),
        'regime_high_rsi_oversold': (int, float),
    }

    for key, value in overrides.items():
        if key in type_checks:
            expected_types = type_checks[key]
            if not isinstance(value, expected_types):
                type_names = '/'.join(t.__name__ for t in expected_types)
                errors.append(f"Override {key}: expected {type_names}, got {type(value).__name__}")

    return errors
