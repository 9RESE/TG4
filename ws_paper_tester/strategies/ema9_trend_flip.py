"""
EMA-9 Trend Flip Strategy - Backward Compatibility Shim

This module provides backward compatibility for code that imports from
ws_paper_tester.strategies.ema9_trend_flip directly.

The actual implementation has been refactored into the ema9_trend_flip/ package.
All imports are re-exported from there.

Usage:
    # Both of these work:
    from ws_paper_tester.strategies.ema9_trend_flip import generate_signal, CONFIG
    from ws_paper_tester.strategies import ema9_trend_flip
"""

# Re-export everything from the package
from .ema9_trend_flip import (
    # Required strategy interface
    STRATEGY_NAME,
    STRATEGY_VERSION,
    SYMBOLS,
    CONFIG,
    SYMBOL_CONFIGS,
    generate_signal,
    on_start,
    on_fill,
    on_stop,
    # Enums
    TrendDirection,
    RejectionReason,
    # Helpers
    get_symbol_config,
    initialize_state,
    # Indicators
    calculate_ema,
    calculate_ema_series,
    build_hourly_candles,
    calculate_atr,
    get_candle_position,
    check_consecutive_positions,
    # Exits
    check_exit_conditions,
    check_ema_flip_exit,
    check_max_hold_time_exit,
    # Risk
    check_position_limits,
    calculate_entry_stops,
    create_entry_signal,
    track_rejection,
    # Signal helpers
    build_indicators,
)


__all__ = [
    # Required strategy interface
    'STRATEGY_NAME',
    'STRATEGY_VERSION',
    'SYMBOLS',
    'CONFIG',
    'SYMBOL_CONFIGS',
    'generate_signal',
    'on_start',
    'on_fill',
    'on_stop',
    # Enums
    'TrendDirection',
    'RejectionReason',
    # Helpers
    'get_symbol_config',
    'initialize_state',
    # Indicators
    'calculate_ema',
    'calculate_ema_series',
    'build_hourly_candles',
    'calculate_atr',
    'get_candle_position',
    'check_consecutive_positions',
    # Exits
    'check_exit_conditions',
    'check_ema_flip_exit',
    'check_max_hold_time_exit',
    # Risk
    'check_position_limits',
    'calculate_entry_stops',
    'create_entry_signal',
    'track_rejection',
    # Signal helpers
    'build_indicators',
]
