"""
WaveTrend Oscillator Strategy - Backward Compatibility Shim

This module provides backward compatibility for code that imports from
ws_paper_tester.strategies.wavetrend directly.

The actual implementation has been refactored into the wavetrend/ package.
All imports are re-exported from there.

Usage:
    # Both of these work:
    from ws_paper_tester.strategies.wavetrend import generate_signal, CONFIG
    from ws_paper_tester.strategies import wavetrend
"""

# Re-export everything from the package
from .wavetrend import (
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
    WaveTrendZone,
    CrossoverType,
    DivergenceType,
    TradingSession,
    RejectionReason,
    # Helpers
    get_symbol_config,
    # Validation
    validate_config,
    validate_config_overrides,
    # Indicators
    calculate_ema,
    calculate_ema_series,
    calculate_sma,
    calculate_sma_series,
    calculate_wavetrend,
    classify_zone,
    get_zone_string,
    detect_crossover,
    detect_divergence,
    calculate_confidence,
    is_in_oversold_zone,
    is_in_overbought_zone,
    is_extreme_zone,
    # Regimes
    classify_trading_session,
    get_session_adjustments,
    get_zone_regime_adjustments,
    should_wait_for_zone_exit,
    # Risk
    check_fee_profitability,
    check_circuit_breaker,
    check_correlation_exposure,
    check_position_limits,
    calculate_position_age,
    calculate_position_pnl,
    # Exits
    check_crossover_exit,
    check_extreme_zone_exit,
    check_stop_loss_exit,
    check_take_profit_exit,
    check_all_exits,
    # Signal helpers
    initialize_state,
    track_rejection,
    build_base_indicators,
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
    'WaveTrendZone',
    'CrossoverType',
    'DivergenceType',
    'TradingSession',
    'RejectionReason',
]
