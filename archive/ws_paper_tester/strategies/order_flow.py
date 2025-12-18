"""
Order Flow Strategy - Backward Compatibility Shim

This module provides backward compatibility for code that imports from
ws_paper_tester.strategies.order_flow directly.

The actual implementation has been refactored into the order_flow/ package.
All imports are re-exported from there.

Usage:
    # Both of these work:
    from ws_paper_tester.strategies.order_flow import generate_signal, CONFIG
    from ws_paper_tester.strategies import order_flow
"""

# Re-export everything from the package
from .order_flow import (
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
    VolatilityRegime,
    TradingSession,
    RejectionReason,
    # Helpers
    get_symbol_config,
    # Validation
    validate_config,
    validate_config_overrides,
    # Indicators
    calculate_volatility,
    calculate_micro_price,
    calculate_vpin,
    # Regimes
    classify_volatility_regime,
    get_regime_adjustments,
    classify_trading_session,
    get_session_adjustments,
    # Risk
    check_fee_profitability,
    calculate_trailing_stop,
    get_progressive_decay_multiplier,
    check_circuit_breaker,
    is_trade_flow_aligned,
    check_correlation_exposure,
    # Exits
    check_trailing_stop_exit,
    check_position_decay_exit,
    # Signal helpers
    initialize_state,
    track_rejection,
    build_base_indicators,
)

# Provide underscore-prefixed versions for any code that used internal functions
_get_symbol_config = get_symbol_config
_validate_config = validate_config
_validate_config_overrides = validate_config_overrides
_calculate_volatility = calculate_volatility
_calculate_micro_price = calculate_micro_price
_calculate_vpin = calculate_vpin
_classify_volatility_regime = classify_volatility_regime
_get_regime_adjustments = get_regime_adjustments
_classify_trading_session = classify_trading_session
_get_session_adjustments = get_session_adjustments
_check_fee_profitability = check_fee_profitability
_calculate_trailing_stop = calculate_trailing_stop
_get_progressive_decay_multiplier = get_progressive_decay_multiplier
_check_circuit_breaker = check_circuit_breaker
_is_trade_flow_aligned = is_trade_flow_aligned
_check_correlation_exposure = check_correlation_exposure
_check_trailing_stop_exit = check_trailing_stop_exit
_check_position_decay_exit = check_position_decay_exit
_initialize_state = initialize_state
_track_rejection = track_rejection
_build_base_indicators = build_base_indicators


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
    'VolatilityRegime',
    'TradingSession',
    'RejectionReason',
]
