"""
Momentum Scalping Strategy - Backward Compatibility Shim

This module provides backward compatibility for code that imports from
ws_paper_tester.strategies.momentum_scalping directly.

The actual implementation has been refactored into the momentum_scalping/ package.
All imports are re-exported from there.

Usage:
    # Both of these work:
    from ws_paper_tester.strategies.momentum_scalping import generate_signal, CONFIG
    from ws_paper_tester.strategies import momentum_scalping
"""

# Re-export everything from the package
from .momentum_scalping import (
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
    MomentumDirection,
    # Helpers
    get_symbol_config,
    # Validation
    validate_config,
    validate_config_overrides,
    # Indicators
    calculate_ema,
    calculate_ema_series,
    calculate_rsi,
    calculate_rsi_series,
    calculate_macd,
    calculate_macd_with_history,
    calculate_volume_ratio,
    calculate_volume_spike,
    calculate_volatility,
    calculate_atr,
    check_ema_alignment,
    check_momentum_signal,
    # v2.0.0 indicator additions
    calculate_correlation,
    calculate_adx,
    check_5m_trend_alignment,
    # Regimes
    classify_volatility_regime,
    get_regime_adjustments,
    classify_trading_session,
    get_session_adjustments,
    # Risk
    check_fee_profitability,
    check_circuit_breaker,
    check_correlation_exposure,
    check_position_limits,
    is_volume_confirmed,
    calculate_position_age,
    calculate_position_pnl,
    # v2.0.0 risk additions
    get_xrp_btc_correlation,
    should_pause_for_low_correlation,
    check_adx_strong_trend,
    # Exits
    check_take_profit_exit,
    check_stop_loss_exit,
    check_time_based_exit,
    check_momentum_exhaustion_exit,
    check_ema_cross_exit,
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
    'VolatilityRegime',
    'TradingSession',
    'RejectionReason',
    'MomentumDirection',
]
