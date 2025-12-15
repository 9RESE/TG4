"""
Whale Sentiment Strategy - Backward Compatibility Shim

This module provides backward compatibility for code that imports from
ws_paper_tester.strategies.whale_sentiment directly.

The actual implementation has been refactored into the whale_sentiment/ package.
All imports are re-exported from there.

Usage:
    # Both of these work:
    from ws_paper_tester.strategies.whale_sentiment import generate_signal, CONFIG
    from ws_paper_tester.strategies import whale_sentiment
"""

# Re-export everything from the package
from .whale_sentiment import (
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
    SentimentZone,
    WhaleSignal,
    SignalDirection,
    TradingSession,
    RejectionReason,
    # Helpers
    get_symbol_config,
    # Validation
    validate_config,
    validate_config_overrides,
    # Indicators
    calculate_ema,
    calculate_sma,
    calculate_rsi,
    detect_volume_spike,
    classify_whale_signal,
    calculate_fear_greed_proxy,
    classify_sentiment_zone,
    get_sentiment_string,
    is_fear_zone,
    is_greed_zone,
    is_extreme_zone,
    detect_rsi_divergence,
    calculate_trade_flow,
    check_trade_flow_confirmation,
    calculate_rolling_correlation,
    calculate_composite_confidence,
    validate_volume_spike,
    # Regimes
    classify_trading_session,
    get_session_adjustments,
    get_sentiment_regime_adjustments,
    is_contrarian_opportunity,
    should_reduce_size_for_sentiment,
    # Risk
    check_fee_profitability,
    check_circuit_breaker,
    check_correlation_exposure,
    check_position_limits,
    check_real_correlation,
    calculate_position_age,
    calculate_position_pnl,
    # Exits
    check_sentiment_reversal_exit,
    check_stop_loss_exit,
    check_take_profit_exit,
    check_trailing_stop_exit,
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
    'SentimentZone',
    'WhaleSignal',
    'SignalDirection',
    'TradingSession',
    'RejectionReason',
]
