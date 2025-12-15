"""
Whale Sentiment Strategy v1.4.0

Trades based on whale activity proxy (volume spikes) and price deviation
sentiment indicators using a contrarian approach.

REC-009: Research Foundation (Updated for v1.4.0)
Based on academic literature analysis:
- "The Moby Dick Effect" (Magner & Sanhueza, 2025): Whale contagion effects
- Philadelphia Federal Reserve (2024): Whale vs retail behavior
- PMC/NIH (2023): RSI ineffectiveness in crypto markets
See deep-review-v4.0.md Section 7 for full research references.

Key Features (v1.4.0):
- Volume spike detection as PRIMARY signal (55% weight)
- Price deviation for sentiment classification (35% weight)
- RSI code REMOVED per REC-032 (deprecated code cleanup)
- EXTREME volatility regime with trading pause (REC-031)
- Candle data persistence for fast restart recovery
- Contrarian mode: buy fear, sell greed
- Trade flow confirmation (10% weight)
- Cross-pair correlation management
- Session-aware position sizing (UTC validated)
- Guide v2.0 compliance: 100%

Entry Logic:
- Long: Fear sentiment (price deviation) + whale accumulation or neutral
- Short: Greed sentiment (price deviation) + whale distribution or neutral

Exit Logic:
- Sentiment reversal (primary - sentiment shifts opposite)
- Stop loss / Take profit
- Trailing stop (optional)

Version History:
- 1.4.0: Deep Review v4.0 Implementation
         - REC-030: CRITICAL - Fixed undefined function reference
         - REC-031: Added EXTREME volatility regime with trading pause
         - REC-032: Removed deprecated RSI code (calculate_rsi, config settings)
         - REC-033: Added scope and limitations documentation
         - Guide v2.0 compliance: 100%
- 1.3.0: Deep Review v3.0 Implementation
         - REC-021 to REC-027: RSI removal, volatility regimes, dynamic confidence
- 1.2.0: Deep Review v2.0 Implementation
         - REC-011 to REC-020: Persistence, warmup, RSI removal from confidence
- 1.1.0: Deep Review v1.0 Implementation
         - REC-001 to REC-010: Initial improvements
- 1.0.0: Initial implementation
"""

# =============================================================================
# Public API - Required exports for strategy interface
# =============================================================================
from .config import (
    STRATEGY_NAME,
    STRATEGY_VERSION,
    SYMBOLS,
    CONFIG,
    SYMBOL_CONFIGS,
    # Enums (for type hints and external use)
    SentimentZone,
    WhaleSignal,
    SignalDirection,
    TradingSession,
    RejectionReason,
    # Helper
    get_symbol_config,
)

from .signal import generate_signal

from .lifecycle import on_start, on_fill, on_stop

# =============================================================================
# Secondary exports (for advanced use / testing)
# =============================================================================
from .validation import validate_config, validate_config_overrides

from .indicators import (
    calculate_ema,
    calculate_sma,
    calculate_atr,  # REC-023: Volatility regime
    detect_volume_spike,
    classify_whale_signal,
    calculate_fear_greed_proxy,
    classify_sentiment_zone,
    get_sentiment_string,
    is_fear_zone,
    is_greed_zone,
    is_extreme_zone,
    detect_rsi_divergence,  # Stub only - returns 'none' always
    calculate_trade_flow,
    check_trade_flow_confirmation,
    calculate_rolling_correlation,
    calculate_composite_confidence,
    validate_volume_spike,
    # REC-032: calculate_rsi REMOVED
)

from .regimes import (
    classify_trading_session,
    get_session_adjustments,
    get_sentiment_regime_adjustments,
    is_contrarian_opportunity,
    should_reduce_size_for_sentiment,
    # REC-023/REC-031: Volatility regime
    VolatilityRegime,
    classify_volatility_regime,
    get_volatility_adjustments,
    # REC-025: Extended fear period
    check_extended_fear_period,
    # REC-027: Dynamic confidence
    calculate_dynamic_confidence_threshold,
)

from .risk import (
    check_fee_profitability,
    check_circuit_breaker,
    check_correlation_exposure,
    check_position_limits,
    check_real_correlation,
    calculate_position_age,
    calculate_position_pnl,
)

from .exits import (
    check_sentiment_reversal_exit,
    check_stop_loss_exit,
    check_take_profit_exit,
    check_trailing_stop_exit,
    check_all_exits,
)

from .signal import (
    track_rejection,
    build_base_indicators,
)

from .lifecycle import initialize_state

# REC-011: Candle persistence exports
from .persistence import (
    save_candles,
    load_candles,
    should_save_candles,
    get_candle_file_path,
    delete_candle_file,
    get_persistence_status,
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
    'VolatilityRegime',  # REC-031
    # Helpers
    'get_symbol_config',
    # Validation
    'validate_config',
    'validate_config_overrides',
    # Indicators
    'calculate_ema',
    'calculate_sma',
    'calculate_atr',  # REC-023
    'detect_volume_spike',
    'classify_whale_signal',
    'calculate_fear_greed_proxy',
    'classify_sentiment_zone',
    'get_sentiment_string',
    'is_fear_zone',
    'is_greed_zone',
    'is_extreme_zone',
    'detect_rsi_divergence',  # Stub only
    'calculate_trade_flow',
    'check_trade_flow_confirmation',
    'calculate_rolling_correlation',
    'calculate_composite_confidence',
    'validate_volume_spike',
    # Regimes
    'classify_trading_session',
    'get_session_adjustments',
    'get_sentiment_regime_adjustments',
    'is_contrarian_opportunity',
    'should_reduce_size_for_sentiment',
    'classify_volatility_regime',  # REC-023
    'get_volatility_adjustments',  # REC-023/REC-031
    'check_extended_fear_period',  # REC-025
    'calculate_dynamic_confidence_threshold',  # REC-027
    # Risk
    'check_fee_profitability',
    'check_circuit_breaker',
    'check_correlation_exposure',
    'check_position_limits',
    'check_real_correlation',
    'calculate_position_age',
    'calculate_position_pnl',
    # Exits
    'check_sentiment_reversal_exit',
    'check_stop_loss_exit',
    'check_take_profit_exit',
    'check_trailing_stop_exit',
    'check_all_exits',
    # Signal helpers
    'initialize_state',
    'track_rejection',
    'build_base_indicators',
    # REC-011: Persistence
    'save_candles',
    'load_candles',
    'should_save_candles',
    'get_candle_file_path',
    'delete_candle_file',
    'get_persistence_status',
]
