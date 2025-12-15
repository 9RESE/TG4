"""
Whale Sentiment Strategy v1.2.0

Trades based on whale activity proxy (volume spikes) and price deviation
sentiment indicators using a contrarian approach.

REC-009: Research Foundation (Updated for v1.2.0)
Based on academic literature analysis:
- "The Moby Dick Effect" (Magner & Sanhueza, 2025): Whale contagion effects
- Philadelphia Federal Reserve (2024): Whale vs retail behavior
- PMC/NIH (2023): RSI ineffectiveness in crypto markets
See deep-review-v2.0.md Section 7 for full research references.

Key Features (v1.2.0):
- Volume spike detection as PRIMARY signal (55% weight per REC-013)
- Price deviation for sentiment classification (35% weight per REC-013)
- RSI REMOVED from confidence calculation (academically ineffective per REC-013)
- Candle data persistence for fast restart recovery (REC-011)
- Contrarian mode: buy fear, sell greed
- Trade flow confirmation (10% weight)
- Cross-pair correlation management
- Session-aware position sizing (UTC validated)

Entry Logic:
- Long: Fear sentiment (price deviation) + whale accumulation or neutral
- Short: Greed sentiment (price deviation) + whale distribution or neutral

Exit Logic:
- Sentiment reversal (primary - sentiment shifts opposite)
- Stop loss / Take profit
- Trailing stop (optional)

Version History:
- 1.2.0: Deep Review v2.0 Implementation
         - REC-011: Candle data persistence for fast restarts
         - REC-012: Warmup progress indicator (pct, ETA)
         - REC-013: REMOVED RSI from confidence (volume 55%, price dev 35%, flow 10%)
         - REC-016: XRP/BTC re-enablement guard with explicit flag
         - REC-017: UTC timezone validation on startup
         - REC-018: Trade flow expected indicator for clarity
         - REC-019: Volume window now per-symbol configurable
         - REC-020: Extracted magic numbers to config
         - REC-014/REC-015: Documented for future (volatility regimes, backtesting)
- 1.1.0: Deep Review v1.0 Implementation
         - REC-001: Recalibrated confidence weights (volume 40%, RSI 15%)
         - REC-005: Enhanced indicator logging on all code paths
         - REC-007: Disabled XRP/BTC by default (liquidity concerns)
         - REC-008: Reduced short size multiplier to 0.5x (squeeze risk)
         - REC-009: Updated research documentation references
         - REC-010: Documented UTC timezone requirement for sessions
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
)

from .regimes import (
    classify_trading_session,
    get_session_adjustments,
    get_sentiment_regime_adjustments,
    is_contrarian_opportunity,
    should_reduce_size_for_sentiment,
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
    # Helpers
    'get_symbol_config',
    # Validation
    'validate_config',
    'validate_config_overrides',
    # Indicators
    'calculate_ema',
    'calculate_sma',
    'calculate_rsi',
    'detect_volume_spike',
    'classify_whale_signal',
    'calculate_fear_greed_proxy',
    'classify_sentiment_zone',
    'get_sentiment_string',
    'is_fear_zone',
    'is_greed_zone',
    'is_extreme_zone',
    'detect_rsi_divergence',
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
