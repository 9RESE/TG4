"""
Whale Sentiment Strategy v1.0.0

Trades based on whale activity proxy (volume spikes) and market sentiment
indicators (RSI, price deviation) using a contrarian approach.

Based on research from master-plan-v1.0.md:
- Volume spike detection as whale activity proxy
- RSI + price deviation for sentiment classification
- Contrarian mode: buy fear, sell greed
- Trade flow confirmation for signal validation
- Cross-pair correlation management
- Session-aware position sizing

Entry Logic:
- Long: Fear sentiment (RSI < 40) + whale accumulation or neutral
- Short: Greed sentiment (RSI > 60) + whale distribution or neutral

Exit Logic:
- Sentiment reversal (primary - sentiment shifts opposite)
- Stop loss / Take profit
- Trailing stop (optional)

Version History:
- 1.0.0: Initial implementation based on master-plan-v1.0.md research
         - Volume spike detection as whale proxy
         - RSI sentiment classification
         - Fear/greed price deviation proxy
         - Contrarian mode signal generation
         - Trade flow confirmation
         - Cross-pair correlation management
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
]
