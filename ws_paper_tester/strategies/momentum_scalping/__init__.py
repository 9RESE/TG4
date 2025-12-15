"""
Momentum Scalping Strategy v2.0.0

Trades based on RSI, MACD, and EMA momentum indicators on 1-minute timeframes.
Targets quick momentum bursts with strict risk management.

Based on research from master-plan-v1.0.md:
- RSI period 7 for fast momentum detection
- MACD settings (6, 13, 5) optimized for 1-minute scalping
- EMA 8/21/50 ribbon for trend direction
- Volume spike confirmation
- 2:1 risk-reward ratio
- Maximum 3-minute hold time

Version History:
- 2.0.0: Deep Review v1.0 Implementation
         - REC-001 (CRITICAL): XRP/BTC correlation monitoring with pause thresholds
           - New: calculate_correlation() for XRP-BTC correlation calculation
           - New: get_xrp_btc_correlation() for correlation tracking
           - New: should_pause_for_low_correlation() to pause XRP/BTC at low correlation
           - New: CORRELATION_BREAKDOWN rejection reason
           - Config: correlation_warn_threshold (0.55), correlation_pause_threshold (0.50)
         - REC-002 (HIGH): 5m trend filter for multi-timeframe confirmation
           - New: check_5m_trend_alignment() function
           - New: TIMEFRAME_MISALIGNMENT rejection reason
           - Entry signals must align with 5m EMA trend direction
         - REC-003 (HIGH): ADX trend strength filter for BTC/USDT
           - New: calculate_adx() for ADX calculation
           - New: check_adx_strong_trend() to filter strong trending markets
           - New: ADX_STRONG_TREND rejection reason
           - Skip BTC/USDT entries when ADX > 25 (strong trend)
         - REC-004 (MEDIUM): Regime-based RSI band adjustment
           - Widen RSI bands (75/25) during HIGH volatility regime
           - Reduces false signals when crypto sustains overbought conditions
         - REC-005, REC-006: Documented for future implementation

- 1.0.0: Initial implementation based on master-plan-v1.0.md research
         - RSI + MACD + EMA signal generation
         - Volume spike confirmation
         - Volatility regime classification
         - Session awareness
         - Time-based and momentum exhaustion exits
         - Cross-pair correlation management
         - Per-symbol configuration for XRP/USDT, BTC/USDT, XRP/BTC
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
    VolatilityRegime,
    TradingSession,
    RejectionReason,
    MomentumDirection,
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
    # v2.0.0 additions
    calculate_correlation,
    calculate_adx,
    check_5m_trend_alignment,
)

from .regimes import (
    classify_volatility_regime,
    get_regime_adjustments,
    classify_trading_session,
    get_session_adjustments,
)

from .risk import (
    check_fee_profitability,
    check_circuit_breaker,
    check_correlation_exposure,
    check_position_limits,
    is_volume_confirmed,
    calculate_position_age,
    calculate_position_pnl,
    # v2.0.0 additions
    get_xrp_btc_correlation,
    should_pause_for_low_correlation,
    check_adx_strong_trend,
)

from .exits import (
    check_take_profit_exit,
    check_stop_loss_exit,
    check_time_based_exit,
    check_momentum_exhaustion_exit,
    check_ema_cross_exit,
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
    'VolatilityRegime',
    'TradingSession',
    'RejectionReason',
    'MomentumDirection',
    # Helpers
    'get_symbol_config',
    # Validation
    'validate_config',
    'validate_config_overrides',
    # Indicators
    'calculate_ema',
    'calculate_ema_series',
    'calculate_rsi',
    'calculate_rsi_series',
    'calculate_macd',
    'calculate_macd_with_history',
    'calculate_volume_ratio',
    'calculate_volume_spike',
    'calculate_volatility',
    'calculate_atr',
    'check_ema_alignment',
    'check_momentum_signal',
    # v2.0.0 indicator additions
    'calculate_correlation',
    'calculate_adx',
    'check_5m_trend_alignment',
    # Regimes
    'classify_volatility_regime',
    'get_regime_adjustments',
    'classify_trading_session',
    'get_session_adjustments',
    # Risk
    'check_fee_profitability',
    'check_circuit_breaker',
    'check_correlation_exposure',
    'check_position_limits',
    'is_volume_confirmed',
    'calculate_position_age',
    'calculate_position_pnl',
    # v2.0.0 risk additions
    'get_xrp_btc_correlation',
    'should_pause_for_low_correlation',
    'check_adx_strong_trend',
    # Exits
    'check_take_profit_exit',
    'check_stop_loss_exit',
    'check_time_based_exit',
    'check_momentum_exhaustion_exit',
    'check_ema_cross_exit',
    'check_all_exits',
    # Signal helpers
    'initialize_state',
    'track_rejection',
    'build_base_indicators',
]
