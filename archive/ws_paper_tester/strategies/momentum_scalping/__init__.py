"""
Momentum Scalping Strategy v2.1.0

Trades based on RSI, MACD, and EMA momentum indicators on 1-minute timeframes.
Targets quick momentum bursts with strict risk management.

Based on research from master-plan-v1.0.md:
- RSI period 7-8 for fast momentum detection (8 for XRP per REC-003)
- MACD settings (6, 13, 5) optimized for 1-minute scalping
- EMA 8/21/50 ribbon for trend direction
- Volume spike confirmation
- Trade flow confirmation (REC-007)
- ATR-based trailing stops (REC-005)
- 2:1 risk-reward ratio
- Maximum 3-minute hold time

Version History:
- 2.1.1: REC-012/REC-013 Monitoring Implementation
         - REC-012 (P3/LOW): XRP Independence Monitoring
           - New: monitoring.py module with CorrelationMonitor class
           - Tracks XRP-BTC correlation for weekly review and trend analysis
           - Escalation triggers for sustained low correlation (30 days <0.70)
           - Escalation triggers for high pause rate (>50% sessions paused)
           - State persisted to logs/monitoring/monitoring_state.json
           - Weekly reports saved to logs/monitoring/correlation_report_*.json
         - REC-013 (P3/LOW): Market Sentiment Monitoring
           - New: SentimentMonitor class for Fear & Greed Index tracking
           - Sentiment classification (Extreme Fear/Fear/Neutral/Greed/Extreme Greed)
           - Prolonged extreme sentiment alerts (7+ consecutive days)
           - Volatility expansion signals for regime awareness
           - Config: enable_sentiment_monitoring, sentiment thresholds

- 2.1.0: Deep Review v2.0 Implementation
         - REC-001 (P0/CRITICAL): Raised XRP/BTC pause threshold
           - Config: correlation_pause_threshold raised from 0.50 to 0.60
           - Config: correlation_warn_threshold raised from 0.55 to 0.60
           - Pauses XRP/BTC trading until correlation stabilizes
         - REC-002 (P1/HIGH): Raised ADX threshold for BTC
           - Config: adx_strong_trend_threshold raised from 25 to 30
           - More conservative filtering of strong trending markets
         - REC-003 (P1/HIGH): Changed RSI period for XRP/USDT
           - Config: XRP/USDT rsi_period changed from 7 to 8
           - Reduces noise while maintaining responsiveness
         - REC-005 (P2/MEDIUM): Implemented ATR-based trailing stops
           - New: check_trailing_stop_exit() in exits.py
           - Config: use_trailing_stop, trail_atr_mult, trail_activation_pct
           - Trails at highest - (ATR * multiplier) after activation
         - REC-006 (P2/LOW): Documented DST handling
           - Module docstring in regimes.py with DST configuration guide
           - Session boundaries are configurable for DST adjustments
         - REC-007 (P2/MEDIUM): Trade flow confirmation
           - Uses data.get_trade_imbalance() for entry confirmation
           - New: TRADE_FLOW_MISALIGNMENT rejection reason
           - Config: use_trade_flow_confirmation, trade_imbalance_threshold
         - REC-008 (P2/LOW): Increased correlation lookback
           - Config: correlation_lookback increased from 50 to 100 (~8.3 hours)
           - More stable correlation readings
         - REC-009 (P3/LOW): Breakeven momentum exit option
           - Config: exit_breakeven_on_momentum_exhaustion (default: False)
           - Optional exit near breakeven on RSI extreme
         - REC-010 (P3/LOW): Structured logging
           - Uses Python logging module instead of print statements
           - Compatible with log aggregation tools

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
    # v2.1.0 additions
    check_trailing_stop_exit,
)

from .signal import (
    track_rejection,
    build_base_indicators,
)

from .lifecycle import initialize_state

# =============================================================================
# REC-012/REC-013 (v2.1.0): Monitoring exports
# =============================================================================
from .monitoring import (
    MonitoringManager,
    CorrelationMonitor,
    SentimentMonitor,
    MonitoringState,
    CorrelationRecord,
    SentimentRecord,
    integrate_monitoring_on_tick,
    get_or_create_monitoring_manager,
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
    # v2.1.0 exit additions
    'check_trailing_stop_exit',
    # Signal helpers
    'initialize_state',
    'track_rejection',
    'build_base_indicators',
    # REC-012/REC-013 (v2.1.0): Monitoring
    'MonitoringManager',
    'CorrelationMonitor',
    'SentimentMonitor',
    'MonitoringState',
    'CorrelationRecord',
    'SentimentRecord',
    'integrate_monitoring_on_tick',
    'get_or_create_monitoring_manager',
]
