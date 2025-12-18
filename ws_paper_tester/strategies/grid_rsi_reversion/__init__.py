"""
Grid RSI Reversion Strategy v1.3.0

Combines grid trading mechanics with RSI-based mean reversion signals.
Grid levels provide primary entry signals, while RSI acts as a confidence
modifier to enhance signal quality and position sizing.

Based on research from master-plan-v1.0.md:
- Geometric grid spacing for crypto volatility
- RSI as confidence modifier (not hard filter)
- Adaptive RSI zones based on ATR volatility
- Cycle-based position management (buy/sell pairs)
- Multi-position tracking across grid levels
- Trend filter (ADX > 30 pauses trading)

Target Pairs: XRP/USDT, BTC/USDT, XRP/BTC

Version History:
- 1.3.0: Configurable Timeframe Support
         - NEW: candle_timeframe_minutes config parameter (5, 60, 1440)
         - NEW: _get_candles_for_timeframe() helper for dynamic selection
         - NEW: Optimizer --timeframes CLI flag for timeframe override
         - NEW: Optimizer --focus adaptive and --focus timeframes modes
         - IMPROVED: Expanded parameter grids with adaptive features
         - IMPROVED: Period-based time estimation in optimizer
         - IMPROVED: Detailed CLI help with examples (epilog)
         - IMPROVED: Timeframe tracking in indicators dict

- 1.2.0: Deep Review v2.1 Implementation (REC-009 through REC-010)
         - REC-009: BTC/USDT grid_spacing_pct increased from 1.0% to 1.5%
           for improved R:R ratio (0.10:1 → 0.15:1)
         - REC-010: Aligned adx_recenter_threshold from 25 to 30 to match
           main trend filter threshold for consistency
         - REC-011: Documented VPIN for regime detection as future enhancement
         - Compliance score: 97% per deep-review-v2.1

- 1.1.0: Deep Review v2.0 Implementation (REC-001 through REC-008)
         - REC-001: Signal rejection tracking verified on all paths
         - REC-002: Complete indicator logging on all early exits
         - REC-003: Trade flow confirmation (volume analysis)
         - REC-004: Widened stop-loss (5-10% per symbol)
         - REC-005: Real correlation monitoring between symbols
         - REC-006: Liquidity validation for XRP/BTC
         - REC-007: Explicit R:R ratio calculation and validation
         - REC-008: Trend check before grid recentering
         - Enhanced per-symbol configuration for XRP/BTC
         - Added new rejection reasons for flow/liquidity
         - Improved indicator metadata on all code paths

- 1.0.0: Initial implementation based on master-plan-v1.0.md research
         - Grid level setup with geometric/arithmetic spacing
         - RSI confidence calculation for entries
         - Adaptive RSI zones based on ATR
         - Grid cycle completion tracking
         - Stop loss below lowest grid level
         - Trend filter using ADX
         - Volatility regime classification
         - Per-symbol configuration

Future Enhancements (documented per REC-011):
- VPIN (Volume-Synchronized Probability of Informed Trading) for regime detection
- Would provide order flow-based regime classification complementing price volatility
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
    GridType,
    VolatilityRegime,
    TradingSession,
    RejectionReason,
    GridLevelStatus,
    RSIZone,
    # Helpers
    get_symbol_config,
    get_grid_type,
)

from .signal import generate_signal

from .lifecycle import on_start, on_fill, on_stop

# =============================================================================
# Secondary exports (for advanced use / testing)
# =============================================================================
from .validation import validate_config, validate_config_overrides, validate_grid_level

from .indicators import (
    calculate_rsi,
    calculate_atr,
    calculate_adx,
    calculate_volatility,
    get_adaptive_rsi_zones,
    classify_rsi_zone,
    calculate_rsi_confidence,
    calculate_position_size_multiplier,
    # REC-003: Trade flow confirmation
    calculate_trade_flow,
    calculate_volume_ratio,
    check_trade_flow_confirmation,
    # REC-005: Correlation monitoring
    calculate_rolling_correlation,
    # REC-006: Liquidity validation
    check_liquidity_threshold,
    # REC-007: R:R ratio calculation
    calculate_grid_rr_ratio,
)

from .grid import (
    calculate_grid_prices,
    setup_grid_levels,
    get_unfilled_levels,
    get_nearest_unfilled_level,
    check_price_at_grid_level,
    mark_level_filled,
    count_filled_levels,
    check_cycle_completion,
    should_recenter_grid,
    recenter_grid,
    calculate_grid_stats,
)

from .regimes import (
    classify_volatility_regime,
    get_regime_adjustments,
    classify_trading_session,
    get_session_adjustments,
)

from .risk import (
    check_accumulation_limit,
    check_position_limits,
    check_correlation_exposure,
    check_circuit_breaker,
    check_trend_filter,
    check_max_drawdown,
    calculate_position_pnl,
    calculate_position_age,
    check_all_risk_limits,
)

from .exits import (
    check_grid_stop_loss,
    check_max_drawdown_exit,
    check_stale_position_exit,
    check_grid_cycle_sell,
    check_all_exits,
)

from .signal import (
    track_rejection,
    build_base_indicators,
)

from .lifecycle import initialize_state, initialize_grid_for_symbol


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
    'GridType',
    'VolatilityRegime',
    'TradingSession',
    'RejectionReason',
    'GridLevelStatus',
    'RSIZone',
    # Helpers
    'get_symbol_config',
    'get_grid_type',
    # Validation
    'validate_config',
    'validate_config_overrides',
    'validate_grid_level',
    # Indicators
    'calculate_rsi',
    'calculate_atr',
    'calculate_adx',
    'calculate_volatility',
    'get_adaptive_rsi_zones',
    'classify_rsi_zone',
    'calculate_rsi_confidence',
    'calculate_position_size_multiplier',
    # REC-003: Trade flow
    'calculate_trade_flow',
    'calculate_volume_ratio',
    'check_trade_flow_confirmation',
    # REC-005: Correlation
    'calculate_rolling_correlation',
    # REC-006: Liquidity
    'check_liquidity_threshold',
    # REC-007: R:R ratio
    'calculate_grid_rr_ratio',
    # Grid
    'calculate_grid_prices',
    'setup_grid_levels',
    'get_unfilled_levels',
    'get_nearest_unfilled_level',
    'check_price_at_grid_level',
    'mark_level_filled',
    'count_filled_levels',
    'check_cycle_completion',
    'should_recenter_grid',
    'recenter_grid',
    'calculate_grid_stats',
    # Regimes
    'classify_volatility_regime',
    'get_regime_adjustments',
    'classify_trading_session',
    'get_session_adjustments',
    # Risk
    'check_accumulation_limit',
    'check_position_limits',
    'check_correlation_exposure',
    'check_circuit_breaker',
    'check_trend_filter',
    'check_max_drawdown',
    'calculate_position_pnl',
    'calculate_position_age',
    'check_all_risk_limits',
    # Exits
    'check_grid_stop_loss',
    'check_max_drawdown_exit',
    'check_stale_position_exit',
    'check_grid_cycle_sell',
    'check_all_exits',
    # Signal helpers
    'track_rejection',
    'build_base_indicators',
    'initialize_state',
    'initialize_grid_for_symbol',
]
