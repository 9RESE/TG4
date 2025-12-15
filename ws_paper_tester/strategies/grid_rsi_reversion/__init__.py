"""
Grid RSI Reversion Strategy v1.0.0

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
- 1.0.0: Initial implementation based on master-plan-v1.0.md research
         - Grid level setup with geometric/arithmetic spacing
         - RSI confidence calculation for entries
         - Adaptive RSI zones based on ATR
         - Grid cycle completion tracking
         - Stop loss below lowest grid level
         - Trend filter using ADX
         - Volatility regime classification
         - Per-symbol configuration
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
