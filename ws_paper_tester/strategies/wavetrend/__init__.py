"""
WaveTrend Oscillator Strategy v1.0.0

Trades based on WaveTrend oscillator crossovers in overbought/oversold zones.
Designed for hourly timeframe but adapts to available data.

Based on research from master-plan-v1.0.md:
- WaveTrend (LazyBear) dual-line crossover system
- Zone-based signal filtering (OB/OS zones)
- Divergence detection for confirmation
- Cross-pair correlation management
- Session-aware position sizing

Entry Logic:
- Long: Bullish crossover (WT1 > WT2) from oversold zone
- Short: Bearish crossover (WT1 < WT2) from overbought zone

Exit Logic:
- Crossover reversal (primary)
- Extreme zone profit taking
- Stop loss / Take profit

Version History:
- 1.0.0: Initial implementation based on master-plan-v1.0.md research
         - WaveTrend calculation with configurable parameters
         - Zone classification (extreme OB/OS, OB/OS, neutral)
         - Crossover detection for entry signals
         - Divergence detection for confidence bonus
         - Session-aware position sizing
         - Cross-pair correlation management
         - Circuit breaker with cooldown
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
    WaveTrendZone,
    CrossoverType,
    DivergenceType,
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
    calculate_ema_series,
    calculate_sma,
    calculate_sma_series,
    calculate_wavetrend,
    classify_zone,
    get_zone_string,
    detect_crossover,
    detect_divergence,
    calculate_confidence,
    is_in_oversold_zone,
    is_in_overbought_zone,
    is_extreme_zone,
)

from .regimes import (
    classify_trading_session,
    get_session_adjustments,
    get_zone_regime_adjustments,
    should_wait_for_zone_exit,
)

from .risk import (
    check_fee_profitability,
    check_circuit_breaker,
    check_correlation_exposure,
    check_position_limits,
    calculate_position_age,
    calculate_position_pnl,
)

from .exits import (
    check_crossover_exit,
    check_extreme_zone_exit,
    check_stop_loss_exit,
    check_take_profit_exit,
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
    'WaveTrendZone',
    'CrossoverType',
    'DivergenceType',
    'TradingSession',
    'RejectionReason',
    # Helpers
    'get_symbol_config',
    # Validation
    'validate_config',
    'validate_config_overrides',
    # Indicators
    'calculate_ema',
    'calculate_ema_series',
    'calculate_sma',
    'calculate_sma_series',
    'calculate_wavetrend',
    'classify_zone',
    'get_zone_string',
    'detect_crossover',
    'detect_divergence',
    'calculate_confidence',
    'is_in_oversold_zone',
    'is_in_overbought_zone',
    'is_extreme_zone',
    # Regimes
    'classify_trading_session',
    'get_session_adjustments',
    'get_zone_regime_adjustments',
    'should_wait_for_zone_exit',
    # Risk
    'check_fee_profitability',
    'check_circuit_breaker',
    'check_correlation_exposure',
    'check_position_limits',
    'calculate_position_age',
    'calculate_position_pnl',
    # Exits
    'check_crossover_exit',
    'check_extreme_zone_exit',
    'check_stop_loss_exit',
    'check_take_profit_exit',
    'check_all_exits',
    # Signal helpers
    'initialize_state',
    'track_rejection',
    'build_base_indicators',
]
