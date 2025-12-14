"""
Order Flow Strategy v4.2.0

Trades based on trade tape analysis and buy/sell imbalance.
Enhanced with VPIN, volatility regimes, session awareness, and advanced risk management.

Version History:
- 1.0.0: Initial implementation
- 2.0.0: Added volatility adjustment, dynamic thresholds
- 3.0.0: Added per-pair PnL, trade flow confirmation, fee check, circuit breaker
- 3.1.0: Fixed asymmetric thresholds, config validation
- 4.0.0: Major refactor per order-flow-strategy-review-v3.1.md
         - REC-001: VPIN (Volume-Synchronized Probability of Informed Trading)
         - REC-002: Volatility regime classification (LOW/MEDIUM/HIGH/EXTREME)
         - REC-003: Time-of-day session awareness (Asia/Europe/US/Overlap)
         - REC-004: Progressive position decay (gradual TP reduction)
         - REC-005: Cross-pair correlation management
- 4.1.0: Improvements per order-flow-strategy-review-v4.0.md
         - REC-001: Signal rejection logging and statistics
         - REC-002: Configuration override validation with type checking
         - REC-003: Configurable session boundaries (DST-aware)
         - REC-004: Enhanced position decay with close-at-profit-after-fees option
         - Finding #1: Improved VPIN bucket overflow logic
         - Finding #5: Better position decay exit at intermediate stages
- 4.1.1: Modular refactoring - split into multiple files for maintainability
- 4.2.0: Improvements per deep-review-v5.0.md
         - REC-002: Circuit breaker now reads max_consecutive_losses from config
         - REC-003: Exit signals use per-symbol position size for multi-symbol accuracy
         - REC-004: VWAP reversion signals now check trade flow confirmation
         - REC-005: Micro-price fallback status logged in indicators
         - REC-006: Per-symbol position limits (max_position_per_symbol_usd)
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
    calculate_volatility,
    calculate_micro_price,
    calculate_vpin,
)

from .regimes import (
    classify_volatility_regime,
    get_regime_adjustments,
    classify_trading_session,
    get_session_adjustments,
)

from .risk import (
    check_fee_profitability,
    calculate_trailing_stop,
    get_progressive_decay_multiplier,
    check_circuit_breaker,
    is_trade_flow_aligned,
    check_correlation_exposure,
)

from .exits import (
    check_trailing_stop_exit,
    check_position_decay_exit,
)

from .signal import (
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
    # Helpers
    'get_symbol_config',
    # Validation
    'validate_config',
    'validate_config_overrides',
    # Indicators
    'calculate_volatility',
    'calculate_micro_price',
    'calculate_vpin',
    # Regimes
    'classify_volatility_regime',
    'get_regime_adjustments',
    'classify_trading_session',
    'get_session_adjustments',
    # Risk
    'check_fee_profitability',
    'calculate_trailing_stop',
    'get_progressive_decay_multiplier',
    'check_circuit_breaker',
    'is_trade_flow_aligned',
    'check_correlation_exposure',
    # Exits
    'check_trailing_stop_exit',
    'check_position_decay_exit',
    # Signal helpers
    'initialize_state',
    'track_rejection',
    'build_base_indicators',
]
