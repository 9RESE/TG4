"""
Grid RSI Reversion Strategy Entry Point

Re-exports the strategy interface from the grid_rsi_reversion package.
This file allows the strategy to be loaded as a module:
    from strategies.grid_rsi_reversion import generate_signal

Or imported by the strategy loader:
    strategies/grid_rsi_reversion
"""

from .grid_rsi_reversion import (
    # Required interface
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

__all__ = [
    'STRATEGY_NAME',
    'STRATEGY_VERSION',
    'SYMBOLS',
    'CONFIG',
    'SYMBOL_CONFIGS',
    'generate_signal',
    'on_start',
    'on_fill',
    'on_stop',
    'GridType',
    'VolatilityRegime',
    'TradingSession',
    'RejectionReason',
    'GridLevelStatus',
    'RSIZone',
    'get_symbol_config',
    'get_grid_type',
]
