"""
Mean Reversion Strategy v4.2.1 - Compatibility Shim

This file maintains backward compatibility by re-exporting from the
mean_reversion/ module package.

For the full implementation, see:
- mean_reversion/__init__.py - Package entry point and documentation
- mean_reversion/config.py - Strategy metadata and configuration
- mean_reversion/indicators.py - Technical indicator calculations
- mean_reversion/regimes.py - Volatility regime classification
- mean_reversion/risk.py - Risk management functions
- mean_reversion/signals.py - Signal generation logic
- mean_reversion/lifecycle.py - on_start, on_fill, on_stop callbacks
"""

# Re-export everything from the module package
from .mean_reversion import (
    # Required strategy interface
    STRATEGY_NAME,
    STRATEGY_VERSION,
    SYMBOLS,
    CONFIG,
    generate_signal,
    # Optional lifecycle callbacks
    on_start,
    on_fill,
    on_stop,
    # Enums
    VolatilityRegime,
    RejectionReason,
    # Additional exports
    SYMBOL_CONFIGS,
)

__all__ = [
    'STRATEGY_NAME',
    'STRATEGY_VERSION',
    'SYMBOLS',
    'CONFIG',
    'generate_signal',
    'on_start',
    'on_fill',
    'on_stop',
    'VolatilityRegime',
    'RejectionReason',
    'SYMBOL_CONFIGS',
]
