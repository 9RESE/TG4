"""
Ratio Trading Strategy v4.2.1 - Compatibility Shim

This file maintains backward compatibility by re-exporting from the
ratio_trading/ module package.

For the full implementation, see:
- ratio_trading/__init__.py - Package entry point and documentation
- ratio_trading/config.py - Strategy metadata and configuration
- ratio_trading/enums.py - Type-safe enumerations
- ratio_trading/indicators.py - Technical indicator calculations
- ratio_trading/regimes.py - Volatility regime classification
- ratio_trading/risk.py - Risk management functions
- ratio_trading/tracking.py - State and rejection tracking
- ratio_trading/signals.py - Signal generation logic
- ratio_trading/lifecycle.py - on_start, on_fill, on_stop callbacks
"""

# Re-export everything from the module package
from .ratio_trading import (
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
    ExitReason,
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
    'ExitReason',
]
