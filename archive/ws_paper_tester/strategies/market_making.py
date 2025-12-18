"""
Market Making Strategy v1.5.0 - Compatibility Shim

This file maintains backward compatibility by re-exporting from the
market_making/ module package.

For the full implementation, see:
- market_making/__init__.py - Package entry point and documentation
- market_making/config.py - Strategy metadata and configuration
- market_making/calculations.py - Pure calculation functions
- market_making/signals.py - Signal generation logic
- market_making/lifecycle.py - on_start, on_fill, on_stop callbacks
"""

# Re-export everything from the module package
from .market_making import (
    # Required strategy interface
    STRATEGY_NAME,
    STRATEGY_VERSION,
    SYMBOLS,
    CONFIG,
    SYMBOL_CONFIGS,
    generate_signal,
    # Optional lifecycle callbacks
    on_start,
    on_fill,
    on_stop,
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
]
