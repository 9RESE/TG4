"""
ML Signal Strategy

A machine learning-based trading strategy that uses trained models from
the model registry to generate signals. Integrates with historic database
features including order flow and multi-timeframe analysis.

Usage:
    The strategy loads models from the registry and uses them to predict
    buy/sell/hold signals based on extracted features from market data.
"""

from .signal import (
    STRATEGY_NAME,
    STRATEGY_VERSION,
    SYMBOLS,
    CONFIG,
    generate_signal,
    on_start,
    on_fill,
    on_stop,
    initialize_state,
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
    'initialize_state',
]
