"""
Production Integration Module

Provides model registry, signal ensemble, and live trading integration.
"""

from .registry import ModelRegistry, ModelInfo
from .ensemble import SignalEnsemble, EnsembleConfig, EnsembleMethod
from .strategy import MLStrategy, MLStrategyConfig

__all__ = [
    'ModelRegistry',
    'ModelInfo',
    'SignalEnsemble',
    'EnsembleConfig',
    'EnsembleMethod',
    'MLStrategy',
    'MLStrategyConfig'
]
