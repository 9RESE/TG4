"""
Training module for ML models.

Provides training pipelines, callbacks, and hyperparameter optimization
using Optuna.
"""

from .trainer import Trainer, train_xgboost, train_lstm
from .optimize import run_optimization, create_study

__all__ = [
    "Trainer",
    "train_xgboost",
    "train_lstm",
    "run_optimization",
    "create_study",
]
