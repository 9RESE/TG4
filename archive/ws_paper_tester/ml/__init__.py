"""
ML Trading System v1.0

Machine learning module for signal classification, price prediction,
and reinforcement learning-based trading strategies.

Modules:
    - config: ML configuration management
    - data: Data loading and dataset classes
    - features: Feature extraction and engineering
    - models: ML model implementations (XGBoost, LSTM, RL)
    - training: Training pipelines and optimization
    - evaluation: Metrics and backtesting
    - rl: Reinforcement learning environments and agents
    - integration: Model registry, ensemble, and strategy integration
"""

__version__ = "1.0.0"
__author__ = "TG4 Trading Bot"

# Core configuration
from .config import MLConfig, default_config

# Feature extraction
from .features import FeatureExtractor

# Data utilities
from .data import TradingDataset, SequenceDataset, DataPreprocessor, create_dataloaders

# Models
from .models import (
    XGBoostClassifier,
    LightGBMClassifier,
    SignalClassifier,
    PriceDirectionLSTM,
    LSTMPredictor
)

# Training
from .training import Trainer, train_xgboost, train_lstm

# Evaluation
from .evaluation import (
    calculate_metrics,
    calculate_trading_metrics,
    TradingMetrics,
    backtest_model,
    BacktestConfig,
    BacktestResult
)

# RL components
from .rl import TradingEnv, TradingEnvConfig, train_ppo, train_sac

# Integration
from .integration import (
    ModelRegistry,
    ModelInfo,
    SignalEnsemble,
    EnsembleConfig,
    MLStrategy,
    MLStrategyConfig
)

__all__ = [
    # Version
    "__version__",
    # Config
    "MLConfig",
    "default_config",
    # Features
    "FeatureExtractor",
    # Data
    "TradingDataset",
    "SequenceDataset",
    "DataPreprocessor",
    "create_dataloaders",
    # Models
    "XGBoostClassifier",
    "LightGBMClassifier",
    "SignalClassifier",
    "PriceDirectionLSTM",
    "LSTMPredictor",
    # Training
    "Trainer",
    "train_xgboost",
    "train_lstm",
    # Evaluation
    "calculate_metrics",
    "calculate_trading_metrics",
    "TradingMetrics",
    "backtest_model",
    "BacktestConfig",
    "BacktestResult",
    # RL
    "TradingEnv",
    "TradingEnvConfig",
    "train_ppo",
    "train_sac",
    # Integration
    "ModelRegistry",
    "ModelInfo",
    "SignalEnsemble",
    "EnsembleConfig",
    "MLStrategy",
    "MLStrategyConfig",
]
