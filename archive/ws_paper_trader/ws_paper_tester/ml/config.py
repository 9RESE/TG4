"""
ML Configuration Management

Centralized configuration for all ML components including:
- Feature extraction parameters
- Model architectures
- Training hyperparameters
- GPU settings
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from pathlib import Path
import os
import yaml


@dataclass
class FeatureConfig:
    """Configuration for feature extraction."""

    # Trend indicators
    ema_periods: List[int] = field(default_factory=lambda: [9, 21, 50, 200])
    sma_periods: List[int] = field(default_factory=lambda: [20, 50])

    # Momentum indicators
    rsi_period: int = 14
    rsi_fast_period: int = 7
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    stoch_k_period: int = 14
    stoch_d_period: int = 3

    # Volatility indicators
    atr_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    adx_period: int = 14

    # Volume indicators
    volume_lookback: int = 20

    # Returns calculation
    return_periods: List[int] = field(default_factory=lambda: [1, 5, 10, 20])

    # Multi-timeframe
    timeframes: List[int] = field(default_factory=lambda: [1, 5, 15, 60, 240])

    # Sequence length for LSTM/Transformer
    sequence_length: int = 60


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""

    # Database connection
    database_url: str = "postgresql://postgres:password@localhost:5433/trading"

    # Symbols
    symbols: List[str] = field(default_factory=lambda: ["XRP/USDT", "BTC/USDT"])

    # Data splits (by time)
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Batch size
    batch_size: int = 128

    # Lookback for training data (days)
    lookback_days: int = 365

    # Label generation
    future_bars: int = 5  # Prediction horizon
    direction_threshold_pct: float = 0.5  # Threshold for buy/sell classification

    # Data normalization
    normalization_method: str = "zscore"  # "zscore" or "minmax"

    # Cache settings
    cache_dir: str = "data/cache"


@dataclass
class XGBoostConfig:
    """Configuration for XGBoost signal classifier."""

    # Model parameters
    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 1
    gamma: float = 0.1
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0

    # Training settings
    early_stopping_rounds: int = 50
    eval_metric: str = "mlogloss"

    # Class weights for imbalanced data
    use_class_weights: bool = True

    # GPU settings
    tree_method: str = "hist"
    device: str = "cuda"


@dataclass
class LSTMConfig:
    """Configuration for LSTM price predictor."""

    # Architecture
    input_size: int = 10
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = True
    use_attention: bool = True

    # Output
    num_classes: int = 3  # buy, sell, hold

    # Training
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5

    # Sequence
    sequence_length: int = 60

    # Mixed precision
    use_amp: bool = True


@dataclass
class RLConfig:
    """Configuration for reinforcement learning agent."""

    # Environment
    initial_balance: float = 1000.0
    max_position_pct: float = 0.25
    fee_rate: float = 0.001
    slippage_pct: float = 0.0005

    # Observation space
    observation_size: int = 36

    # PPO parameters
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5

    # Training
    total_timesteps: int = 1_000_000

    # Reward shaping
    pnl_weight: float = 1.0
    risk_penalty_weight: float = 0.5
    drawdown_penalty_weight: float = 0.3


@dataclass
class OptunaConfig:
    """Configuration for hyperparameter optimization."""

    n_trials: int = 100
    study_name: str = "signal_classifier"
    storage: str = "sqlite:///optuna.db"
    direction: str = "maximize"  # maximize Sharpe ratio

    # Search space bounds
    lr_min: float = 1e-5
    lr_max: float = 1e-2
    hidden_size_options: List[int] = field(default_factory=lambda: [64, 128, 256])
    num_layers_options: List[int] = field(default_factory=lambda: [1, 2, 3, 4])
    dropout_min: float = 0.1
    dropout_max: float = 0.5
    batch_size_options: List[int] = field(default_factory=lambda: [32, 64, 128, 256])


@dataclass
class GPUConfig:
    """GPU configuration for AMD ROCm."""

    # Device settings
    device: str = "cuda"
    use_amp: bool = True  # Automatic mixed precision

    # ROCm specific (for RX 6700 XT)
    hsa_override_gfx_version: str = "10.3.0"
    pytorch_rocm_arch: str = "gfx1030"

    # Memory management
    max_memory_gb: float = 10.0  # Leave 2GB headroom on 12GB card
    gradient_checkpointing: bool = False

    # MIOpen settings
    miopen_find_mode: int = 3  # Use cached kernels


@dataclass
class MLConfig:
    """Master ML configuration combining all sub-configs."""

    # Sub-configurations
    features: FeatureConfig = field(default_factory=FeatureConfig)
    data: DataConfig = field(default_factory=DataConfig)
    xgboost: XGBoostConfig = field(default_factory=XGBoostConfig)
    lstm: LSTMConfig = field(default_factory=LSTMConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    optuna: OptunaConfig = field(default_factory=OptunaConfig)
    gpu: GPUConfig = field(default_factory=GPUConfig)

    # Paths
    model_dir: str = "models"
    log_dir: str = "logs/ml"
    tensorboard_dir: str = "logs/tensorboard"

    # Experiment tracking
    experiment_name: str = "ml_trading_v1"

    # Feature set to use
    feature_set: str = "xgboost"  # "xgboost", "lstm", "rl"

    @classmethod
    def from_yaml(cls, path: str) -> "MLConfig":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls._from_dict(config_dict)

    @classmethod
    def _from_dict(cls, d: Dict[str, Any]) -> "MLConfig":
        """Create config from dictionary."""
        config = cls()

        if "features" in d:
            config.features = FeatureConfig(**d["features"])
        if "data" in d:
            config.data = DataConfig(**d["data"])
        if "xgboost" in d:
            config.xgboost = XGBoostConfig(**d["xgboost"])
        if "lstm" in d:
            config.lstm = LSTMConfig(**d["lstm"])
        if "rl" in d:
            config.rl = RLConfig(**d["rl"])
        if "optuna" in d:
            config.optuna = OptunaConfig(**d["optuna"])
        if "gpu" in d:
            config.gpu = GPUConfig(**d["gpu"])

        # Top-level settings
        for key in ["model_dir", "log_dir", "tensorboard_dir",
                    "experiment_name", "feature_set"]:
            if key in d:
                setattr(config, key, d[key])

        return config

    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = {
            "features": self.features.__dict__,
            "data": self.data.__dict__,
            "xgboost": self.xgboost.__dict__,
            "lstm": self.lstm.__dict__,
            "rl": self.rl.__dict__,
            "optuna": self.optuna.__dict__,
            "gpu": self.gpu.__dict__,
            "model_dir": self.model_dir,
            "log_dir": self.log_dir,
            "tensorboard_dir": self.tensorboard_dir,
            "experiment_name": self.experiment_name,
            "feature_set": self.feature_set,
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    def setup_gpu_env(self) -> None:
        """Set up GPU environment variables for AMD ROCm."""
        os.environ["HSA_OVERRIDE_GFX_VERSION"] = self.gpu.hsa_override_gfx_version
        os.environ["PYTORCH_ROCM_ARCH"] = self.gpu.pytorch_rocm_arch
        os.environ["MIOPEN_FIND_MODE"] = str(self.gpu.miopen_find_mode)

    def ensure_dirs(self) -> None:
        """Ensure all required directories exist."""
        for dir_path in [self.model_dir, self.log_dir, self.tensorboard_dir,
                         self.data.cache_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


# Feature sets for different model types
XGBOOST_FEATURES = [
    # Trend (8)
    'price_vs_ema_9', 'price_vs_ema_21', 'price_vs_ema_50',
    'ema_alignment', 'returns_1', 'returns_5', 'returns_10', 'returns_20',

    # Momentum (8)
    'rsi_14', 'rsi_7', 'macd_histogram', 'stoch_k', 'stoch_d',
    'momentum_10', 'rsi_divergence', 'macd_crossover',

    # Volatility (8)
    'atr_pct', 'bb_position', 'bb_width', 'adx_14', 'di_plus', 'di_minus',
    'volatility_20', 'volatility_regime',

    # Volume (4)
    'volume_ratio', 'volume_zscore', 'trade_imbalance', 'vpin',

    # Temporal (4)
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
]

LSTM_FEATURES = [
    # Price (normalized per-sequence)
    'open_norm', 'high_norm', 'low_norm', 'close_norm', 'volume_norm',

    # Technical (already normalized 0-100 or bounded)
    'rsi_14', 'bb_position', 'adx_14',

    # Returns (inherently normalized)
    'returns_1', 'log_returns_1',
]

RL_FEATURES = [
    # Market state (12)
    'price_vs_ema_9', 'rsi_14', 'atr_pct', 'volume_ratio',
    'bb_position', 'adx_14', 'macd_histogram', 'stoch_k',
    'returns_1', 'returns_5', 'volatility_20', 'trade_imbalance',

    # Position state (8)
    'has_position', 'position_side', 'position_size_pct',
    'unrealized_pnl_pct', 'distance_to_stop', 'distance_to_target',
    'position_duration', 'entry_price_vs_current',

    # Account state (8)
    'equity_pct', 'available_margin_pct', 'current_drawdown_pct',
    'consecutive_wins', 'consecutive_losses', 'win_rate_recent',
    'avg_profit_pct', 'sharpe_recent',

    # Regime (8)
    'volatility_regime', 'trend_strength', 'market_regime',
    'correlation_xrp_btc', 'btc_momentum', 'xrp_momentum',
    'hour_sin', 'hour_cos',
]


def get_feature_list(feature_set: str) -> List[str]:
    """Get feature list for a given model type."""
    feature_sets = {
        "xgboost": XGBOOST_FEATURES,
        "lstm": LSTM_FEATURES,
        "rl": RL_FEATURES,
    }
    return feature_sets.get(feature_set, XGBOOST_FEATURES)


# Default configuration instance
default_config = MLConfig()
