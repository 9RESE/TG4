# ML Architecture - System Design

**Document Version**: 1.0
**Created**: 2025-12-16
**Status**: Design Phase

---

## Overview

This document presents the proposed ML architecture for the TG4 trading system, including model types, training pipelines, and integration patterns.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              ML Trading System v1.0                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────────┐  │
│  │   Data Layer    │    │  Feature Layer  │    │      Model Layer            │  │
│  │                 │    │                 │    │                             │  │
│  │  TimescaleDB    │───▶│  FeatureStore   │───▶│  ┌─────────┐ ┌─────────┐   │  │
│  │  Historical     │    │  Feature Eng.   │    │  │ Signal  │ │  Price  │   │  │
│  │  WebSocket      │    │  Normalization  │    │  │Classifier│ │Predictor│   │  │
│  │                 │    │                 │    │  └─────────┘ └─────────┘   │  │
│  └─────────────────┘    └─────────────────┘    │  ┌─────────┐ ┌─────────┐   │  │
│                                                │  │Position │ │   RL    │   │  │
│                                                │  │  Sizer  │ │  Agent  │   │  │
│                                                │  └─────────┘ └─────────┘   │  │
│                                                └─────────────────────────────┘  │
│                                                              │                   │
│                                                              ▼                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         Integration Layer                                │   │
│  │                                                                          │   │
│  │   ┌───────────────┐    ┌───────────────┐    ┌───────────────────────┐   │   │
│  │   │  ML Strategy  │    │ Signal Fusion │    │  Execution Engine     │   │   │
│  │   │  (New)        │───▶│ (Ensemble)    │───▶│  (Existing)           │   │   │
│  │   └───────────────┘    └───────────────┘    └───────────────────────┘   │   │
│  │                                                                          │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Model Architecture

### Model 1: Signal Classifier (XGBoost/LightGBM)

**Purpose**: Predict buy/sell/hold signals from technical indicators

**Architecture**:
```
Input Features (25-40 features)
         │
         ▼
┌─────────────────────┐
│  XGBoost/LightGBM   │
│  - 500 trees        │
│  - max_depth: 6     │
│  - learning_rate: 0.1│
└─────────────────────┘
         │
         ▼
Output: P(buy), P(sell), P(hold)
```

**Input Features**:
```python
SIGNAL_CLASSIFIER_FEATURES = [
    # Trend (8 features)
    'price_vs_ema_9', 'price_vs_ema_21', 'price_vs_ema_50',
    'ema_alignment', 'consecutive_above_ema',
    'returns_1', 'returns_5', 'returns_20',

    # Momentum (6 features)
    'rsi_14', 'rsi_7', 'macd_histogram',
    'stoch_k', 'stoch_d', 'momentum_10',

    # Volatility (6 features)
    'atr_pct', 'bb_position', 'bb_width',
    'adx_14', 'di_plus', 'di_minus',

    # Volume (4 features)
    'volume_ratio', 'volume_zscore',
    'trade_imbalance', 'vpin',

    # Temporal (4 features)
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'
]
```

**Training Configuration**:
```python
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    objective='multi:softprob',
    num_class=3,  # buy, sell, hold
    tree_method='hist',  # GPU-compatible
    device='cuda',
    random_state=42
)
```

**Output**:
```python
# Prediction output
{
    'buy_probability': 0.65,
    'sell_probability': 0.15,
    'hold_probability': 0.20,
    'predicted_action': 'buy',
    'confidence': 0.65
}
```

### Model 2: Price Direction Predictor (LSTM/Transformer)

**Purpose**: Predict price direction over multiple horizons

**Architecture Option A: LSTM**:
```
Input Sequence (60 bars x 10 features)
         │
         ▼
┌─────────────────────┐
│  LSTM Layer 1       │
│  - hidden: 128      │
│  - bidirectional    │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  LSTM Layer 2       │
│  - hidden: 64       │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  Attention Layer    │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  Dense Layers       │
│  - 64 → 32 → 3      │
└─────────────────────┘
         │
         ▼
Output: P(up), P(down), P(neutral) for each horizon
```

**Architecture Option B: Temporal Fusion Transformer**:
```
Static Features ────────────┐
                            │
Time-Varying Known ─────────┼───▶ Variable Selection
                            │           │
Time-Varying Unknown ───────┘           ▼
                                  LSTM Encoder
                                        │
                                        ▼
                              Interpretable Multi-Head
                                    Attention
                                        │
                                        ▼
                                Quantile Outputs
                                (10%, 50%, 90%)
```

**Implementation (LSTM)**:
```python
import torch
import torch.nn as nn

class PriceDirectionLSTM(nn.Module):
    def __init__(
        self,
        input_size: int = 10,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=4,
            batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)

        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Use last timestep
        last_hidden = attn_out[:, -1, :]

        # Classification
        logits = self.classifier(last_hidden)
        return logits
```

### Model 3: Position Sizer (Regression)

**Purpose**: Determine optimal position size based on market conditions

**Architecture**:
```
Input: Market State + Regime + Risk Metrics
         │
         ▼
┌─────────────────────┐
│  Neural Network     │
│  - 64 → 32 → 16 → 1 │
│  - ReLU activation  │
│  - Sigmoid output   │
└─────────────────────┘
         │
         ▼
Output: Position Size % (0.0 - 1.0)
```

**Implementation**:
```python
class PositionSizer(nn.Module):
    def __init__(self, input_size: int = 15):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Output 0-1
        )

    def forward(self, x):
        return self.network(x)

# Input features
POSITION_SIZER_FEATURES = [
    # Signal strength
    'signal_confidence',
    'signal_classifier_prob',

    # Volatility
    'atr_pct', 'volatility_20', 'volatility_regime',

    # Trend strength
    'adx_14', 'ema_alignment',

    # Risk metrics
    'current_drawdown_pct',
    'consecutive_losses',
    'portfolio_heat',

    # Correlation
    'correlation_xrp_btc',
    'correlation_exposure',

    # Fee impact
    'expected_profit_after_fees',
]
```

### Model 4: Reinforcement Learning Agent (PPO/SAC)

**Purpose**: Learn optimal trading policy through interaction

**Architecture**:
```
Observation Space (36 features)
         │
         ▼
┌─────────────────────┐
│  Actor Network      │
│  - 256 → 256 → 128  │
│  - tanh activation  │
└─────────────────────┘
         │
         ▼
Action Space:
- Continuous: position_size (-1 to 1)
- Discrete: [hold, buy_small, buy_medium, buy_large,
             sell_small, sell_medium, sell_large,
             close_position]
```

**Gymnasium Environment**:
```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TradingEnv(gym.Env):
    """Custom trading environment for RL"""

    def __init__(
        self,
        data_provider,
        initial_balance: float = 1000.0,
        max_position: float = 100.0,
        fee_rate: float = 0.001
    ):
        super().__init__()

        self.data_provider = data_provider
        self.initial_balance = initial_balance
        self.max_position = max_position
        self.fee_rate = fee_rate

        # Observation space: 36 normalized features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(36,), dtype=np.float32
        )

        # Action space: continuous position sizing
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.step_count = 0
        self.data_provider.reset()
        return self._get_observation(), {}

    def step(self, action):
        # Execute action
        position_change = action[0] * self.max_position
        reward = self._execute_trade(position_change)

        # Advance market data
        self.data_provider.step()
        self.step_count += 1

        # Check termination
        done = self.step_count >= self.max_steps
        truncated = self.balance <= 0

        return self._get_observation(), reward, done, truncated, {}

    def _get_observation(self) -> np.ndarray:
        """Build observation vector"""
        features = self.data_provider.get_features()
        position_features = [
            self.position / self.max_position,
            self.balance / self.initial_balance,
            self._get_unrealized_pnl() / self.initial_balance
        ]
        return np.concatenate([features, position_features])

    def _execute_trade(self, position_change: float) -> float:
        """Execute trade and return reward"""
        # Implementation details...
        pass
```

**PPO Training**:
```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Create environment
env = DummyVecEnv([lambda: TradingEnv(data_provider)])

# Create PPO agent
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    tensorboard_log="./logs/ppo_trading/",
    device='cuda',
    verbose=1
)

# Train
model.learn(total_timesteps=1_000_000)

# Save
model.save("models/ppo_trading_v1")
```

## Training Pipeline

### Data Pipeline

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ TimescaleDB  │────▶│   Feature    │────▶│   Training   │
│              │     │   Store      │     │   Dataset    │
└──────────────┘     └──────────────┘     └──────────────┘
                            │
                            ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Model      │────▶│    Model     │────▶│    Model     │
│   Training   │     │  Validation  │     │   Registry   │
└──────────────┘     └──────────────┘     └──────────────┘
```

### Training Script Structure

```python
# ws_paper_tester/ml/train.py

import torch
from torch.utils.data import DataLoader
from pathlib import Path
import mlflow

from ml.data import TradingDataset, create_dataloaders
from ml.models import SignalClassifier, PricePredictor
from ml.features import FeatureExtractor
from ml.evaluation import backtest_model


def train_signal_classifier(
    data_path: Path,
    model_path: Path,
    config: dict
) -> dict:
    """Train signal classification model"""

    # Load and prepare data
    train_loader, val_loader, test_loader = create_dataloaders(
        data_path,
        batch_size=config['batch_size'],
        train_ratio=0.7,
        val_ratio=0.15
    )

    # Initialize model
    model = SignalClassifier(
        input_size=config['input_size'],
        hidden_size=config['hidden_size']
    )

    if torch.cuda.is_available():
        model = model.cuda()

    # Training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_loss = float('inf')

    for epoch in range(config['epochs']):
        # Train epoch
        train_loss = train_epoch(model, train_loader, optimizer, criterion)

        # Validate
        val_loss, val_metrics = validate(model, val_loader, criterion)

        # Log metrics
        mlflow.log_metrics({
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_metrics['accuracy'],
            'val_f1': val_metrics['f1']
        }, step=epoch)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path / 'best_model.pt')

        scheduler.step(val_loss)

    # Final evaluation
    model.load_state_dict(torch.load(model_path / 'best_model.pt'))
    test_metrics = evaluate(model, test_loader)

    # Backtest
    backtest_results = backtest_model(model, test_loader)

    return {
        'test_metrics': test_metrics,
        'backtest': backtest_results
    }
```

### Hyperparameter Optimization

```python
# ws_paper_tester/ml/optimize.py

import optuna
from optuna.integration import PyTorchLightningPruningCallback

def objective(trial: optuna.Trial) -> float:
    """Optuna objective function"""

    # Suggest hyperparameters
    config = {
        'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
        'hidden_size': trial.suggest_categorical('hidden_size', [64, 128, 256]),
        'num_layers': trial.suggest_int('num_layers', 1, 4),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    }

    # Train model
    results = train_signal_classifier(
        data_path=DATA_PATH,
        model_path=MODELS_PATH / f'trial_{trial.number}',
        config=config
    )

    # Return metric to optimize (negative for maximization)
    return -results['backtest']['sharpe_ratio']


def run_optimization(n_trials: int = 100):
    """Run hyperparameter optimization"""

    study = optuna.create_study(
        study_name='signal_classifier',
        direction='minimize',
        storage='sqlite:///optuna.db',
        load_if_exists=True
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=1,  # GPU training
        show_progress_bar=True
    )

    # Best parameters
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best params: {study.best_params}")
    print(f"Best Sharpe: {-study.best_value:.3f}")

    return study.best_params
```

## Model Integration

### ML Strategy Implementation

```python
# ws_paper_tester/strategies/ml_signal/signal.py

from typing import Optional, Dict, Any
import torch
from ws_paper_tester.types import DataSnapshot, Signal
from ml.models import SignalClassifier
from ml.features import FeatureExtractor

STRATEGY_NAME = "ml_signal_v1"
STRATEGY_VERSION = "1.0.0"
SYMBOLS = ["BTC/USDT", "XRP/USDT"]

CONFIG = {
    'model_path': 'models/signal_classifier_v1.pt',
    'confidence_threshold': 0.6,
    'position_size_usd': 50.0,
    'stop_loss_pct': 1.5,
    'take_profit_pct': 3.0
}

# Load model at module level
_model = None
_feature_extractor = None
_device = 'cuda' if torch.cuda.is_available() else 'cpu'


def on_start(config: Dict[str, Any], state: Dict[str, Any]) -> None:
    """Initialize model on strategy start"""
    global _model, _feature_extractor

    _model = SignalClassifier.load(config['model_path'])
    _model = _model.to(_device)
    _model.eval()

    _feature_extractor = FeatureExtractor()
    state['initialized'] = True


def generate_signal(
    data: DataSnapshot,
    config: Dict[str, Any],
    state: Dict[str, Any]
) -> Optional[Signal]:
    """Generate signal using ML model"""

    if not state.get('initialized'):
        return None

    for symbol in SYMBOLS:
        if symbol not in data.candles_1m:
            continue

        # Extract features
        features = _feature_extractor.extract_from_snapshot(data, symbol)
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        features_tensor = features_tensor.to(_device)

        # Model inference
        with torch.no_grad():
            logits = _model(features_tensor)
            probs = torch.softmax(logits, dim=1)

        # Get prediction
        buy_prob = probs[0, 0].item()
        sell_prob = probs[0, 1].item()
        hold_prob = probs[0, 2].item()

        # Generate signal if confident
        if buy_prob > config['confidence_threshold']:
            return Signal(
                symbol=symbol,
                action='buy',
                size=config['position_size_usd'],
                price=data.prices[symbol],
                stop_loss=data.prices[symbol] * (1 - config['stop_loss_pct'] / 100),
                take_profit=data.prices[symbol] * (1 + config['take_profit_pct'] / 100),
                reason=f"ML signal: P(buy)={buy_prob:.2%}",
                metadata={
                    'buy_prob': buy_prob,
                    'sell_prob': sell_prob,
                    'hold_prob': hold_prob,
                    'model': STRATEGY_NAME
                }
            )

        if sell_prob > config['confidence_threshold']:
            # Check if we have a position to sell
            if state.get('has_position', False):
                return Signal(
                    symbol=symbol,
                    action='sell',
                    size=config['position_size_usd'],
                    price=data.prices[symbol],
                    stop_loss=0,
                    take_profit=0,
                    reason=f"ML signal: P(sell)={sell_prob:.2%}",
                    metadata={
                        'buy_prob': buy_prob,
                        'sell_prob': sell_prob,
                        'hold_prob': hold_prob,
                        'model': STRATEGY_NAME
                    }
                )

    return None
```

### Signal Fusion (Ensemble)

```python
# ws_paper_tester/ml/ensemble.py

from typing import List, Optional, Dict
from ws_paper_tester.types import Signal

class SignalEnsemble:
    """Combine signals from multiple models/strategies"""

    def __init__(
        self,
        weights: Dict[str, float] = None,
        voting_threshold: float = 0.6
    ):
        self.weights = weights or {}
        self.voting_threshold = voting_threshold

    def fuse_signals(
        self,
        signals: List[Dict[str, Any]]
    ) -> Optional[Signal]:
        """Fuse multiple signals into consensus"""

        if not signals:
            return None

        # Weight signals
        buy_score = 0.0
        sell_score = 0.0
        total_weight = 0.0

        for signal in signals:
            source = signal.get('source', 'unknown')
            weight = self.weights.get(source, 1.0)

            if signal['action'] == 'buy':
                buy_score += weight * signal.get('confidence', 1.0)
            elif signal['action'] == 'sell':
                sell_score += weight * signal.get('confidence', 1.0)

            total_weight += weight

        # Normalize
        buy_score /= total_weight
        sell_score /= total_weight

        # Make decision
        if buy_score > self.voting_threshold:
            return self._create_consensus_signal(signals, 'buy', buy_score)
        elif sell_score > self.voting_threshold:
            return self._create_consensus_signal(signals, 'sell', sell_score)

        return None
```

## Model Registry

### Version Control

```python
# ws_paper_tester/ml/registry.py

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
import hashlib

@dataclass
class ModelVersion:
    """Model version metadata"""
    name: str
    version: str
    created_at: datetime
    model_path: Path
    config: dict
    metrics: dict
    checksum: str

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'version': self.version,
            'created_at': self.created_at.isoformat(),
            'model_path': str(self.model_path),
            'config': self.config,
            'metrics': self.metrics,
            'checksum': self.checksum
        }


class ModelRegistry:
    """Registry for trained models"""

    def __init__(self, registry_path: Path):
        self.registry_path = registry_path
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self._load_registry()

    def register(
        self,
        name: str,
        version: str,
        model_path: Path,
        config: dict,
        metrics: dict
    ) -> ModelVersion:
        """Register a new model version"""

        # Calculate checksum
        checksum = self._calculate_checksum(model_path)

        # Create version
        model_version = ModelVersion(
            name=name,
            version=version,
            created_at=datetime.utcnow(),
            model_path=model_path,
            config=config,
            metrics=metrics,
            checksum=checksum
        )

        # Save to registry
        self._registry[f"{name}:{version}"] = model_version
        self._save_registry()

        return model_version

    def get_latest(self, name: str) -> Optional[ModelVersion]:
        """Get latest version of a model"""
        versions = [v for k, v in self._registry.items() if k.startswith(f"{name}:")]
        if not versions:
            return None
        return max(versions, key=lambda v: v.created_at)

    def get_best(self, name: str, metric: str) -> Optional[ModelVersion]:
        """Get best version by metric"""
        versions = [v for k, v in self._registry.items() if k.startswith(f"{name}:")]
        if not versions:
            return None
        return max(versions, key=lambda v: v.metrics.get(metric, 0))
```

## Directory Structure

```
ws_paper_tester/
├── ml/
│   ├── __init__.py
│   ├── config.py              # ML configuration
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py         # PyTorch datasets
│   │   ├── loader.py          # Data loading utilities
│   │   └── preprocessing.py   # Data preprocessing
│   ├── features/
│   │   ├── __init__.py
│   │   ├── extractor.py       # Feature extraction
│   │   ├── store.py           # Feature store
│   │   └── transforms.py      # Feature transforms
│   ├── models/
│   │   ├── __init__.py
│   │   ├── classifier.py      # Signal classifier
│   │   ├── predictor.py       # Price predictor
│   │   ├── sizer.py           # Position sizer
│   │   └── rl_agent.py        # RL agent
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py         # Training loops
│   │   ├── callbacks.py       # Training callbacks
│   │   └── optimize.py        # Hyperparameter optimization
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py         # Evaluation metrics
│   │   └── backtest.py        # Backtesting integration
│   ├── ensemble.py            # Signal fusion
│   └── registry.py            # Model registry
├── strategies/
│   └── ml_signal/             # ML strategy
│       ├── __init__.py
│       ├── signal.py
│       └── config.py
└── models/                    # Trained models
    ├── signal_classifier_v1.pt
    ├── price_predictor_v1.pt
    └── registry.json
```

---

**Next Document**: [Implementation Roadmap](./06-implementation-roadmap.md)
