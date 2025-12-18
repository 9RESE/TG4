# Model Catalog - Existing ML Assets

**Document Version**: 1.0
**Created**: 2025-12-16
**Status**: Inventory Complete

---

## Overview

This document catalogs existing ML models, code, and infrastructure from the archived TG4 system that can be reused for the ML v1.0 implementation.

## Pre-Trained Models

### 1. LSTM XRP Model

**Location**: `archive/models/lstm_xrp.pth`
**Size**: 205 KB
**Framework**: PyTorch

**Architecture**:
```python
LSTMPredictor(
    input_size=1,
    hidden_size=64,
    num_layers=2,
    output_size=1,
    dropout=0.2
)
```

**Training Details**:
- Trained on XRP/USDT price data
- Sequence length: 60 periods
- Epochs: 50
- Loss function: MSE
- Optimizer: Adam (lr=0.001)

**Capabilities**:
- Price direction prediction
- Next-bar price estimation
- Bullish/bearish signal generation

**Reusability**: HIGH
- Can be fine-tuned on new data
- Architecture suitable for other symbols
- Requires only 60-bar lookback

**Integration Code** (from `archive/src/strategies/xrp_momentum_lstm/strategy.py`):
```python
from models.lstm_predictor import LSTMPredictor

# Load model
model = LSTMPredictor()
model.load_state_dict(torch.load('models/lstm_xrp.pth'))
model.eval()

# Predict
with torch.no_grad():
    prediction = model(price_sequence)
    direction = 'bullish' if prediction > current_price else 'bearish'
```

---

### 2. PPO Trading Agent

**Location**: `archive/models/rl_ppo_agent.zip`
**Size**: 168 KB
**Framework**: Stable-Baselines3

**Architecture**:
```python
PPO(
    policy='MlpPolicy',
    env=TradingEnv,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2
)
```

**Training Details**:
- Trained on multi-symbol trading environment
- Observation space: 36 features
- Action space: Continuous position sizing
- Training timesteps: 1M+

**Capabilities**:
- Portfolio allocation decisions
- Position sizing
- Buy/sell timing

**Reusability**: MEDIUM
- Environment definition may need updates
- Policy can be loaded and continued
- May need retraining on recent data

**Integration Code**:
```python
from stable_baselines3 import PPO

# Load agent
model = PPO.load('models/rl_ppo_agent.zip')

# Predict action
observation = get_observation()
action, _ = model.predict(observation, deterministic=True)
```

---

### 3. Ensemble RL Agent

**Location**: `archive/models/rl_ensemble_agent.zip`
**Size**: 195 KB
**Framework**: Stable-Baselines3

**Architecture**:
- Multi-strategy ensemble controller
- 6-strategy weighting system
- Dynamic regime-based allocation

**Training Details**:
- Custom ensemble environment
- Reward shaping for accumulation goals
- BTC (45%), XRP (35%), USDT (20%) targets

**Capabilities**:
- Strategy selection
- Weight allocation
- Regime-adaptive behavior

**Reusability**: MEDIUM
- Complex environment dependencies
- Ensemble logic transferable
- May need architecture updates

---

## Reusable Code Components

### 1. LSTM Predictor Class

**Location**: `archive/src/models/lstm_predictor.py`
**Lines**: ~200

**Key Features**:
```python
class LSTMPredictor(nn.Module):
    """LSTM model for price prediction"""

    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        # Architecture definition

    def forward(self, x):
        # Forward pass with LSTM

    def train_model(self, train_data, epochs=50, lr=0.001):
        # Training loop with validation

    def save(self, path):
        # Save model state

    def load(self, path):
        # Load model state

    def predict(self, sequence):
        # Generate prediction from sequence
```

**Reusability**: HIGH - Core architecture ready for extension

---

### 2. Custom Trading Environment

**Location**: `archive/src/models/ensemble_env.py`
**Lines**: ~400

**Key Features**:
```python
class EnsembleTradingEnv(gym.Env):
    """Custom Gymnasium environment for multi-strategy trading"""

    # Observation space: 36 features
    observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(36,))

    # Action space: 12 discrete actions
    action_space = spaces.Discrete(12)

    def reset(self):
        # Reset environment state

    def step(self, action):
        # Execute action, return observation, reward, done, info

    def _calculate_reward(self):
        # Reward shaping for accumulation goals
```

**Observation Features**:
```python
[
    # Market data (12 features)
    'btc_close_norm', 'xrp_close_norm', 'btc_volume_norm', 'xrp_volume_norm',
    'btc_rsi', 'xrp_rsi', 'btc_momentum', 'xrp_momentum',
    'correlation', 'spread_pct', 'btc_volatility', 'xrp_volatility',

    # Position state (8 features)
    'btc_position_norm', 'xrp_position_norm', 'usdt_balance_norm',
    'total_equity_norm', 'unrealized_pnl', 'realized_pnl',
    'position_duration', 'leverage',

    # Regime (8 features)
    'regime_bull', 'regime_bear', 'regime_sideways',
    'volatility_regime', 'trend_strength', 'momentum_regime',
    'fear_greed', 'btc_dominance',

    # Strategy state (8 features)
    'strategy_1_weight', 'strategy_2_weight', ..., 'strategy_6_weight',
    'consecutive_wins', 'consecutive_losses'
]
```

**Reusability**: HIGH - Complete environment ready for modification

---

### 3. Feature Engineering Utilities

**Location**: `archive/src/utils/`
**Components**:

#### Technical Indicators
```python
# From utils/indicators.py
def calculate_rsi(prices, period=14):
    """Calculate RSI"""

def calculate_atr(high, low, close, period=14):
    """Calculate ATR"""

def calculate_ema(prices, period):
    """Calculate EMA"""

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD"""

def calculate_bollinger_bands(prices, period=20, std=2):
    """Calculate Bollinger Bands"""
```

#### Regime Detection
```python
# From utils/regime.py
def classify_volatility_regime(volatility):
    """Classify volatility into LOW/MEDIUM/HIGH/EXTREME"""

def detect_trend(ema_short, ema_long):
    """Detect trend direction"""

def calculate_market_regime(indicators):
    """Calculate composite market regime"""
```

**Reusability**: HIGH - Drop-in utilities for feature engineering

---

### 4. Data Processing

**Location**: Various files in archive

#### Sequence Generation
```python
# From models/lstm_predictor.py
def create_sequences(data, seq_length=60):
    """Create input sequences for LSTM"""
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        targets.append(data[i+seq_length])
    return np.array(sequences), np.array(targets)
```

#### Data Normalization
```python
# From utils/preprocessing.py
class DataNormalizer:
    """Normalize trading data"""

    def __init__(self, method='minmax'):
        self.scaler = MinMaxScaler() if method == 'minmax' else StandardScaler()

    def fit_transform(self, data):
        return self.scaler.fit_transform(data)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
```

**Reusability**: HIGH - Standard preprocessing utilities

---

### 5. Strategy Integration Patterns

**Location**: `archive/src/strategies/xrp_momentum_lstm/strategy.py`
**Lines**: ~300

**Key Pattern**:
```python
# Module-level model loading
_model = None
_scaler = None

def on_start(config, state):
    """Initialize ML model"""
    global _model, _scaler
    _model = LSTMPredictor.load(config['model_path'])
    _scaler = MinMaxScaler()
    state['initialized'] = True

def generate_signal(data, config, state):
    """Generate signal using ML model"""
    if not state.get('initialized'):
        return None

    # Prepare input
    prices = extract_prices(data)
    scaled = _scaler.transform(prices)
    tensor = torch.tensor(scaled).unsqueeze(0)

    # Predict
    with torch.no_grad():
        prediction = _model(tensor)

    # Generate signal based on prediction
    if prediction > threshold:
        return create_buy_signal(...)
    elif prediction < -threshold:
        return create_sell_signal(...)

    return None
```

**Reusability**: HIGH - Template for ML strategy integration

---

## Dependency Analysis

### Current requirements.txt

```
# ML Core (Already Present)
torch==2.5.1+rocm6.2
torchvision==0.20.1+rocm6.2
torchaudio==2.5.1+rocm6.2
stable-baselines3>=2.0.0
gymnasium>=0.26.0
scikit-learn
xgboost

# Technical Analysis
ta>=0.11.0
pandas-ta

# Data
numpy<2
pandas
asyncpg

# Statistics
statsmodels>=0.14.0
```

**Status**: Most dependencies already in place. ROCm support configured.

---

## Migration Path

### From Archive to ws_paper_tester/ml

| Archive Component | Target Location | Migration Effort |
|-------------------|-----------------|------------------|
| `lstm_predictor.py` | `ml/models/predictor.py` | Low |
| `ensemble_env.py` | `ml/models/rl_env.py` | Medium |
| `utils/indicators.py` | `ml/features/indicators.py` | Low |
| `utils/regime.py` | `ml/features/regime.py` | Low |
| Strategy patterns | `strategies/ml_signal/` | Low |

### Steps

1. **Copy Core Models**:
   ```bash
   cp archive/src/models/lstm_predictor.py ws_paper_tester/ml/models/
   ```

2. **Update Imports**:
   ```python
   # Old
   from models.lstm_predictor import LSTMPredictor

   # New
   from ml.models.predictor import LSTMPredictor
   ```

3. **Adapt to New Data Layer**:
   - Update data loading to use `HistoricalDataProvider`
   - Adapt feature extraction to use `DataSnapshot`

4. **Test Integration**:
   - Verify model loading
   - Test inference pipeline
   - Run paper trading validation

---

## Recommendations

### High Priority Reuse

1. **LSTM Predictor Architecture**
   - Proven to work for XRP price prediction
   - Clean implementation
   - Easy to extend for multi-feature input

2. **Trading Environment Structure**
   - Well-defined observation/action spaces
   - Comprehensive reward shaping
   - Ready for Stable-Baselines3

3. **Feature Engineering Utilities**
   - Standard indicator calculations
   - Regime classification logic
   - Data normalization

### Medium Priority Reuse

4. **Pre-trained Models**
   - LSTM XRP model (for fine-tuning baseline)
   - PPO agent (for transfer learning)

5. **Strategy Integration Pattern**
   - Module-level model loading
   - Inference pipeline structure

### Avoid Reusing

- Old data loading code (use new TimescaleDB provider)
- Deprecated dependencies
- Hardcoded paths/configurations

---

## Quick Start: Using Archive Components

```python
# Example: Using LSTM predictor with new data layer

# 1. Copy and adapt model
import sys
sys.path.insert(0, 'archive/src')
from models.lstm_predictor import LSTMPredictor

# 2. Load pre-trained model
model = LSTMPredictor(input_size=1, hidden_size=64, num_layers=2)
model.load_state_dict(torch.load('archive/models/lstm_xrp.pth'))
model.eval()

# 3. Use with new data provider
from data.historical_provider import HistoricalDataProvider

async def predict_direction():
    provider = HistoricalDataProvider()

    # Get recent candles
    candles = await provider.get_latest_candles(
        symbol='XRP/USDT',
        interval_minutes=1,
        count=60
    )

    # Extract prices
    prices = np.array([c.close for c in candles])

    # Normalize
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices.reshape(-1, 1))

    # Predict
    tensor = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        prediction = model(tensor)

    # Inverse transform
    predicted_price = scaler.inverse_transform(prediction.numpy())

    return 'bullish' if predicted_price > prices[-1] else 'bearish'
```

---

## Summary

The archive contains substantial reusable ML infrastructure:

| Category | Items | Reusability |
|----------|-------|-------------|
| Pre-trained Models | 3 | Medium-High |
| Model Classes | 2 | High |
| Environments | 1 | High |
| Feature Utilities | 10+ | High |
| Strategy Patterns | 1 | High |

**Estimated Time Savings**: 40-60% compared to building from scratch

**Recommended Approach**: Start with LSTM predictor and feature utilities, adapt to new data layer, then incrementally add complexity.

---

**End of ML v1.0 Planning Documentation**

---

## Document Index

1. [Overview](./00-overview.md) - Executive summary and project scope
2. [System Analysis](./01-system-analysis.md) - Current architecture analysis
3. [Data Analysis](./02-data-analysis.md) - Historical data for ML training
4. [Feature Engineering](./03-feature-engineering.md) - Feature extraction details
5. [AMD GPU Setup](./04-amd-gpu-setup.md) - ROCm/PyTorch configuration
6. [ML Architecture](./05-ml-architecture.md) - Proposed system design
7. [Implementation Roadmap](./06-implementation-roadmap.md) - Phased implementation plan
8. [Model Catalog](./07-model-catalog.md) - Existing ML assets (this document)
