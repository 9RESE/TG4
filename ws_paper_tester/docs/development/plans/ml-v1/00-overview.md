# ML Trading System v1.0 - Project Overview

**Document Version**: 1.0
**Created**: 2025-12-16
**Status**: Planning Phase
**Author**: Claude Code (Research & Planning)

---

## Executive Summary

This document outlines a comprehensive machine learning system designed to enhance the TG4 (grok-4_1) trading bot platform. The ML system will leverage the existing AMD GPU (RX 6700 XT with ROCm 6.2 support) to:

1. **Fine-tune existing strategies** - Optimize parameters through hyperparameter search
2. **Develop new strategies** - Use ML models for signal generation and prediction
3. **Improve risk management** - ML-based position sizing and stop-loss optimization
4. **Predict market regimes** - Classify market conditions for strategy selection

## Project Scope

### In Scope (Phase 1)

| Component | Description | Priority |
|-----------|-------------|----------|
| Signal Classification | ML models to predict buy/sell/hold signals | High |
| Hyperparameter Optimization | Bayesian optimization for strategy tuning | High |
| Feature Engineering Pipeline | Automated technical indicator extraction | High |
| AMD GPU Training | ROCm-accelerated model training | High |
| Backtesting Integration | ML model validation against historical data | High |

### In Scope (Phase 2)

| Component | Description | Priority |
|-----------|-------------|----------|
| Reinforcement Learning | PPO/SAC agents for portfolio optimization | Medium |
| Transformer Models | Time-series forecasting for price direction | Medium |
| Ensemble Methods | Multi-model voting for signal consensus | Medium |
| Online Learning | Continuous model updates from live data | Medium |

### Out of Scope (v1)

- High-frequency trading (sub-second decisions)
- Cross-exchange arbitrage
- Sentiment analysis from social media
- Large language model integration

## System Goals

### Primary Objectives

1. **Improve Win Rate**: Target 60%+ win rate on ML-enhanced signals (vs. current ~55%)
2. **Reduce Drawdown**: ML-optimized stops to reduce max drawdown by 20%
3. **Adapt to Regimes**: Automatic strategy selection based on predicted market regime
4. **Accelerate Development**: GPU-accelerated training for rapid experimentation

### Success Metrics

| Metric | Current Baseline | Target (v1.0) |
|--------|------------------|---------------|
| Signal Win Rate | ~55% | 60%+ |
| Max Drawdown | Variable | -15% cap |
| Sharpe Ratio | ~1.2 | 1.5+ |
| Training Time (full model) | N/A | <2 hours |
| Backtest Coverage | Manual | 100% automated |

## Technical Foundation

### Hardware Resources

- **CPU**: AMD Ryzen 9 7950X (16-core)
- **RAM**: 128GB DDR5
- **GPU**: AMD RX 6700 XT (12GB VRAM)
- **Storage**: NVMe SSD (fast data access)
- **OS**: Ubuntu Linux 6.8.0

### Software Stack

```
┌─────────────────────────────────────────────────┐
│              Application Layer                   │
│  - Strategy Optimization                         │
│  - Signal Generation                             │
│  - Backtesting                                   │
├─────────────────────────────────────────────────┤
│              ML Framework Layer                  │
│  - PyTorch 2.5.1+ROCm6.2                         │
│  - XGBoost / LightGBM                            │
│  - Stable-Baselines3 (RL)                        │
├─────────────────────────────────────────────────┤
│              Data Layer                          │
│  - TimescaleDB (historical candles)              │
│  - pandas / numpy                                │
│  - pandas-ta (indicators)                        │
├─────────────────────────────────────────────────┤
│              Infrastructure                      │
│  - ROCm 6.2 (GPU acceleration)                   │
│  - Docker (TimescaleDB)                          │
│  - asyncpg (database access)                     │
└─────────────────────────────────────────────────┘
```

## Data Availability

### Historical Data Summary

| Symbol | Timeframes | Est. Records | Training Ready |
|--------|------------|--------------|----------------|
| XRP/USDT | 1m-1w | ~525K 1m candles/year | Yes |
| BTC/USDT | 1m-1w | ~525K 1m candles/year | Yes |
| XRP/BTC | 1m-1w | ~525K 1m candles/year | Yes |

### Multi-Timeframe Coverage

- **1-minute**: Base granularity (365-day retention)
- **5m, 15m, 30m**: Auto-computed continuous aggregates
- **1h, 4h, 12h, 1d, 1w**: Auto-computed continuous aggregates

## Existing ML Assets

### Pre-trained Models (Archive)

| Model | Type | File | Status |
|-------|------|------|--------|
| LSTM XRP | Neural Network | `lstm_xrp.pth` (205KB) | Reusable |
| PPO Agent | Reinforcement Learning | `rl_ppo_agent.zip` (168KB) | Reusable |
| Ensemble Agent | RL Ensemble | `rl_ensemble_agent.zip` (195KB) | Reusable |

### Existing Code (Reusable)

- LSTM Predictor class (`archive/src/models/lstm_predictor.py`)
- Custom Gymnasium Environment (`archive/src/models/ensemble_env.py`)
- RL Agent framework (`archive/src/models/rl_agent.py`)
- Feature engineering utilities (`archive/src/utils/`)

## Document Structure

This planning documentation is organized as follows:

| Document | Contents |
|----------|----------|
| `00-overview.md` | This overview document |
| `01-system-analysis.md` | Current system architecture analysis |
| `02-data-analysis.md` | Historical data structure for ML training |
| `03-feature-engineering.md` | Feature extraction from strategies |
| `04-amd-gpu-setup.md` | AMD ROCm/PyTorch setup guide |
| `05-ml-architecture.md` | Proposed ML architecture design |
| `06-implementation-roadmap.md` | Implementation phases and timeline |
| `07-model-catalog.md` | Existing models and reusability assessment |

## Key Findings Summary

### Strengths

1. **Rich Historical Data**: TimescaleDB with 1-minute base data and auto-aggregates
2. **Clean Strategy Architecture**: Consistent signal generation patterns across 9 strategies
3. **Existing ML Infrastructure**: LSTM models, RL agents, and environments already built
4. **GPU Support Ready**: ROCm 6.2 + PyTorch wheels already in requirements.txt
5. **Comprehensive Logging**: JSONL logs perfect for ML training dataset creation

### Opportunities

1. **Signal Classification**: Current strategies produce clean labeled data for supervised learning
2. **Hyperparameter Tuning**: Well-defined config files ready for Bayesian optimization
3. **Multi-Timeframe Features**: Auto-computed aggregates simplify feature engineering
4. **RL Environment**: Existing ensemble_env.py provides foundation for new agents

### Challenges

1. **AMD GPU Support**: RX 6700 XT not officially supported by ROCm (requires workarounds)
2. **Data Volume**: Need to manage ~15GB+ of historical data efficiently
3. **Look-Ahead Bias**: Must carefully handle data normalization during training
4. **Market Regime Shifts**: Models must adapt to changing market conditions

## Next Steps

1. **Review Documentation**: Read remaining documents for detailed analysis
2. **Validate GPU Setup**: Ensure ROCm 6.2 is properly configured
3. **Create Training Pipeline**: Build data loader from TimescaleDB
4. **Implement Phase 1**: Start with XGBoost signal classifier
5. **Iterate and Improve**: Continuous backtesting and refinement

---

**Related Documents**:
- [System Analysis](./01-system-analysis.md)
- [Data Analysis](./02-data-analysis.md)
- [Feature Engineering](./03-feature-engineering.md)
- [AMD GPU Setup](./04-amd-gpu-setup.md)
- [ML Architecture](./05-ml-architecture.md)
- [Implementation Roadmap](./06-implementation-roadmap.md)
- [Model Catalog](./07-model-catalog.md)
