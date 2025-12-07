# TG4 - Local AI/ML Crypto Trading Platform

**Phase 9: Rebound Capture + Selective Offense + Live Paper Deployment**

**Goal**: Accumulate BTC (45%), XRP (35%), USDT (20%) starting with $1000 USDT + 500 XRP
**Mode**: Paper trading only (no real funds)
**Exchanges**: Kraken (10x margin), Bitrue (3x ETFs)
**Hardware**: AMD Ryzen 9 7950X, 128GB DDR5, RX 6700 XT (ROCm-enabled)
**Focus**: Asymmetric offense - defensive in high-vol, aggressive on dips

## Phase 9 Features
- **Asymmetric Offense**: Deploys leverage only in greed regime (ATR <2% + RSI oversold)
- **LSTM Dip Detection**: Hybrid LSTM + RSI confirmation for leverage trades
- **Volatility-Aware Risk**: Dynamic leverage scaling (1-10x based on ATR%)
- **Live Paper Trading**: 15-minute cycles with CSV logging
- **PPO Reinforcement Learning**: Rebound bonus rewards for correctly-timed dips

## Trading Modes
- **Defensive** (high volatility >5%): Parks in USDT, no leverage
- **Offensive** (low volatility <2% + dip signal): Deploys Kraken 10x / Bitrue 3x ETFs

## Quick Start
```bash
git clone https://github.com/9RESE/TG4.git
cd TG4
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Train RL agent (2M timesteps recommended)
python src/main.py --mode train-rl --timesteps 2000000 --device cuda

# Run live paper trading
python src/live_paper.py --interval 15

# Check logs
tail -f logs/trades.csv
tail -f logs/equity_curve.csv
```

## Notebooks
- `notebooks/08_defensive_backtest.ipynb` - Phase 8 defensive leverage backtest
- `notebooks/09_rebound_simulation.ipynb` - Phase 9 rebound capture simulation
