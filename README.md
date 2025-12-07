# TG4 - Local AI/ML Crypto Trading Platform

**Phase 11: Bear Profit Tuning + Mean Reversion + USDT Yield**

**Goal**: Accumulate BTC (45%), XRP (35%), USDT (20%) starting with $1000 USDT + 500 XRP
**Mode**: Paper trading only (no real funds)
**Exchanges**: Kraken (10x margin), Bitrue (3x ETFs)
**Hardware**: AMD Ryzen 9 7950X, 128GB DDR5, RX 6700 XT (ROCm-enabled)
**Focus**: Profit in both bull AND bear markets via tuned selective shorting

## Phase 11 Features (Tuned from Phase 10)
- **Lowered Short Thresholds**: ATR >4% (from 5%) + RSI >65 (from 70) for earlier entries
- **VWAP Mean Reversion Filter**: Only short when price is above VWAP
- **USDT Yield Simulation**: ~5-8% APY while parked in defensive mode
- **Stronger Short Rewards**: 5x multiplier on profitable shorts in RL training
- **RSI Auto-Exit**: Close shorts when RSI drops below 40 (mean reversion)
- **Reduced Exposure**: Max 15% portfolio in shorts (from 20%)
- **Goal**: Flip Phase 10's -8.6% to flat/positive via tuned downside capture

## Trading Modes
- **Defensive** (neutral volatility): Parks in USDT, earns simulated yield
- **Offensive** (low volatility <2% + RSI oversold <30): Deploys Kraken 10x / Bitrue 3x long ETFs
- **Bear** (high volatility >4% + RSI overbought >65): Deploys 3-5x shorts on XRP/BTC

## Quick Start
```bash
git clone https://github.com/9RESE/TG4.git
cd TG4
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Train RL agent with Phase 11 tuned rewards (2M timesteps recommended)
python src/main.py --mode train-rl --timesteps 2000000 --device cuda

# Run live paper trading
python src/live_paper.py --interval 15

# Check logs
tail -f logs/trades.csv
tail -f logs/equity_curve.csv
```

## RL Action Space (Phase 11)
| Action | Asset | Type |
|--------|-------|------|
| 0-2 | BTC | buy/hold/sell |
| 3-5 | XRP | buy/hold/sell |
| 6-8 | USDT | park/hold/deploy |
| 9 | BTC | short (via BTC3S ETF) |
| 10 | XRP | short (via Kraken margin) |
| 11 | ALL | close shorts |

## Bear Mode Thresholds (Phase 11 Tuned)
- **Entry**: ATR% > 4% AND RSI > 65 AND confidence > 80% AND price > VWAP
- **Short Leverage**: 3-5x (capped for safety)
- **Max Exposure**: 15% of portfolio (reduced from 20%)
- **Stop Loss**: 8% loss (price rises)
- **Take Profit**: 15% gain (price drops)
- **RSI Exit**: Auto-close when RSI < 40 (mean reversion)
- **Decay Timeout**: Close after 14 days

## Notebooks
- `notebooks/08_defensive_backtest.ipynb` - Phase 8 defensive leverage backtest
- `notebooks/09_rebound_simulation.ipynb` - Phase 9 rebound capture simulation
- `notebooks/10_bear_profit_backtest.ipynb` - Phase 10 bear profit backtest
- `notebooks/11_bear_profit_tuned.ipynb` - Phase 11 tuned bear profit backtest

## Phase History
- **Phase 8**: Volatility-aware risk management, dynamic leverage scaling
- **Phase 9**: Rebound capture, asymmetric offense on dips (-9.2% but +7.2% vs XRP)
- **Phase 10**: Bear market profit engine, selective shorting (-8.6%, +0.6% vs Phase 9)
- **Phase 11**: Tuned bear thresholds, VWAP filter, USDT yield, stronger short rewards
