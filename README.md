# TG4 - Local AI/ML Crypto Trading Platform

**Phase 12: Bear Profit Flip - Paired Shorts + Real Yield + Live Prep**

**Goal**: Accumulate BTC (45%), XRP (35%), USDT (20%) starting with $1000 USDT + 500 XRP
**Mode**: Paper trading only (no real funds)
**Exchanges**: Kraken (10x margin), Bitrue (3x ETFs)
**Hardware**: AMD Ryzen 9 7950X, 128GB DDR5, RX 6700 XT (ROCm-enabled)
**Focus**: Flat/positive in bears via yield compounding + precision shorts

## Phase 12 Features (Bear Profit Flip)
- **Paired Bear Mode**: Overbought rip (RSI >72) = aggressive short, Grind-down = park + yield
- **Real USDT Yield**: 6% APY from Kraken/Bitrue lending (applied every 4 hours in backtest)
- **Short Precision Rewards**: 6.0x multiplier for whipsaw-free profitable shorts
- **Yield Factor Doubling**: 2x reward factor for defensive USDT parking
- **Rip Short Threshold**: RSI >72 triggers aggressive shorts (8-12% of portfolio)
- **Lower Decay Penalty**: 0.3 (from 0.5) to encourage timely exits, not panic
- **Goal**: Flip Phase 11's -6.3% to flat/positive via yield + precision shorts

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

## Bear Mode Thresholds (Phase 12 Paired)
- **Rip Short**: ATR% > 4% AND RSI > 72 = aggressive 8-12% of portfolio
- **Selective Short**: ATR% > 4% AND RSI > 65 AND price > VWAP = 5-8% of portfolio
- **Grind-Down**: ATR% > 4% but no short signal = park USDT + yield
- **Short Leverage**: 3-5x (capped for safety)
- **Max Exposure**: 15% of portfolio in shorts
- **Stop Loss**: 8% loss (price rises)
- **Take Profit**: 15% gain (price drops)
- **RSI Exit**: Auto-close when RSI < 40 (mean reversion)
- **Decay Timeout**: Close after 14 days

## Notebooks
- `notebooks/08_defensive_backtest.ipynb` - Phase 8 defensive leverage backtest
- `notebooks/09_rebound_simulation.ipynb` - Phase 9 rebound capture simulation
- `notebooks/10_bear_profit_backtest.ipynb` - Phase 10 bear profit backtest
- `notebooks/11_bear_profit_tuned.ipynb` - Phase 11 tuned bear profit backtest
- `notebooks/12_bear_flip_backtest.ipynb` - Phase 12 bear profit flip backtest

## Phase History
- **Phase 8**: Volatility-aware risk management, dynamic leverage scaling
- **Phase 9**: Rebound capture, asymmetric offense on dips (-9.2% but +7.2% vs XRP)
- **Phase 10**: Bear market profit engine, selective shorting (-8.6%, +0.6% vs Phase 9)
- **Phase 11**: Tuned bear thresholds, VWAP filter, USDT yield, stronger short rewards (-6.3%, +2.3% vs Phase 10)
- **Phase 12**: Paired bear mode (rip vs grind), real USDT yield, short precision rewards
