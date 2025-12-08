# TG4 - Local AI/ML Crypto Trading Platform

**Phase 13: Yield-Max + Opportunistic Shorts + Real Rate Pull + Live Launch**

**Goal**: Accumulate BTC (45%), XRP (35%), USDT (20%) starting with $1000 USDT + 500 XRP
**Mode**: Paper trading only (no real funds)
**Exchanges**: Kraken (10x margin), Bitrue (3x ETFs)
**Hardware**: AMD Ryzen 9 7950X, 128GB DDR5, RX 6700 XT (ROCm-enabled)
**Focus**: Target -4% or better in bears via amplified yield + opportunistic shorts

## Phase 13 Features (Yield-Max + Opportunistic Shorts)
- **YieldManager Class**: Real rate simulation (Kraken 6% + Bitrue 7% = 6.5% avg APY)
- **Opportunistic Shorts**: RSI >70 spike in grind-down mode (ATR >4.5%)
- **Higher Precision Rewards**: 8.0x multiplier for whipsaw-free profitable shorts (up from 6.0x)
- **Paired Bear Mode**: Rip (RSI >72) = aggressive, Opportunistic (RSI >70) = conservative
- **10-Minute Live Loop**: Faster yield accrual and position management
- **Yield Logging**: Track total yield earned in equity curve CSV
- **Goal**: Push toward -4% or better via yield + selective short PNL

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
- `notebooks/13_yield_max_backtest.ipynb` - Phase 13 yield-max + opportunistic shorts

## Phase History
- **Phase 8**: Volatility-aware risk management, dynamic leverage scaling
- **Phase 9**: Rebound capture, asymmetric offense on dips (-9.2% but +7.2% vs XRP)
- **Phase 10**: Bear market profit engine, selective shorting (-8.6%, +0.6% vs Phase 9)
- **Phase 11**: Tuned bear thresholds, VWAP filter, USDT yield, stronger short rewards (-6.3%, +2.3% vs Phase 10)
- **Phase 12**: Paired bear mode (rip vs grind), real USDT yield, short precision rewards (-7.5%, defensive)
- **Phase 13**: YieldManager (6.5% avg APY), opportunistic shorts (RSI >70), 8.0x precision rewards
