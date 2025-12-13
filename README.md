# TG4 - Local AI/ML Crypto Trading Platform

**Current Phase**: Phase 21 - WebSocket Paper Tester + Unified Trading System

**Goal**: Accumulate BTC (45%), XRP (35%), USDT (20%) starting with $1000 USDT + 500 XRP
**Mode**: Paper trading only (no real funds)
**Exchanges**: Kraken (10x margin), Bitrue (3x ETFs)
**Hardware**: AMD Ryzen 9 7950X, 128GB DDR5, RX 6700 XT (ROCm-enabled)
**Focus**: Strategy development and testing with real-time WebSocket data

## Project Structure

```
grok-4_1/
├── src/                      # Main Trading Platform (30+ strategies)
│   ├── unified_trader.py     # Production paper trading orchestrator
│   ├── strategies/           # Trading strategies library
│   └── ...
├── ws_paper_tester/          # NEW: WebSocket Paper Tester
│   ├── ws_tester.py          # Lightweight strategy testing
│   ├── strategies/           # Drop-in test strategies
│   └── ...
├── docs/                     # Documentation (Arc42 + Diataxis)
└── config/                   # Configuration files
```

## Systems Overview

### 1. WebSocket Paper Tester (NEW)

Lightweight, WebSocket-native system for rapid strategy development:

```bash
cd ws_paper_tester
pip install -r requirements.txt
python ws_tester.py --simulated   # Test with simulated data
python ws_tester.py               # Live Kraken WebSocket data
```

**Features:**
- Real-time dashboard at http://localhost:8080
- Isolated $100 portfolio per strategy
- Auto-discovery of strategies (drop `.py` in `strategies/`)
- 3 example strategies: market_making, order_flow, mean_reversion

See [WebSocket Paper Tester Guide](docs/user/how-to/websocket-paper-tester.md)

### 2. Unified Trading Platform (src/)

Production-ready paper trading with 30+ strategies:

```bash
python src/unified_trader.py --mode paper
```

### 3. RL Training System

Train reinforcement learning agents for trading:

```bash
python src/main.py --mode train-rl --timesteps 2000000
```

## Phase 14 Features (Live Launch + Dashboard)
- **LiveDashboard Class**: Real-time fear/greed display with regime visualization
- **Softened Thresholds**: RSI >68 (down from 70), ATR >4.2% (down from 4.5%), conf >0.78
- **Auto-Yield Accrual**: 6.5% avg APY compounding on parked USDT
- **Dashboard Integration**: Terminal + optional matplotlib plots
- **Fear/Greed Index**: Volatility + RSI composite score (0-100)
- **Enhanced Logging**: Equity curve with fear/greed tracking
- **Goal**: Capture Dec grind rips with softened thresholds

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

# Train RL agent with Phase 14 tuned rewards (2M timesteps recommended)
python src/main.py --mode train-rl --timesteps 2000000 --device cuda

# Run live paper trading with dashboard
python src/live_paper.py --interval 10

# Run with matplotlib plots (requires display)
python src/live_paper.py --interval 10 --plots

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

## Phase 14 Opportunistic Thresholds
- **Opportunistic Short**: RSI > 68 AND ATR% > 4.2% AND conf > 0.78 AND above VWAP
- **Rip Short**: RSI > 72 AND ATR% > 4% = aggressive 8-12% of portfolio
- **Selective Short**: RSI > 65 AND ATR% > 4% AND above VWAP = 5-8% of portfolio
- **Grind-Down**: ATR% > 4% but no short signal = park USDT + yield
- **Short Leverage**: 3-5x (capped for safety)
- **Max Exposure**: 15% of portfolio in shorts
- **Stop Loss**: 8% loss (price rises)
- **Take Profit**: 15% gain (price drops)
- **RSI Exit**: Auto-close when RSI < 40 (mean reversion)

## Dashboard Features
- Real-time portfolio value display
- Fear/Greed index (0-100 scale)
- RSI and volatility tracking
- Mode distribution summary
- Yield earnings per cycle
- Optional matplotlib plots

## Notebooks
- `notebooks/08_defensive_backtest.ipynb` - Phase 8 defensive leverage backtest
- `notebooks/09_rebound_simulation.ipynb` - Phase 9 rebound capture simulation
- `notebooks/10_bear_profit_backtest.ipynb` - Phase 10 bear profit backtest
- `notebooks/11_bear_profit_tuned.ipynb` - Phase 11 tuned bear profit backtest
- `notebooks/12_bear_flip_backtest.ipynb` - Phase 12 bear profit flip backtest
- `notebooks/13_yield_max_backtest.ipynb` - Phase 13 yield-max + opportunistic shorts
- `notebooks/14_live_prep.ipynb` - Phase 14 live launch preparation

## Phase History
- **Phase 8**: Volatility-aware risk management, dynamic leverage scaling
- **Phase 9**: Rebound capture, asymmetric offense on dips (-9.2% but +7.2% vs XRP)
- **Phase 10**: Bear market profit engine, selective shorting (-8.6%, +0.6% vs Phase 9)
- **Phase 11**: Tuned bear thresholds, VWAP filter, USDT yield, stronger short rewards (-6.3%, +2.3% vs Phase 10)
- **Phase 12**: Paired bear mode (rip vs grind), real USDT yield, short precision rewards (-7.5%, defensive)
- **Phase 13**: YieldManager (6.5% avg APY), opportunistic shorts (RSI >70), 8.0x precision rewards (-7.1%)
- **Phase 14**: Live launch, dashboard, softened thresholds (RSI >68, ATR >4.2%, conf >0.78)
