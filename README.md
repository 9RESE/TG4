# TG4 - Local AI/ML Crypto Trading Platform

**Goal**: Accumulate BTC, XRP, RLUSD, USDT, and USDC starting with $1000 USDT + 500 XRP
**Mode**: Paper trading only (no real funds)
**Exchanges**: Kraken, Blofin, Bitrue (spot + margin)
**Hardware**: AMD Ryzen 9 7950X, 128GB DDR5, RX 6700 XT (ROCm-enabled)
**Focus**: Ripple ecosystem (XRP/RLUSD) + stablecoin arbitrage + ML-driven momentum

## Features
- Unified data fetching via ccxt
- Accurate portfolio tracking with margin simulation
- Vectorbt-backed backtesting
- PyTorch ML models (LSTM, Transformers) running on AMD GPU
- Strategies tailored to accumulate target assets

## Quick Start
```bash
git clone https://github.com/9RESE/TG4.git
cd TG4
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python src/main.py --help
```
