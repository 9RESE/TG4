# Project Notes

## Quick Start

```bash
cd ws_paper_tester
# Install dependencies
pip install -r requirements.txt
# Run with live Kraken WebSocket data
python ws_tester.py
# Run with simulated data
python ws_tester.py --simulated
# Run for 30 minutes with dashboard disabled
python ws_tester.py --duration 30 --no-dashboard
# Run tests
pytest tests/ -v
```

## Current Strategies

| Strategy | Version | Symbols | Status |
|----------|---------|---------|--------|
| market_making | v1.5.0 | XRP/USDT, BTC/USDT, XRP/BTC | Active |
| mean_reversion | v3.0.0 | XRP/USDT, BTC/USDT, XRP/BTC | Active |
| order_flow | v4.1.0 | XRP/USDT, BTC/USDT | Active |
| ratio_trading | v2.1.0 | XRP/BTC | Active |

## Strategy Ideas (Backlog)

| Strategy Type | Timeframe | Description |
|--------------|-----------|-------------|
| Scalping (momentum) | 1m-5m | Quick momentum bursts |
| Arbitrage | Tick-level | Cross-exchange price differences |

## Research Notes

### XRP/BTC Correlation
- Does XRP follow BTC predictably enough to trade on?
- See ratio_trading strategy for implementation

### Moving Average Strategy
- 9-week MA on 5m and 1h candles
- One candle opposite trend closes position
- 2 candles closed above/below MA = trend signal
- Trend definition open to improvement
