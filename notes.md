# Development Notes

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

## Strategy Ideas (Future Development)

### Moving Average Trend Strategy
- BTC and XRP USDT pairs
- 9-week moving average on 5m and 1h timeframes
- One candle opposite the trend closes position
- 2 candles closed above/below the 9 MA defines a trend
- Trend definition equation is open to improvement

### XRP/BTC Correlation Strategy
- Research question: Does XRP follow BTC predictably enough to trade on?
- Could inform ratio trading or lead/lag strategies

### Additional Strategy Types to Explore
| Strategy | Timeframe | Description |
|----------|-----------|-------------|
| Scalping (momentum) | 1m-5m | Quick momentum bursts |
| Arbitrage | Tick-level | Cross-exchange price differences |
