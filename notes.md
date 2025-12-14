# Project Notes

## WS Paper Tester

### Quick Start
```bash
cd ws_paper_tester
pip install -r requirements.txt
python ws_tester.py                    # Live Kraken WebSocket data
python ws_tester.py --simulated        # Simulated data
python ws_tester.py --duration 30      # Run for 30 minutes
pytest tests/ -v                       # Run tests
```

### Active Strategies

| Strategy | Version | Status | Symbols |
|----------|---------|--------|---------|
| order_flow | 4.1.0 | Production Ready | XRP/USDT, BTC/USDT |
| market_making | 1.5.0 | Production Ready | XRP/USDT, BTC/USDT, XRP/BTC |
| mean_reversion | 2.0.0 | Paper Testing Ready | XRP/USDT, BTC/USDT |
| ratio_trading | 1.0.0 | Under Review | XRP/USDT, BTC/USDT, XRP/BTC |

### Strategy Ideas (Future Development)

- **Scalping (momentum)**: 1m-5m timeframes for quick momentum bursts
- **Arbitrage**: Tick-level for cross-exchange price differences
- **9 Week MA Strategy**: BTC and XRP USDT pairs with 9-week moving average on 5m and 1h. Close position on candle opposite trend, 2 candles above/below MA indicates trend
- **BTC-XRP Correlation**: Investigate if XRP follows BTC predictably enough to trade on

### Documentation Structure

```
ws_paper_tester/docs/
├── development/
│   ├── features/         # Version release notes
│   │   ├── order_flow/
│   │   ├── market_maker/
│   │   └── mean_reversion/
│   └── review/           # Deep strategy reviews
│       ├── order_flow/
│       ├── market_maker/
│       └── mean_reversion/
└── user/
    └── how-to/           # User guides
```
