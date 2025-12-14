# Trading Bots Development Notes

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

## Strategy Development Ideas

### Current Strategies
- **order_flow** (v4.1.0) - Trade tape analysis with VPIN, volatility regimes, session awareness
- **market_maker** (v1.5.0) - Market making with fee-aware trading
- **ratio** - XRP/BTC ratio trading
- **mean_reversion** - Mean reversion strategy

### Future Strategy Ideas
- 9-week moving average on 5min and 1hr timeframes
- XRP/BTC correlation-based trading
- Scalping (momentum) - 1m-5m quick momentum bursts
- Arbitrage - Cross-exchange price differences

## Documentation Structure

```
ws_paper_tester/docs/
├── development/
│   ├── features/      # Feature documentation by strategy
│   │   ├── order_flow/
│   │   ├── market_maker/
│   │   └── ratio/
│   └── review/        # Strategy reviews
│       ├── order_flow/
│       ├── market_maker/
│       └── mean_reversion/
└── user/
    └── how-to/        # User guides
```

## Recent Work

### 2025-12-14
- Order Flow Strategy v4.1.0: Implemented review recommendations
  - Signal rejection logging and statistics
  - Configuration override validation
  - Configurable session boundaries
  - Enhanced position decay with profit-after-fees option
  - Improved VPIN bucket logic
- Order Flow Strategy Review v4.0.0: Deep code and strategy analysis

### 2025-12-13
- Order Flow Strategy v3.1.0: Bug fixes and asymmetric thresholds
- Order Flow Strategy Review v3.1.0: Deep code analysis
- Market Making v1.5.0: Fee-aware trading improvements
