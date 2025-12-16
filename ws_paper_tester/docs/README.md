# WebSocket Paper Tester Documentation

**Version:** 1.15.1
**Last Updated:** 2025-12-15

Welcome to the WebSocket Paper Tester documentation. This system provides real-time paper trading with Kraken WebSocket data and comprehensive strategy backtesting capabilities.

---

## Quick Navigation

### For Users

| Document | Description |
|----------|-------------|
| [Configure Paper Tester](user/how-to/configure-paper-tester.md) | Configuration guide for all settings |
| [Operate Historical Data](user/how-to/operate-historical-data.md) | Historical data system operations |

### For Developers

| Document | Description |
|----------|-------------|
| [Strategy Development Guide](development/strategy-development-guide.md) | Complete guide to writing trading strategies |
| [Code Review Issues](CODE_REVIEW_ISSUES.md) | Issue tracking and resolution status |

### Feature Documentation

Located in `development/features/`:

| Strategy | Latest Version |
|----------|---------------|
| [Mean Reversion](development/features/mean_reversion/) | v4.2 |
| [Market Making](development/features/market_maker/) | v2.2 |
| [Order Flow](development/features/order_flow/) | v5.0 |
| [Ratio Trading](development/features/ratio_trading/) | v4.3 |
| [WaveTrend](development/features/wavetrend/) | v1.1 |
| [Whale Sentiment](development/features/whale_sentiment/) | v1.6 |
| [Grid RSI Reversion](development/features/grid_rsi_reversion/) | v1.0 |
| [Momentum Scalping](development/features/momentum_scalping/) | v2.0 |

### System Components

| Component | Documentation |
|-----------|---------------|
| [Indicator Library](development/features/indicators/indicator-library-v1.0.md) | Centralized indicator functions |
| [Regime Detection](development/features/regime_detection/regime-detection-v1.0.md) | Market regime classification |
| [Historical Data](development/features/historical-data-system/historical-data-system-v1.0.md) | TimescaleDB data storage |

---

## Key Features

### Trading Execution
- Real-time paper trading with Kraken WebSocket v2
- Simulated mode for offline testing
- Configurable fees and slippage
- **Leveraged positions** (longs up to 1.5x, shorts up to 2x)
- **Margin call liquidation** (25% maintenance margin)

### Strategy Framework
- Auto-discovery of strategy modules
- Per-strategy isolated portfolios
- Lifecycle callbacks (`on_start`, `on_fill`, `on_stop`)
- Signal rejection tracking
- Circuit breaker protection

### Data Infrastructure
- TimescaleDB historical data storage
- Multi-timeframe candle aggregation
- Gap detection and filling
- External indicators (Fear & Greed, BTC Dominance)

### Monitoring
- Web dashboard with real-time updates
- Structured JSON logging
- Per-pair P&L tracking
- Strategy performance metrics

---

## Getting Started

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run with simulated data (quick test)
python ws_tester.py --duration 5 --simulated

# 3. Run with live Kraken data
python ws_tester.py --duration 60

# 4. View dashboard
# Open http://127.0.0.1:8787 in browser
```

See [Configure Paper Tester](user/how-to/configure-paper-tester.md) for detailed configuration.

---

## Documentation Structure

This documentation follows two frameworks:
- **[Arc42](https://arc42.org/)** for technical architecture (`architecture/`)
- **[Diataxis](https://diataxis.fr/)** for user documentation (`user/`)

```
docs/
├── README.md                           # This file (entry point)
├── CODE_REVIEW_ISSUES.md               # Issue tracking
│
├── architecture/                       # Arc42 Architecture Docs
│   ├── README.md                       # Architecture overview
│   ├── 01-introduction-and-goals.md    # Requirements, stakeholders
│   ├── 05-building-block-view.md       # Component breakdown
│   └── 09-architecture-decisions.md    # ADRs
│
├── user/                               # Diataxis User Docs
│   ├── tutorials/                      # Learning-oriented
│   │   ├── README.md
│   │   └── 01-first-paper-trading-session.md
│   ├── how-to/                         # Task-oriented
│   │   ├── configure-paper-tester.md
│   │   └── operate-historical-data.md
│   ├── reference/                      # Information-oriented
│   │   ├── README.md
│   │   ├── signal.md
│   │   └── cli.md
│   └── explanation/                    # Understanding-oriented
│       ├── README.md
│       └── how-leverage-works.md
│
└── development/                        # Developer Docs
    ├── strategy-development-guide.md   # Main dev guide
    ├── features/                       # Feature documentation
    │   ├── README.md                   # Version index
    │   ├── mean_reversion/
    │   ├── market_maker/
    │   ├── order_flow/
    │   ├── ratio_trading/
    │   ├── wavetrend/
    │   ├── whale_sentiment/
    │   ├── grid_rsi_reversion/
    │   ├── momentum_scalping/
    │   ├── indicators/
    │   ├── regime_detection/
    │   └── historical-data-system/
    ├── review/                         # Strategy reviews
    └── plans/                          # Design documents
```

---

## Changelog

See [CHANGELOG.md](../CHANGELOG.md) for version history and release notes.

---

*WebSocket Paper Tester v1.15.1*
