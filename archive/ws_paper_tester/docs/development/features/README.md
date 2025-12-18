# Feature Documentation

This directory contains feature documentation for all strategies and system components.

---

## Strategy Features (Current Versions)

| Strategy | Current Version | Latest Doc | Description |
|----------|-----------------|------------|-------------|
| Mean Reversion | v4.2 | [mean-reversion-v4.2.md](mean_reversion/mean-reversion-v4.2.md) | RSI/SMA mean reversion with trailing stops |
| Market Making | v2.2 | [market-making-v2.2.md](market_maker/market-making-v2.2.md) | Spread capture with Avellaneda-Stoikov model |
| Order Flow | v5.0 | [order-flow-v5.0.md](order_flow/order-flow-v5.0.md) | Trade flow imbalance momentum |
| Ratio Trading | v4.3 | [ratio-trading-v4.3.md](ratio_trading/ratio-trading-v4.3.md) | XRP/BTC pair mean reversion |
| WaveTrend | v1.1 | [wavetrend-v1.1.md](wavetrend/wavetrend-v1.1.md) | WaveTrend oscillator crossover |
| Whale Sentiment | v1.6 | [whale-sentiment-v1.6.md](whale_sentiment/whale-sentiment-v1.6.md) | Volume spike contrarian trading |
| Grid RSI Reversion | v1.0 | [grid-rsi-reversion-v1.0.md](grid_rsi_reversion/grid-rsi-reversion-v1.0.md) | Grid trading with RSI confirmation |
| Momentum Scalping | v2.0 | [momentum-scalping-v2.0.md](momentum_scalping/momentum-scalping-v2.0.md) | Short-term momentum capture |

## System Components (Current Versions)

| Component | Current Version | Latest Doc | Description |
|-----------|-----------------|------------|-------------|
| Indicator Library | v1.0 | [indicator-library-v1.0.md](indicators/indicator-library-v1.0.md) | Centralized indicator functions |
| Regime Detection | v1.0 | [regime-detection-v1.0.md](regime_detection/regime-detection-v1.0.md) | Market regime classification |
| Historical Data | v1.0 | [historical-data-system-v1.0.md](historical-data-system/historical-data-system-v1.0.md) | TimescaleDB data storage |

---

## Version History

Each strategy directory contains version history documents (v1.0, v2.0, etc.). These are kept for:
- Tracking feature evolution
- Understanding design decisions
- Rolling back if needed

**Convention:** The highest version number is always the current implementation.

---

## Directory Structure

```
features/
├── README.md                    # This file (index)
├── mean_reversion/
│   ├── mean-reversion-v2.0.md   # Historical
│   ├── mean-reversion-v3.0.md   # Historical
│   ├── mean-reversion-v4.0.md   # Historical
│   └── mean-reversion-v4.2.md   # CURRENT
├── market_maker/
│   ├── market-making-v1.3.md    # Historical
│   ├── ...
│   └── market-making-v2.2.md    # CURRENT
├── order_flow/
│   └── ...
├── ratio_trading/
│   └── ...
├── wavetrend/
│   └── ...
├── whale_sentiment/
│   └── ...
├── grid_rsi_reversion/
│   └── ...
├── momentum_scalping/
│   └── ...
├── indicators/
│   └── indicator-library-v1.0.md
├── regime_detection/
│   └── regime-detection-v1.0.md
└── historical-data-system/
    └── historical-data-system-v1.0.md
```

---

*Last Updated: 2025-12-15*
