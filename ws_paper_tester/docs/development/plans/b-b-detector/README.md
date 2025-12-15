# Bull/Bear/Sideways Market Regime Detector

## Executive Summary

This document outlines the research, design, and implementation plan for a **Market Regime Detection System** that will analyze market conditions across multiple token pairs (XRP/USDT, BTC/USDT, XRP/BTC) and provide actionable regime classifications that strategies can use to optimize their parameters.

## Problem Statement

Current strategies operate with static parameters regardless of market conditions. A strategy optimized for trending markets (like momentum strategies) performs poorly in sideways markets, while mean-reversion strategies fail during strong trends. By detecting the current market regime, strategies can:

1. **Adapt parameters** - Adjust position sizes, stop-losses, and entry thresholds
2. **Select appropriate strategies** - Enable/disable strategies based on conditions
3. **Reduce drawdowns** - Avoid trading in unfavorable conditions
4. **Improve win rates** - Trade only when conditions favor the strategy type

## Market Regimes Defined

| Regime | Characteristics | Optimal Strategies |
|--------|----------------|-------------------|
| **Bull Trending** | Higher highs, higher lows, ADX > 25, positive momentum | Trend following, momentum, breakout |
| **Bear Trending** | Lower highs, lower lows, ADX > 25, negative momentum | Short selling, trend following (short) |
| **Sideways/Range** | No clear direction, ADX < 20, bounded price action | Mean reversion, grid trading, market making |
| **High Volatility** | ATR > 1.5x average, rapid price swings | Scalping, reduced position sizes |
| **Low Volatility** | ATR < 0.5x average, compressed ranges | Accumulation, larger positions |

## Proposed Solution Architecture

### Multi-Layer Detection System

```
Layer 1: Per-Symbol Analysis
    [XRP/USDT Regime] [BTC/USDT Regime] [XRP/BTC Regime]
              ↓               ↓               ↓
Layer 2: Cross-Asset Correlation Analysis
    [BTC Dominance] [Correlation Matrix] [Relative Strength]
              ↓               ↓               ↓
Layer 3: Market-Wide Regime Classification
    [Overall Market Regime] + [Confidence Score]
              ↓
Layer 4: Strategy Parameter Adjustment
    [Strategy-Specific Config Overrides]
```

### Core Components

1. **Technical Indicator Engine** - ADX, RSI, MACD, Choppiness Index, Bollinger Bands
2. **Multi-Timeframe Analyzer** - 1m, 5m, 15m, 1h, 4h confluence scoring
3. **External Data Integrator** - Fear & Greed Index, BTC Dominance
4. **Composite Scoring System** - Weighted regime classification
5. **Strategy Parameter Router** - Dynamic config adjustment

## Key Metrics

### Success Criteria

- Regime detection accuracy > 70% (backtested)
- Strategy performance improvement > 15% with adaptive parameters
- False regime transitions < 3 per day (avoid whipsaw)
- Latency < 100ms for regime updates

### Monitoring

- Regime transition frequency
- Time spent in each regime
- Strategy performance by regime
- Correlation with actual price movements

## Documentation Structure

| Document | Description |
|----------|-------------|
| [research-findings.md](./research-findings.md) | Detailed algorithm research |
| [architecture-design.md](./architecture-design.md) | Technical architecture |
| [implementation-plan.md](./implementation-plan.md) | Step-by-step implementation |
| [data-sources.md](./data-sources.md) | Available data APIs |
| [strategy-integration.md](./strategy-integration.md) | How strategies consume regime data |

## Timeline Estimate

- **Phase 1**: Core indicator engine (ADX, RSI, CHOP, trend slope)
- **Phase 2**: Multi-timeframe analysis and scoring
- **Phase 3**: External data integration (Fear & Greed, BTC Dominance)
- **Phase 4**: Strategy parameter routing
- **Phase 5**: Backtesting and optimization

## References

- [LuxAlgo - Market Regimes Explained](https://www.luxalgo.com/blog/market-regimes-explained-build-winning-trading-strategies/)
- [QuantStart - HMM Market Regime Detection](https://www.quantstart.com/articles/market-regime-detection-using-hidden-markov-models-in-qstrader/)
- [Kraken API Documentation](https://docs.kraken.com/api/)
- [Alternative.me Fear & Greed API](https://alternative.me/crypto/fear-and-greed-index/)

---

*Version: 1.0.0 | Created: 2025-12-15*
