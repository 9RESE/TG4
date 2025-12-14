# Trading Strategies Reference

Technical reference for all trading strategies in the WebSocket Paper Tester.

## Strategy Overview

| Strategy | Version | Pairs | Type | Description |
|----------|---------|-------|------|-------------|
| [Mean Reversion](#mean-reversion) | 4.3.0 | XRP/USDT, BTC/USDT, XRP/BTC | Counter-trend | Trades price deviations from moving average |
| [Ratio Trading](#ratio-trading) | 4.2.1 | XRP/BTC | Statistical arbitrage | Trades ratio deviations between correlated pairs |
| [Order Flow](#order-flow) | 4.1.0 | XRP/USDT | Momentum | Trades based on order book imbalance |
| [Market Making](#market-making) | 1.5.0 | XRP/USDT | Liquidity provision | Provides liquidity with bid-ask spreads |

---

## Mean Reversion

**Location:** `ws_paper_tester/strategies/mean_reversion/`

### Module Structure

```
mean_reversion/
├── __init__.py      # Public exports
├── config.py        # Configuration and enums
├── indicators.py    # Technical indicator calculations
├── regimes.py       # Volatility regime classification
├── risk.py          # Risk management functions
├── signals.py       # Signal generation logic
└── lifecycle.py     # Lifecycle callbacks
```

### Strategy Description

Trades price deviations from moving average and VWAP using Bollinger Bands and RSI for confirmation. Best suited for range-bound markets with moderate volatility (0.3-1.0%).

### Supported Pairs

| Pair | Position Size | Take Profit | Stop Loss | Notes |
|------|---------------|-------------|-----------|-------|
| XRP/USDT | $20 | 0.5% | 0.5% | Primary pair |
| BTC/USDT | $25 | 0.4% | 0.4% | Reduced size (market conditions) |
| XRP/BTC | $15 | 0.8% | 0.8% | Ratio trading, wider spreads |

### Key Configuration

```python
CONFIG = {
    # Core Parameters
    'lookback_candles': 20,
    'deviation_threshold': 0.5,  # % deviation to trigger
    'bb_period': 20,
    'bb_std_dev': 2.0,
    'rsi_period': 14,
    'rsi_oversold': 35,
    'rsi_overbought': 65,

    # Risk Management
    'take_profit_pct': 0.5,
    'stop_loss_pct': 0.5,
    'max_consecutive_losses': 3,
    'circuit_breaker_minutes': 15,

    # Volatility Regimes
    'use_volatility_regimes': True,
    'regime_low_threshold': 0.3,
    'regime_medium_threshold': 0.8,
    'regime_high_threshold': 1.5,

    # Correlation Monitoring (v4.2.0, updated v4.3.0)
    'use_correlation_monitoring': True,
    'correlation_warn_threshold': 0.55,
    'correlation_pause_threshold': 0.5,   # Raised from 0.25 per REC-001/002

    # ADX Filter for BTC (v4.3.0)
    'use_adx_filter': True,
    'adx_period': 14,
    'adx_strong_trend_threshold': 25,     # Pause BTC when ADX > 25
}
```

### Market Conditions to Pause

- Fear & Greed Index < 25 (Extreme Fear)
- ADX > 25 for BTC/USDT (strong trend) - v4.3.0 REC-003
- ADX > 30 (strong trend for other pairs)
- XRP/BTC correlation < 0.5 (raised from 0.25 in v4.3.0 REC-001/002)

### Version History

| Version | Date | Changes |
|---------|------|---------|
| 4.3.0 | 2025-12-14 | Deep Review v8.0: ADX filter for BTC (REC-003), correlation threshold raised to 0.5 (REC-001/002) |
| 4.2.1 | 2025-12-14 | Refactored into modular package structure |
| 4.2.0 | 2025-12-14 | Correlation pause threshold (0.25), tightened warn (0.4) |
| 4.1.0 | 2025-12-13 | Fee profitability checks, reduced BTC position size |
| 4.0.0 | 2025-12-13 | Trailing stops disabled, extended decay timing, correlation monitoring |
| 3.0.0 | 2025-12-12 | XRP/BTC pair, trend filter, trailing stops, position decay |
| 2.0.0 | 2025-12-11 | Multi-symbol support, circuit breaker, volatility regimes |

---

## Ratio Trading

**Location:** `ws_paper_tester/strategies/ratio_trading/`

### Module Structure

```
ratio_trading/
├── __init__.py      # Public exports
├── config.py        # Configuration
├── enums.py         # Enums (ExitReason, RejectionReason)
├── indicators.py    # Z-score and technical indicators
├── regimes.py       # Volatility regime classification
├── risk.py          # Risk management
├── signals.py       # Signal generation
├── tracking.py      # Dual-asset tracking
└── lifecycle.py     # Lifecycle callbacks
```

### Strategy Description

Statistical arbitrage strategy trading the XRP/BTC ratio. Uses Z-score of the ratio to identify deviations from the mean. **Only suitable for XRP/BTC pair** - USDT pairs are NOT appropriate for ratio trading.

### Supported Pairs

| Pair | Position Size | Z-Score Entry | Z-Score Exit | Notes |
|------|---------------|---------------|--------------|-------|
| XRP/BTC | $15-25 | 2.0 | 0.5 | Only supported pair |

### Key Configuration

```python
CONFIG = {
    # Core Parameters
    'lookback_candles': 100,
    'zscore_entry': 2.0,
    'zscore_exit': 0.5,

    # Risk Management
    'take_profit_pct': 1.0,
    'stop_loss_pct': 1.5,
    'max_position_usd': 50.0,

    # Correlation Protection (v4.0.0+)
    'use_correlation_protection': True,
    'correlation_threshold': 0.3,
    'correlation_lookback': 50,
}
```

### Theory

Ratio trading assumes:
1. XRP and BTC prices are cointegrated (long-term relationship)
2. Ratio deviations are temporary and mean-revert
3. Correlation remains stable (> 0.3)

**When to pause:**
- Correlation drops below threshold (currently at ~40%, down from ~80%)
- Major regulatory events affecting one asset
- Extreme market stress

### Version History

| Version | Date | Changes |
|---------|------|---------|
| 4.2.1 | 2025-12-14 | Refactored into modular package structure |
| 4.2.0 | 2025-12-14 | Deep review v7.0 recommendations |
| 4.1.0 | 2025-12-13 | Deep review v6.0 recommendations |
| 4.0.0 | 2025-12-12 | Correlation protection enabled by default |
| 3.0.0 | 2025-12-12 | Review recommendations and enhancements |
| 2.0.0 | 2025-12-11 | Major refactor with review recommendations |

---

## Order Flow

**Location:** `ws_paper_tester/strategies/order_flow.py`

### Strategy Description

Momentum strategy based on order book imbalance. Trades when buy/sell pressure exceeds thresholds.

### Key Configuration

```python
CONFIG = {
    'imbalance_threshold': 0.6,
    'position_size_usd': 20.0,
    'take_profit_pct': 0.3,
    'stop_loss_pct': 0.2,
}
```

### Version History

| Version | Date | Changes |
|---------|------|---------|
| 4.1.0 | 2025-12-11 | Review recommendations implementation |
| 3.1.0 | 2025-12-10 | Bug fixes and asymmetric thresholds |

---

## Market Making

**Location:** `ws_paper_tester/strategies/market_making.py`

### Strategy Description

Provides liquidity by placing limit orders on both sides of the spread. Profits from bid-ask spread while managing inventory risk.

### Key Configuration

```python
CONFIG = {
    'spread_pct': 0.1,
    'order_size_usd': 10.0,
    'max_inventory': 100.0,
    'inventory_skew': 0.5,
}
```

### Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.5.0 | 2025-12-10 | Fee-aware trading and comprehensive improvements |

---

## Common Features

All strategies share these capabilities:

### Volatility Regimes
- **LOW**: < 0.3% - Tighter thresholds
- **MEDIUM**: 0.3-0.8% - Normal operation
- **HIGH**: 0.8-1.5% - Wider thresholds, smaller sizes
- **EXTREME**: > 1.5% - Trading paused

### Circuit Breaker
Pauses trading after consecutive losses (default: 3 losses, 15-minute cooldown).

### Signal Rejection Tracking
All rejected signals are tracked by reason for analysis:
- `CIRCUIT_BREAKER` - Circuit breaker active
- `TIME_COOLDOWN` - Cooldown between signals
- `REGIME_PAUSE` - Extreme volatility
- `TRENDING_MARKET` - Market unsuitable for mean reversion
- `LOW_CORRELATION` - Correlation below threshold

### Fee Profitability Checks
Validates expected profit exceeds round-trip fees (~0.2%) before generating signals.

---

## Strategy Development Guide

For creating new strategies, see:
- [How to Create a Strategy](../how-to/create-strategy.md)
- [Strategy Development Guide](../../development/features/strategy-development-guide.md) (internal)

For strategy reviews and research, see:
- `ws_paper_tester/docs/development/review/` - Review documents by strategy

---

*Last updated: 2025-12-14*
