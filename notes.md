# WS Paper Tester - Development Notes

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

---

## Current Version: v1.5.0

**Release Date:** 2025-12-13

### Market Making Strategy v1.5.0 Features

| Feature | Description |
|---------|-------------|
| MM-E03 | Fee-aware profitability check |
| MM-009 | R:R ratios adjusted to 1:1 |
| MM-E01 | Micro-price calculation |
| MM-E02 | A-S optimal spread calculation |
| MM-010 | Refactored signal generation |
| MM-011 | Configurable fallback prices |
| MM-E04 | Time-based position decay |

### Trading Pairs

| Pair | Type | Notes |
|------|------|-------|
| XRP/USDT | Primary | Most trades |
| BTC/USDT | Primary | High liquidity |
| XRP/BTC | Cross-pair | Dual-asset accumulation |

---

## Strategies

### 1. Market Making (market_making.py) - v1.5.0

**Purpose:** Provides liquidity by trading both sides of the spread

**Key Features:**
- Fee-aware profitability checks
- Micro-price for better price discovery
- Volatility-adjusted spreads
- Trade flow confirmation
- Trailing stops
- Position decay for stale trades

**Signal Logic:**
| Condition | Action |
|-----------|--------|
| Long inventory + sell pressure | Sell to reduce |
| Short inventory + buy pressure | Buy to cover |
| Wide spread + bid imbalance + profitable after fees | Buy |
| Wide spread + ask imbalance + profitable after fees | Sell/Short |
| Stale position with partial profit | Exit with reduced TP |

**Config Defaults (v1.5.0):**
```yaml
min_spread_pct: 0.1
position_size_usd: 20
max_inventory: 100
take_profit_pct: 0.5      # Improved from 0.4 (1:1 R:R)
stop_loss_pct: 0.5
use_fee_check: true
fee_rate: 0.001
use_micro_price: true
use_position_decay: true
```

**Symbols:** XRP/USDT, BTC/USDT, XRP/BTC

---

### 2. Order Flow (order_flow.py) - v2.1

**Purpose:** Trades based on trade tape analysis and buy/sell imbalance

**Config Defaults:**
```yaml
imbalance_threshold: 0.3
volume_spike_mult: 2.0
position_size_usd: 25
lookback_trades: 50
cooldown_trades: 10
take_profit_pct: 0.5
stop_loss_pct: 0.3
```

**Symbols:** XRP/USDT, BTC/USDT

---

### 3. Mean Reversion (mean_reversion.py)

**Purpose:** Trades price deviations from moving average and VWAP

**Config Defaults:**
```yaml
lookback_candles: 20
deviation_threshold: 0.5
position_size_usd: 20
rsi_oversold: 35
rsi_overbought: 65
take_profit_pct: 0.4
stop_loss_pct: 0.6
max_position: 50
```

**Symbols:** XRP/USDT

---

### 4. Ratio Trading (ratio.py) - v1.0

**Purpose:** Trade XRP/BTC ratio for dual-asset accumulation

**Symbols:** XRP/BTC

---

## Strategy Comparison

| Strategy | Style | Data Used | Pairs | Risk |
|----------|-------|-----------|-------|------|
| Market Making | Scalping | Orderbook | XRP/USDT, BTC/USDT, XRP/BTC | Low |
| Order Flow | Momentum | Trade tape | XRP/USDT, BTC/USDT | Medium |
| Mean Reversion | Counter-trend | Candles/VWAP | XRP/USDT | Medium |
| Ratio Trading | Dual-accumulation | Price ratio | XRP/BTC | Low-Medium |

---

## Documentation Structure

```
docs/
├── development/
│   ├── features/
│   │   ├── market_maker/
│   │   │   ├── market-making-v1.3.md
│   │   │   ├── market-making-v1.4.md
│   │   │   └── market-making-v1.5.md
│   │   ├── order_flow/
│   │   │   └── order-flow-v2.1.md
│   │   └── ratio/
│   │       └── ratio-trading-v1.0.md
│   └── review/
│       └── market_maker/
│           ├── strategy-development-guide.md
│           ├── market-making-strategy-review.md
│           ├── market-making-strategy-review-v1.2.md
│           ├── market-making-strategy-review-v1.3.md
│           └── market-making-strategy-review-v1.4.md
└── user/
    └── how-to/
        └── configure-paper-tester.md
```

---

## Version History

| Version | Date | Description |
|---------|------|-------------|
| v1.5.0 | 2025-12-13 | Fee-aware trading, micro-price, position decay |
| v1.4.0 | 2025-12-13 | Per-pair PnL tracking, trailing stops |
| v1.3.0 | 2025-12-12 | Major strategy review improvements |
| v1.2.0 | 2025-12-11 | BTC/USDT support |
| v1.1.0 | 2025-12-10 | XRP/BTC ratio trading |
| v1.0.0 | 2025-12-09 | Initial release |

---

## Future Ideas

- 9-week moving average strategy on 5min and 1hr
- XRP follows BTC correlation trading
- Cross-exchange arbitrage
