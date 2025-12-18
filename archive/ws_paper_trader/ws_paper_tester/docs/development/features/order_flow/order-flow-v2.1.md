# Order Flow Strategy v2.1 - XRP/BTC Ratio Trading

## Overview

Version 2.1 of the Order Flow strategy adds support for XRP/BTC pair trading with the goal of growing holdings of both XRP and BTC through ratio trading.

## Changes from v2.0

### New Symbol Support
- Added `XRP/BTC` to supported symbols list
- Strategy now trades: `XRP/USDT`, `BTC/USDT`, `XRP/BTC`

### Research-Based Configuration

Market analysis was performed on Kraken's 24h data to optimize XRP/BTC parameters:

| Metric | XRP/BTC | XRP/USDT | BTC/USDT |
|--------|---------|----------|----------|
| 24h Volume | 96,604 XRP | 286,505 XRP | 27 BTC |
| 24h Trades | 664 | 787 | 1,894 |
| Spread | 0.0446% | 0.0287% | 0.0174% |
| Trade Frequency | ~1 per 2 min | ~1 per 2 min | ~1.3/min |

### XRP/BTC Specific Configuration

```python
'XRP/BTC': {
    'imbalance_threshold': 0.35,   # Higher threshold (wider spread, less noise)
    'volume_spike_mult': 1.5,      # Lower multiplier (fewer trades to detect spikes)
    'cooldown_seconds': 30.0,      # Longer cooldown (lower trade frequency)
    'cooldown_trades': 5,          # Fewer trades needed for cooldown
    'position_size_xrp': 30.0,     # Trade 30 XRP per signal (~6% of 500 XRP)
    'take_profit_pct': 0.4,        # Wider than spread (0.0446%)
    'stop_loss_pct': 0.4,          # 1:1 R:R
    'base_asset': 'XRP',
    'quote_asset': 'BTC',
}
```

### Strategy Logic Changes

For XRP/BTC pair (different from USD pairs):
- **Buy signal** (buy pressure): Trade BTC for XRP (accumulate XRP)
- **Sell signal** (sell pressure): Trade XRP for BTC (accumulate BTC)
- **No shorting**: XRP/BTC uses direct buy/sell only (no short positions)
- Position limits in XRP (500 max) instead of USD

### Symbol-Specific Settings

The strategy now supports per-symbol overrides for:
- `cooldown_trades`: Trade-based cooldown
- `cooldown_seconds`: Time-based cooldown
- `volume_spike_mult`: Volume spike detection threshold
- `take_profit_pct` / `stop_loss_pct`: TP/SL percentages
- `position_size_xrp`: Position size in XRP (for XRP/BTC)
- `position_size_usd`: Position size in USD (for USDT pairs)

## Portfolio Changes

### Starting Assets Support

The portfolio system now supports starting asset holdings in addition to USDT capital:

```yaml
# config.yaml
starting_assets:
  XRP: 500.0    # 500 XRP starting balance
  BTC: 0.0      # Will accumulate through trading
```

### Implementation Details

- `PortfolioManager` accepts `starting_assets` parameter
- Each strategy portfolio initialized with both USDT and assets
- Assets tracked separately from USDT balance
- Equity calculation includes asset values at current prices

## Configuration

### config.yaml Updates

```yaml
# Starting asset holdings
starting_assets:
  XRP: 500.0
  BTC: 0.0

# Symbols including XRP/BTC
symbols:
  - XRP/USDT
  - BTC/USDT
  - XRP/BTC
```

### Strategy Override Example

```yaml
strategy_overrides:
  order_flow:
    imbalance_threshold: 0.3
    position_size_usd: 25
```

## Trading Goal

The primary goal for XRP/BTC trading is to **grow holdings of both XRP and BTC** through ratio trading:

1. When XRP is relatively undervalued vs BTC (buy pressure): Buy XRP with BTC
2. When XRP is relatively overvalued vs BTC (sell pressure): Sell XRP for BTC
3. Over time, both asset holdings should increase through profitable ratio trades

## Files Changed

- `strategies/order_flow.py` - v2.1.0 with XRP/BTC support
- `config.yaml` - Added starting_assets and XRP/BTC symbol
- `ws_tester/portfolio.py` - Starting assets support in PortfolioManager
- `ws_tester.py` - Pass starting_assets from config
- `ws_tester/data_layer.py` - USDT pair defaults in SimulatedDataManager
- `tests/test_strategies.py` - Updated test fixtures to use USDT pairs

## Testing

All 126 tests pass after changes.

Verified working with Kraken WebSocket:
- XRP/BTC: 0.00002243 BTC
- XRP/USDT: 2.03 USDT
- BTC/USDT: 90,295.6 USDT
