# API Documentation

API reference documentation for TG4 trading system.

## Exchange APIs

| API | Description | Status |
|-----|-------------|--------|
| [Kraken API Reference](kraken-api-reference.md) | Complete Kraken REST/WebSocket API | Complete |
| [Kraken Account Status](kraken-account-status.md) | Account capabilities, balances, permissions | Verified |

## Internal APIs

| API | Description | Status |
|-----|-------------|--------|
| Trading Engine | Core trading execution API | Planned |
| Market Data | OHLCV and orderbook data streams | Planned |
| LLM Interface | LLM query/response protocol | Planned |
| Risk Manager | Risk check and validation API | Planned |

## Kraken API Quick Reference

### REST API Base URL
```
https://api.kraken.com
```

### WebSocket URLs
```
Public:  wss://ws.kraken.com/v2
Private: wss://ws-auth.kraken.com/v2
```

### Key Endpoints

| Category | Endpoints |
|----------|-----------|
| Market Data | Time, Assets, AssetPairs, Ticker, OHLC, Depth, Trades |
| Account | Balance, TradeBalance, OpenOrders, ClosedOrders, OpenPositions |
| Trading | AddOrder, EditOrder, CancelOrder, CancelAllOrders |
| Funding | DepositMethods, DepositAddresses, Withdraw, WithdrawalStatus |

### Order Types

- `market`, `limit`
- `stop-loss`, `stop-loss-limit`
- `take-profit`, `take-profit-limit`
- `trailing-stop`, `trailing-stop-limit`

### Margin Trading

- Up to 5x leverage on major pairs
- Up to 10x on selected assets
- Rollover fees every 4 hours

See [Kraken API Reference](kraken-api-reference.md) for complete documentation.
