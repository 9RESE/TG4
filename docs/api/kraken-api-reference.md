# Kraken API Reference

Comprehensive reference for the Kraken Exchange API covering REST, WebSocket, and trading operations.

## API Overview

| API Type | Base URL | Use Case |
|----------|----------|----------|
| REST API | `https://api.kraken.com` | Request-response operations |
| WebSocket v2 | `wss://ws.kraken.com/v2` | Real-time streaming data |
| WebSocket Auth | `wss://ws-auth.kraken.com/v2` | Authenticated streaming |

## Authentication

### Required Components

| Component | Header/Field | Description |
|-----------|--------------|-------------|
| API-Key | Header | Your public API key |
| API-Sign | Header | HMAC-SHA512 signature |
| nonce | POST body | Always-increasing 64-bit integer |
| otp | POST body | Optional 2FA code |

### Signature Generation

```
API-Sign = HMAC-SHA512(
    URI_path + SHA256(nonce + POST_data),
    base64_decode(private_key)
)
```

**URI path**: Starting from `/0/private/...`

### Nonce Best Practices

- Use Unix timestamp in milliseconds
- Never reuse or decrease nonce values
- Excessive invalid nonces trigger temporary bans

### Python Authentication Example

```python
import hmac
import hashlib
import base64
import urllib.parse
import time

def get_kraken_signature(urlpath, data, secret):
    postdata = urllib.parse.urlencode(data)
    encoded = (str(data['nonce']) + postdata).encode()
    message = urlpath.encode() + hashlib.sha256(encoded).digest()
    mac = hmac.new(base64.b64decode(secret), message, hashlib.sha512)
    return base64.b64encode(mac.digest()).decode()

def kraken_request(uri_path, data, api_key, api_secret):
    headers = {
        'API-Key': api_key,
        'API-Sign': get_kraken_signature(uri_path, data, api_secret)
    }
    data['nonce'] = int(time.time() * 1000)
    # Make POST request with headers and data
```

---

## Rate Limits

### REST API Rate Limits

| Tier | Max Counter | Decay Rate |
|------|-------------|------------|
| Starter | 15 | -0.33/sec |
| Intermediate | 20 | -0.5/sec |
| Pro | 20 | -1/sec |

**Counter Increments**:
- Most calls: +1
- Ledger/trade history: +2
- AddOrder/CancelOrder: Separate limiters

### Trading Engine Rate Limits

| Tier | Threshold | Decay Rate |
|------|-----------|------------|
| Starter | 60 | -1/sec |
| Intermediate | 125 | -2.34/sec |
| Pro | 180 | -3.75/sec |

**Transaction Costs**:

| Action | Fixed Cost | + Decay Penalty (by resting time) |
|--------|------------|-----------------------------------|
| Add Order | +1 | None |
| Amend Order | +1 | +3 to +1 |
| Edit Order | +1 | +6 to +1 |
| Cancel Order | +0 | +8 to +1 |

### Open Order Limits (per pair)

| Tier | Limit |
|------|-------|
| Starter | 60 |
| Intermediate | 80 |
| Pro | 225 |

### Error Codes

| Error | Meaning |
|-------|---------|
| `EAPI:Rate limit exceeded` | REST API counter exceeded |
| `EOrder:Rate limit exceeded` | Trading engine limit exceeded |
| `EOrder:Orders limit exceeded` | Too many open orders |
| `EService:Throttled:[timestamp]` | Retry after timestamp |

---

## REST API Endpoints

### Public Market Data

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/0/public/Time` | GET | Server time |
| `/0/public/SystemStatus` | GET | System status |
| `/0/public/Assets` | GET | Asset information |
| `/0/public/AssetPairs` | GET | Tradable asset pairs |
| `/0/public/Ticker` | GET | Ticker information |
| `/0/public/OHLC` | GET | OHLC candle data |
| `/0/public/Depth` | GET | Order book |
| `/0/public/Trades` | GET | Recent trades |
| `/0/public/Spread` | GET | Recent spreads |

### Private Account Data

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/0/private/Balance` | POST | Account balances |
| `/0/private/ExtendedBalance` | POST | Extended balance info |
| `/0/private/TradeBalance` | POST | Trade balance |
| `/0/private/OpenOrders` | POST | Open orders |
| `/0/private/ClosedOrders` | POST | Closed orders |
| `/0/private/QueryOrders` | POST | Query specific orders |
| `/0/private/TradesHistory` | POST | Trade history |
| `/0/private/QueryTrades` | POST | Query specific trades |
| `/0/private/OpenPositions` | POST | Open margin positions |
| `/0/private/Ledgers` | POST | Ledger entries |
| `/0/private/QueryLedgers` | POST | Query specific ledgers |
| `/0/private/TradeVolume` | POST | 30-day trade volume |
| `/0/private/GetWebSocketsToken` | POST | Get WS auth token |

### Private Trading

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/0/private/AddOrder` | POST | Place new order |
| `/0/private/AddOrderBatch` | POST | Place multiple orders |
| `/0/private/EditOrder` | POST | Edit existing order |
| `/0/private/AmendOrder` | POST | Amend order (price/volume) |
| `/0/private/CancelOrder` | POST | Cancel single order |
| `/0/private/CancelOrderBatch` | POST | Cancel multiple orders |
| `/0/private/CancelAllOrders` | POST | Cancel all orders |
| `/0/private/CancelAllOrdersAfter` | POST | Dead man's switch |

### Private Funding

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/0/private/DepositMethods` | POST | Available deposit methods |
| `/0/private/DepositAddresses` | POST | Deposit addresses |
| `/0/private/DepositStatus` | POST | Recent deposit status |
| `/0/private/WithdrawalMethods` | POST | Available withdrawal methods |
| `/0/private/WithdrawalAddresses` | POST | Withdrawal addresses |
| `/0/private/WithdrawalInfo` | POST | Withdrawal information |
| `/0/private/Withdraw` | POST | Initiate withdrawal |
| `/0/private/WithdrawalStatus` | POST | Recent withdrawal status |
| `/0/private/CancelWithdrawal` | POST | Cancel withdrawal request |
| `/0/private/WalletTransfer` | POST | Transfer between wallets |

### Subaccounts

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/0/private/CreateSubaccount` | POST | Create subaccount |
| `/0/private/AccountTransfer` | POST | Transfer between accounts |

### Earn (Staking)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/0/private/ListStrategies` | POST | List earn strategies |
| `/0/private/ListAllocations` | POST | List current allocations |
| `/0/private/AllocateStrategy` | POST | Allocate to earn |
| `/0/private/DeallocateStrategy` | POST | Deallocate from earn |
| `/0/private/AllocateStrategyStatus` | POST | Allocation status |
| `/0/private/DeallocateStrategyStatus` | POST | Deallocation status |

### Export Reports

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/0/private/AddExport` | POST | Request export report |
| `/0/private/ExportStatus` | POST | Export status |
| `/0/private/RetrieveExport` | POST | Download export |
| `/0/private/RemoveExport` | POST | Delete export |

---

## Order Types

### Basic Order Types

| Type | Description |
|------|-------------|
| `market` | Execute immediately at best available price |
| `limit` | Execute at specified price or better |

### Conditional Order Types

| Type | Trigger | Execution |
|------|---------|-----------|
| `stop-loss` | Price reaches stop (unfavorable) | Market order |
| `stop-loss-limit` | Price reaches stop (unfavorable) | Limit order |
| `take-profit` | Price reaches target (favorable) | Market order |
| `take-profit-limit` | Price reaches target (favorable) | Limit order |
| `trailing-stop` | Price reverts from peak | Market order |
| `trailing-stop-limit` | Price reverts from peak | Limit order |
| `settle-position` | Close margin position | Market order |

### Price Parameters

| Parameter | Description |
|-----------|-------------|
| `price` | Limit price for limit orders; trigger price for stop/take-profit |
| `price2` | Limit price for stop-loss-limit, take-profit-limit, trailing-stop-limit |

### Relative Price Syntax

Prices can be specified relative to last traded price:

| Prefix/Suffix | Meaning | Example |
|---------------|---------|---------|
| `+` | Add to last price | `+100` = last + $100 |
| `-` | Subtract from last price | `-50` = last - $50 |
| `#` | Relative offset | `#100` |
| `%` | Percentage offset | `+2%` = last + 2% |

### Order Flags

| Flag | Description |
|------|-------------|
| `post` | Post-only (limit orders only) |
| `fcib` | Prefer fee in base currency |
| `fciq` | Prefer fee in quote currency |
| `nompp` | No market price protection |
| `viqc` | Volume in quote currency |

---

## Margin Trading

### Leverage Levels

| Asset Type | Max Leverage |
|------------|--------------|
| Major pairs (BTC, ETH) | Up to 5x |
| Selected assets | Up to 10x |
| Futures/Derivatives | Up to 50x |

### Margin Parameters

| Parameter | Description |
|-----------|-------------|
| `leverage` | Leverage level (2, 3, 4, 5, or 10) |
| `reduce_only` | Only reduce existing position |

### Example: Margin Order

```python
# Using CCXT
order = exchange.create_order(
    symbol='BTC/USD',
    type='market',
    side='buy',
    amount=0.1,
    params={'leverage': 3}
)
```

### Margin Costs

| Fee Type | Rate | Timing |
|----------|------|--------|
| Opening fee | 0.01% - 0.05% | On position open |
| Rollover fee | 0.01% - 0.05% | Every 4 hours |

### Position Management

| Endpoint | Purpose |
|----------|---------|
| `/0/private/OpenPositions` | View current margin positions |
| `/0/private/TradeBalance` | View margin equity and levels |
| `/0/private/ExtendedBalance` | View available margin |

---

## WebSocket v2 API

### Connection URLs

| Type | URL |
|------|-----|
| Public | `wss://ws.kraken.com/v2` |
| Authenticated | `wss://ws-auth.kraken.com/v2` |

### Public Channels

| Channel | Description |
|---------|-------------|
| `ticker` | Level 1 top-of-book data |
| `book` | Level 2 order book |
| `trade` | Recent trades |
| `ohlc` | Candle data |
| `instrument` | Market information |

### Private Channels

| Channel | Description |
|---------|-------------|
| `executions` | Trade fills |
| `balances` | Account balances |

### Trading Operations

| Operation | Description |
|-----------|-------------|
| `add_order` | Place new order |
| `amend_order` | Modify order |
| `cancel_order` | Cancel single order |
| `cancel_all` | Cancel all orders |
| `batch_add` | Place multiple orders |
| `batch_cancel` | Cancel multiple orders |

### Subscription Message Format

```json
{
  "method": "subscribe",
  "params": {
    "channel": "ticker",
    "symbol": ["XRP/USD", "BTC/USD"]
  }
}
```

### Authentication for Private Channels

1. Get token via REST: `POST /0/private/GetWebSocketsToken`
2. Include token in subscription:

```json
{
  "method": "subscribe",
  "params": {
    "channel": "executions",
    "token": "your_ws_token"
  }
}
```

---

## Common Asset Pairs

### XRP Pairs

| Pair | API Name | Margin |
|------|----------|--------|
| XRP/USD | XRPUSD | Yes (5x) |
| XRP/USDT | XRPUSDT | Yes (5x) |
| XRP/EUR | XRPEUR | Yes (5x) |
| XRP/BTC | XRPXBT | Yes (5x) |
| XRP/ETH | XRPETH | No |

### BTC Pairs

| Pair | API Name | Margin |
|------|----------|--------|
| BTC/USD | XBTUSD | Yes (5x) |
| BTC/USDT | XBTUSDT | Yes (5x) |
| BTC/EUR | XBTEUR | Yes (5x) |

### Pair Naming Convention

- `XBT` = Bitcoin (Kraken uses ISO standard)
- `XXBT` = Bitcoin with X prefix (older format)
- `ZUSD` = USD with Z prefix (older format)

---

## Error Handling

### Common Errors

| Error | Description | Solution |
|-------|-------------|----------|
| `EAPI:Invalid key` | Invalid API key | Check key configuration |
| `EAPI:Invalid signature` | Signature mismatch | Verify signing algorithm |
| `EAPI:Invalid nonce` | Nonce too low | Use higher nonce value |
| `EOrder:Insufficient funds` | Not enough balance | Check available funds |
| `EOrder:Unknown position` | Position not found | Verify position exists |
| `EGeneral:Permission denied` | Missing API permission | Update API key permissions |

### HTTP Status Codes

| Code | Meaning |
|------|---------|
| 200 | Success (check `error` array in response) |
| 403 | Forbidden (IP banned or access denied) |
| 429 | Too Many Requests (rate limited) |
| 5xx | Server error (retry with backoff) |

---

## Best Practices

### Connection Management

1. Use WebSocket for real-time data
2. Implement exponential backoff for retries
3. Handle disconnections gracefully
4. Use `cancel_on_disconnect` for safety

### Order Management

1. Always use `post` flag for maker orders
2. Implement dead man's switch (`CancelAllOrdersAfter`)
3. Validate order parameters before submission
4. Track orders by `userref` for correlation

### Rate Limit Management

1. Track your rate counter locally
2. Implement request queuing
3. Prioritize critical operations
4. Use batch operations when possible

---

## API Permissions Reference

| Permission | Capabilities |
|------------|--------------|
| Query Funds | View balances, ledgers |
| Query Open Orders & Trades | View orders, trades, positions |
| Query Closed Orders & Trades | View historical orders/trades |
| Modify Orders | Create, edit, cancel orders |
| Cancel/Close Orders | Cancel orders only |
| Deposit | View deposit addresses, initiate deposits |
| Withdraw | Initiate withdrawals |
| Access WebSockets API | Real-time streaming access |

---

## Resources

- [Kraken API Center](https://docs.kraken.com/api/)
- [REST API Documentation](https://docs.kraken.com/rest/)
- [WebSocket v2 Documentation](https://docs.kraken.com/websockets-v2/)
- [API Support](https://support.kraken.com/hc/en-us/categories/360000080686-API)
- [Rate Limits Guide](https://docs.kraken.com/api/docs/guides/spot-rest-ratelimits)
- [Authentication Guide](https://docs.kraken.com/api/docs/guides/spot-rest-auth)
