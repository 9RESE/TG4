# Kraken Account Status & Capabilities

*Last verified: 2025-12-18*

## Account Overview

| Attribute | Value |
|-----------|-------|
| **Account Tier** | Starter/Intermediate |
| **Verification Level** | Full (wire deposits enabled) |
| **Fee Tier** | Standard (volume-based discounts available) |
| **30-day Trading Volume** | $0.00 |
| **Margin Trading** | Enabled |
| **Staking/Earn** | Available |

## Current Balances

| Asset | Balance | USD Equivalent |
|-------|---------|----------------|
| XRP | 25.00000000 | ~$46.49 |
| **Total Equity** | - | $46.49 |
| **Free Margin** | - | $44.17 |

## Fee Structure

### Current Fees (0 volume tier)

| Type | Taker | Maker |
|------|-------|-------|
| Current | 0.40% | 0.25% |
| Minimum achievable | 0.05% | 0.00% |

### Volume-Based Fee Tiers

| 30-day Volume (USD) | Taker | Maker |
|---------------------|-------|-------|
| $0 - $50,000 | 0.40% | 0.25% |
| $50,001 - $100,000 | 0.35% | 0.20% |
| $100,001 - $250,000 | 0.28% | 0.14% |
| $250,001 - $500,000 | 0.22% | 0.12% |
| $500,001 - $1,000,000 | 0.18% | 0.10% |
| $1,000,001 - $5,000,000 | 0.14% | 0.08% |
| $5,000,001 - $10,000,000 | 0.12% | 0.06% |
| $10,000,001+ | 0.10% | 0.04% |

## API Key Permissions

| Permission | Status | Notes |
|------------|--------|-------|
| Query Funds | Enabled | Balance, trade balance |
| Query Open Orders & Trades | Enabled | Open orders, positions |
| Query Closed Orders & Trades | Enabled | History access |
| Modify Orders | Enabled | Create, edit, cancel |
| Cancel/Close Orders | Enabled | Cancel operations |
| Deposit | Enabled | View deposit methods/addresses |
| Withdraw | **Disabled** | Security measure for bot |
| Access WebSockets API | Enabled | Real-time streaming |

## Margin Trading Capabilities

### Account Margin Status

| Metric | Value |
|--------|-------|
| Equity | $46.49 |
| Trade Balance | $44.17 |
| Used Margin | $0.00 |
| Free Margin | $44.17 |
| Open Positions | 0 |

### Margin Costs

| Fee Type | Rate | Timing |
|----------|------|--------|
| Opening Fee | 0.01% - 0.02% | On position open |
| Rollover Fee | 0.01% - 0.02% | Every 4 hours |

### XRP Margin Pairs

| Pair | API Name | Max Leverage | Notes |
|------|----------|--------------|-------|
| XRP/USD | XXRPZUSD | **10x** | Primary USD pair |
| XRP/EUR | XXRPZEUR | **10x** | Primary EUR pair |
| XRP/USDT | XRPUSDT | **10x** | Stablecoin pair |
| XRP/USDC | XRPUSDC | **10x** | Stablecoin pair |
| XRP/BTC | XXRPXXBT | 3x | Bitcoin pair |
| XRP/ETH | XRPETH | 3x | Ethereum pair |
| XRP/GBP | XRPGBP | 3x | British Pound |
| XRP/AUD | XRPAUD | 3x | Australian Dollar |
| XRP/CAD | XXRPZCAD | 3x | Canadian Dollar |

### Other Key Margin Pairs

| Pair | Max Leverage |
|------|--------------|
| BTC/USD | 5x |
| ETH/USD | 5x |
| SOL/USD | 5x |
| ADA/USDT | 10x |
| ETH/USDT | 10x |
| LINK/USDT | 10x |
| SOL/USDT | 10x |

**Total margin-enabled pairs: 253**

## Deposit Methods Available

| Method | Limit | Notes |
|--------|-------|-------|
| Wire Transfer (Dart Bank) | $10,000,000 | Bank wire |
| Plaid US | $10,000,000 | ACH transfer |
| PayPal | $10,000,000 | PayPal integration |
| WalletPay | $10,000,000 | Digital wallet |
| WorldPay | $10,000,000 | Card payments |

## Rate Limits

### REST API Limits

| Tier | Counter Max | Decay Rate | Our Tier |
|------|-------------|------------|----------|
| Starter | 15 | -0.33/sec | Current |
| Intermediate | 20 | -0.5/sec | - |
| Pro | 20 | -1/sec | - |

### Trading Engine Limits

| Tier | Threshold | Decay Rate | Open Orders/Pair |
|------|-----------|------------|------------------|
| Starter | 60 | -1/sec | 60 |
| Intermediate | 125 | -2.34/sec | 80 |
| Pro | 180 | -3.75/sec | 225 |

## WebSocket Access

| Attribute | Value |
|-----------|-------|
| Status | Enabled |
| Token Expiry | 900 seconds (15 min) |
| Public URL | wss://ws.kraken.com/v2 |
| Private URL | wss://ws-auth.kraken.com/v2 |

### Available Channels

**Public:**
- ticker (Level 1)
- book (Level 2 order book)
- trade (recent trades)
- ohlc (candles)

**Private:**
- executions (fills)
- balances (account)
- add_order, cancel_order, etc.

## Ledger History

| Type | Asset | Amount | Timestamp |
|------|-------|--------|-----------|
| Deposit | XRP | 25.00000000 | 2025-12-06 |

## Trading Recommendations

### Optimal Pairs for XRP Trading

1. **XRP/USD (XXRPZUSD)** - Primary, highest liquidity, 10x leverage
2. **XRP/USDT (XRPUSDT)** - Stablecoin, 10x leverage, good for arbitrage
3. **XRP/EUR (XXRPZEUR)** - EUR exposure, 10x leverage

### Fee Optimization

- Use **limit orders** for 0.25% maker fee vs 0.40% taker
- Volume of $50K/month reduces fees to 0.35%/0.20%
- Post-only orders guarantee maker fee

### Margin Considerations

- Start with 2-3x leverage until strategy is proven
- Monitor rollover fees for long-term positions
- Keep margin utilization below 50% for safety

## API Usage Notes

### Authentication

```python
# Nonce: Use millisecond timestamp
nonce = str(int(time.time() * 1000))

# Signature: HMAC-SHA512
signature = HMAC-SHA512(
    uri_path + SHA256(nonce + post_data),
    base64_decode(private_key)
)
```

### Rate Limit Management

- REST counter: Track locally, respect 15 max
- Add 1 per call, 2 for ledger/history
- Wait for decay before retrying (3 sec at Starter)

### Error Handling

| Error | Meaning | Action |
|-------|---------|--------|
| EAPI:Rate limit exceeded | Counter full | Wait 3-5 seconds |
| EOrder:Rate limit exceeded | Too many orders | Slow down trading |
| EOrder:Insufficient funds | Not enough balance | Check available funds |
| EAPI:Invalid nonce | Nonce too low | Use higher timestamp |

## Security Notes

- API key has **no withdrawal permission** (recommended for bots)
- Withdrawal requires separate manual approval
- Consider IP whitelisting for production
- WebSocket tokens expire in 15 minutes

---

*This document reflects the account state as of the verification date. Balances and positions change with trading activity.*
