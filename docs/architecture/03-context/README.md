# 03 - System Context

## Business Context

```
+-------------------+     +-------------------+     +-------------------+
|                   |     |                   |     |                   |
|   Kraken API      |<--->|      TG4          |<--->|   LLM Providers   |
|   (WebSocket)     |     |   Trading System  |     |   (Claude, etc.)  |
|                   |     |                   |     |                   |
+-------------------+     +-------------------+     +-------------------+
        ^                         ^
        |                         |
        v                         v
+-------------------+     +-------------------+
|                   |     |                   |
|   Market Data     |     |   TimescaleDB     |
|   (OHLCV, Trades) |     |   (Historical)    |
|                   |     |                   |
+-------------------+     +-------------------+
```

## External Interfaces

| Interface | Type | Purpose |
|-----------|------|---------|
| Kraken WebSocket | API | Real-time market data, order execution |
| Claude API | LLM | Trading decision agent |
| Grok API | LLM | Trading decision agent (comparison) |
| GPT-4 API | LLM | Trading decision agent (comparison) |
| Ollama (local) | LLM | Cost-efficient local inference |

## Technical Context

### Communication Partners

| Partner | Protocol | Data Format |
|---------|----------|-------------|
| Kraken | WebSocket | JSON |
| LLM APIs | HTTPS/REST | JSON |
| TimescaleDB | PostgreSQL | SQL |
| Redis | TCP | Redis Protocol |

### Trading Pairs

| Pair | Purpose |
|------|---------|
| BTC/USDT | Primary trading pair |
| XRP/USDT | Secondary trading pair |
| XRP/BTC | Cross-pair arbitrage |
