# C4 Context Diagram

## System Context

The TG4 trading system operates within this context:

```
                              ┌─────────────────┐
                              │    Developer    │
                              │    (Operator)   │
                              └────────┬────────┘
                                       │
                                       │ monitors/configures
                                       ▼
┌─────────────────┐           ┌─────────────────┐           ┌─────────────────┐
│                 │  market   │                 │   LLM     │                 │
│   Kraken        │◄─────────►│      TG4        │◄─────────►│  LLM Providers  │
│   Exchange      │   data    │  Trading System │  queries  │ (Claude, etc.)  │
│                 │  orders   │                 │           │                 │
└─────────────────┘           └─────────────────┘           └─────────────────┘
                                       │
                                       │ stores
                                       ▼
                              ┌─────────────────┐
                              │   TimescaleDB   │
                              │  (Historical)   │
                              └─────────────────┘
```

## External Systems

| System | Relationship | Data Flow |
|--------|--------------|-----------|
| Kraken Exchange | Primary exchange | Bidirectional: market data in, orders out |
| LLM Providers | Decision support | Queries out, decisions in |
| TimescaleDB | Data persistence | OHLCV data, trade logs, decisions |

## Users

| User | Role | Interaction |
|------|------|-------------|
| Developer/Operator | System administrator | Configuration, monitoring, maintenance |

## Data Flows

| Flow | Source | Destination | Content |
|------|--------|-------------|---------|
| Market Data | Kraken | TG4 | OHLCV, trades, orderbook |
| Orders | TG4 | Kraken | Buy/sell orders |
| LLM Queries | TG4 | LLM Providers | Market context |
| LLM Decisions | LLM Providers | TG4 | Trade signals |
| Historical Data | TG4 | TimescaleDB | All data for persistence |
