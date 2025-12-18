# C4 Component Diagram

## Trading App Components

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           Trading App (Python)                            │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                        Data Layer                                    │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │ │
│  │  │  WebSocket  │  │   Data      │  │  Indicator  │                 │ │
│  │  │   Client    │  │ Normalizer  │  │ Calculator  │                 │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                 │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                      Strategy Layer                                  │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │ │
│  │  │   Prompt    │  │    LLM      │  │   Signal    │                 │ │
│  │  │   Builder   │  │   Client    │  │  Generator  │                 │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                 │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                     Execution Layer                                  │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │ │
│  │  │    Risk     │  │   Order     │  │  Position   │                 │ │
│  │  │   Manager   │  │  Executor   │  │   Tracker   │                 │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                 │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                    Infrastructure Layer                              │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │ │
│  │  │   Logger    │  │   Metrics   │  │   Config    │                 │ │
│  │  │             │  │  Collector  │  │   Manager   │                 │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                 │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

### Data Layer

| Component | Responsibility |
|-----------|----------------|
| WebSocket Client | Maintain connection to Kraken, receive data |
| Data Normalizer | Convert exchange data to internal format |
| Indicator Calculator | Compute EMA, RSI, Bollinger Bands, etc. |

### Strategy Layer

| Component | Responsibility |
|-----------|----------------|
| Prompt Builder | Format market context for LLM |
| LLM Client | Interface with multiple LLM providers |
| Signal Generator | Parse LLM output into trade signals |

### Execution Layer

| Component | Responsibility |
|-----------|----------------|
| Risk Manager | Validate trades, enforce limits |
| Order Executor | Submit orders to exchange |
| Position Tracker | Track open positions and P&L |

### Infrastructure Layer

| Component | Responsibility |
|-----------|----------------|
| Logger | Structured JSON logging |
| Metrics Collector | Expose Prometheus metrics |
| Config Manager | Load and validate configuration |
