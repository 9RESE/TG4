# 05 - Building Block View

## Level 1: System Overview

```
+------------------------------------------------------------------+
|                        TG4 Trading System                         |
+------------------------------------------------------------------+
|                                                                    |
|  +----------------+  +----------------+  +--------------------+   |
|  | Data Ingestion |  | Strategy Engine|  | Execution Manager  |   |
|  +----------------+  +----------------+  +--------------------+   |
|                                                                    |
|  +----------------+  +----------------+  +--------------------+   |
|  | Risk Manager   |  | Portfolio Mgr  |  | Monitoring/Logging |   |
|  +----------------+  +----------------+  +--------------------+   |
|                                                                    |
+------------------------------------------------------------------+
```

## Level 2: Component Details

### Data Ingestion

| Component | Responsibility |
|-----------|----------------|
| WebSocket Client | Connect to Kraken, receive real-time data |
| Data Normalizer | Standardize data format |
| TimescaleDB Writer | Persist historical data |
| Indicator Calculator | Compute technical indicators |

### Strategy Engine

| Component | Responsibility |
|-----------|----------------|
| LLM Client | Interface with multiple LLM providers |
| Prompt Builder | Format market context for LLM |
| Signal Generator | Parse LLM output into trade signals |
| Strategy Selector | Choose active strategy/LLM |

### Execution Manager

| Component | Responsibility |
|-----------|----------------|
| Order Builder | Create order requests |
| Order Executor | Submit orders via API |
| Position Tracker | Track open positions |
| Fill Handler | Process order fills |

### Risk Manager

| Component | Responsibility |
|-----------|----------------|
| Position Sizer | Calculate position sizes |
| Stop-Loss Manager | Set and update stop-losses |
| Drawdown Monitor | Track portfolio drawdown |
| Kill Switch | Emergency trading halt |

### Portfolio Manager

| Component | Responsibility |
|-----------|----------------|
| Balance Tracker | Track asset balances |
| Rebalancer | Adjust asset allocation |
| P&L Calculator | Compute profit/loss |

### Monitoring/Logging

| Component | Responsibility |
|-----------|----------------|
| Structured Logger | JSON logging |
| Metrics Collector | Prometheus metrics |
| Dashboard | Grafana visualization |
