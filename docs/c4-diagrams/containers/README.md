# C4 Container Diagram

## Container Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           TG4 Trading System                              │
│                                                                           │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐  │
│  │                 │  │                 │  │                         │  │
│  │  Trading App    │  │   TimescaleDB   │  │         Redis           │  │
│  │   (Python)      │  │   (PostgreSQL)  │  │        (Cache)          │  │
│  │                 │  │                 │  │                         │  │
│  │  - WebSocket    │  │  - OHLCV data   │  │  - State cache          │  │
│  │  - Strategies   │  │  - Trade logs   │  │  - Pub/sub              │  │
│  │  - Risk mgmt    │  │  - Decisions    │  │  - Session data         │  │
│  │  - Execution    │  │                 │  │                         │  │
│  │                 │  │                 │  │                         │  │
│  └────────┬────────┘  └────────┬────────┘  └────────────┬────────────┘  │
│           │                    │                         │               │
│           └────────────────────┴─────────────────────────┘               │
│                                                                           │
│  ┌─────────────────┐  ┌─────────────────┐                                │
│  │                 │  │                 │                                │
│  │   Prometheus    │  │    Grafana      │                                │
│  │   (Metrics)     │  │  (Dashboard)    │                                │
│  │                 │  │                 │                                │
│  └─────────────────┘  └─────────────────┘                                │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘
```

## Container Details

| Container | Technology | Purpose |
|-----------|------------|---------|
| Trading App | Python 3.10+ | Core trading logic |
| TimescaleDB | PostgreSQL extension | Time-series data storage |
| Redis | Redis 7.x | Caching, pub/sub |
| Prometheus | Prometheus | Metrics collection |
| Grafana | Grafana | Visualization |

## Communication

| From | To | Protocol | Purpose |
|------|----|----------|---------|
| Trading App | TimescaleDB | PostgreSQL | Data persistence |
| Trading App | Redis | Redis protocol | Caching, state |
| Prometheus | Trading App | HTTP | Scrape metrics |
| Grafana | Prometheus | HTTP | Query metrics |

## External Communication

| Container | External System | Protocol |
|-----------|-----------------|----------|
| Trading App | Kraken | WebSocket |
| Trading App | LLM APIs | HTTPS |
