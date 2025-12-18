# WebSocket Paper Tester - Architecture Documentation

**Version:** 1.15.1
**Last Updated:** 2025-12-15
**Framework:** Arc42

This documentation follows the [Arc42](https://arc42.org/) template for software architecture documentation.

---

## Table of Contents

1. [Introduction and Goals](01-introduction-and-goals.md)
2. [Constraints](02-constraints.md)
3. [Context and Scope](03-context-and-scope.md)
4. [Solution Strategy](04-solution-strategy.md)
5. [Building Block View](05-building-block-view.md)
6. [Runtime View](06-runtime-view.md)
7. [Deployment View](07-deployment-view.md)
8. [Cross-cutting Concepts](08-cross-cutting-concepts.md)
9. [Architecture Decisions](09-architecture-decisions.md)
10. [Quality Requirements](10-quality-requirements.md)
11. [Risks and Technical Debt](11-risks-and-technical-debt.md)
12. [Glossary](12-glossary.md)

---

## Quick Overview

The WebSocket Paper Tester is a real-time paper trading system for cryptocurrency markets. It connects to Kraken's WebSocket API, processes market data, executes trading strategies in simulation, and provides a web dashboard for monitoring.

### Key Architectural Characteristics

| Characteristic | Description |
|----------------|-------------|
| **Architecture Style** | Event-driven, modular monolith |
| **Primary Language** | Python 3.10+ |
| **Data Flow** | WebSocket → DataManager → Strategies → Executor → Dashboard |
| **Concurrency Model** | asyncio with thread-safe portfolio operations |
| **Storage** | TimescaleDB for historical data, in-memory for live trading |

### System Context (C4 Level 1)

```
┌─────────────────────────────────────────────────────────────────┐
│                        External Systems                          │
├─────────────┬─────────────┬─────────────┬─────────────┬─────────┤
│  Kraken WS  │  Kraken REST│  Alternative│  CoinGecko  │  User   │
│  (prices)   │  (backfill) │  (F&G Index)│  (BTC.D)    │ Browser │
└──────┬──────┴──────┬──────┴──────┬──────┴──────┬──────┴────┬────┘
       │             │             │             │           │
       ▼             ▼             ▼             ▼           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   WebSocket Paper Tester                         │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────────────┐│
│  │   Data    │ │ Strategy  │ │  Paper    │ │     Dashboard     ││
│  │  Manager  │→│  Engine   │→│ Executor  │→│   (FastAPI/WS)    ││
│  └───────────┘ └───────────┘ └───────────┘ └───────────────────┘│
│                       │                                          │
│                       ▼                                          │
│              ┌───────────────┐                                   │
│              │  TimescaleDB  │                                   │
│              │  (historical) │                                   │
│              └───────────────┘                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Architecture Decision Records (ADRs)

Key architectural decisions are documented in `09-architecture-decisions.md`:

- **ADR-001**: TimescaleDB for historical data storage
- **ADR-002**: Per-strategy isolated portfolios
- **ADR-003**: Leveraged position support with margin calls
- **ADR-004**: Market regime detection system

---

*Generated following Arc42 template v8.2*
