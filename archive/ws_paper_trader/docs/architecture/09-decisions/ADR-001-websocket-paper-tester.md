# ADR-001: WebSocket Paper Trading System Architecture

**Status:** Accepted
**Date:** 2025-12-13
**Deciders:** Architecture Review
**Technical Story:** Need lightweight paper trading system for strategy development

## Context and Problem Statement

The existing `unified_trader.py` system has grown to 1700+ lines with complex dual portfolio modes, regime detection, and 30+ entangled strategies. Adding new strategies requires significant boilerplate and understanding of the existing architecture.

We needed a simpler system for:
- Rapid strategy prototyping with live WebSocket data
- Clear performance comparison between strategies
- Easy addition of new strategies (drop-in files)
- Real-time visibility into trading decisions

## Decision Drivers

* **Simplicity**: Minimize boilerplate for strategy development
* **Isolation**: Each strategy should operate independently
* **Observability**: Every decision should be logged and visible
* **Performance**: Support tick-level data processing
* **Evolution**: Design should support future live trading

## Considered Options

### Option 1: Extend unified_trader.py
Add WebSocket support and simplified mode to existing system.

**Pros:**
- Reuse existing code
- Single codebase

**Cons:**
- Increases complexity
- Risk of breaking production code
- Hard to simplify entangled strategies

### Option 2: Standalone WebSocket-Native System
Build new system from scratch, WebSocket-first design.

**Pros:**
- Clean architecture
- No legacy constraints
- Purpose-built for the use case

**Cons:**
- More initial development
- Potential code duplication

### Option 3: Fork unified_trader.py
Copy and simplify the existing codebase.

**Pros:**
- Quick start with working code
- Can remove unneeded features

**Cons:**
- Inherits architectural issues
- Diverging codebases

## Decision Outcome

**Chosen Option: Option 2 - Standalone WebSocket-Native System**

### Implementation Details

| Component | Decision | Rationale |
|-----------|----------|-----------|
| **Project Structure** | Standalone `/ws_paper_tester/` | Clean separation, no dependencies |
| **Data Layer** | Async WebSocket client | Native support for real-time data |
| **Strategy Interface** | Simple module with `generate_signal()` | Minimal boilerplate, duck typing |
| **Portfolio Model** | Isolated $100 per strategy | Clear comparison, no interference |
| **State Persistence** | None (logs capture everything) | Fresh start each run simplifies testing |
| **Dashboard** | FastAPI + WebSocket | Real-time updates, no page refresh |
| **Logging** | JSON Lines per stream | Queryable, correlatable logs |

### Architecture

```
ws_paper_tester/
├── ws_tester.py              # Main entry (coordinator)
├── ws_tester/                # Core library
│   ├── types.py              # Immutable data structures
│   ├── data_layer.py         # WebSocket client + data management
│   ├── strategy_loader.py    # Auto-discovery
│   ├── portfolio.py          # Isolated portfolio per strategy
│   ├── executor.py           # Paper execution engine
│   ├── logger.py             # Structured logging
│   └── dashboard/server.py   # Real-time web dashboard
└── strategies/               # Drop-in strategy files
```

### Positive Consequences

* Strategy files are self-contained (~100 lines each)
* Adding a strategy = dropping a `.py` file
* Clear P&L attribution per strategy
* Real-time visibility via dashboard
* 52 unit tests with 100% pass rate
* <500 lines of core code (target was <800)

### Negative Consequences

* Some concepts duplicated from unified_trader.py
* Strategies not directly portable between systems
* Separate deployment/monitoring needed

## Links

* Design Document: [/docs/development/features/WEBSOCKET_PAPER_TESTER_DESIGN.md](../../development/features/WEBSOCKET_PAPER_TESTER_DESIGN.md)
* Implementation: `/ws_paper_tester/`
* How-To Guide: [/docs/user/how-to/websocket-paper-tester.md](../../user/how-to/websocket-paper-tester.md)

---
*ADR Template based on [MADR](https://adr.github.io/madr/)*
