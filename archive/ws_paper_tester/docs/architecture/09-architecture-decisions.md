# 9. Architecture Decisions

This section documents key architectural decisions using the Architecture Decision Record (ADR) format.

---

## ADR-001: TimescaleDB for Historical Data Storage

**Date:** 2025-12-15
**Status:** Accepted

### Context
We need to store historical market data (trades, candles) for backtesting and strategy warmup. Options considered:
- SQLite (simple, file-based)
- PostgreSQL (robust, relational)
- TimescaleDB (PostgreSQL + time-series extensions)
- InfluxDB (native time-series)

### Decision
Use **TimescaleDB** with PostgreSQL 15.

### Rationale
- Native time-series optimizations (hypertables, continuous aggregates)
- 90%+ compression for historical data
- SQL compatibility for complex queries
- Automatic multi-timeframe candle rollup
- Strong Python support via asyncpg

### Consequences
- **Positive:** Efficient storage, automatic aggregation, familiar SQL
- **Negative:** Requires Docker/PostgreSQL infrastructure
- **Mitigation:** Docker Compose provided for easy deployment

---

## ADR-002: Per-Strategy Isolated Portfolios

**Date:** 2025-12-13
**Status:** Accepted

### Context
Multiple strategies run simultaneously. Options for portfolio management:
- Shared portfolio (all strategies trade same capital)
- Isolated portfolios (each strategy has own capital)

### Decision
Use **isolated portfolios** per strategy.

### Rationale
- Clear performance attribution per strategy
- No interference between strategies
- Easier to add/remove strategies
- Matches real-world multi-strategy deployment

### Consequences
- **Positive:** Clean separation, accurate metrics
- **Negative:** More memory usage, can't see cross-strategy effects
- **Mitigation:** Portfolio manager provides aggregate views

---

## ADR-003: Leveraged Positions with Margin Calls

**Date:** 2025-12-15
**Status:** Accepted

### Context
Strategies need leverage capability to maximize capital efficiency. Previously only shorts had 2x leverage; longs were cash-only.

### Decision
- Add leveraged longs (1.5x default, conservative)
- Implement margin call liquidation (25% maintenance margin)
- Make leverage configurable per execution settings

### Rationale
- Symmetric capability for longs and shorts
- Realistic simulation of margin trading
- Conservative long leverage due to unlimited downside risk
- Automatic liquidation prevents negative equity

### Consequences
- **Positive:** More realistic simulation, better capital efficiency
- **Negative:** Complexity in equity calculations
- **Mitigation:** Comprehensive test coverage, margin call logging

---

## ADR-004: Market Regime Detection System

**Date:** 2025-12-15
**Status:** Accepted

### Context
Strategies perform differently in different market conditions. Static parameters lead to poor performance in unfavorable regimes.

### Decision
Implement a composite regime detection system with:
- 5 regimes: STRONG_BULL, BULL, SIDEWAYS, BEAR, STRONG_BEAR
- 4 volatility states: LOW, MEDIUM, HIGH, EXTREME
- Multi-timeframe confluence analysis
- External sentiment (Fear & Greed, BTC Dominance)

### Rationale
- Strategies can adapt parameters to market conditions
- Reduce drawdowns by avoiding unfavorable regimes
- Academic research supports regime-based trading

### Consequences
- **Positive:** Adaptive strategies, reduced drawdowns
- **Negative:** Added latency, external API dependencies
- **Mitigation:** Caching, graceful fallback for external data

---

## ADR-005: Event-Driven Architecture with asyncio

**Date:** 2025-12-13
**Status:** Accepted

### Context
Need to handle multiple concurrent operations:
- WebSocket data reception
- Strategy signal generation
- Dashboard updates
- Historical data persistence

### Decision
Use Python's **asyncio** for the main event loop with thread-safe portfolio operations.

### Rationale
- Efficient I/O handling for WebSocket and HTTP
- Single-threaded simplicity for most operations
- RLock for portfolio thread safety where needed
- Native Python support, no external dependencies

### Consequences
- **Positive:** Efficient, simple concurrency model
- **Negative:** Must be careful about blocking operations
- **Mitigation:** Async database operations, thread-safe portfolio

---

## ADR-006: Strategy Auto-Discovery with Security

**Date:** 2025-12-13
**Status:** Accepted

### Context
Strategies are Python files that get dynamically loaded. This creates security risks from arbitrary code execution.

### Decision
Implement multi-layer security:
- Strategy file whitelist
- SHA256 hash verification for approved strategies
- Optional `ALLOW_UNSIGNED_STRATEGIES` flag
- Explicit module interface requirements

### Rationale
- Prevents unauthorized strategy execution
- Hash verification detects tampering
- Development mode allows unsigned for convenience
- Clear interface contract for strategies

### Consequences
- **Positive:** Security against malicious strategies
- **Negative:** Extra setup step for new strategies
- **Mitigation:** Hash generation utility provided

---

## ADR-007: JSONL Structured Logging

**Date:** 2025-12-13
**Status:** Accepted

### Context
Need comprehensive logging for debugging, analysis, and audit trail. Options:
- Plain text logs
- JSON logs
- Structured binary (protobuf, etc.)

### Decision
Use **JSON Lines (JSONL)** format with:
- Separate log streams (system, strategy, trades, aggregated)
- Correlation IDs for request tracing
- Automatic rotation and gzip compression

### Rationale
- Human-readable yet machine-parseable
- Easy to query with `jq`, load into pandas
- Industry standard for structured logging
- Supports complex nested data

### Consequences
- **Positive:** Great for analysis, debugging
- **Negative:** Larger file size than binary
- **Mitigation:** gzip compression, rotation policy

---

## Decision Log

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| 001 | TimescaleDB for Historical Data | Accepted | 2025-12-15 |
| 002 | Per-Strategy Isolated Portfolios | Accepted | 2025-12-13 |
| 003 | Leveraged Positions with Margin Calls | Accepted | 2025-12-15 |
| 004 | Market Regime Detection System | Accepted | 2025-12-15 |
| 005 | Event-Driven Architecture with asyncio | Accepted | 2025-12-13 |
| 006 | Strategy Auto-Discovery with Security | Accepted | 2025-12-13 |
| 007 | JSONL Structured Logging | Accepted | 2025-12-13 |
