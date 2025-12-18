# ADR-001: Historical Data Storage for Backtesting

**Date:** 2025-12-15
**Status:** Proposed
**Deciders:** Trading Bot Team
**Technical Story:** Enable comprehensive backtesting with persistent historical data

## Context and Problem Statement

The ws_paper_tester trading system currently operates with in-memory data only, limited to 100 candles per timeframe. This prevents:
- True historical backtesting against years of data
- Parameter optimization and walk-forward testing
- Strategy validation before live deployment
- Regime detection training against historical patterns

We need a persistent storage solution for historical market data that supports efficient time-series queries and real-time data ingestion.

## Decision Drivers

* **Query Performance**: Sub-second queries for millions of candles across multiple timeframes
* **Storage Efficiency**: Years of tick/candle data must be stored cost-effectively
* **Real-time Ingestion**: WebSocket data must be persisted with minimal latency
* **Multi-Timeframe Support**: Automatic aggregation from 1-minute to weekly candles
* **Integration**: Compatible with existing asyncio-based Python codebase
* **Operational Simplicity**: Easy to deploy, backup, and maintain

## Considered Options

1. **TimescaleDB (PostgreSQL extension)**
2. **InfluxDB**
3. **QuestDB**
4. **ClickHouse**
5. **Plain PostgreSQL**
6. **SQLite with partitioning**

## Decision Outcome

**Chosen option: TimescaleDB**, because it provides:
- Native PostgreSQL compatibility (existing team expertise)
- Hypertables with automatic time-based partitioning
- Continuous aggregates for automatic multi-timeframe rollups
- 90%+ compression for historical data
- Excellent asyncpg support for Python async operations
- Docker-ready with official images

### Consequences

**Good:**
- Full SQL support - familiar query language
- Automatic chunk management and compression
- Continuous aggregates eliminate manual rollup jobs
- Strong ecosystem (pgAdmin, pg_dump, replication)
- asyncpg provides high-performance async access

**Bad:**
- Heavier resource footprint than specialized time-series DBs
- Compression requires manual policy configuration
- Continuous aggregates have refresh latency (configurable)

**Neutral:**
- Requires Docker or local PostgreSQL installation
- Learning curve for TimescaleDB-specific features (hypertables, continuous aggregates)

## Pros and Cons of the Options

### TimescaleDB

* Good: PostgreSQL compatibility, team familiarity
* Good: Automatic time-based partitioning
* Good: Continuous aggregates for multi-timeframe data
* Good: 90%+ compression
* Good: Excellent Python async support
* Bad: Higher memory usage than InfluxDB
* Bad: More complex setup than SQLite

### InfluxDB

* Good: Purpose-built for time-series
* Good: InfluxQL is intuitive
* Good: Built-in retention policies
* Bad: Different query language from SQL
* Bad: Less flexible for complex queries
* Bad: Python client less mature than asyncpg

### QuestDB

* Good: Extremely fast ingestion
* Good: SQL-like query language
* Good: Low memory footprint
* Bad: Less mature ecosystem
* Bad: Limited aggregation features
* Bad: Fewer deployment options

### ClickHouse

* Good: Excellent for analytical queries
* Good: Very fast aggregations
* Bad: Overkill for single-node deployment
* Bad: Column-oriented design less suited for tick data
* Bad: Complex operational requirements

### Plain PostgreSQL

* Good: Simple, familiar
* Good: Full SQL support
* Bad: No automatic partitioning
* Bad: Manual compression required
* Bad: Poor performance at scale without optimization

### SQLite

* Good: Zero configuration
* Good: Single file storage
* Bad: No concurrent write support
* Bad: Limited to single connection
* Bad: No built-in partitioning

## Links

- [Historical Data System Design](/docs/development/features/historical-data-system.md)
- [TimescaleDB Documentation](https://docs.timescale.com/)
- [Kraken API Documentation](https://docs.kraken.com/api/)
