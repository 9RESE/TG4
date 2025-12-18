# TripleGain Implementation Plan

**Version**: 1.1
**Status**: Phase 1 Complete, Phase 2 Ready
**Date**: December 2025
**Last Updated**: 2025-12-18
**Source Design**: [TripleGain Master Design](../TripleGain-master-design/README.md)

## Current Status

| Phase | Status | Completion | Tests | Coverage |
|-------|--------|------------|-------|----------|
| **Phase 1: Foundation** | **COMPLETE** | 2025-12-18 | 218 | 82% |
| Phase 2: Core Agents | Ready to Start | - | - | - |
| Phase 3: Orchestration | Not Started | - | - | - |
| Phase 4: Extended | Not Started | - | - | - |
| Phase 5: Production | Not Started | - | - | - |

### Phase 1 Deliverables

- Indicator Library (17+ indicators, 91% coverage)
- Market Snapshot Builder (multi-timeframe, 74% coverage)
- Prompt Template System (tier-aware, 92% coverage)
- Database Schema (7 tables with retention/compression)
- API Endpoints (health, indicators, snapshots, debug)
- Configuration System (YAML with validation, 83% coverage)

See [Phase 1 Review](../reviews/phase-1/phase-1-comprehensive-review.md) for details.

---

## Executive Summary

This document provides a detailed implementation roadmap for the TripleGain LLM-assisted cryptocurrency trading system. The implementation follows the 5-phase dependency structure defined in the master design, with each phase building upon the previous phase's deliverables.

### Implementation Principles

| Principle | Description |
|-----------|-------------|
| **Incremental Delivery** | Each phase delivers working functionality |
| **Test-First** | Components must pass tests before integration |
| **Contract-Driven** | Clear interfaces between components |
| **Leverage Existing** | Build upon existing TimescaleDB infrastructure |
| **No Code Duplication** | Reuse existing collectors and data layers |

### LLM Model Assignments

| Role | Assigned Model(s) | Invocation |
|------|-------------------|------------|
| **Technical Analysis** | Qwen 2.5 7B (Local) | Per-minute |
| **Regime Detection** | Qwen 2.5 7B (Local) | Every 5 minutes |
| **Sentiment Analysis** | Grok + GPT (web search) | Every 30 minutes |
| **Trading Decision** | 6-Model A/B: GPT, Grok, DeepSeek V3, Claude Sonnet, Claude Opus, Qwen | Hourly |
| **Portfolio Rebalancing** | DeepSeek V3 | Hourly check |
| **Coordinator** | DeepSeek V3 / Claude Sonnet | On conflict |

---

## Existing Infrastructure (Do Not Redesign)

### TimescaleDB

| Component | Status | Details |
|-----------|--------|---------|
| Database | Operational | TimescaleDB with 5-9 years historical data |
| Hypertables | Active | `trades`, `candles` with chunking |
| Continuous Aggregates | Active | 8 timeframes (1m, 5m, 15m, 30m, 1h, 4h, 12h, 1d, 1w) |
| Retention Policies | Active | 90 days trades, 365 days candles |

### Data Coverage

| Symbol | Start Date | Candle Count |
|--------|------------|--------------|
| XRP/BTC | 2016-07-19 | Full history preserved |
| BTC/USDT | 2019-12-19 | Full history preserved |
| XRP/USDT | 2020-04-30 | Full history preserved |

### Collectors

| Collector | Location | Status |
|-----------|----------|--------|
| WebSocket DB Writer | `data/kraken_db/` | Ready |
| Gap Filler | `data/kraken_db/gap_filler.py` | Ready |
| Order Book Collector | `data/kraken_db/` | Ready |
| Private Data Collector | `data/kraken_db/` | Ready |

### Local LLM

| Component | Location | Status |
|-----------|----------|--------|
| Ollama | `/media/rese/2tb_drive/ollama_config/` | Ready |
| Qwen 2.5 7B | Via Ollama | Available |

---

## Phase Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         IMPLEMENTATION PHASES                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  PHASE 1: FOUNDATION                                                             │
│  ├── 1.1 Data Pipeline Extensions                                               │
│  ├── 1.2 Indicator Library                                                      │
│  ├── 1.3 Market Snapshot Builder                                                │
│  └── 1.4 Prompt Template System                                                 │
│      └── Deliverable: Working data→prompt pipeline                              │
│                                                                                  │
│  PHASE 2: CORE AGENTS                                                            │
│  ├── 2.1 Technical Analysis Agent                                               │
│  ├── 2.2 Regime Detection Agent                                                 │
│  ├── 2.3 Risk Management Engine                                                 │
│  └── 2.4 Trading Decision Agent                                                 │
│      └── Deliverable: Individual agents producing signals                       │
│                                                                                  │
│  PHASE 3: ORCHESTRATION                                                          │
│  ├── 3.1 Agent Communication Protocol                                           │
│  ├── 3.2 Coordinator Agent                                                      │
│  ├── 3.3 Portfolio Rebalancing Agent                                            │
│  └── 3.4 Order Execution Manager                                                │
│      └── Deliverable: Agents working together, executing trades                 │
│                                                                                  │
│  PHASE 4: EXTENDED FEATURES                                                      │
│  ├── 4.1 Sentiment Analysis Agent                                               │
│  ├── 4.2 Hodl Bag System                                                        │
│  ├── 4.3 LLM 6-Model A/B Testing Framework                                      │
│  └── 4.4 Dashboard                                                              │
│      └── Deliverable: Full feature set with monitoring                          │
│                                                                                  │
│  PHASE 5: PRODUCTION                                                             │
│  ├── 5.1 Comprehensive Testing                                                  │
│  ├── 5.2 Paper Trading Validation                                               │
│  ├── 5.3 Live Trading Deployment                                                │
│  └── 5.4 Monitoring & Alerting                                                  │
│      └── Deliverable: Production-ready trading system                           │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase Dependencies

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         DEPENDENCY GRAPH                                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  [Existing TimescaleDB] ─────────────────────────────────────────────┐          │
│  [Existing Collectors]  ─────────────────────────────────────────────┤          │
│  [Existing Ollama]      ─────────────────────────────────────────────┤          │
│                                                                      ↓          │
│                                                              ┌───────────────┐  │
│                                                              │   PHASE 1     │  │
│                                                              │  Foundation   │  │
│                                                              └───────┬───────┘  │
│                                                                      │          │
│                                       ┌──────────────────────────────┼──────┐   │
│                                       ↓                              ↓      ↓   │
│                              ┌───────────────┐              ┌───────────────┐   │
│                              │   PHASE 2     │              │   PHASE 2     │   │
│                              │  TA + Regime  │              │Risk (no deps) │   │
│                              └───────┬───────┘              └───────┬───────┘   │
│                                      │                              │           │
│                                      └──────────────┬───────────────┘           │
│                                                     ↓                           │
│                                            ┌───────────────┐                    │
│                                            │   PHASE 2     │                    │
│                                            │Trading Decision│                    │
│                                            └───────┬───────┘                    │
│                                                    │                            │
│                                                    ↓                            │
│                                            ┌───────────────┐                    │
│                                            │   PHASE 3     │                    │
│                                            │ Orchestration │                    │
│                                            └───────┬───────┘                    │
│                                                    │                            │
│                              ┌─────────────────────┼─────────────────────┐      │
│                              ↓                     ↓                     ↓      │
│                     ┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│                     │   PHASE 4     │     │   PHASE 4     │     │   PHASE 4     │
│                     │  Sentiment    │     │  Hodl Bag     │     │  Dashboard    │
│                     └───────┬───────┘     └───────┬───────┘     └───────┬───────┘
│                             └─────────────────────┼─────────────────────┘       │
│                                                   ↓                             │
│                                           ┌───────────────┐                     │
│                                           │   PHASE 5     │                     │
│                                           │  Production   │                     │
│                                           └───────────────┘                     │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Documents

| Document | Description | Phase |
|----------|-------------|-------|
| [01-phase-1-foundation.md](./01-phase-1-foundation.md) | Data pipeline, indicators, snapshots, prompts | Phase 1 |
| [02-phase-2-core-agents.md](./02-phase-2-core-agents.md) | TA, Regime, Risk, Trading Decision agents | Phase 2 |
| [03-phase-3-orchestration.md](./03-phase-3-orchestration.md) | Communication, Coordinator, Execution | Phase 3 |
| [04-phase-4-extended-features.md](./04-phase-4-extended-features.md) | Sentiment, Hodl Bag, A/B Testing, Dashboard | Phase 4 |
| [05-phase-5-production.md](./05-phase-5-production.md) | Testing, Paper Trading, Live Deployment | Phase 5 |

---

## Project Structure

```
triplegain/
├── src/
│   ├── agents/                    # Agent implementations
│   │   ├── __init__.py
│   │   ├── base_agent.py          # Abstract base agent class
│   │   ├── technical_analysis.py  # Technical Analysis Agent
│   │   ├── regime_detection.py    # Regime Detection Agent
│   │   ├── sentiment_analysis.py  # Sentiment Analysis Agent
│   │   ├── trading_decision.py    # Trading Decision Agent
│   │   ├── risk_management.py     # Risk Management Engine
│   │   ├── portfolio_rebalance.py # Portfolio Rebalancing Agent
│   │   └── coordinator.py         # Coordinator Agent
│   │
│   ├── data/                      # Data layer
│   │   ├── __init__.py
│   │   ├── market_snapshot.py     # Market Snapshot Builder
│   │   ├── indicator_library.py   # Technical Indicator Library
│   │   ├── feature_engineering.py # Feature calculation
│   │   └── data_quality.py        # Data validation
│   │
│   ├── llm/                       # LLM integration
│   │   ├── __init__.py
│   │   ├── client.py              # LLM client (Ollama + API)
│   │   ├── prompt_builder.py      # Prompt assembly
│   │   ├── output_parser.py       # Response parsing
│   │   └── model_comparison.py    # 6-model A/B framework
│   │
│   ├── risk/                      # Risk management
│   │   ├── __init__.py
│   │   ├── rules_engine.py        # Rules-based risk engine
│   │   ├── position_sizing.py     # Position size calculator
│   │   ├── circuit_breakers.py    # Circuit breaker logic
│   │   └── cooldown_manager.py    # Cooldown tracking
│   │
│   ├── execution/                 # Order execution
│   │   ├── __init__.py
│   │   ├── order_manager.py       # Order lifecycle
│   │   ├── position_tracker.py    # Position monitoring
│   │   ├── trade_logger.py        # Trade audit log
│   │   └── hodl_bag.py            # Hodl bag management
│   │
│   ├── orchestration/             # Agent orchestration
│   │   ├── __init__.py
│   │   ├── message_bus.py         # Inter-agent communication
│   │   ├── scheduler.py           # Agent invocation scheduler
│   │   └── consensus.py           # Consensus logic
│   │
│   └── api/                       # API layer
│       ├── __init__.py
│       ├── main.py                # FastAPI application
│       ├── routes/                # API routes
│       └── websocket.py           # WebSocket handlers
│
├── dashboard/                     # React dashboard
│   ├── src/
│   └── package.json
│
├── config/                        # Configuration files
│   ├── agents.yaml                # Agent configuration
│   ├── llm.yaml                   # LLM providers configuration
│   ├── risk.yaml                  # Risk parameters
│   └── system.yaml                # System settings
│
├── tests/                         # Test suite
│   ├── unit/
│   ├── integration/
│   └── backtests/
│
└── scripts/                       # Utility scripts
    ├── backtest.py
    └── paper_trade.py
```

---

## Key Interfaces Summary

### Agent Base Interface

All agents implement:
```python
class BaseAgent(ABC):
    agent_name: str
    llm_tier: str  # "local" | "api"

    @abstractmethod
    async def process(self, snapshot: MarketSnapshot) -> AgentOutput: ...

    @abstractmethod
    def get_output_schema(self) -> dict: ...
```

### Data Flow Interfaces

| Interface | Input | Output | Description |
|-----------|-------|--------|-------------|
| `MarketSnapshot` | Symbol, Timestamp | Structured market data | Complete market state |
| `PromptBuilder` | Snapshot, Context | Assembled prompt | Ready for LLM |
| `AgentOutput` | Agent response | Validated JSON | Parsed agent decision |
| `RiskValidation` | Trade proposal | Approved/Modified/Rejected | Risk-checked trade |

### Database Tables (New)

| Table | Purpose | Phase |
|-------|---------|-------|
| `agent_outputs` | Store agent decisions | Phase 1 |
| `trading_decisions` | Trade decision audit | Phase 2 |
| `trade_executions` | Executed trades | Phase 3 |
| `portfolio_snapshots` | Portfolio history | Phase 3 |
| `model_comparisons` | 6-model A/B results | Phase 4 |

---

## Success Criteria

### Phase Completion Gates

| Phase | Gate Criteria |
|-------|--------------|
| Phase 1 | Indicators calculated correctly, snapshots generated <500ms |
| Phase 2 | Agents produce valid outputs, risk engine rejects invalid trades |
| Phase 3 | Agents communicate, trades execute on paper |
| Phase 4 | Sentiment integrated, dashboard functional, A/B tracking 6 models |
| Phase 5 | Paper trading profitable, all tests pass, live deployment ready |

### System Targets (From Design)

| Metric | Target |
|--------|--------|
| Annual Return | > 50% |
| Maximum Drawdown | < 20% |
| Sharpe Ratio | > 1.5 |
| System Uptime | > 99% |
| Tier 1 Latency | < 500ms |

---

## References

- [Master Design README](../TripleGain-master-design/README.md)
- [Research Synthesis](../TripleGain-master-design/00-research-synthesis.md)
- [Multi-Agent Architecture](../TripleGain-master-design/01-multi-agent-architecture.md)
- [LLM Integration System](../TripleGain-master-design/02-llm-integration-system.md)
- [Risk Management Rules Engine](../TripleGain-master-design/03-risk-management-rules-engine.md)
- [Data Pipeline](../TripleGain-master-design/04-data-pipeline.md)
- [UI Requirements](../TripleGain-master-design/05-user-interface-requirements.md)
- [Evaluation Framework](../TripleGain-master-design/06-evaluation-framework.md)

---

*Implementation Plan v1.0 - December 2025*
*Ready for Phase 1 Development*
