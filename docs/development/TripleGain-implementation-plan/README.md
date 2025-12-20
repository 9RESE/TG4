# TripleGain Implementation Plan

**Version**: 2.2
**Status**: Phase 8 Complete, Phases 9-10 Ready
**Date**: December 2025
**Last Updated**: 2025-12-20
**Source Design**: [TripleGain Master Design](../TripleGain-master-design/README.md)

## Current Status

| Phase | Status | Completion | Tests | Coverage |
|-------|--------|------------|-------|----------|
| **Phase 1: Foundation** | **COMPLETE** | 2025-12-18 | 232 | 82% |
| **Phase 2: Core Agents** | **COMPLETE** | 2025-12-18 | 136 | 67% |
| **Phase 3: Orchestration** | **COMPLETE** | 2025-12-18 | 227 | - |
| **Phase 4: API Security** | **COMPLETE** | 2025-12-18 | 110 | - |
| **Phase 5: Configuration** | **COMPLETE** | 2025-12-18 | - | - |
| **Phase 6: Paper Trading** | **COMPLETE** | 2025-12-19 | 157 | 87% |
| **Phase 7: Sentiment Analysis** | **COMPLETE** | 2025-12-19 | 37 | - |
| **Phase 8: Hodl Bag System** | **COMPLETE** | 2025-12-20 | 56 | - |
| Phase 9: 6-Model A/B Testing | Ready | - | - | - |
| Phase 10: React Dashboard | Ready | - | - | - |
| Phase 11: Production | Not Started | - | - | - |

**Total Tests**: 1202 passing (87% coverage)

### Recent Updates

**Phase 8 Complete (v0.6.0)**: Hodl Bag System with automated 10% profit allocation, 33.33% split across USDT/XRP/BTC, per-asset purchase thresholds ($1/$25/$15), and full paper trading support.

**Phase 7 Complete (v0.5.0)**: Sentiment Analysis Agent with dual-model architecture (Grok + GPT). Grok analyzes social/Twitter sentiment, GPT analyzes news sentiment. Both scores and full analysis reasoning are passed to trading decision LLMs.

**Phase 6 Complete (v0.4.2)**: Paper Trading with session persistence, simulated execution, and deep review fixes.

**Extended Features Split (v2.0)**: The original Phase 4 "Extended Features" has been split into four manageable phases (7-10) for better implementation focus.

---

## Executive Summary

This document provides a detailed implementation roadmap for the TripleGain LLM-assisted cryptocurrency trading system. The implementation follows an 11-phase dependency structure defined in the master design, with each phase building upon the previous phase's deliverables.

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
│  PHASE 1: FOUNDATION ✅                                                          │
│  ├── Data Pipeline Extensions                                                   │
│  ├── Indicator Library                                                          │
│  ├── Market Snapshot Builder                                                    │
│  └── Prompt Template System                                                     │
│                                                                                  │
│  PHASE 2: CORE AGENTS ✅                                                         │
│  ├── Technical Analysis Agent                                                   │
│  ├── Regime Detection Agent                                                     │
│  ├── Risk Management Engine                                                     │
│  └── Trading Decision Agent                                                     │
│                                                                                  │
│  PHASE 3: ORCHESTRATION ✅                                                       │
│  ├── Agent Communication Protocol                                               │
│  ├── Coordinator Agent                                                          │
│  ├── Portfolio Rebalancing Agent                                                │
│  └── Order Execution Manager                                                    │
│                                                                                  │
│  PHASE 4: API SECURITY ✅                                                        │
│  ├── JWT Authentication                                                         │
│  ├── RBAC Authorization                                                         │
│  └── Rate Limiting                                                              │
│                                                                                  │
│  PHASE 5: CONFIGURATION ✅                                                       │
│  ├── Config Validation                                                          │
│  └── Integration Fixes                                                          │
│                                                                                  │
│  PHASE 6: PAPER TRADING ✅                                                       │
│  ├── Simulated Execution                                                        │
│  ├── Session Persistence                                                        │
│  └── Paper Portfolio Tracking                                                   │
│                                                                                  │
│  PHASE 7: SENTIMENT ANALYSIS ✅                                                  │
│  ├── Grok Integration (web + Twitter)                                          │
│  ├── GPT Integration (web search)                                              │
│  └── Dual-Model Aggregation                                                    │
│                                                                                  │
│  PHASE 8: HODL BAG SYSTEM ✅                                                     │
│  ├── Profit Allocation (10%)                                                    │
│  ├── BTC/XRP/USDT Accumulation (33.33% each)                                    │
│  └── Per-Asset Thresholds ($1/$25/$15)                                          │
│                                                                                  │
│  PHASE 9: 6-MODEL A/B TESTING 🔵                                                 │
│  ├── Decision Recording                                                         │
│  ├── Outcome Tracking (1h, 4h, 24h)                                            │
│  ├── Model Leaderboard                                                          │
│  └── Pairwise Significance Tests                                                │
│                                                                                  │
│  PHASE 10: REACT DASHBOARD 🔵                                                    │
│  ├── Portfolio Overview                                                         │
│  ├── Position Monitoring                                                        │
│  ├── Agent Status Display                                                       │
│  ├── Model Comparison Views                                                     │
│  └── Control Panel                                                              │
│                                                                                  │
│  PHASE 11: PRODUCTION ⚪                                                         │
│  ├── Live Trading Deployment                                                    │
│  ├── Monitoring & Alerting                                                      │
│  └── Operational Runbooks                                                       │
│                                                                                  │
│  Legend: ✅ Complete  🔵 Ready  ⚪ Not Started                                   │
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
│                                                              │   PHASE 1 ✅   │  │
│                                                              │  Foundation   │  │
│                                                              └───────┬───────┘  │
│                                                                      │          │
│                                                                      ↓          │
│                                                              ┌───────────────┐  │
│                                                              │   PHASE 2 ✅   │  │
│                                                              │  Core Agents  │  │
│                                                              └───────┬───────┘  │
│                                                                      │          │
│                                                                      ↓          │
│                                                              ┌───────────────┐  │
│                                                              │   PHASE 3 ✅   │  │
│                                                              │ Orchestration │  │
│                                                              └───────┬───────┘  │
│                                                                      │          │
│                                       ┌──────────────────────────────┤          │
│                                       ↓                              ↓          │
│                              ┌───────────────┐              ┌───────────────┐   │
│                              │   PHASE 4 ✅   │              │   PHASE 5 ✅   │   │
│                              │  API Security │              │ Configuration │   │
│                              └───────┬───────┘              └───────┬───────┘   │
│                                      └──────────────┬───────────────┘           │
│                                                     ↓                           │
│                                            ┌───────────────┐                    │
│                                            │   PHASE 6 ✅   │                    │
│                                            │ Paper Trading │                    │
│                                            └───────┬───────┘                    │
│                                                    │                            │
│                  ┌─────────────────────────────────┼──────────────────────────┐ │
│                  ↓                 ↓               ↓                          ↓ │
│         ┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌───────────────┐│
│         │   PHASE 7 ✅   │ │   PHASE 8 ✅   │ │   PHASE 9 🔵   │ │  PHASE 10 🔵  ││
│         │  Sentiment    │ │   Hodl Bag    │ │  A/B Testing  │ │   Dashboard   ││
│         └───────┬───────┘ └───────┬───────┘ └───────┬───────┘ └───────┬───────┘│
│                 └─────────────────┴─────────────────┴─────────────────┘        │
│                                            │                                    │
│                                            ↓                                    │
│                                    ┌───────────────┐                           │
│                                    │  PHASE 11 ⚪   │                           │
│                                    │  Production   │                           │
│                                    └───────────────┘                           │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Documents

### Core Phases (Complete)

| Document | Description | Status |
|----------|-------------|--------|
| [01-phase-1-foundation.md](./01-phase-1-foundation.md) | Data pipeline, indicators, snapshots, prompts | ✅ Complete |
| [02-phase-2-core-agents.md](./02-phase-2-core-agents.md) | TA, Regime, Risk, Trading Decision agents | ✅ Complete |
| [03-phase-3-orchestration.md](./03-phase-3-orchestration.md) | Communication, Coordinator, Execution | ✅ Complete |
| [phase-3_5-paper-trading-plan.md](./phase-3_5-paper-trading-plan.md) | Paper Trading Integration (Phase 6) | ✅ Complete |

### Extended Features

| Document | Description | Status |
|----------|-------------|--------|
| [07-phase-7-sentiment-analysis.md](./07-phase-7-sentiment-analysis.md) | Sentiment Agent (Grok + GPT) | ✅ Complete |
| [08-phase-8-hodl-bag-system.md](./08-phase-8-hodl-bag-system.md) | Hodl Bag Accumulation | ✅ Complete |
| [09-phase-9-model-ab-testing.md](./09-phase-9-model-ab-testing.md) | 6-Model Comparison Framework | 🔵 Ready |
| [10-phase-10-react-dashboard.md](./10-phase-10-react-dashboard.md) | React Monitoring Dashboard | 🔵 Ready |

### Production

| Document | Description | Status |
|----------|-------------|--------|
| [05-phase-5-production.md](./05-phase-5-production.md) | Live Trading, Monitoring | ⚪ Not Started |

### Deprecated

| Document | Description | Status |
|----------|-------------|--------|
| [04-phase-4-extended-features.md](./04-phase-4-extended-features.md) | Original combined extended features | ❌ Deprecated |

> **Note**: The original Phase 4 has been split into Phases 7-10 for better implementation focus. See the individual phase documents above.

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
│   │   ├── sentiment_analysis.py  # Sentiment Analysis Agent (Phase 7)
│   │   ├── trading_decision.py    # Trading Decision Agent
│   │   └── portfolio_rebalance.py # Portfolio Rebalancing Agent
│   │
│   ├── data/                      # Data layer
│   │   ├── __init__.py
│   │   ├── market_snapshot.py     # Market Snapshot Builder
│   │   ├── indicator_library.py   # Technical Indicator Library
│   │   └── database.py            # Database utilities
│   │
│   ├── llm/                       # LLM integration
│   │   ├── __init__.py
│   │   ├── clients/               # Provider clients (OpenAI, Anthropic, etc.)
│   │   ├── prompt_builder.py      # Prompt assembly
│   │   └── model_comparison.py    # 6-model A/B framework (Phase 9)
│   │
│   ├── risk/                      # Risk management
│   │   ├── __init__.py
│   │   └── rules_engine.py        # Rules-based risk engine
│   │
│   ├── execution/                 # Order execution
│   │   ├── __init__.py
│   │   ├── order_manager.py       # Order lifecycle
│   │   ├── position_tracker.py    # Position monitoring
│   │   ├── paper_executor.py      # Paper trading execution
│   │   ├── paper_portfolio.py     # Paper portfolio tracking
│   │   └── hodl_bag.py            # Hodl bag management (Phase 8)
│   │
│   ├── orchestration/             # Agent orchestration
│   │   ├── __init__.py
│   │   ├── message_bus.py         # Inter-agent communication
│   │   └── coordinator.py         # Coordinator agent
│   │
│   └── api/                       # API layer
│       ├── __init__.py
│       ├── app.py                 # FastAPI application
│       ├── security.py            # JWT/RBAC authentication
│       ├── routes_agents.py       # Agent endpoints
│       ├── routes_orchestration.py # Orchestration endpoints
│       ├── routes_paper_trading.py # Paper trading endpoints
│       ├── routes_sentiment.py    # Sentiment endpoints (Phase 7)
│       └── routes_hodl.py         # Hodl bag endpoints (Phase 8)
│
├── dashboard/                     # React dashboard (Phase 10)
│   ├── src/
│   └── package.json
│
├── config/                        # Configuration files
│   ├── agents.yaml                # Agent configuration
│   ├── risk.yaml                  # Risk parameters
│   ├── orchestration.yaml         # Orchestration settings
│   ├── portfolio.yaml             # Portfolio settings
│   ├── execution.yaml             # Execution settings
│   └── hodl.yaml                  # Hodl bag settings (Phase 8)
│
├── tests/                         # Test suite (1106 tests)
│   ├── unit/
│   │   ├── agents/
│   │   ├── risk/
│   │   ├── orchestration/
│   │   ├── execution/
│   │   ├── llm/
│   │   └── api/
│   └── integration/
│
└── migrations/                    # Database migrations
    ├── 001_initial_schema.sql
    ├── 002_phase2_agents.sql
    ├── 003_phase3_orchestration.sql
    ├── 004_paper_trading.sql
    ├── 005_session_persistence.sql
    └── ...
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

### Database Tables

| Table | Purpose | Phase |
|-------|---------|-------|
| `agent_outputs` | Store agent decisions | Phase 1 |
| `trading_decisions` | Trade decision audit | Phase 2 |
| `trade_executions` | Executed trades | Phase 3 |
| `portfolio_snapshots` | Portfolio history | Phase 3 |
| `paper_sessions` | Paper trading sessions | Phase 6 |
| `sentiment_outputs` | Sentiment analysis | Phase 7 |
| `hodl_bags` | Hodl bag holdings | Phase 8 |
| `model_decisions` | Model comparison | Phase 9 |
| `model_leaderboard` | Model rankings | Phase 9 |

---

## Success Criteria

### Phase Completion Gates

| Phase | Gate Criteria | Status |
|-------|--------------|--------|
| Phase 1 | Indicators correct, snapshots <500ms | ✅ |
| Phase 2 | Agents valid outputs, risk rejects invalid | ✅ |
| Phase 3 | Agents communicate, trades execute on paper | ✅ |
| Phase 4 | JWT auth works, RBAC enforced | ✅ |
| Phase 5 | Config validates, integration works | ✅ |
| Phase 6 | Paper trading functional, sessions persist | ✅ |
| Phase 7 | Sentiment aggregates from Grok + GPT | ✅ |
| Phase 8 | Hodl bags accumulate from profits, thresholds work | ✅ |
| Phase 9 | All 6 models tracked, leaderboard accurate | 🔵 |
| Phase 10 | Dashboard displays all data, controls work | 🔵 |
| Phase 11 | Live trading profitable, monitoring active | ⚪ |

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

*Implementation Plan v2.1 - December 2025*
*Phase 7 Complete - Phases 8-10 Ready for Development*
