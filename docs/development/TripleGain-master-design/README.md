# TripleGain Master Design

**Version**: 1.0
**Status**: Design Complete
**Date**: December 2025

---

## Executive Summary

TripleGain is an LLM-assisted cryptocurrency trading system designed to manage a portfolio of BTC, XRP, and USDT with a target allocation of 33/33/33. The system employs a multi-agent architecture with **6-model parallel A/B testing** (GPT, Grok, DeepSeek V3, Claude Sonnet, Claude Opus, Qwen 2.5 7B), rules-based risk management, and comprehensive evaluation frameworks.

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Multi-agent architecture** | Specialized agents outperform monolithic systems (TradingAgents research) |
| **6-Model A/B Testing** | All 6 LLMs run in parallel for Trading Decision to compare performance and build consensus |
| **Specialized Model Roles** | Grok/GPT for sentiment (web search), DeepSeek for rebalancing, DeepSeek/Claude for coordination |
| **Rules-based risk management** | Deterministic, auditable, no LLM dependency for safety |
| **Trend-following primary** | Mean reversion does NOT work on crypto (research finding) |
| **Conservative execution** | Quality over quantity (Claude's 3 trades beat Gemini's 44 in Alpha Arena) |
| **Hodl Bag with USDT** | 33/33/33 BTC/XRP/USDT allocation for profit accumulation |

### Target Metrics

| Metric | Target |
|--------|--------|
| Annual Return | > 50% |
| Maximum Drawdown | < 20% |
| Sharpe Ratio | > 1.5 |
| Win Rate | > 50% |
| System Uptime | > 99% |

---

## Design Documents

### 1. Research Synthesis
**File**: [00-research-synthesis.md](./00-research-synthesis.md)

Synthesizes findings from all research documentation:
- Alpha Arena LLM trading results
- Multi-agent framework analysis
- Algorithmic strategy evidence
- Technology stack evaluation

### 2. Multi-Agent Architecture
**File**: [01-multi-agent-architecture.md](./01-multi-agent-architecture.md)

Defines the six-agent system with specialized LLM assignments:
- **Technical Analysis Agent** (Tier 1 Local: Qwen 2.5 7B) - Indicator calculations
- **Regime Detection Agent** (Tier 1 Local: Qwen 2.5 7B) - Market state classification
- **Sentiment Analysis Agent** (Grok + GPT with web search) - News every 30 min
- **Trading Decision Agent** (6-Model A/B: GPT, Grok, DeepSeek, Claude Sonnet, Claude Opus, Qwen)
- **Risk Management Agent** (Rules-based) - Veto authority, position validation
- **Portfolio Rebalancing Agent** (DeepSeek for edge cases) - 33/33/33 allocation
- **Coordinator Agent** (DeepSeek V3 / Claude Sonnet) - Conflict resolution

### 3. LLM Integration System
**File**: [02-llm-integration-system.md](./02-llm-integration-system.md)

Details the multi-model LLM architecture:
- **Tier 1 (Local)**: Qwen 2.5 7B via Ollama for <500ms execution decisions
- **Tier 2 (API)**: 6 models with specialized roles:
  - GPT + Grok: Sentiment analysis (web search, every 30 min)
  - DeepSeek V3: Portfolio rebalancing, coordination
  - Claude Sonnet/Opus: Trading decisions, coordination
- **6-Model A/B Testing**: All models run in parallel for comparison
- Consensus decision logic (unanimous, majority, split)
- Model leaderboard and performance tracking

### 4. Risk Management Rules Engine
**File**: [03-risk-management-rules-engine.md](./03-risk-management-rules-engine.md)

Comprehensive risk framework:
- Position sizing (ATR-based, confidence-adjusted, regime-adjusted)
- Stop-loss rules (mandatory, ATR-based, trailing)
- Leverage limits (regime and drawdown adjusted)
- Circuit breakers (daily 5%, weekly 10%, max 20% drawdown)
- Confidence thresholds and cooldown periods
- Hodl bag allocation (10% of profits)

### 5. Data Pipeline
**File**: [04-data-pipeline.md](./04-data-pipeline.md)

End-to-end data flow design:
- Kraken WebSocket integration (trades, OHLC, order book, ticker)
- TimescaleDB schema (existing + new tables for agent outputs)
- Feature engineering (technical indicators, order book features)
- Market snapshot builder for LLM prompt injection
- Data quality monitoring

### 6. User Interface Requirements
**File**: [05-user-interface-requirements.md](./05-user-interface-requirements.md)

Dashboard specifications:
- Portfolio summary and allocation visualization
- Price charts with indicator overlays
- Open positions monitoring
- Agent status and decision logs
- Manual override controls
- System health monitoring
- LLM model comparison views

### 7. Evaluation Framework
**File**: [06-evaluation-framework.md](./06-evaluation-framework.md)

Performance measurement system:
- Success metrics from vision document
- Benchmark comparisons (Buy-and-hold, EMA crossover)
- Risk-adjusted metrics (Sharpe, Sortino, Calmar)
- Agent accuracy and confidence calibration
- **6-Model parallel comparison** with leaderboard
- Consensus analysis (unanimous, majority, split accuracy)
- Backtesting and walk-forward validation

---

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          TRIPLEGAIN SYSTEM ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                         EXTERNAL INTERFACES                                 │ │
│  │  Kraken WS/REST  │  News APIs  │  On-Chain  │  Ollama Local  │  LLM APIs  │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                        │                                         │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                         DATA LAYER                                          │ │
│  │  TimescaleDB (5-9 years historical) │ Redis Cache │ Feature Engineering    │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                        │                                         │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                         AGENT LAYER                                         │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐         │ │
│  │  │   TA     │ │  Regime  │ │Sentiment │ │ Trading  │ │Portfolio │         │ │
│  │  │  Agent   │ │  Agent   │ │  Agent   │ │ Decision │ │ Rebalance│         │ │
│  │  │ (Local)  │ │ (Local)  │ │  (API)   │ │  (API)   │ │ (Rules)  │         │ │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘         │ │
│  │                              │                                              │ │
│  │                    ┌─────────────────────┐                                 │ │
│  │                    │  Risk Management    │  ← VETO AUTHORITY               │ │
│  │                    │     (Rules)         │                                 │ │
│  │                    └─────────────────────┘                                 │ │
│  │                              │                                              │ │
│  │                    ┌─────────────────────┐                                 │ │
│  │                    │    Coordinator      │                                 │ │
│  │                    │      (API)          │                                 │ │
│  │                    └─────────────────────┘                                 │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                        │                                         │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                         EXECUTION LAYER                                     │ │
│  │  Order Manager  │  Position Tracker  │  Trade Logger  │  Hodl Bag Manager │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                        │                                         │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                         MONITORING LAYER                                    │ │
│  │  Dashboard  │  Alerts  │  Performance Analytics  │  LLM Comparison         │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Existing Infrastructure Leverage

| Component | Status | Usage in TripleGain |
|-----------|--------|---------------------|
| TimescaleDB | Operational | Primary data store, 5-9 years historical |
| Continuous Aggregates | Operational | 8 timeframes (1m to 1w) |
| WebSocket DB Writer | Ready | Real-time candle ingestion |
| Gap Filler | Ready | Historical data recovery |
| Order Book Collector | Ready | Depth data for features |
| Ollama (Qwen 2.5 7B) | Ready | Tier 1 local LLM |

---

## Implementation Dependencies

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         IMPLEMENTATION DEPENDENCIES                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  PHASE 1: FOUNDATION                                                             │
│  ├── 1.1 Data Pipeline (requires: existing TimescaleDB)                         │
│  ├── 1.2 Indicator Library (requires: 1.1)                                      │
│  ├── 1.3 Market Snapshot Builder (requires: 1.1, 1.2)                          │
│  └── 1.4 Basic Prompt Templates (requires: none)                                │
│                                                                                  │
│  PHASE 2: CORE AGENTS                                                            │
│  ├── 2.1 Technical Analysis Agent (requires: 1.2, 1.3, 1.4)                    │
│  ├── 2.2 Regime Detection Agent (requires: 1.2, 1.3, 1.4)                      │
│  ├── 2.3 Risk Management Engine (requires: none, rules-based)                  │
│  └── 2.4 Trading Decision Agent (requires: 2.1, 2.2, 2.3)                      │
│                                                                                  │
│  PHASE 3: ORCHESTRATION                                                          │
│  ├── 3.1 Agent Communication Protocol (requires: 2.*)                          │
│  ├── 3.2 Coordinator Agent (requires: 3.1)                                      │
│  ├── 3.3 Portfolio Rebalancing Agent (requires: 2.3)                           │
│  └── 3.4 Order Execution Manager (requires: 3.2)                               │
│                                                                                  │
│  PHASE 4: EXTENDED FEATURES                                                      │
│  ├── 4.1 Sentiment Analysis Agent (requires: external API integration)         │
│  ├── 4.2 Hodl Bag System (requires: 3.4)                                       │
│  ├── 4.3 LLM A/B Testing Framework (requires: 2.4)                             │
│  └── 4.4 Advanced Dashboard (requires: 3.*)                                    │
│                                                                                  │
│  PHASE 5: PRODUCTION                                                             │
│  ├── 5.1 Comprehensive Testing (requires: 4.*)                                 │
│  ├── 5.2 Paper Trading Validation (requires: 5.1)                              │
│  ├── 5.3 Live Trading Deployment (requires: 5.2)                               │
│  └── 5.4 Monitoring & Alerting (requires: 5.3)                                 │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Constraints

| Constraint | Source | Impact |
|------------|--------|--------|
| Latency: Tier 1 < 500ms | System requirement | Use local LLM for execution |
| Max Leverage: 5x | Risk policy | Hard cap in risk engine |
| Daily Loss: 5% | Risk policy | Circuit breaker trigger |
| Max Drawdown: 20% | Risk policy | Full halt trigger |
| API Budget: $5/day | Cost management | 6-model parallel testing budget |
| Kraken Rate Limits | Exchange policy | Request throttling required |

---

## Next Steps

1. **Review**: Stakeholder review of design documents
2. **Refinement**: Address feedback, clarify ambiguities
3. **Implementation Planning**: Create detailed implementation roadmap
4. **Development**: Phased implementation following dependencies
5. **Testing**: Unit, integration, and paper trading validation
6. **Deployment**: Staged rollout with monitoring

---

## Document Index

| Document | Description | Status |
|----------|-------------|--------|
| [00-research-synthesis.md](./00-research-synthesis.md) | Research findings synthesis | Complete |
| [01-multi-agent-architecture.md](./01-multi-agent-architecture.md) | Agent system design | Complete |
| [02-llm-integration-system.md](./02-llm-integration-system.md) | LLM configuration and prompts | Complete |
| [03-risk-management-rules-engine.md](./03-risk-management-rules-engine.md) | Risk rules and limits | Complete |
| [04-data-pipeline.md](./04-data-pipeline.md) | Data flow and storage | Complete |
| [05-user-interface-requirements.md](./05-user-interface-requirements.md) | Dashboard specifications | Complete |
| [06-evaluation-framework.md](./06-evaluation-framework.md) | Performance measurement | Complete |

---

## References

- [TripleGain Vision Document](../TripleGain-master-plan/TripleGain-vision.md)
- [Alpha Arena Research](../../research/TripleGain-master-plan_research/alpha-arena-agent-trading-deep-dive.md)
- [TensorTrade Analysis](../../research/TripleGain-master-plan_research/tensortrade-deep-dive.md)
- [Kraken API Reference](../../api/kraken/kraken-api-reference.md)

---

*Document generated: December 2025*
*Design Phase Complete - Ready for Implementation Planning*
