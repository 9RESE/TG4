# C4 Component Diagram

**Last Updated**: 2025-12-18 (Phase 3 Complete)

## Trading App Components

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           Trading App (Python)                                │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                        Data Layer (Phase 1)                              │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │ │
│  │  │  WebSocket  │  │   Market    │  │  Indicator  │  │   Prompt    │    │ │
│  │  │   Client    │  │  Snapshot   │  │   Library   │  │   Builder   │    │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                        Agent Layer (Phase 2)                             │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │ │
│  │  │  Technical  │  │   Regime    │  │   Trading   │  │    Risk     │    │ │
│  │  │  Analysis   │  │  Detection  │  │  Decision   │  │   Engine    │    │ │
│  │  │   Agent     │  │   Agent     │  │   Agent     │  │  (No LLM)   │    │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                     Orchestration Layer (Phase 3)                        │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                      │ │
│  │  │   Message   │  │ Coordinator │  │  Portfolio  │                      │ │
│  │  │     Bus     │  │    Agent    │  │  Rebalance  │                      │ │
│  │  │  (Pub/Sub)  │  │ (DeepSeek)  │  │   Agent     │                      │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                      │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                      Execution Layer (Phase 3)                           │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                      │ │
│  │  │    Order    │  │  Position   │  │   Kraken    │                      │ │
│  │  │  Execution  │  │   Tracker   │  │    API      │                      │ │
│  │  │   Manager   │  │             │  │   Client    │                      │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                      │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                       LLM Layer (Phase 2)                                │ │
│  │  ┌───────┐  ┌───────┐  ┌───────┐  ┌───────┐  ┌───────┐  ┌───────┐     │ │
│  │  │Ollama │  │OpenAI │  │Anthro-│  │ Deep  │  │  xAI  │  │ Base  │     │ │
│  │  │Client │  │Client │  │pic    │  │ Seek  │  │(Grok) │  │Client │     │ │
│  │  │(Qwen) │  │(GPT-4)│  │(Claude)│  │ V3   │  │       │  │       │     │ │
│  │  └───────┘  └───────┘  └───────┘  └───────┘  └───────┘  └───────┘     │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                    Infrastructure Layer                                  │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │ │
│  │  │   Logger    │  │  Database   │  │   Config    │  │  FastAPI    │    │ │
│  │  │             │  │(TimescaleDB)│  │   Manager   │  │    App      │    │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

### Data Layer (Phase 1)

| Component | Responsibility | File Location |
|-----------|----------------|---------------|
| WebSocket Client | Real-time Kraken data stream | `data/kraken_db/` |
| Market Snapshot | Multi-timeframe data aggregation | `src/data/market_snapshot.py` |
| Indicator Library | 17+ technical indicators (EMA, RSI, MACD, etc.) | `src/data/indicator_library.py` |
| Prompt Builder | Tier-aware prompt construction | `src/llm/prompt_builder.py` |

### Agent Layer (Phase 2)

| Component | Responsibility | Model | File Location |
|-----------|----------------|-------|---------------|
| Technical Analysis Agent | Trend/momentum analysis | Qwen 2.5 7B | `src/agents/technical_analysis.py` |
| Regime Detection Agent | Market regime classification | Qwen 2.5 7B | `src/agents/regime_detection.py` |
| Trading Decision Agent | 6-model consensus decisions | Multiple | `src/agents/trading_decision.py` |
| Risk Engine | Deterministic validation (<10ms) | None (rules) | `src/risk/rules_engine.py` |

### Orchestration Layer (Phase 3)

| Component | Responsibility | Model | File Location |
|-----------|----------------|-------|---------------|
| Message Bus | Inter-agent pub/sub communication | None | `src/orchestration/message_bus.py` |
| Coordinator Agent | Agent scheduling, conflict resolution | DeepSeek V3 / Claude | `src/orchestration/coordinator.py` |
| Portfolio Rebalance Agent | 33/33/33 allocation maintenance | DeepSeek V3 | `src/agents/portfolio_rebalance.py` |

### Execution Layer (Phase 3)

| Component | Responsibility | File Location |
|-----------|----------------|---------------|
| Order Execution Manager | Order lifecycle, Kraken API | `src/execution/order_manager.py` |
| Position Tracker | P&L calculation, SL/TP monitoring | `src/execution/position_tracker.py` |
| Kraken API Client | Exchange communication | External dependency |

### LLM Layer (Phase 2)

| Component | Provider | Models | File Location |
|-----------|----------|--------|---------------|
| Ollama Client | Ollama (Local) | Qwen 2.5 7B | `src/llm/clients/ollama.py` |
| OpenAI Client | OpenAI | GPT-4-turbo | `src/llm/clients/openai_client.py` |
| Anthropic Client | Anthropic | Claude Sonnet/Opus | `src/llm/clients/anthropic_client.py` |
| DeepSeek Client | DeepSeek | DeepSeek V3 | `src/llm/clients/deepseek_client.py` |
| xAI Client | xAI | Grok-2-1212 | `src/llm/clients/xai_client.py` |

### Infrastructure Layer

| Component | Responsibility | File Location |
|-----------|----------------|---------------|
| Logger | Structured JSON logging | Python logging |
| Database | TimescaleDB with hypertables | `src/data/database.py` |
| Config Manager | YAML config with env substitution | `src/utils/config.py` |
| FastAPI App | REST API endpoints | `src/api/app.py` |

## Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA FLOW                                       │
└─────────────────────────────────────────────────────────────────────────────┘

  Kraken WebSocket                                              Kraken API
       │                                                            │
       ▼                                                            ▼
  ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐  ┌─────────┐
  │  Trade  │ ──▶ │ Market  │ ──▶ │   TA    │ ──▶ │Coordi-  │  │  Order  │
  │  Data   │     │Snapshot │     │  Agent  │     │ nator   │──▶│Execution│
  └─────────┘     └─────────┘     └─────────┘     └─────────┘  └─────────┘
                       │                               │            │
                       ▼                               │            ▼
                  ┌─────────┐                          │       ┌─────────┐
                  │ Regime  │ ─────────────────────────┤       │Position │
                  │  Agent  │                          │       │ Tracker │
                  └─────────┘                          │       └─────────┘
                       │                               │
                       ▼                               │
                  ┌─────────┐     ┌─────────┐         │
                  │Trading  │ ──▶ │  Risk   │ ────────┘
                  │Decision │     │ Engine  │
                  └─────────┘     └─────────┘
                       │
                       ▼
                  ┌─────────┐
                  │Message  │
                  │  Bus    │
                  └─────────┘
```

## Message Topics

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MESSAGE BUS TOPICS                                 │
└─────────────────────────────────────────────────────────────────────────────┘

  MARKET_DATA ──────┬──────────────────────────────────────────────────────┐
                    │                                                       │
  TA_SIGNALS ───────┼──────────┬──────────────────────────────────────────┤
                    │          │                                           │
  REGIME_UPDATES ───┼──────────┼──────────┬───────────────────────────────┤
                    │          │          │                                │
  TRADING_SIGNALS ──┼──────────┼──────────┼──────────┬────────────────────┤
                    │          │          │          │                     │
  RISK_ALERTS ──────┼──────────┼──────────┼──────────┼──────────┬─────────┤
                    │          │          │          │          │          │
  EXECUTION_EVENTS ─┼──────────┼──────────┼──────────┼──────────┼─────────┤
                    │          │          │          │          │          │
  PORTFOLIO_UPDATES ┼──────────┼──────────┼──────────┼──────────┼─────────┤
                    │          │          │          │          │          │
                    ▼          ▼          ▼          ▼          ▼          ▼
               ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
               │  TA    │ │ Regime │ │Trading │ │  Risk  │ │ Coord  │ │Portf.  │
               │ Agent  │ │ Agent  │ │ Agent  │ │ Engine │ │ Agent  │ │ Agent  │
               └────────┘ └────────┘ └────────┘ └────────┘ └────────┘ └────────┘
```

## References

- [Phase 1 Features](../../development/features/phase-1-foundation.md)
- [Phase 2 Features](../../development/features/phase-2-core-agents.md)
- [Phase 3 Features](../../development/features/phase-3-orchestration.md)
- [Multi-Agent Architecture](../../development/TripleGain-master-design/01-multi-agent-architecture.md)

---

*C4 Component Diagram v3.0 - Phase 3 Complete - December 2025*
