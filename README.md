# TripleGain

**Multi-Asset LLM-Assisted Trading System**

An autonomous cryptocurrency trading system using a multi-agent architecture with 6 LLMs for decision comparison and consensus-based trading.

## Overview

| Aspect | Details |
|--------|---------|
| **Objective** | Grow BTC, USDT, XRP holdings autonomously |
| **Target Allocation** | 33% BTC / 33% XRP / 33% USDT |
| **Trading Pairs** | BTC/USDT, XRP/USDT, XRP/BTC |
| **Starting Capital** | ~$2,100 |
| **Exchange** | Kraken (spot + margin) |
| **Deployment** | Paper trading → Micro-live → Scale |

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        ANALYSIS LAYER                             │
│  Technical Analysis │ Regime Detection │ Sentiment Analysis       │
│     (Qwen Local)    │   (Qwen Local)   │    (Grok + GPT)          │
└──────────────────────────────┬───────────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                    DECISION LAYER (6-Model A/B)                   │
│  GPT │ Grok │ DeepSeek V3 │ Claude Sonnet │ Claude Opus │ Qwen   │
└──────────────────────────────┬───────────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│  Risk Management (Rules) ──► Coordinator ──► Order Execution     │
│     VETO AUTHORITY              │                                │
└──────────────────────────────────────────────────────────────────┘
```

**Key Design Decisions**:
- **6-Model A/B Testing**: All LLMs run in parallel for comparison and consensus
- **Rules-Based Risk**: Deterministic risk management, no LLM override possible
- **Trend-Following**: Research shows mean reversion fails on crypto
- **Conservative Execution**: Quality over quantity

## Quick Start

```bash
# Start database
docker-compose up -d timescaledb

# Fill any data gaps
python -m data.kraken_db.gap_filler --db-url "$DATABASE_URL"

# Run tests
pytest
```

## Project Status

**Current Phase**: Pre-Phase 1 (Infrastructure Ready)

| Phase | Status | Description |
|-------|--------|-------------|
| Infrastructure | Complete | TimescaleDB, data collectors, Ollama |
| 1. Foundation | Not Started | Indicators, snapshots, prompts |
| 2. Core Agents | Not Started | TA, Regime, Risk, Trading Decision |
| 3. Orchestration | Not Started | Communication, Coordinator, Execution |
| 4. Extended | Not Started | Sentiment, Hodl Bag, Dashboard |
| 5. Production | Not Started | Testing, Paper/Live Trading |

## Data Infrastructure

- **Historical Data**: 5-9 years via TimescaleDB continuous aggregates
- **Symbols**: XRP/BTC (2016), BTC/USDT (2019), XRP/USDT (2020)
- **Timeframes**: 1m, 5m, 15m, 30m, 1h, 4h, 12h, 1d, 1w
- **Collectors**: WebSocket writer, gap filler, order book, private trades

## Target Metrics

| Metric | Target |
|--------|--------|
| Annual Return | > 50% |
| Maximum Drawdown | < 20% |
| Sharpe Ratio | > 1.5 |
| Win Rate | > 50% |
| System Uptime | > 99% |

## Risk Controls

- Max Leverage: 5x
- Daily Loss Limit: 5%
- Weekly Loss Limit: 10%
- Max Drawdown Circuit Breaker: 20%
- Required Stop-Loss on all trades

## Documentation

| Document | Description |
|----------|-------------|
| [Master Design](docs/development/TripleGain-master-design/README.md) | Complete system design |
| [Implementation Plan](docs/development/TripleGain-implementation-plan/README.md) | 5-phase roadmap |
| [Multi-Agent Architecture](docs/development/TripleGain-master-design/01-multi-agent-architecture.md) | Agent specifications |
| [Risk Management](docs/development/TripleGain-master-design/03-risk-management-rules-engine.md) | Risk rules engine |
| [Kraken API Reference](docs/api/kraken/kraken-api-reference.md) | Exchange integration |

## Tech Stack

- **Language**: Python 3.11+
- **Database**: TimescaleDB (PostgreSQL extension)
- **LLM Local**: Ollama (Qwen 2.5 7B)
- **LLM API**: OpenAI, Anthropic, xAI, DeepSeek
- **Dashboard**: React (planned)
- **Exchange**: Kraken REST/WebSocket API
