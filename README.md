# TripleGain - Multi-Asset LLM-Assisted Trading System

An autonomous trading system that uses LLMs as decision-making agents to grow holdings across BTC, USDT, and XRP.

## Overview

| Aspect | Details |
|--------|---------|
| **Objective** | Grow BTC, USDT, XRP holdings autonomously |
| **Trading Pairs** | BTC/USDT, XRP/USDT, XRP/BTC |
| **Starting Capital** | ~$2,100 (1,000 USDT + 500 XRP) |
| **Exchange** | Kraken (primary), Bybit (futures expansion) |
| **Mode** | Paper trading → Micro-live → Scale |

## Architecture

Multi-agent system following TradingAgents/Nof1.ai patterns:

```
┌─────────────────────────────────────────────────────────────┐
│                     AGENT LAYER                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  Technical   │  │    Risk      │  │   Trading    │       │
│  │  Analysis    │  │  Management  │  │   Decision   │       │
│  │   (Local)    │  │   (Rules)    │  │    (LLM)     │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
│                           │                                  │
│                    ┌──────────────┐                         │
│                    │  Portfolio   │                         │
│                    │ Rebalancing  │                         │
│                    └──────────────┘                         │
└─────────────────────────────────────────────────────────────┘
```

## Historical Data

The system has access to **5-9 years of historical data** via TimescaleDB continuous aggregates:

| Symbol | Coverage | Daily Candles |
|--------|----------|---------------|
| XRP/BTC | 2016-07-19 → Present | 3,355 (9 years) |
| BTC/USDT | 2019-12-19 → Present | 2,190 (5 years) |
| XRP/USDT | 2020-04-30 → Present | 2,057 (5 years) |

**Aggregate Holdings:**
- 1.5M+ 5-minute candles
- 178K+ hourly candles
- 7.6K+ daily candles

## LLM Models

Testing multiple LLMs with identical prompts:

| Model | Provider | Use Case |
|-------|----------|----------|
| Claude Sonnet 4.5 | Anthropic | Primary (conservative) |
| GPT-4o | OpenAI | Backup (balanced) |
| Grok 4 | xAI | Comparison |
| Deepseek V3 | Deepseek | Cost-efficient |
| Qwen 2.5 7B | Ollama (local) | Routine analysis |

## Trading Constraints

| Parameter | Value |
|-----------|-------|
| Max Leverage | 3x |
| Risk per Trade | 1% of portfolio |
| Max Stop-Loss Distance | 2% from entry |
| Min Risk:Reward | 2:1 |
| Max Drawdown Trigger | 10% (pause trading) |
| Confidence Threshold | 0.6 minimum |
| Trade Cooldown | 30 minutes per pair |

## Strategy Focus

- **Primary**: Trend-following momentum (proven effective in crypto)
- **Secondary**: Volatility breakout (Bollinger squeeze entries)
- **Avoid**: Mean reversion RSI (doesn't work on BTC)

## Development Phases

1. **Phase 1: Foundation** - LLM strategy module, prompt engineering, backtesting
2. **Phase 2: Paper Trading** - Minimum 30 days with all LLMs
3. **Phase 3: Micro-Live** - $100 USDT live with best-performing LLM
4. **Phase 4: Scale** - Gradual increase based on performance

## Success Metrics

- Sharpe Ratio > 1.5
- Max Drawdown < 15%
- Win Rate > 50%
- All three assets increasing over 90-day rolling window

## Infrastructure

- **Database**: TimescaleDB with continuous aggregates
- **Cache**: Redis for state management
- **Local LLM**: Ollama on `/media/rese/2tb_drive/ollama_config/`
- **Monitoring**: Prometheus/Grafana

## Quick Start

```bash
# Start database
docker-compose up -d timescaledb

# Fill data gaps
python -m data.kraken_db.gap_filler --db-url "$DATABASE_URL"

# Run tests
pytest
```

## Documentation

- [System Architecture](docs/development/master-plan/01-system-architecture.md)
- [LLM Prompt Templates](docs/development/master-plan/02-llm-prompt-templates.md)
- [Multi-Agent Coordination](docs/development/master-plan/03-multi-agent-coordination.md)
- [Risk Management Rules](docs/development/master-plan/04-risk-management-rules-engine.md)
- [Implementation Roadmap](docs/development/master-plan/05-implementation-roadmap.md)
- [LLM Evaluation Framework](docs/development/master-plan/06-llm-evaluation-framework.md)
- [Kraken DB Architecture](docs/architecture/05-building-blocks/kraken-db.md)
- [Kraken API Reference](docs/api/kraken-api-reference.md)
