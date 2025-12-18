# TripleGain - LLM-Assisted Trading System

## Project Type
Python trading system with TimescaleDB + 6-model LLM comparison

## Quick Commands
```bash
pytest triplegain/tests/                            # Run all tests (218 passing)
pytest --cov=triplegain/src --cov-report=term       # Run with coverage (82%)
docker-compose up -d timescaledb                    # Start database
python -m data.kraken_db.gap_filler --db-url "$DB_URL"  # Fill data gaps
uvicorn triplegain.src.api.app:app --reload         # Start API server
```

## Current Phase: Phase 2 Ready

**Status**: Phase 1 COMPLETE (2025-12-18), ready for Phase 2 core agents

| Phase | Status | Description |
|-------|--------|-------------|
| 1. Foundation | **COMPLETE** | Data pipeline, indicators, snapshots, prompts |
| 2. Core Agents | Ready to Start | TA, Regime, Risk, Trading Decision agents |
| 3. Orchestration | Not Started | Communication, Coordinator, Execution |
| 4. Extended | Not Started | Sentiment, Hodl Bag, A/B Testing, Dashboard |
| 5. Production | Not Started | Testing, Paper Trading, Live Deployment |

### Phase 1 Deliverables (Completed)
- **Indicator Library**: 17+ indicators (EMA, SMA, RSI, MACD, ATR, BB, ADX, OBV, VWAP, Supertrend, StochRSI, ROC, Keltner, Choppiness, Volume SMA)
- **Market Snapshot Builder**: Multi-timeframe aggregation with compact/full formats
- **Prompt Template System**: Tier-aware (local/API) with token budget management
- **Database Schema**: 7 tables with retention/compression policies
- **API Endpoints**: Health, indicators, snapshots, debug routes
- **Config System**: YAML-based with env var substitution and validation
- **Test Coverage**: 82% (218 tests passing)

## Key Facts (Memory)

### Data Available
- **5-9 years historical** via TimescaleDB continuous aggregates
- Symbols: XRP/USDT (2020), BTC/USDT (2019), XRP/BTC (2016)
- Aggregates: 1m, 5m, 15m, 30m, 1h, 4h, 12h, 1d, 1w
- Base retention: 90d trades, 365d 1m candles (aggregates preserve full history)

### Infrastructure Ready
- Ollama: `/media/rese/2tb_drive/ollama_config/` (Qwen 2.5 7B ready)
- Collectors: WebSocket writer, gap filler, order book, private data

### Critical Design Decisions
| Decision | Rationale |
|----------|-----------|
| 6-Model A/B Testing | GPT, Grok, DeepSeek V3, Claude Sonnet/Opus, Qwen run parallel |
| Rules-based Risk | Deterministic, no LLM dependency, <10ms execution |
| Trend-following | Mean reversion does NOT work on crypto |
| Conservative | Quality over quantity (3 trades beat 44 in Alpha Arena) |
| 33/33/33 Allocation | BTC/XRP/USDT with Hodl Bag (10% of profits) |

### LLM Model Roles
| Role | Model | Frequency |
|------|-------|-----------|
| Technical Analysis | Qwen 2.5 7B (Local) | Per-minute |
| Regime Detection | Qwen 2.5 7B (Local) | Every 5 min |
| Sentiment | Grok + GPT (web search) | Every 30 min |
| Trading Decision | 6-Model A/B | Hourly |
| Coordinator | DeepSeek V3 / Claude | On conflict |

### System Constraints
- Max Leverage: 5x | Daily Loss Limit: 5% | Max Drawdown: 20%
- Tier 1 Latency: <500ms | API Budget: ~$5/day

### Target Metrics
- Annual Return: >50% | Sharpe: >1.5 | Win Rate: >50% | Uptime: >99%

## Project Structure
```
triplegain/
├── src/
│   ├── data/           # Indicator library, market snapshot, database
│   ├── llm/            # Prompt builder, templates
│   ├── api/            # FastAPI endpoints
│   └── utils/          # Config loader
├── tests/
│   ├── unit/           # 218 unit tests
│   └── integration/    # Database integration tests
config/                 # YAML configuration files
data/kraken_db/         # Data collectors (operational)
docs/development/       # Design + Implementation plans + Reviews
migrations/             # Database migrations
```

## Documentation
- [Master Design](docs/development/TripleGain-master-design/README.md)
- [Implementation Plan](docs/development/TripleGain-implementation-plan/README.md)
- [Kraken API](docs/api/kraken/kraken-api-reference.md)

---
*Uses global config from ~/.claude/CLAUDE.md*
