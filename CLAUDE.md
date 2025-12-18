# TripleGain - LLM-Assisted Trading System

## Project Type
Python trading system with TimescaleDB + multi-LLM agents

## Commands
- Test: `pytest`
- Database: `docker-compose up -d timescaledb`
- Gap Filler: `python -m data.kraken_db.gap_filler --db-url "$DATABASE_URL"`

## Key Facts (Memory)

### Historical Database
- **5-9 years of historical data** available via TimescaleDB continuous aggregates
- Symbols: XRP/USDT, BTC/USDT, XRP/BTC
- Coverage: XRP/BTC since 2016-07-19, BTC/USDT since 2019-12-19, XRP/USDT since 2020-04-30
- 1.5M+ 5-minute candles, 178K+ hourly candles, 7.6K+ daily candles
- Base tables have retention (90 days trades, 365 days 1m candles), but aggregates preserve full history

### Infrastructure
- Local LLM models stored at: `/media/rese/2tb_drive/ollama_config/`
- Database: TimescaleDB with continuous aggregates for 8 timeframes
- Data collectors ready: order book depth, private trade history

### LLM Models for Comparison
- Claude Sonnet 4.5 (Primary)
- GPT-4o (Backup)
- Grok 4 (Comparison)
- Deepseek V3 (Cost-efficient)
- Qwen 2.5 7B via Ollama (Local)

## Project Notes
- Multi-agent architecture: Technical Analysis, Risk Management, Trading Decision, Portfolio Rebalancing
- Trading constraints: 3x max leverage, 1% risk per trade, 2:1 min risk:reward
- Strategy focus: Trend-following momentum, volatility breakout

## Architecture
See `docs/architecture/` for detailed documentation.
See `docs/development/master-plan/` for system design and implementation roadmap.

## Key Documentation
- [System Architecture](docs/development/master-plan/01-system-architecture.md)
- [Kraken DB Building Block](docs/architecture/05-building-blocks/kraken-db.md)
- [Kraken API Reference](docs/api/kraken-api-reference.md)
- [Data Gap Analysis](docs/api/kraken-data-gap-analysis.md)

---
*Uses global config from ~/.claude/CLAUDE.md*
