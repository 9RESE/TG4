# Phase 1: Foundation Layer

**Status**: COMPLETE
**Completion Date**: 2025-12-18
**Test Coverage**: 82% (218 tests)

---

## Overview

Phase 1 establishes the foundation layer for the TripleGain LLM-assisted trading system. It provides the data pipeline that transforms raw market data into LLM-ready prompts.

## Components

### 1. Indicator Library

**Location**: `triplegain/src/data/indicator_library.py`
**Coverage**: 91%

Calculates 17+ technical indicators using NumPy vectorization.

| Category | Indicators |
|----------|------------|
| Trend | EMA (9,21,50,200), SMA (20,50,200), ADX (14), Supertrend (10/3.0) |
| Momentum | RSI (14), MACD (12/26/9), Stochastic RSI (14/14/3/3), ROC (10) |
| Volatility | ATR (14), Bollinger Bands (20/2.0), Keltner Channels (20/10/2.0) |
| Volume | OBV, VWAP, Volume SMA (20), Volume vs Average |
| Pattern | Choppiness Index (14), Squeeze Detection |

**Performance**: <50ms for 1000 candles, all indicators

### 2. Market Snapshot Builder

**Location**: `triplegain/src/data/market_snapshot.py`
**Coverage**: 74%

Aggregates multi-timeframe market data into structured snapshots.

**Features**:
- Multi-timeframe candle aggregation (1m, 5m, 15m, 1h, 4h, 1d)
- Order book feature extraction (imbalance, spread, depth)
- Multi-timeframe trend alignment scoring
- Data quality validation and flags
- Two output formats:
  - `to_prompt_format()` - Full JSON for API LLMs
  - `to_compact_format()` - Minimal JSON for local LLMs

### 3. Prompt Builder

**Location**: `triplegain/src/llm/prompt_builder.py`
**Coverage**: 92%

Assembles LLM prompts with token budget management.

**Features**:
- Template-based system prompts per agent
- Tier-aware token budgets (local vs API)
- Automatic truncation to fit budget
- Portfolio context injection
- Agent output aggregation for coordinator

**Token Budgets**:
| Tier | Total | System | Market Data | Buffer |
|------|-------|--------|-------------|--------|
| Local | 8,192 | 1,500 | 3,000 | 2,492 |
| API | 128,000 | 3,000 | 6,000 | 116,000 |

### 4. Database Layer

**Location**: `triplegain/src/data/database.py`
**Coverage**: 82%

Async database operations with connection pooling.

**Features**:
- asyncpg connection pooling
- Candle fetching from continuous aggregates
- Order book data retrieval
- Agent output storage
- Health check queries

### 5. Configuration System

**Location**: `triplegain/src/utils/config.py`
**Coverage**: 83%

YAML-based configuration with validation.

**Features**:
- Environment variable substitution (`${VAR}` syntax)
- Schema validation
- Default value handling
- Sensitive value masking for debug endpoints

### 6. API Endpoints

**Location**: `triplegain/src/api/app.py`
**Coverage**: 62%

FastAPI REST endpoints for testing and monitoring.

| Endpoint | Purpose |
|----------|---------|
| `/health` | Full health check with DB/Ollama status |
| `/health/live` | Kubernetes liveness probe |
| `/health/ready` | Kubernetes readiness probe |
| `/api/v1/indicators/{symbol}/{timeframe}` | Calculate indicators |
| `/api/v1/snapshot/{symbol}` | Build market snapshot |
| `/api/v1/debug/prompt/{agent}` | Preview assembled prompt |
| `/api/v1/debug/config` | View sanitized config |

### 7. Database Schema

**Location**: `migrations/001_agent_tables.sql`

| Table | Purpose | Features |
|-------|---------|----------|
| `agent_outputs` | LLM outputs | Hypertable, 90-day retention |
| `trading_decisions` | Decision audit | CHECK constraints |
| `trade_executions` | Trade records | Foreign key to decisions |
| `portfolio_snapshots` | Portfolio history | Hypertable, compression |
| `risk_state` | Risk tracking | Circuit breaker state |
| `external_data_cache` | API cache | 30-day retention |
| `indicator_cache` | Indicator cache | 7-day retention, compression |

## Usage

### Calculate Indicators

```python
from triplegain.src.data.indicator_library import IndicatorLibrary

config = {
    'ema': {'periods': [9, 21, 50, 200]},
    'rsi': {'period': 14},
    'macd': {'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
}
library = IndicatorLibrary(config)

# candles = list of {'open': x, 'high': x, 'low': x, 'close': x, 'volume': x}
indicators = library.calculate_all('BTC/USDT', '1h', candles)
```

### Build Market Snapshot

```python
from triplegain.src.data.market_snapshot import MarketSnapshotBuilder

builder = MarketSnapshotBuilder(db_pool=None, indicator_library=library, config={})
snapshot = builder.build_snapshot_from_candles(
    symbol='BTC/USDT',
    candles_by_timeframe={'1h': hourly_candles, '4h': four_hour_candles}
)

# For local LLM
compact = snapshot.to_compact_format()

# For API LLM
full = snapshot.to_prompt_format(token_budget=6000)
```

### Build Prompt

```python
from triplegain.src.llm.prompt_builder import PromptBuilder, PortfolioContext

builder = PromptBuilder(config)
prompt = builder.build_prompt(
    agent_name='technical_analysis',
    snapshot=snapshot,
    portfolio_context=PortfolioContext(
        total_equity_usd=Decimal('10000'),
        available_margin_usd=Decimal('8000'),
        positions=[],
        allocation={'BTC': Decimal('0.33'), 'XRP': Decimal('0.33'), 'USDT': Decimal('0.34')},
        daily_pnl_pct=Decimal('0.02'),
        drawdown_pct=Decimal('0.05'),
        consecutive_losses=0,
        win_rate_7d=Decimal('0.55')
    )
)

# Send to LLM
system_prompt = prompt.system_prompt
user_message = prompt.user_message
```

## Testing

```bash
# Run all Phase 1 tests
pytest triplegain/tests/unit/ -v

# Run with coverage
pytest triplegain/tests/unit/ --cov=triplegain/src --cov-report=term-missing

# Run specific test module
pytest triplegain/tests/unit/test_indicator_library.py -v
```

## Reviews

- [Phase 1 Review Report](../reviews/phase-1/phase-1-review-report.md)
- [Phase 1 Deep Review](../reviews/phase-1/phase-1-deep-review.md)
- [Phase 1 Comprehensive Review](../reviews/phase-1/phase-1-comprehensive-review.md)

## References

- [Implementation Plan](../TripleGain-implementation-plan/01-phase-1-foundation.md)
- [ADR-001: Phase 1 Architecture](../../architecture/09-decisions/ADR-001-phase1-foundation-architecture.md)
- [Building Blocks](../../architecture/05-building-blocks/README.md)

---

*Phase 1 Feature Documentation - December 2025*
