# 05 - Building Block View

## Level 1: System Overview

```
+------------------------------------------------------------------+
|                    TripleGain Trading System                      |
+------------------------------------------------------------------+
|                                                                    |
|  +----------------+  +----------------+  +--------------------+   |
|  | Data Layer     |  | LLM Layer      |  | Agent Layer        |   |
|  | (Phase 1 DONE) |  | (Phase 1 DONE) |  | (Phase 2)          |   |
|  +----------------+  +----------------+  +--------------------+   |
|                                                                    |
|  +----------------+  +----------------+  +--------------------+   |
|  | Risk Engine    |  | Orchestration  |  | API/Dashboard      |   |
|  | (Phase 2)      |  | (Phase 3)      |  | (Phase 1/4)        |   |
|  +----------------+  +----------------+  +--------------------+   |
|                                                                    |
+------------------------------------------------------------------+
```

## Level 2: Component Details

### Data Layer (Phase 1 - COMPLETE)

| Component | Location | Status | Responsibility |
|-----------|----------|--------|----------------|
| Indicator Library | `triplegain/src/data/indicator_library.py` | **Done** | Calculate 17+ technical indicators |
| Market Snapshot Builder | `triplegain/src/data/market_snapshot.py` | **Done** | Aggregate multi-timeframe market state |
| Database Pool | `triplegain/src/data/database.py` | **Done** | Async connection pooling, data access |
| Config Loader | `triplegain/src/utils/config.py` | **Done** | YAML config with validation |

#### Indicators Implemented

| Category | Indicators |
|----------|------------|
| Trend | EMA (9,21,50,200), SMA (20,50,200), ADX, Supertrend |
| Momentum | RSI, MACD, Stochastic RSI, ROC |
| Volatility | ATR, Bollinger Bands, Keltner Channels |
| Volume | OBV, VWAP, Volume SMA, Volume vs Avg |
| Pattern | Choppiness Index, Squeeze Detection |

### LLM Layer (Phase 1 - COMPLETE)

| Component | Location | Status | Responsibility |
|-----------|----------|--------|----------------|
| Prompt Builder | `triplegain/src/llm/prompt_builder.py` | **Done** | Assemble prompts with token management |
| Prompt Templates | `config/prompts/` | **Done** | Agent-specific system prompts |

#### Token Budget Management

| Tier | Total Budget | System Prompt | Market Data | Response Buffer |
|------|--------------|---------------|-------------|-----------------|
| Local (Qwen) | 8,192 | 1,500 | 3,000 | 2,492 |
| API (GPT/Claude) | 128,000 | 3,000 | 6,000 | 116,000 |

### API Layer (Phase 1 - COMPLETE)

| Endpoint | Method | Responsibility |
|----------|--------|----------------|
| `/health` | GET | Full system health check |
| `/health/live` | GET | Kubernetes liveness probe |
| `/health/ready` | GET | Kubernetes readiness probe |
| `/api/v1/indicators/{symbol}/{timeframe}` | GET | Calculate and return indicators |
| `/api/v1/snapshot/{symbol}` | GET | Build and return market snapshot |
| `/api/v1/debug/prompt/{agent}` | GET | Preview assembled prompt |
| `/api/v1/debug/config` | GET | View sanitized configuration |

### Database Layer (Phase 1 - COMPLETE)

| Table | Purpose | Hypertable | Retention |
|-------|---------|------------|-----------|
| `agent_outputs` | Store LLM agent outputs | Yes | 90 days |
| `trading_decisions` | Trade decision audit trail | No | Indefinite |
| `trade_executions` | Executed trade records | No | Indefinite |
| `portfolio_snapshots` | Portfolio history | Yes | Indefinite |
| `risk_state` | Risk tracking state | No | Indefinite |
| `external_data_cache` | External API cache | Yes | 30 days |
| `indicator_cache` | Calculated indicators | Yes | 7 days |

### Agent Layer (Phase 2 - PLANNED)

| Component | Responsibility |
|-----------|----------------|
| Technical Analysis Agent | Analyze indicators, identify signals |
| Regime Detection Agent | Detect market regime (trending/ranging) |
| Risk Management Engine | Rules-based risk validation |
| Trading Decision Agent | Generate trade recommendations |

### Orchestration Layer (Phase 3 - PLANNED)

| Component | Responsibility |
|-----------|----------------|
| Message Bus | Inter-agent communication |
| Coordinator Agent | Resolve conflicts, aggregate signals |
| Portfolio Rebalance Agent | Maintain target allocation |
| Order Execution Manager | Execute approved trades |

### Extended Features (Phase 4 - PLANNED)

| Component | Responsibility |
|-----------|----------------|
| Sentiment Agent | Analyze news/social sentiment |
| Hodl Bag System | Long-term profit allocation |
| A/B Testing Framework | Compare 6 LLM models |
| Dashboard | Real-time monitoring UI |

## Level 3: Key Interfaces

### MarketSnapshot

```python
@dataclass
class MarketSnapshot:
    timestamp: datetime
    symbol: str
    current_price: Decimal
    candles: dict[str, list[CandleSummary]]
    indicators: dict[str, any]
    order_book: Optional[OrderBookFeatures]
    mtf_state: Optional[MultiTimeframeState]

    def to_prompt_format(self, token_budget: int) -> str: ...
    def to_compact_format(self) -> str: ...
```

### PromptBuilder

```python
class PromptBuilder:
    def build_prompt(
        self,
        agent_name: str,
        snapshot: MarketSnapshot,
        portfolio_context: Optional[PortfolioContext],
        additional_context: Optional[dict],
        query: Optional[str]
    ) -> AssembledPrompt: ...
```

### IndicatorLibrary

```python
class IndicatorLibrary:
    def calculate_all(
        self,
        symbol: str,
        timeframe: str,
        candles: list[dict]
    ) -> dict[str, any]: ...
```

## References

- [Phase 1 Implementation](../../development/TripleGain-implementation-plan/01-phase-1-foundation.md)
- [ADR-001: Phase 1 Architecture](../09-decisions/ADR-001-phase1-foundation-architecture.md)
- [Phase 1 Review](../../development/reviews/phase-1/phase-1-comprehensive-review.md)
