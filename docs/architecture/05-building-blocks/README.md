# 05 - Building Block View

## Level 1: System Overview

```
+------------------------------------------------------------------+
|                    TripleGain Trading System                      |
+------------------------------------------------------------------+
|                                                                    |
|  +----------------+  +----------------+  +--------------------+   |
|  | Data Layer     |  | LLM Layer      |  | Agent Layer        |   |
|  | (Phase 1 DONE) |  | (Phase 1 DONE) |  | (Phase 2 DONE)     |   |
|  +----------------+  +----------------+  +--------------------+   |
|                                                                    |
|  +----------------+  +----------------+  +--------------------+   |
|  | Risk Engine    |  | Orchestration  |  | API/Dashboard      |   |
|  | (Phase 2 DONE) |  | (Phase 3)      |  | (Phase 1/4)        |   |
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

### Agent Layer (Phase 2 - COMPLETE)

| Component | Location | Status | Responsibility |
|-----------|----------|--------|----------------|
| Base Agent | `triplegain/src/agents/base_agent.py` | **Done** | Abstract interface, AgentOutput dataclass |
| Technical Analysis Agent | `triplegain/src/agents/technical_analysis.py` | **Done** | Analyze indicators, identify signals |
| Regime Detection Agent | `triplegain/src/agents/regime_detection.py` | **Done** | Classify market regime (7 types) |
| Trading Decision Agent | `triplegain/src/agents/trading_decision.py` | **Done** | 6-model consensus decisions |

#### Agent Outputs

| Agent | Output Class | Key Fields |
|-------|--------------|------------|
| Technical Analysis | TAOutput | trend_direction, momentum_score, bias |
| Regime Detection | RegimeOutput | regime, position_size_multiplier, entry_strictness |
| Trading Decision | TradingDecisionOutput | action, consensus_strength, entry_price, stop_loss |

#### Agent Features (v1.2)

| Feature | Description |
|---------|-------------|
| `last_output` Property | Access most recent output without cache lookup |
| Model Outcome Tracking | `update_comparison_outcomes()` for A/B analysis |
| Thread-Safe Caching | `asyncio.Lock` protected cache with TTL |

### Risk Layer (Phase 2 - COMPLETE)

| Component | Location | Status | Responsibility |
|-----------|----------|--------|----------------|
| Rules Engine | `triplegain/src/risk/rules_engine.py` | **Done** | Trade validation, circuit breakers, cooldowns |

#### Risk Validation Layers

| Layer | Check |
|-------|-------|
| Stop-Loss | Required, 0.5-5% distance |
| Confidence | Dynamic threshold (0.6-0.8 based on losses) + entry strictness |
| Position Size | Max 20% of equity |
| Volatility Spike | ATR > 3x average reduces size 50% |
| Leverage | Regime-adjusted (1-5x) |
| Exposure | Max 80% total |
| Correlation | Max 40% correlated exposure (BTC/XRP) |
| Margin | Sufficient available |

#### Risk Engine Features (v1.2)

| Feature | Description |
|---------|-------------|
| State Persistence | Risk state survives restarts (database-backed) |
| Auto Reset | Daily/weekly resets trigger automatically |
| Volatility Detection | ATR spike detection with position reduction |
| Correlation Matrix | BTC/XRP correlation-aware exposure limits |
| Entry Strictness | Regime-based confidence adjustment |
| Drawdown Edge Cases | Handles zero/negative equity scenarios |

### LLM Clients (Phase 2 - COMPLETE)

| Client | Location | Status | Provider |
|--------|----------|--------|----------|
| BaseLLMClient | `triplegain/src/llm/clients/base.py` | **Done** | Abstract interface |
| OllamaClient | `triplegain/src/llm/clients/ollama.py` | **Done** | Local Ollama |
| OpenAIClient | `triplegain/src/llm/clients/openai_client.py` | **Done** | OpenAI GPT |
| AnthropicClient | `triplegain/src/llm/clients/anthropic_client.py` | **Done** | Claude models |
| DeepSeekClient | `triplegain/src/llm/clients/deepseek_client.py` | **Done** | DeepSeek V3 |
| XAIClient | `triplegain/src/llm/clients/xai_client.py` | **Done** | Grok models |

#### LLM Client Features (v1.1)

| Feature | Description |
|---------|-------------|
| Rate Limiter | Sliding window algorithm per provider |
| Retry Logic | Exponential backoff with configurable max retries |
| Cost Tracking | Automatic cost calculation for all models |
| `generate_with_retry()` | Rate-limited generation with automatic retries |

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

### BaseAgent (Phase 2)

```python
class BaseAgent(ABC):
    agent_name: str
    llm_tier: str  # "tier1_local" | "tier2_api"

    @abstractmethod
    async def process(self, snapshot: MarketSnapshot) -> AgentOutput: ...

    @abstractmethod
    def get_output_schema(self) -> dict: ...
```

### RiskManagementEngine (Phase 2)

```python
class RiskManagementEngine:
    def validate_trade(
        self,
        proposal: TradeProposal,
        risk_state: RiskState
    ) -> RiskValidation: ...

    def calculate_position_size(
        self,
        equity: float,
        entry_price: float,
        stop_loss: float,
        regime: str,
        confidence: float
    ) -> float: ...
```

### BaseLLMClient (Phase 2)

```python
class BaseLLMClient(ABC):
    provider_name: str

    @abstractmethod
    async def generate(
        self,
        model: str,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.3,
        max_tokens: int = 2048
    ) -> LLMResponse: ...

    @abstractmethod
    async def health_check(self) -> bool: ...
```

## References

- [Phase 1 Implementation](../../development/TripleGain-implementation-plan/01-phase-1-foundation.md)
- [Phase 2 Implementation](../../development/TripleGain-implementation-plan/02-phase-2-core-agents.md)
- [ADR-001: Phase 1 Architecture](../09-decisions/ADR-001-phase1-foundation-architecture.md)
- [ADR-002: Phase 2 Architecture](../09-decisions/ADR-002-phase2-core-agents-architecture.md)
- [ADR-003: Phase 2 Code Review Fixes](../09-decisions/ADR-003-phase2-code-review-fixes.md)
- [Phase 1 Review](../../development/reviews/phase-1/phase-1-comprehensive-review.md)
- [Phase 2 Review](../../development/reviews/phase-2/phase-2-code-logic-review.md)
- [Phase 2 Deep Audit](../../development/reviews/phase-2/phase-2-deep-audit-review.md)
- [Phase 2 Feature Documentation](../../development/features/phase-2-core-agents.md)
