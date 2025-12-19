# Phase 2: Core Agents - Feature Documentation

**Version**: 1.2
**Status**: COMPLETE (with Deep Audit Fixes)
**Date**: 2025-12-18
**Updated**: 2025-12-18 - Deep audit fixes applied

## Overview

Phase 2 implements the core agent system for TripleGain, including the base agent framework, specialized trading agents, risk management engine, and multi-provider LLM client infrastructure.

## Components Implemented

### 1. Base Agent Framework

**Location**: `triplegain/src/agents/base_agent.py`

The base agent provides an abstract interface for all LLM-powered agents:

```python
class BaseAgent(ABC):
    agent_name: str
    llm_tier: str  # "tier1_local" | "tier2_api"

    @abstractmethod
    async def process(self, snapshot: MarketSnapshot) -> AgentOutput: ...

    @abstractmethod
    def get_output_schema(self) -> dict: ...
```

**AgentOutput Dataclass**:
- `agent_name`: Identifier for the agent
- `timestamp`: UTC timestamp of output
- `symbol`: Trading pair analyzed
- `confidence`: Output confidence (0-1)
- `reasoning`: Human-readable explanation
- `latency_ms`: Processing time
- `tokens_used`: LLM tokens consumed
- `model_used`: Model identifier

**Features**:
- Automatic output validation
- JSON serialization (`to_dict()`, `to_json()`)
- Database storage integration
- Thread-safe output caching with asyncio locks
- TTL-based cache expiration per symbol
- Configurable cache TTL (`cache_ttl_seconds`)

### 2. Technical Analysis Agent

**Location**: `triplegain/src/agents/technical_analysis.py`

**Purpose**: Analyze technical indicators and generate trading signals.

**Model**: Qwen 2.5 7B (Local via Ollama)
**Frequency**: Per-minute

**TAOutput Fields**:
| Field | Type | Description |
|-------|------|-------------|
| trend_direction | enum | bullish, bearish, neutral |
| trend_strength | float | 0-1 strength score |
| timeframe_alignment | list | Aligned timeframes |
| momentum_score | float | -1 to 1 momentum |
| rsi_signal | enum | overbought, oversold, neutral |
| macd_signal | enum | bullish, bearish, neutral |
| resistance_levels | list | Key resistance prices |
| support_levels | list | Key support prices |
| current_position | enum | near_support, near_resistance, mid_range |
| primary_signal | str | Main trading signal |
| secondary_signals | list | Supporting signals |
| warnings | list | Risk warnings |
| bias | enum | long, short, neutral |

**Key Methods**:
- `_parse_response()`: Extract JSON from LLM response
- `_normalize_parsed_output()`: Clamp values to valid ranges
- `_create_output_from_indicators()`: Fallback when LLM fails
- `_create_fallback_output()`: Safe default output

### 3. Regime Detection Agent

**Location**: `triplegain/src/agents/regime_detection.py`

**Purpose**: Classify market regime to adjust strategy parameters.

**Model**: Qwen 2.5 7B (Local via Ollama)
**Frequency**: Every 5 minutes

**Valid Regimes**:
| Regime | Description | Position Mult | Max Leverage |
|--------|-------------|---------------|--------------|
| trending_bull | Strong uptrend, ADX>25 | 1.0 | 5x |
| trending_bear | Strong downtrend, ADX>25 | 1.0 | 3x |
| ranging | Low ADX, no clear direction | 0.75 | 2x |
| volatile_bull | High ATR, bullish bias | 0.5 | 2x |
| volatile_bear | High ATR, bearish bias | 0.5 | 2x |
| choppy | Erratic, whipsaws | 0.25 | 1x |
| breakout_potential | Consolidation, squeeze | 0.75 | 3x |

**RegimeOutput Fields**:
- `regime`: Current market regime
- `volatility`: low, normal, high, extreme
- `trend_strength`: 0-1 strength score
- `volume_profile`: decreasing, stable, increasing, spike
- `choppiness`: 0-100 (higher = more choppy)
- `adx_value`: ADX indicator reading
- `position_size_multiplier`: Regime-adjusted sizing
- `stop_loss_multiplier`: Regime-adjusted stops
- `entry_strictness`: relaxed, normal, strict, very_strict

### 4. Risk Management Engine

**Location**: `triplegain/src/risk/rules_engine.py`

**Purpose**: Deterministic risk validation with <10ms latency.

**Critical**: NO LLM dependency - purely rule-based.

**Validation Layers**:
1. **Stop-Loss Validation**: Required, min/max distance
2. **Confidence Validation**: Dynamic thresholds after losses + entry strictness
3. **Position Size Validation**: Max % of equity
4. **Volatility Spike Check**: Reduce size 50% during ATR > 3x average
5. **Leverage Validation**: Regime-adjusted limits
6. **Exposure Validation**: Total portfolio exposure
7. **Correlated Position Check**: Max correlated exposure (40% default)
8. **Margin Validation**: Sufficient available margin

**Circuit Breakers**:
| Trigger | Action |
|---------|--------|
| Daily loss > 5% | Halt trading until next day |
| Weekly loss > 10% | Halt + reduce positions 50% |
| Max drawdown > 20% | Halt + close all positions |
| 5 consecutive losses | 1x leverage only |
| Volatility spike (ATR > 3x avg) | Reduce size 50% + 15min cooldown |

**New Features** (v1.1):
- **State Persistence**: Risk state persisted to database, survives restarts
- **Auto Reset**: Daily/weekly resets triggered automatically
- **Correlated Positions**: Tracks BTC/XRP correlation to limit combined exposure
- **Entry Strictness**: Regime-based confidence adjustment (relaxed to very_strict)

**Cooldown Periods**:
- Post-trade: 5 minutes
- Post-loss: 10 minutes
- Consecutive losses (3+): 30 minutes
- Consecutive losses (5+): 60 minutes

**Position Sizing**:
```python
def calculate_position_size(
    equity: float,
    entry_price: float,
    stop_loss: float,
    regime: str,
    confidence: float,
) -> float:
    # ATR-based sizing with regime and confidence multipliers
    # Capped at max_position_pct (default 20%)
```

### 5. Trading Decision Agent

**Location**: `triplegain/src/agents/trading_decision.py`

**Purpose**: Generate trading decisions with 6-model A/B testing.

**Models Queried in Parallel**:
| Name | Provider | Model |
|------|----------|-------|
| qwen | Ollama | qwen2.5:7b |
| gpt4 | OpenAI | gpt-4-turbo |
| grok | xAI | grok-2-1212 |
| deepseek | DeepSeek | deepseek-chat |
| sonnet | Anthropic | claude-3-5-sonnet-20241022 |
| opus | Anthropic | claude-3-opus-20240229 |

**Valid Actions**: BUY, SELL, HOLD, CLOSE_LONG, CLOSE_SHORT

**ConsensusResult**:
- `final_action`: Winning action by vote
- `consensus_strength`: % of models agreeing
- `avg_entry_price`: Average from agreeing models
- `avg_stop_loss`: Average from agreeing models
- `avg_take_profit`: Average from agreeing models
- `total_cost_usd`: Sum of all model costs

**Consensus Algorithm**:
1. Query all 6 models in parallel (with `asyncio.wait` for partial results)
2. Filter valid decisions (no errors, valid action)
3. Count votes per action
4. Select action with most votes
5. Calculate agreement type and confidence boost:
   - Unanimous (6/6): +0.15 confidence boost
   - Strong majority (5/6): +0.10 confidence boost
   - Majority (4/6): +0.05 confidence boost
   - Split (<4/6): No boost
6. Average confidence from agreeing models + apply boost
7. Average trade parameters from agreeing models

**New Features** (v1.1):
- Proper base class initialization for stats/caching
- `asyncio.wait` preserves partial results when models timeout
- Timed-out models are logged individually (not all-or-nothing)

### 6. LLM Client Infrastructure

**Location**: `triplegain/src/llm/clients/`

**Base Interface**:
```python
class BaseLLMClient(ABC):
    provider_name: str

    async def generate(
        model: str,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> LLMResponse: ...

    async def health_check() -> bool: ...
```

**LLMResponse**:
- `text`: Generated response
- `tokens_used`: Total tokens
- `model`: Model identifier
- `latency_ms`: Response time
- `cost_usd`: API cost

**New Features** (v1.1):
- **Rate Limiting**: Per-provider sliding window rate limiter
- **Retry Logic**: Exponential backoff with configurable max retries
- **Cost Calculation**: Automatic cost tracking for all models
- `generate_with_retry()`: Rate-limited generation with automatic retries

**Implemented Clients**:

| Client | Provider | Features |
|--------|----------|----------|
| OllamaClient | Ollama | Local inference, no cost |
| OpenAIClient | OpenAI | GPT models, streaming |
| AnthropicClient | Anthropic | Claude models |
| DeepSeekClient | DeepSeek | DeepSeek V3 |
| XAIClient | xAI | Grok models |

## Configuration

### agents.yaml

```yaml
agents:
  technical_analysis:
    model: qwen2.5:7b
    provider: ollama
    timeout_ms: 5000
    retry_count: 2

  regime_detection:
    model: qwen2.5:7b
    provider: ollama
    timeout_ms: 5000

  trading_decision:
    timeout_seconds: 30
    min_consensus: 0.5
    models:
      qwen: {provider: ollama, model: qwen2.5:7b}
      gpt4: {provider: openai, model: gpt-4-turbo}
      # ... other models
```

### risk.yaml

```yaml
limits:
  max_leverage: 5
  max_position_pct: 20
  max_total_exposure_pct: 80
  max_risk_per_trade_pct: 2
  min_confidence: 0.60

stop_loss:
  required: true
  min_distance_pct: 0.5
  max_distance_pct: 5.0
  min_risk_reward: 1.5

circuit_breakers:
  daily_loss_pct: 5.0
  weekly_loss_pct: 10.0
  max_drawdown_pct: 20.0
  consecutive_losses: 5
```

## Database Migration

**Migration**: `migrations/002_model_comparisons.sql`

**model_comparisons Table**:
- Stores individual model decisions for A/B tracking
- Links to consensus decision
- Tracks cost, latency, tokens per model
- Performance views for model comparison

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/agents/ta/{symbol}` | GET | Get cached TA analysis |
| `/api/v1/agents/ta/{symbol}/run` | POST | Trigger fresh TA analysis |
| `/api/v1/agents/regime/{symbol}` | GET | Get cached regime |
| `/api/v1/agents/regime/{symbol}/run` | POST | Trigger fresh regime detection |
| `/api/v1/agents/trading/{symbol}/run` | POST | Trigger trading decision |
| `/api/v1/risk/validate` | POST | Validate trade proposal |
| `/api/v1/risk/state` | GET | Get current risk state |
| `/api/v1/risk/reset` | POST | Reset risk state (admin) |
| `/api/v1/agents/stats` | GET | Get all agent statistics |

## Testing

**Test Files**:
- `tests/unit/agents/test_base_agent.py` - 7 tests
- `tests/unit/agents/test_technical_analysis.py` - 23 tests
- `tests/unit/agents/test_regime_detection.py` - 36 tests
- `tests/unit/agents/test_trading_decision.py` - 29 tests
- `tests/unit/risk/test_rules_engine.py` - 29 tests
- `tests/unit/llm/test_clients.py` - 19 tests

**Total Phase 2 Tests**: 136 (all passing)

**Coverage**:
- Risk Engine: 83%
- Agents: 58-71%
- LLM Clients: 15-94% (network code excluded)

## Integration Points

### With Phase 1

- **MarketSnapshot**: Input to all agents
- **PromptBuilder**: Constructs agent prompts
- **IndicatorLibrary**: Provides data for analysis
- **Database**: Stores agent outputs

### With Phase 3 (Future)

- **Coordinator Agent**: Consumes agent outputs
- **Order Execution**: Receives validated trade proposals
- **Message Bus**: Inter-agent communication

## Design Decisions

1. **Rules-based Risk Engine**: No LLM dependency for deterministic, auditable risk validation.

2. **6-Model A/B Testing**: Compare model performance in production to identify best performers.

3. **Consensus Voting**: Majority voting with averaged parameters reduces individual model errors.

4. **Regime-Adjusted Parameters**: All trading parameters adapt to market conditions.

5. **Fallback Outputs**: When LLM fails, agents produce safe conservative outputs using indicators.

## Code Review Fixes (v1.1)

All issues identified in the Phase 2 code review have been addressed:

### High Priority (Completed)
| Issue | Fix |
|-------|-----|
| Volatility spike circuit breaker missing | Added `update_volatility()` with 3x ATR detection |
| Risk state not persisted | Added `to_dict()`, `from_dict()`, `persist_state()`, `load_state()` |
| Rate limiting for LLM APIs | Added `RateLimiter` class with sliding window |
| Output caching thread safety | Added `asyncio.Lock` for thread-safe cache access |

### Medium Priority (Completed)
| Issue | Fix |
|-------|-----|
| Entry strictness not consumed | Added `entry_strictness` parameter to `validate_trade()` |
| Correlated position check | Added `_validate_correlation()` with pair correlation matrix |
| Confidence boost not applied | Applied boost in `TradingDecisionOutput` creation |
| Daily/weekly reset not automated | Added `_check_and_reset_periods()` |

### Low Priority (Completed)
| Issue | Fix |
|-------|-----|
| TradingDecisionAgent bypasses base class | Fixed to call `super().__init__()` |
| Model timeout handling | Changed to `asyncio.wait` to preserve partial results |
| Missing POST trigger endpoints | Added `/run` POST endpoints for TA and Regime |
| Confidence thresholds from config | Now loaded from `risk.yaml` confidence section |

**Migration**: `migrations/003_risk_state_and_indexes.sql` adds:
- `risk_state` table for persistence
- `risk_state_history` table for audit
- Additional indexes for query optimization

## Deep Audit Fixes (v1.2)

A comprehensive deep audit identified additional issues:

### Model Outcome Tracking (Trading Decision)
**Issue**: `model_comparisons` table stored decisions but not trade outcomes.
**Fix**:
- `_store_model_comparisons()` now includes `price_at_decision`
- Added `update_comparison_outcomes()` method for A/B analysis
- Orchestrator can call after 1h/4h/24h to populate `was_correct`, `outcome_pnl_pct`

### `_last_output` Attribute (Base Agent)
**Issue**: Attribute used but never declared in `__init__`.
**Fix**:
- Added `self._last_output: Optional[AgentOutput] = None` in `__init__`
- Added `@property last_output` getter for clean access

### Drawdown Edge Cases (Risk Engine)
**Issue**: `update_drawdown()` failed with zero or negative equity.
**Fix**: Enhanced with edge case handling:
- Zero/zero equity → 0% drawdown
- Negative equity → >100% drawdown
- New peak detection → Reset drawdown to 0%

**Tests Added**: 10 new unit tests (378 total, was 368)

## References

- [Phase 2 Implementation Plan](../TripleGain-implementation-plan/02-phase-2-core-agents.md)
- [Risk Management Design](../TripleGain-master-design/03-risk-management-rules-engine.md)
- [Multi-Agent Architecture](../TripleGain-master-design/01-multi-agent-architecture.md)
- [Phase 2 Code Review](../reviews/phase-2/phase-2-code-logic-review.md)
- [Phase 2 Deep Audit Review](../reviews/phase-2/phase-2-deep-audit-review.md)
- [ADR-003: Phase 2 Code Review Fixes](../../architecture/09-decisions/ADR-003-phase2-code-review-fixes.md)

---

*Phase 2 Feature Documentation v1.2 - December 2025*
