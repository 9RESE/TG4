# Phase 2: Core Agents

**Phase Status**: Pending Phase 1 Completion
**Dependencies**: Phase 1 (Data Pipeline, Indicators, Snapshots, Prompts)
**Deliverable**: Individual agents producing validated signals

---

## Overview

Phase 2 implements the four core agents that form the trading decision pipeline:
1. Technical Analysis Agent (Qwen 2.5 7B - Local)
2. Regime Detection Agent (Qwen 2.5 7B - Local)
3. Risk Management Engine (Rule-based, no LLM)
4. Trading Decision Agent (6-Model A/B Testing)

### Agent Dependencies

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       PHASE 2 AGENT FLOW                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  MarketSnapshot ──────┬────────────────────────────────────────────────────►│
│                       │                                                     │
│                       ▼                                                     │
│              ┌───────────────────┐                                          │
│              │ Technical Analysis │ ◄─── Qwen 2.5 7B (Local)               │
│              │       Agent        │      Invoked: Per minute               │
│              └─────────┬─────────┘                                          │
│                        │                                                     │
│                        ▼                                                     │
│              ┌───────────────────┐                                          │
│              │  Regime Detection  │ ◄─── Qwen 2.5 7B (Local)               │
│              │       Agent        │      Invoked: Every 5 minutes          │
│              └─────────┬─────────┘                                          │
│                        │                                                     │
│                        ├─────────────────────────────────────┐              │
│                        ▼                                     ▼              │
│              ┌───────────────────┐              ┌───────────────────┐       │
│              │     Trading        │              │  Risk Management  │       │
│              │  Decision Agent    │◄────────────│      Engine       │       │
│              │                    │  Validates   │  (Rule-based)     │       │
│              │  6-Model A/B:      │              └───────────────────┘       │
│              │  • GPT             │                                          │
│              │  • Grok            │                                          │
│              │  • DeepSeek V3     │                                          │
│              │  • Claude Sonnet   │                                          │
│              │  • Claude Opus     │                                          │
│              │  • Qwen 2.5 7B     │                                          │
│              │                    │                                          │
│              │  Invoked: Hourly   │                                          │
│              └─────────┬─────────┘                                          │
│                        │                                                     │
│                        ▼                                                     │
│              ┌───────────────────┐                                          │
│              │  Risk Validation   │                                          │
│              │  (Before output)   │                                          │
│              └─────────┬─────────┘                                          │
│                        │                                                     │
│                        ▼                                                     │
│                Trading Decision                                              │
│                (Approved/Modified/Rejected)                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2.1 Technical Analysis Agent

### Purpose

Analyzes market data using pre-computed technical indicators to identify trading signals and market conditions.

### LLM Assignment

| Property | Value |
|----------|-------|
| Model | Qwen 2.5 7B |
| Provider | Ollama (Local) |
| Invocation | Every minute on candle close |
| Latency Target | < 300ms |
| Tier | Tier 1 (Local) |

### Input/Output Contract

**Input**: MarketSnapshot (compact format)

**Output Schema**:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["timestamp", "symbol", "trend", "momentum", "bias", "confidence"],
  "properties": {
    "timestamp": {
      "type": "string",
      "format": "date-time"
    },
    "symbol": {
      "type": "string"
    },
    "trend": {
      "type": "object",
      "required": ["direction", "strength"],
      "properties": {
        "direction": {
          "type": "string",
          "enum": ["bullish", "bearish", "neutral"]
        },
        "strength": {
          "type": "number",
          "minimum": 0,
          "maximum": 1
        },
        "timeframe_alignment": {
          "type": "array",
          "items": {"type": "string"}
        }
      }
    },
    "momentum": {
      "type": "object",
      "required": ["score"],
      "properties": {
        "score": {
          "type": "number",
          "minimum": -1,
          "maximum": 1
        },
        "rsi_signal": {
          "type": "string",
          "enum": ["oversold", "neutral", "overbought"]
        },
        "macd_signal": {
          "type": "string",
          "enum": ["bullish_cross", "bearish_cross", "bullish", "bearish", "neutral"]
        }
      }
    },
    "key_levels": {
      "type": "object",
      "properties": {
        "resistance": {"type": "array", "items": {"type": "number"}},
        "support": {"type": "array", "items": {"type": "number"}},
        "current_position": {
          "type": "string",
          "enum": ["near_support", "mid_range", "near_resistance"]
        }
      }
    },
    "signals": {
      "type": "object",
      "properties": {
        "primary": {"type": "string"},
        "secondary": {"type": "array", "items": {"type": "string"}},
        "warnings": {"type": "array", "items": {"type": "string"}}
      }
    },
    "bias": {
      "type": "string",
      "enum": ["long", "short", "neutral"]
    },
    "confidence": {
      "type": "number",
      "minimum": 0,
      "maximum": 1
    },
    "reasoning": {
      "type": "string",
      "maxLength": 500
    }
  }
}
```

### Agent Interface

```python
# src/agents/technical_analysis.py

from dataclasses import dataclass
from typing import Optional
from src.agents.base_agent import BaseAgent, AgentOutput
from src.data.market_snapshot import MarketSnapshot

@dataclass
class TAOutput(AgentOutput):
    """Technical Analysis Agent output."""
    trend_direction: str
    trend_strength: float
    momentum_score: float
    rsi_signal: str
    macd_signal: str
    resistance_levels: list[float]
    support_levels: list[float]
    primary_signal: str
    warnings: list[str]
    bias: str
    confidence: float
    reasoning: str


class TechnicalAnalysisAgent(BaseAgent):
    """
    Technical Analysis Agent using local Qwen 2.5 7B.

    Analyzes indicators and price action to identify trading signals.
    Does NOT make trading decisions - only provides analysis.
    """

    agent_name = "technical_analysis"
    llm_tier = "tier1_local"
    model = "qwen2.5:7b"

    def __init__(
        self,
        llm_client,
        prompt_builder,
        output_parser,
        config: dict
    ):
        self.llm = llm_client
        self.prompt_builder = prompt_builder
        self.parser = output_parser
        self.config = config

    async def process(
        self,
        snapshot: MarketSnapshot,
        portfolio_context: Optional[dict] = None
    ) -> TAOutput:
        """
        Analyze market data and produce TA output.

        Args:
            snapshot: Market data snapshot
            portfolio_context: Optional portfolio state

        Returns:
            TAOutput with analysis results
        """
        # Build prompt
        prompt = self.prompt_builder.build_prompt(
            agent_name=self.agent_name,
            snapshot=snapshot,
            portfolio_context=portfolio_context
        )

        # Call LLM
        start_time = time.time()
        response = await self.llm.generate(
            model=self.model,
            system_prompt=prompt.system_prompt,
            user_message=prompt.user_message
        )
        latency_ms = int((time.time() - start_time) * 1000)

        # Parse output
        parsed = self.parser.parse_ta_output(response.text)

        # Store output
        await self._store_output(parsed, latency_ms, response.tokens_used)

        return parsed

    def get_output_schema(self) -> dict:
        """Return JSON schema for validation."""
        return TA_OUTPUT_SCHEMA

    async def _store_output(
        self,
        output: TAOutput,
        latency_ms: int,
        tokens: int
    ) -> None:
        """Store output to agent_outputs table."""
        pass
```

### Configuration

```yaml
# config/agents.yaml (technical_analysis section)

agents:
  technical_analysis:
    enabled: true
    model: qwen2.5:7b
    provider: ollama

    invocation:
      trigger: candle_close
      interval_seconds: 60
      symbols:
        - BTC/USDT
        - XRP/USDT
        - XRP/BTC

    prompt:
      template: technical_analysis.txt
      token_budget: 3000
      include_timeframes:
        - 1m
        - 5m
        - 1h
        - 4h
      max_candles_per_tf: 5

    output:
      store_all: true
      cache_ttl_seconds: 55  # Slightly less than interval

    performance:
      timeout_ms: 5000
      retry_count: 2
      retry_delay_ms: 500
```

### Test Requirements

| Test | Description | Acceptance |
|------|-------------|------------|
| Output schema | Output matches JSON schema | 100% valid |
| Confidence bounds | confidence in [0, 1] | No violations |
| Latency | End-to-end < 500ms | 95th percentile |
| Error handling | Handle LLM timeout | Graceful fallback |
| Storage | All outputs stored | Records in DB |

---

## 2.2 Regime Detection Agent

### Purpose

Classifies current market regime (trending bull, trending bear, ranging, volatile, etc.) to guide strategy selection and risk parameters.

### LLM Assignment

| Property | Value |
|----------|-------|
| Model | Qwen 2.5 7B |
| Provider | Ollama (Local) |
| Invocation | Every 5 minutes |
| Latency Target | < 500ms |
| Tier | Tier 1 (Local) |

### Regime Classifications

| Regime | Description | Trading Implication |
|--------|-------------|---------------------|
| `trending_bull` | Strong upward trend | Favor long positions |
| `trending_bear` | Strong downward trend | Favor short positions |
| `ranging` | Price in range, no clear trend | Mean reversion strategies |
| `volatile_bull` | High volatility, bullish bias | Reduced position size, long bias |
| `volatile_bear` | High volatility, bearish bias | Reduced position size, short bias |
| `choppy` | Erratic, no clear direction | Avoid trading or reduce size |
| `breakout_potential` | Consolidation before breakout | Prepare for directional move |

### Output Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["timestamp", "symbol", "regime", "confidence", "characteristics"],
  "properties": {
    "timestamp": {"type": "string", "format": "date-time"},
    "symbol": {"type": "string"},
    "regime": {
      "type": "string",
      "enum": [
        "trending_bull",
        "trending_bear",
        "ranging",
        "volatile_bull",
        "volatile_bear",
        "choppy",
        "breakout_potential"
      ]
    },
    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    "characteristics": {
      "type": "object",
      "required": ["volatility", "trend_strength", "volume_profile"],
      "properties": {
        "volatility": {
          "type": "string",
          "enum": ["low", "normal", "high", "extreme"]
        },
        "trend_strength": {"type": "number", "minimum": 0, "maximum": 1},
        "volume_profile": {
          "type": "string",
          "enum": ["decreasing", "stable", "increasing", "spike"]
        },
        "choppiness": {"type": "number", "minimum": 0, "maximum": 100},
        "adx_value": {"type": "number"}
      }
    },
    "regime_duration": {
      "type": "object",
      "properties": {
        "current_regime_started": {"type": "string", "format": "date-time"},
        "periods_in_regime": {"type": "integer"}
      }
    },
    "transition_probability": {
      "type": "object",
      "description": "Probability of transitioning to each regime",
      "additionalProperties": {"type": "number"}
    },
    "recommended_adjustments": {
      "type": "object",
      "properties": {
        "position_size_multiplier": {
          "type": "number",
          "minimum": 0.25,
          "maximum": 1.5
        },
        "stop_loss_multiplier": {
          "type": "number",
          "minimum": 0.5,
          "maximum": 2.0
        },
        "take_profit_multiplier": {
          "type": "number",
          "minimum": 0.5,
          "maximum": 3.0
        },
        "entry_strictness": {
          "type": "string",
          "enum": ["relaxed", "normal", "strict", "very_strict"]
        }
      }
    },
    "reasoning": {"type": "string", "maxLength": 300}
  }
}
```

### Agent Interface

```python
# src/agents/regime_detection.py

@dataclass
class RegimeOutput(AgentOutput):
    """Regime Detection Agent output."""
    regime: str
    confidence: float
    volatility: str
    trend_strength: float
    volume_profile: str
    choppiness: float
    position_size_multiplier: float
    stop_loss_multiplier: float
    take_profit_multiplier: float
    entry_strictness: str
    reasoning: str


class RegimeDetectionAgent(BaseAgent):
    """
    Regime Detection Agent using local Qwen 2.5 7B.

    Classifies market regime to guide strategy selection.
    """

    agent_name = "regime_detection"
    llm_tier = "tier1_local"
    model = "qwen2.5:7b"

    async def process(
        self,
        snapshot: MarketSnapshot,
        ta_output: Optional[TAOutput] = None
    ) -> RegimeOutput:
        """
        Detect market regime.

        Args:
            snapshot: Market data snapshot
            ta_output: Optional TA agent output for context

        Returns:
            RegimeOutput with regime classification
        """
        pass

    def get_regime_parameters(self, regime: str) -> dict:
        """Get default trading parameters for a regime."""
        return REGIME_PARAMETERS.get(regime, DEFAULT_PARAMETERS)
```

### Configuration

```yaml
# config/agents.yaml (regime_detection section)

agents:
  regime_detection:
    enabled: true
    model: qwen2.5:7b
    provider: ollama

    invocation:
      trigger: scheduled
      interval_seconds: 300  # 5 minutes
      symbols:
        - BTC/USDT
        - XRP/USDT

    regime_parameters:
      trending_bull:
        position_size_multiplier: 1.0
        stop_loss_multiplier: 1.2
        take_profit_multiplier: 2.0
        entry_strictness: normal

      trending_bear:
        position_size_multiplier: 1.0
        stop_loss_multiplier: 1.2
        take_profit_multiplier: 2.0
        entry_strictness: normal

      ranging:
        position_size_multiplier: 0.75
        stop_loss_multiplier: 0.8
        take_profit_multiplier: 1.5
        entry_strictness: strict

      volatile_bull:
        position_size_multiplier: 0.5
        stop_loss_multiplier: 1.5
        take_profit_multiplier: 2.5
        entry_strictness: strict

      volatile_bear:
        position_size_multiplier: 0.5
        stop_loss_multiplier: 1.5
        take_profit_multiplier: 2.5
        entry_strictness: strict

      choppy:
        position_size_multiplier: 0.25
        stop_loss_multiplier: 1.0
        take_profit_multiplier: 1.0
        entry_strictness: very_strict

      breakout_potential:
        position_size_multiplier: 0.75
        stop_loss_multiplier: 1.0
        take_profit_multiplier: 3.0
        entry_strictness: strict

    output:
      store_all: true
      cache_ttl_seconds: 290
```

### Test Requirements

| Test | Description | Acceptance |
|------|-------------|------------|
| Regime enum | Output in valid regime set | 100% valid |
| Parameter bounds | Multipliers in valid range | No violations |
| Consistency | Same input = same output | Stable classification |
| Transition | Regime changes detected | Within 2 periods |

---

## 2.3 Risk Management Engine

### Purpose

Rule-based engine (no LLM) that validates all trading decisions against risk parameters. Acts as a gatekeeper before order execution.

### NOT an LLM Agent

The Risk Management Engine is **purely rule-based** - it does not use any LLM. This ensures:
- Deterministic behavior
- Sub-10ms validation
- No API dependencies
- 100% auditability

### Risk Rules (From Design 03-risk-management-rules-engine.md)

#### Position Sizing Rules

| Rule | Formula | Default |
|------|---------|---------|
| Risk per trade | `position_value * stop_loss_pct` | 1% of equity |
| Max position size | `equity * 0.2` | 20% of equity |
| Max leverage | `min(regime_allowed, user_max)` | 5x |

#### Circuit Breakers

| Breaker | Trigger | Action | Cooldown |
|---------|---------|--------|----------|
| Daily Loss | -5% daily P&L | Halt trading | Until next day |
| Weekly Loss | -10% weekly P&L | Halt trading | Until next week |
| Max Drawdown | -20% from peak | Halt trading | Manual reset |
| Consecutive Losses | 5 losses in a row | Reduce size 50% | 3 hours |
| Volatility Spike | ATR > 3x average | Reduce size 50% | Until normalized |

### Interface Definition

```python
# src/risk/rules_engine.py

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple
from decimal import Decimal

class ValidationStatus(Enum):
    APPROVED = "approved"
    MODIFIED = "modified"
    REJECTED = "rejected"


@dataclass
class TradeProposal:
    """Proposed trade from Trading Decision Agent."""
    symbol: str
    action: str  # BUY, SELL, CLOSE
    side: str  # long, short
    size: Decimal  # Position size
    entry_price: Decimal
    stop_loss: Decimal
    take_profit: Decimal
    leverage: int
    confidence: Decimal
    reasoning: str


@dataclass
class RiskValidation:
    """Result of risk validation."""
    status: ValidationStatus
    original_proposal: TradeProposal
    modified_proposal: Optional[TradeProposal]
    checks_passed: list[str]
    checks_failed: list[str]
    modifications: list[str]  # What was modified
    rejection_reasons: list[str]
    risk_metrics: dict


@dataclass
class RiskState:
    """Current risk state."""
    daily_pnl_pct: Decimal
    weekly_pnl_pct: Decimal
    current_drawdown_pct: Decimal
    peak_equity: Decimal
    consecutive_losses: int
    open_positions_count: int
    total_exposure_pct: Decimal
    circuit_breakers_active: list[str]
    cooldowns_active: dict[str, datetime]


class RiskManagementEngine:
    """
    Rule-based risk management engine.

    Validates trade proposals against configurable rules.
    NO LLM - purely deterministic.
    """

    def __init__(self, config: dict, db_pool):
        self.config = config
        self.db = db_pool
        self._state: Optional[RiskState] = None

    async def validate_trade(
        self,
        proposal: TradeProposal,
        portfolio_context: dict,
        regime: str
    ) -> RiskValidation:
        """
        Validate a trade proposal.

        Args:
            proposal: Proposed trade
            portfolio_context: Current portfolio state
            regime: Current market regime

        Returns:
            RiskValidation with approval/modification/rejection
        """
        checks_passed = []
        checks_failed = []
        modifications = []
        rejection_reasons = []

        # Load current risk state
        state = await self._load_risk_state()

        # Check circuit breakers
        if state.circuit_breakers_active:
            return RiskValidation(
                status=ValidationStatus.REJECTED,
                original_proposal=proposal,
                modified_proposal=None,
                checks_passed=[],
                checks_failed=["circuit_breaker_active"],
                modifications=[],
                rejection_reasons=[f"Circuit breaker active: {state.circuit_breakers_active}"],
                risk_metrics={}
            )

        # Check confidence threshold
        min_confidence = self._get_min_confidence(regime)
        if proposal.confidence < min_confidence:
            rejection_reasons.append(
                f"Confidence {proposal.confidence} below threshold {min_confidence}"
            )
            checks_failed.append("confidence_threshold")
        else:
            checks_passed.append("confidence_threshold")

        # Check position size
        max_size = self._calculate_max_position_size(
            portfolio_context,
            regime
        )
        modified_size = proposal.size
        if proposal.size > max_size:
            modified_size = max_size
            modifications.append(f"Reduced size from {proposal.size} to {max_size}")
            checks_failed.append("position_size_limit")
        else:
            checks_passed.append("position_size_limit")

        # Check leverage
        max_leverage = self._get_max_leverage(regime)
        modified_leverage = proposal.leverage
        if proposal.leverage > max_leverage:
            modified_leverage = max_leverage
            modifications.append(f"Reduced leverage from {proposal.leverage}x to {max_leverage}x")
            checks_failed.append("leverage_limit")
        else:
            checks_passed.append("leverage_limit")

        # Check R:R ratio
        rr_ratio = self._calculate_rr_ratio(
            proposal.entry_price,
            proposal.stop_loss,
            proposal.take_profit
        )
        if rr_ratio < self.config["min_rr_ratio"]:
            checks_failed.append("risk_reward_ratio")
            rejection_reasons.append(f"R:R ratio {rr_ratio:.2f} below minimum {self.config['min_rr_ratio']}")
        else:
            checks_passed.append("risk_reward_ratio")

        # Check exposure limit
        new_exposure = self._calculate_new_exposure(
            portfolio_context,
            modified_size,
            proposal.entry_price
        )
        if new_exposure > self.config["max_total_exposure_pct"]:
            checks_failed.append("exposure_limit")
            rejection_reasons.append(f"Exposure {new_exposure:.1f}% exceeds limit")
        else:
            checks_passed.append("exposure_limit")

        # Check correlated positions
        if await self._has_correlated_position(proposal.symbol):
            checks_failed.append("correlation_limit")
            modifications.append("Reduced size due to correlated position")
            modified_size = modified_size * Decimal("0.5")

        # Determine final status
        if rejection_reasons:
            status = ValidationStatus.REJECTED
            modified_proposal = None
        elif modifications:
            status = ValidationStatus.MODIFIED
            modified_proposal = TradeProposal(
                symbol=proposal.symbol,
                action=proposal.action,
                side=proposal.side,
                size=modified_size,
                entry_price=proposal.entry_price,
                stop_loss=proposal.stop_loss,
                take_profit=proposal.take_profit,
                leverage=modified_leverage,
                confidence=proposal.confidence,
                reasoning=proposal.reasoning
            )
        else:
            status = ValidationStatus.APPROVED
            modified_proposal = proposal

        return RiskValidation(
            status=status,
            original_proposal=proposal,
            modified_proposal=modified_proposal,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            modifications=modifications,
            rejection_reasons=rejection_reasons,
            risk_metrics=self._calculate_risk_metrics(proposal, portfolio_context)
        )

    async def check_circuit_breakers(
        self,
        portfolio_context: dict
    ) -> list[str]:
        """Check all circuit breakers, return list of triggered ones."""
        pass

    async def update_risk_state(
        self,
        trade_result: dict
    ) -> None:
        """Update risk state after a trade closes."""
        pass

    def _get_min_confidence(self, regime: str) -> Decimal:
        """Get minimum confidence threshold for regime."""
        regime_configs = self.config.get("regime_thresholds", {})
        return Decimal(str(regime_configs.get(regime, {}).get("min_confidence", 0.55)))

    def _calculate_max_position_size(
        self,
        portfolio: dict,
        regime: str
    ) -> Decimal:
        """Calculate maximum allowed position size."""
        pass

    def _get_max_leverage(self, regime: str) -> int:
        """Get maximum leverage for regime."""
        pass

    def _calculate_rr_ratio(
        self,
        entry: Decimal,
        stop: Decimal,
        target: Decimal
    ) -> Decimal:
        """Calculate risk-reward ratio."""
        risk = abs(entry - stop)
        reward = abs(target - entry)
        return reward / risk if risk > 0 else Decimal("0")
```

### Configuration

```yaml
# config/risk.yaml

risk_management:
  # Position sizing
  position_sizing:
    risk_per_trade_pct: 1.0
    max_position_pct: 20.0
    max_total_exposure_pct: 60.0

  # Leverage limits by regime
  leverage_limits:
    trending_bull: 5
    trending_bear: 5
    ranging: 3
    volatile_bull: 2
    volatile_bear: 2
    choppy: 1
    breakout_potential: 3

  # Confidence thresholds by regime
  confidence_thresholds:
    trending_bull: 0.55
    trending_bear: 0.55
    ranging: 0.60
    volatile_bull: 0.65
    volatile_bear: 0.65
    choppy: 0.75
    breakout_potential: 0.60

  # Risk-reward requirements
  min_rr_ratio: 1.5

  # Circuit breakers
  circuit_breakers:
    daily_loss_limit_pct: 5.0
    weekly_loss_limit_pct: 10.0
    max_drawdown_pct: 20.0
    consecutive_loss_limit: 5
    volatility_spike_threshold: 3.0  # ATR multiplier

  # Cooldowns
  cooldowns:
    after_consecutive_losses_hours: 3
    after_circuit_breaker_hours: 24
    reduced_size_period_hours: 6

  # Correlation
  correlation:
    check_enabled: true
    high_correlation_threshold: 0.7
    reduction_factor: 0.5
```

### Test Requirements

| Test | Description | Acceptance |
|------|-------------|------------|
| Approval flow | Valid trade approved | No modifications |
| Modification flow | Over-limit trade modified | Size/leverage reduced |
| Rejection flow | Invalid trade rejected | Clear reasons |
| Circuit breaker | Trigger and reset | Correct timing |
| Performance | Validation < 10ms | 99th percentile |
| Determinism | Same input = same output | 100% reproducible |

---

## 2.4 Trading Decision Agent

### Purpose

Makes final trading decisions by synthesizing TA output, regime, and sentiment (when available). Uses 6-model parallel comparison for A/B testing.

### 6-Model A/B Testing Configuration

| Model | Provider | Tier | Primary Use |
|-------|----------|------|-------------|
| GPT (latest) | OpenAI | API | A/B comparison |
| Grok (latest) | xAI | API | A/B comparison + sentiment |
| DeepSeek V3 | DeepSeek | API | A/B comparison + primary |
| Claude Sonnet | Anthropic | API | A/B comparison |
| Claude Opus | Anthropic | API | A/B comparison + complex |
| Qwen 2.5 7B | Ollama | Local | A/B comparison + baseline |

### Invocation Strategy

All 6 models run **in parallel** on every trading decision query (hourly). Results are:
1. Collected and compared
2. Consensus calculated
3. Best decision selected based on consensus and confidence
4. All results logged for performance tracking

### Output Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["timestamp", "symbol", "action", "confidence", "parameters", "reasoning"],
  "properties": {
    "timestamp": {"type": "string", "format": "date-time"},
    "symbol": {"type": "string"},
    "action": {
      "type": "string",
      "enum": ["BUY", "SELL", "HOLD", "CLOSE"]
    },
    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    "parameters": {
      "type": "object",
      "properties": {
        "entry_type": {
          "type": "string",
          "enum": ["market", "limit"]
        },
        "entry_price": {"type": "number"},
        "size_pct": {
          "type": "number",
          "minimum": 0,
          "maximum": 100,
          "description": "Percentage of available margin"
        },
        "leverage": {
          "type": "integer",
          "minimum": 1,
          "maximum": 5
        },
        "stop_loss_pct": {
          "type": "number",
          "description": "Distance from entry as percentage"
        },
        "take_profit_pct": {
          "type": "number",
          "description": "Distance from entry as percentage"
        },
        "time_horizon": {
          "type": "string",
          "enum": ["scalp", "intraday", "swing", "position"]
        }
      }
    },
    "input_summary": {
      "type": "object",
      "description": "Summary of inputs used for decision",
      "properties": {
        "ta_bias": {"type": "string"},
        "ta_confidence": {"type": "number"},
        "regime": {"type": "string"},
        "regime_confidence": {"type": "number"},
        "sentiment_bias": {"type": "string"},
        "sentiment_confidence": {"type": "number"}
      }
    },
    "risk_assessment": {
      "type": "object",
      "properties": {
        "risk_reward_ratio": {"type": "number"},
        "estimated_risk_pct": {"type": "number"},
        "invalidation_level": {"type": "number"}
      }
    },
    "reasoning": {
      "type": "string",
      "maxLength": 500
    }
  }
}
```

### Agent Interface

```python
# src/agents/trading_decision.py

@dataclass
class TradingDecisionOutput(AgentOutput):
    """Trading Decision Agent output."""
    action: str
    confidence: float
    entry_type: str
    entry_price: Optional[Decimal]
    size_pct: float
    leverage: int
    stop_loss_pct: float
    take_profit_pct: float
    time_horizon: str
    risk_reward_ratio: float
    reasoning: str


@dataclass
class ModelDecision:
    """Single model's decision for comparison."""
    model_id: str
    action: str
    confidence: float
    parameters: dict
    reasoning: str
    latency_ms: int
    tokens_used: int
    cost_usd: Decimal


@dataclass
class ConsensusResult:
    """Result of 6-model consensus calculation."""
    consensus_action: str
    consensus_confidence: float
    agreement_level: str  # unanimous, strong_majority, majority, split
    vote_counts: dict[str, int]  # action -> count
    model_decisions: list[ModelDecision]
    selected_model: str  # Which model's parameters to use
    confidence_boost: float


class TradingDecisionAgent(BaseAgent):
    """
    Trading Decision Agent with 6-model A/B testing.

    Runs all models in parallel, calculates consensus, selects best decision.
    """

    agent_name = "trading_decision"
    llm_tier = "tier2_api"

    MODELS = [
        {"id": "gpt", "provider": "openai", "model": "gpt-4-turbo"},
        {"id": "grok", "provider": "xai", "model": "grok-2"},
        {"id": "deepseek", "provider": "deepseek", "model": "deepseek-v3"},
        {"id": "claude_sonnet", "provider": "anthropic", "model": "claude-sonnet-4-20250514"},
        {"id": "claude_opus", "provider": "anthropic", "model": "claude-opus-4-20250514"},
        {"id": "qwen", "provider": "ollama", "model": "qwen2.5:7b"},
    ]

    def __init__(
        self,
        llm_clients: dict,  # provider -> client
        prompt_builder,
        output_parser,
        config: dict
    ):
        self.llm_clients = llm_clients
        self.prompt_builder = prompt_builder
        self.parser = output_parser
        self.config = config

    async def process(
        self,
        snapshot: MarketSnapshot,
        ta_output: TAOutput,
        regime_output: RegimeOutput,
        sentiment_output: Optional[dict] = None,
        portfolio_context: Optional[dict] = None
    ) -> Tuple[TradingDecisionOutput, ConsensusResult]:
        """
        Make trading decision using 6-model parallel comparison.

        Args:
            snapshot: Market data
            ta_output: Technical analysis output
            regime_output: Regime detection output
            sentiment_output: Optional sentiment analysis
            portfolio_context: Portfolio state

        Returns:
            (TradingDecisionOutput, ConsensusResult) tuple
        """
        # Build prompt with all inputs
        prompt = self.prompt_builder.build_prompt(
            agent_name=self.agent_name,
            snapshot=snapshot,
            portfolio_context=portfolio_context,
            additional_context={
                "technical_analysis": ta_output.to_dict(),
                "regime": regime_output.to_dict(),
                "sentiment": sentiment_output
            }
        )

        # Run all 6 models in parallel
        model_decisions = await self._run_all_models_parallel(prompt)

        # Calculate consensus
        consensus = self._calculate_consensus(model_decisions)

        # Apply confidence boost based on agreement
        boosted_confidence = min(
            1.0,
            consensus.consensus_confidence + consensus.confidence_boost
        )

        # Get parameters from selected model
        selected = next(
            d for d in model_decisions
            if d.model_id == consensus.selected_model
        )

        # Build output
        output = TradingDecisionOutput(
            action=consensus.consensus_action,
            confidence=boosted_confidence,
            entry_type=selected.parameters.get("entry_type", "limit"),
            entry_price=selected.parameters.get("entry_price"),
            size_pct=selected.parameters.get("size_pct", 10),
            leverage=selected.parameters.get("leverage", 1),
            stop_loss_pct=selected.parameters.get("stop_loss_pct", 2.0),
            take_profit_pct=selected.parameters.get("take_profit_pct", 4.0),
            time_horizon=selected.parameters.get("time_horizon", "intraday"),
            risk_reward_ratio=selected.parameters.get("take_profit_pct", 4.0) / selected.parameters.get("stop_loss_pct", 2.0),
            reasoning=selected.reasoning
        )

        # Store all outputs for comparison analysis
        await self._store_comparison_results(model_decisions, consensus)

        return output, consensus

    async def _run_all_models_parallel(
        self,
        prompt: AssembledPrompt
    ) -> list[ModelDecision]:
        """Run all 6 models in parallel, return all decisions."""
        tasks = []
        for model_config in self.MODELS:
            tasks.append(
                self._run_single_model(model_config, prompt)
            )

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out failures
        decisions = []
        for result in results:
            if isinstance(result, ModelDecision):
                decisions.append(result)
            else:
                # Log error but continue
                pass

        return decisions

    def _calculate_consensus(
        self,
        decisions: list[ModelDecision]
    ) -> ConsensusResult:
        """
        Calculate consensus from model decisions.

        Consensus rules:
        - Unanimous (6/6): +0.15 confidence boost
        - Strong majority (5/6): +0.10 confidence boost
        - Majority (4/6): +0.05 confidence boost
        - Split (≤3/6): No boost, defer to top performer or HOLD
        """
        # Count votes
        vote_counts = {}
        for d in decisions:
            vote_counts[d.action] = vote_counts.get(d.action, 0) + 1

        # Find majority action
        majority_action = max(vote_counts, key=vote_counts.get)
        majority_count = vote_counts[majority_action]

        # Determine agreement level and confidence boost
        n_models = len(decisions)
        if majority_count == n_models:
            agreement = "unanimous"
            boost = 0.15
        elif majority_count >= n_models - 1:
            agreement = "strong_majority"
            boost = 0.10
        elif majority_count >= n_models // 2 + 1:
            agreement = "majority"
            boost = 0.05
        else:
            agreement = "split"
            boost = 0.0
            majority_action = "HOLD"  # Default to HOLD on split

        # Calculate average confidence for majority action
        majority_decisions = [d for d in decisions if d.action == majority_action]
        avg_confidence = sum(d.confidence for d in majority_decisions) / len(majority_decisions) if majority_decisions else 0.5

        # Select model (highest confidence among majority voters, or top historical performer)
        if majority_decisions:
            selected = max(majority_decisions, key=lambda d: d.confidence)
        else:
            selected = decisions[0] if decisions else None

        return ConsensusResult(
            consensus_action=majority_action,
            consensus_confidence=avg_confidence,
            agreement_level=agreement,
            vote_counts=vote_counts,
            model_decisions=decisions,
            selected_model=selected.model_id if selected else "none",
            confidence_boost=boost
        )
```

### Configuration

```yaml
# config/agents.yaml (trading_decision section)

agents:
  trading_decision:
    enabled: true

    invocation:
      trigger: scheduled
      interval_seconds: 3600  # Hourly
      symbols:
        - BTC/USDT
        - XRP/USDT

    # 6-Model A/B configuration
    models:
      - id: gpt
        provider: openai
        model: gpt-4-turbo
        enabled: true
        timeout_ms: 30000

      - id: grok
        provider: xai
        model: grok-2
        enabled: true
        timeout_ms: 30000

      - id: deepseek
        provider: deepseek
        model: deepseek-v3
        enabled: true
        timeout_ms: 30000

      - id: claude_sonnet
        provider: anthropic
        model: claude-sonnet-4-20250514
        enabled: true
        timeout_ms: 30000

      - id: claude_opus
        provider: anthropic
        model: claude-opus-4-20250514
        enabled: true
        timeout_ms: 60000

      - id: qwen
        provider: ollama
        model: qwen2.5:7b
        enabled: true
        timeout_ms: 5000

    # Consensus rules
    consensus:
      unanimous_boost: 0.15
      strong_majority_boost: 0.10
      majority_boost: 0.05
      split_action: HOLD

    # Output constraints
    constraints:
      max_size_pct: 20
      max_leverage: 5
      min_stop_loss_pct: 1.0
      max_stop_loss_pct: 5.0
      min_rr_ratio: 1.5

    # Performance tracking
    tracking:
      store_all_model_outputs: true
      leaderboard_update_frequency: hourly
```

### Database Table: Model Comparisons

```sql
-- New table for 6-model comparison results
CREATE TABLE model_comparisons (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    decision_id UUID REFERENCES trading_decisions(id),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    symbol VARCHAR(20) NOT NULL,

    -- Individual model outputs
    model_outputs JSONB NOT NULL,  -- Array of {model_id, action, confidence, latency_ms, cost}

    -- Consensus result
    consensus_action VARCHAR(10) NOT NULL,
    consensus_confidence DECIMAL(4, 3),
    agreement_level VARCHAR(20) NOT NULL,
    vote_counts JSONB,
    selected_model VARCHAR(20),
    confidence_boost DECIMAL(4, 3),

    -- Outcome tracking (updated later)
    outcome_correct BOOLEAN,
    price_after_1h DECIMAL(20, 10),
    price_after_4h DECIMAL(20, 10),
    price_after_24h DECIMAL(20, 10),

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_model_comparisons_ts
    ON model_comparisons (timestamp DESC);
CREATE INDEX idx_model_comparisons_symbol
    ON model_comparisons (symbol, timestamp DESC);
```

### Test Requirements

| Test | Description | Acceptance |
|------|-------------|------------|
| Parallel execution | All 6 models run | All results returned |
| Timeout handling | Model timeout handled | Graceful degradation |
| Consensus calculation | Correct vote counting | Accurate agreement |
| Confidence boost | Correct boost applied | Matches rules |
| Storage | All comparisons logged | Records in DB |

---

## Phase 2 Acceptance Criteria

### Functional Requirements

| Requirement | Test Method | Acceptance |
|-------------|-------------|------------|
| TA agent produces valid output | Schema validation | 100% valid |
| Regime agent classifies correctly | Manual review | Reasonable classifications |
| Risk engine blocks invalid trades | Test cases | 100% blocked |
| Trading agent runs 6 models | Log verification | All 6 invoked |
| Consensus calculated correctly | Unit tests | Correct algorithm |

### Integration Requirements

| Requirement | Acceptance |
|-------------|------------|
| TA → Regime flow | Regime uses TA output |
| TA + Regime → Trading | Trading uses both |
| Trading → Risk | Risk validates output |
| All outputs stored | Records in agent_outputs |

### Deliverables Checklist

- [ ] `src/agents/base_agent.py` - Base agent class
- [ ] `src/agents/technical_analysis.py` - TA agent
- [ ] `src/agents/regime_detection.py` - Regime agent
- [ ] `src/risk/rules_engine.py` - Risk engine
- [ ] `src/agents/trading_decision.py` - Trading agent
- [ ] Configuration files for all agents
- [ ] System prompts in `config/prompts/`
- [ ] Unit tests for each component
- [ ] Integration tests for agent flow
- [ ] Model comparison table migration

---

## API Endpoints (Phase 2)

```yaml
# API Routes for Phase 2

endpoints:
  # Agent outputs
  - path: /api/v1/agents/ta/{symbol}
    method: GET
    description: Get latest TA analysis
    response: TAOutput

  - path: /api/v1/agents/regime/{symbol}
    method: GET
    description: Get current regime
    response: RegimeOutput

  # Manual triggers (for testing)
  - path: /api/v1/agents/ta/{symbol}/run
    method: POST
    description: Manually trigger TA analysis
    response: TAOutput

  - path: /api/v1/agents/trading/{symbol}/run
    method: POST
    description: Manually trigger trading decision
    response: TradingDecisionOutput

  # Risk
  - path: /api/v1/risk/validate
    method: POST
    description: Validate a trade proposal
    body: TradeProposal
    response: RiskValidation

  - path: /api/v1/risk/state
    method: GET
    description: Get current risk state
    response: RiskState
```

---

## References

- Design: [01-multi-agent-architecture.md](../TripleGain-master-design/01-multi-agent-architecture.md)
- Design: [02-llm-integration-system.md](../TripleGain-master-design/02-llm-integration-system.md)
- Design: [03-risk-management-rules-engine.md](../TripleGain-master-design/03-risk-management-rules-engine.md)
- Design: [06-evaluation-framework.md](../TripleGain-master-design/06-evaluation-framework.md)

---

*Phase 2 Implementation Plan v1.0 - December 2025*
