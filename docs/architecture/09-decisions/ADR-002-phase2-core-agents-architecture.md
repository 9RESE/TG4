# ADR-002: Phase 2 Core Agents Architecture

**Status**: Accepted
**Date**: 2025-12-18
**Context**: Phase 2 Core Agents Implementation

## Context

Phase 2 requires implementing the core agent system for TripleGain:
- Multiple specialized agents (TA, Regime, Trading Decision)
- Risk management engine
- Multi-provider LLM integration
- 6-model A/B testing framework

Key considerations:
1. Agents must produce consistent, validated outputs
2. Risk engine must be deterministic and fast (<10ms)
3. 6 different LLM providers must be integrated
4. Model decisions must be tracked for performance comparison

## Decision Drivers

- **Reliability**: Agents must produce valid outputs or safe fallbacks
- **Performance**: Risk validation must be fast enough for real-time trading
- **Flexibility**: Easy to add new models or adjust parameters
- **Observability**: Track all model decisions for A/B analysis
- **Cost Control**: Minimize API costs while maintaining quality

## Decisions Made

### 1. Abstract Base Agent with Dataclass Outputs

**Decision**: Use abstract base class with strongly-typed dataclass outputs.

**Rationale**:
- Consistent interface across all agents
- Built-in validation at output creation
- Automatic serialization for storage
- Type hints for IDE support

**Alternatives Considered**:
- Dict-based outputs: Less type safety, harder to validate
- Pydantic models: More dependencies, similar benefits

### 2. Rules-Based Risk Engine (No LLM)

**Decision**: Implement risk management as deterministic rules engine without LLM.

**Rationale**:
- <10ms latency requirement
- Predictable, auditable behavior
- No network dependency
- No API costs
- Works when LLMs are unavailable

**Alternatives Considered**:
- LLM-assisted risk: Too slow, unpredictable, expensive
- Hybrid approach: Added complexity without clear benefit

### 3. Parallel 6-Model A/B Testing

**Decision**: Query all 6 models in parallel, calculate consensus via voting.

**Rationale**:
- Latency = max(individual latencies) instead of sum
- Real production data for model comparison
- Consensus reduces individual model errors
- Statistical basis for model selection

**Alternatives Considered**:
- Sequential queries: Too slow for hourly decisions
- Single model selection: No comparison data
- Weighted voting: Premature without baseline data

### 4. Consensus Algorithm: Majority Vote with Averaged Parameters

**Decision**: Use simple majority voting for action, average parameters from agreeing models.

**Rationale**:
- Simple and transparent
- Equal weighting until performance data available
- Averaging reduces outlier parameter values

**Future Enhancement**: Weight by historical accuracy once data collected.

### 5. Fallback Outputs from Indicators

**Decision**: When LLM parsing fails, generate output from technical indicators.

**Rationale**:
- System never stops producing signals
- Indicator-based fallback is conservative
- Lower confidence indicates reduced reliability

**Implementation**:
- Parse failure -> indicator-based output (confidence 0.4)
- Complete failure -> neutral output (confidence 0.0)

### 6. Multi-Client LLM Architecture

**Decision**: Separate client class per provider with common interface.

**Rationale**:
- Each provider has unique API quirks
- Easy to add new providers
- Provider-specific error handling
- Independent configuration

**Clients Implemented**:
- OllamaClient (local, no cost)
- OpenAIClient (GPT models)
- AnthropicClient (Claude models)
- DeepSeekClient (DeepSeek V3)
- XAIClient (Grok models)

### 7. Regime-Based Parameter Adjustment

**Decision**: Market regime determines position sizing, leverage limits, and entry strictness.

**Rationale**:
- Adapt to market conditions automatically
- Conservative in choppy/volatile markets
- Aggressive in clear trends
- Reduces losses in difficult conditions

**Regime Multipliers**:
| Regime | Position Size | Max Leverage |
|--------|---------------|--------------|
| trending_bull | 1.0x | 5x |
| choppy | 0.25x | 1x |
| volatile | 0.5x | 2x |

### 8. Circuit Breakers and Cooldowns

**Decision**: Automatic trading halts and cooldowns based on performance.

**Rationale**:
- Prevent catastrophic losses
- Force pause after consecutive losses
- Allow system/human review before resuming

**Triggers**:
- Daily loss > 5%: Halt until next day
- Max drawdown > 20%: Halt + close positions
- 5 consecutive losses: Cooldown + 1x leverage only

## Consequences

### Positive

- Consistent agent outputs with validation
- Fast, deterministic risk management
- Comprehensive model comparison data
- Automatic adaptation to market conditions
- Built-in loss prevention mechanisms

### Negative

- Higher API costs running 6 models in parallel
- Coverage gaps in LLM client network code
- Consensus may lag individual model insights

### Risks

- API provider outages affect multi-model consensus
- Model cost increases over time
- Regime detection errors propagate to all decisions

## Compliance

- **Testing**: 136 unit tests for Phase 2 components
- **Coverage**: 67% overall (83% for risk engine)
- **Latency**: Risk validation measured at <10ms

## References

- [Phase 2 Feature Documentation](../../development/features/phase-2-core-agents.md)
- [Phase 2 Implementation Plan](../../development/TripleGain-implementation-plan/02-phase-2-core-agents.md)
- [Risk Management Design](../../development/TripleGain-master-design/03-risk-management-rules-engine.md)

---

*ADR-002 - Core Agents Architecture - December 2025*
