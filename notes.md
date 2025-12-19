# Notes
Prompt:
# Task: Begin TripleGain Phase 2 Implementation - Core Agents
## Context
Read these files first:
1. CLAUDE.md - Project memory and constraints
2. docs/development/TripleGain-implementation-plan/02-phase-2-core-agents.md - Phase 2 spec
3. docs/development/TripleGain-master-design/01-multi-agent-architecture.md - Agent architecture
4. docs/development/TripleGain-master-design/02-llm-integration-system.md - LLM integration
5. docs/development/TripleGain-master-design/03-risk-management-rules-engine.md - Risk rules
## Phase 1 Complete (DO NOT recreate)
Located in `triplegain/src/`:
- Indicator Library: `data/indicator_library.py` (17+ indicators)
- Market Snapshot: `data/market_snapshot.py` (multi-timeframe aggregation)
- Prompt Builder: `llm/prompt_builder.py` (tier-aware templates)
- Database schema with 7 tables, compression, retention policies
- API endpoints: health, indicators, snapshots, debug
- Config system: YAML-based with env var substitution
- 218 tests passing, 82% coverage
## Phase 2 Deliverables
1. **Base Agent Class** (`triplegain/src/agents/base_agent.py`)
- Abstract base with process(), get_output_schema(), store_output()
- AgentOutput dataclass with common fields
2. **Technical Analysis Agent** (`triplegain/src/agents/technical_analysis.py`)
- Model: Qwen 2.5 7B via Ollama (local)
- Invocation: Every minute on candle close
- Input: MarketSnapshot (compact)
- Output: trend, momentum, key_levels, signals, bias, confidence
3. **Regime Detection Agent** (`triplegain/src/agents/regime_detection.py`)
- Model: Qwen 2.5 7B via Ollama (local)
- Invocation: Every 5 minutes
- Output: regime (trending_bull/bear, ranging, volatile, choppy, breakout_potential)
- Includes recommended_adjustments (position_size, stop_loss, take_profit multipliers)
4. **Risk Management Engine** (`triplegain/src/risk/rules_engine.py`)
- NO LLM - purely rule-based, deterministic
- Circuit breakers: daily loss 5%, weekly 10%, max drawdown 20%
- Position sizing, leverage limits, R:R validation
- Latency target: <10ms
5. **Trading Decision Agent** (`triplegain/src/agents/trading_decision.py`)
- 6-Model A/B Testing: GPT, Grok, DeepSeek V3, Claude Sonnet, Claude Opus, Qwen
- All models run in parallel, consensus calculated
- Invocation: Hourly
- Output: action (BUY/SELL/HOLD/CLOSE), confidence, parameters, reasoning
6. **LLM Clients** (`triplegain/src/llm/clients/`)
- OllamaClient for local models
- OpenAIClient, AnthropicClient, DeepSeekClient, XAIClient for API models
- Common interface: generate(model, system_prompt, user_message)
7. **Database Migration**
- model_comparisons table for 6-model A/B tracking
- Outcome tracking (price_after_1h, 4h, 24h)
8. **Configuration** (`config/agents.yaml`, `config/risk.yaml`)
- Agent invocation settings, model configs
- Risk parameters, circuit breakers, cooldowns
9. **API Endpoints**
- GET /api/v1/agents/ta/{symbol} - Latest TA analysis
- GET /api/v1/agents/regime/{symbol} - Current regime
- POST /api/v1/agents/trading/{symbol}/run - Trigger trading decision
- POST /api/v1/risk/validate - Validate trade proposal
- GET /api/v1/risk/state - Current risk state
## Constraints
- Reuse existing indicator_library.py, market_snapshot.py, prompt_builder.py
- All agent outputs validated against JSON schemas
- Store all outputs to agent_outputs table
- Test-first approach (pytest) - maintain 80%+ coverage
- TA agent latency <500ms, Risk engine <10ms
- Handle LLM timeouts gracefully with fallbacks
## LLM Model Configuration
| Model | Provider | Usage |
|-------|----------|-------|
| Qwen 2.5 7B | Ollama (local) | TA, Regime, baseline A/B |
| GPT-4-turbo | OpenAI | A/B comparison + news |
| Grok-2 | xAI | A/B + sentiment + news |
| DeepSeek V3 | DeepSeek | A/B primary |
| Claude Sonnet | Anthropic | A/B comparison |
| Claude Opus | Anthropic | A/B complex |
## Starting Instructions
1. Create `triplegain/src/agents/` directory structure
2. Implement base_agent.py with abstract interface
3. Build OllamaClient and test with local Qwen 2.5 7B
4. Implement TechnicalAnalysisAgent first (simplest, tests LLM integration)
5. Add unit tests for each component
6. Proceed to RegimeDetectionAgent, then RiskManagementEngine, then TradingDecisionAgent