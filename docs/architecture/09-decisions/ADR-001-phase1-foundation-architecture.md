# ADR-001: Phase 1 Foundation Architecture

## Status

**Accepted** (2025-12-18)

## Context

TripleGain is an LLM-assisted cryptocurrency trading system requiring a solid foundation layer before implementing trading agents. The foundation must:

1. Calculate technical indicators accurately and performantly
2. Build market snapshots aggregating multi-timeframe data
3. Generate LLM prompts that fit within token budgets
4. Store agent outputs and trading decisions for audit
5. Support both local (Ollama/Qwen) and API-based LLMs

## Decision

### 1. Synchronous Indicator Calculation

**Decision**: Implement `IndicatorLibrary.calculate_all()` as a synchronous function despite the original design specifying async.

**Rationale**:
- Indicator calculation is CPU-bound, not I/O-bound
- NumPy vectorization makes calculations extremely fast (<50ms for 1000 candles)
- Async overhead would add complexity without benefit
- Can be wrapped in `asyncio.to_thread()` if needed later

### 2. Tier-Aware Prompt Building

**Decision**: Implement two output formats for market snapshots:
- `to_prompt_format()` - Full JSON for API LLMs (8000+ tokens available)
- `to_compact_format()` - Minimal JSON for local LLMs (3500 tokens)

**Rationale**:
- Local Qwen 2.5 7B has 8192 context limit, need room for response
- API models (GPT-4, Claude) have 128k+ context but still benefit from concise prompts
- Token budget management prevents truncation errors

### 3. Database Schema with TimescaleDB Features

**Decision**: Use TimescaleDB hypertables with retention and compression policies for:
- `agent_outputs` (90-day retention)
- `indicator_cache` (7-day retention, 1-day compression)
- `portfolio_snapshots` (7-day compression, indefinite retention)

**Rationale**:
- Automatic data lifecycle management
- Compression reduces storage costs by 80-90%
- Retention policies prevent unbounded growth
- Hypertables optimize time-series queries

### 4. Configuration Externalization

**Decision**: Implement YAML-based configuration with environment variable substitution via `ConfigLoader` class.

**Rationale**:
- Separates configuration from code
- Enables different configs per environment (dev/staging/prod)
- Secrets via env vars, not committed to repo
- Validation catches misconfiguration early

### 5. API-First Development

**Decision**: Implement FastAPI endpoints for all Phase 1 components:
- `/health`, `/health/live`, `/health/ready`
- `/api/v1/indicators/{symbol}/{timeframe}`
- `/api/v1/snapshot/{symbol}`
- `/api/v1/debug/prompt/{agent}`

**Rationale**:
- Enables testing without building agents
- Kubernetes-compatible health checks
- Debug endpoints aid development
- Foundation for Phase 4 dashboard

## Consequences

### Positive

- **Performance**: 82% test coverage, all tests pass in <2s
- **Maintainability**: Clear separation between data/llm/api/utils layers
- **Extensibility**: New indicators easy to add (18 lines average)
- **Observability**: API endpoints enable monitoring and debugging
- **Production-Ready**: Database policies handle data lifecycle automatically

### Negative

- **Technical Debt**: `IndicatorResult` dataclass defined but not used (returns raw dict)
- **Coverage Gap**: API endpoints at 62% coverage (async paths harder to test)
- **Async Pattern**: Synchronous indicators may need wrapping for true async workflows

### Neutral

- **Token Estimation**: Uses 3.5 chars/token heuristic instead of tiktoken
- **Warmup Inconsistency**: Different indicators have different warmup periods (documented)

## Alternatives Considered

### 1. TA-Lib for Indicators

**Rejected**: Would add C dependency, harder to deploy, less control over edge cases.

### 2. PostgreSQL without TimescaleDB

**Rejected**: Would lose automatic partitioning, retention policies, and compression.

### 3. Redis for Indicator Cache

**Rejected**: TimescaleDB indicator_cache table sufficient, avoids additional infrastructure.

### 4. GraphQL API

**Rejected**: REST simpler for current needs, can add GraphQL in Phase 4 if needed.

## References

- [Phase 1 Implementation Plan](../../development/TripleGain-implementation-plan/01-phase-1-foundation.md)
- [Phase 1 Comprehensive Review](../../development/reviews/phase-1/phase-1-comprehensive-review.md)
- [Master Design - Data Pipeline](../../development/TripleGain-master-design/04-data-pipeline.md)

---

*ADR-001 v1.0 - December 2025*
