# ADR-012: Configuration & Integration Fixes

## Status
Accepted

## Date
2025-12-19

## Context

During the Phase 5 code review focusing on configuration files, migrations, and cross-module integration, 15 issues were identified across different priority levels. The most critical finding was:

1. **P0 Critical**: Duplicate migration file numbering (two `003_*.sql` files) causing non-deterministic schema state
2. **P1 High**: Template file extension mismatch (`agents.yaml` referenced `.md` but actual files were `.txt`)
3. **P1 High**: Missing `.env.example` file for environment variable documentation
4. **P1 High**: Incomplete config validation (only 4 of 9 config files validated at startup)
5. **P1 High**: Symbol list inconsistency between `orchestration.yaml` and other config files

These issues could cause runtime failures, deployment confusion, and schema inconsistencies.

## Decision

We implemented all 15 fixes across configuration, migration, and integration layers:

### Critical Fixes (P0 - 1 issue)
- **F01**: Renumbered duplicate migrations:
  - `003_phase3_orchestration.sql` → `004_phase3_orchestration.sql`
  - Migration order now deterministic: 001 → 002 → 003 → 004

### High Priority Fixes (P1 - 4 issues)
- **F02**: Fixed template file extensions in `agents.yaml`:
  - Changed all `.md` references to `.txt` (technical_analysis, regime_detection, sentiment_analysis, trading_decision, coordinator)

- **F03**: Created `.env.example` with all required environment variables:
  - Database configuration (host, port, name, user, password)
  - Kraken API credentials
  - LLM provider API keys (OpenAI, Anthropic, DeepSeek, xAI, Ollama)

- **F04**: Added validators for all 9 configuration files:
  - New validators: `_validate_agents_config`, `_validate_risk_config`, `_validate_orchestration_config`, `_validate_portfolio_config`, `_validate_execution_config`
  - Added `validate_all_configs_on_startup()` function for application initialization

- **F05**: Fixed symbol list inconsistency:
  - Uncommented `XRP/BTC` in `orchestration.yaml` to match `indicators.yaml` and `execution.yaml`

### Medium Priority Fixes (P2 - 6 issues)
- **F06**: Enabled coordinator agent in `agents.yaml` (Phase 3 is complete)

- **F07**: Updated `CLAUDE.md` to document max exposure (80%) from `risk.yaml`

- **F08**: Added re-exports to `triplegain/src/llm/__init__.py`:
  - Exports: `BaseLLMClient`, `LLMResponse`, `OllamaClient`, `OpenAIClient`, `AnthropicClient`, `DeepSeekClient`, `XAIClient`

- **F09**: Created shared test fixtures in `triplegain/tests/conftest.py`:
  - `temp_config_dir` - Creates temporary config directory with all 9 config files
  - `mock_llm_client` - Mock LLM client for agent testing
  - `sample_candles`, `sample_candles_extended` - Market data fixtures
  - `sample_ticker`, `sample_order_book` - Market data fixtures
  - `sample_risk_state`, `sample_position`, `sample_trading_signal` - Domain fixtures

- **F10**: Fixed risk_state table duplication in migrations:
  - Added `DROP TABLE IF EXISTS risk_state CASCADE` before `CREATE TABLE` in migration 003
  - Documented that 003 schema (JSONB-based) supersedes 001 schema

- **F11**: Aligned token budgets between `prompts.yaml` and `snapshot.yaml`:
  - Updated `snapshot.yaml` to use `market_data` values from `prompts.yaml`
  - tier1_local: 3000 tokens, tier2_api: 6000 tokens

### Low Priority Fixes (P3 - 4 issues)
- **F12**: Cleaned up confusing XAI comment in `agents.yaml`

- **F13**: Added convenience exports to `triplegain/src/__init__.py`:
  - Re-exports all LLM clients for simpler imports

- **F14**: Added startup config validation function:
  - `validate_all_configs_on_startup()` validates critical configs first, then optional
  - Critical: agents, risk, orchestration, execution, database
  - Optional: indicators, prompts, snapshot, portfolio

- **F15**: Added portfolio_rebalance agent configuration to `agents.yaml`:
  - Enabled, tier2_api, deepseek provider, hourly frequency

## Technical Details

### Config Validator Architecture
```python
validators = {
    'indicators': self._validate_indicators_config,
    'snapshot': self._validate_snapshot_config,
    'database': self._validate_database_config,
    'prompts': self._validate_prompts_config,
    'agents': self._validate_agents_config,      # NEW
    'risk': self._validate_risk_config,          # NEW
    'orchestration': self._validate_orchestration_config,  # NEW
    'portfolio': self._validate_portfolio_config,  # NEW
    'execution': self._validate_execution_config,  # NEW
}
```

### Migration Numbering (After Fix)
```
migrations/
  001_agent_tables.sql         # Tables: agent_outputs, trading_decisions, etc.
  002_model_comparisons.sql    # Table: model_comparisons
  003_risk_state_and_indexes.sql  # Table: risk_state (JSONB), indexes
  004_phase3_orchestration.sql  # Tables: positions, orders, coordinator_state
```

### Shared Test Fixtures
```python
# triplegain/tests/conftest.py provides:
@pytest.fixture
def temp_config_dir():          # Full config directory
def mock_llm_client():          # LLM mock with generate()
def sample_candles():           # 5 OHLCV candles
def sample_candles_extended():  # 50 candles for RSI tests
def sample_ticker():            # BTC/USDT ticker
def sample_order_book():        # Bid/ask data
def sample_risk_state():        # Risk engine state
def sample_position():          # Open position
def sample_trading_signal():    # Trading signal
```

## Consequences

### Positive
- **Deployment Reliability**: Consistent migration ordering prevents schema drift between environments
- **Runtime Safety**: Template loading will now succeed with correct file extensions
- **Developer Experience**: `.env.example` documents all required environment variables
- **Early Error Detection**: All configs validated at startup, not at first use
- **Code Reuse**: Shared test fixtures reduce duplication across 1045 tests
- **Import Convenience**: Simpler import paths for LLM clients

### Negative
- **Breaking Change**: Existing databases need migration 004 (renamed from 003)
- **Test Fixture Dependency**: Tests using conftest.py fixtures may behave differently

## Alternatives Considered

### Migration Renumbering
- **Alternative**: Use timestamps instead of sequential numbers (e.g., `20251218_agent_tables.sql`)
- **Rejected**: Would require changing existing migration tracking and is overkill for this project size

### Template Extension Fix
- **Alternative**: Rename actual template files from `.txt` to `.md`
- **Rejected**: `.txt` is more appropriate for plain-text prompt templates; config was wrong

### Config Validation
- **Alternative**: Use Pydantic models for all configuration
- **Deferred**: Would require significant refactoring; current validators are sufficient

## Related Documents
- [Phase 5 Findings](../../development/reviews/full/review-4/findings/phase-5-findings.md)
- [Configuration Loader](../../../triplegain/src/utils/config.py)
