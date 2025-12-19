# Review Phase 5: Configuration & Integration

**Status**: Ready for Review
**Estimated Context**: ~3,000 tokens (config) + review
**Priority**: High - System-wide consistency
**Output**: `findings/phase-5-findings.md`
**DO NOT IMPLEMENT FIXES**

---

## Files to Review

### Configuration Files

| File | Lines | Purpose |
|------|-------|---------|
| `config/agents.yaml` | ~300 | Agent configuration |
| `config/risk.yaml` | ~200 | Risk parameters |
| `config/orchestration.yaml` | ~150 | Coordinator settings |
| `config/portfolio.yaml` | ~100 | Portfolio allocation |
| `config/execution.yaml` | ~150 | Execution settings |
| `config/database.yaml` | ~80 | Database connection |
| `config/indicators.yaml` | ~150 | Indicator parameters |
| `config/prompts.yaml` | ~100 | Prompt configuration |
| `config/snapshot.yaml` | ~80 | Snapshot builder |

### Cross-Cutting Code

| File | Purpose |
|------|---------|
| All `__init__.py` files | Module exports |
| Database migrations | Schema consistency |
| Test configuration | Test settings |

**Total**: ~1,300 lines config + integration review

---

## Pre-Review: Load Files

```bash
# Read all config files
cat config/agents.yaml
cat config/risk.yaml
cat config/orchestration.yaml
cat config/portfolio.yaml
cat config/execution.yaml
cat config/database.yaml
cat config/indicators.yaml
cat config/prompts.yaml
cat config/snapshot.yaml
```

---

## Part 1: Configuration Review

### 1.1 Agents Configuration (`agents.yaml`)

#### Technical Analysis Agent
- [ ] Model: qwen2.5:7b (Ollama)
- [ ] Interval: 60 seconds
- [ ] Symbols: BTC/USDT, XRP/USDT, XRP/BTC
- [ ] Timeout: 5000ms
- [ ] Token budget: 3000

#### Regime Detection Agent
- [ ] Model: qwen2.5:7b (Ollama)
- [ ] Interval: 300 seconds (5 min)
- [ ] Regime parameters for each regime type
- [ ] Multipliers in valid ranges

#### Trading Decision Agent
- [ ] All 6 models configured
- [ ] Correct model names
- [ ] Appropriate timeouts
- [ ] Consensus settings match spec
- [ ] Constraints match design

#### Portfolio Rebalance Agent
- [ ] Model: deepseek-v3
- [ ] Interval: 3600 seconds
- [ ] Threshold: 5%

---

### 1.2 Risk Configuration (`risk.yaml`)

#### Position Sizing
- [ ] risk_per_trade_pct: 1.0 (reasonable)
- [ ] max_position_pct: 20.0 (reasonable)
- [ ] max_total_exposure_pct: 60.0 (reasonable)

#### Leverage Limits
- [ ] By regime, all <= 5
- [ ] Choppy regime = 1 (most conservative)

#### Confidence Thresholds
- [ ] By regime, reasonable values
- [ ] Choppy highest (0.75)

#### Circuit Breakers
- [ ] daily_loss_limit_pct: 5.0
- [ ] weekly_loss_limit_pct: 10.0
- [ ] max_drawdown_pct: 20.0
- [ ] consecutive_loss_limit: 5
- [ ] volatility_spike_threshold: 3.0

#### Cooldowns
- [ ] Reasonable durations
- [ ] Match implementation

---

### 1.3 Orchestration Configuration (`orchestration.yaml`)

#### LLM Settings
- [ ] Primary: deepseek-v3
- [ ] Fallback: claude-sonnet
- [ ] Both valid model names

#### Symbols
- [ ] Match agents.yaml
- [ ] Consistent format

#### Schedules
- [ ] Match implementation plan
- [ ] All enabled as expected

#### Conflict Resolution
- [ ] Threshold: 0.2
- [ ] Max time: 10000ms

---

### 1.4 Portfolio Configuration (`portfolio.yaml`)

#### Target Allocation
- [ ] btc_pct: 33.33
- [ ] xrp_pct: 33.33
- [ ] usdt_pct: 33.33
- [ ] Sums to 100 (within rounding)

#### Rebalancing
- [ ] Threshold: 5.0% (reasonable)
- [ ] Min trade: $10 (reasonable)
- [ ] Execution type: limit (safer)

#### Hodl Bags
- [ ] Enabled: true
- [ ] Allocation: 10%
- [ ] Assets: BTC, XRP

---

### 1.5 Execution Configuration (`execution.yaml`)

#### Kraken Settings
- [ ] API key from env var
- [ ] API secret from env var
- [ ] Rate limits reasonable

#### Order Settings
- [ ] Default type: limit (safer)
- [ ] Time in force: GTC
- [ ] Retry count: 3

#### Position Management
- [ ] Sync interval: 30s
- [ ] Max positions: 6

#### Slippage Protection
- [ ] Max slippage: 0.5%
- [ ] Limit orders: true

---

### 1.6 Database Configuration (`database.yaml`)

#### Connection
- [ ] Host from env var
- [ ] Port default: 5432
- [ ] Pool size appropriate (5-20)

#### Retention
- [ ] agent_outputs: 90 days
- [ ] trading_decisions: indefinite
- [ ] trade_executions: indefinite
- [ ] Reasonable settings

---

### 1.7 Indicators Configuration (`indicators.yaml`)

#### Indicator Parameters
- [ ] EMA periods: [9, 21, 50, 200]
- [ ] SMA periods: [20, 50, 200]
- [ ] RSI period: 14
- [ ] MACD: 12/26/9
- [ ] ATR period: 14
- [ ] ADX period: 14
- [ ] BB: 20, 2.0

#### Timeframes
- [ ] All required: 1m, 5m, 15m, 1h, 4h, 1d

#### Symbols
- [ ] Match other configs

---

### 1.8 Prompts Configuration (`prompts.yaml`)

#### Token Budgets
- [ ] tier1_local appropriate (8192 total)
- [ ] tier2_api appropriate (128000 total)
- [ ] Allocations sum correctly

#### Agent Settings
- [ ] Each agent has template reference
- [ ] Tiers match LLM assignments

---

### 1.9 Snapshot Configuration (`snapshot.yaml`)

#### Candle Lookback
- [ ] Reasonable per timeframe
- [ ] Won't exceed available data

#### Indicators
- [ ] Match indicator_library implementation
- [ ] All used indicators included

#### Token Budgets
- [ ] tier1_local: 3500
- [ ] tier2_api: 8000
- [ ] Consistent with prompts.yaml

---

## Part 2: Configuration Consistency

### Cross-Config Validation

| Setting | Location 1 | Location 2 | Match? |
|---------|------------|------------|--------|
| Symbols | agents.yaml | orchestration.yaml | [ ] |
| Symbols | agents.yaml | indicators.yaml | [ ] |
| Token budgets | prompts.yaml | snapshot.yaml | [ ] |
| Timeframes | indicators.yaml | snapshot.yaml | [ ] |
| Models | agents.yaml | (implementation) | [ ] |
| Thresholds | risk.yaml | (implementation) | [ ] |

---

## Part 3: Environment Variables

### Required Environment Variables

```bash
# Database
DATABASE_HOST
DATABASE_PORT
DATABASE_NAME
DATABASE_USER
DATABASE_PASSWORD

# Kraken
KRAKEN_API_KEY
KRAKEN_API_SECRET

# LLM Providers
OPENAI_API_KEY
ANTHROPIC_API_KEY
DEEPSEEK_API_KEY
XAI_API_KEY

# Ollama (optional - has default)
OLLAMA_HOST
```

- [ ] All required vars documented
- [ ] Defaults provided where sensible
- [ ] No secrets in config files
- [ ] .env.example provided

---

## Part 4: Integration Review

### Module Dependencies

```
data/
  ├── database.py (no deps)
  ├── indicator_library.py (uses: database)
  └── market_snapshot.py (uses: database, indicator_library)

llm/
  ├── clients/ (no internal deps)
  └── prompt_builder.py (no deps)

agents/
  ├── base_agent.py (uses: llm/clients)
  ├── technical_analysis.py (uses: base, data)
  ├── regime_detection.py (uses: base, data)
  ├── trading_decision.py (uses: base, llm/clients)
  └── portfolio_rebalance.py (uses: base)

risk/
  └── rules_engine.py (uses: data)

orchestration/
  ├── message_bus.py (no deps)
  └── coordinator.py (uses: message_bus, agents, risk)

execution/
  ├── order_manager.py (uses: orchestration)
  └── position_tracker.py (uses: data)

api/
  └── *.py (uses: all above)
```

- [ ] No circular dependencies
- [ ] Dependencies match expected
- [ ] Clean module boundaries

### Import Verification

```bash
# Verify imports work
python -c "from triplegain.src import *"
```

- [ ] All modules importable
- [ ] No import errors
- [ ] __all__ exports correct

---

## Part 5: Database Schema Review

### Migration Consistency

```bash
ls migrations/
# Expected:
# 001_initial_schema.sql
# 002_phase2_agents.sql
# 003_phase3_orchestration.sql
```

- [ ] Migrations in order
- [ ] All tables created
- [ ] Indexes created
- [ ] Constraints correct
- [ ] Schema matches code expectations

### Table Inventory

| Table | Created In | Used By |
|-------|------------|---------|
| agent_outputs | 001 | All agents |
| trading_decisions | 001 | Trading, Risk |
| trade_executions | 001 | Execution |
| portfolio_snapshots | 001 | Portfolio |
| risk_state | 001 | Risk |
| external_data_cache | 001 | (Phase 4) |
| indicator_cache | 001 | Data |
| model_comparisons | 002 | Trading |
| order_status_log | 003 | Execution |
| position_snapshots | 003 | Execution |

- [ ] All tables exist
- [ ] Used correctly in code

---

## Part 6: Test Configuration

### pytest.ini / pyproject.toml

- [ ] Test paths configured
- [ ] Coverage settings
- [ ] Markers defined (unit, integration)
- [ ] Async support configured

### Test Environment

- [ ] Test database configured
- [ ] Mock LLM clients available
- [ ] Fixtures defined for common objects

---

## Part 7: Documentation Sync

### CLAUDE.md Accuracy

- [ ] Quick commands work
- [ ] Phase status accurate
- [ ] Key facts accurate
- [ ] Structure matches reality

### Implementation Plan vs Reality

| Plan Component | Implementation Status |
|----------------|----------------------|
| Phase 1 all items | [ ] Verify complete |
| Phase 2 all items | [ ] Verify complete |
| Phase 3 all items | [ ] Verify complete |

---

## Critical Integration Questions

1. **Config Loading**: Is config loaded once at startup or re-read?
2. **Environment Override**: Can env vars override file config?
3. **Hot Reload**: Any support for config hot reload?
4. **Validation**: Is config validated at startup?
5. **Secrets**: Are any secrets accidentally in config files?
6. **Consistency**: Do all components use same config source?

---

## Findings Template

```markdown
## Finding: [Title]

**File**: `config/filename.yaml` or multiple files
**Priority**: P0/P1/P2/P3
**Category**: Configuration/Integration/Consistency

### Description
[What was found]

### Current State
```yaml
# current configuration
```

### Recommended Fix
```yaml
# recommended configuration
```

### Impact
[What could go wrong]
```

---

## Final Integration Checklist

### System Startup
- [ ] All configs load without error
- [ ] Database connection established
- [ ] All agents initialize
- [ ] Coordinator starts
- [ ] API server starts

### System Health
- [ ] /health endpoint returns OK
- [ ] Database query succeeds
- [ ] Ollama connection verified
- [ ] No warning logs

### End-to-End Flow
- [ ] Snapshot builds successfully
- [ ] TA agent runs
- [ ] Regime agent runs
- [ ] Trading decision agent runs
- [ ] Risk validation works
- [ ] (Paper) execution works

---

## Review Completion

After completing this phase:

1. [ ] All config files reviewed
2. [ ] Cross-config consistency verified
3. [ ] Environment variables documented
4. [ ] Module dependencies verified
5. [ ] Database schema verified
6. [ ] Test configuration verified
7. [ ] Documentation sync verified
8. [ ] Findings documented
9. [ ] **Review 4 Complete**

---

## Final Summary Report

After completing all phases, prepare summary including:

1. **Total Issues by Priority**
   - P0 (Critical): X
   - P1 (High): X
   - P2 (Medium): X
   - P3 (Low): X

2. **Issues by Category**
   - Security: X
   - Logic: X
   - Performance: X
   - Quality: X

3. **Critical Paths Affected**
   - List any core flows with issues

4. **Paper Trading Readiness**
   - Yes/No with conditions

5. **Recommended Fix Order**
   - Prioritized list

---

*Phase 5 Review Plan v1.0 - FINAL PHASE*
