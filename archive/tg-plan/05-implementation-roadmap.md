# Implementation Roadmap

**Version:** 1.0
**Date:** December 2025
**Status:** Planning Phase

---

## Overview

This document outlines the implementation roadmap for the TripleGain multi-asset LLM trading system, organized into four phases with clear dependencies and deliverables.

---

## Table of Contents

1. [Phase Summary](#1-phase-summary)
2. [Phase 1: Foundation](#2-phase-1-foundation)
3. [Phase 2: Paper Trading](#3-phase-2-paper-trading)
4. [Phase 3: Micro-Live](#4-phase-3-micro-live)
5. [Phase 4: Scale](#5-phase-4-scale)
6. [Dependency Graph](#6-dependency-graph)
7. [Risk Checkpoints](#7-risk-checkpoints)

---

## 1. Phase Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           IMPLEMENTATION PHASES                              │
└─────────────────────────────────────────────────────────────────────────────┘

Phase 1: FOUNDATION
├── LLM strategy module
├── Prompt engineering
├── Multi-agent framework
└── Backtesting integration

Phase 2: PAPER TRADING
├── Minimum 30 days paper trading
├── All LLMs running in parallel
├── Performance comparison
└── Strategy refinement

Phase 3: MICRO-LIVE
├── $100 USDT live trading
├── Best-performing LLM only
├── Real market validation
└── Gradual confidence building

Phase 4: SCALE
├── Gradual capital increase
├── Multi-model production
├── Full automation
└── Continuous improvement
```

### Success Gates

| Phase | Entry Criteria | Exit Criteria |
|-------|----------------|---------------|
| **Phase 1** | Research complete | All components tested |
| **Phase 2** | Phase 1 complete | 30+ days paper, metrics met |
| **Phase 3** | Phase 2 metrics met | 30+ days live, positive P&L |
| **Phase 4** | Phase 3 profitable | Ongoing operations |

---

## 2. Phase 1: Foundation

### 2.1 Overview

**Objective:** Build core infrastructure for LLM-based trading
**Duration:** 3-4 weeks
**Capital at Risk:** $0 (development only)

### 2.2 Task Breakdown

#### Week 1: Core Agent Framework

| Task | Description | Dependencies | Deliverable |
|------|-------------|--------------|-------------|
| 1.1 | Create `agents/` directory structure | None | Directory structure |
| 1.2 | Implement `BaseAgent` abstract class | 1.1 | `base_agent.py` |
| 1.3 | Implement `TechnicalAgent` | 1.2, Indicator library | `technical_agent.py` |
| 1.4 | Implement `RiskAgent` | 1.2 | `risk_agent.py` |
| 1.5 | Create agent unit tests | 1.3, 1.4 | `tests/test_agents.py` |

**Technical Agent Implementation:**
```python
# Priority indicators to implement
technical_indicators = {
    'trend': ['ema_9', 'ema_21', 'ema_50', 'adx'],
    'momentum': ['rsi_14', 'macd', 'macd_signal', 'roc'],
    'volatility': ['atr_14', 'bollinger_bands', 'keltner_channels'],
    'volume': ['obv', 'vwap', 'volume_ma_ratio']
}
```

#### Week 2: LLM Integration

| Task | Description | Dependencies | Deliverable |
|------|-------------|--------------|-------------|
| 2.1 | Create `llm/` directory structure | None | Directory structure |
| 2.2 | Implement `LLMClient` base class | 2.1 | `client.py` |
| 2.3 | Implement Anthropic provider | 2.2 | `providers/anthropic.py` |
| 2.4 | Implement OpenAI provider | 2.2 | `providers/openai.py` |
| 2.5 | Implement xAI provider (Grok) | 2.2 | `providers/xai.py` |
| 2.6 | Implement Deepseek provider | 2.2 | `providers/deepseek.py` |
| 2.7 | Implement Ollama provider (local) | 2.2 | `providers/ollama.py` |
| 2.8 | Create prompt templates | None | `prompts/` directory |
| 2.9 | Implement output parser | 2.8 | `prompts/output_parsers.py` |
| 2.10 | Create LLM integration tests | 2.3-2.7, 2.9 | `tests/test_llm.py` |

**LLM Provider Priority:**
1. Anthropic (Claude) - Primary
2. xAI (Grok 4) - Comparison
3. Deepseek - Best Alpha Arena performance, cost-efficient
4. OpenAI (GPT-4o) - Backup
5. Ollama (Qwen) - Local cost-free option

#### Week 3: Trading Agent & Coordination

| Task | Description | Dependencies | Deliverable |
|------|-------------|--------------|-------------|
| 3.1 | Implement `TradingAgent` | 2.8, 2.9 | `trading_agent.py` |
| 3.2 | Implement `RebalancingAgent` | 1.2 | `rebalancing_agent.py` |
| 3.3 | Implement `Coordinator` | 1.3, 1.4, 3.1, 3.2 | `coordinator.py` |
| 3.4 | Implement `ModelSelector` | 2.2 | `model_selector.py` |
| 3.5 | Create message bus (Redis) | None | `messaging/bus.py` |
| 3.6 | Implement state manager | 3.5 | `state/manager.py` |
| 3.7 | Integration tests | 3.3, 3.6 | `tests/test_coordination.py` |

#### Week 4: Backtesting & Database

| Task | Description | Dependencies | Deliverable |
|------|-------------|--------------|-------------|
| 4.1 | Extend backtest runner for LLM | Existing backtest | `backtest_runner.py` updates |
| 4.2 | Create trade decision tables | None | SQL migrations |
| 4.3 | Create model performance tables | 4.2 | SQL migrations |
| 4.4 | Implement decision logging | 3.3, 4.2 | `logging/decisions.py` |
| 4.5 | Create Grafana dashboards | 4.3 | Dashboard JSON configs |
| 4.6 | Docker Compose setup | All above | `docker-compose.yml` |
| 4.7 | End-to-end test | All above | `tests/test_e2e.py` |

### 2.3 Phase 1 Milestones

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PHASE 1 MILESTONES                                   │
└─────────────────────────────────────────────────────────────────────────────┘

M1.1: Agent Framework Complete
├── BaseAgent, TechnicalAgent, RiskAgent implemented
├── Unit tests passing
└── Documentation complete

M1.2: LLM Integration Complete
├── All 5 providers working
├── Prompt templates defined
├── Output parsing validated
└── Cost tracking implemented

M1.3: Coordination Complete
├── Coordinator orchestrating all agents
├── Message bus operational
├── State management working
└── Integration tests passing

M1.4: Phase 1 Complete
├── Backtesting with LLM decisions working
├── Database logging operational
├── Monitoring dashboards ready
├── Docker deployment working
└── All tests passing
```

### 2.4 Phase 1 Deliverables

| Deliverable | Description | Acceptance Criteria |
|-------------|-------------|---------------------|
| Agent Framework | All agents implemented | Unit tests pass, <100ms latency |
| LLM Integration | Multi-provider support | All providers respond correctly |
| Coordination Layer | Agent orchestration | Complete trading cycles work |
| Backtest Support | LLM backtesting | Historical data runs complete |
| Monitoring | Grafana dashboards | Key metrics visible |
| Documentation | Technical docs | All components documented |

---

## 3. Phase 2: Paper Trading

### 3.1 Overview

**Objective:** Validate system with paper trading, compare LLM performance
**Duration:** Minimum 30 days
**Capital at Risk:** $0 (paper trading)

### 3.2 Setup Tasks

| Task | Description | Dependencies | Deliverable |
|------|-------------|--------------|-------------|
| 2.1 | Deploy to paper trading environment | Phase 1 complete | Running system |
| 2.2 | Configure all LLM providers | 2.1 | API keys configured |
| 2.3 | Set up parallel LLM comparison | 2.2 | Multi-model running |
| 2.4 | Configure alerting | 2.1 | Telegram/email alerts |
| 2.5 | Create daily reporting | 2.1 | Automated reports |

### 3.3 Paper Trading Configuration

```yaml
# paper_trading_config.yaml

trading:
  enabled: true
  mode: paper
  pairs:
    - BTC/USDT
    - XRP/USDT
    - XRP/BTC

  starting_capital:
    usdt: 1000
    xrp: 500
    btc: 0

  constraints:
    max_leverage: 3
    risk_per_trade_pct: 0.01
    max_drawdown_pct: 0.10
    confidence_threshold: 0.6

models:
  comparison_mode: true  # Run all models in parallel
  active_models:
    - claude-sonnet-4-5
    - gpt-4o
    - grok-4
    - deepseek-v3
    - qwen-2.5-7b

  evaluation_period_days: 7  # Evaluate performance weekly
  auto_switch_enabled: false  # Manual switching during validation

cycle:
  interval_seconds: 60  # Check every minute
  llm_query_interval_seconds: 300  # LLM decisions every 5 minutes
  max_trades_per_day: 10

monitoring:
  log_all_decisions: true
  log_reasoning: true
  alert_on_trade: true
  daily_report_time: "00:00"
```

### 3.4 Validation Schedule

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      30-DAY PAPER TRADING SCHEDULE                           │
└─────────────────────────────────────────────────────────────────────────────┘

Days 1-7: OBSERVATION PERIOD
├── All models running
├── No intervention
├── Collect baseline data
└── Daily metrics review

Days 8-14: FIRST ADJUSTMENT
├── Review week 1 performance
├── Identify prompt improvements
├── Adjust parameters if needed
└── Continue observation

Days 15-21: SECOND ADJUSTMENT
├── Review week 2 performance
├── Model ranking analysis
├── Refine risk parameters
└── Test edge cases

Days 22-30: FINAL VALIDATION
├── Lock configuration
├── No further changes
├── Final performance measurement
└── Go/No-Go decision for Phase 3

Daily Activities:
├── Morning: Review overnight trades
├── Midday: Check system health
├── Evening: Review day's performance
└── Weekly: Comprehensive analysis report
```

### 3.5 Phase 2 Success Metrics

| Metric | Target | Minimum | Measurement |
|--------|--------|---------|-------------|
| **Sharpe Ratio** | > 1.5 | > 1.0 | Risk-adjusted returns |
| **Max Drawdown** | < 10% | < 15% | Peak-to-trough |
| **Win Rate** | > 55% | > 50% | Profitable trades |
| **Profit Factor** | > 1.5 | > 1.2 | Gross profit / loss |
| **System Uptime** | > 99% | > 95% | Availability |
| **Decision Latency** | < 15s | < 30s | LLM response time |
| **Parse Success Rate** | > 98% | > 95% | Valid LLM outputs |

### 3.6 Phase 2 Exit Criteria

**Must Meet ALL:**
- [ ] 30+ days of continuous paper trading
- [ ] Sharpe Ratio >= 1.0
- [ ] Max Drawdown <= 15%
- [ ] Win Rate >= 50%
- [ ] System uptime >= 95%
- [ ] No critical bugs identified
- [ ] Best model clearly identified

**Recommended:**
- [ ] Sharpe Ratio >= 1.5
- [ ] All three assets show growth
- [ ] Model consensus mechanism validated

---

## 4. Phase 3: Micro-Live

### 4.1 Overview

**Objective:** Validate with real money on a small scale
**Duration:** Minimum 30 days
**Capital at Risk:** $100 USDT (5% of total)

### 4.2 Live Trading Configuration

```yaml
# live_trading_config.yaml

trading:
  enabled: true
  mode: live
  exchange: kraken

  capital:
    usdt: 100  # $100 USDT only
    xrp: 0     # No XRP initially
    btc: 0     # No BTC initially

  constraints:
    max_leverage: 2  # Reduced from paper
    risk_per_trade_pct: 0.005  # 0.5% (half of paper)
    max_drawdown_pct: 0.15  # 15% ($15 max loss)
    confidence_threshold: 0.7  # Higher threshold

models:
  comparison_mode: false
  active_model: "best_from_phase_2"  # Single model
  fallback_model: "qwen-2.5-7b"  # Local fallback

cycle:
  interval_seconds: 60
  llm_query_interval_seconds: 600  # 10 minutes (more conservative)
  max_trades_per_day: 5  # Reduced from paper

safety:
  require_human_approval: true  # First week only
  max_single_order_usd: 50
  api_rate_limit_buffer: 0.5
```

### 4.3 Graduated Scale-Up Plan

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PHASE 3 GRADUATION SCHEDULE                             │
└─────────────────────────────────────────────────────────────────────────────┘

Week 1: SUPERVISED ($100)
├── Human approval required for all trades
├── Single model only
├── BTC/USDT pair only
├── Max $50 per trade
└── Daily review mandatory

Week 2: SEMI-AUTOMATED ($100)
├── Human approval for trades > $25
├── Auto-execute small trades
├── Add XRP/USDT pair
├── Increase to 3 trades/day max
└── Bi-daily review

Week 3-4: AUTOMATED ($100)
├── Full automation (with alerts)
├── All three pairs active
├── 5 trades/day max
├── Daily review
└── Prepare for Phase 4 if profitable

Graduation Criteria for Each Week:
├── Positive or near-breakeven P&L
├── No stop-loss failures
├── System stability
└── No unexpected behavior
```

### 4.4 Phase 3 Success Metrics

| Metric | Target | Minimum | Notes |
|--------|--------|---------|-------|
| **Total P&L** | +$5 | +$0 | Must not lose money |
| **Max Drawdown** | < $15 | < $20 | 15-20% of capital |
| **Slippage** | < 0.1% | < 0.2% | Order execution quality |
| **Fill Rate** | > 95% | > 90% | Orders successfully filled |
| **Uptime** | > 99% | > 97% | No missed opportunities |

### 4.5 Phase 3 Exit Criteria

**Must Meet ALL:**
- [ ] 30+ days of live trading
- [ ] Non-negative total P&L
- [ ] No single trade loss > 5% of capital
- [ ] All orders executed successfully
- [ ] No exchange API issues
- [ ] No security incidents

**For Phase 4 Eligibility:**
- [ ] Positive total P&L (any amount)
- [ ] Sharpe Ratio > 0.5 (lower bar for live)
- [ ] Win Rate >= 45%

---

## 5. Phase 4: Scale

### 5.1 Overview

**Objective:** Gradual scale to full capital deployment
**Duration:** Ongoing
**Capital at Risk:** $200 -> $500 -> $1,000 -> $2,100

### 5.2 Capital Deployment Schedule

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PHASE 4 CAPITAL DEPLOYMENT                              │
└─────────────────────────────────────────────────────────────────────────────┘

Stage 4.1: $200 USDT (10% of total)
├── Criteria: Phase 3 profitable
├── Duration: 2 weeks minimum
├── Pairs: All three
├── Leverage: Max 2x
└── Exit: Profitable for 10+ trading days

Stage 4.2: $500 USDT (25% of total)
├── Criteria: Stage 4.1 profitable
├── Duration: 2 weeks minimum
├── Leverage: Max 2x
├── Multi-model comparison resumes
└── Exit: Sharpe > 1.0 for period

Stage 4.3: $1,000 USDT (50% of total)
├── Criteria: Stage 4.2 meets targets
├── Duration: 4 weeks minimum
├── Leverage: Max 3x available
├── Auto model switching enabled
└── Exit: Sharpe > 1.2, max DD < 15%

Stage 4.4: Full Capital ($2,100)
├── Criteria: Stage 4.3 sustained performance
├── Duration: Ongoing
├── All features enabled
├── Full risk budget available
└── Monthly performance review

Rollback Rules:
├── Any stage: 10% drawdown -> return to previous stage
├── Any stage: 3 consecutive losing weeks -> pause and review
└── Full capital: 15% drawdown -> reduce to 50% capital
```

### 5.3 Phase 4 Features

| Stage | Multi-Model | Auto-Switch | Rebalancing | Leverage |
|-------|-------------|-------------|-------------|----------|
| 4.1 | No | No | No | 2x |
| 4.2 | Yes (comparison) | No | No | 2x |
| 4.3 | Yes | Yes | Yes | 3x |
| 4.4 | Yes | Yes | Yes | 3x |

### 5.4 Continuous Improvement Process

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CONTINUOUS IMPROVEMENT CYCLE                              │
└─────────────────────────────────────────────────────────────────────────────┘

Weekly Activities:
├── Performance review
├── Model comparison analysis
├── Prompt refinement candidates
└── Risk parameter review

Monthly Activities:
├── Full backtest with new data
├── Strategy optimization round
├── Infrastructure review
└── Documentation update

Quarterly Activities:
├── Major version upgrade consideration
├── New model evaluation
├── Architecture review
└── Goals reassessment
```

---

## 6. Dependency Graph

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DEPENDENCY GRAPH                                     │
└─────────────────────────────────────────────────────────────────────────────┘

PHASE 1: FOUNDATION
───────────────────

[1.1 Directory Structure]
         │
         ├──────────────────────┐
         │                      │
         ▼                      ▼
[1.2 BaseAgent]          [2.1 LLM Directory]
         │                      │
    ┌────┴────┐            ┌────┴────┐
    │         │            │         │
    ▼         ▼            ▼         ▼
[1.3 Tech] [1.4 Risk]  [2.2 Client] [2.7 Prompts]
    │         │            │         │
    │         │       ┌────┴────┐    │
    │         │       │    │    │    │
    │         │       ▼    ▼    ▼    ▼    ▼
    │         │    [2.3] [2.4] [2.5] [2.6] [2.9 Parser]
    │         │   Claude GPT  Grok Deep       │
    │         │                          │
    │         └──────┬───────────────────┘
    │                │
    │                ▼
    │         [3.1 TradingAgent]
    │                │
    └────────────────┤
                     │
                     ▼
              [3.3 Coordinator]
                     │
         ┌───────────┼───────────┐
         │           │           │
         ▼           ▼           ▼
    [3.5 Bus]   [3.6 State]  [3.4 Selector]
         │           │           │
         └───────────┴───────────┘
                     │
                     ▼
              [4.1 Backtest]
                     │
                     ▼
              [4.6 Docker]
                     │
                     ▼
            [PHASE 1 COMPLETE]


PHASE 2: PAPER TRADING
──────────────────────

[PHASE 1 COMPLETE]
         │
         ▼
[2.1 Deploy Paper Env]
         │
         ├──────────────┐
         │              │
         ▼              ▼
[2.2 Configure]  [2.3 Multi-Model]
         │              │
         └──────┬───────┘
                │
                ▼
         [2.4 Alerting]
                │
                ▼
         [2.5 Reporting]
                │
                ▼
         [30 DAYS PAPER]
                │
                ▼
         [METRICS MET?]
                │
         YES────┼────NO
         │             │
         ▼             ▼
[PHASE 2 COMPLETE]  [ITERATE]


PHASE 3: MICRO-LIVE
───────────────────

[PHASE 2 COMPLETE]
         │
         ▼
[3.1 Live Config]
         │
         ▼
[3.2 API Keys Live]
         │
         ▼
[3.3 Safety Checks]
         │
         ▼
[WEEK 1: SUPERVISED]
         │
         ▼
[WEEK 2: SEMI-AUTO]
         │
         ▼
[WEEK 3-4: AUTO]
         │
         ▼
[PROFITABLE?]
         │
   YES───┼───NO
   │           │
   ▼           ▼
[PHASE 3]  [RETURN TO]
[COMPLETE] [PHASE 2]


PHASE 4: SCALE
──────────────

[PHASE 3 COMPLETE]
         │
         ▼
    [$200 Stage]──── DD>10%? ──▶ [Back to $100]
         │
         ▼
    [$500 Stage]──── DD>10%? ──▶ [Back to $200]
         │
         ▼
   [$1000 Stage]──── DD>10%? ──▶ [Back to $500]
         │
         ▼
   [FULL $2100]──── DD>15%? ──▶ [Back to $1000]
         │
         ▼
   [ONGOING OPS]
```

---

## 7. Risk Checkpoints

### 7.1 Phase 1 Risk Checkpoint

**Before Proceeding to Phase 2:**

- [ ] **Code Quality**
  - All tests passing
  - No critical bugs
  - Error handling complete
  - Logging comprehensive

- [ ] **Security**
  - API keys stored securely (environment variables)
  - No hardcoded credentials
  - Rate limiting implemented
  - Input validation complete

- [ ] **Operations**
  - Docker deployment tested
  - Monitoring dashboards functional
  - Alert system working
  - Backup/restore tested

### 7.2 Phase 2 Risk Checkpoint

**Before Proceeding to Phase 3:**

- [ ] **Performance Validation**
  - Sharpe Ratio >= 1.0
  - Max Drawdown <= 15%
  - Win Rate >= 50%
  - Positive expectancy

- [ ] **System Stability**
  - Uptime >= 95%
  - No data loss events
  - No critical errors in 7+ days
  - LLM parse success >= 95%

- [ ] **Strategy Validation**
  - Best model clearly identified
  - Risk rules functioning correctly
  - Position sizing accurate
  - Stop-losses executing properly

### 7.3 Phase 3 Risk Checkpoint

**Before Proceeding to Phase 4:**

- [ ] **Live Trading Validation**
  - Non-negative P&L
  - All orders executed correctly
  - Slippage within acceptable range
  - No API errors

- [ ] **Exchange Integration**
  - Order types working (limit, market, stop)
  - Balance tracking accurate
  - Position tracking accurate
  - Fee calculation correct

- [ ] **Operational Readiness**
  - 24/7 monitoring in place
  - Incident response plan documented
  - Manual intervention procedures tested
  - Emergency stop mechanism working

### 7.4 Phase 4 Risk Checkpoint (Per Stage)

**Before Each Capital Increase:**

- [ ] Previous stage profitable for minimum period
- [ ] No drawdown exceeding threshold
- [ ] System stability maintained
- [ ] No new bugs introduced
- [ ] Documentation current
- [ ] Backup trading account tested

---

## Appendix: Quick Reference Timeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        IMPLEMENTATION TIMELINE                               │
└─────────────────────────────────────────────────────────────────────────────┘

Week 1-4:   PHASE 1 (Foundation)
            └── Build core infrastructure

Week 5-8:   PHASE 2 (Paper Trading)
            └── 30 days minimum paper trading

Week 9-12:  PHASE 3 (Micro-Live)
            └── $100 live trading

Week 13+:   PHASE 4 (Scale)
            └── Gradual capital increase

Total Minimum: 13 weeks (3 months) to full deployment
Recommended: 16-20 weeks with buffer for iteration
```

---

*Document Version: 1.0*
*Last Updated: December 2025*
