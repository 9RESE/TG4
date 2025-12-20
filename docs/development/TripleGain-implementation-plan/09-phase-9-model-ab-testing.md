# Phase 9: 6-Model A/B Testing Framework

**Phase Status**: Ready to Start
**Dependencies**: Phase 2 (Trading Decision Agent), Phase 3 (Execution), Phase 6 (Paper Trading)
**Deliverable**: Comprehensive framework for tracking and comparing performance across all 6 LLM models

---

## Overview

The 6-Model A/B Testing Framework provides comprehensive performance tracking and comparison across all LLM models used for trading decisions. This enables data-driven model selection and consensus-based decision making.

### Why This Phase Matters

| Benefit | Description |
|---------|-------------|
| Model Selection | Identify best-performing models for different scenarios |
| Consensus Power | Quantify accuracy improvement from model agreement |
| Cost Optimization | Balance accuracy vs API costs |
| Continuous Improvement | Track model performance over time |
| Transparency | Full auditability of all model decisions |

### Models Under Test

| Model | Provider | Tier | Primary Use |
|-------|----------|------|-------------|
| GPT (latest) | OpenAI | API | Trading decisions |
| Grok (latest) | xAI | API | Trading + Sentiment |
| DeepSeek V3 | DeepSeek | API | Trading + Coordination |
| Claude Sonnet | Anthropic | API | Trading + Coordination |
| Claude Opus | Anthropic | API | Trading decisions |
| Qwen 2.5 7B | Ollama | Local | TA, Regime, Trading |

---

## 9.1 Architecture

### Framework Design

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    6-MODEL A/B TESTING FRAMEWORK                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  MODELS UNDER TEST:                                                         │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┬─────────────┐   │
│  │ GPT         │ Grok        │ DeepSeek V3 │Claude Sonnet│Claude Opus  │   │
│  │ (OpenAI)    │ (xAI)       │ (DeepSeek)  │ (Anthropic) │ (Anthropic) │   │
│  └─────────────┴─────────────┴─────────────┴─────────────┴─────────────┘   │
│  ┌─────────────┐                                                           │
│  │ Qwen 2.5 7B │                                                           │
│  │ (Local)     │                                                           │
│  └─────────────┘                                                           │
│                                                                             │
│  TRACKING DIMENSIONS:                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 1. Decision Accuracy                                                 │   │
│  │    - Correct BUY/SELL/HOLD predictions (1h, 4h, 24h horizons)       │   │
│  │    - By market regime                                                │   │
│  │    - By symbol                                                       │   │
│  │                                                                      │   │
│  │ 2. Profitability                                                     │   │
│  │    - Simulated P&L if model's decision was executed alone           │   │
│  │    - Contribution to consensus decisions                             │   │
│  │    - Risk-adjusted returns (Sharpe)                                  │   │
│  │                                                                      │   │
│  │ 3. Confidence Calibration                                            │   │
│  │    - Does confidence match actual accuracy?                          │   │
│  │    - Expected Calibration Error (ECE)                               │   │
│  │                                                                      │   │
│  │ 4. Operational Metrics                                               │   │
│  │    - Latency (p50, p95, p99)                                        │   │
│  │    - Cost per decision                                               │   │
│  │    - Error rate                                                      │   │
│  │    - Timeout rate                                                    │   │
│  │                                                                      │   │
│  │ 5. Consensus Contribution                                            │   │
│  │    - How often on winning side of consensus                          │   │
│  │    - How often selected as parameter source                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  LEADERBOARD UPDATE:                                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Composite Score = (Accuracy × 0.35) + (Profit × 0.25) +             │   │
│  │                   (Calibration × 0.20) + (Latency × 0.10) +         │   │
│  │                   (Cost × 0.10)                                      │   │
│  │                                                                      │   │
│  │ Updated: Hourly                                                      │   │
│  │ Statistical Significance: Pairwise t-tests (p < 0.05)               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MODEL COMPARISON DATA FLOW                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. DECISION COLLECTION                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Trading Decision Agent runs all 6 models in parallel:               │   │
│  │                                                                      │   │
│  │   [Same Market Snapshot] ──→ GPT ──→ Decision + Confidence          │   │
│  │                          ──→ Grok ──→ Decision + Confidence         │   │
│  │                          ──→ DeepSeek ──→ Decision + Confidence     │   │
│  │                          ──→ Claude Sonnet ──→ Decision + Conf.     │   │
│  │                          ──→ Claude Opus ──→ Decision + Confidence  │   │
│  │                          ──→ Qwen ──→ Decision + Confidence         │   │
│  │                                                                      │   │
│  │ Each decision recorded with:                                        │   │
│  │   • Timestamp, symbol, action, confidence                           │   │
│  │   • Latency, cost, error status                                     │   │
│  │   • Market snapshot hash (for reproducibility)                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  2. OUTCOME TRACKING                                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ After 1h, 4h, 24h - update decision with actual outcome:            │   │
│  │                                                                      │   │
│  │   price_change_1h: +1.2%                                            │   │
│  │   price_change_4h: +2.8%                                            │   │
│  │   price_change_24h: +4.1%                                           │   │
│  │                                                                      │   │
│  │   outcome_1h: CORRECT (BUY decision, price increased)              │   │
│  │   outcome_4h: CORRECT                                               │   │
│  │   outcome_24h: CORRECT                                              │   │
│  │                                                                      │   │
│  │   simulated_pnl_1h: +$12.50 (if executed alone)                    │   │
│  │   simulated_pnl_4h: +$28.00                                        │   │
│  │   simulated_pnl_24h: +$41.00                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  3. METRICS CALCULATION (Hourly)                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ For each model, calculate over last 24h/7d/30d:                     │   │
│  │   • Accuracy (total, by action type, by regime)                     │   │
│  │   • Simulated P&L and Sharpe                                        │   │
│  │   • ECE (Expected Calibration Error)                                │   │
│  │   • Latency percentiles                                              │   │
│  │   • Total cost and cost per correct decision                        │   │
│  │   • Consensus win rate                                               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  4. LEADERBOARD UPDATE                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Calculate composite scores and rank models:                          │   │
│  │                                                                      │   │
│  │   1. DeepSeek V3    [0.78] ★                                        │   │
│  │   2. Claude Opus    [0.72]                                          │   │
│  │   3. Grok           [0.70]                                          │   │
│  │   4. Claude Sonnet  [0.68]                                          │   │
│  │   5. GPT            [0.62]                                          │   │
│  │   6. Qwen 2.5 7B    [0.58]                                          │   │
│  │                                                                      │   │
│  │ Run pairwise significance tests (15 pairs for 6 models)             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 9.2 Data Structures

### Database Schema

```sql
-- Individual model decisions (one row per model per decision event)
CREATE TABLE model_decisions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    decision_event_id UUID NOT NULL,  -- Groups all 6 model responses
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model_id VARCHAR(20) NOT NULL,  -- gpt, grok, deepseek, claude_sonnet, claude_opus, qwen
    symbol VARCHAR(20) NOT NULL,

    -- Decision output
    action VARCHAR(10) NOT NULL,  -- BUY, SELL, HOLD
    confidence DECIMAL(5, 4) NOT NULL,
    entry_price DECIMAL(20, 10),
    stop_loss_pct DECIMAL(5, 2),
    take_profit_pct DECIMAL(5, 2),
    leverage INTEGER,
    reasoning TEXT,

    -- Operational metrics
    latency_ms INTEGER NOT NULL,
    cost_usd DECIMAL(10, 6),
    error BOOLEAN DEFAULT FALSE,
    error_message TEXT,
    timeout BOOLEAN DEFAULT FALSE,

    -- Market context (for reproducibility)
    market_snapshot_hash VARCHAR(64),

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_model_decisions_event
    ON model_decisions (decision_event_id);

CREATE INDEX idx_model_decisions_model_ts
    ON model_decisions (model_id, timestamp DESC);

CREATE INDEX idx_model_decisions_symbol_ts
    ON model_decisions (symbol, timestamp DESC);

-- Decision outcomes (updated after 1h, 4h, 24h)
CREATE TABLE decision_outcomes (
    decision_id UUID PRIMARY KEY REFERENCES model_decisions(id),

    -- Price changes
    price_at_decision DECIMAL(20, 10) NOT NULL,
    price_after_1h DECIMAL(20, 10),
    price_after_4h DECIMAL(20, 10),
    price_after_24h DECIMAL(20, 10),

    price_change_1h_pct DECIMAL(10, 4),
    price_change_4h_pct DECIMAL(10, 4),
    price_change_24h_pct DECIMAL(10, 4),

    -- Correctness determination
    outcome_1h VARCHAR(10),  -- CORRECT, INCORRECT, PENDING
    outcome_4h VARCHAR(10),
    outcome_24h VARCHAR(10),

    -- Simulated P&L (if this decision was executed alone)
    simulated_pnl_1h DECIMAL(20, 2),
    simulated_pnl_4h DECIMAL(20, 2),
    simulated_pnl_24h DECIMAL(20, 2),

    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Model performance snapshots (updated hourly)
CREATE TABLE model_performance (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model_id VARCHAR(20) NOT NULL,
    period VARCHAR(10) NOT NULL,  -- 24h, 7d, 30d

    -- Decision metrics
    total_decisions INTEGER NOT NULL DEFAULT 0,
    correct_decisions INTEGER NOT NULL DEFAULT 0,
    accuracy_pct DECIMAL(5, 2),

    -- By decision type
    buy_count INTEGER DEFAULT 0,
    buy_correct INTEGER DEFAULT 0,
    buy_accuracy_pct DECIMAL(5, 2),

    sell_count INTEGER DEFAULT 0,
    sell_correct INTEGER DEFAULT 0,
    sell_accuracy_pct DECIMAL(5, 2),

    hold_count INTEGER DEFAULT 0,
    hold_correct INTEGER DEFAULT 0,
    hold_accuracy_pct DECIMAL(5, 2),

    -- Profitability
    simulated_pnl_usd DECIMAL(20, 2),
    simulated_pnl_pct DECIMAL(10, 4),
    sharpe_ratio DECIMAL(10, 4),

    -- Calibration
    expected_calibration_error DECIMAL(5, 4),
    is_well_calibrated BOOLEAN,

    -- Operational
    avg_latency_ms INTEGER,
    p50_latency_ms INTEGER,
    p95_latency_ms INTEGER,
    p99_latency_ms INTEGER,
    total_cost_usd DECIMAL(10, 4),
    cost_per_decision_usd DECIMAL(10, 6),
    cost_per_correct_usd DECIMAL(10, 6),
    error_count INTEGER DEFAULT 0,
    error_rate_pct DECIMAL(5, 2),
    timeout_count INTEGER DEFAULT 0,

    -- Consensus contribution
    consensus_agreement_count INTEGER DEFAULT 0,
    consensus_win_count INTEGER DEFAULT 0,
    consensus_win_rate_pct DECIMAL(5, 2),
    times_selected_for_params INTEGER DEFAULT 0,

    -- Composite score
    composite_score DECIMAL(5, 4),

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_model_performance_model_ts
    ON model_performance (model_id, timestamp DESC);

CREATE INDEX idx_model_performance_period
    ON model_performance (period, timestamp DESC);

-- Model leaderboard (latest rankings)
CREATE TABLE model_leaderboard (
    model_id VARCHAR(20) PRIMARY KEY,
    rank INTEGER NOT NULL,
    composite_score DECIMAL(5, 4) NOT NULL,
    accuracy_pct DECIMAL(5, 2),
    profit_pct DECIMAL(10, 4),
    calibration_score DECIMAL(5, 4),
    latency_score DECIMAL(5, 4),
    cost_efficiency DECIMAL(10, 4),
    best_at VARCHAR(50),  -- What this model excels at
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Pairwise significance tests (15 pairs for 6 models)
CREATE TABLE model_pairwise_tests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model_a VARCHAR(20) NOT NULL,
    model_b VARCHAR(20) NOT NULL,
    metric VARCHAR(50) NOT NULL,  -- pnl, accuracy, etc.
    period VARCHAR(10) NOT NULL,  -- 7d, 30d

    -- Test results
    mean_a DECIMAL(10, 4),
    mean_b DECIMAL(10, 4),
    std_a DECIMAL(10, 4),
    std_b DECIMAL(10, 4),
    t_statistic DECIMAL(10, 4),
    p_value DECIMAL(10, 6),
    is_significant BOOLEAN,
    winner VARCHAR(20),
    sample_size INTEGER
);

CREATE INDEX idx_pairwise_tests_ts
    ON model_pairwise_tests (timestamp DESC);

-- Consensus analysis
CREATE TABLE consensus_analysis (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    decision_event_id UUID NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Agreement level
    agreement_level INTEGER NOT NULL,  -- 3-6 models agree
    consensus_action VARCHAR(10),  -- BUY, SELL, HOLD, SPLIT
    agreeing_models TEXT[],  -- Array of model IDs

    -- Outcome
    outcome_correct BOOLEAN,
    price_change_pct DECIMAL(10, 4),

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_consensus_agreement
    ON consensus_analysis (agreement_level, timestamp DESC);
```

### Data Classes

| Class | Purpose | Key Fields |
|-------|---------|------------|
| `ModelMetrics` | Comprehensive metrics | accuracy, profit, calibration, latency, cost, consensus |
| `LeaderboardEntry` | Single leaderboard row | rank, model_id, composite_score, best_at |
| `PairwiseTest` | Significance test result | model_a, model_b, t_stat, p_value, winner |
| `ConsensusAnalysis` | Consensus pattern | agreement_level, action, models, outcome |
| `DecisionRecord` | Single model decision | model, action, confidence, latency, cost |

---

## 9.3 Component Details

### 9.3.1 ModelComparisonFramework

**File**: `triplegain/src/llm/model_comparison.py`

**Responsibilities**:
- Record all model decisions with metadata
- Update outcomes after time horizons
- Calculate comprehensive metrics
- Maintain leaderboard rankings
- Run pairwise significance tests
- Analyze consensus patterns

**Key Methods**:

| Method | Purpose | Parameters | Returns |
|--------|---------|------------|---------|
| `record_decision()` | Store model decision | model_id, decision, latency, cost | decision_id |
| `update_outcome()` | Update with actual outcome | decision_id, prices | bool |
| `calculate_metrics()` | Calculate model metrics | model_id, period | ModelMetrics |
| `update_leaderboard()` | Update all rankings | - | list[LeaderboardEntry] |
| `run_pairwise_tests()` | Statistical comparison | metric | list[PairwiseTest] |
| `get_consensus_stats()` | Analyze consensus | period | dict |
| `get_model_rank()` | Get single model rank | model_id | LeaderboardEntry |

### 9.3.2 Outcome Tracker

**File**: `triplegain/src/llm/outcome_tracker.py`

**Responsibilities**:
- Schedule outcome updates at 1h, 4h, 24h after decision
- Fetch actual prices at each horizon
- Determine correctness based on action vs price change
- Calculate simulated P&L

**Correctness Logic**:

| Action | Price Change | Outcome |
|--------|--------------|---------|
| BUY | > 0% | CORRECT |
| BUY | <= 0% | INCORRECT |
| SELL | < 0% | CORRECT |
| SELL | >= 0% | INCORRECT |
| HOLD | abs < 1% | CORRECT |
| HOLD | abs >= 1% | INCORRECT |

### 9.3.3 Consensus Calculator

**File**: `triplegain/src/llm/consensus.py`

**Consensus Rules**:

| Agreement | Threshold | Action |
|-----------|-----------|--------|
| Unanimous | 6/6 | Execute with confidence boost +15% |
| Strong Majority | 5/6 | Execute with confidence boost +10% |
| Majority | 4/6 | Execute with confidence boost +5% |
| Split | 3/6 or less | Defer or hold |

**Tie-Breaking**:
When split, use decision from highest-ranked model on leaderboard.

### 9.3.4 ECE Calculator

**Expected Calibration Error** measures how well confidence matches accuracy.

```python
def calculate_ece(decisions: list, n_bins: int = 10) -> float:
    """
    Calculate Expected Calibration Error.

    Perfect calibration: ECE = 0
    Well calibrated: ECE < 0.10
    Poor calibration: ECE > 0.20
    """
    bins = [[] for _ in range(n_bins)]

    for d in decisions:
        if d.outcome == "PENDING":
            continue
        bin_idx = min(int(d.confidence * n_bins), n_bins - 1)
        bins[bin_idx].append(d)

    ece = 0
    total = sum(len(b) for b in bins)

    for i, bin_decisions in enumerate(bins):
        if not bin_decisions:
            continue

        expected_conf = (i + 0.5) / n_bins
        actual_accuracy = sum(
            1 for d in bin_decisions if d.outcome == "CORRECT"
        ) / len(bin_decisions)

        ece += abs(expected_conf - actual_accuracy) * len(bin_decisions) / total

    return ece
```

---

## 9.4 Configuration

**File**: `config/model_comparison.yaml`

```yaml
model_comparison:
  enabled: true

  models:
    - id: gpt
      name: "GPT (latest)"
      provider: openai
      model: gpt-4-turbo
      enabled: true

    - id: grok
      name: "Grok (latest)"
      provider: xai
      model: grok-2
      enabled: true

    - id: deepseek
      name: "DeepSeek V3"
      provider: deepseek
      model: deepseek-v3
      enabled: true

    - id: claude_sonnet
      name: "Claude Sonnet"
      provider: anthropic
      model: claude-sonnet-4-20250514
      enabled: true

    - id: claude_opus
      name: "Claude Opus"
      provider: anthropic
      model: claude-opus-4-5-20251101
      enabled: true

    - id: qwen
      name: "Qwen 2.5 7B"
      provider: ollama
      model: qwen2.5:7b
      enabled: true

  # Composite score weights
  composite_weights:
    accuracy: 0.35
    profitability: 0.25
    calibration: 0.20
    latency: 0.10
    cost: 0.10

  # Update frequencies
  updates:
    metrics_interval_seconds: 3600  # Hourly
    leaderboard_interval_seconds: 3600  # Hourly
    pairwise_tests_interval_seconds: 86400  # Daily
    outcome_check_interval_seconds: 300  # 5 min

  # Outcome evaluation horizons
  horizons:
    - 1h
    - 4h
    - 24h

  # Primary evaluation horizon
  primary_horizon: 4h

  # Significance testing
  significance:
    p_value_threshold: 0.05
    min_sample_size: 30

  # Consensus rules
  consensus:
    unanimous_threshold: 6
    strong_majority_threshold: 5
    majority_threshold: 4
    confidence_boosts:
      unanimous: 0.15
      strong_majority: 0.10
      majority: 0.05

  # Periods for analysis
  analysis_periods:
    - 24h
    - 7d
    - 30d
```

---

## 9.5 API Endpoints

### New Routes

| Method | Path | Description | Response |
|--------|------|-------------|----------|
| GET | `/api/v1/models/leaderboard` | Current rankings | list[LeaderboardEntry] |
| GET | `/api/v1/models/{id}/metrics` | Single model metrics | ModelMetrics |
| GET | `/api/v1/models/comparison` | All models comparison | ComparisonReport |
| GET | `/api/v1/models/pairwise` | Significance tests | list[PairwiseTest] |
| GET | `/api/v1/models/consensus` | Consensus analysis | ConsensusStats |
| GET | `/api/v1/models/{id}/decisions` | Model decision history | list[DecisionRecord] |
| GET | `/api/v1/models/best-at/{action}` | Best model for action | LeaderboardEntry |

### Route Implementation

**File**: `triplegain/src/api/routes_models.py`

---

## 9.6 Integration with Trading Decision

### Enhanced Trading Decision Agent

Update `trading_decision.py` to record all model responses:

```python
async def process(self, snapshot, ta_output, regime_output, sentiment_output=None):
    """Run all 6 models and determine consensus."""

    # Generate unique event ID
    decision_event_id = str(uuid.uuid4())

    # Run all models in parallel
    tasks = []
    for model_id in self.model_ids:
        tasks.append(self._run_single_model(
            model_id=model_id,
            snapshot=snapshot,
            context={...}
        ))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Record all decisions
    for model_id, result in zip(self.model_ids, results):
        await self.model_comparison.record_decision(
            decision_event_id=decision_event_id,
            model_id=model_id,
            decision=result.decision if not isinstance(result, Exception) else None,
            latency_ms=result.latency_ms if not isinstance(result, Exception) else 0,
            cost_usd=result.cost_usd if not isinstance(result, Exception) else 0,
            error=isinstance(result, Exception),
            error_message=str(result) if isinstance(result, Exception) else None
        )

    # Calculate consensus
    valid_results = [r for r in results if not isinstance(r, Exception)]
    consensus = self._calculate_consensus(valid_results)

    # Record consensus
    await self.model_comparison.record_consensus(
        decision_event_id=decision_event_id,
        consensus=consensus
    )

    return consensus.decision
```

---

## 9.7 Reporting

### Model Comparison Dashboard

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 6-MODEL COMPARISON REPORT                                                    │
│ Period: Last 7 Days                                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ LEADERBOARD                                                                  │
│                                                                             │
│ Rank │ Model         │ Accuracy │ Profit  │ Sharpe │ Calibr. │ Score       │
│ ─────┼───────────────┼──────────┼─────────┼────────┼─────────┼─────────────│
│ 1    │ DeepSeek V3   │ 62.0%    │ +8.5%   │ 1.80   │ 0.88    │ 0.78  ★     │
│ 2    │ Claude Opus   │ 60.0%    │ +7.2%   │ 1.60   │ 0.85    │ 0.72        │
│ 3    │ Grok          │ 58.0%    │ +6.8%   │ 1.50   │ 0.82    │ 0.70        │
│ 4    │ Claude Sonnet │ 57.0%    │ +5.5%   │ 1.40   │ 0.80    │ 0.68        │
│ 5    │ GPT           │ 55.0%    │ +4.2%   │ 1.20   │ 0.75    │ 0.62        │
│ 6    │ Qwen 2.5 7B   │ 52.0%    │ +3.1%   │ 1.00   │ 0.70    │ 0.58        │
│                                                                             │
│ PAIRWISE SIGNIFICANCE (15 comparisons)                                      │
│                                                                             │
│ Significant differences (p < 0.05):                                         │
│ • DeepSeek > GPT (p=0.012)                                                 │
│ • DeepSeek > Qwen (p=0.003)                                                │
│ • Claude Opus > Qwen (p=0.018)                                             │
│                                                                             │
│ Not significant (need more data):                                           │
│ • DeepSeek vs Claude Opus (p=0.15)                                         │
│ • DeepSeek vs Grok (p=0.22)                                                │
│                                                                             │
│ CONSENSUS ANALYSIS                                                          │
│                                                                             │
│ Agreement │ Count │ Accuracy │ Insight                                      │
│ ──────────┼───────┼──────────┼───────────────────────────────────────────── │
│ 6/6       │ 85    │ 75%      │ Strong signal - highest accuracy            │
│ 5/6       │ 120   │ 68%      │ Reliable - execute with confidence          │
│ 4/6       │ 160   │ 62%      │ Moderate - check threshold                  │
│ ≤3/6      │ 67    │ 48%      │ Uncertain - defer to top performer          │
│                                                                             │
│ BEST MODEL BY DECISION TYPE                                                 │
│ • BUY decisions: DeepSeek V3 (65% accuracy)                                │
│ • SELL decisions: Claude Opus (62% accuracy)                               │
│ • HOLD decisions: Grok (70% accuracy)                                      │
│                                                                             │
│ COST ANALYSIS                                                               │
│ Total API cost: $45.85 | Cost per correct decision: $0.18                  │
│ Most cost-effective: DeepSeek V3 ($0.08/correct)                           │
│ Qwen 2.5 7B: $0 (local) but lowest accuracy                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 9.8 Test Requirements

### Unit Tests

| Test | Description | Acceptance Criteria |
|------|-------------|---------------------|
| `test_decision_recording` | Decisions recorded correctly | All fields stored |
| `test_outcome_update` | Outcomes updated correctly | Correct determination |
| `test_accuracy_calculation` | Accuracy computed correctly | Matches manual calc |
| `test_ece_calculation` | ECE computed correctly | Validated algorithm |
| `test_composite_score` | Weighted correctly | Weights sum to 1.0 |
| `test_leaderboard_ranking` | Correct order | Highest score = rank 1 |
| `test_pairwise_tests` | Correct p-values | Verified with scipy |
| `test_consensus_detection` | Correct agreement level | 6/6, 5/6, etc. |
| `test_simulated_pnl` | P&L calculated correctly | Matches manual calc |

### Integration Tests

| Test | Description | Acceptance Criteria |
|------|-------------|---------------------|
| `test_parallel_model_execution` | All 6 run in parallel | Reasonable latency |
| `test_outcome_scheduler` | Outcomes updated on schedule | 1h, 4h, 24h |
| `test_leaderboard_persistence` | Persists to database | Survives restart |
| `test_api_endpoints` | REST endpoints work | Valid responses |
| `test_consensus_execution` | Consensus drives trading | Correct action taken |

---

## 9.9 Deliverables Checklist

- [ ] `triplegain/src/llm/model_comparison.py` - Core framework
- [ ] `triplegain/src/llm/outcome_tracker.py` - Outcome tracking
- [ ] `triplegain/src/llm/consensus.py` - Consensus calculation
- [ ] `triplegain/src/api/routes_models.py` - API endpoints
- [ ] `config/model_comparison.yaml` - Configuration
- [ ] `migrations/009_model_comparison.sql` - Database migration
- [ ] `triplegain/tests/unit/llm/test_model_comparison.py` - Unit tests
- [ ] `triplegain/tests/integration/test_model_integration.py` - Integration tests
- [ ] Update `trading_decision.py` to record all model decisions
- [ ] Update coordinator to run outcome tracker

---

## 9.10 Acceptance Criteria

### Functional Requirements

| Requirement | Test Method | Acceptance |
|-------------|-------------|------------|
| All 6 models recorded | Integration test | All decisions stored |
| Outcomes tracked | Scheduler test | Updates at horizons |
| Metrics accurate | Unit test | Matches manual calc |
| Leaderboard correct | Unit test | Proper ranking |
| Significance tests work | Unit test | Valid p-values |

### Non-Functional Requirements

| Requirement | Target | Measurement |
|-------------|--------|-------------|
| Recording latency | < 10ms | After model response |
| Metrics calculation | < 5s | Hourly update |
| Leaderboard update | < 2s | Hourly |
| API response | < 100ms | Endpoint timing |

---

## References

- Design: [06-evaluation-framework.md](../TripleGain-master-design/06-evaluation-framework.md)
- Design: [02-llm-integration-system.md](../TripleGain-master-design/02-llm-integration-system.md)
- Existing: [trading_decision.py](../../../triplegain/src/agents/trading_decision.py)
- Phase 2: [02-phase-2-core-agents.md](./02-phase-2-core-agents.md)

---

*Phase 9 Implementation Plan v1.0 - December 2025*
