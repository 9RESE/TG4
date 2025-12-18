# TripleGain Evaluation Framework

**Document Version**: 1.0
**Status**: Design Phase

---

## 1. Evaluation Framework Overview

### 1.1 Evaluation Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TRIPLEGAIN EVALUATION FRAMEWORK                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  LEVEL 1: SYSTEM PERFORMANCE                                                │
│  ├── Overall portfolio returns vs benchmarks                                │
│  ├── Risk-adjusted metrics (Sharpe, Sortino, Calmar)                       │
│  └── Success metrics from vision document                                   │
│                                                                             │
│  LEVEL 2: STRATEGY EFFECTIVENESS                                            │
│  ├── Per-strategy win rate and P&L                                         │
│  ├── Regime-strategy alignment                                              │
│  └── Signal quality assessment                                              │
│                                                                             │
│  LEVEL 3: AGENT PERFORMANCE                                                 │
│  ├── Decision accuracy per agent                                            │
│  ├── Confidence calibration                                                 │
│  └── Latency and reliability                                                │
│                                                                             │
│  LEVEL 4: LLM MODEL COMPARISON                                              │
│  ├── Model accuracy comparison                                              │
│  ├── Cost-effectiveness analysis                                            │
│  └── A/B test results                                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Success Metrics (From Vision)

### 2.1 Primary Success Criteria

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **Annual Return** | > 50% | (Final Equity - Initial Equity) / Initial Equity |
| **Maximum Drawdown** | < 20% | Max(Peak - Trough) / Peak |
| **Sharpe Ratio** | > 1.5 | (Annualized Return - Risk-Free) / Annualized Volatility |
| **Portfolio Balance** | 33/33/33 | Within 5% deviation from target |
| **Hodl Bag Growth** | Positive | Net increase in BTC + XRP hodl holdings |
| **System Uptime** | > 99% | Total uptime / Total time |

### 2.2 Secondary Success Criteria

| Metric | Target | Purpose |
|--------|--------|---------|
| Win Rate | > 50% | Signal quality validation |
| Profit Factor | > 1.5 | Gross profit / Gross loss |
| Sortino Ratio | > 2.0 | Downside risk-adjusted returns |
| Calmar Ratio | > 2.0 | Return / Max drawdown |
| Average Trade Duration | 4-48 hours | MLFT style validation |
| Trade Frequency | 1-5 per day | Quality over quantity |

---

## 3. Benchmark Comparisons

### 3.1 Benchmark Definitions

| Benchmark | Description | Purpose |
|-----------|-------------|---------|
| **Buy-and-Hold BTC** | Hold 100% BTC from start | Market beta comparison |
| **Buy-and-Hold XRP** | Hold 100% XRP from start | Market beta comparison |
| **Equal Weight Hold** | 33/33/33 BTC/XRP/USDT rebalanced monthly | Passive allocation comparison |
| **Simple EMA Crossover** | EMA(9) crosses EMA(21) on 4H | Basic strategy baseline |
| **Random Walk** | Random buy/sell signals | Skill vs luck test |

### 3.2 Benchmark Calculation

```python
class BenchmarkCalculator:
    """
    Calculate benchmark returns for comparison.
    """

    def __init__(self, price_data: dict, initial_equity: float):
        self.prices = price_data
        self.initial_equity = initial_equity

    def buy_and_hold_btc(self) -> dict:
        """Calculate buy-and-hold BTC returns."""
        start_price = self.prices["BTC"][0]["close"]
        end_price = self.prices["BTC"][-1]["close"]

        btc_amount = self.initial_equity / start_price
        final_value = btc_amount * end_price

        return {
            "strategy": "Buy-and-Hold BTC",
            "initial_value": self.initial_equity,
            "final_value": final_value,
            "return_pct": ((final_value - self.initial_equity) /
                           self.initial_equity) * 100,
            "max_drawdown": self._calculate_drawdown(
                self._calculate_equity_curve_bh("BTC")
            )
        }

    def equal_weight_hold(self) -> dict:
        """Calculate equal-weight portfolio returns."""
        allocation = {"BTC": 0.333, "XRP": 0.333, "USDT": 0.334}

        # Calculate amounts at start
        btc_amount = (self.initial_equity * allocation["BTC"]) / \
                     self.prices["BTC"][0]["close"]
        xrp_amount = (self.initial_equity * allocation["XRP"]) / \
                     self.prices["XRP"][0]["close"]
        usdt_amount = self.initial_equity * allocation["USDT"]

        # Calculate final values
        final_btc_value = btc_amount * self.prices["BTC"][-1]["close"]
        final_xrp_value = xrp_amount * self.prices["XRP"][-1]["close"]
        final_value = final_btc_value + final_xrp_value + usdt_amount

        return {
            "strategy": "Equal Weight Hold (33/33/33)",
            "initial_value": self.initial_equity,
            "final_value": final_value,
            "return_pct": ((final_value - self.initial_equity) /
                           self.initial_equity) * 100
        }

    def simple_ema_crossover(self) -> dict:
        """
        Calculate returns for simple EMA crossover strategy.
        Uses EMA(9) crosses EMA(21) on 4H timeframe.
        """
        # Implementation: backtest EMA crossover
        signals = self._generate_ema_signals(
            self.prices["BTC"],
            fast_period=9,
            slow_period=21
        )

        equity_curve = self._backtest_signals(signals, self.initial_equity)

        return {
            "strategy": "EMA(9/21) Crossover",
            "initial_value": self.initial_equity,
            "final_value": equity_curve[-1],
            "return_pct": ((equity_curve[-1] - self.initial_equity) /
                           self.initial_equity) * 100,
            "max_drawdown": self._calculate_drawdown(equity_curve),
            "trades": len([s for s in signals if s != 0])
        }
```

### 3.3 Comparison Dashboard

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ STRATEGY COMPARISON (Last 90 Days)                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ Strategy                │ Return   │ Max DD  │ Sharpe │ Sortino │ Calmar  │
│ ────────────────────────┼──────────┼─────────┼────────┼─────────┼─────────│
│ TripleGain System       │ +24.5%   │ -8.3%   │ 1.82   │ 2.45    │ 2.95    │
│ Buy-and-Hold BTC        │ +18.2%   │ -22.1%  │ 0.85   │ 0.92    │ 0.82    │
│ Buy-and-Hold XRP        │ +32.4%   │ -35.8%  │ 0.78   │ 0.81    │ 0.91    │
│ Equal Weight Hold       │ +21.5%   │ -18.5%  │ 1.02   │ 1.15    │ 1.16    │
│ EMA(9/21) Crossover     │ +12.8%   │ -15.2%  │ 0.72   │ 0.88    │ 0.84    │
│ Random Walk             │ -5.2%    │ -28.4%  │ -0.21  │ -0.18   │ -0.18   │
│                                                                             │
│ TripleGain Outperformance:                                                  │
│ • vs Buy-and-Hold BTC: +6.3% with 63% less drawdown                        │
│ • vs Equal Weight:     +3.0% with 55% less drawdown                        │
│ • vs EMA Crossover:    +11.7% with 45% less drawdown                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Risk-Adjusted Return Metrics

### 4.1 Metric Definitions

```python
def calculate_sharpe_ratio(
    returns: list[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 365
) -> float:
    """
    Calculate annualized Sharpe ratio.

    Sharpe = (Annualized Return - Risk-Free Rate) / Annualized Volatility
    """
    import numpy as np

    returns_array = np.array(returns)

    # Annualized return
    total_return = np.prod(1 + returns_array) - 1
    periods = len(returns_array)
    annualized_return = (1 + total_return) ** (periods_per_year / periods) - 1

    # Annualized volatility
    daily_volatility = np.std(returns_array)
    annualized_volatility = daily_volatility * np.sqrt(periods_per_year)

    # Sharpe ratio
    sharpe = (annualized_return - risk_free_rate) / annualized_volatility

    return sharpe


def calculate_sortino_ratio(
    returns: list[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 365
) -> float:
    """
    Calculate annualized Sortino ratio.

    Sortino = (Annualized Return - Risk-Free Rate) / Downside Deviation

    Only considers negative returns for volatility calculation.
    """
    import numpy as np

    returns_array = np.array(returns)

    # Annualized return
    total_return = np.prod(1 + returns_array) - 1
    periods = len(returns_array)
    annualized_return = (1 + total_return) ** (periods_per_year / periods) - 1

    # Downside deviation (only negative returns)
    negative_returns = returns_array[returns_array < 0]
    if len(negative_returns) == 0:
        return float('inf')  # No downside

    downside_deviation = np.std(negative_returns) * np.sqrt(periods_per_year)

    # Sortino ratio
    sortino = (annualized_return - risk_free_rate) / downside_deviation

    return sortino


def calculate_calmar_ratio(
    returns: list[float],
    max_drawdown: float,
    periods_per_year: int = 365
) -> float:
    """
    Calculate Calmar ratio.

    Calmar = Annualized Return / Maximum Drawdown

    Measures return per unit of drawdown risk.
    """
    import numpy as np

    returns_array = np.array(returns)

    # Annualized return
    total_return = np.prod(1 + returns_array) - 1
    periods = len(returns_array)
    annualized_return = (1 + total_return) ** (periods_per_year / periods) - 1

    # Calmar ratio
    if max_drawdown == 0:
        return float('inf')

    calmar = annualized_return / abs(max_drawdown)

    return calmar


def calculate_max_drawdown(equity_curve: list[float]) -> tuple[float, int, int]:
    """
    Calculate maximum drawdown and its duration.

    Returns:
        (max_drawdown_pct, peak_index, trough_index)
    """
    import numpy as np

    equity = np.array(equity_curve)
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak

    max_dd_idx = np.argmin(drawdown)
    max_dd = drawdown[max_dd_idx]

    # Find peak before max drawdown
    peak_idx = np.argmax(equity[:max_dd_idx + 1])

    return abs(max_dd), peak_idx, max_dd_idx
```

### 4.2 Rolling Metrics

| Metric | Window | Update Frequency | Purpose |
|--------|--------|------------------|---------|
| Sharpe (7d) | 7 days | Daily | Short-term performance |
| Sharpe (30d) | 30 days | Daily | Medium-term trend |
| Sharpe (90d) | 90 days | Weekly | Long-term consistency |
| Sortino (30d) | 30 days | Daily | Downside risk tracking |
| Win Rate (7d) | 7 days | Per trade | Recent accuracy |
| Profit Factor (7d) | 7 days | Per trade | Recent profitability |

---

## 5. Agent Performance Evaluation

### 5.1 Agent Accuracy Measurement

```python
@dataclass
class AgentDecisionRecord:
    agent_name: str
    timestamp: datetime
    symbol: str
    decision: str  # BUY, SELL, HOLD
    confidence: float
    actual_outcome: str  # CORRECT, INCORRECT, PENDING
    return_after_1h: float  # Price change after 1 hour
    return_after_4h: float  # Price change after 4 hours
    return_after_24h: float  # Price change after 24 hours


def evaluate_agent_accuracy(
    decisions: list[AgentDecisionRecord],
    evaluation_horizon: str = "4h"
) -> dict:
    """
    Evaluate agent decision accuracy.

    A decision is considered correct if:
    - BUY: price increased after horizon
    - SELL: price decreased after horizon
    - HOLD: price stayed within 1% range
    """
    correct = 0
    incorrect = 0
    pending = 0

    horizon_field = f"return_after_{evaluation_horizon}"

    for d in decisions:
        if d.actual_outcome == "PENDING":
            pending += 1
            continue

        price_change = getattr(d, horizon_field, 0)

        if d.decision == "BUY":
            if price_change > 0:
                correct += 1
            else:
                incorrect += 1
        elif d.decision == "SELL":
            if price_change < 0:
                correct += 1
            else:
                incorrect += 1
        elif d.decision == "HOLD":
            if abs(price_change) < 0.01:  # Within 1%
                correct += 1
            else:
                incorrect += 1

    total = correct + incorrect
    accuracy = correct / total if total > 0 else 0

    return {
        "agent": decisions[0].agent_name if decisions else "Unknown",
        "total_decisions": total,
        "correct": correct,
        "incorrect": incorrect,
        "pending": pending,
        "accuracy": accuracy,
        "evaluation_horizon": evaluation_horizon
    }
```

### 5.2 Confidence Calibration

**Concept**: A well-calibrated model's confidence should match its actual accuracy.

```python
def calculate_confidence_calibration(
    decisions: list[AgentDecisionRecord],
    n_bins: int = 10
) -> dict:
    """
    Calculate confidence calibration.

    Perfect calibration: 70% confident decisions should be correct 70% of time.
    """
    import numpy as np

    # Group decisions into confidence bins
    bins = np.linspace(0, 1, n_bins + 1)
    bin_correct = [0] * n_bins
    bin_total = [0] * n_bins

    for d in decisions:
        if d.actual_outcome == "PENDING":
            continue

        bin_idx = min(int(d.confidence * n_bins), n_bins - 1)
        bin_total[bin_idx] += 1

        if d.actual_outcome == "CORRECT":
            bin_correct[bin_idx] += 1

    # Calculate actual accuracy per bin
    calibration_data = []
    expected_confidence_error = 0

    for i in range(n_bins):
        if bin_total[i] > 0:
            actual_accuracy = bin_correct[i] / bin_total[i]
            expected_confidence = (bins[i] + bins[i+1]) / 2

            calibration_data.append({
                "confidence_range": f"{bins[i]:.1f}-{bins[i+1]:.1f}",
                "expected_accuracy": expected_confidence,
                "actual_accuracy": actual_accuracy,
                "sample_count": bin_total[i]
            })

            expected_confidence_error += abs(actual_accuracy - expected_confidence) * bin_total[i]

    # Overall calibration error (ECE - Expected Calibration Error)
    total_samples = sum(bin_total)
    ece = expected_confidence_error / total_samples if total_samples > 0 else 0

    return {
        "calibration_by_bin": calibration_data,
        "expected_calibration_error": ece,
        "is_well_calibrated": ece < 0.1  # <10% error is good
    }
```

### 5.3 Agent Performance Dashboard

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ AGENT PERFORMANCE (Last 30 Days)                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ Agent                  │ Decisions │ Accuracy │ ECE   │ Avg Latency │ Errors│
│ ───────────────────────┼───────────┼──────────┼───────┼─────────────┼───────│
│ Technical Analysis     │ 2,880     │ 52.1%    │ 0.08  │ 185ms       │ 3     │
│ Regime Detection       │ 1,440     │ 61.5%    │ 0.05  │ 220ms       │ 1     │
│ Sentiment Analysis     │ 288       │ 54.8%    │ 0.12  │ 2.1s        │ 5     │
│ Trading Decision       │ 720       │ 57.4%    │ 0.07  │ 2.8s        │ 2     │
│ Risk Management        │ N/A       │ N/A      │ N/A   │ 8ms         │ 0     │
│ Portfolio Rebalance    │ 15        │ N/A      │ N/A   │ 450ms       │ 0     │
│                                                                             │
│ Best Performing: Regime Detection (61.5% accuracy)                          │
│ Best Calibrated: Regime Detection (ECE: 0.05)                               │
│ Most Reliable: Risk Management (0 errors)                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. LLM Model Comparison Framework (6-Model Parallel Testing)

### 6.1 Comparison Methodology

```
┌─────────────────────────────────────────────────────────────────────────────┐
│               6-MODEL PARALLEL COMPARISON METHODOLOGY                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  MODELS UNDER TEST:                                                         │
│  1. GPT (latest)        - API                                               │
│  2. Grok (latest)       - API (web search capability)                       │
│  3. DeepSeek V3         - API                                               │
│  4. Claude Sonnet       - API                                               │
│  5. Claude Opus         - API                                               │
│  6. Qwen 2.5 7B         - Local (via Ollama)                               │
│                                                                             │
│  PHASE 1: BACKTESTING (Historical Data)                                     │
│  ├── Run all 6 models on same historical data set                          │
│  ├── Compare decision quality against known outcomes                        │
│  ├── Measure latency, cost, and error rates per model                      │
│  └── Generate initial leaderboard ranking                                   │
│                                                                             │
│  PHASE 2: PAPER TRADING (Live Data, Simulated Execution)                    │
│  ├── Run all 6 models in parallel on live market data                      │
│  ├── Track all decisions but don't execute                                  │
│  ├── Compare theoretical P&L per model                                      │
│  └── Analyze consensus patterns and accuracy                                │
│                                                                             │
│  PHASE 3: LIVE TRADING (Consensus-Based Execution)                          │
│  ├── Execute trades based on consensus (majority or unanimous)             │
│  ├── Track which models contributed to winning/losing decisions            │
│  ├── Update leaderboard with real trading performance                      │
│  └── Statistical significance testing across all model pairs               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Six-Model Test Configuration

```yaml
six_model_test_config:
  name: "TripleGain_6_Model_Parallel_Test"
  status: "running"
  start_date: "2025-12-01"
  planned_duration_days: 90

  models:
    - id: "gpt"
      name: "GPT (latest)"
      provider: "openai"
      tier: "api"

    - id: "grok"
      name: "Grok (latest)"
      provider: "xai"
      tier: "api"
      capabilities: ["web_search"]

    - id: "deepseek"
      name: "DeepSeek V3"
      provider: "deepseek"
      tier: "api"

    - id: "claude_sonnet"
      name: "Claude Sonnet"
      provider: "anthropic"
      tier: "api"

    - id: "claude_opus"
      name: "Claude Opus"
      provider: "anthropic"
      tier: "api"

    - id: "qwen"
      name: "Qwen 2.5 7B"
      provider: "ollama"
      tier: "local"

  execution_mode: "parallel_all"  # All models run on every query

  consensus_rules:
    unanimous: { threshold: 6, confidence_boost: 0.15 }
    strong_majority: { threshold: 5, confidence_boost: 0.10 }
    majority: { threshold: 4, confidence_boost: 0.05 }
    split: { threshold: 3, action: "defer_or_hold" }

  metrics_to_compare:
    primary:
      - decision_accuracy
      - profitability
      - confidence_calibration
      - sharpe_ratio

    secondary:
      - latency
      - cost_per_decision
      - error_rate
      - consensus_contribution

  leaderboard:
    update_frequency: "hourly"
    composite_score_weights:
      accuracy: 0.35
      profitability: 0.25
      confidence_calibration: 0.20
      latency: 0.10
      cost: 0.10

  minimum_sample_size: 100  # Decisions per model

  analysis_frequency: "daily"
```

### 6.3 Multi-Model Statistical Analysis

```python
def calculate_multi_model_rankings(
    model_results: dict[str, list[float]],
    metrics: list[str],
    weights: dict[str, float]
) -> dict:
    """
    Calculate rankings and significance for 6-model comparison.

    Uses pairwise comparisons and composite scoring.
    """
    from scipy import stats
    import numpy as np
    from itertools import combinations

    models = list(model_results.keys())
    n_models = len(models)

    # Calculate composite scores for each model
    composite_scores = {}
    for model in models:
        scores = []
        for metric in metrics:
            metric_value = np.mean(model_results[model][metric])
            scores.append(metric_value * weights.get(metric, 0))
        composite_scores[model] = sum(scores)

    # Rank models by composite score
    rankings = sorted(
        composite_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Pairwise significance testing
    pairwise_results = {}
    for model_a, model_b in combinations(models, 2):
        results_a = model_results[model_a]["pnl"]
        results_b = model_results[model_b]["pnl"]

        t_stat, p_value = stats.ttest_ind(results_a, results_b)

        pairwise_results[f"{model_a}_vs_{model_b}"] = {
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "winner": model_a if np.mean(results_a) > np.mean(results_b) else model_b
        }

    # Consensus analysis
    consensus_accuracy = calculate_consensus_accuracy(model_results)

    return {
        "leaderboard": [
            {"rank": i+1, "model": model, "score": score}
            for i, (model, score) in enumerate(rankings)
        ],
        "pairwise_significance": pairwise_results,
        "consensus_analysis": consensus_accuracy,
        "n_comparisons": len(list(combinations(models, 2)))  # 15 pairs for 6 models
    }
```

### 6.4 Six-Model Comparison Report

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 6-MODEL COMPARISON REPORT                                                    │
│ Period: 2025-12-01 to 2025-12-18 (18 days)                                  │
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
│ • Grok > Qwen (p=0.042)                                                    │
│                                                                             │
│ Not significant (need more data):                                           │
│ • DeepSeek vs Claude Opus (p=0.15)                                         │
│ • DeepSeek vs Grok (p=0.22)                                                │
│ • Claude Opus vs Claude Sonnet (p=0.31)                                    │
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
│ Total API cost: $7.85 | Cost per correct decision: $0.029                  │
│ Most cost-effective: DeepSeek V3 ($0.003/correct)                          │
│ Qwen 2.5 7B: $0 (local) but lowest accuracy                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Backtesting Framework

### 7.1 Backtesting Configuration

```yaml
backtest_config:
  name: "TripleGain Full Backtest"

  data:
    symbols: ["BTC/USDT", "XRP/USDT", "XRP/BTC"]
    start_date: "2024-01-01"
    end_date: "2024-12-31"
    timeframes: ["1m", "5m", "15m", "1h", "4h", "1d"]
    source: "timescaledb_historical"

  initial_conditions:
    equity_usd: 2100
    allocation:
      btc: 0.333
      xrp: 0.333
      usdt: 0.334

  execution_model:
    slippage_bps: 5  # 0.05% slippage
    fees_bps: 10     # 0.10% taker fee
    fill_rate: 0.98  # 98% of orders fill

  risk_parameters:
    max_leverage: 5
    risk_per_trade: 0.01
    daily_loss_limit: 0.05
    max_drawdown: 0.20

  agent_configuration:
    use_cached_outputs: true  # Use historical agent outputs if available
    regenerate_decisions: false  # Or re-run agents on historical data

  output:
    save_equity_curve: true
    save_trade_log: true
    save_agent_decisions: true
    output_dir: "backtests/triplegain_2024/"
```

### 7.2 Backtest Execution

```python
class Backtester:
    """
    Run backtests on historical data.
    """

    def __init__(self, config: dict, data_provider, agent_system):
        self.config = config
        self.data = data_provider
        self.agents = agent_system

    async def run(self) -> dict:
        """
        Execute full backtest.
        """
        # Initialize state
        equity = self.config["initial_conditions"]["equity_usd"]
        positions = []
        trade_log = []
        equity_curve = [equity]

        # Get historical data
        candles = await self.data.get_candles(
            self.config["data"]["symbols"],
            self.config["data"]["start_date"],
            self.config["data"]["end_date"]
        )

        # Iterate through time
        for timestamp in self._generate_timestamps(candles):
            # Get market snapshot at this timestamp
            snapshot = self._get_snapshot_at(candles, timestamp)

            # Run agents
            agent_outputs = await self.agents.run_all(snapshot)

            # Get trading decision
            decision = agent_outputs.get("trading_decision")

            if decision and decision["action"] in ["BUY", "SELL"]:
                # Validate with risk management
                validated = self._validate_trade(decision, equity, positions)

                if validated["approved"]:
                    # Execute trade
                    execution = self._execute_trade(
                        validated["trade"],
                        snapshot,
                        self.config["execution_model"]
                    )

                    if execution["filled"]:
                        positions.append(execution["position"])
                        trade_log.append(execution)
                        equity -= execution["fees"]

            # Update positions (check stops, calculate P&L)
            positions, closed_trades = self._update_positions(
                positions,
                snapshot
            )

            for closed in closed_trades:
                equity += closed["realized_pnl"]
                trade_log.append(closed)

            equity_curve.append(equity)

        # Calculate final metrics
        return self._calculate_results(equity_curve, trade_log)

    def _calculate_results(
        self,
        equity_curve: list,
        trade_log: list
    ) -> dict:
        """
        Calculate backtest performance metrics.
        """
        initial_equity = equity_curve[0]
        final_equity = equity_curve[-1]

        returns = [
            (equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1]
            for i in range(1, len(equity_curve))
        ]

        max_dd, peak_idx, trough_idx = calculate_max_drawdown(equity_curve)

        winning_trades = [t for t in trade_log if t.get("realized_pnl", 0) > 0]
        losing_trades = [t for t in trade_log if t.get("realized_pnl", 0) < 0]

        return {
            "summary": {
                "initial_equity": initial_equity,
                "final_equity": final_equity,
                "total_return_pct": ((final_equity - initial_equity) / initial_equity) * 100,
                "max_drawdown_pct": max_dd * 100,
                "sharpe_ratio": calculate_sharpe_ratio(returns),
                "sortino_ratio": calculate_sortino_ratio(returns),
                "calmar_ratio": calculate_calmar_ratio(returns, max_dd)
            },
            "trades": {
                "total_trades": len(trade_log),
                "winning_trades": len(winning_trades),
                "losing_trades": len(losing_trades),
                "win_rate": len(winning_trades) / len(trade_log) if trade_log else 0,
                "profit_factor": (
                    sum(t["realized_pnl"] for t in winning_trades) /
                    abs(sum(t["realized_pnl"] for t in losing_trades))
                    if losing_trades else float('inf')
                ),
                "avg_win": sum(t["realized_pnl"] for t in winning_trades) / len(winning_trades) if winning_trades else 0,
                "avg_loss": sum(t["realized_pnl"] for t in losing_trades) / len(losing_trades) if losing_trades else 0
            },
            "equity_curve": equity_curve,
            "trade_log": trade_log
        }
```

### 7.3 Walk-Forward Validation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WALK-FORWARD VALIDATION                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ PURPOSE: Prevent overfitting by testing on unseen data                      │
│                                                                             │
│ METHODOLOGY:                                                                │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │                                                                          │ │
│ │  Training       │ Test    │ (repeat)                                    │ │
│ │  ═══════════════│═════════│                                             │ │
│ │  [  6 months   ]│[1 month]│                                             │ │
│ │        │←───────│─→ roll forward                                        │ │
│ │                 │                                                        │ │
│ │        [  6 months   ]│[1 month]│                                       │ │
│ │                       │←───────│─→ roll forward                         │ │
│ │                                │                                         │ │
│ │               [  6 months   ]│[1 month]│                                │ │
│ │                              │←───────│─→ roll forward                  │ │
│ │                                                                          │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│ CONFIGURATION:                                                              │
│   Training window: 6 months                                                 │
│   Test window: 1 month                                                      │
│   Roll step: 1 month                                                        │
│   Total periods: 12 (covering 18 months)                                    │
│                                                                             │
│ ANALYSIS:                                                                   │
│   • Compare training vs test performance                                    │
│   • Flag if test significantly underperforms training (overfitting)        │
│   • Calculate aggregate test metrics across all windows                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Reporting & Dashboards

### 8.1 Daily Performance Report

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ DAILY PERFORMANCE REPORT                                                     │
│ Date: 2025-12-18                                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ PORTFOLIO SUMMARY                                                            │
│   Opening Equity: $2,100.00                                                  │
│   Closing Equity: $2,145.32                                                  │
│   Daily P&L: +$45.32 (+2.16%)                                               │
│   Max Drawdown Today: -0.8%                                                  │
│                                                                             │
│ TRADING ACTIVITY                                                             │
│   Trades Executed: 3                                                         │
│   Win Rate: 66.7% (2W / 1L)                                                 │
│   Largest Win: +$38.50 (BTC/USDT)                                           │
│   Largest Loss: -$12.20 (XRP/USDT)                                          │
│                                                                             │
│ AGENT PERFORMANCE                                                            │
│   Trading Decisions Made: 24                                                 │
│   Decisions Executed: 3                                                      │
│   Average Confidence: 0.68                                                  │
│   Agent Errors: 0                                                            │
│                                                                             │
│ RISK STATUS                                                                  │
│   Daily Loss Limit Used: 0% of 5%                                           │
│   Leverage Peak: 2.5x                                                        │
│   Circuit Breakers: None triggered                                           │
│                                                                             │
│ vs BENCHMARKS (Today)                                                        │
│   vs Buy-and-Hold BTC: +1.2% outperformance                                 │
│   vs Equal Weight: +0.8% outperformance                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.2 Weekly Summary Report

Generated every Sunday with:
- Week's total P&L and metrics
- Best/worst performing days
- Agent accuracy for the week
- Benchmark comparison
- Notable events and decisions
- Recommendations for parameter adjustments

### 8.3 Monthly Deep Analysis

Generated monthly with:
- Full performance analytics
- LLM model comparison update
- Strategy effectiveness analysis
- Risk metrics deep dive
- Recommendations for system improvements

---

*Document Version 1.0 - December 2025*
