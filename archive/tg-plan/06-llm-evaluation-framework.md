# LLM Performance Evaluation Framework

**Version:** 1.0
**Date:** December 2025
**Status:** Design Phase

---

## Overview

This document defines the framework for evaluating and comparing LLM performance in trading decisions. The framework enables objective comparison of Claude, GPT-4, Grok, Deepseek, and local models (Qwen) to determine optimal model selection and auto-switching thresholds.

---

## Table of Contents

1. [Evaluation Objectives](#1-evaluation-objectives)
2. [Performance Metrics](#2-performance-metrics)
3. [Comparison Methodology](#3-comparison-methodology)
4. [Data Collection](#4-data-collection)
5. [Analysis Framework](#5-analysis-framework)
6. [Model Selection Algorithm](#6-model-selection-algorithm)
7. [Reporting System](#7-reporting-system)

---

## 1. Evaluation Objectives

### 1.1 Primary Goals

| Goal | Description | Measurement |
|------|-------------|-------------|
| **Profitability** | Identify most profitable model | Total P&L, Sharpe Ratio |
| **Risk Management** | Assess risk discipline | Max Drawdown, Stop-loss adherence |
| **Consistency** | Measure reliability | Win rate stability, variance |
| **Efficiency** | Compare cost vs. performance | P&L per dollar of API cost |
| **Latency** | Operational suitability | Response time, timeout rate |

### 1.2 Secondary Goals

| Goal | Description | Measurement |
|------|-------------|-------------|
| **Reasoning Quality** | Evaluate decision logic | Human review score |
| **Market Regime Adaptation** | Performance across conditions | Per-regime Sharpe |
| **Confidence Calibration** | Accuracy of confidence scores | Brier score |
| **Parse Reliability** | Technical output quality | Parse success rate |

---

## 2. Performance Metrics

### 2.1 Trading Performance Metrics

```python
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

@dataclass
class TradingMetrics:
    """Core trading performance metrics."""

    # Return metrics
    total_pnl: float
    total_pnl_pct: float
    annualized_return: float

    # Risk-adjusted metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Risk metrics
    max_drawdown: float
    max_drawdown_duration_days: int
    volatility: float
    var_95: float  # Value at Risk (95%)

    # Trade metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    avg_holding_time_hours: float

    # Efficiency metrics
    expectancy: float  # Average profit per trade
    risk_reward_achieved: float  # Actual avg win / avg loss

def calculate_trading_metrics(trades: List[Trade]) -> TradingMetrics:
    """Calculate comprehensive trading metrics from trade list."""

    if not trades:
        return None

    # Extract P&L values
    pnls = [t.pnl for t in trades if t.pnl is not None]
    pnl_pcts = [t.pnl_pct for t in trades if t.pnl_pct is not None]

    if not pnls:
        return None

    # Basic metrics
    total_pnl = sum(pnls)
    total_pnl_pct = sum(pnl_pcts)

    # Win/loss separation
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    # Win rate
    win_rate = len(wins) / len(pnls) if pnls else 0

    # Profit factor
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Average metrics
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0

    # Risk-adjusted returns
    returns = np.array(pnl_pcts)
    volatility = np.std(returns) if len(returns) > 1 else 0
    sharpe = (np.mean(returns) / volatility * np.sqrt(252)) if volatility > 0 else 0

    # Drawdown calculation
    cumulative = np.cumsum(pnl_pcts)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative - running_max
    max_drawdown = abs(np.min(drawdowns))

    # Sortino (downside deviation)
    negative_returns = returns[returns < 0]
    downside_std = np.std(negative_returns) if len(negative_returns) > 0 else volatility
    sortino = (np.mean(returns) / downside_std * np.sqrt(252)) if downside_std > 0 else 0

    # Calmar ratio
    calmar = (total_pnl_pct / max_drawdown) if max_drawdown > 0 else float('inf')

    # Expectancy
    expectancy = np.mean(pnls)

    # Risk-reward achieved
    rr_achieved = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

    return TradingMetrics(
        total_pnl=total_pnl,
        total_pnl_pct=total_pnl_pct,
        annualized_return=total_pnl_pct * (365 / max(1, len(set(t.date for t in trades)))),
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        max_drawdown=max_drawdown,
        max_drawdown_duration_days=0,  # Calculate separately
        volatility=volatility,
        var_95=np.percentile(returns, 5) if len(returns) > 0 else 0,
        total_trades=len(trades),
        winning_trades=len(wins),
        losing_trades=len(losses),
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss,
        largest_win=max(wins) if wins else 0,
        largest_loss=min(losses) if losses else 0,
        avg_holding_time_hours=np.mean([t.holding_hours for t in trades if t.holding_hours]),
        expectancy=expectancy,
        risk_reward_achieved=rr_achieved
    )
```

### 2.2 Model-Specific Metrics

```python
@dataclass
class ModelMetrics:
    """Metrics specific to LLM performance."""

    model_name: str

    # Operational metrics
    total_queries: int
    successful_queries: int
    failed_queries: int
    timeout_count: int
    parse_failures: int
    success_rate: float

    # Latency metrics
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    max_latency_ms: float

    # Cost metrics
    total_tokens_input: int
    total_tokens_output: int
    total_cost_usd: float
    cost_per_trade: float
    pnl_per_dollar_cost: float

    # Decision quality
    confidence_accuracy: float  # How often high confidence = profitable
    action_distribution: dict[str, int]  # LONG/SHORT/HOLD/CLOSE counts
    avg_confidence: float
    confidence_when_profitable: float
    confidence_when_unprofitable: float

    # Trading performance (subset for this model)
    trading_metrics: TradingMetrics

def calculate_model_metrics(
    model_name: str,
    decisions: List[TradingDecision],
    trades: List[Trade],
    api_logs: List[APILog]
) -> ModelMetrics:
    """Calculate model-specific metrics."""

    # Filter to this model
    model_decisions = [d for d in decisions if d.model_used == model_name]
    model_trades = [t for t in trades if t.model == model_name]
    model_logs = [l for l in api_logs if l.model == model_name]

    # Operational metrics
    successful = [l for l in model_logs if l.success]
    failed = [l for l in model_logs if not l.success]
    timeouts = [l for l in failed if l.error_type == 'timeout']
    parse_failures = [l for l in failed if l.error_type == 'parse_error']

    success_rate = len(successful) / len(model_logs) if model_logs else 0

    # Latency
    latencies = [l.latency_ms for l in successful]
    avg_latency = np.mean(latencies) if latencies else 0

    # Cost
    total_input = sum(l.tokens_input for l in model_logs)
    total_output = sum(l.tokens_output for l in model_logs)
    total_cost = sum(l.cost_usd for l in model_logs)

    # Decision quality
    confidences = [d.confidence for d in model_decisions]
    avg_confidence = np.mean(confidences) if confidences else 0

    # Confidence calibration
    profitable_trades = [t for t in model_trades if t.pnl and t.pnl > 0]
    unprofitable_trades = [t for t in model_trades if t.pnl and t.pnl < 0]

    conf_when_profit = np.mean([
        d.confidence for d in model_decisions
        if any(t.decision_id == d.id and t.pnl > 0 for t in model_trades)
    ]) if profitable_trades else 0

    conf_when_loss = np.mean([
        d.confidence for d in model_decisions
        if any(t.decision_id == d.id and t.pnl < 0 for t in model_trades)
    ]) if unprofitable_trades else 0

    # Action distribution
    action_dist = {}
    for d in model_decisions:
        action_dist[d.action] = action_dist.get(d.action, 0) + 1

    # Trading metrics
    trading = calculate_trading_metrics(model_trades)

    return ModelMetrics(
        model_name=model_name,
        total_queries=len(model_logs),
        successful_queries=len(successful),
        failed_queries=len(failed),
        timeout_count=len(timeouts),
        parse_failures=len(parse_failures),
        success_rate=success_rate,
        avg_latency_ms=avg_latency,
        p50_latency_ms=np.percentile(latencies, 50) if latencies else 0,
        p95_latency_ms=np.percentile(latencies, 95) if latencies else 0,
        p99_latency_ms=np.percentile(latencies, 99) if latencies else 0,
        max_latency_ms=max(latencies) if latencies else 0,
        total_tokens_input=total_input,
        total_tokens_output=total_output,
        total_cost_usd=total_cost,
        cost_per_trade=total_cost / len(model_trades) if model_trades else 0,
        pnl_per_dollar_cost=trading.total_pnl / total_cost if total_cost > 0 else 0,
        confidence_accuracy=0,  # Calculate separately
        action_distribution=action_dist,
        avg_confidence=avg_confidence,
        confidence_when_profitable=conf_when_profit,
        confidence_when_unprofitable=conf_when_loss,
        trading_metrics=trading
    )
```

### 2.3 Composite Score Calculation

```python
@dataclass
class ModelScore:
    """Composite score for model ranking."""

    model_name: str
    total_score: float
    component_scores: dict[str, float]
    rank: int

def calculate_composite_score(metrics: ModelMetrics) -> ModelScore:
    """Calculate weighted composite score for model comparison."""

    # Define weights for each component
    weights = {
        'sharpe_ratio': 0.25,        # Risk-adjusted returns
        'profit_factor': 0.15,       # Win/loss ratio
        'win_rate': 0.10,            # Consistency
        'max_drawdown_inv': 0.15,    # Risk control (inverted)
        'success_rate': 0.10,        # Operational reliability
        'cost_efficiency': 0.10,     # Cost effectiveness
        'latency_score': 0.05,       # Speed
        'confidence_calibration': 0.10  # Decision quality
    }

    components = {}

    # Sharpe ratio (normalized 0-1, capped at 3)
    sharpe_norm = min(metrics.trading_metrics.sharpe_ratio / 3, 1.0)
    components['sharpe_ratio'] = max(0, sharpe_norm)

    # Profit factor (normalized 0-1, capped at 3)
    pf_norm = min(metrics.trading_metrics.profit_factor / 3, 1.0)
    components['profit_factor'] = max(0, pf_norm)

    # Win rate (already 0-1)
    components['win_rate'] = metrics.trading_metrics.win_rate

    # Max drawdown inverted (lower is better)
    dd_inv = 1 - min(metrics.trading_metrics.max_drawdown / 0.20, 1.0)
    components['max_drawdown_inv'] = max(0, dd_inv)

    # Success rate (already 0-1)
    components['success_rate'] = metrics.success_rate

    # Cost efficiency (PnL per dollar, normalized)
    # Assume good is $100 PnL per $1 cost
    cost_eff = min(metrics.pnl_per_dollar_cost / 100, 1.0)
    components['cost_efficiency'] = max(0, cost_eff)

    # Latency score (faster is better, <1s is perfect)
    latency_score = 1 - min(metrics.avg_latency_ms / 10000, 1.0)
    components['latency_score'] = max(0, latency_score)

    # Confidence calibration
    # High confidence should correlate with profitability
    if metrics.confidence_when_profitable > 0 and metrics.confidence_when_unprofitable > 0:
        calibration = (metrics.confidence_when_profitable - metrics.confidence_when_unprofitable)
        calibration_norm = (calibration + 1) / 2  # Normalize to 0-1
    else:
        calibration_norm = 0.5
    components['confidence_calibration'] = calibration_norm

    # Calculate weighted total
    total_score = sum(
        components[key] * weights[key]
        for key in weights
    )

    return ModelScore(
        model_name=metrics.model_name,
        total_score=total_score,
        component_scores=components,
        rank=0  # Set after comparing all models
    )
```

---

## 3. Comparison Methodology

### 3.1 Parallel Execution Framework

```python
from datetime import datetime
from typing import Dict, List
import asyncio

class ParallelModelEvaluator:
    """Run multiple models in parallel for fair comparison."""

    def __init__(self, models: List[str], llm_client: LLMClient):
        self.models = models
        self.client = llm_client

    async def evaluate_signal(
        self,
        market_context: MarketContext,
        system_prompt: str,
        user_prompt: str
    ) -> Dict[str, TradingDecision]:
        """Query all models with identical context."""

        tasks = []
        for model in self.models:
            task = self._query_model(model, system_prompt, user_prompt)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        decisions = {}
        for model, result in zip(self.models, results):
            if isinstance(result, Exception):
                decisions[model] = None
            else:
                decisions[model] = result

        return decisions

    async def _query_model(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str
    ) -> TradingDecision:
        """Query single model with timeout handling."""
        try:
            response = await asyncio.wait_for(
                self.client.query(model, system_prompt, user_prompt),
                timeout=15.0
            )
            return parse_trading_decision(response, model)
        except asyncio.TimeoutError:
            raise
        except Exception as e:
            raise
```

### 3.2 A/B Testing Framework

```python
@dataclass
class ABTestConfig:
    """Configuration for A/B testing models."""
    model_a: str
    model_b: str
    allocation_a: float = 0.5  # 50% of trades to model A
    min_trades: int = 100      # Minimum trades before evaluation
    confidence_level: float = 0.95  # Statistical confidence required

class ABTestManager:
    """Manage A/B testing between two models."""

    def __init__(self, config: ABTestConfig):
        self.config = config
        self.trades_a: List[Trade] = []
        self.trades_b: List[Trade] = []

    def assign_model(self) -> str:
        """Randomly assign a model for this trade."""
        import random
        if random.random() < self.config.allocation_a:
            return self.config.model_a
        return self.config.model_b

    def record_trade(self, trade: Trade) -> None:
        """Record trade result."""
        if trade.model == self.config.model_a:
            self.trades_a.append(trade)
        else:
            self.trades_b.append(trade)

    def can_conclude(self) -> bool:
        """Check if enough data for statistical conclusion."""
        return (len(self.trades_a) >= self.config.min_trades and
                len(self.trades_b) >= self.config.min_trades)

    def get_winner(self) -> Optional[str]:
        """Determine winner with statistical significance."""
        if not self.can_conclude():
            return None

        metrics_a = calculate_trading_metrics(self.trades_a)
        metrics_b = calculate_trading_metrics(self.trades_b)

        # T-test on returns
        from scipy import stats
        returns_a = [t.pnl_pct for t in self.trades_a if t.pnl_pct]
        returns_b = [t.pnl_pct for t in self.trades_b if t.pnl_pct]

        t_stat, p_value = stats.ttest_ind(returns_a, returns_b)

        if p_value < (1 - self.config.confidence_level):
            # Statistically significant difference
            if np.mean(returns_a) > np.mean(returns_b):
                return self.config.model_a
            return self.config.model_b

        return None  # No significant difference
```

### 3.3 Market Regime Stratification

```python
class RegimeStratifiedEvaluator:
    """Evaluate model performance across market regimes."""

    REGIMES = ['trending_up', 'trending_down', 'ranging', 'volatile', 'quiet']

    def __init__(self):
        self.regime_trades: Dict[str, Dict[str, List[Trade]]] = {
            regime: {} for regime in self.REGIMES
        }

    def record_trade(self, trade: Trade, regime: str) -> None:
        """Record trade with regime context."""
        if regime not in self.regime_trades:
            self.regime_trades[regime] = {}
        if trade.model not in self.regime_trades[regime]:
            self.regime_trades[regime][trade.model] = []
        self.regime_trades[regime][trade.model].append(trade)

    def get_regime_performance(self, model: str) -> Dict[str, TradingMetrics]:
        """Get performance breakdown by regime."""
        results = {}
        for regime in self.REGIMES:
            trades = self.regime_trades.get(regime, {}).get(model, [])
            if trades:
                results[regime] = calculate_trading_metrics(trades)
        return results

    def get_best_model_per_regime(self) -> Dict[str, str]:
        """Identify best model for each regime."""
        best_models = {}

        for regime in self.REGIMES:
            regime_trades = self.regime_trades.get(regime, {})
            if not regime_trades:
                continue

            best_model = None
            best_sharpe = float('-inf')

            for model, trades in regime_trades.items():
                if len(trades) >= 10:  # Minimum trades for comparison
                    metrics = calculate_trading_metrics(trades)
                    if metrics.sharpe_ratio > best_sharpe:
                        best_sharpe = metrics.sharpe_ratio
                        best_model = model

            if best_model:
                best_models[regime] = best_model

        return best_models
```

---

## 4. Data Collection

### 4.1 Database Schema

```sql
-- Decision log table
CREATE TABLE llm_decisions (
    id SERIAL,
    timestamp TIMESTAMPTZ NOT NULL,
    request_id UUID NOT NULL,
    model VARCHAR(50) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    action VARCHAR(10) NOT NULL,
    confidence DECIMAL(5,4) NOT NULL,
    position_size_pct DECIMAL(10,4),
    leverage INTEGER,
    entry_price DECIMAL(20,8),
    stop_loss DECIMAL(20,8),
    take_profit DECIMAL(20,8),
    invalidation TEXT,
    reasoning TEXT,
    raw_response TEXT,
    regime VARCHAR(30),
    executed BOOLEAN DEFAULT FALSE,
    PRIMARY KEY (id, timestamp)
);
SELECT create_hypertable('llm_decisions', 'timestamp');

-- API call log table
CREATE TABLE llm_api_logs (
    id SERIAL,
    timestamp TIMESTAMPTZ NOT NULL,
    request_id UUID NOT NULL,
    model VARCHAR(50) NOT NULL,
    success BOOLEAN NOT NULL,
    error_type VARCHAR(30),
    error_message TEXT,
    latency_ms INTEGER NOT NULL,
    tokens_input INTEGER,
    tokens_output INTEGER,
    cost_usd DECIMAL(10,6),
    PRIMARY KEY (id, timestamp)
);
SELECT create_hypertable('llm_api_logs', 'timestamp');

-- Trade results linked to decisions
CREATE TABLE trade_results (
    id SERIAL,
    timestamp TIMESTAMPTZ NOT NULL,
    decision_id INTEGER NOT NULL,
    model VARCHAR(50) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    entry_price DECIMAL(20,8) NOT NULL,
    exit_price DECIMAL(20,8),
    exit_reason VARCHAR(30),  -- stop_loss, take_profit, signal, manual
    pnl DECIMAL(20,8),
    pnl_pct DECIMAL(10,6),
    holding_time_seconds INTEGER,
    regime_at_entry VARCHAR(30),
    regime_at_exit VARCHAR(30),
    PRIMARY KEY (id, timestamp)
);
SELECT create_hypertable('trade_results', 'timestamp');

-- Model performance snapshots (daily aggregates)
CREATE TABLE model_performance_daily (
    date DATE NOT NULL,
    model VARCHAR(50) NOT NULL,
    total_decisions INTEGER,
    executed_trades INTEGER,
    winning_trades INTEGER,
    losing_trades INTEGER,
    total_pnl DECIMAL(20,8),
    sharpe_ratio DECIMAL(10,6),
    max_drawdown DECIMAL(10,6),
    total_cost_usd DECIMAL(10,6),
    avg_latency_ms INTEGER,
    success_rate DECIMAL(5,4),
    PRIMARY KEY (date, model)
);

-- Indexes for performance
CREATE INDEX idx_decisions_model ON llm_decisions (model, timestamp DESC);
CREATE INDEX idx_decisions_regime ON llm_decisions (regime, timestamp DESC);
CREATE INDEX idx_trades_model ON trade_results (model, timestamp DESC);
CREATE INDEX idx_api_logs_model ON llm_api_logs (model, timestamp DESC);
```

### 4.2 Data Collection Pipeline

```python
from datetime import datetime
from typing import Optional
import asyncpg

class EvaluationDataCollector:
    """Collect and store evaluation data."""

    def __init__(self, db_pool: asyncpg.Pool):
        self.db = db_pool

    async def log_decision(
        self,
        decision: TradingDecision,
        request_id: str,
        regime: str
    ) -> int:
        """Log LLM decision to database."""

        query = """
        INSERT INTO llm_decisions (
            timestamp, request_id, model, symbol, action, confidence,
            position_size_pct, leverage, entry_price, stop_loss,
            take_profit, invalidation, reasoning, raw_response, regime
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
        RETURNING id
        """

        row = await self.db.fetchrow(
            query,
            datetime.utcnow(),
            request_id,
            decision.model_used,
            decision.symbol,
            decision.action,
            decision.confidence,
            decision.position_size_pct,
            decision.leverage,
            decision.entry_price,
            decision.stop_loss,
            decision.take_profit,
            decision.invalidation,
            decision.reasoning,
            decision.raw_response,
            regime
        )

        return row['id']

    async def log_api_call(
        self,
        request_id: str,
        model: str,
        success: bool,
        latency_ms: int,
        tokens_input: int = 0,
        tokens_output: int = 0,
        cost_usd: float = 0.0,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None
    ) -> None:
        """Log API call metrics."""

        query = """
        INSERT INTO llm_api_logs (
            timestamp, request_id, model, success, error_type,
            error_message, latency_ms, tokens_input, tokens_output, cost_usd
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        """

        await self.db.execute(
            query,
            datetime.utcnow(),
            request_id,
            model,
            success,
            error_type,
            error_message,
            latency_ms,
            tokens_input,
            tokens_output,
            cost_usd
        )

    async def log_trade_result(
        self,
        decision_id: int,
        model: str,
        symbol: str,
        entry_price: float,
        exit_price: float,
        exit_reason: str,
        pnl: float,
        pnl_pct: float,
        holding_time_seconds: int,
        regime_at_entry: str,
        regime_at_exit: str
    ) -> None:
        """Log trade result linked to decision."""

        query = """
        INSERT INTO trade_results (
            timestamp, decision_id, model, symbol, entry_price,
            exit_price, exit_reason, pnl, pnl_pct, holding_time_seconds,
            regime_at_entry, regime_at_exit
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
        """

        await self.db.execute(
            query,
            datetime.utcnow(),
            decision_id,
            model,
            symbol,
            entry_price,
            exit_price,
            exit_reason,
            pnl,
            pnl_pct,
            holding_time_seconds,
            regime_at_entry,
            regime_at_exit
        )

        # Mark decision as executed
        await self.db.execute(
            "UPDATE llm_decisions SET executed = TRUE WHERE id = $1",
            decision_id
        )

    async def compute_daily_aggregates(self, date: datetime.date) -> None:
        """Compute and store daily performance aggregates."""

        query = """
        INSERT INTO model_performance_daily (
            date, model, total_decisions, executed_trades,
            winning_trades, losing_trades, total_pnl,
            sharpe_ratio, max_drawdown, total_cost_usd,
            avg_latency_ms, success_rate
        )
        SELECT
            $1::date as date,
            d.model,
            COUNT(d.id) as total_decisions,
            COUNT(t.id) as executed_trades,
            COUNT(CASE WHEN t.pnl > 0 THEN 1 END) as winning_trades,
            COUNT(CASE WHEN t.pnl < 0 THEN 1 END) as losing_trades,
            COALESCE(SUM(t.pnl), 0) as total_pnl,
            0 as sharpe_ratio,  -- Calculate separately
            0 as max_drawdown,  -- Calculate separately
            COALESCE(SUM(a.cost_usd), 0) as total_cost_usd,
            COALESCE(AVG(a.latency_ms), 0) as avg_latency_ms,
            COALESCE(AVG(CASE WHEN a.success THEN 1.0 ELSE 0.0 END), 0) as success_rate
        FROM llm_decisions d
        LEFT JOIN trade_results t ON d.id = t.decision_id
        LEFT JOIN llm_api_logs a ON d.request_id = a.request_id AND d.model = a.model
        WHERE d.timestamp::date = $1
        GROUP BY d.model
        ON CONFLICT (date, model) DO UPDATE SET
            total_decisions = EXCLUDED.total_decisions,
            executed_trades = EXCLUDED.executed_trades,
            winning_trades = EXCLUDED.winning_trades,
            losing_trades = EXCLUDED.losing_trades,
            total_pnl = EXCLUDED.total_pnl,
            total_cost_usd = EXCLUDED.total_cost_usd,
            avg_latency_ms = EXCLUDED.avg_latency_ms,
            success_rate = EXCLUDED.success_rate
        """

        await self.db.execute(query, date)
```

---

## 5. Analysis Framework

### 5.1 Comparative Analysis Report

```python
from dataclasses import dataclass
from typing import List, Dict
from datetime import datetime, timedelta

@dataclass
class ComparativeReport:
    """Complete comparative analysis report."""

    report_date: datetime
    period_start: datetime
    period_end: datetime
    models_evaluated: List[str]
    model_metrics: Dict[str, ModelMetrics]
    model_scores: Dict[str, ModelScore]
    rankings: List[str]  # Ordered best to worst
    regime_performance: Dict[str, Dict[str, TradingMetrics]]
    recommendations: List[str]
    summary: str

class ComparativeAnalyzer:
    """Generate comparative analysis reports."""

    def __init__(self, data_collector: EvaluationDataCollector):
        self.collector = data_collector

    async def generate_report(
        self,
        period_days: int = 7
    ) -> ComparativeReport:
        """Generate comprehensive comparison report."""

        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=period_days)

        # Fetch all data for period
        decisions = await self._fetch_decisions(start_date, end_date)
        trades = await self._fetch_trades(start_date, end_date)
        api_logs = await self._fetch_api_logs(start_date, end_date)

        # Identify all models
        models = list(set(d.model_used for d in decisions))

        # Calculate metrics for each model
        model_metrics = {}
        model_scores = {}

        for model in models:
            metrics = calculate_model_metrics(model, decisions, trades, api_logs)
            model_metrics[model] = metrics
            model_scores[model] = calculate_composite_score(metrics)

        # Rank models
        sorted_models = sorted(
            models,
            key=lambda m: model_scores[m].total_score,
            reverse=True
        )
        for rank, model in enumerate(sorted_models, 1):
            model_scores[model].rank = rank

        # Regime performance
        regime_perf = await self._analyze_regime_performance(trades)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            model_metrics, model_scores, sorted_models
        )

        # Summary
        summary = self._generate_summary(
            model_metrics, model_scores, sorted_models, regime_perf
        )

        return ComparativeReport(
            report_date=datetime.utcnow(),
            period_start=start_date,
            period_end=end_date,
            models_evaluated=models,
            model_metrics=model_metrics,
            model_scores=model_scores,
            rankings=sorted_models,
            regime_performance=regime_perf,
            recommendations=recommendations,
            summary=summary
        )

    def _generate_recommendations(
        self,
        metrics: Dict[str, ModelMetrics],
        scores: Dict[str, ModelScore],
        rankings: List[str]
    ) -> List[str]:
        """Generate actionable recommendations."""

        recs = []

        best = rankings[0]
        best_metrics = metrics[best]

        # Primary model recommendation
        recs.append(f"Use {best} as primary model (score: {scores[best].total_score:.3f})")

        # Cost optimization
        cheapest = min(rankings, key=lambda m: metrics[m].cost_per_trade)
        if cheapest != best:
            cost_diff = metrics[best].cost_per_trade - metrics[cheapest].cost_per_trade
            perf_diff = scores[best].total_score - scores[cheapest].total_score
            if perf_diff < 0.1 and cost_diff > 0.01:
                recs.append(f"Consider {cheapest} for cost savings (${cost_diff:.4f}/trade, only {perf_diff:.2f} score difference)")

        # Reliability concerns
        for model in rankings:
            if metrics[model].success_rate < 0.95:
                recs.append(f"Warning: {model} has low success rate ({metrics[model].success_rate*100:.1f}%)")

        # Latency concerns
        for model in rankings:
            if metrics[model].p95_latency_ms > 10000:
                recs.append(f"Warning: {model} has high latency (p95: {metrics[model].p95_latency_ms:.0f}ms)")

        return recs

    def _generate_summary(
        self,
        metrics: Dict[str, ModelMetrics],
        scores: Dict[str, ModelScore],
        rankings: List[str],
        regime_perf: Dict
    ) -> str:
        """Generate executive summary."""

        best = rankings[0]
        worst = rankings[-1]

        summary_parts = [
            f"**Model Comparison Summary**",
            f"",
            f"Best Performer: {best}",
            f"- Sharpe Ratio: {metrics[best].trading_metrics.sharpe_ratio:.2f}",
            f"- Win Rate: {metrics[best].trading_metrics.win_rate*100:.1f}%",
            f"- Max Drawdown: {metrics[best].trading_metrics.max_drawdown*100:.1f}%",
            f"- Cost/Trade: ${metrics[best].cost_per_trade:.4f}",
            f"",
            f"Worst Performer: {worst}",
            f"- Sharpe Ratio: {metrics[worst].trading_metrics.sharpe_ratio:.2f}",
            f"- Win Rate: {metrics[worst].trading_metrics.win_rate*100:.1f}%",
            f"",
            f"Score Spread: {scores[best].total_score:.3f} to {scores[worst].total_score:.3f}",
        ]

        return "\n".join(summary_parts)
```

---

## 6. Model Selection Algorithm

### 6.1 Dynamic Model Selector

```python
from datetime import datetime, timedelta
from typing import Optional

class DynamicModelSelector:
    """Select optimal model based on recent performance."""

    def __init__(
        self,
        models: List[str],
        lookback_days: int = 7,
        min_trades_for_switch: int = 20,
        performance_threshold: float = 0.1  # 10% score difference to switch
    ):
        self.models = models
        self.lookback_days = lookback_days
        self.min_trades = min_trades_for_switch
        self.threshold = performance_threshold
        self.current_model = models[0]  # Default to first

    async def select_model(
        self,
        regime: Optional[str] = None
    ) -> str:
        """Select best model for current conditions."""

        # Get recent performance
        scores = await self._get_recent_scores()

        if not scores:
            return self.current_model

        # Check if regime-specific selection is better
        if regime:
            regime_best = await self._get_regime_best(regime)
            if regime_best and regime_best != self.current_model:
                # Check if regime specialist significantly outperforms
                if regime_best in scores:
                    improvement = scores[regime_best] - scores.get(self.current_model, 0)
                    if improvement > self.threshold:
                        return regime_best

        # General best model
        best_model = max(scores.keys(), key=lambda m: scores[m])

        # Only switch if improvement is significant
        if best_model != self.current_model:
            improvement = scores[best_model] - scores.get(self.current_model, 0)
            if improvement > self.threshold:
                self.current_model = best_model

        return self.current_model

    async def _get_recent_scores(self) -> Dict[str, float]:
        """Get composite scores for lookback period."""
        # Implementation: Query database for recent trades and calculate scores
        pass

    async def _get_regime_best(self, regime: str) -> Optional[str]:
        """Get best model for specific regime."""
        # Implementation: Query regime-stratified performance
        pass

    def should_switch(
        self,
        current_score: float,
        best_score: float,
        current_trades: int
    ) -> bool:
        """Determine if model switch is warranted."""

        # Need minimum trades for reliable comparison
        if current_trades < self.min_trades:
            return False

        # Score difference must exceed threshold
        improvement = best_score - current_score
        if improvement < self.threshold:
            return False

        return True
```

### 6.2 Ensemble Decision Maker

```python
class EnsembleDecisionMaker:
    """Combine multiple model outputs for more robust decisions."""

    def __init__(
        self,
        models: List[str],
        weights: Optional[Dict[str, float]] = None,
        consensus_threshold: float = 0.6
    ):
        self.models = models
        self.weights = weights or {m: 1/len(models) for m in models}
        self.consensus_threshold = consensus_threshold

    def combine_decisions(
        self,
        decisions: Dict[str, TradingDecision]
    ) -> TradingDecision:
        """Combine multiple model decisions into one."""

        valid_decisions = {
            m: d for m, d in decisions.items()
            if d is not None and d.confidence >= 0.6
        }

        if not valid_decisions:
            return self._default_hold()

        # Weighted voting for action
        action_scores = {'LONG': 0, 'SHORT': 0, 'HOLD': 0, 'CLOSE': 0}
        total_weight = 0

        for model, decision in valid_decisions.items():
            weight = self.weights.get(model, 0)
            weighted_conf = decision.confidence * weight
            action_scores[decision.action] += weighted_conf
            total_weight += weighted_conf

        # Normalize
        if total_weight > 0:
            action_scores = {a: s/total_weight for a, s in action_scores.items()}

        # Find consensus
        best_action = max(action_scores.keys(), key=lambda a: action_scores[a])
        best_score = action_scores[best_action]

        if best_score < self.consensus_threshold:
            return self._default_hold()

        # Get parameters from highest-confidence agreeing decision
        agreeing = [
            d for m, d in valid_decisions.items()
            if d.action == best_action
        ]
        primary = max(agreeing, key=lambda d: d.confidence)

        return TradingDecision(
            action=best_action,
            symbol=primary.symbol,
            confidence=best_score,
            position_size_pct=primary.position_size_pct,
            leverage=primary.leverage,
            entry_price=primary.entry_price,
            stop_loss=primary.stop_loss,
            take_profit=primary.take_profit,
            invalidation=primary.invalidation,
            reasoning=f"Ensemble decision ({len(agreeing)}/{len(valid_decisions)} agree): {primary.reasoning}",
            model_used="ensemble"
        )

    def _default_hold(self) -> TradingDecision:
        return TradingDecision(
            action='HOLD',
            symbol='',
            confidence=0.0,
            reasoning="No consensus among models",
            model_used="ensemble"
        )
```

---

## 7. Reporting System

### 7.1 Daily Report Template

```python
def generate_daily_report(
    date: datetime.date,
    model_metrics: Dict[str, ModelMetrics]
) -> str:
    """Generate daily performance report."""

    report = f"""
# TripleGain Daily LLM Performance Report
**Date:** {date.strftime('%Y-%m-%d')}

## Summary
| Model | Trades | Win Rate | P&L | Sharpe | Cost |
|-------|--------|----------|-----|--------|------|
"""

    for model, m in sorted(model_metrics.items(), key=lambda x: x[1].trading_metrics.total_pnl, reverse=True):
        report += f"| {model} | {m.trading_metrics.total_trades} | {m.trading_metrics.win_rate*100:.1f}% | ${m.trading_metrics.total_pnl:.2f} | {m.trading_metrics.sharpe_ratio:.2f} | ${m.total_cost_usd:.4f} |\n"

    report += """
## Top Performer Analysis
"""

    best = max(model_metrics.items(), key=lambda x: x[1].trading_metrics.total_pnl)
    report += f"- Best: **{best[0]}** with ${best[1].trading_metrics.total_pnl:.2f} P&L\n"

    report += """
## Operational Health
"""

    for model, m in model_metrics.items():
        status = "OK" if m.success_rate > 0.95 else "WARNING"
        report += f"- {model}: {status} (Success: {m.success_rate*100:.1f}%, Avg Latency: {m.avg_latency_ms:.0f}ms)\n"

    return report
```

### 7.2 Grafana Dashboard Queries

```sql
-- Model comparison panel (Sharpe Ratio over time)
SELECT
    date_trunc('day', timestamp) as time,
    model,
    sharpe_ratio
FROM model_performance_daily
WHERE date >= NOW() - INTERVAL '30 days'
ORDER BY time, model;

-- Win rate comparison
SELECT
    model,
    SUM(winning_trades)::float / NULLIF(SUM(executed_trades), 0) as win_rate
FROM model_performance_daily
WHERE date >= NOW() - INTERVAL '7 days'
GROUP BY model
ORDER BY win_rate DESC;

-- Cost efficiency
SELECT
    model,
    SUM(total_pnl) / NULLIF(SUM(total_cost_usd), 0) as pnl_per_dollar
FROM model_performance_daily
WHERE date >= NOW() - INTERVAL '30 days'
GROUP BY model
ORDER BY pnl_per_dollar DESC;

-- Latency percentiles
SELECT
    model,
    percentile_cont(0.50) WITHIN GROUP (ORDER BY latency_ms) as p50,
    percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms) as p95,
    percentile_cont(0.99) WITHIN GROUP (ORDER BY latency_ms) as p99
FROM llm_api_logs
WHERE timestamp >= NOW() - INTERVAL '24 hours'
GROUP BY model;

-- Regime performance heatmap
SELECT
    model,
    regime_at_entry as regime,
    AVG(pnl_pct) * 100 as avg_pnl_pct,
    COUNT(*) as trade_count
FROM trade_results
WHERE timestamp >= NOW() - INTERVAL '30 days'
GROUP BY model, regime_at_entry
ORDER BY model, regime;
```

---

## Appendix: Evaluation Checklist

### Before Going Live with Model Selection

- [ ] Minimum 100 trades per model in paper trading
- [ ] Statistical significance of performance difference
- [ ] Parse success rate > 95% for selected model
- [ ] Latency p95 < 10 seconds
- [ ] Cost per trade acceptable
- [ ] Performance validated across at least 2 market regimes
- [ ] Confidence calibration validated
- [ ] Fallback model tested and ready

### Ongoing Monitoring

- [ ] Daily report reviewed
- [ ] Weekly comparison analysis
- [ ] Monthly strategy review
- [ ] Quarterly model re-evaluation

---

*Document Version: 1.0*
*Last Updated: December 2025*
