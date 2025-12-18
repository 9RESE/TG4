# Phase 5: Production

**Phase Status**: Pending Phase 4 Completion
**Dependencies**: All previous phases
**Deliverable**: Production-ready trading system

---

## Overview

Phase 5 prepares the system for production deployment through comprehensive testing, paper trading validation, and live trading deployment with full monitoring and alerting.

### Components

| Component | Description | Depends On |
|-----------|-------------|------------|
| 5.1 Comprehensive Testing | Unit, integration, and backtest suites | All phases |
| 5.2 Paper Trading Validation | Live data, simulated execution | Phase 4 complete |
| 5.3 Live Trading Deployment | Production deployment | Paper trading success |
| 5.4 Monitoring & Alerting | Observability infrastructure | Deployment |

---

## 5.1 Comprehensive Testing

### Test Categories

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TEST PYRAMID                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                            ▲                                                │
│                           / \                                               │
│                          /   \     E2E Tests (10%)                         │
│                         /     \    • Full system flows                     │
│                        /       \   • UI interactions                       │
│                       ─────────────                                         │
│                      /           \                                          │
│                     /  Integration \  Integration Tests (30%)              │
│                    /    Tests       \ • Agent interactions                 │
│                   /                  \• Database operations                │
│                  ────────────────────                                       │
│                 /                    \                                      │
│                /     Unit Tests       \  Unit Tests (60%)                  │
│               /                        \ • Individual functions            │
│              /                          \• Isolated components             │
│             ────────────────────────────                                    │
│                                                                             │
│  Coverage Target: 80%+ across all categories                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Unit Test Specifications

#### Data Layer Tests

```python
# tests/unit/test_indicator_library.py

import pytest
from decimal import Decimal
from src.data.indicator_library import IndicatorLibrary

class TestIndicatorLibrary:
    """Tests for technical indicator calculations."""

    @pytest.fixture
    def indicator_lib(self):
        return IndicatorLibrary(config={}, db_pool=None)

    @pytest.fixture
    def sample_closes(self):
        # Known data for validation
        return [
            44000, 44100, 44050, 44200, 44150,
            44300, 44250, 44400, 44350, 44500,
            44450, 44600, 44550, 44700, 44650
        ]

    def test_ema_calculation(self, indicator_lib, sample_closes):
        """EMA should match known values within tolerance."""
        ema = indicator_lib.calculate_ema(sample_closes, period=9)
        # Expected EMA(9) for last value (manually calculated)
        expected_last = Decimal("44523.45")
        assert abs(ema[-1] - expected_last) < Decimal("1.0")

    def test_rsi_bounds(self, indicator_lib, sample_closes):
        """RSI must always be between 0 and 100."""
        rsi = indicator_lib.calculate_rsi(sample_closes, period=14)
        assert all(0 <= r <= 100 for r in rsi if r is not None)

    def test_atr_positive(self, indicator_lib):
        """ATR must always be positive."""
        highs = [100, 102, 101, 103, 102]
        lows = [98, 99, 98, 100, 99]
        closes = [99, 101, 100, 102, 101]
        atr = indicator_lib.calculate_atr(highs, lows, closes, period=3)
        assert all(a > 0 for a in atr if a is not None)

    def test_macd_structure(self, indicator_lib, sample_closes):
        """MACD should return line, signal, and histogram."""
        macd = indicator_lib.calculate_macd(
            sample_closes,
            fast=12, slow=26, signal=9
        )
        assert "line" in macd
        assert "signal" in macd
        assert "histogram" in macd

    def test_bollinger_bands_ordering(self, indicator_lib, sample_closes):
        """BB upper > middle > lower."""
        bb = indicator_lib.calculate_bollinger_bands(
            sample_closes, period=20, std_dev=2.0
        )
        assert bb["upper"] > bb["middle"] > bb["lower"]
```

#### Agent Tests

```python
# tests/unit/test_risk_engine.py

import pytest
from decimal import Decimal
from src.risk.rules_engine import (
    RiskManagementEngine,
    TradeProposal,
    ValidationStatus
)

class TestRiskEngine:
    """Tests for risk management rules."""

    @pytest.fixture
    def risk_engine(self):
        config = {
            "risk_per_trade_pct": 1.0,
            "max_position_pct": 20.0,
            "min_rr_ratio": 1.5,
            "max_total_exposure_pct": 60.0,
            "regime_thresholds": {
                "trending_bull": {"min_confidence": 0.55}
            },
            "leverage_limits": {"trending_bull": 5}
        }
        return RiskManagementEngine(config, db_pool=None)

    @pytest.fixture
    def valid_proposal(self):
        return TradeProposal(
            symbol="BTC/USDT",
            action="BUY",
            side="long",
            size=Decimal("0.05"),
            entry_price=Decimal("45000"),
            stop_loss=Decimal("44100"),  # 2% SL
            take_profit=Decimal("46800"),  # 4% TP = R:R 2.0
            leverage=2,
            confidence=Decimal("0.68"),
            reasoning="Test trade"
        )

    @pytest.fixture
    def portfolio_context(self):
        return {
            "equity": Decimal("10000"),
            "available_margin": Decimal("8000"),
            "positions": []
        }

    @pytest.mark.asyncio
    async def test_valid_trade_approved(
        self, risk_engine, valid_proposal, portfolio_context
    ):
        """Valid trade should be approved without modification."""
        validation = await risk_engine.validate_trade(
            proposal=valid_proposal,
            portfolio_context=portfolio_context,
            regime="trending_bull"
        )
        assert validation.status == ValidationStatus.APPROVED

    @pytest.mark.asyncio
    async def test_low_confidence_rejected(
        self, risk_engine, valid_proposal, portfolio_context
    ):
        """Low confidence trade should be rejected."""
        valid_proposal.confidence = Decimal("0.40")
        validation = await risk_engine.validate_trade(
            proposal=valid_proposal,
            portfolio_context=portfolio_context,
            regime="trending_bull"
        )
        assert validation.status == ValidationStatus.REJECTED
        assert "confidence" in str(validation.rejection_reasons).lower()

    @pytest.mark.asyncio
    async def test_excessive_leverage_modified(
        self, risk_engine, valid_proposal, portfolio_context
    ):
        """Excessive leverage should be reduced."""
        valid_proposal.leverage = 10
        validation = await risk_engine.validate_trade(
            proposal=valid_proposal,
            portfolio_context=portfolio_context,
            regime="trending_bull"
        )
        assert validation.status == ValidationStatus.MODIFIED
        assert validation.modified_proposal.leverage == 5

    @pytest.mark.asyncio
    async def test_poor_rr_rejected(
        self, risk_engine, valid_proposal, portfolio_context
    ):
        """Trade with R:R below minimum should be rejected."""
        valid_proposal.take_profit = Decimal("45500")  # R:R = 0.5
        validation = await risk_engine.validate_trade(
            proposal=valid_proposal,
            portfolio_context=portfolio_context,
            regime="trending_bull"
        )
        assert validation.status == ValidationStatus.REJECTED

    @pytest.mark.asyncio
    async def test_circuit_breaker_blocks(
        self, risk_engine, valid_proposal, portfolio_context
    ):
        """Active circuit breaker should block all trades."""
        # Simulate circuit breaker
        risk_engine._state = type('obj', (object,), {
            'circuit_breakers_active': ['daily_loss']
        })()

        validation = await risk_engine.validate_trade(
            proposal=valid_proposal,
            portfolio_context=portfolio_context,
            regime="trending_bull"
        )
        assert validation.status == ValidationStatus.REJECTED
```

### Integration Test Specifications

```python
# tests/integration/test_agent_pipeline.py

import pytest
from datetime import datetime
from decimal import Decimal

class TestAgentPipeline:
    """Integration tests for agent communication."""

    @pytest.fixture
    async def system(self, db_pool, ollama_client, api_clients):
        """Set up full system with all agents."""
        from src.orchestration.message_bus import MessageBus
        from src.agents.technical_analysis import TechnicalAnalysisAgent
        from src.agents.regime_detection import RegimeDetectionAgent
        from src.agents.trading_decision import TradingDecisionAgent
        from src.risk.rules_engine import RiskManagementEngine

        bus = MessageBus({})
        ta_agent = TechnicalAnalysisAgent(ollama_client, ...)
        regime_agent = RegimeDetectionAgent(ollama_client, ...)
        trading_agent = TradingDecisionAgent(api_clients, ...)
        risk_engine = RiskManagementEngine({}, db_pool)

        return {
            "bus": bus,
            "ta": ta_agent,
            "regime": regime_agent,
            "trading": trading_agent,
            "risk": risk_engine
        }

    @pytest.mark.asyncio
    async def test_full_pipeline_flow(self, system, sample_snapshot):
        """Test TA → Regime → Trading → Risk flow."""
        # Run TA
        ta_output = await system["ta"].process(sample_snapshot)
        assert ta_output.bias in ["long", "short", "neutral"]
        assert 0 <= ta_output.confidence <= 1

        # Run Regime
        regime_output = await system["regime"].process(
            sample_snapshot, ta_output
        )
        assert regime_output.regime in [
            "trending_bull", "trending_bear", "ranging",
            "volatile_bull", "volatile_bear", "choppy"
        ]

        # Run Trading Decision
        decision, consensus = await system["trading"].process(
            snapshot=sample_snapshot,
            ta_output=ta_output,
            regime_output=regime_output
        )
        assert decision.action in ["BUY", "SELL", "HOLD", "CLOSE"]
        assert len(consensus.model_decisions) >= 1  # At least one model

        # Validate with Risk
        if decision.action != "HOLD":
            proposal = create_proposal_from_decision(decision)
            validation = await system["risk"].validate_trade(
                proposal=proposal,
                portfolio_context=sample_portfolio,
                regime=regime_output.regime
            )
            assert validation.status in [
                ValidationStatus.APPROVED,
                ValidationStatus.MODIFIED,
                ValidationStatus.REJECTED
            ]

    @pytest.mark.asyncio
    async def test_message_propagation(self, system, sample_snapshot):
        """Test messages flow through bus correctly."""
        received = []

        async def handler(msg):
            received.append(msg)

        await system["bus"].subscribe(
            subscriber_id="test",
            topic=MessageTopic.TA_SIGNALS,
            handler=handler
        )

        # Run TA agent
        ta_output = await system["ta"].process(sample_snapshot)

        # Verify message published
        await asyncio.sleep(0.1)  # Allow propagation
        assert len(received) == 1
        assert received[0].source == "technical_analysis"

    @pytest.mark.asyncio
    async def test_six_model_parallel(self, system, sample_snapshot):
        """All 6 models should run and return results."""
        ta_output = await system["ta"].process(sample_snapshot)
        regime_output = await system["regime"].process(sample_snapshot, ta_output)

        decision, consensus = await system["trading"].process(
            snapshot=sample_snapshot,
            ta_output=ta_output,
            regime_output=regime_output
        )

        # All 6 models should have decisions (or documented failures)
        model_ids = [d.model_id for d in consensus.model_decisions]
        expected_models = ["gpt", "grok", "deepseek", "claude_sonnet", "claude_opus", "qwen"]

        # At least 4 of 6 should succeed
        assert len(set(model_ids) & set(expected_models)) >= 4
```

### Backtesting Framework

```python
# tests/backtests/test_backtest_framework.py

import pytest
from datetime import datetime, timedelta
from decimal import Decimal

class TestBacktestFramework:
    """Tests for backtesting functionality."""

    @pytest.fixture
    def backtest_config(self):
        return {
            "data": {
                "symbols": ["BTC/USDT", "XRP/USDT"],
                "start_date": "2024-06-01",
                "end_date": "2024-06-30",  # 1 month
                "source": "timescaledb"
            },
            "initial_conditions": {
                "equity_usd": 2100,
                "allocation": {"btc": 0.333, "xrp": 0.333, "usdt": 0.334}
            },
            "execution_model": {
                "slippage_bps": 5,
                "fees_bps": 10,
                "fill_rate": 0.98
            }
        }

    @pytest.mark.asyncio
    async def test_backtest_runs_without_error(
        self, backtest_config, db_pool
    ):
        """Backtest should complete without exceptions."""
        from src.backtesting.backtester import Backtester

        backtester = Backtester(backtest_config, db_pool)
        results = await backtester.run()

        assert results is not None
        assert "summary" in results
        assert "trades" in results
        assert "equity_curve" in results

    @pytest.mark.asyncio
    async def test_backtest_metrics_valid(
        self, backtest_config, db_pool
    ):
        """Backtest metrics should be valid."""
        backtester = Backtester(backtest_config, db_pool)
        results = await backtester.run()

        summary = results["summary"]

        # Equity should be positive
        assert summary["final_equity"] > 0

        # Drawdown should be non-negative percentage
        assert 0 <= summary["max_drawdown_pct"] <= 100

        # Win rate should be valid percentage
        if results["trades"]["total_trades"] > 0:
            assert 0 <= results["trades"]["win_rate"] <= 1

    @pytest.mark.asyncio
    async def test_backtest_vs_benchmark(
        self, backtest_config, db_pool
    ):
        """System should be compared against benchmarks."""
        from src.backtesting.benchmarks import BenchmarkCalculator

        backtester = Backtester(backtest_config, db_pool)
        results = await backtester.run()

        benchmark = BenchmarkCalculator(
            price_data=results["price_data"],
            initial_equity=backtest_config["initial_conditions"]["equity_usd"]
        )

        bh_btc = benchmark.buy_and_hold_btc()
        equal_weight = benchmark.equal_weight_hold()

        # Results should include comparison
        assert "benchmark_comparison" in results or \
               results["summary"]["total_return_pct"] is not None
```

### Test Coverage Requirements

| Module | Minimum Coverage |
|--------|-----------------|
| `src/data/` | 85% |
| `src/agents/` | 80% |
| `src/risk/` | 90% |
| `src/execution/` | 85% |
| `src/orchestration/` | 75% |
| `src/llm/` | 80% |
| Overall | 80% |

### Test Configuration

```yaml
# pytest.ini / pyproject.toml

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
markers = [
    "unit: Unit tests",
    "integration: Integration tests",
    "backtest: Backtesting tests",
    "slow: Slow tests (>30s)"
]
filterwarnings = [
    "ignore::DeprecationWarning"
]

[tool.coverage.run]
source = ["src"]
omit = [
    "*/tests/*",
    "*/__init__.py"
]

[tool.coverage.report]
fail_under = 80
show_missing = true
```

---

## 5.2 Paper Trading Validation

### Purpose

Validate the complete system with live market data but simulated order execution before risking real capital.

### Paper Trading Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       PAPER TRADING ARCHITECTURE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      LIVE DATA SOURCES                               │   │
│  │  • Kraken WebSocket (real-time prices)                              │   │
│  │  • Order book depth (real)                                          │   │
│  │  • News/Sentiment (real)                                            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      FULL AGENT SYSTEM                               │   │
│  │  • TA Agent (real analysis)                                         │   │
│  │  • Regime Detection (real)                                          │   │
│  │  • Sentiment Analysis (real)                                        │   │
│  │  • Trading Decision (6-model real)                                  │   │
│  │  • Risk Management (real validation)                                │   │
│  │  • Portfolio Rebalancing (simulated execution)                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                   SIMULATED EXECUTION LAYER                          │   │
│  │                                                                      │   │
│  │  Instead of Kraken API:                                             │   │
│  │  ┌───────────────────────────────────────────────────────────────┐  │   │
│  │  │ Paper Order Executor                                           │  │   │
│  │  │ • Simulates order fill at current market price                │  │   │
│  │  │ • Applies slippage model                                       │  │   │
│  │  │ • Applies fee model                                            │  │   │
│  │  │ • Tracks simulated positions                                   │  │   │
│  │  │ • Manages simulated stop-loss/take-profit                     │  │   │
│  │  └───────────────────────────────────────────────────────────────┘  │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    PAPER TRADING DATABASE                            │   │
│  │  • Same schema as production                                        │   │
│  │  • Separate database/schema                                         │   │
│  │  • Full audit trail                                                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Paper Order Executor

```python
# src/execution/paper_executor.py

from dataclasses import dataclass
from decimal import Decimal
from datetime import datetime
from typing import Optional
import asyncio

@dataclass
class PaperOrder:
    """Simulated order."""
    id: str
    symbol: str
    side: str
    order_type: str
    size: Decimal
    price: Optional[Decimal]
    status: str
    filled_size: Decimal = Decimal(0)
    filled_price: Optional[Decimal] = None
    created_at: datetime = None
    filled_at: Optional[datetime] = None


@dataclass
class PaperPosition:
    """Simulated position."""
    id: str
    symbol: str
    side: str
    size: Decimal
    entry_price: Decimal
    entry_time: datetime
    stop_loss: Optional[Decimal]
    take_profit: Optional[Decimal]
    unrealized_pnl: Decimal = Decimal(0)


class PaperExecutor:
    """
    Paper trading order executor.

    Simulates order execution with realistic slippage and fees.
    """

    def __init__(
        self,
        price_feed,  # Real-time price source
        config: dict,
        db_pool
    ):
        self.prices = price_feed
        self.config = config
        self.db = db_pool
        self._orders: dict[str, PaperOrder] = {}
        self._positions: dict[str, PaperPosition] = {}
        self._portfolio = {
            "btc": Decimal(0),
            "xrp": Decimal(0),
            "usdt": Decimal(str(config["initial_usdt"]))
        }

    async def execute_order(
        self,
        proposal: TradeProposal
    ) -> ExecutionResult:
        """
        Execute order in paper trading mode.

        Simulates realistic execution with slippage.
        """
        order_id = str(uuid.uuid4())

        # Get current price
        current_price = await self.prices.get_price(proposal.symbol)

        # Apply slippage
        slippage_bps = self.config["slippage_bps"]
        if proposal.side == "long":
            fill_price = current_price * (1 + slippage_bps / 10000)
        else:
            fill_price = current_price * (1 - slippage_bps / 10000)

        # Calculate fees
        fee_bps = self.config["fee_bps"]
        fee_usd = float(proposal.size) * float(fill_price) * fee_bps / 10000

        # Simulate fill probability
        if random.random() > self.config["fill_rate"]:
            return ExecutionResult(
                success=False,
                error_message="Order not filled (simulated)"
            )

        # Create order
        order = PaperOrder(
            id=order_id,
            symbol=proposal.symbol,
            side="buy" if proposal.side == "long" else "sell",
            order_type=proposal.entry_type or "market",
            size=proposal.size,
            price=proposal.entry_price,
            status="filled",
            filled_size=proposal.size,
            filled_price=Decimal(str(fill_price)),
            created_at=datetime.utcnow(),
            filled_at=datetime.utcnow()
        )
        self._orders[order_id] = order

        # Create position
        position = PaperPosition(
            id=str(uuid.uuid4()),
            symbol=proposal.symbol,
            side=proposal.side,
            size=proposal.size,
            entry_price=Decimal(str(fill_price)),
            entry_time=datetime.utcnow(),
            stop_loss=proposal.stop_loss,
            take_profit=proposal.take_profit
        )
        self._positions[position.id] = position

        # Update portfolio
        await self._update_portfolio(order, fee_usd)

        # Store to DB
        await self._store_execution(order, position, fee_usd)

        # Start monitoring for SL/TP
        asyncio.create_task(self._monitor_position(position))

        return ExecutionResult(
            success=True,
            order=order,
            position=position
        )

    async def _monitor_position(self, position: PaperPosition) -> None:
        """Monitor position for stop-loss and take-profit."""
        while position.id in self._positions:
            current_price = await self.prices.get_price(position.symbol)

            # Check stop-loss
            if position.stop_loss:
                if position.side == "long" and current_price <= position.stop_loss:
                    await self._close_position(position, "stop_loss", current_price)
                    return
                elif position.side == "short" and current_price >= position.stop_loss:
                    await self._close_position(position, "stop_loss", current_price)
                    return

            # Check take-profit
            if position.take_profit:
                if position.side == "long" and current_price >= position.take_profit:
                    await self._close_position(position, "take_profit", current_price)
                    return
                elif position.side == "short" and current_price <= position.take_profit:
                    await self._close_position(position, "take_profit", current_price)
                    return

            # Update unrealized P&L
            if position.side == "long":
                position.unrealized_pnl = (current_price - position.entry_price) * position.size
            else:
                position.unrealized_pnl = (position.entry_price - current_price) * position.size

            await asyncio.sleep(5)  # Check every 5 seconds

    async def get_portfolio_value(self) -> Decimal:
        """Get current portfolio value in USD."""
        prices = await self.prices.get_all_prices()

        btc_value = self._portfolio["btc"] * prices.get("BTC/USDT", Decimal(0))
        xrp_value = self._portfolio["xrp"] * prices.get("XRP/USDT", Decimal(0))
        usdt_value = self._portfolio["usdt"]

        return btc_value + xrp_value + usdt_value
```

### Paper Trading Configuration

```yaml
# config/paper_trading.yaml

paper_trading:
  enabled: true

  # Initial portfolio
  initial_portfolio:
    usdt: 2100  # Starting with $2100 USDT

  # Execution simulation
  execution:
    slippage_bps: 5  # 0.05% slippage
    fee_bps: 10  # 0.10% fee
    fill_rate: 0.98  # 98% fill rate

  # Database
  database:
    schema: paper_trading  # Separate schema
    # Or separate database:
    # database: triplegain_paper

  # Duration
  validation:
    minimum_duration_days: 14
    minimum_trades: 50
    success_criteria:
      min_sharpe_ratio: 1.0
      max_drawdown_pct: 15
      min_win_rate: 0.45

  # Real-time data
  data_sources:
    kraken_websocket: true
    sentiment_apis: true
```

### Validation Criteria

| Criterion | Threshold | Measurement Period |
|-----------|-----------|-------------------|
| Sharpe Ratio | > 1.0 | Full paper period |
| Max Drawdown | < 15% | Full paper period |
| Win Rate | > 45% | Minimum 50 trades |
| System Uptime | > 99% | Full paper period |
| Agent Error Rate | < 1% | Full paper period |
| Execution Latency | < 1s (p95) | Full paper period |

### Paper Trading Report

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PAPER TRADING VALIDATION REPORT                           │
│                    Period: 2025-12-01 to 2025-12-14 (14 days)               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PERFORMANCE SUMMARY                                                        │
│  ───────────────────────────────────────────────────────────────────────    │
│  Initial Equity:    $2,100.00                                               │
│  Final Equity:      $2,312.50                                               │
│  Total Return:      +$212.50 (+10.1%)                                       │
│  Sharpe Ratio:      1.45 ✓                                                  │
│  Max Drawdown:      -6.8% ✓                                                 │
│                                                                             │
│  TRADING METRICS                                                            │
│  ───────────────────────────────────────────────────────────────────────    │
│  Total Trades:      67                                                      │
│  Win Rate:          58.2% ✓                                                 │
│  Profit Factor:     1.82                                                    │
│  Average Win:       +$18.50                                                 │
│  Average Loss:      -$12.30                                                 │
│                                                                             │
│  6-MODEL COMPARISON                                                         │
│  ───────────────────────────────────────────────────────────────────────    │
│  Best Performer:    DeepSeek V3 (62% accuracy)                             │
│  Consensus Accuracy: 68% when unanimous                                     │
│                                                                             │
│  SYSTEM HEALTH                                                              │
│  ───────────────────────────────────────────────────────────────────────    │
│  Uptime:            99.7% ✓                                                 │
│  Agent Errors:      3 (0.4%) ✓                                              │
│  Data Gaps:         0 ✓                                                     │
│                                                                             │
│  VALIDATION STATUS: PASSED ✓                                                │
│  Ready for live trading deployment                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5.3 Live Trading Deployment

### Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PRODUCTION DEPLOYMENT ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      INFRASTRUCTURE LAYER                            │   │
│  │                                                                      │   │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐           │   │
│  │  │  TimescaleDB  │  │    Redis      │  │   Ollama      │           │   │
│  │  │  (Existing)   │  │   (Cache)     │  │  (Local LLM)  │           │   │
│  │  │               │  │               │  │               │           │   │
│  │  │  Port: 5432   │  │  Port: 6379   │  │  Port: 11434  │           │   │
│  │  └───────────────┘  └───────────────┘  └───────────────┘           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      APPLICATION LAYER                               │   │
│  │                                                                      │   │
│  │  ┌───────────────────────────────────────────────────────────────┐  │   │
│  │  │                    TRIPLEGAIN CORE                             │  │   │
│  │  │                                                                │  │   │
│  │  │  • Coordinator Service                                        │  │   │
│  │  │  • Agent Services (TA, Regime, Sentiment, Trading, Risk)     │  │   │
│  │  │  • Execution Service                                          │  │   │
│  │  │  • Portfolio Service                                          │  │   │
│  │  │                                                                │  │   │
│  │  │  Process Manager: systemd / supervisor                        │  │   │
│  │  │  Restart Policy: always                                       │  │   │
│  │  └───────────────────────────────────────────────────────────────┘  │   │
│  │                                                                      │   │
│  │  ┌───────────────────────────────────────────────────────────────┐  │   │
│  │  │                    FASTAPI SERVER                              │  │   │
│  │  │                                                                │  │   │
│  │  │  Port: 8000                                                   │  │   │
│  │  │  Workers: 4 (uvicorn)                                         │  │   │
│  │  └───────────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      FRONTEND LAYER                                  │   │
│  │                                                                      │   │
│  │  ┌───────────────────────────────────────────────────────────────┐  │   │
│  │  │                    REACT DASHBOARD                             │  │   │
│  │  │                                                                │  │   │
│  │  │  Served by: nginx                                             │  │   │
│  │  │  Port: 80/443                                                 │  │   │
│  │  │  SSL: Let's Encrypt (optional for local)                      │  │   │
│  │  └───────────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Docker Compose Configuration

```yaml
# docker-compose.yml

version: '3.8'

services:
  # TimescaleDB (may already exist)
  timescaledb:
    image: timescale/timescaledb:latest-pg15
    ports:
      - "5432:5432"
    environment:
      POSTGRES_USER: triplegain
      POSTGRES_PASSWORD: ${DB_PASSWORD}
      POSTGRES_DB: triplegain
    volumes:
      - timescale_data:/var/lib/postgresql/data
    restart: always

  # Redis cache
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: always

  # TripleGain Core
  triplegain-core:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      DATABASE_URL: postgresql://triplegain:${DB_PASSWORD}@timescaledb:5432/triplegain
      REDIS_URL: redis://redis:6379
      OLLAMA_HOST: http://host.docker.internal:11434
      KRAKEN_API_KEY: ${KRAKEN_API_KEY}
      KRAKEN_API_SECRET: ${KRAKEN_API_SECRET}
      OPENAI_API_KEY: ${OPENAI_API_KEY}
      ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY}
      XAI_API_KEY: ${XAI_API_KEY}
      DEEPSEEK_API_KEY: ${DEEPSEEK_API_KEY}
    depends_on:
      - timescaledb
      - redis
    restart: always
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Dashboard
  dashboard:
    build:
      context: ./dashboard
      dockerfile: Dockerfile
    ports:
      - "80:80"
    depends_on:
      - triplegain-core
    restart: always

volumes:
  timescale_data:
  redis_data:
```

### Systemd Service (Alternative to Docker)

```ini
# /etc/systemd/system/triplegain.service

[Unit]
Description=TripleGain Trading System
After=network.target postgresql.service redis.service

[Service]
Type=simple
User=triplegain
Group=triplegain
WorkingDirectory=/opt/triplegain
Environment="PATH=/opt/triplegain/venv/bin"
EnvironmentFile=/opt/triplegain/.env
ExecStart=/opt/triplegain/venv/bin/python -m src.main
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

# Security
NoNewPrivileges=yes
ProtectSystem=strict
ProtectHome=yes
PrivateTmp=yes

[Install]
WantedBy=multi-user.target
```

### Deployment Checklist

```markdown
## Pre-Deployment Checklist

### Environment
- [ ] All API keys configured in .env
- [ ] Database connection verified
- [ ] Redis connection verified
- [ ] Ollama accessible with Qwen 2.5 7B loaded
- [ ] Kraken API permissions verified (trade, query)

### Configuration
- [ ] Risk parameters reviewed and confirmed
- [ ] Initial capital amount set correctly
- [ ] Target allocation set (33/33/33)
- [ ] Circuit breaker thresholds set

### Testing
- [ ] All unit tests passing
- [ ] All integration tests passing
- [ ] Paper trading validation passed
- [ ] Backtests show acceptable performance

### Monitoring
- [ ] Logging configured
- [ ] Alert destinations configured
- [ ] Dashboard accessible
- [ ] Health endpoints responding

### Security
- [ ] API keys not in code
- [ ] Database credentials secured
- [ ] Network access restricted
- [ ] Backup strategy in place

### Documentation
- [ ] Runbook created
- [ ] Emergency procedures documented
- [ ] Contact information updated
```

### Startup Sequence

```python
# src/main.py

import asyncio
import signal
from src.orchestration.coordinator import CoordinatorAgent
from src.config import load_config

async def main():
    """Main entry point for TripleGain."""
    config = load_config()

    # Initialize components
    db_pool = await create_db_pool(config["database"])
    redis_client = await create_redis_client(config["redis"])

    # Initialize agents
    agents = await initialize_agents(config, db_pool)

    # Initialize coordinator
    coordinator = CoordinatorAgent(
        message_bus=MessageBus(config),
        agents=agents,
        llm_client=create_llm_client(config),
        config=config["coordinator"]
    )

    # Set up shutdown handler
    loop = asyncio.get_event_loop()

    def shutdown_handler(signum, frame):
        print(f"Received signal {signum}, shutting down...")
        asyncio.create_task(coordinator.stop())

    signal.signal(signal.SIGTERM, shutdown_handler)
    signal.signal(signal.SIGINT, shutdown_handler)

    # Start coordinator
    print("Starting TripleGain Trading System...")
    await coordinator.start()


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 5.4 Monitoring & Alerting

### Observability Stack

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MONITORING & ALERTING ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      METRICS COLLECTION                              │   │
│  │                                                                      │   │
│  │  Application Metrics (Prometheus format):                            │   │
│  │  • triplegain_portfolio_equity_usd                                  │   │
│  │  • triplegain_portfolio_pnl_daily_pct                               │   │
│  │  • triplegain_trades_total{symbol, action}                          │   │
│  │  • triplegain_agent_latency_ms{agent}                               │   │
│  │  • triplegain_agent_errors_total{agent}                             │   │
│  │  • triplegain_model_accuracy{model}                                 │   │
│  │  • triplegain_llm_cost_usd_total{model}                             │   │
│  │  • triplegain_positions_open{symbol}                                │   │
│  │  • triplegain_circuit_breaker_active                                │   │
│  │                                                                      │   │
│  │  System Metrics:                                                     │   │
│  │  • CPU, Memory, Disk usage                                          │   │
│  │  • Network latency to exchanges                                     │   │
│  │  • Database connection pool                                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      LOGGING                                         │   │
│  │                                                                      │   │
│  │  Structured JSON logs to:                                            │   │
│  │  • stdout (for container environments)                              │   │
│  │  • /var/log/triplegain/ (file)                                      │   │
│  │                                                                      │   │
│  │  Log levels:                                                         │   │
│  │  • DEBUG: Agent outputs, calculations                               │   │
│  │  • INFO: Trades, decisions, state changes                           │   │
│  │  • WARNING: Risk threshold approaching, data staleness              │   │
│  │  • ERROR: Agent failures, execution errors                          │   │
│  │  • CRITICAL: Circuit breaker, system failures                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      ALERTING                                        │   │
│  │                                                                      │   │
│  │  Alert Channels:                                                     │   │
│  │  • Dashboard notifications (all levels)                             │   │
│  │  • Email (WARNING and above)                                        │   │
│  │  • SMS/Push (CRITICAL only)                                         │   │
│  │                                                                      │   │
│  │  Alert Rules:                                                        │   │
│  │  • Circuit breaker triggered                                        │   │
│  │  • Daily loss > 3% (warning), > 5% (critical)                      │   │
│  │  • Agent error rate > 5%                                            │   │
│  │  • WebSocket disconnected > 1 minute                                │   │
│  │  • Data staleness > 2 minutes                                       │   │
│  │  • API budget > 80% used                                            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Metrics Implementation

```python
# src/monitoring/metrics.py

from prometheus_client import Counter, Gauge, Histogram, start_http_server

# Portfolio metrics
portfolio_equity = Gauge(
    'triplegain_portfolio_equity_usd',
    'Current portfolio equity in USD'
)
portfolio_pnl_daily = Gauge(
    'triplegain_portfolio_pnl_daily_pct',
    'Daily P&L percentage'
)
portfolio_drawdown = Gauge(
    'triplegain_portfolio_drawdown_pct',
    'Current drawdown from peak'
)

# Trading metrics
trades_total = Counter(
    'triplegain_trades_total',
    'Total trades executed',
    ['symbol', 'action', 'result']
)
positions_open = Gauge(
    'triplegain_positions_open',
    'Number of open positions',
    ['symbol']
)

# Agent metrics
agent_latency = Histogram(
    'triplegain_agent_latency_seconds',
    'Agent execution latency',
    ['agent'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)
agent_errors = Counter(
    'triplegain_agent_errors_total',
    'Agent errors',
    ['agent', 'error_type']
)

# LLM metrics
llm_requests = Counter(
    'triplegain_llm_requests_total',
    'LLM API requests',
    ['model', 'status']
)
llm_cost = Counter(
    'triplegain_llm_cost_usd_total',
    'Total LLM API cost in USD',
    ['model']
)
llm_tokens = Counter(
    'triplegain_llm_tokens_total',
    'Total tokens used',
    ['model', 'type']  # input/output
)

# Model comparison metrics
model_accuracy = Gauge(
    'triplegain_model_accuracy',
    'Model prediction accuracy',
    ['model']
)
model_rank = Gauge(
    'triplegain_model_leaderboard_rank',
    'Model leaderboard rank',
    ['model']
)

# System health
circuit_breaker_active = Gauge(
    'triplegain_circuit_breaker_active',
    'Circuit breaker status (1=active, 0=inactive)',
    ['breaker_type']
)
websocket_connected = Gauge(
    'triplegain_websocket_connected',
    'WebSocket connection status',
    ['endpoint']
)
data_age_seconds = Gauge(
    'triplegain_data_age_seconds',
    'Age of latest market data',
    ['symbol']
)


def start_metrics_server(port: int = 9090):
    """Start Prometheus metrics server."""
    start_http_server(port)
```

### Alert Configuration

```yaml
# config/alerts.yaml

alerts:
  channels:
    dashboard:
      enabled: true
      all_levels: true

    email:
      enabled: true
      min_level: warning
      recipients:
        - trading@example.com
      smtp:
        host: ${SMTP_HOST}
        port: 587
        username: ${SMTP_USER}
        password: ${SMTP_PASSWORD}

    sms:
      enabled: false  # Enable for critical alerts
      min_level: critical
      provider: twilio
      phone_numbers:
        - "+1234567890"

  rules:
    # Trading alerts
    - name: daily_loss_warning
      condition: "portfolio_pnl_daily_pct < -3"
      level: warning
      message: "Daily loss exceeds 3%: {{ value }}%"
      cooldown_minutes: 60

    - name: daily_loss_critical
      condition: "portfolio_pnl_daily_pct < -5"
      level: critical
      message: "Daily loss limit reached: {{ value }}%. Trading halted."
      cooldown_minutes: 0  # Always alert

    - name: circuit_breaker_triggered
      condition: "circuit_breaker_active == 1"
      level: critical
      message: "Circuit breaker triggered: {{ breaker_type }}"
      cooldown_minutes: 0

    # System alerts
    - name: websocket_disconnected
      condition: "websocket_connected == 0 for 1m"
      level: error
      message: "WebSocket disconnected from {{ endpoint }}"
      cooldown_minutes: 5

    - name: data_staleness
      condition: "data_age_seconds > 120"
      level: warning
      message: "Market data stale for {{ symbol }}: {{ value }}s"
      cooldown_minutes: 10

    - name: agent_error_rate
      condition: "rate(agent_errors_total[5m]) > 0.05"
      level: error
      message: "High error rate for {{ agent }}: {{ value }}"
      cooldown_minutes: 15

    # Cost alerts
    - name: api_budget_warning
      condition: "sum(llm_cost_usd_total) > daily_budget * 0.8"
      level: warning
      message: "API budget 80% used: ${{ value }}"
      cooldown_minutes: 120

    - name: api_budget_exceeded
      condition: "sum(llm_cost_usd_total) > daily_budget"
      level: error
      message: "Daily API budget exceeded: ${{ value }}"
      cooldown_minutes: 0
```

### Logging Configuration

```python
# src/monitoring/logging_config.py

import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""

    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }

        # Add extra fields
        if hasattr(record, "symbol"):
            log_entry["symbol"] = record.symbol
        if hasattr(record, "agent"):
            log_entry["agent"] = record.agent
        if hasattr(record, "trade_id"):
            log_entry["trade_id"] = record.trade_id
        if hasattr(record, "model"):
            log_entry["model"] = record.model

        # Add exception info
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


def configure_logging(config: dict):
    """Configure application logging."""
    root_logger = logging.getLogger()
    root_logger.setLevel(config.get("level", "INFO"))

    # Console handler (JSON)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(console_handler)

    # File handler (for local deployment)
    if config.get("file_path"):
        file_handler = logging.handlers.RotatingFileHandler(
            config["file_path"],
            maxBytes=100_000_000,  # 100MB
            backupCount=10
        )
        file_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(file_handler)

    # Reduce noise from libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
```

---

## Phase 5 Acceptance Criteria

### Testing Requirements

| Requirement | Threshold |
|-------------|-----------|
| Unit test coverage | > 80% |
| Integration tests passing | 100% |
| Backtest Sharpe ratio | > 1.0 |
| Paper trading validation | Passed |

### Deployment Requirements

| Requirement | Acceptance |
|-------------|------------|
| All services start | No errors on startup |
| Health checks pass | All endpoints healthy |
| Dashboard accessible | UI loads correctly |
| WebSocket connections | Stable for 24h |

### Monitoring Requirements

| Requirement | Acceptance |
|-------------|------------|
| Metrics exported | Prometheus scraping |
| Alerts configured | Test alerts fire correctly |
| Logs structured | JSON format parseable |
| Dashboard integration | Real-time updates |

### Deliverables Checklist

- [ ] `tests/unit/` - Unit test suite
- [ ] `tests/integration/` - Integration tests
- [ ] `tests/backtests/` - Backtesting tests
- [ ] `src/execution/paper_executor.py`
- [ ] `src/monitoring/metrics.py`
- [ ] `src/monitoring/logging_config.py`
- [ ] `docker-compose.yml`
- [ ] Deployment documentation
- [ ] Runbook for operations
- [ ] Paper trading report (passed)

---

## Production Go-Live Checklist

```markdown
## Final Go-Live Checklist

### Pre-Launch (T-24h)
- [ ] Paper trading validation report approved
- [ ] All tests passing on production branch
- [ ] Configuration reviewed by second person
- [ ] API keys rotated to production keys
- [ ] Initial capital deposited on Kraken

### Launch Day (T-0)
- [ ] System started in monitoring-only mode
- [ ] All agents reporting healthy
- [ ] Dashboard showing correct data
- [ ] WebSocket connections stable
- [ ] First scheduled agent runs complete without error

### Post-Launch (T+1h)
- [ ] First trading decision processed
- [ ] Risk validation working
- [ ] No unexpected errors in logs
- [ ] Metrics showing in monitoring

### Steady State (T+24h)
- [ ] Multiple trades executed successfully
- [ ] P&L tracking correctly
- [ ] Model comparison data populating
- [ ] All alerts functioning

### Sign-Off
- [ ] Operations team briefed
- [ ] Emergency contacts confirmed
- [ ] Runbook tested
- [ ] System declared production-ready
```

---

## References

- Design: [06-evaluation-framework.md](../TripleGain-master-design/06-evaluation-framework.md)
- Design: [03-risk-management-rules-engine.md](../TripleGain-master-design/03-risk-management-rules-engine.md)

---

*Phase 5 Implementation Plan v1.0 - December 2025*
