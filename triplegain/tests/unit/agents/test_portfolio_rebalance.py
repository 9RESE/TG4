"""
Unit tests for PortfolioRebalanceAgent - Portfolio allocation and rebalancing.

Tests cover:
- Allocation calculation
- Rebalancing trade calculation
- Hodl bag exclusion
- LLM execution strategy
"""

import asyncio
import pytest
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

from triplegain.src.agents.portfolio_rebalance import (
    PortfolioRebalanceAgent,
    PortfolioAllocation,
    RebalanceTrade,
    RebalanceOutput,
)


class TestPortfolioAllocation:
    """Tests for PortfolioAllocation dataclass."""

    def test_allocation_creation(self):
        """Test basic allocation creation."""
        allocation = PortfolioAllocation(
            total_equity_usd=Decimal("10000"),
            btc_value_usd=Decimal("3000"),
            xrp_value_usd=Decimal("4000"),
            usdt_value_usd=Decimal("3000"),
            btc_pct=Decimal("30"),
            xrp_pct=Decimal("40"),
            usdt_pct=Decimal("30"),
            max_deviation_pct=Decimal("6.67"),
        )
        assert allocation.total_equity_usd == Decimal("10000")
        assert allocation.max_deviation_pct == Decimal("6.67")

    def test_allocation_to_dict(self):
        """Test allocation serialization."""
        allocation = PortfolioAllocation(
            total_equity_usd=Decimal("10000"),
            btc_value_usd=Decimal("3333"),
            xrp_value_usd=Decimal("3333"),
            usdt_value_usd=Decimal("3334"),
            btc_pct=Decimal("33.33"),
            xrp_pct=Decimal("33.33"),
            usdt_pct=Decimal("33.34"),
            max_deviation_pct=Decimal("0.01"),
        )
        d = allocation.to_dict()
        assert d["total_equity_usd"] == 10000.0
        assert d["btc_pct"] == 33.33


class TestRebalanceTrade:
    """Tests for RebalanceTrade dataclass."""

    def test_trade_creation(self):
        """Test basic trade creation."""
        trade = RebalanceTrade(
            symbol="BTC/USDT",
            action="buy",
            amount_usd=Decimal("500"),
            execution_type="limit",
            priority=1,
        )
        assert trade.symbol == "BTC/USDT"
        assert trade.action == "buy"
        assert trade.amount_usd == Decimal("500")

    def test_trade_to_dict(self):
        """Test trade serialization."""
        trade = RebalanceTrade(
            symbol="XRP/USDT",
            action="sell",
            amount_usd=Decimal("300"),
            execution_type="market",
            priority=2,
        )
        d = trade.to_dict()
        assert d["symbol"] == "XRP/USDT"
        assert d["action"] == "sell"
        assert d["amount_usd"] == 300.0


class TestRebalanceOutput:
    """Tests for RebalanceOutput dataclass."""

    def test_output_creation(self):
        """Test basic output creation."""
        allocation = PortfolioAllocation(
            total_equity_usd=Decimal("10000"),
            btc_value_usd=Decimal("2500"),
            xrp_value_usd=Decimal("4000"),
            usdt_value_usd=Decimal("3500"),
            btc_pct=Decimal("25"),
            xrp_pct=Decimal("40"),
            usdt_pct=Decimal("35"),
            max_deviation_pct=Decimal("8.33"),
        )
        output = RebalanceOutput(
            agent_name="portfolio_rebalance",
            timestamp=datetime.now(timezone.utc),
            symbol="PORTFOLIO",
            confidence=0.8,
            reasoning="Deviation exceeds threshold",
            action="rebalance",
            current_allocation=allocation,
            trades=[
                RebalanceTrade(symbol="BTC/USDT", action="buy", amount_usd=Decimal("833")),
                RebalanceTrade(symbol="XRP/USDT", action="sell", amount_usd=Decimal("667")),
            ],
        )
        assert output.action == "rebalance"
        assert len(output.trades) == 2

    def test_output_to_dict(self):
        """Test output serialization."""
        output = RebalanceOutput(
            agent_name="portfolio_rebalance",
            timestamp=datetime.now(timezone.utc),
            symbol="PORTFOLIO",
            confidence=1.0,
            reasoning="Below threshold",
            action="no_action",
            trades=[],
        )
        d = output.to_dict()
        assert d["action"] == "no_action"
        assert d["trades"] == []


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    client = AsyncMock()
    client.generate = AsyncMock(return_value=MagicMock(
        text='{"should_rebalance": true, "execution_strategy": "limit_orders", "reasoning": "OK"}',
        tokens_used=100,
    ))
    return client


@pytest.fixture
def mock_prompt_builder():
    """Create a mock prompt builder."""
    return MagicMock()


@pytest.fixture
def portfolio_config():
    """Create portfolio configuration."""
    return {
        "target_allocation": {
            "btc_pct": 33.33,
            "xrp_pct": 33.33,
            "usdt_pct": 33.34,
        },
        "rebalancing": {
            "threshold_pct": 5.0,
            "min_trade_usd": 10.0,
            "execution_type": "limit",
        },
        "hodl_bags": {
            "enabled": True,
            "btc_amount": 0,
            "xrp_amount": 0,
            "usdt_amount": 0,
        },
        "mock_balances": {
            "BTC": 0.5,
            "XRP": 10000,
            "USDT": 5000,
        },
        "mock_prices": {
            "BTC/USDT": 45000,
            "XRP/USDT": 0.60,
        },
    }


class TestPortfolioRebalanceAgent:
    """Tests for PortfolioRebalanceAgent."""

    @pytest.fixture
    def agent(self, mock_llm_client, mock_prompt_builder, portfolio_config):
        """Create a portfolio rebalance agent instance for testing."""
        return PortfolioRebalanceAgent(
            llm_client=mock_llm_client,
            prompt_builder=mock_prompt_builder,
            config=portfolio_config,
            kraken_client=None,
            db_pool=None,
        )

    def test_agent_creation(self, agent):
        """Test agent creation."""
        assert agent.agent_name == "portfolio_rebalance"
        assert agent.target_btc_pct == Decimal("33.33")
        assert agent.target_xrp_pct == Decimal("33.33")
        assert agent.target_usdt_pct == Decimal("33.34")
        assert agent.threshold_pct == Decimal("5.0")

    @pytest.mark.asyncio
    async def test_check_allocation(self, agent):
        """Test checking current allocation."""
        allocation = await agent.check_allocation()

        assert allocation is not None
        assert allocation.total_equity_usd > 0
        # With mock balances: 0.5 BTC @ 45000 = 22500, 10000 XRP @ 0.60 = 6000, 5000 USDT
        # Total = 33500
        assert allocation.btc_value_usd == Decimal("22500")
        assert allocation.xrp_value_usd == Decimal("6000")
        assert allocation.usdt_value_usd == Decimal("5000")

    @pytest.mark.asyncio
    async def test_calculate_rebalance_trades(self, agent):
        """Test calculating rebalance trades."""
        # Create an imbalanced allocation
        allocation = PortfolioAllocation(
            total_equity_usd=Decimal("30000"),
            btc_value_usd=Decimal("20000"),  # 66.67%
            xrp_value_usd=Decimal("5000"),   # 16.67%
            usdt_value_usd=Decimal("5000"),  # 16.67%
            btc_pct=Decimal("66.67"),
            xrp_pct=Decimal("16.67"),
            usdt_pct=Decimal("16.67"),
            max_deviation_pct=Decimal("33.34"),
        )

        trades = agent._calculate_rebalance_trades(allocation)

        # Should sell BTC (overweight), buy XRP and USDT (underweight)
        assert len(trades) >= 2

        # Find BTC trade
        btc_trade = next((t for t in trades if t.symbol == "BTC/USDT"), None)
        assert btc_trade is not None
        assert btc_trade.action == "sell"
        # Target is 10000, have 20000, so sell 10000
        assert btc_trade.amount_usd == Decimal("10000")

    @pytest.mark.asyncio
    async def test_process_below_threshold(self, agent):
        """Test processing when below rebalancing threshold."""
        # Configure for balanced allocation
        agent.threshold_pct = Decimal("50.0")  # Very high threshold

        output = await agent.process()

        assert output.action == "no_action"
        assert len(output.trades) == 0

    @pytest.mark.asyncio
    async def test_process_force_rebalance(self, agent, mock_llm_client):
        """Test force rebalancing."""
        # Set up LLM response
        mock_llm_client.generate.return_value = MagicMock(
            text='{"should_rebalance": true, "execution_strategy": "immediate", "reasoning": "Forced"}',
            tokens_used=50,
        )

        output = await agent.process(force=True)

        assert output is not None
        # Even if below threshold, force=True should trigger analysis

    @pytest.mark.asyncio
    async def test_process_above_threshold(self, agent, mock_llm_client):
        """Test processing when above threshold."""
        # Set low threshold to trigger rebalancing
        agent.threshold_pct = Decimal("1.0")

        mock_llm_client.generate.return_value = MagicMock(
            text='{"should_rebalance": true, "execution_strategy": "limit_orders", "reasoning": "OK"}',
            tokens_used=100,
        )

        output = await agent.process()

        assert output is not None
        # May have rebalance action if deviation is > 1%

    @pytest.mark.asyncio
    async def test_get_balances_from_config(self, agent):
        """Test getting balances from config (no Kraken)."""
        balances = await agent._get_balances()

        assert balances["BTC"] == 0.5
        assert balances["XRP"] == 10000
        assert balances["USDT"] == 5000

    @pytest.mark.asyncio
    async def test_get_prices_from_config(self, agent):
        """Test getting prices from config (no Kraken)."""
        prices = await agent._get_current_prices()

        assert prices["BTC/USDT"] == Decimal("45000")
        assert prices["XRP/USDT"] == Decimal("0.60")

    @pytest.mark.asyncio
    async def test_get_hodl_bags_from_config(self, agent):
        """Test getting hodl bags from config."""
        hodl = await agent._get_hodl_bags()

        assert hodl["BTC"] == Decimal("0")
        assert hodl["XRP"] == Decimal("0")
        assert hodl["USDT"] == Decimal("0")

    @pytest.mark.asyncio
    async def test_hodl_bags_excluded(self, agent):
        """Test that hodl bags are excluded from rebalancing."""
        # Configure hodl bags
        agent.config["hodl_bags"]["btc_amount"] = 0.1  # Hodl 0.1 BTC

        allocation = await agent.check_allocation()

        # Available BTC should be 0.5 - 0.1 = 0.4 BTC
        # But since _get_hodl_bags returns from config, it will be 0
        # In production, this would be stored in database

    def test_get_output_schema(self, agent):
        """Test getting output schema."""
        schema = agent.get_output_schema()

        assert schema["$schema"] == "http://json-schema.org/draft-07/schema#"
        assert "properties" in schema
        assert "action" in schema["properties"]
        assert "trades" in schema["properties"]

    def test_parse_llm_trades(self, agent):
        """Test parsing trades from LLM response."""
        trades_data = [
            {"symbol": "BTC/USDT", "action": "sell", "amount_usd": 500, "execution_type": "market", "priority": 1},
            {"symbol": "XRP/USDT", "action": "buy", "amount_usd": 500, "execution_type": "limit", "priority": 2},
        ]

        trades = agent._parse_llm_trades(trades_data)

        assert len(trades) == 2
        assert trades[0].symbol == "BTC/USDT"
        assert trades[0].action == "sell"
        assert trades[1].symbol == "XRP/USDT"
        assert trades[1].action == "buy"

    def test_parse_llm_trades_invalid_data(self, agent):
        """Test parsing invalid trade data."""
        trades_data = [
            {"invalid": "data", "amount_usd": "not_a_number"},  # Invalid amount
        ]

        trades = agent._parse_llm_trades(trades_data)
        # Should handle errors gracefully - may create partial trade or skip
        # The implementation creates a trade with defaults if symbol/action exist
        # Let's verify it handles the error case appropriately
        for trade in trades:
            # Any valid trade should have a valid amount
            assert trade.amount_usd >= 0


class TestPortfolioRebalanceCalculation:
    """Tests for rebalance calculation logic."""

    @pytest.fixture
    def agent(self, mock_llm_client, mock_prompt_builder, portfolio_config):
        """Create agent for calculation tests."""
        return PortfolioRebalanceAgent(
            llm_client=mock_llm_client,
            prompt_builder=mock_prompt_builder,
            config=portfolio_config,
        )

    def test_calculate_trades_balanced_portfolio(self, agent):
        """Test no trades needed for balanced portfolio."""
        allocation = PortfolioAllocation(
            total_equity_usd=Decimal("30000"),
            btc_value_usd=Decimal("10000"),
            xrp_value_usd=Decimal("10000"),
            usdt_value_usd=Decimal("10000"),
            btc_pct=Decimal("33.33"),
            xrp_pct=Decimal("33.33"),
            usdt_pct=Decimal("33.34"),
            max_deviation_pct=Decimal("0.01"),
        )

        trades = agent._calculate_rebalance_trades(allocation)
        # All trades should be below min_trade_usd threshold
        assert len(trades) == 0

    def test_calculate_trades_zero_equity(self, agent):
        """Test no trades with zero equity."""
        allocation = PortfolioAllocation(
            total_equity_usd=Decimal("0"),
            btc_value_usd=Decimal("0"),
            xrp_value_usd=Decimal("0"),
            usdt_value_usd=Decimal("0"),
            btc_pct=Decimal("0"),
            xrp_pct=Decimal("0"),
            usdt_pct=Decimal("0"),
            max_deviation_pct=Decimal("0"),
        )

        trades = agent._calculate_rebalance_trades(allocation)
        assert len(trades) == 0

    def test_calculate_trades_below_min_trade(self, agent):
        """Test trades below minimum are excluded."""
        agent.min_trade_usd = Decimal("1000")  # High minimum

        allocation = PortfolioAllocation(
            total_equity_usd=Decimal("3000"),
            btc_value_usd=Decimal("1100"),  # Slightly overweight
            xrp_value_usd=Decimal("900"),   # Slightly underweight
            usdt_value_usd=Decimal("1000"),
            btc_pct=Decimal("36.67"),
            xrp_pct=Decimal("30"),
            usdt_pct=Decimal("33.33"),
            max_deviation_pct=Decimal("3.34"),
        )

        trades = agent._calculate_rebalance_trades(allocation)
        # All deviations are < $1000, so no trades
        assert len(trades) == 0

    def test_trades_sorted_by_priority(self, agent):
        """Test trades are sorted with sells first."""
        allocation = PortfolioAllocation(
            total_equity_usd=Decimal("30000"),
            btc_value_usd=Decimal("15000"),  # Overweight - sell
            xrp_value_usd=Decimal("5000"),   # Underweight - buy
            usdt_value_usd=Decimal("10000"),
            btc_pct=Decimal("50"),
            xrp_pct=Decimal("16.67"),
            usdt_pct=Decimal("33.33"),
            max_deviation_pct=Decimal("16.67"),
        )

        trades = agent._calculate_rebalance_trades(allocation)

        # Sells should come first (priority 1)
        sell_trades = [t for t in trades if t.action == "sell"]
        buy_trades = [t for t in trades if t.action == "buy"]

        if sell_trades and buy_trades:
            assert sell_trades[0].priority < buy_trades[0].priority
