"""
Unit tests for the Prompt Template System.

Tests validate:
- Template loading and caching
- Token estimation accuracy
- Budget compliance
- Context injection
- Schema validation
- Prompt assembly
"""

import pytest
import os
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
import json

from triplegain.src.llm.prompt_builder import (
    PromptBuilder,
    AssembledPrompt,
    PortfolioContext,
)
from triplegain.src.data.market_snapshot import MarketSnapshot


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def prompt_config(tmp_path) -> dict:
    """Create test configuration with temp template directory."""
    templates_dir = tmp_path / "prompts"
    templates_dir.mkdir()

    # Create test templates
    (templates_dir / "technical_analysis.txt").write_text("""You are an expert quantitative technical analyst for cryptocurrency markets.

ROLE:
You analyze market data using technical indicators to identify trading opportunities.
You DO NOT make final trading decisions - you provide analysis to other agents.

OUTPUT FORMAT (JSON only, no other text):
{
  "timestamp": "ISO8601",
  "symbol": "SYMBOL",
  "bias": "long|short|neutral",
  "confidence": 0.0-1.0
}
""")

    (templates_dir / "regime_detection.txt").write_text("""You are a market regime detection agent.

ROLE:
Classify the current market state into one of: trending_bull, trending_bear, ranging, high_volatility.

OUTPUT FORMAT (JSON only):
{
  "regime": "trending_bull|trending_bear|ranging|high_volatility",
  "confidence": 0.0-1.0
}
""")

    return {
        'token_budgets': {
            'tier1_local': {
                'total': 8192,
                'system_prompt': 1500,
                'context': 800,
                'market_data': 3000,
                'query': 400,
                'buffer': 2492,
            },
            'tier2_api': {
                'total': 128000,
                'system_prompt': 3000,
                'context': 2000,
                'market_data': 6000,
                'query': 1000,
                'buffer': 116000,
            },
        },
        'templates_dir': str(templates_dir),
        'agents': {
            'technical_analysis': {
                'template': 'technical_analysis.txt',
                'tier': 'tier1_local',
                'max_candles_per_tf': 10,
            },
            'regime_detection': {
                'template': 'regime_detection.txt',
                'tier': 'tier1_local',
                'max_candles_per_tf': 5,
            },
        },
    }


@pytest.fixture
def sample_snapshot() -> MarketSnapshot:
    """Create sample market snapshot for testing."""
    now = datetime.now(timezone.utc)
    return MarketSnapshot(
        timestamp=now,
        symbol='BTC/USDT',
        current_price=Decimal('45250.50'),
        indicators={
            'rsi_14': 62.5,
            'macd': {'line': 150.2, 'signal': 120.5, 'histogram': 29.7},
            'ema_9': 45123.45,
            'ema_21': 44890.12,
            'atr_14': 1250.0,
            'adx_14': 28.5,
            'bollinger_bands': {
                'upper': 46500.0,
                'middle': 45000.0,
                'lower': 43500.0,
                'position': 0.65,
            },
        },
    )


@pytest.fixture
def sample_portfolio_context() -> PortfolioContext:
    """Create sample portfolio context for testing."""
    return PortfolioContext(
        total_equity_usd=Decimal('100000'),
        available_margin_usd=Decimal('80000'),
        positions=[
            {'symbol': 'BTC/USDT', 'side': 'long', 'size_usd': 5000, 'unrealized_pnl': 250},
        ],
        allocation={'BTC': Decimal('33'), 'XRP': Decimal('33'), 'USDT': Decimal('34')},
        daily_pnl_pct=Decimal('1.5'),
        drawdown_pct=Decimal('2.0'),
        consecutive_losses=0,
        win_rate_7d=Decimal('0.65'),
    )


# =============================================================================
# AssembledPrompt Tests
# =============================================================================

class TestAssembledPrompt:
    """Tests for AssembledPrompt dataclass."""

    def test_assembled_prompt_creation(self):
        """Test creating an AssembledPrompt."""
        prompt = AssembledPrompt(
            system_prompt="You are a trading agent.",
            user_message="Analyze the current market.",
            estimated_tokens=500,
            agent_name="technical_analysis",
            tier="tier1_local",
        )

        assert prompt.agent_name == "technical_analysis"
        assert prompt.tier == "tier1_local"
        assert prompt.estimated_tokens == 500


# =============================================================================
# PortfolioContext Tests
# =============================================================================

class TestPortfolioContext:
    """Tests for PortfolioContext dataclass."""

    def test_portfolio_context_creation(self, sample_portfolio_context):
        """Test creating a PortfolioContext."""
        ctx = sample_portfolio_context

        assert ctx.total_equity_usd == Decimal('100000')
        assert ctx.win_rate_7d == Decimal('0.65')
        assert len(ctx.positions) == 1

    def test_portfolio_context_to_dict(self, sample_portfolio_context):
        """Test converting portfolio context to dict."""
        ctx = sample_portfolio_context

        d = ctx.to_dict()

        assert 'total_equity_usd' in d
        assert 'positions' in d
        assert d['consecutive_losses'] == 0


# =============================================================================
# PromptBuilder Tests
# =============================================================================

class TestPromptBuilder:
    """Tests for PromptBuilder."""

    def test_builder_creation(self, prompt_config):
        """Test creating a PromptBuilder."""
        builder = PromptBuilder(prompt_config)

        assert builder is not None
        assert builder._templates is not None

    def test_template_loading(self, prompt_config):
        """Test that templates are loaded correctly."""
        builder = PromptBuilder(prompt_config)

        assert 'technical_analysis' in builder._templates
        assert 'regime_detection' in builder._templates

    def test_token_estimation(self, prompt_config):
        """Test token estimation accuracy."""
        builder = PromptBuilder(prompt_config)

        # Known text length
        text = "A" * 350  # ~100 tokens at 3.5 chars/token
        estimated = builder.estimate_tokens(text)

        # Should be close to 100
        assert 90 <= estimated <= 110

    def test_truncate_to_budget(self, prompt_config):
        """Test truncation to fit budget."""
        builder = PromptBuilder(prompt_config)

        long_text = "A" * 10000
        truncated = builder.truncate_to_budget(long_text, max_tokens=100)

        # Check truncated text fits budget
        estimated = builder.estimate_tokens(truncated)
        assert estimated <= 100

    def test_build_prompt_basic(self, prompt_config, sample_snapshot):
        """Test building a basic prompt."""
        builder = PromptBuilder(prompt_config)

        prompt = builder.build_prompt(
            agent_name='technical_analysis',
            snapshot=sample_snapshot,
        )

        assert prompt.agent_name == 'technical_analysis'
        assert prompt.system_prompt != ""
        assert prompt.user_message != ""

    def test_build_prompt_with_context(self, prompt_config, sample_snapshot, sample_portfolio_context):
        """Test building prompt with portfolio context."""
        builder = PromptBuilder(prompt_config)

        prompt = builder.build_prompt(
            agent_name='technical_analysis',
            snapshot=sample_snapshot,
            portfolio_context=sample_portfolio_context,
        )

        # User message should include portfolio info
        assert 'equity' in prompt.user_message.lower() or 'portfolio' in prompt.user_message.lower()

    def test_build_prompt_with_query(self, prompt_config, sample_snapshot):
        """Test building prompt with custom query."""
        builder = PromptBuilder(prompt_config)

        custom_query = "Focus specifically on trend alignment across timeframes."

        prompt = builder.build_prompt(
            agent_name='technical_analysis',
            snapshot=sample_snapshot,
            query=custom_query,
        )

        assert custom_query in prompt.user_message

    def test_budget_compliance(self, prompt_config, sample_snapshot, sample_portfolio_context):
        """Test that generated prompts fit within budget."""
        builder = PromptBuilder(prompt_config)

        prompt = builder.build_prompt(
            agent_name='technical_analysis',
            snapshot=sample_snapshot,
            portfolio_context=sample_portfolio_context,
        )

        # Get tier budget
        tier = prompt_config['agents']['technical_analysis']['tier']
        max_budget = prompt_config['token_budgets'][tier]['total']

        assert prompt.estimated_tokens < max_budget

    def test_format_portfolio_context(self, prompt_config, sample_portfolio_context):
        """Test portfolio context formatting."""
        builder = PromptBuilder(prompt_config)

        formatted = builder._format_portfolio_context(sample_portfolio_context)

        assert isinstance(formatted, str)
        assert 'equity' in formatted.lower() or '100000' in formatted

    def test_format_market_data(self, prompt_config, sample_snapshot):
        """Test market data formatting."""
        builder = PromptBuilder(prompt_config)

        # Test tier1_local (compact format)
        formatted = builder._format_market_data(sample_snapshot, 'tier1_local')
        assert isinstance(formatted, str)
        data = json.loads(formatted)
        assert 'sym' in data  # Compact format uses 'sym'

        # Test tier2_api (full format)
        formatted_full = builder._format_market_data(sample_snapshot, 'tier2_api')
        data_full = json.loads(formatted_full)
        assert 'symbol' in data_full  # Full format uses 'symbol'

    def test_unknown_agent_error(self, prompt_config, sample_snapshot):
        """Test error for unknown agent."""
        builder = PromptBuilder(prompt_config)

        with pytest.raises(ValueError):
            builder.build_prompt(
                agent_name='nonexistent_agent',
                snapshot=sample_snapshot,
            )


# =============================================================================
# Output Format Tests
# =============================================================================

class TestPromptOutputFormat:
    """Tests for prompt output format requirements."""

    def test_system_prompt_has_role(self, prompt_config, sample_snapshot):
        """Test that system prompt includes role definition."""
        builder = PromptBuilder(prompt_config)

        prompt = builder.build_prompt(
            agent_name='technical_analysis',
            snapshot=sample_snapshot,
        )

        assert 'ROLE' in prompt.system_prompt or 'role' in prompt.system_prompt.lower()

    def test_system_prompt_has_output_format(self, prompt_config, sample_snapshot):
        """Test that system prompt includes output format."""
        builder = PromptBuilder(prompt_config)

        prompt = builder.build_prompt(
            agent_name='technical_analysis',
            snapshot=sample_snapshot,
        )

        assert 'OUTPUT FORMAT' in prompt.system_prompt or 'JSON' in prompt.system_prompt

    def test_user_message_has_market_data(self, prompt_config, sample_snapshot):
        """Test that user message includes market data."""
        builder = PromptBuilder(prompt_config)

        prompt = builder.build_prompt(
            agent_name='technical_analysis',
            snapshot=sample_snapshot,
        )

        # Should include the symbol
        assert 'BTC/USDT' in prompt.user_message


# =============================================================================
# Additional Context Tests
# =============================================================================

class TestAdditionalContext:
    """Tests for additional context injection."""

    def test_with_additional_context(self, prompt_config, sample_snapshot):
        """Test prompt with additional context dict."""
        builder = PromptBuilder(prompt_config)

        additional = {
            'recent_trade': {'symbol': 'BTC/USDT', 'pnl': 250},
            'alert': 'RSI entering overbought territory',
        }

        prompt = builder.build_prompt(
            agent_name='technical_analysis',
            snapshot=sample_snapshot,
            additional_context=additional,
        )

        # Additional context should be included
        assert 'overbought' in prompt.user_message or 'recent_trade' in prompt.user_message.lower()

    def test_with_agent_outputs(self, prompt_config, sample_snapshot):
        """Test including other agent outputs in context."""
        builder = PromptBuilder(prompt_config)

        additional = {
            'regime_agent_output': {
                'regime': 'trending_bull',
                'confidence': 0.85,
            },
        }

        prompt = builder.build_prompt(
            agent_name='technical_analysis',
            snapshot=sample_snapshot,
            additional_context=additional,
        )

        assert 'trending_bull' in prompt.user_message or 'regime' in prompt.user_message.lower()


# =============================================================================
# Performance Tests
# =============================================================================

class TestPromptPerformance:
    """Performance tests for prompt building."""

    def test_build_prompt_latency(self, prompt_config, sample_snapshot, sample_portfolio_context):
        """Test prompt build time is fast."""
        import time

        builder = PromptBuilder(prompt_config)

        start = time.time()
        for _ in range(100):
            builder.build_prompt(
                agent_name='technical_analysis',
                snapshot=sample_snapshot,
                portfolio_context=sample_portfolio_context,
            )
        elapsed = (time.time() - start) * 1000 / 100

        assert elapsed < 5, f"Build took {elapsed:.2f}ms, expected <5ms"
