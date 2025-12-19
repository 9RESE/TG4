"""
Prompt Template System - Builds prompts for LLM agents.

This module manages:
- Loading and caching prompt templates
- Token estimation and budget management
- Context injection (portfolio, market data)
- Prompt assembly for different agent types and tiers
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Optional, Any
from pathlib import Path
import json

from ..data.market_snapshot import MarketSnapshot

logger = logging.getLogger(__name__)


@dataclass
class AssembledPrompt:
    """Complete prompt ready for LLM."""
    system_prompt: str
    user_message: str
    estimated_tokens: int
    agent_name: str
    tier: str


@dataclass
class PortfolioContext:
    """Portfolio state for context injection."""
    total_equity_usd: Decimal
    available_margin_usd: Decimal
    positions: list[dict]
    allocation: dict[str, Decimal]
    daily_pnl_pct: Decimal
    drawdown_pct: Decimal
    consecutive_losses: int
    win_rate_7d: Decimal

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'total_equity_usd': float(self.total_equity_usd),
            'available_margin_usd': float(self.available_margin_usd),
            'positions': self.positions,
            'allocation': {k: float(v) for k, v in self.allocation.items()},
            'daily_pnl_pct': float(self.daily_pnl_pct),
            'drawdown_pct': float(self.drawdown_pct),
            'consecutive_losses': self.consecutive_losses,
            'win_rate_7d': float(self.win_rate_7d),
        }


class PromptBuilder:
    """
    Builds prompts for LLM agents.

    Handles template loading, context injection, token management.
    """

    # Characters per token (conservative estimate for JSON)
    CHARS_PER_TOKEN = 3.5

    # Safety margin for token estimation (10% buffer for special tokens, formatting, etc.)
    TOKEN_SAFETY_MARGIN = 1.10

    def __init__(self, config: dict):
        """
        Initialize PromptBuilder.

        Args:
            config: Prompt configuration dictionary
        """
        self.config = config
        self._templates: dict[str, str] = {}
        self._load_templates()

    def build_prompt(
        self,
        agent_name: str,
        snapshot: MarketSnapshot,
        portfolio_context: Optional[PortfolioContext] = None,
        additional_context: Optional[dict] = None,
        query: Optional[str] = None
    ) -> AssembledPrompt:
        """
        Build complete prompt for an agent.

        Args:
            agent_name: Name of the target agent
            snapshot: Market data snapshot
            portfolio_context: Portfolio state (optional)
            additional_context: Extra context (e.g., other agent outputs)
            query: Specific query/task (uses default if not provided)

        Returns:
            AssembledPrompt ready for LLM
        """
        # Validate agent exists
        if agent_name not in self.config.get('agents', {}):
            raise ValueError(f"Unknown agent: {agent_name}")

        agent_config = self.config['agents'][agent_name]
        tier = agent_config.get('tier', 'tier1_local')
        budget = self.config['token_budgets'].get(tier, {})

        # Get system prompt from template
        system_prompt = self._templates.get(agent_name)
        if not system_prompt:
            logger.warning(f"No template found for agent {agent_name}, using minimal prompt")
            system_prompt = f"You are the {agent_name} agent. Analyze the data and provide structured output."

        # Build user message components
        user_parts = []

        # Add portfolio context if provided
        if portfolio_context:
            portfolio_str = self._format_portfolio_context(portfolio_context)
            user_parts.append(f"## Portfolio State\n{portfolio_str}")

        # Add market data
        market_str = self._format_market_data(snapshot, tier)
        user_parts.append(f"## Market Data\n{market_str}")

        # Add additional context if provided
        if additional_context:
            context_str = self._format_additional_context(additional_context)
            user_parts.append(f"## Additional Context\n{context_str}")

        # Add query/task
        if query:
            user_parts.append(f"## Task\n{query}")
        else:
            user_parts.append("## Task\nAnalyze the provided market data and provide your assessment.")

        user_message = "\n\n".join(user_parts)

        # Estimate tokens
        total_text = system_prompt + user_message
        estimated_tokens = self.estimate_tokens(total_text)

        # Truncate if over budget
        max_budget = budget.get('total', 8192) - budget.get('buffer', 2000)
        if estimated_tokens > max_budget:
            original_tokens = estimated_tokens
            user_message = self.truncate_to_budget(
                user_message,
                max_tokens=max_budget - self.estimate_tokens(system_prompt)
            )
            estimated_tokens = self.estimate_tokens(system_prompt + user_message)
            logger.warning(
                f"Prompt truncated for {agent_name}: {original_tokens} -> {estimated_tokens} tokens "
                f"(budget: {max_budget})"
            )

        return AssembledPrompt(
            system_prompt=system_prompt,
            user_message=user_message,
            estimated_tokens=estimated_tokens,
            agent_name=agent_name,
            tier=tier,
        )

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Uses ~3.5 characters per token (conservative for JSON) with a 10%
        safety margin to account for special tokens, BPE encoding variations,
        and formatting overhead.
        """
        if not text:
            return 0
        base_estimate = len(text) / self.CHARS_PER_TOKEN
        return int(base_estimate * self.TOKEN_SAFETY_MARGIN)

    def truncate_to_budget(
        self,
        content: str,
        max_tokens: int
    ) -> str:
        """
        Truncate content to fit token budget.

        Accounts for safety margin so truncated content will fit within
        budget when estimate_tokens() is called on it.
        """
        if max_tokens <= 0:
            return ""

        # Account for safety margin by dividing - the estimate will multiply back
        effective_chars = int(max_tokens * self.CHARS_PER_TOKEN / self.TOKEN_SAFETY_MARGIN)

        if len(content) <= effective_chars:
            return content

        # Truncate with ellipsis indicator
        return content[:effective_chars - 20] + "\n... [truncated]"

    def _load_templates(self) -> None:
        """Load all prompt templates from disk with validation."""
        templates_dir = self.config.get('templates_dir', '')
        if not templates_dir:
            logger.warning("No templates_dir configured")
            return

        templates_path = Path(templates_dir)
        if not templates_path.exists():
            logger.warning(f"Templates directory not found: {templates_path}")
            return

        agents = self.config.get('agents', {})
        for agent_name, agent_config in agents.items():
            template_file = agent_config.get('template', '')
            if template_file:
                template_path = templates_path / template_file
                if template_path.exists():
                    template_content = template_path.read_text()

                    # Validate template
                    validation_result = self._validate_template(agent_name, template_content)
                    if validation_result['valid']:
                        self._templates[agent_name] = template_content
                        logger.debug(f"Loaded template for {agent_name}")
                    else:
                        logger.warning(
                            f"Template validation failed for {agent_name}: "
                            f"{validation_result['errors']}"
                        )
                        # Still load it but log warning
                        self._templates[agent_name] = template_content
                else:
                    logger.warning(f"Template file not found: {template_path}")

    def _validate_template(self, agent_name: str, content: str) -> dict:
        """
        Validate a prompt template for required sections.

        Args:
            agent_name: Name of the agent
            content: Template content

        Returns:
            Dict with 'valid' bool and 'errors' list
        """
        errors = []

        # Check minimum length
        if len(content) < 50:
            errors.append("Template is too short (< 50 characters)")

        # Check for role/persona definition
        role_patterns = [
            r'(?i)you\s+are',
            r'(?i)your\s+role',
            r'(?i)as\s+(?:a|an)',
            r'(?i)system:',
            r'(?i)persona:',
        ]
        has_role = any(re.search(pattern, content) for pattern in role_patterns)
        if not has_role:
            errors.append("Template missing role/persona definition")

        # Check for output format specification
        output_patterns = [
            r'(?i)output\s+format',
            r'(?i)response\s+format',
            r'(?i)json',
            r'(?i)return',
            r'(?i)respond\s+with',
            r'(?i)provide',
        ]
        has_output_format = any(re.search(pattern, content) for pattern in output_patterns)
        if not has_output_format:
            errors.append("Template missing output format specification")

        # Agent-specific validations
        agent_requirements = {
            'technical_analysis': ['trend', 'indicator', 'support', 'resistance'],
            'regime_detection': ['regime', 'trend', 'volatility', 'range'],
            'sentiment_analysis': ['sentiment', 'news', 'social'],
            'trading_decision': ['action', 'position', 'risk'],
            'risk_manager': ['risk', 'stop', 'position', 'limit'],
            'coordinator': ['agent', 'conflict', 'decision'],
        }

        if agent_name in agent_requirements:
            required_keywords = agent_requirements[agent_name]
            content_lower = content.lower()
            missing = [kw for kw in required_keywords if kw not in content_lower]
            # Require at least 2/3 of keywords to be present (stricter validation)
            if len(missing) > len(required_keywords) // 3:
                errors.append(f"Template missing key concepts: {missing}")

        return {
            'valid': len(errors) == 0,
            'errors': errors
        }

    def validate_all_templates(self) -> dict[str, dict]:
        """
        Validate all loaded templates.

        Returns:
            Dict of agent_name -> validation result
        """
        results = {}
        for agent_name, template in self._templates.items():
            results[agent_name] = self._validate_template(agent_name, template)
        return results

    def _format_portfolio_context(
        self,
        context: PortfolioContext
    ) -> str:
        """Format portfolio context for injection."""
        data = {
            'total_equity_usd': float(context.total_equity_usd),
            'available_margin_usd': float(context.available_margin_usd),
            'daily_pnl_pct': float(context.daily_pnl_pct),
            'drawdown_pct': float(context.drawdown_pct),
            'consecutive_losses': context.consecutive_losses,
            'win_rate_7d': float(context.win_rate_7d),
            'allocation': {k: float(v) for k, v in context.allocation.items()},
            'open_positions': len(context.positions),
        }

        if context.positions:
            data['positions'] = context.positions

        return json.dumps(data, indent=2)

    def _format_market_data(
        self,
        snapshot: MarketSnapshot,
        tier: str
    ) -> str:
        """Format market data, adjusting detail for tier."""
        if tier == 'tier1_local':
            # Compact format for local LLM
            return snapshot.to_compact_format()
        else:
            # Full format for API LLMs
            return snapshot.to_prompt_format()

    def _format_additional_context(
        self,
        additional: dict
    ) -> str:
        """Format additional context for injection."""
        return json.dumps(additional, indent=2, default=str)

    def get_default_query(self, agent_name: str) -> str:
        """Get default query for an agent."""
        defaults = {
            'technical_analysis': (
                "Analyze the provided market data. Identify trend direction, "
                "momentum conditions, and key support/resistance levels. "
                "Provide your bias (long/short/neutral) with confidence score."
            ),
            'regime_detection': (
                "Classify the current market regime based on the provided data. "
                "Determine if the market is trending (bull/bear), ranging, or "
                "experiencing high/low volatility."
            ),
            'sentiment_analysis': (
                "Analyze market sentiment based on the provided news and social data. "
                "Provide a sentiment score and key drivers."
            ),
            'trading_decision': (
                "Based on all agent inputs and current portfolio state, "
                "recommend a trading action with position sizing."
            ),
        }
        return defaults.get(agent_name, "Analyze the provided data and share your assessment.")
