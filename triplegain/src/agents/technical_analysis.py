"""
Technical Analysis Agent - Analyzes market data using technical indicators.

Uses Qwen 2.5 7B via Ollama for fast, local inference.
Invoked: Every minute on candle close.
Latency target: <500ms
"""

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, TYPE_CHECKING

from .base_agent import BaseAgent, AgentOutput

if TYPE_CHECKING:
    from ..data.market_snapshot import MarketSnapshot
    from ..llm.prompt_builder import PromptBuilder, PortfolioContext

logger = logging.getLogger(__name__)


# JSON Schema for TA output validation
TA_OUTPUT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["timestamp", "symbol", "trend", "momentum", "bias", "confidence"],
    "properties": {
        "timestamp": {"type": "string", "format": "date-time"},
        "symbol": {"type": "string"},
        "trend": {
            "type": "object",
            "required": ["direction", "strength"],
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["bullish", "bearish", "neutral"]
                },
                "strength": {"type": "number", "minimum": 0, "maximum": 1},
                "timeframe_alignment": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            }
        },
        "momentum": {
            "type": "object",
            "required": ["score"],
            "properties": {
                "score": {"type": "number", "minimum": -1, "maximum": 1},
                "rsi_signal": {
                    "type": "string",
                    "enum": ["oversold", "neutral", "overbought"]
                },
                "macd_signal": {
                    "type": "string",
                    "enum": ["bullish_cross", "bearish_cross", "bullish", "bearish", "neutral"]
                }
            }
        },
        "key_levels": {
            "type": "object",
            "properties": {
                "resistance": {"type": "array", "items": {"type": "number"}},
                "support": {"type": "array", "items": {"type": "number"}},
                "current_position": {
                    "type": "string",
                    "enum": ["near_support", "mid_range", "near_resistance"]
                }
            }
        },
        "signals": {
            "type": "object",
            "properties": {
                "primary": {"type": "string"},
                "secondary": {"type": "array", "items": {"type": "string"}},
                "warnings": {"type": "array", "items": {"type": "string"}}
            }
        },
        "bias": {
            "type": "string",
            "enum": ["long", "short", "neutral"]
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reasoning": {"type": "string", "maxLength": 500}
    }
}


@dataclass
class TAOutput(AgentOutput):
    """Technical Analysis Agent output."""
    # Trend analysis
    trend_direction: str = "neutral"  # bullish, bearish, neutral
    trend_strength: float = 0.0  # 0-1
    timeframe_alignment: list[str] = field(default_factory=list)

    # Momentum analysis
    momentum_score: float = 0.0  # -1 to 1
    rsi_signal: str = "neutral"  # oversold, neutral, overbought
    macd_signal: str = "neutral"  # bullish_cross, bearish_cross, bullish, bearish, neutral

    # Key levels
    resistance_levels: list[float] = field(default_factory=list)
    support_levels: list[float] = field(default_factory=list)
    current_position: str = "mid_range"  # near_support, mid_range, near_resistance

    # Signals
    primary_signal: str = ""
    secondary_signals: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # Bias (long/short/neutral)
    bias: str = "neutral"

    def validate(self) -> tuple[bool, list[str]]:
        """Validate TA-specific constraints."""
        is_valid, errors = super().validate()

        # Trend direction must be valid
        if self.trend_direction not in ["bullish", "bearish", "neutral"]:
            errors.append(f"Invalid trend_direction: {self.trend_direction}")
            is_valid = False

        # Trend strength must be in [0, 1]
        if not 0 <= self.trend_strength <= 1:
            errors.append(f"Trend strength {self.trend_strength} not in [0, 1]")
            is_valid = False

        # Momentum score must be in [-1, 1]
        if not -1 <= self.momentum_score <= 1:
            errors.append(f"Momentum score {self.momentum_score} not in [-1, 1]")
            is_valid = False

        # Bias must be valid
        if self.bias not in ["long", "short", "neutral"]:
            errors.append(f"Invalid bias: {self.bias}")
            is_valid = False

        return is_valid, errors


class TechnicalAnalysisAgent(BaseAgent):
    """
    Technical Analysis Agent using local Qwen 2.5 7B.

    Analyzes indicators and price action to identify trading signals.
    Does NOT make trading decisions - only provides analysis.
    """

    agent_name = "technical_analysis"
    llm_tier = "tier1_local"
    model = "qwen2.5:7b"

    def __init__(
        self,
        llm_client,
        prompt_builder: 'PromptBuilder',
        config: dict,
        db_pool=None
    ):
        """
        Initialize TechnicalAnalysisAgent.

        Args:
            llm_client: Ollama client for local inference
            prompt_builder: Prompt builder
            config: Agent configuration
            db_pool: Database pool for output storage
        """
        super().__init__(llm_client, prompt_builder, config, db_pool)

        # Agent-specific config
        self.model = config.get('model', 'qwen2.5:7b')
        self.timeout_ms = config.get('timeout_ms', 5000)
        self.retry_count = config.get('retry_count', 2)

    async def process(
        self,
        snapshot: 'MarketSnapshot',
        portfolio_context: Optional['PortfolioContext'] = None,
        **kwargs
    ) -> TAOutput:
        """
        Analyze market data and produce TA output.

        Args:
            snapshot: Market data snapshot
            portfolio_context: Optional portfolio state

        Returns:
            TAOutput with analysis results
        """
        logger.debug(f"TA Agent processing {snapshot.symbol}")

        # Build prompt
        prompt = self.prompt_builder.build_prompt(
            agent_name=self.agent_name,
            snapshot=snapshot,
            portfolio_context=portfolio_context,
            query=self._get_analysis_query(),
        )

        # Call LLM with retries
        response_text = None
        latency_ms = 0
        tokens_used = 0

        for attempt in range(self.retry_count + 1):
            try:
                response_text, latency_ms, tokens_used = await self._call_llm(
                    prompt.system_prompt,
                    prompt.user_message,
                )
                break
            except Exception as e:
                if attempt == self.retry_count:
                    logger.error(f"TA Agent failed after {self.retry_count + 1} attempts: {e}")
                    return self._create_fallback_output(snapshot, str(e))
                logger.warning(f"TA Agent attempt {attempt + 1} failed: {e}")

        # Parse LLM response
        try:
            parsed = self._parse_response(response_text, snapshot)
        except Exception as e:
            logger.error(f"Failed to parse TA response: {e}")
            return self._create_fallback_output(snapshot, f"Parse error: {e}")

        # Create output
        output = TAOutput(
            agent_name=self.agent_name,
            timestamp=datetime.now(timezone.utc),
            symbol=snapshot.symbol,
            confidence=parsed.get('confidence', 0.5),
            reasoning=parsed.get('reasoning', ''),
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            model_used=self.model,
            trend_direction=parsed.get('trend', {}).get('direction', 'neutral'),
            trend_strength=parsed.get('trend', {}).get('strength', 0.5),
            timeframe_alignment=parsed.get('trend', {}).get('timeframe_alignment', []),
            momentum_score=parsed.get('momentum', {}).get('score', 0.0),
            rsi_signal=parsed.get('momentum', {}).get('rsi_signal', 'neutral'),
            macd_signal=parsed.get('momentum', {}).get('macd_signal', 'neutral'),
            resistance_levels=parsed.get('key_levels', {}).get('resistance', []),
            support_levels=parsed.get('key_levels', {}).get('support', []),
            current_position=parsed.get('key_levels', {}).get('current_position', 'mid_range'),
            primary_signal=parsed.get('signals', {}).get('primary', ''),
            secondary_signals=parsed.get('signals', {}).get('secondary', []),
            warnings=parsed.get('signals', {}).get('warnings', []),
            bias=parsed.get('bias', 'neutral'),
        )

        # Validate output
        is_valid, validation_errors = output.validate()
        if not is_valid:
            logger.warning(f"TA output validation issues: {validation_errors}")

        # Store output
        await self.store_output(output)

        # Cache for quick retrieval
        self._last_output = output

        logger.info(
            f"TA Agent: {snapshot.symbol} bias={output.bias} "
            f"confidence={output.confidence:.2f} latency={latency_ms}ms"
        )

        return output

    def get_output_schema(self) -> dict:
        """Return JSON schema for validation."""
        return TA_OUTPUT_SCHEMA

    def _get_analysis_query(self) -> str:
        """Get the analysis query for the LLM."""
        return """Analyze the provided market data. Respond with a JSON object containing:

{
  "trend": {
    "direction": "bullish|bearish|neutral",
    "strength": 0.0-1.0,
    "timeframe_alignment": ["aligned timeframes"]
  },
  "momentum": {
    "score": -1.0 to 1.0,
    "rsi_signal": "oversold|neutral|overbought",
    "macd_signal": "bullish_cross|bearish_cross|bullish|bearish|neutral"
  },
  "key_levels": {
    "resistance": [price_levels],
    "support": [price_levels],
    "current_position": "near_support|mid_range|near_resistance"
  },
  "signals": {
    "primary": "main signal description",
    "secondary": ["additional observations"],
    "warnings": ["any concerns"]
  },
  "bias": "long|short|neutral",
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation (max 200 chars)"
}

Return ONLY the JSON object, no additional text."""

    def _parse_response(self, response_text: str, snapshot: 'MarketSnapshot') -> dict:
        """
        Parse LLM response into structured data.

        Args:
            response_text: Raw LLM response
            snapshot: Market snapshot for context

        Returns:
            Parsed dictionary
        """
        # Try to extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                return self._normalize_parsed_output(parsed, snapshot)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode error: {e}")

        # Fallback: try to parse the entire response
        try:
            parsed = json.loads(response_text.strip())
            return self._normalize_parsed_output(parsed, snapshot)
        except json.JSONDecodeError:
            pass

        # If all parsing fails, create minimal output from indicators
        return self._create_output_from_indicators(snapshot)

    def _normalize_parsed_output(self, parsed: dict, snapshot: 'MarketSnapshot') -> dict:
        """Normalize and validate parsed output."""
        # Ensure trend structure
        if 'trend' not in parsed or not isinstance(parsed['trend'], dict):
            parsed['trend'] = {'direction': 'neutral', 'strength': 0.5}

        # Normalize direction
        direction = str(parsed['trend'].get('direction', 'neutral')).lower()
        if direction not in ['bullish', 'bearish', 'neutral']:
            direction = 'neutral'
        parsed['trend']['direction'] = direction

        # Clamp strength
        strength = parsed['trend'].get('strength', 0.5)
        parsed['trend']['strength'] = max(0.0, min(1.0, float(strength)))

        # Ensure momentum structure
        if 'momentum' not in parsed or not isinstance(parsed['momentum'], dict):
            parsed['momentum'] = {'score': 0.0}

        # Clamp momentum score
        score = parsed['momentum'].get('score', 0.0)
        parsed['momentum']['score'] = max(-1.0, min(1.0, float(score)))

        # Normalize RSI signal
        rsi_signal = str(parsed['momentum'].get('rsi_signal', 'neutral')).lower()
        if rsi_signal not in ['oversold', 'neutral', 'overbought']:
            rsi_signal = 'neutral'
        parsed['momentum']['rsi_signal'] = rsi_signal

        # Normalize MACD signal
        macd_signal = str(parsed['momentum'].get('macd_signal', 'neutral')).lower()
        valid_macd = ['bullish_cross', 'bearish_cross', 'bullish', 'bearish', 'neutral']
        if macd_signal not in valid_macd:
            macd_signal = 'neutral'
        parsed['momentum']['macd_signal'] = macd_signal

        # Normalize bias
        bias = str(parsed.get('bias', 'neutral')).lower()
        if bias not in ['long', 'short', 'neutral']:
            bias = 'neutral'
        parsed['bias'] = bias

        # Clamp confidence
        confidence = parsed.get('confidence', 0.5)
        parsed['confidence'] = max(0.0, min(1.0, float(confidence)))

        # Ensure reasoning exists
        if 'reasoning' not in parsed or not parsed['reasoning']:
            parsed['reasoning'] = f"Analysis for {snapshot.symbol}"

        # Truncate reasoning if too long
        if len(parsed['reasoning']) > 500:
            parsed['reasoning'] = parsed['reasoning'][:497] + "..."

        return parsed

    def _create_output_from_indicators(self, snapshot: 'MarketSnapshot') -> dict:
        """Create output from raw indicators when LLM parsing fails."""
        indicators = snapshot.indicators

        # Determine bias from indicators
        rsi = indicators.get('rsi_14', 50)
        macd = indicators.get('macd', {})
        macd_hist = macd.get('histogram', 0) if isinstance(macd, dict) else 0

        # Simple heuristics
        bias = 'neutral'
        if rsi and macd_hist:
            if rsi > 60 and macd_hist > 0:
                bias = 'long'
            elif rsi < 40 and macd_hist < 0:
                bias = 'short'

        rsi_signal = 'neutral'
        if rsi:
            if rsi < 30:
                rsi_signal = 'oversold'
            elif rsi > 70:
                rsi_signal = 'overbought'

        return {
            'trend': {
                'direction': 'bullish' if bias == 'long' else ('bearish' if bias == 'short' else 'neutral'),
                'strength': 0.5,
                'timeframe_alignment': [],
            },
            'momentum': {
                'score': 0.0,
                'rsi_signal': rsi_signal,
                'macd_signal': 'bullish' if macd_hist and macd_hist > 0 else 'bearish' if macd_hist and macd_hist < 0 else 'neutral',
            },
            'key_levels': {
                'resistance': [],
                'support': [],
                'current_position': 'mid_range',
            },
            'signals': {
                'primary': 'Fallback analysis from indicators',
                'secondary': [],
                'warnings': ['LLM parsing failed, using indicator heuristics'],
            },
            'bias': bias,
            'confidence': 0.25,  # Conservative confidence for heuristic fallback
            'reasoning': 'Generated from indicator values due to LLM parse failure',
        }

    def _create_fallback_output(self, snapshot: 'MarketSnapshot', error: str) -> TAOutput:
        """Create fallback output when LLM fails completely."""
        return TAOutput(
            agent_name=self.agent_name,
            timestamp=datetime.now(timezone.utc),
            symbol=snapshot.symbol,
            confidence=0.0,
            reasoning=f"LLM error: {error}",
            latency_ms=0,
            tokens_used=0,
            model_used=self.model,
            trend_direction='neutral',
            trend_strength=0.0,
            momentum_score=0.0,
            rsi_signal='neutral',
            macd_signal='neutral',
            bias='neutral',
            primary_signal='No analysis available',
            warnings=['LLM call failed'],
        )
