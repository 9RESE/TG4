"""
Regime Detection Agent - Classifies current market regime.

Uses Qwen 2.5 7B via Ollama for local inference.
Invoked: Every 5 minutes.
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
    from .technical_analysis import TAOutput

logger = logging.getLogger(__name__)


# Valid regime types
VALID_REGIMES = [
    "trending_bull",
    "trending_bear",
    "ranging",
    "volatile_bull",
    "volatile_bear",
    "choppy",
    "breakout_potential",
]

# Default parameters for each regime
REGIME_PARAMETERS = {
    "trending_bull": {
        "position_size_multiplier": 1.0,
        "stop_loss_multiplier": 1.2,
        "take_profit_multiplier": 2.0,
        "entry_strictness": "normal",
        "max_leverage": 5,
    },
    "trending_bear": {
        "position_size_multiplier": 1.0,
        "stop_loss_multiplier": 1.2,
        "take_profit_multiplier": 2.0,
        "entry_strictness": "normal",
        "max_leverage": 3,
    },
    "ranging": {
        "position_size_multiplier": 0.75,
        "stop_loss_multiplier": 0.8,
        "take_profit_multiplier": 1.5,
        "entry_strictness": "strict",
        "max_leverage": 2,
    },
    "volatile_bull": {
        "position_size_multiplier": 0.5,
        "stop_loss_multiplier": 1.5,
        "take_profit_multiplier": 2.5,
        "entry_strictness": "strict",
        "max_leverage": 2,
    },
    "volatile_bear": {
        "position_size_multiplier": 0.5,
        "stop_loss_multiplier": 1.5,
        "take_profit_multiplier": 2.5,
        "entry_strictness": "strict",
        "max_leverage": 2,
    },
    "choppy": {
        "position_size_multiplier": 0.25,
        "stop_loss_multiplier": 1.0,
        "take_profit_multiplier": 1.0,
        "entry_strictness": "very_strict",
        "max_leverage": 1,
    },
    "breakout_potential": {
        "position_size_multiplier": 0.75,
        "stop_loss_multiplier": 1.0,
        "take_profit_multiplier": 3.0,
        "entry_strictness": "strict",
        "max_leverage": 3,
    },
}


# JSON Schema for Regime output
REGIME_OUTPUT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["timestamp", "symbol", "regime", "confidence", "characteristics"],
    "properties": {
        "timestamp": {"type": "string", "format": "date-time"},
        "symbol": {"type": "string"},
        "regime": {
            "type": "string",
            "enum": VALID_REGIMES
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "characteristics": {
            "type": "object",
            "required": ["volatility", "trend_strength", "volume_profile"],
            "properties": {
                "volatility": {
                    "type": "string",
                    "enum": ["low", "normal", "high", "extreme"]
                },
                "trend_strength": {"type": "number", "minimum": 0, "maximum": 1},
                "volume_profile": {
                    "type": "string",
                    "enum": ["decreasing", "stable", "increasing", "spike"]
                },
                "choppiness": {"type": "number", "minimum": 0, "maximum": 100},
                "adx_value": {"type": "number"}
            }
        },
        "recommended_adjustments": {
            "type": "object",
            "properties": {
                "position_size_multiplier": {"type": "number", "minimum": 0.25, "maximum": 1.5},
                "stop_loss_multiplier": {"type": "number", "minimum": 0.5, "maximum": 2.0},
                "take_profit_multiplier": {"type": "number", "minimum": 0.5, "maximum": 3.0},
                "entry_strictness": {
                    "type": "string",
                    "enum": ["relaxed", "normal", "strict", "very_strict"]
                }
            }
        },
        "reasoning": {"type": "string", "maxLength": 300}
    }
}


@dataclass
class RegimeOutput(AgentOutput):
    """Regime Detection Agent output."""
    # Primary classification
    regime: str = "ranging"

    # Market characteristics
    volatility: str = "normal"  # low, normal, high, extreme
    trend_strength: float = 0.0  # 0-1
    volume_profile: str = "stable"  # decreasing, stable, increasing, spike
    choppiness: float = 50.0  # 0-100 (higher = more choppy)
    adx_value: float = 0.0

    # Regime duration tracking
    regime_started: Optional[datetime] = None
    periods_in_regime: int = 0

    # Transition probabilities
    transition_probabilities: dict = field(default_factory=dict)

    # Recommended trading adjustments
    position_size_multiplier: float = 1.0
    stop_loss_multiplier: float = 1.0
    take_profit_multiplier: float = 1.0
    entry_strictness: str = "normal"  # relaxed, normal, strict, very_strict

    def validate(self) -> tuple[bool, list[str]]:
        """Validate Regime-specific constraints."""
        is_valid, errors = super().validate()

        # Regime must be valid
        if self.regime not in VALID_REGIMES:
            errors.append(f"Invalid regime: {self.regime}")
            is_valid = False

        # Volatility must be valid
        if self.volatility not in ["low", "normal", "high", "extreme"]:
            errors.append(f"Invalid volatility: {self.volatility}")
            is_valid = False

        # Trend strength must be in [0, 1]
        if not 0 <= self.trend_strength <= 1:
            errors.append(f"Trend strength {self.trend_strength} not in [0, 1]")
            is_valid = False

        # Multipliers in valid ranges
        if not 0.25 <= self.position_size_multiplier <= 1.5:
            errors.append(f"Position size multiplier {self.position_size_multiplier} out of range")
            is_valid = False

        if not 0.5 <= self.stop_loss_multiplier <= 2.0:
            errors.append(f"Stop loss multiplier {self.stop_loss_multiplier} out of range")
            is_valid = False

        return is_valid, errors

    def get_regime_parameters(self) -> dict:
        """Get trading parameters for current regime."""
        return REGIME_PARAMETERS.get(self.regime, REGIME_PARAMETERS["ranging"])


class RegimeDetectionAgent(BaseAgent):
    """
    Regime Detection Agent using local Qwen 2.5 7B.

    Classifies market regime to guide strategy selection and risk parameters.
    """

    agent_name = "regime_detection"
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
        Initialize RegimeDetectionAgent.

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

        # Regime tracking
        self._previous_regime: Optional[str] = None
        self._regime_start_time: Optional[datetime] = None
        self._periods_in_current_regime = 0

    async def process(
        self,
        snapshot: 'MarketSnapshot',
        portfolio_context: Optional['PortfolioContext'] = None,
        ta_output: Optional['TAOutput'] = None,
        **kwargs
    ) -> RegimeOutput:
        """
        Detect market regime.

        Args:
            snapshot: Market data snapshot
            portfolio_context: Optional portfolio state
            ta_output: Optional TA agent output for context

        Returns:
            RegimeOutput with regime classification
        """
        logger.debug(f"Regime Agent processing {snapshot.symbol}")

        # Build additional context from TA output
        additional_context = {}
        if ta_output:
            additional_context['technical_analysis'] = {
                'trend_direction': ta_output.trend_direction,
                'trend_strength': ta_output.trend_strength,
                'momentum_score': ta_output.momentum_score,
                'bias': ta_output.bias,
                'confidence': ta_output.confidence,
            }

        # Build prompt
        prompt = self.prompt_builder.build_prompt(
            agent_name=self.agent_name,
            snapshot=snapshot,
            portfolio_context=portfolio_context,
            additional_context=additional_context if additional_context else None,
            query=self._get_regime_query(),
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
                    logger.error(f"Regime Agent failed after {self.retry_count + 1} attempts: {e}")
                    return self._create_fallback_output(snapshot, str(e))
                logger.warning(f"Regime Agent attempt {attempt + 1} failed: {e}")

        # Parse LLM response
        try:
            parsed = self._parse_response(response_text, snapshot)
        except Exception as e:
            logger.error(f"Failed to parse Regime response: {e}")
            return self._create_fallback_output(snapshot, f"Parse error: {e}")

        # Track regime changes
        current_regime = parsed.get('regime', 'ranging')
        if current_regime != self._previous_regime:
            self._regime_start_time = datetime.now(timezone.utc)
            self._periods_in_current_regime = 0
            self._previous_regime = current_regime
        else:
            self._periods_in_current_regime += 1

        # Get parameters for regime
        regime_params = REGIME_PARAMETERS.get(current_regime, REGIME_PARAMETERS["ranging"])
        adjustments = parsed.get('recommended_adjustments', {})

        # Create output
        output = RegimeOutput(
            agent_name=self.agent_name,
            timestamp=datetime.now(timezone.utc),
            symbol=snapshot.symbol,
            confidence=parsed.get('confidence', 0.5),
            reasoning=parsed.get('reasoning', ''),
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            model_used=self.model,
            regime=current_regime,
            volatility=parsed.get('characteristics', {}).get('volatility', 'normal'),
            trend_strength=parsed.get('characteristics', {}).get('trend_strength', 0.5),
            volume_profile=parsed.get('characteristics', {}).get('volume_profile', 'stable'),
            choppiness=parsed.get('characteristics', {}).get('choppiness', 50.0),
            adx_value=parsed.get('characteristics', {}).get('adx_value', 0.0),
            regime_started=self._regime_start_time,
            periods_in_regime=self._periods_in_current_regime,
            transition_probabilities=parsed.get('transition_probability', {}),
            position_size_multiplier=adjustments.get(
                'position_size_multiplier',
                regime_params['position_size_multiplier']
            ),
            stop_loss_multiplier=adjustments.get(
                'stop_loss_multiplier',
                regime_params['stop_loss_multiplier']
            ),
            take_profit_multiplier=adjustments.get(
                'take_profit_multiplier',
                regime_params['take_profit_multiplier']
            ),
            entry_strictness=adjustments.get(
                'entry_strictness',
                regime_params['entry_strictness']
            ),
        )

        # Validate output
        is_valid, validation_errors = output.validate()
        if not is_valid:
            logger.warning(f"Regime output validation issues: {validation_errors}")

        # Store output
        await self.store_output(output)

        # Cache for quick retrieval
        self._last_output = output

        logger.info(
            f"Regime Agent: {snapshot.symbol} regime={output.regime} "
            f"confidence={output.confidence:.2f} latency={latency_ms}ms"
        )

        return output

    def get_output_schema(self) -> dict:
        """Return JSON schema for validation."""
        return REGIME_OUTPUT_SCHEMA

    def get_regime_parameters(self, regime: str) -> dict:
        """Get default trading parameters for a regime."""
        return REGIME_PARAMETERS.get(regime, REGIME_PARAMETERS["ranging"])

    def _get_regime_query(self) -> str:
        """Get the regime detection query for the LLM."""
        return """Classify the current market regime. Respond with a JSON object:

{
  "regime": "trending_bull|trending_bear|ranging|volatile_bull|volatile_bear|choppy|breakout_potential",
  "confidence": 0.0-1.0,
  "characteristics": {
    "volatility": "low|normal|high|extreme",
    "trend_strength": 0.0-1.0,
    "volume_profile": "decreasing|stable|increasing|spike",
    "choppiness": 0-100,
    "adx_value": ADX indicator value
  },
  "recommended_adjustments": {
    "position_size_multiplier": 0.25-1.5,
    "stop_loss_multiplier": 0.5-2.0,
    "take_profit_multiplier": 0.5-3.0,
    "entry_strictness": "relaxed|normal|strict|very_strict"
  },
  "transition_probability": {
    "to_trending_bull": 0.0-1.0,
    "to_trending_bear": 0.0-1.0,
    "regime_change_imminent": true|false
  },
  "reasoning": "Brief explanation (max 200 chars)"
}

REGIMES:
- trending_bull: ADX>25, price above EMAs, higher highs
- trending_bear: ADX>25, price below EMAs, lower lows
- ranging: ADX<20, choppiness>60, no clear direction
- volatile_bull: High ATR, bullish bias
- volatile_bear: High ATR, bearish bias
- choppy: Erratic, whipsaws, hard to trade
- breakout_potential: Consolidation, squeeze forming

Return ONLY the JSON object."""

    def _parse_response(self, response_text: str, snapshot: 'MarketSnapshot') -> dict:
        """Parse LLM response into structured data."""
        # Try to extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                return self._normalize_parsed_output(parsed, snapshot)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode error: {e}")

        # Fallback
        try:
            parsed = json.loads(response_text.strip())
            return self._normalize_parsed_output(parsed, snapshot)
        except json.JSONDecodeError:
            pass

        # Create from indicators
        return self._create_output_from_indicators(snapshot)

    def _normalize_parsed_output(self, parsed: dict, snapshot: 'MarketSnapshot') -> dict:
        """Normalize and validate parsed output."""
        # Normalize regime
        regime = str(parsed.get('regime', 'ranging')).lower().replace(' ', '_')
        if regime not in VALID_REGIMES:
            regime = 'ranging'
        parsed['regime'] = regime

        # Clamp confidence
        parsed['confidence'] = max(0.0, min(1.0, float(parsed.get('confidence', 0.5))))

        # Ensure characteristics
        if 'characteristics' not in parsed:
            parsed['characteristics'] = {}

        chars = parsed['characteristics']

        # Normalize volatility
        volatility = str(chars.get('volatility', 'normal')).lower()
        if volatility not in ['low', 'normal', 'high', 'extreme']:
            volatility = 'normal'
        chars['volatility'] = volatility

        # Clamp trend_strength
        chars['trend_strength'] = max(0.0, min(1.0, float(chars.get('trend_strength', 0.5))))

        # Normalize volume_profile
        volume = str(chars.get('volume_profile', 'stable')).lower()
        if volume not in ['decreasing', 'stable', 'increasing', 'spike']:
            volume = 'stable'
        chars['volume_profile'] = volume

        # Clamp choppiness
        chars['choppiness'] = max(0.0, min(100.0, float(chars.get('choppiness', 50.0))))

        # ADX value
        chars['adx_value'] = float(chars.get('adx_value', 0.0))

        # Ensure recommended_adjustments
        if 'recommended_adjustments' not in parsed:
            parsed['recommended_adjustments'] = REGIME_PARAMETERS.get(regime, {})

        adj = parsed['recommended_adjustments']

        # Clamp multipliers
        adj['position_size_multiplier'] = max(0.25, min(1.5, float(adj.get('position_size_multiplier', 1.0))))
        adj['stop_loss_multiplier'] = max(0.5, min(2.0, float(adj.get('stop_loss_multiplier', 1.0))))
        adj['take_profit_multiplier'] = max(0.5, min(3.0, float(adj.get('take_profit_multiplier', 1.5))))

        # Normalize entry strictness
        strictness = str(adj.get('entry_strictness', 'normal')).lower()
        if strictness not in ['relaxed', 'normal', 'strict', 'very_strict']:
            strictness = 'normal'
        adj['entry_strictness'] = strictness

        # Ensure reasoning
        if 'reasoning' not in parsed or not parsed['reasoning']:
            parsed['reasoning'] = f"Regime detection for {snapshot.symbol}"
        if len(parsed['reasoning']) > 300:
            parsed['reasoning'] = parsed['reasoning'][:297] + "..."

        return parsed

    def _create_output_from_indicators(self, snapshot: 'MarketSnapshot') -> dict:
        """Create regime output from indicators when LLM fails."""
        indicators = snapshot.indicators

        adx = indicators.get('adx_14', 25)
        atr = indicators.get('atr_14', 0)
        choppiness = indicators.get('choppiness_14', 50)
        rsi = indicators.get('rsi_14', 50)

        # Determine regime from indicators
        if adx and adx > 25:
            if rsi and rsi > 50:
                regime = 'trending_bull'
            else:
                regime = 'trending_bear'
        elif choppiness and choppiness > 60:
            regime = 'choppy'
        else:
            regime = 'ranging'

        # Determine volatility
        # Simple heuristic based on ATR/price ratio
        volatility = 'normal'
        current_price = float(snapshot.current_price)
        if atr and current_price > 0:
            atr_pct = (atr / current_price) * 100
            if atr_pct < 1:
                volatility = 'low'
            elif atr_pct < 3:
                volatility = 'normal'
            elif atr_pct < 5:
                volatility = 'high'
            else:
                volatility = 'extreme'

        return {
            'regime': regime,
            'confidence': 0.4,
            'characteristics': {
                'volatility': volatility,
                'trend_strength': (adx / 100) if adx else 0.25,
                'volume_profile': 'stable',
                'choppiness': choppiness if choppiness else 50,
                'adx_value': adx if adx else 25,
            },
            'recommended_adjustments': REGIME_PARAMETERS.get(regime, {}),
            'reasoning': 'Generated from indicators due to LLM parse failure',
        }

    def _create_fallback_output(self, snapshot: 'MarketSnapshot', error: str) -> RegimeOutput:
        """Create fallback output when LLM fails completely."""
        return RegimeOutput(
            agent_name=self.agent_name,
            timestamp=datetime.now(timezone.utc),
            symbol=snapshot.symbol,
            confidence=0.0,
            reasoning=f"LLM error: {error}",
            latency_ms=0,
            tokens_used=0,
            model_used=self.model,
            regime='choppy',  # Default to cautious regime
            volatility='normal',
            trend_strength=0.0,
            volume_profile='stable',
            position_size_multiplier=0.25,  # Very conservative
            entry_strictness='very_strict',
        )
