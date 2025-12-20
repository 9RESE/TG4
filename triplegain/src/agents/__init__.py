"""
TripleGain Agents Module.

Contains LLM-based agents for market analysis and trading decisions:
- BaseAgent: Abstract base class for all agents
- TechnicalAnalysisAgent: Analyzes indicators and price action
- RegimeDetectionAgent: Classifies market regime
- TradingDecisionAgent: 6-model A/B testing for trading decisions
- SentimentAnalysisAgent: Dual-model sentiment from Grok + GPT (Phase 7)
"""

from .base_agent import BaseAgent, AgentOutput
from .technical_analysis import TechnicalAnalysisAgent, TAOutput
from .regime_detection import RegimeDetectionAgent, RegimeOutput
from .trading_decision import TradingDecisionAgent, TradingDecisionOutput, ConsensusResult
from .sentiment_analysis import (
    SentimentAnalysisAgent,
    SentimentOutput,
    SentimentBias,
    FearGreedLevel,
    KeyEvent,
    ProviderResult,
)

__all__ = [
    'BaseAgent',
    'AgentOutput',
    'TechnicalAnalysisAgent',
    'TAOutput',
    'RegimeDetectionAgent',
    'RegimeOutput',
    'TradingDecisionAgent',
    'TradingDecisionOutput',
    'ConsensusResult',
    'SentimentAnalysisAgent',
    'SentimentOutput',
    'SentimentBias',
    'FearGreedLevel',
    'KeyEvent',
    'ProviderResult',
]
