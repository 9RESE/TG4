"""
TripleGain Agents Module.

Contains LLM-based agents for market analysis and trading decisions:
- BaseAgent: Abstract base class for all agents
- TechnicalAnalysisAgent: Analyzes indicators and price action
- RegimeDetectionAgent: Classifies market regime
- TradingDecisionAgent: 6-model A/B testing for trading decisions
"""

from .base_agent import BaseAgent, AgentOutput
from .technical_analysis import TechnicalAnalysisAgent, TAOutput
from .regime_detection import RegimeDetectionAgent, RegimeOutput
from .trading_decision import TradingDecisionAgent, TradingDecisionOutput, ConsensusResult

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
]
