"""
TripleGain Risk Management Module.

Contains rule-based risk management (NO LLM):
- RiskManagementEngine: Validates trades, enforces limits, manages circuit breakers
"""

from .rules_engine import (
    RiskManagementEngine,
    TradeProposal,
    RiskValidation,
    RiskState,
    ValidationStatus,
)

__all__ = [
    'RiskManagementEngine',
    'TradeProposal',
    'RiskValidation',
    'RiskState',
    'ValidationStatus',
]
