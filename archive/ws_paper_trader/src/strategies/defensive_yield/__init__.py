"""
Defensive Yield Strategy
Phase 15: Modular Strategy Factory

This strategy wraps the existing RL-driven orchestrator logic.
It's the live production strategy - do NOT modify without thorough testing.
"""
from .strategy import DefensiveYield

__all__ = ['DefensiveYield']
