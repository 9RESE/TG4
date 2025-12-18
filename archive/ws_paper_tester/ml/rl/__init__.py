"""
Reinforcement Learning Module

Provides Gymnasium environments and agent wrappers for RL-based trading.
"""

from .environment import TradingEnv, TradingEnvConfig
from .rewards import (
    RewardFunction,
    PnLReward,
    SharpeReward,
    RiskAdjustedReward,
    create_reward_function
)
from .agents import (
    train_ppo,
    train_sac,
    load_agent,
    evaluate_agent
)

__all__ = [
    # Environment
    'TradingEnv',
    'TradingEnvConfig',
    # Rewards
    'RewardFunction',
    'PnLReward',
    'SharpeReward',
    'RiskAdjustedReward',
    'create_reward_function',
    # Agents
    'train_ppo',
    'train_sac',
    'load_agent',
    'evaluate_agent'
]
