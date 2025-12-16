"""
Strategy Optimization Framework.

Runs parameter optimization in isolated subprocesses for memory safety.
"""

from .base_optimizer import BaseOptimizer, OptimizationConfig, RunResult

__all__ = ['BaseOptimizer', 'OptimizationConfig', 'RunResult']
