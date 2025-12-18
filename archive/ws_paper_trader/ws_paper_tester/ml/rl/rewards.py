"""
Reward Functions for RL Trading Agents

Provides various reward formulations for different trading objectives.
"""

from typing import Optional, Dict, Any, Callable
from abc import ABC, abstractmethod
import numpy as np


class RewardFunction(ABC):
    """Abstract base class for reward functions."""

    @abstractmethod
    def __call__(
        self,
        step_return: float,
        equity: float,
        position: float,
        trade_executed: bool,
        env: Any
    ) -> float:
        """
        Calculate reward for current step.

        Args:
            step_return: Return for this step (as decimal)
            equity: Current portfolio equity
            position: Current position size
            trade_executed: Whether a trade was executed this step
            env: Reference to environment

        Returns:
            Reward value
        """
        pass


class PnLReward(RewardFunction):
    """
    Simple PnL-based reward.

    Rewards the agent based on the change in equity.
    """

    def __init__(
        self,
        scaling: float = 100.0,
        trade_cost_penalty: float = 0.0,
        holding_penalty: float = 0.0
    ):
        """
        Initialize PnL reward.

        Args:
            scaling: Multiplier for the reward
            trade_cost_penalty: Penalty per trade (to discourage overtrading)
            holding_penalty: Penalty per step with no position
        """
        self.scaling = scaling
        self.trade_cost_penalty = trade_cost_penalty
        self.holding_penalty = holding_penalty

    def __call__(
        self,
        step_return: float,
        equity: float,
        position: float,
        trade_executed: bool,
        env: Any
    ) -> float:
        """Calculate PnL reward."""
        reward = step_return * self.scaling

        if trade_executed:
            reward -= self.trade_cost_penalty

        if position == 0 and self.holding_penalty > 0:
            reward -= self.holding_penalty

        return reward


class SharpeReward(RewardFunction):
    """
    Sharpe ratio-based reward.

    Optimizes risk-adjusted returns by rewarding positive returns
    and penalizing volatility.
    """

    def __init__(
        self,
        lookback: int = 20,
        scaling: float = 1.0,
        volatility_penalty: float = 0.5
    ):
        """
        Initialize Sharpe reward.

        Args:
            lookback: Number of steps to calculate Sharpe over
            scaling: Multiplier for the reward
            volatility_penalty: How much to penalize volatility
        """
        self.lookback = lookback
        self.scaling = scaling
        self.volatility_penalty = volatility_penalty

    def __call__(
        self,
        step_return: float,
        equity: float,
        position: float,
        trade_executed: bool,
        env: Any
    ) -> float:
        """Calculate Sharpe-based reward."""
        # Get recent returns from environment
        if hasattr(env, 'returns_history') and len(env.returns_history) >= self.lookback:
            recent_returns = np.array(env.returns_history[-self.lookback:])
            mean_return = recent_returns.mean()
            std_return = recent_returns.std() + 1e-8

            # Differential Sharpe: reward improvement
            reward = mean_return / std_return
        else:
            # Fallback to simple return reward
            reward = step_return

        return reward * self.scaling


class RiskAdjustedReward(RewardFunction):
    """
    Comprehensive risk-adjusted reward function.

    Combines multiple objectives:
    - Portfolio returns
    - Drawdown penalty
    - Position sizing reward
    - Trade quality metrics
    """

    def __init__(
        self,
        return_weight: float = 1.0,
        drawdown_penalty: float = 2.0,
        max_drawdown_threshold: float = 0.10,
        position_reward: float = 0.1,
        win_rate_bonus: float = 0.5,
        scaling: float = 1.0
    ):
        """
        Initialize risk-adjusted reward.

        Args:
            return_weight: Weight for return component
            drawdown_penalty: Penalty multiplier for drawdowns
            max_drawdown_threshold: Drawdown level that triggers penalty
            position_reward: Bonus for having a position (encourage engagement)
            win_rate_bonus: Bonus based on current win rate
            scaling: Overall reward scaling
        """
        self.return_weight = return_weight
        self.drawdown_penalty = drawdown_penalty
        self.max_drawdown_threshold = max_drawdown_threshold
        self.position_reward = position_reward
        self.win_rate_bonus = win_rate_bonus
        self.scaling = scaling

    def __call__(
        self,
        step_return: float,
        equity: float,
        position: float,
        trade_executed: bool,
        env: Any
    ) -> float:
        """Calculate risk-adjusted reward."""
        reward = 0.0

        # Return component
        reward += step_return * 100 * self.return_weight

        # Drawdown penalty
        if hasattr(env, 'peak_equity') and hasattr(env, 'config'):
            current_drawdown = (env.peak_equity - equity) / env.peak_equity
            if current_drawdown > self.max_drawdown_threshold:
                excess_drawdown = current_drawdown - self.max_drawdown_threshold
                reward -= excess_drawdown * self.drawdown_penalty * 100

        # Position engagement reward
        if position != 0:
            reward += self.position_reward

        # Win rate bonus
        if hasattr(env, 'trade_history') and len(env.trade_history) > 5:
            wins = sum(1 for t in env.trade_history if t['pnl'] > 0)
            win_rate = wins / len(env.trade_history)
            if win_rate > 0.5:
                reward += (win_rate - 0.5) * self.win_rate_bonus

        return reward * self.scaling


class RegimeAwareReward(RewardFunction):
    """
    Reward function that considers market regime.

    Gives bonus for appropriate behavior in different market conditions.
    """

    def __init__(
        self,
        base_scaling: float = 100.0,
        regime_bonus: float = 1.0,
        trend_threshold: float = 0.02,
        volatility_threshold: float = 0.03
    ):
        """
        Initialize regime-aware reward.

        Args:
            base_scaling: Base reward scaling
            regime_bonus: Bonus for regime-appropriate behavior
            trend_threshold: Return threshold to detect trend
            volatility_threshold: ATR threshold for high volatility
        """
        self.base_scaling = base_scaling
        self.regime_bonus = regime_bonus
        self.trend_threshold = trend_threshold
        self.volatility_threshold = volatility_threshold

    def __call__(
        self,
        step_return: float,
        equity: float,
        position: float,
        trade_executed: bool,
        env: Any
    ) -> float:
        """Calculate regime-aware reward."""
        reward = step_return * self.base_scaling

        # Detect regime from environment data
        if hasattr(env, 'features') and hasattr(env, 'current_step'):
            features = env.features

            # Get volatility if available
            volatility = 0.02
            if 'volatility' in features.columns:
                vol_val = features['volatility'].iloc[env.current_step]
                if not np.isnan(vol_val):
                    volatility = vol_val

            # Get trend from returns
            if 'returns' in features.columns:
                recent_returns = features['returns'].iloc[
                    max(0, env.current_step - 20):env.current_step
                ].mean()
            else:
                recent_returns = 0.0

            # Regime detection
            is_trending_up = recent_returns > self.trend_threshold
            is_trending_down = recent_returns < -self.trend_threshold
            is_high_vol = volatility > self.volatility_threshold

            # Regime-appropriate bonuses
            if is_trending_up and position > 0:
                reward += self.regime_bonus  # Long in uptrend
            elif is_trending_down and position < 0:
                reward += self.regime_bonus  # Short in downtrend
            elif not is_trending_up and not is_trending_down and abs(position) < 0.1:
                reward += self.regime_bonus * 0.5  # Flat in choppy market

            if is_high_vol and abs(position) < 0.5:
                reward += self.regime_bonus * 0.5  # Reduced position in high vol

        return reward


class AccumulationReward(RewardFunction):
    """
    Reward for accumulation-focused strategies.

    Encourages building positions at good prices and holding for growth.
    """

    def __init__(
        self,
        return_weight: float = 1.0,
        accumulation_bonus: float = 0.5,
        holding_bonus: float = 0.1,
        dip_buying_bonus: float = 1.0,
        dip_threshold: float = -0.02
    ):
        """
        Initialize accumulation reward.

        Args:
            return_weight: Weight for return component
            accumulation_bonus: Bonus for increasing position
            holding_bonus: Bonus for holding profitable position
            dip_buying_bonus: Bonus for buying dips
            dip_threshold: Return threshold to consider a "dip"
        """
        self.return_weight = return_weight
        self.accumulation_bonus = accumulation_bonus
        self.holding_bonus = holding_bonus
        self.dip_buying_bonus = dip_buying_bonus
        self.dip_threshold = dip_threshold

    def __call__(
        self,
        step_return: float,
        equity: float,
        position: float,
        trade_executed: bool,
        env: Any
    ) -> float:
        """Calculate accumulation reward."""
        reward = step_return * 100 * self.return_weight

        # Accumulation bonus for buying
        if trade_executed and position > 0:
            reward += self.accumulation_bonus

            # Extra bonus for buying dips
            if hasattr(env, 'features') and hasattr(env, 'current_step'):
                features = env.features
                if 'returns' in features.columns:
                    recent_return = features['returns'].iloc[env.current_step - 1]
                    if recent_return < self.dip_threshold:
                        reward += self.dip_buying_bonus

        # Holding bonus for profitable long positions
        if position > 0 and hasattr(env, 'entry_price'):
            current_price = env.data['close'].iloc[env.current_step]
            unrealized_pct = (current_price - env.entry_price) / env.entry_price
            if unrealized_pct > 0:
                reward += self.holding_bonus

        return reward


def create_reward_function(
    reward_type: str = 'pnl',
    **kwargs
) -> RewardFunction:
    """
    Factory function to create reward functions.

    Args:
        reward_type: Type of reward function
        **kwargs: Arguments for the reward function

    Returns:
        RewardFunction instance
    """
    reward_classes = {
        'pnl': PnLReward,
        'sharpe': SharpeReward,
        'risk_adjusted': RiskAdjustedReward,
        'regime_aware': RegimeAwareReward,
        'accumulation': AccumulationReward
    }

    if reward_type not in reward_classes:
        raise ValueError(f"Unknown reward type: {reward_type}. Choose from {list(reward_classes.keys())}")

    return reward_classes[reward_type](**kwargs)


class CompositeReward(RewardFunction):
    """
    Combine multiple reward functions.

    Allows weighted combination of different reward objectives.
    """

    def __init__(
        self,
        reward_functions: list,
        weights: Optional[list] = None
    ):
        """
        Initialize composite reward.

        Args:
            reward_functions: List of RewardFunction instances
            weights: Optional weights for each function (default: equal)
        """
        self.reward_functions = reward_functions

        if weights is None:
            self.weights = [1.0 / len(reward_functions)] * len(reward_functions)
        else:
            if len(weights) != len(reward_functions):
                raise ValueError("Number of weights must match number of reward functions")
            total = sum(weights)
            self.weights = [w / total for w in weights]

    def __call__(
        self,
        step_return: float,
        equity: float,
        position: float,
        trade_executed: bool,
        env: Any
    ) -> float:
        """Calculate weighted combination of rewards."""
        total_reward = 0.0

        for reward_fn, weight in zip(self.reward_functions, self.weights):
            r = reward_fn(step_return, equity, position, trade_executed, env)
            total_reward += r * weight

        return total_reward
