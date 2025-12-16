"""
RL Agent Training and Evaluation

Provides wrappers for Stable-Baselines3 PPO and SAC agents
with trading-specific configurations.
"""

from typing import Optional, Dict, Any, Union, Callable, List, Tuple
from pathlib import Path
import json
import os
import numpy as np

# Set HSA override for AMD GPUs before any torch import
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '10.3.0')

try:
    import torch
except ImportError:
    torch = None

try:
    from stable_baselines3 import PPO, SAC, A2C
    from stable_baselines3.common.callbacks import (
        BaseCallback,
        EvalCallback,
        CheckpointCallback
    )
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.monitor import Monitor
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    PPO = SAC = A2C = None

from .environment import TradingEnv, TradingEnvConfig


class TradingCallback(BaseCallback if SB3_AVAILABLE else object):
    """
    Custom callback for trading-specific logging.

    Logs trading metrics like win rate, Sharpe ratio, and drawdown
    during training.
    """

    def __init__(
        self,
        log_freq: int = 1000,
        verbose: int = 1
    ):
        """
        Initialize trading callback.

        Args:
            log_freq: How often to log metrics (in steps)
            verbose: Verbosity level
        """
        if SB3_AVAILABLE:
            super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_returns = []
        self.episode_trades = []

    def _on_step(self) -> bool:
        """Called after each step."""
        # Get episode info from environment
        if 'infos' in self.locals:
            for info in self.locals['infos']:
                if 'total_return' in info:
                    self.episode_returns.append(info['total_return'])
                if 'num_trades' in info:
                    self.episode_trades.append(info['num_trades'])

        # Log periodically
        if self.num_timesteps % self.log_freq == 0 and self.episode_returns:
            avg_return = np.mean(self.episode_returns[-100:])
            avg_trades = np.mean(self.episode_trades[-100:]) if self.episode_trades else 0

            if self.verbose > 0:
                print(f"Step {self.num_timesteps}: "
                      f"Avg Return={avg_return:.2f}%, "
                      f"Avg Trades={avg_trades:.1f}")

            # Log to tensorboard if available
            if hasattr(self, 'logger') and self.logger is not None:
                self.logger.record('trading/avg_return', avg_return)
                self.logger.record('trading/avg_trades', avg_trades)

        return True


def train_ppo(
    env: Union[TradingEnv, Callable],
    total_timesteps: int = 1_000_000,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    device: str = 'auto',
    tensorboard_log: Optional[str] = None,
    save_path: Optional[str] = None,
    eval_env: Optional[TradingEnv] = None,
    eval_freq: int = 10000,
    n_eval_episodes: int = 10,
    verbose: int = 1,
    seed: Optional[int] = None,
    **kwargs
) -> 'PPO':
    """
    Train a PPO agent on the trading environment.

    Args:
        env: Trading environment or callable that creates environment
        total_timesteps: Total training timesteps
        learning_rate: Learning rate
        n_steps: Steps per update
        batch_size: Minibatch size
        n_epochs: Number of epochs per update
        gamma: Discount factor
        gae_lambda: GAE lambda
        clip_range: PPO clip range
        ent_coef: Entropy coefficient
        vf_coef: Value function coefficient
        max_grad_norm: Max gradient norm
        device: Training device ('auto', 'cuda', 'cpu')
        tensorboard_log: TensorBoard log directory
        save_path: Path to save trained model
        eval_env: Evaluation environment
        eval_freq: Evaluation frequency
        n_eval_episodes: Number of evaluation episodes
        verbose: Verbosity level
        seed: Random seed
        **kwargs: Additional PPO arguments

    Returns:
        Trained PPO model
    """
    if not SB3_AVAILABLE:
        raise ImportError("stable-baselines3 is required. Install with: pip install stable-baselines3")

    # Determine device
    if device == 'auto':
        if torch is not None and torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

    if verbose > 0:
        print(f"Training PPO agent on {device.upper()}")

    # Wrap environment if needed
    if callable(env) and not isinstance(env, TradingEnv):
        env = DummyVecEnv([env])
    elif isinstance(env, TradingEnv):
        env = DummyVecEnv([lambda: env])

    # Create model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        device=device,
        tensorboard_log=tensorboard_log,
        verbose=verbose,
        seed=seed,
        **kwargs
    )

    # Setup callbacks
    callbacks = [TradingCallback(verbose=verbose)]

    if eval_env is not None:
        eval_env = DummyVecEnv([lambda: eval_env])
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=save_path,
            log_path=tensorboard_log,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            verbose=verbose
        )
        callbacks.append(eval_callback)

    if save_path:
        checkpoint_callback = CheckpointCallback(
            save_freq=50000,
            save_path=save_path,
            name_prefix='ppo_trading'
        )
        callbacks.append(checkpoint_callback)

    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=verbose > 0
    )

    # Save final model
    if save_path:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        model.save(Path(save_path) / "ppo_final")
        if verbose > 0:
            print(f"Model saved to {save_path}/ppo_final")

    return model


def train_sac(
    env: Union[TradingEnv, Callable],
    total_timesteps: int = 1_000_000,
    learning_rate: float = 3e-4,
    buffer_size: int = 1_000_000,
    learning_starts: int = 10000,
    batch_size: int = 256,
    tau: float = 0.005,
    gamma: float = 0.99,
    train_freq: int = 1,
    gradient_steps: int = 1,
    ent_coef: str = 'auto',
    target_entropy: str = 'auto',
    device: str = 'auto',
    tensorboard_log: Optional[str] = None,
    save_path: Optional[str] = None,
    eval_env: Optional[TradingEnv] = None,
    eval_freq: int = 10000,
    n_eval_episodes: int = 10,
    verbose: int = 1,
    seed: Optional[int] = None,
    **kwargs
) -> 'SAC':
    """
    Train a SAC agent on the trading environment.

    SAC is better for continuous action spaces and exploration-heavy tasks.

    Args:
        env: Trading environment or callable
        total_timesteps: Total training timesteps
        learning_rate: Learning rate
        buffer_size: Replay buffer size
        learning_starts: Steps before training starts
        batch_size: Batch size for training
        tau: Soft update coefficient
        gamma: Discount factor
        train_freq: Training frequency
        gradient_steps: Gradient steps per update
        ent_coef: Entropy regularization coefficient
        target_entropy: Target entropy for automatic tuning
        device: Training device
        tensorboard_log: TensorBoard log directory
        save_path: Path to save trained model
        eval_env: Evaluation environment
        eval_freq: Evaluation frequency
        n_eval_episodes: Number of evaluation episodes
        verbose: Verbosity level
        seed: Random seed
        **kwargs: Additional SAC arguments

    Returns:
        Trained SAC model
    """
    if not SB3_AVAILABLE:
        raise ImportError("stable-baselines3 is required. Install with: pip install stable-baselines3")

    # Determine device
    if device == 'auto':
        if torch is not None and torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

    if verbose > 0:
        print(f"Training SAC agent on {device.upper()}")

    # Wrap environment if needed
    if callable(env) and not isinstance(env, TradingEnv):
        env = DummyVecEnv([env])
    elif isinstance(env, TradingEnv):
        env = DummyVecEnv([lambda: env])

    # Create model
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        tau=tau,
        gamma=gamma,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        ent_coef=ent_coef,
        target_entropy=target_entropy,
        device=device,
        tensorboard_log=tensorboard_log,
        verbose=verbose,
        seed=seed,
        **kwargs
    )

    # Setup callbacks
    callbacks = [TradingCallback(verbose=verbose)]

    if eval_env is not None:
        eval_env = DummyVecEnv([lambda: eval_env])
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=save_path,
            log_path=tensorboard_log,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            verbose=verbose
        )
        callbacks.append(eval_callback)

    if save_path:
        checkpoint_callback = CheckpointCallback(
            save_freq=50000,
            save_path=save_path,
            name_prefix='sac_trading'
        )
        callbacks.append(checkpoint_callback)

    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=verbose > 0
    )

    # Save final model
    if save_path:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        model.save(Path(save_path) / "sac_final")
        if verbose > 0:
            print(f"Model saved to {save_path}/sac_final")

    return model


def load_agent(
    path: Union[str, Path],
    agent_type: str = 'ppo',
    device: str = 'auto',
    env: Optional[TradingEnv] = None
) -> Union['PPO', 'SAC']:
    """
    Load a trained RL agent.

    Args:
        path: Path to saved model
        agent_type: Type of agent ('ppo' or 'sac')
        device: Device to load model on
        env: Optional environment for the model

    Returns:
        Loaded model
    """
    if not SB3_AVAILABLE:
        raise ImportError("stable-baselines3 is required")

    path = Path(path)

    if agent_type.lower() == 'ppo':
        model = PPO.load(path, device=device, env=env)
    elif agent_type.lower() == 'sac':
        model = SAC.load(path, device=device, env=env)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    return model


def evaluate_agent(
    model: Union['PPO', 'SAC'],
    env: TradingEnv,
    n_episodes: int = 100,
    deterministic: bool = True,
    verbose: int = 0
) -> Dict[str, Any]:
    """
    Evaluate a trained RL agent.

    Args:
        model: Trained model
        env: Evaluation environment
        n_episodes: Number of episodes to evaluate
        deterministic: Use deterministic actions
        verbose: Verbosity level

    Returns:
        Dictionary with evaluation metrics
    """
    episode_rewards = []
    episode_returns = []
    episode_lengths = []
    episode_trades = []
    episode_sharpes = []
    episode_drawdowns = []
    episode_win_rates = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        # Get episode statistics
        stats = env.get_episode_stats()

        episode_rewards.append(episode_reward)
        episode_returns.append(stats.get('total_return', 0))
        episode_lengths.append(info.get('episode_steps', 0))
        episode_trades.append(stats.get('num_trades', 0))
        episode_sharpes.append(stats.get('sharpe_ratio', 0))
        episode_drawdowns.append(stats.get('max_drawdown', 0))
        episode_win_rates.append(stats.get('win_rate', 0))

        if verbose > 0 and (ep + 1) % 10 == 0:
            print(f"Episode {ep + 1}/{n_episodes}: "
                  f"Return={stats.get('total_return', 0):.2f}%, "
                  f"Trades={stats.get('num_trades', 0)}")

    results = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_return': np.mean(episode_returns),
        'std_return': np.std(episode_returns),
        'mean_trades': np.mean(episode_trades),
        'mean_sharpe': np.mean(episode_sharpes),
        'mean_drawdown': np.mean(episode_drawdowns),
        'mean_win_rate': np.mean(episode_win_rates),
        'mean_episode_length': np.mean(episode_lengths),
        'n_episodes': n_episodes
    }

    if verbose > 0:
        print("\n=== Evaluation Results ===")
        print(f"Mean Return: {results['mean_return']:.2f}% (+/- {results['std_return']:.2f}%)")
        print(f"Mean Sharpe: {results['mean_sharpe']:.3f}")
        print(f"Mean Drawdown: {results['mean_drawdown']:.2f}%")
        print(f"Mean Win Rate: {results['mean_win_rate']:.1f}%")
        print(f"Mean Trades: {results['mean_trades']:.1f}")

    return results


def compare_agents(
    agents: Dict[str, Union['PPO', 'SAC']],
    env: TradingEnv,
    n_episodes: int = 50,
    verbose: int = 1
) -> Dict[str, Dict[str, Any]]:
    """
    Compare multiple trained agents.

    Args:
        agents: Dictionary mapping agent name to model
        env: Evaluation environment
        n_episodes: Episodes per agent
        verbose: Verbosity level

    Returns:
        Dictionary mapping agent name to evaluation results
    """
    results = {}

    for name, model in agents.items():
        if verbose > 0:
            print(f"\nEvaluating {name}...")

        agent_results = evaluate_agent(
            model, env, n_episodes,
            deterministic=True, verbose=0
        )
        results[name] = agent_results

        if verbose > 0:
            print(f"  Return: {agent_results['mean_return']:.2f}%")
            print(f"  Sharpe: {agent_results['mean_sharpe']:.3f}")
            print(f"  Drawdown: {agent_results['mean_drawdown']:.2f}%")

    return results


class RLTradingAgent:
    """
    High-level wrapper for RL trading agents.

    Provides easy-to-use interface for training, saving, loading,
    and getting trading actions.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        agent_type: str = 'ppo',
        device: str = 'auto'
    ):
        """
        Initialize RL trading agent.

        Args:
            model_path: Path to saved model (optional)
            agent_type: Type of agent
            device: Device to use
        """
        self.agent_type = agent_type.lower()
        self.device = device
        self.model = None
        self.config = None

        if model_path is not None:
            self.load(model_path)

    def train(
        self,
        env: TradingEnv,
        total_timesteps: int = 500000,
        save_path: Optional[str] = None,
        **kwargs
    ) -> None:
        """Train the agent."""
        if self.agent_type == 'ppo':
            self.model = train_ppo(
                env,
                total_timesteps=total_timesteps,
                save_path=save_path,
                device=self.device,
                **kwargs
            )
        elif self.agent_type == 'sac':
            self.model = train_sac(
                env,
                total_timesteps=total_timesteps,
                save_path=save_path,
                device=self.device,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown agent type: {self.agent_type}")

    def load(self, path: Union[str, Path]) -> None:
        """Load trained model."""
        self.model = load_agent(path, self.agent_type, self.device)

    def save(self, path: Union[str, Path]) -> None:
        """Save trained model."""
        if self.model is None:
            raise RuntimeError("No model to save. Train or load a model first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Get action for observation.

        Args:
            observation: Environment observation
            deterministic: Use deterministic policy

        Returns:
            Tuple of (action, info_dict)
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Train or load a model first.")

        action, _states = self.model.predict(observation, deterministic=deterministic)

        # Map action to trading signal
        if isinstance(action, np.ndarray):
            action_val = action[0] if len(action) == 1 else action
        else:
            action_val = action

        # Convert to trading signal
        action_map = {
            0: {'action': 'strong_sell', 'confidence': 0.9},
            1: {'action': 'sell', 'confidence': 0.7},
            2: {'action': 'hold', 'confidence': 0.5},
            3: {'action': 'buy', 'confidence': 0.7},
            4: {'action': 'strong_buy', 'confidence': 0.9}
        }

        signal = action_map.get(action_val, {'action': 'hold', 'confidence': 0.5})

        return action, signal

    def evaluate(
        self,
        env: TradingEnv,
        n_episodes: int = 100
    ) -> Dict[str, Any]:
        """Evaluate the agent."""
        if self.model is None:
            raise RuntimeError("No model loaded.")

        return evaluate_agent(self.model, env, n_episodes)
