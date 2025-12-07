import gymnasium as gym
from gymnasium import spaces
import numpy as np
import warnings
import os

# Suppress ROCm/HIP warnings for cleaner output
os.environ['HIP_VISIBLE_DEVICES'] = ''  # Force CPU for RL (PPO+MLP runs faster on CPU)
warnings.filterwarnings('ignore', message='.*expandable_segments.*')
warnings.filterwarnings('ignore', message='.*hipBLASLt.*')

from stable_baselines3 import PPO
import sys
sys.path.insert(0, '..')
from portfolio import Portfolio


class TradingEnv(gym.Env):
    """Custom Trading Environment for RL agent focused on accumulation"""

    def __init__(self, data_dict, initial_balance=None):
        super().__init__()
        self.data = data_dict  # multi-symbol OHLCV
        self.symbols = list(data_dict.keys())
        self.initial_balance = initial_balance or {'USDT': 1000.0, 'XRP': 500.0}

        # Actions: 0=hold, 1=buy_small, 2=buy_large, 3=sell_small, 4=sell_large for each asset
        # Simplified: 9 actions (3 assets x 3 actions: buy/hold/sell)
        self.action_space = spaces.Discrete(9)

        # Observations: prices, volumes, portfolio weights, momentum indicators
        n_features = len(self.symbols) * 4 + 5  # 4 features per symbol + 5 portfolio features
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32)

        self.current_step = 0
        self.max_steps = min(len(df) for df in data_dict.values()) - 1
        self.portfolio = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 60  # warmup period for indicators
        self.portfolio = Portfolio(self.initial_balance.copy())
        return self._get_obs(), {}

    def _get_obs(self):
        obs = []
        for sym in self.symbols:
            df = self.data[sym]
            if self.current_step < len(df):
                row = df.iloc[self.current_step]
                # Normalized features
                close = row['close']
                volume = row['volume']
                # Simple momentum: % change from 10 periods ago
                prev_close = df.iloc[max(0, self.current_step - 10)]['close']
                momentum = (close - prev_close) / prev_close if prev_close > 0 else 0
                # Volatility: std of last 20 closes
                volatility = df['close'].iloc[max(0, self.current_step - 20):self.current_step].std()
                obs.extend([close, volume, momentum, volatility if not np.isnan(volatility) else 0])

        # Portfolio state
        prices = self._current_prices()
        total = max(self.portfolio.get_total_usd(prices), 1.0)
        for asset in ['BTC', 'XRP', 'RLUSD', 'USDT', 'USDC']:
            weight = self.portfolio.balances.get(asset, 0) * prices.get(asset, 1.0) / total
            obs.append(weight)

        return np.array(obs, dtype=np.float32)

    def _current_prices(self):
        prices = {'USDT': 1.0, 'USDC': 1.0, 'RLUSD': 1.0}
        for sym in self.symbols:
            df = self.data[sym]
            if self.current_step < len(df):
                base = sym.split('/')[0]
                prices[base] = df.iloc[self.current_step]['close']
        return prices

    def step(self, action):
        prices = self._current_prices()
        prev_value = self.portfolio.get_total_usd(prices)

        # Execute action (simplified trading logic)
        assets = ['BTC', 'XRP', 'RLUSD']
        asset_idx = action // 3
        action_type = action % 3  # 0=buy, 1=hold, 2=sell

        if asset_idx < len(assets):
            asset = assets[asset_idx]
            price = prices.get(asset, 1.0)

            if action_type == 0:  # Buy
                usdt = self.portfolio.balances.get('USDT', 0)
                buy_amount = usdt * 0.1 / price  # 10% of USDT
                if buy_amount > 0 and usdt >= buy_amount * price:
                    self.portfolio.update('USDT', -buy_amount * price)
                    self.portfolio.update(asset, buy_amount)
            elif action_type == 2:  # Sell
                holding = self.portfolio.balances.get(asset, 0)
                sell_amount = holding * 0.1  # 10% of holding
                if sell_amount > 0:
                    self.portfolio.update(asset, -sell_amount)
                    self.portfolio.update('USDT', sell_amount * price)

        self.current_step += 1

        # Calculate reward (focus on accumulation + total value)
        new_value = self.portfolio.get_total_usd(prices)
        value_reward = (new_value - prev_value) / prev_value if prev_value > 0 else 0

        # Bonus for holding target assets (BTC, XRP, RLUSD)
        accumulation_bonus = 0
        for asset in ['BTC', 'XRP', 'RLUSD']:
            if self.portfolio.balances.get(asset, 0) > self.initial_balance.get(asset, 0):
                accumulation_bonus += 0.001

        reward = value_reward + accumulation_bonus

        done = self.current_step >= self.max_steps
        truncated = False

        return self._get_obs(), reward, done, truncated, {}


def train_rl_agent(data_dict, timesteps=100000, device="cpu"):
    """
    Train PPO agent on trading environment.

    Note: PPO with MLP policy runs faster on CPU than GPU.
    Use device="cuda" only for CNN policies or very large batch sizes.
    """
    env = TradingEnv(data_dict)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./tensorboard/",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        device=device  # CPU is faster for MLP policies
    )
    print(f"Training PPO agent for {timesteps} timesteps on {device.upper()}...")
    model.learn(total_timesteps=timesteps)
    model.save("models/rl_ppo_agent")
    print("Model saved to models/rl_ppo_agent")
    return model


def load_rl_agent(path="models/rl_ppo_agent", device="cpu"):
    """Load trained RL agent"""
    return PPO.load(path, device=device)
