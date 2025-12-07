import gymnasium as gym
from gymnasium import spaces
import numpy as np
import warnings
import os

# Enable ROCm/CUDA for GPU training
warnings.filterwarnings('ignore', message='.*expandable_segments.*')
warnings.filterwarnings('ignore', message='.*hipBLASLt.*')

from stable_baselines3 import PPO
import sys
sys.path.insert(0, '..')
from portfolio import Portfolio


class TradingEnv(gym.Env):
    """
    Custom Trading Environment for RL agent.
    Phase 8: Defensive leverage with volatility penalty.
    Goals: Accumulate BTC (45%), XRP (35%), USDT (20% as collateral)
    Features: Vol penalty discourages leverage in choppy markets.
    """

    def __init__(self, data_dict, initial_balance=None):
        super().__init__()
        self.data = data_dict  # multi-symbol OHLCV
        self.symbols = list(data_dict.keys())
        self.initial_balance = initial_balance or {'USDT': 1000.0, 'XRP': 500.0, 'BTC': 0.0}

        # Phase 7: Simplified targets - USDT as collateral/safety
        self.targets = {'BTC': 0.45, 'XRP': 0.35, 'USDT': 0.20}

        # Actions: 0-2 BTC (buy/hold/sell), 3-5 XRP, 6-8 USDT (park/hold/deploy)
        self.action_space = spaces.Discrete(9)

        # Phase 8: Extended observations with volatility features
        # 4 features per symbol + 3 portfolio + 2 volatility features
        n_features = len(self.symbols) * 4 + 3 + 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32)

        self.current_step = 0
        self.max_steps = min(len(df) for df in data_dict.values()) - 1
        self.portfolio = None

        # Leverage tracking for margin positions
        self.margin_positions = {'BTC': 0.0, 'XRP': 0.0}
        self.margin_entries = {'BTC': 0.0, 'XRP': 0.0}  # Track entry prices
        self.leverage = 10  # Kraken 10x

        # Phase 8: Volatility tracking
        self.current_volatility = 0.0
        self.vol_high_threshold = 0.05  # ATR% above this = high vol

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 60  # warmup period for indicators
        self.portfolio = Portfolio(self.initial_balance.copy())
        self.margin_positions = {'BTC': 0.0, 'XRP': 0.0}
        self.margin_entries = {'BTC': 0.0, 'XRP': 0.0}
        self.current_volatility = 0.0
        return self._get_obs(), {}

    def _calculate_atr_pct(self, df, period=14):
        """Calculate ATR as percentage of price."""
        if len(df) < period + 1:
            return 0.05

        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        tr_list = []
        for i in range(1, min(len(close), self.current_step + 1)):
            tr = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
            tr_list.append(tr)

        if len(tr_list) < period:
            return 0.05

        atr = np.mean(tr_list[-period:])
        current_price = close[min(self.current_step, len(close) - 1)]
        return atr / current_price if current_price > 0 else 0.05

    def _get_obs(self):
        obs = []
        volatilities = []

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

                # Phase 8: Track ATR% for vol penalty
                atr_pct = self._calculate_atr_pct(df)
                volatilities.append(atr_pct)

        # Portfolio state - simplified to BTC, XRP, USDT
        prices = self._current_prices()
        total = max(self.portfolio.get_total_usd(prices), 1.0)
        for asset in ['BTC', 'XRP', 'USDT']:
            weight = self.portfolio.balances.get(asset, 0) * prices.get(asset, 1.0) / total
            obs.append(weight)

        # Phase 8: Add volatility features to observation
        avg_volatility = np.mean(volatilities) if volatilities else 0.05
        self.current_volatility = avg_volatility
        obs.append(avg_volatility)  # Current volatility
        obs.append(1.0 if avg_volatility > self.vol_high_threshold else 0.0)  # High vol flag

        return np.array(obs, dtype=np.float32)

    def _current_prices(self):
        prices = {'USDT': 1.0}
        for sym in self.symbols:
            df = self.data[sym]
            if self.current_step < len(df):
                base = sym.split('/')[0]
                prices[base] = df.iloc[self.current_step]['close']
        return prices

    def step(self, action):
        prices = self._current_prices()
        prev_value = self.portfolio.get_total_usd(prices)

        # Execute action
        # Actions 0-2: BTC (buy/hold/sell)
        # Actions 3-5: XRP (buy/hold/sell)
        # Actions 6-8: USDT operations (park more / hold / deploy to margin)
        assets = ['BTC', 'XRP', 'USDT']
        asset_idx = action // 3
        action_type = action % 3  # 0=buy/park, 1=hold, 2=sell/deploy

        if asset_idx < len(assets):
            asset = assets[asset_idx]
            price = prices.get(asset, 1.0)

            if asset == 'USDT':
                # USDT actions: 6=park (sell assets), 7=hold, 8=deploy (buy assets)
                if action_type == 0:  # Park - sell some XRP/BTC to USDT
                    for sell_asset in ['XRP', 'BTC']:
                        holding = self.portfolio.balances.get(sell_asset, 0)
                        if holding > 0:
                            sell_amount = holding * 0.05  # Sell 5%
                            self.portfolio.update(sell_asset, -sell_amount)
                            self.portfolio.update('USDT', sell_amount * prices.get(sell_asset, 1.0))
                elif action_type == 2:  # Deploy - open leveraged position
                    usdt = self.portfolio.balances.get('USDT', 0)
                    # Phase 8: Only deploy if volatility is acceptable
                    if usdt > 100 and self.current_volatility < self.vol_high_threshold:
                        # Simulate 10x leverage position on XRP
                        margin_usdt = usdt * 0.1  # Use 10% as margin
                        exposure = margin_usdt * self.leverage
                        xrp_price = prices.get('XRP', 1.0)
                        position_size = exposure / xrp_price
                        self.margin_positions['XRP'] += position_size
                        self.margin_entries['XRP'] = xrp_price  # Track entry
                        self.portfolio.update('USDT', -margin_usdt)  # Lock margin
            else:
                # BTC/XRP spot trades
                if action_type == 0:  # Buy
                    usdt = self.portfolio.balances.get('USDT', 0)
                    buy_amount = usdt * 0.15 / price  # 15% of USDT (more aggressive)
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

        # Calculate reward with USDT-biased goal alignment + vol penalty
        new_value = self.portfolio.get_total_usd(prices)
        total = max(new_value, 1.0)

        # Base reward: portfolio return
        base_reward = (new_value - prev_value) / prev_value if prev_value > 0 else 0

        # Alignment reward: match target allocation (stronger weight = 3.0)
        alignment_score = 0.0
        weights = {}
        for asset in self.targets:
            asset_value = self.portfolio.balances.get(asset, 0) * prices.get(asset, 1.0)
            weights[asset] = asset_value / total
            alignment_score += min(weights[asset], self.targets[asset])

        # USDT safety bonus: reward for maintaining USDT reserves during drawdowns
        usdt_weight = weights.get('USDT', 0)
        usdt_bonus = 0.0
        if usdt_weight >= 0.15:  # At least 15% in USDT
            usdt_bonus = 0.1  # Safety bonus

        # Phase 8: Enhanced USDT bonus during high volatility
        if self.current_volatility > self.vol_high_threshold and usdt_weight >= 0.20:
            usdt_bonus += 0.2  # Extra bonus for parking during chop

        # Margin P&L from leveraged positions (with actual entry tracking)
        margin_pnl = 0.0
        for asset, position in self.margin_positions.items():
            if position > 0:
                current_price = prices.get(asset, 1.0)
                entry_price = self.margin_entries.get(asset, current_price)
                # Real P&L calculation
                pnl_pct = (current_price - entry_price) / entry_price if entry_price > 0 else 0
                margin_pnl += position * pnl_pct * 0.1  # Scaled down

        # Phase 8: Volatility penalty - discourage leverage in choppy markets
        vol_penalty = 0.0
        has_leverage = sum(self.margin_positions.values()) > 0
        if has_leverage and self.current_volatility > self.vol_high_threshold:
            # Penalize holding leveraged positions during high volatility
            vol_penalty = self.current_volatility * 2.0

        # Combined reward: base + strong alignment + USDT safety + margin - vol penalty
        reward = base_reward + 3.0 * alignment_score + usdt_bonus + margin_pnl - vol_penalty

        done = self.current_step >= self.max_steps
        truncated = False

        return self._get_obs(), reward, done, truncated, {}


def train_rl_agent(data_dict, timesteps=100000, device="cuda"):
    """
    Train PPO agent on trading environment.
    Phase 7: Use GPU (cuda/rocm) for faster training.
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
        device=device
    )
    print(f"Training PPO agent for {timesteps} timesteps on {device.upper()}...")
    model.learn(total_timesteps=timesteps)
    model.save("models/rl_ppo_agent")
    print("Model saved to models/rl_ppo_agent")
    return model


def load_rl_agent(path="models/rl_ppo_agent", device="cuda"):
    """Load trained RL agent"""
    return PPO.load(path, device=device)
