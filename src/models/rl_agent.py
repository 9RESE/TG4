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
    Phase 12: Bear Profit Flip - Paired Shorts + Real Yield + Live Prep.
    Goals: Accumulate BTC (45%), XRP (35%), USDT (20% as collateral)
    Features:
    - Vol penalty discourages leverage in choppy markets
    - Rebound bonus rewards correctly timed dip-buys
    - RSI-based offense triggers in greed regime
    - SHORT actions for bear market profit capture
    - Phase 12: Paired bear mode (rip_short vs grind_down)
    - Phase 12: Real USDT yield (6% APY from Kraken/Bitrue lending)
    - Phase 12: Short precision rewards (6.0x for whipsaw-free shorts)
    - Phase 12: 2x yield bonus during defensive parking
    - Decay penalty prevents holding shorts too long
    """

    def __init__(self, data_dict, initial_balance=None):
        super().__init__()
        self.data = data_dict  # multi-symbol OHLCV
        self.symbols = list(data_dict.keys())
        self.initial_balance = initial_balance or {'USDT': 1000.0, 'XRP': 500.0, 'BTC': 0.0}

        # Phase 7: Simplified targets - USDT as collateral/safety
        self.targets = {'BTC': 0.45, 'XRP': 0.35, 'USDT': 0.20}

        # Phase 10: Extended actions with shorts
        # 0-2: BTC (buy/hold/sell), 3-5: XRP (buy/hold/sell)
        # 6-8: USDT (park/hold/deploy), 9-10: SHORT (BTC/XRP)
        # 11: Close all shorts
        self.action_space = spaces.Discrete(12)

        # Phase 10: Extended observations with volatility + RSI + short position features
        # 4 features per symbol + 3 portfolio + 2 volatility + 2 RSI + 2 short positions
        n_features = len(self.symbols) * 4 + 3 + 4 + 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32)

        self.current_step = 0
        self.max_steps = min(len(df) for df in data_dict.values()) - 1
        self.portfolio = None

        # Leverage tracking for margin positions (longs)
        self.margin_positions = {'BTC': 0.0, 'XRP': 0.0}
        self.margin_entries = {'BTC': 0.0, 'XRP': 0.0}  # Track entry prices
        self.leverage = 10  # Kraken 10x

        # Phase 10: Short position tracking
        self.short_positions = {'BTC': 0.0, 'XRP': 0.0}
        self.short_entries = {'BTC': 0.0, 'XRP': 0.0}
        self.short_open_step = {'BTC': 0, 'XRP': 0}  # Track when shorts opened
        self.short_leverage = 5  # Conservative 5x for shorts
        self.max_short_duration = 336  # ~14 days in hourly steps

        # Phase 8: Volatility tracking
        self.current_volatility = 0.0
        self.vol_high_threshold = 0.04  # Phase 11: Lowered from 0.05 - ATR% for bear signal
        self.vol_low_threshold = 0.02   # ATR% below this = greed/calm

        # Phase 9: RSI tracking for rebound detection
        self.current_rsi = {'XRP': 50.0, 'BTC': 50.0}
        self.rsi_oversold = 30  # RSI below this = oversold dip
        self.rsi_overbought = 65  # Phase 11: Lowered from 70 - earlier overbought detection
        self.rsi_short_exit = 40  # Phase 11: Auto-exit shorts when RSI drops below this
        self.rsi_rip_threshold = 72  # Phase 12: Overbought rip threshold (aggressive short)

        # Phase 12: Real USDT yield (6% APY from Kraken/Bitrue lending)
        self.usdt_yield_apy = 0.06  # 6% APY realistic rate
        self.usdt_yield_per_step = self.usdt_yield_apy / 365 / 24 * 4  # ~4 hours per step

        # Phase 12: Short precision tracking (reward whipsaw-free shorts)
        self.short_peak_price = {'BTC': 0.0, 'XRP': 0.0}  # Track highest price during short
        self.short_whipsaw_threshold = 0.05  # 5% bounce = whipsaw

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 60  # warmup period for indicators
        self.portfolio = Portfolio(self.initial_balance.copy())
        self.margin_positions = {'BTC': 0.0, 'XRP': 0.0}
        self.margin_entries = {'BTC': 0.0, 'XRP': 0.0}
        # Phase 10: Reset short positions
        self.short_positions = {'BTC': 0.0, 'XRP': 0.0}
        self.short_entries = {'BTC': 0.0, 'XRP': 0.0}
        self.short_open_step = {'BTC': 0, 'XRP': 0}
        # Phase 12: Reset short precision tracking
        self.short_peak_price = {'BTC': 0.0, 'XRP': 0.0}
        self.current_volatility = 0.0
        self.current_rsi = {'XRP': 50.0, 'BTC': 50.0}
        return self._get_obs(), {}

    def _calculate_rsi(self, close_prices, period=14):
        """Calculate RSI indicator."""
        if len(close_prices) < period + 1:
            return 50.0

        deltas = np.diff(close_prices[-period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0.001

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

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

        # Phase 9: Add RSI features for rebound detection
        for sym in self.symbols:
            df = self.data[sym]
            if self.current_step < len(df):
                close_prices = df['close'].iloc[:self.current_step+1].values
                rsi = self._calculate_rsi(close_prices)
                asset = sym.split('/')[0]
                self.current_rsi[asset] = rsi
                obs.append(rsi / 100.0)  # Normalized RSI (0-1)

        # Phase 10: Add short position features
        for asset in ['BTC', 'XRP']:
            short_size = self.short_positions.get(asset, 0)
            obs.append(1.0 if short_size > 0 else 0.0)  # Has short flag

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
        prev_prices = prices.copy()  # Track for short P&L

        # Execute action
        # Actions 0-2: BTC (buy/hold/sell)
        # Actions 3-5: XRP (buy/hold/sell)
        # Actions 6-8: USDT operations (park more / hold / deploy to margin)
        # Phase 10: Actions 9-11: SHORT operations
        # 9: Short BTC, 10: Short XRP, 11: Close all shorts

        if action <= 8:
            # Original actions (0-8)
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

        elif action == 9:  # Short BTC
            usdt = self.portfolio.balances.get('USDT', 0)
            if usdt > 100 and self.short_positions['BTC'] == 0:
                # Open BTC short with 5x leverage, 8% of USDT
                margin = usdt * 0.08
                btc_price = prices.get('BTC', 90000.0)
                exposure = margin * self.short_leverage
                size = exposure / btc_price
                self.short_positions['BTC'] = size
                self.short_entries['BTC'] = btc_price
                self.short_open_step['BTC'] = self.current_step
                self.portfolio.update('USDT', -margin)

        elif action == 10:  # Short XRP
            usdt = self.portfolio.balances.get('USDT', 0)
            if usdt > 100 and self.short_positions['XRP'] == 0:
                # Open XRP short with 5x leverage, 8% of USDT
                margin = usdt * 0.08
                xrp_price = prices.get('XRP', 2.0)
                exposure = margin * self.short_leverage
                size = exposure / xrp_price
                self.short_positions['XRP'] = size
                self.short_entries['XRP'] = xrp_price
                self.short_open_step['XRP'] = self.current_step
                self.portfolio.update('USDT', -margin)

        elif action == 11:  # Close all shorts
            for asset in ['BTC', 'XRP']:
                if self.short_positions[asset] > 0:
                    current_price = prices.get(asset, 1.0)
                    entry_price = self.short_entries[asset]
                    size = self.short_positions[asset]
                    # Short P&L: profit when price drops
                    pnl = (entry_price - current_price) * size
                    # Return collateral + P&L
                    collateral = (size * entry_price) / self.short_leverage
                    self.portfolio.update('USDT', collateral + pnl)
                    self.short_positions[asset] = 0.0
                    self.short_entries[asset] = 0.0

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

        # Phase 12: Real USDT yield (6% APY from Kraken/Bitrue lending)
        usdt_yield = 0.0
        usdt_yield_factor = 0.0
        usdt_balance = self.portfolio.balances.get('USDT', 0)
        is_defensive = self.current_volatility > self.vol_high_threshold or usdt_weight >= 0.20
        if usdt_balance > 100 and is_defensive:
            # Apply real yield to balance
            yield_earned = self.usdt_yield_per_step * usdt_balance
            self.portfolio.update('USDT', yield_earned)
            usdt_yield = yield_earned
            # Phase 12: 2x yield factor for reward (double down on parking earnings)
            usdt_yield_factor = yield_earned * 2.0

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

        # Phase 9: Rebound bonus - reward correctly timed dip-buys
        rebound_bonus = 0.0
        action_is_buy = (action % 3 == 0) and (action // 3 < 2)  # BTC or XRP buy

        # Low volatility + buy = greed regime offense (massive bonus)
        if self.current_volatility < self.vol_low_threshold and action_is_buy:
            rebound_bonus += 5.0  # Massive bonus for correctly timed offense

        # RSI oversold + buy = dip confirmation synergy
        if action_is_buy:
            asset = 'BTC' if action < 3 else 'XRP'
            asset_rsi = self.current_rsi.get(asset, 50)
            if asset_rsi < self.rsi_oversold:
                rebound_bonus += 3.0  # LSTM dip confirmation synergy
            elif asset_rsi < 40:  # Moderately oversold
                rebound_bonus += 1.0

        # Deploy action (8) during low vol = aggressive offense bonus
        if action == 8 and self.current_volatility < self.vol_low_threshold:
            rebound_bonus += 4.0  # Reward deploying leverage in calm dips

        # Phase 12: Bear market short rewards with precision tracking
        short_bonus = 0.0
        short_decay = 0.0
        short_precision_bonus = 0.0
        action_is_short = action in [9, 10]  # Short BTC or XRP

        # Calculate short P&L with whipsaw detection
        short_pnl = 0.0
        for asset in ['BTC', 'XRP']:
            if self.short_positions[asset] > 0:
                current_price = prices.get(asset, 1.0)
                entry_price = self.short_entries[asset]
                size = self.short_positions[asset]

                # Phase 12: Track peak price for whipsaw detection
                if current_price > self.short_peak_price[asset]:
                    self.short_peak_price[asset] = current_price

                # Short profits when price drops
                pnl_pct = (entry_price - current_price) / entry_price if entry_price > 0 else 0
                raw_short_pnl = pnl_pct * size

                # Phase 12: Check for whipsaw (price bounced > 5% above entry)
                bounce_pct = (self.short_peak_price[asset] - entry_price) / entry_price if entry_price > 0 else 0
                is_whipsaw_free = bounce_pct < self.short_whipsaw_threshold

                # Phase 12: 6.0x multiplier for whipsaw-free profitable shorts
                if raw_short_pnl > 0:
                    if is_whipsaw_free:
                        short_pnl += 6.0 * abs(raw_short_pnl)  # Phase 12: Precision short reward
                        short_precision_bonus += 2.0  # Extra bonus for clean shorts
                    else:
                        short_pnl += 4.0 * abs(raw_short_pnl)  # Still reward, but less
                else:
                    short_pnl += raw_short_pnl * 0.1  # Scaled loss

                # Phase 12: Lower decay penalty (encourage timely exits, not panic)
                duration = self.current_step - self.short_open_step[asset]
                if duration > self.max_short_duration:
                    short_decay += 0.3  # Phase 12: Reduced from 0.5

                # Phase 11: Auto-exit bonus when RSI drops below exit threshold (mean reversion)
                asset_rsi = self.current_rsi.get(asset, 50)
                if asset_rsi < self.rsi_short_exit and action == 11:
                    short_bonus += 2.0  # Reward mean reversion exit

        # Phase 12: Paired bear mode - rip shorts vs selective shorts
        if action_is_short:
            asset = 'BTC' if action == 9 else 'XRP'
            asset_rsi = self.current_rsi.get(asset, 50)

            # Phase 12: Overbought rip = aggressive short (RSI > 72)
            if self.current_volatility > self.vol_high_threshold and asset_rsi > self.rsi_rip_threshold:
                short_bonus += 6.0 * self.short_leverage  # Phase 12: Rip short bonus
                # Reset peak price tracking for new short
                self.short_peak_price[asset] = 0.0

            # Phase 12: Standard bear signal: ATR >4% + RSI >65
            elif self.current_volatility > self.vol_high_threshold and asset_rsi > self.rsi_overbought:
                short_bonus += 4.0 * self.short_leverage  # Phase 12: Selective short

            # Moderate bear signal: high vol only
            elif self.current_volatility > self.vol_high_threshold:
                short_bonus += 2.0  # Reduced for low-confidence shorts

        # Reward closing shorts with profit
        if action == 11:  # Close shorts
            if short_pnl > 0:
                short_bonus += 6.0  # Phase 12: Increased for profitable exit
                # Reset peak tracking
                self.short_peak_price = {'BTC': 0.0, 'XRP': 0.0}

        # Phase 12: Combined reward with yield factor and precision bonus
        # base + alignment + USDT safety + yield factor + margin + rebound + shorts + precision - penalties
        reward = (base_reward + 3.0 * alignment_score + usdt_bonus + usdt_yield_factor + margin_pnl +
                  rebound_bonus + short_bonus + short_pnl + short_precision_bonus - vol_penalty - short_decay)

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
