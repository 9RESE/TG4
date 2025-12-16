"""
Trading Environment for Reinforcement Learning

Gymnasium-compatible environment for training RL agents on trading tasks.
"""

from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    gym = None
    spaces = None


@dataclass
class TradingEnvConfig:
    """Configuration for trading environment."""
    # Position sizing
    initial_capital: float = 10000.0
    max_position_pct: float = 0.25  # Max 25% of capital per position
    fee_rate: float = 0.001  # 0.1% trading fee
    slippage_pct: float = 0.0005  # 0.05% slippage

    # Risk management
    stop_loss_pct: float = 2.0  # 2% stop loss
    take_profit_pct: float = 4.0  # 4% take profit
    max_drawdown_pct: float = 15.0  # 15% max drawdown (episode ends)

    # Episode settings
    episode_length: int = 500  # Steps per episode
    warmup_steps: int = 50  # Initial steps for indicator calculation

    # Action space
    use_discrete_actions: bool = True
    num_discrete_actions: int = 5  # 0=strong sell, 1=sell, 2=hold, 3=buy, 4=strong buy

    # Observation settings
    lookback_window: int = 20  # Number of historical bars in observation
    include_position_info: bool = True
    include_portfolio_info: bool = True

    # Reward settings
    reward_scaling: float = 1.0
    penalize_holding: float = 0.0  # Small penalty for holding to encourage trading

    # Features to include in observation
    feature_columns: List[str] = field(default_factory=lambda: [
        'returns', 'log_returns', 'volatility', 'rsi', 'macd',
        'bb_position', 'atr_pct', 'volume_ratio'
    ])


class TradingEnv(gym.Env):
    """
    Gymnasium environment for training RL trading agents.

    Observation Space:
        - Market features: OHLCV-derived indicators
        - Position info: Current position, entry price, unrealized PnL
        - Portfolio info: Capital, equity, drawdown

    Action Space:
        - Discrete: 5 actions (strong sell, sell, hold, buy, strong buy)
        - Continuous: Position target [-1, 1] where -1 = max short, +1 = max long

    Reward:
        - PnL-based with optional risk adjustments
        - Configurable reward shaping
    """

    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(
        self,
        data: pd.DataFrame,
        features: Optional[pd.DataFrame] = None,
        config: Optional[TradingEnvConfig] = None,
        reward_function: Optional[Any] = None,
        render_mode: Optional[str] = None
    ):
        """
        Initialize trading environment.

        Args:
            data: DataFrame with OHLCV columns (open, high, low, close, volume)
            features: Optional pre-computed features DataFrame
            config: Environment configuration
            reward_function: Custom reward function (callable)
            render_mode: Rendering mode
        """
        super().__init__()

        if gym is None:
            raise ImportError("gymnasium is required. Install with: pip install gymnasium")

        self.config = config or TradingEnvConfig()
        self.render_mode = render_mode
        self.reward_function = reward_function

        # Store data
        self.data = data.copy()
        self.features = features if features is not None else self._compute_features(data)

        # Validate data length
        min_length = self.config.warmup_steps + self.config.episode_length
        if len(self.data) < min_length:
            raise ValueError(f"Data length {len(self.data)} < minimum {min_length}")

        # Build action space
        if self.config.use_discrete_actions:
            self.action_space = spaces.Discrete(self.config.num_discrete_actions)
        else:
            # Continuous: target position from -1 (max short) to +1 (max long)
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(1,), dtype=np.float32
            )

        # Build observation space
        obs_dim = self._calculate_obs_dim()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Episode state
        self.current_step = 0
        self.start_step = 0
        self.capital = self.config.initial_capital
        self.position = 0.0  # Position size (positive = long, negative = short)
        self.entry_price = 0.0
        self.peak_equity = self.config.initial_capital
        self.equity_history = []
        self.trade_history = []
        self.returns_history = []

    def _calculate_obs_dim(self) -> int:
        """Calculate observation space dimension."""
        dim = len(self.config.feature_columns) * self.config.lookback_window

        if self.config.include_position_info:
            dim += 4  # position, entry_price, unrealized_pnl, holding_time

        if self.config.include_portfolio_info:
            dim += 4  # capital, equity, drawdown, win_rate

        return dim

    def _compute_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute basic features from OHLCV data."""
        df = data.copy()

        # Price returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # Volatility
        df['volatility'] = df['returns'].rolling(20).std()

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'] / 100  # Normalize to 0-1

        # MACD
        exp12 = df['close'].ewm(span=12, adjust=False).mean()
        exp26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = (exp12 - exp26) / df['close']  # Normalized

        # Bollinger Band position
        sma20 = df['close'].rolling(20).mean()
        std20 = df['close'].rolling(20).std()
        bb_upper = sma20 + 2 * std20
        bb_lower = sma20 - 2 * std20
        df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower + 1e-10)

        # ATR percentage
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr_pct'] = tr.rolling(14).mean() / df['close']

        # Volume ratio
        df['volume_ratio'] = df['volume'] / (df['volume'].rolling(20).mean() + 1e-10)

        # Fill NaN values
        df = df.fillna(0)

        return df

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)

        # Random start position (but leave room for episode)
        max_start = len(self.data) - self.config.episode_length - 1
        min_start = self.config.warmup_steps

        if seed is not None:
            np.random.seed(seed)

        self.start_step = np.random.randint(min_start, max(min_start + 1, max_start))
        self.current_step = self.start_step

        # Reset portfolio
        self.capital = self.config.initial_capital
        self.position = 0.0
        self.entry_price = 0.0
        self.peak_equity = self.config.initial_capital
        self.equity_history = [self.config.initial_capital]
        self.trade_history = []
        self.returns_history = []
        self.holding_time = 0

        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        """Build observation vector."""
        obs = []

        # Feature history
        start_idx = max(0, self.current_step - self.config.lookback_window)
        end_idx = self.current_step

        for col in self.config.feature_columns:
            if col in self.features.columns:
                values = self.features[col].iloc[start_idx:end_idx].values
                # Pad if needed
                if len(values) < self.config.lookback_window:
                    values = np.pad(values, (self.config.lookback_window - len(values), 0))
                obs.extend(values)

        # Position info
        if self.config.include_position_info:
            current_price = self.data['close'].iloc[self.current_step]

            # Normalized position
            max_position = self.capital * self.config.max_position_pct / current_price
            norm_position = self.position / (max_position + 1e-10)

            # Unrealized PnL
            if self.position != 0:
                unrealized_pnl = (current_price - self.entry_price) * self.position
                unrealized_pnl_pct = unrealized_pnl / self.capital
            else:
                unrealized_pnl_pct = 0.0

            # Holding time (normalized)
            norm_holding = min(self.holding_time / 100.0, 1.0)

            obs.extend([
                norm_position,
                self.entry_price / current_price if self.entry_price > 0 else 0,
                unrealized_pnl_pct,
                norm_holding
            ])

        # Portfolio info
        if self.config.include_portfolio_info:
            current_price = self.data['close'].iloc[self.current_step]
            equity = self._calculate_equity(current_price)

            # Drawdown
            self.peak_equity = max(self.peak_equity, equity)
            drawdown = (self.peak_equity - equity) / self.peak_equity

            # Win rate
            wins = sum(1 for t in self.trade_history if t['pnl'] > 0)
            win_rate = wins / len(self.trade_history) if self.trade_history else 0.5

            obs.extend([
                self.capital / self.config.initial_capital,
                equity / self.config.initial_capital,
                drawdown,
                win_rate
            ])

        return np.array(obs, dtype=np.float32)

    def _calculate_equity(self, current_price: float) -> float:
        """Calculate current equity (capital + unrealized PnL)."""
        unrealized = (current_price - self.entry_price) * self.position if self.position != 0 else 0
        return self.capital + unrealized

    def step(self, action: Union[int, np.ndarray]) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step."""
        # Get current and next price
        current_price = self.data['close'].iloc[self.current_step]

        # Convert action to target position
        if self.config.use_discrete_actions:
            # Map discrete action to position target
            action_map = {
                0: -1.0,   # Strong sell (max short)
                1: -0.5,   # Sell (half short)
                2: 0.0,    # Hold (flat)
                3: 0.5,    # Buy (half long)
                4: 1.0     # Strong buy (max long)
            }
            target_position_pct = action_map.get(action, 0.0)
        else:
            target_position_pct = float(action[0])

        # Calculate target position size
        max_position_value = self.capital * self.config.max_position_pct
        target_position_value = max_position_value * target_position_pct
        target_position = target_position_value / current_price

        # Execute trade if position change needed
        position_change = target_position - self.position
        trade_pnl = 0.0
        trade_executed = False

        if abs(position_change) > 1e-8:
            # Apply slippage
            if position_change > 0:
                execution_price = current_price * (1 + self.config.slippage_pct)
            else:
                execution_price = current_price * (1 - self.config.slippage_pct)

            # Close existing position if reducing
            if self.position != 0 and np.sign(position_change) != np.sign(self.position):
                # Calculate realized PnL from closing
                close_amount = min(abs(self.position), abs(position_change))
                if self.position > 0:
                    close_pnl = (execution_price - self.entry_price) * close_amount
                else:
                    close_pnl = (self.entry_price - execution_price) * abs(close_amount)

                # Apply fees
                close_fees = close_amount * execution_price * self.config.fee_rate
                trade_pnl = close_pnl - close_fees

                # Record trade
                self.trade_history.append({
                    'step': self.current_step,
                    'side': 'close_long' if self.position > 0 else 'close_short',
                    'price': execution_price,
                    'size': close_amount,
                    'pnl': trade_pnl
                })

                self.capital += trade_pnl
                trade_executed = True

            # Apply fees for new position
            trade_value = abs(position_change) * execution_price
            trade_fees = trade_value * self.config.fee_rate
            self.capital -= trade_fees

            # Update position
            self.position = target_position

            if self.position != 0 and (not trade_executed or abs(target_position) > 1e-8):
                self.entry_price = execution_price
                self.holding_time = 0

        # Update holding time
        if self.position != 0:
            self.holding_time += 1

        # Advance step
        self.current_step += 1

        # Get new price for reward calculation
        if self.current_step < len(self.data):
            new_price = self.data['close'].iloc[self.current_step]
        else:
            new_price = current_price

        # Calculate reward
        equity = self._calculate_equity(new_price)
        self.equity_history.append(equity)

        # Calculate step return
        prev_equity = self.equity_history[-2] if len(self.equity_history) > 1 else self.config.initial_capital
        step_return = (equity - prev_equity) / prev_equity
        self.returns_history.append(step_return)

        # Use custom reward function if provided
        if self.reward_function is not None:
            reward = self.reward_function(
                step_return=step_return,
                equity=equity,
                position=self.position,
                trade_executed=trade_executed,
                env=self
            )
        else:
            # Default reward: scaled PnL
            reward = step_return * 100 * self.config.reward_scaling

            # Optional holding penalty
            if self.position == 0 and self.config.penalize_holding > 0:
                reward -= self.config.penalize_holding

        # Check termination conditions
        episode_steps = self.current_step - self.start_step
        terminated = False
        truncated = False

        # Max drawdown check
        self.peak_equity = max(self.peak_equity, equity)
        current_drawdown = (self.peak_equity - equity) / self.peak_equity * 100

        if current_drawdown >= self.config.max_drawdown_pct:
            terminated = True
            reward -= 10.0  # Penalty for max drawdown

        # Episode length check
        if episode_steps >= self.config.episode_length:
            truncated = True

        # Data end check
        if self.current_step >= len(self.data) - 1:
            truncated = True

        # Stop loss / take profit check
        if self.position != 0:
            unrealized_return = (new_price - self.entry_price) / self.entry_price * 100
            if self.position < 0:
                unrealized_return = -unrealized_return

            if unrealized_return <= -self.config.stop_loss_pct:
                # Force close at stop loss
                terminated = True
                reward -= 5.0
            elif unrealized_return >= self.config.take_profit_pct:
                # Force close at take profit
                reward += 2.0

        # Build info dict
        info = {
            'equity': equity,
            'capital': self.capital,
            'position': self.position,
            'entry_price': self.entry_price,
            'current_price': new_price,
            'drawdown': current_drawdown,
            'step_return': step_return,
            'total_return': (equity / self.config.initial_capital - 1) * 100,
            'num_trades': len(self.trade_history),
            'episode_steps': episode_steps
        }

        return self._get_observation(), reward, terminated, truncated, info

    def render(self):
        """Render environment state."""
        if self.render_mode == 'human':
            current_price = self.data['close'].iloc[self.current_step]
            equity = self._calculate_equity(current_price)
            print(
                f"Step {self.current_step}: "
                f"Price=${current_price:.4f}, "
                f"Equity=${equity:.2f}, "
                f"Position={self.position:.4f}, "
                f"Trades={len(self.trade_history)}"
            )

    def close(self):
        """Clean up environment."""
        pass

    def get_episode_stats(self) -> Dict[str, float]:
        """Get statistics for the current episode."""
        if not self.returns_history:
            return {}

        returns = np.array(self.returns_history)
        equity_arr = np.array(self.equity_history)

        # Calculate metrics
        total_return = (equity_arr[-1] / self.config.initial_capital - 1) * 100

        # Sharpe ratio (simplified)
        if returns.std() > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(252 * 24)  # Annualized
        else:
            sharpe = 0.0

        # Max drawdown
        running_max = np.maximum.accumulate(equity_arr)
        drawdowns = (running_max - equity_arr) / running_max
        max_drawdown = drawdowns.max() * 100

        # Win rate
        wins = sum(1 for t in self.trade_history if t['pnl'] > 0)
        win_rate = wins / len(self.trade_history) * 100 if self.trade_history else 0

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': len(self.trade_history),
            'final_equity': equity_arr[-1]
        }


class MultiAssetTradingEnv(TradingEnv):
    """
    Trading environment supporting multiple assets.

    Extends TradingEnv to handle portfolio of assets with
    allocation actions.
    """

    def __init__(
        self,
        data_dict: Dict[str, pd.DataFrame],
        features_dict: Optional[Dict[str, pd.DataFrame]] = None,
        config: Optional[TradingEnvConfig] = None,
        **kwargs
    ):
        """
        Initialize multi-asset environment.

        Args:
            data_dict: Dictionary mapping symbol to OHLCV DataFrame
            features_dict: Optional pre-computed features per symbol
            config: Environment configuration
        """
        self.symbols = list(data_dict.keys())
        self.data_dict = data_dict

        # Use first symbol's data as primary for base class
        primary_data = data_dict[self.symbols[0]]

        if features_dict is not None:
            primary_features = features_dict[self.symbols[0]]
        else:
            primary_features = None

        super().__init__(
            data=primary_data,
            features=primary_features,
            config=config,
            **kwargs
        )

        # Override action space for multi-asset
        n_assets = len(self.symbols)
        if self.config.use_discrete_actions:
            # Discrete: choose which asset to trade
            self.action_space = spaces.Discrete(n_assets * 5)
        else:
            # Continuous: allocation to each asset
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(n_assets,), dtype=np.float32
            )

        # Multi-asset positions
        self.positions = {sym: 0.0 for sym in self.symbols}
        self.entry_prices = {sym: 0.0 for sym in self.symbols}

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset multi-asset environment."""
        obs, info = super().reset(seed=seed, options=options)

        self.positions = {sym: 0.0 for sym in self.symbols}
        self.entry_prices = {sym: 0.0 for sym in self.symbols}

        return self._get_observation(), info

    def _get_observation(self) -> np.ndarray:
        """Build observation for all assets."""
        obs = []

        for symbol in self.symbols:
            df = self.data_dict[symbol]

            # Price features
            if self.current_step < len(df):
                price = df['close'].iloc[self.current_step]
                returns = df['close'].pct_change().iloc[
                    max(0, self.current_step - self.config.lookback_window):self.current_step
                ].values

                if len(returns) < self.config.lookback_window:
                    returns = np.pad(returns, (self.config.lookback_window - len(returns), 0))

                obs.extend(returns)

                # Position for this asset
                obs.append(self.positions[symbol])

        # Portfolio info
        total_equity = self._calculate_total_equity()
        obs.append(total_equity / self.config.initial_capital)

        return np.array(obs, dtype=np.float32)

    def _calculate_total_equity(self) -> float:
        """Calculate total portfolio equity."""
        equity = self.capital

        for symbol in self.symbols:
            df = self.data_dict[symbol]
            if self.current_step < len(df):
                price = df['close'].iloc[self.current_step]
                position = self.positions[symbol]
                entry = self.entry_prices[symbol]

                if position != 0:
                    unrealized = (price - entry) * position
                    equity += unrealized

        return equity
