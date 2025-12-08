"""
Ensemble RL Training Environment
Phase 21: Multi-Strategy Weight Optimization (6 Strategies)

This environment trains an RL agent to dynamically weight six strategies:
1. MeanReversionVWAP - Dominates in chop (XRP $2.00-2.20 range)
2. XRPBTCPairTrading - Activates on XRP/BTC divergence
3. DefensiveYield - Max weight during high ATR + fear
4. MATrendFollow - 9-SMA trend following with breakout detection
5. XRPBTCLeadLag - BTC breakout → XRP follow with high leverage
6. IntraDayScalper - BB squeeze + RSI extremes, activates on ATR >3%

Observation Space (36 features):
- Regime features (7): ATR XRP/BTC, correlation, RSI, VWAP deviation, z-score
- Strategy signals (18): 6 strategies x (action, confidence, leverage)
- Current weights (6): Mean reversion, pair trading, defensive, ma_trend, leadlag, scalper
- Portfolio state (5): BTC/XRP/USDT weights + volatility + scalp_active flag

Action Space:
- Discrete(8): Weight presets (chop, divergence, fear, balanced, aggressive, trend, breakout, scalp)
- Or Continuous(6): Direct weight values (softmax normalized)

Reward:
- PNL from executed trades
- Accumulation bias for BTC/XRP/USDT
- Yield bonus during defensive mode
- Regime alignment bonus (right strategy for right market)
- Scalp bonus for capturing intra-day swings
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from portfolio import Portfolio


class EnsembleEnv(gym.Env):
    """
    RL Environment for training ensemble strategy weights.

    The agent learns to allocate weights to:
    - Mean Reversion (best in chop/range-bound)
    - Pair Trading (best in divergence)
    - Defensive (best in fear/high volatility)

    Reward is based on:
    - Portfolio PNL (primary)
    - Accumulation bias (BTC 45%, XRP 35%, USDT 20%)
    - Yield accrual during defensive
    - Regime alignment (matching strategy to market)
    """

    def __init__(self, data_dict: dict, initial_balance: dict = None,
                 use_discrete_actions: bool = True):
        """
        Initialize ensemble training environment.

        Args:
            data_dict: Multi-symbol OHLCV data
            initial_balance: Starting portfolio
            use_discrete_actions: If True, use 5 weight presets. If False, use continuous weights.
        """
        super().__init__()

        self.data = data_dict
        self.symbols = list(data_dict.keys())
        self.initial_balance = initial_balance or {'USDT': 1000.0, 'XRP': 500.0, 'BTC': 0.01}
        self.use_discrete_actions = use_discrete_actions

        # Accumulation targets
        self.targets = {'BTC': 0.45, 'XRP': 0.35, 'USDT': 0.20}

        # Action space
        if use_discrete_actions:
            # 8 weight presets: chop, divergence, fear, balanced, aggressive, trend, breakout, scalp
            self.action_space = spaces.Discrete(8)
        else:
            # Continuous weights for 6 strategies (softmax applied in step)
            self.action_space = spaces.Box(low=-2, high=2, shape=(6,), dtype=np.float32)

        # Observation space: 36 features (Phase 21: 6 strategies)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(36,), dtype=np.float32
        )

        # Environment state
        self.current_step = 0
        self.max_steps = min(len(df) for df in data_dict.values()) - 1
        self.portfolio = None

        # Strategy weights (6 strategies - Phase 21)
        self.weights = {
            'mean_reversion': 0.20,
            'pair_trading': 0.12,
            'defensive': 0.22,
            'ma_trend': 0.18,
            'leadlag': 0.13,
            'scalper': 0.15
        }

        # Regime tracking
        self.current_regime = 'neutral'
        self.current_volatility = 0.03
        self.current_correlation = 0.8
        self.current_rsi = {'XRP': 50.0, 'BTC': 50.0}

        # Weight presets for discrete actions (8 presets for 6 strategies - Phase 21)
        self.weight_presets = {
            0: {'mean_reversion': 0.45, 'pair_trading': 0.08, 'defensive': 0.15, 'ma_trend': 0.12, 'leadlag': 0.10, 'scalper': 0.10},  # Chop
            1: {'mean_reversion': 0.12, 'pair_trading': 0.40, 'defensive': 0.12, 'ma_trend': 0.08, 'leadlag': 0.13, 'scalper': 0.15},  # Divergence
            2: {'mean_reversion': 0.10, 'pair_trading': 0.05, 'defensive': 0.55, 'ma_trend': 0.10, 'leadlag': 0.10, 'scalper': 0.10},  # Fear
            3: {'mean_reversion': 0.17, 'pair_trading': 0.17, 'defensive': 0.17, 'ma_trend': 0.17, 'leadlag': 0.16, 'scalper': 0.16},  # Balanced
            4: {'mean_reversion': 0.20, 'pair_trading': 0.20, 'defensive': 0.08, 'ma_trend': 0.22, 'leadlag': 0.15, 'scalper': 0.15},  # Aggressive
            5: {'mean_reversion': 0.12, 'pair_trading': 0.08, 'defensive': 0.12, 'ma_trend': 0.40, 'leadlag': 0.13, 'scalper': 0.15},  # Trend
            6: {'mean_reversion': 0.08, 'pair_trading': 0.08, 'defensive': 0.10, 'ma_trend': 0.25, 'leadlag': 0.35, 'scalper': 0.14},  # Breakout
            7: {'mean_reversion': 0.10, 'pair_trading': 0.08, 'defensive': 0.12, 'ma_trend': 0.15, 'leadlag': 0.15, 'scalper': 0.40},  # Scalp (high vol)
        }

        # Simulated strategy signals (will be computed from data)
        self.strategy_signals = {}

        # USDT yield (6.5% APY)
        self.usdt_yield_per_step = 0.065 / 365 / 24 * 4  # ~4 hours per step

        # Position tracking
        self.positions = {'long': {}, 'short': {}}
        self.entry_prices = {}

    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)

        self.current_step = 60  # Warmup period
        self.portfolio = Portfolio(self.initial_balance.copy())

        # Reset weights to balanced (6 strategies)
        self.weights = {
            'mean_reversion': 0.17,
            'pair_trading': 0.17,
            'defensive': 0.17,
            'ma_trend': 0.17,
            'leadlag': 0.16,
            'scalper': 0.16
        }

        # Reset tracking
        self.positions = {'long': {}, 'short': {}}
        self.entry_prices = {}
        self.current_regime = 'neutral'

        return self._get_obs(), {}

    def _calculate_rsi(self, close_prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI indicator."""
        if len(close_prices) < period + 1:
            return 50.0

        deltas = np.diff(close_prices[-period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0.001

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _calculate_atr_pct(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR as percentage of price."""
        if len(df) < period + 1 or self.current_step < period:
            return 0.03

        idx = min(self.current_step, len(df) - 1)
        high = df['high'].values[:idx+1]
        low = df['low'].values[:idx+1]
        close = df['close'].values[:idx+1]

        tr_list = []
        for i in range(max(1, len(close) - period), len(close)):
            tr = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
            tr_list.append(tr)

        atr = np.mean(tr_list) if tr_list else 0
        current_price = close[-1] if len(close) > 0 else 1
        return atr / current_price if current_price > 0 else 0.03

    def _calculate_vwap_deviation(self, df: pd.DataFrame) -> float:
        """Calculate current price deviation from VWAP."""
        if len(df) < 14 or self.current_step < 14:
            return 0.0

        idx = min(self.current_step, len(df) - 1)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).rolling(14).sum() / df['volume'].rolling(14).sum()

        current_price = df['close'].iloc[idx]
        current_vwap = vwap.iloc[idx]

        if pd.isna(current_vwap) or current_vwap == 0:
            return 0.0

        return (current_price - current_vwap) / current_vwap

    def _calculate_correlation(self, window: int = 24) -> float:
        """Calculate XRP/BTC rolling correlation."""
        if 'XRP/USDT' not in self.data or 'BTC/USDT' not in self.data:
            return 0.8

        xrp_df = self.data['XRP/USDT']
        btc_df = self.data['BTC/USDT']

        min_len = min(len(xrp_df), len(btc_df), self.current_step + 1)
        if min_len < window:
            return 0.8

        xrp_returns = xrp_df['close'].pct_change().iloc[max(0, min_len-window):min_len]
        btc_returns = btc_df['close'].pct_change().iloc[max(0, min_len-window):min_len]

        corr = xrp_returns.corr(btc_returns)
        return corr if not np.isnan(corr) else 0.8

    def _calculate_zscore(self) -> float:
        """Calculate XRP/BTC spread z-score for pair trading signal."""
        if 'XRP/USDT' not in self.data or 'BTC/USDT' not in self.data:
            return 0.0

        xrp_df = self.data['XRP/USDT']
        btc_df = self.data['BTC/USDT']

        lookback = min(168, self.current_step)
        if lookback < 20:
            return 0.0

        xrp_prices = xrp_df['close'].iloc[self.current_step-lookback:self.current_step+1].values
        btc_prices = btc_df['close'].iloc[self.current_step-lookback:self.current_step+1].values

        # Simple hedge ratio via correlation
        if len(xrp_prices) > 0 and len(btc_prices) > 0:
            btc_mean = np.mean(btc_prices)
            xrp_mean = np.mean(xrp_prices)
            covariance = np.sum((btc_prices - btc_mean) * (xrp_prices - xrp_mean))
            variance = np.sum((btc_prices - btc_mean) ** 2)
            hedge_ratio = covariance / variance if variance > 0 else 0

            spread = xrp_prices - hedge_ratio * btc_prices
            zscore = (spread[-1] - np.mean(spread)) / (np.std(spread) + 0.0001)
            return float(np.clip(zscore, -5, 5))

        return 0.0

    def _simulate_strategy_signals(self) -> dict:
        """Simulate signals from each strategy based on current market state."""
        signals = {
            'mean_reversion': {'action': 'hold', 'confidence': 0.0, 'leverage': 1},
            'pair_trading': {'action': 'hold', 'confidence': 0.0, 'leverage': 1},
            'defensive': {'action': 'hold', 'confidence': 0.0, 'leverage': 1},
            'ma_trend': {'action': 'hold', 'confidence': 0.0, 'leverage': 1},
            'leadlag': {'action': 'hold', 'confidence': 0.0, 'leverage': 1},
            'scalper': {'action': 'hold', 'confidence': 0.0, 'leverage': 1}
        }

        if 'XRP/USDT' not in self.data:
            return signals

        xrp_df = self.data['XRP/USDT']
        if self.current_step >= len(xrp_df):
            return signals

        # Get current indicators
        xrp_rsi = self.current_rsi.get('XRP', 50)
        vwap_dev = self._calculate_vwap_deviation(xrp_df)
        zscore = self._calculate_zscore()

        # Mean Reversion signals (VWAP + RSI)
        if xrp_rsi < 32 and vwap_dev < -0.003:
            signals['mean_reversion'] = {
                'action': 'buy',
                'confidence': min(0.5 + (32 - xrp_rsi) / 50 + abs(vwap_dev) * 10, 0.95),
                'leverage': 7
            }
        elif xrp_rsi > 68 and vwap_dev > 0.003:
            signals['mean_reversion'] = {
                'action': 'sell',
                'confidence': min(0.5 + (xrp_rsi - 68) / 50 + abs(vwap_dev) * 10, 0.95),
                'leverage': 7
            }

        # Pair Trading signals (Z-score)
        if zscore > 1.8:
            signals['pair_trading'] = {
                'action': 'short_xrp_long_btc',
                'confidence': min(0.5 + (zscore - 1.8) * 0.15, 0.95),
                'leverage': 10
            }
        elif zscore < -1.8:
            signals['pair_trading'] = {
                'action': 'long_xrp_short_btc',
                'confidence': min(0.5 + (abs(zscore) - 1.8) * 0.15, 0.95),
                'leverage': 10
            }

        # Defensive signals (High volatility = park)
        if self.current_volatility > 0.04:
            signals['defensive'] = {
                'action': 'park',
                'confidence': min(0.6 + self.current_volatility * 5, 0.95),
                'leverage': 1
            }

        # Phase 21: MA Trend signals (SMA9 crossover)
        if 'XRP/USDT' in self.data:
            xrp_df = self.data['XRP/USDT']
            if self.current_step >= 10:
                close_prices = xrp_df['close'].iloc[:self.current_step+1].values
                sma9 = np.mean(close_prices[-9:])
                current_price = close_prices[-1]
                prev_price = close_prices[-2]

                # Uptrend: close above SMA9
                if current_price > sma9 and prev_price <= sma9:
                    signals['ma_trend'] = {
                        'action': 'buy',
                        'confidence': min(0.7 + (current_price / sma9 - 1) * 10, 0.95),
                        'leverage': 5
                    }
                # Downtrend: close below SMA9
                elif current_price < sma9 and prev_price >= sma9:
                    signals['ma_trend'] = {
                        'action': 'sell',
                        'confidence': min(0.65 + (sma9 / current_price - 1) * 10, 0.90),
                        'leverage': 4
                    }

        # Phase 21: Lead-Lag signals (BTC leads XRP)
        if 'BTC/USDT' in self.data and 'XRP/USDT' in self.data:
            btc_df = self.data['BTC/USDT']
            if self.current_step >= 24:
                btc_prices = btc_df['close'].iloc[:self.current_step+1].values
                btc_high_24h = np.max(btc_prices[-24:-1])
                current_btc = btc_prices[-1]
                btc_vol = btc_df['volume'].iloc[-24:-1].mean()
                current_vol = btc_df['volume'].iloc[self.current_step]

                # BTC breakout detection
                if current_btc > btc_high_24h and current_vol > btc_vol * 1.5:
                    # BTC breakout → XRP follow with high leverage
                    signals['leadlag'] = {
                        'action': 'buy',
                        'confidence': min(0.85 + (current_btc / btc_high_24h - 1) * 5, 0.95),
                        'leverage': 7
                    }
                elif self.current_correlation > 0.8:
                    # High correlation: XRP follows BTC direction
                    btc_change = (current_btc - btc_prices[-4]) / btc_prices[-4]
                    if btc_change > 0.01:
                        signals['leadlag'] = {
                            'action': 'buy',
                            'confidence': 0.70 + self.current_correlation * 0.1,
                            'leverage': 5
                        }
                    elif btc_change < -0.01:
                        signals['leadlag'] = {
                            'action': 'sell',
                            'confidence': 0.65 + self.current_correlation * 0.1,
                            'leverage': 5
                        }

        # Phase 21: Scalper signals (BB squeeze + high volatility)
        if 'XRP/USDT' in self.data:
            xrp_df = self.data['XRP/USDT']
            if self.current_step >= 25 and self.current_volatility > 0.03:  # ATR >3% activates
                close_prices = xrp_df['close'].iloc[:self.current_step+1].values
                # Simplified Bollinger Bands
                if len(close_prices) >= 20:
                    sma20 = np.mean(close_prices[-20:])
                    std20 = np.std(close_prices[-20:])
                    bb_upper = sma20 + 2 * std20
                    bb_lower = sma20 - 2 * std20
                    current_price = close_prices[-1]
                    current_rsi = self.current_rsi.get('XRP', 50)

                    # Oversold squeeze: price at lower band + RSI oversold
                    if current_price <= bb_lower and current_rsi < 30:
                        signals['scalper'] = {
                            'action': 'buy',
                            'confidence': min(0.70 + (30 - current_rsi) / 50, 0.90),
                            'leverage': 3
                        }
                    # Overbought squeeze: price at upper band + RSI overbought
                    elif current_price >= bb_upper and current_rsi > 70:
                        signals['scalper'] = {
                            'action': 'sell',
                            'confidence': min(0.65 + (current_rsi - 70) / 50, 0.85),
                            'leverage': 3
                        }

        self.strategy_signals = signals
        return signals

    def _determine_regime(self) -> str:
        """Determine current market regime."""
        # High fear: High ATR + extreme RSI
        if self.current_volatility > 0.04:
            xrp_rsi = self.current_rsi.get('XRP', 50)
            if xrp_rsi > 70 or xrp_rsi < 30:
                return 'fear'

        # Divergence: Low correlation
        if self.current_correlation < 0.5:
            return 'divergence'

        # Chop: Low ATR
        if self.current_volatility < 0.025:
            return 'chop'

        return 'neutral'

    def _get_obs(self) -> np.ndarray:
        """Build observation vector."""
        obs = []

        # Calculate regime features
        if 'XRP/USDT' in self.data:
            xrp_df = self.data['XRP/USDT']
            atr_xrp = self._calculate_atr_pct(xrp_df)
            xrp_rsi = self._calculate_rsi(xrp_df['close'].iloc[:self.current_step+1].values)
            vwap_dev = self._calculate_vwap_deviation(xrp_df)
            self.current_rsi['XRP'] = xrp_rsi
        else:
            atr_xrp = 0.03
            xrp_rsi = 50.0
            vwap_dev = 0.0

        if 'BTC/USDT' in self.data:
            btc_df = self.data['BTC/USDT']
            atr_btc = self._calculate_atr_pct(btc_df)
            btc_rsi = self._calculate_rsi(btc_df['close'].iloc[:self.current_step+1].values)
            self.current_rsi['BTC'] = btc_rsi
        else:
            atr_btc = 0.02
            btc_rsi = 50.0

        # Update volatility and correlation
        self.current_volatility = max(atr_xrp, atr_btc)
        self.current_correlation = self._calculate_correlation()
        zscore = self._calculate_zscore()

        # Update regime
        self.current_regime = self._determine_regime()

        # Regime features (7)
        obs.extend([
            atr_xrp,
            atr_btc,
            self.current_correlation,
            xrp_rsi / 100,
            btc_rsi / 100,
            vwap_dev,
            zscore / 5  # Normalized
        ])

        # Simulate strategy signals
        signals = self._simulate_strategy_signals()

        # Strategy signals (18: 6 strategies x 3 features) - Phase 21
        action_map = {'hold': 0, 'buy': 1, 'sell': -1, 'park': 0,
                      'short_xrp_long_btc': -0.5, 'long_xrp_short_btc': 0.5}

        for name in ['mean_reversion', 'pair_trading', 'defensive', 'ma_trend', 'leadlag', 'scalper']:
            signal = signals.get(name, {})
            action_val = action_map.get(signal.get('action', 'hold'), 0)
            confidence = signal.get('confidence', 0)
            leverage = signal.get('leverage', 1) / 10
            obs.extend([action_val, confidence, leverage])

        # Current weights (6) - Phase 21
        for name in ['mean_reversion', 'pair_trading', 'defensive', 'ma_trend', 'leadlag', 'scalper']:
            obs.append(self.weights.get(name, 0.17))

        # Portfolio state (5) - Phase 21: added volatility + scalp_active
        prices = self._current_prices()
        total = max(self.portfolio.get_total_usd(prices), 1.0)
        for asset in ['BTC', 'XRP', 'USDT']:
            weight = self.portfolio.balances.get(asset, 0) * prices.get(asset, 1.0) / total
            obs.append(weight)

        # Volatility feature
        obs.append(self.current_volatility)

        # Scalp active flag (1.0 if high vol + scalper has signal)
        scalper_active = 1.0 if (self.current_volatility > 0.03 and
                                  signals.get('scalper', {}).get('action', 'hold') != 'hold') else 0.0
        obs.append(scalper_active)

        return np.array(obs, dtype=np.float32)

    def _current_prices(self) -> dict:
        """Get current prices from data."""
        prices = {'USDT': 1.0}
        for sym in self.symbols:
            df = self.data[sym]
            if self.current_step < len(df):
                base = sym.split('/')[0]
                prices[base] = df.iloc[self.current_step]['close']
        return prices

    def _execute_weighted_signal(self, prices: dict) -> float:
        """Execute trade based on weighted strategy signals."""
        pnl = 0.0

        # Get best signal based on weights
        best_action = 'hold'
        best_confidence = 0.0
        best_strategy = 'defensive'

        for name, signal in self.strategy_signals.items():
            weight = self.weights.get(name, 0.33)
            weighted_conf = signal.get('confidence', 0) * weight

            if weighted_conf > best_confidence and signal.get('action', 'hold') != 'hold':
                best_confidence = weighted_conf
                best_action = signal.get('action', 'hold')
                best_strategy = name

        usdt_balance = self.portfolio.balances.get('USDT', 0)

        # Execute only if confidence > 0.5
        if best_confidence > 0.5:
            if best_action == 'buy' and usdt_balance > 50:
                # Buy XRP
                xrp_price = prices.get('XRP', 2.0)
                buy_value = usdt_balance * 0.15
                buy_amount = buy_value / xrp_price
                self.portfolio.update('USDT', -buy_value)
                self.portfolio.update('XRP', buy_amount)
                self.entry_prices['XRP_long'] = xrp_price

            elif best_action == 'sell' and self.portfolio.balances.get('XRP', 0) > 0:
                # Sell XRP
                xrp_balance = self.portfolio.balances.get('XRP', 0)
                xrp_price = prices.get('XRP', 2.0)
                sell_amount = xrp_balance * 0.2
                entry = self.entry_prices.get('XRP_long', xrp_price)
                pnl = (xrp_price - entry) / entry * sell_amount * xrp_price
                self.portfolio.update('XRP', -sell_amount)
                self.portfolio.update('USDT', sell_amount * xrp_price)

            elif best_action == 'park':
                # Sell assets to USDT
                for asset in ['XRP', 'BTC']:
                    holding = self.portfolio.balances.get(asset, 0)
                    if holding > 0:
                        price = prices.get(asset, 1.0)
                        sell_amount = holding * 0.1
                        self.portfolio.update(asset, -sell_amount)
                        self.portfolio.update('USDT', sell_amount * price)

        return pnl

    def step(self, action):
        """Execute one step of the environment."""
        prices = self._current_prices()
        prev_value = self.portfolio.get_total_usd(prices)

        # Apply action to update weights
        if self.use_discrete_actions:
            self.weights = self.weight_presets.get(action, self.weight_presets[3]).copy()
        else:
            # Continuous action: softmax to get weights (6 strategies - Phase 21)
            exp_action = np.exp(action - np.max(action))
            weights_arr = exp_action / exp_action.sum()
            self.weights = {
                'mean_reversion': float(weights_arr[0]),
                'pair_trading': float(weights_arr[1]),
                'defensive': float(weights_arr[2]),
                'ma_trend': float(weights_arr[3]),
                'leadlag': float(weights_arr[4]),
                'scalper': float(weights_arr[5])
            }

        # Execute trade based on weighted signals
        trade_pnl = self._execute_weighted_signal(prices)

        # Apply USDT yield if in defensive mode
        yield_earned = 0.0
        if self.weights['defensive'] > 0.5:
            usdt_balance = self.portfolio.balances.get('USDT', 0)
            if usdt_balance > 100:
                yield_earned = self.usdt_yield_per_step * usdt_balance
                self.portfolio.update('USDT', yield_earned)

        # Advance step
        self.current_step += 1

        # Calculate reward
        new_value = self.portfolio.get_total_usd(prices)
        base_reward = (new_value - prev_value) / prev_value if prev_value > 0 else 0

        # Accumulation alignment bonus
        total = max(new_value, 1.0)
        alignment_score = 0.0
        for asset in self.targets:
            asset_value = self.portfolio.balances.get(asset, 0) * prices.get(asset, 1.0)
            weight = asset_value / total
            alignment_score += min(weight, self.targets[asset])

        # Regime alignment bonus (reward correct strategy for market) - Phase 21
        regime_bonus = 0.0
        if self.current_regime == 'chop' and self.weights['mean_reversion'] > 0.35:
            regime_bonus = 1.0
        elif self.current_regime == 'divergence' and self.weights['pair_trading'] > 0.3:
            regime_bonus = 1.0
        elif self.current_regime == 'fear' and self.weights['defensive'] > 0.4:
            regime_bonus = 1.0
        # Phase 21: Trend bonus for MA trend and Lead-Lag
        elif self.current_regime == 'neutral':
            if self.weights['ma_trend'] > 0.3 or self.weights['leadlag'] > 0.3:
                regime_bonus = 0.5  # Partial bonus for trend strategies in neutral

        # Phase 21: Scalp bonus for high volatility captures
        scalp_bonus = 0.0
        if self.current_volatility > 0.03 and self.weights.get('scalper', 0) > 0.3:
            scalp_signal = self.strategy_signals.get('scalper', {})
            if scalp_signal.get('action', 'hold') != 'hold':
                scalp_bonus = 1.5  # Reward scalper activation during high vol

        # Yield bonus
        yield_bonus = yield_earned * 2.0  # 2x weight for yield

        # Combined reward
        reward = (base_reward * 10 + alignment_score * 3 + regime_bonus + yield_bonus + trade_pnl * 5 + scalp_bonus)

        done = self.current_step >= self.max_steps
        truncated = False

        return self._get_obs(), reward, done, truncated, {}

    def render(self):
        """Render environment state."""
        prices = self._current_prices()
        total_value = self.portfolio.get_total_usd(prices)
        print(f"Step {self.current_step}: Value=${total_value:.2f}, "
              f"Regime={self.current_regime}, Weights={self.weights}")


def train_ensemble_agent(data_dict: dict, timesteps: int = 2000000, device: str = "cuda"):
    """
    Train PPO agent on ensemble environment.

    Args:
        data_dict: Market data
        timesteps: Training timesteps
        device: Training device (cuda/cpu)

    Returns:
        Trained PPO model
    """
    from stable_baselines3 import PPO

    env = EnsembleEnv(data_dict, use_discrete_actions=True)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./tensorboard/ensemble/",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        device=device
    )

    print(f"Training Ensemble PPO agent for {timesteps} timesteps on {device.upper()}...")
    model.learn(total_timesteps=timesteps)
    model.save("models/rl_ensemble_agent")
    print("Ensemble model saved to models/rl_ensemble_agent")

    return model


def load_ensemble_agent(path: str = "models/rl_ensemble_agent", device: str = "cuda"):
    """Load trained ensemble RL agent."""
    from stable_baselines3 import PPO
    return PPO.load(path, device=device)
