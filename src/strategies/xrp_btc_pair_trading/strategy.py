"""
XRP/BTC Pair Trading Strategy
Phase 15: Enhanced with statsmodels for proper cointegration

Statistical arbitrage exploiting cointegration between XRP and BTC.
Uses OLS regression for dynamic hedge ratio and z-score for entry/exit.
Optimized for Kraken XRP/BTC pair with 10x margin.

Entry: Z-score > 2 (short XRP, long BTC) or Z-score < -2 (long XRP, short BTC)
Exit: Z-score crosses 0.5 (mean reversion) or |Z| > 3 (stop loss)

Target: Low-drawdown alpha with Sharpe > 1.5
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple

try:
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import coint, adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels not installed. Using fallback OLS.")

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from strategies.base_strategy import BaseStrategy


class XRPBTCPairTrading(BaseStrategy):
    """
    Pair Trading Strategy using XRP and BTC cointegration with statsmodels.

    The strategy:
    1. Calculates hedge ratio via OLS regression (statsmodels)
    2. Computes spread: XRP - hedge_ratio * BTC
    3. Calculates z-score of spread
    4. Trades mean reversion of spread with 10x leverage

    Entry:
    - Z-score > entry_z: Short XRP, Long BTC (spread too high)
    - Z-score < -entry_z: Long XRP, Short BTC (spread too low)

    Exit:
    - |Z-score| < exit_z (mean reverted)
    - |Z-score| > 3 (stop - divergence)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.name = 'xrp_btc_pair_trading'
        self.xrp_symbol = config.get('xrp_symbol', 'XRP/USDT')
        self.btc_symbol = config.get('btc_symbol', 'BTC/USDT')
        self.lookback = config.get('lookback', 168)  # 1 week of 1h candles
        self.entry_z = config.get('entry_z', 2.0)
        self.exit_z = config.get('exit_z', 0.5)
        self.stop_z = config.get('stop_z', 3.0)
        self.max_leverage = config.get('max_leverage', 10)
        self.position_size_pct = config.get('position_size_pct', 0.10)

        # State
        self.hedge_ratio = None
        self.spread_mean = None
        self.spread_std = None
        self.current_position = None  # 'long_xrp' or 'short_xrp' or None
        self.last_zscore = 0.0
        self.cointegration_pvalue = None

    def update_hedge_ratio(self, data: Dict[str, pd.DataFrame]) -> bool:
        """
        Update hedge ratio using OLS regression from statsmodels.

        Returns True if hedge ratio was successfully updated.
        """
        if self.xrp_symbol not in data or self.btc_symbol not in data:
            return False

        xrp_df = data[self.xrp_symbol]
        btc_df = data[self.btc_symbol]

        if len(xrp_df) < self.lookback or len(btc_df) < self.lookback:
            return False

        xrp_prices = xrp_df['close'].iloc[-self.lookback:].values
        btc_prices = btc_df['close'].iloc[-self.lookback:].values

        if STATSMODELS_AVAILABLE:
            # Use statsmodels OLS for proper regression
            btc_with_const = sm.add_constant(btc_prices)
            model = sm.OLS(xrp_prices, btc_with_const).fit()
            self.hedge_ratio = model.params[1]  # Beta coefficient

            # Optional: Test cointegration
            try:
                _, pvalue, _ = coint(xrp_prices, btc_prices)
                self.cointegration_pvalue = pvalue
            except Exception:
                self.cointegration_pvalue = None
        else:
            # Fallback: Simple OLS without statsmodels
            btc_mean = np.mean(btc_prices)
            xrp_mean = np.mean(xrp_prices)
            covariance = np.sum((btc_prices - btc_mean) * (xrp_prices - xrp_mean))
            variance = np.sum((btc_prices - btc_mean) ** 2)
            self.hedge_ratio = covariance / variance if variance > 0 else 0

        return self.hedge_ratio is not None and self.hedge_ratio != 0

    def _calculate_spread_zscore(self, data: Dict[str, pd.DataFrame]) -> Tuple[float, float]:
        """
        Calculate current spread and z-score.

        Returns (spread, zscore)
        """
        xrp_df = data[self.xrp_symbol]
        btc_df = data[self.btc_symbol]

        # Current prices
        xrp_price = xrp_df['close'].iloc[-1]
        btc_price = btc_df['close'].iloc[-1]

        # Current spread
        current_spread = xrp_price - self.hedge_ratio * btc_price

        # Historical spread for z-score
        xrp_hist = xrp_df['close'].iloc[-self.lookback:].values
        btc_hist = btc_df['close'].iloc[-self.lookback:].values
        historical_spread = xrp_hist - self.hedge_ratio * btc_hist

        self.spread_mean = np.mean(historical_spread)
        self.spread_std = np.std(historical_spread)

        if self.spread_std == 0:
            return current_spread, 0.0

        zscore = (current_spread - self.spread_mean) / self.spread_std

        return current_spread, zscore

    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Generate pair trading signals based on spread z-score.
        """
        # Check both symbols are available
        if self.xrp_symbol not in data or self.btc_symbol not in data:
            return {
                'action': 'hold',
                'symbol': 'PAIR',
                'size': 0.0,
                'leverage': 1,
                'confidence': 0.0,
                'reason': 'Missing XRP or BTC data'
            }

        xrp_df = data[self.xrp_symbol]
        btc_df = data[self.btc_symbol]

        if len(xrp_df) < self.lookback or len(btc_df) < self.lookback:
            return {
                'action': 'hold',
                'symbol': 'PAIR',
                'size': 0.0,
                'leverage': 1,
                'confidence': 0.0,
                'reason': f'Insufficient data for lookback ({self.lookback})'
            }

        # Update hedge ratio if needed
        if self.hedge_ratio is None:
            if not self.update_hedge_ratio(data):
                return {
                    'action': 'hold',
                    'symbol': 'PAIR',
                    'size': 0.0,
                    'leverage': 1,
                    'confidence': 0.0,
                    'reason': 'Could not calculate hedge ratio'
                }

        # Calculate spread and z-score
        spread, zscore = self._calculate_spread_zscore(data)
        self.last_zscore = zscore

        xrp_price = xrp_df['close'].iloc[-1]
        btc_price = btc_df['close'].iloc[-1]

        # Check cointegration strength (if available)
        coint_ok = self.cointegration_pvalue is None or self.cointegration_pvalue < 0.05

        # Default signal
        signal = {
            'action': 'hold',
            'symbol': 'PAIR',
            'size': 0.0,
            'leverage': 1,
            'confidence': 0.0,
            'reason': f'Z-score {zscore:.2f} within range [{-self.entry_z:.1f}, {self.entry_z:.1f}]',
            'indicators': {
                'hedge_ratio': self.hedge_ratio,
                'zscore': zscore,
                'spread': spread,
                'spread_mean': self.spread_mean,
                'spread_std': self.spread_std,
                'xrp_price': xrp_price,
                'btc_price': btc_price,
                'cointegration_pvalue': self.cointegration_pvalue,
                'current_position': self.current_position
            }
        }

        # STOP LOSS: Close on extreme divergence
        if self.current_position and abs(zscore) > self.stop_z:
            signal = {
                'action': 'close_pair',
                'symbol': 'PAIR',
                'size': self.position_size_pct,
                'leverage': 1,
                'confidence': 0.9,
                'reason': f'STOP: Z-score {zscore:.2f} exceeds {self.stop_z}',
                'indicators': signal['indicators'],
                'close_position': self.current_position
            }
            self.current_position = None
            return signal

        # EXIT: Mean reversion complete
        if self.current_position:
            should_exit = abs(zscore) < self.exit_z
            if should_exit:
                signal = {
                    'action': 'close_pair',
                    'symbol': 'PAIR',
                    'size': self.position_size_pct,
                    'leverage': 1,
                    'confidence': 0.85,
                    'reason': f'EXIT: Z-score {zscore:.2f} reverted to mean',
                    'indicators': signal['indicators'],
                    'close_position': self.current_position
                }
                self.current_position = None
                return signal

        # ENTRY signals (only if no current position and cointegration is ok)
        if self.current_position is None and coint_ok:
            # Z-score > entry_z: Spread too high, short XRP / long BTC
            if zscore > self.entry_z:
                confidence = min(0.5 + (zscore - self.entry_z) * 0.15, 0.95)
                signal = {
                    'action': 'short_xrp_long_btc',
                    'symbol': 'PAIR',
                    'size': self.position_size_pct,
                    'leverage': self.max_leverage,
                    'confidence': confidence,
                    'reason': f'ENTRY: Z-score {zscore:.2f} > {self.entry_z} - Short XRP, Long BTC',
                    'indicators': signal['indicators'],
                    'legs': [
                        {'symbol': self.xrp_symbol, 'side': 'short', 'size': self.position_size_pct},
                        {'symbol': self.btc_symbol, 'side': 'long', 'size': self.position_size_pct * abs(self.hedge_ratio)}
                    ]
                }
                self.current_position = 'short_xrp'

            # Z-score < -entry_z: Spread too low, long XRP / short BTC
            elif zscore < -self.entry_z:
                confidence = min(0.5 + (abs(zscore) - self.entry_z) * 0.15, 0.95)
                signal = {
                    'action': 'long_xrp_short_btc',
                    'symbol': 'PAIR',
                    'size': self.position_size_pct,
                    'leverage': self.max_leverage,
                    'confidence': confidence,
                    'reason': f'ENTRY: Z-score {zscore:.2f} < -{self.entry_z} - Long XRP, Short BTC',
                    'indicators': signal['indicators'],
                    'legs': [
                        {'symbol': self.xrp_symbol, 'side': 'long', 'size': self.position_size_pct},
                        {'symbol': self.btc_symbol, 'side': 'short', 'size': self.position_size_pct * abs(self.hedge_ratio)}
                    ]
                }
                self.current_position = 'long_xrp'

        return signal

    def update_model(self, data: Optional[Dict[str, pd.DataFrame]] = None) -> bool:
        """
        Update hedge ratio from new data.
        Should be called periodically to adapt to changing market conditions.
        """
        if data is None:
            return True

        return self.update_hedge_ratio(data)

    def get_status(self) -> Dict[str, Any]:
        """Get strategy status."""
        base_status = super().get_status()
        base_status.update({
            'xrp_symbol': self.xrp_symbol,
            'btc_symbol': self.btc_symbol,
            'hedge_ratio': self.hedge_ratio,
            'current_position': self.current_position,
            'last_zscore': self.last_zscore,
            'cointegration_pvalue': self.cointegration_pvalue,
            'lookback': self.lookback,
            'statsmodels_available': STATSMODELS_AVAILABLE,
            'thresholds': {
                'entry_z': self.entry_z,
                'exit_z': self.exit_z,
                'stop_z': self.stop_z
            }
        })
        return base_status
