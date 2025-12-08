"""
XRP/BTC Pair Trading Strategy
Phase 15: Modular Strategy Factory

Statistical arbitrage exploiting cointegration between XRP and BTC.
Uses OLS regression to calculate hedge ratio and z-score for entry/exit.

Entry: Z-score > 2 (short XRP, long BTC) or Z-score < -2 (long XRP, short BTC)
Exit: Z-score returns to 0 (mean reversion)

Target: 10x leverage on Kraken for both legs.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from strategies.base_strategy import BaseStrategy


class XRPBTCPairTrading(BaseStrategy):
    """
    Pair Trading Strategy using XRP and BTC cointegration.

    The strategy:
    1. Calculates hedge ratio via OLS regression
    2. Computes spread: XRP - hedge_ratio * BTC
    3. Calculates z-score of spread
    4. Trades mean reversion of spread

    Entry:
    - Z-score > 2: Short XRP, Long BTC (spread too high)
    - Z-score < -2: Long XRP, Short BTC (spread too low)

    Exit:
    - Z-score crosses 0 (mean reverted)
    - Stop at Z-score > 3 or < -3 (divergence)
    """

    # Z-score thresholds
    ZSCORE_ENTRY = 2.0
    ZSCORE_EXIT = 0.0
    ZSCORE_STOP = 3.0

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.name = 'xrp_btc_pair_trading'
        self.xrp_symbol = config.get('xrp_symbol', 'XRP/USDT')
        self.btc_symbol = config.get('btc_symbol', 'BTC/USDT')
        self.max_leverage = config.get('max_leverage', 10)
        self.position_size_pct = config.get('position_size_pct', 0.10)
        self.lookback_period = config.get('lookback_period', 100)

        # State
        self.hedge_ratio = None
        self.spread_mean = None
        self.spread_std = None
        self.current_position = None  # 'long_xrp' or 'short_xrp' or None

    def _calculate_hedge_ratio(self, xrp_prices: np.ndarray, btc_prices: np.ndarray) -> float:
        """
        Calculate hedge ratio using OLS regression.
        XRP = alpha + beta * BTC + epsilon
        hedge_ratio = beta
        """
        if len(xrp_prices) < 20 or len(btc_prices) < 20:
            return 0.0

        # Simple OLS without statsmodels dependency
        # beta = Cov(X, Y) / Var(X)
        btc_mean = np.mean(btc_prices)
        xrp_mean = np.mean(xrp_prices)

        covariance = np.sum((btc_prices - btc_mean) * (xrp_prices - xrp_mean))
        variance = np.sum((btc_prices - btc_mean) ** 2)

        if variance == 0:
            return 0.0

        beta = covariance / variance
        return beta

    def _calculate_spread(self, xrp_prices: np.ndarray, btc_prices: np.ndarray) -> np.ndarray:
        """Calculate spread: XRP - hedge_ratio * BTC"""
        if self.hedge_ratio is None or self.hedge_ratio == 0:
            self.hedge_ratio = self._calculate_hedge_ratio(xrp_prices, btc_prices)

        if self.hedge_ratio == 0:
            return np.zeros(len(xrp_prices))

        spread = xrp_prices - self.hedge_ratio * btc_prices
        return spread

    def _calculate_zscore(self, spread: np.ndarray, period: int = 20) -> float:
        """Calculate z-score of current spread."""
        if len(spread) < period:
            return 0.0

        recent_spread = spread[-period:]
        self.spread_mean = np.mean(recent_spread)
        self.spread_std = np.std(recent_spread)

        if self.spread_std == 0:
            return 0.0

        current_spread = spread[-1]
        zscore = (current_spread - self.spread_mean) / self.spread_std

        return zscore

    def _check_cointegration(self, xrp_prices: np.ndarray, btc_prices: np.ndarray) -> Tuple[bool, float]:
        """
        Simple cointegration check using correlation of returns.
        For production, use proper ADF test on residuals.
        """
        if len(xrp_prices) < 30:
            return False, 0.0

        # Calculate returns
        xrp_returns = np.diff(xrp_prices) / xrp_prices[:-1]
        btc_returns = np.diff(btc_prices) / btc_prices[:-1]

        # Correlation of returns
        correlation = np.corrcoef(xrp_returns, btc_returns)[0, 1]

        # Simple heuristic: high correlation suggests cointegration potential
        is_cointegrated = abs(correlation) > 0.5

        return is_cointegrated, correlation

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

        if len(xrp_df) < self.lookback_period or len(btc_df) < self.lookback_period:
            return {
                'action': 'hold',
                'symbol': 'PAIR',
                'size': 0.0,
                'leverage': 1,
                'confidence': 0.0,
                'reason': 'Insufficient data for lookback'
            }

        # Get price arrays (last lookback_period candles)
        xrp_prices = xrp_df['close'].values[-self.lookback_period:]
        btc_prices = btc_df['close'].values[-self.lookback_period:]

        # Check cointegration
        is_cointegrated, correlation = self._check_cointegration(xrp_prices, btc_prices)

        if not is_cointegrated:
            return {
                'action': 'hold',
                'symbol': 'PAIR',
                'size': 0.0,
                'leverage': 1,
                'confidence': 0.0,
                'reason': f'Weak cointegration (corr: {correlation:.2f})',
                'indicators': {
                    'correlation': correlation,
                    'is_cointegrated': False
                }
            }

        # Calculate hedge ratio and spread
        self.hedge_ratio = self._calculate_hedge_ratio(xrp_prices, btc_prices)
        spread = self._calculate_spread(xrp_prices, btc_prices)
        zscore = self._calculate_zscore(spread)

        current_xrp_price = xrp_df['close'].iloc[-1]
        current_btc_price = btc_df['close'].iloc[-1]

        # Default signal
        signal = {
            'action': 'hold',
            'symbol': 'PAIR',
            'size': 0.0,
            'leverage': 1,
            'confidence': 0.0,
            'reason': f'Z-score {zscore:.2f} within range',
            'indicators': {
                'hedge_ratio': self.hedge_ratio,
                'zscore': zscore,
                'spread': spread[-1] if len(spread) > 0 else 0,
                'spread_mean': self.spread_mean,
                'spread_std': self.spread_std,
                'correlation': correlation,
                'xrp_price': current_xrp_price,
                'btc_price': current_btc_price
            }
        }

        # Check for stop loss (divergence)
        if self.current_position and abs(zscore) > self.ZSCORE_STOP:
            signal = {
                'action': 'close',
                'symbol': 'PAIR',
                'size': self.position_size_pct,
                'leverage': 1,
                'confidence': 0.9,
                'reason': f'STOP: Z-score {zscore:.2f} exceeds {self.ZSCORE_STOP}',
                'indicators': signal['indicators'],
                'close_position': self.current_position
            }
            self.current_position = None
            return signal

        # Check for exit (mean reversion complete)
        if self.current_position:
            should_exit = (
                (self.current_position == 'long_xrp' and zscore > self.ZSCORE_EXIT) or
                (self.current_position == 'short_xrp' and zscore < self.ZSCORE_EXIT)
            )
            if should_exit:
                signal = {
                    'action': 'close',
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

        # Entry signals (only if no current position)
        if self.current_position is None:
            # Z-score > 2: Spread too high, short XRP / long BTC
            if zscore > self.ZSCORE_ENTRY:
                confidence = min(0.5 + (zscore - self.ZSCORE_ENTRY) * 0.2, 0.95)
                signal = {
                    'action': 'short_xrp_long_btc',
                    'symbol': 'PAIR',
                    'size': self.position_size_pct,
                    'leverage': self.max_leverage,
                    'confidence': confidence,
                    'reason': f'ENTRY: Z-score {zscore:.2f} > {self.ZSCORE_ENTRY} - Short XRP, Long BTC',
                    'indicators': signal['indicators'],
                    'legs': [
                        {'symbol': self.xrp_symbol, 'side': 'short', 'size': self.position_size_pct},
                        {'symbol': self.btc_symbol, 'side': 'long', 'size': self.position_size_pct * self.hedge_ratio}
                    ]
                }
                self.current_position = 'short_xrp'

            # Z-score < -2: Spread too low, long XRP / short BTC
            elif zscore < -self.ZSCORE_ENTRY:
                confidence = min(0.5 + (abs(zscore) - self.ZSCORE_ENTRY) * 0.2, 0.95)
                signal = {
                    'action': 'long_xrp_short_btc',
                    'symbol': 'PAIR',
                    'size': self.position_size_pct,
                    'leverage': self.max_leverage,
                    'confidence': confidence,
                    'reason': f'ENTRY: Z-score {zscore:.2f} < -{self.ZSCORE_ENTRY} - Long XRP, Short BTC',
                    'indicators': signal['indicators'],
                    'legs': [
                        {'symbol': self.xrp_symbol, 'side': 'long', 'size': self.position_size_pct},
                        {'symbol': self.btc_symbol, 'side': 'short', 'size': self.position_size_pct * self.hedge_ratio}
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

        if self.xrp_symbol in data and self.btc_symbol in data:
            xrp_prices = data[self.xrp_symbol]['close'].values
            btc_prices = data[self.btc_symbol]['close'].values
            self.hedge_ratio = self._calculate_hedge_ratio(xrp_prices, btc_prices)
            return True

        return False

    def get_status(self) -> Dict[str, Any]:
        """Get strategy status."""
        base_status = super().get_status()
        base_status.update({
            'xrp_symbol': self.xrp_symbol,
            'btc_symbol': self.btc_symbol,
            'hedge_ratio': self.hedge_ratio,
            'current_position': self.current_position,
            'lookback_period': self.lookback_period,
            'thresholds': {
                'entry': self.ZSCORE_ENTRY,
                'exit': self.ZSCORE_EXIT,
                'stop': self.ZSCORE_STOP
            }
        })
        return base_status
