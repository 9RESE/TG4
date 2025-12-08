"""
XRP/BTC Pair Trading Strategy
Phase 16: Tuned for Dec 2025 market conditions

Statistical arbitrage exploiting cointegration between XRP and BTC.
Uses OLS regression for dynamic hedge ratio and z-score for entry/exit.
Optimized for Kraken XRP/BTC pair with 10x margin.

Phase 16 Tuning:
- Extended lookback: 336h (2 weeks) for more stable hedge ratio
- Lower entry_z: 1.8 for earlier signals on temporary decorrelation
- BTC momentum filter: Only trade when BTC RSI supports direction
- Dynamic hedge ratio recalculation every 24h

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

    Phase 16 Tuning:
    - Extended lookback (336h = 2 weeks) for stable hedge ratio
    - Lower entry threshold (1.8 z-score vs 2.0)
    - BTC momentum filter for directional confirmation
    - Recalculate hedge ratio every 24 candles (1 day)

    Entry:
    - Z-score > entry_z AND BTC RSI > 50: Short XRP, Long BTC
    - Z-score < -entry_z AND BTC RSI < 50: Long XRP, Short BTC

    Exit:
    - |Z-score| < exit_z (mean reverted)
    - |Z-score| > stop_z (divergence stop)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.name = 'xrp_btc_pair_trading'
        self.xrp_symbol = config.get('xrp_symbol', 'XRP/USDT')
        self.btc_symbol = config.get('btc_symbol', 'BTC/USDT')

        # Phase 16: Extended lookback for stable hedge ratio
        self.lookback = config.get('lookback', 336)  # 2 weeks (was 168)

        # Phase 16: Lower entry for earlier signals
        self.entry_z = config.get('entry_z', 1.8)  # was 2.0
        self.exit_z = config.get('exit_z', 0.5)
        self.stop_z = config.get('stop_z', 3.0)

        self.max_leverage = config.get('max_leverage', 10)
        self.position_size_pct = config.get('position_size_pct', 0.10)

        # Phase 16: BTC momentum filter
        self.use_btc_filter = config.get('use_btc_filter', True)
        self.btc_rsi_window = config.get('btc_rsi_window', 14)

        # Phase 16: Dynamic hedge ratio recalculation
        self.hedge_recalc_period = config.get('hedge_recalc_period', 24)  # hours
        self.candles_since_recalc = 0

        # Phase 16: Cointegration threshold
        self.coint_pvalue_threshold = config.get('coint_pvalue_threshold', 0.10)  # was 0.05

        # State
        self.hedge_ratio = None
        self.spread_mean = None
        self.spread_std = None
        self.current_position = None
        self.last_zscore = 0.0
        self.cointegration_pvalue = None
        self.btc_rsi = 50.0

    def _calculate_rsi(self, prices: np.ndarray, window: int = 14) -> float:
        """Calculate RSI from price array."""
        if len(prices) < window + 1:
            return 50.0

        deltas = np.diff(prices[-(window + 1):])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0.001

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def update_hedge_ratio(self, data: Dict[str, pd.DataFrame], force: bool = False) -> bool:
        """
        Update hedge ratio using OLS regression from statsmodels.
        Phase 16: Recalculate every hedge_recalc_period candles.
        """
        # Check if recalculation needed
        self.candles_since_recalc += 1
        if not force and self.hedge_ratio is not None and self.candles_since_recalc < self.hedge_recalc_period:
            return True

        if self.xrp_symbol not in data or self.btc_symbol not in data:
            return False

        xrp_df = data[self.xrp_symbol]
        btc_df = data[self.btc_symbol]

        if len(xrp_df) < self.lookback or len(btc_df) < self.lookback:
            return False

        xrp_prices = xrp_df['close'].iloc[-self.lookback:].values
        btc_prices = btc_df['close'].iloc[-self.lookback:].values

        # Calculate BTC RSI for filter
        self.btc_rsi = self._calculate_rsi(btc_prices, self.btc_rsi_window)

        if STATSMODELS_AVAILABLE:
            # Use statsmodels OLS for proper regression
            btc_with_const = sm.add_constant(btc_prices)
            model = sm.OLS(xrp_prices, btc_with_const).fit()
            self.hedge_ratio = model.params[1]

            # Test cointegration
            try:
                _, pvalue, _ = coint(xrp_prices, btc_prices)
                self.cointegration_pvalue = pvalue
            except Exception:
                self.cointegration_pvalue = None
        else:
            # Fallback: Simple OLS
            btc_mean = np.mean(btc_prices)
            xrp_mean = np.mean(xrp_prices)
            covariance = np.sum((btc_prices - btc_mean) * (xrp_prices - xrp_mean))
            variance = np.sum((btc_prices - btc_mean) ** 2)
            self.hedge_ratio = covariance / variance if variance > 0 else 0

        self.candles_since_recalc = 0
        return self.hedge_ratio is not None and self.hedge_ratio != 0

    def _calculate_spread_zscore(self, data: Dict[str, pd.DataFrame]) -> Tuple[float, float]:
        """Calculate current spread and z-score."""
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

    def _check_btc_filter(self, action: str) -> bool:
        """
        Phase 16: Check if BTC momentum supports the trade direction.
        - Short XRP/Long BTC: Want BTC strong (RSI > 50)
        - Long XRP/Short BTC: Want BTC weak (RSI < 50)
        """
        if not self.use_btc_filter:
            return True

        if action == 'short_xrp_long_btc':
            return self.btc_rsi > 50  # BTC should be strong
        elif action == 'long_xrp_short_btc':
            return self.btc_rsi < 50  # BTC should be weak

        return True

    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Generate pair trading signals based on spread z-score and BTC filter.
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

        # Update hedge ratio (recalculates if needed)
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

        # Phase 16: Relaxed cointegration check
        coint_ok = (self.cointegration_pvalue is None or
                    self.cointegration_pvalue < self.coint_pvalue_threshold)

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
                'btc_rsi': self.btc_rsi,
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

        # ENTRY signals (only if no current position)
        if self.current_position is None:

            # Z-score > entry_z: Spread too high, short XRP / long BTC
            if zscore > self.entry_z:
                btc_filter_ok = self._check_btc_filter('short_xrp_long_btc')

                if coint_ok and btc_filter_ok:
                    confidence = min(0.5 + (zscore - self.entry_z) * 0.15, 0.95)
                    signal = {
                        'action': 'short_xrp_long_btc',
                        'symbol': 'PAIR',
                        'size': self.position_size_pct,
                        'leverage': self.max_leverage,
                        'confidence': confidence,
                        'reason': f'ENTRY: Z-score {zscore:.2f} > {self.entry_z}, BTC RSI {self.btc_rsi:.1f} > 50',
                        'indicators': signal['indicators'],
                        'legs': [
                            {'symbol': self.xrp_symbol, 'side': 'short', 'size': self.position_size_pct},
                            {'symbol': self.btc_symbol, 'side': 'long', 'size': self.position_size_pct * abs(self.hedge_ratio)}
                        ]
                    }
                    self.current_position = 'short_xrp'
                elif not btc_filter_ok:
                    signal['reason'] = f'Z-score {zscore:.2f} > {self.entry_z} but BTC RSI {self.btc_rsi:.1f} < 50 (filter blocked)'
                elif not coint_ok:
                    signal['reason'] = f'Z-score {zscore:.2f} but weak cointegration (p={self.cointegration_pvalue:.3f})'

            # Z-score < -entry_z: Spread too low, long XRP / short BTC
            elif zscore < -self.entry_z:
                btc_filter_ok = self._check_btc_filter('long_xrp_short_btc')

                if coint_ok and btc_filter_ok:
                    confidence = min(0.5 + (abs(zscore) - self.entry_z) * 0.15, 0.95)
                    signal = {
                        'action': 'long_xrp_short_btc',
                        'symbol': 'PAIR',
                        'size': self.position_size_pct,
                        'leverage': self.max_leverage,
                        'confidence': confidence,
                        'reason': f'ENTRY: Z-score {zscore:.2f} < -{self.entry_z}, BTC RSI {self.btc_rsi:.1f} < 50',
                        'indicators': signal['indicators'],
                        'legs': [
                            {'symbol': self.xrp_symbol, 'side': 'long', 'size': self.position_size_pct},
                            {'symbol': self.btc_symbol, 'side': 'short', 'size': self.position_size_pct * abs(self.hedge_ratio)}
                        ]
                    }
                    self.current_position = 'long_xrp'
                elif not btc_filter_ok:
                    signal['reason'] = f'Z-score {zscore:.2f} < -{self.entry_z} but BTC RSI {self.btc_rsi:.1f} > 50 (filter blocked)'
                elif not coint_ok:
                    signal['reason'] = f'Z-score {zscore:.2f} but weak cointegration (p={self.cointegration_pvalue:.3f})'

        return signal

    def update_model(self, data: Optional[Dict[str, pd.DataFrame]] = None) -> bool:
        """Force recalculation of hedge ratio."""
        if data is None:
            return True
        return self.update_hedge_ratio(data, force=True)

    def get_status(self) -> Dict[str, Any]:
        """Get strategy status."""
        base_status = super().get_status()
        base_status.update({
            'xrp_symbol': self.xrp_symbol,
            'btc_symbol': self.btc_symbol,
            'hedge_ratio': self.hedge_ratio,
            'current_position': self.current_position,
            'last_zscore': self.last_zscore,
            'btc_rsi': self.btc_rsi,
            'cointegration_pvalue': self.cointegration_pvalue,
            'lookback': self.lookback,
            'use_btc_filter': self.use_btc_filter,
            'coint_pvalue_threshold': self.coint_pvalue_threshold,
            'statsmodels_available': STATSMODELS_AVAILABLE,
            'thresholds': {
                'entry_z': self.entry_z,
                'exit_z': self.exit_z,
                'stop_z': self.stop_z
            }
        })
        return base_status
