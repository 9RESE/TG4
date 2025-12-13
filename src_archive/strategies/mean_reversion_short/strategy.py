"""
Phase 24: Mean Reversion Short Strategy
VWAP/RSI filter optimized for short entries on overbought conditions.

Features:
- VWAP deviation detection
- RSI overbought/oversold
- VWAP bands for entry/exit zones
- Focus on shorting overextended moves
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from strategies.base_strategy import BaseStrategy
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple


class MeanReversionShort(BaseStrategy):
    """
    Mean reversion strategy focused on shorting overbought conditions.
    Uses VWAP deviation + RSI for high-probability short entries.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.symbol = config.get('symbol', 'XRP/USDT')

        # VWAP parameters
        self.vwap_period = config.get('vwap_period', 20)
        self.deviation_threshold = config.get('deviation_threshold', 0.02)  # 2%
        self.band_std_mult = config.get('band_std_mult', 2.0)

        # RSI parameters
        self.rsi_period = config.get('rsi_period', 14)
        self.rsi_overbought = config.get('rsi_overbought', 65)
        self.rsi_oversold = config.get('rsi_oversold', 35)

        # State
        self.last_vwap = 0.0
        self.last_deviation = 0.0
        self.last_rsi = 50.0
        self.in_position = False
        self.position_type = None  # 'long' or 'short'

    def _calculate_vwap(self, df: pd.DataFrame) -> float:
        """Calculate Volume Weighted Average Price."""
        if len(df) < self.vwap_period:
            return 0.0

        recent = df.tail(self.vwap_period)
        typical_price = (recent['high'] + recent['low'] + recent['close']) / 3
        volume = recent['volume']

        if volume.sum() <= 0:
            return 0.0

        return (typical_price * volume).sum() / volume.sum()

    def _get_vwap_deviation(self, df: pd.DataFrame) -> Tuple[float, float]:
        """Get current price deviation from VWAP."""
        if len(df) < self.vwap_period:
            return 0.0, 0.0

        current_price = df['close'].iloc[-1]
        vwap = self._calculate_vwap(df)

        if vwap <= 0:
            return 0.0, 0.0

        deviation_pct = (current_price - vwap) / vwap
        return deviation_pct, vwap

    def _calculate_vwap_bands(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate VWAP bands (similar to Bollinger around VWAP)."""
        if len(df) < self.vwap_period:
            return {'vwap': 0.0, 'upper': 0.0, 'lower': 0.0}

        vwap = self._calculate_vwap(df)
        recent = df.tail(self.vwap_period)
        typical_price = (recent['high'] + recent['low'] + recent['close']) / 3
        std_dev = typical_price.std()

        return {
            'vwap': vwap,
            'upper': vwap + (self.band_std_mult * std_dev),
            'lower': vwap - (self.band_std_mult * std_dev),
            'std_dev': std_dev
        }

    def _calculate_rsi(self, close: np.ndarray) -> float:
        """Calculate RSI indicator."""
        if len(close) < self.rsi_period + 1:
            return 50.0

        deltas = np.diff(close[-self.rsi_period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0.001

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Generate mean reversion signals focused on shorting.

        Args:
            data: Dict with symbol -> DataFrame (OHLCV)

        Returns:
            Signal dict with action, confidence, position details
        """
        # Find XRP data
        xrp_key = None
        for key in data.keys():
            if 'XRP' in key.upper():
                xrp_key = key
                break

        if not xrp_key or xrp_key not in data:
            return {
                'action': 'hold',
                'symbol': self.symbol,
                'confidence': 0.0,
                'reason': 'No XRP data available',
                'strategy': 'mean_reversion_short'
            }

        df = data[xrp_key]
        if len(df) < 30:
            return {
                'action': 'hold',
                'symbol': xrp_key,
                'confidence': 0.0,
                'reason': 'Insufficient data',
                'strategy': 'mean_reversion_short'
            }

        close = df['close'].values
        current_price = close[-1]

        # Calculate indicators
        deviation_pct, vwap = self._get_vwap_deviation(df)
        rsi = self._calculate_rsi(close)
        bands = self._calculate_vwap_bands(df)

        # Store state
        self.last_vwap = vwap
        self.last_deviation = deviation_pct
        self.last_rsi = rsi

        # Signal generation
        action = 'hold'
        confidence = 0.0
        reason = ''

        # SHORT signal: Price significantly above VWAP + overbought RSI
        if deviation_pct > self.deviation_threshold and rsi > self.rsi_overbought:
            action = 'short'
            # Confidence scales with deviation and RSI extremity
            confidence = min(0.5 + (deviation_pct * 5) + ((rsi - self.rsi_overbought) / 70), 0.95)
            reason = f'SHORT: {deviation_pct*100:.1f}% above VWAP, RSI={rsi:.1f}'

        # LONG signal: Price significantly below VWAP + oversold RSI
        elif deviation_pct < -self.deviation_threshold and rsi < self.rsi_oversold:
            action = 'buy'
            confidence = min(0.5 + (abs(deviation_pct) * 5) + ((self.rsi_oversold - rsi) / 70), 0.95)
            reason = f'LONG: {abs(deviation_pct)*100:.1f}% below VWAP, RSI={rsi:.1f}'

        # Cover short: Price returned to VWAP from above
        elif self.position_type == 'short' and deviation_pct < 0.005:
            action = 'cover'
            confidence = 0.7
            reason = f'Cover short: Price returned to VWAP ({deviation_pct*100:.2f}%)'

        # Sell long: Price returned to VWAP from below
        elif self.position_type == 'long' and deviation_pct > -0.005:
            action = 'sell'
            confidence = 0.7
            reason = f'Sell long: Price returned to VWAP ({deviation_pct*100:.2f}%)'

        # Band breakout signals (stronger)
        if current_price > bands['upper'] and rsi > 70:
            action = 'short'
            confidence = min(confidence + 0.15, 0.95)
            reason = f'UPPER BAND BREAK + RSI={rsi:.1f}'
        elif current_price < bands['lower'] and rsi < 30:
            action = 'buy'
            confidence = min(confidence + 0.15, 0.95)
            reason = f'LOWER BAND BREAK + RSI={rsi:.1f}'

        if action == 'hold':
            reason = f'No signal (dev={deviation_pct*100:.2f}%, RSI={rsi:.1f})'

        # Update position tracking for exit signals
        if action in ['buy', 'long']:
            self.position_type = 'long'
        elif action == 'short':
            self.position_type = 'short'
        elif action in ['sell', 'cover']:
            self.position_type = None

        return {
            'action': action,
            'symbol': xrp_key,
            'size': self.position_size_pct,
            'leverage': min(self.max_leverage, 5),
            'confidence': confidence,
            'reason': reason,
            'strategy': 'mean_reversion_short',
            'indicators': {
                'vwap': vwap,
                'deviation_pct': deviation_pct,
                'rsi': rsi,
                'upper_band': bands['upper'],
                'lower_band': bands['lower'],
                'current_price': current_price
            }
        }

    def update_model(self, data: Optional[Dict[str, pd.DataFrame]] = None) -> bool:
        """Mean reversion is rule-based, no model to update."""
        return True

    def get_status(self) -> Dict[str, Any]:
        """Get strategy status."""
        base_status = super().get_status()
        base_status.update({
            'last_vwap': self.last_vwap,
            'last_deviation': self.last_deviation,
            'last_rsi': self.last_rsi,
            'position_type': self.position_type,
            'deviation_threshold': self.deviation_threshold,
            'rsi_overbought': self.rsi_overbought,
            'rsi_oversold': self.rsi_oversold
        })
        return base_status
