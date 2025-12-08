"""
Mean Reversion VWAP Strategy
Phase 15: Enhanced with ta library for proper VWAP/RSI

Trades mean reversion around VWAP using RSI as confirmation.
Optimized for XRP/USDT chop in $2.00-2.10 range (Dec 2025).

- Long: RSI < 30 AND price > 0.5% below VWAP
- Short: RSI > 70 AND price > 0.5% above VWAP
- 5x leverage on Kraken for XRP

Target: Capture 5-15% in choppy conditions.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

try:
    import ta.volume as vol
    import ta.momentum as mom
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    print("Warning: ta library not installed. Using fallback calculations.")

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from strategies.base_strategy import BaseStrategy


class MeanReversionVWAP(BaseStrategy):
    """
    Mean Reversion Strategy using VWAP and RSI from ta library.

    Entry Conditions:
    - LONG: RSI < 30 (oversold) AND price < VWAP * (1 - dev_threshold)
    - SHORT: RSI > 70 (overbought) AND price > VWAP * (1 + dev_threshold)

    Exit Conditions:
    - Price reverts to VWAP (take profit)
    - Stop loss at 5% adverse move
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.name = 'mean_reversion_vwap'
        self.symbol = config.get('symbol', 'XRP/USDT')
        self.vwap_window = config.get('vwap_window', 14)
        self.rsi_window = config.get('rsi_window', 14)
        self.dev_threshold = config.get('dev_threshold', 0.005)  # 0.5%
        self.max_leverage = config.get('max_leverage', 5)
        self.long_size = config.get('long_size', 0.12)
        self.short_size = config.get('short_size', 0.10)

    def _calculate_vwap_ta(self, df: pd.DataFrame) -> pd.Series:
        """Calculate VWAP using ta library."""
        if TA_AVAILABLE and len(df) >= self.vwap_window:
            vwap_indicator = vol.VolumeWeightedAveragePrice(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                volume=df['volume'],
                window=self.vwap_window
            )
            return vwap_indicator.volume_weighted_average_price()
        else:
            # Fallback calculation
            return self._calculate_vwap_fallback(df)

    def _calculate_vwap_fallback(self, df: pd.DataFrame) -> pd.Series:
        """Fallback VWAP calculation without ta library."""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).rolling(self.vwap_window).sum() / \
               df['volume'].rolling(self.vwap_window).sum()
        return vwap

    def _calculate_rsi_ta(self, df: pd.DataFrame) -> pd.Series:
        """Calculate RSI using ta library."""
        if TA_AVAILABLE and len(df) >= self.rsi_window:
            rsi_indicator = mom.RSIIndicator(
                close=df['close'],
                window=self.rsi_window
            )
            return rsi_indicator.rsi()
        else:
            # Fallback calculation
            return self._calculate_rsi_fallback(df)

    def _calculate_rsi_fallback(self, df: pd.DataFrame) -> pd.Series:
        """Fallback RSI calculation without ta library."""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_window).mean()
        rs = gain / loss.replace(0, 0.001)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Generate mean reversion signals based on VWAP and RSI.
        """
        if self.symbol not in data:
            return {
                'action': 'hold',
                'symbol': self.symbol,
                'size': 0.0,
                'leverage': 1,
                'confidence': 0.0,
                'reason': f'{self.symbol} not in data'
            }

        df = data[self.symbol].copy()
        min_periods = max(self.vwap_window, self.rsi_window) + 5

        if len(df) < min_periods:
            return {
                'action': 'hold',
                'symbol': self.symbol,
                'size': 0.0,
                'leverage': 1,
                'confidence': 0.0,
                'reason': f'Insufficient data ({len(df)} < {min_periods})'
            }

        # Calculate indicators using ta library
        df['vwap'] = self._calculate_vwap_ta(df)
        df['rsi'] = self._calculate_rsi_ta(df)

        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest

        current_price = latest['close']
        vwap = latest['vwap']
        rsi = latest['rsi']

        # Handle NaN
        if pd.isna(vwap) or pd.isna(rsi):
            return {
                'action': 'hold',
                'symbol': self.symbol,
                'size': 0.0,
                'leverage': 1,
                'confidence': 0.0,
                'reason': 'Indicators not ready (NaN)',
                'indicators': {'vwap': vwap, 'rsi': rsi, 'price': current_price}
            }

        # Calculate deviation from VWAP
        vwap_deviation = (current_price - vwap) / vwap if vwap > 0 else 0

        # Default signal
        signal = {
            'action': 'hold',
            'symbol': self.symbol,
            'size': 0.0,
            'leverage': 1,
            'confidence': 0.0,
            'reason': 'No signal',
            'indicators': {
                'vwap': vwap,
                'rsi': rsi,
                'price': current_price,
                'vwap_deviation': vwap_deviation,
                'prev_rsi': prev['rsi'] if not pd.isna(prev['rsi']) else 50
            }
        }

        # Long signal: RSI oversold AND price below VWAP by threshold
        if rsi < 30 and current_price < vwap * (1 - self.dev_threshold):
            # Confidence based on extremity
            rsi_score = (30 - rsi) / 30
            dev_score = min(abs(vwap_deviation) / self.dev_threshold, 2) / 2
            confidence = min(0.5 + rsi_score * 0.3 + dev_score * 0.2, 0.95)

            signal = {
                'action': 'buy',
                'symbol': self.symbol,
                'asset': 'XRP',
                'size': self.long_size,
                'leverage': self.max_leverage,
                'confidence': confidence,
                'reason': f'Long: RSI {rsi:.1f} < 30, price {vwap_deviation*100:.2f}% below VWAP',
                'indicators': signal['indicators'],
                'stop_loss': current_price * 0.95,  # 5% stop
                'take_profit': vwap  # Target VWAP
            }

        # Short signal: RSI overbought AND price above VWAP by threshold
        elif rsi > 70 and current_price > vwap * (1 + self.dev_threshold):
            rsi_score = (rsi - 70) / 30
            dev_score = min(abs(vwap_deviation) / self.dev_threshold, 2) / 2
            confidence = min(0.5 + rsi_score * 0.3 + dev_score * 0.2, 0.95)

            signal = {
                'action': 'sell',
                'symbol': self.symbol,
                'asset': 'XRP',
                'size': self.short_size,
                'leverage': self.max_leverage,
                'confidence': confidence,
                'reason': f'Short: RSI {rsi:.1f} > 70, price +{vwap_deviation*100:.2f}% above VWAP',
                'indicators': signal['indicators'],
                'stop_loss': current_price * 1.05,  # 5% stop
                'take_profit': vwap  # Target VWAP
            }

        return signal

    def update_model(self, data: Optional[Dict[str, pd.DataFrame]] = None) -> bool:
        """No ML model to update - pure indicator-based strategy."""
        return True

    def get_status(self) -> Dict[str, Any]:
        """Get strategy status."""
        base_status = super().get_status()
        base_status.update({
            'symbol': self.symbol,
            'vwap_window': self.vwap_window,
            'rsi_window': self.rsi_window,
            'dev_threshold': self.dev_threshold,
            'ta_available': TA_AVAILABLE
        })
        return base_status
