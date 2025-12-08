"""
Mean Reversion VWAP Strategy
Phase 15: Modular Strategy Factory

Trades mean reversion around VWAP using RSI as confirmation.
- Long: RSI < 30 AND price > 0.5% below VWAP
- Short: RSI > 70 AND price > 0.5% above VWAP

Target: XRP/USDT with 5x leverage on Kraken.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from strategies.base_strategy import BaseStrategy


class MeanReversionVWAP(BaseStrategy):
    """
    Mean Reversion Strategy using VWAP and RSI.

    Entry Conditions:
    - LONG: RSI < 30 (oversold) AND price < VWAP * 0.995 (0.5% below)
    - SHORT: RSI > 70 (overbought) AND price > VWAP * 1.005 (0.5% above)

    Exit Conditions:
    - Price reverts to VWAP (take profit)
    - RSI crosses 50 (mean reversion complete)
    - Stop loss at 2% adverse move
    """

    # Thresholds
    RSI_OVERSOLD = 30
    RSI_OVERBOUGHT = 70
    RSI_NEUTRAL = 50
    VWAP_DEVIATION_LONG = 0.995   # 0.5% below VWAP
    VWAP_DEVIATION_SHORT = 1.005  # 0.5% above VWAP

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.name = 'mean_reversion_vwap'
        self.symbol = config.get('symbol', 'XRP/USDT')
        self.max_leverage = config.get('max_leverage', 5)
        self.position_size_pct = config.get('position_size_pct', 0.10)
        self.vwap_period = config.get('vwap_period', 20)
        self.rsi_period = config.get('rsi_period', 14)

    def _calculate_vwap(self, df: pd.DataFrame, period: int = 20) -> float:
        """Calculate Volume Weighted Average Price."""
        if len(df) < period:
            return 0.0

        recent = df.tail(period)
        typical_price = (recent['high'] + recent['low'] + recent['close']) / 3
        volume = recent['volume']

        if volume.sum() == 0:
            return 0.0

        vwap = (typical_price * volume).sum() / volume.sum()
        return vwap

    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Relative Strength Index."""
        if len(df) < period + 1:
            return 50.0

        close = df['close'].values
        deltas = np.diff(close[-(period + 1):])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0.001

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range."""
        if len(df) < period + 1:
            return 0.0

        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        tr_list = []
        for i in range(1, len(high)):
            tr = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
            tr_list.append(tr)

        if len(tr_list) < period:
            return 0.0

        atr = np.mean(tr_list[-period:])
        return atr

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

        df = data[self.symbol]
        if len(df) < max(self.vwap_period, self.rsi_period) + 1:
            return {
                'action': 'hold',
                'symbol': self.symbol,
                'size': 0.0,
                'leverage': 1,
                'confidence': 0.0,
                'reason': 'Insufficient data'
            }

        # Calculate indicators
        vwap = self._calculate_vwap(df, self.vwap_period)
        rsi = self._calculate_rsi(df, self.rsi_period)
        current_price = df['close'].iloc[-1]
        atr = self._calculate_atr(df)

        # Calculate price deviation from VWAP
        if vwap > 0:
            price_to_vwap = current_price / vwap
        else:
            price_to_vwap = 1.0

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
                'price_to_vwap': price_to_vwap,
                'atr': atr
            }
        }

        # Long signal: RSI oversold AND price below VWAP
        if rsi < self.RSI_OVERSOLD and price_to_vwap < self.VWAP_DEVIATION_LONG:
            # Confidence based on how extreme the conditions are
            rsi_score = (self.RSI_OVERSOLD - rsi) / self.RSI_OVERSOLD  # 0-1
            vwap_score = (self.VWAP_DEVIATION_LONG - price_to_vwap) * 100  # deviation %
            confidence = min(0.5 + rsi_score * 0.3 + vwap_score * 0.2, 1.0)

            signal = {
                'action': 'buy',
                'symbol': self.symbol,
                'size': self.position_size_pct,
                'leverage': self.max_leverage,
                'confidence': confidence,
                'reason': f'Long: RSI {rsi:.1f} < {self.RSI_OVERSOLD}, price {(price_to_vwap-1)*100:.2f}% vs VWAP',
                'indicators': signal['indicators'],
                'stop_loss': current_price * 0.98,  # 2% stop
                'take_profit': vwap  # Target VWAP
            }

        # Short signal: RSI overbought AND price above VWAP
        elif rsi > self.RSI_OVERBOUGHT and price_to_vwap > self.VWAP_DEVIATION_SHORT:
            rsi_score = (rsi - self.RSI_OVERBOUGHT) / (100 - self.RSI_OVERBOUGHT)
            vwap_score = (price_to_vwap - self.VWAP_DEVIATION_SHORT) * 100
            confidence = min(0.5 + rsi_score * 0.3 + vwap_score * 0.2, 1.0)

            signal = {
                'action': 'sell',  # Short
                'symbol': self.symbol,
                'size': self.position_size_pct * 0.8,  # Slightly smaller shorts
                'leverage': self.max_leverage,
                'confidence': confidence,
                'reason': f'Short: RSI {rsi:.1f} > {self.RSI_OVERBOUGHT}, price +{(price_to_vwap-1)*100:.2f}% vs VWAP',
                'indicators': signal['indicators'],
                'stop_loss': current_price * 1.02,  # 2% stop
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
            'vwap_period': self.vwap_period,
            'rsi_period': self.rsi_period,
            'thresholds': {
                'rsi_oversold': self.RSI_OVERSOLD,
                'rsi_overbought': self.RSI_OVERBOUGHT,
                'vwap_long': self.VWAP_DEVIATION_LONG,
                'vwap_short': self.VWAP_DEVIATION_SHORT
            }
        })
        return base_status
