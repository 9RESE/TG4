"""
Mean Reversion VWAP Strategy
Phase 16: Tuned for Dec 2025 XRP $2.00-2.20 chop

Trades mean reversion around VWAP using RSI as confirmation.
Optimized for XRP/USDT chop in $2.00-2.20 range (Dec 2025).

Phase 16 Tuning:
- Tighter dev_threshold: 0.003 (0.3%) for more trades
- Volume filter: >1.5x avg to avoid false signals
- Leverage: 5-7x on Kraken for amplified accumulation
- RSI thresholds: 32/68 (slightly relaxed for more opportunities)

Target: 5-15% monthly in choppy conditions.
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

    Phase 16 Tuning:
    - Tighter VWAP deviation threshold (0.3% vs 0.5%)
    - Volume confirmation filter (1.5x avg volume)
    - Relaxed RSI thresholds (32/68 vs 30/70)
    - Higher leverage (5-7x)

    Entry Conditions:
    - LONG: RSI < 32 AND price < VWAP * 0.997 AND volume > 1.5x avg
    - SHORT: RSI > 68 AND price > VWAP * 1.003 AND volume > 1.5x avg

    Exit Conditions:
    - Price reverts to VWAP (take profit)
    - Stop loss at 4% adverse move (tighter for leverage)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.name = 'mean_reversion_vwap'
        self.symbol = config.get('symbol', 'XRP/USDT')
        self.vwap_window = config.get('vwap_window', 14)
        self.rsi_window = config.get('rsi_window', 14)

        # Phase 16: Tighter threshold for more trades in tight range
        self.dev_threshold = config.get('dev_threshold', 0.003)  # 0.3% (was 0.5%)

        # Phase 16: Slightly relaxed RSI for more opportunities
        self.rsi_oversold = config.get('rsi_oversold', 32)  # was 30
        self.rsi_overbought = config.get('rsi_overbought', 68)  # was 70

        # Phase 16: Volume filter
        self.volume_filter = config.get('volume_filter', True)
        self.volume_mult = config.get('volume_mult', 1.5)  # 1.5x avg volume
        self.volume_window = config.get('volume_window', 20)

        # Phase 16: Higher leverage for amplified accumulation
        self.max_leverage = config.get('max_leverage', 7)  # was 5
        self.long_size = config.get('long_size', 0.12)
        self.short_size = config.get('short_size', 0.10)

        # Phase 16: Tighter stops for leveraged positions
        self.stop_loss_pct = config.get('stop_loss_pct', 0.04)  # 4% (was 5%)

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
            return self._calculate_rsi_fallback(df)

    def _calculate_rsi_fallback(self, df: pd.DataFrame) -> pd.Series:
        """Fallback RSI calculation without ta library."""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_window).mean()
        rs = gain / loss.replace(0, 0.001)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _check_volume_filter(self, df: pd.DataFrame) -> bool:
        """
        Phase 16: Check if current volume is above threshold.
        Helps avoid false signals in low-liquidity periods.
        """
        if not self.volume_filter:
            return True

        if len(df) < self.volume_window + 1:
            return True  # Not enough data, allow signal

        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].iloc[-self.volume_window-1:-1].mean()

        return current_volume > avg_volume * self.volume_mult

    def _get_volume_ratio(self, df: pd.DataFrame) -> float:
        """Get current volume as multiple of average."""
        if len(df) < self.volume_window + 1:
            return 1.0

        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].iloc[-self.volume_window-1:-1].mean()

        return current_volume / avg_volume if avg_volume > 0 else 1.0

    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Generate mean reversion signals based on VWAP, RSI, and volume.
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
        min_periods = max(self.vwap_window, self.rsi_window, self.volume_window) + 5

        if len(df) < min_periods:
            return {
                'action': 'hold',
                'symbol': self.symbol,
                'size': 0.0,
                'leverage': 1,
                'confidence': 0.0,
                'reason': f'Insufficient data ({len(df)} < {min_periods})'
            }

        # Calculate indicators
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

        # Calculate metrics
        vwap_deviation = (current_price - vwap) / vwap if vwap > 0 else 0
        volume_ratio = self._get_volume_ratio(df)
        volume_ok = self._check_volume_filter(df)

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
                'volume_ratio': volume_ratio,
                'volume_ok': volume_ok,
                'prev_rsi': prev['rsi'] if not pd.isna(prev['rsi']) else 50
            }
        }

        # Phase 16: Long signal with tuned thresholds
        # RSI < 32 AND price 0.3%+ below VWAP AND volume > 1.5x avg
        if (rsi < self.rsi_oversold and
            current_price < vwap * (1 - self.dev_threshold) and
            volume_ok):

            # Confidence based on extremity + volume
            rsi_score = (self.rsi_oversold - rsi) / self.rsi_oversold
            dev_score = min(abs(vwap_deviation) / self.dev_threshold, 2) / 2
            vol_score = min(volume_ratio / self.volume_mult, 2) / 4  # Volume bonus
            confidence = min(0.5 + rsi_score * 0.25 + dev_score * 0.15 + vol_score, 0.95)

            signal = {
                'action': 'buy',
                'symbol': self.symbol,
                'asset': 'XRP',
                'size': self.long_size,
                'leverage': self.max_leverage,
                'confidence': confidence,
                'reason': f'Long: RSI {rsi:.1f} < {self.rsi_oversold}, price {vwap_deviation*100:.2f}% below VWAP, vol {volume_ratio:.1f}x',
                'indicators': signal['indicators'],
                'stop_loss': current_price * (1 - self.stop_loss_pct),
                'take_profit': vwap
            }

        # Phase 16: Short signal with tuned thresholds
        # RSI > 68 AND price 0.3%+ above VWAP AND volume > 1.5x avg
        elif (rsi > self.rsi_overbought and
              current_price > vwap * (1 + self.dev_threshold) and
              volume_ok):

            rsi_score = (rsi - self.rsi_overbought) / (100 - self.rsi_overbought)
            dev_score = min(abs(vwap_deviation) / self.dev_threshold, 2) / 2
            vol_score = min(volume_ratio / self.volume_mult, 2) / 4
            confidence = min(0.5 + rsi_score * 0.25 + dev_score * 0.15 + vol_score, 0.95)

            signal = {
                'action': 'sell',
                'symbol': self.symbol,
                'asset': 'XRP',
                'size': self.short_size,
                'leverage': self.max_leverage,
                'confidence': confidence,
                'reason': f'Short: RSI {rsi:.1f} > {self.rsi_overbought}, price +{vwap_deviation*100:.2f}% above VWAP, vol {volume_ratio:.1f}x',
                'indicators': signal['indicators'],
                'stop_loss': current_price * (1 + self.stop_loss_pct),
                'take_profit': vwap
            }

        # Phase 16: Log near-miss signals for debugging
        elif not volume_ok and (rsi < self.rsi_oversold or rsi > self.rsi_overbought):
            signal['reason'] = f'Near signal blocked by volume filter (vol: {volume_ratio:.1f}x < {self.volume_mult}x)'

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
            'rsi_thresholds': (self.rsi_oversold, self.rsi_overbought),
            'volume_filter': self.volume_filter,
            'volume_mult': self.volume_mult,
            'stop_loss_pct': self.stop_loss_pct,
            'ta_available': TA_AVAILABLE
        })
        return base_status
