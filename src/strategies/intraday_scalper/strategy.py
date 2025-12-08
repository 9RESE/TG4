"""
Phase 21: IntraDay Scalper Strategy - Volatility Harvester
5-15min Bollinger Band squeezes + RSI extremes for quick BTC/XRP entries/exits.

Rules:
- Activates only when daily vol >3% (ATR filter)
- Oversold: close < BB lower + RSI < 30 → buy scalp to upper band
- Overbought: close > BB upper + RSI > 70 → sell/short scalp
- Quick exits: 0.5-1% targets, tight stops
- Max 3x leverage for controlled risk
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from strategies.base_strategy import BaseStrategy
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional


class IntraDayScalper(BaseStrategy):
    """
    Volatility harvester for intra-day swings.
    Captures Bollinger Band squeezes during high volatility periods.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.bb_period = config.get('bb_period', 20)
        self.bb_std = config.get('bb_std', 2.0)
        self.rsi_period = config.get('rsi_period', 14)
        self.vol_threshold = config.get('daily_vol_pct', 3.0)  # ATR >3% activates
        self.rsi_oversold = config.get('rsi_oversold', 30)
        self.rsi_overbought = config.get('rsi_overbought', 70)
        self.scalp_target_pct = config.get('scalp_target_pct', 0.01)  # 1% quick target
        self.scalp_stop_pct = config.get('scalp_stop_pct', 0.005)  # 0.5% tight stop
        self.symbols = config.get('symbols', ['BTC/USDT', 'XRP/USDT'])

        # Track scalp state
        self.active_scalps = {}
        self.last_atr_pct = 0.0
        self.last_rsi = {}
        self.last_bb_position = {}  # 'above', 'below', 'middle'

    def _calculate_bollinger_bands(self, close: pd.Series) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = close.rolling(window=self.bb_period).mean()
        std = close.rolling(window=self.bb_period).std()
        upper = sma + (self.bb_std * std)
        lower = sma - (self.bb_std * std)
        return {'sma': sma, 'upper': upper, 'lower': lower}

    def _calculate_rsi(self, close: pd.Series, period: int = None) -> pd.Series:
        """Calculate RSI indicator."""
        if period is None:
            period = self.rsi_period

        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_atr_pct(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR as percentage of price."""
        if len(df) < period + 1:
            return 0.0

        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        tr_list = []
        for i in range(1, min(period + 1, len(close))):
            tr = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
            tr_list.append(tr)

        atr = np.mean(tr_list) if tr_list else 0
        current_price = close[-1] if len(close) > 0 else 1
        return (atr / current_price * 100) if current_price > 0 else 0.0

    def _detect_squeeze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect Bollinger Band squeeze setup.

        Returns:
            dict: {'is_squeeze': bool, 'direction': 'long'/'short'/None, 'strength': float}
        """
        if len(df) < self.bb_period + 5:
            return {'is_squeeze': False, 'direction': None, 'strength': 0.0}

        close = df['close']
        bb = self._calculate_bollinger_bands(close)
        rsi = self._calculate_rsi(close)

        latest_close = close.iloc[-1]
        latest_upper = bb['upper'].iloc[-1]
        latest_lower = bb['lower'].iloc[-1]
        latest_rsi = rsi.iloc[-1]

        if pd.isna(latest_upper) or pd.isna(latest_lower) or pd.isna(latest_rsi):
            return {'is_squeeze': False, 'direction': None, 'strength': 0.0}

        # Band width (squeeze indicator)
        band_width = (latest_upper - latest_lower) / bb['sma'].iloc[-1]
        avg_width = ((bb['upper'] - bb['lower']) / bb['sma']).rolling(20).mean().iloc[-1]
        is_tight = band_width < avg_width * 0.8  # Bands tightening

        # Oversold squeeze: price at lower band + RSI oversold
        if latest_close <= latest_lower and latest_rsi < self.rsi_oversold:
            strength = (self.rsi_oversold - latest_rsi) / self.rsi_oversold
            return {
                'is_squeeze': True,
                'direction': 'long',
                'strength': min(strength + 0.3, 1.0),
                'entry': latest_close,
                'target': latest_upper,
                'stop': latest_close * (1 - self.scalp_stop_pct)
            }

        # Overbought squeeze: price at upper band + RSI overbought
        if latest_close >= latest_upper and latest_rsi > self.rsi_overbought:
            strength = (latest_rsi - self.rsi_overbought) / (100 - self.rsi_overbought)
            return {
                'is_squeeze': True,
                'direction': 'short',
                'strength': min(strength + 0.3, 1.0),
                'entry': latest_close,
                'target': latest_lower,
                'stop': latest_close * (1 + self.scalp_stop_pct)
            }

        return {'is_squeeze': False, 'direction': None, 'strength': 0.0}

    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Generate scalping signals based on BB squeeze + RSI extremes.

        Args:
            data: Dict with symbol -> DataFrame (OHLCV, preferably 5m or 15m)

        Returns:
            Signal dict with action, size, leverage, targets, etc.
        """
        signals = []

        for symbol in self.symbols:
            # Try 5m data first, fall back to 1h
            df = data.get(f'{symbol}_5m') or data.get(symbol)
            if df is None or len(df) < self.bb_period + 10:
                continue

            # Check volatility filter
            atr_pct = self._calculate_atr_pct(df)
            self.last_atr_pct = atr_pct

            if atr_pct < self.vol_threshold:
                # Low volatility - scalper inactive
                continue

            # Detect squeeze setup
            squeeze = self._detect_squeeze(df)

            if squeeze['is_squeeze']:
                rsi = self._calculate_rsi(df['close']).iloc[-1]
                self.last_rsi[symbol] = rsi

                if squeeze['direction'] == 'long':
                    confidence = 0.65 + squeeze['strength'] * 0.25
                    signals.append({
                        'action': 'buy',
                        'symbol': symbol,
                        'size': self.position_size_pct * 0.8,  # Smaller for scalps
                        'leverage': min(self.max_leverage, 3),  # Max 3x for scalps
                        'confidence': min(confidence, 0.90),
                        'reason': f'BB squeeze LONG: RSI={rsi:.0f}, ATR={atr_pct:.1f}%',
                        'strategy': 'scalper',
                        'target': squeeze.get('target'),
                        'stop': squeeze.get('stop'),
                        'scalp': True
                    })

                elif squeeze['direction'] == 'short':
                    confidence = 0.60 + squeeze['strength'] * 0.25
                    signals.append({
                        'action': 'short',
                        'symbol': symbol,
                        'size': self.position_size_pct * 0.6,
                        'leverage': min(self.max_leverage, 3),
                        'confidence': min(confidence, 0.85),
                        'reason': f'BB squeeze SHORT: RSI={rsi:.0f}, ATR={atr_pct:.1f}%',
                        'strategy': 'scalper',
                        'target': squeeze.get('target'),
                        'stop': squeeze.get('stop'),
                        'scalp': True
                    })

        # Return best signal or hold
        if signals:
            best = max(signals, key=lambda x: x.get('confidence', 0))
            return best

        return {
            'action': 'hold',
            'symbol': self.symbols[0] if self.symbols else 'BTC/USDT',
            'confidence': 0.0,
            'reason': f'No scalp signal (ATR={self.last_atr_pct:.1f}%, threshold={self.vol_threshold}%)',
            'strategy': 'scalper'
        }

    def update_model(self, data: Optional[Dict[str, pd.DataFrame]] = None) -> bool:
        """Scalper is rule-based, no model to update."""
        return True

    def get_status(self) -> Dict[str, Any]:
        """Get strategy status."""
        base_status = super().get_status()
        base_status.update({
            'bb_period': self.bb_period,
            'rsi_period': self.rsi_period,
            'vol_threshold': self.vol_threshold,
            'last_atr_pct': self.last_atr_pct,
            'last_rsi': self.last_rsi.copy(),
            'active_scalps': len(self.active_scalps)
        })
        return base_status
