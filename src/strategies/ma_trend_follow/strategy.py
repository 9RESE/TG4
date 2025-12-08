"""
Phase 20: MA Trend Follow Strategy
9-period SMA trend following on 1h timeframe with 5min confirmation.

Rules:
- Uptrend: 2 closes above SMA9 on 1h → long entry
- Downtrend: 2 closes below SMA9 on 1h → short entry
- Exit: 1 candle closes opposite the trend
- Multi-TF: 1h for trend, 5min for entry confirmation
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from strategies.base_strategy import BaseStrategy
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional


class MATrendFollow(BaseStrategy):
    """
    9-period SMA trend following strategy.
    Captures momentum runs on BTC/XRP with quick reversal exits.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.ma_period = config.get('ma_period', 9)
        self.confirm_candles = config.get('confirm', 2)
        self.exit_opposite = config.get('exit_opposite', True)
        self.primary_tf = config.get('primary_tf', '1h')
        self.confirm_tf = config.get('confirm_tf', '5m')
        self.symbols = config.get('symbols', ['BTC/USDT', 'XRP/USDT'])

        # Track current trend state
        self.trend_state = {}  # {symbol: 'up', 'down', 'none'}
        self.entry_price = {}

    def _calculate_sma(self, df: pd.DataFrame, period: int = None) -> pd.Series:
        """Calculate Simple Moving Average."""
        if period is None:
            period = self.ma_period
        return df['close'].rolling(window=period).mean()

    def _detect_trend(self, df: pd.DataFrame) -> str:
        """
        Detect trend based on consecutive closes above/below SMA.

        Returns:
            str: 'up', 'down', or 'none'
        """
        if len(df) < self.ma_period + self.confirm_candles:
            return 'none'

        df = df.copy()
        df['sma'] = self._calculate_sma(df)

        # Check last N candles for trend confirmation
        recent = df.tail(self.confirm_candles)
        above_sma = (recent['close'] > recent['sma']).all()
        below_sma = (recent['close'] < recent['sma']).all()

        if above_sma:
            return 'up'
        elif below_sma:
            return 'down'
        return 'none'

    def _check_exit(self, df: pd.DataFrame, current_trend: str) -> bool:
        """
        Check if we should exit based on trend reversal.
        One candle opposite the trend triggers exit.
        """
        if not self.exit_opposite or len(df) < self.ma_period + 1:
            return False

        df = df.copy()
        df['sma'] = self._calculate_sma(df)
        latest = df.iloc[-1]

        if current_trend == 'up' and latest['close'] < latest['sma']:
            return True
        if current_trend == 'down' and latest['close'] > latest['sma']:
            return True

        return False

    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Generate trading signals based on MA trend.

        Args:
            data: Dict with symbol -> DataFrame (OHLCV)

        Returns:
            Signal dict with action, size, leverage, etc.
        """
        signals = []

        for symbol in self.symbols:
            if symbol not in data:
                continue

            df = data[symbol]
            if len(df) < self.ma_period + self.confirm_candles + 5:
                continue

            current_trend = self._detect_trend(df)
            prev_trend = self.trend_state.get(symbol, 'none')

            # Check for exit first
            if prev_trend in ['up', 'down'] and self._check_exit(df, prev_trend):
                self.trend_state[symbol] = 'none'
                signals.append({
                    'action': 'close',
                    'symbol': symbol,
                    'confidence': 0.8,
                    'reason': f'MA trend reversal exit ({prev_trend} -> opposite)',
                    'strategy': 'ma_trend'
                })
                continue

            # Check for new trend entry
            if current_trend == 'up' and prev_trend != 'up':
                self.trend_state[symbol] = 'up'
                self.entry_price[symbol] = df['close'].iloc[-1]
                signals.append({
                    'action': 'buy',
                    'symbol': symbol,
                    'size': self.position_size_pct,
                    'leverage': min(self.max_leverage, 5),
                    'confidence': 0.75,
                    'reason': f'MA uptrend confirmed ({self.confirm_candles} closes > SMA{self.ma_period})',
                    'strategy': 'ma_trend'
                })

            elif current_trend == 'down' and prev_trend != 'down':
                self.trend_state[symbol] = 'down'
                self.entry_price[symbol] = df['close'].iloc[-1]
                signals.append({
                    'action': 'short',
                    'symbol': symbol,
                    'size': self.position_size_pct * 0.8,  # Slightly smaller for shorts
                    'leverage': min(self.max_leverage, 5),
                    'confidence': 0.70,
                    'reason': f'MA downtrend confirmed ({self.confirm_candles} closes < SMA{self.ma_period})',
                    'strategy': 'ma_trend'
                })

        # Return best signal or hold
        if signals:
            # Prioritize by confidence
            best = max(signals, key=lambda x: x.get('confidence', 0))
            return best

        return {
            'action': 'hold',
            'symbol': self.symbols[0] if self.symbols else 'BTC/USDT',
            'confidence': 0.0,
            'reason': 'No MA trend signal',
            'strategy': 'ma_trend'
        }

    def update_model(self, data: Optional[Dict[str, pd.DataFrame]] = None) -> bool:
        """MA Trend is rule-based, no model to update."""
        return True

    def get_status(self) -> Dict[str, Any]:
        """Get strategy status."""
        base_status = super().get_status()
        base_status.update({
            'ma_period': self.ma_period,
            'confirm_candles': self.confirm_candles,
            'trend_states': self.trend_state.copy(),
            'entry_prices': self.entry_price.copy()
        })
        return base_status
