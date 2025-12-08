"""
Phase 21: MA Trend Follow Strategy - Momentum Breakout Tuned
9-period SMA trend following with fast 1-candle confirmation for breakouts.

Rules:
- Uptrend: 1 close above SMA9 on 1h → long entry (faster for breakouts)
- Downtrend: 1 close below SMA9 on 1h → short entry
- Exit: 1 candle closes opposite the trend
- Breakout boost: Volume spike + high break → higher leverage
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
        self.confirm_candles = config.get('confirm', 1)  # Phase 21: 1 candle for faster breakout response
        self.exit_opposite = config.get('exit_opposite', True)
        self.primary_tf = config.get('primary_tf', '1h')
        self.confirm_tf = config.get('confirm_tf', '5m')
        self.symbols = config.get('symbols', ['BTC/USDT', 'XRP/USDT'])

        # Phase 21: Breakout detection params
        self.breakout_vol_mult = config.get('breakout_vol_mult', 1.5)  # Volume 1.5x avg = breakout
        self.recent_high_lookback = config.get('high_lookback', 24)  # 24h high break

        # Track current trend state
        self.trend_state = {}  # {symbol: 'up', 'down', 'none'}
        self.entry_price = {}
        self.last_breakout = {}  # Phase 21: Track breakout strength

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

    def _detect_breakout(self, df: pd.DataFrame) -> dict:
        """
        Phase 21: Detect breakout conditions for leverage boost.
        Breakout = price breaks recent high + volume spike.

        Returns:
            dict: {'is_breakout': bool, 'strength': float, 'leverage_boost': int}
        """
        if len(df) < self.recent_high_lookback + 1:
            return {'is_breakout': False, 'strength': 0.0, 'leverage_boost': 0}

        current_price = df['close'].iloc[-1]
        current_vol = df['volume'].iloc[-1]

        # Recent high (24h lookback)
        recent_high = df['high'].iloc[-self.recent_high_lookback:-1].max()
        avg_volume = df['volume'].iloc[-self.recent_high_lookback:-1].mean()

        # Check breakout conditions
        price_break = current_price > recent_high
        volume_spike = current_vol > avg_volume * self.breakout_vol_mult

        if price_break and volume_spike:
            # Strong breakout: +3-4 leverage boost
            strength = (current_vol / avg_volume) * (current_price / recent_high - 1) * 100
            leverage_boost = 4 if strength > 0.5 else 3
            return {'is_breakout': True, 'strength': strength, 'leverage_boost': leverage_boost}
        elif price_break:
            # Weak breakout (no volume): +1-2 leverage
            return {'is_breakout': True, 'strength': 0.2, 'leverage_boost': 2}

        return {'is_breakout': False, 'strength': 0.0, 'leverage_boost': 0}

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

                # Phase 21: Check for breakout - dynamic leverage 3x → 5-7x
                breakout = self._detect_breakout(df)
                base_leverage = 3
                if breakout['is_breakout']:
                    leverage = min(self.max_leverage, base_leverage + breakout['leverage_boost'])
                    confidence = 0.80 + breakout['strength'] * 0.1  # Higher confidence on breakout
                    reason = f'MA uptrend BREAKOUT ({self.confirm_candles} closes > SMA{self.ma_period}, vol+{breakout["strength"]:.1f}x)'
                    self.last_breakout[symbol] = breakout
                else:
                    leverage = base_leverage
                    confidence = 0.75
                    reason = f'MA uptrend confirmed ({self.confirm_candles} closes > SMA{self.ma_period})'

                signals.append({
                    'action': 'buy',
                    'symbol': symbol,
                    'size': self.position_size_pct,
                    'leverage': leverage,
                    'confidence': min(confidence, 0.95),
                    'reason': reason,
                    'strategy': 'ma_trend',
                    'breakout': breakout['is_breakout']
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
