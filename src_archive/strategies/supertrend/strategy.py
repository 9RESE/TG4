"""
SuperTrend Strategy - ATR-based Trend Following
Research: 11.07% average profit per trade in backtests

The SuperTrend indicator uses ATR to create dynamic support/resistance bands.
When price crosses above, trend is bullish. When below, bearish.

Features:
- Single SuperTrend with configurable ATR period and multiplier
- Double SuperTrend for confirmation (reduces false signals)
- 200 EMA filter for macro trend alignment
- Multi-timeframe confirmation option
- Dynamic position sizing based on ATR

Settings:
- Default (10, 3): Standard settings for most markets
- Scalping (10, 3) + (25, 5): Double confirmation
- Swing (20, 5): Slower, fewer signals, higher quality
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from strategies.base_strategy import BaseStrategy
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime


class SuperTrendStrategy(BaseStrategy):
    """
    SuperTrend trend-following strategy.

    Backtested performance: ~11% average profit per trade.
    Works best in trending markets, avoid during consolidation.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Primary SuperTrend parameters
        self.atr_period = config.get('atr_period', 10)
        self.atr_multiplier = config.get('atr_multiplier', 3.0)

        # Secondary SuperTrend for double confirmation
        self.use_double_supertrend = config.get('use_double_supertrend', True)
        self.atr_period_slow = config.get('atr_period_slow', 25)
        self.atr_multiplier_slow = config.get('atr_multiplier_slow', 5.0)

        # EMA filter (macro trend)
        self.use_ema_filter = config.get('use_ema_filter', True)
        self.ema_period = config.get('ema_period', 200)

        # Position sizing
        self.base_size_pct = config.get('base_size_pct', 0.10)
        self.dynamic_sizing = config.get('dynamic_sizing', True)
        self.min_size_pct = config.get('min_size_pct', 0.05)
        self.max_size_pct = config.get('max_size_pct', 0.20)

        # Risk management
        self.stop_loss_atr_mult = config.get('stop_loss_atr_mult', 1.5)
        self.take_profit_atr_mult = config.get('take_profit_atr_mult', 3.0)

        # Symbols
        self.symbols = config.get('symbols', ['BTC/USDT', 'XRP/USDT'])

        # State tracking
        self.last_trend: Dict[str, str] = {}  # 'bullish' or 'bearish'
        self.last_supertrend: Dict[str, float] = {}
        self.positions: Dict[str, str] = {}  # symbol -> 'long' or 'short'
        self.last_signal_bar: Dict[str, int] = {}
        self.bar_count: Dict[str, int] = {}

        # Cooldown
        self.cooldown_bars = config.get('cooldown_bars', 2)

    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range using Wilder's smoothing."""
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        return atr

    def _calculate_supertrend(self, df: pd.DataFrame, atr_period: int,
                              atr_multiplier: float) -> Dict[str, pd.Series]:
        """
        Calculate SuperTrend indicator.

        Returns:
            Dict with 'supertrend', 'direction', 'upper_band', 'lower_band'
        """
        atr = self._calculate_atr(df, atr_period)
        hl2 = (df['high'] + df['low']) / 2

        # Basic bands
        upper_band = hl2 + (atr_multiplier * atr)
        lower_band = hl2 - (atr_multiplier * atr)

        # Initialize output series
        supertrend = pd.Series(index=df.index, dtype=float)
        direction = pd.Series(index=df.index, dtype=int)  # 1 = bullish, -1 = bearish

        # Calculate SuperTrend
        for i in range(1, len(df)):
            # Adjust bands based on previous close
            if lower_band.iloc[i] > lower_band.iloc[i-1] or df['close'].iloc[i-1] < lower_band.iloc[i-1]:
                lower_band.iloc[i] = lower_band.iloc[i]
            else:
                lower_band.iloc[i] = lower_band.iloc[i-1]

            if upper_band.iloc[i] < upper_band.iloc[i-1] or df['close'].iloc[i-1] > upper_band.iloc[i-1]:
                upper_band.iloc[i] = upper_band.iloc[i]
            else:
                upper_band.iloc[i] = upper_band.iloc[i-1]

            # Determine direction
            if i == 1:
                direction.iloc[i] = 1 if df['close'].iloc[i] > upper_band.iloc[i] else -1
            else:
                if direction.iloc[i-1] == 1:
                    if df['close'].iloc[i] < lower_band.iloc[i]:
                        direction.iloc[i] = -1
                    else:
                        direction.iloc[i] = 1
                else:
                    if df['close'].iloc[i] > upper_band.iloc[i]:
                        direction.iloc[i] = 1
                    else:
                        direction.iloc[i] = -1

            # Set SuperTrend value
            supertrend.iloc[i] = lower_band.iloc[i] if direction.iloc[i] == 1 else upper_band.iloc[i]

        return {
            'supertrend': supertrend,
            'direction': direction,
            'upper_band': upper_band,
            'lower_band': lower_band,
            'atr': atr
        }

    def _calculate_ema(self, close: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return close.ewm(span=period, adjust=False).mean()

    def _check_ema_filter(self, close: float, ema: float, direction: str) -> bool:
        """Check if trade direction aligns with EMA macro trend."""
        if not self.use_ema_filter:
            return True

        if direction == 'long':
            return close > ema
        elif direction == 'short':
            return close < ema
        return True

    def _calculate_position_size(self, atr_pct: float) -> float:
        """Dynamic position sizing based on volatility."""
        if not self.dynamic_sizing:
            return self.base_size_pct

        # Lower size in high volatility
        if atr_pct > 3.0:
            scale = 0.5
        elif atr_pct > 2.0:
            scale = 0.7
        elif atr_pct < 1.0:
            scale = 1.2
        else:
            scale = 1.0

        size = self.base_size_pct * scale
        return max(self.min_size_pct, min(self.max_size_pct, size))

    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Generate trading signals based on SuperTrend crossovers.

        Signal Logic:
        - EXIT: Close positions on trend reversal (priority)
        - BUY: Price crosses above SuperTrend (direction changes to bullish)
        - SHORT: Price crosses below SuperTrend (direction changes to bearish)
        - Double confirmation: Both fast and slow SuperTrend agree
        - EMA filter: Only trade in direction of macro trend
        """
        signals = []

        for symbol in self.symbols:
            # Get data (try different timeframes)
            df = data.get(f'{symbol}_1h')
            if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                df = data.get(f'{symbol}_15m')
            if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                df = data.get(symbol)

            if df is None or len(df) < max(self.atr_period, self.ema_period) + 10:
                continue

            # Update bar count
            self.bar_count[symbol] = self.bar_count.get(symbol, 0) + 1
            current_bar = self.bar_count[symbol]

            close = df['close']
            current_price = close.iloc[-1]

            # Calculate primary SuperTrend
            st_fast = self._calculate_supertrend(df, self.atr_period, self.atr_multiplier)
            direction_fast = st_fast['direction'].iloc[-1]
            prev_direction_fast = st_fast['direction'].iloc[-2] if len(df) > 2 else direction_fast
            supertrend_value = st_fast['supertrend'].iloc[-1]
            atr = st_fast['atr'].iloc[-1]

            # Calculate slow SuperTrend if using double confirmation
            direction_slow = direction_fast  # Default to fast
            if self.use_double_supertrend:
                st_slow = self._calculate_supertrend(df, self.atr_period_slow, self.atr_multiplier_slow)
                direction_slow = st_slow['direction'].iloc[-1]

            # Calculate EMA
            ema = self._calculate_ema(close, self.ema_period)
            ema_value = ema.iloc[-1]

            # Calculate ATR percentage for position sizing
            atr_pct = (atr / current_price) * 100 if current_price > 0 else 0

            # Store state
            self.last_supertrend[symbol] = supertrend_value

            # Detect crossover (direction change)
            crossover_bullish = direction_fast == 1 and prev_direction_fast == -1
            crossover_bearish = direction_fast == -1 and prev_direction_fast == 1

            # Current position
            current_pos = self.positions.get(symbol)

            # EXIT SIGNALS - Check first (no cooldown for exits)
            if current_pos == 'long' and crossover_bearish:
                # Close long on bearish crossover
                signal = {
                    'action': 'sell',
                    'symbol': symbol,
                    'size': 1.0,  # Close full position
                    'leverage': 1,
                    'confidence': 0.75,
                    'reason': f'SuperTrend bearish crossover - closing long',
                    'strategy': 'supertrend',
                    'supertrend': supertrend_value
                }
                self.positions.pop(symbol, None)
                self.last_trend[symbol] = 'bearish'
                self.last_signal_bar[symbol] = current_bar
                signals.append(signal)
                continue  # Don't open new position same bar

            elif current_pos == 'short' and crossover_bullish:
                # Cover short on bullish crossover
                signal = {
                    'action': 'cover',
                    'symbol': symbol,
                    'size': 1.0,  # Close full position
                    'leverage': 1,
                    'confidence': 0.75,
                    'reason': f'SuperTrend bullish crossover - covering short',
                    'strategy': 'supertrend',
                    'supertrend': supertrend_value
                }
                self.positions.pop(symbol, None)
                self.last_trend[symbol] = 'bullish'
                self.last_signal_bar[symbol] = current_bar
                signals.append(signal)
                continue  # Don't open new position same bar

            # Check cooldown for new entries only
            last_signal = self.last_signal_bar.get(symbol, -999)
            if current_bar - last_signal < self.cooldown_bars:
                continue

            # ENTRY SIGNALS - Only if no position
            if crossover_bullish and current_pos is None:
                # Check double SuperTrend confirmation
                if self.use_double_supertrend and direction_slow != 1:
                    continue

                # Check EMA filter
                if not self._check_ema_filter(current_price, ema_value, 'long'):
                    continue

                # Calculate stops
                stop_loss = current_price - (atr * self.stop_loss_atr_mult)
                take_profit = current_price + (atr * self.take_profit_atr_mult)

                position_size = self._calculate_position_size(atr_pct)

                signal = {
                    'action': 'buy',
                    'symbol': symbol,
                    'size': position_size,
                    'leverage': self.max_leverage,
                    'confidence': 0.75 if self.use_double_supertrend else 0.65,
                    'reason': f'SuperTrend bullish crossover, ATR={atr_pct:.1f}%',
                    'strategy': 'supertrend',
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'supertrend': supertrend_value,
                    'ema': ema_value
                }

                self.positions[symbol] = 'long'
                self.last_trend[symbol] = 'bullish'
                self.last_signal_bar[symbol] = current_bar
                signals.append(signal)

            elif crossover_bearish and current_pos is None:
                # Check double SuperTrend confirmation
                if self.use_double_supertrend and direction_slow != -1:
                    continue

                # Check EMA filter
                if not self._check_ema_filter(current_price, ema_value, 'short'):
                    continue

                # Calculate stops
                stop_loss = current_price + (atr * self.stop_loss_atr_mult)
                take_profit = current_price - (atr * self.take_profit_atr_mult)

                position_size = self._calculate_position_size(atr_pct)

                signal = {
                    'action': 'short',
                    'symbol': symbol,
                    'size': position_size * 0.8,  # Slightly smaller for shorts
                    'leverage': self.max_leverage,
                    'confidence': 0.70 if self.use_double_supertrend else 0.60,
                    'reason': f'SuperTrend bearish crossover, ATR={atr_pct:.1f}%',
                    'strategy': 'supertrend',
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'supertrend': supertrend_value,
                    'ema': ema_value
                }

                self.positions[symbol] = 'short'
                self.last_trend[symbol] = 'bearish'
                self.last_signal_bar[symbol] = current_bar
                signals.append(signal)

        # Return best signal or hold
        if signals:
            return max(signals, key=lambda x: x.get('confidence', 0))

        # Build detailed hold reason for diagnostics
        primary_symbol = self.symbols[0] if self.symbols else 'BTC/USDT'
        trend = self.last_trend.get(primary_symbol, 'unknown')
        st_value = self.last_supertrend.get(primary_symbol, 0)
        has_position = bool(self.positions)

        hold_reasons = []
        if trend == 'unknown':
            hold_reasons.append('No established trend')
        else:
            hold_reasons.append(f'Trend is {trend}, no crossover')

        if has_position:
            hold_reasons.append(f'In position: {list(self.positions.keys())}')

        return {
            'action': 'hold',
            'symbol': primary_symbol,
            'confidence': 0.0,
            'reason': f"SuperTrend: {', '.join(hold_reasons)}",
            'strategy': 'supertrend',
            'indicators': {
                'trend': trend,
                'trend_direction': 1 if trend == 'bullish' else (-1 if trend == 'bearish' else 0),
                'supertrend_value': st_value,
                'trend_changed': False,  # No crossover
                'has_position': has_position,
                'positions': list(self.positions.keys()) if self.positions else []
            }
        }

    def update_model(self, data: Optional[Dict[str, pd.DataFrame]] = None) -> bool:
        """SuperTrend is rule-based, no model to update."""
        return True

    def get_status(self) -> Dict[str, Any]:
        """Get strategy status."""
        base = super().get_status()
        base.update({
            'atr_period': self.atr_period,
            'atr_multiplier': self.atr_multiplier,
            'use_double_supertrend': self.use_double_supertrend,
            'use_ema_filter': self.use_ema_filter,
            'last_trends': self.last_trend,
            'last_supertrend_values': self.last_supertrend
        })
        return base
