"""
Ichimoku Cloud Strategy - All-in-One Trend System
Research: BTC broke above cloud at $93K -> rallied to $120K+, XRP targets $6-$30

The Ichimoku Kinko Hyo provides trend, momentum, and support/resistance in one view.

Components:
- Tenkan-sen (Conversion Line): (9-period high + low) / 2
- Kijun-sen (Base Line): (26-period high + low) / 2
- Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2, plotted 26 periods ahead
- Senkou Span B (Leading Span B): (52-period high + low) / 2, plotted 26 periods ahead
- Chikou Span (Lagging Span): Close plotted 26 periods behind

Signals:
- Kumo Breakout: Price breaks above/below the cloud
- TK Cross: Tenkan crosses Kijun (like a fast MA cross)
- Chikou Confirmation: Lagging span confirms trend

Adapted Settings for Crypto:
- Scalping/Day: (6, 13, 26)
- Default: (9, 26, 52)
- Swing: (12, 24, 120)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from strategies.base_strategy import BaseStrategy
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime


class IchimokuCloud(BaseStrategy):
    """
    Ichimoku Cloud trading strategy.

    All-in-one indicator for trend, momentum, and S/R levels.
    Particularly effective for BTC and XRP trading.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Trading style presets
        style = config.get('style', 'default')  # 'scalping', 'default', 'swing'

        if style == 'scalping':
            default_tenkan, default_kijun, default_senkou = 6, 13, 26
        elif style == 'swing':
            default_tenkan, default_kijun, default_senkou = 12, 24, 120
        else:  # default
            default_tenkan, default_kijun, default_senkou = 9, 26, 52

        # Ichimoku parameters
        self.tenkan_period = config.get('tenkan_period', default_tenkan)
        self.kijun_period = config.get('kijun_period', default_kijun)
        self.senkou_b_period = config.get('senkou_b_period', default_senkou)
        self.displacement = config.get('displacement', 26)

        # Signal settings
        self.require_chikou_confirm = config.get('require_chikou_confirm', True)
        self.require_price_above_cloud = config.get('require_price_above_cloud', True)
        self.use_tk_cross = config.get('use_tk_cross', True)
        self.use_kumo_breakout = config.get('use_kumo_breakout', True)

        # Position sizing
        self.base_size_pct = config.get('base_size_pct', 0.12)
        self.min_size_pct = config.get('min_size_pct', 0.05)
        self.max_size_pct = config.get('max_size_pct', 0.25)

        # Risk management
        self.use_kijun_stop = config.get('use_kijun_stop', True)  # Use Kijun as stop

        # Symbols
        self.symbols = config.get('symbols', ['BTC/USDT', 'XRP/USDT'])

        # State tracking
        self.last_signal: Dict[str, str] = {}
        self.cloud_position: Dict[str, str] = {}  # 'above', 'below', 'inside'
        self.positions: Dict[str, str] = {}  # symbol -> 'long' or 'short'

    def _donchian_middle(self, high: pd.Series, low: pd.Series, period: int) -> pd.Series:
        """Calculate Donchian middle (used in Ichimoku calculations)."""
        return (high.rolling(window=period).max() + low.rolling(window=period).min()) / 2

    def _calculate_ichimoku(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate all Ichimoku components."""
        high = df['high']
        low = df['low']
        close = df['close']

        # Tenkan-sen (Conversion Line)
        tenkan = self._donchian_middle(high, low, self.tenkan_period)

        # Kijun-sen (Base Line)
        kijun = self._donchian_middle(high, low, self.kijun_period)

        # Senkou Span A (Leading Span A) - shifted forward
        senkou_a = ((tenkan + kijun) / 2).shift(self.displacement)

        # Senkou Span B (Leading Span B) - shifted forward
        senkou_b = self._donchian_middle(high, low, self.senkou_b_period).shift(self.displacement)

        # Chikou Span (Lagging Span) - close shifted backward
        chikou = close.shift(-self.displacement)

        # Cloud boundaries (for current position)
        cloud_top = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)
        cloud_bottom = pd.concat([senkou_a, senkou_b], axis=1).min(axis=1)

        # Future cloud (for projection)
        future_senkou_a = (tenkan + kijun) / 2
        future_senkou_b = self._donchian_middle(high, low, self.senkou_b_period)

        return {
            'tenkan': tenkan,
            'kijun': kijun,
            'senkou_a': senkou_a,
            'senkou_b': senkou_b,
            'chikou': chikou,
            'cloud_top': cloud_top,
            'cloud_bottom': cloud_bottom,
            'future_senkou_a': future_senkou_a,
            'future_senkou_b': future_senkou_b
        }

    def _get_cloud_position(self, close: float, cloud_top: float, cloud_bottom: float) -> str:
        """Determine price position relative to cloud."""
        if pd.isna(cloud_top) or pd.isna(cloud_bottom):
            return 'unknown'
        if close > cloud_top:
            return 'above'
        elif close < cloud_bottom:
            return 'below'
        return 'inside'

    def _check_tk_cross(self, tenkan: pd.Series, kijun: pd.Series) -> Optional[str]:
        """Check for Tenkan/Kijun crossover."""
        if len(tenkan) < 2 or len(kijun) < 2:
            return None

        current_tenkan = tenkan.iloc[-1]
        prev_tenkan = tenkan.iloc[-2]
        current_kijun = kijun.iloc[-1]
        prev_kijun = kijun.iloc[-2]

        if pd.isna(current_tenkan) or pd.isna(current_kijun):
            return None

        # Bullish cross: Tenkan crosses above Kijun
        if current_tenkan > current_kijun and prev_tenkan <= prev_kijun:
            return 'bullish'

        # Bearish cross: Tenkan crosses below Kijun
        if current_tenkan < current_kijun and prev_tenkan >= prev_kijun:
            return 'bearish'

        return None

    def _check_kumo_breakout(self, close: pd.Series, cloud_top: pd.Series,
                             cloud_bottom: pd.Series) -> Optional[str]:
        """Check for Kumo (cloud) breakout."""
        if len(close) < 2:
            return None

        current_close = close.iloc[-1]
        prev_close = close.iloc[-2]
        current_top = cloud_top.iloc[-1]
        current_bottom = cloud_bottom.iloc[-1]
        prev_top = cloud_top.iloc[-2]
        prev_bottom = cloud_bottom.iloc[-2]

        if pd.isna(current_top) or pd.isna(current_bottom):
            return None

        # Bullish breakout: price breaks above cloud
        if current_close > current_top and prev_close <= prev_top:
            return 'bullish'

        # Bearish breakout: price breaks below cloud
        if current_close < current_bottom and prev_close >= prev_bottom:
            return 'bearish'

        return None

    def _check_chikou_confirm(self, chikou: pd.Series, close: pd.Series) -> Optional[str]:
        """Check if Chikou Span confirms trend (above/below price 26 periods ago)."""
        if len(chikou) < self.displacement + 1:
            return None

        # Chikou is the close shifted back, so we compare current chikou
        # (which represents current close) to the close 26 periods ago
        current_close = close.iloc[-1]
        past_close = close.iloc[-self.displacement - 1] if len(close) > self.displacement else close.iloc[0]

        if current_close > past_close:
            return 'bullish'
        elif current_close < past_close:
            return 'bearish'

        return None

    def _get_cloud_strength(self, senkou_a: float, senkou_b: float) -> str:
        """Assess cloud strength based on Senkou relationship."""
        if pd.isna(senkou_a) or pd.isna(senkou_b):
            return 'unknown'

        diff = abs(senkou_a - senkou_b) / ((senkou_a + senkou_b) / 2) * 100

        if diff > 2:
            return 'strong'
        elif diff > 0.5:
            return 'moderate'
        return 'weak'

    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Generate Ichimoku-based trading signals.

        Signal Priority:
        1. Exit signals (TK reversal, stop loss, cloud re-entry)
        2. Kumo Breakout (strongest entry)
        3. TK Cross with cloud position confirmation
        4. Chikou confirmation adds confidence
        """
        signals = []

        # CHECK EXITS FIRST - Priority over new entries
        for symbol in list(self.positions.keys()):
            current_pos = self.positions[symbol]
            df = data.get(f'{symbol}_1h')
            if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                df = data.get(f'{symbol}_4h')
            if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                df = data.get(symbol)

            if df is None or len(df) < 30:
                continue

            close = df['close']
            current_price = close.iloc[-1]

            # Calculate Ichimoku for exit checks
            ichi = self._calculate_ichimoku(df)
            kijun = ichi['kijun'].iloc[-1]
            cloud_top = ichi['cloud_top'].iloc[-1]
            cloud_bottom = ichi['cloud_bottom'].iloc[-1]
            tk_cross = self._check_tk_cross(ichi['tenkan'], ichi['kijun'])

            should_exit = False
            exit_reason = ''

            if current_pos == 'long':
                # Exit on bearish TK cross
                if tk_cross == 'bearish':
                    should_exit = True
                    exit_reason = 'TK bearish cross'
                # Exit on Kijun stop (price breaks below Kijun)
                elif self.use_kijun_stop and current_price < kijun:
                    should_exit = True
                    exit_reason = f'Kijun stop ${kijun:.2f}'
                # Exit on cloud re-entry from above
                elif not pd.isna(cloud_top) and current_price < cloud_top and self.cloud_position.get(symbol) == 'above':
                    should_exit = True
                    exit_reason = 'Cloud re-entry'

                if should_exit:
                    self.positions.pop(symbol, None)
                    return {
                        'action': 'sell',
                        'symbol': symbol,
                        'size': 1.0,
                        'leverage': 1,
                        'confidence': 0.75,
                        'reason': f'Ichimoku exit LONG: {exit_reason}',
                        'strategy': 'ichimoku'
                    }

            elif current_pos == 'short':
                # Exit on bullish TK cross
                if tk_cross == 'bullish':
                    should_exit = True
                    exit_reason = 'TK bullish cross'
                # Exit on Kijun stop (price breaks above Kijun)
                elif self.use_kijun_stop and current_price > kijun:
                    should_exit = True
                    exit_reason = f'Kijun stop ${kijun:.2f}'
                # Exit on cloud re-entry from below
                elif not pd.isna(cloud_bottom) and current_price > cloud_bottom and self.cloud_position.get(symbol) == 'below':
                    should_exit = True
                    exit_reason = 'Cloud re-entry'

                if should_exit:
                    self.positions.pop(symbol, None)
                    return {
                        'action': 'cover',
                        'symbol': symbol,
                        'size': 1.0,
                        'leverage': 1,
                        'confidence': 0.75,
                        'reason': f'Ichimoku exit SHORT: {exit_reason}',
                        'strategy': 'ichimoku'
                    }

        for symbol in self.symbols:
            # Skip if already in a position for this symbol
            if symbol in self.positions:
                continue

            df = data.get(f'{symbol}_1h')
            if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                df = data.get(f'{symbol}_4h')
            if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                df = data.get(symbol)

            min_bars = max(self.senkou_b_period, self.displacement) + 10
            if df is None or len(df) < min_bars:
                continue

            close = df['close']
            current_price = close.iloc[-1]

            # Calculate Ichimoku
            ichi = self._calculate_ichimoku(df)

            # Get current values
            tenkan = ichi['tenkan'].iloc[-1]
            kijun = ichi['kijun'].iloc[-1]
            cloud_top = ichi['cloud_top'].iloc[-1]
            cloud_bottom = ichi['cloud_bottom'].iloc[-1]
            senkou_a = ichi['future_senkou_a'].iloc[-1]
            senkou_b = ichi['future_senkou_b'].iloc[-1]

            # Determine cloud position
            cloud_pos = self._get_cloud_position(current_price, cloud_top, cloud_bottom)
            self.cloud_position[symbol] = cloud_pos

            # Check signals
            tk_cross = self._check_tk_cross(ichi['tenkan'], ichi['kijun'])
            kumo_breakout = self._check_kumo_breakout(close, ichi['cloud_top'], ichi['cloud_bottom'])
            chikou_confirm = self._check_chikou_confirm(ichi['chikou'], close)
            cloud_strength = self._get_cloud_strength(senkou_a, senkou_b)

            # Build signal
            signal_direction = None
            confidence = 0.0
            reason_parts = []

            # Priority 1: Kumo Breakout
            if self.use_kumo_breakout and kumo_breakout:
                signal_direction = 'long' if kumo_breakout == 'bullish' else 'short'
                confidence = 0.75
                reason_parts.append(f'Kumo breakout {kumo_breakout}')

            # Priority 2: TK Cross with cloud confirmation
            elif self.use_tk_cross and tk_cross:
                if tk_cross == 'bullish' and cloud_pos != 'below':
                    signal_direction = 'long'
                    confidence = 0.65
                    reason_parts.append(f'TK cross bullish')
                elif tk_cross == 'bearish' and cloud_pos != 'above':
                    signal_direction = 'short'
                    confidence = 0.60
                    reason_parts.append(f'TK cross bearish')

            if signal_direction is None:
                continue

            # Chikou confirmation boost
            if chikou_confirm:
                if (signal_direction == 'long' and chikou_confirm == 'bullish') or \
                   (signal_direction == 'short' and chikou_confirm == 'bearish'):
                    confidence += 0.10
                    reason_parts.append('Chikou confirms')
                elif self.require_chikou_confirm:
                    continue  # Skip if chikou doesn't confirm

            # Cloud position boost
            if signal_direction == 'long' and cloud_pos == 'above':
                confidence += 0.05
                reason_parts.append('Above cloud')
            elif signal_direction == 'short' and cloud_pos == 'below':
                confidence += 0.05
                reason_parts.append('Below cloud')

            # Cloud strength factor
            if cloud_strength == 'strong':
                confidence += 0.05
            elif cloud_strength == 'weak':
                confidence -= 0.05

            # Calculate stop loss (Kijun-based)
            stop_loss = kijun if self.use_kijun_stop else None

            signal = {
                'action': 'buy' if signal_direction == 'long' else 'short',
                'symbol': symbol,
                'size': self.base_size_pct,
                'leverage': self.max_leverage,
                'confidence': min(confidence, 0.92),
                'reason': f"Ichimoku: {', '.join(reason_parts)}",
                'strategy': 'ichimoku',
                'stop_loss': stop_loss,
                'cloud_position': cloud_pos,
                'cloud_strength': cloud_strength,
                'tenkan': tenkan,
                'kijun': kijun
            }

            # Track position for exit logic
            self.positions[symbol] = signal_direction
            self.last_signal[symbol] = signal_direction
            signals.append(signal)

        if signals:
            return max(signals, key=lambda x: x.get('confidence', 0))

        # Build detailed hold reason for diagnostics
        primary_symbol = self.symbols[0] if self.symbols else 'BTC/USDT'
        cloud_pos = self.cloud_position.get(primary_symbol, 'unknown')
        has_position = bool(self.positions)

        hold_reasons = []
        if cloud_pos == 'inside':
            hold_reasons.append('Price inside cloud (Kumo)')
        elif cloud_pos == 'unknown':
            hold_reasons.append('Cloud position unknown')
        else:
            hold_reasons.append(f'No TK cross or Kumo breakout')

        if has_position:
            hold_reasons.append(f'In position: {list(self.positions.keys())}')

        return {
            'action': 'hold',
            'symbol': primary_symbol,
            'confidence': 0.0,
            'reason': f"Ichimoku: {', '.join(hold_reasons)}, cloud={cloud_pos}",
            'strategy': 'ichimoku',
            'indicators': {
                'cloud_position': cloud_pos,
                'in_cloud': cloud_pos == 'inside',
                'price_vs_cloud': 1 if cloud_pos == 'above' else (-1 if cloud_pos == 'below' else 0),
                'has_position': has_position,
                'positions': list(self.positions.keys()) if self.positions else [],
                'tk_cross': False,  # No cross detected
                'kumo_breakout': False
            }
        }

    def update_model(self, data: Optional[Dict[str, pd.DataFrame]] = None) -> bool:
        """Rule-based strategy, no model to update."""
        return True

    def get_status(self) -> Dict[str, Any]:
        """Get strategy status."""
        base = super().get_status()
        base.update({
            'tenkan_period': self.tenkan_period,
            'kijun_period': self.kijun_period,
            'senkou_b_period': self.senkou_b_period,
            'cloud_positions': self.cloud_position,
            'last_signals': self.last_signal
        })
        return base
