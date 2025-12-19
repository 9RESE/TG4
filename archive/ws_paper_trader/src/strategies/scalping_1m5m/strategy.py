"""
1-Minute and 5-Minute Scalping Strategies
Research: SMA crossover on 1m with 200 EMA filter achieves consistent small profits

Multiple scalping techniques:
1. SMA Crossover (5/12): Fast/slow MA cross
2. Previous Day High/Low: Liquidity zone bounces
3. Price Action: CHOCH (Change of Character) detection

Best Practices:
- BTC/ETH for 1-2 min (high liquidity, tight spreads)
- ADA/MATIC for 5-10 min (moderate volatility)
- Always use 200 EMA as macro filter
- Automation required (manual too slow)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from strategies.base_strategy import BaseStrategy
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta


class Scalping1m5m(BaseStrategy):
    """
    Ultra-short timeframe scalping strategy.

    Combines multiple scalping techniques for 1m and 5m charts.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Timeframe preference
        self.primary_timeframe = config.get('primary_timeframe', '5m')  # '1m' or '5m'

        # SMA Crossover parameters
        self.sma_fast = config.get('sma_fast', 5)
        self.sma_slow = config.get('sma_slow', 12)

        # EMA filter (macro trend)
        self.ema_period = config.get('ema_period', 200)
        self.use_ema_filter = config.get('use_ema_filter', True)

        # Previous day high/low
        self.use_prev_day_levels = config.get('use_prev_day_levels', True)
        self.level_tolerance = config.get('level_tolerance', 0.002)  # 0.2%

        # Change of Character (CHOCH) detection
        self.use_choch = config.get('use_choch', True)
        self.choch_lookback = config.get('choch_lookback', 10)

        # Risk management
        self.scalp_target_pct = config.get('scalp_target_pct', 0.005)  # 0.5%
        self.scalp_stop_pct = config.get('scalp_stop_pct', 0.003)  # 0.3%
        self.max_positions = config.get('max_positions', 2)

        # Position sizing
        self.base_size_pct = config.get('base_size_pct', 0.05)  # Small for scalping

        # Cooldown
        self.cooldown_seconds = config.get('cooldown_seconds', 60)

        # Symbols
        self.symbols = config.get('symbols', ['BTC/USDT', 'XRP/USDT'])

        # State
        self.active_scalps: Dict[str, Dict] = {}
        self.last_trade_time: Dict[str, datetime] = {}
        self.prev_day_levels: Dict[str, Dict] = {}

    def _calculate_sma(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average."""
        return series.rolling(window=period).mean()

    def _calculate_ema(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate EMA."""
        return series.ewm(span=period, adjust=False).mean()

    def _get_prev_day_levels(self, df: pd.DataFrame, symbol: str) -> Dict[str, float]:
        """Calculate previous day's high and low."""
        if len(df) < 288:  # Need at least ~1 day of 5m data
            return {}

        # Find previous day boundary
        now = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else datetime.now()

        # Simple approach: use data from 288-576 bars ago for "yesterday" on 5m
        if self.primary_timeframe == '5m':
            prev_day_data = df.iloc[-576:-288] if len(df) > 576 else df.iloc[:-288]
        else:  # 1m
            prev_day_data = df.iloc[-2880:-1440] if len(df) > 2880 else df.iloc[:-1440]

        if len(prev_day_data) < 10:
            return {}

        return {
            'high': prev_day_data['high'].max(),
            'low': prev_day_data['low'].min()
        }

    def _check_sma_cross(self, sma_fast: pd.Series, sma_slow: pd.Series) -> Optional[str]:
        """Check for SMA crossover."""
        if len(sma_fast) < 2:
            return None

        current_fast = sma_fast.iloc[-1]
        prev_fast = sma_fast.iloc[-2]
        current_slow = sma_slow.iloc[-1]
        prev_slow = sma_slow.iloc[-2]

        if pd.isna(current_fast) or pd.isna(current_slow):
            return None

        if current_fast > current_slow and prev_fast <= prev_slow:
            return 'bullish'
        elif current_fast < current_slow and prev_fast >= prev_slow:
            return 'bearish'

        return None

    def _check_level_test(self, close: float, levels: Dict[str, float]) -> Optional[str]:
        """Check if price is testing previous day levels."""
        if not levels:
            return None

        high = levels.get('high')
        low = levels.get('low')

        if high and abs(close - high) / high < self.level_tolerance:
            return 'at_prev_high'  # Potential short
        if low and abs(close - low) / low < self.level_tolerance:
            return 'at_prev_low'  # Potential long

        return None

    def _detect_choch(self, df: pd.DataFrame) -> Optional[str]:
        """
        Detect Change of Character (CHOCH).

        CHOCH occurs when:
        - After higher highs, price makes lower low (bearish CHOCH)
        - After lower lows, price makes higher high (bullish CHOCH)
        """
        if not self.use_choch or len(df) < self.choch_lookback + 5:
            return None

        highs = df['high'].iloc[-self.choch_lookback:]
        lows = df['low'].iloc[-self.choch_lookback:]

        # Recent swing structure
        recent_highs = [highs.iloc[i] for i in range(len(highs)-1) if highs.iloc[i] > highs.iloc[i-1]]
        recent_lows = [lows.iloc[i] for i in range(len(lows)-1) if lows.iloc[i] < lows.iloc[i-1]]

        current_high = df['high'].iloc[-1]
        current_low = df['low'].iloc[-1]
        prev_high = df['high'].iloc[-2]
        prev_low = df['low'].iloc[-2]

        # Bullish CHOCH: After lower lows, make higher high
        if len(recent_lows) >= 2 and current_high > max(highs.iloc[-5:-1]):
            return 'bullish'

        # Bearish CHOCH: After higher highs, make lower low
        if len(recent_highs) >= 2 and current_low < min(lows.iloc[-5:-1]):
            return 'bearish'

        return None

    def _check_cooldown(self, symbol: str) -> bool:
        """Check if cooldown period has passed."""
        last_time = self.last_trade_time.get(symbol)
        if last_time is None:
            return True
        return datetime.now() - last_time >= timedelta(seconds=self.cooldown_seconds)

    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Generate scalping signals from multiple techniques.

        Combines:
        1. Exit signals (target/stop/reversal)
        2. SMA crossover
        3. Previous day level tests
        4. CHOCH detection
        """
        signals = []

        # CHECK EXITS FIRST
        for symbol in list(self.active_scalps.keys()):
            scalp = self.active_scalps[symbol]
            df = data.get(f'{symbol}_{self.primary_timeframe}')
            if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                df = data.get(symbol)

            if df is None or len(df) < 2:
                continue

            current_price = df['close'].iloc[-1]
            direction = scalp['direction']
            target = scalp['target']
            stop = scalp['stop']

            # Check for exit conditions
            should_exit = False
            exit_reason = ''

            if direction == 'long':
                if current_price >= target:
                    should_exit = True
                    exit_reason = f'Target hit ${target:.2f}'
                elif current_price <= stop:
                    should_exit = True
                    exit_reason = f'Stop hit ${stop:.2f}'

                if should_exit:
                    self.active_scalps.pop(symbol, None)
                    return {
                        'action': 'sell',
                        'symbol': symbol,
                        'size': 1.0,
                        'leverage': 1,
                        'confidence': 0.80,
                        'reason': f'Scalp exit LONG: {exit_reason}',
                        'strategy': 'scalping_1m5m'
                    }

            elif direction == 'short':
                if current_price <= target:
                    should_exit = True
                    exit_reason = f'Target hit ${target:.2f}'
                elif current_price >= stop:
                    should_exit = True
                    exit_reason = f'Stop hit ${stop:.2f}'

                if should_exit:
                    self.active_scalps.pop(symbol, None)
                    return {
                        'action': 'cover',
                        'symbol': symbol,
                        'size': 1.0,
                        'leverage': 1,
                        'confidence': 0.80,
                        'reason': f'Scalp exit SHORT: {exit_reason}',
                        'strategy': 'scalping_1m5m'
                    }

        # NEW ENTRY SIGNALS
        for symbol in self.symbols:
            # Skip if already in a scalp for this symbol
            if symbol in self.active_scalps:
                continue

            # Check cooldown
            if not self._check_cooldown(symbol):
                continue

            # Check max positions
            if len(self.active_scalps) >= self.max_positions:
                continue

            # Get appropriate timeframe data
            tf = self.primary_timeframe
            df = data.get(f'{symbol}_{tf}')
            if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                df = data.get(symbol)

            min_bars = max(self.sma_slow, self.ema_period, self.choch_lookback) + 10
            if df is None or len(df) < min_bars:
                continue

            close = df['close']
            current_price = close.iloc[-1]

            # Calculate indicators
            sma_fast = self._calculate_sma(close, self.sma_fast)
            sma_slow = self._calculate_sma(close, self.sma_slow)
            ema = self._calculate_ema(close, self.ema_period)

            # EMA filter
            above_ema = current_price > ema.iloc[-1] if not pd.isna(ema.iloc[-1]) else True
            below_ema = current_price < ema.iloc[-1] if not pd.isna(ema.iloc[-1]) else True

            # Get signals from each technique
            sma_cross = self._check_sma_cross(sma_fast, sma_slow)

            # Previous day levels
            if self.use_prev_day_levels:
                self.prev_day_levels[symbol] = self._get_prev_day_levels(df, symbol)
            level_test = self._check_level_test(current_price, self.prev_day_levels.get(symbol, {}))

            # CHOCH
            choch = self._detect_choch(df)

            # Combine signals
            signal = None
            confidence = 0.0
            reason_parts = []

            # Priority 1: SMA cross + EMA alignment
            if sma_cross == 'bullish' and (not self.use_ema_filter or above_ema):
                confidence = 0.65
                reason_parts.append('SMA cross bullish')

                # Bonus for level test
                if level_test == 'at_prev_low':
                    confidence += 0.10
                    reason_parts.append('at prev day low')

                # Bonus for CHOCH confirmation
                if choch == 'bullish':
                    confidence += 0.08
                    reason_parts.append('CHOCH bullish')

                signal = {
                    'action': 'buy',
                    'symbol': symbol,
                    'size': self.base_size_pct,
                    'leverage': min(self.max_leverage, 5),
                    'confidence': min(confidence, 0.88),
                    'reason': f"Scalp LONG: {', '.join(reason_parts)}",
                    'strategy': 'scalping_1m5m',
                    'target': current_price * (1 + self.scalp_target_pct),
                    'stop': current_price * (1 - self.scalp_stop_pct),
                    'timeframe': tf
                }

            elif sma_cross == 'bearish' and (not self.use_ema_filter or below_ema):
                confidence = 0.60
                reason_parts.append('SMA cross bearish')

                if level_test == 'at_prev_high':
                    confidence += 0.10
                    reason_parts.append('at prev day high')

                if choch == 'bearish':
                    confidence += 0.08
                    reason_parts.append('CHOCH bearish')

                signal = {
                    'action': 'short',
                    'symbol': symbol,
                    'size': self.base_size_pct * 0.8,
                    'leverage': min(self.max_leverage, 5),
                    'confidence': min(confidence, 0.82),
                    'reason': f"Scalp SHORT: {', '.join(reason_parts)}",
                    'strategy': 'scalping_1m5m',
                    'target': current_price * (1 - self.scalp_target_pct),
                    'stop': current_price * (1 + self.scalp_stop_pct),
                    'timeframe': tf
                }

            # Priority 2: Level test without SMA cross (needs CHOCH)
            elif level_test == 'at_prev_low' and choch == 'bullish':
                signal = {
                    'action': 'buy',
                    'symbol': symbol,
                    'size': self.base_size_pct * 0.7,
                    'leverage': min(self.max_leverage, 3),
                    'confidence': 0.62,
                    'reason': f"Scalp LONG: Prev day low bounce + CHOCH",
                    'strategy': 'scalping_1m5m',
                    'target': current_price * (1 + self.scalp_target_pct),
                    'stop': current_price * (1 - self.scalp_stop_pct),
                    'timeframe': tf
                }

            elif level_test == 'at_prev_high' and choch == 'bearish':
                signal = {
                    'action': 'short',
                    'symbol': symbol,
                    'size': self.base_size_pct * 0.6,
                    'leverage': min(self.max_leverage, 3),
                    'confidence': 0.58,
                    'reason': f"Scalp SHORT: Prev day high rejection + CHOCH",
                    'strategy': 'scalping_1m5m',
                    'target': current_price * (1 - self.scalp_target_pct),
                    'stop': current_price * (1 + self.scalp_stop_pct),
                    'timeframe': tf
                }

            if signal:
                signals.append(signal)

        if signals:
            best = max(signals, key=lambda x: x.get('confidence', 0))

            # Get entry price safely
            entry_price = 0.0
            symbol_df = data.get(best['symbol'])
            if symbol_df is not None and isinstance(symbol_df, pd.DataFrame) and not symbol_df.empty:
                if 'close' in symbol_df.columns and len(symbol_df) > 0:
                    entry_price = symbol_df['close'].iloc[-1]

            # Track the scalp
            self.active_scalps[best['symbol']] = {
                'direction': 'long' if best['action'] == 'buy' else 'short',
                'entry': entry_price,
                'target': best['target'],
                'stop': best['stop'],
                'timestamp': datetime.now()
            }
            self.last_trade_time[best['symbol']] = datetime.now()

            return best

        return {
            'action': 'hold',
            'symbol': self.symbols[0] if self.symbols else 'BTC/USDT',
            'confidence': 0.0,
            'reason': 'No scalp setup',
            'strategy': 'scalping_1m5m',
            'active_scalps': len(self.active_scalps)
        }

    def update_model(self, data: Optional[Dict[str, pd.DataFrame]] = None) -> bool:
        """Rule-based strategy, no model to update."""
        return True

    def get_status(self) -> Dict[str, Any]:
        """Get strategy status."""
        base = super().get_status()
        base.update({
            'primary_timeframe': self.primary_timeframe,
            'sma_periods': (self.sma_fast, self.sma_slow),
            'active_scalps': len(self.active_scalps),
            'prev_day_levels': self.prev_day_levels
        })
        return base
