"""
Phase 26: MA Trend Follow Strategy - Enhanced with ADX, EMA, ATR Stops
EMA-based trend following with comprehensive filters to reduce whipsaws.

Improvements from Phase 21:
- Switched from SMA to EMA for better responsiveness
- Added ADX filter to avoid ranging markets
- Added ATR-based dynamic stop loss
- Added volume confirmation
- Added RSI divergence detection
- Added MA ribbon for trend confirmation
- Added cooldown period to prevent overtrading
- Improved confidence scaling based on multiple factors

Rules:
- Uptrend: Price crosses above EMA + ADX > threshold + Volume confirmed
- Downtrend: Price crosses below EMA + ADX > threshold + Volume confirmed
- Exit: ATR-based stop loss OR trend reversal
- Breakout boost: Volume spike + high break → higher leverage
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from strategies.base_strategy import BaseStrategy
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta


class MATrendFollow(BaseStrategy):
    """
    Enhanced EMA-based trend following strategy with comprehensive filters.

    Key Features:
    - EMA instead of SMA for faster response
    - ADX filter to trade only in trending markets
    - ATR-based stop loss for risk management
    - Volume confirmation to filter weak signals
    - RSI divergence detection for early reversal warnings
    - MA ribbon for trend strength confirmation
    - Cooldown period to prevent overtrading
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # MA Settings
        self.ma_type = config.get('ma_type', 'ema')  # 'ema' or 'sma'
        self.ma_period = config.get('ma_period', 12)  # Increased from 9
        self.confirm_candles = config.get('confirmation_candles', 2)  # Increased from 1
        self.min_distance_pct = config.get('min_distance_pct', 0.003)  # 0.3% min distance
        self.exit_opposite = config.get('exit_opposite', True)

        # Timeframes
        self.primary_tf = config.get('primary_tf', '1h')
        self.confirm_tf = config.get('confirm_tf', '5m')
        self.symbols = config.get('symbols', ['BTC/USDT', 'XRP/USDT'])

        # ADX Filter Settings
        self.use_adx_filter = config.get('use_adx_filter', True)
        self.adx_period = config.get('adx_period', 14)
        self.min_adx = config.get('min_adx', 20)  # Only trade when ADX > 20
        self.strong_adx = config.get('strong_adx', 30)  # Strong trend threshold

        # ATR Stop Loss Settings
        self.use_atr_stop = config.get('use_atr_stop', True)
        self.atr_period = config.get('atr_period', 14)
        self.atr_stop_mult = config.get('atr_stop_mult', 2.0)  # 2x ATR stop

        # Volume Filter Settings
        self.use_volume_filter = config.get('use_volume_filter', True)
        self.volume_window = config.get('volume_window', 20)
        self.volume_mult = config.get('volume_mult', 1.3)  # 1.3x avg volume

        # Breakout Detection
        self.breakout_vol_mult = config.get('breakout_vol_mult', 2.0)  # Increased from 1.5
        self.recent_high_lookback = config.get('high_lookback', 24)

        # RSI Settings
        self.use_rsi_filter = config.get('use_rsi_filter', True)
        self.rsi_period = config.get('rsi_period', 14)
        self.rsi_overbought = config.get('rsi_overbought', 70)
        self.rsi_oversold = config.get('rsi_oversold', 30)

        # MA Ribbon Settings
        self.use_ribbon = config.get('use_ribbon', True)
        self.ribbon_periods = config.get('ribbon_periods', [5, 10, 20, 30, 50])

        # Cooldown Settings
        self.cooldown_hours = config.get('cooldown_hours', 4)
        self.last_trade_time: Dict[str, datetime] = {}

        # Track current state
        self.trend_state: Dict[str, str] = {}  # {symbol: 'up', 'down', 'none'}
        self.entry_price: Dict[str, float] = {}
        self.stop_loss: Dict[str, float] = {}
        self.last_breakout: Dict[str, dict] = {}

        # Indicator cache for logging
        self.last_indicators: Dict[str, dict] = {}

    # ==================== INDICATOR CALCULATIONS ====================

    def _calculate_ema(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return series.ewm(span=period, adjust=False).mean()

    def _calculate_sma(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average."""
        return series.rolling(window=period).mean()

    def _calculate_ma(self, df: pd.DataFrame, period: int = None) -> pd.Series:
        """Calculate Moving Average based on configured type (EMA or SMA)."""
        if period is None:
            period = self.ma_period

        if self.ma_type == 'ema':
            return self._calculate_ema(df['close'], period)
        else:
            return self._calculate_sma(df['close'], period)

    def _calculate_atr(self, df: pd.DataFrame, period: int = None) -> pd.Series:
        """Calculate Average True Range."""
        if period is None:
            period = self.atr_period

        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        return atr

    def _calculate_adx(self, df: pd.DataFrame, period: int = None) -> pd.Series:
        """
        Calculate Average Directional Index (ADX) for trend strength.

        ADX > 20: Trending market (tradeable)
        ADX > 30: Strong trend
        ADX < 20: Ranging/choppy market (avoid)
        """
        if period is None:
            period = self.adx_period

        high = df['high']
        low = df['low']
        close = df['close']

        # Calculate +DM and -DM
        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        # Calculate ATR
        atr = self._calculate_atr(df, period)

        # Calculate +DI and -DI
        plus_di = 100 * self._calculate_ema(plus_dm, period) / atr.replace(0, 1e-10)
        minus_di = 100 * self._calculate_ema(minus_dm, period) / atr.replace(0, 1e-10)

        # Calculate DX and ADX
        di_sum = plus_di + minus_di
        di_diff = (plus_di - minus_di).abs()
        dx = 100 * di_diff / di_sum.replace(0, 1e-10)
        adx = self._calculate_ema(dx, period)

        return adx

    def _calculate_rsi(self, df: pd.DataFrame, period: int = None) -> pd.Series:
        """Calculate Relative Strength Index."""
        if period is None:
            period = self.rsi_period

        delta = df['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        rs = avg_gain / avg_loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_ribbon(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate MA ribbon for trend strength confirmation.

        Returns alignment status:
        - 'bullish': All MAs aligned (shorter above longer)
        - 'bearish': All MAs aligned (shorter below longer)
        - 'mixed': MAs not aligned (weak trend)
        """
        ribbon = {}
        close = df['close']

        for period in self.ribbon_periods:
            if self.ma_type == 'ema':
                ribbon[f'ma_{period}'] = self._calculate_ema(close, period).iloc[-1]
            else:
                ribbon[f'ma_{period}'] = self._calculate_sma(close, period).iloc[-1]

        # Check alignment
        values = [ribbon[f'ma_{p}'] for p in self.ribbon_periods]

        # Bullish: shorter MAs above longer MAs (ascending order when sorted by period)
        bullish = all(values[i] > values[i+1] for i in range(len(values)-1))
        # Bearish: shorter MAs below longer MAs (descending order)
        bearish = all(values[i] < values[i+1] for i in range(len(values)-1))

        if bullish:
            ribbon['alignment'] = 'bullish'
        elif bearish:
            ribbon['alignment'] = 'bearish'
        else:
            ribbon['alignment'] = 'mixed'

        return ribbon

    # ==================== FILTER FUNCTIONS ====================

    def _check_adx_filter(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Check ADX filter conditions.

        Returns dict with:
        - passed: bool - whether filter passes
        - adx: float - current ADX value
        - strength: str - 'strong', 'moderate', 'weak'
        """
        if not self.use_adx_filter:
            return {'passed': True, 'adx': 0, 'strength': 'disabled'}

        if len(df) < self.adx_period + 5:
            return {'passed': False, 'adx': 0, 'strength': 'insufficient_data'}

        adx = self._calculate_adx(df)
        current_adx = adx.iloc[-1]

        if pd.isna(current_adx):
            return {'passed': False, 'adx': 0, 'strength': 'nan'}

        passed = current_adx >= self.min_adx

        if current_adx >= self.strong_adx:
            strength = 'strong'
        elif current_adx >= self.min_adx:
            strength = 'moderate'
        else:
            strength = 'weak'

        return {
            'passed': passed,
            'adx': float(current_adx),
            'strength': strength
        }

    def _check_volume_filter(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Check volume confirmation.

        Returns dict with:
        - passed: bool
        - ratio: float - current volume / avg volume
        """
        if not self.use_volume_filter:
            return {'passed': True, 'ratio': 1.0}

        if len(df) < self.volume_window + 1:
            return {'passed': False, 'ratio': 0}

        avg_volume = df['volume'].iloc[-self.volume_window-1:-1].mean()
        current_volume = df['volume'].iloc[-1]

        if avg_volume <= 0:
            return {'passed': False, 'ratio': 0}

        ratio = current_volume / avg_volume
        passed = ratio >= self.volume_mult

        return {
            'passed': passed,
            'ratio': float(ratio)
        }

    def _check_rsi_filter(self, df: pd.DataFrame, direction: str) -> Dict[str, Any]:
        """
        Check RSI conditions and divergence.

        Args:
            direction: 'long' or 'short'

        Returns dict with:
        - passed: bool
        - rsi: float
        - divergence: str or None
        """
        if not self.use_rsi_filter:
            return {'passed': True, 'rsi': 50, 'divergence': None}

        if len(df) < self.rsi_period + 5:
            return {'passed': False, 'rsi': 50, 'divergence': None}

        rsi = self._calculate_rsi(df)
        current_rsi = rsi.iloc[-1]

        if pd.isna(current_rsi):
            return {'passed': False, 'rsi': 50, 'divergence': None}

        # Check divergence
        divergence = self._detect_divergence(df, rsi)

        # Filter logic
        if direction == 'long':
            # Don't buy when overbought, unless there's bullish divergence
            if current_rsi >= self.rsi_overbought and divergence != 'bullish':
                passed = False
            else:
                passed = True
        else:  # short
            # Don't short when oversold, unless there's bearish divergence
            if current_rsi <= self.rsi_oversold and divergence != 'bearish':
                passed = False
            else:
                passed = True

        return {
            'passed': passed,
            'rsi': float(current_rsi),
            'divergence': divergence
        }

    def _detect_divergence(self, df: pd.DataFrame, rsi: pd.Series, lookback: int = 14) -> Optional[str]:
        """
        Detect RSI divergence.

        Bullish divergence: Price makes lower low, RSI makes higher low
        Bearish divergence: Price makes higher high, RSI makes lower high
        """
        if len(df) < lookback + 5:
            return None

        price = df['close'].iloc[-lookback:]
        rsi_vals = rsi.iloc[-lookback:]

        # Find recent price extremes
        price_high_idx = price.idxmax()
        price_low_idx = price.idxmin()

        current_price = price.iloc[-1]
        current_rsi = rsi_vals.iloc[-1]

        # Bearish divergence: price near high but RSI declining
        high_price = price.max()
        high_rsi = rsi_vals.loc[price_high_idx] if price_high_idx in rsi_vals.index else rsi_vals.iloc[0]

        if current_price >= high_price * 0.995:  # Within 0.5% of high
            if current_rsi < high_rsi - 5:  # RSI 5+ points lower
                return 'bearish'

        # Bullish divergence: price near low but RSI rising
        low_price = price.min()
        low_rsi = rsi_vals.loc[price_low_idx] if price_low_idx in rsi_vals.index else rsi_vals.iloc[0]

        if current_price <= low_price * 1.005:  # Within 0.5% of low
            if current_rsi > low_rsi + 5:  # RSI 5+ points higher
                return 'bullish'

        return None

    def _check_cooldown(self, symbol: str) -> bool:
        """Check if symbol is in cooldown period."""
        if symbol not in self.last_trade_time:
            return True  # No cooldown

        hours_since = (datetime.now() - self.last_trade_time[symbol]).total_seconds() / 3600
        return hours_since >= self.cooldown_hours

    def _check_distance_from_ma(self, price: float, ma: float) -> bool:
        """Check if price is far enough from MA to avoid whipsaw zone."""
        if ma <= 0:
            return False

        distance_pct = abs(price - ma) / ma
        return distance_pct >= self.min_distance_pct

    # ==================== TREND DETECTION ====================

    def _detect_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect trend with comprehensive filtering.

        Returns dict with:
        - trend: 'up', 'down', 'none'
        - ma: current MA value
        - price: current price
        - distance_pct: distance from MA
        """
        min_len = max(self.ma_period, max(self.ribbon_periods)) + self.confirm_candles + 5

        if len(df) < min_len:
            return {'trend': 'none', 'ma': 0, 'price': 0, 'distance_pct': 0}

        df = df.copy()
        df['ma'] = self._calculate_ma(df)

        # Get recent candles for confirmation
        recent = df.tail(self.confirm_candles)
        current_price = df['close'].iloc[-1]
        current_ma = df['ma'].iloc[-1]

        if pd.isna(current_ma):
            return {'trend': 'none', 'ma': 0, 'price': current_price, 'distance_pct': 0}

        # Check consecutive closes above/below MA
        above_ma = (recent['close'] > recent['ma']).all()
        below_ma = (recent['close'] < recent['ma']).all()

        # Calculate distance
        distance_pct = abs(current_price - current_ma) / current_ma if current_ma > 0 else 0

        # Check minimum distance (avoid whipsaw zone)
        if distance_pct < self.min_distance_pct:
            trend = 'none'
        elif above_ma:
            trend = 'up'
        elif below_ma:
            trend = 'down'
        else:
            trend = 'none'

        return {
            'trend': trend,
            'ma': float(current_ma),
            'price': float(current_price),
            'distance_pct': float(distance_pct)
        }

    def _detect_breakout(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect breakout conditions for leverage boost.

        Breakout = price breaks recent high + volume spike (2x avg).
        """
        if len(df) < self.recent_high_lookback + 1:
            return {'is_breakout': False, 'strength': 0.0, 'leverage_boost': 0}

        current_price = df['close'].iloc[-1]
        current_vol = df['volume'].iloc[-1]

        # Recent high (24h lookback)
        recent_high = df['high'].iloc[-self.recent_high_lookback:-1].max()
        avg_volume = df['volume'].iloc[-self.recent_high_lookback:-1].mean()

        if avg_volume <= 0:
            return {'is_breakout': False, 'strength': 0.0, 'leverage_boost': 0}

        # Check breakout conditions
        price_break = current_price > recent_high
        volume_spike = current_vol > avg_volume * self.breakout_vol_mult

        if price_break and volume_spike:
            # Strong breakout: +3-4 leverage boost
            vol_ratio = current_vol / avg_volume
            price_break_pct = (current_price / recent_high - 1) * 100
            strength = vol_ratio * price_break_pct
            leverage_boost = 4 if strength > 0.5 else 3
            return {
                'is_breakout': True,
                'strength': float(strength),
                'leverage_boost': leverage_boost,
                'volume_ratio': float(vol_ratio),
                'price_break_pct': float(price_break_pct)
            }
        elif price_break:
            # Weak breakout (no volume): +1-2 leverage
            return {
                'is_breakout': True,
                'strength': 0.2,
                'leverage_boost': 2,
                'volume_ratio': float(current_vol / avg_volume) if avg_volume > 0 else 0,
                'price_break_pct': float((current_price / recent_high - 1) * 100)
            }

        return {'is_breakout': False, 'strength': 0.0, 'leverage_boost': 0}

    # ==================== STOP LOSS ====================

    def _calculate_stop_loss(self, df: pd.DataFrame, entry_price: float,
                            direction: str) -> float:
        """
        Calculate ATR-based dynamic stop loss.

        Args:
            entry_price: Trade entry price
            direction: 'long' or 'short'

        Returns:
            Stop loss price
        """
        if not self.use_atr_stop:
            # Fallback to fixed percentage stop
            stop_pct = 0.03  # 3%
            if direction == 'long':
                return entry_price * (1 - stop_pct)
            else:
                return entry_price * (1 + stop_pct)

        atr = self._calculate_atr(df)
        current_atr = atr.iloc[-1]

        if pd.isna(current_atr) or current_atr <= 0:
            current_atr = entry_price * 0.02  # Fallback to 2%

        if direction == 'long':
            return entry_price - (current_atr * self.atr_stop_mult)
        else:  # short
            return entry_price + (current_atr * self.atr_stop_mult)

    def _check_stop_loss(self, symbol: str, current_price: float) -> bool:
        """Check if stop loss has been hit."""
        if symbol not in self.stop_loss or symbol not in self.trend_state:
            return False

        stop = self.stop_loss[symbol]
        trend = self.trend_state[symbol]

        if trend == 'up' and current_price <= stop:
            return True
        elif trend == 'down' and current_price >= stop:
            return True

        return False

    # ==================== CONFIDENCE CALCULATION ====================

    def _calculate_confidence(self, adx_result: dict, volume_result: dict,
                             rsi_result: dict, ribbon: dict,
                             breakout: dict) -> float:
        """
        Calculate confidence score based on multiple factors.

        Base confidence: 0.50
        Boosts:
        - Strong ADX (>30): +0.15
        - Volume confirmed: +0.10
        - RSI divergence favorable: +0.10
        - Ribbon aligned: +0.10
        - Breakout: +strength * 0.1

        Maximum: 0.95
        """
        base_confidence = 0.50
        boosts = []

        # ADX boost
        if adx_result.get('strength') == 'strong':
            boosts.append(0.15)
        elif adx_result.get('strength') == 'moderate':
            boosts.append(0.08)

        # Volume boost
        if volume_result.get('passed'):
            ratio = volume_result.get('ratio', 1.0)
            boost = min(0.10, (ratio - 1) * 0.05)  # Scale with ratio
            boosts.append(max(0, boost))

        # RSI boost (favorable divergence)
        if rsi_result.get('divergence') in ['bullish', 'bearish']:
            boosts.append(0.10)

        # Ribbon alignment boost
        if ribbon.get('alignment') in ['bullish', 'bearish']:
            boosts.append(0.10)

        # Breakout boost
        if breakout.get('is_breakout'):
            strength = breakout.get('strength', 0)
            boosts.append(min(0.15, strength * 0.1))

        final_confidence = base_confidence + sum(boosts)
        return min(0.95, final_confidence)

    # ==================== MAIN SIGNAL GENERATION ====================

    def _check_exit(self, df: pd.DataFrame, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Check if we should exit based on:
        1. ATR stop loss hit
        2. Trend reversal (1 candle opposite)
        """
        if symbol not in self.trend_state or self.trend_state[symbol] == 'none':
            return None

        current_trend = self.trend_state[symbol]
        current_price = df['close'].iloc[-1]

        # Check stop loss
        if self._check_stop_loss(symbol, current_price):
            entry = self.entry_price.get(symbol, current_price)
            pnl_pct = ((current_price - entry) / entry) * 100
            if current_trend == 'down':
                pnl_pct = -pnl_pct  # Invert for short

            return {
                'action': 'close',
                'symbol': symbol,
                'confidence': 0.90,
                'reason': f'ATR stop loss hit (entry: {entry:.2f}, stop: {self.stop_loss.get(symbol, 0):.2f}, PnL: {pnl_pct:.2f}%)',
                'strategy': 'ma_trend_follow',
                'exit_type': 'stop_loss'
            }

        # Check trend reversal
        if self.exit_opposite and len(df) >= self.ma_period + 1:
            df_copy = df.copy()
            df_copy['ma'] = self._calculate_ma(df_copy)
            latest = df_copy.iloc[-1]

            if current_trend == 'up' and latest['close'] < latest['ma']:
                entry = self.entry_price.get(symbol, current_price)
                pnl_pct = ((current_price - entry) / entry) * 100
                return {
                    'action': 'close',
                    'symbol': symbol,
                    'confidence': 0.80,
                    'reason': f'MA trend reversal (up -> cross below EMA, PnL: {pnl_pct:.2f}%)',
                    'strategy': 'ma_trend_follow',
                    'exit_type': 'trend_reversal'
                }

            if current_trend == 'down' and latest['close'] > latest['ma']:
                entry = self.entry_price.get(symbol, current_price)
                pnl_pct = ((entry - current_price) / entry) * 100  # Inverted for short
                return {
                    'action': 'close',
                    'symbol': symbol,
                    'confidence': 0.80,
                    'reason': f'MA trend reversal (down -> cross above EMA, PnL: {pnl_pct:.2f}%)',
                    'strategy': 'ma_trend_follow',
                    'exit_type': 'trend_reversal'
                }

        return None

    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Generate trading signals based on enhanced MA trend following.

        Signal Generation Flow:
        1. Check for exit conditions first (stop loss, trend reversal)
        2. Check cooldown period
        3. Detect trend (EMA crossover with distance filter)
        4. Apply ADX filter (trend strength)
        5. Apply volume filter
        6. Apply RSI filter (overbought/oversold, divergence)
        7. Check MA ribbon alignment
        8. Detect breakout for leverage boost
        9. Calculate final confidence

        Args:
            data: Dict with symbol -> DataFrame (OHLCV)

        Returns:
            Signal dict with action, size, leverage, confidence, etc.
        """
        signals = []

        for symbol in self.symbols:
            if symbol not in data:
                continue

            df = data[symbol]
            min_len = max(self.ma_period, self.adx_period, self.atr_period,
                         max(self.ribbon_periods)) + self.confirm_candles + 10

            if len(df) < min_len:
                continue

            current_price = df['close'].iloc[-1]

            # Store indicators for logging
            indicators = {}

            # 1. Check for exit first
            exit_signal = self._check_exit(df, symbol)
            if exit_signal:
                self.trend_state[symbol] = 'none'
                self.entry_price.pop(symbol, None)
                self.stop_loss.pop(symbol, None)
                signals.append(exit_signal)
                continue

            # 2. Check cooldown
            if not self._check_cooldown(symbol):
                continue

            # 3. Detect trend
            trend_result = self._detect_trend(df)
            current_trend = trend_result['trend']
            prev_trend = self.trend_state.get(symbol, 'none')

            indicators['ma'] = trend_result['ma']
            indicators['price'] = trend_result['price']
            indicators['distance_pct'] = trend_result['distance_pct']

            # Skip if no new trend or same trend
            if current_trend == 'none' or current_trend == prev_trend:
                continue

            # 4. Apply ADX filter
            adx_result = self._check_adx_filter(df)
            indicators['adx'] = adx_result['adx']
            indicators['adx_strength'] = adx_result['strength']

            if not adx_result['passed']:
                continue  # Skip - not trending enough

            # 5. Apply volume filter
            volume_result = self._check_volume_filter(df)
            indicators['volume_ratio'] = volume_result['ratio']

            # Volume filter is soft - we use it for confidence but don't reject

            # 6. Apply RSI filter
            direction = 'long' if current_trend == 'up' else 'short'
            rsi_result = self._check_rsi_filter(df, direction)
            indicators['rsi'] = rsi_result['rsi']
            indicators['divergence'] = rsi_result['divergence']

            if not rsi_result['passed']:
                continue  # Skip - RSI conditions not favorable

            # 7. Check MA ribbon
            ribbon = self._calculate_ribbon(df)
            indicators['ribbon_alignment'] = ribbon['alignment']

            # Optional: require ribbon alignment (strict mode)
            # if self.use_ribbon and ribbon['alignment'] == 'mixed':
            #     continue

            # 8. Detect breakout
            breakout = self._detect_breakout(df)
            indicators['breakout'] = breakout['is_breakout']
            indicators['breakout_strength'] = breakout.get('strength', 0)

            # 9. Calculate confidence
            confidence = self._calculate_confidence(
                adx_result, volume_result, rsi_result, ribbon, breakout
            )

            # Store indicators
            self.last_indicators[symbol] = indicators

            # Generate signal
            if current_trend == 'up':
                # Calculate leverage
                base_leverage = 3
                if breakout['is_breakout']:
                    leverage = min(self.max_leverage, base_leverage + breakout['leverage_boost'])
                else:
                    leverage = base_leverage

                # Calculate stop loss
                entry_price = current_price
                stop = self._calculate_stop_loss(df, entry_price, 'long')

                # Build reason string
                reason_parts = [f'{self.ma_type.upper()}-{self.ma_period} uptrend']
                reason_parts.append(f'ADX={adx_result["adx"]:.1f}')
                if volume_result['passed']:
                    reason_parts.append(f'Vol={volume_result["ratio"]:.1f}x')
                if ribbon['alignment'] == 'bullish':
                    reason_parts.append('Ribbon aligned')
                if breakout['is_breakout']:
                    reason_parts.append(f'BREAKOUT +{breakout["leverage_boost"]}x')
                if rsi_result['divergence'] == 'bullish':
                    reason_parts.append('Bullish divergence')

                # Update state
                self.trend_state[symbol] = 'up'
                self.entry_price[symbol] = entry_price
                self.stop_loss[symbol] = stop
                self.last_breakout[symbol] = breakout

                signals.append({
                    'action': 'buy',
                    'symbol': symbol,
                    'size': self.position_size_pct,
                    'leverage': leverage,
                    'confidence': confidence,
                    'reason': ' | '.join(reason_parts),
                    'strategy': 'ma_trend_follow',
                    'breakout': breakout['is_breakout'],
                    'stop_loss': stop,
                    'indicators': indicators
                })

            elif current_trend == 'down':
                # Calculate leverage (slightly lower for shorts)
                base_leverage = 3
                leverage = min(self.max_leverage, base_leverage + 1)

                # Calculate stop loss
                entry_price = current_price
                stop = self._calculate_stop_loss(df, entry_price, 'short')

                # Build reason string
                reason_parts = [f'{self.ma_type.upper()}-{self.ma_period} downtrend']
                reason_parts.append(f'ADX={adx_result["adx"]:.1f}')
                if volume_result['passed']:
                    reason_parts.append(f'Vol={volume_result["ratio"]:.1f}x')
                if ribbon['alignment'] == 'bearish':
                    reason_parts.append('Ribbon aligned')
                if rsi_result['divergence'] == 'bearish':
                    reason_parts.append('Bearish divergence')

                # Update state
                self.trend_state[symbol] = 'down'
                self.entry_price[symbol] = entry_price
                self.stop_loss[symbol] = stop

                signals.append({
                    'action': 'short',
                    'symbol': symbol,
                    'size': self.position_size_pct * 0.8,  # Slightly smaller for shorts
                    'leverage': leverage,
                    'confidence': confidence * 0.95,  # Slightly lower for shorts
                    'reason': ' | '.join(reason_parts),
                    'strategy': 'ma_trend_follow',
                    'stop_loss': stop,
                    'indicators': indicators
                })

        # Return best signal or hold
        if signals:
            # Sort by confidence and return best
            signals.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            best = signals[0]

            # Update cooldown for executed symbol
            if best['action'] in ['buy', 'short']:
                self.last_trade_time[best['symbol']] = datetime.now()

            return best

        # Default hold signal
        return {
            'action': 'hold',
            'symbol': self.symbols[0] if self.symbols else 'BTC/USDT',
            'confidence': 0.0,
            'reason': 'No MA trend signal (filters not passed)',
            'strategy': 'ma_trend_follow',
            'indicators': self.last_indicators.get(self.symbols[0], {}) if self.symbols else {}
        }

    def update_model(self, data: Optional[Dict[str, pd.DataFrame]] = None) -> bool:
        """MA Trend is rule-based, no model to update."""
        return True

    def get_status(self) -> Dict[str, Any]:
        """Get strategy status."""
        base_status = super().get_status()
        base_status.update({
            'ma_type': self.ma_type,
            'ma_period': self.ma_period,
            'confirm_candles': self.confirm_candles,
            'use_adx_filter': self.use_adx_filter,
            'min_adx': self.min_adx,
            'use_atr_stop': self.use_atr_stop,
            'atr_stop_mult': self.atr_stop_mult,
            'use_volume_filter': self.use_volume_filter,
            'use_rsi_filter': self.use_rsi_filter,
            'use_ribbon': self.use_ribbon,
            'cooldown_hours': self.cooldown_hours,
            'trend_states': self.trend_state.copy(),
            'entry_prices': self.entry_price.copy(),
            'stop_losses': self.stop_loss.copy(),
            'last_indicators': self.last_indicators.copy()
        })
        return base_status

    def on_order_filled(self, order: Dict[str, Any]) -> None:
        """Update internal state when order is filled."""
        symbol = order.get('symbol', '')
        action = order.get('action', '')

        # Reset cooldown on fill
        if symbol and action in ['buy', 'short']:
            self.last_trade_time[symbol] = datetime.now()
