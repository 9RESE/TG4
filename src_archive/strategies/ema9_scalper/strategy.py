"""
Phase 25: EMA-9 Scalper Strategy - Improved Implementation
Fast EMA-9 crossover signals with comprehensive filtering and risk management.

Improvements over Phase 22:
- Added SL/TP tracking with active_scalps dict
- Added position awareness and max_positions enforcement
- Added trade cooldown mechanism (bars and time-based)
- Added RSI filter for signal confirmation
- Added volume confirmation filter
- Added EMA trend filter (EMA-21) for trade direction
- Added ADX filter for ranging market detection
- Fixed ATR calculation (removed incorrect sqrt(24) scaling)
- Added on_order_filled() for position tracking
- Dynamic position sizing based on volatility
- Reduced default leverage to 3x

Rules:
- Activates when ATR > threshold (volatility filter)
- Buy signal: price crosses above EMA-9 + RSI < 40 + price > EMA-21 + volume confirmation
- Sell signal: price crosses below EMA-9 + RSI > 60 + price < EMA-21 + volume confirmation
- ADX < 30 preferred (ranging market for crossover strategies)
- Override mode: bypasses RL confidence gate when ATR > override threshold
- Max 3x leverage for controlled risk
- Automatic SL/TP tracking and exit signals
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from strategies.base_strategy import BaseStrategy
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime, timedelta


class EMA9Scalper(BaseStrategy):
    """
    EMA-9 crossover scalper with comprehensive filtering and risk management.

    Phase 25 improvements:
    - Proper SL/TP tracking with automatic exits
    - Position awareness (no duplicate entries)
    - Multi-filter confirmation (RSI, volume, ADX, trend)
    - Trade cooldown to prevent overtrading
    - Dynamic position sizing based on ATR
    - Fixed ATR calculation for correct volatility measurement
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Core EMA parameters
        self.timeframe = config.get('timeframe', '5m')
        # Phase 30: Support both ema_fast_period and ema_period for compatibility
        self.ema_fast_period = config.get('ema_fast_period', config.get('ema_period', 9))
        self.ema_slow_period = config.get('ema_slow_period', 21)  # Trend filter

        # Leverage and sizing (reduced from 5x to 3x)
        self.leverage = config.get('max_leverage', config.get('leverage', 3))
        # Phase 30: Support both base_size_pct and size_pct for compatibility
        self.base_size_pct = config.get('base_size_pct', config.get('size_pct', 0.05))
        self.dynamic_sizing = config.get('dynamic_sizing', True)
        self.min_size_pct = config.get('min_size_pct', 0.02)
        self.max_size_pct = config.get('max_size_pct', 0.10)

        # ATR / Volatility thresholds
        self.atr_period = config.get('atr_period', 14)
        self.atr_threshold = config.get('atr_threshold', 0.5)  # Min ATR% to trade
        self.atr_threshold_high = config.get('atr_threshold_high', 3.0)  # Max ATR%
        self.override_atr_threshold = config.get('override_atr_threshold', 1.8)  # Override RL gate

        # RSI filter parameters
        self.rsi_period = config.get('rsi_period', 14)
        self.rsi_buy_max = config.get('rsi_buy_max', 40)  # Only buy when RSI < 40
        self.rsi_sell_min = config.get('rsi_sell_min', 60)  # Only sell when RSI > 60
        self.use_rsi_filter = config.get('use_rsi_filter', True)

        # ADX filter (ranging market detection)
        self.adx_period = config.get('adx_period', 14)
        self.adx_max = config.get('adx_max', 30)  # Only trade when ADX < 30
        self.use_adx_filter = config.get('use_adx_filter', True)

        # Volume filter
        self.volume_mult = config.get('volume_mult', 1.2)
        self.volume_window = config.get('volume_window', 20)
        self.use_volume_filter = config.get('use_volume_filter', True)

        # Trend filter (require price aligned with slow EMA)
        self.use_trend_filter = config.get('use_trend_filter', True)
        self.trend_tolerance = config.get('trend_tolerance', 0.002)  # 0.2% tolerance

        # Risk management - SL/TP
        self.stop_loss_pct = config.get('stop_loss_pct', 0.01)  # 1% stop loss
        self.take_profit_pct = config.get('take_profit_pct', 0.02)  # 2% take profit

        # Position limits
        self.max_positions = config.get('max_positions', 2)
        self.symbols = config.get('symbols', ['BTC/USDT', 'XRP/USDT'])

        # Trade cooldown
        self.cooldown_bars = config.get('cooldown_bars', 3)
        self.cooldown_minutes = config.get('cooldown_minutes', 10)

        # Crossover confirmation
        self.require_confirmation = config.get('require_confirmation', True)
        self.confirmation_threshold = config.get('confirmation_threshold', 0.001)  # 0.1% above/below EMA

        # State tracking
        self.active_scalps: Dict[str, Dict[str, Any]] = {}
        self.last_trade_time: Dict[str, datetime] = {}
        self.last_trade_bar: Dict[str, int] = {}
        self.bar_count: Dict[str, int] = {}

        # Previous bar state for crossover detection
        self.prev_price: Dict[str, float] = {}
        self.prev_ema_fast: Dict[str, float] = {}

        # Cached indicators for status
        self.current_atr_pct = 0.0
        self.current_adx = 0.0
        self.override_active = False
        self.last_rsi: Dict[str, float] = {}
        self.last_ema_fast: Dict[str, float] = {}
        self.last_ema_slow: Dict[str, float] = {}
        self.last_signal_reason: str = ""

    def _calculate_ema(self, close: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return close.ewm(span=period, adjust=False).mean()

    def _calculate_rsi(self, close: pd.Series, period: int = None) -> pd.Series:
        """
        Calculate RSI using Wilder's smoothing (proper RSI calculation).
        """
        if period is None:
            period = self.rsi_period

        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        # Wilder's smoothing (EMA with alpha = 1/period)
        avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_atr(self, df: pd.DataFrame, period: int = None) -> pd.Series:
        """Calculate Average True Range with Wilder's smoothing."""
        if period is None:
            period = self.atr_period

        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        return atr

    def _calculate_atr_pct(self, df: pd.DataFrame) -> float:
        """
        Calculate ATR as percentage of current price.
        Fixed: Uses direct ATR percentage without incorrect sqrt scaling.
        """
        if len(df) < self.atr_period + 1:
            return 0.0

        atr = self._calculate_atr(df)
        current_price = df['close'].iloc[-1]

        if current_price > 0 and not pd.isna(atr.iloc[-1]):
            return (atr.iloc[-1] / current_price) * 100
        return 0.0

    def _calculate_adx(self, df: pd.DataFrame, period: int = None) -> pd.Series:
        """
        Calculate Average Directional Index (ADX).
        ADX < 25-30 indicates ranging market (good for crossover strategies).
        """
        if period is None:
            period = self.adx_period

        high = df['high']
        low = df['low']

        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        atr = self._calculate_atr(df, period)

        plus_di = 100 * (plus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / (atr + 1e-10))
        minus_di = 100 * (minus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / (atr + 1e-10))

        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10))
        adx = dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

        return adx

    def _check_volume_confirmation(self, df: pd.DataFrame) -> bool:
        """Check if current volume is above average."""
        if not self.use_volume_filter:
            return True

        if 'volume' not in df.columns or len(df) < self.volume_window + 1:
            return True  # Skip filter if no volume data

        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].iloc[-self.volume_window-1:-1].mean()

        if avg_volume > 0:
            return current_volume > avg_volume * self.volume_mult
        return True

    def _check_cooldown(self, symbol: str, current_bar: int) -> bool:
        """Check if we're in cooldown period after a trade."""
        # Bar-based cooldown
        last_bar = self.last_trade_bar.get(symbol, -999)
        if current_bar - last_bar < self.cooldown_bars:
            return False

        # Time-based cooldown
        last_time = self.last_trade_time.get(symbol)
        if last_time:
            elapsed = datetime.now() - last_time
            if elapsed < timedelta(minutes=self.cooldown_minutes):
                return False

        return True

    def _calculate_dynamic_size(self, atr_pct: float) -> float:
        """
        Calculate position size based on volatility.
        Higher volatility = smaller position for risk control.
        """
        if not self.dynamic_sizing:
            return self.base_size_pct

        vol_range = self.atr_threshold_high - self.atr_threshold
        if vol_range <= 0:
            return self.base_size_pct

        vol_normalized = (atr_pct - self.atr_threshold) / vol_range
        vol_normalized = max(0, min(1, vol_normalized))

        # Inverse scaling: 1.0 at low vol, 0.4 at high vol
        scale_factor = 1.0 - (vol_normalized * 0.6)

        size = self.base_size_pct * scale_factor
        return max(self.min_size_pct, min(self.max_size_pct, size))

    def _detect_crossover(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Detect EMA-9 crossover with multi-filter confirmation.

        Returns dict with signal details including all filter states.
        """
        min_bars = max(self.ema_fast_period, self.ema_slow_period,
                       self.rsi_period, self.adx_period, self.atr_period) + 5

        if len(df) < min_bars:
            return {'signal': 'hold', 'strength': 0.0, 'reason': 'Insufficient data'}

        close = df['close']

        # Calculate indicators
        ema_fast = self._calculate_ema(close, self.ema_fast_period)
        ema_slow = self._calculate_ema(close, self.ema_slow_period)
        rsi = self._calculate_rsi(close)
        adx = self._calculate_adx(df)
        atr_pct = self._calculate_atr_pct(df)

        # Get current values
        price = close.iloc[-1]
        prev_price = close.iloc[-2]
        ema_fast_now = ema_fast.iloc[-1]
        ema_fast_prev = ema_fast.iloc[-2]
        ema_slow_now = ema_slow.iloc[-1]
        rsi_now = rsi.iloc[-1]
        adx_now = adx.iloc[-1]

        # Store for status
        self.current_atr_pct = atr_pct
        self.current_adx = adx_now if not pd.isna(adx_now) else 0
        self.last_rsi[symbol] = rsi_now if not pd.isna(rsi_now) else 50
        self.last_ema_fast[symbol] = ema_fast_now
        self.last_ema_slow[symbol] = ema_slow_now

        # Check override status
        self.override_active = atr_pct > self.override_atr_threshold

        # Validate values
        if any(pd.isna(x) for x in [ema_fast_now, ema_fast_prev, ema_slow_now, rsi_now]):
            return {'signal': 'hold', 'strength': 0.0, 'reason': 'NaN in indicators'}

        # Build base result with all indicator values
        base_result = {
            'price': price,
            'ema_fast': ema_fast_now,
            'ema_slow': ema_slow_now,
            'rsi': rsi_now,
            'adx': adx_now,
            'atr_pct': atr_pct,
            'override': self.override_active,
            'indicators': {
                'ema_fast': ema_fast_now,
                'ema_slow': ema_slow_now,
                'rsi': rsi_now,
                'adx': adx_now,
                'atr_pct': atr_pct
            }
        }

        # === FILTER CHECKS ===

        # 1. Volatility filter
        if atr_pct < self.atr_threshold:
            return {**base_result, 'signal': 'hold', 'strength': 0.0,
                    'reason': f'Low volatility: ATR={atr_pct:.2f}% < {self.atr_threshold}%'}

        if atr_pct > self.atr_threshold_high:
            return {**base_result, 'signal': 'hold', 'strength': 0.0,
                    'reason': f'Extreme volatility: ATR={atr_pct:.2f}% > {self.atr_threshold_high}%'}

        # 2. ADX filter (prefer ranging markets)
        if self.use_adx_filter and not pd.isna(adx_now) and adx_now > self.adx_max:
            return {**base_result, 'signal': 'hold', 'strength': 0.0,
                    'reason': f'Trending market: ADX={adx_now:.1f} > {self.adx_max}'}

        # 3. Volume filter
        if not self._check_volume_confirmation(df):
            return {**base_result, 'signal': 'hold', 'strength': 0.0,
                    'reason': 'Volume below threshold'}

        # Calculate dynamic position size
        position_size = self._calculate_dynamic_size(atr_pct)

        # === CROSSOVER DETECTION ===

        # Bullish crossover: price crosses above EMA-fast
        bullish_crossover = (prev_price <= ema_fast_prev) and (price > ema_fast_now)

        # Confirmation: price is at least threshold% above EMA
        if self.require_confirmation:
            bullish_confirmed = (price - ema_fast_now) / ema_fast_now > self.confirmation_threshold
        else:
            bullish_confirmed = True

        if bullish_crossover and bullish_confirmed:
            # RSI filter: only buy when not overbought
            if self.use_rsi_filter and rsi_now > self.rsi_buy_max:
                return {**base_result, 'signal': 'hold', 'strength': 0.0,
                        'reason': f'RSI too high for buy: {rsi_now:.0f} > {self.rsi_buy_max}'}

            # Trend filter: price should be above or near slow EMA
            if self.use_trend_filter:
                trend_aligned = price >= ema_slow_now * (1 - self.trend_tolerance)
                if not trend_aligned:
                    return {**base_result, 'signal': 'hold', 'strength': 0.0,
                            'reason': f'Price below EMA-{self.ema_slow_period} trend'}

            # Calculate signal strength
            strength = 0.6
            # Bonus for RSI being more oversold
            if rsi_now < 30:
                strength += 0.15
            elif rsi_now < self.rsi_buy_max:
                strength += 0.1
            # Bonus for low ADX (good range)
            if adx_now < 20:
                strength += 0.1
            # Bonus for override active (high vol opportunity)
            if self.override_active:
                strength += 0.1

            return {
                **base_result,
                'signal': 'buy',
                'strength': min(strength, 0.95),
                'entry': price,
                'target': price * (1 + self.take_profit_pct),
                'stop': price * (1 - self.stop_loss_pct),
                'size': position_size,
                'reason': f'EMA-{self.ema_fast_period} bullish crossover: RSI={rsi_now:.0f}, ATR={atr_pct:.1f}%'
            }

        # Bearish crossover: price crosses below EMA-fast
        bearish_crossover = (prev_price >= ema_fast_prev) and (price < ema_fast_now)

        if self.require_confirmation:
            bearish_confirmed = (ema_fast_now - price) / ema_fast_now > self.confirmation_threshold
        else:
            bearish_confirmed = True

        if bearish_crossover and bearish_confirmed:
            # RSI filter: only sell when not oversold
            if self.use_rsi_filter and rsi_now < self.rsi_sell_min:
                return {**base_result, 'signal': 'hold', 'strength': 0.0,
                        'reason': f'RSI too low for sell: {rsi_now:.0f} < {self.rsi_sell_min}'}

            # Trend filter: price should be below or near slow EMA
            if self.use_trend_filter:
                trend_aligned = price <= ema_slow_now * (1 + self.trend_tolerance)
                if not trend_aligned:
                    return {**base_result, 'signal': 'hold', 'strength': 0.0,
                            'reason': f'Price above EMA-{self.ema_slow_period} trend'}

            # Calculate signal strength
            strength = 0.55  # Slightly lower base for shorts
            if rsi_now > 70:
                strength += 0.15
            elif rsi_now > self.rsi_sell_min:
                strength += 0.1
            if adx_now < 20:
                strength += 0.1
            if self.override_active:
                strength += 0.1

            return {
                **base_result,
                'signal': 'sell',
                'strength': min(strength, 0.90),
                'entry': price,
                'target': price * (1 - self.take_profit_pct),
                'stop': price * (1 + self.stop_loss_pct),
                'size': position_size * 0.8,  # Smaller size for shorts
                'reason': f'EMA-{self.ema_fast_period} bearish crossover: RSI={rsi_now:.0f}, ATR={atr_pct:.1f}%'
            }

        return {
            **base_result,
            'signal': 'hold',
            'strength': 0.0,
            'reason': f'No crossover (ATR={atr_pct:.1f}%, ADX={adx_now:.0f}, RSI={rsi_now:.0f})'
        }

    def _update_active_scalps(self, symbol: str, current_price: float) -> Optional[Dict[str, Any]]:
        """
        Check active scalps for stop-loss or take-profit hits.
        Returns exit signal if SL/TP triggered.
        """
        if symbol not in self.active_scalps:
            return None

        scalp = self.active_scalps[symbol]
        direction = scalp['direction']
        entry_price = scalp['entry']
        stop_price = scalp['stop']
        target_price = scalp['target']

        # Check stop-loss for long
        if direction == 'long' and current_price <= stop_price:
            del self.active_scalps[symbol]
            pnl_pct = (current_price - entry_price) / entry_price * 100
            return {
                'action': 'sell',
                'symbol': symbol,
                'confidence': 0.95,
                'size': 1.0,  # Exit full position
                'reason': f'STOP-LOSS: {pnl_pct:.2f}% (entry={entry_price:.4f}, exit={current_price:.4f})',
                'exit_type': 'stop_loss',
                'pnl_pct': pnl_pct,
                'strategy': 'ema9_scalper'
            }

        # Check stop-loss for short
        if direction == 'short' and current_price >= stop_price:
            del self.active_scalps[symbol]
            pnl_pct = (entry_price - current_price) / entry_price * 100
            return {
                'action': 'cover',
                'symbol': symbol,
                'confidence': 0.95,
                'size': 1.0,
                'reason': f'STOP-LOSS: {pnl_pct:.2f}% (entry={entry_price:.4f}, exit={current_price:.4f})',
                'exit_type': 'stop_loss',
                'pnl_pct': pnl_pct,
                'strategy': 'ema9_scalper'
            }

        # Check take-profit for long
        if direction == 'long' and current_price >= target_price:
            del self.active_scalps[symbol]
            pnl_pct = (current_price - entry_price) / entry_price * 100
            return {
                'action': 'sell',
                'symbol': symbol,
                'confidence': 0.95,
                'size': 1.0,
                'reason': f'TAKE-PROFIT: +{pnl_pct:.2f}% (entry={entry_price:.4f}, exit={current_price:.4f})',
                'exit_type': 'take_profit',
                'pnl_pct': pnl_pct,
                'strategy': 'ema9_scalper'
            }

        # Check take-profit for short
        if direction == 'short' and current_price <= target_price:
            del self.active_scalps[symbol]
            pnl_pct = (entry_price - current_price) / entry_price * 100
            return {
                'action': 'cover',
                'symbol': symbol,
                'confidence': 0.95,
                'size': 1.0,
                'reason': f'TAKE-PROFIT: +{pnl_pct:.2f}% (entry={entry_price:.4f}, exit={current_price:.4f})',
                'exit_type': 'take_profit',
                'pnl_pct': pnl_pct,
                'strategy': 'ema9_scalper'
            }

        return None

    def should_override_rl(self, data: Dict[str, pd.DataFrame]) -> bool:
        """
        Check if EMA-9 scalper should override RL confidence gate.
        Override when daily ATR exceeds threshold.
        """
        for symbol in self.symbols:
            # Phase 30: Check 1m first for 60s polling, then 5m, 15m, 1h
            df = None
            for key in [f'{symbol}_1m', f'{symbol}_5m', f'{symbol}_15m', symbol]:
                if key in data and data[key] is not None and len(data[key]) > 0:
                    df = data[key]
                    break

            if df is not None and len(df) >= self.atr_period + 5:
                self.current_atr_pct = self._calculate_atr_pct(df)
                break

        self.override_active = self.current_atr_pct > self.override_atr_threshold
        return self.override_active

    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Generate EMA-9 crossover signals with comprehensive filtering.

        Features:
        - Multi-filter confirmation (RSI, volume, ADX, trend)
        - SL/TP tracking with automatic exits
        - Position awareness (respects max_positions)
        - Trade cooldown to prevent overtrading
        - Dynamic position sizing

        Args:
            data: Dict with symbol -> DataFrame (OHLCV)

        Returns:
            Signal dict with action, size, leverage, SL/TP levels, etc.
        """
        entry_signals = []
        exit_signals = []

        # Check override status
        self.should_override_rl(data)

        for symbol in self.symbols:
            # Phase 30: Prefer 1m for 60s polling, then 5m, 15m, 1h
            # Check each key explicitly to avoid DataFrame truth value ambiguity
            df = None
            for key in [f'{symbol}_1m', f'{symbol}_5m', f'{symbol}_15m', f'{symbol}_{self.timeframe}', symbol]:
                if key in data and data[key] is not None and len(data[key]) > 0:
                    df = data[key]
                    break

            if df is None or len(df) < max(self.ema_fast_period, self.ema_slow_period,
                                            self.rsi_period, self.adx_period) + 10:
                continue

            # Update bar count
            self.bar_count[symbol] = self.bar_count.get(symbol, 0) + 1
            current_bar = self.bar_count[symbol]
            current_price = df['close'].iloc[-1]

            # Check active scalps for SL/TP first (highest priority)
            exit_signal = self._update_active_scalps(symbol, current_price)
            if exit_signal:
                exit_signals.append(exit_signal)
                continue  # Don't open new position if we just exited

            # Check cooldown
            if not self._check_cooldown(symbol, current_bar):
                continue

            # Check max positions
            if len(self.active_scalps) >= self.max_positions:
                continue

            # Skip if already have position in this symbol
            if symbol in self.active_scalps:
                continue

            # Detect crossover with all filters
            crossover = self._detect_crossover(df, symbol)
            self.last_signal_reason = crossover.get('reason', '')

            if crossover['signal'] == 'buy':
                signal = {
                    'action': 'buy',
                    'symbol': symbol,
                    'size': crossover.get('size', self.base_size_pct),
                    'leverage': self.leverage,
                    'confidence': crossover['strength'],
                    'reason': crossover['reason'],
                    'strategy': 'ema9_scalper',
                    'source': 'ema9',
                    'override': self.override_active,
                    'atr_pct': crossover.get('atr_pct', 0),
                    'entry': crossover.get('entry', current_price),
                    'target': crossover.get('target', current_price * 1.02),
                    'stop': crossover.get('stop', current_price * 0.99),
                    'indicators': crossover.get('indicators', {}),
                    'scalp': True
                }
                entry_signals.append(signal)

            elif crossover['signal'] == 'sell':
                signal = {
                    'action': 'sell',
                    'symbol': symbol,
                    'size': crossover.get('size', self.base_size_pct * 0.8),
                    'leverage': self.leverage,
                    'confidence': crossover['strength'],
                    'reason': crossover['reason'],
                    'strategy': 'ema9_scalper',
                    'source': 'ema9',
                    'override': self.override_active,
                    'atr_pct': crossover.get('atr_pct', 0),
                    'entry': crossover.get('entry', current_price),
                    'target': crossover.get('target', current_price * 0.98),
                    'stop': crossover.get('stop', current_price * 1.01),
                    'indicators': crossover.get('indicators', {}),
                    'scalp': True
                }
                entry_signals.append(signal)

        # Prioritize exit signals over new entries
        if exit_signals:
            return max(exit_signals, key=lambda x: x.get('confidence', 0))

        # Return best entry signal
        if entry_signals:
            best = max(entry_signals, key=lambda x: x.get('confidence', 0))

            # Track the new scalp
            symbol = best['symbol']
            self.active_scalps[symbol] = {
                'direction': 'long' if best['action'] == 'buy' else 'short',
                'entry': best['entry'],
                'target': best['target'],
                'stop': best['stop'],
                'size': best['size'],
                'timestamp': datetime.now()
            }
            self.last_trade_time[symbol] = datetime.now()
            self.last_trade_bar[symbol] = self.bar_count.get(symbol, 0)

            return best

        # No signal - hold
        active_info = f", {len(self.active_scalps)} active" if self.active_scalps else ""
        primary_symbol = self.symbols[0] if self.symbols else 'BTC/USDT'
        return {
            'action': 'hold',
            'symbol': primary_symbol,
            'confidence': 0.0,
            'reason': f'{self.last_signal_reason}{active_info}',
            'strategy': 'ema9_scalper',
            'override': self.override_active,
            'indicators': {
                'atr_pct': self.current_atr_pct,
                'adx': self.current_adx,
                'rsi': self.last_rsi.get(primary_symbol, 50),
                'ema_fast': self.last_ema_fast.get(primary_symbol, 0),
                'ema_slow': self.last_ema_slow.get(primary_symbol, 0),
                'override_active': self.override_active,
                'atr_threshold': self.atr_threshold,
                'atr_threshold_high': self.atr_threshold_high,
                'adx_max': self.adx_max,
                'rsi_buy_max': self.rsi_buy_max,
                'rsi_sell_min': self.rsi_sell_min,
                'active_scalps': len(self.active_scalps),
                'max_positions': self.max_positions,
                'in_cooldown': not self._check_cooldown(primary_symbol, self.bar_count.get(primary_symbol, 0)),
                'waiting_for_crossover': True
            }
        }

    def on_order_filled(self, order: Dict[str, Any]) -> None:
        """
        Callback when an order is filled.
        Updates active scalp tracking.
        """
        symbol = order.get('symbol', '')
        action = order.get('action', '')

        # If this is an exit order, remove from active scalps
        if action in ['sell', 'cover'] and symbol in self.active_scalps:
            del self.active_scalps[symbol]

        # Update cooldown tracking
        if symbol:
            self.last_trade_time[symbol] = datetime.now()
            self.last_trade_bar[symbol] = self.bar_count.get(symbol, 0)

    def update_model(self, data: Optional[Dict[str, pd.DataFrame]] = None) -> bool:
        """EMA-9 scalper is rule-based, no model to update."""
        return True

    def get_status(self) -> Dict[str, Any]:
        """Get detailed strategy status including active positions."""
        base_status = super().get_status()
        base_status.update({
            # Core parameters
            'ema_fast_period': self.ema_fast_period,
            'ema_slow_period': self.ema_slow_period,
            'timeframe': self.timeframe,
            'leverage': self.leverage,

            # Position sizing
            'base_size_pct': self.base_size_pct,
            'dynamic_sizing': self.dynamic_sizing,

            # Risk management
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct,
            'max_positions': self.max_positions,

            # Thresholds
            'atr_threshold': self.atr_threshold,
            'override_atr_threshold': self.override_atr_threshold,
            'rsi_buy_max': self.rsi_buy_max,
            'rsi_sell_min': self.rsi_sell_min,
            'adx_max': self.adx_max,

            # Current state
            'current_atr_pct': self.current_atr_pct,
            'current_adx': self.current_adx,
            'override_active': self.override_active,
            'last_rsi': self.last_rsi.copy(),
            'last_ema_fast': self.last_ema_fast.copy(),
            'last_ema_slow': self.last_ema_slow.copy(),

            # Active positions
            'active_scalps_count': len(self.active_scalps),
            'active_scalps': {
                sym: {
                    'direction': s['direction'],
                    'entry': s['entry'],
                    'target': s['target'],
                    'stop': s['stop'],
                    'size': s['size'],
                    'age_seconds': (datetime.now() - s['timestamp']).total_seconds()
                }
                for sym, s in self.active_scalps.items()
            },

            # Filter status
            'use_rsi_filter': self.use_rsi_filter,
            'use_adx_filter': self.use_adx_filter,
            'use_volume_filter': self.use_volume_filter,
            'use_trend_filter': self.use_trend_filter,

            # Cooldown
            'cooldown_bars': self.cooldown_bars,
            'cooldown_minutes': self.cooldown_minutes
        })
        return base_status
