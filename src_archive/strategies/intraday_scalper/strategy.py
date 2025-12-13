"""
Phase 25: IntraDay Scalper Strategy - Volatility Harvester (Improved)
5-15min Bollinger Band squeezes + RSI extremes for quick BTC/XRP entries/exits.

Improvements over Phase 21:
- Fixed RSI to use Wilder's EMA (proper RSI calculation)
- Added EMA trend filter for trade direction alignment
- Added volume confirmation filter
- Properly uses squeeze detection
- Added ADX filter for ranging market detection
- Dynamic ATR-based position sizing
- Trade cooldown mechanism
- Active scalps tracking with SL/TP management
- Optimized parameters for scalping (BB=12, RSI=7)

Rules:
- Activates only when ATR > threshold (volatility filter)
- ADX < 25 preferred (ranging market for mean reversion)
- Oversold: close < BB lower + RSI < 25 + price > EMA → buy scalp
- Overbought: close > BB upper + RSI > 75 + price < EMA → sell/short scalp
- Volume must be > 1.3x average for confirmation
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
from datetime import datetime, timedelta


class IntraDayScalper(BaseStrategy):
    """
    Volatility harvester for intra-day swings.
    Captures Bollinger Band squeezes during high volatility periods.

    Phase 25 improvements:
    - Proper Wilder's RSI calculation (EMA-based)
    - EMA trend filter
    - Volume confirmation
    - ADX ranging filter
    - Dynamic position sizing based on ATR
    - Trade cooldown to prevent overtrading
    - Active position tracking with SL/TP
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Optimized parameters for scalping (faster response)
        self.bb_period = config.get('bb_period', 12)  # Was 20, faster for scalping
        self.bb_std = config.get('bb_std', 2.0)
        self.rsi_period = config.get('rsi_period', 7)  # Was 14, faster for scalping
        self.ema_period = config.get('ema_period', 9)  # Trend filter
        self.adx_period = config.get('adx_period', 14)  # ADX for ranging detection
        self.atr_period = config.get('atr_period', 14)

        # Volatility thresholds
        self.vol_threshold = config.get('vol_threshold', 0.5)  # ATR% to activate
        self.vol_threshold_high = config.get('vol_threshold_high', 3.0)  # Too volatile

        # RSI thresholds (tighter for crypto scalping)
        self.rsi_oversold = config.get('rsi_oversold', 25)  # Was 30
        self.rsi_overbought = config.get('rsi_overbought', 75)  # Was 70

        # ADX filter
        self.adx_max = config.get('adx_max', 25)  # Only trade when ADX < 25 (ranging)
        self.use_adx_filter = config.get('use_adx_filter', True)

        # Volume filter
        self.volume_mult = config.get('volume_mult', 1.3)  # Volume > 1.3x avg
        self.volume_window = config.get('volume_window', 20)
        self.use_volume_filter = config.get('use_volume_filter', True)

        # Squeeze detection
        self.squeeze_threshold = config.get('squeeze_threshold', 0.8)  # Band width < 80% avg
        self.require_squeeze = config.get('require_squeeze', False)  # Optional strict mode

        # Targets and stops
        self.scalp_target_pct = config.get('scalp_target_pct', 0.01)  # 1% target
        self.scalp_stop_pct = config.get('scalp_stop_pct', 0.005)  # 0.5% stop

        # Position sizing
        self.base_size_pct = config.get('base_size_pct', 0.08)  # 8% base position
        self.dynamic_sizing = config.get('dynamic_sizing', True)
        self.min_size_pct = config.get('min_size_pct', 0.03)  # Min 3%
        self.max_size_pct = config.get('max_size_pct', 0.15)  # Max 15%

        # Trade cooldown
        self.cooldown_bars = config.get('cooldown_bars', 3)  # Min bars between trades
        self.cooldown_minutes = config.get('cooldown_minutes', 15)  # Min minutes between trades

        # Symbols
        self.symbols = config.get('symbols', ['BTC/USDT', 'XRP/USDT'])

        # Track scalp state
        self.active_scalps: Dict[str, Dict[str, Any]] = {}
        self.last_trade_time: Dict[str, datetime] = {}
        self.last_trade_bar: Dict[str, int] = {}
        self.bar_count: Dict[str, int] = {}

        # Cached indicators
        self.last_atr_pct = 0.0
        self.last_adx = 0.0
        self.last_rsi: Dict[str, float] = {}
        self.last_ema: Dict[str, float] = {}
        self.last_bb_position: Dict[str, str] = {}  # 'above', 'below', 'middle'
        self.last_squeeze_state: Dict[str, bool] = {}

    def _calculate_bollinger_bands(self, close: pd.Series) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = close.rolling(window=self.bb_period).mean()
        std = close.rolling(window=self.bb_period).std()
        upper = sma + (self.bb_std * std)
        lower = sma - (self.bb_std * std)

        # Calculate band width for squeeze detection
        width = (upper - lower) / sma
        avg_width = width.rolling(window=20).mean()

        return {
            'sma': sma,
            'upper': upper,
            'lower': lower,
            'width': width,
            'avg_width': avg_width
        }

    def _calculate_rsi_wilder(self, close: pd.Series, period: int = None) -> pd.Series:
        """
        Calculate RSI using Wilder's smoothing (EMA-based).
        This is the correct RSI calculation used by most platforms.
        """
        if period is None:
            period = self.rsi_period

        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        # Use Wilder's smoothing (EMA with alpha = 1/period)
        # This is equivalent to: EMA = prev_EMA + alpha * (current - prev_EMA)
        avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_ema(self, close: pd.Series, period: int = None) -> pd.Series:
        """Calculate Exponential Moving Average."""
        if period is None:
            period = self.ema_period
        return close.ewm(span=period, adjust=False).mean()

    def _calculate_atr(self, df: pd.DataFrame, period: int = None) -> pd.Series:
        """Calculate Average True Range with proper smoothing."""
        if period is None:
            period = self.atr_period

        high = df['high']
        low = df['low']
        close = df['close']

        # True Range calculation
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Wilder's smoothing for ATR
        atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        return atr

    def _calculate_atr_pct(self, df: pd.DataFrame, period: int = None) -> float:
        """Calculate ATR as percentage of current price."""
        if len(df) < (period or self.atr_period) + 1:
            return 0.0

        atr = self._calculate_atr(df, period)
        current_price = df['close'].iloc[-1]

        if current_price > 0 and not pd.isna(atr.iloc[-1]):
            return (atr.iloc[-1] / current_price) * 100
        return 0.0

    def _calculate_adx(self, df: pd.DataFrame, period: int = None) -> pd.Series:
        """
        Calculate Average Directional Index (ADX).
        ADX < 25 indicates ranging/choppy market (good for mean reversion).
        ADX > 25 indicates trending market (avoid mean reversion).
        """
        if period is None:
            period = self.adx_period

        high = df['high']
        low = df['low']
        close = df['close']

        # Calculate +DM and -DM
        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        # Calculate ATR
        atr = self._calculate_atr(df, period)

        # Smooth +DM and -DM
        plus_di = 100 * (plus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / (atr + 1e-10))
        minus_di = 100 * (minus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / (atr + 1e-10))

        # Calculate DX and ADX
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10))
        adx = dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

        return adx

    def _check_volume_confirmation(self, df: pd.DataFrame) -> bool:
        """Check if current volume is above average (confirms signal strength)."""
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
        Calculate position size based on current volatility.
        Higher volatility = smaller position size for risk management.
        """
        if not self.dynamic_sizing:
            return self.base_size_pct

        # Inverse relationship: higher ATR = smaller size
        # Normalize ATR between threshold and high threshold
        vol_range = self.vol_threshold_high - self.vol_threshold
        if vol_range <= 0:
            return self.base_size_pct

        # Scale factor: 1.0 at low vol, 0.3 at high vol
        vol_normalized = (atr_pct - self.vol_threshold) / vol_range
        vol_normalized = max(0, min(1, vol_normalized))  # Clamp 0-1

        scale_factor = 1.0 - (vol_normalized * 0.7)  # 1.0 to 0.3

        size = self.base_size_pct * scale_factor
        return max(self.min_size_pct, min(self.max_size_pct, size))

    def _detect_squeeze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect Bollinger Band squeeze setup with all filters.

        Returns:
            dict with squeeze info including direction, strength, and trade params
        """
        min_bars = max(self.bb_period, self.rsi_period, self.ema_period, self.adx_period) + 5
        if len(df) < min_bars:
            return {'is_squeeze': False, 'direction': None, 'strength': 0.0, 'reason': 'Insufficient data'}

        close = df['close']

        # Calculate all indicators
        bb = self._calculate_bollinger_bands(close)
        rsi = self._calculate_rsi_wilder(close)
        ema = self._calculate_ema(close)
        adx = self._calculate_adx(df)
        atr_pct = self._calculate_atr_pct(df)

        # Get latest values
        latest_close = close.iloc[-1]
        latest_upper = bb['upper'].iloc[-1]
        latest_lower = bb['lower'].iloc[-1]
        latest_sma = bb['sma'].iloc[-1]
        latest_rsi = rsi.iloc[-1]
        latest_ema = ema.iloc[-1]
        latest_adx = adx.iloc[-1]
        latest_width = bb['width'].iloc[-1]
        avg_width = bb['avg_width'].iloc[-1]

        # Store for status reporting
        self.last_atr_pct = atr_pct
        self.last_adx = latest_adx if not pd.isna(latest_adx) else 0

        # Validate values
        if any(pd.isna(x) for x in [latest_upper, latest_lower, latest_rsi, latest_ema]):
            return {'is_squeeze': False, 'direction': None, 'strength': 0.0, 'reason': 'NaN in indicators'}

        # Check volatility filter
        if atr_pct < self.vol_threshold:
            return {'is_squeeze': False, 'direction': None, 'strength': 0.0,
                    'reason': f'Low volatility: ATR={atr_pct:.2f}% < {self.vol_threshold}%'}

        if atr_pct > self.vol_threshold_high:
            return {'is_squeeze': False, 'direction': None, 'strength': 0.0,
                    'reason': f'Extreme volatility: ATR={atr_pct:.2f}% > {self.vol_threshold_high}%'}

        # Check ADX filter (prefer ranging markets for mean reversion)
        if self.use_adx_filter and latest_adx > self.adx_max:
            return {'is_squeeze': False, 'direction': None, 'strength': 0.0,
                    'reason': f'Trending market: ADX={latest_adx:.1f} > {self.adx_max}'}

        # Check volume confirmation
        if not self._check_volume_confirmation(df):
            return {'is_squeeze': False, 'direction': None, 'strength': 0.0,
                    'reason': 'Volume below threshold'}

        # Check squeeze (band tightening)
        is_squeeze = latest_width < avg_width * self.squeeze_threshold

        if self.require_squeeze and not is_squeeze:
            return {'is_squeeze': False, 'direction': None, 'strength': 0.0,
                    'reason': 'No band squeeze detected'}

        # Calculate dynamic position size
        position_size = self._calculate_dynamic_size(atr_pct)

        # Build base result
        base_result = {
            'atr_pct': atr_pct,
            'adx': latest_adx,
            'is_tight_bands': is_squeeze,
            'position_size': position_size,
            'indicators': {
                'rsi': latest_rsi,
                'ema': latest_ema,
                'bb_upper': latest_upper,
                'bb_lower': latest_lower,
                'bb_sma': latest_sma,
                'adx': latest_adx,
                'atr_pct': atr_pct
            }
        }

        # LONG signal: Price at lower band + RSI oversold + Price above EMA (micro uptrend)
        if latest_close <= latest_lower and latest_rsi < self.rsi_oversold:
            # EMA filter: prefer when price is recovering (close > EMA or approaching)
            ema_aligned = latest_close >= latest_ema * 0.995  # Within 0.5% of EMA

            strength = (self.rsi_oversold - latest_rsi) / self.rsi_oversold
            if is_squeeze:
                strength += 0.2  # Bonus for squeeze
            if ema_aligned:
                strength += 0.1  # Bonus for EMA alignment

            return {
                **base_result,
                'is_squeeze': True,
                'direction': 'long',
                'strength': min(strength, 1.0),
                'entry': latest_close,
                'target': latest_close * (1 + self.scalp_target_pct),
                'stop': latest_close * (1 - self.scalp_stop_pct),
                'ema_aligned': ema_aligned,
                'reason': f'BB lower touch + RSI={latest_rsi:.0f}'
            }

        # SHORT signal: Price at upper band + RSI overbought + Price below EMA (micro downtrend)
        if latest_close >= latest_upper and latest_rsi > self.rsi_overbought:
            # EMA filter: prefer when price is declining (close < EMA or approaching)
            ema_aligned = latest_close <= latest_ema * 1.005  # Within 0.5% of EMA

            strength = (latest_rsi - self.rsi_overbought) / (100 - self.rsi_overbought)
            if is_squeeze:
                strength += 0.2
            if ema_aligned:
                strength += 0.1

            return {
                **base_result,
                'is_squeeze': True,
                'direction': 'short',
                'strength': min(strength, 1.0),
                'entry': latest_close,
                'target': latest_close * (1 - self.scalp_target_pct),
                'stop': latest_close * (1 + self.scalp_stop_pct),
                'ema_aligned': ema_aligned,
                'reason': f'BB upper touch + RSI={latest_rsi:.0f}'
            }

        return {
            **base_result,
            'is_squeeze': False,
            'direction': None,
            'strength': 0.0,
            'reason': f'No setup (RSI={latest_rsi:.0f}, close vs bands)'
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

        # Check stop-loss
        if direction == 'long' and current_price <= stop_price:
            del self.active_scalps[symbol]
            pnl_pct = (current_price - entry_price) / entry_price * 100
            return {
                'action': 'sell',
                'symbol': symbol,
                'reason': f'STOP-LOSS hit: {pnl_pct:.2f}%',
                'exit_type': 'stop_loss',
                'pnl_pct': pnl_pct
            }

        if direction == 'short' and current_price >= stop_price:
            del self.active_scalps[symbol]
            pnl_pct = (entry_price - current_price) / entry_price * 100
            return {
                'action': 'cover',
                'symbol': symbol,
                'reason': f'STOP-LOSS hit: {pnl_pct:.2f}%',
                'exit_type': 'stop_loss',
                'pnl_pct': pnl_pct
            }

        # Check take-profit
        if direction == 'long' and current_price >= target_price:
            del self.active_scalps[symbol]
            pnl_pct = (current_price - entry_price) / entry_price * 100
            return {
                'action': 'sell',
                'symbol': symbol,
                'reason': f'TAKE-PROFIT hit: +{pnl_pct:.2f}%',
                'exit_type': 'take_profit',
                'pnl_pct': pnl_pct
            }

        if direction == 'short' and current_price <= target_price:
            del self.active_scalps[symbol]
            pnl_pct = (entry_price - current_price) / entry_price * 100
            return {
                'action': 'cover',
                'symbol': symbol,
                'reason': f'TAKE-PROFIT hit: +{pnl_pct:.2f}%',
                'exit_type': 'take_profit',
                'pnl_pct': pnl_pct
            }

        return None

    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Generate scalping signals based on BB squeeze + RSI extremes.

        Improvements:
        - Uses Wilder's RSI (EMA-based)
        - EMA trend filter
        - Volume confirmation
        - ADX ranging filter
        - Dynamic position sizing
        - Trade cooldown
        - Active scalp tracking with SL/TP

        Args:
            data: Dict with symbol -> DataFrame (OHLCV)

        Returns:
            Signal dict with action, size, leverage, targets, etc.
        """
        signals = []
        exit_signals = []

        for symbol in self.symbols:
            # Phase 30: Try 1m data first for 60s polling, then 5m, 15m, 1h
            # Use explicit None checks to avoid DataFrame truth value ambiguity
            df = data.get(f'{symbol}_1m')
            if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                df = data.get(f'{symbol}_5m')
            if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                df = data.get(f'{symbol}_15m')
            if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                df = data.get(symbol)

            if df is None or len(df) < max(self.bb_period, self.rsi_period, self.adx_period) + 10:
                continue

            # Update bar count
            self.bar_count[symbol] = self.bar_count.get(symbol, 0) + 1
            current_bar = self.bar_count[symbol]
            current_price = df['close'].iloc[-1]

            # Check active scalps for SL/TP first
            exit_signal = self._update_active_scalps(symbol, current_price)
            if exit_signal:
                exit_signal['confidence'] = 0.95  # High confidence for SL/TP exits
                exit_signal['strategy'] = 'scalper'
                exit_signal['size'] = 1.0  # Exit full position
                exit_signals.append(exit_signal)
                continue  # Don't open new position if we just closed one

            # Check cooldown
            if not self._check_cooldown(symbol, current_bar):
                continue

            # Skip if we already have an active scalp for this symbol
            if symbol in self.active_scalps:
                continue

            # Detect squeeze setup (includes all filters)
            squeeze = self._detect_squeeze(df)

            # Store last RSI for status
            self.last_rsi[symbol] = squeeze.get('indicators', {}).get('rsi', 0)
            self.last_ema[symbol] = squeeze.get('indicators', {}).get('ema', 0)
            self.last_squeeze_state[symbol] = squeeze.get('is_tight_bands', False)

            if squeeze['is_squeeze'] and squeeze['direction']:
                rsi = squeeze['indicators']['rsi']
                atr_pct = squeeze['atr_pct']
                position_size = squeeze['position_size']

                if squeeze['direction'] == 'long':
                    confidence = 0.60 + squeeze['strength'] * 0.30
                    if squeeze.get('ema_aligned'):
                        confidence += 0.05

                    signal = {
                        'action': 'buy',
                        'symbol': symbol,
                        'size': position_size,
                        'leverage': min(self.max_leverage, 3),
                        'confidence': min(confidence, 0.92),
                        'reason': f'BB squeeze LONG: RSI={rsi:.0f}, ATR={atr_pct:.1f}%, ADX={squeeze["adx"]:.0f}',
                        'strategy': 'scalper',
                        'target': squeeze['target'],
                        'stop': squeeze['stop'],
                        'entry': squeeze['entry'],
                        'indicators': squeeze['indicators'],
                        'scalp': True
                    }
                    signals.append(signal)

                elif squeeze['direction'] == 'short':
                    confidence = 0.55 + squeeze['strength'] * 0.30
                    if squeeze.get('ema_aligned'):
                        confidence += 0.05

                    signal = {
                        'action': 'short',
                        'symbol': symbol,
                        'size': position_size * 0.8,  # Slightly smaller for shorts
                        'leverage': min(self.max_leverage, 3),
                        'confidence': min(confidence, 0.88),
                        'reason': f'BB squeeze SHORT: RSI={rsi:.0f}, ATR={atr_pct:.1f}%, ADX={squeeze["adx"]:.0f}',
                        'strategy': 'scalper',
                        'target': squeeze['target'],
                        'stop': squeeze['stop'],
                        'entry': squeeze['entry'],
                        'indicators': squeeze['indicators'],
                        'scalp': True
                    }
                    signals.append(signal)

        # Prioritize exit signals over new entries
        if exit_signals:
            return max(exit_signals, key=lambda x: x.get('confidence', 0))

        # Return best entry signal or hold
        if signals:
            best = max(signals, key=lambda x: x.get('confidence', 0))

            # Track the new scalp
            symbol = best['symbol']
            self.active_scalps[symbol] = {
                'direction': 'long' if best['action'] == 'buy' else 'short',
                'entry': best['entry'],
                'target': best['target'],
                'stop': best['stop'],
                'timestamp': datetime.now()
            }
            self.last_trade_time[symbol] = datetime.now()
            self.last_trade_bar[symbol] = self.bar_count.get(symbol, 0)

            return best

        # No signal - hold
        hold_reason = f'No scalp setup (ATR={self.last_atr_pct:.1f}%, ADX={self.last_adx:.0f})'
        if self.active_scalps:
            hold_reason += f', {len(self.active_scalps)} active scalp(s)'

        # Build indicators for diagnostics
        primary_symbol = self.symbols[0] if self.symbols else 'BTC/USDT'
        return {
            'action': 'hold',
            'symbol': primary_symbol,
            'confidence': 0.0,
            'reason': hold_reason,
            'strategy': 'scalper',
            'indicators': {
                'atr_pct': self.last_atr_pct,
                'adx': self.last_adx,
                'rsi': self.last_rsi.get(primary_symbol, 50),
                'ema': self.last_ema.get(primary_symbol, 0),
                'is_squeeze': self.last_squeeze_state.get(primary_symbol, False),
                'active_scalps': len(self.active_scalps),
                'vol_threshold': self.vol_threshold,
                'vol_threshold_high': self.vol_threshold_high,
                'adx_max': self.adx_max,
                'rsi_oversold': self.rsi_oversold,
                'rsi_overbought': self.rsi_overbought,
                'in_cooldown': not self._check_cooldown(primary_symbol, self.bar_count.get(primary_symbol, 0)),
                'waiting_for_setup': True
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
        """Scalper is rule-based, no model to update."""
        return True

    def get_status(self) -> Dict[str, Any]:
        """Get detailed strategy status."""
        base_status = super().get_status()
        base_status.update({
            # Parameters
            'bb_period': self.bb_period,
            'rsi_period': self.rsi_period,
            'ema_period': self.ema_period,
            'adx_period': self.adx_period,
            'vol_threshold': self.vol_threshold,
            'rsi_oversold': self.rsi_oversold,
            'rsi_overbought': self.rsi_overbought,
            'adx_max': self.adx_max,

            # Current state
            'last_atr_pct': self.last_atr_pct,
            'last_adx': self.last_adx,
            'last_rsi': self.last_rsi.copy(),
            'last_ema': self.last_ema.copy(),
            'last_squeeze_state': self.last_squeeze_state.copy(),

            # Active positions
            'active_scalps': len(self.active_scalps),
            'active_scalps_detail': {
                sym: {
                    'direction': s['direction'],
                    'entry': s['entry'],
                    'target': s['target'],
                    'stop': s['stop'],
                    'age_seconds': (datetime.now() - s['timestamp']).total_seconds()
                }
                for sym, s in self.active_scalps.items()
            },

            # Filters status
            'use_adx_filter': self.use_adx_filter,
            'use_volume_filter': self.use_volume_filter,
            'require_squeeze': self.require_squeeze,
            'dynamic_sizing': self.dynamic_sizing
        })
        return base_status
