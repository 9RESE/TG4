"""
Mean Reversion VWAP Strategy - Enhanced Version
Phase 25: Complete rewrite with best practices

Improvements over Phase 16/24:
- 24-hour rolling VWAP for proper session coverage
- VWAP Standard Deviation bands (adaptive to volatility)
- ADX trend filter to avoid mean reversion in trending markets
- RSI divergence detection for higher conviction entries
- Risk-adjusted position sizing based on leverage
- Position tracking to avoid duplicate signals
- Multi-asset support (XRP/USDT and BTC/USDT)
- Partial take-profit logic
- ATR-based dynamic stops

Target: 5-15% monthly in ranging/choppy conditions.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

try:
    import ta.volume as vol
    import ta.momentum as mom
    import ta.trend as trend
    import ta.volatility as volatility
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    print("Warning: ta library not installed. Using fallback calculations.")

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from strategies.base_strategy import BaseStrategy


class PositionState(Enum):
    """Position state tracking."""
    FLAT = "flat"
    LONG = "long"
    SHORT = "short"


@dataclass
class Position:
    """Track an open position."""
    side: PositionState
    entry_price: float
    size: float
    stop_loss: float
    take_profit_1: float  # Partial TP at ±1σ
    take_profit_2: float  # Full TP at VWAP
    entry_time: str
    symbol: str
    partial_closed: bool = False


class MeanReversionVWAP(BaseStrategy):
    """
    Enhanced Mean Reversion Strategy using VWAP Standard Deviation Bands.

    Phase 25 Improvements:
    - 24-hour rolling VWAP (proper session coverage for crypto)
    - Standard deviation bands for volatility-adaptive entries
    - ADX filter to avoid trending markets
    - RSI divergence detection for high-conviction entries
    - Risk-adjusted stops based on leverage
    - Position tracking to prevent duplicate signals

    Entry Conditions (Long):
    - Price below VWAP - 2σ (oversold zone)
    - RSI < 30 OR RSI bullish divergence
    - ADX < 25 (ranging market) OR price between 20/50 SMA
    - Volume > 1.5x average (confirmation)

    Entry Conditions (Short):
    - Price above VWAP + 2σ (overbought zone)
    - RSI > 70 OR RSI bearish divergence
    - ADX < 25 (ranging market)
    - Volume > 1.5x average

    Exit Conditions:
    - Partial TP (50%) at ±1σ band
    - Full TP at VWAP
    - Stop loss at ±3σ or ATR-based (whichever is tighter)
    """

    # Supported symbols for multi-asset trading
    SUPPORTED_SYMBOLS = ['XRP/USDT', 'BTC/USDT']

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.name = 'mean_reversion_vwap'

        # Multi-asset support
        self.symbols = config.get('symbols', ['XRP/USDT'])
        if isinstance(self.symbols, str):
            self.symbols = [self.symbols]

        # Legacy single symbol support
        self.symbol = config.get('symbol', self.symbols[0])

        # ===== VWAP Configuration =====
        # 24-hour window for proper crypto session VWAP
        self.vwap_window = config.get('vwap_window', 24)

        # Standard deviation multipliers for bands
        self.band_1_std = config.get('band_1_std', 1.0)  # ±1σ for partial TP
        self.band_2_std = config.get('band_2_std', 2.0)  # ±2σ for entry
        self.band_3_std = config.get('band_3_std', 3.0)  # ±3σ for stop loss

        # ===== RSI Configuration =====
        self.rsi_window = config.get('rsi_window', 14)
        # Restored proper mean reversion thresholds
        self.rsi_oversold = config.get('rsi_oversold', 30)
        self.rsi_overbought = config.get('rsi_overbought', 70)
        # Divergence lookback
        self.divergence_lookback = config.get('divergence_lookback', 5)

        # ===== Trend Filter (ADX) =====
        self.adx_window = config.get('adx_window', 14)
        self.adx_threshold = config.get('adx_threshold', 25)  # Below = ranging
        self.use_trend_filter = config.get('use_trend_filter', True)

        # SMA trend filter as backup
        self.sma_fast = config.get('sma_fast', 20)
        self.sma_slow = config.get('sma_slow', 50)

        # ===== Volume Filter =====
        self.volume_filter = config.get('volume_filter', True)
        self.volume_mult = config.get('volume_mult', 1.5)  # Restored proper threshold
        self.volume_window = config.get('volume_window', 20)

        # ===== Risk Management =====
        self.max_leverage = config.get('max_leverage', 5)  # Reduced from 7

        # Risk per trade (% of account)
        self.risk_per_trade = config.get('risk_per_trade', 0.03)  # 3% max loss per trade

        # Position sizing
        self.long_size = config.get('long_size', 0.10)
        self.short_size = config.get('short_size', 0.08)

        # ATR-based stops
        self.use_atr_stops = config.get('use_atr_stops', True)
        self.atr_window = config.get('atr_window', 14)
        self.atr_multiplier = config.get('atr_multiplier', 2.0)

        # Partial take profit
        self.partial_tp_pct = config.get('partial_tp_pct', 0.5)  # Close 50% at ±1σ

        # ===== Position Tracking =====
        self.positions: Dict[str, Position] = {}

        # ===== Cached Indicators =====
        self._indicator_cache: Dict[str, Dict[str, Any]] = {}

    def _calculate_vwap_with_bands(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Calculate VWAP with standard deviation bands.

        Returns:
            Tuple of (vwap, upper_2std, lower_2std, std)
        """
        if len(df) < self.vwap_window:
            return pd.Series(), pd.Series(), pd.Series(), pd.Series()

        # Typical price
        typical_price = (df['high'] + df['low'] + df['close']) / 3

        # VWAP calculation
        tp_volume = typical_price * df['volume']
        cumulative_tp_vol = tp_volume.rolling(self.vwap_window).sum()
        cumulative_vol = df['volume'].rolling(self.vwap_window).sum()

        vwap = cumulative_tp_vol / cumulative_vol

        # Standard deviation of price from VWAP
        # Using squared deviations weighted by volume
        squared_dev = ((typical_price - vwap) ** 2) * df['volume']
        variance = squared_dev.rolling(self.vwap_window).sum() / cumulative_vol
        std = np.sqrt(variance)

        # Bands
        upper_1std = vwap + (std * self.band_1_std)
        lower_1std = vwap - (std * self.band_1_std)
        upper_2std = vwap + (std * self.band_2_std)
        lower_2std = vwap - (std * self.band_2_std)
        upper_3std = vwap + (std * self.band_3_std)
        lower_3std = vwap - (std * self.band_3_std)

        return {
            'vwap': vwap,
            'std': std,
            'upper_1std': upper_1std,
            'lower_1std': lower_1std,
            'upper_2std': upper_2std,
            'lower_2std': lower_2std,
            'upper_3std': upper_3std,
            'lower_3std': lower_3std
        }

    def _calculate_rsi(self, df: pd.DataFrame) -> pd.Series:
        """Calculate RSI with proper handling."""
        if TA_AVAILABLE and len(df) >= self.rsi_window:
            rsi_indicator = mom.RSIIndicator(
                close=df['close'],
                window=self.rsi_window
            )
            return rsi_indicator.rsi()
        else:
            return self._calculate_rsi_fallback(df)

    def _calculate_rsi_fallback(self, df: pd.DataFrame) -> pd.Series:
        """Fallback RSI calculation with proper edge case handling."""
        delta = df['close'].diff()
        gain = delta.clip(lower=0).rolling(window=self.rsi_window).mean()
        loss = (-delta.clip(upper=0)).rolling(window=self.rsi_window).mean()

        # Proper handling of zero loss (avoid division issues)
        rs = pd.Series(index=df.index, dtype=float)

        # Where loss is 0 but gain > 0: RSI = 100
        # Where both are 0: RSI = 50
        # Normal case: RSI = 100 - (100 / (1 + rs))
        mask_zero_loss = (loss == 0) | (loss.isna())
        mask_zero_gain = (gain == 0) | (gain.isna())

        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        # Handle edge cases
        rsi = rsi.where(~(mask_zero_loss & ~mask_zero_gain), 100)
        rsi = rsi.where(~(mask_zero_loss & mask_zero_gain), 50)

        return rsi

    def _calculate_adx(self, df: pd.DataFrame) -> pd.Series:
        """Calculate ADX for trend strength."""
        if TA_AVAILABLE and len(df) >= self.adx_window:
            adx_indicator = trend.ADXIndicator(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=self.adx_window
            )
            return adx_indicator.adx()
        else:
            return self._calculate_adx_fallback(df)

    def _calculate_adx_fallback(self, df: pd.DataFrame) -> pd.Series:
        """Simplified ADX fallback calculation."""
        high = df['high']
        low = df['low']
        close = df['close']

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(self.adx_window).mean()

        # Directional Movement
        up_move = high - high.shift()
        down_move = low.shift() - low

        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)

        plus_di = 100 * (plus_dm.rolling(self.adx_window).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(self.adx_window).mean() / atr)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(self.adx_window).mean()

        return adx

    def _calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range."""
        if TA_AVAILABLE and len(df) >= self.atr_window:
            atr_indicator = volatility.AverageTrueRange(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=self.atr_window
            )
            return atr_indicator.average_true_range()
        else:
            high = df['high']
            low = df['low']
            close = df['close']

            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            return tr.rolling(self.atr_window).mean()

    def _detect_rsi_divergence(self, df: pd.DataFrame, rsi: pd.Series) -> Dict[str, bool]:
        """
        Detect RSI divergence (bullish and bearish).

        Bullish divergence: Price makes lower low, RSI makes higher low
        Bearish divergence: Price makes higher high, RSI makes lower high
        """
        result = {'bullish': False, 'bearish': False}

        if len(df) < self.divergence_lookback + 2:
            return result

        lookback = self.divergence_lookback

        # Get recent price and RSI values
        recent_close = df['close'].iloc[-lookback:]
        recent_rsi = rsi.iloc[-lookback:]

        if recent_close.isna().any() or recent_rsi.isna().any():
            return result

        # Find local minima/maxima
        price_min_idx = recent_close.idxmin()
        price_max_idx = recent_close.idxmax()
        rsi_min_idx = recent_rsi.idxmin()
        rsi_max_idx = recent_rsi.idxmax()

        current_price = df['close'].iloc[-1]
        current_rsi = rsi.iloc[-1]

        # Bullish divergence: price at lower low, RSI at higher low
        if (current_price <= recent_close.min() * 1.005 and  # Near/at low
            current_rsi > recent_rsi.min() * 1.02):  # RSI higher than its low
            result['bullish'] = True

        # Bearish divergence: price at higher high, RSI at lower high
        if (current_price >= recent_close.max() * 0.995 and  # Near/at high
            current_rsi < recent_rsi.max() * 0.98):  # RSI lower than its high
            result['bearish'] = True

        return result

    def _check_volume_filter(self, df: pd.DataFrame) -> Tuple[bool, float]:
        """
        Check if current volume exceeds threshold.

        Returns:
            Tuple of (passes_filter, volume_ratio)
        """
        if not self.volume_filter:
            return True, 1.0

        if len(df) < self.volume_window + 1:
            return True, 1.0

        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].iloc[-self.volume_window-1:-1].mean()

        if avg_volume <= 0:
            return True, 1.0

        volume_ratio = current_volume / avg_volume
        passes = volume_ratio >= self.volume_mult

        return passes, volume_ratio

    def _check_trend_filter(self, df: pd.DataFrame, adx: float) -> Tuple[bool, str]:
        """
        Check if market is ranging (suitable for mean reversion).

        Returns:
            Tuple of (is_ranging, trend_description)
        """
        if not self.use_trend_filter:
            return True, "filter_disabled"

        # Primary check: ADX
        if pd.isna(adx):
            return True, "adx_unavailable"

        if adx < self.adx_threshold:
            return True, f"ranging_adx_{adx:.1f}"

        # Secondary check: Price between SMAs (choppy)
        if len(df) >= self.sma_slow:
            sma_fast = df['close'].rolling(self.sma_fast).mean().iloc[-1]
            sma_slow = df['close'].rolling(self.sma_slow).mean().iloc[-1]
            current_price = df['close'].iloc[-1]

            # Price between SMAs = choppy/ranging
            if min(sma_fast, sma_slow) <= current_price <= max(sma_fast, sma_slow):
                return True, f"choppy_between_smas"

        return False, f"trending_adx_{adx:.1f}"

    def _calculate_position_size(self, price: float, stop_distance: float) -> float:
        """
        Calculate position size based on risk management.

        Risk-adjusted sizing: position size = risk_amount / stop_distance
        """
        if stop_distance <= 0:
            return self.long_size

        # With leverage, effective stop distance is amplified
        effective_stop = stop_distance * self.max_leverage

        # Position size to risk only risk_per_trade of account
        # size = risk_per_trade / effective_stop
        risk_adjusted_size = self.risk_per_trade / effective_stop

        # Cap at configured max size
        return min(risk_adjusted_size, self.long_size)

    def _calculate_stops_and_targets(
        self,
        side: str,
        entry_price: float,
        vwap_bands: Dict[str, pd.Series],
        atr: float
    ) -> Dict[str, float]:
        """
        Calculate stop loss and take profit levels.

        Uses tighter of:
        - VWAP ±3σ band
        - ATR-based stop (2x ATR)
        """
        vwap = vwap_bands['vwap'].iloc[-1]

        if side == 'long':
            # Stop loss: lower of 3σ band or ATR-based
            band_stop = vwap_bands['lower_3std'].iloc[-1]
            atr_stop = entry_price - (atr * self.atr_multiplier) if self.use_atr_stops else band_stop
            stop_loss = max(band_stop, atr_stop)  # Use tighter (higher) stop for longs

            # Take profits
            tp_1 = vwap_bands['lower_1std'].iloc[-1]  # Partial at -1σ
            tp_2 = vwap  # Full at VWAP

        else:  # short
            band_stop = vwap_bands['upper_3std'].iloc[-1]
            atr_stop = entry_price + (atr * self.atr_multiplier) if self.use_atr_stops else band_stop
            stop_loss = min(band_stop, atr_stop)  # Use tighter (lower) stop for shorts

            tp_1 = vwap_bands['upper_1std'].iloc[-1]
            tp_2 = vwap

        return {
            'stop_loss': stop_loss,
            'take_profit_1': tp_1,
            'take_profit_2': tp_2,
            'stop_distance_pct': abs(entry_price - stop_loss) / entry_price
        }

    def _get_position(self, symbol: str) -> Optional[Position]:
        """Get current position for symbol."""
        return self.positions.get(symbol)

    def _has_position(self, symbol: str) -> bool:
        """Check if we have an open position."""
        pos = self._get_position(symbol)
        return pos is not None and pos.side != PositionState.FLAT

    def _check_exit_conditions(
        self,
        symbol: str,
        current_price: float,
        vwap_bands: Dict[str, pd.Series]
    ) -> Optional[Dict[str, Any]]:
        """
        Check if current position should be exited.

        Returns exit signal if conditions met, None otherwise.
        """
        pos = self._get_position(symbol)
        if not pos or pos.side == PositionState.FLAT:
            return None

        signal = None

        if pos.side == PositionState.LONG:
            # Check stop loss
            if current_price <= pos.stop_loss:
                signal = {
                    'action': 'sell',
                    'reason': f'Stop loss hit: ${current_price:.4f} <= ${pos.stop_loss:.4f}',
                    'exit_type': 'stop_loss',
                    'size': 1.0  # Full exit
                }

            # Check partial TP (at -1σ moving toward VWAP)
            elif not pos.partial_closed and current_price >= pos.take_profit_1:
                signal = {
                    'action': 'sell',
                    'reason': f'Partial TP at -1σ: ${current_price:.4f}',
                    'exit_type': 'partial_tp',
                    'size': self.partial_tp_pct
                }

            # Check full TP (at VWAP)
            elif current_price >= pos.take_profit_2:
                signal = {
                    'action': 'sell',
                    'reason': f'Full TP at VWAP: ${current_price:.4f}',
                    'exit_type': 'full_tp',
                    'size': 1.0
                }

        elif pos.side == PositionState.SHORT:
            # Check stop loss
            if current_price >= pos.stop_loss:
                signal = {
                    'action': 'buy',  # Buy to cover short
                    'reason': f'Stop loss hit: ${current_price:.4f} >= ${pos.stop_loss:.4f}',
                    'exit_type': 'stop_loss',
                    'size': 1.0
                }

            # Check partial TP
            elif not pos.partial_closed and current_price <= pos.take_profit_1:
                signal = {
                    'action': 'buy',
                    'reason': f'Partial TP at +1σ: ${current_price:.4f}',
                    'exit_type': 'partial_tp',
                    'size': self.partial_tp_pct
                }

            # Check full TP
            elif current_price <= pos.take_profit_2:
                signal = {
                    'action': 'buy',
                    'reason': f'Full TP at VWAP: ${current_price:.4f}',
                    'exit_type': 'full_tp',
                    'size': 1.0
                }

        return signal

    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Generate mean reversion signals based on VWAP bands, RSI, and filters.

        Multi-asset: Checks all configured symbols and returns the highest confidence signal.
        """
        best_signal = None
        best_confidence = 0

        for symbol in self.symbols:
            if symbol not in data:
                continue

            signal = self._generate_signal_for_symbol(symbol, data[symbol])

            if signal and signal.get('confidence', 0) > best_confidence:
                best_confidence = signal['confidence']
                best_signal = signal

        # Return best signal or default hold
        if best_signal:
            return best_signal

        return {
            'action': 'hold',
            'symbol': self.symbol,
            'size': 0.0,
            'leverage': 1,
            'confidence': 0.0,
            'reason': 'No valid signals across symbols'
        }

    def _generate_signal_for_symbol(self, symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate signal for a specific symbol."""

        min_periods = max(self.vwap_window, self.rsi_window, self.adx_window,
                         self.volume_window, self.sma_slow) + 5

        if len(df) < min_periods:
            return {
                'action': 'hold',
                'symbol': symbol,
                'size': 0.0,
                'leverage': 1,
                'confidence': 0.0,
                'reason': f'Insufficient data ({len(df)} < {min_periods})'
            }

        # Calculate all indicators
        vwap_bands = self._calculate_vwap_with_bands(df)
        rsi = self._calculate_rsi(df)
        adx = self._calculate_adx(df)
        atr = self._calculate_atr(df)
        divergence = self._detect_rsi_divergence(df, rsi)
        volume_ok, volume_ratio = self._check_volume_filter(df)

        # Get latest values
        current_price = df['close'].iloc[-1]
        current_rsi = rsi.iloc[-1]
        current_adx = adx.iloc[-1]
        current_atr = atr.iloc[-1]
        current_vwap = vwap_bands['vwap'].iloc[-1]

        # Handle NaN indicators
        if pd.isna(current_vwap) or pd.isna(current_rsi):
            return {
                'action': 'hold',
                'symbol': symbol,
                'size': 0.0,
                'leverage': 1,
                'confidence': 0.0,
                'reason': 'Indicators not ready (NaN)',
                'indicators': {
                    'vwap': current_vwap,
                    'rsi': current_rsi,
                    'price': current_price
                }
            }

        # Check trend filter
        is_ranging, trend_desc = self._check_trend_filter(df, current_adx)

        # Calculate VWAP deviation
        vwap_deviation = (current_price - current_vwap) / current_vwap if current_vwap > 0 else 0

        # Get band values
        upper_2std = vwap_bands['upper_2std'].iloc[-1]
        lower_2std = vwap_bands['lower_2std'].iloc[-1]
        std = vwap_bands['std'].iloc[-1]
        std_pct = std / current_vwap if current_vwap > 0 else 0

        # Build indicators dict for logging
        indicators = {
            'vwap': current_vwap,
            'std': std,
            'std_pct': std_pct,
            'upper_2std': upper_2std,
            'lower_2std': lower_2std,
            'rsi': current_rsi,
            'adx': current_adx,
            'atr': current_atr,
            'price': current_price,
            'vwap_deviation': vwap_deviation,
            'volume_ratio': volume_ratio,
            'volume_ok': volume_ok,
            'is_ranging': is_ranging,
            'trend_desc': trend_desc,
            'divergence': divergence
        }

        # Default signal
        signal = {
            'action': 'hold',
            'symbol': symbol,
            'asset': symbol.split('/')[0],
            'size': 0.0,
            'leverage': 1,
            'confidence': 0.0,
            'reason': 'No signal',
            'indicators': indicators
        }

        # First check for exit signals on existing positions
        exit_signal = self._check_exit_conditions(symbol, current_price, vwap_bands)
        if exit_signal:
            pos = self._get_position(symbol)
            exit_signal['symbol'] = symbol
            exit_signal['asset'] = symbol.split('/')[0]
            exit_signal['confidence'] = 0.9  # High confidence for managed exits
            exit_signal['leverage'] = 1
            exit_signal['indicators'] = indicators

            # Handle partial close
            if exit_signal.get('exit_type') == 'partial_tp':
                exit_signal['size'] = pos.size * self.partial_tp_pct
                pos.partial_closed = True
                pos.size *= (1 - self.partial_tp_pct)
            else:
                exit_signal['size'] = pos.size if pos else self.long_size
                # Clear position on full exit
                if symbol in self.positions:
                    del self.positions[symbol]

            return exit_signal

        # Skip new entry signals if we already have a position
        if self._has_position(symbol):
            signal['reason'] = f'Already in {self.positions[symbol].side.value} position'
            return signal

        # ===== LONG SIGNAL CONDITIONS =====
        # Price below -2σ band AND (RSI oversold OR bullish divergence) AND ranging AND volume
        long_price_condition = current_price < lower_2std
        long_rsi_condition = current_rsi < self.rsi_oversold or divergence['bullish']

        if long_price_condition and long_rsi_condition and is_ranging and volume_ok:
            # Calculate stops and targets
            stops = self._calculate_stops_and_targets('long', current_price, vwap_bands, current_atr)

            # Risk-adjusted position size
            position_size = self._calculate_position_size(current_price, stops['stop_distance_pct'])

            # Confidence scoring
            # Base: 0.5
            # +0.15 for RSI extremity
            # +0.15 for price beyond 2σ
            # +0.10 for divergence
            # +0.10 for strong volume
            rsi_score = max(0, (self.rsi_oversold - current_rsi) / self.rsi_oversold) * 0.15

            price_below_2std = (lower_2std - current_price) / std if std > 0 else 0
            price_score = min(price_below_2std, 1) * 0.15

            div_score = 0.10 if divergence['bullish'] else 0
            vol_score = min((volume_ratio - self.volume_mult) / self.volume_mult, 1) * 0.10 if volume_ratio > self.volume_mult else 0

            confidence = min(0.5 + rsi_score + price_score + div_score + vol_score, 0.95)

            reason_parts = [f'Long: price ${current_price:.4f} < -2σ ${lower_2std:.4f}']
            if current_rsi < self.rsi_oversold:
                reason_parts.append(f'RSI {current_rsi:.1f}')
            if divergence['bullish']:
                reason_parts.append('bullish div')
            reason_parts.append(f'vol {volume_ratio:.1f}x')

            signal = {
                'action': 'buy',
                'symbol': symbol,
                'asset': symbol.split('/')[0],
                'size': position_size,
                'leverage': self.max_leverage,
                'confidence': confidence,
                'reason': ', '.join(reason_parts),
                'indicators': indicators,
                'stop_loss': stops['stop_loss'],
                'take_profit': stops['take_profit_2'],
                'take_profit_partial': stops['take_profit_1']
            }

            # Track position
            from datetime import datetime
            self.positions[symbol] = Position(
                side=PositionState.LONG,
                entry_price=current_price,
                size=position_size,
                stop_loss=stops['stop_loss'],
                take_profit_1=stops['take_profit_1'],
                take_profit_2=stops['take_profit_2'],
                entry_time=datetime.now().isoformat(),
                symbol=symbol
            )

        # ===== SHORT SIGNAL CONDITIONS =====
        # Price above +2σ band AND (RSI overbought OR bearish divergence) AND ranging AND volume
        elif current_price > upper_2std and (current_rsi > self.rsi_overbought or divergence['bearish']) and is_ranging and volume_ok:
            short_price_condition = True
            short_rsi_condition = True
            stops = self._calculate_stops_and_targets('short', current_price, vwap_bands, current_atr)
            position_size = self._calculate_position_size(current_price, stops['stop_distance_pct'])

            rsi_score = max(0, (current_rsi - self.rsi_overbought) / (100 - self.rsi_overbought)) * 0.15

            price_above_2std = (current_price - upper_2std) / std if std > 0 else 0
            price_score = min(price_above_2std, 1) * 0.15

            div_score = 0.10 if divergence['bearish'] else 0
            vol_score = min((volume_ratio - self.volume_mult) / self.volume_mult, 1) * 0.10 if volume_ratio > self.volume_mult else 0

            confidence = min(0.5 + rsi_score + price_score + div_score + vol_score, 0.95)

            reason_parts = [f'Short: price ${current_price:.4f} > +2σ ${upper_2std:.4f}']
            if current_rsi > self.rsi_overbought:
                reason_parts.append(f'RSI {current_rsi:.1f}')
            if divergence['bearish']:
                reason_parts.append('bearish div')
            reason_parts.append(f'vol {volume_ratio:.1f}x')

            signal = {
                'action': 'sell',
                'symbol': symbol,
                'asset': symbol.split('/')[0],
                'size': position_size,
                'leverage': self.max_leverage,
                'confidence': confidence,
                'reason': ', '.join(reason_parts),
                'indicators': indicators,
                'stop_loss': stops['stop_loss'],
                'take_profit': stops['take_profit_2'],
                'take_profit_partial': stops['take_profit_1']
            }

            from datetime import datetime
            self.positions[symbol] = Position(
                side=PositionState.SHORT,
                entry_price=current_price,
                size=position_size,
                stop_loss=stops['stop_loss'],
                take_profit_1=stops['take_profit_1'],
                take_profit_2=stops['take_profit_2'],
                entry_time=datetime.now().isoformat(),
                symbol=symbol
            )

        # Log near-miss signals for analysis
        elif not is_ranging:
            signal['reason'] = f'Trend filter blocked: {trend_desc}'
        elif not volume_ok:
            signal['reason'] = f'Volume filter blocked: {volume_ratio:.1f}x < {self.volume_mult}x'
        elif current_price < lower_2std and not (current_rsi < self.rsi_oversold or divergence['bullish']):
            signal['reason'] = f'Price at -2σ but RSI {current_rsi:.1f} not oversold'
        elif current_price > upper_2std and not (current_rsi > self.rsi_overbought or divergence['bearish']):
            signal['reason'] = f'Price at +2σ but RSI {current_rsi:.1f} not overbought'

        return signal

    def on_order_filled(self, order: Dict[str, Any]) -> None:
        """
        Update position tracking when orders are filled.
        """
        symbol = order.get('symbol', '')
        action = order.get('action', '')

        if action in ['sell', 'cover'] and symbol in self.positions:
            # Check if this is a full exit
            if order.get('exit_type') in ['stop_loss', 'full_tp']:
                del self.positions[symbol]
            elif order.get('exit_type') == 'partial_tp':
                self.positions[symbol].partial_closed = True

    def update_model(self, data: Optional[Dict[str, pd.DataFrame]] = None) -> bool:
        """No ML model to update - pure indicator-based strategy."""
        return True

    def get_status(self) -> Dict[str, Any]:
        """Get strategy status with enhanced metrics."""
        base_status = super().get_status()

        positions_info = {}
        for symbol, pos in self.positions.items():
            positions_info[symbol] = {
                'side': pos.side.value,
                'entry_price': pos.entry_price,
                'size': pos.size,
                'stop_loss': pos.stop_loss,
                'take_profit_1': pos.take_profit_1,
                'take_profit_2': pos.take_profit_2,
                'partial_closed': pos.partial_closed
            }

        base_status.update({
            'symbols': self.symbols,
            'vwap_window': self.vwap_window,
            'band_multipliers': {
                '1std': self.band_1_std,
                '2std': self.band_2_std,
                '3std': self.band_3_std
            },
            'rsi_window': self.rsi_window,
            'rsi_thresholds': (self.rsi_oversold, self.rsi_overbought),
            'adx_threshold': self.adx_threshold,
            'volume_filter': self.volume_filter,
            'volume_mult': self.volume_mult,
            'risk_per_trade': self.risk_per_trade,
            'max_leverage': self.max_leverage,
            'use_atr_stops': self.use_atr_stops,
            'positions': positions_info,
            'ta_available': TA_AVAILABLE
        })
        return base_status
