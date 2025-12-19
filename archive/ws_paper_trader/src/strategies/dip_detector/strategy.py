"""
Phase 26: Enhanced Dip Detector Strategy
Detects high-probability dip entries using RSI divergence, volume surge,
drawdown analysis, BTC correlation, and multi-timeframe confirmation.

Improvements over Phase 24:
- Fixed RSI calculation (Wilder's smoothing)
- RSI divergence detection (bullish/bearish)
- BTC correlation filter
- Multi-timeframe confirmation
- Position tracking with cooldown
- Stop-loss and take-profit integration
- Multi-asset support (not just XRP)
- Dynamic thresholds based on volatility
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from strategies.base_strategy import BaseStrategy
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta


class DipDetector(BaseStrategy):
    """
    Enhanced dip detection strategy for high-probability entries.

    Features:
    - Wilder's RSI with divergence detection
    - Volume surge with time-of-day normalization
    - Drawdown from multiple lookback periods
    - BTC correlation filter (XRP follows BTC)
    - Multi-timeframe confirmation (1h + 4h alignment)
    - Position tracking with cooldown
    - Integrated stop-loss and take-profit
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Primary symbol (default XRP, but configurable)
        self.symbol = config.get('symbol', 'XRP/USDT')
        self.btc_symbol = config.get('btc_symbol', 'BTC/USDT')

        # RSI parameters (relaxed for crypto volatility)
        self.rsi_period = config.get('rsi_period', 14)
        self.rsi_oversold = config.get('rsi_oversold', 38)  # Relaxed from 30
        self.rsi_overbought = config.get('rsi_overbought', 65)  # Relaxed from 70
        self.rsi_extreme_oversold = config.get('rsi_extreme_oversold', 25)  # Strong signal

        # RSI Divergence detection
        self.use_divergence = config.get('use_divergence', True)
        self.divergence_lookback = config.get('divergence_lookback', 14)
        self.divergence_min_bars = config.get('divergence_min_bars', 3)

        # Volume parameters
        self.volume_lookback = config.get('volume_lookback', 20)
        self.volume_surge_threshold = config.get('volume_surge_threshold', 1.5)
        self.volume_capitulation_threshold = config.get('volume_capitulation_threshold', 2.5)

        # Drawdown parameters
        self.drawdown_lookback = config.get('drawdown_lookback', 20)
        self.drawdown_threshold = config.get('drawdown_threshold', -0.05)  # 5% dip
        self.drawdown_strong = config.get('drawdown_strong', -0.08)  # 8% = strong dip

        # ATR for volatility context
        self.atr_period = config.get('atr_period', 14)
        self.max_atr_for_leverage = config.get('max_atr_for_leverage', 0.08)
        self.min_atr_for_trade = config.get('min_atr_for_trade', 0.005)  # Min volatility

        # BTC Correlation Filter
        self.use_btc_filter = config.get('use_btc_filter', True)
        self.btc_stable_threshold = config.get('btc_stable_threshold', 0.02)  # 2% move
        self.btc_correlation_period = config.get('btc_correlation_period', 24)

        # Multi-timeframe confirmation
        self.use_mtf = config.get('use_mtf', True)
        self.mtf_rsi_threshold = config.get('mtf_rsi_threshold', 45)  # Higher TF RSI

        # Position tracking and cooldown
        self.cooldown_minutes = config.get('cooldown_minutes', 30)
        self.max_positions = config.get('max_positions', 2)
        self.last_signal_time: Dict[str, datetime] = {}
        self.position_count = 0

        # Risk management
        self.stop_loss_pct = config.get('stop_loss_pct', 0.03)  # 3% stop
        self.take_profit_pct = config.get('take_profit_pct', 0.06)  # 6% target (2:1 RRR)
        self.use_trailing_stop = config.get('use_trailing_stop', False)
        self.trailing_stop_pct = config.get('trailing_stop_pct', 0.02)

        # Multi-asset support
        self.symbols = config.get('symbols', ['XRP/USDT'])
        if isinstance(self.symbols, str):
            self.symbols = [self.symbols]

        # State tracking
        self.last_rsi: Dict[str, float] = {}
        self.last_drawdown: Dict[str, float] = {}
        self.last_volume_surge: Dict[str, float] = {}
        self.entry_prices: Dict[str, float] = {}
        self.highest_since_entry: Dict[str, float] = {}

        # RSI history for divergence detection
        self.rsi_history: Dict[str, List[float]] = {}
        self.price_history: Dict[str, List[float]] = {}

    def _calculate_rsi_wilders(self, close: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Calculate RSI using Wilder's smoothing (EMA-based).
        Returns current RSI and full RSI series for divergence detection.
        """
        if len(close) < self.rsi_period + 1:
            return 50.0, np.array([50.0])

        deltas = np.diff(close)
        gains = np.where(deltas > 0, deltas, 0).astype(float)
        losses = np.where(deltas < 0, -deltas, 0).astype(float)

        # Initialize with SMA for first period
        avg_gain = np.mean(gains[:self.rsi_period])
        avg_loss = np.mean(losses[:self.rsi_period])

        rsi_values = []

        # Wilder's smoothing: prev * (n-1)/n + current/n
        for i in range(self.rsi_period, len(gains)):
            avg_gain = (avg_gain * (self.rsi_period - 1) + gains[i]) / self.rsi_period
            avg_loss = (avg_loss * (self.rsi_period - 1) + losses[i]) / self.rsi_period

            if avg_loss == 0:
                rsi = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))

            rsi_values.append(rsi)

        return rsi_values[-1] if rsi_values else 50.0, np.array(rsi_values)

    def _detect_bullish_divergence(self, price: np.ndarray, rsi: np.ndarray) -> Tuple[bool, float]:
        """
        Detect bullish RSI divergence: price makes lower lows, RSI makes higher lows.
        Returns (is_divergence, strength 0-1)
        """
        if len(price) < self.divergence_lookback or len(rsi) < self.divergence_lookback:
            return False, 0.0

        # Use recent data for divergence detection
        price_recent = price[-self.divergence_lookback:]
        rsi_recent = rsi[-self.divergence_lookback:]

        # Find local lows (simple approach: compare to neighbors)
        price_lows = []
        rsi_lows = []

        for i in range(1, len(price_recent) - 1):
            # Check if this is a local low
            if price_recent[i] <= price_recent[i-1] and price_recent[i] <= price_recent[i+1]:
                price_lows.append((i, price_recent[i]))
                rsi_lows.append((i, rsi_recent[i]))

        if len(price_lows) < 2:
            return False, 0.0

        # Check for bullish divergence: price lower low, RSI higher low
        # Compare last two lows
        for i in range(len(price_lows) - 1):
            price_low1 = price_lows[i][1]
            price_low2 = price_lows[i + 1][1]
            rsi_low1 = rsi_lows[i][1]
            rsi_low2 = rsi_lows[i + 1][1]

            # Bullish divergence: price makes lower low, RSI makes higher low
            if price_low2 < price_low1 and rsi_low2 > rsi_low1:
                # Calculate strength based on divergence magnitude
                price_diff = (price_low1 - price_low2) / price_low1
                rsi_diff = rsi_low2 - rsi_low1
                strength = min(1.0, (price_diff * 10 + rsi_diff / 30))
                return True, strength

        return False, 0.0

    def _detect_bearish_divergence(self, price: np.ndarray, rsi: np.ndarray) -> Tuple[bool, float]:
        """
        Detect bearish RSI divergence: price makes higher highs, RSI makes lower highs.
        Returns (is_divergence, strength 0-1)
        """
        if len(price) < self.divergence_lookback or len(rsi) < self.divergence_lookback:
            return False, 0.0

        price_recent = price[-self.divergence_lookback:]
        rsi_recent = rsi[-self.divergence_lookback:]

        # Find local highs
        price_highs = []
        rsi_highs = []

        for i in range(1, len(price_recent) - 1):
            if price_recent[i] >= price_recent[i-1] and price_recent[i] >= price_recent[i+1]:
                price_highs.append((i, price_recent[i]))
                rsi_highs.append((i, rsi_recent[i]))

        if len(price_highs) < 2:
            return False, 0.0

        # Check for bearish divergence
        for i in range(len(price_highs) - 1):
            price_high1 = price_highs[i][1]
            price_high2 = price_highs[i + 1][1]
            rsi_high1 = rsi_highs[i][1]
            rsi_high2 = rsi_highs[i + 1][1]

            # Bearish divergence: price makes higher high, RSI makes lower high
            if price_high2 > price_high1 and rsi_high2 < rsi_high1:
                price_diff = (price_high2 - price_high1) / price_high1
                rsi_diff = rsi_high1 - rsi_high2
                strength = min(1.0, (price_diff * 10 + rsi_diff / 30))
                return True, strength

        return False, 0.0

    def _calculate_volume_surge(self, volume: np.ndarray) -> float:
        """Calculate volume surge ratio vs recent average."""
        if len(volume) < self.volume_lookback:
            return 1.0

        avg_volume = np.mean(volume[-self.volume_lookback:-1])
        current_volume = volume[-1]

        return current_volume / avg_volume if avg_volume > 0 else 1.0

    def _calculate_drawdown(self, close: np.ndarray) -> float:
        """Calculate current drawdown from recent high."""
        if len(close) < self.drawdown_lookback:
            return 0.0

        recent_high = np.max(close[-self.drawdown_lookback:])
        current = close[-1]

        return (current - recent_high) / recent_high if recent_high > 0 else 0.0

    def _calculate_atr_pct(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> float:
        """Calculate ATR as percentage of price using Wilder's smoothing."""
        if len(close) < self.atr_period + 1:
            return 0.05

        # Calculate True Range
        tr_list = []
        for i in range(-self.atr_period, 0):
            tr = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
            tr_list.append(tr)

        # Wilder's smoothing for ATR
        atr = tr_list[0]
        for i in range(1, len(tr_list)):
            atr = (atr * (self.atr_period - 1) + tr_list[i]) / self.atr_period

        return atr / close[-1] if close[-1] > 0 else 0.05

    def _check_btc_correlation(self, data: Dict[str, pd.DataFrame]) -> Tuple[bool, str]:
        """
        Check BTC status for correlation filter.
        Returns (is_safe_to_buy, reason)

        Safe to buy XRP dip when:
        - BTC is stable (not dumping)
        - BTC is rising (even better)
        - BTC RSI is not extremely oversold (potential more downside)
        """
        btc_key = None
        for key in data.keys():
            if 'BTC' in key.upper() and '_' not in key:  # Main timeframe only
                btc_key = key
                break

        if not btc_key or btc_key not in data:
            return True, "No BTC data - proceeding without filter"

        btc_df = data[btc_key]
        if len(btc_df) < self.btc_correlation_period:
            return True, "Insufficient BTC data"

        btc_close = btc_df['close'].values

        # Calculate BTC move over correlation period
        btc_change = (btc_close[-1] - btc_close[-self.btc_correlation_period]) / btc_close[-self.btc_correlation_period]

        # Calculate BTC RSI
        btc_rsi, _ = self._calculate_rsi_wilders(btc_close)

        # Decision logic
        if btc_change < -self.btc_stable_threshold:
            # BTC is dumping - risky to buy XRP dip
            if btc_rsi < 30:
                return False, f"BTC dumping ({btc_change*100:.1f}%) with RSI {btc_rsi:.0f} - wait"
            else:
                return False, f"BTC down {btc_change*100:.1f}% - caution"

        if btc_change > self.btc_stable_threshold:
            # BTC rising - good environment for dip buying
            return True, f"BTC up {btc_change*100:.1f}% - favorable"

        # BTC stable
        return True, f"BTC stable ({btc_change*100:.1f}%)"

    def _check_mtf_confirmation(self, data: Dict[str, pd.DataFrame], symbol: str) -> Tuple[bool, str]:
        """
        Check multi-timeframe confirmation.
        Look for higher timeframe (4h/15m suffix) RSI alignment.
        """
        # Look for 15m or 4h data
        mtf_key = None
        for suffix in ['_15m', '_4h']:
            test_key = f"{symbol}{suffix}"
            if test_key in data:
                mtf_key = test_key
                break

        if not mtf_key:
            return True, "No MTF data available"

        mtf_df = data[mtf_key]
        if len(mtf_df) < self.rsi_period + 1:
            return True, "Insufficient MTF data"

        mtf_close = mtf_df['close'].values
        mtf_rsi, _ = self._calculate_rsi_wilders(mtf_close)

        if mtf_rsi < self.mtf_rsi_threshold:
            return True, f"MTF RSI {mtf_rsi:.0f} confirms oversold"
        else:
            return False, f"MTF RSI {mtf_rsi:.0f} not oversold"

    def _check_cooldown(self, symbol: str) -> bool:
        """Check if cooldown period has passed since last signal."""
        if symbol not in self.last_signal_time:
            return True

        elapsed = datetime.now() - self.last_signal_time[symbol]
        return elapsed > timedelta(minutes=self.cooldown_minutes)

    def _calculate_confidence(self,
                             rsi: float,
                             volume_surge: float,
                             drawdown: float,
                             has_divergence: bool,
                             divergence_strength: float,
                             btc_favorable: bool) -> float:
        """
        Calculate confidence score (0-1) based on multiple factors.
        Enhanced with divergence and BTC correlation bonuses.
        """
        confidence = 0.0

        # RSI contribution (0-0.30)
        if rsi < self.rsi_extreme_oversold:
            confidence += 0.30
        elif rsi < 30:
            confidence += 0.25
        elif rsi < 35:
            confidence += 0.20
        elif rsi < self.rsi_oversold:
            confidence += 0.15
        elif rsi < 45:
            confidence += 0.05

        # Volume surge contribution (0-0.20)
        if volume_surge > self.volume_capitulation_threshold:
            confidence += 0.20  # Capitulation volume
        elif volume_surge > 2.0:
            confidence += 0.15
        elif volume_surge > self.volume_surge_threshold:
            confidence += 0.10

        # Drawdown contribution (0-0.20)
        if drawdown < self.drawdown_strong:
            confidence += 0.20
        elif drawdown < self.drawdown_threshold:
            confidence += 0.15
        elif drawdown < -0.03:
            confidence += 0.10

        # RSI Divergence bonus (0-0.20) - HIGH VALUE SIGNAL
        if has_divergence:
            confidence += 0.10 + (divergence_strength * 0.10)

        # BTC correlation bonus (0-0.10)
        if btc_favorable:
            confidence += 0.10

        return min(1.0, confidence)

    def _check_stop_loss_take_profit(self, symbol: str, current_price: float) -> Optional[Dict[str, Any]]:
        """
        Check if we should exit based on stop-loss or take-profit.
        Returns exit signal if triggered, None otherwise.
        """
        if symbol not in self.entry_prices:
            return None

        entry_price = self.entry_prices[symbol]
        pnl_pct = (current_price - entry_price) / entry_price

        # Update highest price for trailing stop
        if symbol not in self.highest_since_entry:
            self.highest_since_entry[symbol] = current_price
        else:
            self.highest_since_entry[symbol] = max(self.highest_since_entry[symbol], current_price)

        # Check stop-loss
        if pnl_pct <= -self.stop_loss_pct:
            return {
                'action': 'sell',
                'symbol': symbol,
                'confidence': 1.0,
                'reason': f'STOP-LOSS triggered at {pnl_pct*100:.1f}%',
                'strategy': 'dip_detector',
                'exit_type': 'stop_loss'
            }

        # Check take-profit
        if pnl_pct >= self.take_profit_pct:
            return {
                'action': 'sell',
                'symbol': symbol,
                'confidence': 1.0,
                'reason': f'TAKE-PROFIT triggered at {pnl_pct*100:.1f}%',
                'strategy': 'dip_detector',
                'exit_type': 'take_profit'
            }

        # Check trailing stop
        if self.use_trailing_stop and symbol in self.highest_since_entry:
            highest = self.highest_since_entry[symbol]
            drawdown_from_high = (current_price - highest) / highest

            if drawdown_from_high <= -self.trailing_stop_pct and pnl_pct > 0:
                return {
                    'action': 'sell',
                    'symbol': symbol,
                    'confidence': 0.9,
                    'reason': f'TRAILING-STOP triggered (down {drawdown_from_high*100:.1f}% from high)',
                    'strategy': 'dip_detector',
                    'exit_type': 'trailing_stop'
                }

        return None

    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Generate dip-buy signals based on RSI, volume, drawdown, divergence,
        and BTC correlation.

        Args:
            data: Dict with symbol -> DataFrame (OHLCV)

        Returns:
            Signal dict with action, confidence, leverage approval
        """
        best_signal = {
            'action': 'hold',
            'symbol': self.symbol,
            'confidence': 0.0,
            'reason': 'No opportunity detected',
            'strategy': 'dip_detector'
        }

        # Check each configured symbol
        for symbol in self.symbols:
            # Find data key for this symbol
            data_key = None
            for key in data.keys():
                base = symbol.split('/')[0]
                if base in key.upper() and '_' not in key:  # Main timeframe
                    data_key = key
                    break

            if not data_key or data_key not in data:
                continue

            df = data[data_key]
            if len(df) < 50:
                continue

            close = df['close'].values
            volume = df['volume'].values
            high = df['high'].values
            low = df['low'].values
            current_price = close[-1]

            # Check for exit signals first (stop-loss, take-profit)
            exit_signal = self._check_stop_loss_take_profit(data_key, current_price)
            if exit_signal:
                return exit_signal

            # Check cooldown
            if not self._check_cooldown(data_key):
                continue

            # Check max positions
            if self.position_count >= self.max_positions:
                continue

            # Calculate indicators
            rsi, rsi_series = self._calculate_rsi_wilders(close)
            volume_surge = self._calculate_volume_surge(volume)
            drawdown = self._calculate_drawdown(close)
            atr_pct = self._calculate_atr_pct(high, low, close)

            # Store state
            self.last_rsi[data_key] = rsi
            self.last_drawdown[data_key] = drawdown
            self.last_volume_surge[data_key] = volume_surge

            # Detect RSI divergence
            has_bullish_div, div_strength = False, 0.0
            has_bearish_div, bear_div_strength = False, 0.0

            if self.use_divergence and len(rsi_series) >= self.divergence_lookback:
                has_bullish_div, div_strength = self._detect_bullish_divergence(close, rsi_series)
                has_bearish_div, bear_div_strength = self._detect_bearish_divergence(close, rsi_series)

            # Check BTC correlation
            btc_safe, btc_reason = True, "BTC filter disabled"
            if self.use_btc_filter:
                btc_safe, btc_reason = self._check_btc_correlation(data)

            # Check multi-timeframe confirmation
            mtf_confirmed, mtf_reason = True, "MTF disabled"
            if self.use_mtf:
                mtf_confirmed, mtf_reason = self._check_mtf_confirmation(data, data_key)

            # Dip conditions (relaxed)
            is_oversold = rsi < self.rsi_oversold
            has_volume_surge = volume_surge > self.volume_surge_threshold
            has_significant_drawdown = drawdown < self.drawdown_threshold
            has_min_volatility = atr_pct > self.min_atr_for_trade

            # Calculate confidence
            confidence = self._calculate_confidence(
                rsi, volume_surge, drawdown,
                has_bullish_div, div_strength,
                btc_safe
            )

            # Determine if this is a dip opportunity
            # Enhanced logic: divergence alone can trigger, or traditional oversold + (volume or drawdown)
            is_dip = False
            reason_parts = []

            if has_bullish_div and div_strength > 0.3:
                # Bullish divergence is a strong signal by itself
                is_dip = True
                reason_parts.append(f"BULLISH DIVERGENCE (str={div_strength:.2f})")

            if is_oversold:
                if has_volume_surge or has_significant_drawdown:
                    is_dip = True
                    if is_oversold:
                        reason_parts.append(f"RSI={rsi:.1f}")
                    if has_volume_surge:
                        reason_parts.append(f"vol={volume_surge:.1f}x")
                    if has_significant_drawdown:
                        reason_parts.append(f"DD={drawdown*100:.1f}%")
                elif rsi < self.rsi_extreme_oversold:
                    # Extreme oversold is enough by itself
                    is_dip = True
                    reason_parts.append(f"EXTREME RSI={rsi:.1f}")

            # Apply filters
            if is_dip and not btc_safe:
                confidence *= 0.5  # Reduce confidence but don't block
                reason_parts.append(f"[BTC caution: {btc_reason}]")

            if is_dip and not mtf_confirmed:
                confidence *= 0.8  # Slight reduction
                reason_parts.append(f"[MTF: {mtf_reason}]")

            if is_dip and not has_min_volatility:
                is_dip = False
                reason_parts.append(f"[Low volatility: ATR={atr_pct*100:.2f}%]")

            # Determine leverage approval
            leverage_ok = is_dip and confidence > 0.6 and atr_pct < self.max_atr_for_leverage and btc_safe

            # Generate signal
            if is_dip and confidence > best_signal['confidence']:
                action = 'buy'
                reason = f"DIP: {', '.join(reason_parts)}"

                # Determine signal strength
                if confidence > 0.8:
                    reason = f"STRONG {reason}"
                elif confidence > 0.6:
                    reason = f"MODERATE {reason}"
                elif confidence > 0.4:
                    reason = f"WEAK {reason}"
                    confidence *= 0.9  # Penalize weak signals

                best_signal = {
                    'action': action,
                    'symbol': data_key,
                    'size': self.position_size_pct * (1.3 if leverage_ok else 1.0),
                    'leverage': min(self.max_leverage, 10) if leverage_ok else min(self.max_leverage, 3),
                    'confidence': confidence,
                    'reason': reason,
                    'strategy': 'dip_detector',
                    'leverage_ok': leverage_ok,
                    'stop_loss': current_price * (1 - self.stop_loss_pct),
                    'take_profit': current_price * (1 + self.take_profit_pct),
                    'indicators': {
                        'rsi': rsi,
                        'volume_surge': volume_surge,
                        'drawdown': drawdown,
                        'atr_pct': atr_pct,
                        'bullish_divergence': has_bullish_div,
                        'divergence_strength': div_strength,
                        'is_oversold': is_oversold,
                        'has_volume_surge': has_volume_surge,
                        'has_significant_drawdown': has_significant_drawdown,
                        'btc_safe': btc_safe,
                        'btc_reason': btc_reason,
                        'mtf_confirmed': mtf_confirmed
                    }
                }

            # Check for sell signal (overbought or bearish divergence)
            elif rsi > self.rsi_overbought or (has_bearish_div and bear_div_strength > 0.3):
                sell_confidence = 0.5 + (rsi - self.rsi_overbought) / 60

                if has_bearish_div:
                    sell_confidence += 0.2
                    sell_reason = f"BEARISH DIVERGENCE + overbought RSI={rsi:.1f}"
                else:
                    sell_reason = f"Overbought RSI={rsi:.1f}"

                if sell_confidence > best_signal['confidence'] and best_signal['action'] == 'hold':
                    best_signal = {
                        'action': 'sell',
                        'symbol': data_key,
                        'confidence': min(1.0, sell_confidence),
                        'reason': sell_reason,
                        'strategy': 'dip_detector',
                        'indicators': {
                            'rsi': rsi,
                            'bearish_divergence': has_bearish_div,
                            'divergence_strength': bear_div_strength
                        }
                    }

        # If no opportunity found, provide detailed reason with indicators
        if best_signal['action'] == 'hold':
            status_parts = []
            primary_symbol = self.symbols[0] if self.symbols else 'BTC/USDT'
            primary_rsi = 50
            primary_drawdown = 0

            for symbol in self.symbols:
                base = symbol.split('/')[0]
                rsi_val = self.last_rsi.get(symbol, self.last_rsi.get(f"{base}/USDT", 50))
                dd_val = self.last_drawdown.get(symbol, 0)
                status_parts.append(f"{base}: RSI={rsi_val:.0f}, DD={dd_val*100:.1f}%")

                if symbol == primary_symbol:
                    primary_rsi = rsi_val
                    primary_drawdown = dd_val

            best_signal['reason'] = f"DipDetector: No dip ({'; '.join(status_parts)})"
            best_signal['indicators'] = {
                'rsi': primary_rsi,
                'drawdown': primary_drawdown,
                'position_count': self.position_count,
                'max_positions': self.max_positions,
                'has_position': self.position_count > 0,
                'cooldown_active': any(
                    (datetime.now() - t).total_seconds() < self.cooldown_minutes * 60
                    for t in self.last_signal_time.values()
                ) if self.last_signal_time else False
            }

        return best_signal

    def on_order_filled(self, order: Dict[str, Any]) -> None:
        """Track position entries for stop-loss/take-profit management."""
        symbol = order.get('symbol', '')
        action = order.get('action', '')
        price = order.get('price', 0)

        if action == 'buy' and price > 0:
            self.entry_prices[symbol] = price
            self.highest_since_entry[symbol] = price
            self.last_signal_time[symbol] = datetime.now()
            self.position_count += 1

        elif action in ['sell', 'close']:
            if symbol in self.entry_prices:
                del self.entry_prices[symbol]
            if symbol in self.highest_since_entry:
                del self.highest_since_entry[symbol]
            self.position_count = max(0, self.position_count - 1)

    def update_model(self, data: Optional[Dict[str, pd.DataFrame]] = None) -> bool:
        """Dip detector is rule-based, no model to update."""
        return True

    def get_status(self) -> Dict[str, Any]:
        """Get strategy status."""
        base_status = super().get_status()
        base_status.update({
            'last_rsi': self.last_rsi,
            'last_drawdown': self.last_drawdown,
            'last_volume_surge': self.last_volume_surge,
            'entry_prices': self.entry_prices,
            'position_count': self.position_count,
            'parameters': {
                'rsi_oversold': self.rsi_oversold,
                'rsi_overbought': self.rsi_overbought,
                'drawdown_threshold': self.drawdown_threshold,
                'volume_surge_threshold': self.volume_surge_threshold,
                'use_divergence': self.use_divergence,
                'use_btc_filter': self.use_btc_filter,
                'use_mtf': self.use_mtf,
                'stop_loss_pct': self.stop_loss_pct,
                'take_profit_pct': self.take_profit_pct
            }
        })
        return base_status
