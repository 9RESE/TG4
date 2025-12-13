"""
Phase 25: XRP Momentum LSTM Strategy (Improved)
LSTM-based momentum strategy with dip detection for leveraged entries.

Improvements over Phase 24:
- LSTM integration with pre-trained model loading
- Priority-based signal scoring (no more cascading overrides)
- BTC correlation as leading indicator
- Multi-timeframe confirmation (5m data)
- Position management with stop-loss/take-profit
- Lowered thresholds for realistic trading
- RSI confirmation filter
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from strategies.base_strategy import BaseStrategy
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class SignalCandidate:
    """Represents a potential trading signal with priority scoring."""
    action: str
    confidence: float
    leverage_ok: bool
    reason: str
    priority: int = 0  # Higher = more important


class XRPMomentumLSTM(BaseStrategy):
    """
    LSTM-based momentum strategy for XRP with dip detection.

    Phase 25 Improvements:
    - Actually integrates LSTM predictions from pre-trained model
    - Uses BTC momentum as leading indicator
    - Priority-based signal selection (highest confidence wins)
    - Multi-timeframe confirmation with 5m data
    - Position management with stop-loss/take-profit tracking
    - Configurable thresholds tuned for XRP volatility
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.symbol = config.get('symbol', 'XRP/USDT')
        self.btc_symbol = config.get('btc_symbol', 'BTC/USDT')

        # ATR parameters
        self.atr_period = config.get('atr_period', 14)
        self.momentum_period = config.get('momentum_period', 10)

        # Dip detection (lowered thresholds for more activity)
        self.dip_lookback = config.get('dip_lookback', 24)
        self.dip_threshold = config.get('dip_threshold', -0.02)  # 2% dip (was -3%)

        # Volume thresholds (lowered for more activity)
        self.volume_spike_mult = config.get('volume_spike_mult', 1.5)  # Was 2.0
        self.volume_lookback = config.get('volume_lookback', 50)

        # Momentum thresholds (lowered for XRP volatility)
        self.momentum_buy_threshold = config.get('momentum_buy_threshold', 0.03)  # Was 0.05
        self.momentum_sell_threshold = config.get('momentum_sell_threshold', -0.03)  # Was -0.05
        self.momentum_recovery_threshold = config.get('momentum_recovery_threshold', -0.015)  # Was -0.02

        # RSI parameters
        self.rsi_period = config.get('rsi_period', 14)
        self.rsi_oversold = config.get('rsi_oversold', 35)
        self.rsi_overbought = config.get('rsi_overbought', 65)
        self.use_rsi_filter = config.get('use_rsi_filter', True)

        # BTC correlation
        self.use_btc_correlation = config.get('use_btc_correlation', True)
        self.btc_lead_threshold = config.get('btc_lead_threshold', 0.02)  # 2% BTC move

        # Multi-timeframe
        self.use_multi_timeframe = config.get('use_multi_timeframe', True)
        self.mtf_agreement_required = config.get('mtf_agreement_required', False)  # Require 5m alignment

        # Position management
        self.stop_loss_pct = config.get('stop_loss_pct', 0.03)  # 3% stop loss
        self.take_profit_pct = config.get('take_profit_pct', 0.05)  # 5% take profit
        self.trailing_stop = config.get('trailing_stop', False)
        self.trailing_stop_pct = config.get('trailing_stop_pct', 0.02)

        # LSTM integration
        self.use_lstm = config.get('use_lstm', True)  # Now defaults to True!
        self.model_path = config.get('model_path', 'models/lstm_xrp.pth')
        self.lstm_seq_len = config.get('lstm_seq_len', 60)
        self.lstm_predictor = None
        self._lstm_initialized = False

        # State tracking
        self.last_atr = 0.0
        self.last_momentum = 0.0
        self.last_rsi = 50.0
        self.last_btc_momentum = 0.0
        self.is_dip = False

        # Position tracking for stop-loss/take-profit
        self.entry_prices: Dict[str, float] = {}
        self.highest_since_entry: Dict[str, float] = {}

        # Initialize LSTM if enabled
        self._init_lstm()

    def _init_lstm(self):
        """Initialize LSTM predictor with pre-trained model."""
        if not self.use_lstm or self._lstm_initialized:
            return

        try:
            from models.lstm_predictor import LSTMPredictor
            import torch

            self.lstm_predictor = LSTMPredictor()

            # Load pre-trained model if exists
            if os.path.exists(self.model_path):
                self.lstm_predictor.model.load_state_dict(
                    torch.load(self.model_path, map_location=self.lstm_predictor.device, weights_only=True)
                )
                self.lstm_predictor.model.eval()
                print(f"[XRPMomentumLSTM] Loaded LSTM model from {self.model_path}")
            else:
                print(f"[XRPMomentumLSTM] No pre-trained model at {self.model_path}, LSTM will train on first data")

            self._lstm_initialized = True

        except ImportError as e:
            print(f"[XRPMomentumLSTM] Could not import LSTM dependencies: {e}")
            self.use_lstm = False
        except Exception as e:
            print(f"[XRPMomentumLSTM] LSTM initialization error: {e}")
            self.use_lstm = False

    def _calculate_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """Calculate Average True Range with safe handling."""
        if len(close) < 2:
            return np.array([0.0])

        # True Range calculation
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]  # Avoid using rolled value for first element

        tr = np.maximum(
            high - low,
            np.abs(high - prev_close),
            np.abs(low - prev_close)
        )

        atr = pd.Series(tr).rolling(window=self.atr_period, min_periods=1).mean().values
        return atr

    def _calculate_momentum(self, close: np.ndarray) -> np.ndarray:
        """Calculate price momentum (rate of change) with zero-division protection."""
        if len(close) < self.momentum_period + 1:
            return np.zeros(len(close))

        prev_close = np.roll(close, self.momentum_period)
        # Protect against zero division
        prev_close = np.where(prev_close == 0, 1e-10, prev_close)

        roc = (close - prev_close) / prev_close
        roc[:self.momentum_period] = 0
        return roc

    def _calculate_rsi(self, close: np.ndarray) -> np.ndarray:
        """Calculate Relative Strength Index."""
        if len(close) < self.rsi_period + 1:
            return np.full(len(close), 50.0)

        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        avg_gain = pd.Series(gain).rolling(window=self.rsi_period, min_periods=1).mean().values
        avg_loss = pd.Series(loss).rolling(window=self.rsi_period, min_periods=1).mean().values

        # Avoid division by zero
        avg_loss = np.where(avg_loss == 0, 1e-10, avg_loss)
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        # Prepend a neutral RSI for the first element lost in diff
        return np.concatenate([[50.0], rsi])

    def _detect_dip(self, close: np.ndarray) -> Tuple[bool, float]:
        """
        Detect if price is in a dip from recent high.
        Returns (is_dip, drawdown_pct).
        """
        if len(close) < self.dip_lookback:
            return False, 0.0

        recent_high = np.max(close[-self.dip_lookback:])
        current = close[-1]

        if recent_high == 0:
            return False, 0.0

        drawdown = (current - recent_high) / recent_high
        return drawdown < self.dip_threshold, drawdown

    def _check_volume_spike(self, volume: np.ndarray) -> Tuple[bool, float]:
        """
        Check for volume spike vs average.
        Returns (is_spike, volume_ratio).
        """
        if len(volume) < 2:
            return False, 1.0

        lookback = min(self.volume_lookback, len(volume) - 1)
        avg_vol = np.mean(volume[-lookback-1:-1]) if lookback > 0 else volume[-1]

        if avg_vol == 0:
            return False, 1.0

        ratio = volume[-1] / avg_vol
        return ratio > self.volume_spike_mult, ratio

    def _get_lstm_signal(self, close: np.ndarray) -> Tuple[bool, float]:
        """
        Get LSTM prediction (if enabled and trained).
        Returns (is_bullish, confidence_boost).
        """
        if not self.use_lstm or self.lstm_predictor is None:
            return True, 0.0  # Neutral, no confidence boost

        try:
            # Need enough data for LSTM sequence
            if len(close) < self.lstm_seq_len:
                return True, 0.0

            # Fit scaler on recent data
            recent_data = close[-500:] if len(close) > 500 else close
            self.lstm_predictor.scaler.fit(recent_data.reshape(-1, 1))

            # Get prediction
            window = close[-self.lstm_seq_len:]
            is_bullish = self.lstm_predictor.predict_signal(window)

            # LSTM provides confidence boost of 0.1 when aligned
            confidence_boost = 0.10 if is_bullish else -0.05

            return is_bullish, confidence_boost

        except Exception as e:
            # Fail silently, don't block signals
            return True, 0.0

    def _get_btc_momentum(self, data: Dict[str, pd.DataFrame]) -> float:
        """Get BTC momentum as a leading indicator."""
        if not self.use_btc_correlation:
            return 0.0

        # Try to find BTC data
        btc_key = None
        for key in data.keys():
            if 'BTC' in key.upper() and '5m' not in key and '15m' not in key:
                btc_key = key
                break

        if btc_key is None or btc_key not in data:
            return 0.0

        btc_df = data[btc_key]
        if len(btc_df) < self.momentum_period + 1:
            return 0.0

        btc_momentum = self._calculate_momentum(btc_df['close'].values)
        return btc_momentum[-1] if len(btc_momentum) > 0 else 0.0

    def _get_5m_confirmation(self, data: Dict[str, pd.DataFrame]) -> Tuple[bool, float]:
        """
        Get 5-minute timeframe confirmation.
        Returns (is_aligned, momentum_5m).
        """
        if not self.use_multi_timeframe:
            return True, 0.0

        # Look for 5m data
        xrp_5m_key = None
        for key in data.keys():
            if 'XRP' in key.upper() and '5m' in key.lower():
                xrp_5m_key = key
                break

        if xrp_5m_key is None or xrp_5m_key not in data:
            return True, 0.0  # No 5m data, don't block

        df_5m = data[xrp_5m_key]
        if len(df_5m) < self.momentum_period + 1:
            return True, 0.0

        momentum_5m = self._calculate_momentum(df_5m['close'].values)
        current_5m_momentum = momentum_5m[-1] if len(momentum_5m) > 0 else 0.0

        return True, current_5m_momentum

    def _find_xrp_data(self, data: Dict[str, pd.DataFrame]) -> Optional[str]:
        """Find the correct XRP data key (prefer 1h over 5m/15m)."""
        # Priority: exact match > 1h suffix > no suffix > any XRP
        candidates = []

        for key in data.keys():
            if 'XRP' in key.upper():
                if key == self.symbol:
                    return key  # Exact match
                elif '5m' not in key.lower() and '15m' not in key.lower():
                    candidates.insert(0, key)  # Prefer non-5m/15m
                else:
                    candidates.append(key)

        return candidates[0] if candidates else None

    def _check_position_exit(self, symbol: str, current_price: float) -> Optional[SignalCandidate]:
        """Check if we should exit based on stop-loss or take-profit."""
        if symbol not in self.entry_prices:
            return None

        entry_price = self.entry_prices[symbol]
        pnl_pct = (current_price - entry_price) / entry_price

        # Update highest price for trailing stop
        if symbol not in self.highest_since_entry:
            self.highest_since_entry[symbol] = current_price
        else:
            self.highest_since_entry[symbol] = max(self.highest_since_entry[symbol], current_price)

        # Check take profit
        if pnl_pct >= self.take_profit_pct:
            return SignalCandidate(
                action='sell',
                confidence=0.90,
                leverage_ok=False,
                reason=f'Take profit hit: {pnl_pct*100:.1f}% gain',
                priority=100  # Highest priority
            )

        # Check stop loss
        if pnl_pct <= -self.stop_loss_pct:
            return SignalCandidate(
                action='sell',
                confidence=0.95,
                leverage_ok=False,
                reason=f'Stop loss hit: {pnl_pct*100:.1f}% loss',
                priority=100  # Highest priority
            )

        # Check trailing stop
        if self.trailing_stop and symbol in self.highest_since_entry:
            highest = self.highest_since_entry[symbol]
            drawdown_from_high = (current_price - highest) / highest
            if drawdown_from_high <= -self.trailing_stop_pct:
                return SignalCandidate(
                    action='sell',
                    confidence=0.85,
                    leverage_ok=False,
                    reason=f'Trailing stop: {drawdown_from_high*100:.1f}% from high',
                    priority=90
                )

        return None

    def _generate_signal_candidates(
        self,
        close: np.ndarray,
        current_momentum: float,
        current_atr: float,
        avg_atr: float,
        current_rsi: float,
        is_dip: bool,
        drawdown: float,
        volume_spike: bool,
        volume_ratio: float,
        lstm_bullish: bool,
        lstm_confidence: float,
        btc_momentum: float,
        mtf_aligned: bool,
        momentum_5m: float
    ) -> List[SignalCandidate]:
        """
        Generate all potential signal candidates with priority scoring.
        This replaces the cascading if-else logic with explicit priority.
        """
        candidates = []

        # === BUY SIGNALS ===

        # 1. Dip buy with recovery (highest priority buy)
        if is_dip and current_momentum > self.momentum_recovery_threshold:
            confidence = 0.75
            leverage_ok = True

            # Boost confidence with confirmations
            if lstm_bullish:
                confidence += lstm_confidence
            if btc_momentum > 0:
                confidence += 0.05
            if self.use_rsi_filter and current_rsi < self.rsi_oversold:
                confidence += 0.05
            if volume_spike:
                confidence += 0.05

            confidence = min(confidence, 0.95)

            candidates.append(SignalCandidate(
                action='buy',
                confidence=confidence,
                leverage_ok=leverage_ok,
                reason=f'Dip buy: {drawdown*100:.1f}% drawdown, momentum recovering ({current_momentum*100:.2f}%)',
                priority=80
            ))

        # 2. Strong momentum with volume (medium-high priority)
        if current_momentum > self.momentum_buy_threshold and volume_spike:
            confidence = 0.65
            leverage_ok = current_atr < avg_atr * 1.5  # Only leverage if not too volatile

            if lstm_bullish:
                confidence += lstm_confidence
            if btc_momentum > self.btc_lead_threshold:
                confidence += 0.10  # BTC leading up
            if self.use_rsi_filter and current_rsi < 60:  # Not overbought
                confidence += 0.05

            confidence = min(confidence, 0.90)

            candidates.append(SignalCandidate(
                action='buy',
                confidence=confidence,
                leverage_ok=leverage_ok,
                reason=f'Strong momentum: {current_momentum*100:.1f}% + volume spike ({volume_ratio:.1f}x)',
                priority=70
            ))

        # 3. Volume spike with LSTM bullish (medium priority)
        if lstm_bullish and volume_spike and current_atr > avg_atr * 0.5:
            confidence = 0.60 + lstm_confidence

            if btc_momentum > 0:
                confidence += 0.05
            if self.use_rsi_filter and current_rsi < 55:
                confidence += 0.05

            confidence = min(confidence, 0.85)

            candidates.append(SignalCandidate(
                action='buy',
                confidence=confidence,
                leverage_ok=False,
                reason=f'LSTM bullish + volume spike ({volume_ratio:.1f}x) + volatility',
                priority=60
            ))

        # 4. BTC leading indicator (lower priority)
        if self.use_btc_correlation and btc_momentum > self.btc_lead_threshold:
            if current_momentum > 0 and lstm_bullish:
                confidence = 0.55 + lstm_confidence

                candidates.append(SignalCandidate(
                    action='buy',
                    confidence=min(confidence, 0.75),
                    leverage_ok=False,
                    reason=f'BTC leading: {btc_momentum*100:.1f}% + XRP aligned',
                    priority=50
                ))

        # 5. RSI oversold bounce (medium priority)
        if self.use_rsi_filter and current_rsi < self.rsi_oversold:
            if current_momentum > self.momentum_recovery_threshold:
                confidence = 0.60

                if lstm_bullish:
                    confidence += lstm_confidence
                if volume_spike:
                    confidence += 0.10

                confidence = min(confidence, 0.80)

                candidates.append(SignalCandidate(
                    action='buy',
                    confidence=confidence,
                    leverage_ok=False,
                    reason=f'RSI oversold bounce: RSI={current_rsi:.0f}, momentum recovering',
                    priority=55
                ))

        # === SELL SIGNALS ===

        # 1. Strong negative momentum with volatility (high priority sell)
        if current_momentum < self.momentum_sell_threshold and current_atr > avg_atr * 1.2:
            confidence = 0.65

            if not lstm_bullish:
                confidence += 0.10
            if btc_momentum < -self.btc_lead_threshold:
                confidence += 0.10
            if self.use_rsi_filter and current_rsi > self.rsi_overbought:
                confidence += 0.05

            confidence = min(confidence, 0.90)

            candidates.append(SignalCandidate(
                action='sell',
                confidence=confidence,
                leverage_ok=False,
                reason=f'Negative momentum ({current_momentum*100:.1f}%) + high volatility',
                priority=75
            ))

        # 2. RSI overbought with momentum reversal
        if self.use_rsi_filter and current_rsi > self.rsi_overbought:
            if current_momentum < 0:
                confidence = 0.55

                if not lstm_bullish:
                    confidence += 0.10
                if btc_momentum < 0:
                    confidence += 0.05

                confidence = min(confidence, 0.75)

                candidates.append(SignalCandidate(
                    action='sell',
                    confidence=confidence,
                    leverage_ok=False,
                    reason=f'RSI overbought: RSI={current_rsi:.0f}, momentum reversing',
                    priority=60
                ))

        # 3. BTC leading down
        if self.use_btc_correlation and btc_momentum < -self.btc_lead_threshold:
            if current_momentum < 0:
                confidence = 0.50

                if not lstm_bullish:
                    confidence += 0.10

                candidates.append(SignalCandidate(
                    action='sell',
                    confidence=min(confidence, 0.70),
                    leverage_ok=False,
                    reason=f'BTC leading down: {btc_momentum*100:.1f}%',
                    priority=45
                ))

        return candidates

    def _select_best_signal(self, candidates: List[SignalCandidate]) -> Optional[SignalCandidate]:
        """Select the best signal from candidates based on priority and confidence."""
        if not candidates:
            return None

        # Sort by priority (descending), then confidence (descending)
        candidates.sort(key=lambda x: (x.priority, x.confidence), reverse=True)

        return candidates[0]

    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Generate trading signals based on momentum, dip detection, and LSTM.
        Uses priority-based signal selection instead of cascading if-else.

        Args:
            data: Dict with symbol -> DataFrame (OHLCV)

        Returns:
            Signal dict with action, confidence, leverage approval
        """
        # Find XRP data (improved matching)
        xrp_key = self._find_xrp_data(data)

        if not xrp_key:
            return {
                'action': 'hold',
                'symbol': self.symbol,
                'confidence': 0.0,
                'reason': 'No XRP data available',
                'strategy': 'xrp_momentum_lstm'
            }

        df = data[xrp_key]
        min_data = max(self.atr_period, self.momentum_period, self.dip_lookback, self.rsi_period) + 5

        if len(df) < min_data:
            return {
                'action': 'hold',
                'symbol': xrp_key,
                'confidence': 0.0,
                'reason': f'Insufficient data ({len(df)}/{min_data})',
                'strategy': 'xrp_momentum_lstm'
            }

        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        current_price = close[-1]

        # Calculate all indicators
        atr = self._calculate_atr(high, low, close)
        momentum = self._calculate_momentum(close)
        rsi = self._calculate_rsi(close)

        current_atr = atr[-1] if not np.isnan(atr[-1]) else 0
        current_momentum = momentum[-1]
        current_rsi = rsi[-1] if len(rsi) > 0 else 50.0
        avg_atr = np.nanmean(atr)

        # Store state for status reporting
        self.last_atr = current_atr
        self.last_momentum = current_momentum
        self.last_rsi = current_rsi

        # Volume analysis
        volume_spike, volume_ratio = self._check_volume_spike(volume)

        # Dip detection
        is_dip, drawdown = self._detect_dip(close)
        self.is_dip = is_dip

        # LSTM prediction
        lstm_bullish, lstm_confidence = self._get_lstm_signal(close)

        # BTC correlation
        btc_momentum = self._get_btc_momentum(data)
        self.last_btc_momentum = btc_momentum

        # Multi-timeframe confirmation
        mtf_aligned, momentum_5m = self._get_5m_confirmation(data)

        # Check for position exit signals first (stop-loss/take-profit)
        exit_signal = self._check_position_exit(xrp_key, current_price)
        if exit_signal:
            return {
                'action': exit_signal.action,
                'symbol': xrp_key,
                'size': 1.0,  # Close full position
                'leverage': 1,
                'confidence': exit_signal.confidence,
                'reason': exit_signal.reason,
                'strategy': 'xrp_momentum_lstm',
                'leverage_ok': False,
                'indicators': {
                    'momentum': current_momentum,
                    'rsi': current_rsi,
                    'atr': current_atr,
                    'avg_atr': avg_atr,
                    'is_dip': is_dip,
                    'volume_spike': volume_spike,
                    'btc_momentum': btc_momentum,
                    'lstm_bullish': lstm_bullish
                }
            }

        # Generate signal candidates
        candidates = self._generate_signal_candidates(
            close=close,
            current_momentum=current_momentum,
            current_atr=current_atr,
            avg_atr=avg_atr,
            current_rsi=current_rsi,
            is_dip=is_dip,
            drawdown=drawdown,
            volume_spike=volume_spike,
            volume_ratio=volume_ratio,
            lstm_bullish=lstm_bullish,
            lstm_confidence=lstm_confidence,
            btc_momentum=btc_momentum,
            mtf_aligned=mtf_aligned,
            momentum_5m=momentum_5m
        )

        # Select best signal
        best_signal = self._select_best_signal(candidates)

        if best_signal:
            # Apply multi-timeframe filter if required
            if self.mtf_agreement_required and self.use_multi_timeframe:
                # Check if 5m momentum agrees with signal direction
                if best_signal.action == 'buy' and momentum_5m < 0:
                    best_signal = None  # Block signal
                elif best_signal.action == 'sell' and momentum_5m > 0:
                    best_signal = None  # Block signal

        if best_signal:
            # Calculate position size and leverage
            base_size = self.position_size_pct
            if best_signal.leverage_ok:
                size = base_size * 1.2
                leverage = min(self.max_leverage, 10)
            else:
                size = base_size
                leverage = min(self.max_leverage, 3)

            return {
                'action': best_signal.action,
                'symbol': xrp_key,
                'size': size,
                'leverage': leverage,
                'confidence': best_signal.confidence,
                'reason': best_signal.reason,
                'strategy': 'xrp_momentum_lstm',
                'leverage_ok': best_signal.leverage_ok,
                'indicators': {
                    'momentum': current_momentum,
                    'rsi': current_rsi,
                    'atr': current_atr,
                    'avg_atr': avg_atr,
                    'is_dip': is_dip,
                    'drawdown': drawdown,
                    'volume_spike': volume_spike,
                    'volume_ratio': volume_ratio,
                    'btc_momentum': btc_momentum,
                    'momentum_5m': momentum_5m,
                    'lstm_bullish': lstm_bullish
                }
            }

        # No signal - hold
        return {
            'action': 'hold',
            'symbol': xrp_key,
            'size': self.position_size_pct,
            'leverage': 1,
            'confidence': 0.0,
            'reason': f'No signal (mom: {current_momentum*100:.2f}%, RSI: {current_rsi:.0f}, dip: {is_dip})',
            'strategy': 'xrp_momentum_lstm',
            'leverage_ok': False,
            'indicators': {
                'momentum': current_momentum,
                'rsi': current_rsi,
                'atr': current_atr,
                'avg_atr': avg_atr,
                'is_dip': is_dip,
                'volume_spike': volume_spike,
                'btc_momentum': btc_momentum,
                'lstm_bullish': lstm_bullish
            }
        }

    def on_order_filled(self, order: Dict[str, Any]) -> None:
        """
        Track position for stop-loss/take-profit management.
        Called when an order is executed.
        """
        symbol = order.get('symbol', '')
        action = order.get('action', '')
        price = order.get('price', 0)

        if action == 'buy':
            self.entry_prices[symbol] = price
            self.highest_since_entry[symbol] = price
            print(f"[XRPMomentumLSTM] Position opened: {symbol} @ ${price:.4f}")

        elif action in ['sell', 'close']:
            if symbol in self.entry_prices:
                entry = self.entry_prices.pop(symbol, price)
                pnl_pct = (price - entry) / entry * 100
                print(f"[XRPMomentumLSTM] Position closed: {symbol} @ ${price:.4f} (PnL: {pnl_pct:.2f}%)")

            if symbol in self.highest_since_entry:
                del self.highest_since_entry[symbol]

    def update_model(self, data: Optional[Dict[str, pd.DataFrame]] = None) -> bool:
        """
        Update LSTM model with new data (if enabled).
        """
        if not self.use_lstm or self.lstm_predictor is None:
            return True

        if data is None:
            return True

        try:
            # Find XRP data for training
            xrp_key = self._find_xrp_data(data)
            if not xrp_key or xrp_key not in data:
                return True

            df = data[xrp_key]
            if len(df) < 100:
                return True

            close = df['close'].values

            # Only retrain periodically or if model doesn't exist
            train_end = int(len(close) * 0.8)
            if train_end > self.lstm_seq_len:
                # Quiet training - verbose=False to reduce log noise
                self.lstm_predictor.train(close[:train_end], epochs=20, seq_len=self.lstm_seq_len, verbose=False)
                return True

        except Exception as e:
            print(f"[XRPMomentumLSTM] Model update error: {e}")

        return True

    def get_status(self) -> Dict[str, Any]:
        """Get strategy status including all indicator values."""
        base_status = super().get_status()
        base_status.update({
            'last_atr': self.last_atr,
            'last_momentum': self.last_momentum,
            'last_rsi': self.last_rsi,
            'last_btc_momentum': self.last_btc_momentum,
            'is_dip': self.is_dip,
            'use_lstm': self.use_lstm,
            'lstm_loaded': self._lstm_initialized,
            'use_btc_correlation': self.use_btc_correlation,
            'use_multi_timeframe': self.use_multi_timeframe,
            'dip_threshold': self.dip_threshold,
            'volume_spike_mult': self.volume_spike_mult,
            'momentum_buy_threshold': self.momentum_buy_threshold,
            'momentum_sell_threshold': self.momentum_sell_threshold,
            'active_positions': list(self.entry_prices.keys()),
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct
        })
        return base_status
