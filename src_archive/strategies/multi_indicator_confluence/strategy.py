"""
Multi-Indicator Confluence Strategy
Research: RSI + MACD achieves 77% win rate, multi-indicator improves to 75-85%

Combines multiple indicators for high-probability entries:
- Trend Filter: 50 EMA vs 200 EMA (Golden/Death Cross)
- Momentum: RSI > 50 for bullish, < 50 for bearish
- Entry Timing: MACD crossover + histogram direction
- Volatility: Bollinger Bands for breakout confirmation
- Volume: Confirm with volume spike (>1.5x average)

The "confluent conviction" approach increases win rates substantially.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from strategies.base_strategy import BaseStrategy
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime


class MultiIndicatorConfluence(BaseStrategy):
    """
    Multi-Indicator Confluence strategy.

    Waits for multiple indicators to align before entering.
    Higher win rate through confirmation, fewer but better trades.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # EMA Trend Filter
        self.ema_fast = config.get('ema_fast', 50)
        self.ema_slow = config.get('ema_slow', 200)

        # RSI Parameters
        self.rsi_period = config.get('rsi_period', 14)
        self.rsi_bullish_threshold = config.get('rsi_bullish_threshold', 50)
        self.rsi_oversold = config.get('rsi_oversold', 30)
        self.rsi_overbought = config.get('rsi_overbought', 70)

        # MACD Parameters
        self.macd_fast = config.get('macd_fast', 12)
        self.macd_slow = config.get('macd_slow', 26)
        self.macd_signal = config.get('macd_signal', 9)

        # Bollinger Bands
        self.bb_period = config.get('bb_period', 20)
        self.bb_std = config.get('bb_std', 2.0)

        # Volume Filter
        self.volume_mult = config.get('volume_mult', 1.5)
        self.volume_window = config.get('volume_window', 20)

        # Confluence requirements
        self.min_confirmations = config.get('min_confirmations', 3)  # Min indicators agreeing
        self.require_trend_alignment = config.get('require_trend_alignment', True)
        self.require_volume_confirm = config.get('require_volume_confirm', True)

        # Position sizing
        self.base_size_pct = config.get('base_size_pct', 0.12)
        self.confidence_scaling = config.get('confidence_scaling', True)

        # Symbols
        self.symbols = config.get('symbols', ['BTC/USDT', 'XRP/USDT'])

        # State
        self.confluence_scores: Dict[str, Dict] = {}
        self.last_signals: Dict[str, str] = {}

    def _calculate_ema(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate EMA."""
        return series.ewm(span=period, adjust=False).mean()

    def _calculate_rsi(self, close: pd.Series) -> pd.Series:
        """Calculate RSI using Wilder's smoothing."""
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.ewm(alpha=1/self.rsi_period, min_periods=self.rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/self.rsi_period, min_periods=self.rsi_period, adjust=False).mean()

        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(self, close: pd.Series) -> Dict[str, pd.Series]:
        """Calculate MACD, Signal, and Histogram."""
        ema_fast = close.ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.macd_slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=self.macd_signal, adjust=False).mean()
        histogram = macd - signal

        return {
            'macd': macd,
            'signal': signal,
            'histogram': histogram
        }

    def _calculate_bollinger(self, close: pd.Series) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = close.rolling(window=self.bb_period).mean()
        std = close.rolling(window=self.bb_period).std()

        upper = sma + (self.bb_std * std)
        lower = sma - (self.bb_std * std)
        width = (upper - lower) / sma

        return {
            'upper': upper,
            'lower': lower,
            'middle': sma,
            'width': width
        }

    def _check_trend(self, ema50: float, ema200: float, close: float) -> Dict[str, Any]:
        """Check EMA trend alignment."""
        if pd.isna(ema50) or pd.isna(ema200):
            return {'direction': 'neutral', 'strength': 0, 'golden_cross': False}

        if ema50 > ema200:
            direction = 'bullish'
            golden_cross = True
        else:
            direction = 'bearish'
            golden_cross = False

        # Trend strength
        diff_pct = abs(ema50 - ema200) / ema200 * 100
        strength = min(diff_pct / 5, 1.0)  # Normalize to 0-1

        # Price position
        price_above_ema = close > ema50

        return {
            'direction': direction,
            'strength': strength,
            'golden_cross': golden_cross,
            'price_above_fast': price_above_ema
        }

    def _check_rsi(self, rsi: float) -> Dict[str, Any]:
        """Analyze RSI for momentum."""
        if pd.isna(rsi):
            return {'direction': 'neutral', 'signal': None, 'value': 0}

        if rsi < self.rsi_oversold:
            return {'direction': 'bullish', 'signal': 'oversold', 'value': rsi}
        elif rsi > self.rsi_overbought:
            return {'direction': 'bearish', 'signal': 'overbought', 'value': rsi}
        elif rsi > self.rsi_bullish_threshold:
            return {'direction': 'bullish', 'signal': 'above_50', 'value': rsi}
        else:
            return {'direction': 'bearish', 'signal': 'below_50', 'value': rsi}

    def _check_macd(self, macd: float, signal: float, histogram: float,
                    prev_macd: float, prev_signal: float) -> Dict[str, Any]:
        """Analyze MACD for entry timing."""
        if pd.isna(macd) or pd.isna(signal):
            return {'direction': 'neutral', 'crossover': None, 'histogram': 0}

        # Check for crossover
        crossover = None
        if macd > signal and prev_macd <= prev_signal:
            crossover = 'bullish'
        elif macd < signal and prev_macd >= prev_signal:
            crossover = 'bearish'

        # Histogram direction
        if histogram > 0 and histogram > prev_macd - prev_signal:
            hist_direction = 'bullish'
        elif histogram < 0 and histogram < prev_macd - prev_signal:
            hist_direction = 'bearish'
        else:
            hist_direction = 'neutral'

        direction = 'bullish' if macd > signal else 'bearish'

        return {
            'direction': direction,
            'crossover': crossover,
            'histogram': histogram,
            'hist_direction': hist_direction
        }

    def _check_bollinger(self, close: float, upper: float, lower: float,
                        width: float) -> Dict[str, Any]:
        """Analyze Bollinger Bands position."""
        if pd.isna(upper) or pd.isna(lower):
            return {'position': 'middle', 'signal': None}

        if close >= upper:
            return {'position': 'upper', 'signal': 'overbought', 'width': width}
        elif close <= lower:
            return {'position': 'lower', 'signal': 'oversold', 'width': width}
        else:
            # Calculate position within bands
            band_range = upper - lower
            position_pct = (close - lower) / band_range if band_range > 0 else 0.5
            return {'position': 'middle', 'signal': None, 'position_pct': position_pct, 'width': width}

    def _check_volume(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check volume confirmation."""
        if 'volume' not in df.columns or len(df) < self.volume_window + 1:
            return {'confirmed': True, 'ratio': 1.0}  # Skip if no volume data

        current_vol = df['volume'].iloc[-1]
        avg_vol = df['volume'].iloc[-self.volume_window-1:-1].mean()

        ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
        confirmed = ratio >= self.volume_mult

        return {
            'confirmed': confirmed,
            'ratio': ratio,
            'above_average': ratio > 1.0
        }

    def _calculate_confluence(self, trend: Dict, rsi: Dict, macd: Dict,
                              bb: Dict, volume: Dict) -> Dict[str, Any]:
        """
        Calculate confluence score.

        Returns number of indicators agreeing and overall direction.
        """
        bullish_score = 0
        bearish_score = 0
        signals = []

        # Trend (weight: 2)
        if trend['direction'] == 'bullish':
            bullish_score += 2
            signals.append('Trend: Bullish')
        elif trend['direction'] == 'bearish':
            bearish_score += 2
            signals.append('Trend: Bearish')

        # RSI (weight: 1.5)
        if rsi['direction'] == 'bullish':
            bullish_score += 1.5
            signals.append(f"RSI: {rsi['signal']} ({rsi['value']:.0f})")
        elif rsi['direction'] == 'bearish':
            bearish_score += 1.5
            signals.append(f"RSI: {rsi['signal']} ({rsi['value']:.0f})")

        # MACD (weight: 1.5, extra for crossover)
        if macd['direction'] == 'bullish':
            bullish_score += 1.5
            if macd['crossover'] == 'bullish':
                bullish_score += 0.5
                signals.append('MACD: Bullish crossover')
            else:
                signals.append('MACD: Bullish')
        elif macd['direction'] == 'bearish':
            bearish_score += 1.5
            if macd['crossover'] == 'bearish':
                bearish_score += 0.5
                signals.append('MACD: Bearish crossover')
            else:
                signals.append('MACD: Bearish')

        # Bollinger (weight: 1)
        if bb['signal'] == 'oversold':
            bullish_score += 1
            signals.append('BB: Oversold')
        elif bb['signal'] == 'overbought':
            bearish_score += 1
            signals.append('BB: Overbought')

        # Volume confirmation (weight: 1)
        if volume['confirmed']:
            signals.append(f"Volume: {volume['ratio']:.1f}x avg")
            # Add to whichever direction is winning
            if bullish_score > bearish_score:
                bullish_score += 1
            elif bearish_score > bullish_score:
                bearish_score += 1

        # Determine direction
        if bullish_score > bearish_score:
            direction = 'bullish'
            score = bullish_score
        elif bearish_score > bullish_score:
            direction = 'bearish'
            score = bearish_score
        else:
            direction = 'neutral'
            score = 0

        return {
            'direction': direction,
            'bullish_score': bullish_score,
            'bearish_score': bearish_score,
            'total_score': score,
            'signals': signals,
            'confirmations': len([s for s in signals if 'Bullish' in s or 'Bearish' in s or 'oversold' in s.lower() or 'overbought' in s.lower()])
        }

    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Generate signals based on multi-indicator confluence.

        Only trades when minimum number of indicators agree.
        """
        signals = []

        for symbol in self.symbols:
            df = data.get(f'{symbol}_1h')
            if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                df = data.get(f'{symbol}_4h')
            if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                df = data.get(symbol)

            min_bars = max(self.ema_slow, self.bb_period, self.macd_slow) + 10
            if df is None or len(df) < min_bars:
                continue

            close = df['close']
            current_price = close.iloc[-1]

            # Calculate all indicators
            ema50 = self._calculate_ema(close, self.ema_fast)
            ema200 = self._calculate_ema(close, self.ema_slow)
            rsi = self._calculate_rsi(close)
            macd_data = self._calculate_macd(close)
            bb = self._calculate_bollinger(close)

            # Analyze each indicator
            trend = self._check_trend(ema50.iloc[-1], ema200.iloc[-1], current_price)
            rsi_analysis = self._check_rsi(rsi.iloc[-1])
            macd_analysis = self._check_macd(
                macd_data['macd'].iloc[-1],
                macd_data['signal'].iloc[-1],
                macd_data['histogram'].iloc[-1],
                macd_data['macd'].iloc[-2],
                macd_data['signal'].iloc[-2]
            )
            bb_analysis = self._check_bollinger(
                current_price,
                bb['upper'].iloc[-1],
                bb['lower'].iloc[-1],
                bb['width'].iloc[-1]
            )
            volume = self._check_volume(df)

            # Calculate confluence
            confluence = self._calculate_confluence(
                trend, rsi_analysis, macd_analysis, bb_analysis, volume
            )

            self.confluence_scores[symbol] = confluence

            # Check if we have enough confirmations
            if confluence['confirmations'] < self.min_confirmations:
                continue

            # Check trend alignment requirement
            if self.require_trend_alignment:
                if confluence['direction'] == 'bullish' and trend['direction'] != 'bullish':
                    continue
                if confluence['direction'] == 'bearish' and trend['direction'] != 'bearish':
                    continue

            # Check volume requirement
            if self.require_volume_confirm and not volume['confirmed']:
                continue

            # Calculate confidence based on confluence score
            max_possible_score = 7.5  # 2 + 1.5 + 2 + 1 + 1
            confidence = 0.50 + (confluence['total_score'] / max_possible_score) * 0.40

            # MACD crossover bonus
            if macd_analysis['crossover']:
                confidence += 0.05

            # Position size based on confidence
            size = self.base_size_pct
            if self.confidence_scaling:
                size *= (0.8 + confidence * 0.4)  # Scale 80-120%

            action = 'buy' if confluence['direction'] == 'bullish' else 'short'

            signal = {
                'action': action,
                'symbol': symbol,
                'size': min(size, self.base_size_pct * 1.5),
                'leverage': self.max_leverage,
                'confidence': min(confidence, 0.92),
                'reason': f"Confluence: {', '.join(confluence['signals'][:3])}",
                'strategy': 'confluence',
                'confluence_score': confluence['total_score'],
                'confirmations': confluence['confirmations'],
                'rsi': rsi_analysis['value'],
                'trend': trend['direction']
            }

            self.last_signals[symbol] = action
            signals.append(signal)

        if signals:
            return max(signals, key=lambda x: x.get('confidence', 0))

        # Build detailed hold reason for diagnostics
        primary_symbol = self.symbols[0] if self.symbols else 'BTC/USDT'
        conf_score = self.confluence_scores.get(primary_symbol, {})
        confirmations = conf_score.get('confirmations', 0)

        hold_reasons = []
        hold_reasons.append(f'{confirmations}/{self.min_confirmations} confirmations')
        if conf_score.get('direction') == 'neutral':
            hold_reasons.append('Mixed signals')

        return {
            'action': 'hold',
            'symbol': primary_symbol,
            'confidence': 0.0,
            'reason': f"Confluence: {', '.join(hold_reasons)}",
            'strategy': 'confluence',
            'indicators': {
                'confluence': confirmations,
                'min_required': self.min_confirmations,
                'bullish_score': conf_score.get('bullish_score', 0),
                'bearish_score': conf_score.get('bearish_score', 0),
                'total_score': conf_score.get('total_score', 0),
                'low_confluence': confirmations < self.min_confirmations,
                'direction': conf_score.get('direction', 'neutral')
            }
        }

    def update_model(self, data: Optional[Dict[str, pd.DataFrame]] = None) -> bool:
        """Rule-based strategy, no model to update."""
        return True

    def get_status(self) -> Dict[str, Any]:
        """Get strategy status."""
        base = super().get_status()
        base.update({
            'min_confirmations': self.min_confirmations,
            'confluence_scores': self.confluence_scores,
            'last_signals': self.last_signals
        })
        return base
