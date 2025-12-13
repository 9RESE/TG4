"""
Whale Tracking and Sentiment Analysis Strategy
Research: Whale wallets (+1000 BTC) increased 2.3% during market fear (accumulation signal)

Combines on-chain whale tracking with social sentiment analysis.

Data Sources (requires external API integration):
- Whale Alert: Large transaction monitoring
- Glassnode: On-chain metrics
- Social Sentiment: Twitter/Reddit/TikTok analysis

Signals:
- Whale accumulation + positive sentiment = bullish
- Whale distribution + negative sentiment = bearish
- Divergence (whales buying, retail panic) = contrarian opportunity

This strategy requires external data feeds. Without them, it uses
technical indicators as proxy sentiment measures.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from strategies.base_strategy import BaseStrategy
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta


class WhaleSentiment(BaseStrategy):
    """
    Whale tracking + sentiment strategy.

    Uses large holder behavior and market sentiment for trading decisions.
    Falls back to technical indicators if external data unavailable.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Whale detection thresholds
        self.whale_threshold_btc = config.get('whale_threshold_btc', 100)  # 100+ BTC
        self.whale_threshold_xrp = config.get('whale_threshold_xrp', 1_000_000)  # 1M+ XRP
        self.whale_threshold_usd = config.get('whale_threshold_usd', 1_000_000)  # $1M+

        # Sentiment thresholds (0-1 scale)
        self.bullish_threshold = config.get('bullish_threshold', 0.60)
        self.bearish_threshold = config.get('bearish_threshold', 0.40)

        # Volume spike detection (proxy for whale activity)
        self.volume_spike_mult = config.get('volume_spike_mult', 2.0)
        self.volume_window = config.get('volume_window', 24)  # 24 bars

        # Price deviation for fear/greed proxy
        self.fear_deviation = config.get('fear_deviation', -0.05)  # -5% = fear
        self.greed_deviation = config.get('greed_deviation', 0.05)  # +5% = greed

        # RSI as sentiment proxy
        self.rsi_period = config.get('rsi_period', 14)
        self.rsi_fear = config.get('rsi_fear', 25)  # Extreme fear
        self.rsi_greed = config.get('rsi_greed', 75)  # Extreme greed

        # Contrarian mode
        self.contrarian_mode = config.get('contrarian_mode', True)

        # Position sizing
        self.base_size_pct = config.get('base_size_pct', 0.10)

        # Symbols
        self.symbols = config.get('symbols', ['BTC/USDT', 'XRP/USDT'])

        # State
        self.whale_signals: Dict[str, str] = {}
        self.sentiment_scores: Dict[str, float] = {}
        self.last_large_trades: Dict[str, List[Dict]] = {}

    def _calculate_rsi(self, close: pd.Series) -> pd.Series:
        """Calculate RSI."""
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.ewm(alpha=1/self.rsi_period, min_periods=self.rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/self.rsi_period, min_periods=self.rsi_period, adjust=False).mean()

        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _detect_volume_spike(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect unusual volume (proxy for whale activity).

        Large volume spikes often indicate institutional activity.
        """
        if 'volume' not in df.columns or len(df) < self.volume_window + 1:
            return {'detected': False, 'ratio': 1.0}

        current_vol = df['volume'].iloc[-1]
        avg_vol = df['volume'].iloc[-self.volume_window-1:-1].mean()

        ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
        detected = ratio >= self.volume_spike_mult

        # Determine direction from price movement during spike
        if detected:
            price_change = (df['close'].iloc[-1] - df['open'].iloc[-1]) / df['open'].iloc[-1]
            direction = 'accumulation' if price_change > 0 else 'distribution'
        else:
            direction = 'neutral'

        return {
            'detected': detected,
            'ratio': ratio,
            'direction': direction
        }

    def _calculate_fear_greed_proxy(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate fear/greed proxy from price action.

        - Large drops from recent high = fear
        - Large rises from recent low = greed
        """
        close = df['close']
        current = close.iloc[-1]

        # Recent high/low (20 periods)
        recent_high = close.iloc[-21:-1].max() if len(close) > 21 else close.max()
        recent_low = close.iloc[-21:-1].min() if len(close) > 21 else close.min()

        from_high = (current - recent_high) / recent_high
        from_low = (current - recent_low) / recent_low if recent_low > 0 else 0

        # Determine sentiment
        if from_high <= self.fear_deviation:
            sentiment = 'fear'
            score = max(0, 0.5 + from_high * 5)  # Scale to 0-0.5
        elif from_low >= self.greed_deviation:
            sentiment = 'greed'
            score = min(1, 0.5 + from_low * 5)  # Scale to 0.5-1
        else:
            sentiment = 'neutral'
            score = 0.5

        return {
            'sentiment': sentiment,
            'score': score,
            'from_high_pct': from_high * 100,
            'from_low_pct': from_low * 100
        }

    def _analyze_rsi_sentiment(self, rsi: float) -> Dict[str, Any]:
        """Use RSI as sentiment indicator."""
        if pd.isna(rsi):
            return {'sentiment': 'neutral', 'score': 0.5}

        if rsi <= self.rsi_fear:
            return {'sentiment': 'extreme_fear', 'score': rsi / 100}
        elif rsi >= self.rsi_greed:
            return {'sentiment': 'extreme_greed', 'score': rsi / 100}
        elif rsi < 40:
            return {'sentiment': 'fear', 'score': rsi / 100}
        elif rsi > 60:
            return {'sentiment': 'greed', 'score': rsi / 100}
        else:
            return {'sentiment': 'neutral', 'score': 0.5}

    def _parse_external_sentiment(self, data: Dict[str, pd.DataFrame],
                                  symbol: str) -> Optional[Dict[str, Any]]:
        """
        Parse external sentiment data if available.

        Expected format in data:
        - {symbol}_sentiment: DataFrame with 'score', 'source', 'timestamp'
        - {symbol}_whales: DataFrame with 'type' (in/out), 'amount', 'usd_value'
        """
        sentiment_key = f'{symbol.replace("/", "_")}_sentiment'
        whale_key = f'{symbol.replace("/", "_")}_whales'

        result = {'has_external_data': False}

        # Parse sentiment data
        if sentiment_key in data:
            df = data[sentiment_key]
            if isinstance(df, pd.DataFrame) and 'score' in df.columns:
                result['sentiment_score'] = df['score'].iloc[-1]
                result['has_external_data'] = True

        # Parse whale data
        if whale_key in data:
            df = data[whale_key]
            if isinstance(df, pd.DataFrame) and len(df) > 0:
                # Aggregate recent whale activity
                inflows = df[df['type'] == 'in']['usd_value'].sum() if 'type' in df.columns else 0
                outflows = df[df['type'] == 'out']['usd_value'].sum() if 'type' in df.columns else 0

                net_flow = inflows - outflows
                result['whale_net_flow'] = net_flow
                result['whale_direction'] = 'accumulation' if net_flow > 0 else 'distribution'
                result['has_external_data'] = True

        return result if result['has_external_data'] else None

    def _generate_composite_signal(self, volume_signal: Dict, fear_greed: Dict,
                                   rsi_sentiment: Dict, external: Optional[Dict]) -> Dict[str, Any]:
        """
        Generate composite signal from all sentiment sources.

        In contrarian mode, fear = buy, greed = sell.
        """
        signals = []
        scores = []

        # Volume spike (whale proxy)
        if volume_signal['detected']:
            if volume_signal['direction'] == 'accumulation':
                signals.append('bullish')
                scores.append(0.7)
            elif volume_signal['direction'] == 'distribution':
                signals.append('bearish')
                scores.append(0.7)

        # Fear/Greed
        if fear_greed['sentiment'] == 'fear':
            if self.contrarian_mode:
                signals.append('bullish')  # Buy the fear
                scores.append(0.6)
            else:
                signals.append('bearish')
                scores.append(0.6)
        elif fear_greed['sentiment'] == 'greed':
            if self.contrarian_mode:
                signals.append('bearish')  # Sell the greed
                scores.append(0.6)
            else:
                signals.append('bullish')
                scores.append(0.6)

        # RSI sentiment
        if rsi_sentiment['sentiment'] == 'extreme_fear':
            if self.contrarian_mode:
                signals.append('bullish')
                scores.append(0.75)
            else:
                signals.append('bearish')
                scores.append(0.75)
        elif rsi_sentiment['sentiment'] == 'extreme_greed':
            if self.contrarian_mode:
                signals.append('bearish')
                scores.append(0.75)
            else:
                signals.append('bullish')
                scores.append(0.75)

        # External data (highest weight)
        if external:
            if 'whale_direction' in external:
                if external['whale_direction'] == 'accumulation':
                    signals.append('bullish')
                    scores.append(0.85)
                else:
                    signals.append('bearish')
                    scores.append(0.85)

            if 'sentiment_score' in external:
                ext_score = external['sentiment_score']
                if ext_score > self.bullish_threshold:
                    signals.append('bullish')
                    scores.append(ext_score)
                elif ext_score < self.bearish_threshold:
                    signals.append('bearish')
                    scores.append(1 - ext_score)

        # Aggregate signals
        if not signals:
            return {'direction': 'neutral', 'confidence': 0.0, 'signals': []}

        bullish_count = signals.count('bullish')
        bearish_count = signals.count('bearish')

        if bullish_count > bearish_count:
            direction = 'bullish'
            confidence = sum(s for sig, s in zip(signals, scores) if sig == 'bullish') / bullish_count
        elif bearish_count > bullish_count:
            direction = 'bearish'
            confidence = sum(s for sig, s in zip(signals, scores) if sig == 'bearish') / bearish_count
        else:
            direction = 'neutral'
            confidence = 0.0

        return {
            'direction': direction,
            'confidence': confidence * 0.9,  # Cap at 90%
            'signals': signals,
            'bullish_count': bullish_count,
            'bearish_count': bearish_count
        }

    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Generate signals from whale activity and sentiment analysis.
        """
        signals = []

        for symbol in self.symbols:
            df = data.get(f'{symbol}_1h')
            if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                df = data.get(symbol)

            if df is None or len(df) < max(self.volume_window, self.rsi_period) + 5:
                continue

            close = df['close']
            current_price = close.iloc[-1]

            # Calculate proxy indicators
            rsi = self._calculate_rsi(close)
            volume_signal = self._detect_volume_spike(df)
            fear_greed = self._calculate_fear_greed_proxy(df)
            rsi_sentiment = self._analyze_rsi_sentiment(rsi.iloc[-1])

            # Try to get external data
            external = self._parse_external_sentiment(data, symbol)

            # Generate composite signal
            composite = self._generate_composite_signal(
                volume_signal, fear_greed, rsi_sentiment, external
            )

            self.sentiment_scores[symbol] = fear_greed['score']
            self.whale_signals[symbol] = volume_signal['direction']

            if composite['direction'] == 'neutral' or composite['confidence'] < 0.55:
                continue

            action = 'buy' if composite['direction'] == 'bullish' else 'short'

            reason_parts = []
            if volume_signal['detected']:
                reason_parts.append(f"Volume spike {volume_signal['ratio']:.1f}x ({volume_signal['direction']})")
            reason_parts.append(f"Sentiment: {fear_greed['sentiment']}")
            reason_parts.append(f"RSI: {rsi_sentiment['sentiment']} ({rsi.iloc[-1]:.0f})")

            if self.contrarian_mode:
                reason_parts.append("Contrarian mode")

            signal = {
                'action': action,
                'symbol': symbol,
                'size': self.base_size_pct,
                'leverage': self.max_leverage,
                'confidence': composite['confidence'],
                'reason': f"Whale/Sentiment: {', '.join(reason_parts[:2])}",
                'strategy': 'whale_sentiment',
                'volume_spike': volume_signal['detected'],
                'sentiment': fear_greed['sentiment'],
                'rsi': rsi.iloc[-1],
                'contrarian_mode': self.contrarian_mode
            }

            signals.append(signal)

        if signals:
            return max(signals, key=lambda x: x.get('confidence', 0))

        # Build detailed hold reason for diagnostics
        primary_symbol = self.symbols[0] if self.symbols else 'BTC/USDT'
        sentiment = self.sentiment_scores.get(primary_symbol, 0.5)
        whale_signal = self.whale_signals.get(primary_symbol, 'neutral')

        hold_reasons = []
        if whale_signal == 'neutral':
            hold_reasons.append('No whale activity detected')
        if abs(sentiment - 0.5) < 0.1:
            hold_reasons.append('Sentiment neutral')

        if not hold_reasons:
            hold_reasons.append('Low confidence composite signal')

        return {
            'action': 'hold',
            'symbol': primary_symbol,
            'confidence': 0.0,
            'reason': f"WhaleSentiment: {', '.join(hold_reasons)}",
            'strategy': 'whale_sentiment',
            'indicators': {
                'sentiment_score': sentiment,
                'whale_signal': whale_signal,
                'whale_score': 0.0 if whale_signal == 'neutral' else 0.7,
                'no_whale_activity': whale_signal == 'neutral',
                'contrarian_mode': self.contrarian_mode
            }
        }

    def update_model(self, data: Optional[Dict[str, pd.DataFrame]] = None) -> bool:
        """Rule-based strategy, no model to update."""
        return True

    def get_status(self) -> Dict[str, Any]:
        """Get strategy status."""
        base = super().get_status()
        base.update({
            'contrarian_mode': self.contrarian_mode,
            'sentiment_scores': self.sentiment_scores,
            'whale_signals': self.whale_signals
        })
        return base
