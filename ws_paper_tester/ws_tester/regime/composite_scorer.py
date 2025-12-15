"""
Composite Regime Scorer

Combines multiple technical indicators into a single composite score
for market regime classification.

The composite score ranges from -1.0 to +1.0:
- > 0.4: STRONG_BULL
- 0.15 to 0.4: BULL
- -0.15 to 0.15: SIDEWAYS
- -0.4 to -0.15: BEAR
- < -0.4: STRONG_BEAR

Default Indicator Weights:
- ADX (direction + strength): 25%
- Choppiness Index: 20%
- MA Alignment: 20%
- RSI: 15%
- Volume: 10%
- External Sentiment: 10%
"""
from collections import deque
from typing import Dict, Optional, Tuple, List

from ws_tester.types import Candle
from ws_tester.indicators import (
    calculate_rsi,
    calculate_adx_with_di,
    calculate_choppiness,
    calculate_sma,
    calculate_atr,
    calculate_volume_ratio,
)

from .types import (
    MarketRegime,
    VolatilityState,
    TrendStrength,
    IndicatorScores,
    SymbolRegime,
    ExternalSentiment,
)


class CompositeScorer:
    """
    Weighted combination of multiple regime indicators.

    Calculates a composite score from -1.0 to +1.0 that represents
    the overall market regime. Individual indicator scores are weighted
    and combined to produce the final classification.
    """

    DEFAULT_WEIGHTS = {
        'adx': 0.25,
        'chop': 0.20,
        'ma': 0.20,
        'rsi': 0.15,
        'volume': 0.10,
        'sentiment': 0.10,
    }

    DEFAULT_THRESHOLDS = {
        'strong_bull': 0.4,
        'bull': 0.15,
        'bear': -0.15,
        'strong_bear': -0.4,
    }

    VOLATILITY_THRESHOLDS = {
        'low': 0.3,       # ATR% < 0.3%
        'medium': 0.8,    # ATR% 0.3% - 0.8%
        'high': 1.5,      # ATR% 0.8% - 1.5%
        # > 1.5% = EXTREME
    }

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the composite scorer.

        Args:
            config: Optional configuration dict with:
                - weights: Dict of indicator weights
                - thresholds: Dict of regime classification thresholds
                - smoothing_period: Number of periods for score smoothing
        """
        config = config or {}

        self.weights = config.get('weights', self.DEFAULT_WEIGHTS)
        self.thresholds = config.get('thresholds', self.DEFAULT_THRESHOLDS)
        self.smoothing_period = config.get('smoothing_period', 3)

        # Score history for smoothing
        self._score_history: deque = deque(maxlen=20)

    def calculate_symbol_regime(
        self,
        symbol: str,
        candles: Tuple[Candle, ...],
        external: Optional[ExternalSentiment] = None
    ) -> Optional[SymbolRegime]:
        """
        Calculate regime classification for a single symbol.

        Args:
            symbol: Trading symbol (e.g., 'XRP/USDT')
            candles: Tuple of Candle objects (at least 200 required)
            external: Optional external sentiment data

        Returns:
            SymbolRegime with full classification and raw indicator values,
            or None if insufficient data
        """
        if len(candles) < 200:
            return None

        # Extract price data
        closes = [c.close for c in candles]
        volumes = [c.volume for c in candles]

        # Calculate indicators
        adx_result = calculate_adx_with_di(candles, period=14)
        choppiness = calculate_choppiness(candles, period=14)
        rsi = calculate_rsi(candles, period=14)
        sma_20 = calculate_sma(candles, 20)
        sma_50 = calculate_sma(candles, 50)
        sma_200 = calculate_sma(candles, 200)
        atr = calculate_atr(candles, period=14)

        # Validate all indicators calculated successfully
        if any(v is None for v in [adx_result, choppiness, rsi, sma_20, sma_50, sma_200, atr]):
            return None

        current_price = closes[-1]

        # Calculate volume ratio
        vol_ratio = calculate_volume_ratio(candles, lookback=20)
        if vol_ratio is None:
            vol_ratio = 1.0

        # Calculate ATR as percentage of price
        atr_pct = (atr / current_price) * 100 if current_price > 0 else 0

        # Score each component
        indicator_scores = IndicatorScores(
            adx_score=self._score_adx(adx_result.adx, adx_result.plus_di, adx_result.minus_di),
            chop_score=self._score_chop(choppiness),
            ma_score=self._score_ma(current_price, sma_20, sma_50, sma_200),
            rsi_score=self._score_rsi(rsi),
            volume_score=self._score_volume(vol_ratio),
            sentiment_score=self._score_sentiment(external) if external else 0.0,
        )

        # Calculate weighted composite score
        composite = (
            indicator_scores.adx_score * self.weights['adx'] +
            indicator_scores.chop_score * self.weights['chop'] +
            indicator_scores.ma_score * self.weights['ma'] +
            indicator_scores.rsi_score * self.weights['rsi'] +
            indicator_scores.volume_score * self.weights['volume'] +
            indicator_scores.sentiment_score * self.weights['sentiment']
        )

        # Apply smoothing
        smoothed_score = self._smooth_score(composite)

        # Classify regime
        regime = self._classify_regime(smoothed_score)

        # Calculate confidence (higher absolute score = higher confidence)
        confidence = min(abs(smoothed_score) * 2, 1.0)

        # Classify trend strength from ADX
        trend_strength = self._classify_trend_strength(adx_result.adx)

        # Classify volatility from ATR%
        volatility_state = self._classify_volatility(atr_pct)

        return SymbolRegime(
            symbol=symbol,
            regime=regime,
            confidence=confidence,
            trend_strength=trend_strength,
            volatility_state=volatility_state,
            composite_score=smoothed_score,
            indicator_scores=indicator_scores,
            adx=adx_result.adx,
            plus_di=adx_result.plus_di,
            minus_di=adx_result.minus_di,
            choppiness=choppiness,
            rsi=rsi,
            sma_20=sma_20,
            sma_50=sma_50,
            sma_200=sma_200,
            atr_pct=atr_pct,
        )

    def calculate_overall(
        self,
        symbol_regimes: Dict[str, SymbolRegime],
        mtf_confluence: Optional[any] = None,
        external_sentiment: Optional[ExternalSentiment] = None,
    ) -> Dict:
        """
        Calculate overall market regime from symbol regimes.

        Combines multiple symbol regimes into an overall market view.
        Optionally considers MTF confluence and external sentiment.

        Args:
            symbol_regimes: Dict of symbol -> SymbolRegime
            mtf_confluence: Optional MTF confluence data
            external_sentiment: Optional external sentiment data

        Returns:
            Dict with keys: regime, confidence, is_trending, trend_direction,
            composite_score
        """
        if not symbol_regimes:
            return {
                'regime': MarketRegime.SIDEWAYS,
                'confidence': 0.0,
                'is_trending': False,
                'trend_direction': 'NONE',
                'composite_score': 0.0,
            }

        # Average composite scores across symbols
        scores = [sr.composite_score for sr in symbol_regimes.values()]
        avg_score = sum(scores) / len(scores)

        # Average confidence
        confidences = [sr.confidence for sr in symbol_regimes.values()]
        avg_confidence = sum(confidences) / len(confidences)

        # Boost confidence if MTF aligns
        if mtf_confluence and mtf_confluence.alignment_score > 0.7:
            avg_confidence = min(1.0, avg_confidence * 1.2)

        # Adjust score based on external sentiment
        if external_sentiment:
            sentiment_adjustment = self._score_sentiment(external_sentiment) * 0.1
            avg_score = avg_score + sentiment_adjustment

        # Classify
        regime = self._classify_regime(avg_score)
        is_trending = regime in (
            MarketRegime.STRONG_BULL,
            MarketRegime.BULL,
            MarketRegime.BEAR,
            MarketRegime.STRONG_BEAR,
        ) and avg_confidence > 0.4

        # Determine trend direction
        if avg_score > self.thresholds['bull']:
            trend_direction = 'UP'
        elif avg_score < self.thresholds['bear']:
            trend_direction = 'DOWN'
        else:
            trend_direction = 'NONE'

        return {
            'regime': regime,
            'confidence': avg_confidence,
            'is_trending': is_trending,
            'trend_direction': trend_direction,
            'composite_score': avg_score,
        }

    def _score_adx(self, adx: float, plus_di: float, minus_di: float) -> float:
        """
        Convert ADX values to directional score (-1 to +1).

        ADX < 20: No trend (score = 0)
        ADX >= 20: Direction from +DI vs -DI, strength from ADX value

        Args:
            adx: ADX value (0-100)
            plus_di: +DI value
            minus_di: -DI value

        Returns:
            Score from -1.0 to +1.0
        """
        if adx < 20:
            return 0.0  # No trend

        # Direction based on +DI vs -DI
        direction = 1.0 if plus_di > minus_di else -1.0

        # Strength scaled by ADX (cap at 50)
        strength = min(adx / 50.0, 1.0)

        return direction * strength

    def _score_chop(self, chop: float) -> float:
        """
        Convert Choppiness Index to trend/ranging score.

        > 61.8: Choppy (score = -1.0)
        < 38.2: Trending (score = +1.0)
        38.2-61.8: Linear interpolation

        Note: This score indicates trending vs ranging, not direction.
        A positive score means "trending" (favorable for trend strategies),
        negative means "ranging" (unfavorable for trend strategies).

        Args:
            chop: Choppiness Index value (0-100)

        Returns:
            Score from -1.0 to +1.0
        """
        if chop > 61.8:
            return -1.0  # Very choppy
        elif chop < 38.2:
            return 1.0   # Trending
        else:
            # Linear interpolation
            # At 50, score = 0
            # At 61.8, score = -1
            # At 38.2, score = 1
            return (50 - chop) / 23.6

    def _score_ma(
        self,
        price: float,
        sma_20: float,
        sma_50: float,
        sma_200: float
    ) -> float:
        """
        Score based on moving average alignment.

        Perfect Bull (Price > SMA20 > SMA50 > SMA200): +1.0
        Perfect Bear (Price < SMA20 < SMA50 < SMA200): -1.0
        Partial alignments get intermediate scores.

        Args:
            price: Current price
            sma_20: 20-period SMA
            sma_50: 50-period SMA
            sma_200: 200-period SMA

        Returns:
            Score from -1.0 to +1.0
        """
        # Perfect bull alignment
        if price > sma_20 > sma_50 > sma_200:
            return 1.0

        # Perfect bear alignment
        if price < sma_20 < sma_50 < sma_200:
            return -1.0

        # Bull bias (golden cross)
        if sma_50 > sma_200:
            if price > sma_50:
                return 0.7
            elif price > sma_200:
                return 0.4
            else:
                return 0.1

        # Bear bias (death cross)
        if sma_50 < sma_200:
            if price < sma_50:
                return -0.7
            elif price < sma_200:
                return -0.4
            else:
                return -0.1

        # Neutral
        return 0.0

    def _score_rsi(self, rsi: float) -> float:
        """
        Convert RSI to momentum score.

        RSI 50 = neutral (0)
        RSI 100 = +1.0
        RSI 0 = -1.0

        Args:
            rsi: RSI value (0-100)

        Returns:
            Score from -1.0 to +1.0
        """
        return (rsi - 50) / 50.0

    def _score_volume(self, vol_ratio: float) -> float:
        """
        Score volume relative to average.

        Volume ratio 1.0 = average (score = 0)
        Higher volume = positive score (confirms moves)
        Lower volume = negative score (weak moves)

        Args:
            vol_ratio: Current volume / average volume

        Returns:
            Score from -1.0 to +1.0
        """
        # vol_ratio of 2.0 = score of 1.0
        # vol_ratio of 0.5 = score of -0.5
        return min(max(vol_ratio - 1.0, -1.0), 1.0)

    def _score_sentiment(self, external: ExternalSentiment) -> float:
        """
        Score external sentiment data.

        Fear & Greed Index:
        - 0 (Extreme Fear) = -1.0
        - 50 (Neutral) = 0
        - 100 (Extreme Greed) = +1.0

        Args:
            external: ExternalSentiment data

        Returns:
            Score from -1.0 to +1.0
        """
        fg = external.fear_greed_value
        return (fg - 50) / 50.0

    def _smooth_score(self, score: float) -> float:
        """
        Apply exponential smoothing to reduce whipsaw.

        Args:
            score: Raw composite score

        Returns:
            Smoothed score
        """
        self._score_history.append(score)

        if len(self._score_history) < self.smoothing_period:
            return score

        # Simple moving average smoothing
        recent = list(self._score_history)[-self.smoothing_period:]
        return sum(recent) / len(recent)

    def _classify_regime(self, score: float) -> MarketRegime:
        """
        Classify regime from composite score.

        Args:
            score: Composite score (-1 to +1)

        Returns:
            MarketRegime enum value
        """
        if score > self.thresholds['strong_bull']:
            return MarketRegime.STRONG_BULL
        elif score > self.thresholds['bull']:
            return MarketRegime.BULL
        elif score < self.thresholds['strong_bear']:
            return MarketRegime.STRONG_BEAR
        elif score < self.thresholds['bear']:
            return MarketRegime.BEAR
        else:
            return MarketRegime.SIDEWAYS

    def _classify_trend_strength(self, adx: float) -> TrendStrength:
        """
        Classify trend strength from ADX value.

        Args:
            adx: ADX value (0-100)

        Returns:
            TrendStrength enum value
        """
        if adx < 15:
            return TrendStrength.ABSENT
        elif adx < 20:
            return TrendStrength.WEAK
        elif adx < 25:
            return TrendStrength.EMERGING
        elif adx < 40:
            return TrendStrength.STRONG
        else:
            return TrendStrength.VERY_STRONG

    def _classify_volatility(self, atr_pct: float) -> VolatilityState:
        """
        Classify volatility from ATR percentage.

        Args:
            atr_pct: ATR as percentage of price

        Returns:
            VolatilityState enum value
        """
        if atr_pct < self.VOLATILITY_THRESHOLDS['low']:
            return VolatilityState.LOW
        elif atr_pct < self.VOLATILITY_THRESHOLDS['medium']:
            return VolatilityState.MEDIUM
        elif atr_pct < self.VOLATILITY_THRESHOLDS['high']:
            return VolatilityState.HIGH
        else:
            return VolatilityState.EXTREME
