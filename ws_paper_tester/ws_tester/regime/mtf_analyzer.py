"""
Multi-Timeframe Analyzer

Analyzes regime confluence across multiple timeframes to increase
confidence in regime classification.

Timeframe Hierarchy (weighted):
- 4H: Strategic direction (weight 5)
- 1H: Tactical direction (weight 4)
- 15m: Operational direction (weight 3)
- 5m: Execution timing (weight 2)
- 1m: Entry precision (weight 1)

Higher timeframes have more weight because they represent
stronger/more established trends.

Note: Currently DataSnapshot only provides 1m and 5m candles.
This analyzer works with available timeframes and can be extended
when more timeframes become available.
"""
from typing import Dict, Optional, Tuple

from ws_tester.types import Candle, DataSnapshot
from ws_tester.indicators import calculate_sma

from .types import MarketRegime, MTFConfluence


class MTFAnalyzer:
    """
    Analyze regime confluence across multiple timeframes.

    Higher timeframe alignment increases confidence in regime classification.
    Conflicting timeframes reduce confidence.
    """

    DEFAULT_WEIGHTS = {
        '4h': 5,
        '1h': 4,
        '15m': 3,
        '5m': 2,
        '1m': 1,
    }

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the MTF analyzer.

        Args:
            config: Optional configuration dict with:
                - mtf_weights: Dict of timeframe weights
        """
        config = config or {}
        self.weights = config.get('mtf_weights', self.DEFAULT_WEIGHTS)

    def analyze(
        self,
        data: DataSnapshot,
        symbol: str = None
    ) -> Optional[MTFConfluence]:
        """
        Calculate multi-timeframe confluence.

        Analyzes all available timeframes and calculates a weighted
        alignment score.

        Args:
            data: DataSnapshot with candle data
            symbol: Optional specific symbol to analyze (uses first available if None)

        Returns:
            MTFConfluence data or None if insufficient data
        """
        per_timeframe: Dict[str, MarketRegime] = {}

        # Determine symbol to analyze
        if symbol is None:
            if data.candles_1m:
                symbol = next(iter(data.candles_1m.keys()))
            else:
                return None

        # 1-minute regime
        candles_1m = data.candles_1m.get(symbol)
        if candles_1m and len(candles_1m) >= 50:
            regime_1m = self._classify_timeframe(candles_1m)
            per_timeframe['1m'] = regime_1m

        # 5-minute regime
        candles_5m = data.candles_5m.get(symbol)
        if candles_5m and len(candles_5m) >= 50:
            regime_5m = self._classify_timeframe(candles_5m)
            per_timeframe['5m'] = regime_5m

        # Note: 15m, 1h, 4h would be added here when available
        # For now, we can build longer timeframes from 1m data if needed
        if candles_1m and len(candles_1m) >= 60:
            # Build pseudo-15m from last 15 1-minute candles
            regime_15m = self._classify_from_recent(candles_1m, 15)
            if regime_15m:
                per_timeframe['15m'] = regime_15m

        if candles_1m and len(candles_1m) >= 60:
            # Build pseudo-1h from last 60 1-minute candles
            regime_1h = self._classify_from_recent(candles_1m, 60)
            if regime_1h:
                per_timeframe['1h'] = regime_1h

        if not per_timeframe:
            return None

        # Calculate weighted alignment
        bull_weight = 0
        bear_weight = 0
        total_weight = 0

        for tf, regime in per_timeframe.items():
            weight = self.weights.get(tf, 1)
            total_weight += weight

            if regime in (MarketRegime.BULL, MarketRegime.STRONG_BULL):
                bull_weight += weight
            elif regime in (MarketRegime.BEAR, MarketRegime.STRONG_BEAR):
                bear_weight += weight

        # Determine dominant regime and alignment score
        if total_weight == 0:
            dominant = MarketRegime.SIDEWAYS
            alignment_score = 0.0
        elif bull_weight > bear_weight:
            bull_pct = bull_weight / total_weight
            if bull_pct > 0.7:
                dominant = MarketRegime.STRONG_BULL if bull_pct > 0.85 else MarketRegime.BULL
            else:
                dominant = MarketRegime.SIDEWAYS
            alignment_score = bull_pct
        elif bear_weight > bull_weight:
            bear_pct = bear_weight / total_weight
            if bear_pct > 0.7:
                dominant = MarketRegime.STRONG_BEAR if bear_pct > 0.85 else MarketRegime.BEAR
            else:
                dominant = MarketRegime.SIDEWAYS
            alignment_score = bear_pct
        else:
            dominant = MarketRegime.SIDEWAYS
            alignment_score = 0.5

        # Count aligned timeframes
        aligned = sum(1 for r in per_timeframe.values() if r == dominant)

        return MTFConfluence(
            timeframes_aligned=aligned,
            total_timeframes=len(per_timeframe),
            alignment_score=alignment_score,
            dominant_regime=dominant,
            per_timeframe=per_timeframe,
        )

    def _classify_timeframe(self, candles: Tuple[Candle, ...]) -> MarketRegime:
        """
        Quick regime classification for a single timeframe.

        Uses simple SMA comparison for fast classification:
        - Price > SMA20 > SMA50: BULL
        - Price < SMA20 < SMA50: BEAR
        - Otherwise: SIDEWAYS

        Args:
            candles: Tuple of Candle objects

        Returns:
            MarketRegime classification
        """
        if len(candles) < 50:
            return MarketRegime.SIDEWAYS

        closes = [c.close for c in candles]
        current = closes[-1]

        sma_20 = sum(closes[-20:]) / 20
        sma_50 = sum(closes[-50:]) / 50

        # Check for strong trend (Perfect alignment)
        if len(candles) >= 200:
            sma_200 = sum(closes[-200:]) / 200
            if current > sma_20 > sma_50 > sma_200:
                return MarketRegime.STRONG_BULL
            if current < sma_20 < sma_50 < sma_200:
                return MarketRegime.STRONG_BEAR

        # Regular trend
        if current > sma_20 > sma_50:
            return MarketRegime.BULL
        elif current < sma_20 < sma_50:
            return MarketRegime.BEAR
        else:
            return MarketRegime.SIDEWAYS

    def _classify_from_recent(
        self,
        candles: Tuple[Candle, ...],
        lookback: int
    ) -> Optional[MarketRegime]:
        """
        Classify regime from recent candles (pseudo-timeframe).

        Builds a synthetic view of a longer timeframe by looking at
        recent price action.

        Args:
            candles: Source candles (typically 1m)
            lookback: Number of candles to consider

        Returns:
            MarketRegime or None if insufficient data
        """
        if len(candles) < lookback:
            return None

        recent = candles[-lookback:]
        closes = [c.close for c in recent]

        if len(closes) < 20:
            return MarketRegime.SIDEWAYS

        current = closes[-1]
        start = closes[0]

        # Calculate trend from start to end
        change_pct = ((current - start) / start) * 100 if start > 0 else 0

        # Simple classification based on change
        if change_pct > 1.0:  # > 1% move up
            return MarketRegime.BULL
        elif change_pct < -1.0:  # > 1% move down
            return MarketRegime.BEAR
        else:
            # Check SMA alignment for more nuance
            if len(closes) >= 20:
                sma_10 = sum(closes[-10:]) / 10
                sma_20 = sum(closes[-20:]) / 20
                if current > sma_10 > sma_20:
                    return MarketRegime.BULL
                elif current < sma_10 < sma_20:
                    return MarketRegime.BEAR

            return MarketRegime.SIDEWAYS

    def get_alignment_description(self, confluence: MTFConfluence) -> str:
        """
        Get a human-readable description of MTF alignment.

        Args:
            confluence: MTFConfluence data

        Returns:
            Description string
        """
        if confluence.alignment_score > 0.85:
            strength = "Strong"
        elif confluence.alignment_score > 0.7:
            strength = "Good"
        elif confluence.alignment_score > 0.5:
            strength = "Moderate"
        else:
            strength = "Weak"

        aligned = confluence.timeframes_aligned
        total = confluence.total_timeframes
        regime = confluence.dominant_regime.name

        return f"{strength} {regime} alignment ({aligned}/{total} timeframes)"
