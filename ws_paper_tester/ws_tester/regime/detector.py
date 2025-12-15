"""
Market Regime Detector

Main orchestrator for the regime detection system. Combines:
- CompositeScorer: Per-symbol regime classification
- MTFAnalyzer: Multi-timeframe confluence
- ExternalDataFetcher: Fear & Greed, BTC Dominance
- ParameterRouter: Strategy parameter adjustments

Usage:
    detector = RegimeDetector(symbols=['XRP/USDT', 'BTC/USDT'])

    # On each tick
    regime_snapshot = await detector.detect(data_snapshot)

    # For strategies
    if detector.should_trade('mean_reversion', regime_snapshot):
        adjustments = detector.get_strategy_adjustments('mean_reversion', regime_snapshot)
        config = detector.apply_adjustments(base_config, adjustments)
"""
import logging
from collections import deque
from datetime import datetime
from typing import List, Dict, Optional

from ws_tester.types import DataSnapshot

from .types import (
    MarketRegime,
    VolatilityState,
    RegimeSnapshot,
    SymbolRegime,
    RegimeAdjustments,
)
from .composite_scorer import CompositeScorer
from .mtf_analyzer import MTFAnalyzer
from .external_data import ExternalDataFetcher
from .parameter_router import ParameterRouter

logger = logging.getLogger(__name__)


class RegimeDetector:
    """
    Main orchestrator for market regime detection.

    Combines multiple data sources and analysis methods to produce
    a comprehensive market regime classification.
    """

    def __init__(
        self,
        symbols: List[str],
        config: Optional[Dict] = None
    ):
        """
        Initialize the regime detector.

        Args:
            symbols: List of trading symbols to analyze
            config: Optional configuration dict with:
                - weights: Indicator weights for composite scorer
                - thresholds: Regime classification thresholds
                - smoothing_period: Score smoothing period
                - external_enabled: Enable external data fetching
                - min_regime_duration: Minimum seconds before regime change
        """
        self.symbols = symbols
        self.config = config or {}

        # Sub-components
        self.composite_scorer = CompositeScorer(self.config)
        self.mtf_analyzer = MTFAnalyzer(self.config)
        self.external_fetcher = ExternalDataFetcher(
            enabled=self.config.get('external_enabled', True)
        )
        self.parameter_router = ParameterRouter(
            self.config.get('adjustments')
        )

        # State tracking
        self._current_regime: Optional[MarketRegime] = None
        self._regime_start_time: Optional[datetime] = None
        self._transition_history: deque = deque(maxlen=100)
        self._last_snapshot: Optional[RegimeSnapshot] = None

        # Hysteresis settings with validation
        self._min_regime_duration = self.config.get('min_regime_duration', 60)  # seconds
        self._confirmation_count = 0
        self._confirmation_threshold = self.config.get('confirmation_bars', 3)

        # Validate hysteresis parameters
        if self._min_regime_duration < 10:
            logger.warning(
                f"min_regime_duration={self._min_regime_duration}s is very low, "
                "may cause excessive regime switching. Recommended: >= 30s"
            )
        if self._confirmation_threshold < 2:
            logger.warning(
                f"confirmation_bars={self._confirmation_threshold} is very low, "
                "reduces regime stability. Recommended: >= 2"
            )
        if self._min_regime_duration > 600:
            logger.warning(
                f"min_regime_duration={self._min_regime_duration}s is very high, "
                "may cause slow regime adaptation. Recommended: <= 300s"
            )

    async def detect(self, data: DataSnapshot) -> RegimeSnapshot:
        """
        Calculate current market regime from data snapshot.

        This is the main entry point called on each tick.

        Args:
            data: DataSnapshot with current market data

        Returns:
            RegimeSnapshot with complete regime classification
        """
        timestamp = data.timestamp

        # 1. Calculate per-symbol regimes
        symbol_regimes: Dict[str, SymbolRegime] = {}
        for symbol in self.symbols:
            candles = data.candles_1m.get(symbol, ())
            if len(candles) < 200:
                logger.debug(f"Insufficient candles for {symbol}: {len(candles)}")
                continue

            # Get external sentiment (cached)
            external = await self.external_fetcher.fetch()

            symbol_regime = self.composite_scorer.calculate_symbol_regime(
                symbol, candles, external
            )
            if symbol_regime:
                symbol_regimes[symbol] = symbol_regime

        if not symbol_regimes:
            # Return neutral regime if no symbols could be analyzed
            return self._create_neutral_snapshot(timestamp)

        # 2. Multi-timeframe confluence (for first symbol)
        first_symbol = list(symbol_regimes.keys())[0]
        mtf = self.mtf_analyzer.analyze(data, first_symbol)

        # 3. Get external sentiment
        external = await self.external_fetcher.fetch()

        # 4. Calculate overall regime
        overall = self.composite_scorer.calculate_overall(
            symbol_regimes=symbol_regimes,
            mtf_confluence=mtf,
            external_sentiment=external,
        )

        # 5. Aggregate volatility state
        volatility_state = self._aggregate_volatility(symbol_regimes)

        # 6. Apply hysteresis (prevent rapid regime switching)
        final_regime = self._apply_hysteresis(overall['regime'], timestamp)

        # 7. Build snapshot
        snapshot = RegimeSnapshot(
            timestamp=timestamp,
            overall_regime=final_regime,
            overall_confidence=overall['confidence'],
            is_trending=overall['is_trending'],
            trend_direction=overall['trend_direction'],
            volatility_state=volatility_state,
            symbol_regimes=symbol_regimes,
            mtf_confluence=mtf,
            external_sentiment=external,
            composite_score=overall['composite_score'],
            indicator_weights=self.composite_scorer.weights,
            regime_age_seconds=self._calculate_regime_age(timestamp),
            recent_transitions=self._count_recent_transitions(timestamp),
        )

        # 8. Track transitions
        self._track_transition(snapshot, timestamp)
        self._last_snapshot = snapshot

        # 9. Log significant changes
        if self._current_regime != final_regime:
            logger.info(
                f"Regime changed: {self._current_regime} -> {final_regime} "
                f"(confidence: {overall['confidence']:.0%})"
            )
            self._current_regime = final_regime
            self._regime_start_time = timestamp

        return snapshot

    def get_strategy_adjustments(
        self,
        strategy_name: str,
        regime: RegimeSnapshot
    ) -> RegimeAdjustments:
        """
        Get parameter adjustments for a strategy given current regime.

        Args:
            strategy_name: Name of the strategy
            regime: Current regime snapshot

        Returns:
            RegimeAdjustments with parameter multipliers
        """
        return self.parameter_router.get_adjustments(strategy_name, regime)

    def apply_adjustments(
        self,
        base_config: dict,
        adjustments: RegimeAdjustments
    ) -> dict:
        """
        Apply regime adjustments to a strategy configuration.

        Args:
            base_config: Strategy's base configuration
            adjustments: Adjustments to apply

        Returns:
            Adjusted configuration
        """
        return self.parameter_router.apply_to_config(base_config, adjustments)

    def should_trade(
        self,
        strategy_name: str,
        regime: RegimeSnapshot
    ) -> bool:
        """
        Check if a strategy should trade in the current regime.

        Args:
            strategy_name: Strategy name
            regime: Current regime snapshot

        Returns:
            True if strategy should be allowed to trade
        """
        return self.parameter_router.should_trade(strategy_name, regime)

    def get_current_regime(self) -> Optional[RegimeSnapshot]:
        """
        Get the most recent regime snapshot.

        Returns:
            Last RegimeSnapshot or None if detect() hasn't been called
        """
        return self._last_snapshot

    def _aggregate_volatility(
        self,
        symbol_regimes: Dict[str, SymbolRegime]
    ) -> VolatilityState:
        """
        Aggregate volatility state across symbols.

        Uses the highest volatility state among all symbols.

        Args:
            symbol_regimes: Dict of symbol regimes

        Returns:
            Aggregated VolatilityState
        """
        if not symbol_regimes:
            return VolatilityState.MEDIUM

        # Use highest volatility (most conservative)
        volatility_order = [
            VolatilityState.LOW,
            VolatilityState.MEDIUM,
            VolatilityState.HIGH,
            VolatilityState.EXTREME,
        ]

        max_vol = VolatilityState.LOW
        for sr in symbol_regimes.values():
            if volatility_order.index(sr.volatility_state) > volatility_order.index(max_vol):
                max_vol = sr.volatility_state

        return max_vol

    def _apply_hysteresis(
        self,
        new_regime: MarketRegime,
        timestamp: datetime
    ) -> MarketRegime:
        """
        Apply hysteresis to prevent rapid regime switching.

        Requires confirmation before changing regime:
        1. Minimum time in current regime
        2. Multiple consecutive readings of new regime

        Args:
            new_regime: Newly detected regime
            timestamp: Current timestamp

        Returns:
            Final regime (may stay with old if not confirmed)
        """
        if self._current_regime is None:
            return new_regime

        # If same regime, reset confirmation counter
        if new_regime == self._current_regime:
            self._confirmation_count = 0
            return new_regime

        # Check minimum duration
        if self._regime_start_time:
            age = (timestamp - self._regime_start_time).total_seconds()
            if age < self._min_regime_duration:
                return self._current_regime

        # Require confirmation
        self._confirmation_count += 1
        if self._confirmation_count >= self._confirmation_threshold:
            self._confirmation_count = 0
            return new_regime

        return self._current_regime

    def _calculate_regime_age(self, timestamp: datetime) -> float:
        """
        Calculate how long in the current regime.

        Args:
            timestamp: Current timestamp

        Returns:
            Seconds in current regime
        """
        if self._regime_start_time is None:
            return 0.0
        return (timestamp - self._regime_start_time).total_seconds()

    def _track_transition(self, snapshot: RegimeSnapshot, timestamp: datetime) -> None:
        """
        Track regime transitions for stability analysis.

        Args:
            snapshot: Current regime snapshot
            timestamp: Current timestamp
        """
        self._transition_history.append({
            'timestamp': timestamp,
            'regime': snapshot.overall_regime,
        })

    def _count_recent_transitions(self, timestamp: datetime) -> int:
        """
        Count regime transitions in the last hour.

        Args:
            timestamp: Current timestamp

        Returns:
            Number of transitions in last hour
        """
        one_hour_ago = timestamp.timestamp() - 3600
        transitions = 0
        prev_regime = None

        for entry in self._transition_history:
            if entry['timestamp'].timestamp() > one_hour_ago:
                if prev_regime is not None and entry['regime'] != prev_regime:
                    transitions += 1
                prev_regime = entry['regime']

        return transitions

    def _create_neutral_snapshot(self, timestamp: datetime) -> RegimeSnapshot:
        """
        Create a neutral regime snapshot when analysis isn't possible.

        Args:
            timestamp: Current timestamp

        Returns:
            Neutral RegimeSnapshot
        """
        return RegimeSnapshot(
            timestamp=timestamp,
            overall_regime=MarketRegime.SIDEWAYS,
            overall_confidence=0.0,
            is_trending=False,
            trend_direction='NONE',
            volatility_state=VolatilityState.MEDIUM,
            symbol_regimes={},
            mtf_confluence=None,
            external_sentiment=None,
            composite_score=0.0,
            indicator_weights=self.composite_scorer.weights,
            regime_age_seconds=0.0,
            recent_transitions=0,
        )

    def get_regime_summary(self) -> str:
        """
        Get a human-readable summary of the current regime.

        Returns:
            Summary string
        """
        if self._last_snapshot is None:
            return "No regime data available"

        return self.parameter_router.get_regime_summary(self._last_snapshot)
