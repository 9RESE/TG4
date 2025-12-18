"""
Market Regime Detection Types

Contains all type definitions for the regime detection system:
- MarketRegime: Overall market classification (STRONG_BULL, BULL, SIDEWAYS, BEAR, STRONG_BEAR)
- VolatilityState: Volatility classification (LOW, MEDIUM, HIGH, EXTREME)
- TrendStrength: ADX-based trend strength classification
- IndicatorScores: Individual indicator contribution scores
- SymbolRegime: Per-symbol regime classification
- MTFConfluence: Multi-timeframe alignment data
- ExternalSentiment: Fear & Greed, BTC Dominance data
- RegimeSnapshot: Complete market regime state at a point in time
- RegimeAdjustments: Strategy parameter adjustments based on regime
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Dict, Optional


class MarketRegime(Enum):
    """Overall market regime classification."""
    STRONG_BULL = auto()  # Very strong uptrend (score > 0.4)
    BULL = auto()         # Uptrend (score 0.15 to 0.4)
    SIDEWAYS = auto()     # Ranging/consolidating (score -0.15 to 0.15)
    BEAR = auto()         # Downtrend (score -0.4 to -0.15)
    STRONG_BEAR = auto()  # Very strong downtrend (score < -0.4)


class VolatilityState(Enum):
    """Volatility classification based on ATR percentage."""
    LOW = auto()       # ATR < 0.3% of price - Calm markets
    MEDIUM = auto()    # 0.3% - 0.8% - Normal volatility
    HIGH = auto()      # 0.8% - 1.5% - Elevated volatility
    EXTREME = auto()   # > 1.5% - Extreme volatility, caution advised


class TrendStrength(Enum):
    """ADX-based trend strength classification."""
    ABSENT = auto()      # ADX < 15 - No trend present
    WEAK = auto()        # ADX 15-20 - Trend developing
    EMERGING = auto()    # ADX 20-25 - Trend emerging
    STRONG = auto()      # ADX 25-40 - Strong trend
    VERY_STRONG = auto() # ADX > 40 - Very strong trend


@dataclass(frozen=True)
class IndicatorScores:
    """
    Individual indicator contributions to regime classification.

    Each score ranges from -1.0 to +1.0:
    - Positive values indicate bullish bias
    - Negative values indicate bearish bias
    - Zero indicates neutral
    """
    adx_score: float           # ADX direction * strength (-1.0 to +1.0)
    chop_score: float          # Choppiness contribution (-1.0 = choppy, +1.0 = trending)
    ma_score: float            # Moving average alignment (-1.0 to +1.0)
    rsi_score: float           # RSI momentum (-1.0 to +1.0)
    volume_score: float        # Volume relative to average (-1.0 to +1.0)
    sentiment_score: float     # External sentiment (-1.0 to +1.0)


@dataclass(frozen=True)
class SymbolRegime:
    """
    Regime classification for a single trading symbol.

    Contains both the classification and all raw indicator values
    for transparency and debugging.
    """
    symbol: str
    regime: MarketRegime
    confidence: float          # 0.0 - 1.0
    trend_strength: TrendStrength
    volatility_state: VolatilityState
    composite_score: float     # -1.0 to +1.0
    indicator_scores: IndicatorScores

    # Raw indicator values for debugging/logging
    adx: float
    plus_di: float
    minus_di: float
    choppiness: float
    rsi: float
    sma_20: float
    sma_50: float
    sma_200: float
    atr_pct: float  # ATR as percentage of price


@dataclass(frozen=True)
class MTFConfluence:
    """
    Multi-timeframe alignment data.

    Higher alignment (more timeframes agreeing) increases confidence
    in the detected regime.
    """
    timeframes_aligned: int    # Count of timeframes with same regime
    total_timeframes: int      # Total timeframes analyzed
    alignment_score: float     # 0.0 - 1.0 (weighted alignment)
    dominant_regime: MarketRegime
    per_timeframe: Dict[str, MarketRegime]  # {'1m': BULL, '5m': BULL, ...}


@dataclass(frozen=True)
class ExternalSentiment:
    """
    External market sentiment data from third-party APIs.

    Sources:
    - Fear & Greed Index: Alternative.me
    - BTC Dominance: CoinGecko
    """
    fear_greed_value: int      # 0-100 (0 = Extreme Fear, 100 = Extreme Greed)
    fear_greed_classification: str  # "Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"
    btc_dominance: float       # Percentage (e.g., 56.5)
    last_updated: datetime


@dataclass(frozen=True)
class RegimeSnapshot:
    """
    Complete market regime state at a point in time.

    This is the main output of the RegimeDetector and contains:
    - Overall market classification
    - Per-symbol breakdown
    - Multi-timeframe confluence (if available)
    - External sentiment (if available)
    - Stability metrics for regime reliability assessment
    """
    timestamp: datetime

    # Overall market classification
    overall_regime: MarketRegime
    overall_confidence: float      # 0.0 - 1.0
    is_trending: bool              # True if in directional regime
    trend_direction: str           # "UP", "DOWN", "NONE"

    # Volatility context
    volatility_state: VolatilityState

    # Per-symbol breakdown
    symbol_regimes: Dict[str, SymbolRegime]

    # Multi-timeframe analysis (may be None if not available)
    mtf_confluence: Optional[MTFConfluence]

    # External data (may be None if fetch failed)
    external_sentiment: Optional[ExternalSentiment]

    # Composite scoring details
    composite_score: float         # -1.0 to +1.0
    indicator_weights: Dict[str, float]

    # Stability metrics (for filtering unstable regimes)
    regime_age_seconds: float      # How long in current regime
    recent_transitions: int        # Regime changes in last hour

    def is_favorable_for_trend_strategy(self) -> bool:
        """
        Check if conditions favor trend-following strategies.

        Favorable conditions:
        - Market is trending (is_trending=True)
        - Confidence is reasonable (> 0.5)
        - Volatility is not extreme

        Returns:
            True if conditions favor trend-following strategies
        """
        return (
            self.is_trending and
            self.overall_confidence > 0.5 and
            self.volatility_state != VolatilityState.EXTREME
        )

    def is_favorable_for_mean_reversion(self) -> bool:
        """
        Check if conditions favor mean-reversion strategies.

        Favorable conditions:
        - Market is ranging (SIDEWAYS regime)
        - Not trending
        - Volatility is low to medium

        Returns:
            True if conditions favor mean-reversion strategies
        """
        return (
            not self.is_trending and
            self.overall_regime == MarketRegime.SIDEWAYS and
            self.volatility_state in (VolatilityState.LOW, VolatilityState.MEDIUM)
        )

    def is_favorable_for_scalping(self) -> bool:
        """
        Check if conditions favor scalping strategies.

        Favorable conditions:
        - Moderate volatility (not too low, not extreme)
        - Market has some direction

        Returns:
            True if conditions favor scalping strategies
        """
        return (
            self.volatility_state in (VolatilityState.MEDIUM, VolatilityState.HIGH) and
            self.overall_confidence > 0.4
        )

    def should_reduce_exposure(self) -> bool:
        """
        Check if overall exposure should be reduced.

        Reduce exposure when:
        - Extreme volatility
        - Low confidence
        - Many recent regime transitions (unstable)

        Returns:
            True if exposure should be reduced
        """
        return (
            self.volatility_state == VolatilityState.EXTREME or
            self.overall_confidence < 0.3 or
            self.recent_transitions > 5
        )


@dataclass
class RegimeAdjustments:
    """
    Parameter adjustments based on regime.

    These multipliers are applied to strategy base configurations
    to adapt behavior to current market conditions.

    Example:
        If position_size_multiplier = 0.5 and base_size = $20,
        the adjusted size would be $10.
    """
    position_size_multiplier: float = 1.0      # 0.5 = half size, 2.0 = double
    stop_loss_multiplier: float = 1.0          # Widen/tighten stops
    take_profit_multiplier: float = 1.0        # Adjust profit targets
    entry_threshold_shift: float = 0.0         # Require stronger signals
    strategy_enabled: bool = True              # Can disable strategy entirely
    cooldown_multiplier: float = 1.0           # Adjust trading frequency
    max_position_multiplier: float = 1.0       # Cap maximum exposure

    def apply_volatility_modifier(self, volatility_state: VolatilityState) -> 'RegimeAdjustments':
        """
        Apply volatility-based modifiers to the adjustments.

        Args:
            volatility_state: Current volatility classification

        Returns:
            New RegimeAdjustments with volatility modifiers applied
        """
        vol_modifiers = {
            VolatilityState.LOW: (1.2, 0.8, 0.8),     # (position_mult, stop_mult, tp_mult)
            VolatilityState.MEDIUM: (1.0, 1.0, 1.0),
            VolatilityState.HIGH: (0.7, 1.5, 1.5),
            VolatilityState.EXTREME: (0.3, 2.0, 2.0),
        }

        pos_mult, stop_mult, tp_mult = vol_modifiers.get(
            volatility_state, (1.0, 1.0, 1.0)
        )

        return RegimeAdjustments(
            position_size_multiplier=self.position_size_multiplier * pos_mult,
            stop_loss_multiplier=self.stop_loss_multiplier * stop_mult,
            take_profit_multiplier=self.take_profit_multiplier * tp_mult,
            entry_threshold_shift=self.entry_threshold_shift,
            strategy_enabled=self.strategy_enabled if volatility_state != VolatilityState.EXTREME else False,
            cooldown_multiplier=self.cooldown_multiplier,
            max_position_multiplier=self.max_position_multiplier * pos_mult,
        )


# Default regime adjustments by strategy type
DEFAULT_REGIME_ADJUSTMENTS: Dict[str, Dict[MarketRegime, RegimeAdjustments]] = {
    'mean_reversion': {
        MarketRegime.STRONG_BULL: RegimeAdjustments(
            position_size_multiplier=0.3,
            stop_loss_multiplier=2.0,
            entry_threshold_shift=0.5,  # Require much stronger signals
            strategy_enabled=False      # Don't mean-revert in strong trends
        ),
        MarketRegime.BULL: RegimeAdjustments(
            position_size_multiplier=0.6,
            stop_loss_multiplier=1.5,
            entry_threshold_shift=0.3,
        ),
        MarketRegime.SIDEWAYS: RegimeAdjustments(
            position_size_multiplier=1.0,
            take_profit_multiplier=0.8,  # Take profits quicker in ranges
        ),
        MarketRegime.BEAR: RegimeAdjustments(
            position_size_multiplier=0.6,
            stop_loss_multiplier=1.5,
            entry_threshold_shift=0.3,
        ),
        MarketRegime.STRONG_BEAR: RegimeAdjustments(
            position_size_multiplier=0.3,
            stop_loss_multiplier=2.0,
            entry_threshold_shift=0.5,
            strategy_enabled=False
        ),
    },
    'momentum_scalping': {
        MarketRegime.STRONG_BULL: RegimeAdjustments(
            position_size_multiplier=1.2,
            take_profit_multiplier=1.5,  # Let winners run
        ),
        MarketRegime.BULL: RegimeAdjustments(
            position_size_multiplier=1.0,
        ),
        MarketRegime.SIDEWAYS: RegimeAdjustments(
            position_size_multiplier=0.3,
            strategy_enabled=False      # Momentum fails in ranges
        ),
        MarketRegime.BEAR: RegimeAdjustments(
            position_size_multiplier=1.0,
        ),
        MarketRegime.STRONG_BEAR: RegimeAdjustments(
            position_size_multiplier=1.2,
            take_profit_multiplier=1.5,
        ),
    },
    'grid_trading': {
        MarketRegime.STRONG_BULL: RegimeAdjustments(
            strategy_enabled=False      # Grid gets one-sided in trends
        ),
        MarketRegime.BULL: RegimeAdjustments(
            position_size_multiplier=0.5,
        ),
        MarketRegime.SIDEWAYS: RegimeAdjustments(
            position_size_multiplier=1.2,  # Best conditions for grid
        ),
        MarketRegime.BEAR: RegimeAdjustments(
            position_size_multiplier=0.5,
        ),
        MarketRegime.STRONG_BEAR: RegimeAdjustments(
            strategy_enabled=False
        ),
    },
    'whale_sentiment': {
        MarketRegime.STRONG_BULL: RegimeAdjustments(
            position_size_multiplier=0.8,
            entry_threshold_shift=0.2,  # Don't fight strong trends
        ),
        MarketRegime.BULL: RegimeAdjustments(
            position_size_multiplier=1.0,
        ),
        MarketRegime.SIDEWAYS: RegimeAdjustments(
            position_size_multiplier=1.2,  # Contrarian works well
        ),
        MarketRegime.BEAR: RegimeAdjustments(
            position_size_multiplier=1.0,
        ),
        MarketRegime.STRONG_BEAR: RegimeAdjustments(
            position_size_multiplier=0.8,
            entry_threshold_shift=0.2,
        ),
    },
}
