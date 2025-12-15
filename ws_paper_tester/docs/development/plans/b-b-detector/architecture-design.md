# Market Regime Detector: Architecture Design

## 1. System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MARKET REGIME DETECTION SYSTEM                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐    │
│  │   Kraken WS      │     │  External APIs   │     │  Historical      │    │
│  │   Price Data     │     │  (F&G, BTC.D)    │     │  Data Store      │    │
│  └────────┬─────────┘     └────────┬─────────┘     └────────┬─────────┘    │
│           │                        │                        │              │
│           ▼                        ▼                        ▼              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        DATA AGGREGATION LAYER                        │   │
│  │  • DataSnapshot enhancement • External data cache • MTF candle mgmt  │   │
│  └────────────────────────────────────┬────────────────────────────────┘   │
│                                       │                                    │
│                                       ▼                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      INDICATOR CALCULATION LAYER                     │   │
│  │                                                                      │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐   │   │
│  │  │   ADX   │  │  CHOP   │  │   MA    │  │   RSI   │  │  Volume │   │   │
│  │  │ +DI/-DI │  │  Index  │  │Alignment│  │  Zones  │  │  Ratio  │   │   │
│  │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘   │   │
│  └───────┼────────────┼────────────┼────────────┼────────────┼────────┘   │
│          │            │            │            │            │            │
│          ▼            ▼            ▼            ▼            ▼            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     REGIME CLASSIFICATION LAYER                      │   │
│  │                                                                      │   │
│  │  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐    │   │
│  │  │   Per-Symbol    │   │  Multi-Timeframe│   │    Composite    │    │   │
│  │  │   Classifier    │   │   Confluence    │   │   Score Engine  │    │   │
│  │  └────────┬────────┘   └────────┬────────┘   └────────┬────────┘    │   │
│  └───────────┼──────────────────────┼────────────────────┼─────────────┘   │
│              │                      │                    │                 │
│              ▼                      ▼                    ▼                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        REGIME OUTPUT LAYER                           │   │
│  │                                                                      │   │
│  │   RegimeSnapshot                                                     │   │
│  │   ├── overall_regime: "BULL" | "BEAR" | "SIDEWAYS"                  │   │
│  │   ├── confidence: 0.0 - 1.0                                         │   │
│  │   ├── is_trending: bool                                             │   │
│  │   ├── volatility_state: "LOW" | "MEDIUM" | "HIGH" | "EXTREME"       │   │
│  │   ├── per_symbol_regimes: {symbol: RegimeData}                      │   │
│  │   └── external_sentiment: {fear_greed, btc_dominance}               │   │
│  └────────────────────────────────────┬────────────────────────────────┘   │
│                                       │                                    │
│                                       ▼                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    STRATEGY PARAMETER ROUTER                         │   │
│  │                                                                      │   │
│  │   ┌─────────────────────┐    ┌──────────────────────────────────┐   │   │
│  │   │  Config Mapping     │───▶│  Strategy Config Overrides       │   │   │
│  │   │  Tables             │    │  • Position sizing adjustments   │   │   │
│  │   └─────────────────────┘    │  • Stop-loss/take-profit scaling │   │   │
│  │                              │  • Entry threshold modifications │   │   │
│  │                              │  • Strategy enable/disable flags │   │   │
│  │                              └──────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## 2. Module Structure

```
ws_tester/
├── indicators/                      # Existing indicator library
│   ├── __init__.py
│   ├── moving_averages.py          # SMA, EMA (existing)
│   ├── oscillators.py              # RSI, ADX, MACD (existing + enhance)
│   ├── volatility.py               # ATR, Bollinger (existing)
│   ├── trend.py                    # Trend detection (existing + enhance)
│   ├── volume.py                   # Volume analysis (existing)
│   └── choppiness.py               # NEW: Choppiness Index
│
├── regime/                          # NEW: Regime detection module
│   ├── __init__.py                 # Public API exports
│   ├── types.py                    # RegimeSnapshot, RegimeData types
│   ├── detector.py                 # Main RegimeDetector class
│   ├── composite_scorer.py         # Composite scoring algorithm
│   ├── mtf_analyzer.py             # Multi-timeframe confluence
│   ├── parameter_router.py         # Strategy config adjustments
│   └── external_data.py            # Fear & Greed, BTC Dominance fetchers
│
└── types.py                         # Extend with RegimeSnapshot
```

## 3. Data Types

### 3.1 Core Types (`regime/types.py`)

```python
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Dict, Optional

class MarketRegime(Enum):
    STRONG_BULL = auto()
    BULL = auto()
    SIDEWAYS = auto()
    BEAR = auto()
    STRONG_BEAR = auto()

class VolatilityState(Enum):
    LOW = auto()       # ATR < 0.3% of price
    MEDIUM = auto()    # 0.3% - 0.8%
    HIGH = auto()      # 0.8% - 1.5%
    EXTREME = auto()   # > 1.5%

class TrendStrength(Enum):
    ABSENT = auto()    # ADX < 15
    WEAK = auto()      # ADX 15-20
    EMERGING = auto()  # ADX 20-25
    STRONG = auto()    # ADX 25-40
    VERY_STRONG = auto()  # ADX > 40

@dataclass(frozen=True)
class IndicatorScores:
    """Individual indicator contributions to regime classification."""
    adx_score: float           # -1.0 to +1.0
    chop_score: float          # -1.0 to +1.0
    ma_score: float            # -1.0 to +1.0
    rsi_score: float           # -1.0 to +1.0
    volume_score: float        # -1.0 to +1.0
    sentiment_score: float     # -1.0 to +1.0

@dataclass(frozen=True)
class SymbolRegime:
    """Regime classification for a single symbol."""
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

@dataclass(frozen=True)
class MTFConfluence:
    """Multi-timeframe alignment data."""
    timeframes_aligned: int    # Count of aligned timeframes
    total_timeframes: int
    alignment_score: float     # 0.0 - 1.0
    dominant_regime: MarketRegime
    per_timeframe: Dict[str, MarketRegime]  # {'1m': BULL, '5m': BULL, ...}

@dataclass(frozen=True)
class ExternalSentiment:
    """External market sentiment data."""
    fear_greed_value: int      # 0-100
    fear_greed_classification: str  # "Extreme Fear", "Fear", etc.
    btc_dominance: float       # Percentage (e.g., 56.5)
    last_updated: datetime

@dataclass(frozen=True)
class RegimeSnapshot:
    """Complete market regime state at a point in time."""
    timestamp: datetime

    # Overall market classification
    overall_regime: MarketRegime
    overall_confidence: float
    is_trending: bool
    trend_direction: str       # "UP", "DOWN", "NONE"

    # Volatility context
    volatility_state: VolatilityState

    # Per-symbol breakdown
    symbol_regimes: Dict[str, SymbolRegime]

    # Multi-timeframe analysis
    mtf_confluence: Optional[MTFConfluence]

    # External data
    external_sentiment: Optional[ExternalSentiment]

    # Composite scoring details
    composite_score: float
    indicator_weights: Dict[str, float]

    # Stability metrics
    regime_age_seconds: float  # How long in current regime
    recent_transitions: int    # Regime changes in last hour

    def is_favorable_for_trend_strategy(self) -> bool:
        """Check if conditions favor trend-following strategies."""
        return (
            self.is_trending and
            self.overall_confidence > 0.5 and
            self.volatility_state != VolatilityState.EXTREME
        )

    def is_favorable_for_mean_reversion(self) -> bool:
        """Check if conditions favor mean-reversion strategies."""
        return (
            not self.is_trending and
            self.overall_regime == MarketRegime.SIDEWAYS and
            self.volatility_state in (VolatilityState.LOW, VolatilityState.MEDIUM)
        )
```

### 3.2 Strategy Parameter Adjustments

```python
@dataclass(frozen=True)
class RegimeAdjustments:
    """Parameter adjustments based on regime."""
    position_size_multiplier: float = 1.0      # 0.5 = half size, 2.0 = double
    stop_loss_multiplier: float = 1.0          # Widen/tighten stops
    take_profit_multiplier: float = 1.0        # Adjust profit targets
    entry_threshold_shift: float = 0.0         # Require stronger signals
    strategy_enabled: bool = True              # Can disable strategy entirely
    cooldown_multiplier: float = 1.0           # Adjust trading frequency
    max_position_multiplier: float = 1.0       # Cap maximum exposure

REGIME_ADJUSTMENTS: Dict[str, Dict[MarketRegime, RegimeAdjustments]] = {
    'mean_reversion': {
        MarketRegime.STRONG_BULL: RegimeAdjustments(
            position_size_multiplier=0.5,
            stop_loss_multiplier=1.5,
            strategy_enabled=False  # Don't mean-revert in strong trends
        ),
        MarketRegime.SIDEWAYS: RegimeAdjustments(
            position_size_multiplier=1.2,
            take_profit_multiplier=0.8  # Take profits quicker in ranges
        ),
        # ... etc
    },
    'momentum_scalping': {
        MarketRegime.SIDEWAYS: RegimeAdjustments(
            position_size_multiplier=0.5,
            strategy_enabled=False  # Momentum fails in ranges
        ),
        MarketRegime.STRONG_BULL: RegimeAdjustments(
            position_size_multiplier=1.5,
            take_profit_multiplier=1.5  # Let winners run
        ),
    }
}
```

## 4. Core Components

### 4.1 RegimeDetector Class (`regime/detector.py`)

```python
class RegimeDetector:
    """Main orchestrator for regime detection."""

    def __init__(
        self,
        symbols: List[str],
        config: Optional[dict] = None
    ):
        self.symbols = symbols
        self.config = config or DEFAULT_CONFIG

        # Sub-components
        self.composite_scorer = CompositeScorer(self.config)
        self.mtf_analyzer = MTFAnalyzer(self.config)
        self.external_fetcher = ExternalDataFetcher()
        self.parameter_router = ParameterRouter()

        # State tracking
        self._current_regime: Optional[RegimeSnapshot] = None
        self._regime_start_time: Optional[datetime] = None
        self._transition_history: deque = deque(maxlen=100)
        self._external_cache: Optional[ExternalSentiment] = None
        self._external_cache_time: Optional[datetime] = None

    async def detect(self, data: DataSnapshot) -> RegimeSnapshot:
        """Calculate current market regime from data snapshot."""
        timestamp = data.timestamp

        # 1. Calculate per-symbol regimes
        symbol_regimes = {}
        for symbol in self.symbols:
            candles = data.candles_1m.get(symbol, ())
            if len(candles) < 200:
                continue

            symbol_regime = self._calculate_symbol_regime(symbol, candles)
            symbol_regimes[symbol] = symbol_regime

        # 2. Multi-timeframe confluence (if available)
        mtf = None
        if data.candles_5m:
            mtf = self.mtf_analyzer.analyze(data)

        # 3. External sentiment (cached, async refresh)
        external = await self._get_external_sentiment()

        # 4. Composite overall regime
        overall = self.composite_scorer.calculate_overall(
            symbol_regimes=symbol_regimes,
            mtf_confluence=mtf,
            external_sentiment=external
        )

        # 5. Build snapshot
        snapshot = RegimeSnapshot(
            timestamp=timestamp,
            overall_regime=overall['regime'],
            overall_confidence=overall['confidence'],
            is_trending=overall['is_trending'],
            trend_direction=overall['trend_direction'],
            volatility_state=self._aggregate_volatility(symbol_regimes),
            symbol_regimes=symbol_regimes,
            mtf_confluence=mtf,
            external_sentiment=external,
            composite_score=overall['composite_score'],
            indicator_weights=self.composite_scorer.weights,
            regime_age_seconds=self._calculate_regime_age(overall['regime']),
            recent_transitions=self._count_recent_transitions()
        )

        # 6. Track regime transitions
        self._track_transition(snapshot)

        return snapshot

    def get_strategy_adjustments(
        self,
        strategy_name: str,
        regime: RegimeSnapshot
    ) -> RegimeAdjustments:
        """Get parameter adjustments for a strategy given current regime."""
        return self.parameter_router.get_adjustments(strategy_name, regime)
```

### 4.2 Composite Scorer (`regime/composite_scorer.py`)

```python
class CompositeScorer:
    """Weighted combination of multiple regime indicators."""

    DEFAULT_WEIGHTS = {
        'adx': 0.25,
        'chop': 0.20,
        'ma': 0.20,
        'rsi': 0.15,
        'volume': 0.10,
        'sentiment': 0.10
    }

    def __init__(self, config: dict = None):
        self.weights = config.get('weights', self.DEFAULT_WEIGHTS) if config else self.DEFAULT_WEIGHTS
        self.smoothing_period = config.get('smoothing_period', 3) if config else 3
        self._score_history: deque = deque(maxlen=20)

    def calculate_symbol_score(
        self,
        candles: Tuple[Candle, ...],
        external: Optional[ExternalSentiment] = None
    ) -> Tuple[float, IndicatorScores]:
        """Calculate composite score for a single symbol."""

        closes = [c.close for c in candles]
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        volumes = [c.volume for c in candles]

        # Calculate indicators
        adx_data = calculate_adx(highs, lows, closes)
        chop = calculate_choppiness(highs, lows, closes)
        sma_20 = calculate_sma(closes, 20)
        sma_50 = calculate_sma(closes, 50)
        sma_200 = calculate_sma(closes, 200)
        rsi = calculate_rsi(closes)
        vol_ratio = volumes[-1] / (sum(volumes[-20:]) / 20) if volumes else 1.0

        # Score each component (-1 to +1)
        scores = IndicatorScores(
            adx_score=self._score_adx(adx_data),
            chop_score=self._score_chop(chop),
            ma_score=self._score_ma(closes[-1], sma_20, sma_50, sma_200),
            rsi_score=self._score_rsi(rsi),
            volume_score=self._score_volume(vol_ratio),
            sentiment_score=self._score_sentiment(external) if external else 0.0
        )

        # Weighted composite
        composite = (
            scores.adx_score * self.weights['adx'] +
            scores.chop_score * self.weights['chop'] +
            scores.ma_score * self.weights['ma'] +
            scores.rsi_score * self.weights['rsi'] +
            scores.volume_score * self.weights['volume'] +
            scores.sentiment_score * self.weights['sentiment']
        )

        return composite, scores

    def _score_adx(self, adx_data: dict) -> float:
        """Convert ADX to directional score."""
        adx = adx_data.get('adx', 0)
        plus_di = adx_data.get('plus_di', 0)
        minus_di = adx_data.get('minus_di', 0)

        if adx < 20:
            return 0.0  # No trend

        direction = 1.0 if plus_di > minus_di else -1.0
        strength = min(adx / 50.0, 1.0)
        return direction * strength

    def _score_chop(self, chop: float) -> float:
        """Convert Choppiness Index to trending score."""
        if chop > 61.8:
            return -1.0  # Very choppy = sideways
        elif chop < 38.2:
            return 1.0   # Trending
        else:
            return (50 - chop) / 23.6

    def _score_ma(
        self,
        price: float,
        sma_20: float,
        sma_50: float,
        sma_200: float
    ) -> float:
        """Score based on MA alignment."""
        if price > sma_20 > sma_50 > sma_200:
            return 1.0
        elif price < sma_20 < sma_50 < sma_200:
            return -1.0
        elif sma_50 > sma_200:
            return 0.5 if price > sma_50 else 0.25
        elif sma_50 < sma_200:
            return -0.5 if price < sma_50 else -0.25
        else:
            return 0.0

    def _score_rsi(self, rsi: float) -> float:
        """Convert RSI to momentum score."""
        return (rsi - 50) / 50.0  # -1 to +1

    def _score_volume(self, vol_ratio: float) -> float:
        """Score volume relative to average."""
        return min(max(vol_ratio - 1.0, -1.0), 1.0)

    def _score_sentiment(self, external: ExternalSentiment) -> float:
        """Score external sentiment data."""
        fg = external.fear_greed_value
        return (fg - 50) / 50.0
```

### 4.3 External Data Fetcher (`regime/external_data.py`)

```python
import aiohttp
import asyncio
from datetime import datetime, timedelta

class ExternalDataFetcher:
    """Fetch external market sentiment data."""

    FEAR_GREED_URL = "https://api.alternative.me/fng/"
    COINGECKO_GLOBAL_URL = "https://api.coingecko.com/api/v3/global"

    CACHE_TTL = timedelta(minutes=5)  # Cache for 5 minutes

    def __init__(self):
        self._cache: Optional[ExternalSentiment] = None
        self._cache_time: Optional[datetime] = None
        self._lock = asyncio.Lock()

    async def fetch(self) -> Optional[ExternalSentiment]:
        """Fetch external sentiment data with caching."""
        async with self._lock:
            now = datetime.utcnow()

            # Return cached if fresh
            if (
                self._cache is not None and
                self._cache_time is not None and
                now - self._cache_time < self.CACHE_TTL
            ):
                return self._cache

            # Fetch fresh data
            try:
                fear_greed = await self._fetch_fear_greed()
                btc_dom = await self._fetch_btc_dominance()

                self._cache = ExternalSentiment(
                    fear_greed_value=fear_greed['value'],
                    fear_greed_classification=fear_greed['classification'],
                    btc_dominance=btc_dom,
                    last_updated=now
                )
                self._cache_time = now

                return self._cache

            except Exception as e:
                # Return stale cache on error
                if self._cache is not None:
                    return self._cache
                return None

    async def _fetch_fear_greed(self) -> dict:
        """Fetch Fear & Greed Index from Alternative.me."""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                self.FEAR_GREED_URL,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                data = await resp.json()
                return {
                    'value': int(data['data'][0]['value']),
                    'classification': data['data'][0]['value_classification']
                }

    async def _fetch_btc_dominance(self) -> float:
        """Fetch BTC dominance from CoinGecko."""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                self.COINGECKO_GLOBAL_URL,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                data = await resp.json()
                return data['data']['market_cap_percentage']['btc']
```

### 4.4 Multi-Timeframe Analyzer (`regime/mtf_analyzer.py`)

```python
class MTFAnalyzer:
    """Analyze regime confluence across multiple timeframes."""

    TIMEFRAME_WEIGHTS = {
        '4h': 5,
        '1h': 4,
        '15m': 3,
        '5m': 2,
        '1m': 1
    }

    def __init__(self, config: dict = None):
        self.weights = config.get('mtf_weights', self.TIMEFRAME_WEIGHTS) if config else self.TIMEFRAME_WEIGHTS

    def analyze(self, data: DataSnapshot) -> MTFConfluence:
        """Calculate multi-timeframe confluence."""

        # Get regime for each available timeframe
        per_timeframe = {}

        # 1-minute regime (always available)
        if data.candles_1m:
            for symbol in data.candles_1m:
                regime = self._classify_timeframe(data.candles_1m[symbol])
                per_timeframe['1m'] = regime

        # 5-minute regime
        if data.candles_5m:
            for symbol in data.candles_5m:
                regime = self._classify_timeframe(data.candles_5m[symbol])
                per_timeframe['5m'] = regime

        # Calculate weighted alignment
        bull_weight = sum(
            self.weights[tf]
            for tf, regime in per_timeframe.items()
            if regime in (MarketRegime.BULL, MarketRegime.STRONG_BULL)
        )
        bear_weight = sum(
            self.weights[tf]
            for tf, regime in per_timeframe.items()
            if regime in (MarketRegime.BEAR, MarketRegime.STRONG_BEAR)
        )
        total_weight = sum(self.weights[tf] for tf in per_timeframe)

        # Determine dominant regime
        if total_weight == 0:
            dominant = MarketRegime.SIDEWAYS
            alignment_score = 0.0
        elif bull_weight > bear_weight:
            dominant = MarketRegime.BULL if bull_weight / total_weight > 0.7 else MarketRegime.SIDEWAYS
            alignment_score = bull_weight / total_weight
        elif bear_weight > bull_weight:
            dominant = MarketRegime.BEAR if bear_weight / total_weight > 0.7 else MarketRegime.SIDEWAYS
            alignment_score = bear_weight / total_weight
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
            per_timeframe=per_timeframe
        )

    def _classify_timeframe(self, candles: Tuple[Candle, ...]) -> MarketRegime:
        """Quick regime classification for a single timeframe."""
        if len(candles) < 50:
            return MarketRegime.SIDEWAYS

        closes = [c.close for c in candles]
        sma_20 = sum(closes[-20:]) / 20
        sma_50 = sum(closes[-50:]) / 50

        current = closes[-1]

        if current > sma_20 > sma_50:
            return MarketRegime.BULL
        elif current < sma_20 < sma_50:
            return MarketRegime.BEAR
        else:
            return MarketRegime.SIDEWAYS
```

### 4.5 Parameter Router (`regime/parameter_router.py`)

```python
class ParameterRouter:
    """Route regime-based parameter adjustments to strategies."""

    def __init__(self, adjustments_config: dict = None):
        self.adjustments = adjustments_config or REGIME_ADJUSTMENTS

    def get_adjustments(
        self,
        strategy_name: str,
        regime: RegimeSnapshot
    ) -> RegimeAdjustments:
        """Get parameter adjustments for a strategy."""

        strategy_adjustments = self.adjustments.get(strategy_name, {})
        current_regime = regime.overall_regime

        # Get regime-specific adjustments
        base = strategy_adjustments.get(
            current_regime,
            RegimeAdjustments()  # Default: no adjustments
        )

        # Apply volatility modifiers
        vol_multiplier = self._volatility_modifier(regime.volatility_state)

        return RegimeAdjustments(
            position_size_multiplier=base.position_size_multiplier * vol_multiplier,
            stop_loss_multiplier=base.stop_loss_multiplier,
            take_profit_multiplier=base.take_profit_multiplier,
            entry_threshold_shift=base.entry_threshold_shift,
            strategy_enabled=base.strategy_enabled,
            cooldown_multiplier=base.cooldown_multiplier,
            max_position_multiplier=base.max_position_multiplier * vol_multiplier
        )

    def _volatility_modifier(self, vol_state: VolatilityState) -> float:
        """Reduce position sizes in high volatility."""
        modifiers = {
            VolatilityState.LOW: 1.2,
            VolatilityState.MEDIUM: 1.0,
            VolatilityState.HIGH: 0.7,
            VolatilityState.EXTREME: 0.3
        }
        return modifiers.get(vol_state, 1.0)

    def apply_to_config(
        self,
        base_config: dict,
        adjustments: RegimeAdjustments
    ) -> dict:
        """Apply adjustments to a strategy config dict."""

        adjusted = base_config.copy()

        if 'position_size_usd' in adjusted:
            adjusted['position_size_usd'] *= adjustments.position_size_multiplier

        if 'stop_loss_pct' in adjusted:
            adjusted['stop_loss_pct'] *= adjustments.stop_loss_multiplier

        if 'take_profit_pct' in adjusted:
            adjusted['take_profit_pct'] *= adjustments.take_profit_multiplier

        if 'max_position' in adjusted:
            adjusted['max_position'] *= adjustments.max_position_multiplier

        if 'cooldown_seconds' in adjusted:
            adjusted['cooldown_seconds'] *= adjustments.cooldown_multiplier

        return adjusted
```

## 5. Integration Points

### 5.1 DataSnapshot Enhancement

Add regime data to the existing DataSnapshot:

```python
@dataclass(frozen=True)
class DataSnapshot:
    # ... existing fields ...

    # NEW: Regime data
    regime: Optional[RegimeSnapshot] = None
```

### 5.2 Strategy Interface Enhancement

Strategies can access regime data and adjusted configs:

```python
def generate_signal(
    data: DataSnapshot,
    config: dict,
    state: dict
) -> Optional[Signal]:
    """Generate trading signal with regime awareness."""

    # Access current regime
    regime = data.regime
    if regime is None:
        # Fallback to conservative behavior
        return None

    # Check if strategy should trade in current regime
    if not regime.is_favorable_for_mean_reversion():
        return None

    # Adjust position size based on regime
    adjusted_size = config['position_size_usd']
    if regime.volatility_state == VolatilityState.HIGH:
        adjusted_size *= 0.5

    # ... rest of signal generation ...
```

### 5.3 Main Loop Integration

```python
async def main_loop():
    # Initialize regime detector
    regime_detector = RegimeDetector(symbols=SYMBOLS)

    while running:
        # Get market data
        data_snapshot = await data_manager.get_snapshot()

        # Calculate regime
        regime_snapshot = await regime_detector.detect(data_snapshot)

        # Enhance data snapshot with regime
        enhanced_snapshot = DataSnapshot(
            **{**data_snapshot.__dict__, 'regime': regime_snapshot}
        )

        # Run strategies with regime-aware data
        for strategy in strategies:
            adjustments = regime_detector.get_strategy_adjustments(
                strategy.name,
                regime_snapshot
            )
            adjusted_config = parameter_router.apply_to_config(
                strategy.config,
                adjustments
            )

            signal = strategy.generate_signal(
                enhanced_snapshot,
                adjusted_config,
                strategy.state
            )

            if signal:
                await executor.execute(signal)
```

## 6. Configuration

### 6.1 Regime Detector Config (`config.yaml`)

```yaml
regime_detection:
  enabled: true

  # Indicator weights for composite scoring
  weights:
    adx: 0.25
    chop: 0.20
    ma: 0.20
    rsi: 0.15
    volume: 0.10
    sentiment: 0.10

  # Multi-timeframe weights
  mtf_weights:
    4h: 5
    1h: 4
    15m: 3
    5m: 2
    1m: 1

  # Smoothing to prevent whipsaw
  smoothing_period: 3
  min_regime_duration_seconds: 60  # Minimum time before regime can change
  transition_confirmation_bars: 3   # Bars to confirm regime change

  # External data
  external_data:
    enabled: true
    fear_greed_enabled: true
    btc_dominance_enabled: true
    cache_ttl_minutes: 5

  # Thresholds
  thresholds:
    strong_bull: 0.4
    bull: 0.15
    bear: -0.15
    strong_bear: -0.4
    trending_threshold: 0.15
    high_confidence: 0.6

  # Logging
  log_regime_changes: true
  log_regime_details: false
```

---

*Version: 1.0.0 | Created: 2025-12-15*
