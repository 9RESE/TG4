"""
Market Regime Detection Module

Identifies different market regimes (trending, ranging, volatile, quiet)
to help understand model performance across different conditions.

Regimes:
- TRENDING_UP: Strong upward trend with momentum
- TRENDING_DOWN: Strong downward trend with momentum
- RANGING: Sideways price action with low volatility
- VOLATILE: High volatility with no clear direction
- BREAKOUT: Transitioning from ranging to trending
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd


class MarketRegime(Enum):
    """Market regime categories."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    BREAKOUT = "breakout"


@dataclass
class RegimeState:
    """Current regime state with confidence."""
    regime: MarketRegime
    confidence: float  # 0-1, how confident in this regime
    duration: int  # Number of bars in this regime
    volatility: float  # Current volatility percentile
    trend_strength: float  # ADX or similar metric
    support_level: float
    resistance_level: float


@dataclass
class RegimeConfig:
    """Configuration for regime detection."""
    # Trend detection
    ema_short: int = 20
    ema_long: int = 50
    adx_period: int = 14
    adx_trend_threshold: float = 25.0  # ADX above this = trending

    # Volatility detection
    atr_period: int = 14
    volatility_lookback: int = 100  # For percentile calculation
    high_volatility_pct: float = 75.0  # Above this percentile = volatile
    low_volatility_pct: float = 25.0  # Below this = quiet/ranging

    # Range detection
    range_lookback: int = 20
    range_threshold: float = 0.02  # 2% range is considered ranging

    # Breakout detection
    breakout_threshold: float = 0.015  # 1.5% move from range
    volume_surge_threshold: float = 1.5  # Volume > 1.5x average


class RegimeDetector:
    """
    Detects market regimes from price data.

    Uses multiple indicators:
    - ADX for trend strength
    - ATR for volatility
    - Price range analysis
    - Volume analysis for breakouts
    """

    def __init__(self, config: Optional[RegimeConfig] = None):
        self.config = config or RegimeConfig()
        self._history: List[RegimeState] = []

    def detect_regime(
        self,
        prices: np.ndarray,
        highs: Optional[np.ndarray] = None,
        lows: Optional[np.ndarray] = None,
        volumes: Optional[np.ndarray] = None
    ) -> RegimeState:
        """
        Detect current market regime.

        Args:
            prices: Close prices
            highs: High prices (optional, will approximate from close)
            lows: Low prices (optional, will approximate from close)
            volumes: Volume data (optional, for breakout detection)

        Returns:
            RegimeState with detected regime
        """
        if len(prices) < self.config.ema_long + self.config.adx_period:
            return RegimeState(
                regime=MarketRegime.RANGING,
                confidence=0.0,
                duration=0,
                volatility=0.5,
                trend_strength=0.0,
                support_level=prices[-1] * 0.98,
                resistance_level=prices[-1] * 1.02
            )

        # Approximate high/low if not provided
        if highs is None:
            highs = self._estimate_highs(prices)
        if lows is None:
            lows = self._estimate_lows(prices)

        # Calculate indicators
        ema_short = self._ema(prices, self.config.ema_short)
        ema_long = self._ema(prices, self.config.ema_long)
        adx = self._calculate_adx(highs, lows, prices)
        atr = self._calculate_atr(highs, lows, prices)
        volatility_pct = self._volatility_percentile(atr)

        # Calculate support/resistance
        support, resistance = self._calculate_sr_levels(prices, highs, lows)

        # Determine regime
        regime, confidence = self._classify_regime(
            prices, ema_short, ema_long, adx, volatility_pct,
            support, resistance, volumes
        )

        # Track duration
        duration = self._calculate_duration(regime)

        state = RegimeState(
            regime=regime,
            confidence=confidence,
            duration=duration,
            volatility=volatility_pct,
            trend_strength=adx[-1] if len(adx) > 0 else 0.0,
            support_level=support,
            resistance_level=resistance
        )

        self._history.append(state)
        return state

    def detect_regime_series(
        self,
        df: pd.DataFrame,
        close_col: str = 'close',
        high_col: str = 'high',
        low_col: str = 'low',
        volume_col: str = 'volume'
    ) -> pd.DataFrame:
        """
        Detect regimes for a full time series.

        Args:
            df: DataFrame with OHLCV data
            close_col, high_col, low_col, volume_col: Column names

        Returns:
            DataFrame with regime columns added
        """
        result = df.copy()

        prices = df[close_col].values
        highs = df[high_col].values if high_col in df.columns else None
        lows = df[low_col].values if low_col in df.columns else None
        volumes = df[volume_col].values if volume_col in df.columns else None

        # Calculate indicators once
        ema_short = self._ema(prices, self.config.ema_short)
        ema_long = self._ema(prices, self.config.ema_long)

        if highs is None:
            highs = self._estimate_highs(prices)
        if lows is None:
            lows = self._estimate_lows(prices)

        adx = self._calculate_adx(highs, lows, prices)
        atr = self._calculate_atr(highs, lows, prices)

        # Detect regime at each point
        regimes = []
        confidences = []
        volatilities = []
        trend_strengths = []

        min_lookback = self.config.ema_long + self.config.adx_period

        for i in range(len(prices)):
            if i < min_lookback:
                regimes.append(MarketRegime.RANGING.value)
                confidences.append(0.0)
                volatilities.append(0.5)
                trend_strengths.append(0.0)
            else:
                # Get historical window
                vol_pct = self._volatility_percentile(atr[:i+1])
                support, resistance = self._calculate_sr_levels(
                    prices[:i+1], highs[:i+1], lows[:i+1]
                )

                vols = volumes[:i+1] if volumes is not None else None
                regime, conf = self._classify_regime(
                    prices[:i+1], ema_short[:i+1], ema_long[:i+1],
                    adx[:i+1], vol_pct, support, resistance, vols
                )

                regimes.append(regime.value)
                confidences.append(conf)
                volatilities.append(vol_pct)
                trend_strengths.append(adx[i] if i < len(adx) else 0.0)

        result['regime'] = regimes
        result['regime_confidence'] = confidences
        result['regime_volatility'] = volatilities
        result['regime_trend_strength'] = trend_strengths

        return result

    def get_regime_features(
        self,
        prices: np.ndarray,
        highs: Optional[np.ndarray] = None,
        lows: Optional[np.ndarray] = None,
        volumes: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Get regime-based features for ML model input.

        Returns numerical features derived from regime analysis.
        """
        state = self.detect_regime(prices, highs, lows, volumes)

        # One-hot encode regime
        regime_features = {
            'regime_trending_up': 1.0 if state.regime == MarketRegime.TRENDING_UP else 0.0,
            'regime_trending_down': 1.0 if state.regime == MarketRegime.TRENDING_DOWN else 0.0,
            'regime_ranging': 1.0 if state.regime == MarketRegime.RANGING else 0.0,
            'regime_volatile': 1.0 if state.regime == MarketRegime.VOLATILE else 0.0,
            'regime_breakout': 1.0 if state.regime == MarketRegime.BREAKOUT else 0.0,
        }

        # Add continuous features
        regime_features.update({
            'regime_confidence': state.confidence,
            'regime_duration': min(state.duration / 100.0, 1.0),  # Normalize
            'regime_volatility': state.volatility,
            'regime_trend_strength': state.trend_strength / 100.0,  # ADX 0-100
            'price_to_support': (prices[-1] - state.support_level) / state.support_level,
            'price_to_resistance': (state.resistance_level - prices[-1]) / state.resistance_level,
        })

        return regime_features

    def _classify_regime(
        self,
        prices: np.ndarray,
        ema_short: np.ndarray,
        ema_long: np.ndarray,
        adx: np.ndarray,
        volatility_pct: float,
        support: float,
        resistance: float,
        volumes: Optional[np.ndarray] = None
    ) -> Tuple[MarketRegime, float]:
        """
        Classify the current regime based on indicators.

        Returns (regime, confidence)
        """
        current_price = prices[-1]
        current_adx = adx[-1] if len(adx) > 0 else 0.0
        current_ema_short = ema_short[-1] if len(ema_short) > 0 else current_price
        current_ema_long = ema_long[-1] if len(ema_long) > 0 else current_price

        # Check for trending
        is_trending = current_adx >= self.config.adx_trend_threshold
        ema_bullish = current_ema_short > current_ema_long
        ema_bearish = current_ema_short < current_ema_long

        # Check for high volatility
        is_volatile = volatility_pct >= self.config.high_volatility_pct
        is_quiet = volatility_pct <= self.config.low_volatility_pct

        # Check for ranging
        if len(prices) >= self.config.range_lookback:
            price_range = (prices[-self.config.range_lookback:].max() -
                          prices[-self.config.range_lookback:].min()) / prices[-1]
            is_ranging = price_range <= self.config.range_threshold
        else:
            is_ranging = False

        # Check for breakout
        is_breakout = False
        breakout_conf = 0.0
        if is_ranging or is_quiet:
            # Price breaking out of range
            if current_price > resistance * (1 - self.config.breakout_threshold):
                is_breakout = True
                breakout_conf = min((current_price - resistance) / (resistance * self.config.breakout_threshold), 1.0)
            elif current_price < support * (1 + self.config.breakout_threshold):
                is_breakout = True
                breakout_conf = min((support - current_price) / (support * self.config.breakout_threshold), 1.0)

            # Volume confirmation
            if volumes is not None and len(volumes) >= 20:
                avg_volume = volumes[-20:-1].mean()
                if volumes[-1] > avg_volume * self.config.volume_surge_threshold:
                    breakout_conf = min(breakout_conf + 0.3, 1.0)

        # Decision logic
        if is_breakout and breakout_conf > 0.3:
            return MarketRegime.BREAKOUT, breakout_conf

        if is_trending:
            if ema_bullish:
                confidence = min(current_adx / 50.0, 1.0)  # ADX confidence
                return MarketRegime.TRENDING_UP, confidence
            elif ema_bearish:
                confidence = min(current_adx / 50.0, 1.0)
                return MarketRegime.TRENDING_DOWN, confidence

        if is_volatile:
            confidence = (volatility_pct - 50) / 50.0  # Higher vol = more confident
            return MarketRegime.VOLATILE, confidence

        if is_ranging or is_quiet:
            confidence = 1.0 - (current_adx / self.config.adx_trend_threshold)
            return MarketRegime.RANGING, max(confidence, 0.0)

        # Default to ranging with low confidence
        return MarketRegime.RANGING, 0.3

    def _calculate_duration(self, regime: MarketRegime) -> int:
        """Calculate how long we've been in this regime."""
        if not self._history:
            return 1

        duration = 1
        for state in reversed(self._history):
            if state.regime == regime:
                duration += 1
            else:
                break
        return duration

    def _ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average."""
        alpha = 2 / (period + 1)
        ema = np.zeros_like(prices)
        ema[0] = prices[0]

        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]

        return ema

    def _calculate_adx(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray
    ) -> np.ndarray:
        """Calculate Average Directional Index."""
        period = self.config.adx_period
        n = len(closes)

        if n < period + 1:
            return np.zeros(n)

        # True Range
        tr = np.zeros(n)
        tr[0] = highs[0] - lows[0]
        for i in range(1, n):
            tr[i] = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )

        # Directional Movement
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)

        for i in range(1, n):
            up_move = highs[i] - highs[i-1]
            down_move = lows[i-1] - lows[i]

            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            if down_move > up_move and down_move > 0:
                minus_dm[i] = down_move

        # Smoothed TR, +DM, -DM
        atr = self._ema(tr, period)
        smooth_plus_dm = self._ema(plus_dm, period)
        smooth_minus_dm = self._ema(minus_dm, period)

        # +DI and -DI
        plus_di = np.zeros(n)
        minus_di = np.zeros(n)

        for i in range(n):
            if atr[i] > 0:
                plus_di[i] = 100 * smooth_plus_dm[i] / atr[i]
                minus_di[i] = 100 * smooth_minus_dm[i] / atr[i]

        # DX and ADX
        dx = np.zeros(n)
        for i in range(n):
            denom = plus_di[i] + minus_di[i]
            if denom > 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / denom

        adx = self._ema(dx, period)
        return adx

    def _calculate_atr(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray
    ) -> np.ndarray:
        """Calculate Average True Range."""
        n = len(closes)
        tr = np.zeros(n)

        tr[0] = highs[0] - lows[0]
        for i in range(1, n):
            tr[i] = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )

        return self._ema(tr, self.config.atr_period)

    def _volatility_percentile(self, atr: np.ndarray) -> float:
        """Calculate current ATR as percentile of historical."""
        lookback = min(self.config.volatility_lookback, len(atr))
        if lookback < 2:
            return 50.0

        current = atr[-1]
        historical = atr[-lookback:]

        percentile = (historical < current).sum() / len(historical) * 100
        return percentile

    def _calculate_sr_levels(
        self,
        prices: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray
    ) -> Tuple[float, float]:
        """Calculate support and resistance levels."""
        lookback = min(self.config.range_lookback, len(prices))

        recent_highs = highs[-lookback:]
        recent_lows = lows[-lookback:]

        resistance = recent_highs.max()
        support = recent_lows.min()

        return support, resistance

    def _estimate_highs(self, prices: np.ndarray) -> np.ndarray:
        """Estimate high prices from close prices."""
        # Use rolling max with small window
        highs = np.zeros_like(prices)
        window = 3
        for i in range(len(prices)):
            start = max(0, i - window + 1)
            highs[i] = prices[start:i+1].max() * 1.001  # Small premium
        return highs

    def _estimate_lows(self, prices: np.ndarray) -> np.ndarray:
        """Estimate low prices from close prices."""
        lows = np.zeros_like(prices)
        window = 3
        for i in range(len(prices)):
            start = max(0, i - window + 1)
            lows[i] = prices[start:i+1].min() * 0.999  # Small discount
        return lows


def analyze_performance_by_regime(
    predictions: np.ndarray,
    labels: np.ndarray,
    regimes: np.ndarray,
    returns: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Analyze model performance broken down by market regime.

    Args:
        predictions: Model predictions (0=sell, 1=hold, 2=buy)
        labels: True labels
        regimes: Regime labels for each sample
        returns: Optional actual returns for P&L analysis

    Returns:
        Dictionary with performance by regime
    """
    results = {}
    unique_regimes = np.unique(regimes)

    for regime in unique_regimes:
        mask = regimes == regime
        n_samples = mask.sum()

        if n_samples == 0:
            continue

        regime_preds = predictions[mask]
        regime_labels = labels[mask]

        # Classification metrics
        accuracy = (regime_preds == regime_labels).mean()

        # Buy/Sell precision
        buy_mask = regime_preds == 2
        sell_mask = regime_preds == 0

        buy_precision = ((regime_labels[buy_mask] == 2).sum() / buy_mask.sum()
                        if buy_mask.sum() > 0 else 0.0)
        sell_precision = ((regime_labels[sell_mask] == 0).sum() / sell_mask.sum()
                         if sell_mask.sum() > 0 else 0.0)

        regime_result = {
            'n_samples': int(n_samples),
            'accuracy': float(accuracy),
            'buy_precision': float(buy_precision),
            'sell_precision': float(sell_precision),
            'buy_count': int(buy_mask.sum()),
            'sell_count': int(sell_mask.sum()),
            'hold_rate': float((regime_preds == 1).sum() / n_samples)
        }

        # P&L analysis if returns provided
        if returns is not None:
            regime_returns = returns[mask]

            # Returns from following signals
            trade_returns = []
            for i in range(len(regime_preds)):
                if regime_preds[i] == 2:  # Buy
                    trade_returns.append(regime_returns[i])
                elif regime_preds[i] == 0:  # Sell
                    trade_returns.append(-regime_returns[i])

            if trade_returns:
                trade_returns = np.array(trade_returns)
                regime_result['avg_trade_return'] = float(trade_returns.mean())
                regime_result['total_return'] = float(trade_returns.sum())
                regime_result['win_rate'] = float((trade_returns > 0).sum() / len(trade_returns))

        results[str(regime)] = regime_result

    return results
