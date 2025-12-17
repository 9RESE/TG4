"""
Multi-Timeframe Features - Extract features across multiple timeframes.

Computes features from multiple timeframes (1m, 5m, 15m, 1h, 4h) to provide
richer context for ML models. Uses TimescaleDB continuous aggregates for
efficient data access.

Features computed per timeframe:
- Trend alignment (price vs EMA)
- RSI
- MACD histogram
- ATR (volatility)
- Volume ratio

Aggregate features:
- MTF trend alignment score
- Timeframe divergence score
- Multi-resolution volatility
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import asyncpg
except ImportError:
    asyncpg = None

try:
    import pandas_ta as ta
except ImportError:
    ta = None

logger = logging.getLogger(__name__)


@dataclass
class TimeframeFeatures:
    """Features for a single timeframe."""
    interval_minutes: int
    trend_direction: float  # -1 to 1
    rsi: float
    macd_histogram: float
    atr_pct: float
    volume_ratio: float


@dataclass
class MTFFeatures:
    """Aggregated multi-timeframe features."""
    # Per-timeframe features
    tf_features: Dict[int, TimeframeFeatures]

    # Aggregate features
    mtf_trend_alignment: float  # -1 to 1, all TFs agreeing
    tf_divergence_score: float  # 0 to 1, how much TFs disagree
    multi_resolution_volatility: float  # weighted avg volatility
    dominant_trend: float  # -1, 0, 1 weighted by timeframe
    momentum_confluence: float  # -1 to 1, momentum agreement


class MultiTimeframeFeatureProvider:
    """
    Provider for multi-timeframe features from TimescaleDB.

    Uses continuous aggregates (candles_5m, candles_1h, etc.) to efficiently
    query different timeframe data and compute features across resolutions.
    """

    SUPPORTED_INTERVALS = [1, 5, 15, 60, 240]  # 1m, 5m, 15m, 1h, 4h

    INTERVAL_VIEWS = {
        1: 'candles',
        5: 'candles_5m',
        15: 'candles_15m',
        30: 'candles_30m',
        60: 'candles_1h',
        240: 'candles_4h',
        720: 'candles_12h',
        1440: 'candles_1d',
    }

    # Weights for combining timeframes (higher = more important)
    TF_WEIGHTS = {
        1: 0.10,   # 1m - noise
        5: 0.15,   # 5m - short-term
        15: 0.20,  # 15m - scalping
        60: 0.30,  # 1h - swing
        240: 0.25  # 4h - trend
    }

    def __init__(
        self,
        db_url: str,
        pool_min_size: int = 2,
        pool_max_size: int = 10
    ):
        if asyncpg is None:
            raise ImportError("asyncpg is required")

        self.db_url = db_url
        self.pool_min_size = pool_min_size
        self.pool_max_size = pool_max_size
        self.pool: Optional[asyncpg.Pool] = None

    async def connect(self):
        """Establish database connection."""
        self.pool = await asyncpg.create_pool(
            self.db_url,
            min_size=self.pool_min_size,
            max_size=self.pool_max_size
        )
        logger.info("MultiTimeframeFeatureProvider connected")

    async def close(self):
        """Close database connection."""
        if self.pool:
            await self.pool.close()

    async def get_candles(
        self,
        symbol: str,
        interval_minutes: int,
        end_time: datetime,
        lookback_candles: int = 100
    ) -> pd.DataFrame:
        """
        Query candles for a specific timeframe.

        Args:
            symbol: Trading pair
            interval_minutes: Candle interval
            end_time: End of query range
            lookback_candles: Number of candles to fetch

        Returns:
            DataFrame with OHLCV data
        """
        if not self.pool:
            raise RuntimeError("Not connected")

        view = self.INTERVAL_VIEWS.get(interval_minutes, 'candles')
        lookback = timedelta(minutes=interval_minutes * lookback_candles)
        start_time = end_time - lookback

        if view == 'candles':
            query = """
                SELECT timestamp, open, high, low, close, volume
                FROM candles
                WHERE symbol = $1
                  AND interval_minutes = $2
                  AND timestamp >= $3
                  AND timestamp < $4
                ORDER BY timestamp ASC
            """
            params = [symbol, interval_minutes, start_time, end_time]
        else:
            query = f"""
                SELECT timestamp, open, high, low, close, volume
                FROM {view}
                WHERE symbol = $1
                  AND timestamp >= $2
                  AND timestamp < $3
                ORDER BY timestamp ASC
            """
            params = [symbol, start_time, end_time]

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        if not rows:
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        df = pd.DataFrame([dict(row) for row in rows])
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)

        return df

    def compute_timeframe_features(
        self,
        df: pd.DataFrame,
        interval_minutes: int
    ) -> Optional[TimeframeFeatures]:
        """Compute features for a single timeframe."""
        if len(df) < 20:
            return None

        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']

        # EMA and trend direction
        ema_9 = close.ewm(span=9, adjust=False).mean()
        ema_21 = close.ewm(span=21, adjust=False).mean()

        # Trend direction based on EMA alignment and price position
        current_close = close.iloc[-1]
        current_ema_9 = ema_9.iloc[-1]
        current_ema_21 = ema_21.iloc[-1]

        if current_close > current_ema_9 > current_ema_21:
            trend_direction = 1.0
        elif current_close < current_ema_9 < current_ema_21:
            trend_direction = -1.0
        else:
            # Partial alignment
            trend_direction = (
                np.sign(current_close - current_ema_21) * 0.5 +
                np.sign(current_ema_9 - current_ema_21) * 0.5
            )

        # RSI
        rsi = self._calculate_rsi(close, 14)

        # MACD histogram
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        macd_hist = histogram.iloc[-1] / close.iloc[-1] * 100  # Normalize

        # ATR percentage
        atr = self._calculate_atr(df, 14)
        atr_pct = atr / close.iloc[-1] * 100

        # Volume ratio
        volume_sma = volume.rolling(20).mean()
        volume_ratio = volume.iloc[-1] / volume_sma.iloc[-1] if volume_sma.iloc[-1] > 0 else 1.0

        return TimeframeFeatures(
            interval_minutes=interval_minutes,
            trend_direction=float(trend_direction),
            rsi=float(rsi),
            macd_histogram=float(macd_hist),
            atr_pct=float(atr_pct),
            volume_ratio=float(volume_ratio)
        )

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        if loss.iloc[-1] == 0:
            return 100.0 if gain.iloc[-1] > 0 else 50.0

        rs = gain.iloc[-1] / loss.iloc[-1]
        return 100 - (100 / (1 + rs))

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR."""
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean().iloc[-1]

    async def compute_mtf_features(
        self,
        symbol: str,
        end_time: datetime,
        intervals: List[int] = None,
        lookback_candles: int = 100
    ) -> MTFFeatures:
        """
        Compute multi-timeframe features.

        Args:
            symbol: Trading pair
            end_time: Point in time for feature computation
            intervals: List of intervals to use (default: [1, 5, 15, 60, 240])
            lookback_candles: Candles to fetch per timeframe

        Returns:
            MTFFeatures with per-timeframe and aggregate features
        """
        if intervals is None:
            intervals = self.SUPPORTED_INTERVALS

        # Fetch all timeframes in parallel
        tasks = [
            self.get_candles(symbol, interval, end_time, lookback_candles)
            for interval in intervals
        ]
        candle_data = await asyncio.gather(*tasks)

        # Compute features per timeframe
        tf_features = {}
        for interval, df in zip(intervals, candle_data):
            features = self.compute_timeframe_features(df, interval)
            if features:
                tf_features[interval] = features

        # Compute aggregate features
        return self._aggregate_mtf_features(tf_features)

    def _aggregate_mtf_features(
        self,
        tf_features: Dict[int, TimeframeFeatures]
    ) -> MTFFeatures:
        """Aggregate per-timeframe features into composite metrics."""

        if not tf_features:
            return MTFFeatures(
                tf_features={},
                mtf_trend_alignment=0.0,
                tf_divergence_score=0.5,
                multi_resolution_volatility=0.0,
                dominant_trend=0.0,
                momentum_confluence=0.0
            )

        # Weighted trend alignment
        total_weight = 0.0
        weighted_trend = 0.0
        trends = []

        for interval, features in tf_features.items():
            weight = self.TF_WEIGHTS.get(interval, 0.1)
            weighted_trend += features.trend_direction * weight
            total_weight += weight
            trends.append(features.trend_direction)

        mtf_trend_alignment = weighted_trend / total_weight if total_weight > 0 else 0.0

        # Divergence score (how much timeframes disagree)
        if len(trends) > 1:
            tf_divergence_score = np.std(trends)
        else:
            tf_divergence_score = 0.0

        # Multi-resolution volatility (weighted average ATR)
        total_weight = 0.0
        weighted_vol = 0.0
        for interval, features in tf_features.items():
            weight = self.TF_WEIGHTS.get(interval, 0.1)
            weighted_vol += features.atr_pct * weight
            total_weight += weight

        multi_resolution_volatility = weighted_vol / total_weight if total_weight > 0 else 0.0

        # Dominant trend (higher timeframes have more influence)
        sorted_intervals = sorted(tf_features.keys(), reverse=True)
        dominant_trend = 0.0
        for interval in sorted_intervals[:2]:  # Top 2 timeframes
            dominant_trend += tf_features[interval].trend_direction

        dominant_trend = np.clip(dominant_trend / 2, -1.0, 1.0)

        # Momentum confluence (RSI and MACD agreement across timeframes)
        momentum_scores = []
        for features in tf_features.values():
            # RSI momentum: >50 bullish, <50 bearish
            rsi_momentum = (features.rsi - 50) / 50  # -1 to 1

            # MACD momentum
            macd_momentum = np.sign(features.macd_histogram)

            momentum_scores.append((rsi_momentum + macd_momentum) / 2)

        momentum_confluence = np.mean(momentum_scores) if momentum_scores else 0.0

        return MTFFeatures(
            tf_features=tf_features,
            mtf_trend_alignment=float(mtf_trend_alignment),
            tf_divergence_score=float(tf_divergence_score),
            multi_resolution_volatility=float(multi_resolution_volatility),
            dominant_trend=float(dominant_trend),
            momentum_confluence=float(momentum_confluence)
        )

    async def compute_features_for_candles(
        self,
        symbol: str,
        candle_timestamps: List[datetime],
        intervals: List[int] = None
    ) -> pd.DataFrame:
        """
        Compute MTF features for a list of timestamps (for ML training).

        Args:
            symbol: Trading pair
            candle_timestamps: List of timestamps to compute features for
            intervals: Timeframes to use

        Returns:
            DataFrame with MTF features indexed by timestamp
        """
        if intervals is None:
            intervals = self.SUPPORTED_INTERVALS

        results = []

        for ts in candle_timestamps:
            mtf = await self.compute_mtf_features(symbol, ts, intervals)

            row = {
                'timestamp': ts,
                'mtf_trend_alignment': mtf.mtf_trend_alignment,
                'tf_divergence_score': mtf.tf_divergence_score,
                'multi_resolution_volatility': mtf.multi_resolution_volatility,
                'dominant_trend': mtf.dominant_trend,
                'momentum_confluence': mtf.momentum_confluence
            }

            # Add per-timeframe features
            for interval, tf in mtf.tf_features.items():
                prefix = f'tf_{interval}m_'
                row[f'{prefix}trend'] = tf.trend_direction
                row[f'{prefix}rsi'] = tf.rsi
                row[f'{prefix}macd_hist'] = tf.macd_histogram
                row[f'{prefix}atr_pct'] = tf.atr_pct
                row[f'{prefix}volume_ratio'] = tf.volume_ratio

            results.append(row)

        return pd.DataFrame(results).set_index('timestamp')


async def enrich_features_with_mtf(
    features_df: pd.DataFrame,
    symbol: str,
    db_url: str,
    intervals: List[int] = None
) -> pd.DataFrame:
    """
    Enrich a features DataFrame with multi-timeframe features.

    Args:
        features_df: DataFrame with 'timestamp' column
        symbol: Trading pair
        db_url: Database connection URL
        intervals: Timeframes to use

    Returns:
        DataFrame with added MTF features
    """
    provider = MultiTimeframeFeatureProvider(db_url)

    try:
        await provider.connect()

        timestamps = features_df['timestamp'].tolist()
        mtf_df = await provider.compute_features_for_candles(
            symbol, timestamps, intervals
        )

        # Merge MTF features
        result = features_df.merge(
            mtf_df.reset_index(),
            on='timestamp',
            how='left'
        )

        # Fill NaN with neutral values
        fill_values = {
            'mtf_trend_alignment': 0.0,
            'tf_divergence_score': 0.0,
            'multi_resolution_volatility': 0.0,
            'dominant_trend': 0.0,
            'momentum_confluence': 0.0
        }

        for interval in (intervals or provider.SUPPORTED_INTERVALS):
            prefix = f'tf_{interval}m_'
            fill_values[f'{prefix}trend'] = 0.0
            fill_values[f'{prefix}rsi'] = 50.0
            fill_values[f'{prefix}macd_hist'] = 0.0
            fill_values[f'{prefix}atr_pct'] = 0.0
            fill_values[f'{prefix}volume_ratio'] = 1.0

        result.fillna(fill_values, inplace=True)

        return result

    finally:
        await provider.close()


def enrich_features_with_mtf_sync(
    features_df: pd.DataFrame,
    symbol: str,
    db_url: str,
    intervals: List[int] = None
) -> pd.DataFrame:
    """Synchronous wrapper for enrich_features_with_mtf."""
    return asyncio.run(
        enrich_features_with_mtf(features_df, symbol, db_url, intervals)
    )
