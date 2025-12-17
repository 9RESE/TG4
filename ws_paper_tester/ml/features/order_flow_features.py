"""
Order Flow Features - Compute order flow features from historic trades.

Computes real trade_imbalance and VPIN from the trades table in TimescaleDB,
replacing placeholder values in the feature extractor.

Features computed:
- trade_imbalance: Buy volume - Sell volume / Total volume
- vpin: Volume-Synchronized Probability of Informed Trading
- order_flow_toxicity: Combined toxicity score
- buy_sell_ratio: Buy volume / Sell volume
- trade_intensity: Trades per minute
- avg_trade_size: Average trade size
- large_trade_ratio: Ratio of large trades (>2x avg size)
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd

try:
    import asyncpg
except ImportError:
    asyncpg = None

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HistoricTrade:
    """Trade data from database."""
    symbol: str
    timestamp: datetime
    price: Decimal
    volume: Decimal
    side: str  # 'buy' or 'sell'

    @classmethod
    def from_row(cls, row) -> 'HistoricTrade':
        """Create from database row (asyncpg.Record or dict-like object)."""
        return cls(
            symbol=row['symbol'],
            timestamp=row['timestamp'],
            price=row['price'],
            volume=row['volume'],
            side=row['side']
        )


@dataclass
class OrderFlowFeatures:
    """Container for computed order flow features."""
    trade_imbalance: float
    vpin: float
    order_flow_toxicity: float
    buy_sell_ratio: float
    trade_intensity: float
    avg_trade_size: float
    large_trade_ratio: float
    trade_count: int


class OrderFlowFeatureProvider:
    """
    Provider for order flow features computed from historic trades.

    Connects to TimescaleDB and computes order flow features for
    specified time windows, enabling ML models to use real order
    flow data instead of placeholders.
    """

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
        logger.info("OrderFlowFeatureProvider connected")

    async def close(self):
        """Close database connection."""
        if self.pool:
            await self.pool.close()

    async def get_trades(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        limit: Optional[int] = None
    ) -> List[HistoricTrade]:
        """
        Query trades from database.

        Args:
            symbol: Trading pair
            start: Start time (inclusive)
            end: End time (exclusive)
            limit: Maximum number of trades

        Returns:
            List of HistoricTrade objects
        """
        if not self.pool:
            raise RuntimeError("Not connected. Call await connect() first.")

        query = """
            SELECT symbol, timestamp, price, volume, side
            FROM trades
            WHERE symbol = $1
              AND timestamp >= $2
              AND timestamp < $3
            ORDER BY timestamp ASC
        """

        if limit:
            query += f" LIMIT {limit}"

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, symbol, start, end)

        return [HistoricTrade.from_row(row) for row in rows]

    async def compute_order_flow_features(
        self,
        symbol: str,
        end_time: datetime,
        lookback_minutes: int = 60,
        vpin_buckets: int = 50
    ) -> OrderFlowFeatures:
        """
        Compute order flow features for a time window.

        Args:
            symbol: Trading pair
            end_time: End of computation window
            lookback_minutes: Minutes of trade data to use
            vpin_buckets: Number of buckets for VPIN calculation

        Returns:
            OrderFlowFeatures with computed metrics
        """
        start_time = end_time - timedelta(minutes=lookback_minutes)
        trades = await self.get_trades(symbol, start_time, end_time)

        if not trades:
            return OrderFlowFeatures(
                trade_imbalance=0.0,
                vpin=0.5,
                order_flow_toxicity=0.5,
                buy_sell_ratio=1.0,
                trade_intensity=0.0,
                avg_trade_size=0.0,
                large_trade_ratio=0.0,
                trade_count=0
            )

        return self._compute_features_from_trades(
            trades, lookback_minutes, vpin_buckets
        )

    def _compute_features_from_trades(
        self,
        trades: List[HistoricTrade],
        window_minutes: int,
        vpin_buckets: int
    ) -> OrderFlowFeatures:
        """Compute features from trade list."""

        # Basic volume calculations
        buy_volume = sum(
            float(t.volume) for t in trades if t.side == 'buy'
        )
        sell_volume = sum(
            float(t.volume) for t in trades if t.side == 'sell'
        )
        total_volume = buy_volume + sell_volume

        # Trade imbalance: [-1, 1] range
        if total_volume > 0:
            trade_imbalance = (buy_volume - sell_volume) / total_volume
        else:
            trade_imbalance = 0.0

        # Buy/Sell ratio
        if sell_volume > 0:
            buy_sell_ratio = buy_volume / sell_volume
        else:
            buy_sell_ratio = 2.0 if buy_volume > 0 else 1.0

        # VPIN calculation
        vpin = self._calculate_vpin(trades, vpin_buckets)

        # Order flow toxicity (combine imbalance and VPIN)
        order_flow_toxicity = (abs(trade_imbalance) + vpin) / 2

        # Trade intensity (trades per minute)
        trade_intensity = len(trades) / window_minutes if window_minutes > 0 else 0.0

        # Average trade size
        volumes = [float(t.volume) for t in trades]
        avg_trade_size = np.mean(volumes) if volumes else 0.0

        # Large trade ratio (trades > 2x average)
        if avg_trade_size > 0:
            large_trades = sum(1 for v in volumes if v > 2 * avg_trade_size)
            large_trade_ratio = large_trades / len(trades)
        else:
            large_trade_ratio = 0.0

        return OrderFlowFeatures(
            trade_imbalance=trade_imbalance,
            vpin=vpin,
            order_flow_toxicity=order_flow_toxicity,
            buy_sell_ratio=min(buy_sell_ratio, 10.0),  # Cap at 10
            trade_intensity=trade_intensity,
            avg_trade_size=avg_trade_size,
            large_trade_ratio=large_trade_ratio,
            trade_count=len(trades)
        )

    def _calculate_vpin(
        self,
        trades: List[HistoricTrade],
        bucket_count: int = 50
    ) -> float:
        """
        Calculate VPIN from trade list.

        VPIN (Volume-Synchronized Probability of Informed Trading) measures
        order flow toxicity by dividing trades into equal-volume buckets
        and measuring the imbalance in each.
        """
        if len(trades) < bucket_count:
            return 0.5  # Neutral when insufficient data

        # Calculate total volume and bucket size
        total_volume = sum(float(t.volume) for t in trades)
        if total_volume <= 0:
            return 0.5

        bucket_volume = total_volume / bucket_count

        # Build volume buckets
        buckets = []
        current_bucket_buy = 0.0
        current_bucket_sell = 0.0
        cumulative_volume = 0.0
        bucket_boundary = bucket_volume

        for trade in trades:
            trade_volume = float(trade.volume)
            trade_buy = trade_volume if trade.side == 'buy' else 0.0
            trade_sell = trade_volume if trade.side == 'sell' else 0.0

            remaining_buy = trade_buy
            remaining_sell = trade_sell
            remaining_volume = trade_volume

            while remaining_volume > 0 and len(buckets) < bucket_count:
                space_in_bucket = bucket_boundary - cumulative_volume
                volume_for_bucket = min(remaining_volume, space_in_bucket)

                if remaining_volume > 0:
                    proportion = volume_for_bucket / remaining_volume
                    buy_portion = remaining_buy * proportion
                    sell_portion = remaining_sell * proportion

                    current_bucket_buy += buy_portion
                    current_bucket_sell += sell_portion
                    cumulative_volume += volume_for_bucket

                    remaining_buy -= buy_portion
                    remaining_sell -= sell_portion
                    remaining_volume -= volume_for_bucket

                # Check if bucket is complete
                if cumulative_volume >= bucket_boundary - 1e-10:
                    bucket_total = current_bucket_buy + current_bucket_sell
                    if bucket_total > 0:
                        bucket_imbalance = abs(
                            current_bucket_buy - current_bucket_sell
                        ) / bucket_total
                        buckets.append(bucket_imbalance)

                    current_bucket_buy = 0.0
                    current_bucket_sell = 0.0
                    bucket_boundary += bucket_volume

        # Handle final partial bucket
        if current_bucket_buy + current_bucket_sell > bucket_volume * 0.5:
            bucket_total = current_bucket_buy + current_bucket_sell
            if bucket_total > 0:
                bucket_imbalance = abs(
                    current_bucket_buy - current_bucket_sell
                ) / bucket_total
                buckets.append(bucket_imbalance)

        if not buckets:
            return 0.5

        return sum(buckets) / len(buckets)

    async def compute_features_for_candles(
        self,
        symbol: str,
        candle_timestamps: List[datetime],
        lookback_minutes: int = 60,
        vpin_buckets: int = 50
    ) -> pd.DataFrame:
        """
        Compute order flow features for a list of candle timestamps.

        This is optimized for ML training - computes features in bulk
        for multiple timestamps.

        Args:
            symbol: Trading pair
            candle_timestamps: List of candle end times
            lookback_minutes: Minutes of trade data per candle
            vpin_buckets: VPIN bucket count

        Returns:
            DataFrame with order flow features indexed by timestamp
        """
        if not candle_timestamps:
            return pd.DataFrame()

        # Query all trades for the full range
        start_time = min(candle_timestamps) - timedelta(minutes=lookback_minutes)
        end_time = max(candle_timestamps)

        all_trades = await self.get_trades(symbol, start_time, end_time)

        if not all_trades:
            # Return DataFrame with neutral values
            return pd.DataFrame({
                'timestamp': candle_timestamps,
                'trade_imbalance': 0.0,
                'vpin': 0.5,
                'order_flow_toxicity': 0.5,
                'buy_sell_ratio': 1.0,
                'trade_intensity': 0.0,
                'avg_trade_size': 0.0,
                'large_trade_ratio': 0.0,
                'trade_count': 0
            }).set_index('timestamp')

        # Convert to DataFrame for efficient filtering
        trades_df = pd.DataFrame([
            {
                'timestamp': t.timestamp,
                'price': float(t.price),
                'volume': float(t.volume),
                'side': t.side
            }
            for t in all_trades
        ])
        trades_df.set_index('timestamp', inplace=True)

        # Compute features for each candle
        results = []
        for ts in candle_timestamps:
            window_start = ts - timedelta(minutes=lookback_minutes)
            window_trades = trades_df[
                (trades_df.index >= window_start) &
                (trades_df.index < ts)
            ]

            if len(window_trades) == 0:
                results.append({
                    'timestamp': ts,
                    'trade_imbalance': 0.0,
                    'vpin': 0.5,
                    'order_flow_toxicity': 0.5,
                    'buy_sell_ratio': 1.0,
                    'trade_intensity': 0.0,
                    'avg_trade_size': 0.0,
                    'large_trade_ratio': 0.0,
                    'trade_count': 0
                })
                continue

            # Compute features
            buy_vol = window_trades[window_trades['side'] == 'buy']['volume'].sum()
            sell_vol = window_trades[window_trades['side'] == 'sell']['volume'].sum()
            total_vol = buy_vol + sell_vol

            imbalance = (buy_vol - sell_vol) / total_vol if total_vol > 0 else 0.0

            # Simplified VPIN for bulk computation
            vpin = self._compute_vpin_from_df(window_trades, vpin_buckets)

            toxicity = (abs(imbalance) + vpin) / 2

            buy_sell = buy_vol / sell_vol if sell_vol > 0 else (2.0 if buy_vol > 0 else 1.0)

            intensity = len(window_trades) / lookback_minutes

            avg_size = window_trades['volume'].mean()

            large_count = (window_trades['volume'] > 2 * avg_size).sum()
            large_ratio = large_count / len(window_trades) if len(window_trades) > 0 else 0.0

            results.append({
                'timestamp': ts,
                'trade_imbalance': imbalance,
                'vpin': vpin,
                'order_flow_toxicity': toxicity,
                'buy_sell_ratio': min(buy_sell, 10.0),
                'trade_intensity': intensity,
                'avg_trade_size': avg_size,
                'large_trade_ratio': large_ratio,
                'trade_count': len(window_trades)
            })

        return pd.DataFrame(results).set_index('timestamp')

    def _compute_vpin_from_df(
        self,
        trades_df: pd.DataFrame,
        bucket_count: int
    ) -> float:
        """Compute VPIN from DataFrame (for bulk processing)."""
        if len(trades_df) < bucket_count:
            return 0.5

        total_volume = trades_df['volume'].sum()
        if total_volume <= 0:
            return 0.5

        bucket_volume = total_volume / bucket_count

        # Simpler approach: group into buckets by cumulative volume
        trades_df = trades_df.copy()
        trades_df['cum_vol'] = trades_df['volume'].cumsum()
        trades_df['bucket'] = (trades_df['cum_vol'] / bucket_volume).astype(int)
        trades_df['bucket'] = trades_df['bucket'].clip(upper=bucket_count - 1)

        bucket_imbalances = []
        for bucket_id in range(bucket_count):
            bucket_trades = trades_df[trades_df['bucket'] == bucket_id]
            if len(bucket_trades) == 0:
                continue

            buy_vol = bucket_trades[bucket_trades['side'] == 'buy']['volume'].sum()
            sell_vol = bucket_trades[bucket_trades['side'] == 'sell']['volume'].sum()
            total = buy_vol + sell_vol

            if total > 0:
                imbalance = abs(buy_vol - sell_vol) / total
                bucket_imbalances.append(imbalance)

        if not bucket_imbalances:
            return 0.5

        return sum(bucket_imbalances) / len(bucket_imbalances)


async def enrich_features_with_order_flow(
    features_df: pd.DataFrame,
    symbol: str,
    db_url: str,
    lookback_minutes: int = 60,
    vpin_buckets: int = 50
) -> pd.DataFrame:
    """
    Enrich a features DataFrame with real order flow features.

    Replaces placeholder trade_imbalance and vpin values with
    actual computed values from historic trades.

    Args:
        features_df: DataFrame with 'timestamp' column
        symbol: Trading pair
        db_url: Database connection URL
        lookback_minutes: Trade lookback window
        vpin_buckets: VPIN bucket count

    Returns:
        DataFrame with real order flow features
    """
    provider = OrderFlowFeatureProvider(db_url)

    try:
        await provider.connect()

        timestamps = features_df['timestamp'].tolist()
        order_flow_df = await provider.compute_features_for_candles(
            symbol, timestamps, lookback_minutes, vpin_buckets
        )

        # Merge order flow features
        result = features_df.copy()

        # Replace placeholder columns with real values
        for col in ['trade_imbalance', 'vpin']:
            if col in result.columns and col in order_flow_df.columns:
                result = result.drop(columns=[col])

        # Add all order flow features
        result = result.merge(
            order_flow_df.reset_index(),
            on='timestamp',
            how='left'
        )

        # Fill any NaN with neutral values
        fill_values = {
            'trade_imbalance': 0.0,
            'vpin': 0.5,
            'order_flow_toxicity': 0.5,
            'buy_sell_ratio': 1.0,
            'trade_intensity': 0.0,
            'avg_trade_size': 0.0,
            'large_trade_ratio': 0.0,
            'trade_count': 0
        }
        result.fillna(fill_values, inplace=True)

        return result

    finally:
        await provider.close()


# Convenience function for synchronous usage
def enrich_features_with_order_flow_sync(
    features_df: pd.DataFrame,
    symbol: str,
    db_url: str,
    lookback_minutes: int = 60,
    vpin_buckets: int = 50
) -> pd.DataFrame:
    """Synchronous wrapper for enrich_features_with_order_flow."""
    return asyncio.run(
        enrich_features_with_order_flow(
            features_df, symbol, db_url, lookback_minutes, vpin_buckets
        )
    )
