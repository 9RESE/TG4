"""
Market Snapshot Builder - Aggregates market data for LLM consumption.

This module builds complete market snapshots that include:
- Current price and 24h change
- Candles across multiple timeframes
- Pre-computed technical indicators
- Order book features
- Multi-timeframe analysis
- Data quality metrics

Snapshots can be converted to full or compact prompt formats.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional, Any, TYPE_CHECKING
import json

from .indicator_library import IndicatorLibrary

if TYPE_CHECKING:
    from .database import DatabasePool

logger = logging.getLogger(__name__)


@dataclass
class CandleSummary:
    """Compact candle representation for prompts."""
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal

    def to_compact(self) -> dict:
        """Convert to compact dictionary format."""
        return {
            't': self.timestamp.isoformat(),
            'o': float(self.open),
            'h': float(self.high),
            'l': float(self.low),
            'c': float(self.close),
            'v': float(self.volume),
        }

    def to_dict(self) -> dict:
        """Convert to full dictionary format."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'open': float(self.open),
            'high': float(self.high),
            'low': float(self.low),
            'close': float(self.close),
            'volume': float(self.volume),
        }


@dataclass
class OrderBookFeatures:
    """Extracted order book features."""
    bid_depth_usd: Decimal
    ask_depth_usd: Decimal
    imbalance: Decimal  # -1 to 1 (negative = more asks)
    spread_bps: Decimal
    weighted_mid: Decimal

    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        return {
            'bid_depth_usd': float(self.bid_depth_usd),
            'ask_depth_usd': float(self.ask_depth_usd),
            'imbalance': float(self.imbalance),
            'spread_bps': float(self.spread_bps),
            'weighted_mid': float(self.weighted_mid),
        }


@dataclass
class MultiTimeframeState:
    """Aggregated state across timeframes."""
    trend_alignment_score: Decimal  # -1 to 1
    aligned_bullish_count: int
    aligned_bearish_count: int
    total_timeframes: int
    rsi_by_timeframe: dict[str, Decimal]
    atr_by_timeframe: dict[str, Decimal]

    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        return {
            'trend_alignment_score': float(self.trend_alignment_score),
            'aligned_bullish': self.aligned_bullish_count,
            'aligned_bearish': self.aligned_bearish_count,
            'total_timeframes': self.total_timeframes,
            'rsi_by_tf': {k: float(v) for k, v in self.rsi_by_timeframe.items()},
            'atr_by_tf': {k: float(v) for k, v in self.atr_by_timeframe.items()},
        }


@dataclass
class MarketSnapshot:
    """
    Complete market state snapshot for LLM consumption.

    This is the primary data structure passed to agents.
    """
    # Identification
    timestamp: datetime
    symbol: str

    # Current price
    current_price: Decimal
    price_24h_ago: Optional[Decimal] = None
    price_change_24h_pct: Optional[Decimal] = None

    # Candles by timeframe
    candles: dict[str, list[CandleSummary]] = field(default_factory=dict)

    # Pre-computed indicators
    indicators: dict[str, Any] = field(default_factory=dict)

    # Order book
    order_book: Optional[OrderBookFeatures] = None

    # Multi-timeframe analysis
    mtf_state: Optional[MultiTimeframeState] = None

    # Volume analysis
    volume_24h: Optional[Decimal] = None
    volume_vs_avg: Optional[Decimal] = None  # Ratio vs 20-period avg

    # Cached regime (from previous detection)
    regime_hint: Optional[str] = None
    regime_confidence: Optional[Decimal] = None

    # Data quality
    data_age_seconds: int = 0
    missing_data_flags: list[str] = field(default_factory=list)

    def to_prompt_format(self, token_budget: int = 4000) -> str:
        """
        Convert snapshot to LLM-friendly JSON string.

        Automatically truncates to fit within token budget.
        """
        data = {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'current_price': float(self.current_price),
        }

        if self.price_change_24h_pct is not None:
            data['price_change_24h_pct'] = float(self.price_change_24h_pct)

        # Add indicators
        if self.indicators:
            data['indicators'] = self._serialize_indicators(self.indicators)

        # Add candles (limited to save tokens)
        if self.candles:
            data['candles'] = {}
            for tf, candles in self.candles.items():
                # Limit to last 10 candles per timeframe
                data['candles'][tf] = [c.to_dict() for c in candles[-10:]]

        # Add order book features
        if self.order_book:
            data['order_book'] = self.order_book.to_dict()

        # Add MTF state
        if self.mtf_state:
            data['mtf_state'] = self.mtf_state.to_dict()

        # Add volume metrics
        if self.volume_vs_avg:
            data['volume_vs_avg'] = float(self.volume_vs_avg)

        # Add regime hint
        if self.regime_hint:
            data['regime_hint'] = self.regime_hint

        # Add data quality info
        data['data_age_seconds'] = self.data_age_seconds
        if self.missing_data_flags:
            data['data_quality_issues'] = self.missing_data_flags

        # Serialize and check size
        json_str = json.dumps(data, separators=(',', ':'))

        # Estimate tokens (conservative: 3.5 chars per token)
        estimated_tokens = len(json_str) / 3.5

        # If over budget, truncate candles further
        if estimated_tokens > token_budget and 'candles' in data:
            for tf in data['candles']:
                data['candles'][tf] = data['candles'][tf][-5:]
            json_str = json.dumps(data, separators=(',', ':'))

        return json_str

    def to_compact_format(self) -> str:
        """
        Minimal format for Tier 1 (local) LLM prompts.

        Includes only essential data to minimize latency.
        """
        data = {
            'ts': self.timestamp.isoformat(),
            'sym': self.symbol,
            'px': float(self.current_price),
        }

        # Add key indicators with short keys
        if self.indicators:
            if 'rsi_14' in self.indicators and self.indicators['rsi_14'] is not None:
                data['rsi'] = round(self.indicators['rsi_14'], 1)

            macd = self.indicators.get('macd', {})
            if isinstance(macd, dict) and macd.get('histogram') is not None:
                data['macd_h'] = round(macd['histogram'], 2)

            if 'atr_14' in self.indicators and self.indicators['atr_14'] is not None:
                data['atr'] = round(self.indicators['atr_14'], 1)

            if 'adx_14' in self.indicators and self.indicators['adx_14'] is not None:
                data['adx'] = round(self.indicators['adx_14'], 1)

            bb = self.indicators.get('bollinger_bands', {})
            if isinstance(bb, dict) and bb.get('position') is not None:
                data['bb_pos'] = round(bb['position'], 2)

        # Add trend alignment
        if self.mtf_state:
            data['trend'] = round(float(self.mtf_state.trend_alignment_score), 2)

        # Add regime hint (shortened)
        if self.regime_hint:
            regime_map = {
                'trending_bull': 'bull',
                'trending_bear': 'bear',
                'ranging': 'range',
                'high_volatility': 'hvol',
                'low_volatility': 'lvol',
            }
            data['regime'] = regime_map.get(self.regime_hint, self.regime_hint[:4])

        return json.dumps(data, separators=(',', ':'))

    def _serialize_indicators(self, indicators: dict) -> dict:
        """Serialize indicators, handling nested dicts and special types."""
        result = {}
        for key, value in indicators.items():
            if value is None:
                continue
            if isinstance(value, dict):
                # Handle nested dicts (like MACD, Bollinger Bands)
                result[key] = {k: v for k, v in value.items() if v is not None}
            elif isinstance(value, (int, float, Decimal)):
                result[key] = float(value) if isinstance(value, Decimal) else value
            else:
                result[key] = value
        return result


class MarketSnapshotBuilder:
    """
    Builds complete market snapshots for agent consumption.
    """

    def __init__(
        self,
        db_pool: Optional['DatabasePool'],
        indicator_library: IndicatorLibrary,
        config: dict
    ):
        """
        Initialize MarketSnapshotBuilder.

        Args:
            db_pool: DatabasePool instance (or None for sync operations)
            indicator_library: IndicatorLibrary instance
            config: Snapshot configuration
        """
        self.db = db_pool
        self.indicators = indicator_library
        self.config = config

    async def build_snapshot(
        self,
        symbol: str,
        include_order_book: bool = True
    ) -> MarketSnapshot:
        """
        Build complete market snapshot for a symbol.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            include_order_book: Whether to fetch order book data

        Returns:
            MarketSnapshot with all data populated
        """
        if self.db is None:
            raise RuntimeError("Database pool required for async build_snapshot")

        start_time = datetime.now(timezone.utc)
        logger.debug(f"Building snapshot for {symbol}")

        # Get timeframe config
        lookback_config = self.config.get('candle_lookback', {
            '1m': 60, '5m': 48, '15m': 32, '1h': 48, '4h': 30, '1d': 30
        })

        # Fetch candles for all timeframes in parallel
        timeframes = list(lookback_config.keys())
        candle_tasks = [
            self.db.fetch_candles(symbol, tf, lookback_config.get(tf, 50))
            for tf in timeframes
        ]

        # Fetch 24h data and order book in parallel with candles
        data_24h_task = self.db.fetch_24h_data(symbol)
        order_book_task = self.db.fetch_order_book(symbol) if include_order_book else None

        # Gather all async tasks
        if order_book_task:
            results = await asyncio.gather(
                *candle_tasks, data_24h_task, order_book_task,
                return_exceptions=True
            )
            candle_results = results[:len(timeframes)]
            data_24h = results[len(timeframes)]
            order_book_data = results[len(timeframes) + 1]
        else:
            results = await asyncio.gather(
                *candle_tasks, data_24h_task,
                return_exceptions=True
            )
            candle_results = results[:len(timeframes)]
            data_24h = results[len(timeframes)]
            order_book_data = None

        # Build candles dict by timeframe and track failures
        candles_by_tf = {}
        failed_timeframes = []
        for tf, candles in zip(timeframes, candle_results):
            if isinstance(candles, Exception):
                logger.warning(f"Failed to fetch {tf} candles for {symbol}: {candles}")
                failed_timeframes.append(tf)
                continue
            if candles:
                candles_by_tf[tf] = candles

        # Handle 24h data errors
        data_24h_failed = False
        if isinstance(data_24h, Exception):
            logger.warning(f"Failed to fetch 24h data for {symbol}: {data_24h}")
            data_24h = {}
            data_24h_failed = True

        # Check failure threshold - fail if >50% of data sources failed
        primary_timeframe = self.config.get('primary_timeframe', '1h')
        failure_threshold = self.config.get('data_quality', {}).get('max_failure_rate', 0.5)
        total_sources = len(timeframes) + 1  # timeframes + 24h data
        failed_sources = len(failed_timeframes) + (1 if data_24h_failed else 0)
        failure_rate = failed_sources / total_sources if total_sources > 0 else 0

        if failure_rate > failure_threshold:
            logger.error(
                f"Data source failure rate {failure_rate:.1%} exceeds threshold {failure_threshold:.0%} "
                f"for {symbol}. Failed: {failed_timeframes}"
            )
            raise RuntimeError(
                f"Too many data sources failed ({failed_sources}/{total_sources}). "
                f"Cannot build reliable snapshot for {symbol}"
            )

        # Warn if primary timeframe failed but we can still build with fallback
        if primary_timeframe in failed_timeframes:
            logger.warning(
                f"Primary timeframe {primary_timeframe} failed for {symbol}, "
                f"using fallback timeframe"
            )

        # Get current price from most recent candle
        current_price = Decimal('0')
        snapshot_timestamp = start_time

        # Use primary timeframe if available, otherwise use first available
        if primary_timeframe in candles_by_tf and candles_by_tf[primary_timeframe]:
            candles = candles_by_tf[primary_timeframe]
            current_price = Decimal(str(candles[-1].get('close', 0)))
            snapshot_timestamp = candles[-1].get('timestamp', start_time)
        else:
            for tf_candles in candles_by_tf.values():
                if tf_candles:
                    current_price = Decimal(str(tf_candles[-1].get('close', 0)))
                    snapshot_timestamp = tf_candles[-1].get('timestamp', start_time)
                    break

        # Calculate indicators from primary timeframe
        indicators = {}
        if primary_timeframe in candles_by_tf:
            indicators = self.indicators.calculate_all(
                symbol, primary_timeframe, candles_by_tf[primary_timeframe]
            )

        # Process order book
        order_book_features = None
        if order_book_data and not isinstance(order_book_data, Exception):
            order_book_features = self._process_order_book(order_book_data)

        # Calculate MTF state
        mtf_state = self._calculate_mtf_state(candles_by_tf)

        # Convert candles to CandleSummary
        candle_summaries = {}
        for tf, candles in candles_by_tf.items():
            candle_summaries[tf] = [
                CandleSummary(
                    timestamp=c.get('timestamp', start_time),
                    open=Decimal(str(c.get('open', 0))),
                    high=Decimal(str(c.get('high', 0))),
                    low=Decimal(str(c.get('low', 0))),
                    close=Decimal(str(c.get('close', 0))),
                    volume=Decimal(str(c.get('volume', 0))),
                )
                for c in candles
            ]

        # Calculate data age
        elapsed = (datetime.now(timezone.utc) - snapshot_timestamp).total_seconds()

        snapshot = MarketSnapshot(
            timestamp=snapshot_timestamp,
            symbol=symbol,
            current_price=current_price,
            price_24h_ago=Decimal(str(data_24h.get('price_24h_ago', 0))) if data_24h.get('price_24h_ago') else None,
            price_change_24h_pct=Decimal(str(data_24h.get('price_change_24h_pct', 0))) if data_24h.get('price_change_24h_pct') else None,
            candles=candle_summaries,
            indicators=indicators,
            order_book=order_book_features,
            mtf_state=mtf_state,
            volume_24h=Decimal(str(data_24h.get('volume_24h', 0))) if data_24h.get('volume_24h') else None,
            volume_vs_avg=Decimal(str(indicators.get('volume_vs_avg', 0))) if indicators.get('volume_vs_avg') else None,
            data_age_seconds=int(elapsed),
        )

        # Validate data quality
        issues = self._validate_data_quality(snapshot)
        snapshot.missing_data_flags = issues

        logger.debug(f"Built snapshot for {symbol} in {(datetime.now(timezone.utc) - start_time).total_seconds():.3f}s")

        return snapshot

    async def build_multi_symbol_snapshot(
        self,
        symbols: list[str]
    ) -> dict[str, MarketSnapshot]:
        """
        Build snapshots for multiple symbols in parallel.

        Args:
            symbols: List of trading pairs

        Returns:
            Dict of symbol -> MarketSnapshot
        """
        tasks = [self.build_snapshot(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        snapshots = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to build snapshot for {symbol}: {result}")
                continue
            snapshots[symbol] = result

        return snapshots

    def build_snapshot_from_candles(
        self,
        symbol: str,
        candles_by_tf: dict[str, list[dict]],
        order_book: Optional[dict] = None,
    ) -> MarketSnapshot:
        """
        Build snapshot from provided candles (synchronous, for testing).

        Args:
            symbol: Trading pair
            candles_by_tf: Dict of timeframe -> candle list
            order_book: Optional order book data

        Returns:
            MarketSnapshot with calculated indicators
        """
        now = datetime.now(timezone.utc)

        # Get current price and timestamp from most recent candle
        current_price = Decimal('0')
        snapshot_timestamp = now
        latest_candles = None
        primary_timeframe = self.config.get('primary_timeframe', '1h')

        # Prefer primary timeframe if available
        if primary_timeframe in candles_by_tf and candles_by_tf[primary_timeframe]:
            latest_candles = candles_by_tf[primary_timeframe]
            current_price = Decimal(str(latest_candles[-1].get('close', 0)))
            snapshot_timestamp = latest_candles[-1].get('timestamp', now)
        else:
            for tf, candles in candles_by_tf.items():
                if candles:
                    latest_candles = candles
                    current_price = Decimal(str(candles[-1].get('close', 0)))
                    snapshot_timestamp = candles[-1].get('timestamp', now)
                    break

        # Calculate 24h price change if we have enough data
        price_24h_ago = None
        price_change_24h_pct = None
        volume_24h = None

        # Calculate hours per candle based on timeframe
        timeframe_hours = {
            '1m': 1/60, '5m': 5/60, '15m': 0.25, '30m': 0.5,
            '1h': 1, '4h': 4, '12h': 12, '1d': 24, '1w': 168
        }
        candles_per_24h = int(24 / timeframe_hours.get(primary_timeframe, 1))

        if latest_candles and len(latest_candles) >= candles_per_24h and candles_per_24h > 0:
            price_24h_ago = Decimal(str(latest_candles[-candles_per_24h].get('close', 0)))
            if price_24h_ago and price_24h_ago > 0:
                price_change_24h_pct = ((current_price - price_24h_ago) / price_24h_ago) * 100
            # Calculate 24h volume
            volume_24h = Decimal(str(sum(c.get('volume', 0) for c in latest_candles[-candles_per_24h:])))

        # Calculate indicators from primary timeframe
        indicators = {}
        if latest_candles:
            indicators = self.indicators.calculate_all(symbol, primary_timeframe, latest_candles)

        # Process order book
        order_book_features = None
        if order_book:
            order_book_features = self._process_order_book(order_book)

        # Calculate MTF state
        mtf_state = self._calculate_mtf_state(candles_by_tf)

        # Convert candles to CandleSummary
        candle_summaries = {}
        for tf, candles in candles_by_tf.items():
            candle_summaries[tf] = [
                CandleSummary(
                    timestamp=c.get('timestamp', now),
                    open=Decimal(str(c.get('open', 0))),
                    high=Decimal(str(c.get('high', 0))),
                    low=Decimal(str(c.get('low', 0))),
                    close=Decimal(str(c.get('close', 0))),
                    volume=Decimal(str(c.get('volume', 0))),
                )
                for c in candles
            ]

        # Calculate data age
        if isinstance(snapshot_timestamp, datetime):
            data_age = int((now - snapshot_timestamp).total_seconds())
        else:
            data_age = 0

        snapshot = MarketSnapshot(
            timestamp=snapshot_timestamp,
            symbol=symbol,
            current_price=current_price,
            price_24h_ago=price_24h_ago,
            price_change_24h_pct=price_change_24h_pct,
            candles=candle_summaries,
            indicators=indicators,
            order_book=order_book_features,
            mtf_state=mtf_state,
            volume_24h=volume_24h,
            volume_vs_avg=Decimal(str(indicators.get('volume_vs_avg', 0))) if indicators.get('volume_vs_avg') else None,
            data_age_seconds=data_age,
        )

        # Validate data quality
        issues = self._validate_data_quality(snapshot)
        snapshot.missing_data_flags = issues

        return snapshot

    def _process_order_book(self, order_book: dict) -> OrderBookFeatures:
        """
        Process raw order book data into features.

        Args:
            order_book: Raw order book with 'bids' and 'asks' lists

        Returns:
            OrderBookFeatures extracted from the book
        """
        bids = order_book.get('bids', [])
        asks = order_book.get('asks', [])

        # Calculate depth in USD
        bid_depth = sum(b.get('price', 0) * b.get('size', 0) for b in bids)
        ask_depth = sum(a.get('price', 0) * a.get('size', 0) for a in asks)

        # Calculate imbalance (-1 to 1)
        total_depth = bid_depth + ask_depth
        if total_depth > 0:
            imbalance = (bid_depth - ask_depth) / total_depth
        else:
            imbalance = 0

        # Calculate spread in basis points
        best_bid = bids[0].get('price', 0) if bids else 0
        best_ask = asks[0].get('price', 0) if asks else 0

        if best_bid > 0 and best_ask > 0:
            mid_price = (best_bid + best_ask) / 2
            spread_bps = ((best_ask - best_bid) / mid_price) * 10000
        else:
            mid_price = best_bid or best_ask
            spread_bps = 0

        # Calculate volume-weighted mid price
        total_bid_vol = sum(b.get('size', 0) for b in bids)
        total_ask_vol = sum(a.get('size', 0) for a in asks)

        if total_bid_vol > 0 and total_ask_vol > 0:
            weighted_bid = sum(b.get('price', 0) * b.get('size', 0) for b in bids) / total_bid_vol
            weighted_ask = sum(a.get('price', 0) * a.get('size', 0) for a in asks) / total_ask_vol
            weighted_mid = (weighted_bid + weighted_ask) / 2
        else:
            weighted_mid = mid_price

        return OrderBookFeatures(
            bid_depth_usd=Decimal(str(bid_depth)),
            ask_depth_usd=Decimal(str(ask_depth)),
            imbalance=Decimal(str(round(imbalance, 4))),
            spread_bps=Decimal(str(round(spread_bps, 2))),
            weighted_mid=Decimal(str(round(weighted_mid, 2))),
        )

    def _calculate_mtf_state(
        self,
        candles_by_tf: dict[str, list[dict]]
    ) -> MultiTimeframeState:
        """
        Calculate multi-timeframe alignment state.

        Args:
            candles_by_tf: Dict of timeframe -> candle list

        Returns:
            MultiTimeframeState with trend alignment analysis
        """
        bullish_count = 0
        bearish_count = 0
        rsi_by_tf = {}
        atr_by_tf = {}

        for tf, candles in candles_by_tf.items():
            if not candles or len(candles) < 20:
                continue

            # Calculate indicators for this timeframe
            indicators = self.indicators.calculate_all('', tf, candles)

            # Get EMA trend direction
            ema_9 = indicators.get('ema_9')
            ema_21 = indicators.get('ema_21')

            if ema_9 is not None and ema_21 is not None:
                if ema_9 > ema_21:
                    bullish_count += 1
                else:
                    bearish_count += 1

            # Store RSI and ATR by timeframe
            rsi = indicators.get('rsi_14')
            if rsi is not None:
                rsi_by_tf[tf] = Decimal(str(round(rsi, 1)))

            atr = indicators.get('atr_14')
            if atr is not None:
                atr_by_tf[tf] = Decimal(str(round(atr, 2)))

        total = bullish_count + bearish_count
        if total > 0:
            # Score from -1 (all bearish) to 1 (all bullish)
            alignment_score = (bullish_count - bearish_count) / total
        else:
            alignment_score = 0

        return MultiTimeframeState(
            trend_alignment_score=Decimal(str(round(alignment_score, 2))),
            aligned_bullish_count=bullish_count,
            aligned_bearish_count=bearish_count,
            total_timeframes=total,
            rsi_by_timeframe=rsi_by_tf,
            atr_by_timeframe=atr_by_tf,
        )

    def _validate_data_quality(
        self,
        snapshot: MarketSnapshot
    ) -> list[str]:
        """
        Validate snapshot data quality, return list of issues.

        Args:
            snapshot: MarketSnapshot to validate

        Returns:
            List of issue strings
        """
        issues = []

        # Check data age
        max_age = self.config.get('data_quality', {}).get('max_age_seconds', 60)
        if snapshot.data_age_seconds > max_age:
            issues.append('stale_data')

        # Check for missing indicators
        if not snapshot.indicators:
            issues.append('no_indicators')

        # Check candle counts
        min_candles = self.config.get('data_quality', {}).get('min_candles_required', 20)
        for tf, candles in snapshot.candles.items():
            if len(candles) < min_candles:
                issues.append(f'insufficient_{tf}_candles')

        # Check for order book
        if snapshot.order_book is None:
            issues.append('no_order_book')

        return issues
