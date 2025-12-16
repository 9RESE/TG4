"""
EMA-9 Trend Flip Strategy - Database Warmup Integration

Provides historical data warmup from TimescaleDB for faster strategy initialization.
Uses the data.historical_provider module to fetch historical candles.

Code Review Fix: Issue #8 - Database/Historical Data Integration
"""
import asyncio
import logging
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple

from .config import STRATEGY_NAME, SYMBOLS
from .indicators import calculate_ema_series

# Configure logger
logger = logging.getLogger(STRATEGY_NAME)


# Type alias for candle data
CandleDict = Dict[str, Any]


async def fetch_warmup_candles(
    symbol: str,
    interval_minutes: int,
    num_candles: int,
    db_url: Optional[str] = None
) -> List[CandleDict]:
    """
    Fetch historical candles from database for warmup.

    Args:
        symbol: Trading pair (e.g., 'BTC/USDT')
        interval_minutes: Candle interval in minutes (60 for 1H)
        num_candles: Number of candles to fetch
        db_url: Database URL (uses env var WS_TESTER_DB_URL if None)

    Returns:
        List of candle dicts with timestamp, open, high, low, close, volume
    """
    try:
        # Import here to handle optional dependency
        from data.historical_provider import HistoricalDataProvider
    except ImportError:
        logger.warning("HistoricalDataProvider not available, skipping DB warmup")
        return []

    # Get database URL
    if db_url is None:
        db_url = os.environ.get('WS_TESTER_DB_URL')
        if not db_url:
            logger.debug("No database URL configured, skipping DB warmup")
            return []

    provider = None
    try:
        provider = HistoricalDataProvider(db_url)
        await provider.connect()

        # Calculate time range
        end_time = datetime.now(timezone.utc)
        # Add buffer for partial candles
        start_time = end_time - timedelta(minutes=interval_minutes * (num_candles + 5))

        logger.info(f"Fetching {num_candles} candles for {symbol} from database", extra={
            'symbol': symbol,
            'interval_minutes': interval_minutes,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
        })

        # Fetch candles
        candles = await provider.get_candles(
            symbol=symbol,
            interval_minutes=interval_minutes,
            start_time=start_time,
            end_time=end_time
        )

        # Convert to dict format expected by indicators
        candle_dicts = []
        for candle in candles:
            candle_dicts.append({
                'timestamp': candle.timestamp,
                'open': float(candle.open),
                'high': float(candle.high),
                'low': float(candle.low),
                'close': float(candle.close),
                'volume': float(candle.volume),
            })

        logger.info(f"Fetched {len(candle_dicts)} candles from database", extra={
            'symbol': symbol,
            'candle_count': len(candle_dicts),
        })

        return candle_dicts[-num_candles:]  # Return only requested number

    except Exception as e:
        logger.warning(f"Database warmup failed: {e}", extra={
            'error': str(e),
            'symbol': symbol,
        })
        return []

    finally:
        if provider:
            await provider.close()


def warmup_from_db_sync(
    symbol: str,
    interval_minutes: int,
    num_candles: int,
    db_url: Optional[str] = None
) -> List[CandleDict]:
    """
    Synchronous wrapper for fetch_warmup_candles.

    Args:
        symbol: Trading pair
        interval_minutes: Candle interval in minutes
        num_candles: Number of candles to fetch
        db_url: Database URL

    Returns:
        List of candle dicts
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If already in an async context, create a new loop
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    fetch_warmup_candles(symbol, interval_minutes, num_candles, db_url)
                )
                return future.result(timeout=30)
        else:
            return loop.run_until_complete(
                fetch_warmup_candles(symbol, interval_minutes, num_candles, db_url)
            )
    except Exception as e:
        logger.warning(f"Sync warmup failed: {e}")
        return []


def initialize_warmup_state(
    state: Dict[str, Any],
    config: Dict[str, Any]
) -> bool:
    """
    Initialize strategy state with historical data from database.

    This should be called during on_start() if use_db_warmup is enabled.
    Pre-calculates EMA values to avoid cold-start period.

    Args:
        state: Strategy state dict to update
        config: Strategy configuration

    Returns:
        True if warmup was successful, False otherwise
    """
    if not config.get('use_db_warmup', True):
        logger.debug("Database warmup disabled")
        return False

    symbol = SYMBOLS[0] if SYMBOLS else 'BTC/USDT'
    interval_minutes = config.get('candle_timeframe_minutes', 60)
    num_candles = config.get('warmup_candles_1h', 100)
    db_url = config.get('db_url')

    logger.info(f"Starting database warmup for {symbol}", extra={
        'symbol': symbol,
        'interval_minutes': interval_minutes,
        'num_candles': num_candles,
    })

    # Fetch historical candles
    candles = warmup_from_db_sync(symbol, interval_minutes, num_candles, db_url)

    if not candles:
        logger.info("No candles fetched, warmup skipped")
        return False

    # Store warmed-up candles
    state['hourly_candles'] = candles
    state['warmup_candle_count'] = len(candles)
    state['warmup_complete'] = True
    state['warmup_symbol'] = symbol

    # Pre-calculate EMA series
    ema_period = config.get('ema_period', 9)
    use_open = config.get('use_open_price', True)

    if use_open:
        prices = [c['open'] for c in candles]
    else:
        prices = [c['close'] for c in candles]

    ema_values = calculate_ema_series(prices, ema_period)
    state['ema_values'] = ema_values

    # Log warmup summary
    current_ema = ema_values[-1] if ema_values and ema_values[-1] is not None else None

    logger.info(f"Warmup complete: {len(candles)} candles loaded", extra={
        'symbol': symbol,
        'candle_count': len(candles),
        'ema_period': ema_period,
        'current_ema': round(current_ema, 2) if current_ema else None,
        'oldest_candle': candles[0]['timestamp'].isoformat() if candles else None,
        'newest_candle': candles[-1]['timestamp'].isoformat() if candles else None,
    })

    return True


def merge_warmup_with_realtime(
    warmup_candles: List[CandleDict],
    realtime_candles: List[CandleDict],
    timeframe_minutes: int = 60
) -> List[CandleDict]:
    """
    Merge warmed-up historical candles with incoming real-time candles.

    Handles deduplication based on timestamp and ensures chronological order.

    Args:
        warmup_candles: Historical candles from database
        realtime_candles: Real-time candles from WebSocket
        timeframe_minutes: Candle timeframe in minutes

    Returns:
        Merged list of candles in chronological order
    """
    if not warmup_candles:
        return realtime_candles

    if not realtime_candles:
        return warmup_candles

    # Create timestamp-based lookup for deduplication
    seen_timestamps = set()
    merged = []

    # Add warmup candles first
    for candle in warmup_candles:
        ts = candle['timestamp']
        # Normalize timestamp to timeframe boundary
        if hasattr(ts, 'minute'):
            minute_floor = (ts.minute // timeframe_minutes) * timeframe_minutes
            ts_key = ts.replace(minute=minute_floor, second=0, microsecond=0)
        else:
            ts_key = ts

        if ts_key not in seen_timestamps:
            seen_timestamps.add(ts_key)
            merged.append(candle)

    # Add real-time candles, updating existing or adding new
    for candle in realtime_candles:
        ts = candle['timestamp']
        if hasattr(ts, 'minute'):
            minute_floor = (ts.minute // timeframe_minutes) * timeframe_minutes
            ts_key = ts.replace(minute=minute_floor, second=0, microsecond=0)
        else:
            ts_key = ts

        if ts_key not in seen_timestamps:
            seen_timestamps.add(ts_key)
            merged.append(candle)
        else:
            # Update existing candle with more recent data
            for i, existing in enumerate(merged):
                existing_ts = existing['timestamp']
                if hasattr(existing_ts, 'minute'):
                    existing_floor = (existing_ts.minute // timeframe_minutes) * timeframe_minutes
                    existing_key = existing_ts.replace(minute=existing_floor, second=0, microsecond=0)
                else:
                    existing_key = existing_ts

                if existing_key == ts_key:
                    # Update high/low/close/volume
                    merged[i] = {
                        'timestamp': existing['timestamp'],  # Keep original timestamp
                        'open': existing['open'],  # Keep original open
                        'high': max(existing['high'], candle['high']),
                        'low': min(existing['low'], candle['low']),
                        'close': candle['close'],  # Use latest close
                        'volume': existing['volume'] + candle.get('volume', 0),
                    }
                    break

    # Sort by timestamp
    merged.sort(key=lambda x: x['timestamp'])

    return merged


def check_warmup_status(state: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Check if warmup is complete and ready for trading.

    Args:
        state: Strategy state dict

    Returns:
        Tuple of (is_ready, status_message)
    """
    if state.get('warmup_complete'):
        candle_count = state.get('warmup_candle_count', 0)
        return (True, f"Warmup complete with {candle_count} candles")

    if 'warmup_error' in state:
        return (False, f"Warmup failed: {state['warmup_error']}")

    return (False, "Warmup not started")
