"""
Whale Sentiment Strategy - Data Persistence

REC-011: Implements candle data persistence for fast restart recovery.
REC-037: Implements extreme zone state persistence for accurate tracking (v1.5.0).

This module saves strategy data to disk periodically and reloads it on startup:
- Candle data: eliminates 25+ hour warmup requirement after restarts
- Extreme zone state: maintains accurate extended fear tracking across restarts

File Formats:
- Candles: JSON array of candle objects ({timestamp, open, high, low, close, volume})
- State: JSON object with extreme zone tracking data

Validation:
- Checks file exists and is valid JSON
- Validates timestamps within max age limits
- Gracefully handles corruption (logs warning, starts fresh)
"""
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger('whale_sentiment')


def get_candle_file_path(
    symbol: str,
    config: Dict[str, Any],
    base_dir: str = None
) -> Path:
    """
    Get the file path for candle persistence.

    Args:
        symbol: Trading symbol (e.g., 'XRP/USDT')
        config: Strategy configuration
        base_dir: Optional base directory override

    Returns:
        Path object for the candle file
    """
    persistence_dir = base_dir or config.get('candle_persistence_dir', 'data/candles')
    file_format = config.get('candle_file_format', '{symbol}_5m.json')

    # Sanitize symbol for filename (replace / with _)
    safe_symbol = symbol.replace('/', '_')
    filename = file_format.format(symbol=safe_symbol)

    return Path(persistence_dir) / filename


def save_candles(
    symbol: str,
    candles: tuple,
    config: Dict[str, Any],
    state: Dict[str, Any],
    base_dir: str = None
) -> bool:
    """
    Save candle data to disk.

    REC-011: Saves candle buffer to JSON file for restart recovery.

    Args:
        symbol: Trading symbol
        candles: Tuple of candle objects
        config: Strategy configuration
        state: Strategy state (for tracking save count)
        base_dir: Optional base directory override

    Returns:
        True if save successful, False otherwise
    """
    if not config.get('use_candle_persistence', True):
        return False

    if not candles or len(candles) == 0:
        return False

    try:
        file_path = get_candle_file_path(symbol, config, base_dir)

        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert candles to serializable format
        candle_data = []
        for candle in candles:
            candle_dict = {
                'timestamp': candle.timestamp.isoformat() if hasattr(candle.timestamp, 'isoformat') else str(candle.timestamp),
                'open': float(candle.open),
                'high': float(candle.high),
                'low': float(candle.low),
                'close': float(candle.close),
                'volume': float(candle.volume),
            }
            candle_data.append(candle_dict)

        # Write to file
        with open(file_path, 'w') as f:
            json.dump({
                'symbol': symbol,
                'saved_at': datetime.now(timezone.utc).isoformat(),
                'candle_count': len(candle_data),
                'candles': candle_data,
            }, f, indent=2)

        # Track save count in state
        if 'candle_saves' not in state:
            state['candle_saves'] = {}
        state['candle_saves'][symbol] = state['candle_saves'].get(symbol, 0) + 1

        logger.debug(
            "Saved %d candles for %s to %s",
            len(candle_data), symbol, file_path
        )
        return True

    except Exception as e:
        logger.warning("Failed to save candles for %s: %s", symbol, e)
        return False


def load_candles(
    symbol: str,
    config: Dict[str, Any],
    candle_class: type,
    base_dir: str = None
) -> Tuple[Optional[List], Dict[str, Any]]:
    """
    Load candle data from disk.

    REC-011: Loads persisted candle data for fast restart recovery.

    Args:
        symbol: Trading symbol
        config: Strategy configuration
        candle_class: Candle class for reconstructing objects
        base_dir: Optional base directory override

    Returns:
        Tuple of (candles list or None, metadata dict)
    """
    metadata = {
        'loaded': False,
        'candle_count': 0,
        'file_path': None,
        'saved_at': None,
        'age_hours': None,
        'rejection_reason': None,
    }

    if not config.get('use_candle_persistence', True):
        metadata['rejection_reason'] = 'persistence_disabled'
        return None, metadata

    try:
        file_path = get_candle_file_path(symbol, config, base_dir)
        metadata['file_path'] = str(file_path)

        if not file_path.exists():
            metadata['rejection_reason'] = 'file_not_found'
            return None, metadata

        # Read file
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Validate structure
        if 'candles' not in data or not isinstance(data['candles'], list):
            metadata['rejection_reason'] = 'invalid_structure'
            return None, metadata

        candle_data = data['candles']
        if len(candle_data) == 0:
            metadata['rejection_reason'] = 'empty_file'
            return None, metadata

        # Check age of last candle
        last_candle = candle_data[-1]
        last_timestamp = datetime.fromisoformat(last_candle['timestamp'].replace('Z', '+00:00'))

        # Make timestamp timezone-aware if not already
        if last_timestamp.tzinfo is None:
            last_timestamp = last_timestamp.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)
        age_hours = (now - last_timestamp).total_seconds() / 3600
        metadata['age_hours'] = round(age_hours, 2)
        metadata['saved_at'] = data.get('saved_at')

        max_age = config.get('max_candle_age_hours', 4.0)
        if age_hours > max_age:
            metadata['rejection_reason'] = f'too_old ({age_hours:.1f}h > {max_age}h)'
            logger.warning(
                "Candle data for %s is too old (%.1fh > %.1fh), starting fresh",
                symbol, age_hours, max_age
            )
            return None, metadata

        # Reconstruct candle objects
        candles = []
        for cd in candle_data:
            timestamp = datetime.fromisoformat(cd['timestamp'].replace('Z', '+00:00'))
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)

            candle = candle_class(
                timestamp=timestamp,
                open=float(cd['open']),
                high=float(cd['high']),
                low=float(cd['low']),
                close=float(cd['close']),
                volume=float(cd['volume']),
            )
            candles.append(candle)

        metadata['loaded'] = True
        metadata['candle_count'] = len(candles)

        logger.info(
            "Loaded %d candles for %s (age: %.1fh)",
            len(candles), symbol, age_hours
        )
        return candles, metadata

    except json.JSONDecodeError as e:
        metadata['rejection_reason'] = f'json_error: {e}'
        logger.warning("Corrupted candle file for %s: %s", symbol, e)
        return None, metadata

    except Exception as e:
        metadata['rejection_reason'] = f'error: {e}'
        logger.warning("Failed to load candles for %s: %s", symbol, e)
        return None, metadata


def should_save_candles(
    symbol: str,
    candles: tuple,
    state: Dict[str, Any],
    config: Dict[str, Any]
) -> bool:
    """
    Determine if candles should be saved now.

    REC-011: Saves every N new candles based on configuration.

    Args:
        symbol: Trading symbol
        candles: Current candle buffer
        state: Strategy state
        config: Strategy configuration

    Returns:
        True if candles should be saved
    """
    if not config.get('use_candle_persistence', True):
        return False

    if not candles or len(candles) == 0:
        return False

    # Track last saved candle count per symbol
    if 'last_saved_candle_count' not in state:
        state['last_saved_candle_count'] = {}

    last_count = state['last_saved_candle_count'].get(symbol, 0)
    current_count = len(candles)
    save_interval = config.get('candle_save_interval_candles', 1)

    # Save if we have new candles beyond the interval
    if current_count >= last_count + save_interval:
        state['last_saved_candle_count'][symbol] = current_count
        return True

    return False


def delete_candle_file(
    symbol: str,
    config: Dict[str, Any],
    base_dir: str = None
) -> bool:
    """
    Delete persisted candle file.

    Args:
        symbol: Trading symbol
        config: Strategy configuration
        base_dir: Optional base directory override

    Returns:
        True if deleted successfully, False otherwise
    """
    try:
        file_path = get_candle_file_path(symbol, config, base_dir)
        if file_path.exists():
            file_path.unlink()
            logger.info("Deleted candle file for %s", symbol)
            return True
        return False
    except Exception as e:
        logger.warning("Failed to delete candle file for %s: %s", symbol, e)
        return False


def get_persistence_status(
    symbols: List[str],
    config: Dict[str, Any],
    base_dir: str = None
) -> Dict[str, Dict[str, Any]]:
    """
    Get status of persisted candle data for all symbols.

    Args:
        symbols: List of trading symbols
        config: Strategy configuration
        base_dir: Optional base directory override

    Returns:
        Dict mapping symbol to status info
    """
    status = {}

    for symbol in symbols:
        file_path = get_candle_file_path(symbol, config, base_dir)
        symbol_status = {
            'file_exists': file_path.exists(),
            'file_path': str(file_path),
            'file_size_kb': None,
            'candle_count': None,
            'age_hours': None,
        }

        if file_path.exists():
            try:
                symbol_status['file_size_kb'] = round(file_path.stat().st_size / 1024, 2)

                with open(file_path, 'r') as f:
                    data = json.load(f)

                symbol_status['candle_count'] = len(data.get('candles', []))

                if data.get('candles'):
                    last_ts = datetime.fromisoformat(
                        data['candles'][-1]['timestamp'].replace('Z', '+00:00')
                    )
                    if last_ts.tzinfo is None:
                        last_ts = last_ts.replace(tzinfo=timezone.utc)
                    age = (datetime.now(timezone.utc) - last_ts).total_seconds() / 3600
                    symbol_status['age_hours'] = round(age, 2)

            except Exception as e:
                symbol_status['error'] = str(e)

        status[symbol] = symbol_status

    return status


# =============================================================================
# REC-037: Extreme Zone State Persistence
# =============================================================================


def get_state_file_path(
    config: Dict[str, Any],
    base_dir: str = None
) -> Path:
    """
    REC-037: Get the file path for extreme zone state persistence.

    Args:
        config: Strategy configuration
        base_dir: Optional base directory override

    Returns:
        Path object for the state file
    """
    persistence_dir = base_dir or config.get('candle_persistence_dir', 'data/candles')
    return Path(persistence_dir) / 'whale_sentiment_state.json'


def save_extreme_zone_state(
    state: Dict[str, Any],
    config: Dict[str, Any],
    base_dir: str = None
) -> bool:
    """
    REC-037: Save extreme zone state to disk for restart recovery.

    Persists:
    - extreme_zone_start: When the strategy entered an extreme sentiment zone
    - extreme_zone_type: The type of extreme zone (EXTREME_FEAR or EXTREME_GREED)

    Args:
        state: Strategy state dict containing extreme zone tracking
        config: Strategy configuration
        base_dir: Optional base directory override

    Returns:
        True if save successful, False otherwise
    """
    if not config.get('use_candle_persistence', True):
        return False

    try:
        file_path = get_state_file_path(config, base_dir)

        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Extract extreme zone state
        extreme_zone_start = state.get('extreme_zone_start')
        extreme_zone_type = state.get('extreme_zone_type')

        state_data = {
            'saved_at': datetime.now(timezone.utc).isoformat(),
            'extreme_zone_start': extreme_zone_start.isoformat() if extreme_zone_start else None,
            'extreme_zone_type': extreme_zone_type,
        }

        # Write to file
        with open(file_path, 'w') as f:
            json.dump(state_data, f, indent=2)

        logger.debug(
            "Saved extreme zone state: zone_type=%s, start=%s",
            extreme_zone_type, extreme_zone_start
        )
        return True

    except Exception as e:
        logger.warning("Failed to save extreme zone state: %s", e)
        return False


def load_extreme_zone_state(
    config: Dict[str, Any],
    base_dir: str = None
) -> Dict[str, Any]:
    """
    REC-037: Load extreme zone state from disk.

    Validates that the saved state is still relevant (not too old).

    Args:
        config: Strategy configuration
        base_dir: Optional base directory override

    Returns:
        Dict with extreme_zone_start, extreme_zone_type, or empty values if not found/invalid
    """
    result = {
        'extreme_zone_start': None,
        'extreme_zone_type': None,
        'loaded': False,
        'rejection_reason': None,
    }

    if not config.get('use_candle_persistence', True):
        result['rejection_reason'] = 'persistence_disabled'
        return result

    try:
        file_path = get_state_file_path(config, base_dir)

        if not file_path.exists():
            result['rejection_reason'] = 'file_not_found'
            return result

        # Read file
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Parse extreme zone start
        extreme_zone_start_str = data.get('extreme_zone_start')
        if extreme_zone_start_str:
            extreme_zone_start = datetime.fromisoformat(
                extreme_zone_start_str.replace('Z', '+00:00')
            )
            if extreme_zone_start.tzinfo is None:
                extreme_zone_start = extreme_zone_start.replace(tzinfo=timezone.utc)

            # Validate age - state is valid if zone start is within reasonable time
            # Use extended_fear_pause_hours as max age (if zone was this old, it would pause anyway)
            max_age_hours = config.get('extended_fear_pause_hours', 168) * 1.5  # 150% of pause threshold
            age_hours = (datetime.now(timezone.utc) - extreme_zone_start).total_seconds() / 3600

            if age_hours > max_age_hours:
                result['rejection_reason'] = f'too_old ({age_hours:.1f}h > {max_age_hours:.1f}h)'
                logger.info(
                    "Extreme zone state too old (%.1fh > %.1fh), starting fresh",
                    age_hours, max_age_hours
                )
                return result

            result['extreme_zone_start'] = extreme_zone_start
            result['extreme_zone_type'] = data.get('extreme_zone_type')
            result['loaded'] = True

            logger.info(
                "Loaded extreme zone state: type=%s, duration=%.1fh",
                result['extreme_zone_type'], age_hours
            )

    except json.JSONDecodeError as e:
        result['rejection_reason'] = f'json_error: {e}'
        logger.warning("Corrupted state file: %s", e)

    except Exception as e:
        result['rejection_reason'] = f'error: {e}'
        logger.warning("Failed to load extreme zone state: %s", e)

    return result


def delete_extreme_zone_state(
    config: Dict[str, Any],
    base_dir: str = None
) -> bool:
    """
    REC-037: Delete persisted extreme zone state file.

    Should be called when exiting an extreme zone to clean up stale state.

    Args:
        config: Strategy configuration
        base_dir: Optional base directory override

    Returns:
        True if deleted successfully, False otherwise
    """
    try:
        file_path = get_state_file_path(config, base_dir)
        if file_path.exists():
            file_path.unlink()
            logger.debug("Deleted extreme zone state file")
            return True
        return False
    except Exception as e:
        logger.warning("Failed to delete state file: %s", e)
        return False
