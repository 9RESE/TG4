"""
API Validation Utilities - Common validation helpers for API endpoints.

This module provides:
- Symbol format validation (Finding 23: loaded from config)
- Common parameter validation helpers (Finding 26: centralized)
- UUID validation
- Trade side validation
- Order type validation
- Position status validation

Security Fixes Applied (Phase 4 Review):
- Finding 23: SUPPORTED_SYMBOLS loaded from config
- Finding 26: Centralized validation functions
"""

import re
import logging
from typing import Optional, FrozenSet
from uuid import UUID

logger = logging.getLogger(__name__)


# Pattern for valid trading pair format: BASE/QUOTE or BASE_QUOTE (for URL paths)
SYMBOL_PATTERN = re.compile(r'^[A-Z0-9]{2,10}[/_][A-Z0-9]{2,10}$')

# Valid trade sides
VALID_SIDES = frozenset({"buy", "sell"})

# Valid order types
VALID_ORDER_TYPES = frozenset({"market", "limit", "stop-loss", "take-profit"})

# Valid position statuses
VALID_POSITION_STATUSES = frozenset({"open", "closed", "all"})

# Valid timeframes
VALID_TIMEFRAMES = frozenset({'1m', '5m', '15m', '30m', '1h', '4h', '12h', '1d', '1w'})

# Cache for supported symbols (loaded from config on first access)
_supported_symbols_cache: Optional[FrozenSet[str]] = None


def _load_symbols_from_config() -> FrozenSet[str]:
    """Load supported symbols from config file (Finding 23)."""
    try:
        from ..utils.config import get_config_loader
        config_loader = get_config_loader()

        # Try indicators.yaml first, then orchestration.yaml
        try:
            indicators_config = config_loader.get_indicators_config()
            symbols = indicators_config.get('symbols', [])
            if symbols:
                return frozenset(symbols)
        except Exception:
            pass

        # Fallback to orchestration config
        try:
            orch_config = config_loader.get_orchestration_config()
            symbols = orch_config.get('symbols', [])
            if symbols:
                return frozenset(symbols)
        except Exception:
            pass

    except Exception as e:
        logger.warning(f"Failed to load symbols from config: {e}. Using defaults.")

    # Fallback to hardcoded defaults
    return frozenset({
        "BTC/USDT",
        "XRP/USDT",
        "XRP/BTC",
    })


def get_supported_symbols() -> FrozenSet[str]:
    """Get the set of supported trading symbols (cached)."""
    global _supported_symbols_cache
    if _supported_symbols_cache is None:
        _supported_symbols_cache = _load_symbols_from_config()
        logger.info(f"Loaded {len(_supported_symbols_cache)} supported symbols")
    return _supported_symbols_cache


def reset_symbols_cache() -> None:
    """Reset the symbols cache (useful for testing or config reload)."""
    global _supported_symbols_cache
    _supported_symbols_cache = None


# Legacy constant for backwards compatibility (use get_supported_symbols() for dynamic loading)
SUPPORTED_SYMBOLS = frozenset({
    "BTC/USDT",
    "XRP/USDT",
    "XRP/BTC",
    "ETH/USDT",
    "ETH/BTC",
})


def normalize_symbol(symbol: str) -> str:
    """
    Normalize a symbol to standard format (BASE/QUOTE).

    Handles both slash and underscore separators.

    Args:
        symbol: Symbol string to normalize (e.g., "BTC_USDT" or "BTC/USDT")

    Returns:
        Normalized symbol with slash separator (e.g., "BTC/USDT")
    """
    return symbol.upper().strip().replace('_', '/')


def validate_symbol(symbol: str, strict: bool = True) -> tuple[bool, Optional[str]]:
    """
    Validate a trading symbol format.

    Args:
        symbol: Symbol string to validate (e.g., "BTC/USDT" or "BTC_USDT")
        strict: If True, only allows symbols in supported list (from config).
                If False, allows any valid format.

    Returns:
        Tuple of (is_valid, error_message)
        - (True, None) if valid
        - (False, "error description") if invalid
    """
    if not symbol:
        return False, "Symbol is required"

    # Normalize to uppercase
    symbol_upper = symbol.upper().strip()

    # Check format (accepts both / and _ separators)
    if not SYMBOL_PATTERN.match(symbol_upper):
        return False, f"Invalid symbol format: '{symbol}'. Expected format: BASE/QUOTE (e.g., BTC/USDT)"

    # Normalize to standard format for checking
    normalized = normalize_symbol(symbol_upper)

    # Check if in supported list (strict mode) - Finding 23: use dynamic loading
    if strict:
        supported = get_supported_symbols()
        if normalized not in supported:
            return False, f"Unsupported symbol: '{normalized}'. Supported: {', '.join(sorted(supported))}"

    return True, None


def validate_symbol_or_raise(symbol: str, strict: bool = True) -> str:
    """
    Validate symbol and raise HTTPException if invalid.

    Args:
        symbol: Symbol string to validate (e.g., "BTC/USDT" or "BTC_USDT")
        strict: If True, only allows symbols in SUPPORTED_SYMBOLS

    Returns:
        The validated and normalized symbol (e.g., "BTC/USDT")

    Raises:
        HTTPException: If symbol is invalid
    """
    try:
        from fastapi import HTTPException
    except ImportError:
        raise RuntimeError("FastAPI not available")

    is_valid, error_msg = validate_symbol(symbol, strict=strict)
    if not is_valid:
        logger.warning(f"Symbol validation failed: {error_msg}")
        raise HTTPException(status_code=400, detail=error_msg)

    # Return normalized symbol (with slash)
    return normalize_symbol(symbol)


def is_valid_side(side: str) -> bool:
    """Check if trade side is valid."""
    return side.lower() in VALID_SIDES


def is_valid_order_type(order_type: str) -> bool:
    """Check if order type is valid."""
    return order_type.lower() in VALID_ORDER_TYPES


def is_valid_position_status(status: str) -> bool:
    """Check if position status filter is valid."""
    return status.lower() in VALID_POSITION_STATUSES


def is_valid_timeframe(timeframe: str) -> bool:
    """Check if timeframe is valid."""
    return timeframe.lower() in VALID_TIMEFRAMES


def validate_uuid(value: str, field_name: str = "id") -> str:
    """
    Validate that a string is a valid UUID v4.

    Args:
        value: String to validate
        field_name: Name of the field for error messages

    Returns:
        The validated UUID string

    Raises:
        ValueError: If not a valid UUID
    """
    try:
        UUID(value, version=4)
        return value
    except (ValueError, AttributeError) as e:
        raise ValueError(f"Invalid {field_name} format: must be a valid UUID")


def validate_uuid_or_raise(value: str, field_name: str = "id") -> str:
    """
    Validate UUID and raise HTTPException if invalid.

    Args:
        value: String to validate
        field_name: Name of the field for error messages

    Returns:
        The validated UUID string

    Raises:
        HTTPException: If not a valid UUID
    """
    try:
        from fastapi import HTTPException
    except ImportError:
        raise RuntimeError("FastAPI not available")

    try:
        return validate_uuid(value, field_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


def validate_timeframe_or_raise(timeframe: str) -> str:
    """
    Validate timeframe and raise HTTPException if invalid.

    Args:
        timeframe: Timeframe string to validate

    Returns:
        The validated timeframe string

    Raises:
        HTTPException: If timeframe is invalid
    """
    try:
        from fastapi import HTTPException
    except ImportError:
        raise RuntimeError("FastAPI not available")

    if not is_valid_timeframe(timeframe):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid timeframe: '{timeframe}'. Valid: {', '.join(sorted(VALID_TIMEFRAMES))}"
        )
    return timeframe


def validate_side_or_raise(side: str) -> str:
    """
    Validate trade side and raise HTTPException if invalid.

    Args:
        side: Trade side to validate (buy/sell)

    Returns:
        The validated and lowercased side

    Raises:
        HTTPException: If side is invalid
    """
    try:
        from fastapi import HTTPException
    except ImportError:
        raise RuntimeError("FastAPI not available")

    side_lower = side.lower()
    if side_lower not in VALID_SIDES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid side: '{side}'. Valid: {', '.join(sorted(VALID_SIDES))}"
        )
    return side_lower
