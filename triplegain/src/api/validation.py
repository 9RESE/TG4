"""
API Validation Utilities - Common validation helpers for API endpoints.

This module provides:
- Symbol format validation
- Common parameter validation helpers
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# Supported symbols in the trading system
SUPPORTED_SYMBOLS = frozenset({
    "BTC/USDT",
    "XRP/USDT",
    "XRP/BTC",
    "ETH/USDT",
    "ETH/BTC",
})

# Pattern for valid trading pair format: BASE/QUOTE or BASE_QUOTE (for URL paths)
SYMBOL_PATTERN = re.compile(r'^[A-Z0-9]{2,10}[/_][A-Z0-9]{2,10}$')


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
        strict: If True, only allows symbols in SUPPORTED_SYMBOLS.
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

    # Check if in supported list (strict mode)
    if strict and normalized not in SUPPORTED_SYMBOLS:
        return False, f"Unsupported symbol: '{normalized}'. Supported: {', '.join(sorted(SUPPORTED_SYMBOLS))}"

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
    return side.lower() in {"buy", "sell"}


def is_valid_order_type(order_type: str) -> bool:
    """Check if order type is valid."""
    return order_type.lower() in {"market", "limit", "stop-loss", "take-profit"}


def is_valid_position_status(status: str) -> bool:
    """Check if position status filter is valid."""
    return status.lower() in {"open", "closed", "all"}
