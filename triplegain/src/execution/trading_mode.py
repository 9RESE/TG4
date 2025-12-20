"""
Trading Mode Infrastructure - Paper vs Live execution control.

CRITICAL SAFETY:
- Paper trading is the DEFAULT mode
- Live trading requires dual confirmation (env + config)
- This prevents accidental real money trades during testing

Phase 6: Paper Trading Integration
"""

import logging
import os
from enum import Enum
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


class TradingMode(Enum):
    """Trading execution mode."""
    PAPER = "paper"
    LIVE = "live"


class TradingModeError(Exception):
    """Raised when trading mode validation fails."""
    pass


def load_execution_config() -> dict:
    """
    Load execution configuration from YAML file.

    Returns:
        Configuration dictionary
    """
    config_paths = [
        Path("config/execution.yaml"),
        Path("../config/execution.yaml"),
        Path(__file__).parent.parent.parent.parent / "config" / "execution.yaml",
    ]

    for path in config_paths:
        if path.exists():
            with open(path, 'r') as f:
                return yaml.safe_load(f) or {}

    logger.warning("execution.yaml not found, using defaults")
    return {}


def get_trading_mode(config: Optional[dict] = None) -> TradingMode:
    """
    Get the current trading mode.

    CRITICAL SAFETY: Defaults to PAPER unless explicitly set to LIVE
    via BOTH environment variable AND config file.

    Requires BOTH for LIVE mode:
    - TRIPLEGAIN_TRADING_MODE=live (env var)
    - execution.yaml trading_mode: live (config)

    This dual-confirmation prevents accidental live trading.

    Args:
        config: Optional pre-loaded config dict. If None, loads from file.

    Returns:
        TradingMode.PAPER or TradingMode.LIVE
    """
    if config is None:
        config = load_execution_config()

    env_mode = os.environ.get("TRIPLEGAIN_TRADING_MODE", "paper").lower().strip()
    config_mode = config.get("trading_mode", "paper")

    # Handle both string and dict config values
    if isinstance(config_mode, dict):
        config_mode = config_mode.get("mode", "paper")
    config_mode = str(config_mode).lower().strip()

    # Require BOTH env and config to agree on "live" mode
    if env_mode == "live" and config_mode == "live":
        logger.warning("⚠️ LIVE TRADING MODE ENABLED - REAL MONEY AT RISK ⚠️")
        return TradingMode.LIVE

    if env_mode == "live" or config_mode == "live":
        logger.warning(
            f"Live mode requested but not confirmed in both env and config. "
            f"Env: {env_mode}, Config: {config_mode}. Defaulting to PAPER."
        )

    return TradingMode.PAPER


def validate_trading_mode_on_startup(config: Optional[dict] = None) -> TradingMode:
    """
    Validate trading mode configuration at startup.

    CRITICAL: This function MUST be called during application startup
    to prevent accidental live trading.

    For LIVE mode, requires:
    1. Both env and config agree on "live"
    2. TRIPLEGAIN_CONFIRM_LIVE_TRADING='I_UNDERSTAND_THE_RISKS' env var
    3. KRAKEN_API_KEY and KRAKEN_API_SECRET env vars present

    Args:
        config: Optional pre-loaded config dict

    Returns:
        Current TradingMode

    Raises:
        TradingModeError: If live trading validation fails
    """
    if config is None:
        config = load_execution_config()

    mode = get_trading_mode(config)

    if mode == TradingMode.LIVE:
        # Require explicit confirmation for live trading
        confirm_value = os.environ.get("TRIPLEGAIN_CONFIRM_LIVE_TRADING", "")
        if confirm_value != "I_UNDERSTAND_THE_RISKS":
            raise TradingModeError(
                "SAFETY CHECK FAILED: Live trading requires explicit confirmation. "
                "Set TRIPLEGAIN_CONFIRM_LIVE_TRADING='I_UNDERSTAND_THE_RISKS' to proceed."
            )

        # Check Kraken credentials exist
        if not os.environ.get("KRAKEN_API_KEY"):
            raise TradingModeError("KRAKEN_API_KEY required for live trading")
        if not os.environ.get("KRAKEN_API_SECRET"):
            raise TradingModeError("KRAKEN_API_SECRET required for live trading")

        logger.critical("=" * 60)
        logger.critical("🔴 LIVE TRADING MODE - REAL MONEY TRANSACTIONS 🔴")
        logger.critical("=" * 60)
    else:
        logger.info("=" * 60)
        logger.info("🟢 PAPER TRADING MODE - SIMULATED EXECUTION 🟢")
        logger.info("=" * 60)

    return mode


def is_paper_mode(config: Optional[dict] = None) -> bool:
    """
    Check if currently in paper trading mode.

    Convenience function for quick checks.

    Args:
        config: Optional pre-loaded config dict

    Returns:
        True if in PAPER mode, False if LIVE
    """
    return get_trading_mode(config) == TradingMode.PAPER


def is_live_mode(config: Optional[dict] = None) -> bool:
    """
    Check if currently in live trading mode.

    Convenience function for quick checks.

    Args:
        config: Optional pre-loaded config dict

    Returns:
        True if in LIVE mode, False if PAPER
    """
    return get_trading_mode(config) == TradingMode.LIVE


def get_db_table_prefix(config: Optional[dict] = None) -> str:
    """
    Get database table prefix based on trading mode.

    Paper trading uses 'paper_' prefix for isolation.
    Live trading uses no prefix (production tables).

    Args:
        config: Optional pre-loaded config dict

    Returns:
        Table prefix string (e.g., "paper_" or "")
    """
    if config is None:
        config = load_execution_config()

    mode = get_trading_mode(config)

    if mode == TradingMode.PAPER:
        paper_config = config.get("paper_trading", {})
        return paper_config.get("db_table_prefix", "paper_")

    return ""  # Live mode uses production tables (no prefix)
