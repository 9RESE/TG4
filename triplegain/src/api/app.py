"""
FastAPI Application - Main API server for TripleGain.

This module provides:
- Health check endpoints
- Indicator endpoints for testing
- Snapshot endpoints for testing
- Debug endpoints for prompt inspection

SECURITY: All endpoints (except health) require authentication.
See security.py for rate limiting, CORS, and other protections.
"""

import logging
import re
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

try:
    from fastapi import FastAPI, HTTPException, Query, Depends
    from fastapi.responses import JSONResponse
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = None

from ..data.database import DatabasePool, create_pool_from_config
from ..data.indicator_library import IndicatorLibrary
from ..data.market_snapshot import MarketSnapshotBuilder
from ..llm.prompt_builder import PromptBuilder
from ..utils.config import get_config_loader, ConfigError
from .security import setup_security, get_security_config

logger = logging.getLogger(__name__)

# Global instances (set during lifespan)
_db_pool: Optional[DatabasePool] = None
_indicator_library: Optional[IndicatorLibrary] = None
_snapshot_builder: Optional[MarketSnapshotBuilder] = None
_prompt_builder: Optional[PromptBuilder] = None

# Symbol validation regex: BASE/QUOTE or BASE_QUOTE format (e.g., BTC/USDT, XRP_BTC)
# Allows alphanumeric symbols with optional numbers (e.g., USDT, 1INCH)
# Underscore variant is URL-safe and commonly used in API paths
SYMBOL_PATTERN = re.compile(r'^[A-Z0-9]{2,10}[/_][A-Z0-9]{2,10}$')

# Valid timeframes
VALID_TIMEFRAMES = {'1m', '5m', '15m', '30m', '1h', '4h', '12h', '1d', '1w'}


def validate_symbol(symbol: str) -> str:
    """
    Validate trading symbol format.

    Args:
        symbol: Trading pair (e.g., "BTC/USDT")

    Returns:
        Validated symbol (uppercase)

    Raises:
        HTTPException: If symbol format is invalid
    """
    symbol = symbol.upper()
    if not SYMBOL_PATTERN.match(symbol):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid symbol format: '{symbol}'. Expected format: BASE/QUOTE (e.g., BTC/USDT)"
        )
    return symbol


def validate_timeframe(timeframe: str) -> str:
    """
    Validate timeframe format.

    Args:
        timeframe: Candle timeframe (e.g., "1h")

    Returns:
        Validated timeframe

    Raises:
        HTTPException: If timeframe is invalid
    """
    if timeframe not in VALID_TIMEFRAMES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid timeframe: '{timeframe}'. Valid: {', '.join(sorted(VALID_TIMEFRAMES))}"
        )
    return timeframe


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown."""
    global _db_pool, _indicator_library, _snapshot_builder, _prompt_builder

    try:
        # Load configurations
        config_loader = get_config_loader()
        db_config = config_loader.get_database_config()
        indicator_config = config_loader.get_indicators_config()
        snapshot_config = config_loader.get_snapshot_config()
        prompts_config = config_loader.get_prompts_config()

        # Initialize database pool
        _db_pool = create_pool_from_config(db_config)
        await _db_pool.connect()
        logger.info("Database pool connected")

        # Initialize indicator library
        _indicator_library = IndicatorLibrary(indicator_config, _db_pool)
        logger.info("Indicator library initialized")

        # Initialize snapshot builder
        _snapshot_builder = MarketSnapshotBuilder(_db_pool, _indicator_library, snapshot_config)
        logger.info("Snapshot builder initialized")

        # Initialize prompt builder
        _prompt_builder = PromptBuilder(prompts_config)
        logger.info("Prompt builder initialized")

        yield

    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        # Still yield to allow graceful shutdown
        yield

    finally:
        # Shutdown
        if _db_pool:
            await _db_pool.disconnect()
            logger.info("Database pool disconnected")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    if not FASTAPI_AVAILABLE:
        raise RuntimeError("FastAPI is not installed. Install with: pip install fastapi")

    app = FastAPI(
        title="TripleGain API",
        description="LLM-Assisted Trading System API",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Set up security FIRST (middleware order matters)
    # This adds: request size limits, timeouts, rate limiting, CORS
    security_config = get_security_config()
    setup_security(app, security_config)

    # Register routes
    register_health_routes(app)
    register_indicator_routes(app)
    register_snapshot_routes(app)
    register_debug_routes(app)

    return app


def register_health_routes(app: FastAPI):
    """Register health check endpoints."""

    @app.get("/health")
    async def health_check():
        """
        Basic health check endpoint.

        Returns:
            Health status with component statuses
        """
        status = {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "components": {}
        }

        # Check database
        if _db_pool:
            db_health = await _db_pool.check_health()
            status["components"]["database"] = db_health
            if db_health.get("status") != "healthy":
                status["status"] = "degraded"
        else:
            status["components"]["database"] = {"status": "not_initialized"}
            status["status"] = "degraded"

        # Check indicator library
        status["components"]["indicator_library"] = {
            "status": "healthy" if _indicator_library else "not_initialized"
        }

        # Check snapshot builder
        status["components"]["snapshot_builder"] = {
            "status": "healthy" if _snapshot_builder else "not_initialized"
        }

        # Check prompt builder
        if _prompt_builder:
            template_count = len(_prompt_builder._templates)
            status["components"]["prompt_builder"] = {
                "status": "healthy",
                "templates_loaded": template_count
            }
        else:
            status["components"]["prompt_builder"] = {"status": "not_initialized"}

        return status

    @app.get("/health/live")
    async def liveness_check():
        """Kubernetes liveness probe endpoint."""
        return {"status": "alive"}

    @app.get("/health/ready")
    async def readiness_check():
        """Kubernetes readiness probe endpoint."""
        if _db_pool and _db_pool.is_connected:
            return {"status": "ready"}
        raise HTTPException(status_code=503, detail="Database not ready")


def register_indicator_routes(app: FastAPI):
    """Register indicator-related endpoints."""

    @app.get("/api/v1/indicators/{symbol}/{timeframe}")
    async def get_indicators(
        symbol: str,
        timeframe: str,
        limit: int = Query(default=100, ge=1, le=1000)
    ):
        """
        Calculate and return indicators for a symbol/timeframe.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            timeframe: Candle timeframe (e.g., "1h")
            limit: Number of candles to fetch

        Returns:
            Calculated indicators
        """
        # Validate inputs
        symbol = validate_symbol(symbol)
        timeframe = validate_timeframe(timeframe)

        if not _db_pool or not _indicator_library:
            raise HTTPException(status_code=503, detail="Service not initialized")

        try:
            # Fetch candles
            candles = await _db_pool.fetch_candles(symbol, timeframe, limit)

            if not candles:
                raise HTTPException(status_code=404, detail="No data found for symbol/timeframe")

            # Calculate indicators
            indicators = _indicator_library.calculate_all(symbol, timeframe, candles)

            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "candle_count": len(candles),
                "indicators": indicators,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to calculate indicators: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error calculating indicators")


def register_snapshot_routes(app: FastAPI):
    """Register snapshot-related endpoints."""

    @app.get("/api/v1/snapshot/{symbol}")
    async def get_snapshot(
        symbol: str,
        include_order_book: bool = Query(default=True)
    ):
        """
        Build and return a market snapshot for a symbol.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            include_order_book: Whether to include order book data

        Returns:
            Market snapshot in prompt format
        """
        # Validate inputs
        symbol = validate_symbol(symbol)

        if not _snapshot_builder:
            raise HTTPException(status_code=503, detail="Service not initialized")

        try:
            snapshot = await _snapshot_builder.build_snapshot(symbol, include_order_book)

            return {
                "symbol": symbol,
                "snapshot": {
                    "timestamp": snapshot.timestamp.isoformat(),
                    "current_price": float(snapshot.current_price),
                    "price_24h_ago": float(snapshot.price_24h_ago) if snapshot.price_24h_ago else None,
                    "price_change_24h_pct": float(snapshot.price_change_24h_pct) if snapshot.price_change_24h_pct else None,
                    "data_age_seconds": snapshot.data_age_seconds,
                    "data_quality_issues": snapshot.missing_data_flags,
                },
                "indicators": snapshot.indicators,
                "mtf_state": snapshot.mtf_state.to_dict() if snapshot.mtf_state else None,
                "order_book": snapshot.order_book.to_dict() if snapshot.order_book else None,
            }

        except Exception as e:
            logger.error(f"Failed to build snapshot: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error building snapshot")


def register_debug_routes(app: FastAPI):
    """Register debug endpoints for testing."""

    @app.get("/api/v1/debug/prompt/{agent}")
    async def get_debug_prompt(
        agent: str,
        symbol: str = Query(default="BTC/USDT")
    ):
        """
        Generate and return a debug prompt for an agent.

        Args:
            agent: Agent name (e.g., "technical_analysis")
            symbol: Trading pair for snapshot

        Returns:
            Assembled prompt components
        """
        # Validate inputs
        symbol = validate_symbol(symbol)

        if not _prompt_builder or not _snapshot_builder:
            raise HTTPException(status_code=503, detail="Service not initialized")

        try:
            # Build snapshot
            snapshot = await _snapshot_builder.build_snapshot(symbol)

            # Build prompt
            prompt = _prompt_builder.build_prompt(agent, snapshot)

            return {
                "agent": agent,
                "symbol": symbol,
                "prompt": {
                    "system_prompt": prompt.system_prompt,
                    "user_message": prompt.user_message,
                    "estimated_tokens": prompt.estimated_tokens,
                    "tier": prompt.tier,
                }
            }

        except ValueError as e:
            # ValueError is expected for invalid agent names - safe to expose
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Failed to generate debug prompt: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error generating prompt")

    @app.get("/api/v1/debug/config")
    async def get_config():
        """Return current configuration (sanitized)."""
        try:
            config_loader = get_config_loader()
            return {
                "indicators": list(config_loader.get_indicators_config().keys()),
                "snapshot": {
                    "timeframes": list(config_loader.get_snapshot_config().get('candle_lookback', {}).keys()),
                },
                "database": {
                    "connected": _db_pool.is_connected if _db_pool else False,
                }
            }
        except Exception as e:
            logger.error(f"Failed to get config: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error retrieving config")


# Singleton app instance
_app: Optional[FastAPI] = None


def get_app() -> FastAPI:
    """Get or create the singleton FastAPI app instance."""
    global _app
    if _app is None:
        _app = create_app()
    return _app


# For running directly with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("triplegain.src.api.app:get_app", host="0.0.0.0", port=8000, reload=True)
