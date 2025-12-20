"""
FastAPI Application - Main API server for TripleGain.

This module provides:
- Health check endpoints
- Indicator endpoints for testing
- Snapshot endpoints for testing
- Debug endpoints for prompt inspection (protected in production)

SECURITY: All endpoints (except health) require authentication.
See security.py for rate limiting, CORS, and other protections.

Security Fixes Applied (Phase 4 Review):
- Finding 15: Global exception handler added
- Finding 16: Debug endpoints require authentication in production
- Finding 22: Exception handling order fixed (HTTPException before generic Exception)
"""

import logging
import traceback
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

try:
    from fastapi import FastAPI, HTTPException, Query, Depends, Request
    from fastapi.responses import JSONResponse
    from fastapi.exceptions import RequestValidationError
    from pydantic import ValidationError
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = None

from ..data.database import DatabasePool, create_pool_from_config
from ..data.indicator_library import IndicatorLibrary
from ..data.market_snapshot import MarketSnapshotBuilder
from ..llm.prompt_builder import PromptBuilder
from ..utils.config import get_config_loader, ConfigError
from .security import (
    setup_security,
    get_security_config,
    get_current_user,
    require_role,
    User,
    UserRole,
)
from .validation import (
    validate_symbol_or_raise,
    validate_timeframe_or_raise,
)

logger = logging.getLogger(__name__)

# Global instances (set during lifespan)
_db_pool: Optional[DatabasePool] = None
_indicator_library: Optional[IndicatorLibrary] = None
_snapshot_builder: Optional[MarketSnapshotBuilder] = None
_prompt_builder: Optional[PromptBuilder] = None

# NOTE: Symbol and timeframe validation moved to validation.py (Finding 26)
# Use validate_symbol_or_raise() and validate_timeframe_or_raise() from validation module


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

    # Finding 15: Add global exception handlers
    _register_exception_handlers(app)

    # Register routes
    register_health_routes(app)
    register_indicator_routes(app)
    register_snapshot_routes(app)
    register_debug_routes(app, debug_mode=security_config.debug_mode)

    return app


def _register_exception_handlers(app: FastAPI) -> None:
    """Register global exception handlers (Finding 15)."""

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle Pydantic validation errors with detailed messages."""
        errors = []
        for error in exc.errors():
            location = " -> ".join(str(loc) for loc in error["loc"])
            errors.append({
                "field": location,
                "message": error["msg"],
                "type": error["type"],
            })

        return JSONResponse(
            status_code=422,
            content={
                "detail": "Validation error",
                "errors": errors,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions consistently."""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "detail": exc.detail,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            headers=exc.headers,
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """
        Handle all unhandled exceptions.

        Logs the full traceback but returns a generic error to the client.
        This prevents leaking internal details to potential attackers.
        """
        # Log the full exception for debugging
        logger.error(
            f"Unhandled exception on {request.method} {request.url.path}: {exc}",
            exc_info=True,
        )

        # Don't expose internal details in production
        security_config = get_security_config()
        if security_config.debug_mode:
            detail = f"{type(exc).__name__}: {str(exc)}"
        else:
            detail = "An internal server error occurred"

        return JSONResponse(
            status_code=500,
            content={
                "detail": detail,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )


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
        # Validate inputs - uses centralized validation which normalizes symbol format
        symbol = validate_symbol_or_raise(symbol, strict=False)
        timeframe = validate_timeframe_or_raise(timeframe)

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

        except HTTPException:
            # Finding 22: Re-raise HTTPException to prevent it being caught below
            raise
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
        # Validate inputs - uses centralized validation which normalizes symbol format
        symbol = validate_symbol_or_raise(symbol, strict=False)

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

        except HTTPException:
            # Finding 22: Re-raise HTTPException to prevent it being caught below
            raise
        except Exception as e:
            logger.error(f"Failed to build snapshot: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error building snapshot")


def register_debug_routes(app: FastAPI, debug_mode: bool = False):
    """
    Register debug endpoints for testing.

    Finding 16: Debug endpoints require authentication unless in debug mode.
    In production (debug_mode=False), these endpoints require authentication.
    """

    # Core implementation shared by both protected and unprotected routes
    async def _get_debug_prompt_impl(agent: str, symbol: str):
        """Core implementation for debug prompt generation."""
        # Validate inputs - uses centralized validation which normalizes symbol format
        symbol = validate_symbol_or_raise(symbol, strict=False)

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
            # Finding 22: Handle ValueError before generic Exception
            raise HTTPException(status_code=400, detail=str(e))
        except HTTPException:
            # Finding 22: Re-raise HTTPException to prevent it being caught below
            raise
        except Exception as e:
            logger.error(f"Failed to generate debug prompt: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error generating prompt")

    async def _get_config_impl(show_debug_mode: bool):
        """Core implementation for config retrieval."""
        try:
            config_loader = get_config_loader()
            return {
                "indicators": list(config_loader.get_indicators_config().keys()),
                "snapshot": {
                    "timeframes": list(config_loader.get_snapshot_config().get('candle_lookback', {}).keys()),
                },
                "database": {
                    "connected": _db_pool.is_connected if _db_pool else False,
                },
                "debug_mode": show_debug_mode,
            }
        except Exception as e:
            logger.error(f"Failed to get config: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error retrieving config")

    if debug_mode:
        # Debug mode: no authentication required
        @app.get("/api/v1/debug/prompt/{agent}")
        async def get_debug_prompt(
            agent: str,
            symbol: str = Query(default="BTC/USDT"),
        ):
            """
            Generate and return a debug prompt for an agent.

            DEBUG MODE: No authentication required.

            Args:
                agent: Agent name (e.g., "technical_analysis")
                symbol: Trading pair for snapshot

            Returns:
                Assembled prompt components
            """
            return await _get_debug_prompt_impl(agent, symbol)

        @app.get("/api/v1/debug/config")
        async def get_config():
            """
            Return current configuration (sanitized).

            DEBUG MODE: No authentication required.
            """
            return await _get_config_impl(True)
    else:
        # Production mode: authentication required
        @app.get("/api/v1/debug/prompt/{agent}")
        async def get_debug_prompt_protected(
            agent: str,
            symbol: str = Query(default="BTC/USDT"),
            user: User = Depends(get_current_user),
        ):
            """
            Generate and return a debug prompt for an agent.

            PRODUCTION MODE: Requires authentication.

            Args:
                agent: Agent name (e.g., "technical_analysis")
                symbol: Trading pair for snapshot

            Returns:
                Assembled prompt components
            """
            return await _get_debug_prompt_impl(agent, symbol)

        @app.get("/api/v1/debug/config")
        async def get_config_protected(
            user: User = Depends(get_current_user),
        ):
            """
            Return current configuration (sanitized).

            PRODUCTION MODE: Requires authentication.
            """
            return await _get_config_impl(False)


# Singleton app instance
_app: Optional[FastAPI] = None

# Application state for routes that need shared components
_app_state: dict = {}


def register_paper_trading_routes(app: FastAPI, app_state: dict) -> None:
    """
    Register paper trading routes with the application.

    Phase 6: Paper trading API endpoints.
    Should be called after coordinator is initialized.

    Args:
        app: FastAPI application instance
        app_state: Dictionary with coordinator and other shared state
    """
    global _app_state
    _app_state = app_state

    try:
        from .routes_paper_trading import get_paper_trading_router

        router = get_paper_trading_router(app_state)
        if router:
            app.include_router(router)
            logger.info("Paper trading routes registered")
    except Exception as e:
        logger.warning(f"Failed to register paper trading routes: {e}")


def register_sentiment_routes(app: FastAPI, app_state: dict) -> None:
    """
    Register sentiment analysis routes with the application.

    Phase 7: Sentiment analysis API endpoints.
    Should be called after sentiment agent is initialized.

    Args:
        app: FastAPI application instance
        app_state: Dictionary with sentiment_agent and db_pool
    """
    global _app_state
    _app_state.update(app_state)

    try:
        from .routes_sentiment import create_sentiment_router

        sentiment_agent = app_state.get('sentiment_agent')
        db_pool = app_state.get('db_pool') or _db_pool

        router = create_sentiment_router(
            sentiment_agent=sentiment_agent,
            db_pool=db_pool,
        )
        if router:
            app.include_router(router)
            logger.info("Sentiment routes registered")
    except Exception as e:
        logger.warning(f"Failed to register sentiment routes: {e}")


def register_hodl_routes(app: FastAPI, app_state: dict) -> None:
    """
    Register hodl bag routes with the application.

    Phase 8: Hodl bag API endpoints.
    Should be called after hodl_manager is initialized.

    Args:
        app: FastAPI application instance
        app_state: Dictionary with hodl_manager
    """
    global _app_state
    _app_state.update(app_state)

    try:
        from .routes_hodl import get_hodl_router

        router = get_hodl_router(app_state)
        if router:
            app.include_router(router)
            logger.info("Hodl bag routes registered")
    except Exception as e:
        logger.warning(f"Failed to register hodl routes: {e}")


def get_app() -> FastAPI:
    """Get or create the singleton FastAPI app instance."""
    global _app
    if _app is None:
        _app = create_app()
    return _app


def get_app_state() -> dict:
    """Get current application state dictionary."""
    return _app_state


# For running directly with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("triplegain.src.api.app:get_app", host="0.0.0.0", port=8000, reload=True)
