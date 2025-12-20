"""
Hodl Bag API Routes - Endpoints for hodl bag management.

Phase 8 Endpoints:
- GET /api/v1/hodl/status - Current hodl bag states
- GET /api/v1/hodl/pending - Pending accumulations per asset
- GET /api/v1/hodl/thresholds - Per-asset purchase thresholds
- GET /api/v1/hodl/history - Transaction history
- GET /api/v1/hodl/metrics - Performance metrics
- POST /api/v1/hodl/force-accumulation - Force pending purchase
- GET /api/v1/hodl/snapshots - Historical snapshots

Security:
- All endpoints require authentication
- Force accumulation requires ADMIN role
"""

import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

try:
    from fastapi import APIRouter, HTTPException, Query, Body, Depends, Request
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    APIRouter = None
    BaseModel = object

from .security import (
    get_current_user,
    require_role,
    User,
    UserRole,
    log_security_event,
    SecurityEventType,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic Models
# =============================================================================

if FASTAPI_AVAILABLE:
    class ForceAccumulationRequest(BaseModel):
        """Request body for forcing hodl accumulation."""
        asset: str = Field(
            ...,
            pattern="^(BTC|XRP|USDT)$",
            description="Asset to force accumulate (BTC, XRP, or USDT)"
        )
        confirm: bool = Field(
            ...,
            description="Must be true to confirm force accumulation"
        )

    class HodlBagResponse(BaseModel):
        """Response model for hodl bag state."""
        asset: str
        balance: str
        cost_basis_usd: str
        current_value_usd: Optional[str]
        unrealized_pnl_usd: Optional[str]
        unrealized_pnl_pct: Optional[str]
        pending_usd: str
        first_accumulation: Optional[str]
        last_accumulation: Optional[str]
        accumulation_count: int


def create_hodl_routes(app_state: Dict[str, Any]) -> 'APIRouter':
    """
    Create hodl bag API routes.

    Args:
        app_state: Application state with hodl_manager, config, etc.

    Returns:
        FastAPI router with hodl bag endpoints
    """
    if not FASTAPI_AVAILABLE:
        raise RuntimeError("FastAPI not available")

    router = APIRouter(prefix="/api/v1/hodl", tags=["Hodl Bags"])

    def get_hodl_manager():
        """Get hodl bag manager from app state."""
        hodl_manager = app_state.get("hodl_manager")
        if not hodl_manager:
            raise HTTPException(status_code=503, detail="Hodl bag manager not initialized")
        return hodl_manager

    # =========================================================================
    # GET /api/v1/hodl/status - Hodl bag status
    # =========================================================================

    @router.get("/status")
    async def get_hodl_status(
        current_user: User = Depends(get_current_user),
    ) -> Dict[str, Any]:
        """
        Get current hodl bag status for all assets.

        Returns balances, cost basis, unrealized P&L, and pending amounts.
        """
        hodl_manager = get_hodl_manager()

        hodl_state = await hodl_manager.get_hodl_state()
        pending = await hodl_manager.get_pending()

        bags = {}
        for asset, state in hodl_state.items():
            state.pending_usd = pending.get(asset, Decimal(0))
            bags[asset] = state.to_dict()

        return {
            "bags": bags,
            "enabled": hodl_manager.enabled,
            "is_paper_mode": hodl_manager.is_paper_mode,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # =========================================================================
    # GET /api/v1/hodl/pending - Pending accumulations
    # =========================================================================

    @router.get("/pending")
    async def get_hodl_pending(
        current_user: User = Depends(get_current_user),
        asset: Optional[str] = Query(None, pattern="^(BTC|XRP|USDT)$"),
    ) -> Dict[str, Any]:
        """
        Get pending accumulation amounts per asset.

        Returns USD amounts waiting to reach threshold for purchase.
        """
        hodl_manager = get_hodl_manager()

        pending = await hodl_manager.get_pending(asset)
        thresholds = hodl_manager.thresholds

        result = {}
        for asset_name, amount in pending.items():
            threshold = thresholds.get(asset_name)
            progress_pct = (amount / threshold * 100) if threshold > 0 else Decimal(0)
            result[asset_name] = {
                "pending_usd": str(amount),
                "threshold_usd": str(threshold),
                "progress_pct": float(progress_pct),
                "ready_to_execute": amount >= threshold,
            }

        return {
            "pending": result,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # =========================================================================
    # GET /api/v1/hodl/thresholds - Purchase thresholds
    # =========================================================================

    @router.get("/thresholds")
    async def get_hodl_thresholds(
        current_user: User = Depends(get_current_user),
    ) -> Dict[str, Any]:
        """
        Get per-asset purchase thresholds.

        Returns minimum USD amount before each asset is purchased.
        """
        hodl_manager = get_hodl_manager()

        return {
            "thresholds": hodl_manager.thresholds.to_dict(),
            "allocation_pct": float(hodl_manager.allocation_pct),
            "split": {
                "usdt_pct": float(hodl_manager.usdt_pct),
                "xrp_pct": float(hodl_manager.xrp_pct),
                "btc_pct": float(hodl_manager.btc_pct),
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # =========================================================================
    # GET /api/v1/hodl/history - Transaction history
    # =========================================================================

    @router.get("/history")
    async def get_hodl_history(
        current_user: User = Depends(get_current_user),
        asset: Optional[str] = Query(None, pattern="^(BTC|XRP|USDT)$"),
        limit: int = Query(50, ge=1, le=500),
    ) -> Dict[str, Any]:
        """
        Get hodl bag transaction history.

        Returns list of accumulations, withdrawals, and adjustments.
        """
        hodl_manager = get_hodl_manager()

        transactions = await hodl_manager.get_transaction_history(asset=asset, limit=limit)

        return {
            "transactions": [t.to_dict() for t in transactions],
            "count": len(transactions),
            "asset_filter": asset,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # =========================================================================
    # GET /api/v1/hodl/metrics - Performance metrics
    # =========================================================================

    @router.get("/metrics")
    async def get_hodl_metrics(
        current_user: User = Depends(get_current_user),
    ) -> Dict[str, Any]:
        """
        Get hodl bag performance metrics.

        Returns total value, unrealized P&L, accumulation statistics.
        """
        hodl_manager = get_hodl_manager()

        metrics = await hodl_manager.calculate_metrics()
        stats = hodl_manager.get_stats()

        return {
            "metrics": metrics,
            "stats": stats,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # =========================================================================
    # POST /api/v1/hodl/force-accumulation - Force pending purchase
    # =========================================================================

    @router.post("/force-accumulation")
    async def force_hodl_accumulation(
        request: ForceAccumulationRequest,
        current_user: User = Depends(require_role(UserRole.ADMIN)),
    ) -> Dict[str, Any]:
        """
        Force immediate accumulation for an asset.

        Executes pending accumulation regardless of threshold.
        Requires ADMIN role.
        """
        if not request.confirm:
            raise HTTPException(
                status_code=400,
                detail="Force accumulation requires confirm=true"
            )

        hodl_manager = get_hodl_manager()

        # Get pending amount before execution
        pending = await hodl_manager.get_pending(request.asset)
        pending_amount = pending.get(request.asset, Decimal(0))

        if pending_amount <= 0:
            return {
                "success": False,
                "message": f"No pending {request.asset} to accumulate",
                "pending_usd": "0",
            }

        success = await hodl_manager.force_accumulation(request.asset)

        log_security_event(
            SecurityEventType.DATA_ACCESS,
            current_user.id,
            f"Force hodl accumulation: {request.asset} "
            f"(${float(pending_amount):.2f}), success={success}",
        )

        return {
            "success": success,
            "asset": request.asset,
            "executed_usd": str(pending_amount) if success else "0",
            "message": (
                f"Successfully accumulated ${float(pending_amount):.2f} worth of {request.asset}"
                if success
                else f"Failed to accumulate {request.asset}"
            ),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # =========================================================================
    # GET /api/v1/hodl/snapshots - Historical snapshots
    # =========================================================================

    @router.get("/snapshots")
    async def get_hodl_snapshots(
        current_user: User = Depends(get_current_user),
        asset: Optional[str] = Query(None, pattern="^(BTC|XRP|USDT)$"),
        days: int = Query(30, ge=1, le=365),
    ) -> Dict[str, Any]:
        """
        Get historical hodl bag snapshots.

        Returns daily snapshots for value tracking over time.
        """
        hodl_manager = get_hodl_manager()

        if not hodl_manager.db:
            return {
                "snapshots": [],
                "message": "Database not available for snapshots",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        try:
            if asset:
                query = """
                    SELECT timestamp, asset, balance, price_usd, value_usd,
                           cost_basis_usd, unrealized_pnl_usd, unrealized_pnl_pct
                    FROM hodl_bag_snapshots
                    WHERE asset = $1 AND timestamp > NOW() - $2::INTERVAL
                    ORDER BY timestamp DESC
                    LIMIT 1000
                """
                rows = await hodl_manager.db.fetch(query, asset, f"{days} days")
            else:
                query = """
                    SELECT timestamp, asset, balance, price_usd, value_usd,
                           cost_basis_usd, unrealized_pnl_usd, unrealized_pnl_pct
                    FROM hodl_bag_snapshots
                    WHERE timestamp > NOW() - $1::INTERVAL
                    ORDER BY timestamp DESC
                    LIMIT 1000
                """
                rows = await hodl_manager.db.fetch(query, f"{days} days")

            snapshots = [
                {
                    "timestamp": row["timestamp"].isoformat(),
                    "asset": row["asset"],
                    "balance": str(row["balance"]),
                    "price_usd": str(row["price_usd"]),
                    "value_usd": str(row["value_usd"]),
                    "cost_basis_usd": str(row["cost_basis_usd"]),
                    "unrealized_pnl_usd": str(row["unrealized_pnl_usd"]),
                    "unrealized_pnl_pct": str(row["unrealized_pnl_pct"]),
                }
                for row in rows
            ]

            return {
                "snapshots": snapshots,
                "count": len(snapshots),
                "asset_filter": asset,
                "days": days,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to get hodl snapshots: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve snapshots")

    return router


# =============================================================================
# Router factory for app initialization
# =============================================================================

def get_hodl_router(app_state: Dict[str, Any]) -> Optional['APIRouter']:
    """
    Get hodl bag router if FastAPI is available.

    Args:
        app_state: Application state dictionary with hodl_manager

    Returns:
        Router or None if FastAPI unavailable
    """
    if not FASTAPI_AVAILABLE:
        logger.warning("FastAPI not available - hodl routes disabled")
        return None

    return create_hodl_routes(app_state)


def register_hodl_routes(app, app_state: Dict[str, Any]) -> None:
    """
    Register hodl bag routes with the application.

    Phase 8: Hodl bag API endpoints.
    Should be called after hodl_manager is initialized.

    Args:
        app: FastAPI application instance
        app_state: Dictionary with hodl_manager
    """
    try:
        router = get_hodl_router(app_state)
        if router:
            app.include_router(router)
            logger.info("Hodl bag routes registered")
    except Exception as e:
        logger.warning(f"Failed to register hodl routes: {e}")
