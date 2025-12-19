"""
Agent API Routes - Endpoints for LLM agents and risk management.

Phase 2 Endpoints:
- GET /api/v1/agents/ta/{symbol} - Latest TA analysis
- GET /api/v1/agents/regime/{symbol} - Current regime
- POST /api/v1/agents/trading/{symbol}/run - Trigger trading decision
- POST /api/v1/risk/validate - Validate trade proposal
- GET /api/v1/risk/state - Current risk state
"""

import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional, Any

try:
    from fastapi import APIRouter, HTTPException, Query, Body
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    APIRouter = None
    BaseModel = object

logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic Models
# =============================================================================

if FASTAPI_AVAILABLE:
    class TradeProposalRequest(BaseModel):
        """Request body for trade validation."""
        symbol: str = Field(..., description="Trading symbol")
        side: str = Field(..., description="Trade side: buy or sell")
        size_usd: float = Field(..., gt=0, description="Trade size in USD")
        entry_price: float = Field(..., gt=0, description="Entry price")
        stop_loss: Optional[float] = Field(None, description="Stop-loss price")
        take_profit: Optional[float] = Field(None, description="Take-profit price")
        leverage: int = Field(1, ge=1, le=5, description="Leverage (1-5)")
        confidence: float = Field(0.5, ge=0, le=1, description="Signal confidence")
        regime: str = Field("ranging", description="Current market regime")

    class TradingDecisionRequest(BaseModel):
        """Request body for trading decision."""
        use_ta: bool = Field(True, description="Include TA agent output")
        use_regime: bool = Field(True, description="Include Regime agent output")
        force_refresh: bool = Field(False, description="Force fresh analysis")


# =============================================================================
# Router Factory
# =============================================================================

def create_agent_router(
    indicator_library,
    snapshot_builder,
    prompt_builder,
    db_pool,
    ta_agent=None,
    regime_agent=None,
    trading_agent=None,
    risk_engine=None,
) -> 'APIRouter':
    """
    Create agent router with dependencies.

    Args:
        indicator_library: IndicatorLibrary instance
        snapshot_builder: MarketSnapshotBuilder instance
        prompt_builder: PromptBuilder instance
        db_pool: Database pool
        ta_agent: TechnicalAnalysisAgent instance (optional)
        regime_agent: RegimeDetectionAgent instance (optional)
        trading_agent: TradingDecisionAgent instance (optional)
        risk_engine: RiskManagementEngine instance (optional)

    Returns:
        Configured APIRouter
    """
    if not FASTAPI_AVAILABLE:
        raise RuntimeError("FastAPI not available")

    router = APIRouter(prefix="/api/v1", tags=["agents"])

    # ---------------------------------------------------------------------
    # Technical Analysis Endpoints
    # ---------------------------------------------------------------------

    @router.get("/agents/ta/{symbol}")
    async def get_ta_analysis(
        symbol: str,
        max_age_seconds: int = Query(default=60, ge=0, le=300)
    ):
        """
        Get latest Technical Analysis for a symbol.

        Args:
            symbol: Trading symbol (e.g., "BTC/USDT")
            max_age_seconds: Maximum age of cached result

        Returns:
            TA output with trend, momentum, signals, and bias
        """
        if not ta_agent:
            raise HTTPException(status_code=503, detail="TA Agent not initialized")

        try:
            # Check for cached output
            cached = await ta_agent.get_latest_output(symbol, max_age_seconds)
            if cached:
                return {
                    "symbol": symbol,
                    "cached": True,
                    "output": cached.to_dict(),
                    "stats": ta_agent.get_stats(),
                }

            # Build fresh snapshot
            snapshot = await snapshot_builder.build_snapshot(symbol)

            # Process with TA agent
            output = await ta_agent.process(snapshot)

            return {
                "symbol": symbol,
                "cached": False,
                "output": output.to_dict(),
                "stats": ta_agent.get_stats(),
            }

        except Exception as e:
            logger.error(f"TA analysis failed for {symbol}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"TA analysis failed: {str(e)}")

    @router.post("/agents/ta/{symbol}/run")
    async def run_ta_analysis(
        symbol: str,
        force_refresh: bool = Query(default=True)
    ):
        """
        Trigger a fresh Technical Analysis for a symbol.

        Args:
            symbol: Trading symbol (e.g., "BTC/USDT")
            force_refresh: If True, always run fresh analysis

        Returns:
            TA output with trend, momentum, signals, and bias
        """
        if not ta_agent:
            raise HTTPException(status_code=503, detail="TA Agent not initialized")

        try:
            # Build fresh snapshot
            snapshot = await snapshot_builder.build_snapshot(symbol)

            # Process with TA agent (always fresh for POST)
            output = await ta_agent.process(snapshot)

            return {
                "symbol": symbol,
                "fresh": True,
                "output": output.to_dict(),
                "stats": ta_agent.get_stats(),
            }

        except Exception as e:
            logger.error(f"TA analysis failed for {symbol}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"TA analysis failed: {str(e)}")

    # ---------------------------------------------------------------------
    # Regime Detection Endpoints
    # ---------------------------------------------------------------------

    @router.get("/agents/regime/{symbol}")
    async def get_regime(
        symbol: str,
        max_age_seconds: int = Query(default=300, ge=0, le=600)
    ):
        """
        Get current market regime for a symbol.

        Args:
            symbol: Trading symbol (e.g., "BTC/USDT")
            max_age_seconds: Maximum age of cached result

        Returns:
            Regime classification with recommended parameters
        """
        if not regime_agent:
            raise HTTPException(status_code=503, detail="Regime Agent not initialized")

        try:
            # Check for cached output
            cached = await regime_agent.get_latest_output(symbol, max_age_seconds)
            if cached:
                return {
                    "symbol": symbol,
                    "cached": True,
                    "output": cached.to_dict(),
                    "parameters": cached.get_regime_parameters(),
                    "stats": regime_agent.get_stats(),
                }

            # Build fresh snapshot
            snapshot = await snapshot_builder.build_snapshot(symbol)

            # Get TA output if available
            ta_output = None
            if ta_agent:
                ta_output = await ta_agent.get_latest_output(symbol, 120)

            # Process with Regime agent
            output = await regime_agent.process(snapshot, ta_output=ta_output)

            return {
                "symbol": symbol,
                "cached": False,
                "output": output.to_dict(),
                "parameters": output.get_regime_parameters(),
                "stats": regime_agent.get_stats(),
            }

        except Exception as e:
            logger.error(f"Regime detection failed for {symbol}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Regime detection failed: {str(e)}")

    @router.post("/agents/regime/{symbol}/run")
    async def run_regime_detection(
        symbol: str,
        force_refresh: bool = Query(default=True)
    ):
        """
        Trigger a fresh Regime Detection for a symbol.

        Args:
            symbol: Trading symbol (e.g., "BTC/USDT")
            force_refresh: If True, always run fresh analysis

        Returns:
            Regime classification with recommended parameters
        """
        if not regime_agent:
            raise HTTPException(status_code=503, detail="Regime Agent not initialized")

        try:
            # Build fresh snapshot
            snapshot = await snapshot_builder.build_snapshot(symbol)

            # Get TA output if available
            ta_output = None
            if ta_agent:
                if force_refresh:
                    ta_output = await ta_agent.process(snapshot)
                else:
                    ta_output = await ta_agent.get_latest_output(symbol, 120)

            # Process with Regime agent (always fresh for POST)
            output = await regime_agent.process(snapshot, ta_output=ta_output)

            return {
                "symbol": symbol,
                "fresh": True,
                "output": output.to_dict(),
                "parameters": output.get_regime_parameters(),
                "stats": regime_agent.get_stats(),
            }

        except Exception as e:
            logger.error(f"Regime detection failed for {symbol}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Regime detection failed: {str(e)}")

    # ---------------------------------------------------------------------
    # Trading Decision Endpoints
    # ---------------------------------------------------------------------

    @router.post("/agents/trading/{symbol}/run")
    async def run_trading_decision(
        symbol: str,
        request: TradingDecisionRequest = Body(default_factory=TradingDecisionRequest)
    ):
        """
        Trigger a trading decision for a symbol.

        Runs all 6 models in parallel and calculates consensus.

        Args:
            symbol: Trading symbol (e.g., "BTC/USDT")
            request: Options for the trading decision

        Returns:
            Consensus trading decision with all model outputs
        """
        if not trading_agent:
            raise HTTPException(status_code=503, detail="Trading Decision Agent not initialized")

        try:
            # Build snapshot
            snapshot = await snapshot_builder.build_snapshot(symbol)

            # Get supporting agent outputs if requested
            ta_output = None
            regime_output = None

            if request.use_ta and ta_agent:
                if request.force_refresh:
                    ta_output = await ta_agent.process(snapshot)
                else:
                    ta_output = await ta_agent.get_latest_output(symbol, 120)
                    if not ta_output:
                        ta_output = await ta_agent.process(snapshot)

            if request.use_regime and regime_agent:
                if request.force_refresh:
                    regime_output = await regime_agent.process(snapshot, ta_output=ta_output)
                else:
                    regime_output = await regime_agent.get_latest_output(symbol, 300)
                    if not regime_output:
                        regime_output = await regime_agent.process(snapshot, ta_output=ta_output)

            # Run trading decision
            output = await trading_agent.process(
                snapshot,
                ta_output=ta_output,
                regime_output=regime_output,
            )

            # Format model decisions for response
            model_results = []
            for decision in output.model_decisions:
                model_results.append({
                    "model": decision.model_name,
                    "provider": decision.provider,
                    "action": decision.action,
                    "confidence": decision.confidence,
                    "latency_ms": decision.latency_ms,
                    "cost_usd": decision.cost_usd,
                    "error": decision.error,
                })

            return {
                "symbol": symbol,
                "consensus": {
                    "action": output.action,
                    "confidence": output.confidence,
                    "consensus_strength": output.consensus_strength,
                    "entry_price": output.entry_price,
                    "stop_loss": output.stop_loss,
                    "take_profit": output.take_profit,
                },
                "votes": output.votes,
                "agreeing_models": output.agreeing_models,
                "total_models": output.total_models,
                "model_results": model_results,
                "context": {
                    "regime": output.regime,
                    "ta_bias": output.ta_bias,
                },
                "cost": {
                    "total_usd": output.total_cost_usd,
                    "tokens_used": output.tokens_used,
                    "latency_ms": output.latency_ms,
                },
                "stats": trading_agent.get_stats(),
            }

        except Exception as e:
            logger.error(f"Trading decision failed for {symbol}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Trading decision failed: {str(e)}")

    # ---------------------------------------------------------------------
    # Risk Management Endpoints
    # ---------------------------------------------------------------------

    @router.post("/risk/validate")
    async def validate_trade(
        proposal: TradeProposalRequest
    ):
        """
        Validate a trade proposal against risk rules.

        Args:
            proposal: Trade proposal to validate

        Returns:
            Validation result with any modifications or rejections
        """
        if not risk_engine:
            raise HTTPException(status_code=503, detail="Risk Engine not initialized")

        try:
            from ..risk.rules_engine import TradeProposal

            # Convert request to TradeProposal
            trade_proposal = TradeProposal(
                symbol=proposal.symbol,
                side=proposal.side,
                size_usd=proposal.size_usd,
                entry_price=proposal.entry_price,
                stop_loss=proposal.stop_loss,
                take_profit=proposal.take_profit,
                leverage=proposal.leverage,
                confidence=proposal.confidence,
                regime=proposal.regime,
            )

            # Validate
            result = risk_engine.validate_trade(trade_proposal)

            response = {
                "status": result.status.value,
                "approved": result.is_approved(),
                "validation_time_ms": result.validation_time_ms,
            }

            if result.rejections:
                response["rejections"] = result.rejections

            if result.warnings:
                response["warnings"] = result.warnings

            if result.modifications:
                response["modifications"] = result.modifications

            if result.modified_proposal:
                response["modified_proposal"] = {
                    "size_usd": result.modified_proposal.size_usd,
                    "leverage": result.modified_proposal.leverage,
                }

            response["risk_metrics"] = {
                "risk_per_trade_pct": result.risk_per_trade_pct,
                "portfolio_exposure_pct": result.portfolio_exposure_pct,
            }

            return response

        except Exception as e:
            logger.error(f"Trade validation failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

    @router.get("/risk/state")
    async def get_risk_state():
        """
        Get current risk state.

        Returns:
            Current risk state including circuit breakers, cooldowns, P&L
        """
        if not risk_engine:
            raise HTTPException(status_code=503, detail="Risk Engine not initialized")

        try:
            state = risk_engine.get_state()

            return {
                "equity": {
                    "peak_usd": float(state.peak_equity),
                    "current_usd": float(state.current_equity),
                    "available_margin_usd": float(state.available_margin),
                },
                "pnl": {
                    "daily_pct": state.daily_pnl_pct,
                    "weekly_pct": state.weekly_pnl_pct,
                },
                "drawdown": {
                    "current_pct": state.current_drawdown_pct,
                    "max_pct": state.max_drawdown_pct,
                },
                "trades": {
                    "consecutive_losses": state.consecutive_losses,
                    "consecutive_wins": state.consecutive_wins,
                    "open_positions": state.open_positions,
                    "total_exposure_pct": state.total_exposure_pct,
                },
                "circuit_breakers": {
                    "trading_halted": state.trading_halted,
                    "halt_reason": state.halt_reason,
                    "halt_until": state.halt_until.isoformat() if state.halt_until else None,
                    "triggered": state.triggered_breakers,
                },
                "cooldowns": {
                    "active": state.in_cooldown,
                    "until": state.cooldown_until.isoformat() if state.cooldown_until else None,
                    "reason": state.cooldown_reason,
                },
                "limits": {
                    "max_leverage": risk_engine.get_max_allowed_leverage(
                        "ranging",
                        state.current_drawdown_pct,
                        state.consecutive_losses
                    ),
                },
            }

        except Exception as e:
            logger.error(f"Failed to get risk state: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to get risk state: {str(e)}")

    @router.post("/risk/reset")
    async def reset_risk_state(
        admin_override: bool = Query(default=False)
    ):
        """
        Manually reset risk state (requires admin override for max drawdown).

        Args:
            admin_override: Required for resetting max drawdown halt

        Returns:
            Reset result
        """
        if not risk_engine:
            raise HTTPException(status_code=503, detail="Risk Engine not initialized")

        try:
            success = risk_engine.manual_reset(admin_override=admin_override)

            if success:
                return {
                    "status": "reset_complete",
                    "state": await get_risk_state()
                }
            else:
                raise HTTPException(
                    status_code=403,
                    detail="Max drawdown halt requires admin_override=true"
                )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to reset risk state: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")

    # ---------------------------------------------------------------------
    # Agent Stats Endpoints
    # ---------------------------------------------------------------------

    @router.get("/agents/stats")
    async def get_agent_stats():
        """Get statistics for all agents."""
        stats = {}

        if ta_agent:
            stats["technical_analysis"] = ta_agent.get_stats()

        if regime_agent:
            stats["regime_detection"] = regime_agent.get_stats()

        if trading_agent:
            stats["trading_decision"] = trading_agent.get_stats()

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agents": stats,
        }

    return router
