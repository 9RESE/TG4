# Phase 4 Findings: API Layer Review

**Review Date**: 2025-12-19
**Reviewer**: Claude Opus 4.5
**Status**: Complete - DO NOT IMPLEMENT FIXES
**Files Reviewed**: 5 (~2,079 lines)

---

## Executive Summary

The API layer has **critical security vulnerabilities** that must be addressed before production deployment. While the security infrastructure (authentication, authorization, rate limiting) is well-designed and implemented, **it is not enforced on any route endpoints**. This renders all trading, position, and order management endpoints accessible without authentication.

| Priority | Count | Description |
|----------|-------|-------------|
| P0 (Critical) | 3 | Authentication bypass, authorization not enforced |
| P1 (High) | 8 | Missing security headers, no UUID validation, etc. |
| P2 (Medium) | 11 | Missing validation, audit logging, etc. |
| P3 (Low) | 5 | Code quality, hardcoded values |
| **Total** | **27** | |

---

## Critical Findings (P0)

### Finding 1: Authentication Not Enforced on Routes

**File**: `triplegain/src/api/routes_agents.py`, `triplegain/src/api/routes_orchestration.py`
**Priority**: P0
**Category**: Security - OWASP A01 Broken Access Control

#### Description
The `get_current_user` dependency is implemented in `security.py` but is NOT used by any endpoint. All trading, position management, order management, and coordinator control endpoints are accessible without authentication.

#### Current Code
```python
# routes_orchestration.py:102
@router.post("/coordinator/pause")
async def pause_coordinator():
    """Pause trading..."""
    # NO AUTHENTICATION CHECK - ANYONE CAN PAUSE TRADING
```

#### Recommended Fix
```python
from .security import get_current_user, require_role, UserRole, User

@router.post("/coordinator/pause")
async def pause_coordinator(user: User = Depends(get_current_user)):
    """Pause trading..."""
    # Now requires valid API key
```

#### Security Impact
- **Attack Scenario**: Any attacker with network access can pause/resume trading, close positions, cancel orders, and trigger rebalancing
- **Business Impact**: Complete loss of trading control, potential for significant financial losses

---

### Finding 2: Authorization Decorators Implemented But Not Used

**File**: `triplegain/src/api/security.py:359-385`, all route files
**Priority**: P0
**Category**: Security - OWASP A01 Broken Access Control

#### Description
The `require_role` decorator is fully implemented with role hierarchy (VIEWER < TRADER < ADMIN) but is not applied to any endpoint. Critical operations like `/risk/reset` with `admin_override=true` should require ADMIN role.

#### Current Code
```python
# security.py:359 - Decorator exists
def require_role(required_role: UserRole):
    """Decorator to require a specific role."""
    # Implementation exists but NEVER USED

# routes_agents.py:520 - Should use it but doesn't
@router.post("/risk/reset")
async def reset_risk_state(admin_override: bool = Query(default=False)):
    # NO ROLE CHECK - ANYONE CAN RESET WITH ADMIN OVERRIDE
```

#### Recommended Fix
```python
from .security import require_role, UserRole

@router.post("/risk/reset")
@require_role(UserRole.ADMIN)
async def reset_risk_state(
    admin_override: bool = Query(default=False),
    user: User = Depends(get_current_user)
):
    """Now requires ADMIN role."""
```

#### Security Impact
- **Attack Scenario**: Viewer-level user can reset risk state, potentially bypassing max drawdown circuit breakers
- **Business Impact**: Circumvention of all risk controls designed to protect capital

---

### Finding 3: Critical Operations Accessible Without Authentication

**File**: `triplegain/src/api/routes_orchestration.py`
**Priority**: P0
**Category**: Security - OWASP A01 Broken Access Control

#### Description
The following critical operations have no authentication:
- `POST /coordinator/pause` - Stops all trading
- `POST /coordinator/resume` - Resumes trading
- `POST /portfolio/rebalance` - Executes rebalancing trades
- `POST /positions/{id}/close` - Closes positions (real money impact)
- `POST /orders/{id}/cancel` - Cancels orders
- `POST /risk/reset` - Resets risk state

#### Security Impact
- **Attack Scenario**: Malicious actor on the same network can close all positions at unfavorable prices, causing significant losses
- **Business Impact**: Complete compromise of trading system control

---

## High Priority Findings (P1)

### Finding 4: Missing Security Headers

**File**: `triplegain/src/api/security.py`
**Priority**: P1
**Category**: Security - OWASP A05 Security Misconfiguration

#### Description
Critical security headers are not set:
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `Strict-Transport-Security` (for HTTPS)
- `Content-Security-Policy`

#### Recommended Fix
```python
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        # Add HSTS in production with HTTPS
        return response
```

---

### Finding 5: No UUID Validation for Position/Order IDs

**File**: `triplegain/src/api/routes_orchestration.py:323,439`
**Priority**: P1
**Category**: Security - Input Validation

#### Description
Position and order IDs accept any string without UUID format validation. This could allow injection of malformed IDs or exploitation of downstream systems.

#### Current Code
```python
@router.get("/positions/{position_id}")
async def get_position(position_id: str):  # Accepts ANY string
    position = await position_tracker.get_position(position_id)
```

#### Recommended Fix
```python
from uuid import UUID
from pydantic import UUID4

@router.get("/positions/{position_id}")
async def get_position(position_id: UUID4):  # Now validates UUID format
    position = await position_tracker.get_position(str(position_id))
```

---

### Finding 6: In-Memory API Key Storage

**File**: `triplegain/src/api/security.py:117`
**Priority**: P1
**Category**: Security - OWASP A04 Insecure Design

#### Description
API keys are stored in a plain dict `_api_keys: dict[str, User] = {}` which:
1. Loses all keys on server restart
2. Not shared across multiple instances
3. No persistence mechanism

#### Current Code
```python
# Simple in-memory API key store (production should use database)
_api_keys: dict[str, User] = {}
```

#### Recommended Fix
```python
# Use database-backed storage or Redis
class APIKeyStore:
    def __init__(self, db_pool: DatabasePool):
        self._pool = db_pool

    async def validate_key(self, api_key: str) -> Optional[User]:
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        row = await self._pool.fetchone(
            "SELECT * FROM api_keys WHERE key_hash = $1 AND revoked = FALSE",
            key_hash
        )
        return User.from_row(row) if row else None
```

---

### Finding 7: JWT Referenced But Not Implemented

**File**: `triplegain/src/api/security.py:46-48`
**Priority**: P1
**Category**: Security - OWASP A07 Auth Failures

#### Description
JWT configuration exists (`jwt_secret`, `jwt_algorithm`, `jwt_expiry_hours`) but no JWT validation is implemented. Only API key authentication is functional.

#### Current Code
```python
@dataclass
class SecurityConfig:
    jwt_secret: str = ""  # MUST be set via environment variable
    jwt_algorithm: str = "HS256"
    jwt_expiry_hours: int = 24
    # ... but JWT validation is never implemented
```

#### Recommended Fix
Either implement JWT validation or remove misleading configuration.

---

### Finding 8: Global State Instead of Dependency Injection

**File**: `triplegain/src/api/app.py:37-41`
**Priority**: P1
**Category**: Quality - Design Pattern

#### Description
Dependencies are stored as global module-level variables instead of using FastAPI's `Depends()` system. This makes testing harder and creates potential race conditions.

#### Current Code
```python
# Global instances (set during lifespan)
_db_pool: Optional[DatabasePool] = None
_indicator_library: Optional[IndicatorLibrary] = None
_snapshot_builder: Optional[MarketSnapshotBuilder] = None
_prompt_builder: Optional[PromptBuilder] = None
```

#### Recommended Fix
```python
async def get_db_pool() -> DatabasePool:
    if not app.state.db_pool:
        raise HTTPException(503, "Database not initialized")
    return app.state.db_pool

@app.get("/api/v1/indicators/{symbol}/{timeframe}")
async def get_indicators(
    symbol: str,
    db_pool: DatabasePool = Depends(get_db_pool),
    indicator_lib: IndicatorLibrary = Depends(get_indicator_library),
):
    ...
```

---

### Finding 9: No Confirmation for Destructive Operations

**File**: `triplegain/src/api/routes_orchestration.py`
**Priority**: P1
**Category**: Security - OWASP A04 Insecure Design

#### Description
Destructive operations (`close_position`, `cancel_order`, `rebalance`) have no confirmation mechanism or idempotency tokens to prevent accidental or replayed requests.

#### Recommended Fix
```python
class ClosePositionRequest(BaseModel):
    exit_price: float
    reason: str = "manual"
    confirmation_token: str = Field(..., description="One-time token from GET request")

@router.post("/positions/{position_id}/close")
async def close_position(
    position_id: str,
    request: ClosePositionRequest,
):
    if not verify_confirmation_token(position_id, request.confirmation_token):
        raise HTTPException(400, "Invalid or expired confirmation token")
```

---

### Finding 10: risk/reset Without Admin Check in Code Path

**File**: `triplegain/src/api/routes_agents.py:520-554`
**Priority**: P1
**Category**: Security - Logic Flaw

#### Description
The `/risk/reset` endpoint allows bypassing max drawdown halt with `admin_override=true` but only checks if `risk_engine.manual_reset()` returns True/False. The authentication/authorization check for admin is missing.

#### Current Code
```python
@router.post("/risk/reset")
async def reset_risk_state(admin_override: bool = Query(default=False)):
    # No check that user is actually an admin!
    success = risk_engine.manual_reset(admin_override=admin_override)
```

---

### Finding 11: No Rate Limiting on Risk/Reset

**File**: `triplegain/src/api/security.py:267-274`
**Priority**: P1
**Category**: Security - OWASP A04 Insecure Design

#### Description
The `_expensive_paths` for rate limiting includes `/api/v1/agents/` and `/api/v1/coordinator/` but the `/api/v1/risk/reset` endpoint uses default rate limiting, allowing potential abuse.

---

## Medium Priority Findings (P2)

### Finding 12: No Audit Logging

**File**: `triplegain/src/api/security.py`
**Priority**: P2
**Category**: Security - OWASP A09 Logging Failures

#### Description
No audit logging for:
- Authentication attempts (success/failure)
- Authorization failures
- Critical operations (position close, order cancel, risk reset)
- IP addresses of requests

#### Recommended Fix
```python
async def log_security_event(
    event_type: str,
    user_id: Optional[str],
    ip_address: str,
    details: dict
):
    logger.info(
        f"SECURITY: {event_type}",
        extra={
            "user_id": user_id,
            "ip": ip_address,
            "details": details,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    )
```

---

### Finding 13: MD5 Used for Rate Limit Client ID

**File**: `triplegain/src/api/security.py:281`
**Priority**: P2
**Category**: Security - OWASP A02 Cryptographic Failures

#### Description
MD5 is used to hash the authorization header for rate limiting. While not security-critical here, it's bad practice.

#### Current Code
```python
client_id = f"{client_ip}:{hashlib.md5(auth_header.encode()).hexdigest()[:8]}"
```

#### Recommended Fix
```python
client_id = f"{client_ip}:{hashlib.sha256(auth_header.encode()).hexdigest()[:16]}"
```

---

### Finding 14: No Cross-Field Validation for Trade Proposals

**File**: `triplegain/src/api/routes_agents.py:36-46`
**Priority**: P2
**Category**: Logic - Input Validation

#### Description
`TradeProposalRequest` validates individual fields but not cross-field constraints:
- stop_loss should be < entry_price for buy (long)
- stop_loss should be > entry_price for sell (short)
- take_profit should be > entry_price for buy
- take_profit should be < entry_price for sell

#### Recommended Fix
```python
class TradeProposalRequest(BaseModel):
    # ... fields ...

    @model_validator(mode='after')
    def validate_sl_tp(self) -> 'TradeProposalRequest':
        if self.side == "buy":
            if self.stop_loss and self.stop_loss >= self.entry_price:
                raise ValueError("Stop-loss must be below entry for long")
            if self.take_profit and self.take_profit <= self.entry_price:
                raise ValueError("Take-profit must be above entry for long")
        elif self.side == "sell":
            # Opposite validations
        return self
```

---

### Finding 15: No Global Exception Handler

**File**: `triplegain/src/api/app.py`
**Priority**: P2
**Category**: Quality - Error Handling

#### Description
No global exception handler to catch uncaught exceptions and return consistent error responses.

#### Recommended Fix
```python
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "request_id": request.state.request_id}
    )
```

---

### Finding 16: Debug Endpoints Exposed Without Authentication

**File**: `triplegain/src/api/app.py:324-389`
**Priority**: P2
**Category**: Security - Information Disclosure

#### Description
Debug endpoints `/api/v1/debug/prompt/{agent}` and `/api/v1/debug/config` expose:
- Prompt templates and system prompts (IP)
- Configuration structure
- Internal component status

#### Recommended Fix
Either disable in production or require authentication:
```python
@app.get("/api/v1/debug/config")
async def get_config(user: User = Depends(get_current_user)):
    if not os.environ.get("DEBUG_MODE"):
        raise HTTPException(404, "Not found")
```

---

### Finding 17: Missing Endpoint - GET /agents/outputs/{agent_name}

**File**: `triplegain/src/api/routes_agents.py`
**Priority**: P2
**Category**: Logic - Missing Feature

#### Description
The review checklist specifies `GET /api/v1/agents/outputs/{agent_name}` endpoint but it's not implemented.

---

### Finding 18: No Pydantic Response Models

**File**: `triplegain/src/api/validation.py`
**Priority**: P2
**Category**: Quality - API Contract

#### Description
No Pydantic response models defined for OpenAPI documentation. All responses are raw dicts, making API contract unclear.

#### Recommended Fix
```python
class AgentStatsResponse(BaseModel):
    timestamp: datetime
    agents: dict[str, AgentStats]

@router.get("/agents/stats", response_model=AgentStatsResponse)
async def get_agent_stats() -> AgentStatsResponse:
    ...
```

---

### Finding 19: No Symbol Validation in Orchestration Routes

**File**: `triplegain/src/api/routes_orchestration.py:269,413`
**Priority**: P2
**Category**: Logic - Input Validation

#### Description
The `symbol` query parameter in `/positions` and `/orders` endpoints is not validated using `validate_symbol_or_raise`.

#### Current Code
```python
@router.get("/positions")
async def get_positions(
    symbol: Optional[str] = Query(None),  # Not validated!
```

---

### Finding 20: task_name Not Validated

**File**: `triplegain/src/api/routes_orchestration.py:146`
**Priority**: P2
**Category**: Logic - Input Validation

#### Description
`/coordinator/task/{task_name}/run` accepts any task name without validation against known tasks.

---

### Finding 21: Inconsistent Symbol Handling in Test

**File**: `triplegain/tests/unit/api/test_app.py:251-254`
**Priority**: P2
**Category**: Logic - Inconsistency

#### Description
Test expects symbol returned as `BTC_USDT` but validation normalizes to `BTC/USDT`. Responses should use normalized format.

#### Current Code (test_app.py)
```python
assert data['symbol'] == 'BTC_USDT'  # But validation normalizes to BTC/USDT
```

---

### Finding 22: Exception Handling Order Issue

**File**: `triplegain/src/api/app.py:254-274`
**Priority**: P2
**Category**: Logic - Error Handling

#### Description
Test comment notes: "Currently returns 500 due to exception handling order in app.py. HTTPException is caught by generic Exception handler."

The `HTTPException` for "No data found" gets caught by the outer `except Exception` block and returns 500 instead of 404.

---

## Low Priority Findings (P3)

### Finding 23: SUPPORTED_SYMBOLS Hardcoded

**File**: `triplegain/src/api/validation.py:17-23`
**Priority**: P3
**Category**: Quality - Configuration

#### Description
Supported symbols are hardcoded instead of loaded from configuration.

---

### Finding 24: No Request/Response Logging

**File**: `triplegain/src/api/app.py`
**Priority**: P3
**Category**: Quality - Observability

#### Description
No structured request/response logging for debugging and observability.

---

### Finding 25: JWT Warning Doesn't Block Startup

**File**: `triplegain/src/api/security.py:433-437`
**Priority**: P3
**Category**: Security - Configuration

#### Description
Missing JWT secret only logs a warning but doesn't prevent server startup. In production, this should fail fast.

---

### Finding 26: Validation Functions Scattered

**File**: `triplegain/src/api/validation.py`, `app.py`, route files
**Priority**: P3
**Category**: Quality - Code Organization

#### Description
Validation logic is spread across multiple files:
- `validation.py` has symbol validation
- `app.py` has its own `validate_symbol` and `validate_timeframe`
- Route files have Pydantic models

Should be centralized in `validation.py`.

---

### Finding 27: No Pagination for List Endpoints

**File**: `triplegain/src/api/routes_orchestration.py:269,413`
**Priority**: P3
**Category**: Performance

#### Description
`/positions` and `/orders` endpoints return all items without pagination. Could cause performance issues with large datasets.

---

## OWASP Top 10 Summary

| Risk | Status | Notes |
|------|--------|-------|
| A01 Broken Access Control | **CRITICAL** | Auth not enforced on any route |
| A02 Cryptographic Failures | **MEDIUM** | MD5 used, JWT secret can be empty |
| A03 Injection | **LOW** | Uses parameterized queries |
| A04 Insecure Design | **HIGH** | In-memory keys, no confirmation |
| A05 Security Misconfiguration | **HIGH** | No security headers |
| A06 Vulnerable Components | N/A | Needs dependency scan |
| A07 Auth Failures | **CRITICAL** | Auth implemented but not enforced |
| A08 Data Integrity | **MEDIUM** | No request signing |
| A09 Logging Failures | **HIGH** | No audit logging |
| A10 SSRF | **LOW** | No external URL fetching |

---

## Review Checklist Completion

### FastAPI Application (`app.py`)
- [x] FastAPI instance configured correctly
- [x] CORS configured
- [x] Middleware applied
- [x] Routers included
- [x] Startup/shutdown events defined
- [ ] Global exception handler - **MISSING**
- [ ] Request/response logging - **MISSING**

### Agent Routes (`routes_agents.py`)
- [x] Endpoint coverage (mostly)
- [x] Input validation present
- [x] Response format consistent
- [x] Error responses appropriate
- [ ] Rate limiting per-endpoint - Uses middleware
- [ ] Authentication enforced - **MISSING**

### Orchestration Routes (`routes_orchestration.py`)
- [x] Endpoint coverage complete
- [ ] Critical operations require auth - **MISSING**
- [ ] UUID validation for IDs - **MISSING**
- [ ] Confirmation for destructive ops - **MISSING**

### Validation (`validation.py`)
- [x] Symbol validation present
- [ ] Response models - **MISSING**
- [ ] Cross-field validation - **MISSING**

### Security (`security.py`)
- [x] Rate limiting implemented
- [x] Authentication mechanism exists
- [x] Authorization mechanism exists
- [ ] Actually enforced on routes - **CRITICAL**
- [ ] Security headers - **MISSING**
- [ ] Audit logging - **MISSING**

---

## Test Coverage Notes

Tests exist for:
- Health endpoints (covered)
- Indicator endpoints (covered)
- Snapshot endpoints (covered)
- Debug endpoints (covered)
- Agent endpoints (covered)
- Orchestration endpoints (covered)
- Error handling (covered)

Missing tests:
- Authentication/authorization (auth not enforced so not tested)
- Rate limiting behavior
- Security header verification
- Concurrent request handling

---

## Recommended Fix Priority

1. **Immediate (P0)**: Enforce authentication on all non-public endpoints
2. **Week 1 (P1)**: Add security headers, UUID validation, audit logging
3. **Week 2 (P2)**: Cross-field validation, response models, exception handling
4. **Backlog (P3)**: Configuration improvements, code organization

---

*Phase 4 Review Complete - v1.0*
