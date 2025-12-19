# API Security Standards - TripleGain

**Version**: 1.0
**Last Updated**: 2025-12-19
**Status**: DRAFT (Security improvements pending)

## Overview

This document defines security standards for FastAPI endpoints in the TripleGain trading system. These standards are derived from the 2025-12-19 comprehensive API security review.

---

## Authentication & Authorization

### REQUIRED: All Endpoints Must Be Authenticated

**Status**: ❌ NOT IMPLEMENTED

All API endpoints MUST implement authentication except:
- `/health/live` (Kubernetes liveness)
- `/health/ready` (Kubernetes readiness)
- `/metrics` (Prometheus - should be internal only)

### Standard Pattern

```python
from fastapi import Security, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

security = HTTPBearer()

async def verify_token(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> dict:
    """Verify JWT token and return claims."""
    try:
        token = credentials.credentials
        payload = jwt.decode(
            token,
            settings.JWT_SECRET,
            algorithms=["HS256"]
        )
        return payload
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

# Apply to all protected endpoints
@router.post("/coordinator/pause")
async def pause_coordinator(
    token_data: dict = Depends(verify_token)
):
    # Endpoint implementation
    ...
```

### Role-Based Access Control (RBAC)

```python
from enum import Enum

class UserRole(str, Enum):
    ADMIN = "admin"      # Full access
    TRADER = "trader"    # Trading operations
    VIEWER = "viewer"    # Read-only

def require_role(required_role: UserRole):
    async def role_checker(token_data: dict = Depends(verify_token)):
        user_role = UserRole(token_data.get("role", "viewer"))

        role_hierarchy = {
            UserRole.ADMIN: 3,
            UserRole.TRADER: 2,
            UserRole.VIEWER: 1,
        }

        if role_hierarchy[user_role] < role_hierarchy[required_role]:
            raise HTTPException(
                status_code=403,
                detail=f"Requires {required_role} role"
            )

        return token_data
    return role_checker

# Usage
@router.post("/risk/reset")
async def reset_risk_state(
    admin_override: bool = Query(default=False),
    user: dict = Depends(require_role(UserRole.ADMIN))
):
    # Only admins can reset risk state
    ...
```

### Endpoint Security Classification

| Endpoint Pattern | Required Role | Rationale |
|-----------------|---------------|-----------|
| `/coordinator/pause` | ADMIN | Halts trading |
| `/coordinator/resume` | ADMIN | Resumes trading |
| `/portfolio/rebalance` | ADMIN | Forces trades |
| `/positions/{id}/close` | TRADER | Closes positions |
| `/orders/{id}/cancel` | TRADER | Cancels orders |
| `/risk/reset` | ADMIN | Bypasses safety |
| `/agents/*/run` | TRADER | Triggers analysis |
| `GET /positions` | VIEWER | Read-only |
| `GET /orders` | VIEWER | Read-only |
| `GET /health` | PUBLIC | Monitoring |

---

## Rate Limiting

### REQUIRED: Tiered Rate Limits

**Status**: ❌ NOT IMPLEMENTED

All endpoints MUST implement rate limiting to prevent:
- DoS attacks
- Excessive LLM API costs
- Database overload
- Resource exhaustion

### Standard Pattern

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Tiered limits based on endpoint cost
RATE_LIMITS = {
    "expensive": "5/minute",    # LLM calls (6 models)
    "moderate": "30/minute",    # Database writes
    "cheap": "100/minute",      # Database reads
    "health": "1000/minute",    # Health checks
}
```

### Endpoint Rate Limit Classification

```python
# EXPENSIVE (High LLM cost, ~$1 per call)
@router.post("/agents/trading/{symbol}/run")
@limiter.limit(RATE_LIMITS["expensive"])
async def run_trading_decision(...):
    # 6 model calls, expensive
    ...

# MODERATE (Database writes, position changes)
@router.post("/positions/{position_id}/close")
@limiter.limit(RATE_LIMITS["moderate"])
async def close_position(...):
    # Database write + position tracking
    ...

# CHEAP (Database reads, caching available)
@router.get("/positions")
@limiter.limit(RATE_LIMITS["cheap"])
async def get_positions(...):
    # Simple database read
    ...

# HEALTH (Kubernetes probes hit frequently)
@router.get("/health/ready")
@limiter.limit(RATE_LIMITS["health"])
async def readiness_check():
    # Must be high for K8s
    ...
```

### User-Based Rate Limiting

```python
def get_user_id(request: Request) -> str:
    """Extract user ID from JWT for per-user limits."""
    token_data = request.state.user  # Set by auth middleware
    return token_data.get("user_id", get_remote_address(request))

user_limiter = Limiter(key_func=get_user_id)

@router.post("/portfolio/rebalance")
@user_limiter.limit("1/hour")  # Per user, not per IP
async def force_rebalance(...):
    # Prevent user from spam rebalancing
    ...
```

---

## Input Validation

### REQUIRED: Validate All Inputs

**Status**: ⚠️ PARTIAL (inconsistent)

### Symbol Validation

```python
import re
from fastapi import HTTPException

SYMBOL_PATTERN = re.compile(r'^[A-Z0-9]{3,6}[/_][A-Z0-9]{3,6}$')

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
            detail=f"Invalid symbol format: '{symbol}'. Expected: BASE/QUOTE"
        )
    return symbol

# MANDATORY: Apply to ALL symbol parameters
@router.get("/agents/ta/{symbol}")
async def get_ta_analysis(symbol: str, ...):
    symbol = validate_symbol(symbol)  # ✅ REQUIRED
    ...
```

### Numeric Validation

```python
from pydantic import Field, condecimal
from decimal import Decimal

class TradeProposalRequest(BaseModel):
    """CORRECT: Use condecimal for financial fields."""
    symbol: str = Field(..., regex=r'^[A-Z0-9]{3,6}[/_][A-Z0-9]{3,6}$')
    side: str = Field(..., regex=r'^(buy|sell)$')

    # ✅ Use Decimal, not float
    size_usd: condecimal(gt=0, le=100000, decimal_places=2) = Field(
        ...,
        description="Trade size in USD (max $100k)"
    )

    entry_price: condecimal(gt=0, decimal_places=8) = Field(...)
    stop_loss: Optional[condecimal(gt=0, decimal_places=8)] = None
    take_profit: Optional[condecimal(gt=0, decimal_places=8)] = None

    leverage: int = Field(1, ge=1, le=5)
    confidence: float = Field(0.5, ge=0, le=1)

class ClosePositionRequest(BaseModel):
    """CORRECT: Native Decimal prevents precision loss."""
    exit_price: condecimal(gt=0, decimal_places=8) = Field(...)
    reason: str = Field("manual", max_length=100)
```

### Enum Validation

```python
from enum import Enum

class ExecutionStrategy(str, Enum):
    IMMEDIATE = "immediate"
    DCA_24H = "dca_24h"
    LIMIT_ORDERS = "limit_orders"

class ForceRebalanceRequest(BaseModel):
    # ✅ Use Enum, not raw string
    execution_strategy: ExecutionStrategy = Field(
        default=ExecutionStrategy.LIMIT_ORDERS
    )
```

### Path Parameter Validation

```python
from fastapi import Path
import uuid

@router.get("/positions/{position_id}")
async def get_position(
    # ✅ Validate UUID format in path
    position_id: str = Path(..., regex=r'^[a-f0-9-]{36}$')
):
    try:
        uuid.UUID(position_id)  # Additional validation
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid position ID")
    ...
```

---

## Error Handling

### REQUIRED: Never Expose Internal Details

**Status**: ⚠️ PARTIAL (some endpoints expose str(e))

### Standard Error Response

```python
from typing import Optional
from pydantic import BaseModel

class ErrorResponse(BaseModel):
    """Standardized error response."""
    detail: str
    error_code: Optional[str] = None
    request_id: Optional[str] = None

# INCORRECT: Exposes internal details
@router.get("/coordinator/status")
async def get_coordinator_status():
    try:
        return coordinator.get_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))  # ❌ BAD

# CORRECT: Generic message, detailed logging
@router.get("/coordinator/status")
async def get_coordinator_status(request: Request):
    try:
        return coordinator.get_status()
    except CoordinatorNotReadyError as e:
        # Specific exception = safe to expose
        logger.warning(f"Coordinator not ready: {e}")
        raise HTTPException(
            status_code=503,
            detail="Coordinator is initializing"
        )
    except Exception as e:
        # Generic exception = log details, hide from user
        logger.error(
            f"Failed to get coordinator status: {e}",
            exc_info=True,
            extra={"request_id": request.state.request_id}
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve coordinator status"
        )
```

### HTTP Status Code Standards

| Status | Use Case | Example |
|--------|----------|---------|
| 200 | Success | GET /positions |
| 201 | Created | POST /positions (if creating) |
| 204 | Success, no content | DELETE /orders/{id} |
| 400 | Bad request | Invalid symbol format |
| 401 | Unauthorized | Missing/invalid token |
| 403 | Forbidden | Insufficient role |
| 404 | Not found | Position ID doesn't exist |
| 409 | Conflict | Order already cancelled |
| 413 | Payload too large | Request > 1MB |
| 422 | Validation error | Pydantic validation failed |
| 429 | Rate limit exceeded | Too many requests |
| 500 | Internal error | Database connection failed |
| 503 | Service unavailable | Component not initialized |
| 504 | Gateway timeout | LLM call timeout |

---

## Request Size Limits

### REQUIRED: Limit Request Body Size

**Status**: ❌ NOT IMPLEMENTED

```python
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Limit request body size to prevent memory exhaustion."""

    def __init__(self, app, max_size: int = 1024 * 1024):  # 1MB default
        super().__init__(app)
        self.max_size = max_size

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")

        if content_length and int(content_length) > self.max_size:
            raise HTTPException(
                status_code=413,
                detail=f"Request body too large (max {self.max_size} bytes)"
            )

        return await call_next(request)

# Apply middleware
app.add_middleware(RequestSizeLimitMiddleware, max_size=1024 * 1024)
```

---

## CORS Configuration

### REQUIRED: Whitelist Origins

**Status**: ❌ NOT IMPLEMENTED

```python
from fastapi.middleware.cors import CORSMiddleware

# INCORRECT: Allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ❌ NEVER DO THIS
    allow_credentials=True,
)

# CORRECT: Whitelist specific origins
ALLOWED_ORIGINS = [
    "https://triplegain.example.com",  # Production UI
    "http://localhost:3000",           # Development UI
    "http://localhost:8080",           # Alternative dev port
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # ✅ Explicit whitelist
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
    max_age=600,  # Cache preflight for 10 minutes
)
```

---

## Timeout Configuration

### REQUIRED: Set Timeouts on Async Operations

**Status**: ❌ NOT IMPLEMENTED

```python
import asyncio
from fastapi import HTTPException

# Timeout configuration
TIMEOUTS = {
    "llm_call": 30.0,       # 30s for 6-model consensus
    "database_query": 5.0,  # 5s for DB operations
    "agent_process": 45.0,  # 45s for full agent pipeline
}

@router.post("/agents/trading/{symbol}/run")
async def run_trading_decision(...):
    try:
        output = await asyncio.wait_for(
            trading_agent.process(snapshot, ta_output, regime_output),
            timeout=TIMEOUTS["agent_process"]
        )
        return {...}

    except asyncio.TimeoutError:
        logger.error(f"Trading decision timed out for {symbol}")
        raise HTTPException(
            status_code=504,
            detail="Trading decision timed out after 45s"
        )
```

---

## Request Tracing

### REQUIRED: Request ID Middleware

**Status**: ❌ NOT IMPLEMENTED

```python
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from contextvars import ContextVar

request_id_var: ContextVar[str] = ContextVar("request_id", default="")

class RequestIDMiddleware(BaseHTTPMiddleware):
    """Add unique request ID to all requests."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        request_id_var.set(request_id)

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id

        return response

app.add_middleware(RequestIDMiddleware)

# Use in logging
logger.info(
    "Processing trading decision",
    extra={"request_id": request_id_var.get()}
)
```

---

## Sensitive Data Handling

### REQUIRED: Never Log Sensitive Data

**Status**: ✅ COMPLIANT (no sensitive data in logs currently)

### Prohibited in Logs/Responses
- API keys
- Database passwords
- JWT tokens
- Private keys
- User passwords
- Credit card numbers

### Safe Logging Pattern

```python
import re

def sanitize_for_logging(data: dict) -> dict:
    """Remove sensitive fields from log data."""
    sensitive_patterns = [
        r'password',
        r'token',
        r'secret',
        r'api[_-]?key',
        r'private[_-]?key',
    ]

    sanitized = data.copy()
    for key in data:
        if any(re.search(pattern, key, re.IGNORECASE) for pattern in sensitive_patterns):
            sanitized[key] = "***REDACTED***"

    return sanitized

logger.info(
    "User authentication",
    extra=sanitize_for_logging(request_data)
)
```

---

## Response Model Standards

### REQUIRED: Use Pydantic Response Models

**Status**: ❌ NOT IMPLEMENTED (90% of endpoints return dict)

```python
from typing import List
from pydantic import BaseModel

# INCORRECT: Untyped response
@router.get("/positions")
async def get_positions(...):
    positions = await position_tracker.get_open_positions()
    return {"positions": [p.to_dict() for p in positions]}  # ❌ No type safety

# CORRECT: Typed response model
class PositionResponse(BaseModel):
    id: str
    symbol: str
    side: str
    entry_price: Decimal
    current_price: Decimal
    unrealized_pnl: Decimal
    unrealized_pnl_pct: Decimal

    class Config:
        json_encoders = {Decimal: float}

class PositionsListResponse(BaseModel):
    count: int
    positions: List[PositionResponse]

@router.get("/positions", response_model=PositionsListResponse)
async def get_positions(...) -> PositionsListResponse:
    positions = await position_tracker.get_open_positions()
    return PositionsListResponse(
        count=len(positions),
        positions=[PositionResponse(**p.to_dict()) for p in positions]
    )
```

---

## Testing Requirements

### Security Test Coverage

All security features MUST have tests:

```python
# Authentication tests
def test_endpoint_requires_authentication(client):
    """Verify protected endpoints reject unauthenticated requests."""
    response = client.post("/coordinator/pause")
    assert response.status_code == 401

def test_endpoint_rejects_invalid_token(client):
    """Verify invalid tokens are rejected."""
    headers = {"Authorization": "Bearer invalid_token"}
    response = client.post("/coordinator/pause", headers=headers)
    assert response.status_code == 401

# Rate limiting tests
def test_rate_limit_enforced(client):
    """Verify rate limits are enforced."""
    # Make requests up to limit
    for i in range(5):
        response = client.post("/agents/trading/BTC_USDT/run")
        assert response.status_code in (200, 503)

    # Next request should be rate limited
    response = client.post("/agents/trading/BTC_USDT/run")
    assert response.status_code == 429

# Input validation tests
def test_rejects_sql_injection_in_symbol(client):
    """Verify SQL injection attempts are rejected."""
    response = client.get("/api/v1/indicators/'; DROP TABLE--/1h")
    assert response.status_code == 400

def test_rejects_oversized_request(client):
    """Verify oversized requests are rejected."""
    large_payload = {"data": "x" * (1024 * 1024 * 2)}  # 2MB
    response = client.post("/portfolio/rebalance", json=large_payload)
    assert response.status_code == 413

# Error handling tests
def test_error_does_not_leak_internals(client, mock_db_pool):
    """Verify errors don't expose internal details."""
    mock_db_pool.fetch_candles = AsyncMock(
        side_effect=Exception("Database connection failed: password=secret")
    )
    response = client.get("/api/v1/indicators/BTC_USDT/1h")

    if response.status_code == 500:
        detail = response.json().get("detail", "")
        assert "password" not in detail.lower()
        assert "secret" not in detail.lower()
```

---

## Deployment Checklist

Before deploying to production, verify:

- [ ] Authentication implemented on all protected endpoints
- [ ] Rate limiting configured and tested
- [ ] CORS whitelist configured (no wildcards)
- [ ] Request size limits enforced
- [ ] All inputs validated (symbols, prices, IDs)
- [ ] Response models defined for all endpoints
- [ ] Error messages don't expose internals
- [ ] Timeouts configured on async operations
- [ ] Request ID tracing implemented
- [ ] Security tests passing (auth, rate limit, injection)
- [ ] OpenAPI security schemes documented
- [ ] SSL/TLS certificate configured
- [ ] API keys rotated from defaults
- [ ] Database credentials use secrets management
- [ ] Logging excludes sensitive data
- [ ] Admin endpoints require elevated permissions

---

## Review History

| Date | Reviewer | Changes |
|------|----------|---------|
| 2025-12-19 | Code Review Agent | Initial security standards from API review |

---

## See Also

- [API Layer Review 2025-12-19](../reviews/api-layer-review-2025-12-19.md)
- [Code Standards](./code-standards.md)
- [Testing Standards](./testing-standards.md)
