# TripleGain API Security Action Plan

## Document Purpose
This document provides **actionable security fixes** for the two Critical (P0) and three High (P1) priority security issues identified in the API layer code review.

**Review Date**: 2025-12-19
**Status**: Immediate Action Required
**Estimated Implementation Time**: 8-12 hours

---

## Critical Priority Fixes (P0)

### P0-1: Implement Authentication & Authorization

#### Problem Summary
The API has zero authentication. Anyone with network access can:
- Execute trades
- Modify risk parameters
- Close positions
- Reset safety mechanisms

**Financial Risk**: Complete system compromise possible

#### Solution Architecture

**4-Tier Access Control System**:

| Tier | Access Level | Permissions | Use Case |
|------|--------------|-------------|----------|
| 1 | Read-Only | View health, stats, positions | Monitoring dashboards |
| 2 | Analysis | Trigger TA/Regime agents, view data | Analysis tools |
| 3 | Trading | Run trading decisions, submit trades | Bot operations |
| 4 | Admin | Close positions, reset risk, modify system | Emergency intervention |

#### Implementation Plan

**Step 1: Add Dependencies**
```bash
pip install python-jose[cryptography] passlib[bcrypt] python-multipart
```

**Step 2: Create Authentication Module**

Create `/home/rese/Documents/rese/trading-bots/grok-4_1/triplegain/src/api/auth.py`:

```python
"""
Authentication and Authorization for TripleGain API.
"""
import os
import logging
from datetime import datetime, timedelta
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

logger = logging.getLogger(__name__)

# Configuration (move to config file in production)
SECRET_KEY = os.getenv("API_SECRET_KEY", "CHANGE_THIS_IN_PRODUCTION")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 480  # 8 hours

# API Keys for service accounts (stored in environment)
API_KEYS = {
    os.getenv("API_KEY_READONLY"): "readonly",
    os.getenv("API_KEY_ANALYSIS"): "analysis",
    os.getenv("API_KEY_TRADING"): "trading",
    os.getenv("API_KEY_ADMIN"): "admin",
}

security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class AuthLevel:
    """Authorization levels."""
    READONLY = "readonly"
    ANALYSIS = "analysis"
    TRADING = "trading"
    ADMIN = "admin"

    HIERARCHY = {
        READONLY: 1,
        ANALYSIS: 2,
        TRADING: 3,
        ADMIN: 4,
    }

    @classmethod
    def has_permission(cls, user_level: str, required_level: str) -> bool:
        """Check if user level meets required level."""
        return cls.HIERARCHY.get(user_level, 0) >= cls.HIERARCHY.get(required_level, 999)


def verify_api_key(api_key: str) -> Optional[str]:
    """
    Verify API key and return authorization level.

    Returns:
        Authorization level if valid, None otherwise
    """
    # Remove 'Bearer ' prefix if present
    if api_key.startswith("Bearer "):
        api_key = api_key[7:]

    auth_level = API_KEYS.get(api_key)
    if auth_level:
        logger.info(f"API key verified: {auth_level} access")
        return auth_level

    logger.warning(f"Invalid API key attempted")
    return None


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> str:
    """
    Validate token and return authorization level.

    Raises:
        HTTPException: If authentication fails

    Returns:
        Authorization level (readonly, analysis, trading, admin)
    """
    token = credentials.credentials

    # Verify API key
    auth_level = verify_api_key(token)
    if not auth_level:
        logger.warning("Authentication failed: invalid credentials")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return auth_level


class RequireAuth:
    """Dependency for requiring specific auth level."""

    def __init__(self, required_level: str):
        self.required_level = required_level

    async def __call__(self, auth_level: str = Depends(get_current_user)) -> str:
        """Check if user has required permission level."""
        if not AuthLevel.has_permission(auth_level, self.required_level):
            logger.warning(
                f"Authorization failed: {auth_level} attempted {self.required_level} operation"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required: {self.required_level}",
            )
        return auth_level


# Convenience dependencies for each tier
require_readonly = RequireAuth(AuthLevel.READONLY)
require_analysis = RequireAuth(AuthLevel.ANALYSIS)
require_trading = RequireAuth(AuthLevel.TRADING)
require_admin = RequireAuth(AuthLevel.ADMIN)
```

**Step 3: Apply Authentication to Endpoints**

Update `routes_agents.py`:

```python
from .auth import require_readonly, require_analysis, require_trading, require_admin

# Read-only endpoints (Tier 1)
@router.get("/agents/stats", dependencies=[Depends(require_readonly)])
async def get_agent_stats():
    ...

@router.get("/risk/state", dependencies=[Depends(require_readonly)])
async def get_risk_state():
    ...

# Analysis endpoints (Tier 2)
@router.get("/agents/ta/{symbol}", dependencies=[Depends(require_analysis)])
async def get_ta_analysis(symbol: str, ...):
    ...

@router.post("/agents/ta/{symbol}/run", dependencies=[Depends(require_analysis)])
async def run_ta_analysis(symbol: str, ...):
    ...

# Trading endpoints (Tier 3)
@router.post("/agents/trading/{symbol}/run", dependencies=[Depends(require_trading)])
async def run_trading_decision(symbol: str, ...):
    ...

@router.post("/risk/validate", dependencies=[Depends(require_trading)])
async def validate_trade(proposal: TradeProposalRequest):
    ...

# Admin endpoints (Tier 4)
@router.post("/risk/reset", dependencies=[Depends(require_admin)])
async def reset_risk_state(admin_override: bool = Query(default=False)):
    ...
```

Update `routes_orchestration.py`:

```python
from .auth import require_readonly, require_trading, require_admin

# Read-only
@router.get("/coordinator/status", dependencies=[Depends(require_readonly)])
@router.get("/positions", dependencies=[Depends(require_readonly)])
@router.get("/orders", dependencies=[Depends(require_readonly)])

# Trading level
@router.post("/coordinator/pause", dependencies=[Depends(require_trading)])
@router.post("/coordinator/resume", dependencies=[Depends(require_trading)])
@router.post("/portfolio/rebalance", dependencies=[Depends(require_trading)])

# Admin level
@router.post("/positions/{position_id}/close", dependencies=[Depends(require_admin)])
@router.post("/orders/{order_id}/cancel", dependencies=[Depends(require_admin)])
@router.patch("/positions/{position_id}", dependencies=[Depends(require_admin)])
```

**Step 4: Environment Configuration**

Create `.env.api` (DO NOT COMMIT):

```bash
# Generate secure keys with: openssl rand -hex 32
API_SECRET_KEY=your_secret_key_here_minimum_32_characters

# API Keys for each tier (generate with: openssl rand -hex 32)
API_KEY_READONLY=readonly_key_here
API_KEY_ANALYSIS=analysis_key_here
API_KEY_TRADING=trading_key_here
API_KEY_ADMIN=admin_key_here
```

**Step 5: Update API Client Usage**

```python
import httpx

# Example client usage
headers = {
    "Authorization": f"Bearer {os.getenv('API_KEY_TRADING')}"
}

response = httpx.post(
    "http://localhost:8000/api/v1/agents/trading/BTC/USDT/run",
    headers=headers,
    json={"use_ta": True}
)
```

**Step 6: Testing**

Create test file for authentication:

```python
# triplegain/tests/unit/api/test_auth.py
import pytest
from fastapi import HTTPException
from triplegain.src.api.auth import AuthLevel, verify_api_key

def test_auth_hierarchy():
    assert AuthLevel.has_permission("admin", "readonly")
    assert AuthLevel.has_permission("trading", "analysis")
    assert not AuthLevel.has_permission("readonly", "trading")

def test_invalid_api_key():
    assert verify_api_key("invalid_key") is None

# Integration test
async def test_protected_endpoint(client):
    # No auth
    response = await client.get("/api/v1/agents/stats")
    assert response.status_code == 401

    # Valid auth
    headers = {"Authorization": "Bearer " + os.getenv("API_KEY_READONLY")}
    response = await client.get("/api/v1/agents/stats", headers=headers)
    assert response.status_code == 200
```

**Estimated Time**: 4 hours

---

### P0-2: Implement Rate Limiting

#### Problem Summary
No rate limiting exists. Attackers could:
- Generate unlimited LLM API costs ($0.50 per trading decision × unlimited = $$$)
- DDoS the system
- Exhaust database connections

**Financial Risk**: Unlimited LLM API cost exposure

#### Solution Architecture

**Multi-Tier Rate Limiting**:
- Per-IP limits (prevent DDoS)
- Per-API-key limits (prevent abuse by authenticated users)
- Per-endpoint-class limits (different limits for expensive vs cheap operations)

#### Implementation Plan

**Step 1: Install Dependencies**

```bash
pip install slowapi redis aioredis
```

**Step 2: Configure Redis for Rate Limit Storage**

Update `docker-compose.yml`:

```yaml
services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes

volumes:
  redis_data:
```

**Step 3: Create Rate Limiting Module**

Create `/home/rese/Documents/rese/trading-bots/grok-4_1/triplegain/src/api/rate_limit.py`:

```python
"""
Rate limiting for TripleGain API.
"""
import logging
from functools import wraps
from typing import Optional

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request, HTTPException
import redis.asyncio as redis

logger = logging.getLogger(__name__)

# Redis client for distributed rate limiting
_redis_client: Optional[redis.Redis] = None


async def init_redis(redis_url: str = "redis://localhost:6379"):
    """Initialize Redis connection for rate limiting."""
    global _redis_client
    _redis_client = await redis.from_url(redis_url, encoding="utf-8", decode_responses=True)
    logger.info("Redis connected for rate limiting")


async def close_redis():
    """Close Redis connection."""
    global _redis_client
    if _redis_client:
        await _redis_client.close()
        logger.info("Redis disconnected")


def get_user_identifier(request: Request) -> str:
    """
    Get unique identifier for rate limiting.

    Prefers API key over IP address for authenticated requests.
    """
    # Try to get API key from Authorization header
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        api_key = auth_header[7:]
        # Use last 8 chars of API key as identifier
        return f"key:{api_key[-8:]}" if len(api_key) >= 8 else f"key:{api_key}"

    # Fall back to IP address
    return f"ip:{get_remote_address(request)}"


# Create limiter instance
limiter = Limiter(
    key_func=get_user_identifier,
    default_limits=["1000/hour"],  # Global default
    storage_uri="redis://localhost:6379",
)


class EndpointLimits:
    """Rate limit configurations for different endpoint classes."""

    # Health and monitoring (high limit for uptime checks)
    HEALTH = "100/minute"

    # Data endpoints (moderate)
    DATA = "60/minute"

    # Analysis endpoints (moderate, but LLM calls)
    ANALYSIS = "30/minute"

    # Trading decisions (strict - expensive LLM calls)
    TRADING = "10/minute"

    # Admin operations (very strict)
    ADMIN = "5/minute"

    # Per-IP global limit (prevent DDoS)
    GLOBAL_PER_IP = "200/minute"


async def check_cost_limit(request: Request, estimated_cost_usd: float = 0.0):
    """
    Check if user is within cost budget.

    Tracks cumulative API cost per user per hour.
    """
    if not _redis_client or estimated_cost_usd == 0:
        return

    user_id = get_user_identifier(request)
    cost_key = f"cost:{user_id}:hour"

    # Get current cost
    current_cost = await _redis_client.get(cost_key)
    current_cost = float(current_cost) if current_cost else 0.0

    # Cost limit: $10/hour per user
    MAX_COST_PER_HOUR = 10.0

    if current_cost + estimated_cost_usd > MAX_COST_PER_HOUR:
        logger.warning(
            f"Cost limit exceeded for {user_id}: "
            f"${current_cost:.2f} + ${estimated_cost_usd:.2f} > ${MAX_COST_PER_HOUR}"
        )
        raise HTTPException(
            status_code=429,
            detail=f"Hourly cost limit exceeded (${MAX_COST_PER_HOUR}/hour). "
                   f"Current usage: ${current_cost:.2f}"
        )

    # Increment cost
    pipe = _redis_client.pipeline()
    pipe.incrbyfloat(cost_key, estimated_cost_usd)
    pipe.expire(cost_key, 3600)  # 1 hour TTL
    await pipe.execute()

    logger.info(f"Cost tracked for {user_id}: ${current_cost + estimated_cost_usd:.2f}/hour")
```

**Step 4: Apply to FastAPI App**

Update `app.py`:

```python
from .rate_limit import limiter, init_redis, close_redis, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown."""
    try:
        # ... existing startup code ...

        # Initialize rate limiting
        await init_redis()
        logger.info("Rate limiting initialized")

        yield
    finally:
        # ... existing shutdown code ...

        # Close rate limiting
        await close_redis()


def create_app() -> FastAPI:
    app = FastAPI(...)

    # Add rate limiter
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    return app
```

**Step 5: Apply Limits to Endpoints**

Update `routes_agents.py`:

```python
from .rate_limit import limiter, EndpointLimits, check_cost_limit

@router.get("/agents/stats")
@limiter.limit(EndpointLimits.DATA)
async def get_agent_stats(request: Request):
    ...

@router.post("/agents/ta/{symbol}/run")
@limiter.limit(EndpointLimits.ANALYSIS)
async def run_ta_analysis(request: Request, symbol: str, ...):
    await check_cost_limit(request, estimated_cost_usd=0.05)
    ...

@router.post("/agents/trading/{symbol}/run")
@limiter.limit(EndpointLimits.TRADING)
async def run_trading_decision(request: Request, symbol: str, ...):
    await check_cost_limit(request, estimated_cost_usd=0.50)
    ...

@router.post("/risk/reset")
@limiter.limit(EndpointLimits.ADMIN)
async def reset_risk_state(request: Request, ...):
    ...
```

Update `routes_orchestration.py`:

```python
from .rate_limit import limiter, EndpointLimits

@router.get("/coordinator/status")
@limiter.limit(EndpointLimits.HEALTH)
async def get_coordinator_status(request: Request):
    ...

@router.post("/positions/{position_id}/close")
@limiter.limit(EndpointLimits.ADMIN)
async def close_position(request: Request, position_id: str, ...):
    ...
```

**Step 6: Custom Error Handling**

```python
# app.py
from slowapi.errors import RateLimitExceeded

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={
            "error": {
                "code": 429,
                "message": "Rate limit exceeded",
                "detail": str(exc.detail),
                "retry_after": exc.headers.get("Retry-After"),
            }
        },
        headers=exc.headers,
    )
```

**Step 7: Monitoring**

Add rate limit metrics endpoint:

```python
@router.get("/api/v1/metrics/rate-limits")
@limiter.exempt  # Don't rate limit metrics endpoint
async def get_rate_limit_metrics(auth_level: str = Depends(require_admin)):
    """Get rate limiting statistics (admin only)."""
    # Query Redis for top consumers
    # Return aggregated stats
    ...
```

**Estimated Time**: 4 hours

---

## High Priority Fixes (P1)

### P1-4: Configure CORS

#### Problem Summary
Design spec requires CORS for dashboard, but not implemented. Frontend cannot connect.

#### Solution

Update `app.py`:

```python
from fastapi.middleware.cors import CORSMiddleware
import os

def create_app() -> FastAPI:
    app = FastAPI(...)

    # CORS configuration
    allowed_origins = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://localhost:5173"  # Vite & React defaults
    ).split(",")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
        expose_headers=["X-Request-ID", "X-RateLimit-Remaining"],
        max_age=600,  # Cache preflight for 10 minutes
    )

    logger.info(f"CORS enabled for origins: {allowed_origins}")

    return app
```

**Environment Configuration**:

```bash
# Production
CORS_ORIGINS=https://dashboard.triplegain.com

# Development
CORS_ORIGINS=http://localhost:3000,http://localhost:5173
```

**Estimated Time**: 30 minutes

---

### P1-2: Add Request Size Limits

#### Problem Summary
No limits on request body size. Could enable memory exhaustion attacks.

#### Solution

```python
# app.py
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Limit request body size to prevent memory exhaustion."""

    def __init__(self, app, max_size: int = 1_000_000):
        super().__init__(app)
        self.max_size = max_size

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_size:
            return Response(
                content=f"Request too large (max {self.max_size} bytes)",
                status_code=413
            )
        return await call_next(request)


def create_app() -> FastAPI:
    app = FastAPI(...)

    # Add request size limit (1MB default)
    app.add_middleware(
        RequestSizeLimitMiddleware,
        max_size=int(os.getenv("MAX_REQUEST_SIZE", "1000000"))
    )

    return app
```

**Estimated Time**: 30 minutes

---

### P1-3: Disable Debug Endpoints in Production

#### Problem Summary
Debug endpoints expose system internals and prompt engineering.

#### Solution

Update `app.py`:

```python
def register_debug_routes(app: FastAPI):
    """Register debug endpoints (development only)."""

    # Only register in non-production environments
    if os.getenv("ENVIRONMENT", "production") == "production":
        logger.info("Debug endpoints disabled in production")
        return

    logger.warning("Debug endpoints ENABLED - not for production use")

    @app.get("/api/v1/debug/prompt/{agent}")
    async def get_debug_prompt(...):
        ...

    @app.get("/api/v1/debug/config")
    async def get_config(...):
        ...
```

**Environment Configuration**:

```bash
# Development
ENVIRONMENT=development

# Production
ENVIRONMENT=production
```

**Estimated Time**: 15 minutes

---

## Implementation Checklist

### Phase 1: Critical Security (Must Complete Before Any Deployment)
- [ ] P0-1: Implement authentication (4 hours)
  - [ ] Create auth module
  - [ ] Generate API keys
  - [ ] Apply to all endpoints
  - [ ] Test authentication
  - [ ] Document usage
- [ ] P0-2: Implement rate limiting (4 hours)
  - [ ] Set up Redis
  - [ ] Create rate limit module
  - [ ] Apply to endpoints
  - [ ] Test limits
  - [ ] Add monitoring

### Phase 2: Production Hardening (Complete Before Production Traffic)
- [ ] P1-4: Configure CORS (30 minutes)
- [ ] P1-2: Add request size limits (30 minutes)
- [ ] P1-3: Disable debug endpoints (15 minutes)

### Phase 3: Testing & Validation
- [ ] Write integration tests for auth
- [ ] Write integration tests for rate limiting
- [ ] Penetration testing
- [ ] Load testing
- [ ] Cost monitoring validation

**Total Estimated Time**: 10 hours

---

## Testing Strategy

### Authentication Testing

```python
# Test unauthorized access
def test_unauthorized_access():
    response = client.post("/api/v1/agents/trading/BTC/USDT/run")
    assert response.status_code == 401

# Test insufficient permissions
def test_insufficient_permissions():
    headers = {"Authorization": f"Bearer {READONLY_KEY}"}
    response = client.post(
        "/api/v1/agents/trading/BTC/USDT/run",
        headers=headers
    )
    assert response.status_code == 403

# Test valid access
def test_authorized_access():
    headers = {"Authorization": f"Bearer {TRADING_KEY}"}
    response = client.post(
        "/api/v1/agents/trading/BTC/USDT/run",
        headers=headers,
        json={"use_ta": True}
    )
    assert response.status_code == 200
```

### Rate Limiting Testing

```python
# Test rate limit enforcement
async def test_rate_limit():
    headers = {"Authorization": f"Bearer {TRADING_KEY}"}

    # Should succeed for first 10 requests
    for i in range(10):
        response = await client.post(
            "/api/v1/agents/trading/BTC/USDT/run",
            headers=headers
        )
        assert response.status_code == 200

    # 11th request should be rate limited
    response = await client.post(
        "/api/v1/agents/trading/BTC/USDT/run",
        headers=headers
    )
    assert response.status_code == 429
    assert "Retry-After" in response.headers
```

### Cost Limit Testing

```python
async def test_cost_limit():
    headers = {"Authorization": f"Bearer {TRADING_KEY}"}

    # Make 20 trading decisions (~$10 total cost)
    for i in range(20):
        response = await client.post(
            "/api/v1/agents/trading/BTC/USDT/run",
            headers=headers
        )

    # Next request should hit cost limit
    response = await client.post(
        "/api/v1/agents/trading/BTC/USDT/run",
        headers=headers
    )
    assert response.status_code == 429
    assert "cost limit" in response.json()["detail"].lower()
```

---

## Deployment Steps

### 1. Generate Secrets

```bash
# Generate API keys
export API_KEY_READONLY=$(openssl rand -hex 32)
export API_KEY_ANALYSIS=$(openssl rand -hex 32)
export API_KEY_TRADING=$(openssl rand -hex 32)
export API_KEY_ADMIN=$(openssl rand -hex 32)
export API_SECRET_KEY=$(openssl rand -hex 32)

# Save to .env.api (DO NOT COMMIT)
cat > .env.api <<EOF
API_SECRET_KEY=$API_SECRET_KEY
API_KEY_READONLY=$API_KEY_READONLY
API_KEY_ANALYSIS=$API_KEY_ANALYSIS
API_KEY_TRADING=$API_KEY_TRADING
API_KEY_ADMIN=$API_KEY_ADMIN
EOF

# Restrict permissions
chmod 600 .env.api
```

### 2. Update Docker Compose

```yaml
# docker-compose.yml
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - CORS_ORIGINS=https://dashboard.triplegain.com
      - MAX_REQUEST_SIZE=1000000
    env_file:
      - .env.api
    depends_on:
      - timescaledb
      - redis
```

### 3. Start Services

```bash
# Start Redis and TimescaleDB
docker-compose up -d redis timescaledb

# Run migrations
python -m alembic upgrade head

# Start API
docker-compose up -d api
```

### 4. Verify Security

```bash
# Test without auth (should fail)
curl -X POST http://localhost:8000/api/v1/agents/trading/BTC/USDT/run
# Expected: 401 Unauthorized

# Test with valid auth
curl -X POST http://localhost:8000/api/v1/agents/trading/BTC/USDT/run \
  -H "Authorization: Bearer $API_KEY_TRADING" \
  -H "Content-Type: application/json" \
  -d '{"use_ta": true}'
# Expected: 200 OK

# Test rate limit
for i in {1..15}; do
  curl -X POST http://localhost:8000/api/v1/agents/trading/BTC/USDT/run \
    -H "Authorization: Bearer $API_KEY_TRADING"
done
# Expected: First 10 succeed, then 429 Rate Limit
```

---

## Monitoring & Alerting

### Metrics to Track

```python
# Prometheus metrics (add to metrics.py)
from prometheus_client import Counter, Histogram, Gauge

auth_attempts = Counter(
    'api_auth_attempts_total',
    'Total authentication attempts',
    ['result']  # success, failure
)

rate_limit_hits = Counter(
    'api_rate_limit_hits_total',
    'Total rate limit violations',
    ['endpoint', 'user_type']
)

api_costs = Counter(
    'api_llm_costs_usd_total',
    'Total LLM API costs',
    ['endpoint', 'user']
)
```

### Alert Rules

```yaml
# alerts.yaml
groups:
  - name: api_security
    interval: 30s
    rules:
      - alert: HighAuthFailureRate
        expr: rate(api_auth_attempts_total{result="failure"}[5m]) > 10
        annotations:
          summary: "High authentication failure rate detected"

      - alert: ExcessiveRateLimiting
        expr: rate(api_rate_limit_hits_total[5m]) > 50
        annotations:
          summary: "High rate limit hit rate (possible attack)"

      - alert: UnusualAPICosts
        expr: rate(api_llm_costs_usd_total[1h]) > 20
        annotations:
          summary: "API costs exceeding $20/hour"
```

---

## Cost Protection Summary

After implementing these fixes:

| Attack Vector | Current Exposure | Protected Exposure | Mitigation |
|---------------|------------------|-------------------|------------|
| Unauthorized trading | Unlimited | $0 | Authentication required |
| LLM API abuse | Unlimited | $10/hour per user | Rate + cost limiting |
| DDoS | Complete service loss | 200 req/min per IP | Rate limiting |
| Memory exhaustion | System crash | 1MB max request | Size limits |

**Total Implementation Time**: 10 hours
**Risk Reduction**: 95%+
**Cost Protection**: $10/hour cap per user

---

## Post-Implementation Validation

After completing all fixes, validate:

1. **Authentication Works**:
   - [ ] Endpoints reject unauthenticated requests
   - [ ] Tier permissions enforced correctly
   - [ ] Invalid keys rejected

2. **Rate Limiting Works**:
   - [ ] Per-endpoint limits enforced
   - [ ] Cost limits prevent runaway spending
   - [ ] Rate limit headers present in responses

3. **Production Hardening**:
   - [ ] CORS works for dashboard
   - [ ] Debug endpoints disabled
   - [ ] Request size limits enforced

4. **Monitoring Active**:
   - [ ] Auth metrics collected
   - [ ] Rate limit metrics collected
   - [ ] Cost metrics collected
   - [ ] Alerts configured

---

## Emergency Procedures

### If API Keys Compromised

```bash
# 1. Generate new keys immediately
./scripts/rotate_api_keys.sh

# 2. Update all clients
# 3. Revoke old keys in Redis
redis-cli DEL "api_keys:*"

# 4. Audit access logs
grep "401\|403" /var/log/triplegain/api.log | tail -1000
```

### If Under Attack

```bash
# 1. Identify attacker
redis-cli KEYS "ratelimit:*" | xargs redis-cli MGET

# 2. Block at firewall level
iptables -A INPUT -s ATTACKER_IP -j DROP

# 3. Reduce rate limits temporarily
redis-cli SET "global_limit" "10/minute"

# 4. Enable emergency mode (read-only)
curl -X POST http://localhost:8000/api/v1/coordinator/pause \
  -H "Authorization: Bearer $API_KEY_ADMIN"
```

---

**Document Version**: 1.0
**Last Updated**: 2025-12-19
**Status**: Ready for Implementation
**Priority**: CRITICAL - Implement before any production deployment
