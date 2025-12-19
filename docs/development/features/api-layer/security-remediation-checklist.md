# API Security Remediation Checklist

**Created**: 2025-12-19
**Based On**: Deep Code Review (api-deep-review-2025-12-19.md)
**Status**: PENDING
**Target Completion**: Week of 2025-12-23

---

## Priority 0: Critical Security Issues (MUST FIX)

### 1. Authentication & Authorization
**Status**: ❌ NOT STARTED
**Estimated Effort**: 2-3 days
**Assigned To**: _____________

#### Tasks
- [ ] Create `triplegain/src/api/auth.py` module
- [ ] Install PyJWT: `pip install pyjwt[crypto]`
- [ ] Implement `verify_token()` dependency
  - [ ] Validate JWT signature
  - [ ] Check expiration
  - [ ] Extract user claims
- [ ] Implement RBAC with roles
  - [ ] Define UserRole enum (Admin, Trader, Viewer)
  - [ ] Create `require_role()` dependency factory
  - [ ] Implement role hierarchy
- [ ] Protect all endpoints except:
  - [ ] `/health`
  - [ ] `/health/live`
  - [ ] `/health/ready`
- [ ] Apply correct role requirements:
  - [ ] Admin: `/coordinator/pause`, `/coordinator/resume`, `/portfolio/rebalance`, `/risk/reset`
  - [ ] Trader: `/positions/{id}/close`, `/orders/{id}/cancel`, `/agents/*/run`
  - [ ] Viewer: All GET endpoints
- [ ] Write authentication tests
  - [ ] Test missing token → 401
  - [ ] Test invalid token → 401
  - [ ] Test expired token → 401
  - [ ] Test insufficient role → 403
  - [ ] Test valid token → 200

**Verification**:
```bash
# Should fail without auth:
curl -X POST http://localhost:8000/api/v1/coordinator/pause
# Expected: 401 Unauthorized

# Should succeed with admin token:
curl -X POST http://localhost:8000/api/v1/coordinator/pause \
  -H "Authorization: Bearer <admin_token>"
# Expected: 200 OK
```

---

### 2. Rate Limiting
**Status**: ❌ NOT STARTED
**Estimated Effort**: 1 day
**Assigned To**: _____________

#### Tasks
- [ ] Install slowapi: `pip install slowapi`
- [ ] Create rate limiter instance in `app.py`
- [ ] Configure tiered limits:
  ```python
  RATE_LIMITS = {
      "expensive": "5/minute",    # 6 LLM models
      "moderate": "30/minute",    # Database writes
      "cheap": "100/minute",      # Database reads
      "health": "1000/minute",    # K8s probes
  }
  ```
- [ ] Apply to endpoints:
  - [ ] `@limiter.limit("5/minute")` on `/agents/trading/{symbol}/run`
  - [ ] `@limiter.limit("30/minute")` on `/positions/{id}/close`, `/orders/{id}/cancel`
  - [ ] `@limiter.limit("100/minute")` on all GET endpoints
- [ ] Configure per-user limits for critical operations
- [ ] Handle RateLimitExceeded exception (429 status)
- [ ] Write rate limit tests
  - [ ] Test limit enforcement (make N+1 requests)
  - [ ] Test different tiers
  - [ ] Test per-user vs per-IP limits

**Verification**:
```bash
# Make 6 requests to expensive endpoint:
for i in {1..6}; do
  curl -X POST http://localhost:8000/api/v1/agents/trading/BTC_USDT/run
done
# 6th request should return 429
```

---

### 3. CORS Configuration
**Status**: ❌ NOT STARTED
**Estimated Effort**: 2 hours
**Assigned To**: _____________

#### Tasks
- [ ] Define allowed origins in environment variables
  ```python
  ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
  ```
- [ ] Add CORS middleware to `app.py`:
  ```python
  from fastapi.middleware.cors import CORSMiddleware

  app.add_middleware(
      CORSMiddleware,
      allow_origins=ALLOWED_ORIGINS,
      allow_credentials=True,
      allow_methods=["GET", "POST", "PATCH", "DELETE"],
      allow_headers=["Authorization", "Content-Type"],
      max_age=600,
  )
  ```
- [ ] Document CORS configuration in README
- [ ] Add to deployment guide

**Verification**:
```bash
# Check CORS headers:
curl -H "Origin: http://localhost:3000" \
  -H "Access-Control-Request-Method: POST" \
  -X OPTIONS http://localhost:8000/api/v1/positions

# Should include Access-Control-Allow-Origin header
```

---

### 4. Request Size Limits
**Status**: ❌ NOT STARTED
**Estimated Effort**: 2 hours
**Assigned To**: _____________

#### Tasks
- [ ] Create `RequestSizeLimitMiddleware` in `app.py`
  ```python
  class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
      def __init__(self, app, max_size: int = 1024 * 1024):
          super().__init__(app)
          self.max_size = max_size

      async def dispatch(self, request: Request, call_next):
          content_length = request.headers.get("content-length")
          if content_length and int(content_length) > self.max_size:
              raise HTTPException(status_code=413, detail="Payload too large")
          return await call_next(request)
  ```
- [ ] Apply middleware: `app.add_middleware(RequestSizeLimitMiddleware, max_size=1024*1024)`
- [ ] Write test for oversized request (>1MB)

**Verification**:
```bash
# Send 2MB payload:
python -c "import requests; r = requests.post('http://localhost:8000/api/v1/portfolio/rebalance', json={'data': 'x' * (1024*1024*2)}); print(r.status_code)"
# Expected: 413
```

---

### 5. Async Operation Timeouts
**Status**: ❌ NOT STARTED
**Estimated Effort**: 1 day
**Assigned To**: _____________

#### Tasks
- [ ] Define timeout constants:
  ```python
  TIMEOUTS = {
      "llm_single": 10.0,     # Single LLM call
      "llm_parallel": 45.0,   # 6 parallel LLM calls
      "database_read": 5.0,   # DB SELECT
      "database_write": 10.0, # DB INSERT/UPDATE
      "snapshot_build": 15.0, # Full snapshot
  }
  ```
- [ ] Wrap async calls in all routes:
  - [ ] `/agents/ta/{symbol}` - 10s timeout
  - [ ] `/agents/regime/{symbol}` - 10s timeout
  - [ ] `/agents/trading/{symbol}/run` - 45s timeout
  - [ ] `/snapshot/{symbol}` - 15s timeout
  - [ ] `/indicators/{symbol}/{timeframe}` - 10s timeout
- [ ] Handle `asyncio.TimeoutError` with 504 status
- [ ] Write timeout tests (mock slow operations)

**Example**:
```python
try:
    output = await asyncio.wait_for(
        trading_agent.process(snapshot, ta_output, regime_output),
        timeout=TIMEOUTS["llm_parallel"]
    )
except asyncio.TimeoutError:
    logger.error(f"Trading decision timed out for {symbol}")
    raise HTTPException(status_code=504, detail="Trading decision timed out")
```

**Verification**:
```python
# In test:
mock_agent.process = AsyncMock(side_effect=asyncio.TimeoutError)
response = client.post("/api/v1/agents/trading/BTC_USDT/run")
assert response.status_code == 504
```

---

## Priority 1: High Issues (SHOULD FIX)

### 6. Float → Decimal Conversion
**Status**: ❌ NOT STARTED
**Estimated Effort**: 4 hours
**Assigned To**: _____________

#### Tasks
- [ ] Update Pydantic models in `routes_agents.py`:
  ```python
  from pydantic import condecimal

  class TradeProposalRequest(BaseModel):
      size_usd: condecimal(gt=Decimal('0'), decimal_places=2) = Field(...)
      entry_price: condecimal(gt=Decimal('0'), decimal_places=8) = Field(...)
      stop_loss: Optional[condecimal(gt=Decimal('0'), decimal_places=8)] = None
      take_profit: Optional[condecimal(gt=Decimal('0'), decimal_places=8)] = None
  ```
- [ ] Update `ClosePositionRequest`:
  ```python
  exit_price: condecimal(gt=Decimal('0'), decimal_places=8) = Field(...)
  ```
- [ ] Update `ModifyPositionRequest`:
  ```python
  stop_loss: Optional[condecimal(gt=Decimal('0'), decimal_places=8)] = None
  take_profit: Optional[condecimal(gt=Decimal('0'), decimal_places=8)] = None
  ```
- [ ] Update all tests to use Decimal assertions
- [ ] Verify no precision loss in calculations

**Verification**:
```python
# Test:
from decimal import Decimal
request_data = {
    "symbol": "BTC/USDT",
    "side": "buy",
    "size_usd": "1234.56",
    "entry_price": "45123.45678901",
}
response = client.post("/api/v1/risk/validate", json=request_data)
# Should preserve exact precision
```

---

### 7. Add Response Models
**Status**: ❌ NOT STARTED
**Estimated Effort**: 1 day
**Assigned To**: _____________

#### Tasks
- [ ] Create response models for all endpoints in separate file `triplegain/src/api/models.py`
- [ ] Add to routes:
  ```python
  from .models import PositionResponse, PositionsListResponse

  @router.get("/positions", response_model=PositionsListResponse)
  async def get_positions(...) -> PositionsListResponse:
      ...
  ```
- [ ] Create models for:
  - [ ] `TAAnalysisResponse`
  - [ ] `RegimeDetectionResponse`
  - [ ] `TradingDecisionResponse`
  - [ ] `RiskValidationResponse`
  - [ ] `RiskStateResponse`
  - [ ] `PortfolioAllocationResponse`
  - [ ] `PositionResponse`, `PositionsListResponse`
  - [ ] `OrderResponse`, `OrdersListResponse`
  - [ ] `CoordinatorStatusResponse`
  - [ ] `HealthCheckResponse`
- [ ] Update OpenAPI documentation automatically

---

### 8. Harden Error Handling
**Status**: ❌ NOT STARTED
**Estimated Effort**: 4 hours
**Assigned To**: _____________

#### Tasks
- [ ] Create structured error response model:
  ```python
  class ErrorResponse(BaseModel):
      detail: str
      error_code: Optional[str] = None
      request_id: Optional[str] = None
      timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
  ```
- [ ] Never catch generic exceptions without re-raising
- [ ] Only expose safe exception details (ValueError → specific exception)
- [ ] Add error codes for common errors:
  ```python
  ERROR_CODES = {
      "AGENT_NOT_INITIALIZED": "Agent not initialized",
      "INVALID_SYMBOL": "Invalid symbol format",
      "TIMEOUT": "Operation timed out",
      "RATE_LIMITED": "Rate limit exceeded",
  }
  ```
- [ ] Review all `except ValueError as e: raise HTTPException(..., detail=str(e))` → change to specific exceptions

---

### 9. Add Security Tests
**Status**: ❌ NOT STARTED
**Estimated Effort**: 2 hours
**Assigned To**: _____________

#### Tasks
- [ ] Create `triplegain/tests/security/test_api_security.py`
- [ ] Test authentication enforcement
- [ ] Test rate limiting enforcement
- [ ] Test SQL injection rejection
- [ ] Test oversized request rejection
- [ ] Test error message sanitization
- [ ] Test CORS headers

---

## Priority 2: Medium Issues (NICE TO FIX)

### 10. Centralize Symbol Validation
**Status**: ❌ NOT STARTED
**Estimated Effort**: 2 hours

#### Tasks
- [ ] Remove duplicate `SYMBOL_PATTERN` from `app.py`
- [ ] Use `validation.validate_symbol_or_raise()` everywhere
- [ ] Consistent strict/non-strict behavior

---

### 11. Add Structured Error Responses
**Status**: ❌ NOT STARTED
**Estimated Effort**: 3 hours

#### Tasks
- [ ] Implement `ErrorResponse` model
- [ ] Add error codes to all error responses
- [ ] Include request_id in all errors

---

### 12. Add Request Tracing
**Status**: ❌ NOT STARTED
**Estimated Effort**: 4 hours

#### Tasks
- [ ] Create `RequestIDMiddleware`
- [ ] Add X-Request-ID to all responses
- [ ] Include request_id in all log statements
- [ ] Use contextvars for thread-safe request ID storage

---

### 13. Add Cache Headers
**Status**: ❌ NOT STARTED
**Estimated Effort**: 2 hours

#### Tasks
- [ ] Set Cache-Control on GET endpoints
- [ ] 60s for TA analysis
- [ ] 300s for regime detection
- [ ] 5s for positions/orders

---

### 14. Improve Logging
**Status**: ❌ NOT STARTED
**Estimated Effort**: 2 hours

#### Tasks
- [ ] Add structured logging with request context
- [ ] Log all authentication attempts
- [ ] Log all rate limit violations
- [ ] Log all errors with request_id

---

### 15. Document Security
**Status**: ❌ NOT STARTED
**Estimated Effort**: 2 hours

#### Tasks
- [ ] Document authentication flow in README
- [ ] Document rate limits per endpoint
- [ ] Update OpenAPI with security schemes
- [ ] Create deployment security guide

---

## Priority 3: Optional Improvements

### 16. Complete Type Hints
**Status**: ❌ NOT STARTED
**Estimated Effort**: 1 hour

#### Tasks
- [ ] Add `-> None` to all void functions
- [ ] Add return types to all functions

---

### 17. OpenAPI Security Schemes
**Status**: ❌ NOT STARTED
**Estimated Effort**: 2 hours

#### Tasks
- [ ] Add security schemes to OpenAPI spec
- [ ] Document Bearer token requirement
- [ ] Add examples for authentication

---

### 18. Dependency Override Documentation
**Status**: ❌ NOT STARTED
**Estimated Effort**: 1 hour

#### Tasks
- [ ] Document how to override dependencies in tests
- [ ] Add examples for mocking authentication

---

## Testing Checklist

### Unit Tests
- [ ] All existing tests still pass (916 tests)
- [ ] Authentication tests added (10+ tests)
- [ ] Rate limiting tests added (5+ tests)
- [ ] Security tests added (10+ tests)
- [ ] Timeout tests added (5+ tests)

### Integration Tests
- [ ] API works with real authentication
- [ ] Rate limits enforce correctly under load
- [ ] Timeouts trigger on slow operations
- [ ] CORS works with frontend

### Security Tests
- [ ] Penetration testing passes
- [ ] OWASP top 10 vulnerabilities checked
- [ ] No sensitive data in logs
- [ ] No internal details in error messages

---

## Deployment Checklist

### Pre-Deployment
- [ ] All P0 issues fixed
- [ ] All P1 issues fixed
- [ ] Security tests passing
- [ ] Documentation updated

### Environment Variables
- [ ] `JWT_SECRET` configured (strong random key)
- [ ] `ALLOWED_ORIGINS` configured (whitelist)
- [ ] `RATE_LIMIT_REDIS_URL` configured (if using Redis)
- [ ] `MAX_REQUEST_SIZE` configured

### Infrastructure
- [ ] SSL/TLS certificate installed
- [ ] Load balancer timeout > app timeout
- [ ] Health checks configured
- [ ] Monitoring/alerting configured

### Verification
- [ ] Manual testing with Postman/curl
- [ ] Load testing with locust/k6
- [ ] Security scan with OWASP ZAP
- [ ] Performance testing

---

## Progress Tracking

| Week | Target | Status | Notes |
|------|--------|--------|-------|
| Week 1 | P0 issues fixed | ⏳ PENDING | Auth, rate limiting, CORS, size limits, timeouts |
| Week 2 | P1 issues fixed | ⏳ PENDING | Decimal, response models, error handling, security tests |
| Week 3 | P2 issues + deployment | ⏳ PENDING | Polish, documentation, deployment |

---

## Sign-off

### Code Review Approval
- [ ] All P0 issues resolved
- [ ] All P1 issues resolved
- [ ] Security review passed
- [ ] Load testing passed

**Reviewer**: _____________
**Date**: _____________

### Security Approval
- [ ] Authentication implemented and tested
- [ ] Rate limiting enforced
- [ ] No sensitive data leakage
- [ ] OWASP top 10 mitigated

**Security Reviewer**: _____________
**Date**: _____________

### Deployment Approval
- [ ] All tests passing
- [ ] Documentation updated
- [ ] Environment configured
- [ ] Rollback plan ready

**Tech Lead**: _____________
**Date**: _____________

---

**Last Updated**: 2025-12-19
**Status**: PENDING WORK
**Blocking**: Production deployment
