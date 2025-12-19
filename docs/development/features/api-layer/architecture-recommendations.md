# TripleGain API Architecture Recommendations

## Document Purpose
This document provides architectural guidance and best practices for maintaining and scaling the TripleGain API layer based on the comprehensive code review conducted on 2025-12-19.

**Audience**: Development team, system architects
**Status**: Advisory recommendations for future iterations

---

## Current Architecture Assessment

### Overall Architecture: Grade A-

The current API architecture demonstrates **excellent software engineering practices**:

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Application                   │
├─────────────────────────────────────────────────────────┤
│  Lifespan Manager                                        │
│  ├─ DatabasePool (TimescaleDB)                          │
│  ├─ IndicatorLibrary                                     │
│  ├─ MarketSnapshotBuilder                               │
│  └─ PromptBuilder                                        │
├─────────────────────────────────────────────────────────┤
│  Route Groups (Modular Design)                           │
│  ├─ Health Routes (/health/*)                           │
│  ├─ Indicator Routes (/api/v1/indicators/*)             │
│  ├─ Snapshot Routes (/api/v1/snapshot/*)                │
│  ├─ Debug Routes (/api/v1/debug/*)                      │
│  ├─ Agent Routes (/api/v1/agents/*)                     │
│  └─ Orchestration Routes (/api/v1/{coord,port,pos}/*)   │
├─────────────────────────────────────────────────────────┤
│  Cross-Cutting Concerns                                  │
│  ├─ Validation (Pydantic + Custom)                      │
│  ├─ Error Handling (try/except + HTTPException)         │
│  ├─ Logging (structured logging)                        │
│  └─ [TODO] Auth, Rate Limiting, CORS                    │
└─────────────────────────────────────────────────────────┘
```

### Strengths

1. **Clean Separation of Concerns**
   - Business logic in agents, not API layer
   - Validation separated into dedicated module
   - Route registration modular and testable

2. **Proper Async Patterns**
   - All I/O operations properly awaited
   - Lifespan manager for resource management
   - Database connection pooling

3. **Excellent Error Handling**
   - Consistent exception wrapping
   - Generic error messages (no detail leakage)
   - Comprehensive logging with context

4. **Type Safety**
   - Pydantic models for validation
   - Type hints throughout
   - FastAPI auto-generates OpenAPI spec

### Areas for Improvement

1. **Security Layer Missing** (see security-action-plan.md)
2. **Response Format Inconsistency**
3. **No Request Tracing**
4. **Limited Observability**

---

## Architectural Recommendations

### 1. Standardize Response Envelope

**Problem**: Inconsistent response formats make client integration harder.

**Current State**:
```python
# Some endpoints
return {"symbol": symbol, "output": output.to_dict(), "stats": stats}

# Others
return position.to_dict()

# Others
return {"status": "success", "data": result}
```

**Recommended Standard**:

```python
# src/api/responses.py
from typing import Generic, TypeVar, Optional, Any
from pydantic import BaseModel
from datetime import datetime

T = TypeVar('T')

class APIResponse(BaseModel, Generic[T]):
    """Standard API response envelope."""

    data: T
    meta: Optional[dict] = None
    links: Optional[dict] = None

    class Config:
        arbitrary_types_allowed = True


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: dict  # {code, message, details, request_id}


class ResponseMeta(BaseModel):
    """Response metadata."""

    timestamp: datetime
    request_id: str
    cached: bool = False
    execution_time_ms: Optional[float] = None


# Usage
@router.get("/agents/ta/{symbol}")
async def get_ta_analysis(...) -> APIResponse[TAOutput]:
    output = await ta_agent.process(snapshot)

    return APIResponse(
        data=output.to_dict(),
        meta={
            "timestamp": datetime.now(timezone.utc),
            "request_id": request.state.request_id,
            "cached": False,
            "execution_time_ms": 123.45
        },
        links={
            "self": f"/api/v1/agents/ta/{symbol}",
            "refresh": f"/api/v1/agents/ta/{symbol}/run",
            "regime": f"/api/v1/agents/regime/{symbol}"
        }
    )
```

**Benefits**:
- Consistent client parsing
- Built-in metadata for debugging
- HATEOAS support for API discoverability
- Type-safe responses

**Implementation Time**: 4 hours

---

### 2. Add Request Context & Tracing

**Problem**: No way to trace requests across services or correlate logs.

**Recommended Implementation**:

```python
# src/api/middleware.py
import uuid
import time
from contextvars import ContextVar
from starlette.middleware.base import BaseHTTPMiddleware

# Context variables for request state
request_id_var: ContextVar[str] = ContextVar('request_id', default='')
user_id_var: ContextVar[str] = ContextVar('user_id', default='')

class RequestContextMiddleware(BaseHTTPMiddleware):
    """Add request context and tracing."""

    async def dispatch(self, request, call_next):
        # Generate or extract request ID
        request_id = request.headers.get('X-Request-ID', str(uuid.uuid4()))
        request_id_var.set(request_id)

        # Extract user ID from auth
        user_id = getattr(request.state, 'user_id', 'anonymous')
        user_id_var.set(user_id)

        # Track timing
        start_time = time.time()

        # Process request
        response = await call_next(request)

        # Add headers
        response.headers['X-Request-ID'] = request_id
        response.headers['X-Response-Time'] = f"{(time.time() - start_time) * 1000:.2f}ms"

        return response


# src/api/logging_config.py
import logging
from pythonjsonlogger import jsonlogger

class RequestContextLogFilter(logging.Filter):
    """Add request context to all logs."""

    def filter(self, record):
        record.request_id = request_id_var.get('')
        record.user_id = user_id_var.get('')
        return True


# Configure structured logging
def setup_logging():
    handler = logging.StreamHandler()
    formatter = jsonlogger.JsonFormatter(
        '%(timestamp)s %(level)s %(name)s %(request_id)s %(user_id)s %(message)s'
    )
    handler.setFormatter(formatter)
    handler.addFilter(RequestContextLogFilter())

    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
```

**Benefits**:
- Trace requests across logs
- Correlate user actions with errors
- Measure endpoint performance
- Support distributed tracing (OpenTelemetry compatible)

**Implementation Time**: 3 hours

---

### 3. Implement Dependency Injection Pattern

**Problem**: Global state makes testing harder and limits scalability.

**Current Pattern**:
```python
# Global variables
_db_pool: Optional[DatabasePool] = None
_indicator_library: Optional[IndicatorLibrary] = None

# Used in handlers
@router.get("/api/v1/indicators/{symbol}/{timeframe}")
async def get_indicators(symbol: str, timeframe: str):
    if not _db_pool or not _indicator_library:
        raise HTTPException(status_code=503, detail="Service not initialized")
    # ... use globals
```

**Recommended Pattern**:

```python
# src/api/dependencies.py
from fastapi import Depends, HTTPException

# Container holds singleton instances
class ServiceContainer:
    def __init__(self):
        self.db_pool = None
        self.indicator_library = None
        self.snapshot_builder = None
        self.ta_agent = None
        # ... etc

    def initialize(self, config):
        """Initialize all services."""
        self.db_pool = create_pool_from_config(config.database)
        self.indicator_library = IndicatorLibrary(config.indicators, self.db_pool)
        # ... etc

# Global container instance
_container = ServiceContainer()


# Dependency functions
async def get_db_pool() -> DatabasePool:
    """Get database pool dependency."""
    if not _container.db_pool:
        raise HTTPException(status_code=503, detail="Database not initialized")
    return _container.db_pool


async def get_indicator_library() -> IndicatorLibrary:
    """Get indicator library dependency."""
    if not _container.indicator_library:
        raise HTTPException(status_code=503, detail="Indicator library not initialized")
    return _container.indicator_library


# Usage in routes
@router.get("/api/v1/indicators/{symbol}/{timeframe}")
async def get_indicators(
    symbol: str,
    timeframe: str,
    db_pool: DatabasePool = Depends(get_db_pool),
    indicator_lib: IndicatorLibrary = Depends(get_indicator_library),
):
    # Now easily mockable in tests
    candles = await db_pool.fetch_candles(symbol, timeframe, limit)
    indicators = indicator_lib.calculate_all(symbol, timeframe, candles)
    return indicators
```

**Benefits**:
- Easier testing (inject mocks)
- Clear dependency graph
- Type-safe injection
- Supports multiple instances (blue/green deployment)

**Implementation Time**: 6 hours (requires refactoring)

---

### 4. Add Comprehensive Observability

**Problem**: Limited visibility into system health and performance.

**Recommended Implementation**:

```python
# src/api/metrics.py
from prometheus_client import Counter, Histogram, Gauge, Info
from prometheus_client import make_asgi_app

# Define metrics
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code']
)

http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint']
)

agent_calls_total = Counter(
    'agent_calls_total',
    'Total agent calls',
    ['agent_type', 'symbol', 'status']
)

agent_call_duration_seconds = Histogram(
    'agent_call_duration_seconds',
    'Agent call duration',
    ['agent_type']
)

llm_api_cost_usd = Counter(
    'llm_api_cost_usd_total',
    'Total LLM API costs in USD',
    ['provider', 'model']
)

db_pool_size = Gauge(
    'db_pool_size',
    'Database connection pool size'
)

db_pool_idle = Gauge(
    'db_pool_idle',
    'Idle database connections'
)

# Middleware to collect metrics
class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        start_time = time.time()

        response = await call_next(request)

        duration = time.time() - start_time

        # Record metrics
        http_requests_total.labels(
            method=request.method,
            endpoint=request.url.path,
            status_code=response.status_code
        ).inc()

        http_request_duration_seconds.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(duration)

        return response


# Add to app
def create_app():
    app = FastAPI(...)

    # Add metrics middleware
    app.add_middleware(MetricsMiddleware)

    # Mount metrics endpoint
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    return app
```

**Grafana Dashboard Configuration**:

```json
{
  "dashboard": {
    "title": "TripleGain API",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])"
          }
        ]
      },
      {
        "title": "Request Latency (p95)",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))"
          }
        ]
      },
      {
        "title": "LLM API Costs (24h)",
        "targets": [
          {
            "expr": "increase(llm_api_cost_usd_total[24h])"
          }
        ]
      },
      {
        "title": "Database Pool Usage",
        "targets": [
          {
            "expr": "db_pool_idle / db_pool_size"
          }
        ]
      }
    ]
  }
}
```

**Benefits**:
- Real-time performance monitoring
- Cost tracking and alerting
- Capacity planning data
- SLA compliance verification

**Implementation Time**: 5 hours

---

### 5. Implement Circuit Breaker Pattern

**Problem**: Cascading failures if LLM providers or database go down.

**Recommended Implementation**:

```python
# src/api/circuit_breaker.py
from enum import Enum
from datetime import datetime, timedelta
from typing import Callable, Any
import asyncio

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered

class CircuitBreaker:
    """Circuit breaker for external service calls."""

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout_seconds: int = 60,
        name: str = "default"
    ):
        self.failure_threshold = failure_threshold
        self.timeout = timedelta(seconds=timeout_seconds)
        self.name = name

        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to try again."""
        return (
            self.state == CircuitState.OPEN
            and self.last_failure_time
            and datetime.now() - self.last_failure_time > self.timeout
        )

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""

        # Open circuit - reject immediately
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                logger.info(f"Circuit {self.name} entering HALF_OPEN")
            else:
                raise HTTPException(
                    status_code=503,
                    detail=f"Service temporarily unavailable: {self.name}"
                )

        # Try to execute
        try:
            result = await func(*args, **kwargs)

            # Success - reset counter
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.CLOSED
                logger.info(f"Circuit {self.name} CLOSED (recovered)")

            self.failure_count = 0
            return result

        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = datetime.now()

            logger.warning(
                f"Circuit {self.name} failure {self.failure_count}/{self.failure_threshold}"
            )

            # Trip circuit if threshold reached
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                logger.error(f"Circuit {self.name} OPENED (too many failures)")

            raise


# Usage
llm_circuit_breaker = CircuitBreaker(
    failure_threshold=5,
    timeout_seconds=60,
    name="llm_api"
)

db_circuit_breaker = CircuitBreaker(
    failure_threshold=3,
    timeout_seconds=30,
    name="database"
)


@router.post("/agents/trading/{symbol}/run")
async def run_trading_decision(symbol: str, ...):
    try:
        # Protected LLM call
        result = await llm_circuit_breaker.call(
            trading_agent.process,
            snapshot
        )
        return result

    except HTTPException:
        # Circuit is open - return cached decision or safe default
        cached = await trading_agent.get_latest_output(symbol, max_age=3600)
        if cached:
            return {
                "warning": "Using cached decision (LLM service degraded)",
                "data": cached.to_dict()
            }
        raise
```

**Benefits**:
- Prevent cascading failures
- Faster failure detection
- Automatic recovery testing
- Graceful degradation

**Implementation Time**: 4 hours

---

### 6. Add API Versioning Strategy

**Problem**: No strategy for handling breaking changes.

**Recommended Approach**:

```python
# Approach 1: URL Path Versioning (Current)
# /api/v1/agents/ta/{symbol}
# /api/v2/agents/ta/{symbol}  # Breaking changes

# Approach 2: Header Versioning
# X-API-Version: 1.0
# X-API-Version: 2.0

# Approach 3: Content Negotiation
# Accept: application/vnd.triplegain.v1+json
# Accept: application/vnd.triplegain.v2+json
```

**Recommended: URL Path Versioning (already started)**

**Version Lifecycle Policy**:

```python
# config/api_versions.yaml
versions:
  v1:
    status: stable
    released: 2024-01-01
    deprecated: null
    sunset: null
    supported_until: 2025-12-31

  v2:
    status: beta
    released: 2024-06-01
    deprecated: null
    sunset: null
    supported_until: 2026-12-31

policies:
  # How long to support old versions
  support_window_months: 24

  # Deprecation warning period
  deprecation_warning_months: 6

  # Beta period before stable
  beta_period_months: 3
```

**Implementation**:

```python
# src/api/versioning.py
from enum import Enum
from datetime import datetime

class APIVersion(Enum):
    V1 = "v1"
    V2 = "v2"

class VersionInfo:
    version: str
    status: str  # stable, deprecated, sunset
    sunset_date: datetime = None

# Middleware to add version warnings
class VersionWarningMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        version = self._extract_version(request.url.path)

        response = await call_next(request)

        # Add deprecation warning header
        if self._is_deprecated(version):
            response.headers['X-API-Deprecated'] = 'true'
            response.headers['X-API-Sunset'] = self._get_sunset_date(version)

        response.headers['X-API-Version'] = version
        return response
```

**Migration Guide Template**:

```markdown
# API v1 to v2 Migration Guide

## Breaking Changes

### 1. Response Format
**v1**:
```json
{"symbol": "BTC/USDT", "output": {...}}
```

**v2**:
```json
{
  "data": {...},
  "meta": {"timestamp": "...", "request_id": "..."}
}
```

### 2. Authentication
**v1**: Optional (security issue)
**v2**: Required on all endpoints

## Migration Steps
1. Update client to include `Authorization` header
2. Update response parsing for new envelope format
3. Test in staging environment
4. Deploy to production
5. Monitor for errors
```

**Implementation Time**: 3 hours

---

### 7. Add Health Check Maturity

**Current State**: Basic health checks exist
**Recommended Enhancement**: Detailed component health

```python
# src/api/health.py
from enum import Enum
from typing import Dict, List
from dataclasses import dataclass

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

@dataclass
class ComponentHealth:
    name: str
    status: HealthStatus
    message: str = ""
    details: dict = None
    last_check: datetime = None

class HealthChecker:
    """Comprehensive health checking."""

    async def check_database(self) -> ComponentHealth:
        """Check database health with details."""
        try:
            # Check connection
            await db_pool.execute("SELECT 1")

            # Check pool status
            pool_size = db_pool.get_size()
            idle = db_pool.get_idle_size()

            if idle < 2:
                return ComponentHealth(
                    name="database",
                    status=HealthStatus.DEGRADED,
                    message="Connection pool nearly exhausted",
                    details={
                        "pool_size": pool_size,
                        "idle_connections": idle,
                        "usage_pct": ((pool_size - idle) / pool_size * 100)
                    }
                )

            return ComponentHealth(
                name="database",
                status=HealthStatus.HEALTHY,
                details={"pool_size": pool_size, "idle": idle}
            )

        except Exception as e:
            return ComponentHealth(
                name="database",
                status=HealthStatus.UNHEALTHY,
                message=str(e)
            )

    async def check_llm_providers(self) -> List[ComponentHealth]:
        """Check each LLM provider health."""
        results = []

        for provider in ["openai", "anthropic", "grok", "deepseek"]:
            try:
                # Simple health check call (cached)
                response = await llm_client.health_check(provider)

                results.append(ComponentHealth(
                    name=f"llm_{provider}",
                    status=HealthStatus.HEALTHY if response.ok else HealthStatus.DEGRADED
                ))
            except Exception as e:
                results.append(ComponentHealth(
                    name=f"llm_{provider}",
                    status=HealthStatus.UNHEALTHY,
                    message=str(e)
                ))

        return results

    async def check_all(self) -> Dict[str, ComponentHealth]:
        """Run all health checks."""
        results = {}

        # Database
        results["database"] = await self.check_database()

        # LLM providers
        for component in await self.check_llm_providers():
            results[component.name] = component

        # Redis (rate limiting)
        results["redis"] = await self.check_redis()

        # Agents
        results["agents"] = await self.check_agents()

        return results


@router.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with all components."""
    checker = HealthChecker()
    components = await checker.check_all()

    # Determine overall status
    statuses = [c.status for c in components.values()]
    if any(s == HealthStatus.UNHEALTHY for s in statuses):
        overall = HealthStatus.UNHEALTHY
        http_status = 503
    elif any(s == HealthStatus.DEGRADED for s in statuses):
        overall = HealthStatus.DEGRADED
        http_status = 200  # Still operational
    else:
        overall = HealthStatus.HEALTHY
        http_status = 200

    return JSONResponse(
        status_code=http_status,
        content={
            "status": overall.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "components": {
                name: {
                    "status": component.status.value,
                    "message": component.message,
                    "details": component.details
                }
                for name, component in components.items()
            }
        }
    )
```

**Benefits**:
- Kubernetes readiness probes can be more intelligent
- Operations team has better visibility
- Automatic alerting on degradation

**Implementation Time**: 3 hours

---

## Testing Strategy Recommendations

### Current State
- 102 unit tests for API endpoints
- Good coverage of happy paths
- Limited error scenario testing

### Recommended Additions

#### 1. Integration Tests

```python
# tests/integration/api/test_trading_flow.py
@pytest.mark.integration
async def test_complete_trading_flow():
    """Test end-to-end trading decision flow."""

    # 1. Check system health
    response = await client.get("/health")
    assert response.status_code == 200

    # 2. Trigger TA analysis
    response = await client.post("/api/v1/agents/ta/BTC/USDT/run")
    assert response.status_code == 200
    ta_output = response.json()

    # 3. Trigger regime detection
    response = await client.post("/api/v1/agents/regime/BTC/USDT/run")
    assert response.status_code == 200
    regime_output = response.json()

    # 4. Make trading decision
    response = await client.post("/api/v1/agents/trading/BTC/USDT/run")
    assert response.status_code == 200
    decision = response.json()

    # 5. Validate with risk engine
    if decision["consensus"]["action"] != "hold":
        response = await client.post("/api/v1/risk/validate", json={
            "symbol": "BTC/USDT",
            "side": decision["consensus"]["action"],
            "size_usd": 1000.0,
            "entry_price": decision["consensus"]["entry_price"],
            # ... etc
        })
        assert response.status_code == 200
        assert response.json()["approved"] in [True, False]
```

#### 2. Chaos Testing

```python
# tests/chaos/test_resilience.py
@pytest.mark.chaos
async def test_database_connection_loss():
    """Test behavior when database connection is lost."""

    # Normal operation
    response = await client.get("/api/v1/indicators/BTC/USDT/1h")
    assert response.status_code == 200

    # Simulate DB connection loss
    await db_pool.disconnect()

    # Should return 503
    response = await client.get("/api/v1/indicators/BTC/USDT/1h")
    assert response.status_code == 503
    assert "database" in response.json()["detail"].lower()

    # Reconnect
    await db_pool.connect()

    # Should recover
    response = await client.get("/api/v1/indicators/BTC/USDT/1h")
    assert response.status_code == 200


@pytest.mark.chaos
async def test_llm_provider_timeout():
    """Test behavior when LLM provider times out."""

    # Mock slow LLM response
    with mock.patch.object(llm_client, 'call', side_effect=asyncio.TimeoutError):
        response = await client.post("/api/v1/agents/trading/BTC/USDT/run")

        # Should fallback to cached decision or fail gracefully
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            assert "cached" in response.json()
```

#### 3. Load Testing

```python
# tests/load/locustfile.py
from locust import HttpUser, task, between

class TradingBotUser(HttpUser):
    wait_time = between(1, 5)

    def on_start(self):
        """Setup auth token."""
        self.headers = {
            "Authorization": f"Bearer {os.getenv('API_KEY_TRADING')}"
        }

    @task(1)
    def view_positions(self):
        """View open positions."""
        self.client.get("/api/v1/positions", headers=self.headers)

    @task(2)
    def get_ta_analysis(self):
        """Get TA analysis (cached)."""
        self.client.get("/api/v1/agents/ta/BTC/USDT", headers=self.headers)

    @task(1)
    def trading_decision(self):
        """Trigger trading decision (expensive)."""
        self.client.post(
            "/api/v1/agents/trading/BTC/USDT/run",
            json={"use_ta": True, "use_regime": True},
            headers=self.headers
        )

# Run: locust -f tests/load/locustfile.py --host http://localhost:8000
```

**Target Metrics**:
- 95th percentile latency <500ms for data endpoints
- 95th percentile latency <2s for analysis endpoints
- 95th percentile latency <10s for trading decisions
- Handle 100 concurrent users
- Zero errors under normal load

**Implementation Time**: 8 hours

---

## Deployment Architecture Recommendations

### Current: Single-Process Application
```
┌──────────────┐
│   FastAPI    │
│   Process    │
│  (Uvicorn)   │
└──────────────┘
```

### Recommended: Production Architecture

```
                                    ┌─────────────┐
                                    │   Nginx /   │
                                    │  Load Bal.  │
                                    └──────┬──────┘
                                           │
                        ┌──────────────────┼──────────────────┐
                        │                  │                  │
                   ┌────▼────┐        ┌───▼────┐        ┌────▼────┐
                   │ FastAPI │        │FastAPI │        │ FastAPI │
                   │Instance │        │Instance│        │Instance │
                   │   #1    │        │  #2    │        │   #3    │
                   └────┬────┘        └───┬────┘        └────┬────┘
                        │                  │                  │
                        └──────────────────┼──────────────────┘
                                           │
                        ┌──────────────────┼──────────────────┐
                        │                  │                  │
                   ┌────▼─────┐      ┌────▼─────┐      ┌─────▼─────┐
                   │TimescaleDB│      │  Redis   │      │ Ollama    │
                   │  Primary  │      │ (Cache & │      │ (Local    │
                   │           │      │  Rate    │      │  LLM)     │
                   └───────────┘      │ Limit)   │      └───────────┘
                                      └──────────┘
```

### Docker Compose Configuration

```yaml
version: '3.8'

services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - api1
      - api2
      - api3

  api1:
    build: .
    environment:
      - INSTANCE_ID=1
      - ENVIRONMENT=production
    env_file:
      - .env.api
    depends_on:
      - timescaledb
      - redis
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G

  api2:
    build: .
    environment:
      - INSTANCE_ID=2
      - ENVIRONMENT=production
    env_file:
      - .env.api
    depends_on:
      - timescaledb
      - redis
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G

  api3:
    build: .
    environment:
      - INSTANCE_ID=3
      - ENVIRONMENT=production
    env_file:
      - .env.api
    depends_on:
      - timescaledb
      - redis
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G

  timescaledb:
    image: timescale/timescaledb:latest-pg14
    volumes:
      - timescale_data:/var/lib/postgresql/data
    environment:
      - POSTGRES_PASSWORD=${DATABASE_PASSWORD}
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G

  redis:
    image: redis:7-alpine
    command: redis-server --maxmemory 2gb --maxmemory-policy allkeys-lru
    volumes:
      - redis_data:/data
    deploy:
      resources:
        limits:
          cpus: '1'
          memory: 2G

  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana:latest
    volumes:
      - grafana_data:/var/lib/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}

volumes:
  timescale_data:
  redis_data:
  prometheus_data:
  grafana_data:
```

---

## Documentation Recommendations

### Current State
- Good code-level docstrings
- No external API documentation

### Recommended Additions

#### 1. OpenAPI Enhancement

```python
# Enhance metadata
app = FastAPI(
    title="TripleGain API",
    description="""
    LLM-Assisted Trading System API

    ## Features
    - Real-time technical analysis
    - Multi-model consensus trading decisions
    - Risk management and position tracking
    - Portfolio rebalancing

    ## Authentication
    All endpoints require Bearer token authentication.
    Contact admin for API keys.

    ## Rate Limits
    - Analysis endpoints: 30/minute
    - Trading decisions: 10/minute
    - Data endpoints: 60/minute
    """,
    version="1.0.0",
    contact={
        "name": "TripleGain Support",
        "email": "support@triplegain.com",
    },
    license_info={
        "name": "Proprietary",
    },
)

# Add tags with descriptions
tags_metadata = [
    {
        "name": "agents",
        "description": "LLM agent endpoints for analysis and trading decisions",
    },
    {
        "name": "risk",
        "description": "Risk management and validation",
    },
    {
        "name": "orchestration",
        "description": "System coordination, portfolio, and position management",
    },
]

app = FastAPI(..., openapi_tags=tags_metadata)
```

#### 2. API Usage Examples

```python
# Add examples to Pydantic models
class TradingDecisionRequest(BaseModel):
    use_ta: bool = Field(True, description="Include TA agent output")

    class Config:
        schema_extra = {
            "example": {
                "use_ta": True,
                "use_regime": True,
                "force_refresh": False
            }
        }
```

#### 3. Postman Collection

Generate automatically:
```bash
# Generate OpenAPI spec
curl http://localhost:8000/openapi.json > openapi.json

# Convert to Postman collection
openapi2postmanv2 -s openapi.json -o triplegain-api.postman_collection.json

# Import into Postman
```

---

## Priority Implementation Roadmap

### Phase 1: Security & Stability (Week 1-2)
1. ✅ Authentication & Authorization (8h)
2. ✅ Rate Limiting (6h)
3. ✅ CORS Configuration (1h)
4. ✅ Request Size Limits (1h)
5. ✅ Debug Endpoint Gating (1h)

**Total: ~17 hours**

### Phase 2: Observability (Week 3)
1. Request Tracing & Context (3h)
2. Prometheus Metrics (5h)
3. Enhanced Health Checks (3h)
4. Structured Logging (2h)

**Total: ~13 hours**

### Phase 3: Resilience (Week 4)
1. Circuit Breakers (4h)
2. Response Envelope Standardization (4h)
3. Error Handling Enhancement (2h)

**Total: ~10 hours**

### Phase 4: Testing & Documentation (Week 5-6)
1. Integration Test Suite (8h)
2. Load Testing (4h)
3. Chaos Testing (4h)
4. API Documentation (4h)

**Total: ~20 hours**

### Phase 5: Production Readiness (Week 7)
1. Deployment Architecture (4h)
2. Monitoring Dashboards (4h)
3. Runbooks & Procedures (4h)
4. Performance Tuning (4h)

**Total: ~16 hours**

---

## Conclusion

The TripleGain API is **architecturally sound** with a strong foundation. With the recommended security additions and observability enhancements, it will be production-ready with enterprise-grade reliability.

**Key Takeaways**:
1. Add authentication/rate-limiting before any deployment (P0)
2. Implement observability for operational visibility (P1)
3. Standardize responses for better client integration (P2)
4. Add resilience patterns for fault tolerance (P2)
5. Enhance testing for confidence in changes (P3)

**Total Implementation Effort**: ~76 hours (approximately 2 sprints)

---

**Document Version**: 1.0
**Last Updated**: 2025-12-19
**Status**: Advisory Recommendations
**Next Review**: After Phase 3 implementation
