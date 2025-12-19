# TripleGain Code Standards

**Version**: 1.0
**Last Updated**: 2025-12-19
**Status**: Active

---

## Overview

This document defines code quality standards for the TripleGain trading system based on code review findings from Phase 3 implementation.

---

## 1. SQL Query Standards

### Rule: No String Interpolation in SQL

**Severity**: High
**Rationale**: Prevents SQL injection attacks and sets proper security precedent

**Bad**:
```python
query = """
    SELECT * FROM table
    WHERE timestamp > NOW() - INTERVAL '%s seconds'
""" % max_age_seconds
```

**Good**:
```python
query = """
    SELECT * FROM table
    WHERE timestamp > NOW() - INTERVAL $1
"""
result = await db.fetchrow(query, f'{max_age_seconds} seconds')
```

**Exceptions**: None. Always use parameterized queries with `$1, $2, ...` placeholders.

---

## 2. State Management Standards

### Rule: Symbol-Specific State for Multi-Symbol Agents

**Severity**: Medium
**Rationale**: Prevents state mixing when agent processes multiple trading pairs

**Bad**:
```python
class Agent:
    def __init__(self):
        self._previous_value = None  # Single value for all symbols
```

**Good**:
```python
class Agent:
    def __init__(self):
        self._state: dict[str, dict] = {}  # symbol -> state
```

**Exception**: Single-symbol deployments where agent instances are symbol-specific

---

## 3. Consensus Algorithm Standards

### Rule: Explicit Tie-Breaking with Safety Priority

**Severity**: High
**Rationale**: Ensures predictable, safe behavior on split decisions

**Bad**:
```python
winning_action = max(votes.keys(), key=lambda a: votes[a])  # Arbitrary on tie
```

**Good**:
```python
SAFETY_PRIORITY = {'HOLD': 0, 'CLOSE': 1, 'SELL': 2, 'BUY': 3}

winning_action = max(
    votes.keys(),
    key=lambda a: (votes[a], -SAFETY_PRIORITY.get(a, 99))
)
```

**Applies to**: Any voting/consensus mechanism in trading decisions

---

## 4. Magic Number Standards

### Rule: Named Constants for Thresholds

**Severity**: Low
**Rationale**: Improves maintainability and configurability

**Bad**:
```python
if rsi > 60 and macd_hist > 0:
    bias = 'long'
```

**Good**:
```python
RSI_BULLISH_THRESHOLD = 60
RSI_BEARISH_THRESHOLD = 40

if rsi > RSI_BULLISH_THRESHOLD and macd_hist > 0:
    bias = 'long'
```

**Exception**: Mathematical constants (π, e), well-known standards (HTTP 200, 404)

---

## 5. Financial Calculation Standards

### Rule: Use Decimal for All Financial Math

**Severity**: Critical
**Rationale**: Prevents floating-point rounding errors in money calculations

**Bad**:
```python
total = btc_value + xrp_value  # float arithmetic
```

**Good**:
```python
from decimal import Decimal

total = Decimal(btc_value) + Decimal(xrp_value)
```

**Applies to**: All price, amount, fee, PnL calculations

---

## 6. Error Handling Standards

### Rule: Graceful Degradation with Fallback Outputs

**Severity**: High
**Rationale**: System must remain operational despite LLM/API failures

**Pattern**:
```python
async def process(self, snapshot):
    try:
        # Primary path: LLM call
        response = await self._call_llm(prompt)
        output = self._parse_response(response)
    except Exception as e:
        logger.error(f"LLM failed: {e}")
        # Fallback path: Indicator heuristics
        output = self._create_fallback_output(snapshot, error=str(e))

    return output
```

**Requirements**:
1. Always return valid output (never None)
2. Set confidence=0.0 for fallback outputs
3. Log failure with context
4. Include error message in reasoning

---

## 7. Validation Standards

### Rule: Fail-Fast on Configuration Errors

**Severity**: Medium
**Rationale**: Catch configuration issues at startup, not during trading

**Bad**:
```python
if abs(total_allocation - 100) > 0.1:
    logger.warning("Allocation doesn't sum to 100%")  # Continues anyway
```

**Good**:
```python
if abs(total_allocation - 100) > 0.1:
    raise ValueError(f"Allocation must sum to 100%, got {total_allocation}%")
```

**Applies to**: Configuration validation in `__init__()` methods

---

## 8. Data Integrity Standards

### Rule: Error on Data Inconsistencies, Don't Clamp

**Severity**: Medium
**Rationale**: Data problems should be fixed, not hidden

**Bad**:
```python
if available_btc < 0:
    logger.warning("Negative balance, clamping to 0")
    available_btc = 0  # Masks underlying issue
```

**Good**:
```python
if available_btc < 0:
    raise ValueError(
        f"Data integrity error: hodl bag exceeds balance "
        f"(bag={hodl_bag}, balance={balance})"
    )
```

**Exception**: User input sanitization where clamping is expected behavior

---

## 9. Async Method Standards

### Rule: Document Timeout and Partial Result Behavior

**Severity**: Medium
**Rationale**: Callers need to understand timeout semantics

**Required in Docstring**:
```python
async def query_models(self, ...):
    """
    Query all models with timeout.

    Timeout Behavior:
        - Returns partial results if some models respond
        - Minimum 3/6 models required for valid consensus
        - Timed-out models logged but excluded from vote

    Args:
        timeout_seconds: Max wait time for all models

    Returns:
        ConsensusResult with at least min_models responses

    Raises:
        InsufficientDataError: If <3 models respond
    """
```

---

## 10. Module Export Standards

### Rule: All Public Classes in `__all__`

**Severity**: High
**Rationale**: Enables clean imports, explicit API surface

**Template** (`__init__.py`):
```python
from .module_a import ClassA, ClassB
from .module_b import ClassC

__all__ = [
    'ClassA',
    'ClassB',
    'ClassC',
]
```

**Checklist**:
- [ ] All agent classes exported
- [ ] All output dataclasses exported
- [ ] Public helper classes exported
- [ ] Private classes (prefixed with `_`) excluded

---

## 11. Confidence Score Standards

### Rule: Separate Base and Boosted Confidence

**Severity**: Medium
**Rationale**: Downstream components (Risk Engine) need unmodified confidence

**Pattern**:
```python
@dataclass
class Output:
    confidence: float  # Base confidence before adjustments
    boosted_confidence: float  # After consensus/agreement bonus
    confidence_adjustments: dict  # Details of boost calculation
```

**Applies to**: Multi-model consensus, confidence-modified outputs

---

## 12. Testing Standards

### Unit Test Coverage Requirements

**Minimum**: 80% line coverage
**Target**: 90% line coverage

**Required Test Categories**:
1. Happy path (normal operation)
2. Edge cases (boundaries, empty inputs, zero values)
3. Error cases (LLM failures, timeouts, invalid data)
4. Validation (schema compliance, constraint checking)

**Critical Paths Requiring 100% Coverage**:
- Financial calculations (prices, amounts, fees)
- Consensus algorithms
- Risk validation logic
- State management (caching, persistence)

---

## 13. Logging Standards

### Log Levels

| Level | Use Case | Example |
|-------|----------|---------|
| DEBUG | Detailed flow, cache hits | `Cache hit for BTC (age=2.3s)` |
| INFO | Major operations, decisions | `TA Agent: BTC bias=long confidence=0.82` |
| WARNING | Recoverable issues | `LLM parse failed, using indicators` |
| ERROR | Failures requiring attention | `Database write failed: {error}` |

### Required Context in Logs

**Always Include**:
- Agent name
- Symbol being processed
- Relevant identifiers (output_id, trade_id)

**Good**:
```python
logger.info(f"TA Agent: {symbol} bias={bias} confidence={conf:.2f} latency={ms}ms")
```

**Bad**:
```python
logger.info(f"Analysis complete")  # Missing context
```

---

## 14. Performance Standards

### Latency Targets

| Agent | Target | Timeout | Status |
|-------|--------|---------|--------|
| Technical Analysis | <500ms | 5000ms | Enforced |
| Regime Detection | <500ms | 5000ms | Enforced |
| Trading Decision | <10000ms | 30000ms | Monitored |
| Portfolio Rebalance | N/A | 30000ms | Monitored |

### Caching Requirements

**Must Cache**:
- LLM outputs (TTL: 5 minutes default)
- Market prices (TTL: 5 seconds)
- Indicator calculations (TTL: 1 minute)

**Cache Implementation**:
```python
async def get_cached(self, key: str) -> Optional[T]:
    async with self._cache_lock:
        if key in self._cache:
            value, cached_at = self._cache[key]
            if (datetime.now() - cached_at).seconds < self.ttl:
                return value
    return None
```

---

## 15. Observability Standards

### Required Metrics Per Agent

```python
def get_stats(self) -> dict:
    return {
        "agent_name": self.agent_name,
        "total_invocations": ...,
        "total_latency_ms": ...,
        "average_latency_ms": ...,
        "total_tokens": ...,
        "total_cost_usd": ...,
        "cache_hit_rate": ...,
        "error_rate": ...,
    }
```

### Model Comparison Tracking

For multi-model agents:
```python
# Store individual model decisions
await self.db.execute(
    "INSERT INTO model_comparisons ...",
    model_name, action, confidence, cost, was_consensus
)
```

---

## 16. Schema Validation Standards

### Rule: All Outputs Must Have JSON Schema

**Required**:
```python
class Agent(BaseAgent):
    def get_output_schema(self) -> dict:
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "required": ["timestamp", "symbol", "confidence"],
            "properties": {
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1
                },
                # ...
            }
        }
```

**Validation**:
```python
# In tests
def test_output_schema_valid():
    schema = agent.get_output_schema()
    output = agent.process(snapshot)
    jsonschema.validate(output.to_dict(), schema)
```

---

## 17. Type Hints Standards

### Rule: Comprehensive Type Hints

**Required Coverage**: All public methods, class attributes

**Example**:
```python
from typing import Optional, TYPE_CHECKING
from decimal import Decimal

if TYPE_CHECKING:
    from ..data.market_snapshot import MarketSnapshot

async def process(
    self,
    snapshot: 'MarketSnapshot',
    portfolio_context: Optional['PortfolioContext'] = None,
) -> AgentOutput:
    ...
```

**Use `TYPE_CHECKING`**: For forward references to avoid circular imports

---

## 18. Security Standards

### Sensitive Data Handling

**Never Log**:
- API keys
- Private keys
- Authentication tokens

**Redact in Logs**:
```python
logger.info(f"API call to {url} with key={key[:8]}...")  # First 8 chars only
```

**Environment Variables**:
```python
API_KEY = os.getenv("KRAKEN_API_KEY")
if not API_KEY:
    raise ValueError("KRAKEN_API_KEY environment variable required")
```

---

## 19. Code Review Checklist

Before submitting code for review:

- [ ] All SQL queries use parameterized placeholders
- [ ] Magic numbers replaced with named constants
- [ ] Financial calculations use Decimal
- [ ] Error handling with fallback outputs
- [ ] Configuration validated in `__init__()`
- [ ] Public classes exported in `__all__`
- [ ] Type hints on all methods
- [ ] Docstrings with Args/Returns/Raises
- [ ] Unit tests added (80%+ coverage)
- [ ] Schema validation test added
- [ ] No sensitive data in logs
- [ ] Performance metrics recorded

---

## 20. Anti-Patterns to Avoid

### 1. Silent Failures
```python
# BAD
try:
    result = risky_operation()
except Exception:
    pass  # Silently fails

# GOOD
try:
    result = risky_operation()
except Exception as e:
    logger.error(f"Operation failed: {e}")
    return fallback_result()
```

### 2. Returning None on Error
```python
# BAD
def process(self, snapshot):
    if error:
        return None  # Caller has to check everywhere

# GOOD
def process(self, snapshot):
    if error:
        return self._create_fallback_output(snapshot, error)
```

### 3. Using Float for Money
```python
# BAD
total = price * quantity  # float arithmetic

# GOOD
total = Decimal(price) * Decimal(quantity)
```

### 4. Forgetting to Await
```python
# BAD
result = async_function()  # Forgot await, gets coroutine

# GOOD
result = await async_function()
```

---

## Review Process

### When to Trigger Code Review

1. New agent implementation
2. Changes to consensus/voting logic
3. Financial calculation changes
4. Security-sensitive code (auth, API calls)
5. Before merging to main

### Review Checklist

Reviewer must verify:
- [ ] Standards compliance (this document)
- [ ] Security standards (see security-standards.md)
- [ ] Risk management standards (see risk-management-standards.md)
- [ ] Test coverage meets requirements
- [ ] No regressions in existing tests

---

## 21. Async/Await Standards (Phase 1 Review)

### Rule: CPU-Bound Operations Must Not Block Event Loop

**Severity**: High
**Rationale**: Prevents event loop blocking, maintains concurrency for parallel operations

**Bad**:
```python
async def build_snapshot(self, symbol: str):
    # ... async database calls
    indicators = self.indicator_lib.calculate_all(...)  # Blocks event loop for 30-50ms
    return snapshot
```

**Good**:
```python
from concurrent.futures import ThreadPoolExecutor

class Builder:
    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=4)

    async def build_snapshot(self, symbol: str):
        # ... async database calls
        indicators = await asyncio.get_event_loop().run_in_executor(
            self._executor,
            self.indicator_lib.calculate_all,
            symbol, timeframe, candles
        )
        return snapshot
```

**Applies to**: Indicator calculations, data processing, any CPU-bound operations in async code

---

## 22. Database Schema Validation

### Rule: Validate Required Schema on Startup

**Severity**: Medium
**Rationale**: Fail fast with clear errors instead of runtime failures

**Pattern**:
```python
class DatabasePool:
    async def connect(self):
        await self._create_pool()
        await self._validate_schema()  # Check required tables exist

    async def _validate_schema(self):
        """Verify required tables exist."""
        required_tables = [
            'candles_1m', 'candles_5m', 'candles_1h',
            'order_book_snapshots', 'agent_outputs'
        ]
        async with self.acquire() as conn:
            result = await conn.fetch(
                "SELECT table_name FROM information_schema.tables WHERE table_name = ANY($1)",
                required_tables
            )
            found = {row['table_name'] for row in result}
            missing = set(required_tables) - found
            if missing:
                raise RuntimeError(f"Missing required tables: {missing}")
```

**Applies to**: Database connection initialization, migration validation

---

## 23. Query Performance Monitoring

### Rule: Log Query Execution Times

**Severity**: Medium
**Rationale**: Enables identification of slow queries, performance optimization

**Pattern**:
```python
import time

async def fetch_candles(self, symbol: str, timeframe: str):
    start = time.perf_counter()
    try:
        async with self.acquire() as conn:
            rows = await conn.fetch(query, symbol, timeframe)
        return rows
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.debug(f"Query fetch_candles took {elapsed_ms:.2f}ms")
        if elapsed_ms > 100:
            logger.warning(f"Slow query: fetch_candles took {elapsed_ms:.2f}ms")
```

**Requirements**:
1. Log execution time in debug mode
2. Warn on queries exceeding threshold (100ms default)
3. Include query identifier for troubleshooting

---

## 24. Input Validation Standards (Phase 1 Review)

### Rule: Validate Inputs at Function Entry

**Severity**: High
**Rationale**: Fail fast with clear error messages, prevent invalid calculations

**Pattern**:
```python
def calculate_ema(self, closes: list, period: int) -> np.ndarray:
    """Calculate Exponential Moving Average."""
    if not closes:
        raise ValueError("Input data cannot be empty")
    if period <= 0:
        raise ValueError("Period must be positive")
    if len(closes) < period:
        logger.warning(f"Insufficient data: {len(closes)} < {period}")
        return np.full(len(closes), np.nan)

    # ... calculation
```

**Required Checks**:
- Empty/None inputs
- Positive periods/limits
- Sufficient data length
- Valid ranges (0-100 for RSI, etc.)

---

## 25. Symbol Validation Standards

### Rule: Validate and Normalize Trading Symbols

**Severity**: Medium
**Rationale**: Prevents injection, ensures consistency

**Pattern**:
```python
import re

def validate_symbol(symbol: str) -> str:
    """Validate and normalize trading pair symbol."""
    # Remove slash if present
    normalized = symbol.replace('/', '').upper()

    # Validate format (6-12 alphanumeric characters)
    if not re.match(r'^[A-Z0-9]{6,12}$', normalized):
        raise ValueError(f"Invalid symbol format: {symbol}")

    return normalized
```

**Applies to**: All functions accepting symbol parameters

---

---

## 26. LLM Response Validation Standards (LLM Integration Review)

### Rule: All LLM Responses Must Be Validated Before Use

**Severity**: Critical
**Rationale**: LLMs are unreliable and may return malformed, incomplete, or malicious data

**Bad**:
```python
response = await llm_client.generate(...)
action = json.loads(response.text)['action']  # May fail!
```

**Good**:
```python
response = await llm_client.generate(...)
validated = response_parser.parse_and_validate(
    response.text,
    schema=self.get_output_schema(),
    strict=True
)
action = validated['action']  # Guaranteed to exist
```

**Required Components**:
1. JSON extraction (handles markdown code blocks, extra text)
2. Schema validation (using jsonschema library)
3. Default values for optional fields
4. Clear error messages on validation failures
5. Fallback output generation on parse errors

**Example ResponseParser**:
```python
class ResponseParser:
    def parse_and_validate(self, response_text: str, schema: dict) -> dict:
        # 1. Extract JSON from markdown/text
        json_str = self._extract_json(response_text)

        # 2. Parse JSON
        try:
            data = json.loads(json_str)
        except JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in LLM response: {e}")

        # 3. Validate against schema
        from jsonschema import validate, ValidationError
        validate(instance=data, schema=schema)

        return data
```

---

## 27. Token Counting Standards (LLM Integration Review)

### Rule: Use Actual Tokenizers, Not Character Estimates

**Severity**: Medium
**Rationale**: Character-based estimates can be off by 20-50%, causing budget overruns

**Bad**:
```python
def estimate_tokens(text: str) -> int:
    return len(text) // 3  # Crude estimate
```

**Good**:
```python
import tiktoken

def estimate_tokens(text: str, model: str = "gpt-4") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception:
        # Fallback only if tiktoken unavailable
        return int(len(text) / 3.5 * 1.1)
```

**Applies to**: Prompt budget management, cost estimation, truncation logic

---

## 28. Cost Calculation Standards (LLM Integration Review)

### Rule: Use Correct Pricing Units (per 1M tokens, not 1K)

**Severity**: Critical
**Rationale**: Incorrect divisors cause cost tracking to be wrong by 1000x

**Common Bug**:
```python
# WRONG - Model costs are per 1M tokens
cost = (tokens / 1000) * price_per_1k  # Off by 1000x!

# CORRECT
cost = (tokens / 1_000_000) * price_per_1m
```

**Pattern**:
```python
# Define pricing clearly
MODEL_PRICING = {
    'gpt-4-turbo': {'input': 10.00, 'output': 30.00},  # Per 1M tokens
    'gpt-4o': {'input': 2.50, 'output': 10.00},
}

# Calculate cost
def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = MODEL_PRICING.get(model, {'input': 0, 'output': 0})
    cost = (
        (input_tokens / 1_000_000) * pricing['input'] +
        (output_tokens / 1_000_000) * pricing['output']
    )
    return cost
```

---

## 29. Error Logging Security Standards (LLM Integration Review)

### Rule: Never Log Raw Errors That May Contain Secrets

**Severity**: High
**Rationale**: API keys, tokens, and sensitive data can leak into logs via error messages

**Bad**:
```python
try:
    response = await session.post(url, headers={'Authorization': f'Bearer {api_key}'})
except Exception as e:
    logger.error(f"Request failed: {e}")  # May log API key!
```

**Good**:
```python
try:
    response = await session.post(url, headers={'Authorization': f'Bearer {api_key}'})
except Exception as e:
    # Log only safe, structured data
    logger.error(
        "API request failed",
        extra={
            'provider': self.provider_name,
            'error_type': type(e).__name__,
            'status': getattr(e, 'status', None),
        },
        exc_info=False  # Don't log full traceback
    )
    raise RuntimeError(f"{self.provider_name} request failed") from None
```

**Required Practices**:
1. Use structured logging with explicit fields
2. Never log full exception objects that may contain secrets
3. Sanitize error messages before logging
4. Use `exc_info=False` to prevent traceback logging
5. Redact API keys in logs (show only first 8 chars)

---

## 30. Distributed Rate Limiting Standards (LLM Integration Review)

### Rule: Rate Limiters Must Work Across Processes

**Severity**: High
**Rationale**: Single-process rate limiters fail in production multi-process deployments

**Bad** (Only works in single process):
```python
class RateLimiter:
    def __init__(self):
        self._requests = []  # Process-local state
        self._lock = asyncio.Lock()  # Only works in one event loop
```

**Good** (Works across processes):
```python
import redis.asyncio as redis

class DistributedRateLimiter:
    def __init__(self, redis_url: str, key_prefix: str, max_rpm: int):
        self.redis = redis.from_url(redis_url)
        self.key_prefix = key_prefix
        self.max_rpm = max_rpm

    async def acquire(self) -> float:
        """Acquire rate limit slot using Redis."""
        key = f"{self.key_prefix}:requests"
        now = time.time()
        window_start = now - 60

        # Use Redis sorted set for sliding window
        pipe = self.redis.pipeline()
        # Remove old entries
        pipe.zremrangebyscore(key, 0, window_start)
        # Count current requests
        pipe.zcard(key)
        # Add this request
        pipe.zadd(key, {str(now): now})
        pipe.expire(key, 60)
        results = await pipe.execute()

        count = results[1]
        if count >= self.max_rpm:
            # Wait until oldest expires
            oldest = await self.redis.zrange(key, 0, 0, withscores=True)
            wait_time = (oldest[0][1] + 60) - now + 0.01
            await asyncio.sleep(wait_time)
            return wait_time

        return 0
```

**Alternative**: Use central rate limiting service with HTTP API

---

## 31. Atomic State Transitions (Execution Layer Review)

### Rule: Add to New Collection Before Removing from Old

**Severity**: Critical
**Rationale**: Prevents temporary object invisibility that causes race conditions

**Bad**:
```python
# Remove from open orders
async with self._lock:
    del self._open_orders[order.id]

# Add to history (GAP - order not findable!)
async with self._history_lock:
    self._order_history.append(order)
```

**Good**:
```python
async with self._lock:
    # Add to history first
    self._order_history.append(order)
    if len(self._order_history) > self._max_history_size:
        self._order_history = self._order_history[-self._max_history_size:]
    # Then remove from open orders
    if order.id in self._open_orders:
        del self._open_orders[order.id]
```

**Applies to**: Any state transitions between collections, order/position status changes

---

## 32. Position State Validation (Execution Layer Review)

### Rule: Always Verify Object State Before Operations

**Severity**: High
**Rationale**: Prevents double-processing, data corruption, and duplicate transactions

**Pattern**:
```python
async def close_position(self, position_id: str, exit_price: Decimal) -> Optional[Position]:
    """Close position with idempotency check."""
    async with self._lock:
        position = self._positions.get(position_id)

        if not position:
            logger.warning(f"Position not found: {position_id}")
            return None

        if position.status != PositionStatus.OPEN:
            # Return the actual closed position, not stale reference
            for closed_pos in self._closed_positions:
                if closed_pos.id == position_id:
                    return closed_pos
            return position

        # ... proceed with closing logic
```

**Required Checks**:
1. Object exists
2. Object is in expected state for operation
3. Return correct version of object (not stale reference)
4. Log state violations for debugging

---

## 33. Fail-Safe Default Patterns (Execution Layer Review)

### Rule: Safety Checks Must Fail-Safe on Component Unavailability

**Severity**: Critical
**Rationale**: Missing dependencies should cause rejection, not bypass of safety checks

**Bad**:
```python
async def _check_position_limits(self, symbol: str) -> dict:
    if not self.position_tracker:
        return {"allowed": True, "reason": None}  # DANGEROUS: Always allows
```

**Good**:
```python
async def _check_position_limits(self, symbol: str) -> dict:
    if not self.position_tracker:
        return {
            "allowed": False,
            "reason": "Position tracker unavailable - safety check failed"
        }
```

**Applies to**: All risk limits, position limits, margin checks, balance validations

---

## 34. Lock Granularity (Execution Layer Review)

### Rule: Minimize Lock Hold Time by Preparing Data Outside Locks

**Severity**: Medium
**Rationale**: Reduces lock contention, improves concurrency, prevents deadlocks

**Bad**:
```python
async with self._lock:  # Lock held during entire calculation
    for position in self._positions.values():
        current_price = self._price_cache.get(position.symbol)
        pnl, pnl_pct = position.calculate_pnl(current_price)  # Expensive
        position.unrealized_pnl = pnl
        position.unrealized_pnl_pct = pnl_pct
```

**Good**:
```python
# Snapshot positions outside lock
async with self._lock:
    positions_snapshot = list(self._positions.values())

# Calculate P&L without lock (allows concurrent operations)
updates = []
for position in positions_snapshot:
    price = self._price_cache.get(position.symbol)
    if price:
        pnl, pnl_pct = position.calculate_pnl(price)
        updates.append((position.id, pnl, pnl_pct))

# Quick update with lock
async with self._lock:
    for pos_id, pnl, pnl_pct in updates:
        if pos_id in self._positions:
            pos = self._positions[pos_id]
            pos.unrealized_pnl = pnl
            pos.unrealized_pnl_pct = pnl_pct
```

**Techniques**:
1. Snapshot data structure under lock
2. Process data without lock
3. Acquire lock again for quick update
4. Verify object still exists before updating

---

## 35. Configuration Enforcement (Execution Layer Review)

### Rule: All Configured Constraints Must Be Validated

**Severity**: High
**Rationale**: Prevents silent failures where config exists but isn't used

**Pattern**:
```python
# Config defines constraints
symbols:
  BTC/USDT:
    min_order_size: 0.0001
    max_order_size: 100
    price_decimals: 2

# Code MUST enforce these
def validate_order_size(self, symbol: str, size: Decimal) -> None:
    """Validate order size against configured limits."""
    config = self.config.get('symbols', {}).get(symbol, {})
    min_size = config.get('min_order_size')
    max_size = config.get('max_order_size')

    if min_size and size < min_size:
        raise ValueError(f"Order size {size} below minimum {min_size} for {symbol}")

    if max_size and size > max_size:
        raise ValueError(f"Order size {size} exceeds maximum {max_size} for {symbol}")
```

**Validation Points**:
- Startup: Verify all required config keys exist
- Runtime: Check values against configured limits
- Tests: Verify constraints are actually enforced

---

## 36. Collection Size Management (Execution Layer Review)

### Rule: All Unbounded Collections Must Have Size Limits

**Severity**: Medium
**Rationale**: Prevents memory exhaustion from unbounded growth

**Bad**:
```python
self._order_history: list[Order] = []  # Grows forever

async def _monitor_order(...):
    # ...
    self._order_history.append(order)  # No limit check
```

**Good**:
```python
self._order_history: list[Order] = []
self._max_history_size = config.get('max_history_size', 100)  # Reasonable default

async def _monitor_order(...):
    # ...
    async with self._history_lock:
        self._order_history.append(order)
        # Enforce size limit
        if len(self._order_history) > self._max_history_size:
            self._order_history = self._order_history[-self._max_history_size:]
```

**Requirements**:
1. Define maximum size in config (with sensible default)
2. Enforce limit when adding items
3. Use LRU or time-based eviction for caches
4. Log when trimming occurs (at debug level)
5. Consider persisting to database before evicting

---

## 37. Idempotent Operations (Execution Layer Review)

### Rule: State-Changing Operations Must Be Idempotent

**Severity**: High
**Rationale**: Prevents duplicate transactions from retries, race conditions, or bugs

**Pattern**:
```python
async def close_position(self, position_id: str, exit_price: Decimal) -> Optional[Position]:
    """Close position (idempotent - safe to call multiple times)."""
    async with self._lock:
        position = self._positions.get(position_id)

        if not position:
            # Already closed or never existed
            for closed in self._closed_positions:
                if closed.id == position_id:
                    return closed  # Return existing result
            return None

        if position.status != PositionStatus.OPEN:
            # Already closed in this call
            return position

        # Proceed with closing...
```

**Idempotent Design Checklist**:
- [ ] Check current state before applying changes
- [ ] Return existing result if operation already completed
- [ ] Log but don't error on duplicate attempts
- [ ] Track operation IDs for deduplication
- [ ] Use database transactions for atomicity

---

## 38. Partial Result Handling (Execution Layer Review)

### Rule: Handle Partial Fills and Incomplete Operations

**Severity**: High
**Rationale**: Real exchanges return partial results; code must handle them correctly

**Bad**:
```python
if kraken_status == "closed":
    # Assumes order is fully filled
    position_size = order.size  # Wrong! Could be partial fill
```

**Good**:
```python
if kraken_status == "closed" or kraken_status == "partially_filled":
    filled_size = Decimal(str(order_info.get("vol_exec", 0)))
    if filled_size < order.size:
        logger.warning(
            f"Partial fill: {filled_size}/{order.size} for order {order.id}"
        )

    # Create position with actual filled size
    position_size = filled_size

    # Place contingent orders for filled amount, not original
    if proposal.stop_loss:
        await self._place_stop_loss(order, proposal, filled_size)
```

**Scenarios to Handle**:
1. Partial order fills (common on exchanges)
2. Timeout with some results received
3. Network errors after partial execution
4. User cancellation of partially filled orders

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-12-19 | Initial version based on Phase 3 review findings |
| 1.1 | 2025-12-19 | Added Phase 1 review standards (async/await, validation, monitoring) |
| 1.2 | 2025-12-19 | Added LLM Integration standards (response validation, token counting, security) |
| 1.3 | 2025-12-19 | Added Execution Layer standards (state management, atomicity, idempotency) |

---

**Approved by**: Code Review Agent
**Last Updated**: 2025-12-19 (Execution Layer Review)
**Next Review**: After Phase 4 completion

