# Critical Fixes Checklist - Agent Implementation

**Status**: 🔴 BLOCKING ISSUES - Must Fix Before Production
**Estimated Time**: 2.5 hours
**Priority**: HIGH

---

## High-Priority Fixes (6 Issues)

### ✅ Fix Tracker

- [ ] **Issue 1**: SQL Injection Risk
- [ ] **Issue 2**: DCA Rounding Error
- [ ] **Issue 3**: Task Cancellation Leak
- [ ] **Issue 4**: Thread-Unsafe Stats
- [ ] **Issue 5**: Data Integrity Masking
- [ ] **Issue 6**: Production Fallback Handling

---

## Issue 1: SQL Injection Risk

**File**: `triplegain/src/agents/base_agent.py`
**Line**: 254
**Time**: 15 minutes
**Severity**: HIGH - Security Vulnerability

### Current Code
```python
query = """
    SELECT output_data, timestamp
    FROM agent_outputs
    WHERE agent_name = $1 AND symbol = $2
    AND timestamp > NOW() - INTERVAL '%s seconds'
    ORDER BY timestamp DESC
    LIMIT 1
""" % max_age_seconds
```

### Fixed Code
```python
query = """
    SELECT output_data, timestamp
    FROM agent_outputs
    WHERE agent_name = $1 AND symbol = $2
    AND timestamp > NOW() - $3 * INTERVAL '1 second'
    ORDER BY timestamp DESC
    LIMIT 1
"""
await self.db.fetchrow(query, self.agent_name, symbol, max_age_seconds)
```

### Verification
```bash
pytest triplegain/tests/unit/agents/test_base_agent.py::TestBaseAgent::test_get_latest_output
```

---

## Issue 2: DCA Rounding Error

**File**: `triplegain/src/agents/portfolio_rebalance.py`
**Lines**: 495-510
**Time**: 30 minutes
**Severity**: HIGH - Financial Calculation Error

### Current Code
```python
rounded_batch_amount = base_batch_amount.quantize(Decimal('0.01'))
remainder = trade.amount_usd - (rounded_batch_amount * num_batches)
```

### Problem
With $1000 / 6 batches:
- Base: $166.666...
- Rounded: $166.67 × 6 = $1000.02 ❌
- Remainder: -$0.02 (negative!)

### Fixed Code
```python
from decimal import ROUND_DOWN

# Round DOWN to ensure we never exceed original amount
rounded_batch_amount = base_batch_amount.quantize(
    Decimal('0.01'),
    rounding=ROUND_DOWN
)

# Calculate remainder (should always be >= 0)
remainder = trade.amount_usd - (rounded_batch_amount * num_batches)

# Safety check
if remainder < 0:
    logger.error(f"DCA rounding error: remainder={remainder}, trade={trade.amount_usd}")
    raise ValueError(f"DCA rounding produced negative remainder: {remainder}")

# Add remainder to first batch
batch_amount = rounded_batch_amount
if batch_idx == 0:
    batch_amount += remainder
```

### Verification
```python
# Add to test_portfolio_rebalance.py
def test_dca_rounding_edge_case():
    """Test DCA doesn't produce negative remainders."""
    trade = RebalanceTrade(
        symbol="BTC/USDT",
        action="buy",
        amount_usd=Decimal("1000.00"),
    )

    dca_trades, num_batches, _ = agent._create_dca_batches([trade])

    # Verify total equals original
    total = sum(t.amount_usd for t in dca_trades)
    assert total == Decimal("1000.00"), f"DCA total {total} != 1000.00"

    # Verify all amounts positive
    for t in dca_trades:
        assert t.amount_usd > 0, f"Negative DCA amount: {t.amount_usd}"
```

---

## Issue 3: Task Cancellation Leak

**File**: `triplegain/src/agents/trading_decision.py`
**Lines**: 399-414
**Time**: 30 minutes
**Severity**: HIGH - Resource Leak

### Current Code
```python
for task in pending:
    model_name = tasks[task]
    logger.warning(f"Model {model_name} timed out after {self.timeout_seconds}s")
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass  # Silently ignored, task may still be running
```

### Fixed Code
```python
for task in pending:
    model_name = tasks[task]
    logger.warning(f"Model {model_name} timed out after {self.timeout_seconds}s")

    # Cancel and wait for cleanup
    task.cancel()
    try:
        # Give task 1 second to clean up
        await asyncio.wait_for(task, timeout=1.0)
    except asyncio.CancelledError:
        logger.debug(f"Model {model_name} cancelled successfully")
    except asyncio.TimeoutError:
        logger.error(f"Model {model_name} did not respond to cancellation")
    except Exception as e:
        logger.error(f"Error cancelling model {model_name}: {e}")
```

### Verification
```python
# Add to test_trading_decision.py
@pytest.mark.asyncio
async def test_model_timeout_cleanup():
    """Test that timed-out models are properly cancelled."""
    # Create agent with 1 second timeout
    agent = TradingDecisionAgent(..., config={'timeout_seconds': 1})

    # Mock a slow model
    async def slow_model(*args, **kwargs):
        await asyncio.sleep(10)  # Much longer than timeout
        return MagicMock(text='{"action": "BUY"}')

    agent.llm_clients['slow_model'].generate = slow_model

    # Process should complete in ~1 second, not 10
    start = time.time()
    output = await agent.process(mock_snapshot)
    duration = time.time() - start

    assert duration < 2.5, f"Timeout did not work: {duration}s"
```

---

## Issue 4: Thread-Unsafe Stats Tracking

**File**: All agents (`base_agent.py`, etc.)
**Lines**: Various (331-336, etc.)
**Time**: 45 minutes
**Severity**: HIGH - Race Condition

### Current Code
```python
# Update stats
self._total_invocations += 1
self._total_latency_ms += latency_ms
self._total_tokens += tokens_used
```

### Fixed Code
```python
# In __init__
self._stats_lock = asyncio.Lock()

# In _call_llm
async def _call_llm(self, system_prompt: str, user_message: str) -> tuple[str, int, int]:
    start_time = time.perf_counter()

    try:
        response = await self.llm.generate(
            model=self.model,
            system_prompt=system_prompt,
            user_message=user_message,
        )

        latency_ms = int((time.perf_counter() - start_time) * 1000)
        tokens_used = response.tokens_used

        # Thread-safe stats update
        async with self._stats_lock:
            self._total_invocations += 1
            self._total_latency_ms += latency_ms
            self._total_tokens += tokens_used

        return response.text, latency_ms, tokens_used

    except Exception as e:
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        logger.error(f"LLM call failed for {self.agent_name}: {e}")
        raise

# In get_stats
def get_stats(self) -> dict:
    """Get agent performance statistics (thread-safe)."""
    # No lock needed for read-only access to immutable values
    # But we can add it for consistency
    return {
        "agent_name": self.agent_name,
        "total_invocations": self._total_invocations,
        "total_latency_ms": self._total_latency_ms,
        "average_latency_ms": (
            self._total_latency_ms / self._total_invocations
            if self._total_invocations > 0 else 0
        ),
        "total_tokens": self._total_tokens,
        "average_tokens": (
            self._total_tokens / self._total_invocations
            if self._total_invocations > 0 else 0
        ),
    }
```

### Verification
```python
# Add to test_base_agent.py
@pytest.mark.asyncio
async def test_concurrent_stats_update():
    """Test stats tracking under concurrent updates."""
    agent = create_test_agent()

    # Simulate 100 concurrent calls
    tasks = []
    for i in range(100):
        async def mock_call():
            async with agent._stats_lock:
                agent._total_invocations += 1
        tasks.append(mock_call())

    await asyncio.gather(*tasks)

    # Should be exactly 100
    assert agent._total_invocations == 100
```

---

## Issue 5: Data Integrity Masking

**File**: `triplegain/src/agents/portfolio_rebalance.py`
**Lines**: 345-362
**Time**: 30 minutes
**Severity**: HIGH - Silent Data Corruption

### Current Code
```python
if available_btc < 0:
    logger.warning(
        f"BTC hodl bag ({float(hodl_bags.get('BTC', 0))}) exceeds balance "
        f"({float(balances.get('BTC', 0))}), clamping to 0"
    )
    available_btc = Decimal(0)
```

### Fixed Code
```python
async def _send_critical_alert(self, message: str, details: dict = None):
    """Send critical alert to monitoring system."""
    logger.critical(f"CRITICAL ALERT: {message}")

    if details:
        logger.critical(f"Details: {json.dumps(details, indent=2)}")

    # Store in database for tracking
    if self.db:
        try:
            await self.db.execute(
                """
                INSERT INTO system_alerts (timestamp, severity, message, details)
                VALUES ($1, $2, $3, $4)
                """,
                datetime.now(timezone.utc),
                'CRITICAL',
                message,
                json.dumps(details) if details else None
            )
        except Exception as e:
            logger.error(f"Failed to store alert: {e}")

# In check_allocation
if available_btc < 0:
    await self._send_critical_alert(
        "Hodl bag exceeds balance - data integrity issue",
        {
            'asset': 'BTC',
            'balance': float(balances.get('BTC', 0)),
            'hodl_bag': float(hodl_bags.get('BTC', 0)),
            'difference': float(available_btc),
        }
    )
    # Still clamp to prevent negative trades, but alert is raised
    available_btc = Decimal(0)
```

### Database Migration
```sql
-- Add to migrations/004_system_alerts.sql
CREATE TABLE IF NOT EXISTS system_alerts (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    severity VARCHAR(20) NOT NULL,
    message TEXT NOT NULL,
    details JSONB,
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_at TIMESTAMPTZ,
    acknowledged_by VARCHAR(100)
);

CREATE INDEX idx_system_alerts_timestamp ON system_alerts(timestamp DESC);
CREATE INDEX idx_system_alerts_severity ON system_alerts(severity);
```

### Verification
```python
# Add to test_portfolio_rebalance.py
@pytest.mark.asyncio
async def test_hodl_bag_exceeds_balance_alert(agent, mock_db):
    """Test that exceeding hodl bag triggers alert."""
    # Mock balances: 0.5 BTC balance, 1.0 BTC hodl bag
    agent._get_balances = AsyncMock(return_value={'BTC': 0.5, 'XRP': 1000, 'USDT': 1000})
    agent._get_hodl_bags = AsyncMock(return_value={'BTC': Decimal('1.0')})
    agent._send_critical_alert = AsyncMock()

    allocation = await agent.check_allocation()

    # Should have sent alert
    agent._send_critical_alert.assert_called_once()
    alert_message = agent._send_critical_alert.call_args[0][0]
    assert "Hodl bag exceeds balance" in alert_message
```

---

## Issue 6: Production Fallback Handling

**File**: `triplegain/src/agents/portfolio_rebalance.py`
**Lines**: 605, 634
**Time**: 15 minutes
**Severity**: HIGH - Wrong Data in Production

### Current Code
```python
except Exception as e:
    logger.warning(f"Failed to get Kraken balances: {e}")

# Fallback to config or mock data
balances = self.config.get('mock_balances', {})
```

### Fixed Code
```python
async def _get_balances(self) -> dict[str, float]:
    """Get current balances from Kraken or config."""
    if self.kraken:
        try:
            result = await self.kraken.get_balances()
            return {
                'BTC': float(result.get('XXBT', 0)),
                'XRP': float(result.get('XXRP', 0)),
                'USDT': float(result.get('USDT', 0)),
            }
        except Exception as e:
            # In production, fail fast
            if self.config.get('environment') == 'production':
                logger.error(f"Failed to get balances in production: {e}")
                raise RuntimeError(
                    "Cannot get live balances in production mode. "
                    "Trading disabled for safety."
                ) from e

            # In development/testing, warn and use mock
            logger.warning(f"Failed to get Kraken balances, using mock: {e}")

    # Fallback to config or mock data (only in dev/test)
    balances = self.config.get('mock_balances', {})

    if not balances:
        raise ValueError("No balances available (no Kraken client and no mock data)")

    return {
        'BTC': float(balances.get('BTC', 0)),
        'XRP': float(balances.get('XRP', 0)),
        'USDT': float(balances.get('USDT', 0)),
    }
```

### Configuration
```yaml
# config/portfolio.yaml
environment: production  # or 'development', 'testing'

# Mock balances (only used in dev/test)
mock_balances:
  BTC: 0.5
  XRP: 10000
  USDT: 5000

# Mock prices (only used in dev/test)
mock_prices:
  BTC/USDT: 45000
  XRP/USDT: 0.60
```

### Verification
```python
# Add to test_portfolio_rebalance.py
@pytest.mark.asyncio
async def test_production_fails_without_kraken():
    """Test that production mode fails without live data."""
    config = {
        'environment': 'production',
        'mock_balances': {'BTC': 1.0},
        'target_allocation': {...},
        'rebalancing': {...},
    }

    # No Kraken client
    agent = PortfolioRebalanceAgent(
        llm_client=mock_llm,
        prompt_builder=mock_builder,
        config=config,
        kraken_client=None,
        db_pool=None,
    )

    # Should raise error, not use mock
    with pytest.raises(RuntimeError, match="Cannot get live balances"):
        await agent._get_balances()

@pytest.mark.asyncio
async def test_development_allows_mock():
    """Test that development mode allows mock data."""
    config = {
        'environment': 'development',
        'mock_balances': {'BTC': 1.0, 'XRP': 1000, 'USDT': 1000},
        ...
    }

    agent = PortfolioRebalanceAgent(..., kraken_client=None, ...)

    # Should succeed with mock
    balances = await agent._get_balances()
    assert balances['BTC'] == 1.0
```

---

## Final Verification Checklist

After implementing all fixes:

```bash
# 1. Run unit tests
pytest triplegain/tests/unit/agents/ -v

# 2. Run integration tests (if available)
pytest triplegain/tests/integration/ -v

# 3. Check test coverage
pytest --cov=triplegain/src/agents --cov-report=term

# 4. Run type checking
mypy triplegain/src/agents/

# 5. Run linting
ruff check triplegain/src/agents/

# 6. Manual verification
# - Start system in development mode
# - Verify mock fallbacks work
# - Switch to production mode
# - Verify production fails without live data
# - Test DCA with edge amounts ($999.99, $1000.01)
# - Test concurrent agent execution
```

---

## Sign-Off

- [ ] All 6 issues fixed
- [ ] All tests passing
- [ ] Coverage maintained (>82%)
- [ ] Type checking passing
- [ ] Linting clean
- [ ] Manual verification complete
- [ ] Documentation updated
- [ ] Configuration reviewed
- [ ] Ready for production deployment

**Developer**: ________________
**Date**: ________________
**Review Sign-Off**: ________________

---

**Total Time Investment**: 2.5 hours
**Impact**: Prevents security, financial, and data integrity issues
**Priority**: BLOCKING - Must complete before production

---

**Document Version**: 1.0
**Last Updated**: 2025-12-19
