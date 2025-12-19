# Test Improvement Action Plan

**Generated**: 2025-12-19
**Priority**: HIGH - Production Blockers Identified
**Estimated Effort**: 96 hours (12 days)

## Executive Summary

The test suite review identified **5 critical production risks** that must be addressed before deployment. This document provides concrete test implementations to close these gaps.

---

## Critical Risk #1: Order Execution Partial Failures

### Risk Assessment
**Severity**: CRITICAL
**Likelihood**: HIGH
**Impact**: Catastrophic loss potential (unprotected positions)

### Missing Test Coverage
```python
# File: triplegain/tests/unit/execution/test_order_manager_failures.py

import pytest
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

class TestOrderExecutionPartialFailures:
    """Test scenarios where order execution partially succeeds."""

    @pytest.mark.asyncio
    async def test_entry_fills_but_stop_loss_fails(self):
        """
        CRITICAL: Entry order fills successfully, but stop-loss placement fails.
        System must:
        1. Detect the failure
        2. Log critical alert
        3. Retry stop-loss placement
        4. If retry fails, close position immediately
        """
        mock_kraken = AsyncMock()

        # Entry order succeeds
        mock_kraken.add_order.side_effect = [
            {"result": {"txid": ["ORDER-ENTRY"]}},  # Entry succeeds
            Exception("Stop-loss placement failed"),  # Stop fails
        ]

        order_manager = OrderExecutionManager(kraken_client=mock_kraken, ...)

        result = await order_manager.execute_trade(proposal)

        # Should report partial success with critical alert
        assert result.success is False  # Overall failure
        assert result.partial_fill is True
        assert result.alert_level == "CRITICAL"
        assert "stop-loss failed" in result.error_message.lower()

        # Should have attempted to close position
        assert mock_kraken.add_order.call_count == 3  # Entry, Stop (failed), Close

    @pytest.mark.asyncio
    async def test_entry_fills_stop_fails_retry_succeeds(self):
        """Test successful retry of stop-loss placement."""
        mock_kraken = AsyncMock()

        mock_kraken.add_order.side_effect = [
            {"result": {"txid": ["ORDER-ENTRY"]}},  # Entry succeeds
            Exception("Temporary failure"),  # Stop fails
            {"result": {"txid": ["ORDER-STOP"]}},  # Stop retry succeeds
        ]

        order_manager = OrderExecutionManager(kraken_client=mock_kraken, ...)

        result = await order_manager.execute_trade(proposal)

        assert result.success is True
        assert result.retry_count > 0
        assert len(result.warnings) > 0  # Should warn about retry

    @pytest.mark.asyncio
    async def test_position_recorded_but_db_write_fails(self):
        """Test database write failure after successful order placement."""
        mock_kraken = AsyncMock()
        mock_db = AsyncMock()

        mock_kraken.add_order.return_value = {"result": {"txid": ["ORDER-1"]}}
        mock_db.execute.side_effect = Exception("DB connection lost")

        order_manager = OrderExecutionManager(
            kraken_client=mock_kraken,
            db_pool=mock_db,
            ...
        )

        result = await order_manager.execute_trade(proposal)

        # Order is on exchange but not in database
        assert result.success is True
        assert result.warnings_contains("database write failed")

        # Should cache order locally for retry
        assert order_manager._pending_db_writes[0].external_id == "ORDER-1"

    @pytest.mark.asyncio
    async def test_kraken_returns_success_but_order_not_found(self):
        """Test Kraken API returning success but order doesn't appear in query."""
        mock_kraken = AsyncMock()

        mock_kraken.add_order.return_value = {"result": {"txid": ["ORDER-1"]}}
        mock_kraken.query_orders.return_value = {"result": {}}  # Order not found!

        order_manager = OrderExecutionManager(kraken_client=mock_kraken, ...)

        result = await order_manager.execute_trade(proposal)

        # Should detect inconsistency and report
        assert result.status == "UNKNOWN"
        assert result.alert_level == "HIGH"
        assert "order not found after placement" in result.error_message.lower()


class TestOrderExecutionRateLimits:
    """Test order execution under rate limit constraints."""

    @pytest.mark.asyncio
    async def test_rate_limit_hit_during_order_placement(self):
        """Test handling when rate limit is hit mid-execution."""
        mock_kraken = AsyncMock()

        # First order hits rate limit
        mock_kraken.add_order.side_effect = [
            Exception("Rate limit exceeded"),
            {"result": {"txid": ["ORDER-1"]}},  # Succeeds after wait
        ]

        order_manager = OrderExecutionManager(kraken_client=mock_kraken, ...)

        result = await order_manager.execute_trade(proposal)

        # Should wait and retry
        assert result.success is True
        assert result.latency_ms > 1000  # Should have waited
        assert "rate limit" in result.warnings[0].lower()

    @pytest.mark.asyncio
    async def test_token_bucket_prevents_burst_orders(self):
        """Test that token bucket rate limiter prevents burst orders."""
        order_manager = OrderExecutionManager(
            kraken_client=AsyncMock(),
            config={
                "kraken": {
                    "rate_limit": {
                        "calls_per_minute": 60,
                        "order_calls_per_minute": 15,  # 1 per 4 seconds
                    }
                }
            },
        )

        # Try to place 5 orders in rapid succession
        start_time = time.time()
        results = []

        for i in range(5):
            result = await order_manager.execute_trade(proposals[i])
            results.append(result)

        elapsed = time.time() - start_time

        # Should take at least 16 seconds (4 orders * 4 seconds between)
        assert elapsed >= 16.0
        assert all(r.success for r in results)
```

---

## Critical Risk #2: Orchestration Deadlocks

### Risk Assessment
**Severity**: CRITICAL
**Likelihood**: MEDIUM
**Impact**: Trading stops, requires manual restart

### Missing Test Coverage
```python
# File: triplegain/tests/integration/test_orchestration_flows.py

import asyncio
import pytest
from datetime import datetime, timezone

class TestFullTradeLifecycle:
    """Integration tests for complete trade lifecycle."""

    @pytest.mark.asyncio
    async def test_end_to_end_buy_trade_execution(self):
        """
        Test full trade flow from TA signal to position opened.

        Flow:
        1. Technical Analysis Agent analyzes BTC/USDT → LONG bias
        2. Regime Detection Agent confirms TRENDING_BULL
        3. Trading Decision Agent reaches BUY consensus
        4. Risk Management validates trade
        5. Order Execution Manager places orders
        6. Position Tracker records position
        7. Stop-loss and take-profit orders placed
        """
        # Use REAL agents (not mocks) with mock LLM clients
        ta_agent = TechnicalAnalysisAgent(...)
        regime_agent = RegimeDetectionAgent(...)
        trading_agent = TradingDecisionAgent(...)
        risk_engine = RiskManagementEngine(...)
        order_manager = OrderExecutionManager(...)
        position_tracker = PositionTracker(...)

        # Set up mock LLM responses
        mock_llm.generate.side_effect = [
            # TA response
            MagicMock(text='{"trend": "up", "bias": "long", "confidence": 0.8}'),
            # Regime response
            MagicMock(text='{"regime": "trending_bull", "confidence": 0.9}'),
            # Trading decision responses (6 models)
            *[MagicMock(text='{"action": "BUY", "entry": 45000, "stop": 44000}') for _ in range(6)],
        ]

        # Execute full flow
        snapshot = await build_market_snapshot("BTC/USDT")

        ta_output = await ta_agent.process(snapshot)
        assert ta_output.bias == "long"

        regime_output = await regime_agent.process(snapshot)
        assert regime_output.regime == "trending_bull"

        decision_output = await trading_agent.process(snapshot, ta_output, regime_output)
        assert decision_output.action == "BUY"

        proposal = create_trade_proposal(decision_output)
        validation = risk_engine.validate_trade(proposal, risk_state)
        assert validation.is_approved()

        result = await order_manager.execute_trade(proposal)
        assert result.success is True
        assert result.order.external_id is not None

        position = await position_tracker.add_position(result.order)
        assert position.status == "open"
        assert position.stop_loss is not None
        assert position.take_profit is not None

    @pytest.mark.asyncio
    async def test_concurrent_agent_execution_same_symbol(self):
        """
        Test multiple agents processing same symbol simultaneously.
        Should NOT cause deadlocks, race conditions, or double-trading.
        """
        coordinator = CoordinatorAgent(...)

        # Start coordinator
        await coordinator.start()

        # Trigger multiple scheduled tasks simultaneously
        tasks = [
            coordinator._run_scheduled_task("technical_analysis", "BTC/USDT"),
            coordinator._run_scheduled_task("regime_detection", "BTC/USDT"),
            coordinator._run_scheduled_task("trading_decision", "BTC/USDT"),
        ]

        # All should complete without deadlock
        results = await asyncio.gather(*tasks, timeout=10.0)

        assert len(results) == 3
        assert all(r is not None for r in results)

        # Check for message ordering consistency
        messages = await coordinator._message_bus.get_history()
        assert_messages_properly_ordered(messages)

    @pytest.mark.asyncio
    async def test_coordinator_conflict_resolution_with_llm(self):
        """
        Test coordinator resolving conflicts using LLM.

        Scenario:
        - TA Agent says LONG
        - Regime Agent says HIGH_VOLATILITY (suggests caution)
        - Trading Decision has split vote (3 BUY, 3 HOLD)
        - Coordinator invokes LLM to resolve
        """
        coordinator = CoordinatorAgent(
            llm_client=mock_deepseek,  # Real DeepSeek client (mocked API)
            config={
                "conflicts": {
                    "invoke_llm_threshold": 0.2,  # Low threshold to trigger
                }
            },
        )

        # Create conflicting signals
        ta_output = AgentOutput(bias="long", confidence=0.75, ...)
        regime_output = AgentOutput(regime="high_volatility", confidence=0.85, ...)
        decision_output = TradingDecisionOutput(
            action="BUY",
            consensus_strength=0.50,  # Split vote
            model_decisions=[
                ModelDecision(action="BUY", confidence=0.8),
                ModelDecision(action="BUY", confidence=0.7),
                ModelDecision(action="BUY", confidence=0.6),
                ModelDecision(action="HOLD", confidence=0.7),
                ModelDecision(action="HOLD", confidence=0.75),
                ModelDecision(action="HOLD", confidence=0.8),
            ],
        )

        # Mock LLM resolution
        mock_deepseek.generate.return_value = MagicMock(
            text='{"action": "modify", "reasoning": "Reduce position size by 50% due to high volatility", "confidence": 0.85}'
        )

        # Process conflict
        resolution = await coordinator._resolve_conflict(
            ConflictInfo(
                conflict_type="low_consensus",
                ta_output=ta_output,
                regime_output=regime_output,
                decision_output=decision_output,
            )
        )

        assert resolution.action == "modify"
        assert resolution.confidence > 0.7
        assert "reduce position" in resolution.reasoning.lower()

        # Verify conflict logged to database
        conflict_record = await db_pool.fetchrow(
            "SELECT * FROM coordinator_conflicts WHERE symbol = $1 ORDER BY timestamp DESC LIMIT 1",
            "BTC/USDT"
        )
        assert conflict_record is not None
        assert conflict_record["resolution_action"] == "modify"


class TestMessageBusReliability:
    """Integration tests for message bus under load."""

    @pytest.mark.asyncio
    async def test_message_bus_1000_messages_no_loss(self):
        """Test message bus handles 1000 messages without loss."""
        message_bus = MessageBus(config={"max_history_size": 2000})

        received_counts = {f"subscriber-{i}": 0 for i in range(10)}

        async def handler(subscriber_id):
            async def _handler(msg):
                received_counts[subscriber_id] += 1
            return _handler

        # Subscribe 10 subscribers
        for i in range(10):
            await message_bus.subscribe(
                subscriber_id=f"subscriber-{i}",
                topic=MessageTopic.TA_SIGNALS,
                handler=await handler(f"subscriber-{i}"),
            )

        # Publish 1000 messages
        for i in range(1000):
            msg = create_message(
                topic=MessageTopic.TA_SIGNALS,
                source="test",
                payload={"index": i},
            )
            await message_bus.publish(msg)

        # Wait for all handlers to complete
        await asyncio.sleep(1.0)

        # Verify all subscribers received all messages
        for subscriber_id, count in received_counts.items():
            assert count == 1000, f"{subscriber_id} received {count}/1000"

    @pytest.mark.asyncio
    async def test_message_bus_subscriber_crash_doesnt_block(self):
        """Test that one crashing subscriber doesn't block others."""
        message_bus = MessageBus()

        received = {"good": 0}

        async def crashing_handler(msg):
            raise Exception("Subscriber crashed!")

        async def good_handler(msg):
            received["good"] += 1

        await message_bus.subscribe("crash", MessageTopic.TA_SIGNALS, crashing_handler)
        await message_bus.subscribe("good", MessageTopic.TA_SIGNALS, good_handler)

        # Publish 10 messages
        for i in range(10):
            await message_bus.publish(
                create_message(MessageTopic.TA_SIGNALS, "test", {})
            )

        await asyncio.sleep(0.5)

        # Good subscriber should have received all 10
        assert received["good"] == 10
```

---

## Critical Risk #3: Rate Limiter Edge Cases

### Risk Assessment
**Severity**: HIGH
**Likelihood**: MEDIUM
**Impact**: Account banned, trading halted for hours/days

### Missing Test Coverage
```python
# File: triplegain/tests/unit/execution/test_rate_limiter.py

import asyncio
import pytest
import time

class TestTokenBucketRateLimiter:
    """Comprehensive tests for TokenBucketRateLimiter."""

    @pytest.mark.asyncio
    async def test_concurrent_token_acquisition(self):
        """Test multiple coroutines acquiring tokens simultaneously."""
        limiter = TokenBucketRateLimiter(rate=10, capacity=10)

        acquired_times = []

        async def acquire_and_record():
            start = time.time()
            await limiter.acquire(tokens=1)
            acquired_times.append(time.time() - start)

        # 20 concurrent requests for 1 token each
        # Should take ~1 second (10 tokens/sec)
        tasks = [acquire_and_record() for _ in range(20)]
        await asyncio.gather(*tasks)

        # First 10 should be immediate (<0.1s)
        assert sum(1 for t in acquired_times[:10] if t < 0.1) >= 9

        # Next 10 should wait (~1 second)
        assert sum(1 for t in acquired_times[10:] if t >= 0.9) >= 8

    @pytest.mark.asyncio
    async def test_capacity_overflow_after_long_idle(self):
        """Test that tokens don't exceed capacity after long idle period."""
        limiter = TokenBucketRateLimiter(rate=10, capacity=10)

        # Wait for 10 seconds (should refill 100 tokens, but capped at 10)
        await asyncio.sleep(10)

        # Should be able to acquire 10 tokens immediately
        start = time.time()
        await limiter.acquire(tokens=10)
        elapsed = time.time() - start

        assert elapsed < 0.1  # Should be immediate

        # Next token should require wait
        start = time.time()
        await limiter.acquire(tokens=1)
        elapsed = time.time() - start

        assert elapsed >= 0.09  # Should wait ~0.1 seconds

    @pytest.mark.asyncio
    async def test_refill_accuracy_over_time(self):
        """Test that token refill is accurate over extended periods."""
        limiter = TokenBucketRateLimiter(rate=10, capacity=100)

        # Drain bucket
        await limiter.acquire(tokens=100)

        # Wait 5 seconds, should refill 50 tokens
        await asyncio.sleep(5.0)

        # Should be able to acquire 50 tokens immediately
        start = time.time()
        await limiter.acquire(tokens=50)
        elapsed = time.time() - start

        assert elapsed < 0.1

        # Verify available tokens is approximately 0
        assert limiter.available_tokens < 1.0

    @pytest.mark.asyncio
    async def test_burst_then_sustained_rate(self):
        """Test burst of requests followed by sustained rate."""
        limiter = TokenBucketRateLimiter(rate=10, capacity=20)

        # Burst: 20 requests immediately
        burst_start = time.time()
        burst_tasks = [limiter.acquire(1) for _ in range(20)]
        await asyncio.gather(*burst_tasks)
        burst_time = time.time() - burst_start

        # Burst should complete in <1 second (using capacity)
        assert burst_time < 1.0

        # Sustained: 100 requests should take ~10 seconds
        sustained_start = time.time()
        sustained_tasks = [limiter.acquire(1) for _ in range(100)]
        await asyncio.gather(*sustained_tasks)
        sustained_time = time.time() - sustained_start

        # Should take approximately 10 seconds (100 tokens / 10 per second)
        assert 9.0 <= sustained_time <= 11.0
```

---

## High Priority: Database Transaction Failures

### Test Coverage
```python
# File: triplegain/tests/integration/test_database_transactions.py

import pytest
from decimal import Decimal

class TestDatabaseTransactionFailures:
    """Test database transaction handling and rollback."""

    @pytest.mark.asyncio
    async def test_order_placement_rollback_on_db_error(self):
        """Test transaction rollback when order placement fails."""
        async with db_pool.acquire() as conn:
            async with conn.transaction():
                # Insert order
                await conn.execute(
                    "INSERT INTO orders (id, symbol, side, size) VALUES ($1, $2, $3, $4)",
                    "order-1", "BTC/USDT", "buy", Decimal("0.1")
                )

                # Simulate failure (constraint violation)
                with pytest.raises(Exception):
                    await conn.execute(
                        "INSERT INTO orders (id, symbol, side, size) VALUES ($1, $2, $3, $4)",
                        "order-1", "XRP/USDT", "sell", Decimal("100")  # Same ID - should fail
                    )

        # Verify rollback: first order should NOT exist
        order = await db_pool.fetchrow("SELECT * FROM orders WHERE id = $1", "order-1")
        assert order is None

    @pytest.mark.asyncio
    async def test_position_update_with_order_insertion(self):
        """Test atomic position update with order insertion."""
        async with db_pool.acquire() as conn:
            async with conn.transaction():
                # Insert position
                position_id = await conn.fetchval(
                    """
                    INSERT INTO positions (symbol, side, size, entry_price, leverage, status)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    RETURNING id
                    """,
                    "BTC/USDT", "long", Decimal("0.1"), Decimal("45000"), 2, "open"
                )

                # Insert stop-loss order
                await conn.execute(
                    """
                    INSERT INTO orders (symbol, side, order_type, stop_price, position_id)
                    VALUES ($1, $2, $3, $4, $5)
                    """,
                    "BTC/USDT", "sell", "stop-loss", Decimal("44000"), position_id
                )

                # Insert take-profit order
                await conn.execute(
                    """
                    INSERT INTO orders (symbol, side, order_type, price, position_id)
                    VALUES ($1, $2, $3, $4, $5)
                    """,
                    "BTC/USDT", "sell", "take-profit", Decimal("48000"), position_id
                )

        # Verify all three records exist
        position = await db_pool.fetchrow("SELECT * FROM positions WHERE id = $1", position_id)
        orders = await db_pool.fetch("SELECT * FROM orders WHERE position_id = $1", position_id)

        assert position is not None
        assert len(orders) == 2
        assert any(o["order_type"] == "stop-loss" for o in orders)
        assert any(o["order_type"] == "take-profit" for o in orders)
```

---

## Implementation Timeline

### Week 1: Critical Order Execution Tests (32 hours)
- Day 1-2: Order execution partial failures (16h)
- Day 3-4: Rate limiter edge cases (16h)

### Week 2: Orchestration Integration Tests (32 hours)
- Day 1-2: Full trade lifecycle tests (16h)
- Day 3-4: Concurrent agent execution tests (16h)

### Week 3: Message Bus & Database Tests (24 hours)
- Day 1-2: Message bus reliability tests (12h)
- Day 3: Database transaction tests (12h)

### Week 4: Performance & Stress Tests (8 hours)
- Day 1: Performance benchmarks (4h)
- Day 2: Load testing (4h)

**Total**: 96 hours (12 working days)

---

## Success Criteria

Tests are production-ready when:

1. ✅ All critical failure scenarios have test coverage
2. ✅ Integration test count reaches 50+ (currently 14)
3. ✅ Concurrent execution tests pass with 10+ concurrent agents
4. ✅ Rate limiter stress test handles 1000 requests without violation
5. ✅ Database transaction tests verify atomicity
6. ✅ Full trade lifecycle test completes end-to-end in <30 seconds
7. ✅ Message bus handles 10,000 messages without loss
8. ✅ No flaky tests (0 failures in 10 consecutive runs)

---

## Appendix: Test Template for New Features

When adding new features, use this template:

```python
# test_new_feature.py

class TestNewFeatureHappyPath:
    """Test successful execution scenarios."""

    def test_basic_functionality(self):
        """Test feature works with valid input."""
        pass

class TestNewFeatureEdgeCases:
    """Test boundary conditions and edge cases."""

    def test_boundary_values(self):
        """Test min/max values."""
        pass

    def test_empty_input(self):
        """Test behavior with empty/null input."""
        pass

    def test_invalid_input(self):
        """Test error handling for invalid input."""
        pass

class TestNewFeatureFailures:
    """Test failure scenarios and recovery."""

    def test_partial_failure(self):
        """Test handling when operation partially succeeds."""
        pass

    def test_complete_failure(self):
        """Test handling when operation completely fails."""
        pass

    def test_timeout(self):
        """Test behavior when operation times out."""
        pass

class TestNewFeatureIntegration:
    """Test integration with other components."""

    def test_integration_with_component_a(self):
        """Test interaction with ComponentA."""
        pass

    def test_concurrent_usage(self):
        """Test thread-safety and concurrent access."""
        pass
```

---

**Action Plan Complete**: Ready for implementation

**Next Steps**:
1. Review with team
2. Prioritize which tests to implement first
3. Assign test development tasks
4. Schedule daily progress reviews
