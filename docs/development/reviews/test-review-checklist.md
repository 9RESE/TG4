# Test Review Checklist - Quick Reference

**Use this checklist when reviewing or writing tests**

---

## Test Quality Checklist

### Coverage
- [ ] Happy path tested
- [ ] Error paths tested
- [ ] Edge cases tested (boundaries, empty, null)
- [ ] Concurrent execution tested (if applicable)
- [ ] Integration test exists (if component interacts with others)

### Assertions
- [ ] Tests verify behavior, not just coverage
- [ ] Assertions are specific (not just `assert result is not None`)
- [ ] Error messages are checked, not just that exception is raised
- [ ] Side effects are verified (database writes, message bus publishes)

### Mocks
- [ ] Mocks are minimal (only I/O boundaries)
- [ ] Mock responses are realistic (include malformed/unexpected data)
- [ ] Mock failures include realistic error scenarios
- [ ] Real implementations used where cheap (no I/O)

### Async Tests
- [ ] `@pytest.mark.asyncio` decorator present
- [ ] `AsyncMock` used for async methods
- [ ] Timeouts added for long-running tests
- [ ] Background tasks properly awaited or cancelled

### Fixtures
- [ ] Fixtures have appropriate scope (function/module/session)
- [ ] Async fixtures properly clean up resources
- [ ] Fixtures are documented (docstrings)
- [ ] Parametrized fixtures used for repeated patterns

---

## Production Readiness Checklist

### Critical Scenarios Tested
- [ ] Order execution partial failures
  - [ ] Entry succeeds, stop-loss fails
  - [ ] Position recorded, database write fails
  - [ ] Order placed, not found in query
- [ ] Concurrent agent execution
  - [ ] Multiple agents, same symbol
  - [ ] Race conditions in position tracking
  - [ ] Cache invalidation under load
- [ ] Rate limiter stress tested
  - [ ] Burst traffic handling
  - [ ] Concurrent token acquisition
  - [ ] Long idle period recovery
- [ ] Message bus reliability
  - [ ] 1000+ messages without loss
  - [ ] Subscriber crash doesn't block others
  - [ ] TTL cleanup under load
- [ ] Database transaction failures
  - [ ] Rollback on error
  - [ ] Deadlock detection
  - [ ] Connection pool exhaustion

### Integration Tests Present
- [ ] Full trade lifecycle (TA → Regime → Decision → Risk → Execution → Position)
- [ ] Multi-agent coordination with real agents
- [ ] Portfolio rebalance execution flow
- [ ] Message bus throughput test

### Performance Tests Present
- [ ] Latency benchmarks (risk validation <10ms)
- [ ] Throughput tests (message bus, order placement)
- [ ] Memory leak detection
- [ ] Load tests (sustained high traffic)

---

## Test Smells (Anti-Patterns)

### Red Flags
- ❌ Test always passes (mocks return success every time)
- ❌ No assertions (or only `assert True`)
- ❌ Tests database but doesn't clean up
- ❌ Sleeps for fixed time (use events/conditions instead)
- ❌ Tests implementation details (internal methods)
- ❌ Tests are order-dependent (run each test in isolation)
- ❌ Flaky tests (pass/fail randomly)
- ❌ Tests are too slow (>1s for unit tests)

### Green Flags
- ✅ Tests verify external behavior
- ✅ Tests are fast (<100ms for unit tests)
- ✅ Tests are isolated (no shared state)
- ✅ Tests have clear names describing what's tested
- ✅ Tests fail when feature breaks
- ✅ Tests use realistic data
- ✅ Tests verify error messages and recovery

---

## Test Naming Convention

Good test names answer: **What happens when I do X?**

### Template
```python
def test_<component>_<scenario>_<expected_outcome>():
    """Docstring explains why this test matters."""
    pass
```

### Examples

**Good**:
```python
def test_order_manager_entry_succeeds_stop_fails_closes_position():
    """CRITICAL: Ensures unprotected position is immediately closed."""
    pass

def test_rate_limiter_concurrent_requests_respects_capacity():
    """Prevents burst traffic from exceeding API rate limits."""
    pass
```

**Bad**:
```python
def test_order_manager_1():
    """Test order manager."""
    pass

def test_success():
    """Test that it works."""
    pass
```

---

## Coverage Targets by Module

| Module | Unit Test Coverage | Integration Tests Required |
|--------|-------------------|---------------------------|
| agents/ | 90%+ | 10+ (full lifecycle) |
| risk/ | 95%+ (deterministic) | 5+ (edge cases) |
| orchestration/ | 85%+ | 20+ (coordination flows) |
| execution/ | 90%+ | 15+ (order lifecycle) |
| llm/ | 80%+ | 5+ (API failures) |
| api/ | 90%+ | 10+ (endpoint E2E) |
| data/ | 85%+ | 10+ (database ops) |

---

## Quick Decision Tree

### Should I write a unit test or integration test?

```
Does this component interact with external systems?
├─ NO → Write unit test (mock external dependencies)
└─ YES → Does the interaction involve complex state?
    ├─ NO → Write unit test (mock the interaction)
    └─ YES → Write BOTH:
        ├─ Unit test (verify logic with mocks)
        └─ Integration test (verify interaction works)
```

### Should I use mocks?

```
Does this dependency involve I/O (network, disk, external API)?
├─ YES → Mock it
└─ NO → Is it expensive to instantiate (>10ms)?
    ├─ YES → Mock it
    └─ NO → Use real implementation
```

### How many test cases do I need?

```
Minimum per feature:
├─ 1 happy path test
├─ 2 edge case tests (boundaries)
├─ 1 error handling test
└─ 1 integration test (if interacts with other components)

Total: 5 tests minimum per feature
```

---

## Test Maintenance Rules

### When to Update Tests

1. **Feature Change**: Update tests FIRST (TDD)
2. **Bug Fix**: Add test that reproduces bug, then fix
3. **Refactor**: Tests should still pass (if not, tests were too coupled)
4. **Performance Fix**: Add benchmark test to prevent regression

### When to Delete Tests

1. Feature is removed
2. Test is redundant (covered by integration test)
3. Test is flaky and can't be fixed (document why)

### When to Skip Tests

Only skip tests for:
1. External dependency unavailable (database, API key)
2. Platform-specific tests (OS, architecture)

**NEVER** skip tests because they're failing - fix them or delete them.

---

## Pre-Commit Test Checklist

Before committing code, run:

```bash
# 1. Run all tests
pytest triplegain/tests/

# 2. Check coverage
pytest --cov=triplegain/src --cov-report=term

# 3. Run only new/changed tests
pytest triplegain/tests/ -k "test_new_feature"

# 4. Check for flaky tests (run 10 times)
pytest triplegain/tests/ --count=10

# 5. Run integration tests
pytest triplegain/tests/integration/ -v
```

All must pass before pushing to main.

---

## Emergency Production Issue Response

If production issue occurs:

1. **Reproduce in test** (add test that fails)
2. **Fix issue**
3. **Verify test passes**
4. **Add related edge case tests**
5. **Deploy with confidence**

**Example workflow**:
```python
# 1. Reproduce issue
def test_order_execution_kraken_rate_limit_exceeded():
    """Reproduces production issue where orders failed during high load."""
    # Simulate the exact conditions that caused failure
    pass

# 2. Fix issue in source code
# ...

# 3. Verify test now passes
# pytest test_order_execution_kraken_rate_limit_exceeded.py

# 4. Add related tests
def test_order_execution_rate_limit_recovery():
    """Verify system recovers after rate limit is lifted."""
    pass
```

---

## Resources

- Full review: `/docs/development/reviews/test-suite-review-2025-12-19.md`
- Action plan: `/docs/development/reviews/test-improvement-action-plan.md`
- pytest docs: https://docs.pytest.org/
- pytest-asyncio: https://pytest-asyncio.readthedocs.io/

---

**Last Updated**: 2025-12-19
**Reviewer**: Code Review Agent
