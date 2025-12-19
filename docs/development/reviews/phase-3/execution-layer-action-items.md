# Execution Layer - Critical Action Items

**Generated**: 2025-12-19
**Source**: Execution Layer Code Review
**Status**: BLOCKING PRODUCTION DEPLOYMENT

---

## Critical Priority (Must Fix Before Live Trading)

### CRITICAL-1: Minimum Order Size Validation
**File**: `triplegain/src/execution/order_manager.py`
**Location**: `execute_trade()` method, before line 330
**Severity**: Critical - Orders will fail at exchange

**Task**:
```python
# Add before size validation (line 330)
min_size = self._get_min_order_size(proposal.symbol)
if size < min_size:
    return ExecutionResult(
        success=False,
        error_message=f"Order size {size} below minimum {min_size} for {proposal.symbol}",
        execution_time_ms=int((time.perf_counter() - start_time) * 1000),
    )

# Add helper method
def _get_min_order_size(self, symbol: str) -> Decimal:
    """Get minimum order size for symbol from config."""
    symbol_config = self.config.get('symbols', {}).get(symbol, {})
    return Decimal(str(symbol_config.get('min_order_size', 0)))
```

**Test**: Add test case for below-minimum order rejection
**Effort**: 1 hour

---

### CRITICAL-2: Partial Fill Handling
**File**: `triplegain/src/execution/order_manager.py`
**Location**: `_monitor_order()` method, lines 486-565
**Severity**: Critical - Position size mismatch risk

**Task**:
1. Distinguish between partial and full fills
2. Update position with actual filled size
3. Handle contingent orders for partial fills
4. Add configuration for partial fill behavior

**Implementation**:
```python
# In _monitor_order(), around line 524
if kraken_status == "closed":
    vol_executed = Decimal(str(order_info.get("vol_exec", 0)))
    vol_ordered = order.size

    if vol_executed < vol_ordered:
        # Partial fill
        order.status = OrderStatus.PARTIALLY_FILLED
        logger.warning(
            f"Partial fill: {vol_executed}/{vol_ordered} for {order.id}"
        )

    order.filled_size = vol_executed
    order.filled_price = Decimal(str(order_info.get("price", 0)))
    order.updated_at = datetime.now(timezone.utc)

    # Handle fill (will create position with actual filled size)
    await self._handle_order_fill(order, proposal)

    # Update proposal for contingent orders to match actual fill
    if vol_executed < vol_ordered:
        # Adjust SL/TP based on actual size
        pass
```

**Test**: Add test for partial fill scenarios
**Effort**: 4 hours

---

### CRITICAL-3: Slippage Protection for Market Orders
**File**: `triplegain/src/execution/order_manager.py`
**Location**: `execute_trade()` method, line 324
**Severity**: Critical - Unprotected market exposure

**Task**:
Replace pure market orders with IOC limit orders that have slippage protection.

**Implementation**:
```python
# Replace line 324 with:
if proposal.entry_price:
    order_type = OrderType.LIMIT
    limit_price = Decimal(str(proposal.entry_price))
else:
    # Market order → Use IOC limit with slippage protection
    order_type = OrderType.LIMIT
    max_slippage_pct = Decimal(str(
        self.config.get('orders', {}).get('market_order_slippage_pct', 0.5)
    ))

    # Get current market price
    current_price = await self._get_current_market_price(proposal.symbol)

    if proposal.side == "buy":
        limit_price = current_price * (1 + max_slippage_pct / 100)
    else:
        limit_price = current_price * (1 - max_slippage_pct / 100)

    logger.info(
        f"Market order with slippage protection: {current_price} → {limit_price} "
        f"(max slippage: {max_slippage_pct}%)"
    )

# Update order creation to use limit_price and IOC time-in-force
```

**Test**: Add test for slippage protection activation
**Effort**: 3 hours

---

### CRITICAL-5: Fix Take-Profit Price Field
**File**: `triplegain/src/execution/order_manager.py`
**Location**: `_place_take_profit()` line 676
**Severity**: Critical - Orders may not execute correctly

**Task**:
Review Kraken API documentation and fix field mapping for TP orders.

**Investigation Needed**:
1. Verify if Kraken take-profit orders use `price` or `price2` (stop_price)
2. Determine if take-profit-limit requires both fields
3. Test in mock mode and verify order format

**Implementation** (pending API verification):
```python
# Option 1: If TP uses stop_price like SL
tp_order = Order(
    ...
    order_type=OrderType.TAKE_PROFIT,
    stop_price=Decimal(str(proposal.take_profit)),  # Trigger price
    parent_order_id=parent_order.id,
)

# Option 2: If TP-limit requires both
tp_order = Order(
    ...
    order_type=OrderType.TAKE_PROFIT_LIMIT,
    price=Decimal(str(proposal.take_profit)),  # Limit price
    stop_price=Decimal(str(proposal.take_profit)),  # Trigger price
    parent_order_id=parent_order.id,
)
```

**Test**: Add test for TP order field validation
**Effort**: 2 hours (+ API research)

---

### CRITICAL-6: Account for Fees in P&L
**File**: `triplegain/src/execution/position_tracker.py`
**Location**: `Position.calculate_pnl()` lines 91-113
**Severity**: Critical - Incorrect profitability reporting

**Task**:
Modify P&L calculation to deduct trading fees.

**Implementation**:
```python
def calculate_pnl(
    self,
    current_price: Decimal,
    include_fees: bool = True
) -> tuple[Decimal, Decimal]:
    """
    Calculate unrealized P&L.

    Args:
        current_price: Current market price
        include_fees: Whether to deduct estimated exit fees (default True)

    Returns:
        Tuple of (unrealized_pnl_usd, unrealized_pnl_pct)
    """
    if self.entry_price == 0:
        return Decimal(0), Decimal(0)

    # Calculate price-based P&L
    if self.side == PositionSide.LONG:
        price_diff = current_price - self.entry_price
        pnl = price_diff * self.size * self.leverage
        pnl_pct = (price_diff / self.entry_price) * 100 * self.leverage
    else:
        price_diff = self.entry_price - current_price
        pnl = price_diff * self.size * self.leverage
        pnl_pct = (price_diff / self.entry_price) * 100 * self.leverage

    # Deduct fees if requested
    if include_fees:
        position_value = self.size * current_price
        # Entry fee (0.26% taker) was already paid
        # Estimate exit fee (0.26% taker)
        exit_fee = position_value * Decimal("0.0026")
        pnl -= exit_fee

        # Recalculate percentage
        initial_value = self.size * self.entry_price
        if initial_value > 0:
            pnl_pct = (pnl / initial_value) * 100
        else:
            pnl_pct = Decimal(0)

    return pnl, pnl_pct
```

**Test**: Add test comparing P&L with/without fees
**Effort**: 2 hours

---

### CRITICAL-8: Fix Trailing Stop Logic
**File**: `triplegain/src/execution/position_tracker.py`
**Location**: `update_trailing_stops()` lines 637-710
**Severity**: High - Could overwrite manual stops

**Task**:
1. Only activate trailing stop if it improves existing SL
2. Add safeguard to never loosen stop-loss
3. Make trailing stop strictly opt-in

**Implementation**:
```python
# In update_trailing_stops(), around line 690 (LONG)
if price > (position.trailing_stop_highest_price or Decimal(0)):
    position.trailing_stop_highest_price = price
    new_stop = price * (1 - trail_distance / 100)

    # Only update if new stop is tighter (higher) than current
    if position.stop_loss is None or new_stop > position.stop_loss:
        old_stop = position.stop_loss
        position.stop_loss = new_stop
        logger.info(
            f"Trailing stop updated for {position.id}: "
            f"SL {old_stop} → {new_stop} (tightened)"
        )
    else:
        logger.debug(
            f"Trailing stop {new_stop} not applied (current SL {position.stop_loss} is tighter)"
        )
```

**Test**: Add test for trailing stop not loosening manual SL
**Effort**: 2 hours

---

### CRITICAL-9: Add Comprehensive Position Validation
**File**: `triplegain/src/execution/position_tracker.py`
**Location**: `Position.__post_init__()` lines 76-90
**Severity**: High - Bad data could corrupt system

**Task**:
Add validation for stop-loss and take-profit placement.

**Implementation**:
```python
def __post_init__(self):
    """Validate position fields after initialization."""
    # Existing validations...
    if self.leverage < 1:
        raise ValueError(f"Leverage must be >= 1, got {self.leverage}")
    if self.leverage > 5:
        raise ValueError(f"Leverage must be <= 5 (system limit), got {self.leverage}")
    if self.size <= 0:
        raise ValueError(f"Position size must be > 0, got {self.size}")

    # NEW: Entry price must be positive
    if self.entry_price <= 0:
        raise ValueError(f"Entry price must be > 0, got {self.entry_price}")

    # NEW: Validate stop-loss placement
    if self.stop_loss is not None:
        if self.side == PositionSide.LONG:
            if self.stop_loss >= self.entry_price:
                raise ValueError(
                    f"LONG stop-loss {self.stop_loss} must be < entry {self.entry_price}"
                )
        else:  # SHORT
            if self.stop_loss <= self.entry_price:
                raise ValueError(
                    f"SHORT stop-loss {self.stop_loss} must be > entry {self.entry_price}"
                )

    # NEW: Validate take-profit placement
    if self.take_profit is not None:
        if self.side == PositionSide.LONG:
            if self.take_profit <= self.entry_price:
                raise ValueError(
                    f"LONG take-profit {self.take_profit} must be > entry {self.entry_price}"
                )
        else:  # SHORT
            if self.take_profit >= self.entry_price:
                raise ValueError(
                    f"SHORT take-profit {self.take_profit} must be < entry {self.entry_price}"
                )
```

**Test**: Add tests for invalid SL/TP placement
**Effort**: 1 hour

---

### SAFETY-1: Pre-Flight Safety Checks
**File**: `triplegain/src/execution/order_manager.py`
**Location**: `execute_trade()` beginning (after line 300)
**Severity**: High - Could execute trades when unsafe

**Task**:
Add pre-flight validation before execution.

**Implementation**:
```python
async def execute_trade(
    self,
    proposal: 'TradeProposal',
) -> ExecutionResult:
    """Execute a validated trade proposal."""
    start_time = time.perf_counter()

    # PRE-FLIGHT SAFETY CHECKS
    # 1. Check circuit breaker status
    if self.risk_engine:
        circuit_status = self.risk_engine.get_circuit_breaker_status()
        if circuit_status["active"]:
            return ExecutionResult(
                success=False,
                error_message=f"Circuit breaker active: {circuit_status['reason']}",
                execution_time_ms=int((time.perf_counter() - start_time) * 1000),
            )

    # 2. Check daily loss limit proximity
    if self.risk_engine:
        current_loss = self.risk_engine.get_daily_loss()
        daily_limit = self.risk_engine.config.get("daily_loss_limit_pct", 5.0)
        if current_loss >= daily_limit * 0.9:  # 90% of limit
            logger.warning(
                f"Near daily loss limit: {current_loss:.2f}% / {daily_limit}%"
            )
            # Could reject or require higher confidence

    # 3. Validate proposal size
    if proposal.size_usd <= 0:
        # ... existing validation

    # Rest of method...
```

**Test**: Add tests for pre-flight check failures
**Effort**: 2 hours

---

### SAFETY-2: Position Size Sanity Checks
**File**: `triplegain/src/execution/order_manager.py`
**Location**: `_calculate_size()` lines 819-823
**Severity**: High - Could attempt massive orders

**Task**:
1. Remove dangerous fallback
2. Add upper bound validation
3. Cross-check with risk limits

**Implementation**:
```python
async def _calculate_size(self, proposal: 'TradeProposal') -> Decimal:
    """
    Calculate order size in base currency.

    Returns:
        Order size in base currency (BTC, XRP, etc.)

    Raises:
        ValueError: If size cannot be calculated or exceeds limits
    """
    if not proposal.entry_price or proposal.entry_price <= 0:
        raise ValueError(
            f"Cannot calculate size: invalid entry_price {proposal.entry_price}"
        )

    # Calculate size
    size = Decimal(str(proposal.size_usd)) / Decimal(str(proposal.entry_price))

    # Sanity check: Maximum position size
    max_position_usd = Decimal("50000")  # $50k max per trade
    calculated_usd = size * Decimal(str(proposal.entry_price))
    if calculated_usd > max_position_usd:
        raise ValueError(
            f"Position size ${calculated_usd} exceeds maximum ${max_position_usd}"
        )

    # Cross-check with risk engine limits if available
    if self.risk_engine:
        max_per_trade = self.risk_engine.config.get("max_position_size_usd", 10000)
        if calculated_usd > Decimal(str(max_per_trade)):
            raise ValueError(
                f"Position size ${calculated_usd} exceeds risk limit ${max_per_trade}"
            )

    return size
```

**Test**: Add tests for oversized position rejection
**Effort**: 2 hours

---

## High Priority (Should Fix Soon)

### CRITICAL-4: Fix Race Condition in Order History
**File**: `triplegain/src/execution/order_manager.py`
**Location**: `_monitor_order()` cleanup, lines 570-584
**Severity**: High - Data consistency

**Task**: Use single lock or deque for atomic transitions

**Effort**: 1 hour

---

### CRITICAL-7: Clarify SL/TP Trigger Logic
**File**: `triplegain/src/execution/position_tracker.py`
**Location**: `check_sl_tp_triggers()` lines 567-619
**Severity**: Medium - Logic clarification

**Task**: Add comments explaining precedence, ensure only one trigger per position

**Effort**: 30 minutes

---

## Test Coverage (High Priority)

### Required New Test Cases
**File**: `triplegain/tests/unit/execution/test_order_manager.py`

1. `test_execute_trade_below_minimum_size()` - CRITICAL-1
2. `test_execute_trade_partial_fill()` - CRITICAL-2
3. `test_market_order_slippage_protection()` - CRITICAL-3
4. `test_take_profit_order_field_mapping()` - CRITICAL-5
5. `test_execute_trade_oversized_position()` - SAFETY-2
6. `test_execute_trade_circuit_breaker_active()` - SAFETY-1
7. `test_concurrent_order_placement()`
8. `test_order_monitoring_timeout()`
9. `test_rate_limiter_token_exhaustion()`

**File**: `triplegain/tests/unit/execution/test_position_tracker.py`

10. `test_calculate_pnl_with_fees()` - CRITICAL-6
11. `test_trailing_stop_does_not_loosen_sl()` - CRITICAL-8
12. `test_position_invalid_stop_loss_placement()` - CRITICAL-9
13. `test_position_invalid_take_profit_placement()` - CRITICAL-9
14. `test_position_with_zero_entry_price()` - CRITICAL-9
15. `test_sl_tp_both_triggered_priority()`
16. `test_snapshot_with_stale_price_cache()`

**Total**: 16 critical tests
**Effort**: 8 hours

---

## Timeline

### Week 1: Critical Fixes
- **Day 1**: CRITICAL-1, 2 (order validation + partial fills) - 5 hours
- **Day 2**: CRITICAL-3 (slippage protection) - 3 hours
- **Day 3**: CRITICAL-5, 6 (TP fix + fee accounting) - 4 hours
- **Day 4**: CRITICAL-8, 9 (trailing stop + validation) - 3 hours
- **Day 5**: SAFETY-1, 2 (pre-flight checks) - 4 hours

**Total**: ~20 hours development

### Week 2: Testing + High Priority
- **Days 1-2**: Add 16 critical test cases - 8 hours
- **Day 3**: Fix CRITICAL-4, 7 (race condition + clarity) - 2 hours
- **Days 4-5**: Integration testing + bug fixes - 8 hours

**Total**: ~18 hours

### Weeks 3-4: Paper Trading Validation
- Extended paper trading with all fixes
- Monitor for edge cases
- Performance validation
- Documentation updates

---

## Success Criteria

### Before Moving to Paper Trading
- [ ] All 9 CRITICAL issues resolved
- [ ] All 16 critical tests passing
- [ ] Unit test coverage > 80% for execution layer
- [ ] Mock mode tests pass 100 consecutive executions
- [ ] Code review by second developer
- [ ] All safety checks validated

### Before Live Trading
- [ ] 30+ days paper trading with zero critical errors
- [ ] P&L calculations verified against exchange
- [ ] Position tracking accuracy 100%
- [ ] All edge cases tested
- [ ] Performance within latency targets
- [ ] Monitoring and alerting operational

---

**Status**: READY FOR DEVELOPMENT
**Owner**: TBD
**Priority**: BLOCKING - Cannot deploy to production until complete
