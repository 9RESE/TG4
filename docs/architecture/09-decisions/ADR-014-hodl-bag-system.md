# ADR-014: Hodl Bag System Architecture

**Status**: Accepted
**Date**: 2025-12-20
**Decision Makers**: Development Team
**Context**: Phase 8 Implementation

---

## Context

TripleGain is an LLM-assisted trading system that actively trades BTC/USDT, XRP/USDT, and XRP/BTC pairs. While the primary goal is generating short-term trading profits, there's a desire to build long-term wealth separate from active trading capital.

### Problem Statement

1. **Profit Volatility**: Trading profits can be re-invested and lost in subsequent trades
2. **No Long-Term Accumulation**: All capital remains in the active trading pool
3. **Psychological Factor**: Traders benefit from seeing tangible long-term growth separate from daily P&L
4. **Tax Efficiency**: Short-term trading gains are taxed at higher rates than long-term holdings

### Requirements

1. Automatically allocate a percentage of realized profits to long-term holdings
2. Split allocation across multiple assets for diversification
3. Use per-asset purchase thresholds to minimize trading fees
4. Completely isolate hodl bags from active trading and rebalancing
5. Support both paper and live trading modes
6. Provide API for monitoring and manual intervention

---

## Decision

### 1. Profit Allocation Percentage

**Decision**: Allocate 10% of realized trading profits to hodl bags.

**Rationale**:
- 10% is meaningful for wealth building but doesn't significantly impact trading capital
- Trades yielding $100 profit contribute $10 to hodl bags
- At 50% annual return target, 5% of portfolio value flows to hodl bags annually
- Conservative allocation allows the system to compound trading capital

**Alternatives Considered**:
- 5%: Too small to be meaningful for wealth building
- 20%: Would significantly reduce trading capital and compounding
- Variable (based on portfolio size): Added complexity without clear benefit

### 2. Asset Split Strategy

**Decision**: Split hodl allocation equally across USDT, XRP, and BTC (33.33% each).

**Rationale**:
- **USDT**: Stable reserve for emergencies or opportunistic buying
- **XRP**: Primary trading asset, long-term growth potential
- **BTC**: Store of value, different risk profile from XRP
- Equal split simplifies logic and provides balanced exposure
- Matches the 33/33/33 portfolio allocation philosophy

**Alternatives Considered**:
- 100% BTC: Concentrated risk, no diversification
- Weighted by market cap: Added complexity, requires external data
- User-configurable: Added complexity, decision paralysis

### 3. Per-Asset Purchase Thresholds

**Decision**: Execute purchases only when pending accumulation reaches asset-specific thresholds.

| Asset | Threshold | Rationale |
|-------|-----------|-----------|
| USDT  | $1        | No purchase needed, just rounding threshold |
| XRP   | $25       | Kraken minimum ~$10 + buffer for fees |
| BTC   | $15       | Kraken minimum ~$10 + buffer for fees |

**Rationale**:
- Minimizes trading fees by batching small amounts
- Respects exchange minimum order sizes
- Prevents dust accumulation from failed micro-orders
- USDT requires no purchase (already in quote currency)

**Implementation**:
```
Profit $100 -> Hodl $10 -> Split $3.33 each
- USDT: Immediately held (above $1 threshold)
- XRP:  Pending until total reaches $25
- BTC:  Pending until total reaches $15
```

### 4. Separation from Trading Capital

**Decision**: Hodl bags are completely separate from active trading.

**Rationale**:
- Portfolio rebalance agent excludes hodl balances from allocation calculations
- Position limits don't include hodl positions
- Risk engine exposure calculations ignore hodl holdings
- Prevents hodl bags from being liquidated during drawdowns

**Implementation**:
- `PortfolioRebalanceAgent._get_hodl_bags()` queries hodl balances from database
- `check_allocation()` subtracts hodl balances before calculating target allocations
- `hodl_bags` table has separate schema from `positions` table

### 5. Dedicated Manager vs Agent

**Decision**: Implement as `HodlBagManager` class, not an LLM agent.

**Rationale**:
- Hodl logic is deterministic (percentage split, threshold comparison)
- No market analysis or decision-making required
- LLM calls would add latency and cost without benefit
- Manager integrates directly with PositionTracker for efficiency

**Alternatives Considered**:
- LLM Agent: Overkill for deterministic logic
- Part of PortfolioRebalance: Would complicate that agent's responsibilities
- Part of PositionTracker: Violates single responsibility principle

### 6. Paper Trading Support

**Decision**: Full paper trading simulation with simulated purchases.

**Implementation**:
- `is_paper_mode` flag in HodlBagManager
- Paper mode simulates purchases using fallback prices from config
- Generates paper order IDs for tracking
- Same API and state management as live mode

**Rationale**:
- Allows complete end-to-end testing
- Paper trades tracked separately in database (`is_paper` column)
- No risk of accidental real purchases during testing

### 7. Database Schema

**Decision**: Four dedicated tables for hodl bag tracking.

| Table | Purpose |
|-------|---------|
| `hodl_bags` | Current balance and cost basis per asset |
| `hodl_transactions` | All accumulations/withdrawals for audit |
| `hodl_pending` | Pending amounts waiting for threshold |
| `hodl_bag_snapshots` | Time-series value tracking |

**Rationale**:
- Clean separation from trading tables
- Complete audit trail for tax purposes
- Efficient queries for each use case
- TimescaleDB hypertable for snapshots (if available)

---

## Consequences

### Positive

1. **Wealth Building**: 10% of profits automatically flow to long-term holdings
2. **Diversification**: Split across USDT/XRP/BTC reduces concentration risk
3. **Fee Efficiency**: Thresholds prevent wasteful micro-purchases
4. **Psychological Benefit**: Visible long-term growth separate from trading P&L
5. **Tax Optimization**: Long-term holdings may qualify for lower tax rates
6. **Isolation**: Hodl bags protected from trading volatility and rebalancing

### Negative

1. **Reduced Compounding**: 10% of profits unavailable for trading capital growth
2. **Threshold Delays**: Small profits may take multiple trades to trigger purchase
3. **Added Complexity**: New manager, tables, API endpoints to maintain
4. **Price Risk**: Pending accumulations may miss favorable prices

### Neutral

1. **Configuration Required**: New `config/hodl.yaml` file to manage
2. **API Expansion**: 7 new endpoints for monitoring and control
3. **Test Coverage**: 56 new tests to maintain

---

## Implementation Details

### Integration Points

```
PositionTracker.close_position()
    └── HodlBagManager.process_trade_profit()
            ├── Calculate allocation (10% of profit)
            ├── Split across assets (33.33% each)
            ├── Record pending accumulations
            └── Check thresholds and execute if reached
```

### Configuration (hodl.yaml)

```yaml
hodl_bags:
  enabled: true
  allocation_pct: 10
  split:
    usdt_pct: 33.34
    xrp_pct: 33.33
    btc_pct: 33.33
  min_accumulation:
    usdt: 1
    xrp: 25
    btc: 15
  limits:
    max_single_accumulation_usd: 1000
    daily_accumulation_limit_usd: 5000
```

### Safety Limits

- **Single Accumulation Cap**: $1,000 max per trade to prevent large single allocations
- **Daily Limit**: $5,000 max per day to prevent runaway accumulation
- **Minimum Profit**: $1 minimum profit to trigger allocation (avoids micro-allocations)

---

## References

- [Phase 8 Implementation Plan](../../development/TripleGain-implementation-plan/08-phase-8-hodl-bag-system.md)
- [Hodl Bag Feature Documentation](../../development/features/phase-8-hodl-bag-system.md)
- [ADR-013: Paper Trading Design](./ADR-013-paper-trading-design.md) (paper mode pattern)
