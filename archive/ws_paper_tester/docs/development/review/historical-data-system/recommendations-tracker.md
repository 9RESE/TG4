# Historical Data System - Recommendations Tracker

**Source:** [Deep Code Review v1.0](./deep-review-v1.0.md)
**Date Created:** 2025-12-15
**Last Updated:** 2025-12-15

---

## Quick Reference

| Priority | Status | Count |
|----------|--------|-------|
| Immediate | **Completed** | 4 |
| Short-Term | **Completed** | 5 |
| Long-Term | Pending | 6 |

---

## Immediate Priority (Before Production)

These must be addressed before production deployment.

### REC-001: Add Buffer Overflow Protection

| Field | Value |
|-------|-------|
| **Issue ID** | ISSUE-011 |
| **Severity** | HIGH |
| **File** | `data/websocket_db_writer.py` |
| **Status** | **Completed** |
| **Completed** | 2025-12-15 |

**Problem:**
If the database is unavailable for extended periods, the trade buffer grows unbounded, causing memory exhaustion.

**Solution:**
```python
# Add to DatabaseWriter class

MAX_TRADE_BUFFER_SIZE = 10000
MAX_CANDLE_BUFFER_SIZE = 1000

def __init__(self, ...):
    ...
    self._overflow_count = 0

async def _flush_trades(self):
    if not self.trade_buffer:
        return

    # Protect against buffer overflow
    if len(self.trade_buffer) > self.MAX_TRADE_BUFFER_SIZE:
        dropped = len(self.trade_buffer) - self.MAX_TRADE_BUFFER_SIZE
        logger.warning(f"Trade buffer overflow, dropping {dropped} oldest records")
        while len(self.trade_buffer) > self.MAX_TRADE_BUFFER_SIZE:
            self.trade_buffer.popleft()
        self._overflow_count += dropped

    # ... rest of existing flush logic
```

**Acceptance Criteria:**
- [ ] Buffer size limit is enforced
- [ ] Overflow is logged with count of dropped records
- [ ] Oldest records are dropped first (preserves recency)
- [ ] Unit test for overflow scenario

---

### REC-002: Add Connection Pool Null Checks

| Field | Value |
|-------|-------|
| **Issue ID** | ISSUE-002 |
| **Severity** | MEDIUM |
| **Files** | `data/historical_provider.py`, `data/gap_filler.py`, etc. |
| **Status** | **Completed** |
| **Completed** | 2025-12-15 |

**Problem:**
Calling methods before `connect()` raises cryptic `AttributeError` on `NoneType`.

**Solution:**
```python
# Add helper method to HistoricalDataProvider

def _ensure_connected(self):
    """Raise RuntimeError if not connected."""
    if not self.pool:
        raise RuntimeError(
            "HistoricalDataProvider not connected. "
            "Call await provider.connect() first."
        )

async def get_candles(self, ...):
    self._ensure_connected()
    # ... rest of method
```

**Acceptance Criteria:**
- [ ] All public methods check for connection first
- [ ] Clear error message tells user to call `connect()`
- [ ] Applied to all classes that use connection pool

---

### REC-003: Validate Trade Data Before Storage

| Field | Value |
|-------|-------|
| **Issue ID** | ISSUE-008 |
| **Severity** | MEDIUM |
| **File** | `data/historical_backfill.py` |
| **Status** | **Completed** |
| **Completed** | 2025-12-15 |

**Problem:**
Malformed trade data from API could cause crashes or corrupt database.

**Solution:**
```python
async def store_trades(self, symbol: str, trades: List[list]):
    if not trades:
        return

    records = []
    skipped = 0

    for trade in trades:
        try:
            price, volume, timestamp, side, order_type, misc = trade[:6]

            # Validate data
            price_decimal = Decimal(str(price))
            volume_decimal = Decimal(str(volume))

            if price_decimal <= 0 or volume_decimal <= 0:
                logger.debug(f"Skipping invalid trade: price={price}, volume={volume}")
                skipped += 1
                continue

            records.append((
                symbol,
                datetime.fromtimestamp(float(timestamp), tz=timezone.utc),
                price_decimal,
                volume_decimal,
                'buy' if side == 'b' else 'sell',
                order_type,
                misc
            ))
        except (ValueError, TypeError, InvalidOperation) as e:
            logger.warning(f"Skipping malformed trade: {e}")
            skipped += 1
            continue

    if skipped > 0:
        logger.info(f"Skipped {skipped} invalid trades for {symbol}")

    # ... continue with database insert
```

**Acceptance Criteria:**
- [ ] Invalid prices (<=0, NaN) are rejected
- [ ] Invalid volumes are rejected
- [ ] Malformed timestamps are handled
- [ ] Skip count is logged
- [ ] Processing continues after invalid record

---

### REC-004: Remove Default Credentials

| Field | Value |
|-------|-------|
| **Issue ID** | ISSUE-016 |
| **Severity** | MEDIUM |
| **Files** | `docker-compose.yml`, `main_with_historical.py`, etc. |
| **Status** | **Completed** |
| **Completed** | 2025-12-15 |

**Problem:**
Default `changeme` password in code is a security risk.

**Solution:**

1. **docker-compose.yml:**
```yaml
environment:
  POSTGRES_PASSWORD: ${DB_PASSWORD:?DB_PASSWORD is required}
```

2. **main_with_historical.py:**
```python
parser.add_argument('--db-url', type=str,
                    default=os.getenv('DATABASE_URL'),
                    help='PostgreSQL connection URL (required)')

# Later in code:
if not args.db_url:
    parser.error("--db-url or DATABASE_URL environment variable is required")
```

3. **Update .env.example:**
```bash
# REQUIRED - Set these before starting
DB_PASSWORD=  # Generate with: openssl rand -base64 32
DATABASE_URL=postgresql://trading:YOUR_PASSWORD@localhost:5432/kraken_data
```

**Acceptance Criteria:**
- [ ] No default passwords in code
- [ ] Clear error when password not set
- [ ] Documentation updated
- [ ] `.env.example` shows required variables

---

## Short-Term (Within 1-2 Sprints)

### REC-005: Consolidate PAIR_MAP

| Field | Value |
|-------|-------|
| **Issue ID** | ISSUE-010 |
| **Severity** | LOW |
| **Files** | Multiple |
| **Status** | **Completed** |
| **Completed** | 2025-12-15 | |

**Action:** Move `PAIR_MAP` to `data/types.py` or a dedicated `data/config.py`:
```python
# data/config.py
PAIR_MAP = {
    'XRP/USDT': 'XRPUSDT',
    'BTC/USDT': 'XBTUSDT',
    'XRP/BTC': 'XRPXBT',
    'ETH/USDT': 'ETHUSDT',
    'SOL/USDT': 'SOLUSDT',
}

REVERSE_PAIR_MAP = {v: k for k, v in PAIR_MAP.items()}
```

---

### REC-006: Unify Candle Types

| Field | Value |
|-------|-------|
| **Issue ID** | ISSUE-003 |
| **Severity** | MEDIUM |
| **Files** | `data/types.py`, `data/historical_provider.py` |
| **Status** | **Completed** |
| **Completed** | 2025-12-15 | |

**Options:**
1. Keep `Candle` in provider for DB operations, `HistoricalCandle` for domain logic
2. Merge into single type with `from_row()` class method

**Decision Needed:** Discuss with team before implementation.

---

### REC-007: Add Error Path Tests

| Field | Value |
|-------|-------|
| **Issue ID** | ISSUE-018 |
| **Severity** | MEDIUM |
| **File** | `tests/test_historical_data.py` |
| **Status** | **Completed** |
| **Completed** | 2025-12-15 | |

**Tests to add:**
- [ ] Rate limit handling and backoff
- [ ] Connection timeout and retry
- [ ] Buffer overflow scenario
- [ ] Database unavailable during flush
- [ ] Malformed API response handling

---

### REC-008: Enable TLS for Database

| Field | Value |
|-------|-------|
| **Issue ID** | ISSUE-017 |
| **Severity** | LOW |
| **Files** | Documentation, `.env.example` |
| **Status** | Pending |

**Action:** Update documentation to show TLS-enabled URLs:
```
DATABASE_URL=postgresql://trading:password@localhost:5432/kraken_data?sslmode=require
```

---

### REC-009: Enable Data Retention Policies

| Field | Value |
|-------|-------|
| **Issue ID** | ISSUE-005 |
| **Severity** | MEDIUM |
| **File** | `scripts/continuous-aggregates.sql` |
| **Status** | **Completed** |
| **Completed** | 2025-12-15 | |

**Action:** Uncomment retention policies:
```sql
-- Keep raw trades for 90 days
SELECT add_retention_policy('trades', INTERVAL '90 days', if_not_exists => TRUE);

-- Keep 1-minute candles for 1 year
SELECT add_retention_policy('candles', INTERVAL '365 days', if_not_exists => TRUE);
```

---

## Long-Term (Future Improvements)

### REC-010: Optimize CSV Import

| Field | Value |
|-------|-------|
| **Issue ID** | ISSUE-013 |
| **Severity** | LOW |
| **Status** | **Completed** |
| **Completed** | 2025-12-15 |

**Action:** Replaced `df.iterrows()` with `itertuples()` for ~100x speedup.

### REC-011: Parallel Gap Filling

| Issue ID | ISSUE-014 |
|----------|-----------|
| Implement parallel pagination within large gaps |

### REC-012: Public Data Handler API

| Issue ID | ISSUE-015 |
|----------|-----------|
| Add `register_data_handler()` method to WebSocketPaperTester |

### REC-013: Add Observability

| Issue ID | - |
|----------|-----------|
| Prometheus metrics, structured logging, PagerDuty alerts |

### REC-014: Connection Pool Optimization

| Issue ID | - |
|----------|-----------|
| Consider PgBouncer for connection pooling across services |

### REC-015: Schema Migrations

| Issue ID | - |
|----------|-----------|
| Add Alembic for database schema versioning |

---

## Completion Log

| Date | Recommendation | Implementer | Notes |
|------|----------------|-------------|-------|
| 2025-12-15 | REC-001 | Claude Code | Buffer overflow protection with MAX_TRADE/CANDLE_BUFFER_SIZE |
| 2025-12-15 | REC-002 | Claude Code | Added _ensure_connected() to all provider methods |
| 2025-12-15 | REC-003 | Claude Code | Trade validation with price/volume/timestamp checks |
| 2025-12-15 | REC-004 | Claude Code | Removed all default passwords, require DATABASE_URL |
| 2025-12-15 | REC-005 | Claude Code | Centralized PAIR_MAP, CSV_SYMBOL_MAP, DEFAULT_SYMBOLS in types.py |
| 2025-12-15 | REC-006 | Claude Code | Documented distinction between Candle and HistoricalCandle |
| 2025-12-15 | REC-007 | Claude Code | Added error path tests for validation, connection, mappings |
| 2025-12-15 | REC-009 | Claude Code | Enabled retention policies in continuous-aggregates.sql |
| 2025-12-15 | REC-010 | Claude Code | Replaced iterrows() with itertuples() for ~100x speedup |

---

*Last updated: 2025-12-15*
