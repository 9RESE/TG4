# Kraken Database - Current Data Holdings

*Last updated: 2025-12-17*

## Summary

| Metric | Value |
|--------|-------|
| **Symbols Tracked** | 3 (XRP/USDT, BTC/USDT, XRP/BTC) |
| **Total Trades** | 924,412 |
| **Total 1m Candles** | 783,223 |
| **Database Size** | ~700 MB (including aggregates) |
| **Data Range** | Dec 2024 - Present (candles), Sep 2025 - Present (trades) |

## Data By Symbol

### Trades (Raw Tick Data)

| Symbol | Count | Oldest | Newest |
|--------|-------|--------|--------|
| BTC/USDT | 538,692 | 2025-09-18 00:00:00 UTC | 2025-12-16 06:30:00 UTC |
| XRP/USDT | 247,116 | 2025-09-18 00:01:36 UTC | 2025-12-16 13:34:45 UTC |
| XRP/BTC | 138,604 | 2025-09-18 00:07:56 UTC | 2025-12-16 13:29:37 UTC |

**Note**: Trades are subject to a 90-day retention policy. Older trades are automatically deleted, but the candles built from them are preserved.

### 1-Minute Candles (Base Data)

| Symbol | Count | Oldest | Newest |
|--------|-------|--------|--------|
| BTC/USDT | 343,519 | 2024-12-12 00:00:00 UTC | 2025-12-16 13:46:00 UTC |
| XRP/USDT | 219,965 | 2024-12-12 00:00:00 UTC | 2025-12-16 13:46:00 UTC |
| XRP/BTC | 219,739 | 2024-12-12 00:00:00 UTC | 2025-12-16 13:46:00 UTC |

### Historical Sync Status

The `data_sync_status` table tracks the full data history that has been processed:

| Symbol | Data Type | Historical Range | Records Processed |
|--------|-----------|------------------|-------------------|
| BTC/USDT | trades | 2019-12-19 → 2025-12-16 | 10,404,404 |
| BTC/USDT | candles_1m | 2019-12-19 → 2025-12-16 | - |
| XRP/USDT | trades | 2020-04-30 → 2025-12-16 | 3,985,004 |
| XRP/USDT | candles_1m | 2020-04-30 → 2025-12-16 | 738,990 |
| XRP/BTC | trades | 2016-07-19 → 2025-12-16 | 8,335,808 |
| XRP/BTC | candles_1m | 2016-07-19 → 2025-12-16 | 1,505,829 |

**Note**: This shows the total data processed over time. Due to retention policies, only recent data is stored in the tables.

## Storage Usage

### Base Tables

| Table | Size | Notes |
|-------|------|-------|
| candles | 57 MB | 1-minute base data |
| trades | 34 MB | Rolling 90-day window |
| external_indicators | 24 KB | Fear & Greed, etc. |

### Continuous Aggregates

| Aggregate | Size | Interval |
|-----------|------|----------|
| candles_5m | 325 MB | 5 minutes |
| candles_15m | 138 MB | 15 minutes |
| candles_30m | 76 MB | 30 minutes |
| candles_1h | 41 MB | 1 hour |
| candles_4h | 13 MB | 4 hours |
| candles_12h | 6 MB | 12 hours |
| candles_1d | 4 MB | 1 day |
| candles_1w | 2 MB | 1 week |

**Total Database Size**: ~700 MB

## Retention Policies

| Table | Retention | Schedule |
|-------|-----------|----------|
| trades | 90 days | Daily check |
| candles | 365 days | Daily check |

Continuous aggregates inherit retention from their source tables.

## Data Quality

### Coverage Analysis

Based on the candle counts vs expected candles:

| Symbol | Expected (Dec 12 - Dec 16) | Actual | Coverage |
|--------|---------------------------|--------|----------|
| BTC/USDT | ~5,000 min/day × 35 days = 175,000 | 343,519 | 196% (includes overlap) |
| XRP/USDT | ~175,000 | 219,965 | 126% |
| XRP/BTC | ~175,000 | 219,739 | 126% |

**Note**: Coverage > 100% indicates potential duplicate entries or overlapping imports. The ON CONFLICT handling prevents actual duplicates.

### Gap Detection

Use the GapFiller to check for gaps:

```python
from data.kraken_db import GapFiller

filler = GapFiller(db_url)
await filler.start()
gaps = await filler.detect_gaps(min_gap_minutes=2)
for gap in gaps:
    print(f"{gap.symbol}: {gap.duration} gap from {gap.start_time}")
```

## Updating This Document

To get current statistics, run:

```sql
-- Quick summary
SELECT
    symbol,
    data_type,
    oldest_timestamp,
    newest_timestamp,
    total_records
FROM data_sync_status
ORDER BY symbol, data_type;

-- Candle counts
SELECT
    symbol,
    COUNT(*) as count,
    MIN(timestamp) as oldest,
    MAX(timestamp) as newest
FROM candles
WHERE interval_minutes = 1
GROUP BY symbol;

-- Storage sizes
SELECT
    hypertable_name,
    pg_size_pretty(hypertable_size(format('%I.%I', hypertable_schema, hypertable_name)::regclass)) as size
FROM timescaledb_information.hypertables;
```

## Related Documentation

- [API Reference](kraken-db-api.md) - Query methods for data access
- [Operations Guide](../how-to/kraken-db-operations.md) - Backfill and gap filling
- [Architecture](../explanation/kraken-db-architecture.md) - Retention policy rationale
