# Kraken Database Data Gap Analysis

*Analysis Date: 2025-12-17*

## Executive Summary

This analysis examines the current state of the Kraken historical database, identifies data gaps in existing holdings, and evaluates untapped Kraken API data offerings that could benefit trading operations.

**Key Findings:**
1. **Significant candle data gaps** - 65% of recent candles show gaps >1 minute
2. **Sync status mismatch** - `data_sync_status` reports years of data, but actual tables contain only 3 months
3. **Limited symbol coverage** - Only 3 pairs tracked (XRP/USDT, BTC/USDT, XRP/BTC) vs 253+ available
4. **No order book data** - Missing Level 2 depth data critical for spread analysis
5. **No account data integration** - Ledger history, trade history not being collected

---

## Part 1: Current Data Holdings Analysis

### Database Overview

| Metric | Value |
|--------|-------|
| Database Size | 845 MB |
| Trades Table Size | 24 KB (compressed/retained) |
| Candles Table Size | 32 KB |
| Symbols Tracked | 3 |
| Continuous Aggregates | 8 timeframes |

### Data Sync Status vs Reality

The `data_sync_status` table reports historical coverage that does not match actual table contents:

| Symbol | Reported Oldest | Reported Newest | Actual Oldest (trades) | Actual Newest |
|--------|-----------------|-----------------|------------------------|---------------|
| XRP/BTC | 2016-07-19 | 2025-12-16 | 2025-09-18 | 2025-12-16 |
| BTC/USDT | 2019-12-19 | 2025-12-16 | 2025-09-18 | 2025-12-16 |
| XRP/USDT | 2020-04-30 | 2025-12-16 | 2025-09-18 | 2025-12-16 |

**Finding:** The sync status table is tracking metadata correctly for backfill operations, but retention policies have deleted older data. The 90-day retention policy on trades means only ~3 months of tick data is preserved.

### Candle Data Coverage

| Symbol | 1m Candles | Oldest | Newest | Coverage Period |
|--------|------------|--------|--------|-----------------|
| BTC/USDT | 343,519 | 2024-12-12 | 2025-12-16 | 35 days |
| XRP/BTC | 219,739 | 2024-12-12 | 2025-12-16 | 35 days |
| XRP/USDT | 219,965 | 2024-12-12 | 2025-12-16 | 35 days |

**Finding:** Candle data starts from December 12, 2024 - only 35 days of 1-minute data despite claims of years of historical data.

### Candle Gaps Analysis (Last 7 Days - XRP/USDT)

| Metric | Value |
|--------|-------|
| Total Candles (7 days) | 1,657 |
| Expected Candles (7 days) | 10,080 |
| Gaps Found (>1.5 min) | 1,082 |
| Max Gap | 54 minutes |
| Average Gap | 4.69 minutes |
| Gap Percentage | 65% |

**Finding:** The candle data has significant gaps. Only 16.4% of expected 1-minute candles are present. This indicates either:
1. WebSocket collection not running continuously
2. Low trading volume periods not generating candles
3. Gap filler not running on startup

### Trade Data Holdings

| Symbol | Trade Count | Oldest | Newest | Avg Trades/Day |
|--------|-------------|--------|--------|----------------|
| BTC/USDT | 538,692 | 2025-09-18 | 2025-12-16 | ~6,000 |
| XRP/USDT | 247,116 | 2025-09-18 | 2025-12-16 | ~2,700 |
| XRP/BTC | 138,604 | 2025-09-18 | 2025-12-16 | ~1,500 |

---

## Part 2: Data Gaps Identified

### Critical Gap 1: Missing Historical Data (High Priority)

**Impact:** Cannot perform accurate backtesting beyond 35 days

| Issue | Current State | Recommendation |
|-------|---------------|----------------|
| 1m candles limited | 35 days | Extend retention to 2+ years |
| Trades limited | 90 days | Store in cold storage or extend |
| No CSV import | CSV directory empty | Import Kraken historical CSVs |

**Action Required:**
1. Download Kraken historical CSV files from [Kraken Data](https://support.kraken.com/hc/en-us/articles/360047543791-Downloadable-historical-market-data-time-and-sales-)
2. Run `BulkCSVImporter` to load historical data
3. Adjust retention policies for longer-term analysis

### Critical Gap 2: Candle Continuity (High Priority)

**Impact:** Incomplete data causes indicator calculation errors

| Issue | Severity | Fix |
|-------|----------|-----|
| 65% gap rate in candles | Critical | Run gap_filler on startup |
| Missing overnight data | High | Ensure 24/7 WebSocket collection |
| No fill for zero-volume periods | Medium | Generate empty candles |

**Recommendation:** Implement startup hook that runs `run_gap_filler()` before any strategy initialization.

### Gap 3: Limited Symbol Coverage (Medium Priority)

**Current:** 3 symbols tracked
**Available:** 253+ margin-enabled pairs on Kraken

| Recommended Additional Symbols | Rationale |
|-------------------------------|-----------|
| ETH/USDT | 2nd largest by volume |
| SOL/USDT | High volatility, good for momentum |
| XRP/USD | Primary USD pair with higher liquidity |
| XRP/EUR | Geographic diversification |
| ETH/BTC | Cross-pair analysis |

**Note:** Already mapped in `types.py` but not actively collected:
- ETH/USDT, SOL/USDT, ETH/BTC, LTC/USDT, DOT/USDT, ADA/USDT, LINK/USDT

### Gap 4: Sync Status Accuracy (Medium Priority)

The `data_sync_status` table shows stale information due to retention policies:

```sql
-- Example: data_sync_status claims 8.3M XRP/BTC trades
-- Reality: Only 138,604 trades in table (90-day retention)
```

**Recommendation:** Update sync tracking to reflect actual retained data, not historical totals.

---

## Part 3: Untapped Kraken API Data Offerings

### High-Value Unused Endpoints

#### 1. Order Book Depth (`/0/public/Depth`) - HIGH PRIORITY

**Current State:** Not collected
**Value:** Critical for spread analysis, market microstructure, liquidity assessment

| Data Available | Use Case |
|----------------|----------|
| Bid/Ask prices (up to 500 levels) | Support/resistance detection |
| Volume at each level | Liquidity depth analysis |
| Order book imbalance | Short-term direction prediction |
| Spread calculation | Transaction cost optimization |

**WebSocket Alternative:** `book` channel provides real-time order book updates

**Database Schema Needed:**
```sql
CREATE TABLE order_book_snapshots (
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    side VARCHAR(3) NOT NULL,  -- 'bid' or 'ask'
    price DECIMAL(20, 10) NOT NULL,
    volume DECIMAL(20, 10) NOT NULL,
    level SMALLINT NOT NULL,
    PRIMARY KEY (timestamp, symbol, side, level)
);
```

#### 2. Ticker Data (`/0/public/Ticker`) - MEDIUM PRIORITY

**Current State:** Not collected
**Value:** Aggregated market summary without building from trades

| Field | Description |
|-------|-------------|
| a | Ask [price, whole lot volume, lot volume] |
| b | Bid [price, whole lot volume, lot volume] |
| c | Last trade closed [price, lot volume] |
| v | Volume [today, last 24 hours] |
| p | Volume weighted average price [today, last 24 hours] |
| t | Number of trades [today, last 24 hours] |
| l | Low [today, last 24 hours] |
| h | High [today, last 24 hours] |
| o | Today's opening price |

**Use Case:** Quick market snapshots, multi-symbol screening

#### 3. Spread History (`/0/public/Spread`) - MEDIUM PRIORITY

**Current State:** Not collected
**Value:** Historical bid-ask spreads for cost analysis

**Data Format:** `[timestamp, bid, ask]`

**Use Cases:**
- Optimal execution timing (narrow spread periods)
- Market maker strategy development
- Slippage estimation for backtesting

#### 4. Private Trade History (`/0/private/TradesHistory`) - HIGH PRIORITY

**Current State:** Not collected
**Value:** Your actual executed trades for performance analysis

| Data Available | Use Case |
|----------------|----------|
| Execution price | Slippage analysis |
| Fees paid | True cost calculation |
| Order ID correlation | Strategy attribution |
| Position tracking | P&L calculation |

**Permission Required:** Query Closed Orders & Trades (currently enabled)

#### 5. Ledger Entries (`/0/private/Ledgers`) - MEDIUM PRIORITY

**Current State:** Not collected
**Value:** Complete account history

| Entry Types | Use |
|-------------|-----|
| Deposits | Fund tracking |
| Withdrawals | Fund tracking |
| Trades | Execution record |
| Margin | Position fees |
| Rollover | Holding costs |
| Staking | Earn income |

#### 6. Account Balance Tracking (`/0/private/Balance`) - LOW PRIORITY

**Current State:** Not stored historically
**Value:** Portfolio tracking over time

### WebSocket Channels Not Utilized

| Channel | Status | Potential Value |
|---------|--------|-----------------|
| `ticker` | Unused | Real-time summary data |
| `book` | Unused | **HIGH** - Order book depth |
| `balances` | Unused | Real-time balance tracking |
| `executions` | Unused | Trade fill notifications |

**Current WebSocket Usage:** Only `trade` and `ohlc` channels subscribed

---

## Part 4: Recommendations

### Immediate Actions (This Week)

1. **Run Gap Filler**
   ```bash
   cd data/kraken_db
   python -m gap_filler --db-url "$DATABASE_URL"
   ```

2. **Import Historical CSVs**
   - Download from Kraken
   - Place in `data/kraken_db/candles/`
   - Run `BulkCSVImporter`

3. **Add Startup Gap Check**
   - Integrate `run_gap_filler()` into application startup

### Short-Term Improvements (This Month)

4. **Add Order Book Collection**
   - Implement periodic order book snapshots (every 1-5 minutes)
   - Create `order_book_snapshots` table
   - Use for spread analysis features

5. **Expand Symbol Coverage**
   - Add ETH/USDT, SOL/USDT, XRP/USD to collection
   - Update `DEFAULT_SYMBOLS` in `types.py`

6. **Fix Sync Status Tracking**
   - Create view showing actual vs claimed data
   - Add data quality monitoring

### Long-Term Enhancements (This Quarter)

7. **Private Data Integration**
   - Collect trade history for performance analysis
   - Store ledger entries for fund tracking
   - Enable balance history for portfolio analytics

8. **Order Book Analytics**
   - Implement order flow imbalance indicators
   - Add spread analysis to strategy framework
   - Create liquidity heat maps

9. **Data Quality Dashboard**
   - Monitor gap frequency
   - Alert on collection failures
   - Track data freshness

---

## Part 5: Data Model Recommendations

### Proposed New Tables

```sql
-- Order book depth snapshots
CREATE TABLE order_book_snapshots (
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    bids JSONB NOT NULL,  -- [[price, volume], ...]
    asks JSONB NOT NULL,
    spread DECIMAL(20, 10),
    mid_price DECIMAL(20, 10),
    PRIMARY KEY (timestamp, symbol)
);

-- Ticker snapshots
CREATE TABLE ticker_snapshots (
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    bid DECIMAL(20, 10),
    ask DECIMAL(20, 10),
    last DECIMAL(20, 10),
    volume_24h DECIMAL(20, 10),
    vwap_24h DECIMAL(20, 10),
    high_24h DECIMAL(20, 10),
    low_24h DECIMAL(20, 10),
    trades_24h INTEGER,
    PRIMARY KEY (timestamp, symbol)
);

-- Account balance history
CREATE TABLE balance_history (
    timestamp TIMESTAMPTZ NOT NULL,
    asset VARCHAR(10) NOT NULL,
    balance DECIMAL(20, 10) NOT NULL,
    PRIMARY KEY (timestamp, asset)
);

-- Trade execution history (our trades)
CREATE TABLE execution_history (
    order_id VARCHAR(50) NOT NULL,
    trade_id VARCHAR(50) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    side VARCHAR(4) NOT NULL,
    price DECIMAL(20, 10) NOT NULL,
    volume DECIMAL(20, 10) NOT NULL,
    fee DECIMAL(20, 10),
    fee_currency VARCHAR(10),
    PRIMARY KEY (trade_id)
);
```

---

## Appendix A: Current API Usage Summary

| Endpoint/Channel | Used | Purpose |
|------------------|------|---------|
| `/0/public/Trades` | Yes | Historical backfill |
| `/0/public/OHLC` | Yes | Gap filling |
| WebSocket `trade` | Yes | Real-time trades |
| WebSocket `ohlc` | Yes | Real-time candles |
| `/0/public/Ticker` | No | - |
| `/0/public/Depth` | No | - |
| `/0/public/Spread` | No | - |
| `/0/private/TradesHistory` | No | - |
| `/0/private/Ledgers` | No | - |
| `/0/private/Balance` | No | - |
| WebSocket `book` | No | - |
| WebSocket `ticker` | No | - |
| WebSocket `executions` | No | - |
| WebSocket `balances` | No | - |

## Appendix B: Retention Policy Impact

| Table | Retention | Records Lost Per Day |
|-------|-----------|---------------------|
| trades | 90 days | ~10,000 |
| candles | 365 days | ~4,320 |

**Cost of Extended Retention:**
- 1 year of trades: ~365 * 10K * 100 bytes = ~365 MB
- 2 years of candles: ~730 * 4,320 * 50 bytes = ~158 MB
- Total for 2-year history: ~0.5-1 GB per symbol

---

*Document Version: 1.0*
*Author: Claude Code Analysis*
*Next Review: 2026-01-17*
