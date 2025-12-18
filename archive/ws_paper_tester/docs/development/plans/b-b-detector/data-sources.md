# Market Regime Detector: Data Sources

## Overview

This document catalogs all data sources available for market regime detection, including existing Kraken WebSocket data, REST API endpoints, and external free APIs.

---

## 1. Kraken WebSocket API (Primary Data Source)

**Currently Integrated:** Yes, via `ws_tester/data_layer.py`

### 1.1 Trade Channel

**Subscription:** `trade`
**Data Received:**
- Symbol
- Price
- Volume
- Side (buy/sell)
- Timestamp

**Usage for Regime Detection:**
- Real-time price updates
- Volume analysis
- Trade flow direction (buy/sell imbalance)

### 1.2 Ticker Channel

**Subscription:** `ticker`
**Data Received:**
- Best bid/ask prices
- 24h volume
- 24h high/low
- Last trade price

**Usage for Regime Detection:**
- Current price for indicator calculations
- Spread analysis for liquidity assessment

### 1.3 OHLC Channel

**Subscription:** `ohlc`
**Intervals:** 1, 5, 15, 30, 60, 240, 1440, 10080, 21600 minutes

**Data Received per Candle:**
- Open, High, Low, Close prices
- Volume
- VWAP (Volume Weighted Average Price)
- Trade count

**Usage for Regime Detection:**
- Primary data for all technical indicators
- Multi-timeframe analysis
- Volatility calculations (ATR, Bollinger Bands)

**Current Implementation:**
- 1-minute candles: Built from trade data
- 5-minute candles: Built from 1-minute candles

**Extension Opportunity:**
- Subscribe to 15m, 1h, 4h OHLC channels directly
- Build longer timeframes from shorter ones

### 1.4 Order Book Channel

**Subscription:** `book`
**Depth:** 10, 25, 100, 500, 1000 levels

**Data Received:**
- Bid/ask price levels
- Volume at each level
- Order book snapshots and updates

**Usage for Regime Detection:**
- Order book imbalance (buy vs sell pressure)
- Support/resistance level detection
- Liquidity analysis

---

## 2. Kraken REST API (Supplemental Data)

### 2.1 OHLC Historical Data

**Endpoint:** `GET /public/OHLC`

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `pair` | string | Asset pair (e.g., "XRPUSD") |
| `interval` | int | Timeframe in minutes |
| `since` | int | Unix timestamp for start |

**Limitations:**
- Returns up to 720 most recent entries
- Older data not retrievable via REST

**Intervals Supported:**
- 1, 5, 15, 30, 60, 240, 1440, 10080, 21600 minutes

**Usage:**
- Bootstrap historical data on startup
- Fill gaps after reconnection

### 2.2 Ticker Information

**Endpoint:** `GET /public/Ticker`

**Data Returned:**
- Ask price and volume
- Bid price and volume
- Last trade closed
- 24h volume
- 24h VWAP
- Number of trades (24h)
- 24h low/high
- Today's opening price

**Usage:**
- Daily context for regime detection
- Volume comparison to 24h average

### 2.3 System Status

**Endpoint:** `GET /public/SystemStatus`

**Usage:**
- Verify API availability before external calls

---

## 3. External Free APIs

### 3.1 Alternative.me Fear & Greed Index

**API Endpoint:** `https://api.alternative.me/fng/`

**Method:** GET

**Rate Limit:** Not documented (appears generous)

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | int | 1 | Number of entries (0 = all) |
| `format` | string | json | Response format (json/csv) |
| `date_format` | string | unix | Date format (us/cn/kr/world/unix) |

**Response Example:**
```json
{
  "name": "Fear and Greed Index",
  "data": [
    {
      "value": "42",
      "value_classification": "Fear",
      "timestamp": "1702598400",
      "time_until_update": "12345"
    }
  ],
  "metadata": {
    "error": null
  }
}
```

**Value Classifications:**
| Value Range | Classification |
|-------------|---------------|
| 0-24 | Extreme Fear |
| 25-44 | Fear |
| 45-55 | Neutral |
| 56-75 | Greed |
| 76-100 | Extreme Greed |

**Index Components:**
1. **Volatility (25%)** - Current vs 30/90 day averages
2. **Market Momentum/Volume (25%)** - Current vs 30/90 day averages
3. **Social Media (15%)** - Twitter/Reddit sentiment
4. **Surveys (15%)** - Weekly polls
5. **Bitcoin Dominance (10%)** - Market share analysis
6. **Google Trends (10%)** - Search interest

**Implementation:**
```python
async def fetch_fear_greed():
    async with aiohttp.ClientSession() as session:
        async with session.get('https://api.alternative.me/fng/') as resp:
            data = await resp.json()
            return {
                'value': int(data['data'][0]['value']),
                'classification': data['data'][0]['value_classification'],
                'timestamp': int(data['data'][0]['timestamp'])
            }
```

**Caching Strategy:**
- Updates once per day
- Cache for 5-15 minutes to reduce calls

---

### 3.2 CoinGecko API (Global Market Data)

**Base URL:** `https://api.coingecko.com/api/v3`

**Rate Limit (Demo/Free):**
- 30 calls/minute
- 10,000 calls/month

#### Global Market Data

**Endpoint:** `/global`

**Response Fields:**
```json
{
  "data": {
    "active_cryptocurrencies": 15234,
    "total_market_cap": {
      "btc": 45678901,
      "usd": 3123456789012
    },
    "total_volume": {
      "usd": 123456789012
    },
    "market_cap_percentage": {
      "btc": 56.74,
      "eth": 12.34,
      "xrp": 1.23
    },
    "market_cap_change_percentage_24h_usd": 2.45,
    "updated_at": 1702598400
  }
}
```

**Key Data for Regime Detection:**
- `market_cap_percentage.btc` - BTC Dominance
- `market_cap_change_percentage_24h_usd` - Market momentum
- `total_volume` - Overall market activity

**Implementation:**
```python
async def fetch_global_market():
    async with aiohttp.ClientSession() as session:
        async with session.get('https://api.coingecko.com/api/v3/global') as resp:
            data = await resp.json()
            return {
                'btc_dominance': data['data']['market_cap_percentage']['btc'],
                'eth_dominance': data['data']['market_cap_percentage']['eth'],
                'market_cap_change_24h': data['data']['market_cap_change_percentage_24h_usd'],
                'updated_at': data['data']['updated_at']
            }
```

#### Simple Price (Backup)

**Endpoint:** `/simple/price`

**Parameters:**
- `ids`: Comma-separated coin IDs
- `vs_currencies`: Target currencies
- `include_market_cap`: Boolean
- `include_24hr_vol`: Boolean
- `include_24hr_change`: Boolean

**Usage:** Backup price source if Kraken connection fails

---

### 3.3 CoinyBubble Sentiment API

**API Endpoint:** `https://api.coinybubble.com/` (verify current endpoint)

**Features:**
- Free, no registration
- Real-time sentiment index
- Similar methodology to Binance F&G

**Limitations:**
- Less documentation than Alternative.me
- Newer service, less proven reliability

**Usage:** Alternative to Alternative.me if rate limited

---

### 3.4 CryptoCompare API (Optional)

**Base URL:** `https://min-api.cryptocompare.com`

**Free Tier:**
- 100,000 calls/month
- Rate limit: 50 calls/second

**Useful Endpoints:**
- `/data/blockchain/hashing/hashrate/latest` - Mining data
- `/data/social/coin/latest` - Social metrics
- `/data/exchange/histohour` - Exchange volume

**Usage:** Supplemental data for advanced regime detection

---

## 4. Data Integration Architecture

### 4.1 Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      DATA SOURCES                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ Kraken WS   │  │ Alternative │  │ CoinGecko           │ │
│  │ (Primary)   │  │ (F&G Index) │  │ (BTC Dom, Global)   │ │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
│         │                │                    │            │
└─────────┼────────────────┼────────────────────┼────────────┘
          │                │                    │
          ▼                ▼                    ▼
┌─────────────────────────────────────────────────────────────┐
│                    DATA AGGREGATOR                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────┐  ┌──────────────────────────────────┐│
│  │ Price/Volume     │  │ External Data Cache              ││
│  │ (Real-time)      │  │ (5-min TTL)                      ││
│  │                  │  │                                  ││
│  │ • Latest price   │  │ • Fear & Greed: 42 (Fear)       ││
│  │ • 1m candles     │  │ • BTC Dominance: 56.7%          ││
│  │ • 5m candles     │  │ • Last updated: 5 min ago       ││
│  │ • Order book     │  │                                  ││
│  └──────────────────┘  └──────────────────────────────────┘│
│                                                             │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   REGIME DETECTOR                           │
│                                                             │
│  Combined DataSnapshot + External Sentiment                 │
│        ↓                                                    │
│  Indicator Calculations                                     │
│        ↓                                                    │
│  Composite Score                                            │
│        ↓                                                    │
│  RegimeSnapshot                                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Caching Strategy

| Data Source | Update Frequency | Cache TTL | Fallback |
|-------------|-----------------|-----------|----------|
| Kraken WS Trades | Real-time | None | None (essential) |
| Kraken WS Ticker | Real-time | None | REST API |
| Fear & Greed | Daily | 5-15 min | Last known value |
| BTC Dominance | Every few min | 5 min | Last known value |
| Global Market Cap | Every few min | 5 min | Last known value |

### 4.3 Error Handling

```python
class ExternalDataFetcher:
    async def fetch_with_fallback(self):
        try:
            fresh_data = await self._fetch_all()
            self._update_cache(fresh_data)
            return fresh_data
        except aiohttp.ClientError as e:
            logger.warning(f"External API error: {e}")
            if self._cache is not None:
                logger.info("Using cached external data")
                return self._cache
            else:
                logger.warning("No cache available, using defaults")
                return self._default_sentiment()
        except asyncio.TimeoutError:
            logger.warning("External API timeout")
            return self._cache or self._default_sentiment()

    def _default_sentiment(self):
        """Neutral defaults when external data unavailable."""
        return ExternalSentiment(
            fear_greed_value=50,
            fear_greed_classification="Neutral",
            btc_dominance=55.0,  # Approximate historical average
            last_updated=datetime.utcnow()
        )
```

---

## 5. Data Requirements Summary

### Minimum Required Data (Phase 1)
- Kraken WebSocket: trades, ticker, OHLC (1m)
- Local indicators: ADX, RSI, CHOP, MA calculations

### Recommended Data (Phase 2)
- Kraken OHLC: Multiple timeframes (5m, 15m, 1h)
- Fear & Greed Index (Alternative.me)

### Full Implementation (Phase 3)
- BTC Dominance (CoinGecko)
- Global market data
- Extended timeframes (4h, daily)

---

## 6. API Rate Limit Management

### Request Budget (per minute)

| API | Limit | Planned Usage | Margin |
|-----|-------|---------------|--------|
| Kraken WS | Unlimited | N/A (push) | N/A |
| Kraken REST | 15/min | 1-2/min | 13+ |
| Alternative.me | ~30/min* | 1/5min | 29+ |
| CoinGecko Demo | 30/min | 1/5min | 29+ |

*Estimated based on typical free API limits

### Implementation

```python
class RateLimiter:
    def __init__(self, calls_per_minute: int):
        self.calls_per_minute = calls_per_minute
        self.call_times: deque = deque(maxlen=calls_per_minute)
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = time.time()
            # Remove calls older than 1 minute
            while self.call_times and now - self.call_times[0] > 60:
                self.call_times.popleft()

            if len(self.call_times) >= self.calls_per_minute:
                wait_time = 60 - (now - self.call_times[0])
                await asyncio.sleep(wait_time)

            self.call_times.append(now)
```

---

## 7. Future Data Source Opportunities

### 7.1 On-Chain Data
- **Glassnode API** (paid) - HODL waves, exchange flows
- **IntoTheBlock** - Large transaction analysis
- **Whale Alert API** - Large transfer monitoring

### 7.2 Social Sentiment
- **LunarCrush** - Social engagement metrics
- **Santiment** - Developer activity, social volume

### 7.3 Derivatives Data
- **Coinglass** - Funding rates, open interest
- **Laevitas** - Options data

### 7.4 Alternative Free Sources
- **CoinMarketCap** - Market data (limited free tier)
- **Messari** - Fundamental metrics

---

*Version: 1.0.0 | Created: 2025-12-15*
