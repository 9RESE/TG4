# Kraken WebSocket Integration + HFT Strategies Implementation Plan

**Version:** 2.0 (Revised)
**Date:** 2025-12-13
**Status:** Planning
**Last Updated:** 2025-12-13 - Architecture review and integration fixes

---

## Executive Summary

This document outlines the implementation plan for adding real-time WebSocket data feeds and 3 High-Frequency Trading (HFT) strategies to the existing trading bot system.

### Key Decisions
- **Live data feeds** with **paper trading execution** (real market data, simulated fills)
- **Kraken only** for triangular arbitrage (internal pairs)
- **Best effort latency** optimization (target: 100-500ms)
- **REVISED**: Extend UnifiedOrchestrator instead of parallel HFTOrchestrator
- **REVISED**: Add `hft` mode to unified_trader.py CLI

### New Strategies (3, not 4)
| Strategy | Type | Interval | Description |
|----------|------|----------|-------------|
| Market Making | Tick-level | Real-time | Bid/ask spread capture with inventory management |
| Order Flow | Tick-level | Real-time | Trade tape imbalance detection |
| Triangular Arb (Enhanced) | Tick-level | Real-time | **Upgrade existing** `triangular_arb` to use WS |

> **Note**: `scalping_momentum` removed - overlaps with existing `intraday_scalper` and `scalping_1m5m`

---

## CRITICAL REVIEW: Issues with Original Plan

### Problem 1: Dual Orchestrator Complexity
**Original**: Run HFTOrchestrator alongside UnifiedOrchestrator
**Issue**: Two systems managing same Portfolio = race conditions, duplicate logic
**Solution**: Extend UnifiedOrchestrator with WebSocket data source

### Problem 2: No unified_trader.py Integration
**Original**: Example shows theoretical `main.py` integration
**Issue**: Doesn't integrate with existing CLI (`paper`, `dual`, `experiment` modes)
**Solution**: Add `hft` mode to unified_trader.py

### Problem 3: Strategy Duplication
**Original**: 4 new strategies including `triangular_arb` and `scalping_momentum`
**Issue**: `triangular_arb` already exists in registry; scalping overlaps with existing
**Solution**: Upgrade existing triangular_arb, skip scalping_momentum

### Problem 4: Missing Thread Safety Implementation
**Original**: Mentions adding RLock to Portfolio
**Issue**: No actual implementation shown
**Solution**: Detailed implementation below

---

## System Architecture (REVISED)

### Current State (Synchronous REST Polling)
```
DataFetcher (REST/CCXT) ─► Dict[str, DataFrame] ─► Strategies ─► Orchestrator
     │
     └── 60s cache TTL, blocking calls
```

### Target State (Extended Orchestrator with Pluggable Data Sources)
```
┌─────────────────────────────────────────────────────────────────────────┐
│                          UnifiedOrchestrator (Extended)                  │
│                                                                          │
│  ┌──────────────┐     ┌──────────────┐                                  │
│  │ REST Mode    │     │ WebSocket    │ ◄── Pluggable data source        │
│  │ (DataFetcher)│ OR  │ Mode         │                                  │
│  └──────────────┘     │ (RealtimeData│                                  │
│                       │  Manager)    │                                  │
│                       └──────────────┘                                  │
│                              │                                          │
│                              ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  Strategies (30+ existing + 2 new HFT)                           │  │
│  │  ├─ General: defensive_yield, mean_reversion_vwap, etc.         │  │
│  │  ├─ Grid: grid_arithmetic, grid_geometric, etc.                 │  │
│  │  ├─ Scalper: intraday_scalper, ema9_scalper, scalping_1m5m      │  │
│  │  ├─ Margin: grid_trend_margin, grid_dual_hedge, grid_liq_hunter │  │
│  │  ├─ HFT (NEW): hft_market_making, hft_order_flow                │  │
│  │  └─ Arb (UPGRADE): triangular_arb (now uses WS orderbook)       │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                              │                                          │
│                              ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  Execution Layer                                                  │  │
│  │  ├─ Executor (paper trading with limit orders)                   │  │
│  │  ├─ Portfolio (with RLock for thread safety)                     │  │
│  │  └─ RiskManager (unified + HFT profiles)                         │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Why This is Better
1. **Single source of truth** - One orchestrator, one portfolio state
2. **Simpler code paths** - No thread coordination between orchestrators
3. **Reuses existing infrastructure** - Isolated portfolios, regime detection, logging
4. **Easy testing** - Same CLI commands with different data source

---

## File Structure (REVISED)

### New Files to Create

```
src/
├── realtime/                           # Real-Time Data Layer
│   ├── __init__.py
│   ├── kraken_ws_client.py            # WebSocket client (combined)
│   ├── data_manager.py                # Central data coordinator
│   ├── ohlc_builder.py                # Build candles from trades
│   └── orderbook.py                   # L2 orderbook state
│
├── strategies/hft/                     # HFT Strategies (2 new)
│   ├── __init__.py
│   ├── base_hft.py                    # Base class with RT data access
│   ├── market_making.py               # Spread capture strategy
│   └── order_flow.py                  # Trade tape strategy
```

> **Removed**: `hft_orchestrator.py`, `risk_manager_hft.py`, `scalping_momentum.py`, `triangular_arb.py`
> **Reason**: Use extended UnifiedOrchestrator; existing risk_manager; existing strategies

### Files to Modify

| File | Changes |
|------|---------|
| `src/unified_trader.py` | Add `hft` mode to CLI |
| `src/unified_orchestrator.py` | Add `use_websocket` parameter, HFT loop |
| `src/strategy_registry.py` | Register 2 new HFT strategies |
| `src/strategies/triangular_arb/strategy.py` | Upgrade to use WS orderbook |
| `src/risk_manager.py` | Add HFT risk profiles |
| `src/portfolio.py` | Add thread-safe locking (RLock) |
| `requirements.txt` | Add python-kraken-sdk, websockets |

---

## Component Specifications

### 1. WebSocket Client (`kraken_ws_client.py`)

**Purpose:** Maintain persistent WebSocket connection to Kraken API v2

**Key Features:**
- Uses `python-kraken-sdk` SpotWSClient
- Subscribes to: `trade`, `ticker`, `book` channels
- Automatic reconnection with exponential backoff
- Callback-based message delivery

```python
class KrakenWSClient:
    WS_URL = "wss://ws.kraken.com/v2"

    def __init__(self, symbols: List[str], callbacks: Dict[str, Callable]):
        self.symbols = symbols  # ['XRP/USD', 'BTC/USD', 'XRP/BTC']
        self.on_trade = callbacks.get('trade')
        self.on_ticker = callbacks.get('ticker')
        self.on_book = callbacks.get('book')

    async def connect(self) -> None
    async def subscribe(self, channels: List[str]) -> None
    async def run_forever(self) -> None  # Main event loop
    async def close(self) -> None
```

**Reconnection Logic:**
- Exponential backoff: 1s, 2s, 4s, 8s... max 30s
- Ping/pong heartbeat every 30s
- Auto-resubscribe on reconnect
- State recovery from REST API if needed

### 2. OHLC Builder (`ohlc_builder.py`)

**Purpose:** Aggregate trade stream into OHLC candles in real-time

**Algorithm:**
1. Each trade updates current candle: `high = max(high, price)`, `low = min(low, price)`, `close = price`, `volume += qty`
2. At interval boundary (60s/300s), close candle and start new one
3. Maintain rolling window of last 500 candles

```python
class OHLCBuilder:
    def __init__(self, symbol: str, interval_seconds: int = 60):
        self.current_candle: Dict = None
        self.candles: List[Dict] = []  # Rolling window

    def on_trade(self, price: float, volume: float, timestamp: datetime) -> None
    def to_dataframe(self) -> pd.DataFrame  # Compatible with existing strategies
```

### 3. OrderBook (`orderbook.py`)

**Purpose:** Maintain L2 order book state from WebSocket updates

**Features:**
- Sorted bid/ask levels using `sortedcontainers.SortedDict`
- Checksum validation (Kraken CRC32)
- Fast spread/mid calculations

```python
class OrderBook:
    def __init__(self, symbol: str, depth: int = 25):
        self.bids: SortedDict = SortedDict()  # price -> size
        self.asks: SortedDict = SortedDict()

    def apply_snapshot(self, bids: List, asks: List) -> None
    def apply_update(self, bids: List, asks: List) -> None

    @property
    def spread(self) -> float  # best_ask - best_bid
    def get_mid_price(self) -> float
    def get_depth(self, levels: int = 5) -> Tuple[List, List]
```

### 4. Realtime Data Manager (`data_manager.py`)

**Purpose:** Thread-safe bridge between WebSocket and strategies

```python
class RealtimeDataManager:
    def __init__(self, symbols: List[str]):
        self.ohlc_builders: Dict[str, Dict[str, OHLCBuilder]] = {}
        self.orderbooks: Dict[str, OrderBook] = {}
        self.trade_tape: Dict[str, deque] = {}  # Last 100 trades
        self._lock = threading.RLock()

    # Callbacks for WebSocket
    def on_trade(self, msg: TradeMessage) -> None
    def on_ticker(self, msg: TickerMessage) -> None
    def on_book_update(self, msg: OrderBookUpdate) -> None

    # Strategy access (thread-safe)
    def get_data(self) -> Dict[str, pd.DataFrame]  # Same format as DataFetcher
    def get_orderbook(self, symbol: str) -> OrderBook
    def get_trade_tape(self, symbol: str, n: int = 100) -> List[TradeMessage]
```

---

## Strategy Implementations

### 1. Market Making Strategy

**File:** `src/strategies/hft/market_making.py`

**Logic:**
1. Calculate fair value from orderbook mid-price
2. Place quotes at `mid ± spread/2`
3. Apply inventory skew: if long, lower ask to encourage sells
4. Cancel and requote on 0.05% price move

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_spread_bps` | 15 | Base spread in basis points (0.15%) |
| `inventory_skew_factor` | 0.5 | How much to skew based on position |
| `max_position` | 500 | Max inventory in USD |
| `requote_threshold_bps` | 5 | Price move that triggers requote |

**Signal Format:**
```python
{
    'action': 'quote',
    'symbol': 'XRP/USD',
    'bid_price': 2.3450,
    'ask_price': 2.3485,
    'bid_size': 100,
    'ask_size': 100,
    'cancel_ids': ['order123', 'order124'],
    'strategy': 'hft_market_making'
}
```

### 2. Scalping Momentum Strategy

**File:** `src/strategies/hft/scalping_momentum.py`

**Logic:**
1. Monitor 1m candles built from trade stream
2. Entry: Price breaks 5-bar high/low + volume spike (2x average)
3. Quick exit: 0.4% target, 0.3% stop

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `breakout_lookback` | 5 | Candles for range calculation |
| `volume_spike_mult` | 2.0 | Volume must exceed this × average |
| `target_pct` | 0.004 | Take profit at 0.4% |
| `stop_pct` | 0.003 | Stop loss at 0.3% |
| `leverage` | 3 | Default leverage |

### 3. Triangular Arbitrage Strategy

**File:** `src/strategies/hft/triangular_arb.py`

**Logic:**
1. Monitor three pairs: XRP/USD, BTC/USD, XRP/BTC
2. Calculate forward path: USD → XRP → BTC → USD (using best bid/ask)
3. Calculate reverse path: USD → BTC → XRP → USD
4. Execute if profit > 5bps after fees (3 × 0.16% = 0.48%)

**Forward Path Calculation:**
```
1 USD → (1 / XRP_ask) XRP → (XRP × XRP_BTC_bid) BTC → (BTC × BTC_USD_bid) USD
Profit = final_USD - 1.0
```

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_profit_bps` | 5 | Minimum profit to execute (0.05%) |
| `fee_per_trade` | 0.0016 | Fee per leg (0.16%) |
| `trade_size_usd` | 100 | Size per arb execution |

### 4. Order Flow Strategy

**File:** `src/strategies/hft/order_flow.py`

**Logic:**
1. Analyze last 100 trades from WebSocket stream
2. Calculate buy/sell volume imbalance
3. Detect large trades (> $10k)
4. Enter on strong imbalance (buy_vol > 2× sell_vol) with large trade confirmation

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `lookback_trades` | 100 | Number of trades to analyze |
| `large_trade_threshold` | 10000 | USD value for "large" trade |
| `imbalance_threshold` | 2.0 | Buy/sell ratio for signal |
| `target_pct` | 0.003 | Take profit 0.3% |
| `stop_pct` | 0.002 | Stop loss 0.2% |

---

## HFT Risk Controls

### Risk Manager (`risk_manager_hft.py`)

**Rate Limiting:**
```python
max_orders_per_second = 2
max_orders_per_minute = 30
```

**Position Limits:**
```python
max_position_per_strategy_usd = 500
max_total_hft_exposure_usd = 2000
```

**Circuit Breakers:**
```python
max_daily_loss_pct = 0.02          # 2% daily loss triggers pause
max_consecutive_losses = 5          # 5 losses in a row triggers pause
cooldown_after_break_seconds = 300  # 5 minute cooldown
```

### Risk Profiles

| Strategy | min_confidence | stop_loss | take_profit | max_hold |
|----------|---------------|-----------|-------------|----------|
| market_making | 0.15 | 0.2% | 0.1% | 3 min |
| scalping_momentum | 0.30 | 0.3% | 0.4% | 5 min |
| triangular_arb | 0.10 | 0.1% | 0.05% | 1 min |
| order_flow | 0.35 | 0.2% | 0.3% | 3 min |

---

## Integration with UnifiedOrchestrator (REVISED)

### Extended UnifiedOrchestrator

**Changes to `src/unified_orchestrator.py`:**

```python
class UnifiedOrchestrator:
    def __init__(self,
                 portfolio: Portfolio,
                 config_path: str = None,
                 experiment_id: str = None,
                 # ... existing params ...
                 use_websocket: bool = False,      # NEW: Enable WS data
                 hft_mode: bool = False,           # NEW: Fast loop (100ms)
                 hft_symbols: List[str] = None):   # NEW: WS symbols

        # ... existing init ...

        # NEW: WebSocket mode
        self.use_websocket = use_websocket
        self.hft_mode = hft_mode
        self.realtime_data: Optional[RealtimeDataManager] = None

        if use_websocket:
            self._init_websocket(hft_symbols or ['XRP/USD', 'BTC/USD', 'XRP/BTC'])

    def _init_websocket(self, symbols: List[str]):
        """Initialize WebSocket data source."""
        from realtime.data_manager import RealtimeDataManager
        self.realtime_data = RealtimeDataManager(symbols)
        # WS connection started in run_loop_hft

    def get_data_source(self) -> Dict[str, pd.DataFrame]:
        """Get data from appropriate source (REST or WebSocket)."""
        if self.use_websocket and self.realtime_data:
            return self.realtime_data.get_data()
        else:
            return self.get_live_data()  # Existing REST method

    async def run_loop_hft(self,
                           duration_minutes: int = 60,
                           interval_ms: int = 100):
        """
        Async HFT trading loop with WebSocket data.

        Uses same strategies, same portfolio, same logging.
        Just faster loop and real-time data.
        """
        if not self.realtime_data:
            raise RuntimeError("WebSocket not initialized. Use use_websocket=True")

        # Start WebSocket connection
        await self.realtime_data.connect()

        print(f"\n{'='*60}")
        print(f"UNIFIED ORCHESTRATOR - HFT Mode (WebSocket)")
        print(f"Duration: {duration_minutes} min | Interval: {interval_ms}ms")
        print(f"Experiment ID: {self.experiment_id}")
        print(f"HFT Strategies: {[n for n in self.strategies if n.startswith('hft_')]}")
        print(f"{'='*60}\n")

        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        loop_count = 0

        try:
            while time.time() < end_time:
                loop_count += 1

                # Get real-time data
                data = self.realtime_data.get_data()

                # Only run HFT-category strategies in fast loop
                hft_decisions = self.decide_independent_hft(data)

                for decision in hft_decisions:
                    result = self.execute_paper(decision)
                    if result['executed']:
                        strat = decision.get('strategy', '')
                        print(f"[HFT {loop_count}] {strat}: {decision['action']} "
                              f"@ ${result.get('price', 0):.4f}")

                await asyncio.sleep(interval_ms / 1000)

        except KeyboardInterrupt:
            print("\nStopping HFT loop...")
        finally:
            await self.realtime_data.close()
            self.close()

    def decide_independent_hft(self, data: Dict) -> List[Dict]:
        """Get decisions from HFT strategies only."""
        decisions = []
        for name, strategy in self.strategies.items():
            # Only HFT strategies in fast loop
            if not name.startswith('hft_') and name != 'triangular_arb':
                continue
            try:
                signal = strategy.generate_signals(data)
                if signal and signal.get('action') != 'hold':
                    decisions.append({**signal, 'strategy': name})
            except Exception as e:
                print(f"HFT signal error {name}: {e}")
        return decisions
```

---

## unified_trader.py CLI Integration (NEW)

### Add `hft` Command

**Changes to `src/unified_trader.py`:**

```python
def cmd_hft(args):
    """Run HFT mode with WebSocket data feeds."""
    print("\n" + "="*70)
    print("UNIFIED TRADING PLATFORM - HFT Mode (WebSocket)")
    print("="*70)

    # Load config
    config_path = args.config or DEFAULT_CONFIG_PATH
    if not os.path.exists(config_path):
        print(f"Config not found: {config_path}")
        return

    with open(config_path) as f:
        config = yaml.safe_load(f)

    starting_balance = config.get('global', {}).get('starting_balance', {
        'USDT': 5000.0,
        'XRP': 0.0,
        'BTC': 0.0
    })

    portfolio = Portfolio(starting_balance)
    experiment_id = args.experiment_id or f"hft_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Create orchestrator with WebSocket mode
    orchestrator = UnifiedOrchestrator(
        portfolio=portfolio,
        config_path=config_path,
        experiment_id=experiment_id,
        use_websocket=True,
        hft_mode=True,
        hft_symbols=['XRP/USD', 'BTC/USD', 'XRP/BTC']
    )

    print(f"\nHFT Strategies Enabled:")
    hft_strats = [n for n in orchestrator.strategies if n.startswith('hft_') or n == 'triangular_arb']
    for name in hft_strats:
        print(f"  + {name}")

    print(f"\nExperiment ID: {experiment_id}")
    print("="*70 + "\n")

    # Run async HFT loop
    import asyncio
    asyncio.run(orchestrator.run_loop_hft(
        duration_minutes=args.duration,
        interval_ms=args.interval_ms
    ))


# Add to argparse in main():
hft_parser = subparsers.add_parser('hft', help='Run HFT mode with WebSocket data')
hft_parser.add_argument('--config', type=str, help='Path to config file')
hft_parser.add_argument('--duration', type=int, default=60, help='Duration in minutes')
hft_parser.add_argument('--interval-ms', type=int, default=100, help='Loop interval in ms')
hft_parser.add_argument('--experiment-id', type=str, help='Custom experiment ID')

# Add to command dispatch:
elif args.command == 'hft':
    cmd_hft(args)
```

### Usage Examples

```bash
# Run HFT mode with default settings (100ms loop, 60 min duration)
python unified_trader.py hft

# Run HFT with custom interval and duration
python unified_trader.py hft --interval-ms 200 --duration 120

# Run HFT with custom experiment ID
python unified_trader.py hft --experiment-id hft_test_001
```

---

## Thread Safety Implementation (DETAILED)

### Portfolio Modifications (`src/portfolio.py`)

```python
import threading
from typing import Dict

class Portfolio:
    """Thread-safe portfolio manager."""

    def __init__(self, starting_balances: Dict[str, float]):
        self._lock = threading.RLock()  # NEW: Thread safety

        self.balances = {
            'BTC': starting_balances.get('BTC', 0.0),
            'XRP': starting_balances.get('XRP', 0.0),
            'USDT': starting_balances.get('USDT', 0.0)
        }
        self.history = []
        self.margin_positions = {}
        # ... rest of init ...

    def update(self, asset: str, amount: float):
        """Thread-safe balance update."""
        with self._lock:  # NEW: Lock
            if asset not in ['BTC', 'XRP', 'USDT']:
                return
            self.balances[asset] = self.balances.get(asset, 0.0) + amount
            if self.balances[asset] < 1e-8:
                self.balances[asset] = 0.0

    def get_total_usd(self, prices: Dict[str, float]) -> float:
        """Thread-safe total calculation."""
        with self._lock:  # NEW: Lock
            spot_value = sum(self.balances.get(a, 0) * prices.get(a, 1.0)
                           for a in self.balances)
            # ... margin P&L calculation ...
            return spot_value + margin_pnl

    def open_margin_position(self, asset: str, usdt_collateral: float,
                            leverage: float, price: float, direction: str = 'long'):
        """Thread-safe margin position opening."""
        with self._lock:  # NEW: Lock
            # ... existing logic ...

    def close_margin_position(self, asset: str, price: float):
        """Thread-safe margin position closing."""
        with self._lock:  # NEW: Lock
            # ... existing logic ...
```

---

## Dependencies

### New Python Packages

```bash
pip install python-kraken-sdk>=3.2.7   # Kraken WebSocket API v2
pip install websockets>=12.0            # WebSocket support
pip install sortedcontainers>=2.4.0     # Efficient sorted dict for orderbook
```

### requirements.txt Addition

```
# HFT / WebSocket
python-kraken-sdk>=3.2.7
websockets>=12.0
sortedcontainers>=2.4.0
```

---

## Implementation Timeline (REVISED)

| Phase | Tasks | Files | Complexity |
|-------|-------|-------|------------|
| **1. Thread Safety** | Add RLock to Portfolio | `portfolio.py` | Low |
| **2. Real-Time Data Layer** | WS client, OHLC builder, orderbook | `realtime/*.py` | Medium |
| **3. HFT Strategies** | Market Making, Order Flow | `strategies/hft/*.py` | Medium |
| **4. Orchestrator Extension** | Add WS mode, HFT loop | `unified_orchestrator.py` | Medium |
| **5. CLI Integration** | Add `hft` command | `unified_trader.py` | Low |
| **6. Triangular Arb Upgrade** | Use WS orderbook | `triangular_arb/strategy.py` | Low |
| **7. Testing** | Paper trade 24-48h | N/A | N/A |

### Implementation Order (Efficient Path)

```
Day 1: Thread Safety + Real-Time Data Layer skeleton
       - Add RLock to Portfolio (30 min)
       - Create realtime/ module structure
       - Implement KrakenWSClient with python-kraken-sdk
       - Basic OHLCBuilder and OrderBook classes

Day 2: Complete Real-Time Data Layer
       - RealtimeDataManager with thread-safe access
       - Integration tests with live WS feed

Day 3: HFT Strategies
       - BaseHFT class (extends BaseStrategy)
       - Market Making strategy
       - Order Flow strategy

Day 4: Orchestrator + CLI
       - Extend UnifiedOrchestrator with use_websocket
       - Add run_loop_hft async method
       - Add hft command to unified_trader.py

Day 5: Triangular Arb + Testing
       - Upgrade triangular_arb to use WS orderbook
       - End-to-end paper trading test
```

**Revised Total: ~5 days** (reduced from 12 days)

---

## Testing Strategy

### Unit Tests
- OHLC builder: candle generation, boundary handling
- OrderBook: snapshot, updates, spread calculation
- Risk limits: rate limiting, circuit breakers

### Integration Tests
- WebSocket connection and reconnection
- Message flow: WS → DataManager → Strategy
- Thread safety: concurrent portfolio access

### Paper Trading Tests
- Run all 4 strategies with live feeds for 24-48 hours
- Monitor for: signal quality, latency, position tracking

### Latency Profiling
- Measure: WebSocket message → signal generation → order placement
- Target: < 500ms end-to-end

---

## Monitoring & Logging

### Key Metrics to Track
- WebSocket latency (message receive time)
- Signal generation latency
- Fill rate (limit orders)
- Spread capture (market making)
- Arb opportunity count and success rate
- Order flow prediction accuracy

### Log Levels
- `DEBUG`: All WebSocket messages, calculations
- `INFO`: Signals generated, orders placed
- `WARNING`: Rate limits hit, reconnections
- `ERROR`: Execution failures, circuit breakers

---

## Future Enhancements

1. **Live Trading**: Add real order execution via Kraken REST API
2. **Multi-Exchange Arb**: Add Binance/Coinbase for cross-exchange arb
3. **ML Order Flow**: Train model on historical trade tape data
4. **Latency Optimization**: Move to co-located servers if profitable

---

## References

- [python-kraken-sdk Documentation](https://python-kraken-sdk.readthedocs.io/)
- [Kraken WebSocket API v2](https://docs.kraken.com/websockets-v2/)
- [Kraken REST API](https://docs.kraken.com/api/)

---

## REVISION SUMMARY (v2.0)

### Key Changes from v1.0

| Aspect | v1.0 (Original) | v2.0 (Revised) |
|--------|-----------------|----------------|
| **Architecture** | Separate HFTOrchestrator | Extended UnifiedOrchestrator |
| **New Strategies** | 4 | 2 (+ 1 upgrade) |
| **Implementation Time** | ~12 days | ~5 days |
| **CLI Integration** | None | `hft` command in unified_trader.py |
| **Thread Safety** | Mentioned | Detailed implementation |
| **Complexity** | High (dual orchestrators) | Medium (single extended) |

### Benefits of Revised Approach

1. **50% Less Code** - No separate HFTOrchestrator, no parallel systems
2. **Reuses Infrastructure** - Isolated portfolios, logging, diagnostics all work
3. **Simpler Testing** - Same CLI, just `python unified_trader.py hft`
4. **No Race Conditions** - Single orchestrator manages all execution
5. **Easy Rollback** - If HFT underperforms, just use `paper` or `dual` modes

### What Stays the Same

- WebSocket infrastructure (realtime/ module)
- OHLC builder and OrderBook components
- Market Making and Order Flow strategy logic
- Risk profiles and rate limiting

### Next Steps

1. **Approve revised plan** - Confirm architecture change is acceptable
2. **Start Phase 1** - Thread safety is quick win, do it first
3. **Implement incrementally** - Test each component before moving on
4. **24-48h paper test** - Run HFT mode with isolated portfolio before live

---

*Document revised 2025-12-13 after architecture review*
