# TripleGain User Interface Requirements

**Document Version**: 1.0
**Status**: Design Phase

---

## 1. Dashboard Overview

### 1.1 Dashboard Layout

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                             TRIPLEGAIN TRADING DASHBOARD                                  │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────────────────┐    │
│  │ HEADER: System Status Bar                                                        │    │
│  │ [System: ONLINE] [Exchange: CONNECTED] [Agents: 6/6] [Last Update: 10:30:45]   │    │
│  └─────────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                          │
│  ┌──────────────────────────────┐  ┌────────────────────────────────────────────────┐  │
│  │ PORTFOLIO SUMMARY            │  │ ALLOCATION VISUALIZATION                        │  │
│  │                              │  │                                                  │  │
│  │ Total Equity: $2,145.32     │  │  ┌────────┐                                     │  │
│  │ Daily P&L: +$45.12 (+2.1%)  │  │  │ BTC    │  35%  ████████████                  │  │
│  │ Available Margin: $1,450    │  │  │ XRP    │  32%  ██████████                    │  │
│  │ Unrealized P&L: +$28.50     │  │  │ USDT   │  33%  ███████████                   │  │
│  │                              │  │  └────────┘                                     │  │
│  │ Drawdown: 3.2%              │  │  Target: 33/33/33 | Deviation: 2%               │  │
│  │ Win Rate (7d): 58%          │  │                                                  │  │
│  └──────────────────────────────┘  └────────────────────────────────────────────────┘  │
│                                                                                          │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐   │
│  │ PRICE CHARTS                                                                      │   │
│  │ ┌─────────────────────────────────────────────────────────────────────────────┐  │   │
│  │ │ [BTC/USDT] [XRP/USDT] [XRP/BTC]  |  [1m] [5m] [15m] [1H] [4H] [1D]         │  │   │
│  │ ├─────────────────────────────────────────────────────────────────────────────┤  │   │
│  │ │                                                                              │  │   │
│  │ │          Interactive TradingView-style Chart                                │  │   │
│  │ │          • Price candles with volume                                        │  │   │
│  │ │          • Indicator overlays (EMA, BB)                                     │  │   │
│  │ │          • Entry/exit markers                                               │  │   │
│  │ │          • Support/resistance levels                                        │  │   │
│  │ │                                                                              │  │   │
│  │ └─────────────────────────────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                          │
│  ┌─────────────────────────────────┐  ┌─────────────────────────────────────────────┐  │
│  │ OPEN POSITIONS                  │  │ AGENT STATUS                                 │  │
│  │                                 │  │                                              │  │
│  │ BTC/USDT LONG 0.05 @ $43,200   │  │ TA Agent      [READY]  Last: 10:30:00       │  │
│  │   P&L: +$85.00 (+3.9%)         │  │ Regime Agent  [READY]  Last: 10:30:00       │  │
│  │   SL: $42,100 | TP: $45,000    │  │ Sentiment     [READY]  Last: 10:15:00       │  │
│  │                                 │  │ Trading Dec.  [READY]  Last: 10:00:00       │  │
│  │ XRP/USDT LONG 500 @ $0.58      │  │ Risk Mgmt     [ACTIVE] Monitoring           │  │
│  │   P&L: +$15.00 (+5.1%)         │  │ Portfolio     [READY]  Last: 10:00:00       │  │
│  │   SL: $0.55 | TP: $0.65        │  │                                              │  │
│  └─────────────────────────────────┘  └─────────────────────────────────────────────┘  │
│                                                                                          │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐   │
│  │ RECENT DECISIONS                                                                  │   │
│  │                                                                                   │   │
│  │ 10:30:12 | HOLD    | BTC/USDT | Conf: 0.72 | "Uptrend intact, maintain long"   │   │
│  │ 10:15:45 | BUY     | XRP/USDT | Conf: 0.68 | "Breakout confirmed, enter long"  │   │
│  │ 09:00:00 | REBAL   | PORTFOLIO| Conf: N/A  | "Deviation 5.2% - rebalancing"    │   │
│  │ 08:30:22 | SELL    | BTC/USDT | Conf: 0.65 | "TP hit at $43,500"               │   │
│  │                                                                                   │   │
│  │ [View All Decisions]                                                              │   │
│  └──────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐    │
│  │ MANUAL CONTROLS                                                                  │    │
│  │                                                                                  │    │
│  │ [PAUSE TRADING] [CLOSE ALL POSITIONS] [FORCE REBALANCE] [EMERGENCY STOP]       │    │
│  │                                                                                  │    │
│  └─────────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Technology Stack for Dashboard

| Component | Technology | Rationale |
|-----------|------------|-----------|
| **Frontend Framework** | React + TypeScript | Modern, type-safe, component-based |
| **Charting Library** | Lightweight Charts (TradingView) | Professional trading charts |
| **State Management** | Zustand or Jotai | Simple, performant |
| **Real-time Updates** | WebSocket + React Query | Efficient data sync |
| **Styling** | Tailwind CSS | Rapid development, dark mode |
| **Backend** | FastAPI (Python) | Consistent with trading system |

---

## 2. View Specifications

### 2.1 Portfolio Summary View

**Purpose**: At-a-glance portfolio health and performance

**Components**:

| Component | Data Source | Update Frequency |
|-----------|-------------|------------------|
| Total Equity | Portfolio snapshots table | Real-time |
| Daily P&L | Calculated from snapshots | Real-time |
| Weekly P&L | Aggregated from daily | Every hour |
| Available Margin | Kraken API | Real-time |
| Unrealized P&L | Position tracker | Real-time |
| Current Drawdown | Calculated from peak | Real-time |
| Win Rate | Trade history (7d rolling) | Every trade close |

**Visual Elements**:
- Large numeric displays with color coding (green/red)
- Sparkline charts for trending metrics
- Warning icons when approaching limits

### 2.2 Allocation Visualization

**Purpose**: Show current vs target portfolio allocation

**Display Requirements**:
```
┌─────────────────────────────────────────────────────────────────┐
│ PORTFOLIO ALLOCATION                                             │
│                                                                  │
│ Current Allocation          Target: 33.33% each                 │
│ ┌────────────────────────────────────────────────────────────┐  │
│ │ BTC  [████████████░░░] 35.2%  (+1.9%)  $756.12             │  │
│ │ XRP  [██████████░░░░░] 31.8%  (-1.5%)  $682.45             │  │
│ │ USDT [███████████░░░░] 33.0%  (-0.3%)  $706.75             │  │
│ └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│ Hodl Bags (Excluded from rebalancing)                           │
│ ┌────────────────────────────────────────────────────────────┐  │
│ │ BTC Hodl: 0.0023 BTC ($98.50)                               │  │
│ │ XRP Hodl: 150 XRP ($88.50)                                  │  │
│ └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│ Rebalance Status: [Not needed - deviation < 5%]                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 Price Charts View

**Purpose**: Technical analysis visualization

**Features**:
- Multi-symbol tabs (BTC/USDT, XRP/USDT, XRP/BTC)
- Multiple timeframe selector (1m to 1D)
- Indicator overlays:
  - Moving Averages (EMA 9, 21, 50, 200)
  - Bollinger Bands
  - RSI (separate pane)
  - MACD (separate pane)
  - Volume histogram
- Trade markers:
  - Entry points (green/red arrows)
  - Exit points (with P&L annotation)
  - Stop-loss/take-profit levels (horizontal lines)
- Support/resistance levels (auto-detected)
- Regime indicator band (colored background)

**Interactions**:
- Zoom and pan
- Crosshair with price/time info
- Click on trade markers to see details
- Toggle indicators on/off

### 2.4 Open Positions View

**Purpose**: Real-time position monitoring

**Table Columns**:
| Column | Description |
|--------|-------------|
| Symbol | Trading pair |
| Side | LONG / SHORT |
| Size | Position size |
| Entry Price | Average entry |
| Current Price | Real-time |
| P&L ($) | Unrealized profit/loss |
| P&L (%) | Percentage return |
| Stop Loss | Current SL price |
| Take Profit | Current TP price |
| Duration | Time since entry |
| Actions | [Close] [Modify] |

**Row Styling**:
- Green background tint for profitable
- Red background tint for losing
- Pulsing animation when near SL/TP

### 2.5 Agent Status View

**Purpose**: Monitor agent health and decisions

**Agent Status Card**:
```
┌─────────────────────────────────────────────┐
│ TECHNICAL ANALYSIS AGENT                     │
│                                              │
│ Status: [●] READY                            │
│ Last Execution: 10:30:00 (30s ago)          │
│ Latency: 185ms                               │
│ Model: Qwen 2.5 7B (Local)                  │
│                                              │
│ Last Output Summary:                         │
│ • Trend: Bullish (0.72 confidence)          │
│ • RSI: 58.3 (Neutral)                       │
│ • Regime: Trending Bull                      │
│                                              │
│ [View Full Output] [View History]            │
└─────────────────────────────────────────────┘
```

**Agent Status Indicators**:
| Status | Meaning | Color |
|--------|---------|-------|
| READY | Waiting for trigger | Green |
| RUNNING | Currently executing | Blue (animated) |
| ERROR | Last execution failed | Red |
| TIMEOUT | Response exceeded limit | Orange |
| DISABLED | Manually disabled | Gray |

### 2.6 Decision Log View

**Purpose**: Audit trail of all trading decisions

**Log Entry Format**:
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Decision ID: dec_20251218_103012_001                                         │
│ Timestamp: 2025-12-18 10:30:12 UTC                                          │
│                                                                              │
│ ACTION: BUY XRP/USDT                                                         │
│ Confidence: 0.68                                                             │
│ Parameters:                                                                  │
│   Entry: $0.62 (Limit)                                                      │
│   Size: 300 XRP ($186)                                                      │
│   Leverage: 2x                                                              │
│   Stop Loss: $0.595 (-4%)                                                   │
│   Take Profit: $0.67 (+8%)                                                  │
│                                                                              │
│ Agent Inputs:                                                                │
│   TA: Bullish (0.72) - "Breakout above resistance"                          │
│   Regime: Trending Bull (0.78)                                              │
│   Sentiment: Neutral (0.52)                                                 │
│                                                                              │
│ Risk Evaluation:                                                             │
│   Status: APPROVED                                                           │
│   Risk per trade: 1.1% of equity                                            │
│   R:R Ratio: 2.0                                                            │
│                                                                              │
│ Reasoning:                                                                   │
│ "Technical breakout confirmed with bullish regime. Sentiment neutral        │
│  but not contradicting. Entry at retest of breakout level provides          │
│  favorable risk/reward."                                                     │
│                                                                              │
│ Execution: FILLED @ $0.619 at 10:30:15 UTC                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Filtering Options**:
- By date range
- By symbol
- By action type (BUY, SELL, HOLD, etc.)
- By approval status
- By agent

### 2.7 Manual Override Controls

**Purpose**: Human intervention capabilities

**Control Panel**:
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ MANUAL CONTROLS                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│ TRADING STATE                                                                │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ Current: [ACTIVE]                                                        │ │
│ │                                                                          │ │
│ │ [PAUSE TRADING]    Halt new trades, maintain existing positions         │ │
│ │ [RESUME TRADING]   Re-enable automated trading                          │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│ POSITION MANAGEMENT                                                          │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ [CLOSE ALL POSITIONS]     Immediately close all open positions          │ │
│ │ [CLOSE LOSING POSITIONS]  Close only positions with negative P&L        │ │
│ │ [TIGHTEN ALL STOPS]       Move stops to break-even where possible       │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│ PORTFOLIO                                                                    │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ [FORCE REBALANCE]   Execute rebalancing regardless of threshold         │ │
│ │ [SKIP NEXT REBALANCE]   Defer next scheduled rebalance                  │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│ EMERGENCY                                                                    │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ [⚠️ EMERGENCY STOP]                                                       │ │
│ │ Closes all positions, halts all trading, requires manual restart        │ │
│ │ Use only in critical situations                                          │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│ Confirmation required for all destructive actions                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Confirmation Dialog**:
```
┌─────────────────────────────────────────────────────────┐
│ ⚠️ Confirm Action                                        │
│                                                         │
│ You are about to: CLOSE ALL POSITIONS                   │
│                                                         │
│ This will:                                              │
│ • Close 2 open positions                                │
│ • Realize P&L of approximately +$100.50                │
│ • This action cannot be undone                          │
│                                                         │
│ Type "CONFIRM" to proceed:                              │
│ ┌─────────────────────────────────────────────────────┐│
│ │                                                      ││
│ └─────────────────────────────────────────────────────┘│
│                                                         │
│ [Cancel]                              [Execute]         │
└─────────────────────────────────────────────────────────┘
```

---

## 3. System Health View

### 3.1 Health Dashboard

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ SYSTEM HEALTH                                                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│ CONNECTIONS                                                                  │
│ ┌───────────────────────────────────────────────────────────────────────┐   │
│ │ Kraken WebSocket    [●] Connected    Latency: 45ms    Uptime: 99.9%  │   │
│ │ Kraken REST API     [●] Healthy      Last call: 2s ago               │   │
│ │ TimescaleDB         [●] Connected    Queries/min: 120                │   │
│ │ Redis Cache         [●] Connected    Hit rate: 94%                   │   │
│ │ Ollama (Local LLM)  [●] Ready        Model: Qwen 2.5 7B              │   │
│ │ DeepSeek API        [●] Available    Credits: $0.89 remaining        │   │
│ └───────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│ DATA PIPELINE                                                                │
│ ┌───────────────────────────────────────────────────────────────────────┐   │
│ │ Candle Freshness    [●] OK          Last: 15s ago                    │   │
│ │ Indicator Calc      [●] OK          Avg: 45ms                        │   │
│ │ Data Quality        [●] OK          Warnings: 0                      │   │
│ │ Order Book          [●] OK          Depth: $1.2M                     │   │
│ └───────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│ RISK STATE                                                                   │
│ ┌───────────────────────────────────────────────────────────────────────┐   │
│ │ Daily Loss Limit    [●] OK          Used: 1.2% of 5%                 │   │
│ │ Weekly Loss Limit   [●] OK          Used: 3.5% of 10%                │   │
│ │ Max Drawdown        [●] OK          Current: 3.2% of 20%             │   │
│ │ Consecutive Losses  [●] OK          Current: 1 of 5                  │   │
│ │ Circuit Breakers    [●] None Active                                   │   │
│ └───────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│ RECENT ALERTS                                                                │
│ ┌───────────────────────────────────────────────────────────────────────┐   │
│ │ 10:25:00 [INFO] Rebalancing completed successfully                    │   │
│ │ 09:45:22 [WARN] High volatility detected - leverage reduced           │   │
│ │ 09:00:00 [INFO] Daily reset - counters cleared                        │   │
│ └───────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Alert System

**Alert Levels**:
| Level | Color | Notification |
|-------|-------|--------------|
| INFO | Blue | Dashboard only |
| WARNING | Yellow | Dashboard + Sound |
| ERROR | Red | Dashboard + Sound + Browser notification |
| CRITICAL | Red (pulsing) | All above + Email (optional) |

**Alert Categories**:
- Connection issues
- Data quality problems
- Risk threshold approaching
- Circuit breaker triggered
- Trade execution issues
- Agent errors

---

## 4. Performance Analytics View

### 4.1 Performance Dashboard

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ PERFORMANCE ANALYTICS                                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│ TIME PERIOD: [Today] [7 Days] [30 Days] [90 Days] [All Time]               │
│                                                                              │
│ KEY METRICS                                                                  │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ Total Return    Sharpe Ratio   Max Drawdown   Win Rate   Profit Factor │ │
│ │   +24.5%          1.82           -8.3%         58%          1.75       │ │
│ │   ▲ +2.1%        ▲ +0.12        ▼ improved    ▲ +3%       ▲ +0.08     │ │
│ │   vs last period                                                        │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│ EQUITY CURVE                                                                 │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │                                                     ___                  │ │
│ │                                              ___---'                     │ │
│ │                                        __---'                            │ │
│ │                                  ___--'                                  │ │
│ │                             __--'                                        │ │
│ │                        __--'                                             │ │
│ │                   __--'                                                  │ │
│ │              __--'                                                       │ │
│ │         __--'                                                            │ │
│ │    ___--                                                                 │ │
│ │ ---                                                                      │ │
│ │ $2,000                                                           $2,500 │ │
│ │ Start                                                              Now   │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│ TRADE DISTRIBUTION                                                           │
│ ┌──────────────────────────┐  ┌──────────────────────────────────────────┐ │
│ │ By Symbol                │  │ By Time of Day (UTC)                     │ │
│ │ BTC/USDT: 45 trades     │  │                                          │ │
│ │ XRP/USDT: 38 trades     │  │  ████                                    │ │
│ │ XRP/BTC:  12 trades     │  │  ██████████                              │ │
│ │                          │  │  ████████                                │ │
│ │ Most profitable:         │  │  ██████████████                          │ │
│ │ BTC/USDT (+18.2%)       │  │  ████████████                            │ │
│ └──────────────────────────┘  │  00  04  08  12  16  20  24              │ │
│                                └──────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Trade History Table

**Columns**:
| Column | Description |
|--------|-------------|
| ID | Trade reference |
| Open Time | Entry timestamp |
| Close Time | Exit timestamp |
| Symbol | Trading pair |
| Side | LONG / SHORT |
| Size | Position size |
| Entry | Entry price |
| Exit | Exit price |
| P&L ($) | Realized profit/loss |
| P&L (%) | Percentage return |
| Fees | Trading fees paid |
| Duration | Hold time |
| Exit Reason | SL / TP / Signal / Manual |

**Export Options**:
- CSV download
- JSON export
- PDF report

---

## 5. LLM Model Comparison View (6-Model A/B Testing)

### 5.1 Model Leaderboard Dashboard

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ LLM MODEL LEADERBOARD - 6-Model Parallel Comparison                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│ COMPARISON PERIOD: Last 30 Days  |  All models run in parallel on each query │
│                                                                              │
│ MODEL PERFORMANCE RANKING                                                    │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ Rank│ Model         │ Decisions │ Accuracy │ Profit │ Sharpe │ Cost    │ │
│ ├─────┼───────────────┼───────────┼──────────┼────────┼────────┼─────────┤ │
│ │ 1.  │ DeepSeek V3   │ 432       │ 62.0%    │ +8.5%  │ 1.80   │ $0.89   │ │
│ │ 2.  │ Claude Opus   │ 432       │ 60.0%    │ +7.2%  │ 1.60   │ $2.45   │ │
│ │ 3.  │ Grok          │ 432       │ 58.0%    │ +6.8%  │ 1.50   │ $1.20   │ │
│ │ 4.  │ Claude Sonnet │ 432       │ 57.0%    │ +5.5%  │ 1.40   │ $1.50   │ │
│ │ 5.  │ GPT           │ 432       │ 55.0%    │ +4.2%  │ 1.20   │ $1.80   │ │
│ │ 6.  │ Qwen 2.5 7B   │ 432       │ 52.0%    │ +3.1%  │ 1.00   │ $0.00   │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│ ACCURACY BY DECISION TYPE                                                    │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ Model         │ BUY      │ SELL     │ HOLD     │ Overall  │ Best At    │ │
│ ├───────────────┼──────────┼──────────┼──────────┼──────────┼────────────┤ │
│ │ DeepSeek V3   │ 65%  ★   │ 58%      │ 63%      │ 62.0%    │ BUY        │ │
│ │ Claude Opus   │ 58%      │ 62%  ★   │ 60%      │ 60.0%    │ SELL       │ │
│ │ Grok          │ 56%      │ 55%      │ 70%  ★   │ 58.0%    │ HOLD       │ │
│ │ Claude Sonnet │ 55%      │ 58%      │ 58%      │ 57.0%    │ -          │ │
│ │ GPT           │ 54%      │ 54%      │ 57%      │ 55.0%    │ -          │ │
│ │ Qwen 2.5 7B   │ 50%      │ 51%      │ 55%      │ 52.0%    │ -          │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│ CONSENSUS ANALYSIS                                                           │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ Agreement Level  │ Occurrences │ Accuracy │ Action Taken               │ │
│ ├──────────────────┼─────────────┼──────────┼────────────────────────────┤ │
│ │ Unanimous (6/6)  │ 85          │ 75%      │ Execute with high conf.    │ │
│ │ Strong (5/6)     │ 120         │ 68%      │ Execute with confidence    │ │
│ │ Majority (4/6)   │ 160         │ 62%      │ Execute if threshold met   │ │
│ │ Split (≤3/6)     │ 67          │ 48%      │ Defer / Hold               │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Model Comparison Charts

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ CONFIDENCE CALIBRATION - All 6 Models                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│ Expected Accuracy (Confidence) vs Actual Accuracy                            │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │     100% ├────────────────────────────────────/                         │ │
│ │          │                              ___/  ← Perfect calibration     │ │
│ │          │                        ___--'  DeepSeek (0.88)               │ │
│ │          │                   ___--'       Claude Opus (0.85)            │ │
│ │      50% ├──────────────___--'            Grok (0.82)                   │ │
│ │          │         ___---'                Claude Sonnet (0.80)          │ │
│ │          │    __--'                       GPT (0.75)                    │ │
│ │          │ __'                            Qwen (0.70)                   │ │
│ │       0% └────────────────────────────────────                          │ │
│ │          0%            50%           100%                               │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│ COST EFFICIENCY (Accuracy per Dollar)                                        │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ Qwen 2.5 7B   ████████████████████████████████ ∞  (free)               │ │
│ │ DeepSeek V3   ███████████████████████████ 69.7 acc/$                   │ │
│ │ Grok          █████████████████████ 48.3 acc/$                         │ │
│ │ Claude Sonnet ████████████████ 38.0 acc/$                              │ │
│ │ GPT           ████████████ 30.6 acc/$                                  │ │
│ │ Claude Opus   ████████ 24.5 acc/$                                      │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.3 Real-Time Model Decisions

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ LATEST TRADING DECISION - 6-Model Parallel Query                             │
│ Symbol: BTC/USDT  |  Timestamp: 2025-12-18 10:30:00                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│ MODEL RESPONSES                                                              │
│ ┌───────────────┬────────┬────────┬─────────┬──────────────────────────────┐│
│ │ Model         │ Action │ Conf.  │ Latency │ Key Reasoning                ││
│ ├───────────────┼────────┼────────┼─────────┼──────────────────────────────┤│
│ │ DeepSeek V3   │ BUY    │ 0.72   │ 2.8s    │ Bullish breakout confirmed   ││
│ │ Claude Opus   │ BUY    │ 0.68   │ 4.5s    │ Strong momentum, low risk    ││
│ │ Grok          │ BUY    │ 0.65   │ 2.9s    │ News sentiment positive      ││
│ │ Claude Sonnet │ BUY    │ 0.62   │ 2.1s    │ Technical alignment          ││
│ │ GPT           │ HOLD   │ 0.55   │ 2.0s    │ Waiting for confirmation     ││
│ │ Qwen 2.5 7B   │ BUY    │ 0.58   │ 0.2s    │ Trend continuation pattern   ││
│ └───────────────┴────────┴────────┴─────────┴──────────────────────────────┘│
│                                                                              │
│ CONSENSUS: 5/6 BUY (Strong Majority)                                         │
│ WEIGHTED DECISION: BUY @ 0.68 confidence (boosted +0.10)                     │
│                                                                              │
│ [View Full Reasoning] [Compare Details] [Override Decision]                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Configuration Interface

### 6.1 Settings Panel

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ SYSTEM CONFIGURATION                                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│ RISK PARAMETERS                                                              │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ Max Leverage:                    [5x ▼]                                 │ │
│ │ Risk Per Trade:                  [1.0 %]                                │ │
│ │ Daily Loss Limit:                [5.0 %]                                │ │
│ │ Weekly Loss Limit:               [10.0 %]                               │ │
│ │ Max Drawdown:                    [20.0 %]                               │ │
│ │ Min Confidence Threshold:        [0.60]                                 │ │
│ │ Consecutive Loss Cooldown:       [5 trades]                             │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│ PORTFOLIO                                                                    │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ Target Allocation:                                                       │ │
│ │   BTC: [33.33 %]  XRP: [33.33 %]  USDT: [33.33 %]                      │ │
│ │ Rebalance Threshold:             [5.0 %]                                │ │
│ │ Hodl Bag Allocation:             [10 % of profits]                      │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│ LLM CONFIGURATION                                                            │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ Tier 1 (Local):                  [Qwen 2.5 7B ▼]                        │ │
│ │ Tier 2 (Strategic):              [DeepSeek V3 ▼]                        │ │
│ │ Tier 2 Backup:                   [Claude Sonnet ▼]                      │ │
│ │ Daily API Budget:                [$1.00]                                │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│ NOTIFICATIONS                                                                │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ [✓] Sound alerts                                                        │ │
│ │ [✓] Browser notifications                                               │ │
│ │ [ ] Email notifications                                                  │ │
│ │ Email: [________________________]                                        │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│ [Save Changes]  [Reset to Defaults]                                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Mobile Responsiveness

### 7.1 Mobile Layout

For mobile devices, the dashboard prioritizes:

1. **Portfolio Summary** (always visible at top)
2. **Current Price** (large display)
3. **Open Positions** (expandable list)
4. **Quick Actions** (Pause, Emergency Stop)
5. **Alerts** (notification style)

**Navigation**: Bottom tab bar
- Dashboard
- Positions
- History
- Settings

---

## 8. Accessibility Requirements

| Requirement | Implementation |
|-------------|----------------|
| Color contrast | WCAG AA compliant (4.5:1 minimum) |
| Screen reader | ARIA labels on all interactive elements |
| Keyboard navigation | Full tab navigation support |
| Color blindness | Non-color-dependent status indicators |
| Font scaling | Responsive to browser font settings |

---

*Document Version 1.0 - December 2025*
