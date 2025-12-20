# Phase 10: React Dashboard

**Phase Status**: Ready to Start
**Dependencies**: All Phase 3 components, Phase 7-9 features
**Deliverable**: Full-featured React web dashboard for system monitoring and control

---

## Overview

The React Dashboard provides a comprehensive web interface for monitoring system health, viewing positions, tracking performance, comparing LLM models, and managing trading operations. It connects to the existing FastAPI backend via REST and WebSocket APIs.

### Why This Phase Matters

| Benefit | Description |
|---------|-------------|
| Visibility | Real-time view of all system operations |
| Control | Pause, resume, close positions, emergency stop |
| Analysis | LLM model comparison, performance metrics |
| Trust | Transparency into agent decisions |
| Debugging | Audit trail of all activities |

### Dashboard Capabilities

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DASHBOARD OVERVIEW                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  VIEWS:                                                                      │
│  ├── Portfolio Summary      - Equity, P&L, allocation pie chart            │
│  ├── Price Charts           - Multi-symbol, multi-timeframe, indicators    │
│  ├── Open Positions         - Real-time position monitoring                │
│  ├── Agent Status           - Health and outputs per agent                 │
│  ├── Decision Log           - Audit trail of trading decisions            │
│  ├── Model Leaderboard      - 6-model comparison dashboard                 │
│  ├── Hodl Bags              - Long-term holdings tracking                  │
│  ├── System Health          - Connections, data pipeline, risk state      │
│  └── Manual Controls        - Pause, close positions, emergency stop      │
│                                                                             │
│  REAL-TIME FEATURES:                                                        │
│  ├── WebSocket price feeds                                                  │
│  ├── Live position P&L updates                                              │
│  ├── Agent output streaming                                                 │
│  ├── Alert notifications                                                    │
│  └── System status indicators                                               │
│                                                                             │
│  CONTROLS:                                                                   │
│  ├── Pause/Resume trading                                                   │
│  ├── Close individual positions                                             │
│  ├── Close all positions                                                    │
│  ├── Force portfolio rebalance                                              │
│  ├── Trigger sentiment refresh                                              │
│  └── Emergency stop                                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 10.1 Technology Stack

### Frontend Technologies

| Component | Technology | Version | Rationale |
|-----------|------------|---------|-----------|
| Framework | React | 18.x | Modern, component-based |
| Language | TypeScript | 5.x | Type safety |
| Build Tool | Vite | 5.x | Fast development |
| Styling | Tailwind CSS | 3.x | Utility-first, rapid development |
| State | Zustand | 4.x | Simple, performant |
| Data Fetching | TanStack Query | 5.x | Caching, synchronization |
| Charts | Lightweight Charts | 4.x | TradingView quality |
| Icons | Lucide React | Latest | Consistent iconography |
| WebSocket | Native + reconnecting | - | Real-time updates |

### Backend APIs (Existing)

| Type | Base URL | Description |
|------|----------|-------------|
| REST | `/api/v1/*` | CRUD operations |
| WebSocket | `/ws/*` | Real-time streams |

---

## 10.2 Architecture

### Application Structure

```
dashboard/
├── src/
│   ├── components/
│   │   ├── layout/
│   │   │   ├── Header.tsx           # Top navigation bar
│   │   │   ├── Sidebar.tsx          # Left navigation
│   │   │   ├── StatusBar.tsx        # Bottom status indicators
│   │   │   └── Layout.tsx           # Main layout wrapper
│   │   │
│   │   ├── portfolio/
│   │   │   ├── PortfolioSummary.tsx # Equity, P&L overview
│   │   │   ├── AllocationChart.tsx  # Pie chart of allocation
│   │   │   ├── EquityCurve.tsx      # Historical equity line
│   │   │   └── HodlBagCard.tsx      # Hodl bag status
│   │   │
│   │   ├── charts/
│   │   │   ├── PriceChart.tsx       # TradingView-style chart
│   │   │   ├── MultiChart.tsx       # Multiple symbols
│   │   │   ├── IndicatorPanel.tsx   # Indicator controls
│   │   │   └── TimeframeSelector.tsx# Timeframe buttons
│   │   │
│   │   ├── positions/
│   │   │   ├── PositionTable.tsx    # All positions table
│   │   │   ├── PositionCard.tsx     # Single position details
│   │   │   ├── PositionRow.tsx      # Table row component
│   │   │   └── ClosePositionDialog.tsx # Confirmation modal
│   │   │
│   │   ├── agents/
│   │   │   ├── AgentStatusGrid.tsx  # All agents overview
│   │   │   ├── AgentCard.tsx        # Single agent status
│   │   │   ├── AgentOutputViewer.tsx# Latest outputs
│   │   │   └── AgentTimeline.tsx    # Historical outputs
│   │   │
│   │   ├── decisions/
│   │   │   ├── DecisionLog.tsx      # Decision history table
│   │   │   ├── DecisionCard.tsx     # Single decision details
│   │   │   ├── DecisionDetail.tsx   # Full decision modal
│   │   │   └── ConsensusView.tsx    # Model agreement visual
│   │   │
│   │   ├── models/
│   │   │   ├── Leaderboard.tsx      # Model rankings table
│   │   │   ├── ModelComparison.tsx  # Side-by-side compare
│   │   │   ├── ModelMetrics.tsx     # Single model stats
│   │   │   └── PairwiseMatrix.tsx   # Significance heatmap
│   │   │
│   │   ├── system/
│   │   │   ├── HealthDashboard.tsx  # System health overview
│   │   │   ├── RiskState.tsx        # Risk engine status
│   │   │   ├── AlertPanel.tsx       # Active alerts
│   │   │   └── ConnectionStatus.tsx # API/WS status
│   │   │
│   │   ├── controls/
│   │   │   ├── ControlPanel.tsx     # All controls
│   │   │   ├── PauseResumeButton.tsx
│   │   │   ├── CloseAllDialog.tsx   # Confirmation modal
│   │   │   └── EmergencyStop.tsx    # Big red button
│   │   │
│   │   └── common/
│   │       ├── Card.tsx             # Reusable card
│   │       ├── Badge.tsx            # Status badges
│   │       ├── Button.tsx           # Button variants
│   │       ├── Modal.tsx            # Modal wrapper
│   │       ├── Table.tsx            # Data table
│   │       ├── Spinner.tsx          # Loading indicator
│   │       └── ErrorBoundary.tsx    # Error handling
│   │
│   ├── hooks/
│   │   ├── useWebSocket.ts          # WebSocket connection
│   │   ├── usePortfolio.ts          # Portfolio data
│   │   ├── usePositions.ts          # Position data
│   │   ├── useAgents.ts             # Agent status
│   │   ├── useDecisions.ts          # Decision history
│   │   ├── useModels.ts             # Model comparison
│   │   ├── useHodl.ts               # Hodl bag data
│   │   └── useSystem.ts             # System health
│   │
│   ├── stores/
│   │   ├── portfolioStore.ts        # Portfolio state
│   │   ├── positionStore.ts         # Positions state
│   │   ├── agentStore.ts            # Agent state
│   │   ├── priceStore.ts            # Real-time prices
│   │   ├── alertStore.ts            # Alerts state
│   │   └── settingsStore.ts         # User preferences
│   │
│   ├── services/
│   │   ├── api.ts                   # REST API client
│   │   ├── websocket.ts             # WebSocket client
│   │   └── auth.ts                  # Authentication
│   │
│   ├── types/
│   │   ├── portfolio.ts             # Portfolio types
│   │   ├── position.ts              # Position types
│   │   ├── agent.ts                 # Agent types
│   │   ├── decision.ts              # Decision types
│   │   ├── model.ts                 # Model types
│   │   └── api.ts                   # API response types
│   │
│   ├── pages/
│   │   ├── Dashboard.tsx            # Main dashboard view
│   │   ├── Positions.tsx            # Positions page
│   │   ├── Agents.tsx               # Agents page
│   │   ├── Models.tsx               # Model comparison page
│   │   ├── Decisions.tsx            # Decision log page
│   │   ├── Settings.tsx             # Settings page
│   │   └── Login.tsx                # Login page
│   │
│   ├── App.tsx                      # Root component
│   ├── main.tsx                     # Entry point
│   └── index.css                    # Global styles
│
├── public/
│   └── favicon.ico
├── package.json
├── tsconfig.json
├── tailwind.config.js
├── vite.config.ts
└── .env.example
```

---

## 10.3 Key Views

### 10.3.1 Portfolio Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ PORTFOLIO SUMMARY                                          Last updated: now │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐    │
│  │  Total Equity      │  │  Today's P&L       │  │  Open Positions    │    │
│  │  $4,285.50         │  │  +$45.32 (+1.07%)  │  │  3                 │    │
│  │  ▲ from $4,100.00  │  │  ██████████░░░░░░  │  │  BTC: 2, XRP: 1   │    │
│  └────────────────────┘  └────────────────────┘  └────────────────────┘    │
│                                                                             │
│  ┌─────────────────────────────┐  ┌──────────────────────────────────────┐ │
│  │  ALLOCATION                  │  │  EQUITY CURVE (30 Days)              │ │
│  │                              │  │                                      │ │
│  │       ┌─────┐                │  │  $4,400 ┤           ╱╲  ╱────        │ │
│  │      /       \               │  │  $4,200 ┤      ╱──╱  ╲╱              │ │
│  │     │  BTC    │              │  │  $4,000 ┤─────╱                      │ │
│  │     │  37.4%  │              │  │  $3,800 ┤                            │ │
│  │      \       /   ┌───────┐   │  │         └────────────────────────    │ │
│  │       └─────┘    │  XRP  │   │  │          Dec 1        Today          │ │
│  │                  │ 31.5% │   │  └──────────────────────────────────────┘ │
│  │   ┌───────┐      └───────┘   │                                          │
│  │   │ USDT  │                  │                                          │
│  │   │ 31.1% │                  │                                          │
│  │   └───────┘                  │                                          │
│  └─────────────────────────────┘                                           │
│                                                                             │
│  ┌─────────────────────────────┐  ┌──────────────────────────────────────┐ │
│  │  HODL BAGS                   │  │  TRADING METRICS (7 Days)            │ │
│  │                              │  │                                      │ │
│  │  BTC: 0.0155 BTC ($698.25)  │  │  Win Rate: 62.5%    │ Trades: 24    │ │
│  │       +$85.85 (+14.0%) ▲    │  │  Profit Factor: 1.8 │ Sharpe: 1.65  │ │
│  │                              │  │  Avg Win: $38.50   │ Avg Loss: $21 │ │
│  │  XRP: 850 XRP ($510.00)     │  │                                      │ │
│  │       +$68.00 (+15.4%) ▲    │  │  Max Drawdown: 4.2%                  │ │
│  └─────────────────────────────┘  └──────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 10.3.2 Price Charts

- Multi-symbol support (BTC/USDT, XRP/USDT, XRP/BTC)
- Multiple timeframes (1m, 5m, 15m, 1h, 4h, 1d)
- Technical indicators overlay (EMA, RSI, MACD, Bollinger)
- Position markers on chart
- Volume display
- Crosshair with OHLC data

### 10.3.3 Open Positions

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ OPEN POSITIONS                                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ Symbol    │ Side  │ Size      │ Entry    │ Current  │ P&L         │ Actions│
│ ──────────┼───────┼───────────┼──────────┼──────────┼─────────────┼────────│
│ BTC/USDT  │ LONG  │ 0.05 BTC  │ $45,200  │ $45,850  │ +$32.50 ▲   │ [Close]│
│           │       │           │ SL: $44,300 │ TP: $47,000 │ 2x        │        │
│ ──────────┼───────┼───────────┼──────────┼──────────┼─────────────┼────────│
│ BTC/USDT  │ LONG  │ 0.03 BTC  │ $45,500  │ $45,850  │ +$10.50 ▲   │ [Close]│
│           │       │           │ SL: $44,600 │ TP: $47,200 │ 2x        │        │
│ ──────────┼───────┼───────────┼──────────┼──────────┼─────────────┼────────│
│ XRP/USDT  │ LONG  │ 500 XRP   │ $0.58    │ $0.60    │ +$10.00 ▲   │ [Close]│
│           │       │           │ SL: $0.55 │ TP: $0.65 │ 1x          │        │
│                                                                             │
│ Total Unrealized P&L: +$53.00                          [Close All Positions]│
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 10.3.4 Agent Status

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ AGENT STATUS                                                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐ │
│  │ Technical Analysis  │  │ Regime Detection    │  │ Sentiment Analysis │ │
│  │ ● RUNNING           │  │ ● RUNNING           │  │ ● RUNNING          │ │
│  │                     │  │                     │  │                    │ │
│  │ Last Run: 30s ago   │  │ Last Run: 2m ago    │  │ Last Run: 8m ago   │ │
│  │ Latency: 185ms      │  │ Latency: 220ms      │  │ Latency: 2.1s      │ │
│  │ Errors: 0           │  │ Errors: 0           │  │ Errors: 0          │ │
│  │                     │  │                     │  │                    │ │
│  │ BTC: Bullish (0.72) │  │ BTC: Trending Bull  │  │ BTC: Bullish (0.65)│ │
│  │ XRP: Neutral (0.55) │  │ XRP: Neutral        │  │ XRP: Neutral (0.50)│ │
│  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘ │
│                                                                             │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐ │
│  │ Trading Decision    │  │ Risk Management     │  │ Coordinator        │ │
│  │ ● RUNNING           │  │ ● RUNNING           │  │ ● RUNNING          │ │
│  │                     │  │                     │  │                    │ │
│  │ Last Run: 45m ago   │  │ Always Active       │  │ Last Conflict: N/A │ │
│  │ Latency: 2.8s       │  │ Latency: 8ms        │  │ State: RUNNING     │ │
│  │ Errors: 0           │  │ Errors: 0           │  │                    │ │
│  │                     │  │                     │  │ Scheduled Tasks: 4 │ │
│  │ Consensus: 5/6 BUY  │  │ Limits OK ✓         │  │ All Running ✓      │ │
│  │ Conf: 0.68          │  │ No Circuit Breakers │  │                    │ │
│  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 10.3.5 Model Leaderboard

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ MODEL LEADERBOARD (Last 7 Days)                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ Rank │ Model         │ Accuracy │ Profit  │ Sharpe │ Calibr. │ Score       │
│ ─────┼───────────────┼──────────┼─────────┼────────┼─────────┼─────────────│
│ 1    │ DeepSeek V3   │ 62.0%    │ +8.5%   │ 1.80   │ 0.88    │ 0.78  ★     │
│ 2    │ Claude Opus   │ 60.0%    │ +7.2%   │ 1.60   │ 0.85    │ 0.72        │
│ 3    │ Grok          │ 58.0%    │ +6.8%   │ 1.50   │ 0.82    │ 0.70        │
│ 4    │ Claude Sonnet │ 57.0%    │ +5.5%   │ 1.40   │ 0.80    │ 0.68        │
│ 5    │ GPT           │ 55.0%    │ +4.2%   │ 1.20   │ 0.75    │ 0.62        │
│ 6    │ Qwen 2.5 7B   │ 52.0%    │ +3.1%   │ 1.00   │ 0.70    │ 0.58        │
│                                                                             │
│ CONSENSUS ANALYSIS                                                          │
│ ┌────────────────────────────────────────────────────────────────────────┐ │
│ │ 6/6 Agreement: 75% accurate │ 5/6: 68% │ 4/6: 62% │ ≤3/6: 48%         │ │
│ └────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│ BEST BY DECISION TYPE                                                       │
│ • BUY: DeepSeek V3 (65%)  • SELL: Claude Opus (62%)  • HOLD: Grok (70%)   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 10.3.6 Control Panel

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ SYSTEM CONTROLS                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  TRADING STATE: ● RUNNING                                                   │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  [ PAUSE TRADING ]        [ RESUME TRADING ]                           │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  POSITION CONTROLS                                                          │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  [ Close All Positions ]   Open: 3 │ Total Value: $1,250.00           │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  MANUAL TRIGGERS                                                            │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  [ Force Rebalance ]  [ Refresh Sentiment ]  [ Run TA Analysis ]      │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  EMERGENCY                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │               ┌─────────────────────────────────────────┐              │ │
│  │               │      ⚠️  EMERGENCY STOP  ⚠️              │              │ │
│  │               │                                         │              │ │
│  │               │  Closes all positions and halts system  │              │ │
│  │               └─────────────────────────────────────────┘              │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 10.4 WebSocket Integration

### WebSocket Topics

| Topic | Message Types | Use |
|-------|---------------|-----|
| `/ws/portfolio` | portfolio_snapshot, position_update, trade_execution | Portfolio updates |
| `/ws/prices` | ticker_update, candle_update | Real-time prices |
| `/ws/agents` | agent_output, agent_status_change | Agent status |
| `/ws/alerts` | risk_alert, system_alert | Notifications |

### WebSocket Client

```typescript
// src/services/websocket.ts

class WebSocketManager {
  private connections: Map<string, WebSocket> = new Map();
  private reconnectAttempts: Map<string, number> = new Map();

  connect(topic: string, onMessage: (data: any) => void): void {
    const ws = new WebSocket(`${WS_BASE_URL}${topic}`);

    ws.onopen = () => {
      console.log(`Connected to ${topic}`);
      this.reconnectAttempts.set(topic, 0);
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      onMessage(data);
    };

    ws.onclose = () => {
      this.handleReconnect(topic, onMessage);
    };

    ws.onerror = (error) => {
      console.error(`WebSocket error on ${topic}:`, error);
    };

    this.connections.set(topic, ws);
  }

  private handleReconnect(topic: string, onMessage: (data: any) => void): void {
    const attempts = this.reconnectAttempts.get(topic) || 0;
    if (attempts < 5) {
      setTimeout(() => {
        this.reconnectAttempts.set(topic, attempts + 1);
        this.connect(topic, onMessage);
      }, Math.min(1000 * Math.pow(2, attempts), 30000));
    }
  }

  disconnect(topic: string): void {
    const ws = this.connections.get(topic);
    if (ws) {
      ws.close();
      this.connections.delete(topic);
    }
  }

  disconnectAll(): void {
    this.connections.forEach((ws, topic) => {
      ws.close();
    });
    this.connections.clear();
  }
}
```

---

## 10.5 State Management

### Zustand Stores

```typescript
// src/stores/portfolioStore.ts

interface PortfolioState {
  equity: number;
  dailyPnL: number;
  dailyPnLPct: number;
  allocation: {
    btc: { amount: number; valuePct: number };
    xrp: { amount: number; valuePct: number };
    usdt: { amount: number; valuePct: number };
  };
  equityCurve: { timestamp: Date; value: number }[];
  hodlBags: {
    btc: HodlBagState;
    xrp: HodlBagState;
  };

  // Actions
  setPortfolio: (data: PortfolioSnapshot) => void;
  updateFromWebSocket: (message: PortfolioUpdate) => void;
}

const usePortfolioStore = create<PortfolioState>((set) => ({
  equity: 0,
  dailyPnL: 0,
  dailyPnLPct: 0,
  allocation: {
    btc: { amount: 0, valuePct: 0 },
    xrp: { amount: 0, valuePct: 0 },
    usdt: { amount: 0, valuePct: 0 },
  },
  equityCurve: [],
  hodlBags: {
    btc: initialHodlState,
    xrp: initialHodlState,
  },

  setPortfolio: (data) => set({ ...data }),

  updateFromWebSocket: (message) =>
    set((state) => ({
      ...state,
      equity: message.equity ?? state.equity,
      dailyPnL: message.dailyPnL ?? state.dailyPnL,
    })),
}));
```

---

## 10.6 API Integration

### REST API Client

```typescript
// src/services/api.ts

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1',
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add auth interceptor
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('auth_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

export const portfolioApi = {
  getSummary: () => api.get<PortfolioSummary>('/portfolio/summary'),
  getHistory: (period: string) => api.get<PortfolioSnapshot[]>(`/portfolio/history?period=${period}`),
};

export const positionApi = {
  getAll: () => api.get<Position[]>('/positions'),
  close: (id: string, reason: string) => api.post(`/positions/${id}/close`, { reason }),
  modify: (id: string, data: PositionModify) => api.put(`/positions/${id}/modify`, data),
  closeAll: () => api.post('/controls/close-all', { confirm: 'CLOSE_ALL' }),
};

export const agentApi = {
  getAll: () => api.get<AgentStatus[]>('/agents'),
  getOutputs: (name: string, limit: number) => api.get<AgentOutput[]>(`/agents/${name}/outputs?limit=${limit}`),
};

export const modelApi = {
  getLeaderboard: () => api.get<LeaderboardEntry[]>('/models/leaderboard'),
  getMetrics: (id: string, period: string) => api.get<ModelMetrics>(`/models/${id}/metrics?period=${period}`),
  getComparison: () => api.get<ComparisonReport>('/models/comparison'),
};

export const controlApi = {
  pause: () => api.post('/controls/pause'),
  resume: () => api.post('/controls/resume'),
  emergencyStop: () => api.post('/controls/emergency-stop', { confirm: 'EMERGENCY_STOP' }),
};
```

---

## 10.7 Configuration

### Environment Variables

```bash
# .env.example

# API Configuration
VITE_API_URL=http://localhost:8000/api/v1
VITE_WS_URL=ws://localhost:8000

# Feature Flags
VITE_ENABLE_PAPER_TRADING=true
VITE_ENABLE_LIVE_TRADING=false

# Refresh Intervals (ms)
VITE_PORTFOLIO_REFRESH=5000
VITE_AGENT_REFRESH=10000
VITE_LEADERBOARD_REFRESH=60000
```

### Tailwind Configuration

```javascript
// tailwind.config.js

module.exports = {
  content: ['./src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        profit: '#10b981',  // Green
        loss: '#ef4444',    // Red
        neutral: '#6b7280', // Gray
        primary: '#3b82f6', // Blue
        warning: '#f59e0b', // Amber
        danger: '#dc2626',  // Red-600
      },
    },
  },
  plugins: [],
};
```

---

## 10.8 Test Requirements

### Component Tests

| Test | Description | Acceptance |
|------|-------------|------------|
| `PortfolioSummary` | Renders with data | Values displayed correctly |
| `PositionTable` | Renders positions | All positions shown |
| `AgentCard` | Renders agent status | Status indicator correct |
| `Leaderboard` | Renders rankings | Correct order |
| `ClosePositionDialog` | Confirmation flow | Requires confirmation |
| `EmergencyStop` | Double confirmation | Two-step confirmation |

### Integration Tests

| Test | Description | Acceptance |
|------|-------------|------------|
| `WebSocket connection` | Connects to all topics | Stable connection |
| `REST API calls` | All endpoints respond | Valid responses |
| `Real-time updates` | UI updates on message | < 100ms latency |
| `State persistence` | Survives refresh | Data restored |

### E2E Tests

| Test | Description | Acceptance |
|------|-------------|------------|
| `Login flow` | User can authenticate | Token stored |
| `View portfolio` | Dashboard loads | All widgets render |
| `Close position` | Close via UI | Position closed |
| `Pause trading` | Pause via controls | System paused |

---

## 10.9 Deliverables Checklist

- [ ] Initialize React project with Vite + TypeScript
- [ ] Configure Tailwind CSS
- [ ] Set up Zustand stores
- [ ] Implement API client
- [ ] Implement WebSocket client
- [ ] Create layout components
- [ ] Create portfolio components
- [ ] Create chart components (Lightweight Charts)
- [ ] Create position components
- [ ] Create agent components
- [ ] Create model comparison components
- [ ] Create control components
- [ ] Create page routes
- [ ] Add authentication flow
- [ ] Write component tests
- [ ] Write integration tests
- [ ] Build and deployment configuration

---

## 10.10 Backend Updates Required

### New/Updated API Endpoints

| Method | Path | Description | Status |
|--------|------|-------------|--------|
| GET | `/api/v1/portfolio/summary` | Portfolio overview | Exists |
| GET | `/api/v1/portfolio/history` | Historical equity | Needs update |
| GET | `/api/v1/positions` | All positions | Exists |
| GET | `/api/v1/agents` | Agent status | Exists |
| GET | `/api/v1/models/leaderboard` | Model rankings | Phase 9 |
| GET | `/api/v1/hodl/status` | Hodl bags | Phase 8 |
| POST | `/api/v1/controls/*` | Control actions | Exists |

### WebSocket Handlers

**File**: `triplegain/src/api/websocket.py`

Implement WebSocket handlers for all topics.

---

## 10.11 Acceptance Criteria

### Functional Requirements

| Requirement | Test Method | Acceptance |
|-------------|-------------|------------|
| Portfolio displays correctly | Visual inspection | All data accurate |
| Positions update in real-time | WebSocket test | < 100ms latency |
| Controls work | Integration test | Actions execute |
| Model leaderboard accurate | Data verification | Matches backend |
| Charts render | Visual inspection | Smooth performance |

### Non-Functional Requirements

| Requirement | Target | Measurement |
|-------------|--------|-------------|
| Initial load time | < 2s | Lighthouse |
| WebSocket reconnect | < 5s | Manual disconnect test |
| Mobile responsive | Yes | Device testing |
| Accessibility | WCAG 2.1 AA | Lighthouse |

---

## References

- Design: [05-user-interface-requirements.md](../TripleGain-master-design/05-user-interface-requirements.md)
- Existing: [routes_orchestration.py](../../../triplegain/src/api/routes_orchestration.py)
- Phase 6: [Paper Trading](./phase-3_5-paper-trading-plan.md)

---

*Phase 10 Implementation Plan v1.0 - December 2025*
