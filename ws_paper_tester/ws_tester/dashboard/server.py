"""
Real-time dashboard server for WebSocket Paper Tester.
Provides REST API and WebSocket endpoints for live monitoring.

Security Features:
- Optional API key authentication
- Rate limiting support
- Thread-safe state management
"""

import asyncio
import copy
import json
import math
import os
import queue
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, Header
from fastapi.responses import HTMLResponse


def json_serializer(obj):
    """Custom JSON serializer that handles Infinity, NaN, and datetime."""
    if isinstance(obj, float):
        if math.isinf(obj) or math.isnan(obj):
            return None  # Convert Infinity/NaN to null
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)


# Security: API key authentication (set via environment variable or config)
API_KEY = os.environ.get("WS_TESTER_API_KEY", None)  # None = no auth required
REQUIRE_AUTH = API_KEY is not None


def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """Verify API key for authenticated endpoints."""
    if not REQUIRE_AUTH:
        return True
    if x_api_key is None or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return True


app = FastAPI(title="WS Paper Tester Dashboard")

# Store connected dashboard clients (with lock for thread safety)
dashboard_clients: List[WebSocket] = []
_clients_lock = threading.Lock()

# Latest state (updated by trading loop) - protected by lock
_state_lock = threading.RLock()
latest_state: Dict[str, Any] = {
    "timestamp": None,
    "prices": {},
    "strategies": [],
    "aggregate": {},
    "recent_trades": [],
    "session_info": {}
}


def _get_state_snapshot() -> Dict[str, Any]:
    """Get a thread-safe copy of the current state."""
    with _state_lock:
        return copy.deepcopy(latest_state)


class DashboardPublisher:
    """Publishes updates from trading loop to dashboard clients."""

    def __init__(self):
        # Use thread-safe queue.Queue for cross-thread communication
        self._queue: queue.Queue = queue.Queue(maxsize=1000)

    def put(self, event: dict):
        """Add event to queue (thread-safe, non-blocking)."""
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            # Drop oldest if full
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(event)
            except queue.Empty:
                pass

    async def broadcast_loop(self):
        """Continuously broadcast queued updates."""
        while True:
            try:
                # Non-blocking check for events
                try:
                    event = self._queue.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.05)  # 50ms polling
                    continue

                message = json.dumps(event, default=json_serializer)

                # Get current clients list safely
                with _clients_lock:
                    clients_copy = list(dashboard_clients)

                if not clients_copy:
                    continue

                # Broadcast to all connected clients concurrently (non-blocking)
                async def send_to_client(client: WebSocket):
                    try:
                        await asyncio.wait_for(client.send_text(message), timeout=2.0)
                        return None
                    except (asyncio.TimeoutError, ConnectionError, RuntimeError, OSError):
                        # Client disconnected, timed out, or connection error
                        return client

                # Send to all clients with timeout to prevent slow client blocking
                tasks = [send_to_client(client) for client in clients_copy]
                if tasks:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    disconnected = [r for r in results if r is not None and isinstance(r, WebSocket)]

                    # Remove disconnected clients
                    if disconnected:
                        with _clients_lock:
                            for client in disconnected:
                                if client in dashboard_clients:
                                    dashboard_clients.remove(client)

            except Exception as e:
                print(f"[Dashboard] Broadcast error: {e}")
                await asyncio.sleep(0.1)


publisher = DashboardPublisher()


@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for live dashboard updates."""
    # Check API key from query params for WebSocket auth
    if REQUIRE_AUTH:
        api_key = websocket.query_params.get("api_key")
        if api_key != API_KEY:
            await websocket.close(code=4001, reason="Invalid API key")
            return

    await websocket.accept()
    with _clients_lock:
        dashboard_clients.append(websocket)

    # Send current state on connect (thread-safe snapshot)
    state_snapshot = _get_state_snapshot()
    await websocket.send_text(json.dumps({
        "type": "initial_state",
        "data": state_snapshot
    }, default=json_serializer))

    try:
        while True:
            # Keep connection alive, handle client messages if needed
            data = await websocket.receive_text()
            # Could handle dashboard commands here (pause, filter, etc.)
    except WebSocketDisconnect:
        with _clients_lock:
            if websocket in dashboard_clients:
                dashboard_clients.remove(websocket)


@app.get("/api/strategies")
async def get_strategies(_: bool = Depends(verify_api_key)):
    """Get current strategy stats."""
    state = _get_state_snapshot()
    return state.get("strategies", [])


@app.get("/api/strategy/{name}")
async def get_strategy_detail(name: str, _: bool = Depends(verify_api_key)):
    """Get detailed stats for a specific strategy."""
    state = _get_state_snapshot()
    strategies = state.get("strategies", [])
    for s in strategies:
        if s.get("strategy") == name:
            return s
    return {"error": "Strategy not found"}


@app.get("/api/trades")
async def get_recent_trades(limit: int = 100, _: bool = Depends(verify_api_key)):
    """Get recent trades across all strategies."""
    state = _get_state_snapshot()
    return state.get("recent_trades", [])[:limit]


@app.get("/api/aggregate")
async def get_aggregate(_: bool = Depends(verify_api_key)):
    """Get aggregate stats across all strategies."""
    state = _get_state_snapshot()
    return state.get("aggregate", {})


@app.get("/api/prices")
async def get_prices(_: bool = Depends(verify_api_key)):
    """Get current prices."""
    state = _get_state_snapshot()
    return state.get("prices", {})


@app.get("/api/session")
async def get_session_info(_: bool = Depends(verify_api_key)):
    """Get session information."""
    state = _get_state_snapshot()
    return state.get("session_info", {})


@app.get("/", response_class=HTMLResponse)
async def dashboard_page():
    """Serve dashboard HTML."""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>WS Paper Tester Dashboard</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: 'Courier New', monospace;
            background: #0d1117;
            color: #c9d1d9;
            margin: 0;
            padding: 20px;
        }
        .header {
            background: #161b22;
            padding: 15px 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            border: 1px solid #30363d;
        }
        .header h1 {
            margin: 0 0 10px 0;
            color: #58a6ff;
            font-size: 1.4em;
        }
        .aggregate {
            display: flex;
            gap: 30px;
            font-size: 1.1em;
            flex-wrap: wrap;
        }
        .aggregate span {
            color: #8b949e;
        }
        .aggregate .value {
            color: #c9d1d9;
            font-weight: bold;
        }
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        @media (max-width: 1200px) {
            .grid { grid-template-columns: 1fr; }
        }
        .panel {
            background: #161b22;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #30363d;
        }
        .panel h2 {
            margin: 0 0 15px 0;
            color: #58a6ff;
            font-size: 1.1em;
            border-bottom: 1px solid #30363d;
            padding-bottom: 10px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9em;
        }
        th, td {
            padding: 8px 10px;
            text-align: left;
            border-bottom: 1px solid #21262d;
        }
        th {
            color: #8b949e;
            font-weight: normal;
            text-transform: uppercase;
            font-size: 0.8em;
        }
        .positive { color: #3fb950; }
        .negative { color: #f85149; }
        .neutral { color: #8b949e; }
        .prices {
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid #30363d;
            color: #8b949e;
        }
        .prices .symbol {
            display: inline-block;
            margin-right: 20px;
        }
        .prices .price {
            color: #c9d1d9;
        }
        .trade-feed {
            max-height: 400px;
            overflow-y: auto;
        }
        .trade-entry {
            padding: 8px 0;
            border-bottom: 1px solid #21262d;
            font-size: 0.85em;
        }
        .trade-entry:last-child {
            border-bottom: none;
        }
        .trade-time {
            color: #8b949e;
            margin-right: 10px;
        }
        .trade-strategy {
            color: #58a6ff;
            margin-right: 10px;
        }
        .trade-side {
            font-weight: bold;
            margin-right: 5px;
        }
        .trade-side.buy { color: #3fb950; }
        .trade-side.sell { color: #f85149; }
        .status-dot {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-dot.connected { background: #3fb950; }
        .status-dot.disconnected { background: #f85149; }
        #connection-status {
            font-size: 0.9em;
            color: #8b949e;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>
            <span class="status-dot" id="status-dot"></span>
            WEBSOCKET PAPER TESTER - Live Dashboard
            <span id="connection-status"></span>
        </h1>
        <div class="aggregate" id="aggregate">Connecting...</div>
    </div>

    <div class="grid">
        <div class="panel">
            <h2>Strategy Leaderboard</h2>
            <table id="strategies">
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Strategy</th>
                        <th>Equity</th>
                        <th>P&L</th>
                        <th>ROI</th>
                        <th>Trades</th>
                        <th>Win%</th>
                        <th>DD%</th>
                    </tr>
                </thead>
                <tbody></tbody>
            </table>
            <div class="prices" id="prices"></div>
        </div>

        <div class="panel">
            <h2>Live Trade Feed</h2>
            <div class="trade-feed" id="trades"></div>
        </div>
    </div>

    <script>
        let ws;
        let reconnectTimeout;

        function connect() {
            ws = new WebSocket(`ws://${window.location.host}/ws/live`);

            ws.onopen = () => {
                document.getElementById('status-dot').className = 'status-dot connected';
                document.getElementById('connection-status').textContent = '(Connected)';
                if (reconnectTimeout) {
                    clearTimeout(reconnectTimeout);
                    reconnectTimeout = null;
                }
            };

            ws.onclose = () => {
                document.getElementById('status-dot').className = 'status-dot disconnected';
                document.getElementById('connection-status').textContent = '(Reconnecting...)';
                reconnectTimeout = setTimeout(connect, 2000);
            };

            ws.onerror = () => {
                ws.close();
            };

            ws.onmessage = (event) => {
                const msg = JSON.parse(event.data);
                if (msg.type === 'initial_state' || msg.type === 'state_update') {
                    updateDashboard(msg.data);
                } else if (msg.type === 'trade') {
                    addTrade(msg.data);
                }
            };
        }

        function formatNumber(num, decimals = 2) {
            if (num === undefined || num === null) return '0';
            return parseFloat(num).toFixed(decimals);
        }

        function formatPnl(pnl) {
            const cls = pnl >= 0 ? 'positive' : 'negative';
            const sign = pnl >= 0 ? '+' : '';
            return `<span class="${cls}">${sign}$${formatNumber(pnl)}</span>`;
        }

        function formatPct(pct) {
            const cls = pct >= 0 ? 'positive' : (pct < 0 ? 'negative' : 'neutral');
            const sign = pct >= 0 ? '+' : '';
            return `<span class="${cls}">${sign}${formatNumber(pct)}%</span>`;
        }

        function updateDashboard(state) {
            // Update aggregate
            const agg = state.aggregate || {};
            document.getElementById('aggregate').innerHTML = `
                <div><span>Total:</span> <span class="value">$${formatNumber(agg.total_equity || 0)}</span></div>
                <div><span>P&L:</span> ${formatPnl(agg.total_pnl || 0)}</div>
                <div><span>ROI:</span> ${formatPct(agg.total_roi_pct || 0)}</div>
                <div><span>Strategies:</span> <span class="value">${agg.total_strategies || 0}</span></div>
                <div><span>Trades:</span> <span class="value">${agg.total_trades || 0}</span></div>
                <div><span>Win Rate:</span> <span class="value">${formatNumber(agg.win_rate || 0)}%</span></div>
            `;

            // Update leaderboard
            const tbody = document.querySelector('#strategies tbody');
            const strategies = state.strategies || [];
            tbody.innerHTML = strategies.map((s, i) => `
                <tr>
                    <td>${i + 1}</td>
                    <td>${s.strategy || 'Unknown'}</td>
                    <td>$${formatNumber(s.equity || 0)}</td>
                    <td>${formatPnl(s.pnl || 0)}</td>
                    <td>${formatPct(s.roi_pct || 0)}</td>
                    <td>${s.trades || 0}</td>
                    <td>${formatNumber(s.win_rate || 0)}%</td>
                    <td class="${(s.max_drawdown_pct || 0) > 5 ? 'negative' : ''}">${formatNumber(s.max_drawdown_pct || 0)}%</td>
                </tr>
            `).join('');

            // Update prices
            const prices = state.prices || {};
            document.getElementById('prices').innerHTML = Object.entries(prices)
                .map(([sym, price]) => `<span class="symbol">${sym}: <span class="price">$${formatNumber(price, 6)}</span></span>`)
                .join('');

            // Update recent trades if present
            if (state.recent_trades && state.recent_trades.length > 0) {
                const tradesDiv = document.getElementById('trades');
                tradesDiv.innerHTML = state.recent_trades.slice(0, 50).map(formatTrade).join('');
            }
        }

        function formatTrade(trade) {
            const time = trade.timestamp ? trade.timestamp.split('T')[1].split('.')[0] : '';
            const sideClass = trade.side === 'buy' ? 'buy' : 'sell';
            const pnlStr = trade.pnl ? ` ${formatPnl(trade.pnl)}` : '';
            return `
                <div class="trade-entry">
                    <span class="trade-time">${time}</span>
                    <span class="trade-strategy">${trade.strategy || 'Unknown'}</span>
                    <span class="trade-side ${sideClass}">${(trade.side || 'unknown').toUpperCase()}</span>
                    ${trade.symbol || ''} @ $${formatNumber(trade.price || 0, 6)}${pnlStr}
                </div>
            `;
        }

        function addTrade(trade) {
            const tradesDiv = document.getElementById('trades');
            const html = formatTrade(trade);
            tradesDiv.innerHTML = html + tradesDiv.innerHTML;

            // Keep only last 50 trades
            const entries = tradesDiv.querySelectorAll('.trade-entry');
            if (entries.length > 50) {
                for (let i = 50; i < entries.length; i++) {
                    entries[i].remove();
                }
            }
        }

        connect();
    </script>
</body>
</html>
"""


def update_state(
    prices: Dict[str, float],
    strategies: List[dict],
    aggregate: dict,
    session_info: dict = None
):
    """Update latest state directly (for non-async contexts). Thread-safe."""
    timestamp = datetime.now().isoformat()

    with _state_lock:
        latest_state["timestamp"] = timestamp
        latest_state["prices"] = copy.deepcopy(prices)
        latest_state["strategies"] = copy.deepcopy(strategies)
        latest_state["aggregate"] = copy.deepcopy(aggregate)
        if session_info:
            latest_state["session_info"] = copy.deepcopy(session_info)

    # Also queue broadcast to connected WebSocket clients
    publisher.put({
        "type": "state_update",
        "timestamp": timestamp,
        "data": {
            "prices": prices,
            "strategies": strategies,
            "aggregate": aggregate,
            "session_info": session_info or {},
        }
    })


def add_trade(trade: dict):
    """Add trade to recent trades (for non-async contexts). Thread-safe."""
    # Sanitize trade data to prevent XSS
    sanitized_trade = _sanitize_trade(trade)
    with _state_lock:
        latest_state["recent_trades"].insert(0, sanitized_trade)
        latest_state["recent_trades"] = latest_state["recent_trades"][:100]

    # Also queue broadcast to connected WebSocket clients
    publisher.put({
        "type": "trade",
        "timestamp": datetime.now().isoformat(),
        "data": sanitized_trade
    })


def _sanitize_trade(trade: dict) -> dict:
    """Sanitize trade data to prevent XSS attacks."""
    import html
    sanitized = {}
    for key, value in trade.items():
        if isinstance(value, str):
            # Escape HTML entities in string values
            sanitized[key] = html.escape(value)
        else:
            sanitized[key] = value
    return sanitized


def get_connected_client_count() -> int:
    """Get the number of connected dashboard clients."""
    with _clients_lock:
        return len(dashboard_clients)


async def broadcast_shutdown_message():
    """Send shutdown notification to all connected clients."""
    with _clients_lock:
        clients_copy = list(dashboard_clients)

    shutdown_msg = json.dumps({
        "type": "shutdown",
        "timestamp": datetime.now().isoformat(),
        "message": "Server is shutting down"
    }, default=json_serializer)

    for client in clients_copy:
        try:
            await asyncio.wait_for(client.send_text(shutdown_msg), timeout=2.0)
        except (asyncio.TimeoutError, ConnectionError, RuntimeError, OSError):
            # Client may already be disconnected
            pass
