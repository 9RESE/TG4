"""
Shared test fixtures for TripleGain tests.

This module provides common fixtures used across multiple test files
to reduce code duplication and ensure consistent test data.
"""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from decimal import Decimal


# =============================================================================
# Configuration Fixtures
# =============================================================================

@pytest.fixture
def temp_config_dir():
    """Create a temporary config directory with standard test config files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir)

        # Create indicators.yaml
        (config_path / "indicators.yaml").write_text("""
indicators:
  ema:
    periods: [9, 21, 50, 200]
    source: close
  sma:
    periods: [20, 50, 200]
    source: close
  rsi:
    period: 14
    source: close
  macd:
    fast_period: 12
    slow_period: 26
    signal_period: 9
  atr:
    period: 14
  bollinger_bands:
    period: 20
    std_dev: 2.0
""")

        # Create snapshot.yaml
        (config_path / "snapshot.yaml").write_text("""
snapshot_builder:
  candle_lookback:
    1m: 60
    5m: 48
    15m: 32
    1h: 48
  data_quality:
    max_age_seconds: 60
    min_candles_required: 20
  token_budgets:
    tier1_local: 3000
    tier2_api: 6000
""")

        # Create database.yaml
        (config_path / "database.yaml").write_text("""
database:
  connection:
    host: localhost
    port: 5432
    database: test_db
    user: test_user
    password: test_pass
  retention:
    agent_outputs_days: 90
""")

        # Create prompts.yaml
        (config_path / "prompts.yaml").write_text("""
prompts:
  agents:
    technical_analysis:
      tier: tier1_local
      template: ta_agent.txt
    risk_manager:
      tier: tier2_api
      template: risk_agent.txt
  token_budgets:
    tier1_local:
      total: 8192
      market_data: 3000
      buffer: 2000
    tier2_api:
      total: 128000
      market_data: 6000
      buffer: 116000
  templates_dir: config/prompts/
""")

        # Create agents.yaml
        (config_path / "agents.yaml").write_text("""
providers:
  ollama:
    base_url: "http://localhost:11434"
    timeout_seconds: 30
    default_model: "qwen2.5:7b"
agents:
  technical_analysis:
    enabled: true
    tier: tier1_local
    provider: ollama
    model: "qwen2.5:7b"
    template: "technical_analysis.txt"
token_budgets:
  tier1_local:
    total: 8192
    input: 6000
    output: 2000
""")

        # Create risk.yaml
        (config_path / "risk.yaml").write_text("""
limits:
  max_leverage: 5
  max_position_pct: 20
  max_total_exposure_pct: 80
  max_risk_per_trade_pct: 2
circuit_breakers:
  daily_loss:
    threshold_pct: 5.0
    action: halt_new_trades
""")

        # Create orchestration.yaml
        (config_path / "orchestration.yaml").write_text("""
llm:
  primary:
    provider: deepseek
    model: deepseek-chat
schedules:
  technical_analysis:
    enabled: true
    interval_seconds: 60
symbols:
  - BTC/USDT
  - XRP/USDT
  - XRP/BTC
""")

        # Create portfolio.yaml
        (config_path / "portfolio.yaml").write_text("""
target_allocation:
  btc_pct: 33.33
  xrp_pct: 33.33
  usdt_pct: 33.34
rebalancing:
  enabled: true
  threshold_pct: 5.0
""")

        # Create execution.yaml
        (config_path / "execution.yaml").write_text("""
kraken:
  rest_url: https://api.kraken.com
  websocket_url: wss://ws.kraken.com
symbols:
  BTC/USDT:
    kraken_pair: XBTUSDT
    min_order_size: 0.0001
    price_decimals: 2
    size_decimals: 8
  XRP/USDT:
    kraken_pair: XRPUSDT
    min_order_size: 10
    price_decimals: 5
    size_decimals: 0
""")

        yield config_path


# =============================================================================
# Mock LLM Client Fixtures
# =============================================================================

@pytest.fixture
def mock_llm_client():
    """Mock LLM client for testing agents."""
    client = MagicMock()
    client.generate = AsyncMock(return_value={
        "response": "Test LLM response",
        "tokens_used": 100,
        "latency_ms": 50,
    })
    client.is_available = MagicMock(return_value=True)
    return client


@pytest.fixture
def mock_llm_response():
    """Standard mock LLM response data."""
    return {
        "response": '{"action": "HOLD", "confidence": 0.75, "reasoning": "Test reasoning for the decision"}',
        "tokens_input": 500,
        "tokens_output": 100,
        "latency_ms": 150,
        "model": "test-model",
        "cost_usd": 0.001,
    }


# =============================================================================
# Sample Market Data Fixtures
# =============================================================================

@pytest.fixture
def sample_candles():
    """Sample OHLCV candle data for indicator tests."""
    return [
        {"timestamp": 1700000000, "open": 100.0, "high": 105.0, "low": 99.0, "close": 104.0, "volume": 1000},
        {"timestamp": 1700000060, "open": 104.0, "high": 107.0, "low": 103.0, "close": 106.0, "volume": 1200},
        {"timestamp": 1700000120, "open": 106.0, "high": 108.0, "low": 104.0, "close": 105.0, "volume": 900},
        {"timestamp": 1700000180, "open": 105.0, "high": 106.0, "low": 102.0, "close": 103.0, "volume": 1100},
        {"timestamp": 1700000240, "open": 103.0, "high": 105.0, "low": 101.0, "close": 104.5, "volume": 1000},
    ]


@pytest.fixture
def sample_candles_extended():
    """Extended candle data for tests requiring more data points (e.g., RSI)."""
    import random
    random.seed(42)  # Reproducible
    candles = []
    price = 100.0
    ts = 1700000000

    for i in range(50):
        change = random.uniform(-2, 2)
        open_price = price
        close_price = price + change
        high = max(open_price, close_price) + random.uniform(0, 1)
        low = min(open_price, close_price) - random.uniform(0, 1)
        volume = random.uniform(500, 2000)

        candles.append({
            "timestamp": ts + (i * 60),
            "open": open_price,
            "high": high,
            "low": low,
            "close": close_price,
            "volume": volume,
        })
        price = close_price

    return candles


@pytest.fixture
def sample_ticker():
    """Sample ticker data for market snapshot tests."""
    return {
        "symbol": "BTC/USDT",
        "bid": Decimal("45000.00"),
        "ask": Decimal("45001.50"),
        "last": Decimal("45000.75"),
        "volume_24h": Decimal("1234.5678"),
        "timestamp": datetime.now(timezone.utc),
    }


@pytest.fixture
def sample_order_book():
    """Sample order book data."""
    return {
        "bids": [
            (Decimal("45000.00"), Decimal("1.5")),
            (Decimal("44999.50"), Decimal("2.0")),
            (Decimal("44998.00"), Decimal("3.5")),
        ],
        "asks": [
            (Decimal("45001.50"), Decimal("1.2")),
            (Decimal("45002.00"), Decimal("1.8")),
            (Decimal("45003.50"), Decimal("2.5")),
        ],
        "timestamp": datetime.now(timezone.utc),
    }


# =============================================================================
# Sample Agent Output Fixtures
# =============================================================================

@pytest.fixture
def sample_agent_output():
    """Sample AgentOutput data for testing."""
    return {
        "agent_name": "test_agent",
        "timestamp": datetime.now(timezone.utc),
        "symbol": "BTC/USDT",
        "confidence": 0.75,
        "reasoning": "Test analysis with sufficient reasoning for validation",
        "latency_ms": 100,
        "tokens_used": 500,
        "model_used": "test-model",
    }


@pytest.fixture
def sample_trading_signal():
    """Sample trading signal for execution tests."""
    return {
        "symbol": "BTC/USDT",
        "action": "BUY",
        "confidence": 0.80,
        "entry_price": Decimal("45000.00"),
        "stop_loss": Decimal("44000.00"),
        "take_profit": Decimal("47000.00"),
        "size_pct": 10.0,
        "leverage": 2,
    }


# =============================================================================
# Risk State Fixtures
# =============================================================================

@pytest.fixture
def sample_risk_state():
    """Sample risk state for testing risk engine."""
    return {
        "daily_loss_pct": 2.5,
        "weekly_loss_pct": 4.0,
        "max_drawdown_pct": 8.0,
        "peak_equity_usd": Decimal("10500.00"),
        "current_equity_usd": Decimal("9660.00"),
        "consecutive_losses": 2,
        "trades_today": 5,
        "trading_halted": False,
        "circuit_breakers_active": [],
        "active_cooldowns": {},
    }


# =============================================================================
# Position Fixtures
# =============================================================================

@pytest.fixture
def sample_position():
    """Sample open position for testing."""
    return {
        "position_id": "pos_12345",
        "symbol": "BTC/USDT",
        "side": "long",
        "size": Decimal("0.1"),
        "entry_price": Decimal("45000.00"),
        "current_price": Decimal("45500.00"),
        "leverage": 2,
        "stop_loss": Decimal("44000.00"),
        "take_profit": Decimal("47000.00"),
        "unrealized_pnl": Decimal("50.00"),
        "unrealized_pnl_pct": Decimal("1.11"),
        "opened_at": datetime.now(timezone.utc),
    }
