# Unified Trading Platform Documentation

**Phase 24 - December 2024**

A consolidated trading system combining all strategies into a single, configurable platform with per-strategy logging and experiment support.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Quick Start](#quick-start)
4. [CLI Reference](#cli-reference)
5. [Configuration](#configuration)
6. [Strategy Registry](#strategy-registry)
7. [Unified Orchestrator](#unified-orchestrator)
8. [Logging System](#logging-system)
9. [Experiments & A/B Testing](#experiments--ab-testing)
10. [Adding New Strategies](#adding-new-strategies)
11. [API Reference](#api-reference)

---

## Overview

The Unified Trading Platform consolidates multiple trading scripts and strategies into a single entry point with:

- **16 built-in strategies** (general, scalper, grid, margin)
- **Per-strategy logging** for individual performance tracking
- **Experiment support** for parameter tuning and A/B testing
- **Regime-aware weighting** that adjusts strategy importance based on market conditions
- **YAML configuration** for easy strategy management

### Previous Architecture
```
main.py                    → RLOrchestrator / EnsembleOrchestrator
grid_ensemble_orchestrator.py → GridEnsembleOrchestrator
```

### New Architecture
```
unified_trader.py → UnifiedOrchestrator → All 16 strategies
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    unified_trader.py (CLI)                       │
│  Commands: paper, experiment, list, config, analyze, compare     │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                    UnifiedOrchestrator                           │
│  - Loads strategies from registry                                │
│  - Detects market regime                                         │
│  - Weighted voting across strategies                             │
│  - Paper trading execution                                       │
└─────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ StrategyRegistry│  │StrategyLogger  │  │    Portfolio    │
│ - 16 strategies │  │Manager         │  │ - Balance track │
│ - YAML config   │  │- Per-strategy  │  │ - Paper trading │
│ - Enable/disable│  │  logs          │  │                 │
└─────────────────┘  └─────────────────┘  └─────────────────┘
          │
┌─────────────────────────────────────────────────────────────────┐
│                        Strategies                                │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐            │
│  │ General  │ │ Scalpers │ │   Grid   │ │  Margin  │            │
│  │ (6)      │ │ (2)      │ │ (4)      │ │ (4)      │            │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

### File Structure

```
src/
├── unified_trader.py           # Main CLI entry point
├── unified_orchestrator.py     # Master orchestrator
├── strategy_registry.py        # Strategy loading & config
├── utils/
│   ├── strategy_logger.py      # Per-strategy logging
│   └── diagnostic_logger.py    # Legacy diagnostic logger
├── strategies_config/
│   └── unified.yaml            # Unified configuration
└── strategies/
    ├── base_strategy.py        # Abstract base class
    ├── defensive_yield/
    ├── mean_reversion_vwap/
    ├── xrp_btc_pair_trading/
    ├── intraday_scalper/
    ├── ma_trend_follow/
    ├── xrp_btc_leadlag/
    ├── ema9_scalper/
    └── grid_base.py            # Grid strategy classes
```

---

## Quick Start

### 1. Initialize Configuration

```bash
cd src
python unified_trader.py init-config
```

This creates `strategies_config/unified.yaml` with all 16 strategies configured.

### 2. Run Paper Trading

```bash
# Default: 60 minutes, 5-minute intervals
python unified_trader.py paper

# Custom duration and interval
python unified_trader.py paper --duration 120 --interval 60
```

### 3. List Available Strategies

```bash
python unified_trader.py list
```

Output:
```
[GENERAL]
  + defensive_yield              RL-driven defensive strategy with yield
  + mean_reversion_vwap          VWAP + RSI mean reversion for choppy markets
  + xrp_btc_pair_trading         XRP/BTC cointegration stat arb
  ...

[SCALPER]
  + intraday_scalper             BB squeeze + RSI volatility scalper
  - ema9_scalper                 EMA-9 crossover with privileged override

[GRID]
  - grid_arithmetic              Fixed spacing grid - consistent fills
  ...
```

### 4. Run an Experiment

```bash
# Use preset experiment
python unified_trader.py experiment --preset aggressive

# Custom parameter override
python unified_trader.py experiment --override mean_reversion_vwap:rsi_oversold:30
```

### 5. Analyze Results

```bash
# List recent experiments
python unified_trader.py analyze

# Analyze specific experiment
python unified_trader.py analyze --experiment exp_aggressive_20251212

# Compare experiments
python unified_trader.py compare exp_1 exp_2 exp_3
```

---

## CLI Reference

### `unified_trader.py paper`

Run paper trading with unified orchestrator.

```bash
python unified_trader.py paper [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--config` | str | `strategies_config/unified.yaml` | Path to config file |
| `--duration` | int | 60 | Duration in minutes |
| `--interval` | int | 300 | Decision interval in seconds |
| `--experiment-id` | str | auto-generated | Custom experiment ID |
| `--enable` | list | - | Enable specific strategies |
| `--disable` | list | - | Disable specific strategies |

**Examples:**
```bash
# Run for 2 hours
python unified_trader.py paper --duration 120

# Enable grid strategies at runtime
python unified_trader.py paper --enable grid_arithmetic grid_geometric

# Disable a strategy
python unified_trader.py paper --disable ema9_scalper
```

---

### `unified_trader.py experiment`

Run an experiment with parameter overrides.

```bash
python unified_trader.py experiment [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--config` | str | `strategies_config/unified.yaml` | Path to config file |
| `--preset` | str | - | Experiment preset name |
| `--override` | list | - | Parameter overrides (`strategy:param:value`) |
| `--duration` | int | 60 | Duration in minutes |
| `--interval` | int | 300 | Decision interval in seconds |
| `--experiment-id` | str | auto-generated | Custom experiment ID |

**Examples:**
```bash
# Use aggressive preset
python unified_trader.py experiment --preset aggressive

# Custom overrides
python unified_trader.py experiment \
    --override mean_reversion_vwap:rsi_oversold:30 \
    --override xrp_btc_pair_trading:entry_z:1.5

# Combined preset + overrides
python unified_trader.py experiment --preset conservative \
    --override defensive_yield:max_leverage:5
```

---

### `unified_trader.py list`

List all available strategies and their status.

```bash
python unified_trader.py list [--config PATH]
```

---

### `unified_trader.py config`

Manage strategy configuration.

```bash
python unified_trader.py config [OPTIONS]
```

| Option | Type | Description |
|--------|------|-------------|
| `--config` | str | Path to config file |
| `--enable` | list | Enable strategies by name |
| `--disable` | list | Disable strategies by name |
| `--set` | list | Set parameters (`path.to.param=value`) |

**Examples:**
```bash
# Enable grid strategies
python unified_trader.py config --enable grid_arithmetic grid_geometric

# Disable scalpers
python unified_trader.py config --disable intraday_scalper ema9_scalper

# Set a parameter
python unified_trader.py config --set strategies.mean_reversion_vwap.rsi_oversold=32

# Set global parameter
python unified_trader.py config --set global.starting_balance.USDT=5000
```

---

### `unified_trader.py init-config`

Create a new configuration template.

```bash
python unified_trader.py init-config [--output PATH] [--force]
```

---

### `unified_trader.py analyze`

Analyze experiment logs.

```bash
python unified_trader.py analyze [--experiment ID]
```

Without `--experiment`, lists recent experiments. With ID, shows detailed analysis.

---

### `unified_trader.py compare`

Compare multiple experiments.

```bash
python unified_trader.py compare EXP_ID_1 EXP_ID_2 [EXP_ID_3 ...]
```

Output:
```
Experiment                     Strategies     Trades          PnL
----------------------------------------------------------------------
exp_aggressive_20251212                 6         45       $125.50
exp_conservative_20251212               6         23        $78.20
exp_grid_focus_20251212                 4         89       $156.80
----------------------------------------------------------------------
Best: exp_grid_focus_20251212 ($156.80)
```

---

## Configuration

### Configuration File Structure

```yaml
# strategies_config/unified.yaml

global:
  paper_trading: true
  starting_balance:
    USDT: 2000.0
    XRP: 0.0
    BTC: 0.0
  fee_rate: 0.001
  max_drawdown: 0.10
  risk_per_trade: 0.10
  max_concurrent_positions: 5

strategies:
  strategy_name:
    enabled: true|false
    category: general|scalper|grid|margin
    description: "Human readable description"
    # Strategy-specific parameters...
    param1: value1
    param2: value2

experiments:
  preset_name:
    description: "Preset description"
    overrides:
      strategy_name:
        param: value
    enable_only:  # Optional: only enable these strategies
      - strategy1
      - strategy2
```

### Available Strategies

| Name | Category | Description | Default |
|------|----------|-------------|---------|
| `defensive_yield` | general | RL-driven with yield accrual | enabled |
| `mean_reversion_vwap` | general | VWAP + RSI mean reversion | enabled |
| `xrp_btc_pair_trading` | general | XRP/BTC cointegration stat arb | enabled |
| `ma_trend_follow` | general | SMA-9 trend following | enabled |
| `xrp_btc_leadlag` | general | XRP/BTC correlation lead-lag | enabled |
| `intraday_scalper` | scalper | BB squeeze + RSI scalper | enabled |
| `ema9_scalper` | scalper | EMA-9 privileged override | disabled |
| `grid_arithmetic` | grid | Fixed spacing grid | disabled |
| `grid_geometric` | grid | Percentage spacing grid | disabled |
| `grid_rsi_reversion` | grid | RSI-filtered grid | disabled |
| `grid_bb_squeeze` | grid | BB squeeze detection grid | disabled |
| `grid_trend_margin` | margin | 5x trend-following margin | disabled |
| `grid_dual_hedge` | margin | Dual grid with hedge | disabled |
| `grid_time_weighted` | grid | Time-weighted entry grid | disabled |
| `grid_liq_hunter` | margin | Liquidation zone scalper | disabled |

### Strategy Parameters

#### `mean_reversion_vwap`
```yaml
mean_reversion_vwap:
  enabled: true
  symbol: "XRP/USDT"
  dev_threshold: 0.003      # VWAP deviation (0.3%)
  rsi_oversold: 35          # RSI buy threshold
  rsi_overbought: 65        # RSI sell threshold
  volume_mult: 1.3          # Volume filter (1.3x average)
  max_leverage: 5
  stop_loss_pct: 0.04       # 4% stop loss
```

#### `xrp_btc_pair_trading`
```yaml
xrp_btc_pair_trading:
  enabled: true
  lookback: 336             # Hours for hedge ratio calc
  entry_z: 1.8              # Z-score entry threshold
  exit_z: 0.5               # Z-score exit threshold
  stop_z: 3.0               # Z-score stop loss
  max_leverage: 10
```

#### `intraday_scalper`
```yaml
intraday_scalper:
  enabled: true
  atr_threshold: 0.007      # 0.7% daily ATR to activate
  rsi_oversold: 30
  rsi_overbought: 70
  max_leverage: 3
  stop_loss_pct: 0.005      # 0.5%
  take_profit_pct: 0.01     # 1%
```

---

## Strategy Registry

### Overview

The `StrategyRegistry` class manages all trading strategies with:

- Auto-discovery of built-in strategies
- YAML-based configuration loading
- Runtime enable/disable
- Experiment parameter overrides
- Lazy class loading for performance

### Usage

```python
from strategy_registry import StrategyRegistry

# Initialize with config
registry = StrategyRegistry("strategies_config/unified.yaml")

# List enabled strategies
enabled = registry.get_enabled_strategies()
# ['defensive_yield', 'mean_reversion_vwap', ...]

# Enable/disable strategies
registry.enable('grid_arithmetic')
registry.disable('ema9_scalper')

# Enable entire category
registry.enable_category('grid')

# Get strategy parameters
params = registry.get_params('mean_reversion_vwap')
# {'dev_threshold': 0.003, 'rsi_oversold': 35, ...}

# Override parameter for experiment
registry.override_param('mean_reversion_vwap', 'rsi_oversold', 30)

# Instantiate a strategy
strategy = registry.instantiate('mean_reversion_vwap')

# Instantiate all enabled strategies
strategies = registry.instantiate_all_enabled()

# Print status
registry.print_status()
```

### API Reference

#### `StrategyRegistry.__init__(config_path: str = None)`
Initialize registry, optionally loading configuration from YAML.

#### `registry.load_config(config_path: str)`
Load or reload configuration from YAML file.

#### `registry.enable(name: str) -> bool`
Enable a strategy by name. Returns True if successful.

#### `registry.disable(name: str) -> bool`
Disable a strategy by name. Returns True if successful.

#### `registry.enable_category(category: str)`
Enable all strategies in a category (general, scalper, grid, margin).

#### `registry.disable_category(category: str)`
Disable all strategies in a category.

#### `registry.enable_all()`
Enable all registered strategies.

#### `registry.disable_all()`
Disable all registered strategies.

#### `registry.override_param(strategy_name: str, param_name: str, value: Any)`
Override a parameter value for experiments.

#### `registry.clear_overrides(strategy_name: str = None)`
Clear experiment overrides. If no name given, clears all.

#### `registry.get_params(name: str) -> Dict[str, Any]`
Get merged parameters (default + config + overrides).

#### `registry.get_enabled_strategies() -> List[str]`
Get list of enabled strategy names.

#### `registry.get_strategies_by_category(category: str) -> List[str]`
Get strategies in a specific category.

#### `registry.instantiate(name: str, extra_config: Dict = None) -> BaseStrategy`
Instantiate a strategy with its configuration.

#### `registry.instantiate_all_enabled(extra_config: Dict = None) -> Dict[str, BaseStrategy]`
Instantiate all enabled strategies.

#### `registry.register(name, class_type, category, description, default_params)`
Register a custom strategy class.

#### `registry.get_status() -> Dict[str, Any]`
Get comprehensive registry status.

#### `registry.print_status()`
Print formatted registry status.

---

## Unified Orchestrator

### Overview

The `UnifiedOrchestrator` combines all strategies with:

- **Weighted voting** - Each strategy contributes based on its weight
- **Regime detection** - Automatically adjusts weights based on market conditions
- **Paper execution** - Simulates trades with portfolio tracking
- **Per-strategy logging** - Every strategy gets its own log file

### Usage

```python
from unified_orchestrator import UnifiedOrchestrator
from portfolio import Portfolio

# Initialize
portfolio = Portfolio({'USDT': 2000.0, 'XRP': 0.0, 'BTC': 0.0})
orchestrator = UnifiedOrchestrator(
    portfolio=portfolio,
    config_path="strategies_config/unified.yaml",
    experiment_id="my_experiment"
)

# Add/remove strategies at runtime
orchestrator.add_strategy('grid_arithmetic')
orchestrator.remove_strategy('ema9_scalper')

# Set strategy weight
orchestrator.set_weight('mean_reversion_vwap', 0.25)

# Set experiment parameter
orchestrator.set_experiment_param('mean_reversion_vwap', 'rsi_oversold', 30)

# Manual decision cycle
orchestrator.update_data()           # Fetch market data
orchestrator.update_prices()         # Update current prices
regime = orchestrator.detect_regime() # Detect market regime
signals = orchestrator.get_all_signals()  # Get all strategy signals
decision = orchestrator.weighted_vote(signals)  # Combine via voting
result = orchestrator.execute_paper(decision)   # Execute paper trade

# Or use the main loop
orchestrator.run_loop(
    duration_minutes=60,
    interval_seconds=300,
    on_decision=lambda decision, result: print(f"Decided: {decision['action']}")
)

# Get status
status = orchestrator.get_status()
orchestrator.print_status()

# Close (generates summaries)
orchestrator.close()
```

### Regime Detection

The orchestrator automatically detects market regime and adjusts strategy weights:

| Regime | Condition | Favored Strategies |
|--------|-----------|-------------------|
| `high_volatility` | ATR > 4% | intraday_scalper, ema9_scalper |
| `low_volatility` | ATR < 1.5% | defensive_yield, grid_arithmetic |
| `trend_up` | Price > SMA20 > SMA50 | ma_trend_follow, xrp_btc_leadlag |
| `trend_down` | Price < SMA20 < SMA50 | defensive_yield, grid_dual_hedge |
| `chop` | Sideways market | mean_reversion_vwap, xrp_btc_pair_trading |

### API Reference

#### `UnifiedOrchestrator.__init__(portfolio, config_path, experiment_id)`
Initialize orchestrator with portfolio and configuration.

#### `orchestrator.add_strategy(name: str, config: Dict = None) -> bool`
Add a strategy at runtime.

#### `orchestrator.remove_strategy(name: str) -> bool`
Remove a strategy at runtime.

#### `orchestrator.set_weight(name: str, weight: float)`
Set a specific strategy's weight.

#### `orchestrator.apply_regime_weights(regime: str)`
Apply preset weights for a specific regime.

#### `orchestrator.set_experiment_param(strategy_name: str, param: str, value: Any)`
Override a parameter for experiments.

#### `orchestrator.update_data(symbols: List[str] = None)`
Fetch fresh market data from exchanges.

#### `orchestrator.update_prices(prices: Dict[str, float] = None)`
Update current prices (fetches if not provided).

#### `orchestrator.detect_regime() -> RegimeState`
Detect current market regime from data.

#### `orchestrator.get_all_signals() -> Dict[str, Dict]`
Get signals from all enabled strategies.

#### `orchestrator.weighted_vote(signals: Dict) -> Dict[str, Any]`
Combine signals via weighted voting.

#### `orchestrator.decide() -> Dict[str, Any]`
Main decision loop (update + detect + vote).

#### `orchestrator.execute_paper(decision: Dict) -> Dict[str, Any]`
Execute decision in paper trading mode.

#### `orchestrator.run_loop(duration_minutes, interval_seconds, on_decision)`
Run the main trading loop.

#### `orchestrator.get_status() -> Dict[str, Any]`
Get comprehensive orchestrator status.

#### `orchestrator.print_status()`
Print formatted status.

#### `orchestrator.close() -> Dict[str, Any]`
Close orchestrator and generate summaries.

---

## Logging System

### Overview

The logging system provides:

- **Per-strategy logs** - Each strategy gets `logs/strategies/{name}_{timestamp}.jsonl`
- **Experiment logs** - Combined experiment log `logs/experiments/{experiment_id}.jsonl`
- **Performance metrics** - Rolling metrics `logs/performance/{name}_metrics.json`
- **Orchestrator logs** - Master log `logs/orchestrator/unified_{timestamp}.jsonl`

### Log File Structure

```
logs/
├── strategies/
│   ├── defensive_yield_20251212_143022.jsonl
│   ├── mean_reversion_vwap_20251212_143022.jsonl
│   └── ...
├── experiments/
│   └── exp_aggressive_20251212_143022.jsonl
├── performance/
│   ├── defensive_yield_metrics.json
│   ├── mean_reversion_vwap_metrics.json
│   └── ...
└── orchestrator/
    └── unified_20251212_143022.jsonl
```

### Log Entry Types

#### Strategy Log Entries

```json
{"type": "session_start", "strategy": "mean_reversion_vwap", "timestamp": "..."}
{"type": "config", "strategy": "...", "config": {...}}
{"type": "signal", "action": "buy", "symbol": "XRP/USDT", "confidence": 0.75, ...}
{"type": "execution", "action": "buy", "executed": true, "price": 2.15, ...}
{"type": "trade_close", "side": "long", "entry_price": 2.15, "exit_price": 2.18, "net_pnl": 12.50}
{"type": "session_end", "summary": {...}}
```

#### Orchestrator Log Entries

```json
{"type": "unified_session_start", "experiment_id": "...", "timestamp": "..."}
{"type": "experiment_config", "config": {...}}
{"type": "orchestrator_decision", "strategy_signals": {...}, "final_action": "buy", "weights": {...}}
{"type": "unified_session_end", "combined_pnl": 125.50, "combined_trades": 45}
```

### Using the Logger

```python
from utils.strategy_logger import StrategyLogger, StrategyLoggerManager

# Single strategy logger
logger = StrategyLogger(
    strategy_name="my_strategy",
    experiment_id="exp_001",
    log_dir="logs"
)

logger.log_config({'param1': 10, 'param2': 0.5})
logger.log_signal(action='buy', symbol='XRP/USDT', confidence=0.75)
logger.log_execution(action='buy', executed=True, price=2.15, size=100)
logger.log_trade_close(entry_price=2.15, exit_price=2.18, size=100, leverage=5, side='long')

summary = logger.close()

# Multi-strategy manager
manager = StrategyLoggerManager(experiment_id="exp_001")

# Get logger for each strategy
logger1 = manager.get_logger("strategy_a")
logger2 = manager.get_logger("strategy_b")

# Log orchestrator decision
manager.log_orchestrator_decision(
    strategy_signals={'strategy_a': {...}, 'strategy_b': {...}},
    final_action='buy',
    final_confidence=0.8,
    weights={'strategy_a': 0.6, 'strategy_b': 0.4},
    regime='chop'
)

# Close all and get combined summary
summary = manager.close_all()
```

### API Reference

#### `StrategyLogger.__init__(strategy_name, experiment_id, log_dir)`
Initialize logger for a specific strategy.

#### `logger.log_config(config: Dict)`
Log strategy configuration.

#### `logger.log_signal(action, symbol, confidence, leverage, size, reason, indicators, price)`
Log a generated signal.

#### `logger.log_execution(action, executed, price, size, reason, result)`
Log trade execution attempt.

#### `logger.log_trade_close(entry_price, exit_price, size, leverage, side, fee)`
Log a closed trade with PnL calculation.

#### `logger.log_market_state(prices, indicators, regime)`
Log market state snapshot.

#### `logger.log_experiment_param(param_name, param_value, description)`
Log an experiment parameter variation.

#### `logger.log_insight(category, insight, data)`
Log a tuning insight or observation.

#### `logger.get_summary() -> Dict`
Generate summary statistics.

#### `logger.close() -> Dict`
Close logger and generate final summary.

#### `StrategyLoggerManager.__init__(experiment_id, log_dir)`
Initialize manager for multiple strategies.

#### `manager.get_logger(strategy_name) -> StrategyLogger`
Get or create a logger for a strategy.

#### `manager.log_orchestrator_decision(strategy_signals, final_action, final_confidence, weights, regime)`
Log the orchestrator's combined decision.

#### `manager.log_experiment_config(config)`
Log full experiment configuration.

#### `manager.close_all() -> Dict`
Close all loggers and generate combined summary.

---

## Experiments & A/B Testing

### Defining Experiment Presets

In `unified.yaml`:

```yaml
experiments:
  aggressive:
    description: "Higher leverage, tighter thresholds"
    overrides:
      mean_reversion_vwap:
        max_leverage: 7
        rsi_oversold: 32
        rsi_overbought: 68
      xrp_btc_pair_trading:
        entry_z: 1.5

  conservative:
    description: "Lower leverage, wider thresholds"
    overrides:
      mean_reversion_vwap:
        max_leverage: 3
        rsi_oversold: 38
      xrp_btc_pair_trading:
        entry_z: 2.2

  grid_focus:
    description: "Enable grid strategies only"
    enable_only:
      - grid_arithmetic
      - grid_geometric
      - grid_rsi_reversion
      - grid_bb_squeeze
```

### Running Experiments

```bash
# Run preset
python unified_trader.py experiment --preset aggressive --duration 60

# Custom overrides
python unified_trader.py experiment \
    --override mean_reversion_vwap:rsi_oversold:28 \
    --override mean_reversion_vwap:rsi_overbought:72

# Combined
python unified_trader.py experiment --preset conservative \
    --override defensive_yield:max_leverage:8
```

### Comparing Results

```bash
# Run multiple experiments
python unified_trader.py experiment --preset aggressive --experiment-id exp_agg
python unified_trader.py experiment --preset conservative --experiment-id exp_con

# Compare
python unified_trader.py compare exp_agg exp_con
```

### A/B Testing Workflow

1. **Define hypothesis**: "RSI 32/68 will generate more trades than 35/65"

2. **Create presets**:
```yaml
experiments:
  rsi_tight:
    overrides:
      mean_reversion_vwap:
        rsi_oversold: 32
        rsi_overbought: 68
  rsi_loose:
    overrides:
      mean_reversion_vwap:
        rsi_oversold: 35
        rsi_overbought: 65
```

3. **Run experiments**:
```bash
python unified_trader.py experiment --preset rsi_tight --duration 120
python unified_trader.py experiment --preset rsi_loose --duration 120
```

4. **Analyze**:
```bash
python unified_trader.py analyze
python unified_trader.py compare rsi_tight rsi_loose
```

5. **Review logs**:
```bash
cat logs/strategies/mean_reversion_vwap_*.jsonl | jq 'select(.type=="signal")'
```

---

## Adding New Strategies

### Step 1: Create Strategy Class

```python
# src/strategies/my_strategy/strategy.py

from strategies.base_strategy import BaseStrategy
from typing import Dict, Any
import pandas as pd

class MyStrategy(BaseStrategy):
    """My custom trading strategy."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Extract parameters from config
        self.threshold = config.get('threshold', 0.5)
        self.leverage = config.get('max_leverage', 5)

    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Generate trading signal from market data."""

        # Get XRP data
        xrp_df = data.get('XRP/USDT')
        if xrp_df is None or len(xrp_df) < 20:
            return {'action': 'hold', 'confidence': 0, 'reason': 'Insufficient data'}

        close = xrp_df['close'].iloc[-1]
        sma = xrp_df['close'].rolling(20).mean().iloc[-1]

        # Generate signal
        if close < sma * (1 - self.threshold):
            return {
                'action': 'buy',
                'symbol': 'XRP/USDT',
                'confidence': 0.7,
                'leverage': self.leverage,
                'size': 0.1,
                'reason': f'Price {close:.4f} below SMA {sma:.4f}',
                'indicators': {'close': close, 'sma': sma}
            }
        elif close > sma * (1 + self.threshold):
            return {
                'action': 'sell',
                'symbol': 'XRP/USDT',
                'confidence': 0.7,
                'leverage': 1,
                'size': 0.5,
                'reason': f'Price {close:.4f} above SMA {sma:.4f}',
                'indicators': {'close': close, 'sma': sma}
            }
        else:
            return {
                'action': 'hold',
                'confidence': 0,
                'symbol': 'XRP/USDT',
                'reason': 'No signal'
            }

    def update_model(self, data=None) -> bool:
        """Update strategy model (no-op for this strategy)."""
        return True
```

### Step 2: Register Strategy

Option A: Add to built-in strategies in `strategy_registry.py`:

```python
BUILTIN_STRATEGIES = {
    # ... existing strategies ...
    'my_strategy': ('strategies.my_strategy.strategy', 'MyStrategy'),
}

CATEGORY_MAP = {
    # ... existing mappings ...
    'my_strategy': 'general',
}
```

Option B: Register at runtime:

```python
from strategies.my_strategy.strategy import MyStrategy

registry.register(
    name='my_strategy',
    strategy_class=MyStrategy,
    category='custom',
    description='My custom trading strategy',
    default_params={'threshold': 0.5, 'max_leverage': 5}
)
```

### Step 3: Add Configuration

```yaml
# In unified.yaml
strategies:
  my_strategy:
    enabled: true
    category: custom
    description: "My custom trading strategy"
    threshold: 0.05
    max_leverage: 5
```

### Step 4: Test

```bash
# Enable and run
python unified_trader.py config --enable my_strategy
python unified_trader.py paper --duration 10

# Check logs
cat logs/strategies/my_strategy_*.jsonl | head -20
```

---

## API Reference

### Data Classes

#### `RegimeState`
```python
@dataclass
class RegimeState:
    name: str = "neutral"      # chop, trend_up, trend_down, high_volatility, low_volatility
    volatility: float = 0.0    # ATR percentage
    correlation: float = 0.0   # XRP/BTC correlation
    trend: str = "sideways"    # up, down, sideways
    rsi: Dict[str, float]      # {'BTC': 55.0, 'XRP': 48.0}
    atr: Dict[str, float]      # {'BTC': 0.025}
```

#### `StrategyInfo`
```python
@dataclass
class StrategyInfo:
    name: str
    class_type: Type[BaseStrategy]
    module_path: str
    description: str = ""
    category: str = "general"
    enabled: bool = True
    config: Dict[str, Any]
    default_params: Dict[str, Any]
```

### Signal Format

```python
{
    'action': str,        # 'buy', 'sell', 'short', 'cover', 'hold', 'close'
    'symbol': str,        # 'XRP/USDT', 'BTC/USDT'
    'confidence': float,  # 0.0 to 1.0
    'leverage': int,      # 1 for spot, 2-10 for margin
    'size': float,        # Position size as fraction (0.0 to 1.0)
    'reason': str,        # Human-readable explanation
    'indicators': Dict,   # Strategy-specific indicators
}
```

### Decision Format (from weighted_vote)

```python
{
    'action': str,
    'confidence': float,
    'weighted_score': float,
    'contributing_strategies': List[str],
    'regime': str,
    'reason': str,
    'symbol': str,          # If action != 'hold'
    'leverage': int,        # If action != 'hold'
    'size': float,          # If action != 'hold'
    'primary_strategy': str # Highest confidence contributor
}
```

### Execution Result Format

```python
{
    'executed': bool,
    'action': str,
    'reason': str,
    'decision': Dict,       # Original decision
    'amount': float,        # If executed
    'price': float,         # If executed
    'cost': float,          # For buys
    'proceeds': float,      # For sells
}
```

---

## Troubleshooting

### NumPy Version Conflict

If you see `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x`:

```bash
pip install "numpy<2"
# or
pip install --upgrade pandas pyarrow bottleneck numexpr
```

### Strategy Not Loading

Check the module path in `strategy_registry.py`:
```python
BUILTIN_STRATEGIES = {
    'strategy_name': ('strategies.module.file', 'ClassName'),
}
```

Ensure the file exists at `src/strategies/module/file.py` and contains the class.

### No Signals Generated

1. Check if strategy is enabled: `python unified_trader.py list`
2. Check strategy thresholds in config
3. Review strategy logs: `cat logs/strategies/{name}*.jsonl | jq 'select(.type=="signal")'`

### Low Execution Rate

1. Check confidence threshold (default: 0.35)
2. Review rejection reasons in logs
3. Consider lowering strategy thresholds

---

## Migration from Old Scripts

### From `main.py`

```python
# Old
python main.py --mode paper

# New
python unified_trader.py paper
```

### From `grid_ensemble_orchestrator.py`

```python
# Old
python grid_ensemble_orchestrator.py --duration 60

# New
python unified_trader.py config --enable grid_arithmetic grid_geometric grid_rsi_reversion grid_bb_squeeze
python unified_trader.py paper --duration 60
```

### Using Both Systems

The old scripts (`main.py`, `grid_ensemble_orchestrator.py`) still work independently. You can run either system based on your needs.

---

*Documentation generated for Phase 24 - December 2024*
