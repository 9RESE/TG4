# Strategy Parameter Optimization System v1.0

## Overview

The optimization system provides automated parameter tuning for all 9 trading strategies using grid search with optional parallel execution. It leverages the backtest runner and pre-aggregated candle data for efficient parameter exploration.

## Components

### Base Optimizer (`optimization/base_optimizer.py`)

Abstract base class providing:
- Parameter grid generation and combination counting
- Sequential and parallel execution modes
- Results collection and ranking
- JSON report generation

```python
from optimization.base_optimizer import BaseOptimizer, OptimizationConfig

config = OptimizationConfig(
    strategy_name='ema9_trend_flip',
    symbol='BTC/USDT',
    param_grid={},
    db_url='postgresql://...',
    period='3m',
    starting_capital=100.0,
    parallel=True,
    max_workers=8,
)
```

### Strategy-Specific Optimizers

Each strategy has a dedicated optimizer with domain-specific parameter grids:

| Optimizer | Strategy | Focus Areas |
|-----------|----------|-------------|
| `optimize_ema9.py` | EMA-9 Trend Flip | EMA period, consecutive candles, buffer % |
| `optimize_wavetrend.py` | WaveTrend Oscillator | Channel length, thresholds, zones |
| `optimize_momentum.py` | Momentum Scalping | Volume thresholds, flow settings |
| `optimize_mean_reversion.py` | Mean Reversion | RSI periods, deviation thresholds |
| `optimize_grid_rsi.py` | Grid RSI Reversion | Grid spacing, RSI zones, ADX filter |
| `optimize_whale_sentiment.py` | Whale Sentiment | Volume spike mult, sentiment zones |
| `optimize_order_flow.py` | Order Flow | VPIN thresholds, imbalance settings |
| `optimize_market_making.py` | Market Making | Spread %, inventory skew, gamma |
| `optimize_ratio_trading.py` | Ratio Trading | Bollinger settings, entry/exit thresholds |

### Batch Runner (`optimization/run_optimization.py`)

Orchestrates multiple optimization jobs:

```bash
# Quick optimization for all strategies
python run_optimization.py --quick

# Specific strategy and symbol
python run_optimization.py --strategy ema9 --symbol BTC/USDT

# Full optimization with parallel execution
python run_optimization.py --full --parallel --workers 8

# List available jobs
python run_optimization.py --list
```

## Usage

### Quick Mode (Reduced Grid)

```bash
cd ws_paper_tester/optimization
python optimize_ema9.py --symbol BTC/USDT --period 3m --quick
```

Quick mode uses smaller parameter grids for faster exploration:
- ~200-500 combinations instead of 2000+
- ~10-30 minutes instead of hours
- Good for initial exploration

### Full Mode

```bash
python optimize_ema9.py --symbol BTC/USDT --period 6m
```

Full mode explores comprehensive parameter space:
- 1000-5000+ combinations
- Several hours to complete
- Recommended for final optimization

### Focused Mode

```bash
python optimize_ema9.py --symbol BTC/USDT --focus ema_period
```

Focused mode optimizes specific parameter groups:
- `ema_period`: EMA and consecutive candle settings
- `risk_reward`: Stop loss and take profit ratios
- `thresholds`: Entry/exit threshold tuning

### Parallel Execution

```bash
python optimize_ema9.py --symbol BTC/USDT --parallel --workers 8 --chunk-size 50
```

Parallel execution runs multiple backtests concurrently:
- `--workers`: Number of parallel processes (default: 8)
- `--chunk-size`: Batch size for parallel processing (default: 50)
- Requires sufficient RAM (~2GB per worker)

## Output

### Console Output

```
EMA-9 Optimizer - BTC/USDT
Combinations: 243, Est. time: ~41 min

Proceed? [y/N]: y

[1/243] ema_period=7, consecutive_candles=2, buffer_pct=0.05
  Result: 12 trades, $2.45 P&L, 58.3% win rate
[2/243] ema_period=7, consecutive_candles=2, buffer_pct=0.10
  Result: 10 trades, $1.87 P&L, 50.0% win rate
...

=== OPTIMIZATION COMPLETE ===
Best by P&L: $4.56 (ema_period=9, consecutive_candles=3, buffer_pct=0.10)
Best by Win Rate: 66.7% (ema_period=11, consecutive_candles=4, buffer_pct=0.15)
```

### JSON Report

Results saved to `optimization_results/`:

```json
{
  "strategy": "ema9_trend_flip",
  "symbol": "BTC/USDT",
  "period": "3m",
  "timestamp": "20251216_143022",
  "total_combinations": 243,
  "best_by_pnl": {
    "params": {"ema_period": 9, "consecutive_candles": 3},
    "total_pnl": 4.56,
    "win_rate": 58.3,
    "total_trades": 15
  },
  "best_by_win_rate": {
    "params": {"ema_period": 11, "consecutive_candles": 4},
    "total_pnl": 2.12,
    "win_rate": 66.7,
    "total_trades": 9
  },
  "all_results": [...]
}
```

## Parameter Grids

### EMA-9 Trend Flip

```python
# Quick mode
{
    'ema_period': [7, 9, 11],
    'consecutive_candles': [2, 3, 4],
    'buffer_pct': [0.05, 0.10, 0.15],
    'stop_loss_pct': [0.8, 1.0, 1.2],
    'take_profit_pct': [1.5, 2.0, 2.5],
}

# Full mode
{
    'ema_period': [5, 7, 9, 11, 13, 15],
    'consecutive_candles': [2, 3, 4, 5],
    'buffer_pct': [0.0, 0.05, 0.10, 0.15, 0.20],
    'stop_loss_pct': [0.5, 0.8, 1.0, 1.2, 1.5],
    'take_profit_pct': [1.0, 1.5, 2.0, 2.5, 3.0],
}
```

### Market Making

```python
{
    'min_spread_pct': [0.03, 0.05, 0.08, 0.10],
    'imbalance_threshold': [0.05, 0.08, 0.10, 0.12],
    'inventory_skew': [0.2, 0.3, 0.5, 0.7],
    'gamma': [0.05, 0.1, 0.15, 0.2],  # A-S risk aversion
}
```

## Best Practices

1. **Start with Quick Mode**: Get initial results in minutes
2. **Use Focused Mode**: Drill down on promising parameter ranges
3. **Validate with Full Mode**: Final optimization with comprehensive grid
4. **Monitor Memory**: Parallel mode requires ~2GB RAM per worker
5. **Use Recent Data**: Optimize on recent market conditions (3-6 months)
6. **Walk-Forward Validation**: Test optimized params on out-of-sample data

## Integration with Backtest Runner

The optimization system uses `backtest_runner.py` internally:

```python
from backtest_runner import BacktestConfig, BacktestExecutor

# Each parameter combination runs a full backtest
config = BacktestConfig(
    start_time=start,
    end_time=end,
    symbols=[symbol],
    strategies=[strategy_name],
    starting_capital=100.0,
)

result = await executor.run_strategy(strategy, start, end)
```

## Database Requirements

- TimescaleDB with pre-aggregated candle tables
- Minimum 3 months of historical data recommended
- Tables: `candles`, `candles_5m`, `candles_1h`, `candles_1d`

## Version History

- **v1.0.0** (2025-12-16): Initial release with all 9 strategy optimizers
