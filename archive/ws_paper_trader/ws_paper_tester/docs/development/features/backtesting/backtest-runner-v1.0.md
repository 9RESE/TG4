# Backtest Runner v1.0

## Overview

The backtest runner provides comprehensive historical backtesting for all strategies using data from TimescaleDB. It supports multi-timeframe candle data, realistic trade execution simulation, and detailed performance reporting.

## Features

- **Multi-Strategy Support**: Auto-discovers and backtests all strategies in `strategies/`
- **Pre-Aggregated Data**: Uses pre-computed 5m, 1h, 1d candles for performance
- **Realistic Execution**: Configurable fees (0.1%) and slippage (0.05%)
- **Detailed Metrics**: P&L, win rate, Sharpe ratio, profit factor, max drawdown
- **Trade Logging**: Complete trade history with timestamps, prices, and reasons
- **Equity Curve**: Tracks portfolio value over time for drawdown analysis

## Usage

### Basic Usage

```bash
cd ws_paper_tester

# Run all strategies for last year
python backtest_runner.py

# Specific strategies
python backtest_runner.py --strategies ema9_trend_flip,wavetrend

# Specific date range
python backtest_runner.py --start 2024-01-01 --end 2024-06-01

# Specific symbols
python backtest_runner.py --symbols XRP/USDT BTC/USDT
```

### Period Shortcuts

```bash
python backtest_runner.py --period 1w   # Last week
python backtest_runner.py --period 1m   # Last month
python backtest_runner.py --period 3m   # Last 3 months
python backtest_runner.py --period 6m   # Last 6 months
python backtest_runner.py --period 1y   # Last year
python backtest_runner.py --period all  # All available data
```

### Configuration Options

```bash
python backtest_runner.py \
    --db-url postgresql://user:pass@localhost:5433/kraken_data \
    --capital 1000.0 \
    --output backtest_results \
    --log-level DEBUG
```

## Configuration

### BacktestConfig

```python
@dataclass
class BacktestConfig:
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    symbols: List[str] = ['XRP/USDT', 'BTC/USDT', 'XRP/BTC']
    strategies: Optional[List[str]] = None  # None = all
    starting_capital: float = 100.0
    fee_rate: float = 0.001      # 0.1%
    slippage_rate: float = 0.0005  # 0.05%
    interval_minutes: int = 1     # Base candle interval
    warmup_periods: int = 2000    # Candles for strategy warmup
```

## Data Flow

### Candle Loading

The backtest runner loads candles from pre-aggregated tables:

```
candles (1m)     → Primary iteration timeframe
candles_5m       → Strategies using 5-minute data
candles_1h       → Hourly strategies (EMA-9, WaveTrend)
candles_1d       → Daily strategies (future)
```

### Snapshot Creation

Each tick creates a `DataSnapshot` with all available timeframes:

```python
snapshot = DataSnapshot(
    timestamp=timestamp,
    prices={'BTC/USDT': 87000.0},
    candles_1m={...},     # Last 2000 1-minute candles
    candles_5m={...},     # Last 2000 5-minute candles
    candles_1h={...},     # Last 500 hourly candles
    candles_1d={...},     # Last 365 daily candles
    orderbooks={...},     # Simulated from spread
    trades={},            # Empty for historical backtest
)
```

## Trade Execution

### Signal Processing

```python
def _execute_signal(signal, portfolio, current_price):
    # 1. Apply slippage
    if signal.action in ('buy', 'cover'):
        exec_price = current_price * 1.0005  # Buy higher
    else:
        exec_price = current_price * 0.9995  # Sell lower

    # 2. Detect USD vs base asset sizing
    if signal.size <= 1000 and exec_price > 100:
        base_size = signal.size / exec_price  # USD to base
        usd_value = signal.size
    else:
        base_size = signal.size
        usd_value = signal.size * exec_price

    # 3. Calculate fee
    fee = usd_value * 0.001

    # 4. Execute and update portfolio
    ...
```

### P&L Calculation

Exit signals include `entry_price` in metadata for accurate P&L:

```python
# Long exit
entry_price = signal.metadata.get('entry_price', signal.price)
entry_value = size * entry_price
exit_value = size * exec_price
pnl = exit_value - entry_value - (fee * 2)

# Short exit (inverted)
pnl = entry_value - exit_value - (fee * 2)
```

## Output

### Console Summary

```
================================================================================
BACKTEST RESULTS
================================================================================

Strategy                       P&L       P&L%   Trades   Win%   MaxDD%       PF
--------------------------------------------------------------------------------
ema9_trend_flip              $4.56      4.6%       15   60.0%     2.1%     1.85
wavetrend                    $2.12      2.1%       28   53.6%     3.2%     1.42
mean_reversion              -$0.89     -0.9%       42   47.6%     4.5%     0.91
--------------------------------------------------------------------------------
TOTAL                        $5.79             85

PER-SYMBOL BREAKDOWN
--------------------------------------------------------------------------------
BTC/USDT:
  ema9_trend_flip             $3.21 (8 trades)
  wavetrend                   $1.45 (12 trades)

PERFORMANCE HIGHLIGHTS
--------------------------------------------------------------------------------
Best Performer:  ema9_trend_flip (+$4.56, 60.0% win rate)
Most Active:     mean_reversion (42 trades)
Best Win Rate:   ema9_trend_flip (60.0% on 15 trades)
```

### JSON Export

Results saved to `backtest_results/backtest_summary_YYYYMMDD_HHMMSS.json`:

```json
{
  "timestamp": "20251216_143022",
  "total_pnl": 5.79,
  "total_trades": 85,
  "strategies": [
    {
      "name": "ema9_trend_flip",
      "version": "1.0.0",
      "start_time": "2024-09-16T00:00:00+00:00",
      "end_time": "2024-12-16T00:00:00+00:00",
      "symbols": ["BTC/USDT"],
      "starting_capital": 100.0,
      "ending_capital": 104.56,
      "total_pnl": 4.56,
      "total_pnl_pct": 4.56,
      "max_drawdown": 2.1,
      "max_drawdown_pct": 2.1,
      "total_trades": 15,
      "winning_trades": 9,
      "losing_trades": 6,
      "win_rate": 60.0,
      "profit_factor": 1.85,
      "avg_win": 0.89,
      "avg_loss": -0.52,
      "pnl_by_symbol": {"BTC/USDT": 4.56},
      "trades_by_symbol": {"BTC/USDT": 15}
    }
  ]
}
```

## Performance Metrics

### Calculated Metrics

| Metric | Description |
|--------|-------------|
| `total_pnl` | Net profit/loss in USD |
| `total_pnl_pct` | P&L as percentage of starting capital |
| `win_rate` | Percentage of profitable trades |
| `profit_factor` | Gross profit / Gross loss |
| `max_drawdown` | Maximum peak-to-trough decline |
| `avg_win` | Average profit on winning trades |
| `avg_loss` | Average loss on losing trades |
| `largest_win` | Best single trade |
| `largest_loss` | Worst single trade |

### Equity Curve

Tracked every 60 candles (hourly) for drawdown analysis:

```python
equity_curve: List[Tuple[datetime, float]]
# [(2024-09-16 00:00, 100.0), (2024-09-16 01:00, 100.45), ...]
```

## Database Requirements

### Required Tables

```sql
-- 1-minute candles (primary)
SELECT * FROM candles WHERE interval_minutes = 1;

-- Pre-aggregated (loaded directly)
SELECT * FROM candles_5m;
SELECT * FROM candles_1h;
SELECT * FROM candles_1d;
```

### Data Range

Query available data range:

```python
health = await provider.health_check()
print(f"Data range: {health['oldest_data']} to {health['newest_data']}")
print(f"Total candles: {health['total_candles']:,}")
```

## Integration

### With Optimization System

```python
from backtest_runner import BacktestConfig, BacktestExecutor

# Used by optimizers for each parameter combination
config = BacktestConfig(
    start_time=start,
    end_time=end,
    symbols=[symbol],
    strategies=[strategy_name],
    starting_capital=100.0,
)

executor = BacktestExecutor(config, provider)
result = await executor.run_strategy(strategy, start, end)

# Access results
print(f"P&L: {result.total_pnl}")
print(f"Win Rate: {result.win_rate}")
```

### With Strategies

Strategies receive `DataSnapshot` with all timeframes:

```python
def generate_signal(data: DataSnapshot, config, state):
    # Use pre-aggregated hourly candles
    hourly = data.candles_1h.get('BTC/USDT', ())

    # Calculate indicators
    ema = calculate_ema([c.close for c in hourly], 9)
    ...
```

## Version History

- **v1.0.0** (2025-12-16): Initial release
  - Multi-strategy backtesting
  - Pre-aggregated candle support
  - USD size detection fix
  - Short position handling
  - JSON export
