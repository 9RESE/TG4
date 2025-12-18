#!/usr/bin/env python3
"""
Comprehensive Backtest Runner for ws_paper_tester Strategies.

Runs all strategies against historical data from TimescaleDB and generates
detailed performance reports.

Usage:
    python backtest_runner.py                           # Run all strategies, full history
    python backtest_runner.py --strategies ema9,wavetrend  # Specific strategies
    python backtest_runner.py --start 2023-01-01 --end 2024-01-01  # Date range
    python backtest_runner.py --symbols XRP/USDT BTC/USDT  # Specific symbols
    python backtest_runner.py --period 1y               # Last 1 year
"""

import asyncio
import argparse
import logging
import os
import sys
import json
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Add project root and ws_paper_tester to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from data.kraken_db import HistoricalDataProvider
from ws_tester.strategy_loader import discover_strategies, StrategyWrapper
from ws_tester.types import DataSnapshot, Candle, Signal, Fill, Position, Trade, OrderbookSnapshot
from ws_tester.portfolio import StrategyPortfolio, STARTING_CAPITAL

logger = logging.getLogger(__name__)


# Timeframe constants (in minutes)
TIMEFRAME_1M = 1
TIMEFRAME_5M = 5
TIMEFRAME_1H = 60
TIMEFRAME_1D = 1440

# Map timeframe minutes to human-readable names
TIMEFRAME_NAMES = {
    1: '1m',
    5: '5m',
    60: '1h',
    1440: '1d',
}


def get_required_timeframes(primary_tf_minutes: int) -> list:
    """
    Determine which timeframes to load based on primary timeframe.

    Returns list of timeframe minutes to load, including higher timeframes
    for context but excluding lower (unnecessary) timeframes.
    """
    all_timeframes = [TIMEFRAME_1M, TIMEFRAME_5M, TIMEFRAME_1H, TIMEFRAME_1D]

    # Only load primary timeframe and higher (for context)
    required = [tf for tf in all_timeframes if tf >= primary_tf_minutes]

    return required


@dataclass
class BacktestConfig:
    """Configuration for backtest run."""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    symbols: List[str] = field(default_factory=lambda: ['XRP/USDT', 'BTC/USDT', 'XRP/BTC'])
    strategies: Optional[List[str]] = None  # None = all strategies
    starting_capital: float = 100.0
    fee_rate: float = 0.001  # 0.1%
    slippage_rate: float = 0.0005  # 0.05%
    interval_minutes: int = 1  # Candle interval
    warmup_periods: int = 2000  # Candles for strategy warmup (needs to be large for 1H strategies)


@dataclass
class TradeRecord:
    """Record of a single trade."""
    timestamp: datetime
    strategy: str
    symbol: str
    side: str
    size: float
    price: float
    fee: float
    pnl: float
    reason: str


@dataclass
class StrategyResult:
    """Result of backtesting a single strategy."""
    strategy_name: str
    strategy_version: str
    start_time: datetime
    end_time: datetime
    symbols: List[str]

    # Capital
    starting_capital: float
    ending_capital: float
    peak_capital: float
    trough_capital: float

    # Performance
    total_pnl: float
    total_pnl_pct: float
    max_drawdown: float
    max_drawdown_pct: float

    # Trades
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float

    # Per-symbol breakdown
    pnl_by_symbol: Dict[str, float] = field(default_factory=dict)
    trades_by_symbol: Dict[str, int] = field(default_factory=dict)

    # Risk metrics
    sharpe_ratio: Optional[float] = None
    profit_factor: Optional[float] = None
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0

    # Time metrics
    avg_hold_time_minutes: float = 0.0
    total_candles_processed: int = 0
    signals_generated: int = 0

    # Trade log
    trades: List[TradeRecord] = field(default_factory=list)
    equity_curve: List[Tuple[datetime, float]] = field(default_factory=list)


class BacktestExecutor:
    """Executes backtests for strategies against historical data."""

    def __init__(
        self,
        config: BacktestConfig,
        provider: HistoricalDataProvider,
    ):
        self.config = config
        self.provider = provider

    def _create_snapshot(
        self,
        timestamp: datetime,
        candles_1m: Dict[str, List[Candle]],
        candles_5m: Dict[str, List[Candle]],
        candles_1h: Dict[str, List[Candle]] = None,
        candles_1d: Dict[str, List[Candle]] = None,
    ) -> DataSnapshot:
        """Create a DataSnapshot from candle data."""
        candles_1h = candles_1h or {}
        candles_1d = candles_1d or {}

        # Get current prices from available candles (prefer higher resolution)
        # Try 1m -> 5m -> 1h -> 1d in order of preference
        prices = {}
        for candle_dict in [candles_1m, candles_5m, candles_1h, candles_1d]:
            for symbol, candles in candle_dict.items():
                if symbol not in prices and candles:
                    prices[symbol] = float(candles[-1].close)

        # Helper to convert candle list to tuples
        def to_tuples(candle_dict):
            return {
                sym: tuple(
                    Candle(
                        timestamp=c.timestamp,
                        open=float(c.open),
                        high=float(c.high),
                        low=float(c.low),
                        close=float(c.close),
                        volume=float(c.volume),
                    )
                    for c in candles
                )
                for sym, candles in candle_dict.items()
            }

        candles_1m_tuples = to_tuples(candles_1m)
        candles_5m_tuples = to_tuples(candles_5m)
        candles_1h_tuples = to_tuples(candles_1h)
        candles_1d_tuples = to_tuples(candles_1d)

        # Create minimal orderbook from current prices
        orderbooks = {}
        for symbol, price in prices.items():
            spread = price * 0.0001  # 0.01% spread
            orderbooks[symbol] = OrderbookSnapshot(
                bids=((price - spread/2, 1000.0),),
                asks=((price + spread/2, 1000.0),),
            )

        # Create empty trades (historical backtest doesn't have tick data)
        trades_dict = {sym: tuple() for sym in prices}

        return DataSnapshot(
            timestamp=timestamp,
            prices=prices,
            candles_1m=candles_1m_tuples,
            candles_5m=candles_5m_tuples,
            candles_1h=candles_1h_tuples,
            candles_1d=candles_1d_tuples,
            orderbooks=orderbooks,
            trades=trades_dict,
            regime=None,
        )

    def _execute_signal(
        self,
        signal: Signal,
        portfolio: StrategyPortfolio,
        current_price: float,
    ) -> Optional[Fill]:
        """Execute a signal against the portfolio."""
        # Apply slippage
        if signal.action in ('buy', 'cover'):
            exec_price = current_price * (1 + self.config.slippage_rate)
        else:  # sell, short
            exec_price = current_price * (1 - self.config.slippage_rate)

        # Determine if signal.size is in USD or base asset units
        # Check signal metadata for size_unit hint, otherwise use heuristics
        # Most strategies use position_size_usd which is typically 10-100 USD
        size_unit = None
        if signal.metadata:
            size_unit = signal.metadata.get('size_unit')

        if size_unit == 'usd' or (size_unit is None and signal.size <= 1000):
            # Size is USD value - convert to base asset quantity
            base_size = signal.size / exec_price
            usd_value = signal.size
        elif size_unit == 'base':
            # Size is in base asset units
            base_size = signal.size
            usd_value = signal.size * exec_price
        else:
            # Heuristic: if size is small and price is small, assume USD
            # This handles low-priced assets like XRP ($2) with position_size_usd ($10-20)
            if signal.size <= 1000:
                base_size = signal.size / exec_price
                usd_value = signal.size
            else:
                base_size = signal.size
                usd_value = signal.size * exec_price

        # Calculate fee based on USD value
        fee = usd_value * self.config.fee_rate

        # Check if we have enough capital
        if signal.action == 'buy':
            cost = usd_value + fee
            if cost > portfolio.usdt:
                return None

        # Create fill (including signal metadata for strategy callbacks like grid level marking)
        fill = Fill(
            fill_id=f"bt_{datetime.now().timestamp()}",
            timestamp=datetime.now(timezone.utc),
            symbol=signal.symbol,
            side=signal.action,
            size=base_size,
            price=exec_price,
            fee=fee,
            signal_reason=signal.reason,
            pnl=0.0,
            metadata=signal.metadata.copy() if signal.metadata else {},
        )

        # Update portfolio (simplified for backtest)
        with portfolio._lock:
            if signal.action == 'buy':
                # Long entry
                portfolio.usdt -= (usd_value + fee)
                asset = signal.symbol.split('/')[0]  # XRP from XRP/USDT
                portfolio.assets[asset] = portfolio.assets.get(asset, 0) + base_size
                portfolio.total_trades += 1

            elif signal.action == 'sell':
                # Long exit
                asset = signal.symbol.split('/')[0]
                # Check if we have enough of the asset (use base_size for comparison)
                if portfolio.assets.get(asset, 0) >= base_size * 0.99:  # Allow 1% tolerance
                    # Sell whatever we have, up to base_size
                    actual_sell_size = min(portfolio.assets.get(asset, 0), base_size)
                    portfolio.assets[asset] -= actual_sell_size
                    portfolio.usdt += (actual_sell_size * exec_price - fee)

                    # Calculate P&L - use entry_price from metadata if available
                    # (Strategy exit signals set signal.price to exit price, not entry)
                    entry_price = signal.price
                    if signal.metadata and 'entry_price' in signal.metadata:
                        entry_price = signal.metadata['entry_price']

                    entry_value = actual_sell_size * entry_price
                    exit_value = actual_sell_size * exec_price
                    fill.pnl = exit_value - entry_value - fee * 2
                    fill.size = actual_sell_size
                    portfolio.total_pnl += fill.pnl
                    portfolio.total_trades += 1

                    if fill.pnl > 0:
                        portfolio.winning_trades += 1
                    else:
                        portfolio.losing_trades += 1
                else:
                    return None  # Can't sell what we don't have

            elif signal.action == 'short':
                # Short entry - track short position (conceptual for spot market backtest)
                # Reserve margin equivalent to position value
                cost = usd_value + fee
                if cost > portfolio.usdt:
                    return None

                portfolio.usdt -= cost  # Reserve margin
                asset_short = f"{signal.symbol.split('/')[0]}_short"
                portfolio.assets[asset_short] = portfolio.assets.get(asset_short, 0) + base_size
                # Store entry price for P&L calculation
                if not hasattr(portfolio, 'short_entries'):
                    portfolio.short_entries = {}
                portfolio.short_entries[signal.symbol] = exec_price
                portfolio.total_trades += 1

            elif signal.action == 'cover':
                # Short exit - close short position
                asset_short = f"{signal.symbol.split('/')[0]}_short"
                if portfolio.assets.get(asset_short, 0) >= base_size * 0.99:  # Allow 1% tolerance
                    actual_cover_size = min(portfolio.assets.get(asset_short, 0), base_size)
                    portfolio.assets[asset_short] -= actual_cover_size

                    # Get entry price from metadata or stored short entries
                    entry_price = signal.price
                    if signal.metadata and 'entry_price' in signal.metadata:
                        entry_price = signal.metadata['entry_price']
                    elif hasattr(portfolio, 'short_entries') and signal.symbol in portfolio.short_entries:
                        entry_price = portfolio.short_entries[signal.symbol]

                    # Short P&L: profit when price goes down
                    # Entry value (sold high) - Exit value (bought low) - fees
                    entry_value = actual_cover_size * entry_price
                    exit_value = actual_cover_size * exec_price
                    fill.pnl = entry_value - exit_value - fee * 2  # Opposite of long
                    fill.size = actual_cover_size

                    # Return margin + P&L
                    portfolio.usdt += (actual_cover_size * entry_price + fill.pnl - fee)
                    portfolio.total_pnl += fill.pnl
                    portfolio.total_trades += 1

                    if fill.pnl > 0:
                        portfolio.winning_trades += 1
                    else:
                        portfolio.losing_trades += 1
                else:
                    return None  # Can't cover position we don't have

        return fill

    async def run_strategy(
        self,
        strategy: StrategyWrapper,
        start_time: datetime,
        end_time: datetime,
    ) -> StrategyResult:
        """Run backtest for a single strategy."""
        logger.info(f"Starting backtest for {strategy.name} v{strategy.version}")
        logger.info(f"  Period: {start_time} to {end_time}")
        logger.info(f"  Symbols: {strategy.symbols}")

        # Initialize portfolio
        portfolio = StrategyPortfolio(strategy.name, self.config.starting_capital)

        # Initialize strategy
        strategy.on_start()

        # Track metrics
        trades: List[TradeRecord] = []
        equity_curve: List[Tuple[datetime, float]] = []
        peak_capital = self.config.starting_capital
        trough_capital = self.config.starting_capital
        max_drawdown = 0.0
        signals_generated = 0
        candles_processed = 0

        # Get symbols to trade (intersection of strategy symbols and config symbols)
        trade_symbols = [s for s in strategy.symbols if s in self.config.symbols]
        if not trade_symbols:
            logger.warning(f"  No matching symbols for {strategy.name}")
            trade_symbols = strategy.symbols[:1] if strategy.symbols else ['XRP/USDT']

        # Determine primary timeframe from strategy config (default to 1m for backward compatibility)
        primary_tf_minutes = strategy.config.get('candle_timeframe_minutes', TIMEFRAME_1M)
        required_timeframes = get_required_timeframes(primary_tf_minutes)
        primary_tf_name = TIMEFRAME_NAMES.get(primary_tf_minutes, f'{primary_tf_minutes}m')

        logger.info(f"  Primary timeframe: {primary_tf_name}")
        logger.info(f"  Loading timeframes: {[TIMEFRAME_NAMES.get(tf, f'{tf}m') for tf in required_timeframes]}")

        # Load only required timeframes (PERF: skip unnecessary data)
        all_candles = {tf: {} for tf in required_timeframes}

        for symbol in trade_symbols:
            loaded_info = []
            for tf in required_timeframes:
                candles = await self.provider.get_candles(symbol, tf, start_time, end_time)
                all_candles[tf][symbol] = candles
                loaded_info.append(f"{len(candles)} {TIMEFRAME_NAMES.get(tf, f'{tf}m')}")

            logger.info(f"    {symbol}: {', '.join(loaded_info)} candles")

        # Verify we have data for the primary timeframe
        if primary_tf_minutes not in all_candles or not any(all_candles[primary_tf_minutes].values()):
            logger.warning(f"  No candle data found for {strategy.name}")
            return self._create_empty_result(strategy, start_time, end_time)

        # Use the primary symbol and timeframe for iteration
        primary_symbol = trade_symbols[0]
        primary_candles = all_candles[primary_tf_minutes][primary_symbol]

        # Adjust warmup periods based on timeframe
        # Higher timeframes need fewer candles but still need enough for EMA calculation
        # 1m: use full warmup_periods (2000)
        # 5m: ~400 candles
        # 1h: ~33 candles (but min 20 for EMA-9 + buffer)
        # 1d: ~1-2 candles from formula, but need min 15 for indicators
        warmup_by_timeframe = {
            TIMEFRAME_1M: self.config.warmup_periods,
            TIMEFRAME_5M: max(100, self.config.warmup_periods // 5),
            TIMEFRAME_1H: max(20, self.config.warmup_periods // 60),
            TIMEFRAME_1D: 15,  # Minimum for EMA-9 + ATR-14
        }
        adjusted_warmup = warmup_by_timeframe.get(
            primary_tf_minutes,
            max(15, self.config.warmup_periods // max(1, primary_tf_minutes))
        )

        if len(primary_candles) < adjusted_warmup:
            logger.warning(f"  Insufficient data for warmup ({len(primary_candles)} < {adjusted_warmup})")

        # Process candles at primary timeframe (PERF: iterate at strategy's native timeframe)
        logger.info(f"  Processing {len(primary_candles)} {primary_tf_name} candles...")

        # Initialize windows for each loaded timeframe
        windows = {tf: {sym: [] for sym in trade_symbols} for tf in required_timeframes}

        # Track candle indices for higher timeframes (relative to primary)
        higher_tf_idx = {}
        for tf in required_timeframes:
            if tf > primary_tf_minutes:
                higher_tf_idx[tf] = {sym: 0 for sym in trade_symbols}

        # Window size limits by timeframe
        window_limits = {
            TIMEFRAME_1M: self.config.warmup_periods,
            TIMEFRAME_5M: self.config.warmup_periods,
            TIMEFRAME_1H: 500,   # ~20 days
            TIMEFRAME_1D: 365,   # ~1 year
        }

        for i, candle in enumerate(primary_candles):
            candles_processed += 1
            timestamp = candle.timestamp

            # Update windows for all symbols
            for symbol in trade_symbols:
                # Add primary timeframe candle
                symbol_candles = all_candles[primary_tf_minutes].get(symbol, [])
                if i < len(symbol_candles):
                    windows[primary_tf_minutes][symbol].append(symbol_candles[i])
                    limit = window_limits.get(primary_tf_minutes, self.config.warmup_periods)
                    if len(windows[primary_tf_minutes][symbol]) > limit:
                        windows[primary_tf_minutes][symbol] = windows[primary_tf_minutes][symbol][-limit:]

                # Add higher timeframe candles up to current time
                for tf in higher_tf_idx:
                    tf_candles = all_candles[tf].get(symbol, [])
                    while (higher_tf_idx[tf][symbol] < len(tf_candles) and
                           tf_candles[higher_tf_idx[tf][symbol]].timestamp <= timestamp):
                        windows[tf][symbol].append(tf_candles[higher_tf_idx[tf][symbol]])
                        higher_tf_idx[tf][symbol] += 1
                        limit = window_limits.get(tf, 500)
                        if len(windows[tf][symbol]) > limit:
                            windows[tf][symbol] = windows[tf][symbol][-limit:]

            # Build window dicts for snapshot (backward compatible with existing code)
            window_1m = windows.get(TIMEFRAME_1M, {sym: [] for sym in trade_symbols})
            window_5m = windows.get(TIMEFRAME_5M, {sym: [] for sym in trade_symbols})
            window_1h = windows.get(TIMEFRAME_1H, {sym: [] for sym in trade_symbols})
            window_1d = windows.get(TIMEFRAME_1D, {sym: [] for sym in trade_symbols})

            # Skip warmup period (adjusted for timeframe)
            if i < adjusted_warmup:
                continue

            # Create snapshot with all timeframes
            try:
                snapshot = self._create_snapshot(timestamp, window_1m, window_5m, window_1h, window_1d)
            except Exception as e:
                logger.debug(f"  Failed to create snapshot at {timestamp}: {e}")
                continue

            # Generate signal
            try:
                signal = strategy.generate_signal(snapshot)
            except Exception as e:
                logger.debug(f"  Strategy error at {timestamp}: {e}")
                continue

            if signal:
                signals_generated += 1
                current_price = snapshot.prices.get(signal.symbol, 0)

                if current_price > 0:
                    fill = self._execute_signal(signal, portfolio, current_price)

                    if fill:
                        strategy.on_fill(fill.__dict__)
                        trades.append(TradeRecord(
                            timestamp=timestamp,
                            strategy=strategy.name,
                            symbol=fill.symbol,
                            side=fill.side,
                            size=fill.size,
                            price=fill.price,
                            fee=fill.fee,
                            pnl=fill.pnl,
                            reason=fill.signal_reason,
                        ))

            # Update equity curve (adjusted for timeframe: ~hourly equivalent)
            equity_update_interval = max(1, 60 // primary_tf_minutes)
            if i % equity_update_interval == 0:
                current_equity = portfolio.usdt
                for asset, amount in portfolio.assets.items():
                    # Find the price for this asset
                    for sym in snapshot.prices:
                        if sym.startswith(asset + '/'):
                            current_equity += amount * snapshot.prices[sym]
                            break

                equity_curve.append((timestamp, current_equity))

                # Track peak/trough
                if current_equity > peak_capital:
                    peak_capital = current_equity
                if current_equity < trough_capital:
                    trough_capital = current_equity

                # Track drawdown
                if peak_capital > 0:
                    current_dd = (peak_capital - current_equity) / peak_capital
                    if current_dd > max_drawdown:
                        max_drawdown = current_dd

            # Progress logging (adjusted for timeframe: ~10% intervals)
            progress_interval = max(100, len(primary_candles) // 10)
            if i % progress_interval == 0 and i > 0:
                logger.info(f"    Processed {i}/{len(primary_candles)} {primary_tf_name} candles, {len(trades)} trades")

        # Stop strategy
        strategy.on_stop()

        # Calculate final metrics
        final_equity = portfolio.usdt
        for asset, amount in portfolio.assets.items():
            # Use last known price
            for sym in snapshot.prices if 'snapshot' in dir() else {}:
                if sym.startswith(asset + '/'):
                    final_equity += amount * snapshot.prices[sym]
                    break

        total_pnl = final_equity - self.config.starting_capital

        # Calculate per-symbol metrics
        pnl_by_symbol = defaultdict(float)
        trades_by_symbol = defaultdict(int)
        wins = []
        losses = []

        for trade in trades:
            pnl_by_symbol[trade.symbol] += trade.pnl
            trades_by_symbol[trade.symbol] += 1
            if trade.pnl > 0:
                wins.append(trade.pnl)
            elif trade.pnl < 0:
                losses.append(trade.pnl)

        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        largest_win = max(wins) if wins else 0
        largest_loss = min(losses) if losses else 0

        # Profit factor
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0

        # Win rate
        total_trades = len(trades)
        winning_trades = len(wins)
        losing_trades = len(losses)
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0

        logger.info(f"  Completed: {total_trades} trades, ${total_pnl:.2f} P&L ({win_rate:.1f}% win rate)")

        return StrategyResult(
            strategy_name=strategy.name,
            strategy_version=strategy.version,
            start_time=start_time,
            end_time=end_time,
            symbols=trade_symbols,
            starting_capital=self.config.starting_capital,
            ending_capital=final_equity,
            peak_capital=peak_capital,
            trough_capital=trough_capital,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl / self.config.starting_capital * 100,
            max_drawdown=max_drawdown * self.config.starting_capital,
            max_drawdown_pct=max_drawdown * 100,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            pnl_by_symbol=dict(pnl_by_symbol),
            trades_by_symbol=dict(trades_by_symbol),
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            total_candles_processed=candles_processed,
            signals_generated=signals_generated,
            trades=trades,
            equity_curve=equity_curve,
        )

    def _create_empty_result(
        self,
        strategy: StrategyWrapper,
        start_time: datetime,
        end_time: datetime,
    ) -> StrategyResult:
        """Create an empty result for strategies with no data."""
        return StrategyResult(
            strategy_name=strategy.name,
            strategy_version=strategy.version,
            start_time=start_time,
            end_time=end_time,
            symbols=strategy.symbols,
            starting_capital=self.config.starting_capital,
            ending_capital=self.config.starting_capital,
            peak_capital=self.config.starting_capital,
            trough_capital=self.config.starting_capital,
            total_pnl=0,
            total_pnl_pct=0,
            max_drawdown=0,
            max_drawdown_pct=0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0,
        )


def print_results(results: List[StrategyResult]):
    """Print backtest results to console."""
    print("\n" + "=" * 80)
    print("BACKTEST RESULTS")
    print("=" * 80)

    if not results:
        print("No results to display.")
        return

    # Summary table
    print(f"\n{'Strategy':<25} {'P&L':>10} {'P&L%':>8} {'Trades':>8} {'Win%':>8} {'MaxDD%':>8} {'PF':>8}")
    print("-" * 80)

    # Sort by P&L descending
    sorted_results = sorted(results, key=lambda r: r.total_pnl, reverse=True)

    total_pnl = 0
    total_trades = 0

    for r in sorted_results:
        pf_str = f"{r.profit_factor:.2f}" if r.profit_factor and r.profit_factor != float('inf') else "∞" if r.profit_factor == float('inf') else "N/A"
        print(f"{r.strategy_name:<25} ${r.total_pnl:>9.2f} {r.total_pnl_pct:>7.1f}% {r.total_trades:>8} {r.win_rate:>7.1f}% {r.max_drawdown_pct:>7.1f}% {pf_str:>8}")
        total_pnl += r.total_pnl
        total_trades += r.total_trades

    print("-" * 80)
    print(f"{'TOTAL':<25} ${total_pnl:>9.2f} {'-':>8} {total_trades:>8}")

    # Per-symbol breakdown
    print(f"\n\nPER-SYMBOL BREAKDOWN")
    print("-" * 80)

    all_symbols = set()
    for r in results:
        all_symbols.update(r.pnl_by_symbol.keys())

    for symbol in sorted(all_symbols):
        print(f"\n{symbol}:")
        for r in sorted_results:
            if symbol in r.pnl_by_symbol:
                pnl = r.pnl_by_symbol[symbol]
                trades = r.trades_by_symbol.get(symbol, 0)
                print(f"  {r.strategy_name:<23} ${pnl:>9.2f} ({trades} trades)")

    # Best/worst performers
    print(f"\n\nPERFORMANCE HIGHLIGHTS")
    print("-" * 80)

    if sorted_results:
        best = sorted_results[0]
        worst = sorted_results[-1]

        print(f"Best Performer:  {best.strategy_name} (+${best.total_pnl:.2f}, {best.win_rate:.1f}% win rate)")
        print(f"Worst Performer: {worst.strategy_name} (${worst.total_pnl:.2f}, {worst.win_rate:.1f}% win rate)")

        # Most active
        most_active = max(sorted_results, key=lambda r: r.total_trades)
        print(f"Most Active:     {most_active.strategy_name} ({most_active.total_trades} trades)")

        # Best win rate (min 10 trades)
        qualified = [r for r in sorted_results if r.total_trades >= 10]
        if qualified:
            best_wr = max(qualified, key=lambda r: r.win_rate)
            print(f"Best Win Rate:   {best_wr.strategy_name} ({best_wr.win_rate:.1f}% on {best_wr.total_trades} trades)")

    print("\n" + "=" * 80)


def save_results(results: List[StrategyResult], output_dir: str = "backtest_results"):
    """Save backtest results to JSON files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Summary file
    summary = {
        "timestamp": timestamp,
        "strategies": [],
        "total_pnl": sum(r.total_pnl for r in results),
        "total_trades": sum(r.total_trades for r in results),
    }

    for r in results:
        strategy_data = {
            "name": r.strategy_name,
            "version": r.strategy_version,
            "start_time": r.start_time.isoformat() if r.start_time else None,
            "end_time": r.end_time.isoformat() if r.end_time else None,
            "symbols": r.symbols,
            "starting_capital": r.starting_capital,
            "ending_capital": r.ending_capital,
            "total_pnl": r.total_pnl,
            "total_pnl_pct": r.total_pnl_pct,
            "max_drawdown": r.max_drawdown,
            "max_drawdown_pct": r.max_drawdown_pct,
            "total_trades": r.total_trades,
            "winning_trades": r.winning_trades,
            "losing_trades": r.losing_trades,
            "win_rate": r.win_rate,
            "profit_factor": r.profit_factor if r.profit_factor != float('inf') else None,
            "avg_win": r.avg_win,
            "avg_loss": r.avg_loss,
            "largest_win": r.largest_win,
            "largest_loss": r.largest_loss,
            "pnl_by_symbol": r.pnl_by_symbol,
            "trades_by_symbol": r.trades_by_symbol,
            "signals_generated": r.signals_generated,
            "candles_processed": r.total_candles_processed,
        }
        summary["strategies"].append(strategy_data)

    summary_file = output_path / f"backtest_summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nResults saved to: {summary_file}")

    return summary_file


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Comprehensive Backtest Runner")
    parser.add_argument('--db-url', type=str, default=os.getenv('DATABASE_URL'),
                       help='Database URL (or set DATABASE_URL env var)')
    parser.add_argument('--strategies', type=str, default=None,
                       help='Comma-separated list of strategy names to test')
    parser.add_argument('--symbols', nargs='+', default=['XRP/USDT', 'BTC/USDT', 'XRP/BTC'],
                       help='Symbols to trade')
    parser.add_argument('--start', type=str, default=None,
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None,
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--period', type=str, default=None,
                       help='Period to test (e.g., 1y, 6m, 3m, 1m, 1w)')
    parser.add_argument('--capital', type=float, default=100.0,
                       help='Starting capital per strategy')
    parser.add_argument('--output', type=str, default='backtest_results',
                       help='Output directory for results')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Check database URL
    if not args.db_url:
        print("ERROR: --db-url or DATABASE_URL environment variable required")
        print("Example: DATABASE_URL=postgresql://trading:password@localhost:5433/kraken_data")
        sys.exit(1)

    # Initialize provider
    provider = HistoricalDataProvider(args.db_url)
    await provider.connect()

    try:
        # Get data range
        health = await provider.health_check()
        print("\n" + "=" * 80)
        print("HISTORICAL DATA STATUS")
        print("=" * 80)
        print(f"Symbols available: {health.get('symbols', [])}")
        print(f"Total candles: {health.get('total_candles', 0):,}")
        print(f"Data range: {health.get('oldest_data')} to {health.get('newest_data')}")

        # Determine time range
        end_time = datetime.now(timezone.utc)
        if args.end:
            end_time = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)

        start_time = None
        if args.start:
            start_time = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
        elif args.period:
            periods = {
                '1w': timedelta(weeks=1),
                '2w': timedelta(weeks=2),
                '1m': timedelta(days=30),
                '2m': timedelta(days=60),
                '3m': timedelta(days=90),
                '6m': timedelta(days=180),
                '9m': timedelta(days=270),
                '12m': timedelta(days=365),  # Alias for 1y
                '1y': timedelta(days=365),
                '2y': timedelta(days=730),
                '3y': timedelta(days=1095),
                '4y': timedelta(days=1460),
                '5y': timedelta(days=1825),
                'all': None,
            }
            period_key = args.period.lower()
            if period_key in periods:
                delta = periods[period_key]
                if delta:
                    start_time = end_time - delta
            else:
                print(f"Unknown period: {args.period}")
                print(f"Valid periods: {list(periods.keys())}")
                sys.exit(1)

        if start_time is None:
            # Default to 1 year
            start_time = end_time - timedelta(days=365)

        print(f"\nBacktest period: {start_time.date()} to {end_time.date()}")

        # Create config
        config = BacktestConfig(
            start_time=start_time,
            end_time=end_time,
            symbols=args.symbols,
            strategies=args.strategies.split(',') if args.strategies else None,
            starting_capital=args.capital,
        )

        # Load strategies
        strategies_path = Path(__file__).parent / "strategies"
        all_strategies = discover_strategies(str(strategies_path))

        if config.strategies:
            strategies = {k: v for k, v in all_strategies.items() if k in config.strategies}
        else:
            strategies = all_strategies

        print(f"\nStrategies to test: {list(strategies.keys())}")
        print(f"Symbols: {config.symbols}")
        print(f"Starting capital: ${config.starting_capital}")

        # Run backtests
        executor = BacktestExecutor(config, provider)
        results = []

        for name, strategy in strategies.items():
            try:
                result = await executor.run_strategy(strategy, start_time, end_time)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to backtest {name}: {e}")
                import traceback
                traceback.print_exc()

        # Print and save results
        print_results(results)
        save_results(results, args.output)

    finally:
        await provider.close()


if __name__ == '__main__':
    asyncio.run(main())
