"""
Grid Ensemble Orchestrator
Combines multiple grid strategies for BTC trading with paper trading support.

Strategies:
1. ArithmeticGrid - Fixed spacing, consistent fills
2. GeometricGrid - Percentage spacing, extreme bias
3. RSIMeanReversionGrid - RSI-filtered entries
4. BBSqueezeGrid - Squeeze detection + breakout

Based on BTC 5-minute chart analysis (Dec 2024)
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import json
import time
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict

from strategies.grid_base import (
    ArithmeticGrid,
    GeometricGrid,
    RSIMeanReversionGrid,
    BBSqueezeGrid,
    GridStats,
    # Margin strategies
    TrendFollowingMargin,
    DualGridHedge,
    TimeWeightedGrid,
    LiquidationHuntScalper
)
from utils.diagnostic_logger import get_diagnostic_logger, close_diagnostic_logger


@dataclass
class PaperTrade:
    """Record of a paper trade."""
    timestamp: datetime
    strategy: str
    action: str
    symbol: str
    price: float
    size: float
    leverage: int
    reason: str
    pnl: float = 0.0
    closed: bool = False
    close_price: float = 0.0
    close_time: Optional[datetime] = None


@dataclass
class GridEnsembleStats:
    """Ensemble-level statistics."""
    start_time: datetime = field(default_factory=datetime.now)
    total_capital: float = 10000.0
    current_value: float = 10000.0
    total_pnl: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_fees: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    max_drawdown: float = 0.0
    peak_value: float = 10000.0
    strategy_pnl: Dict[str, float] = field(default_factory=dict)


class GridEnsembleOrchestrator:
    """
    Orchestrates multiple grid strategies for BTC trading.

    Features:
    - Parallel grid strategies with different configurations
    - Paper trading with full order simulation
    - Real-time price fetching
    - Performance tracking and logging
    - Risk management across all strategies
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the grid ensemble.

        Args:
            config: Configuration dict with strategy parameters
        """
        self.config = config or self._default_config()
        self.symbol = self.config.get('symbol', 'BTC/USDT')
        self.total_capital = self.config.get('total_capital', 10000.0)

        # Allocate capital across strategies (grid + margin)
        self.strategy_allocation = self.config.get('strategy_allocation', {
            # Grid strategies (50% total)
            'arithmetic': 0.15,           # 15% to arithmetic grid
            'geometric': 0.10,            # 10% to geometric grid
            'rsi_reversion': 0.15,        # 15% to RSI mean reversion
            'bb_squeeze': 0.10,           # 10% to BB squeeze
            # Margin strategies (50% total)
            'trend_margin': 0.15,         # 15% to trend-following margin (5x)
            'dual_grid_hedge': 0.15,      # 15% to dual grid with hedge
            'time_weighted': 0.10,        # 10% to time-weighted grid
            'liq_hunter': 0.10            # 10% to liquidation hunter (3x)
        })

        # Initialize strategies
        self.strategies: Dict[str, Any] = {}
        self._init_strategies()

        # Paper trading state
        self.paper_trades: List[PaperTrade] = []
        self.open_positions: Dict[str, List[PaperTrade]] = {name: [] for name in self.strategies}
        self.stats = GridEnsembleStats(
            total_capital=self.total_capital,
            current_value=self.total_capital,
            peak_value=self.total_capital
        )

        # Market data
        self.current_price = 0.0
        self.price_history: List[Dict] = []
        self.data: Optional[pd.DataFrame] = None

        # Logging
        self.logger = get_diagnostic_logger()
        self.last_update = datetime.now()

        print(f"GridEnsembleOrchestrator initialized with {len(self.strategies)} strategies")
        print(f"Total capital: ${self.total_capital:,.2f}")
        for name, alloc in self.strategy_allocation.items():
            print(f"  {name}: ${self.total_capital * alloc:,.2f} ({alloc*100:.0f}%)")

    def _default_config(self) -> Dict[str, Any]:
        """Default configuration based on BTC chart analysis."""
        return {
            'symbol': 'BTC/USDT',
            'total_capital': 10000.0,

            # Grid range from chart analysis
            'upper_price': 94500,
            'lower_price': 90000,

            # Risk management
            'max_drawdown': 0.10,  # 10% max drawdown
            'stop_loss': 88500,    # Below major support
            'take_profit': 96000,  # Above resistance

            # Fee structure
            'fee_rate': 0.001,     # 0.1%

            # Strategy-specific configs
            'arithmetic': {
                'num_grids': 20,
                'leverage': 1
            },
            'geometric': {
                'num_grids': 15,
                'grid_ratio': 1.008,
                'leverage': 1
            },
            'rsi_reversion': {
                'num_grids': 15,
                'rsi_period': 14,
                'rsi_oversold': 25,
                'rsi_overbought': 75,
                'leverage': 2
            },
            'bb_squeeze': {
                'num_grids': 12,
                'bb_period': 20,
                'bb_std': 2.0,
                'squeeze_threshold': 0.02,
                'leverage': 2
            },
            # Margin strategy configs
            'trend_margin': {
                'leverage': 5,
                'trendline_start_price': 94000,
                'trendline_slope': 50,  # $/hour
                'rsi_threshold': 35,
                'size_pct': 0.10
            },
            'dual_grid_hedge': {
                'num_grids': 18,
                'hedge_leverage': 2
            },
            'time_weighted': {
                'num_grids': 15
            },
            'liq_hunter': {
                'leverage': 3,
                'liq_zone_pct': 0.02,
                'size_pct': 0.06
            }
        }

    def _init_strategies(self):
        """Initialize all grid strategies."""
        base_config = {
            'symbol': self.symbol,
            'upper_price': self.config['upper_price'],
            'lower_price': self.config['lower_price'],
            'stop_loss': self.config.get('stop_loss'),
            'take_profit': self.config.get('take_profit'),
            'max_drawdown': self.config['max_drawdown'],
            'fee_rate': self.config['fee_rate']
        }

        # Arithmetic Grid
        arith_config = {
            **base_config,
            'total_capital': self.total_capital * self.strategy_allocation['arithmetic'],
            **self.config.get('arithmetic', {})
        }
        self.strategies['arithmetic'] = ArithmeticGrid(arith_config)

        # Geometric Grid
        geo_config = {
            **base_config,
            'total_capital': self.total_capital * self.strategy_allocation['geometric'],
            **self.config.get('geometric', {})
        }
        self.strategies['geometric'] = GeometricGrid(geo_config)

        # RSI Mean Reversion Grid
        rsi_config = {
            **base_config,
            'total_capital': self.total_capital * self.strategy_allocation['rsi_reversion'],
            **self.config.get('rsi_reversion', {})
        }
        self.strategies['rsi_reversion'] = RSIMeanReversionGrid(rsi_config)

        # BB Squeeze Grid
        bb_config = {
            **base_config,
            'total_capital': self.total_capital * self.strategy_allocation['bb_squeeze'],
            **self.config.get('bb_squeeze', {})
        }
        self.strategies['bb_squeeze'] = BBSqueezeGrid(bb_config)

        # === MARGIN STRATEGIES ===

        # Trend-Following Margin (5x leverage)
        trend_config = {
            **base_config,
            'total_capital': self.total_capital * self.strategy_allocation.get('trend_margin', 0.15),
            **self.config.get('trend_margin', {})
        }
        self.strategies['trend_margin'] = TrendFollowingMargin(trend_config)

        # Dual Grid with Hedge
        dual_config = {
            **base_config,
            'total_capital': self.total_capital * self.strategy_allocation.get('dual_grid_hedge', 0.15),
            **self.config.get('dual_grid_hedge', {})
        }
        self.strategies['dual_grid_hedge'] = DualGridHedge(dual_config)

        # Time-Weighted Grid
        time_config = {
            **base_config,
            'total_capital': self.total_capital * self.strategy_allocation.get('time_weighted', 0.10),
            **self.config.get('time_weighted', {})
        }
        self.strategies['time_weighted'] = TimeWeightedGrid(time_config)

        # Liquidation Hunter Scalper (3x leverage)
        liq_config = {
            **base_config,
            'total_capital': self.total_capital * self.strategy_allocation.get('liq_hunter', 0.10),
            **self.config.get('liq_hunter', {})
        }
        self.strategies['liq_hunter'] = LiquidationHuntScalper(liq_config)

    def update(self, current_price: float, data: pd.DataFrame = None) -> List[Dict[str, Any]]:
        """
        Update all strategies with current price.

        Args:
            current_price: Current BTC price
            data: Optional OHLCV DataFrame for technical analysis

        Returns:
            List of all signals generated
        """
        self.current_price = current_price
        self.data = data
        self.last_update = datetime.now()

        # Store price history
        self.price_history.append({
            'timestamp': self.last_update,
            'price': current_price
        })

        # Keep last 1000 prices
        if len(self.price_history) > 1000:
            self.price_history = self.price_history[-1000:]

        all_signals = []

        # Update each strategy
        for name, strategy in self.strategies.items():
            try:
                signals = strategy.update(current_price, data)
                for signal in signals:
                    signal['ensemble_strategy'] = name
                    all_signals.append(signal)
            except Exception as e:
                print(f"Error updating {name}: {e}")

        # Update unrealized PnL
        self._update_unrealized_pnl()

        # Check ensemble-level risk
        self._check_risk()

        # Log market state
        self._log_state()

        return all_signals

    def execute_paper(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute signals in paper trading mode.

        Args:
            signals: List of signals from update()

        Returns:
            List of execution results
        """
        results = []

        for signal in signals:
            try:
                result = self._execute_paper_signal(signal)
                results.append(result)
            except Exception as e:
                results.append({
                    'signal': signal,
                    'executed': False,
                    'error': str(e)
                })

        return results

    def _execute_paper_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single paper trade with Phase 32 order type support."""
        strategy_name = signal.get('ensemble_strategy', 'unknown')
        action = signal.get('action', 'hold')
        order_type = signal.get('order_type', 'market')
        limit_price = signal.get('limit_price')
        signal_price = signal.get('price', self.current_price)
        size = signal.get('size', 0)
        leverage = signal.get('leverage', 1)

        # Phase 32: Simulate limit order behavior
        # For limit orders, only fill if current price is favorable
        if order_type == 'limit' and limit_price is not None:
            if action == 'buy' and self.current_price > limit_price * 1.005:
                # Market moved up too fast, limit order wouldn't fill
                return {
                    'signal': signal,
                    'executed': False,
                    'reason': f'Limit buy not filled - market ${self.current_price:,.2f} > limit ${limit_price:,.2f}'
                }
            elif action == 'sell' and self.current_price < limit_price * 0.995:
                # Market moved down too fast, limit order wouldn't fill
                return {
                    'signal': signal,
                    'executed': False,
                    'reason': f'Limit sell not filled - market ${self.current_price:,.2f} < limit ${limit_price:,.2f}'
                }
            # Fill at limit price (better execution than market)
            price = limit_price
        else:
            # Market order - fill at current price with slippage
            slippage = 0.001 if action == 'buy' else -0.001
            price = self.current_price * (1 + slippage)

        result = {
            'signal': signal,
            'executed': False,
            'timestamp': datetime.now()
        }

        # Max positions per strategy to prevent accumulation blowup
        MAX_POSITIONS_PER_STRATEGY = 5

        if action == 'buy':
            # Check if we have capital
            strategy = self.strategies.get(strategy_name)
            if not strategy:
                result['error'] = f'Unknown strategy: {strategy_name}'
                return result

            # Check position limit
            current_positions = len(self.open_positions.get(strategy_name, []))
            if current_positions >= MAX_POSITIONS_PER_STRATEGY:
                result['error'] = f'Max positions reached for {strategy_name}'
                return result

            # Create paper trade
            trade = PaperTrade(
                timestamp=datetime.now(),
                strategy=strategy_name,
                action='buy',
                symbol=self.symbol,
                price=price,
                size=size,
                leverage=leverage,
                reason=signal.get('reason', 'Grid buy')
            )

            self.paper_trades.append(trade)
            self.open_positions[strategy_name].append(trade)
            self.stats.total_trades += 1

            # Fill the order on the strategy (for both grid and margin strategies)
            grid_level = signal.get('grid_level')
            if grid_level is not None:
                strategy.fill_order(grid_level, price)
            elif hasattr(strategy, 'fill_order'):
                # Margin strategies need to track their positions
                strategy.fill_order(signal, price)

            result['executed'] = True
            result['trade'] = asdict(trade)
            result['order_type'] = order_type

            print(f"[PAPER BUY] {strategy_name}: {size:.6f} BTC @ ${price:,.2f} "
                  f"({order_type}, lev: {leverage}x) - {signal.get('reason', '')}")

        elif action == 'sell':
            # Find matching position to close
            positions = self.open_positions.get(strategy_name, [])

            if not positions:
                result['error'] = 'No position to close'
                return result

            # Close oldest position (FIFO)
            trade = positions.pop(0)
            trade.closed = True
            trade.close_price = price
            trade.close_time = datetime.now()

            # Calculate PnL
            pnl = (price - trade.price) * trade.size * trade.leverage
            fee = (trade.price + price) * trade.size * self.config['fee_rate']
            trade.pnl = pnl - fee

            self.stats.realized_pnl += trade.pnl
            self.stats.total_fees += fee

            if trade.pnl > 0:
                self.stats.winning_trades += 1
            else:
                self.stats.losing_trades += 1

            # Update strategy stats
            if strategy_name not in self.stats.strategy_pnl:
                self.stats.strategy_pnl[strategy_name] = 0.0
            self.stats.strategy_pnl[strategy_name] += trade.pnl

            # Fill the grid level
            grid_level = signal.get('grid_level')
            if grid_level is not None:
                strategy = self.strategies.get(strategy_name)
                if strategy:
                    strategy.fill_order(grid_level, price)

            result['executed'] = True
            result['trade'] = asdict(trade)
            result['pnl'] = trade.pnl

            pnl_str = f"+${trade.pnl:.2f}" if trade.pnl > 0 else f"-${abs(trade.pnl):.2f}"
            print(f"[PAPER SELL] {strategy_name}: {trade.size:.6f} BTC @ ${price:,.2f} "
                  f"({order_type}, entry: ${trade.price:,.2f}) -> {pnl_str}")

        elif action == 'short':
            # Short position (for breakout strategies)
            strategy = self.strategies.get(strategy_name)
            if not strategy:
                result['error'] = f'Unknown strategy: {strategy_name}'
                return result

            # Check position limit
            current_positions = len(self.open_positions.get(strategy_name, []))
            if current_positions >= MAX_POSITIONS_PER_STRATEGY:
                result['error'] = f'Max positions reached for {strategy_name}'
                return result

            trade = PaperTrade(
                timestamp=datetime.now(),
                strategy=strategy_name,
                action='short',
                symbol=self.symbol,
                price=price,
                size=size,
                leverage=leverage,
                reason=signal.get('reason', 'Grid short')
            )

            self.paper_trades.append(trade)
            self.open_positions[strategy_name].append(trade)
            self.stats.total_trades += 1

            # Fill the order on the strategy (critical for margin strategies!)
            if hasattr(strategy, 'fill_order'):
                strategy.fill_order(signal, price)

            result['executed'] = True
            result['trade'] = asdict(trade)

            print(f"[PAPER SHORT] {strategy_name}: {size:.6f} BTC @ ${price:,.2f} "
                  f"({order_type}, lev: {leverage}x) - {signal.get('reason', '')}")

        elif action in ['close', 'close_hedge']:
            # Close a specific position (for margin strategies)
            strategy = self.strategies.get(strategy_name)
            positions = self.open_positions.get(strategy_name, [])

            if not positions:
                result['error'] = 'No position to close'
                return result

            # Close most recent position (or by matching criteria)
            trade = positions.pop()  # LIFO for margin positions
            trade.closed = True
            trade.close_price = price
            trade.close_time = datetime.now()

            # Calculate PnL based on position type
            if trade.action == 'buy':
                pnl = (price - trade.price) * trade.size * trade.leverage
            else:  # short
                pnl = (trade.price - price) * trade.size * trade.leverage

            fee = (trade.price + price) * trade.size * self.config['fee_rate']
            trade.pnl = pnl - fee

            self.stats.realized_pnl += trade.pnl
            self.stats.total_fees += fee

            if trade.pnl > 0:
                self.stats.winning_trades += 1
            else:
                self.stats.losing_trades += 1

            # Update strategy stats
            if strategy_name not in self.stats.strategy_pnl:
                self.stats.strategy_pnl[strategy_name] = 0.0
            self.stats.strategy_pnl[strategy_name] += trade.pnl

            # Notify the strategy that position was closed
            if strategy and hasattr(strategy, 'fill_order'):
                strategy.fill_order(signal, price)

            result['executed'] = True
            result['trade'] = asdict(trade)
            result['pnl'] = trade.pnl

            pnl_str = f"+${trade.pnl:.2f}" if trade.pnl > 0 else f"-${abs(trade.pnl):.2f}"
            print(f"[PAPER CLOSE] {strategy_name}: {trade.size:.6f} BTC @ ${price:,.2f} "
                  f"(entry: ${trade.price:,.2f}) -> {pnl_str}")

        elif action == 'close_all':
            # Emergency close all positions
            closed_count = 0
            total_pnl = 0.0

            for strat_name, positions in self.open_positions.items():
                for trade in positions:
                    trade.closed = True
                    trade.close_price = price
                    trade.close_time = datetime.now()

                    if trade.action == 'buy':
                        pnl = (price - trade.price) * trade.size * trade.leverage
                    else:  # short
                        pnl = (trade.price - price) * trade.size * trade.leverage

                    fee = (trade.price + price) * trade.size * self.config['fee_rate']
                    trade.pnl = pnl - fee
                    total_pnl += trade.pnl
                    closed_count += 1

                    self.stats.realized_pnl += trade.pnl
                    self.stats.total_fees += fee

            # Clear all positions
            for strat_name in self.open_positions:
                self.open_positions[strat_name] = []

            result['executed'] = True
            result['closed_count'] = closed_count
            result['total_pnl'] = total_pnl

            print(f"[PAPER CLOSE ALL] Closed {closed_count} positions, PnL: ${total_pnl:,.2f}")

        return result

    def _update_unrealized_pnl(self):
        """Update unrealized PnL for all open positions."""
        unrealized = 0.0

        for strategy_name, positions in self.open_positions.items():
            for trade in positions:
                if trade.action == 'buy':
                    unrealized += (self.current_price - trade.price) * trade.size * trade.leverage
                else:  # short
                    unrealized += (trade.price - self.current_price) * trade.size * trade.leverage

        self.stats.unrealized_pnl = unrealized
        self.stats.total_pnl = self.stats.realized_pnl + unrealized
        self.stats.current_value = self.total_capital + self.stats.total_pnl - self.stats.total_fees

        # Update peak and drawdown
        if self.stats.current_value > self.stats.peak_value:
            self.stats.peak_value = self.stats.current_value

        drawdown = (self.stats.peak_value - self.stats.current_value) / self.stats.peak_value
        if drawdown > self.stats.max_drawdown:
            self.stats.max_drawdown = drawdown

    def _check_risk(self):
        """Check ensemble-level risk limits."""
        current_drawdown = (self.stats.peak_value - self.stats.current_value) / self.stats.peak_value

        if current_drawdown > self.config['max_drawdown']:
            print(f"[RISK] Max drawdown exceeded: {current_drawdown*100:.1f}% > {self.config['max_drawdown']*100:.1f}%")
            # Could trigger emergency close here

        if self.config.get('stop_loss') and self.current_price < self.config['stop_loss']:
            print(f"[RISK] Stop loss triggered: ${self.current_price:,.2f} < ${self.config['stop_loss']:,.2f}")

        if self.config.get('take_profit') and self.current_price > self.config['take_profit']:
            print(f"[RISK] Take profit triggered: ${self.current_price:,.2f} > ${self.config['take_profit']:,.2f}")

    def _log_state(self):
        """Log current state for diagnostics."""
        state = {
            'timestamp': self.last_update.isoformat(),
            'price': self.current_price,
            'total_value': self.stats.current_value,
            'realized_pnl': self.stats.realized_pnl,
            'unrealized_pnl': self.stats.unrealized_pnl,
            'open_positions': sum(len(p) for p in self.open_positions.values()),
            'drawdown': self.stats.max_drawdown
        }

        self.logger.log_market_state(
            prices={'BTC/USDT': self.current_price},
            features={'total_value': self.stats.current_value},
            regime='grid_trading',
            rsi={'BTC': 50},  # Would be calculated from data
            volatility=0.0
        )

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive ensemble status."""
        open_pos_count = sum(len(p) for p in self.open_positions.values())
        runtime = (datetime.now() - self.stats.start_time).total_seconds() / 3600  # hours

        status = {
            'ensemble': {
                'mode': 'paper_trading',
                'symbol': self.symbol,
                'current_price': self.current_price,
                'last_update': self.last_update.isoformat(),
                'runtime_hours': round(runtime, 2)
            },
            'capital': {
                'initial': self.total_capital,
                'current': round(self.stats.current_value, 2),
                'realized_pnl': round(self.stats.realized_pnl, 2),
                'unrealized_pnl': round(self.stats.unrealized_pnl, 2),
                'total_pnl': round(self.stats.total_pnl, 2),
                'pnl_pct': round(self.stats.total_pnl / self.total_capital * 100, 2),
                'total_fees': round(self.stats.total_fees, 2)
            },
            'performance': {
                'total_trades': self.stats.total_trades,
                'winning_trades': self.stats.winning_trades,
                'losing_trades': self.stats.losing_trades,
                'win_rate': round(self.stats.winning_trades / max(self.stats.total_trades, 1) * 100, 1),
                'max_drawdown': round(self.stats.max_drawdown * 100, 2),
                'open_positions': open_pos_count
            },
            'strategy_pnl': {k: round(v, 2) for k, v in self.stats.strategy_pnl.items()},
            'strategies': {name: strat.get_status() for name, strat in self.strategies.items()}
        }

        return status

    def print_status(self):
        """Print formatted status."""
        status = self.get_status()

        print("\n" + "="*60)
        print("GRID ENSEMBLE STATUS")
        print("="*60)
        print(f"Price: ${self.current_price:,.2f}  |  Runtime: {status['ensemble']['runtime_hours']:.1f}h")
        print("-"*60)
        print(f"Capital: ${status['capital']['current']:,.2f} "
              f"(PnL: {'+' if status['capital']['total_pnl'] >= 0 else ''}"
              f"${status['capital']['total_pnl']:,.2f} / {status['capital']['pnl_pct']:+.2f}%)")
        print(f"Realized: ${status['capital']['realized_pnl']:,.2f}  |  "
              f"Unrealized: ${status['capital']['unrealized_pnl']:,.2f}  |  "
              f"Fees: ${status['capital']['total_fees']:,.2f}")
        print("-"*60)
        print(f"Trades: {status['performance']['total_trades']}  |  "
              f"Wins: {status['performance']['winning_trades']}  |  "
              f"Losses: {status['performance']['losing_trades']}  |  "
              f"Win Rate: {status['performance']['win_rate']:.1f}%")
        print(f"Max Drawdown: {status['performance']['max_drawdown']:.2f}%  |  "
              f"Open Positions: {status['performance']['open_positions']}")
        print("-"*60)
        print("Strategy PnL:")
        for name, pnl in status['strategy_pnl'].items():
            pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
            print(f"  {name}: {pnl_str}")
        print("="*60 + "\n")

    def save_state(self, filepath: str = "grid_ensemble_state.json"):
        """Save current state to file."""
        state = {
            'config': self.config,
            'stats': asdict(self.stats),
            'paper_trades': [asdict(t) for t in self.paper_trades],
            'price_history': self.price_history[-100:],  # Last 100 prices
            'timestamp': datetime.now().isoformat()
        }

        # Convert datetime objects
        state['stats']['start_time'] = state['stats']['start_time'].isoformat() if isinstance(state['stats']['start_time'], datetime) else state['stats']['start_time']

        for trade in state['paper_trades']:
            if isinstance(trade.get('timestamp'), datetime):
                trade['timestamp'] = trade['timestamp'].isoformat()
            if isinstance(trade.get('close_time'), datetime):
                trade['close_time'] = trade['close_time'].isoformat()

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)

        print(f"State saved to {filepath}")

    def close(self):
        """Close the orchestrator and generate summary."""
        summary = close_diagnostic_logger()

        self.print_status()

        print("\n" + "="*60)
        print("FINAL SUMMARY")
        print("="*60)
        print(f"Total Runtime: {(datetime.now() - self.stats.start_time).total_seconds() / 3600:.2f} hours")
        print(f"Final Value: ${self.stats.current_value:,.2f}")
        print(f"Total Return: {self.stats.total_pnl / self.total_capital * 100:+.2f}%")
        print(f"Max Drawdown: {self.stats.max_drawdown * 100:.2f}%")
        print(f"Total Trades: {self.stats.total_trades}")
        print(f"Win Rate: {self.stats.winning_trades / max(self.stats.total_trades, 1) * 100:.1f}%")
        print("="*60)

        self.save_state()

        return summary


def run_paper_grid_ensemble(
    duration_minutes: int = 60,
    update_interval: int = 30,
    initial_capital: float = 10000.0
):
    """
    Run the grid ensemble in paper trading mode.

    Args:
        duration_minutes: How long to run
        update_interval: Seconds between updates
        initial_capital: Starting capital
    """
    import ccxt
    import os

    # Initialize Kraken exchange for price data (public API - no auth needed for market data)
    exchange = ccxt.kraken({
        'enableRateLimit': True,
    })

    # Kraken uses BTC/USD (not USDT)
    config = {
        'symbol': 'BTC/USD',
        'total_capital': initial_capital,
        'upper_price': 100000,
        'lower_price': 94000,
        'max_drawdown': 0.10,
        'stop_loss': 92000,
        'take_profit': 102000,
        'fee_rate': 0.0026  # Kraken taker fee
    }

    # Symbol for Kraken API
    trading_symbol = 'BTC/USD'

    orchestrator = GridEnsembleOrchestrator(config)

    print(f"\nStarting Grid Ensemble Paper Trading")
    print(f"Duration: {duration_minutes} minutes")
    print(f"Update interval: {update_interval} seconds")
    print(f"Capital: ${initial_capital:,.2f}")
    print("-"*50)

    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)

    try:
        while time.time() < end_time:
            # Fetch current price and OHLCV from Kraken
            try:
                ticker = exchange.fetch_ticker(trading_symbol)
                current_price = ticker['last']

                ohlcv = exchange.fetch_ohlcv(trading_symbol, '5m', limit=100)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            except Exception as e:
                print(f"Error fetching data: {e}")
                time.sleep(update_interval)
                continue

            # Update strategies
            signals = orchestrator.update(current_price, df)

            # Execute paper trades
            if signals:
                results = orchestrator.execute_paper(signals)

            # Print status every 5 minutes
            elapsed = time.time() - start_time
            if int(elapsed) % 300 < update_interval:
                orchestrator.print_status()

            time.sleep(update_interval)

    except KeyboardInterrupt:
        print("\n\nStopping paper trading...")

    finally:
        orchestrator.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Grid Ensemble Paper Trading')
    parser.add_argument('--duration', type=int, default=60, help='Duration in minutes')
    parser.add_argument('--interval', type=int, default=30, help='Update interval in seconds')
    parser.add_argument('--capital', type=float, default=10000.0, help='Initial capital')

    args = parser.parse_args()

    run_paper_grid_ensemble(
        duration_minutes=args.duration,
        update_interval=args.interval,
        initial_capital=args.capital
    )
