"""
Grid Trading Base Strategy - Phase 26 Enhanced
Provides common functionality for all grid-based trading strategies.

Improvements over Phase 24:
- Dynamic ATR-based grid spacing
- Compound position sizing with profit reinvestment
- Fixed fee calculation (was double-counting)
- Proper position-level tracking with order IDs
- Multi-asset support (BTC + XRP)
- Slippage tolerance handling
- Grid state consistency protection
- Session-aware adjustments
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod
import uuid


@dataclass
class GridLevel:
    """Represents a single grid level."""
    price: float
    side: str  # 'buy' or 'sell'
    size: float
    filled: bool = False
    fill_price: float = 0.0
    fill_time: Optional[datetime] = None
    order_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    original_side: str = ""  # Preserve original side to prevent state corruption
    level_index: int = 0  # Track position in grid

    def __post_init__(self):
        if not self.original_side:
            self.original_side = self.side


@dataclass
class GridPosition:
    """Tracks a position from a filled grid order."""
    entry_price: float
    size: float
    side: str  # 'long' or 'short'
    entry_time: datetime
    grid_level: int
    order_id: str = ""  # Link to grid level order
    unrealized_pnl: float = 0.0
    target_exit_price: float = 0.0  # Expected exit price for this position
    entry_fee: float = 0.0  # Fee paid on entry


@dataclass
class GridStats:
    """Statistics for grid performance."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    realized_pnl: float = 0.0  # Separated from unrealized
    total_fees: float = 0.0
    max_drawdown: float = 0.0
    cycles_completed: int = 0  # Full buy->sell cycles
    avg_cycle_time: float = 0.0
    cycle_times: List[float] = field(default_factory=list)  # Track individual cycle times
    profit_reinvested: float = 0.0  # Track compounding


class GridBaseStrategy(ABC):
    """
    Abstract base class for grid trading strategies.

    Phase 26 Enhanced with:
    - Dynamic ATR-based grid spacing
    - Compound position sizing
    - Multi-asset support
    - Proper fee handling
    - State consistency protection
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize grid strategy.

        Args:
            config: Strategy configuration dict
        """
        self.symbol = config.get('symbol', 'BTC/USDT')
        self.upper_price = config.get('upper_price', 95000)
        self.lower_price = config.get('lower_price', 90000)
        self.num_grids = config.get('num_grids', 20)
        self.total_capital = config.get('total_capital', 10000)
        self.leverage = config.get('leverage', 1)

        # Risk parameters
        self.stop_loss = config.get('stop_loss', None)
        self.take_profit = config.get('take_profit', None)
        self.max_drawdown = config.get('max_drawdown', 0.10)
        self.fee_rate = config.get('fee_rate', 0.001)  # 0.1%

        # Phase 26: Dynamic grid parameters
        self.use_atr_spacing = config.get('use_atr_spacing', False)
        self.atr_multiplier = config.get('atr_multiplier', 0.3)
        self.atr_period = config.get('atr_period', 14)

        # Phase 26: Compound position sizing
        self.compound_profits = config.get('compound_profits', True)
        self.max_compound = config.get('max_compound', 1.5)  # Max 150% of initial
        self.profit_distribution = config.get('profit_distribution', {
            'reinvest': 0.6,   # 60% reinvested
            'realized': 0.4    # 40% taken as profit
        })

        # Phase 26: Slippage handling - INCREASED for live trading robustness
        # 0.5% default handles volatile crypto markets better
        self.slippage_tolerance = config.get('slippage_tolerance', 0.005)  # 0.5%
        self.min_profit_per_cycle = config.get('min_profit_per_cycle', 0.001)  # 0.1% min

        # Phase 26: Multi-asset support
        self.secondary_symbol = config.get('secondary_symbol', None)  # e.g., 'XRP/USDT'
        self.secondary_allocation = config.get('secondary_allocation', 0.0)  # 0-1

        # State
        self.grid_levels: List[GridLevel] = []
        self.positions: List[GridPosition] = []
        self.position_map: Dict[str, GridPosition] = {}  # order_id -> position
        self.stats = GridStats()
        self.current_price = 0.0
        self.initial_capital = self.total_capital
        self.peak_capital = self.total_capital
        self.current_atr = 0.0
        self.last_atr_update = None

        # Phase 26: Grid state tracking
        self.grid_center_price = (self.upper_price + self.lower_price) / 2
        self.grid_initialized = False
        self.pending_rebalance = False

        # Initialize grid
        self._setup_grid()

    @abstractmethod
    def _setup_grid(self):
        """Set up grid levels. Implemented by subclasses."""
        pass

    @abstractmethod
    def _calculate_position_size(self, level: int, price: float) -> float:
        """Calculate position size for a grid level."""
        pass

    def calculate_atr(self, data: pd.DataFrame) -> float:
        """Calculate ATR from OHLCV data."""
        if data is None or len(data) < self.atr_period + 1:
            return 0.0

        high = data['high'].iloc[-self.atr_period:]
        low = data['low'].iloc[-self.atr_period:]
        close = data['close'].iloc[-self.atr_period:]

        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.mean()

        return atr

    def update(self, current_price: float, data: pd.DataFrame = None) -> List[Dict[str, Any]]:
        """
        Update grid state with current price.

        Phase 26 Enhanced:
        - ATR-based dynamic spacing updates
        - Smart grid level detection with slippage tolerance
        - Position state consistency checks

        Args:
            current_price: Current market price
            data: Optional OHLCV data for advanced strategies

        Returns:
            List of signals/orders to execute
        """
        self.current_price = current_price
        signals = []

        # Update ATR if data provided
        if data is not None and self.use_atr_spacing:
            new_atr = self.calculate_atr(data)
            if new_atr > 0:
                self.current_atr = new_atr
                self.last_atr_update = datetime.now()

        # Check stop loss
        if self.stop_loss and current_price < self.stop_loss:
            signals.append(self._create_exit_signal('stop_loss'))
            return signals

        # Check take profit
        if self.take_profit and current_price > self.take_profit:
            signals.append(self._create_exit_signal('take_profit'))
            return signals

        # Check max drawdown
        current_value = self._calculate_portfolio_value()
        drawdown = (self.peak_capital - current_value) / self.peak_capital if self.peak_capital > 0 else 0
        if drawdown > self.max_drawdown:
            signals.append(self._create_exit_signal('max_drawdown'))
            self.stats.max_drawdown = max(self.stats.max_drawdown, drawdown)
            return signals

        # Update peak and track drawdown
        if current_value > self.peak_capital:
            self.peak_capital = current_value
        self.stats.max_drawdown = max(self.stats.max_drawdown, drawdown)

        # Phase 26: Check grid levels with slippage tolerance
        for i, level in enumerate(self.grid_levels):
            if not level.filled:
                # Calculate price tolerance for this level
                tolerance = level.price * self.slippage_tolerance

                # Check if price crossed this level (with tolerance)
                if level.side == 'buy':
                    if current_price <= level.price + tolerance:
                        # Verify we're not buying above target + tolerance
                        effective_price = min(current_price, level.price)
                        signal = self._create_buy_signal(i, level)
                        signal['effective_price'] = effective_price
                        signals.append(signal)
                elif level.side == 'sell':
                    if current_price >= level.price - tolerance:
                        # Verify we're not selling below target - tolerance
                        effective_price = max(current_price, level.price)
                        signal = self._create_sell_signal(i, level)
                        signal['effective_price'] = effective_price
                        signals.append(signal)

        # Update unrealized PnL
        self._update_positions_pnl()

        # Phase 26: Check if rebalancing is needed based on completed cycles
        if self.compound_profits and self.stats.cycles_completed > 0:
            self._check_profit_rebalance()

        return signals

    def fill_order(self, level_idx: int, fill_price: float, fill_time: datetime = None, order_id: str = None):
        """
        Mark a grid order as filled.

        Phase 26 Enhanced:
        - Proper fee calculation (separate entry/exit fees)
        - Order ID tracking for position matching
        - Cycle time tracking for statistics
        - Compound profit handling

        Args:
            level_idx: Index of the grid level
            fill_price: Actual fill price
            fill_time: Time of fill
            order_id: Optional order ID for tracking
        """
        if level_idx >= len(self.grid_levels):
            return

        level = self.grid_levels[level_idx]
        level.filled = True
        level.fill_price = fill_price
        level.fill_time = fill_time or datetime.now()

        # Phase 26: Calculate fee correctly (single leg only)
        entry_fee = level.size * fill_price * self.fee_rate
        self.stats.total_fees += entry_fee

        if level.side == 'buy':
            # Calculate target exit price (next level up)
            target_exit = self._get_next_sell_price(level_idx)

            # Create new position with proper tracking
            position = GridPosition(
                entry_price=fill_price,
                size=level.size,
                side='long',
                entry_time=level.fill_time,
                grid_level=level_idx,
                order_id=level.order_id,
                target_exit_price=target_exit,
                entry_fee=entry_fee
            )
            self.positions.append(position)
            self.position_map[level.order_id] = position
            self.stats.total_trades += 1

            # Set up sell order at next level up
            self._activate_sell_level(level_idx)

        elif level.side == 'sell':
            # Close corresponding position with proper matching
            pnl, cycle_time = self._close_position_for_level(level_idx, fill_price)

            if pnl > 0:
                self.stats.winning_trades += 1
            else:
                self.stats.losing_trades += 1

            # Phase 26: Handle profit distribution
            if self.compound_profits and pnl > 0:
                reinvest_amount = pnl * self.profit_distribution.get('reinvest', 0.6)
                realized_amount = pnl * self.profit_distribution.get('realized', 0.4)
                self.stats.profit_reinvested += reinvest_amount
                self.stats.realized_pnl += realized_amount
                self.total_capital += reinvest_amount  # Compound the reinvested portion
            else:
                self.stats.realized_pnl += pnl

            self.stats.total_pnl += pnl
            self.stats.cycles_completed += 1

            # Track cycle time
            if cycle_time > 0:
                self.stats.cycle_times.append(cycle_time)
                self.stats.avg_cycle_time = np.mean(self.stats.cycle_times)

            # Reactivate buy order at the level below
            self._reactivate_buy_level(level_idx)

    def _get_next_sell_price(self, buy_level_idx: int) -> float:
        """Get the target sell price for a buy at given level."""
        sell_idx = buy_level_idx + 1
        if sell_idx < len(self.grid_levels):
            return self.grid_levels[sell_idx].price
        return self.upper_price

    def _create_buy_signal(self, level_idx: int, level: GridLevel) -> Dict[str, Any]:
        """Create a buy signal with Phase 26 enhancements + Phase 32 limit orders."""
        # Phase 26: Calculate compound-adjusted size
        adjusted_size = self._get_compound_adjusted_size(level.size)

        return {
            'action': 'buy',
            'symbol': self.symbol,
            'price': level.price,
            'limit_price': level.price,  # Phase 32: Use limit orders for better fills
            'order_type': 'limit',  # Phase 32: Default to limit orders
            'size': adjusted_size,
            'original_size': level.size,
            'leverage': self.leverage,
            'grid_level': level_idx,
            'order_id': level.order_id,
            'strategy': self.__class__.__name__,
            'reason': f'Grid buy at level {level_idx} (${level.price:,.2f})',
            'confidence': 0.7,
            'target_sell_price': self._get_next_sell_price(level_idx)
        }

    def _create_sell_signal(self, level_idx: int, level: GridLevel) -> Dict[str, Any]:
        """Create a sell signal with Phase 26 enhancements + Phase 32 limit orders."""
        # Find the corresponding position for accurate sizing
        position = self._find_position_for_sell(level_idx)
        size = position.size if position else level.size

        return {
            'action': 'sell',
            'symbol': self.symbol,
            'price': level.price,
            'limit_price': level.price,  # Phase 32: Use limit orders for better fills
            'order_type': 'limit',  # Phase 32: Default to limit orders
            'size': size,
            'leverage': self.leverage,
            'grid_level': level_idx,
            'order_id': level.order_id,
            'strategy': self.__class__.__name__,
            'reason': f'Grid sell at level {level_idx} (${level.price:,.2f})',
            'confidence': 0.7,
            'position_id': position.order_id if position else None
        }

    def _create_exit_signal(self, reason: str) -> Dict[str, Any]:
        """Create an exit all positions signal - uses market order for immediate exit."""
        total_size = sum(p.size for p in self.positions)
        total_unrealized = sum(p.unrealized_pnl for p in self.positions)

        return {
            'action': 'close_all',
            'symbol': self.symbol,
            'size': total_size,
            'order_type': 'market',  # Phase 32: Market order for emergency exits
            'strategy': self.__class__.__name__,
            'reason': f'Grid exit: {reason}',
            'unrealized_pnl': total_unrealized,
            'position_count': len(self.positions),
            'confidence': 0.9
        }

    def _activate_sell_level(self, buy_level_idx: int):
        """
        Activate the sell level above a filled buy.

        Phase 26: State consistency protection - only modify sell level,
        preserve original_side for recovery.
        """
        sell_idx = buy_level_idx + 1
        if sell_idx < len(self.grid_levels):
            sell_level = self.grid_levels[sell_idx]
            # Only activate if this level should be a sell
            if sell_level.original_side == 'sell' or sell_level.level_index > buy_level_idx:
                sell_level.filled = False
                sell_level.side = 'sell'
                # Generate new order ID for this activation
                sell_level.order_id = str(uuid.uuid4())[:8]

    def _reactivate_buy_level(self, sell_level_idx: int):
        """
        Reactivate a buy level after sell is complete.

        Phase 26: Reactivate at the level below the sell,
        preserving original grid structure.
        """
        buy_idx = sell_level_idx - 1
        if buy_idx >= 0:
            buy_level = self.grid_levels[buy_idx]
            # Only reactivate if this level should be a buy
            if buy_level.original_side == 'buy' or buy_level.level_index < sell_level_idx:
                buy_level.filled = False
                buy_level.side = 'buy'
                # Generate new order ID for this reactivation
                buy_level.order_id = str(uuid.uuid4())[:8]

    def _find_position_for_sell(self, sell_level_idx: int) -> Optional[GridPosition]:
        """Find the position that corresponds to this sell level."""
        buy_level_idx = sell_level_idx - 1
        for pos in self.positions:
            if pos.grid_level == buy_level_idx:
                return pos
        return None

    def _close_position_for_level(self, level_idx: int, exit_price: float) -> Tuple[float, float]:
        """
        Close position associated with a grid level.

        Phase 26 Fixed:
        - Correct fee calculation (exit fee only, entry already paid)
        - Returns (pnl, cycle_time) tuple
        - Uses order_id matching when available
        """
        buy_level_idx = level_idx - 1

        for i, pos in enumerate(self.positions):
            if pos.grid_level == buy_level_idx:
                # Calculate gross PnL
                gross_pnl = (exit_price - pos.entry_price) * pos.size * self.leverage

                # Phase 26: Fixed fee calculation - only exit fee here
                # Entry fee was already deducted when position opened
                exit_fee = exit_price * pos.size * self.fee_rate
                self.stats.total_fees += exit_fee

                # Net PnL = gross - exit_fee (entry fee already tracked)
                net_pnl = gross_pnl - exit_fee

                # Calculate cycle time
                cycle_time = 0.0
                if pos.entry_time:
                    cycle_time = (datetime.now() - pos.entry_time).total_seconds()

                # Remove position
                self.positions.pop(i)
                if pos.order_id in self.position_map:
                    del self.position_map[pos.order_id]

                return net_pnl, cycle_time

        return 0.0, 0.0

    def _get_compound_adjusted_size(self, base_size: float) -> float:
        """
        Calculate position size adjusted for compound profits.

        Phase 26: Increases size based on reinvested profits,
        capped at max_compound multiplier.
        """
        if not self.compound_profits or self.initial_capital <= 0:
            return base_size

        # Calculate compound multiplier based on reinvested profits
        compound_ratio = 1 + (self.stats.profit_reinvested / self.initial_capital)

        # Cap at max_compound
        compound_ratio = min(compound_ratio, self.max_compound)

        return base_size * compound_ratio

    def _check_profit_rebalance(self):
        """
        Check if grid sizes should be rebalanced based on profits.

        Phase 26: After significant profit accumulation, flag for rebalance.
        """
        if self.stats.profit_reinvested > self.initial_capital * 0.1:  # 10% profit threshold
            self.pending_rebalance = True

    def _update_positions_pnl(self):
        """Update unrealized PnL for all positions."""
        for pos in self.positions:
            if pos.side == 'long':
                pos.unrealized_pnl = (self.current_price - pos.entry_price) * pos.size * self.leverage
            else:
                pos.unrealized_pnl = (pos.entry_price - self.current_price) * pos.size * self.leverage

    def _calculate_portfolio_value(self) -> float:
        """Calculate total portfolio value."""
        unrealized = sum(p.unrealized_pnl for p in self.positions)
        return self.total_capital + self.stats.realized_pnl + unrealized

    def get_status(self) -> Dict[str, Any]:
        """Get current grid status with Phase 26 enhanced metrics."""
        current_value = self._calculate_portfolio_value()
        drawdown = (self.peak_capital - current_value) / self.peak_capital if self.peak_capital > 0 else 0

        # Count active grid levels
        active_buys = sum(1 for l in self.grid_levels if l.side == 'buy' and not l.filled)
        active_sells = sum(1 for l in self.grid_levels if l.side == 'sell' and not l.filled)

        return {
            'strategy': self.__class__.__name__,
            'symbol': self.symbol,
            'current_price': self.current_price,
            'upper_price': self.upper_price,
            'lower_price': self.lower_price,
            'grid_center': self.grid_center_price,
            'num_grids': self.num_grids,

            # Position tracking
            'active_positions': len(self.positions),
            'active_buy_levels': active_buys,
            'active_sell_levels': active_sells,

            # PnL metrics
            'total_pnl': self.stats.total_pnl,
            'realized_pnl': self.stats.realized_pnl,
            'unrealized_pnl': sum(p.unrealized_pnl for p in self.positions),
            'total_fees': self.stats.total_fees,

            # Phase 26: Compound metrics
            'profit_reinvested': self.stats.profit_reinvested,
            'compound_multiplier': 1 + (self.stats.profit_reinvested / self.initial_capital) if self.initial_capital > 0 else 1,

            # Trade statistics
            'cycles_completed': self.stats.cycles_completed,
            'total_trades': self.stats.total_trades,
            'winning_trades': self.stats.winning_trades,
            'losing_trades': self.stats.losing_trades,
            'win_rate': self.stats.winning_trades / max(self.stats.total_trades, 1),
            'avg_cycle_time_seconds': self.stats.avg_cycle_time,

            # Risk metrics
            'portfolio_value': current_value,
            'drawdown': drawdown,
            'max_drawdown': self.stats.max_drawdown,
            'initial_capital': self.initial_capital,
            'peak_capital': self.peak_capital,

            # Phase 26: ATR info
            'current_atr': self.current_atr,
            'use_atr_spacing': self.use_atr_spacing,
            'pending_rebalance': self.pending_rebalance
        }

    def reset_grid(self, new_center_price: float = None):
        """
        Reset grid around a new center price.

        Phase 26: Preserves positions but rebuilds grid levels.
        """
        if new_center_price:
            half_range = (self.upper_price - self.lower_price) / 2
            self.upper_price = new_center_price + half_range
            self.lower_price = new_center_price - half_range
            self.grid_center_price = new_center_price

        self._setup_grid()
        self.pending_rebalance = False


class ArithmeticGrid(GridBaseStrategy):
    """
    Arithmetic Grid Strategy - Phase 26 Enhanced.

    Features:
    - Fixed price spacing between grid levels
    - Optional ATR-based dynamic spacing
    - Smart initial grid placement based on current price
    - Compound position sizing
    """

    def _setup_grid(self, current_price: float = None):
        """
        Set up arithmetic grid with equal spacing.

        Phase 26 Enhanced:
        - ATR-based spacing option
        - Proper buy/sell placement relative to current price
        - Level indexing for state tracking
        """
        price_range = self.upper_price - self.lower_price

        # Phase 26: Calculate spacing - use ATR if available and enabled
        if self.use_atr_spacing and self.current_atr > 0:
            # ATR-based spacing: each grid level is atr_multiplier * ATR apart
            spacing = self.current_atr * self.atr_multiplier
            # Recalculate num_grids to fit the range
            self.num_grids = max(5, int(price_range / spacing))
        else:
            spacing = price_range / self.num_grids

        self.grid_spacing = spacing
        self.grid_center_price = (self.upper_price + self.lower_price) / 2

        # Use provided current price or center price for buy/sell determination
        reference_price = current_price or self.grid_center_price

        self.grid_levels = []
        for i in range(self.num_grids + 1):
            price = self.lower_price + (i * spacing)
            size = self._calculate_position_size(i, price)

            # Phase 26: Determine side based on current price, not grid midpoint
            # Levels below current price are buy orders
            # Levels above current price are sell orders
            # Use 0.2% buffer to prevent edge-case failures at grid boundaries
            if price < reference_price * 0.998:
                side = 'buy'
            elif price > reference_price * 1.002:
                side = 'sell'
            else:
                # Price within buffer zone - default to buy for accumulation bias
                side = 'buy'

            self.grid_levels.append(GridLevel(
                price=price,
                side=side,
                size=size,
                original_side=side,
                level_index=i
            ))

        self.grid_initialized = True

    def _calculate_position_size(self, level: int, price: float) -> float:
        """
        Calculate position size for a grid level.

        Phase 26: Uses actual fill price expectation, not just level price.
        """
        if price <= 0:
            return 0

        # Base capital per grid
        effective_capital = self.total_capital
        if self.compound_profits:
            # Include reinvested profits in capital calculation
            effective_capital += self.stats.profit_reinvested
            # Cap at max compound
            effective_capital = min(effective_capital, self.initial_capital * self.max_compound)

        capital_per_grid = effective_capital / self.num_grids
        return capital_per_grid / price

    def recenter_grid(self, new_center_price: float):
        """
        Recenter the grid around a new price point.

        Phase 26: Smart recentering that preserves existing positions.
        """
        half_range = self.grid_spacing * self.num_grids / 2

        self.upper_price = new_center_price + half_range
        self.lower_price = new_center_price - half_range
        self.grid_center_price = new_center_price

        # Rebuild grid with new center
        self._setup_grid(new_center_price)


class GeometricGrid(GridBaseStrategy):
    """
    Geometric Grid Strategy - Phase 26 Enhanced.

    Percentage-based spacing with larger positions at extremes for mean reversion.

    Phase 26 Improvements:
    - Dynamic ATR-based grid ratio calculation
    - Smart buy/sell placement based on current price (not midpoint)
    - Compound position sizing with profit reinvestment
    - State consistency with original_side tracking
    - Multi-asset support
    - Proper recentering logic
    """

    def __init__(self, config: Dict[str, Any]):
        # Grid ratio - default 2% between levels for crypto volatility
        self.grid_ratio = config.get('grid_ratio', 1.02)
        self.base_grid_ratio = self.grid_ratio  # Store original for ATR adjustment

        # Position sizing at extremes
        self.size_multiplier = config.get('size_multiplier', 1.3)  # Increased from 1.15
        self.extreme_multiplier = config.get('extreme_multiplier', 1.5)  # Extra boost at extremes

        # Phase 26: ATR-based dynamic ratio
        self.use_atr_ratio = config.get('use_atr_ratio', True)
        self.atr_ratio_mult = config.get('atr_ratio_mult', 0.4)  # ratio = 1 + (ATR% * mult)
        self.min_grid_ratio = config.get('min_grid_ratio', 1.005)  # Min 0.5% spacing
        self.max_grid_ratio = config.get('max_grid_ratio', 1.05)   # Max 5% spacing

        # Phase 26: Recentering control
        self.recenter_after_cycles = config.get('recenter_after_cycles', 5)
        self.min_recenter_interval = config.get('min_recenter_interval', 3600)  # 1 hour
        self.last_recenter_time = None
        self.cycles_at_last_recenter = 0

        # Phase 26: Profit distribution for geometric (more aggressive reinvestment)
        if 'profit_distribution' not in config:
            config['profit_distribution'] = {
                'reinvest': 0.7,   # 70% reinvested (more aggressive for geometric)
                'realized': 0.3    # 30% taken as profit
            }

        super().__init__(config)

    def _setup_grid(self, current_price: float = None):
        """
        Set up geometric grid with percentage spacing.

        Phase 26 Enhanced:
        - Accepts current_price for smart buy/sell placement
        - Uses ATR-based dynamic grid ratio when available
        - Proper state tracking with original_side and level_index
        - Calculates grid levels from center outward for symmetry
        """
        self.grid_levels = []

        # Use provided current price or calculate center
        reference_price = current_price or (self.upper_price + self.lower_price) / 2
        self.grid_center_price = reference_price

        # Phase 26: Calculate ATR-adjusted grid ratio
        effective_ratio = self._calculate_effective_ratio()

        # Calculate how many levels fit in the range
        # For geometric: upper/lower = ratio^n, so n = log(upper/lower) / log(ratio)
        price_range_ratio = self.upper_price / self.lower_price

        if effective_ratio > 1:
            calculated_grids = int(np.log(price_range_ratio) / np.log(effective_ratio))
            # Use the smaller of calculated or configured grids
            actual_grids = min(calculated_grids, self.num_grids)
        else:
            actual_grids = self.num_grids

        # Generate grid prices geometrically from lower to upper
        grid_prices = []
        price = self.lower_price
        for i in range(actual_grids + 1):
            grid_prices.append(price)
            price *= effective_ratio
            if price > self.upper_price * 1.001:  # Small tolerance
                break

        # Phase 26: Assign buy/sell based on current price position
        for i, price in enumerate(grid_prices):
            size = self._calculate_position_size(i, price, len(grid_prices))

            # Smart side assignment based on current price
            if price < reference_price * 0.998:  # Small buffer to avoid edge cases
                side = 'buy'
            elif price > reference_price * 1.002:
                side = 'sell'
            else:
                # Price at reference - use accumulation bias (buy)
                side = 'buy'

            self.grid_levels.append(GridLevel(
                price=price,
                side=side,
                size=size,
                original_side=side,
                level_index=i
            ))

        self.num_grids = len(self.grid_levels) - 1 if self.grid_levels else 0
        self.grid_spacing = effective_ratio  # Store for reference
        self.grid_initialized = True

    def _calculate_effective_ratio(self) -> float:
        """
        Calculate the effective grid ratio, optionally adjusted by ATR.

        Phase 26: Dynamic ratio based on market volatility.
        """
        if not self.use_atr_ratio or self.current_atr <= 0:
            return self.grid_ratio

        # ATR-based ratio: wider spacing in volatile markets
        # ratio = 1 + (ATR% * multiplier)
        atr_pct = self.current_atr  # Already as percentage
        dynamic_ratio = 1 + (atr_pct * self.atr_ratio_mult)

        # Clamp to min/max bounds
        effective_ratio = max(self.min_grid_ratio, min(self.max_grid_ratio, dynamic_ratio))

        return effective_ratio

    def _calculate_position_size(self, level: int, price: float, total_levels: int = None) -> float:
        """
        Calculate position size with larger positions at extremes.

        Phase 26 Enhanced:
        - More aggressive multiplier at extremes for mean reversion
        - Compound-aware sizing
        - Proper center calculation
        """
        if price <= 0:
            return 0

        total_levels = total_levels or self.num_grids + 1
        if total_levels <= 1:
            total_levels = self.num_grids + 1

        # Phase 26: Compound-adjusted capital
        effective_capital = self.total_capital
        if self.compound_profits:
            effective_capital += self.stats.profit_reinvested
            effective_capital = min(effective_capital, self.initial_capital * self.max_compound)

        # Base size per grid
        base_size = effective_capital / (total_levels * price)

        # Distance from center for extreme bias
        center = total_levels // 2
        if center == 0:
            center = 1

        distance = abs(level - center)
        distance_ratio = distance / center if center > 0 else 0

        # Phase 26: Graduated multiplier - more aggressive at true extremes
        if distance_ratio > 0.8:  # Top/bottom 20% of grid
            # Extra boost at extremes
            multiplier = self.size_multiplier ** distance_ratio * self.extreme_multiplier
        else:
            multiplier = self.size_multiplier ** distance_ratio

        return base_size * multiplier * self.leverage

    def recenter_grid(self, new_center_price: float):
        """
        Recenter the grid around a new price point.

        Phase 26: Smart recentering that preserves stats and adjusts range.
        """
        # Calculate new range maintaining the same ratio spread
        if self.grid_levels and len(self.grid_levels) > 1:
            # Use actual grid spread to calculate new range
            current_spread = self.grid_levels[-1].price / self.grid_levels[0].price
            half_spread = np.sqrt(current_spread)

            self.upper_price = new_center_price * half_spread
            self.lower_price = new_center_price / half_spread
        else:
            # Fallback to configured range_pct
            range_pct = (self.upper_price - self.lower_price) / (self.upper_price + self.lower_price)
            self.upper_price = new_center_price * (1 + range_pct)
            self.lower_price = new_center_price * (1 - range_pct)

        self.grid_center_price = new_center_price

        # Rebuild grid with new center
        self._setup_grid(new_center_price)

        # Update recentering tracking
        self.last_recenter_time = datetime.now()
        self.cycles_at_last_recenter = self.stats.cycles_completed
        self.pending_rebalance = False

    def should_recenter(self, current_price: float) -> bool:
        """
        Check if grid should be recentered.

        Phase 26: Multiple criteria for smart recentering.
        """
        if not self.grid_initialized or self.grid_center_price == 0:
            return True

        # Time-based check
        if self.last_recenter_time:
            time_since = (datetime.now() - self.last_recenter_time).total_seconds()
            if time_since < self.min_recenter_interval:
                return False

        # Price deviation check - for geometric, use ratio
        if current_price > 0 and self.grid_center_price > 0:
            price_ratio = max(current_price / self.grid_center_price,
                             self.grid_center_price / current_price)
            # Recenter if price moved more than half the grid range
            max_ratio = np.sqrt(self.upper_price / self.lower_price)
            if price_ratio > max_ratio:
                return True

        # Cycle-based check
        cycles_since = self.stats.cycles_completed - self.cycles_at_last_recenter
        if cycles_since >= self.recenter_after_cycles:
            return True

        return False

    def update(self, current_price: float, data: pd.DataFrame = None) -> List[Dict[str, Any]]:
        """
        Update grid state with current price.

        Phase 26 Enhanced:
        - ATR-based dynamic ratio updates
        - Smart recentering when needed
        - Proper signal generation
        """
        self.current_price = current_price
        signals = []

        # Update ATR if data provided
        if data is not None and self.use_atr_spacing:
            new_atr = self.calculate_atr(data)
            if new_atr > 0:
                self.current_atr = new_atr
                self.last_atr_update = datetime.now()

        # Phase 26: Check for recentering
        if self.should_recenter(current_price):
            old_center = self.grid_center_price
            self.recenter_grid(current_price)
            # Don't skip signals after recenter - continue processing

        # Check stop loss
        if self.stop_loss and current_price < self.stop_loss:
            signals.append(self._create_exit_signal('stop_loss'))
            return signals

        # Check take profit
        if self.take_profit and current_price > self.take_profit:
            signals.append(self._create_exit_signal('take_profit'))
            return signals

        # Check max drawdown
        current_value = self._calculate_portfolio_value()
        drawdown = (self.peak_capital - current_value) / self.peak_capital if self.peak_capital > 0 else 0
        if drawdown > self.max_drawdown:
            signals.append(self._create_exit_signal('max_drawdown'))
            self.stats.max_drawdown = max(self.stats.max_drawdown, drawdown)
            return signals

        # Update peak and track drawdown
        if current_value > self.peak_capital:
            self.peak_capital = current_value
        self.stats.max_drawdown = max(self.stats.max_drawdown, drawdown)

        # Phase 26: Check grid levels with geometric-appropriate tolerance
        for i, level in enumerate(self.grid_levels):
            if not level.filled:
                # For geometric grids, tolerance should scale with price
                tolerance = level.price * self.slippage_tolerance

                if level.side == 'buy':
                    if current_price <= level.price + tolerance:
                        effective_price = min(current_price, level.price)
                        signal = self._create_buy_signal(i, level)
                        signal['effective_price'] = effective_price
                        signals.append(signal)

                elif level.side == 'sell':
                    if current_price >= level.price - tolerance:
                        effective_price = max(current_price, level.price)
                        signal = self._create_sell_signal(i, level)
                        signal['effective_price'] = effective_price
                        signals.append(signal)

        # Update unrealized PnL
        self._update_positions_pnl()

        # Phase 26: Check if rebalancing is needed
        if self.compound_profits and self.stats.cycles_completed > 0:
            self._check_profit_rebalance()

        return signals

    def get_status(self) -> Dict[str, Any]:
        """Get status with geometric-specific metrics."""
        status = super().get_status()
        status.update({
            'grid_ratio': self.grid_ratio,
            'effective_ratio': self._calculate_effective_ratio(),
            'size_multiplier': self.size_multiplier,
            'extreme_multiplier': self.extreme_multiplier,
            'use_atr_ratio': self.use_atr_ratio,
            'cycles_since_recenter': self.stats.cycles_completed - self.cycles_at_last_recenter,
        })
        return status


class RSIMeanReversionGrid(GridBaseStrategy):
    """
    Phase 26 Enhanced RSI Mean Reversion Grid.

    Uses RSI to enhance grid trading with mean reversion signals.
    RSI serves as confidence modifier rather than hard filter.

    Phase 26 Improvements:
    - Smart grid setup based on current price (not index)
    - RSI as confidence modifier, not hard filter
    - Dynamic ATR-based grid spacing
    - Cycle-based recentering
    - Compound position sizing
    - Slippage tolerance handling
    - Proper state tracking with order IDs
    """

    def __init__(self, config: Dict[str, Any]):
        # RSI parameters
        self.rsi_period = config.get('rsi_period', 14)
        self.rsi_oversold = config.get('rsi_oversold', 30)  # Relaxed from 25
        self.rsi_overbought = config.get('rsi_overbought', 70)  # Relaxed from 75
        self.current_rsi = 50.0
        self.rsi_history: List[float] = []

        # Phase 26: RSI behavior mode
        # 'confidence_only' = RSI modifies confidence, doesn't block trades
        # 'filter' = RSI must be in zone to trade (legacy behavior)
        self.rsi_mode = config.get('rsi_mode', 'confidence_only')

        # Phase 26: Adaptive RSI zones
        self.use_adaptive_rsi = config.get('use_adaptive_rsi', True)
        self.rsi_zone_expansion = config.get('rsi_zone_expansion', 5)  # Expand zones by ATR%

        # Phase 26: Recentering control
        self.recenter_after_cycles = config.get('recenter_after_cycles', 4)
        self.min_recenter_interval = config.get('min_recenter_interval', 3600)  # 1 hour
        self.last_recenter_time = None
        self.cycles_at_last_recenter = 0

        # Phase 26: Position sizing at RSI extremes
        self.rsi_extreme_multiplier = config.get('rsi_extreme_multiplier', 1.3)

        super().__init__(config)

    def _setup_grid(self, current_price: float = None):
        """
        Set up grid with smart buy/sell assignment based on current price.

        Phase 26 Enhanced:
        - Accepts current_price for proper buy/sell placement
        - Uses ATR for dynamic spacing when available
        - Proper level indexing for state tracking
        """
        price_range = self.upper_price - self.lower_price

        # Phase 26: Calculate spacing - use ATR if available and enabled
        if self.use_atr_spacing and self.current_atr > 0:
            spacing = self.current_atr * self.atr_multiplier
            # Recalculate num_grids to fit the range
            self.num_grids = max(5, int(price_range / spacing))
        else:
            spacing = price_range / self.num_grids

        self.grid_spacing = spacing

        # Use provided current price or calculate center
        reference_price = current_price or (self.upper_price + self.lower_price) / 2
        self.grid_center_price = reference_price

        self.grid_levels = []
        for i in range(self.num_grids + 1):
            price = self.lower_price + (i * spacing)
            size = self._calculate_position_size(i, price)

            # Phase 26: Smart buy/sell assignment based on current price
            # Levels below current price = buy orders
            # Levels above current price = sell orders
            if price < reference_price * 0.998:  # Small buffer
                side = 'buy'
            elif price > reference_price * 1.002:
                side = 'sell'
            else:
                # Price at reference - use buy for accumulation bias
                side = 'buy'

            self.grid_levels.append(GridLevel(
                price=price,
                side=side,
                size=size,
                original_side=side,
                level_index=i
            ))

        self.grid_initialized = True

    def _calculate_position_size(self, level: int, price: float) -> float:
        """
        Calculate position size with RSI-based extreme bias.

        Phase 26: Larger positions when RSI is at extremes.
        """
        if price <= 0:
            return 0

        # Phase 26: Compound-adjusted capital
        effective_capital = self.total_capital
        if self.compound_profits:
            effective_capital += self.stats.profit_reinvested
            effective_capital = min(effective_capital, self.initial_capital * self.max_compound)

        base_size = effective_capital / (self.num_grids * price)

        # Phase 26: RSI extreme multiplier
        # Increase size when RSI is at extremes (better mean reversion opportunity)
        rsi_multiplier = 1.0
        if self.current_rsi < self.rsi_oversold:
            # Very oversold - increase buy size
            rsi_multiplier = self.rsi_extreme_multiplier
        elif self.current_rsi > self.rsi_overbought:
            # Very overbought - increase sell size
            rsi_multiplier = self.rsi_extreme_multiplier
        elif self.current_rsi < self.rsi_oversold + 10:
            # Moderately oversold
            rsi_multiplier = 1.0 + (self.rsi_extreme_multiplier - 1.0) * 0.5
        elif self.current_rsi > self.rsi_overbought - 10:
            # Moderately overbought
            rsi_multiplier = 1.0 + (self.rsi_extreme_multiplier - 1.0) * 0.5

        return base_size * self.leverage * rsi_multiplier

    def _calculate_rsi(self, closes: np.ndarray) -> float:
        """Calculate RSI using Wilder's smoothing method."""
        if len(closes) < self.rsi_period + 1:
            return 50.0

        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        # Use exponential moving average for smoother RSI
        alpha = 1.0 / self.rsi_period
        avg_gain = np.mean(gains[:self.rsi_period])
        avg_loss = np.mean(losses[:self.rsi_period])

        for i in range(self.rsi_period, len(gains)):
            avg_gain = alpha * gains[i] + (1 - alpha) * avg_gain
            avg_loss = alpha * losses[i] + (1 - alpha) * avg_loss

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _calculate_confidence(self, side: str, base_confidence: float = 0.5) -> float:
        """
        Calculate confidence based on RSI position.

        Phase 26: Proper confidence calculation with RSI boost.
        Returns value between 0.0 and 1.0.
        """
        if side == 'buy':
            if self.current_rsi < self.rsi_oversold:
                # Very oversold - high confidence for buy
                boost = min(0.4, (self.rsi_oversold - self.current_rsi) / 50)
                return min(1.0, 0.7 + boost)
            elif self.current_rsi < self.rsi_oversold + 15:
                # Moderately oversold
                boost = (self.rsi_oversold + 15 - self.current_rsi) / 30
                return min(1.0, base_confidence + boost * 0.3)
            else:
                # Neutral or overbought - lower confidence for buy
                return max(0.2, base_confidence - 0.1)
        else:  # sell
            if self.current_rsi > self.rsi_overbought:
                # Very overbought - high confidence for sell
                boost = min(0.4, (self.current_rsi - self.rsi_overbought) / 50)
                return min(1.0, 0.7 + boost)
            elif self.current_rsi > self.rsi_overbought - 15:
                # Moderately overbought
                boost = (self.current_rsi - self.rsi_overbought + 15) / 30
                return min(1.0, base_confidence + boost * 0.3)
            else:
                # Neutral or oversold - lower confidence for sell
                return max(0.2, base_confidence - 0.1)

    def _get_adaptive_rsi_zones(self) -> Tuple[float, float]:
        """
        Get RSI zones adjusted by volatility.

        Phase 26: Wider zones in volatile markets.
        """
        if not self.use_adaptive_rsi or self.current_atr <= 0:
            return self.rsi_oversold, self.rsi_overbought

        # ATR-based zone expansion
        # Higher ATR = wider zones (more tolerance)
        atr_pct = (self.current_atr / self.current_price * 100) if self.current_price > 0 else 0
        expansion = min(self.rsi_zone_expansion, atr_pct * 2)

        adaptive_oversold = max(15, self.rsi_oversold - expansion)
        adaptive_overbought = min(85, self.rsi_overbought + expansion)

        return adaptive_oversold, adaptive_overbought

    def should_recenter(self, current_price: float) -> bool:
        """
        Check if grid should be recentered.

        Phase 26: Multiple criteria for smart recentering.
        """
        if not self.grid_initialized or self.grid_center_price == 0:
            return True

        # Time-based check
        if self.last_recenter_time:
            time_since = (datetime.now() - self.last_recenter_time).total_seconds()
            if time_since < self.min_recenter_interval:
                return False

        # Price deviation check
        if current_price > 0 and self.grid_center_price > 0:
            deviation = abs(current_price - self.grid_center_price) / self.grid_center_price
            # Recenter if price moved beyond grid range
            range_pct = (self.upper_price - self.lower_price) / self.grid_center_price / 2
            if deviation > range_pct * 1.2:  # 20% beyond range
                return True

        # Cycle-based check
        cycles_since = self.stats.cycles_completed - self.cycles_at_last_recenter
        if cycles_since >= self.recenter_after_cycles:
            return True

        return False

    def recenter_grid(self, new_center_price: float):
        """
        Recenter the grid around a new price point.

        Phase 26: Smart recentering that preserves stats.
        """
        # Calculate new range maintaining the same spread
        half_range = (self.upper_price - self.lower_price) / 2

        self.upper_price = new_center_price + half_range
        self.lower_price = new_center_price - half_range
        self.grid_center_price = new_center_price

        # Rebuild grid with new center
        self._setup_grid(new_center_price)

        # Update recentering tracking
        self.last_recenter_time = datetime.now()
        self.cycles_at_last_recenter = self.stats.cycles_completed
        self.pending_rebalance = False

    def update(self, current_price: float, data: pd.DataFrame = None) -> List[Dict[str, Any]]:
        """
        Update grid state with current price and RSI.

        Phase 26 Enhanced:
        - ATR-based dynamic spacing
        - RSI as confidence modifier (not hard filter)
        - Smart recentering
        - Slippage tolerance
        """
        self.current_price = current_price
        signals = []

        # Update ATR if data provided
        if data is not None and self.use_atr_spacing:
            new_atr = self.calculate_atr(data)
            if new_atr > 0:
                self.current_atr = new_atr
                self.last_atr_update = datetime.now()

        # Calculate RSI if data provided
        if data is not None and 'close' in data.columns:
            self.current_rsi = self._calculate_rsi(data['close'].values)
            self.rsi_history.append(self.current_rsi)
            if len(self.rsi_history) > 100:
                self.rsi_history = self.rsi_history[-100:]

        # Phase 26: Check for recentering
        if self.should_recenter(current_price):
            self.recenter_grid(current_price)

        # Check stop loss
        if self.stop_loss and current_price < self.stop_loss:
            signals.append(self._create_exit_signal('stop_loss'))
            return signals

        # Check take profit
        if self.take_profit and current_price > self.take_profit:
            signals.append(self._create_exit_signal('take_profit'))
            return signals

        # Check max drawdown
        current_value = self._calculate_portfolio_value()
        drawdown = (self.peak_capital - current_value) / self.peak_capital if self.peak_capital > 0 else 0
        if drawdown > self.max_drawdown:
            signals.append(self._create_exit_signal('max_drawdown'))
            self.stats.max_drawdown = max(self.stats.max_drawdown, drawdown)
            return signals

        # Update peak and track drawdown
        if current_value > self.peak_capital:
            self.peak_capital = current_value
        self.stats.max_drawdown = max(self.stats.max_drawdown, drawdown)

        # Get adaptive RSI zones
        adaptive_oversold, adaptive_overbought = self._get_adaptive_rsi_zones()

        # Phase 26: Check grid levels with slippage tolerance
        for i, level in enumerate(self.grid_levels):
            if not level.filled:
                tolerance = level.price * self.slippage_tolerance

                if level.side == 'buy' and current_price <= level.price + tolerance:
                    # Phase 26: RSI mode handling
                    should_signal = True
                    if self.rsi_mode == 'filter':
                        # Legacy mode: RSI must be below threshold
                        should_signal = self.current_rsi < adaptive_oversold + 15

                    if should_signal:
                        effective_price = min(current_price, level.price)
                        signal = self._create_buy_signal(i, level)
                        signal['effective_price'] = effective_price
                        signal['rsi'] = self.current_rsi
                        signal['confidence'] = self._calculate_confidence('buy')
                        signal['rsi_zone'] = 'oversold' if self.current_rsi < adaptive_oversold else 'neutral'
                        signals.append(signal)

                elif level.side == 'sell' and current_price >= level.price - tolerance:
                    # Phase 26: RSI mode handling
                    should_signal = True
                    if self.rsi_mode == 'filter':
                        # Legacy mode: RSI must be above threshold
                        should_signal = self.current_rsi > adaptive_overbought - 15

                    if should_signal:
                        effective_price = max(current_price, level.price)
                        signal = self._create_sell_signal(i, level)
                        signal['effective_price'] = effective_price
                        signal['rsi'] = self.current_rsi
                        signal['confidence'] = self._calculate_confidence('sell')
                        signal['rsi_zone'] = 'overbought' if self.current_rsi > adaptive_overbought else 'neutral'
                        signals.append(signal)

        # Update unrealized PnL
        self._update_positions_pnl()

        # Phase 26: Check if rebalancing is needed
        if self.compound_profits and self.stats.cycles_completed > 0:
            self._check_profit_rebalance()

        return signals

    def get_status(self) -> Dict[str, Any]:
        """Get status with RSI and Phase 26 metrics."""
        status = super().get_status()

        adaptive_oversold, adaptive_overbought = self._get_adaptive_rsi_zones()

        # Determine RSI zone
        if self.current_rsi < adaptive_oversold:
            rsi_zone = 'oversold'
        elif self.current_rsi > adaptive_overbought:
            rsi_zone = 'overbought'
        else:
            rsi_zone = 'neutral'

        status.update({
            'current_rsi': self.current_rsi,
            'rsi_oversold': self.rsi_oversold,
            'rsi_overbought': self.rsi_overbought,
            'adaptive_oversold': adaptive_oversold,
            'adaptive_overbought': adaptive_overbought,
            'rsi_zone': rsi_zone,
            'rsi_mode': self.rsi_mode,
            'use_adaptive_rsi': self.use_adaptive_rsi,
            'rsi_extreme_multiplier': self.rsi_extreme_multiplier,
            'cycles_since_recenter': self.stats.cycles_completed - self.cycles_at_last_recenter,
        })
        return status


class BBSqueezeGrid(GridBaseStrategy):
    """
    TTM Squeeze Enhanced Grid Strategy.

    Implements the full TTM Squeeze indicator with:
    - Bollinger Bands + Keltner Channel squeeze detection
    - Momentum histogram for directional bias
    - Multi-level squeeze classification (high/mid/low)
    - Proper grid level tracking and filling
    - ATR-based stop losses and position sizing

    Phase 27 Improvements:
    - Fixed grid level filling bug
    - Added Keltner Channel confirmation
    - Added momentum histogram for breakout direction
    - Multiple squeeze levels (TTM Squeeze Pro style)
    - Improved breakout signal handling
    """

    def __init__(self, config: Dict[str, Any]):
        # Bollinger Band parameters
        self.bb_period = config.get('bb_period', 20)
        self.bb_std = config.get('bb_std', 2.0)

        # Keltner Channel parameters (for TTM Squeeze)
        self.kc_period = config.get('kc_period', 20)
        self.kc_atr_mult_high = config.get('kc_atr_mult_high', 1.0)   # Tightest squeeze
        self.kc_atr_mult_mid = config.get('kc_atr_mult_mid', 1.5)    # Standard squeeze
        self.kc_atr_mult_low = config.get('kc_atr_mult_low', 2.0)    # Loose squeeze

        # Momentum parameters
        self.momentum_period = config.get('momentum_period', 12)

        # Squeeze detection parameters
        self.squeeze_min_bars = config.get('squeeze_min_bars', 6)
        self.squeeze_confirmation = config.get('squeeze_confirmation', True)

        # Breakout parameters
        self.breakout_size_pct = config.get('breakout_size_pct', 0.15)  # 15% for breakout
        self.breakout_leverage_mult = config.get('breakout_leverage_mult', 2.0)
        self.max_breakout_leverage = config.get('max_breakout_leverage', 5)

        # Risk management
        self.stop_loss_atr_mult = config.get('stop_loss_atr_mult', 1.5)
        self.take_profit_atr_mult = config.get('take_profit_atr_mult', 3.0)

        # State - Bollinger Bands
        self.bb_upper = 0.0
        self.bb_lower = 0.0
        self.bb_middle = 0.0
        self.bb_width = 0.0
        self.bb_width_percentile = 0.0

        # State - Keltner Channels
        self.kc_upper_high = 0.0
        self.kc_lower_high = 0.0
        self.kc_upper_mid = 0.0
        self.kc_lower_mid = 0.0
        self.kc_upper_low = 0.0
        self.kc_lower_low = 0.0
        self.kc_middle = 0.0

        # State - Momentum
        self.momentum = 0.0
        self.momentum_prev = 0.0
        self.momentum_rising = False
        self.momentum_positive = False
        self.momentum_histogram = []

        # State - Squeeze
        self.squeeze_level = 'none'  # 'none', 'low', 'mid', 'high'
        self.in_squeeze = False
        self.was_in_squeeze = False
        self.squeeze_bars = 0
        self.squeeze_fired = False
        self.squeeze_direction = 'neutral'  # 'long', 'short', 'neutral'

        # State - ATR
        self.current_atr = 0.0

        # Tracking for level fills
        self.last_filled_levels = set()
        self.pending_breakout = None

        # Historical data for percentile calculation
        self.bb_width_history = []
        self.max_history_size = 100

        super().__init__(config)

    def _setup_grid(self, current_price: float = None):
        """Set up dynamic grid based on current price or config."""
        if current_price:
            # Center grid around current price
            range_pct = (self.upper_price - self.lower_price) / ((self.upper_price + self.lower_price) / 2)
            self.upper_price = current_price * (1 + range_pct / 2)
            self.lower_price = current_price * (1 - range_pct / 2)

        price_range = self.upper_price - self.lower_price
        spacing = price_range / self.num_grids

        self.grid_levels = []
        mid_level = self.num_grids // 2

        for i in range(self.num_grids + 1):
            price = self.lower_price + (i * spacing)
            size = self._calculate_position_size(i, price)
            side = 'buy' if i < mid_level else 'sell'

            self.grid_levels.append(GridLevel(
                price=price,
                side=side,
                size=size,
                level_index=i,
                original_side=side
            ))

        self.grid_initialized = True

    def _calculate_position_size(self, level: int, price: float) -> float:
        """Calculate position size with volatility adjustment."""
        base_size = (self.total_capital / self.num_grids) / price

        # Adjust size based on ATR if available (smaller size in high volatility)
        if self.current_atr > 0 and price > 0:
            atr_pct = self.current_atr / price
            # Reduce size when volatility is high (above 2%)
            if atr_pct > 0.02:
                volatility_factor = 0.02 / atr_pct
                base_size *= max(0.5, min(1.0, volatility_factor))

        return base_size

    def _calculate_bollinger_bands(self, closes: np.ndarray) -> Tuple[float, float, float, float]:
        """Calculate Bollinger Bands with width percentile."""
        if len(closes) < self.bb_period:
            return closes[-1], closes[-1], closes[-1], 0.0

        sma = np.mean(closes[-self.bb_period:])
        std = np.std(closes[-self.bb_period:])

        upper = sma + (self.bb_std * std)
        lower = sma - (self.bb_std * std)
        width = (upper - lower) / sma if sma > 0 else 0.0

        # Track BB width history for percentile calculation
        self.bb_width_history.append(width)
        if len(self.bb_width_history) > self.max_history_size:
            self.bb_width_history.pop(0)

        # Calculate percentile (lower = tighter squeeze)
        if len(self.bb_width_history) > 10:
            self.bb_width_percentile = (
                sum(1 for w in self.bb_width_history if w > width) /
                len(self.bb_width_history) * 100
            )

        return upper, lower, sma, width

    def _calculate_keltner_channels(self, data: pd.DataFrame, atr_mult: float) -> Tuple[float, float, float]:
        """Calculate Keltner Channels using EMA and ATR."""
        if data is None or len(data) < self.kc_period:
            return 0.0, 0.0, 0.0

        # EMA for middle line
        ema = data['close'].ewm(span=self.kc_period, adjust=False).mean().iloc[-1]

        # ATR calculation
        if self.current_atr <= 0:
            self.current_atr = self.calculate_atr(data)

        upper = ema + (atr_mult * self.current_atr)
        lower = ema - (atr_mult * self.current_atr)

        return upper, lower, ema

    def _calculate_momentum(self, data: pd.DataFrame) -> Tuple[float, float]:
        """
        Calculate momentum histogram (TTM Squeeze style).
        Uses linear regression of price deviation from midline.
        """
        if data is None or len(data) < self.momentum_period:
            return 0.0, 0.0

        closes = data['close'].values
        highs = data['high'].values
        lows = data['low'].values

        # Calculate donchian midline
        period = self.momentum_period
        highest = np.max(highs[-period:])
        lowest = np.min(lows[-period:])
        donchian_mid = (highest + lowest) / 2

        # SMA of close
        sma = np.mean(closes[-period:])

        # Midline average
        midline = (donchian_mid + sma) / 2

        # Current momentum = close - midline (simplified linear regression)
        current_momentum = closes[-1] - midline

        # Previous momentum for direction
        if len(closes) >= period + 1:
            prev_highest = np.max(highs[-(period+1):-1])
            prev_lowest = np.min(lows[-(period+1):-1])
            prev_donchian_mid = (prev_highest + prev_lowest) / 2
            prev_sma = np.mean(closes[-(period+1):-1])
            prev_midline = (prev_donchian_mid + prev_sma) / 2
            prev_momentum = closes[-2] - prev_midline
        else:
            prev_momentum = current_momentum

        return current_momentum, prev_momentum

    def _detect_squeeze(self) -> dict:
        """
        Detect TTM Squeeze state using BB inside KC.
        Returns squeeze classification and momentum direction.
        """
        squeeze_state = {
            'in_squeeze': False,
            'level': 'none',
            'fired': False,
            'direction': 'neutral',
            'momentum': self.momentum,
            'momentum_rising': self.momentum_rising
        }

        # Check if BB is inside KC at each level
        squeeze_high = (self.bb_lower > self.kc_lower_high) and (self.bb_upper < self.kc_upper_high)
        squeeze_mid = (self.bb_lower > self.kc_lower_mid) and (self.bb_upper < self.kc_upper_mid)
        squeeze_low = (self.bb_lower > self.kc_lower_low) and (self.bb_upper < self.kc_upper_low)

        if squeeze_high:
            squeeze_state['in_squeeze'] = True
            squeeze_state['level'] = 'high'
        elif squeeze_mid:
            squeeze_state['in_squeeze'] = True
            squeeze_state['level'] = 'mid'
        elif squeeze_low:
            squeeze_state['in_squeeze'] = True
            squeeze_state['level'] = 'low'

        # Determine momentum direction
        if self.momentum > 0 and self.momentum_rising:
            squeeze_state['direction'] = 'long'
        elif self.momentum < 0 and not self.momentum_rising:
            squeeze_state['direction'] = 'short'

        # Check if squeeze just fired (was in squeeze, now released)
        if self.was_in_squeeze and not squeeze_state['in_squeeze']:
            squeeze_state['fired'] = True

        return squeeze_state

    def update(self, current_price: float, data: pd.DataFrame = None) -> List[Dict[str, Any]]:
        """
        Update with TTM Squeeze detection and proper grid management.

        Phase 27 Enhanced:
        - Full TTM Squeeze detection
        - Momentum-based direction confirmation
        - Proper grid level filling
        - Smart position management
        """
        signals = []
        self.current_price = current_price

        # Update ATR first (needed for KC and position sizing)
        if data is not None:
            self.current_atr = self.calculate_atr(data)

        # Calculate indicators if data available
        if data is not None and 'close' in data.columns and len(data) >= self.bb_period:
            closes = data['close'].values

            # Calculate Bollinger Bands
            self.bb_upper, self.bb_lower, self.bb_middle, self.bb_width = \
                self._calculate_bollinger_bands(closes)

            # Calculate Keltner Channels at 3 levels (TTM Squeeze Pro style)
            self.kc_upper_high, self.kc_lower_high, self.kc_middle = \
                self._calculate_keltner_channels(data, self.kc_atr_mult_high)
            self.kc_upper_mid, self.kc_lower_mid, _ = \
                self._calculate_keltner_channels(data, self.kc_atr_mult_mid)
            self.kc_upper_low, self.kc_lower_low, _ = \
                self._calculate_keltner_channels(data, self.kc_atr_mult_low)

            # Calculate momentum
            self.momentum_prev = self.momentum
            self.momentum, prev_momentum = self._calculate_momentum(data)
            self.momentum_rising = self.momentum > prev_momentum
            self.momentum_positive = self.momentum > 0

            # Store for histogram visualization
            self.momentum_histogram.append({
                'value': self.momentum,
                'rising': self.momentum_rising,
                'positive': self.momentum_positive
            })
            if len(self.momentum_histogram) > 50:
                self.momentum_histogram.pop(0)

            # Detect squeeze state
            squeeze_state = self._detect_squeeze()

            # Update squeeze tracking
            self.was_in_squeeze = self.in_squeeze
            self.in_squeeze = squeeze_state['in_squeeze']
            self.squeeze_level = squeeze_state['level']
            self.squeeze_direction = squeeze_state['direction']

            if self.in_squeeze:
                self.squeeze_bars += 1
            else:
                # Squeeze fired - potential breakout
                if squeeze_state['fired'] and self.squeeze_bars >= self.squeeze_min_bars:
                    self.squeeze_fired = True

                    # Create breakout signal if momentum confirms
                    if self.squeeze_confirmation:
                        if squeeze_state['direction'] in ['long', 'short']:
                            breakout_signal = self._create_breakout_signal(
                                squeeze_state['direction'],
                                squeeze_state
                            )
                            if breakout_signal:
                                signals.append(breakout_signal)
                    else:
                        # Without confirmation, use price position relative to BB
                        if current_price > self.bb_upper:
                            signals.append(self._create_breakout_signal('long', squeeze_state))
                        elif current_price < self.bb_lower:
                            signals.append(self._create_breakout_signal('short', squeeze_state))

                self.squeeze_bars = 0
                self.squeeze_fired = False

        # Check stop/take profit first
        risk_signals = self._check_risk_limits(current_price)
        if risk_signals:
            return risk_signals

        # Normal grid operation with proper level tracking
        grid_signals = self._process_grid_levels(current_price)
        signals.extend(grid_signals)

        # Update position PnL
        self._update_positions_pnl()

        return signals

    def _process_grid_levels(self, current_price: float) -> List[Dict[str, Any]]:
        """
        Process grid levels with proper filling and state management.

        Fixes the repeated signal bug by:
        1. Marking levels as filled immediately when signal generated
        2. Tracking which levels were recently filled
        3. Only generating one signal per level crossing
        """
        signals = []

        for i, level in enumerate(self.grid_levels):
            if level.filled:
                continue

            # Skip if this level was just filled
            if i in self.last_filled_levels:
                continue

            signal = None

            if level.side == 'buy':
                # Check if price crossed below buy level
                if current_price <= level.price * (1 + self.slippage_tolerance):
                    signal = self._create_grid_buy_signal(i, level)

            elif level.side == 'sell':
                # Check if price crossed above sell level
                if current_price >= level.price * (1 - self.slippage_tolerance):
                    signal = self._create_grid_sell_signal(i, level)

            if signal:
                # Mark level as filled IMMEDIATELY
                level.filled = True
                level.fill_price = current_price
                level.fill_time = datetime.now()

                # Track this fill to prevent duplicate signals
                self.last_filled_levels.add(i)

                # Keep track limited
                if len(self.last_filled_levels) > self.num_grids:
                    self.last_filled_levels = set(list(self.last_filled_levels)[-self.num_grids:])

                signals.append(signal)

                # Activate corresponding opposite level
                if level.side == 'buy':
                    self._activate_sell_level(i)
                else:
                    self._reactivate_buy_level(i)

        return signals

    def _create_grid_buy_signal(self, level_idx: int, level: GridLevel) -> Dict[str, Any]:
        """Create a buy signal with squeeze context."""
        # Adjust confidence based on squeeze state
        confidence = 0.7
        if self.in_squeeze:
            confidence = 0.8  # Higher confidence during squeeze
        if self.squeeze_direction == 'long':
            confidence = 0.85

        # Adjust size based on momentum
        size = level.size
        if self.momentum_positive and self.momentum_rising:
            size *= 1.2  # Increase size when momentum favors longs

        return {
            'action': 'buy',
            'symbol': self.symbol,
            'price': level.price,
            'size': size,
            'leverage': self.leverage,
            'grid_level': level_idx,
            'order_id': level.order_id,
            'strategy': self.__class__.__name__,
            'reason': f'Grid buy at level {level_idx} (${level.price:,.2f})',
            'confidence': confidence,
            'target_sell_price': self._get_next_sell_price(level_idx),
            'squeeze_state': {
                'in_squeeze': self.in_squeeze,
                'level': self.squeeze_level,
                'momentum': self.momentum,
                'direction': self.squeeze_direction
            }
        }

    def _create_grid_sell_signal(self, level_idx: int, level: GridLevel) -> Dict[str, Any]:
        """Create a sell signal with squeeze context."""
        # Find corresponding position
        position = self._find_position_for_sell(level_idx)
        size = position.size if position else level.size

        # Adjust confidence based on squeeze state
        confidence = 0.7
        if self.in_squeeze:
            confidence = 0.8
        if self.squeeze_direction == 'short':
            confidence = 0.85

        return {
            'action': 'sell',
            'symbol': self.symbol,
            'price': level.price,
            'size': size,
            'leverage': self.leverage,
            'grid_level': level_idx,
            'order_id': level.order_id,
            'strategy': self.__class__.__name__,
            'reason': f'Grid sell at level {level_idx} (${level.price:,.2f})',
            'confidence': confidence,
            'position_id': position.order_id if position else None,
            'squeeze_state': {
                'in_squeeze': self.in_squeeze,
                'level': self.squeeze_level,
                'momentum': self.momentum,
                'direction': self.squeeze_direction
            }
        }

    def _create_breakout_signal(self, direction: str, squeeze_state: dict) -> Dict[str, Any]:
        """
        Create a breakout signal with momentum confirmation.

        Only triggers when:
        1. Squeeze has fired (BB expanded outside KC)
        2. Squeeze lasted minimum bars
        3. Momentum confirms direction (if confirmation enabled)
        """
        # Calculate position size based on volatility
        if self.current_atr > 0:
            # Size inversely proportional to volatility
            risk_amount = self.total_capital * self.breakout_size_pct
            stop_distance = self.current_atr * self.stop_loss_atr_mult
            size = risk_amount / stop_distance
        else:
            size = (self.total_capital * self.breakout_size_pct) / self.current_price

        # Calculate leverage
        leverage = min(
            int(self.leverage * self.breakout_leverage_mult),
            self.max_breakout_leverage
        )

        # Calculate stop loss and take profit
        if direction == 'long':
            stop_loss = self.current_price - (self.current_atr * self.stop_loss_atr_mult)
            take_profit = self.current_price + (self.current_atr * self.take_profit_atr_mult)
        else:
            stop_loss = self.current_price + (self.current_atr * self.stop_loss_atr_mult)
            take_profit = self.current_price - (self.current_atr * self.take_profit_atr_mult)

        # Confidence based on squeeze strength and momentum alignment
        confidence = 0.75
        if squeeze_state['level'] == 'high':
            confidence += 0.1  # Tighter squeeze = stronger breakout
        if squeeze_state['momentum_rising'] == (direction == 'long'):
            confidence += 0.1  # Momentum confirms direction

        confidence = min(0.95, confidence)

        return {
            'action': 'buy' if direction == 'long' else 'short',
            'symbol': self.symbol,
            'price': self.current_price,
            'size': size,
            'leverage': leverage,
            'strategy': self.__class__.__name__,
            'reason': f'TTM Squeeze breakout {direction} (level: {squeeze_state["level"]}, bars: {self.squeeze_bars})',
            'confidence': confidence,
            'breakout': True,
            'squeeze_bars': self.squeeze_bars,
            'squeeze_level': squeeze_state['level'],
            'momentum': squeeze_state['momentum'],
            'momentum_direction': squeeze_state['direction'],
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'atr': self.current_atr
        }

    def _check_risk_limits(self, current_price: float) -> List[Dict[str, Any]]:
        """Check stop loss, take profit, and max drawdown."""
        signals = []

        # Check stop loss
        if self.stop_loss and current_price < self.stop_loss:
            signals.append(self._create_exit_signal('stop_loss'))
            return signals

        # Check take profit
        if self.take_profit and current_price > self.take_profit:
            signals.append(self._create_exit_signal('take_profit'))
            return signals

        # Check max drawdown
        current_value = self._calculate_portfolio_value()
        if self.peak_capital > 0:
            drawdown = (self.peak_capital - current_value) / self.peak_capital
            if drawdown > self.max_drawdown:
                signals.append(self._create_exit_signal(f'max_drawdown ({drawdown:.1%})'))
                self.stats.max_drawdown = max(self.stats.max_drawdown, drawdown)
                return signals

        # Update peak
        if current_value > self.peak_capital:
            self.peak_capital = current_value

        return signals

    def _calculate_portfolio_value(self) -> float:
        """Calculate current portfolio value including open positions."""
        value = self.total_capital
        for pos in self.positions:
            if pos.side == 'long':
                pnl = (self.current_price - pos.entry_price) * pos.size * self.leverage
            else:
                pnl = (pos.entry_price - self.current_price) * pos.size * self.leverage
            value += pnl
        return value

    def _adjust_grid_for_squeeze(self):
        """
        Tighten grid during squeeze WITHOUT resetting fills.
        Only adjusts unfilled levels to concentrate around BB bands.
        """
        if self.bb_upper <= 0 or self.bb_lower <= 0:
            return

        # Only adjust unfilled levels
        new_upper = self.bb_upper * 1.005
        new_lower = self.bb_lower * 0.995
        new_spacing = (new_upper - new_lower) / self.num_grids

        for i, level in enumerate(self.grid_levels):
            if not level.filled:  # IMPORTANT: Don't reset filled levels
                new_price = new_lower + (i * new_spacing)
                level.price = new_price
                level.size = self._calculate_position_size(i, new_price)

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status with TTM Squeeze info."""
        status = super().get_status()
        status.update({
            # Bollinger Bands
            'bb_upper': self.bb_upper,
            'bb_lower': self.bb_lower,
            'bb_middle': self.bb_middle,
            'bb_width': self.bb_width,
            'bb_width_percentile': self.bb_width_percentile,

            # Keltner Channels
            'kc_upper_high': self.kc_upper_high,
            'kc_lower_high': self.kc_lower_high,
            'kc_upper_mid': self.kc_upper_mid,
            'kc_lower_mid': self.kc_lower_mid,
            'kc_middle': self.kc_middle,

            # Momentum
            'momentum': self.momentum,
            'momentum_rising': self.momentum_rising,
            'momentum_positive': self.momentum_positive,

            # Squeeze State
            'in_squeeze': self.in_squeeze,
            'squeeze_level': self.squeeze_level,
            'squeeze_bars': self.squeeze_bars,
            'squeeze_direction': self.squeeze_direction,
            'squeeze_fired': self.squeeze_fired,

            # ATR
            'atr': self.current_atr,

            # Grid State
            'filled_levels': sum(1 for l in self.grid_levels if l.filled),
            'unfilled_levels': sum(1 for l in self.grid_levels if not l.filled)
        })
        return status


# =============================================================================
# MARGIN STRATEGIES
# =============================================================================

@dataclass
class MarginPosition:
    """Tracks a leveraged margin position."""
    entry_price: float
    size: float
    side: str  # 'long' or 'short'
    leverage: int
    entry_time: datetime
    stop_loss: float
    take_profit: float
    trailing_stop: float = 0.0
    highest_price: float = 0.0  # For trailing stop
    lowest_price: float = 0.0   # For trailing stop on shorts
    unrealized_pnl: float = 0.0
    margin_used: float = 0.0


class TrendFollowingMargin:
    """
    Phase 27 Enhanced: Trend-Following Margin Grid Strategy

    A true grid strategy that uses dynamically calculated trendlines from market data
    and creates multiple entry levels along the trend for margin trading.

    Key Improvements:
    - Dynamic trendline calculation from swing lows (not static)
    - ATR-based stop loss and take profit levels
    - Multiple grid entry levels along the trendline
    - Relaxed entry conditions (RSI OR trendline proximity)
    - Support for both long and short positions
    - Volatility-adjusted leverage
    - Multi-timeframe trend confirmation

    Entry Conditions (any of):
    - Price touches dynamic trendline within tolerance AND RSI < threshold
    - Price at grid level with RSI oversold (< rsi_oversold)
    - Strong bounce from trendline with volume confirmation

    Exit: ATR-based trailing stop, take profit targets, or trend break
    """

    def __init__(self, config: Dict[str, Any]):
        self.symbol = config.get('symbol', 'BTC/USD')
        self.base_leverage = config.get('leverage', 3)  # Reduced default for safety
        self.leverage = self.base_leverage
        self.total_capital = config.get('total_capital', 2000)
        self.initial_capital = self.total_capital
        self.size_pct = config.get('size_pct', 0.08)  # 8% per trade

        # Phase 27: Dynamic trendline parameters
        self.use_dynamic_trendline = config.get('use_dynamic_trendline', True)
        self.trendline_lookback = config.get('trendline_lookback', 72)  # Hours
        self.trendline_recalc_interval = config.get('trendline_recalc_interval', 4)  # Hours
        self.min_swing_points = config.get('min_swing_points', 3)  # Min points for trendline

        # Fallback static trendline (only used if dynamic fails)
        self.trendline_start_price = config.get('trendline_start_price', 0)  # 0 = auto-calibrate
        self.trendline_slope = config.get('trendline_slope', 0)  # 0 = calculate from data
        self.trendline_start_time = datetime.now()

        # Phase 27: Grid parameters along trendline
        self.num_grid_levels = config.get('num_grid_levels', 5)  # Entry levels
        self.grid_spacing_pct = config.get('grid_spacing_pct', 0.015)  # 1.5% between levels
        self.max_positions = config.get('max_positions', 3)  # Max concurrent positions

        # Phase 27: Relaxed entry conditions
        self.entry_tolerance = config.get('entry_tolerance', 0.01)  # 1% (relaxed from 0.3%)
        self.rsi_threshold = config.get('rsi_threshold', 40)  # Relaxed from 35
        self.rsi_oversold = config.get('rsi_oversold', 35)
        self.rsi_overbought = config.get('rsi_overbought', 65)
        self.entry_mode = config.get('entry_mode', 'or')  # 'or' = RSI OR trendline, 'and' = both required

        # Phase 27: ATR-based risk management
        self.use_atr_stops = config.get('use_atr_stops', True)
        self.atr_period = config.get('atr_period', 14)
        self.stop_loss_atr_mult = config.get('stop_loss_atr_mult', 2.0)  # 2x ATR stop
        self.take_profit_atr_mult = config.get('take_profit_atr_mult', 3.0)  # 3x ATR TP

        # Fallback fixed percentages (used if ATR unavailable)
        self.stop_loss_pct = config.get('stop_loss_pct', 0.02)  # 2% (increased from 0.5%)
        self.take_profit_1 = config.get('take_profit_1', 0.025)  # 2.5% first target
        self.take_profit_2 = config.get('take_profit_2', 0.04)  # 4% second target
        self.trailing_stop_pct = config.get('trailing_stop_pct', 0.012)  # 1.2%

        # Phase 27: Volatility-adjusted leverage
        self.use_dynamic_leverage = config.get('use_dynamic_leverage', True)
        self.max_leverage = config.get('max_leverage', 5)
        self.min_leverage = config.get('min_leverage', 1)
        self.high_vol_threshold = config.get('high_vol_threshold', 0.03)  # 3% ATR = high vol
        self.low_vol_threshold = config.get('low_vol_threshold', 0.015)  # 1.5% ATR = low vol

        # Phase 27: Trend direction and short support
        self.allow_shorts = config.get('allow_shorts', True)
        self.trend_direction = 'up'  # 'up', 'down', 'sideways'

        # Phase 27: Fee and slippage
        self.fee_rate = config.get('fee_rate', 0.001)  # 0.1%
        self.slippage_tolerance = config.get('slippage_tolerance', 0.005)  # 0.5% for live trading

        # State
        self.positions: List[MarginPosition] = []
        self.grid_levels: List[float] = []  # Entry price levels
        self.current_price = 0.0
        self.current_rsi = 50.0
        self.current_atr = 0.0
        self.current_atr_pct = 0.0
        self.dynamic_trendline = 0.0
        self.trendline_slope_calculated = 0.0
        self.last_trendline_calc = None
        self.swing_lows: List[Tuple[datetime, float]] = []

        # Statistics
        self.total_pnl = 0.0
        self.realized_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_fees = 0.0
        self.stats = GridStats()  # Use GridStats for consistency

        # Grid levels will be initialized on first update with price data
        self.initialized = False

    def _calculate_swing_lows(self, data: pd.DataFrame, lookback: int = 72) -> List[Tuple[datetime, float]]:
        """
        Calculate swing lows from price data for dynamic trendline.

        A swing low is a candle where the low is lower than the lows of
        the candles on either side.
        """
        if data is None or len(data) < lookback:
            return []

        swing_lows = []
        lows = data['low'].iloc[-lookback:].values
        times = data.index[-lookback:] if hasattr(data.index, '__iter__') else range(len(lows))

        # Find swing lows (local minima)
        for i in range(2, len(lows) - 2):
            if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and
                lows[i] < lows[i+1] and lows[i] < lows[i+2]):
                try:
                    time_val = times[i] if hasattr(times[i], 'timestamp') else datetime.now()
                    swing_lows.append((time_val, lows[i]))
                except:
                    swing_lows.append((datetime.now(), lows[i]))

        return swing_lows

    def _calculate_dynamic_trendline(self, data: pd.DataFrame) -> Tuple[float, float]:
        """
        Calculate dynamic trendline from swing lows using linear regression.

        Returns:
            Tuple of (current_trendline_price, slope_per_hour)
        """
        swing_lows = self._calculate_swing_lows(data, self.trendline_lookback)

        if len(swing_lows) < self.min_swing_points:
            # Not enough swing points - use current price as base
            return self.current_price * 0.98, 0  # 2% below current price

        # Use last N swing lows for trendline
        recent_lows = swing_lows[-self.min_swing_points:]

        # Simple linear regression on swing lows
        prices = [sl[1] for sl in recent_lows]

        # Calculate slope (price change per point)
        if len(prices) >= 2:
            slope = (prices[-1] - prices[0]) / max(len(prices) - 1, 1)
            # Extrapolate to current time
            current_trendline = prices[-1] + slope * 2  # Project 2 periods ahead
        else:
            current_trendline = prices[-1] if prices else self.current_price * 0.98
            slope = 0

        self.swing_lows = swing_lows
        return current_trendline, slope

    def _calculate_atr(self, data: pd.DataFrame) -> float:
        """Calculate ATR from OHLCV data."""
        if data is None or len(data) < self.atr_period + 1:
            return 0.0

        high = data['high'].iloc[-self.atr_period:]
        low = data['low'].iloc[-self.atr_period:]
        close = data['close'].iloc[-self.atr_period:]

        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.mean()

    def _calculate_rsi(self, closes: np.ndarray, period: int = 14) -> float:
        """Calculate RSI."""
        if len(closes) < period + 1:
            return 50.0
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _detect_trend(self, data: pd.DataFrame) -> str:
        """Detect current trend direction from price data."""
        if data is None or len(data) < 50:
            return 'sideways'

        close = data['close'].iloc[-50:]
        sma_20 = close.rolling(20).mean().iloc[-1]
        sma_50 = close.rolling(50).mean().iloc[-1]
        current = close.iloc[-1]

        if current > sma_20 > sma_50:
            return 'up'
        elif current < sma_20 < sma_50:
            return 'down'
        return 'sideways'

    def _calculate_dynamic_leverage(self) -> int:
        """Calculate leverage based on current volatility."""
        if not self.use_dynamic_leverage or self.current_atr_pct == 0:
            return self.base_leverage

        # High volatility = lower leverage
        if self.current_atr_pct > self.high_vol_threshold:
            return self.min_leverage
        elif self.current_atr_pct < self.low_vol_threshold:
            return self.max_leverage
        else:
            # Linear interpolation
            vol_range = self.high_vol_threshold - self.low_vol_threshold
            vol_position = (self.current_atr_pct - self.low_vol_threshold) / vol_range
            leverage_range = self.max_leverage - self.min_leverage
            return int(self.max_leverage - (vol_position * leverage_range))

    def _setup_grid_levels(self, base_price: float):
        """Set up grid entry levels around the trendline."""
        self.grid_levels = []

        for i in range(self.num_grid_levels):
            # Grid levels below current price (buy zones)
            level = base_price * (1 - (i + 1) * self.grid_spacing_pct)
            self.grid_levels.append(level)

        self.initialized = True

    def _get_trendline_price(self) -> float:
        """Get current trendline price (dynamic or static)."""
        if self.use_dynamic_trendline and self.dynamic_trendline > 0:
            return self.dynamic_trendline

        # Fallback to static calculation
        if self.trendline_start_price == 0:
            # Auto-calibrate to current price
            return self.current_price * 0.98  # 2% below

        hours_elapsed = (datetime.now() - self.trendline_start_time).total_seconds() / 3600
        return self.trendline_start_price + (self.trendline_slope * hours_elapsed)

    def _check_entry_conditions(self, trendline: float) -> Tuple[bool, str, float]:
        """
        Check if entry conditions are met.

        Returns:
            Tuple of (should_enter, reason, confidence)
        """
        distance_pct = abs(self.current_price - trendline) / trendline
        near_trendline = distance_pct < self.entry_tolerance
        rsi_oversold = self.current_rsi < self.rsi_threshold

        confidence = 0.0
        reasons = []

        if self.entry_mode == 'or':
            # Relaxed mode: RSI OR trendline proximity
            if near_trendline:
                confidence += 0.4
                reasons.append(f'Near trendline ({distance_pct*100:.1f}%)')

            if rsi_oversold:
                confidence += 0.4
                reasons.append(f'RSI oversold ({self.current_rsi:.1f})')

            # Bonus confidence for both conditions
            if near_trendline and rsi_oversold:
                confidence += 0.2

            should_enter = near_trendline or rsi_oversold
        else:
            # Strict mode: both required
            should_enter = near_trendline and rsi_oversold
            if should_enter:
                confidence = 0.8
                reasons = [f'Trendline+RSI ({distance_pct*100:.1f}%, RSI={self.current_rsi:.1f})']

        # Check grid levels for additional entry opportunities
        for level in self.grid_levels:
            level_distance = abs(self.current_price - level) / level
            if level_distance < 0.005:  # Within 0.5% of grid level
                confidence += 0.2
                reasons.append(f'At grid level ${level:,.0f}')
                should_enter = True
                break

        reason = ' + '.join(reasons) if reasons else 'No entry signal'
        return should_enter, reason, min(confidence, 0.95)

    def update(self, current_price: float, data: pd.DataFrame = None) -> List[Dict[str, Any]]:
        """
        Update strategy with current price and generate signals.

        Phase 27 Enhanced with:
        - Dynamic trendline recalculation
        - ATR-based parameters
        - Multiple entry level detection
        - Trend-aware position management
        """
        self.current_price = current_price
        signals = []

        # Calculate indicators if data provided
        if data is not None and 'close' in data.columns:
            self.current_rsi = self._calculate_rsi(data['close'].values)

            # Calculate ATR
            self.current_atr = self._calculate_atr(data)
            if self.current_atr > 0:
                self.current_atr_pct = self.current_atr / current_price

            # Update dynamic leverage
            self.leverage = self._calculate_dynamic_leverage()

            # Detect trend direction
            self.trend_direction = self._detect_trend(data)

            # Recalculate dynamic trendline periodically
            should_recalc = (
                self.last_trendline_calc is None or
                (datetime.now() - self.last_trendline_calc).total_seconds() > self.trendline_recalc_interval * 3600
            )

            if self.use_dynamic_trendline and should_recalc:
                self.dynamic_trendline, self.trendline_slope_calculated = self._calculate_dynamic_trendline(data)
                self.last_trendline_calc = datetime.now()

        # Initialize grid levels if not done
        if not self.initialized:
            trendline = self._get_trendline_price()
            self._setup_grid_levels(trendline)

        # Get current trendline
        trendline = self._get_trendline_price()

        # Check for entry signals
        if len(self.positions) < self.max_positions:
            should_enter, reason, confidence = self._check_entry_conditions(trendline)

            if should_enter and confidence > 0.3:
                # Determine direction based on trend
                if self.trend_direction == 'up' or (self.trend_direction == 'sideways' and self.current_rsi < 50):
                    signals.append(self._create_entry_signal(trendline, 'long', reason, confidence))
                elif self.trend_direction == 'down' and self.allow_shorts:
                    signals.append(self._create_entry_signal(trendline, 'short', reason, confidence))

        # Manage existing positions
        for pos in self.positions[:]:
            signal = self._manage_position(pos)
            if signal:
                signals.append(signal)

        return signals

    def _create_entry_signal(self, trendline: float, side: str, reason: str, confidence: float) -> Dict[str, Any]:
        """Create an entry signal with ATR-based stops."""
        size_usd = self.total_capital * self.size_pct
        size_asset = size_usd / self.current_price

        # ATR-based or fixed stops
        if self.use_atr_stops and self.current_atr > 0:
            if side == 'long':
                stop_loss = self.current_price - (self.current_atr * self.stop_loss_atr_mult)
                take_profit = self.current_price + (self.current_atr * self.take_profit_atr_mult)
            else:  # short
                stop_loss = self.current_price + (self.current_atr * self.stop_loss_atr_mult)
                take_profit = self.current_price - (self.current_atr * self.take_profit_atr_mult)
        else:
            if side == 'long':
                stop_loss = trendline * (1 - self.stop_loss_pct)
                take_profit = self.current_price * (1 + self.take_profit_1)
            else:  # short
                stop_loss = self.current_price * (1 + self.stop_loss_pct)
                take_profit = self.current_price * (1 - self.take_profit_1)

        action = 'buy' if side == 'long' else 'short'

        return {
            'action': action,
            'symbol': self.symbol,
            'price': self.current_price,
            'size': size_asset,
            'size_usd': size_usd,
            'leverage': self.leverage,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'strategy': 'TrendFollowingMargin',
            'reason': f'{reason} | Trend={self.trend_direction}, Lev={self.leverage}x',
            'confidence': confidence,
            'side': side,
            'trendline': trendline,
            'atr': self.current_atr,
            'rsi': self.current_rsi
        }

    def _manage_position(self, pos: MarginPosition) -> Optional[Dict[str, Any]]:
        """Manage an open position with dynamic trailing stops."""
        # Update extreme prices for trailing stop
        if pos.side == 'long':
            if self.current_price > pos.highest_price:
                pos.highest_price = self.current_price
            pos.unrealized_pnl = (self.current_price - pos.entry_price) * pos.size * pos.leverage
        else:  # short
            if self.current_price < pos.lowest_price or pos.lowest_price == 0:
                pos.lowest_price = self.current_price
            pos.unrealized_pnl = (pos.entry_price - self.current_price) * pos.size * pos.leverage

        # Check stop loss
        if pos.side == 'long' and self.current_price <= pos.stop_loss:
            return self._create_exit_signal(pos, 'stop_loss')
        elif pos.side == 'short' and self.current_price >= pos.stop_loss:
            return self._create_exit_signal(pos, 'stop_loss')

        # Check take profit and activate trailing stop
        if pos.side == 'long' and self.current_price >= pos.take_profit:
            if pos.trailing_stop == 0:
                # First TP hit - activate trailing stop
                pos.trailing_stop = pos.highest_price * (1 - self.trailing_stop_pct)
                pos.take_profit = pos.entry_price * (1 + self.take_profit_2)
            else:
                # Update trailing stop
                new_trail = pos.highest_price * (1 - self.trailing_stop_pct)
                pos.trailing_stop = max(pos.trailing_stop, new_trail)

        elif pos.side == 'short' and self.current_price <= pos.take_profit:
            if pos.trailing_stop == 0:
                pos.trailing_stop = pos.lowest_price * (1 + self.trailing_stop_pct)
                pos.take_profit = pos.entry_price * (1 - self.take_profit_2)
            else:
                new_trail = pos.lowest_price * (1 + self.trailing_stop_pct)
                pos.trailing_stop = min(pos.trailing_stop, new_trail)

        # Check trailing stop
        if pos.trailing_stop > 0:
            if pos.side == 'long' and self.current_price <= pos.trailing_stop:
                return self._create_exit_signal(pos, 'trailing_stop')
            elif pos.side == 'short' and self.current_price >= pos.trailing_stop:
                return self._create_exit_signal(pos, 'trailing_stop')

        return None

    def _create_exit_signal(self, pos: MarginPosition, reason: str) -> Dict[str, Any]:
        """Create an exit signal."""
        pnl = pos.unrealized_pnl
        action = 'close' if pos.side == 'long' else 'cover'

        return {
            'action': action,
            'symbol': self.symbol,
            'price': self.current_price,
            'size': pos.size,
            'leverage': pos.leverage,
            'pnl': pnl,
            'strategy': 'TrendFollowingMargin',
            'reason': f'Exit: {reason} (PnL: ${pnl:.2f})',
            'confidence': 0.9,
            'position': pos,
            'side': pos.side
        }

    def fill_order(self, signal: Dict[str, Any], fill_price: float):
        """Process a filled order."""
        action = signal.get('action', '')

        if action in ['buy', 'short']:
            side = 'long' if action == 'buy' else 'short'
            pos = MarginPosition(
                entry_price=fill_price,
                size=signal['size'],
                side=side,
                leverage=signal['leverage'],
                entry_time=datetime.now(),
                stop_loss=signal['stop_loss'],
                take_profit=signal['take_profit'],
                highest_price=fill_price if side == 'long' else 0,
                lowest_price=fill_price if side == 'short' else 0,
                margin_used=signal.get('size_usd', signal['size'] * fill_price) / signal['leverage']
            )
            self.positions.append(pos)
            self.total_trades += 1
            self.stats.total_trades += 1

            # Track entry fee
            fee = signal['size'] * fill_price * self.fee_rate
            self.total_fees += fee
            self.stats.total_fees += fee

        elif action in ['close', 'cover']:
            pos = signal.get('position')
            if pos and pos in self.positions:
                self.positions.remove(pos)
                pnl = signal.get('pnl', 0)

                # Track exit fee
                fee = pos.size * fill_price * self.fee_rate
                self.total_fees += fee
                self.stats.total_fees += fee

                # Net PnL after fees
                net_pnl = pnl - fee
                self.total_pnl += net_pnl
                self.realized_pnl += net_pnl
                self.stats.total_pnl += net_pnl
                self.stats.realized_pnl += net_pnl

                if net_pnl > 0:
                    self.winning_trades += 1
                    self.stats.winning_trades += 1
                else:
                    self.losing_trades += 1
                    self.stats.losing_trades += 1

                self.stats.cycles_completed += 1

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive strategy status."""
        trendline = self._get_trendline_price()
        distance_to_trend = (self.current_price - trendline) / trendline if trendline > 0 else 0

        return {
            'strategy': 'TrendFollowingMargin',
            'leverage': self.leverage,
            'base_leverage': self.base_leverage,
            'current_price': self.current_price,
            'trendline': trendline,
            'trendline_type': 'dynamic' if self.use_dynamic_trendline else 'static',
            'trendline_slope': self.trendline_slope_calculated,
            'distance_to_trendline_pct': distance_to_trend * 100,
            'current_rsi': self.current_rsi,
            'current_atr': self.current_atr,
            'current_atr_pct': self.current_atr_pct * 100,
            'trend_direction': self.trend_direction,
            'grid_levels': self.grid_levels,
            'swing_lows_count': len(self.swing_lows),
            'open_positions': len(self.positions),
            'max_positions': self.max_positions,
            'total_pnl': self.total_pnl,
            'realized_pnl': self.realized_pnl,
            'total_fees': self.total_fees,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.winning_trades / max(self.total_trades, 1),
            'unrealized_pnl': sum(p.unrealized_pnl for p in self.positions),
            'entry_mode': self.entry_mode,
            'use_atr_stops': self.use_atr_stops
        }


@dataclass
class DualGridHedgeStats:
    """Statistics for DualGridHedge strategy."""
    # Overall stats
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    realized_pnl: float = 0.0
    max_drawdown: float = 0.0
    peak_capital: float = 0.0

    # Grid component stats
    grid_trades: int = 0
    grid_pnl: float = 0.0
    grid_cycles: int = 0

    # Hedge component stats
    hedge_trades: int = 0
    hedge_winning: int = 0
    hedge_losing: int = 0
    hedge_pnl: float = 0.0
    false_breakouts: int = 0  # Hedges that hit stop loss quickly

    # Net exposure tracking
    net_delta: float = 0.0  # Positive = net long, negative = net short

    # Recentering stats
    recenters: int = 0
    last_recenter_time: Optional[datetime] = None


class DualGridHedge:
    """
    Phase 28: Enhanced Dual-Grid with Margin Hedge

    A sophisticated strategy combining spot grid trading with dynamic margin hedging
    for breakout protection. Improvements over original:

    1. ATR-based dynamic hedge triggers (not static %)
    2. Correct stop loss placement (below entry for longs, above for shorts)
    3. Trailing stops for hedges to capture extended moves
    4. Volatility-adjusted position sizing
    5. Net delta exposure tracking
    6. Grid recentering after hedge closes
    7. Session-aware trigger adjustments
    8. Compound profit reinvestment
    9. Comprehensive statistics tracking

    Capital Allocation:
    - Main grid: 80% (spot trading within range)
    - Hedge reserve: 20% (margin positions on breakouts)

    Entry Logic:
    - Long hedge: Price breaks above grid by ATR-adjusted threshold
    - Short hedge: Price breaks below grid by ATR-adjusted threshold

    Risk Management:
    - ATR-based stop losses (2x ATR default)
    - Trailing stops activated after 1.5x ATR profit
    - Max 1 hedge position at a time
    - Grid recenter after hedge close to realign with new price level
    """

    def __init__(self, config: Dict[str, Any]):
        self.symbol = config.get('symbol', 'BTC/USD')
        self.total_capital = config.get('total_capital', 10000)
        self.initial_capital = self.total_capital

        # Capital allocation (configurable)
        self.grid_allocation = config.get('grid_allocation', 0.80)
        self.hedge_allocation = config.get('hedge_allocation', 0.20)
        self.grid_capital = self.total_capital * self.grid_allocation
        self.hedge_capital = self.total_capital * self.hedge_allocation

        # Grid parameters
        self.upper_price = config.get('upper_price', 100000)
        self.lower_price = config.get('lower_price', 94000)
        self.num_grids = config.get('num_grids', 18)
        self.fee_rate = config.get('fee_rate', 0.001)

        # ===== PHASE 28: ATR-Based Dynamic Hedge Triggers =====
        self.use_atr_triggers = config.get('use_atr_triggers', True)
        self.atr_period = config.get('atr_period', 14)
        self.atr_trigger_mult = config.get('atr_trigger_mult', 1.5)  # Trigger at 1.5x ATR beyond grid
        self.current_atr = 0.0

        # Fallback static triggers (used if ATR not available)
        self.static_trigger_pct = config.get('static_trigger_pct', 0.015)  # 1.5% default
        self._base_long_trigger = config.get('long_hedge_trigger', 0)  # 0 = auto-calculate
        self._base_short_trigger = config.get('short_hedge_trigger', 0)

        # ===== PHASE 28: Hedge Parameters =====
        self.hedge_leverage = config.get('hedge_leverage', 2)
        self.max_hedge_leverage = config.get('max_hedge_leverage', 3)
        self.min_hedge_leverage = config.get('min_hedge_leverage', 1)

        # ===== PHASE 28: Correct Stop Loss/Take Profit =====
        self.stop_loss_atr_mult = config.get('stop_loss_atr_mult', 2.0)  # 2x ATR stop
        self.take_profit_atr_mult = config.get('take_profit_atr_mult', 3.0)  # 3x ATR target
        self.stop_loss_pct = config.get('stop_loss_pct', 0.02)  # Fallback: 2%
        self.take_profit_pct = config.get('take_profit_pct', 0.03)  # Fallback: 3%

        # ===== PHASE 28: Trailing Stop =====
        self.use_trailing_stop = config.get('use_trailing_stop', True)
        self.trailing_activation_atr = config.get('trailing_activation_atr', 1.5)  # Activate after 1.5x ATR profit
        self.trailing_distance_atr = config.get('trailing_distance_atr', 1.0)  # Trail by 1x ATR

        # ===== PHASE 28: Volatility-Adjusted Sizing =====
        self.use_vol_sizing = config.get('use_vol_sizing', True)
        self.base_size_pct = config.get('base_size_pct', 0.5)  # 50% of hedge capital
        self.min_size_pct = config.get('min_size_pct', 0.25)  # Min 25%
        self.max_size_pct = config.get('max_size_pct', 0.75)  # Max 75%
        self.target_vol = config.get('target_vol', 0.02)  # Target 2% volatility

        # ===== PHASE 28: Grid Recentering =====
        self.recenter_after_hedge = config.get('recenter_after_hedge', True)
        self.recenter_threshold = config.get('recenter_threshold', 0.03)  # 3% price move
        self.min_recenter_interval = config.get('min_recenter_interval', 3600)  # 1 hour min

        # ===== PHASE 28: Session Awareness =====
        self.use_session_adjustment = config.get('use_session_adjustment', True)
        self.session_multipliers = config.get('session_multipliers', {
            'asia': 0.8,       # 00:00-08:00 UTC - tighter triggers
            'europe': 1.0,    # 08:00-14:00 UTC - standard
            'us': 1.2,        # 14:00-21:00 UTC - wider triggers (more volatile)
            'overnight': 0.9  # 21:00-00:00 UTC - slightly tighter
        })

        # ===== PHASE 28: Compound Profits =====
        self.compound_profits = config.get('compound_profits', True)
        self.profit_reinvest_ratio = config.get('profit_reinvest_ratio', 0.6)  # 60% reinvested

        # Initialize grid with enhanced config
        self.grid = ArithmeticGrid({
            'symbol': self.symbol,
            'upper_price': self.upper_price,
            'lower_price': self.lower_price,
            'num_grids': self.num_grids,
            'total_capital': self.grid_capital,
            'fee_rate': self.fee_rate,
            'compound_profits': self.compound_profits,
        })

        # State
        self.hedge_position: Optional[MarginPosition] = None
        self.current_price = 0.0
        self.last_price = 0.0
        self.price_history: List[float] = []

        # Statistics
        self.stats = DualGridHedgeStats(peak_capital=self.total_capital)

        # Tracking
        self.last_recenter_time = datetime.now()
        self.hedge_entry_time: Optional[datetime] = None
        self.trailing_stop_active = False
        self.trailing_stop_price = 0.0

    # ===== ATR Calculation =====
    def _calculate_atr(self, data: pd.DataFrame) -> float:
        """Calculate ATR from OHLCV data."""
        if data is None or len(data) < self.atr_period + 1:
            return 0.0

        high = data['high'].iloc[-self.atr_period:]
        low = data['low'].iloc[-self.atr_period:]
        close = data['close'].iloc[-self.atr_period:]

        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.mean()

    # ===== Session Detection =====
    def _get_current_session(self) -> str:
        """Determine current trading session based on UTC time."""
        hour = datetime.utcnow().hour
        if 0 <= hour < 8:
            return 'asia'
        elif 8 <= hour < 14:
            return 'europe'
        elif 14 <= hour < 21:
            return 'us'
        else:
            return 'overnight'

    def _get_session_multiplier(self) -> float:
        """Get trigger multiplier for current session."""
        if not self.use_session_adjustment:
            return 1.0
        session = self._get_current_session()
        return self.session_multipliers.get(session, 1.0)

    # ===== Dynamic Trigger Calculation =====
    def _calculate_hedge_triggers(self) -> Tuple[float, float]:
        """
        Calculate dynamic hedge trigger prices based on ATR and session.

        Returns:
            Tuple of (long_trigger, short_trigger)
        """
        session_mult = self._get_session_multiplier()

        if self.use_atr_triggers and self.current_atr > 0:
            # ATR-based triggers
            atr_offset = self.current_atr * self.atr_trigger_mult * session_mult
            long_trigger = self.upper_price + atr_offset
            short_trigger = self.lower_price - atr_offset
        else:
            # Fallback to static percentage
            offset = (self.upper_price - self.lower_price) * self.static_trigger_pct * session_mult
            long_trigger = self._base_long_trigger if self._base_long_trigger > 0 else self.upper_price + offset
            short_trigger = self._base_short_trigger if self._base_short_trigger > 0 else self.lower_price - offset

        return long_trigger, short_trigger

    # ===== Volatility-Adjusted Sizing =====
    def _calculate_hedge_size(self) -> Tuple[float, float]:
        """
        Calculate hedge position size adjusted for volatility.

        Returns:
            Tuple of (size_btc, size_usd)
        """
        base_size_usd = self.hedge_capital * self.base_size_pct

        if self.use_vol_sizing and self.current_atr > 0:
            # Calculate current volatility as percentage
            current_vol = self.current_atr / self.current_price if self.current_price > 0 else 0.02

            # Adjust size inversely to volatility (smaller in high vol)
            vol_factor = self.target_vol / current_vol if current_vol > 0 else 1.0
            vol_factor = max(0.5, min(1.5, vol_factor))  # Clamp to 0.5x - 1.5x

            size_pct = self.base_size_pct * vol_factor
            size_pct = max(self.min_size_pct, min(self.max_size_pct, size_pct))
            size_usd = self.hedge_capital * size_pct
        else:
            size_usd = base_size_usd

        size_btc = size_usd / self.current_price if self.current_price > 0 else 0
        return size_btc, size_usd

    # ===== Dynamic Leverage =====
    def _calculate_hedge_leverage(self) -> int:
        """Calculate hedge leverage based on volatility."""
        if self.current_atr <= 0 or self.current_price <= 0:
            return self.hedge_leverage

        vol_pct = self.current_atr / self.current_price

        # Lower leverage in high volatility
        if vol_pct > 0.03:  # >3% volatility
            leverage = self.min_hedge_leverage
        elif vol_pct < 0.015:  # <1.5% volatility
            leverage = self.max_hedge_leverage
        else:
            # Linear interpolation
            leverage = int(self.max_hedge_leverage - (vol_pct - 0.015) / 0.015 * (self.max_hedge_leverage - self.min_hedge_leverage))

        return max(self.min_hedge_leverage, min(self.max_hedge_leverage, leverage))

    # ===== Net Delta Calculation =====
    def _calculate_net_delta(self) -> float:
        """
        Calculate net delta exposure across grid and hedge.

        Positive = net long, Negative = net short
        """
        # Grid exposure (sum of long positions from filled buy orders)
        grid_delta = sum(pos.size for pos in self.grid.positions)

        # Hedge exposure
        hedge_delta = 0.0
        if self.hedge_position:
            if self.hedge_position.side == 'long':
                hedge_delta = self.hedge_position.size * self.hedge_position.leverage
            else:
                hedge_delta = -self.hedge_position.size * self.hedge_position.leverage

        self.stats.net_delta = grid_delta + hedge_delta
        return self.stats.net_delta

    # ===== Main Update =====
    def update(self, current_price: float, data: pd.DataFrame = None) -> List[Dict[str, Any]]:
        """
        Update both grid and hedge components.

        Phase 28 Enhanced:
        - ATR updates for dynamic triggers
        - Trailing stop management
        - Net delta tracking
        - Session-aware triggers
        """
        self.last_price = self.current_price
        self.current_price = current_price
        signals = []

        # Track price history for analysis
        self.price_history.append(current_price)
        if len(self.price_history) > 1000:
            self.price_history = self.price_history[-500:]

        # Update ATR if data provided
        if data is not None:
            new_atr = self._calculate_atr(data)
            if new_atr > 0:
                self.current_atr = new_atr

        # Update main grid
        grid_signals = self.grid.update(current_price, data)
        for sig in grid_signals:
            sig['component'] = 'grid'
            sig['strategy'] = 'DualGridHedge'
        signals.extend(grid_signals)

        # Sync grid stats
        self._sync_grid_stats()

        # Calculate net exposure
        self._calculate_net_delta()

        # Calculate dynamic triggers
        long_trigger, short_trigger = self._calculate_hedge_triggers()

        # Hedge logic
        if self.hedge_position is None:
            # Check for breakout hedge triggers
            if current_price > long_trigger:
                hedge_signal = self._create_hedge_signal('long')
                if hedge_signal:
                    signals.append(hedge_signal)
            elif current_price < short_trigger:
                hedge_signal = self._create_hedge_signal('short')
                if hedge_signal:
                    signals.append(hedge_signal)
        else:
            # Manage existing hedge (trailing stop, exits)
            hedge_signal = self._manage_hedge()
            if hedge_signal:
                signals.append(hedge_signal)

        # Update stats
        self._update_stats()

        return signals

    def _create_hedge_signal(self, direction: str) -> Optional[Dict[str, Any]]:
        """
        Create a hedge position signal with Phase 28 improvements.

        - Correct stop loss placement
        - ATR-based stops and targets
        - Volatility-adjusted sizing
        """
        if self.current_price <= 0:
            return None

        size_btc, size_usd = self._calculate_hedge_size()
        leverage = self._calculate_hedge_leverage()

        # PHASE 28 FIX: Correct stop loss placement
        if self.use_atr_triggers and self.current_atr > 0:
            stop_distance = self.current_atr * self.stop_loss_atr_mult
            tp_distance = self.current_atr * self.take_profit_atr_mult
        else:
            stop_distance = self.current_price * self.stop_loss_pct
            tp_distance = self.current_price * self.take_profit_pct

        if direction == 'long':
            # Long: stop BELOW entry, TP ABOVE entry
            stop_loss = self.current_price - stop_distance
            take_profit = self.current_price + tp_distance
        else:
            # Short: stop ABOVE entry, TP BELOW entry
            stop_loss = self.current_price + stop_distance
            take_profit = self.current_price - tp_distance

        session = self._get_current_session()

        return {
            'action': 'buy' if direction == 'long' else 'short',
            'symbol': self.symbol,
            'price': self.current_price,
            'size': size_btc,
            'size_usd': size_usd,
            'leverage': leverage,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'strategy': 'DualGridHedge',
            'component': 'hedge',
            'session': session,
            'atr': self.current_atr,
            'confidence': 0.7,
            'reason': f'Breakout hedge {direction} @ ${self.current_price:,.0f} (ATR: ${self.current_atr:.0f}, Session: {session})'
        }

    def _manage_hedge(self) -> Optional[Dict[str, Any]]:
        """
        Manage existing hedge position with trailing stop.

        Phase 28 Enhanced:
        - Trailing stop activation after profit threshold
        - Dynamic trailing distance based on ATR
        """
        if not self.hedge_position:
            return None

        pos = self.hedge_position

        # Calculate unrealized PnL
        if pos.side == 'long':
            pos.unrealized_pnl = (self.current_price - pos.entry_price) * pos.size * pos.leverage
            price_move = self.current_price - pos.entry_price

            # Track highest price for trailing stop
            if self.current_price > pos.highest_price:
                pos.highest_price = self.current_price

            # Check trailing stop activation
            if self.use_trailing_stop and self.current_atr > 0:
                activation_threshold = pos.entry_price + (self.current_atr * self.trailing_activation_atr)
                if pos.highest_price >= activation_threshold and not self.trailing_stop_active:
                    self.trailing_stop_active = True
                    self.trailing_stop_price = pos.highest_price - (self.current_atr * self.trailing_distance_atr)

                # Update trailing stop
                if self.trailing_stop_active:
                    new_trail = pos.highest_price - (self.current_atr * self.trailing_distance_atr)
                    self.trailing_stop_price = max(self.trailing_stop_price, new_trail)

                    # Check trailing stop hit
                    if self.current_price <= self.trailing_stop_price:
                        return self._close_hedge('trailing_stop', self.trailing_stop_price)

            # Check fixed stop/TP
            if self.current_price <= pos.stop_loss:
                self.stats.false_breakouts += 1
                return self._close_hedge('stop_loss', pos.stop_loss)
            if self.current_price >= pos.take_profit:
                return self._close_hedge('take_profit', pos.take_profit)

        else:  # short
            pos.unrealized_pnl = (pos.entry_price - self.current_price) * pos.size * pos.leverage
            price_move = pos.entry_price - self.current_price

            # Track lowest price for trailing stop
            if pos.lowest_price == 0 or self.current_price < pos.lowest_price:
                pos.lowest_price = self.current_price

            # Check trailing stop activation for short
            if self.use_trailing_stop and self.current_atr > 0:
                activation_threshold = pos.entry_price - (self.current_atr * self.trailing_activation_atr)
                if pos.lowest_price <= activation_threshold and not self.trailing_stop_active:
                    self.trailing_stop_active = True
                    self.trailing_stop_price = pos.lowest_price + (self.current_atr * self.trailing_distance_atr)

                # Update trailing stop for short
                if self.trailing_stop_active:
                    new_trail = pos.lowest_price + (self.current_atr * self.trailing_distance_atr)
                    self.trailing_stop_price = min(self.trailing_stop_price, new_trail) if self.trailing_stop_price > 0 else new_trail

                    # Check trailing stop hit
                    if self.current_price >= self.trailing_stop_price:
                        return self._close_hedge('trailing_stop', self.trailing_stop_price)

            # Check fixed stop/TP
            if self.current_price >= pos.stop_loss:
                self.stats.false_breakouts += 1
                return self._close_hedge('stop_loss', pos.stop_loss)
            if self.current_price <= pos.take_profit:
                return self._close_hedge('take_profit', pos.take_profit)

        return None

    def _close_hedge(self, reason: str, exit_price: float = None) -> Dict[str, Any]:
        """Close the hedge position and trigger recentering if configured."""
        pos = self.hedge_position
        close_price = exit_price or self.current_price

        # Recalculate final PnL at close price
        if pos.side == 'long':
            final_pnl = (close_price - pos.entry_price) * pos.size * pos.leverage
        else:
            final_pnl = (pos.entry_price - close_price) * pos.size * pos.leverage

        # Deduct fees
        entry_fee = pos.entry_price * pos.size * self.fee_rate
        exit_fee = close_price * pos.size * self.fee_rate
        final_pnl -= (entry_fee + exit_fee)

        return {
            'action': 'close_hedge',
            'symbol': self.symbol,
            'price': close_price,
            'size': pos.size,
            'leverage': pos.leverage,
            'pnl': final_pnl,
            'entry_price': pos.entry_price,
            'side': pos.side,
            'strategy': 'DualGridHedge',
            'component': 'hedge',
            'trailing_stop_active': self.trailing_stop_active,
            'trigger_recenter': self.recenter_after_hedge,
            'reason': f'Close {pos.side} hedge: {reason} @ ${close_price:,.0f} (PnL: ${final_pnl:,.2f})'
        }

    def _sync_grid_stats(self):
        """Sync statistics from grid component."""
        self.stats.grid_trades = self.grid.stats.total_trades
        self.stats.grid_pnl = self.grid.stats.total_pnl
        self.stats.grid_cycles = self.grid.stats.cycles_completed

    def _update_stats(self):
        """Update overall statistics."""
        # Calculate total PnL
        self.stats.total_pnl = self.stats.grid_pnl + self.stats.hedge_pnl

        # Add unrealized hedge PnL
        if self.hedge_position:
            self.stats.total_pnl += self.hedge_position.unrealized_pnl

        # Calculate current value
        current_value = self.initial_capital + self.stats.total_pnl

        # Update peak and drawdown
        if current_value > self.stats.peak_capital:
            self.stats.peak_capital = current_value

        if self.stats.peak_capital > 0:
            drawdown = (self.stats.peak_capital - current_value) / self.stats.peak_capital
            self.stats.max_drawdown = max(self.stats.max_drawdown, drawdown)

    def _recenter_grid(self):
        """Recenter grid around current price after hedge closes."""
        now = datetime.now()
        time_since_recenter = (now - self.last_recenter_time).total_seconds()

        if time_since_recenter < self.min_recenter_interval:
            return  # Too soon to recenter

        # Calculate new grid bounds centered on current price
        grid_range = self.upper_price - self.lower_price
        half_range = grid_range / 2

        self.upper_price = self.current_price + half_range
        self.lower_price = self.current_price - half_range

        # Rebuild grid
        self.grid = ArithmeticGrid({
            'symbol': self.symbol,
            'upper_price': self.upper_price,
            'lower_price': self.lower_price,
            'num_grids': self.num_grids,
            'total_capital': self.grid_capital,
            'fee_rate': self.fee_rate,
            'compound_profits': self.compound_profits,
        })

        self.last_recenter_time = now
        self.stats.recenters += 1
        self.stats.last_recenter_time = now

    def fill_order(self, signal: Dict[str, Any], fill_price: float):
        """
        Process filled orders.

        Phase 28 Enhanced:
        - Proper statistics tracking
        - Compound profit reinvestment
        - Grid recentering after hedge close
        """
        component = signal.get('component', 'grid')

        if component == 'grid':
            grid_level = signal.get('grid_level')
            if grid_level is not None:
                self.grid.fill_order(grid_level, fill_price)
                self.stats.grid_trades = self.grid.stats.total_trades

        elif component == 'hedge':
            action = signal.get('action')

            if action in ['buy', 'short']:
                # Open new hedge position
                self.hedge_position = MarginPosition(
                    entry_price=fill_price,
                    size=signal['size'],
                    side='long' if action == 'buy' else 'short',
                    leverage=signal['leverage'],
                    entry_time=datetime.now(),
                    stop_loss=signal['stop_loss'],
                    take_profit=signal['take_profit'],
                    margin_used=signal['size_usd'] / signal['leverage'],
                    highest_price=fill_price if action == 'buy' else 0,
                    lowest_price=fill_price if action == 'short' else 0
                )
                self.hedge_entry_time = datetime.now()
                self.trailing_stop_active = False
                self.trailing_stop_price = 0.0
                self.stats.hedge_trades += 1
                self.stats.total_trades += 1

            elif action == 'close_hedge':
                pnl = signal.get('pnl', 0)
                self.stats.hedge_pnl += pnl

                if pnl > 0:
                    self.stats.winning_trades += 1
                    self.stats.hedge_winning += 1

                    # Compound profits
                    if self.compound_profits:
                        reinvest = pnl * self.profit_reinvest_ratio
                        self.grid_capital += reinvest * self.grid_allocation
                        self.hedge_capital += reinvest * self.hedge_allocation
                        self.stats.realized_pnl += pnl * (1 - self.profit_reinvest_ratio)
                    else:
                        self.stats.realized_pnl += pnl
                else:
                    self.stats.losing_trades += 1
                    self.stats.hedge_losing += 1
                    self.stats.realized_pnl += pnl

                # Reset hedge state
                self.hedge_position = None
                self.hedge_entry_time = None
                self.trailing_stop_active = False
                self.trailing_stop_price = 0.0

                # Trigger grid recentering if configured
                if signal.get('trigger_recenter', False) and self.recenter_after_hedge:
                    self._recenter_grid()

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status with Phase 28 enhancements."""
        grid_status = self.grid.get_status()
        long_trigger, short_trigger = self._calculate_hedge_triggers()
        session = self._get_current_session()

        # Calculate win rates
        total_hedge = self.stats.hedge_winning + self.stats.hedge_losing
        hedge_win_rate = (self.stats.hedge_winning / total_hedge * 100) if total_hedge > 0 else 0

        return {
            'strategy': 'DualGridHedge',
            'version': 'Phase28',
            'current_price': self.current_price,
            'session': session,

            # Capital allocation
            'capital': {
                'initial': self.initial_capital,
                'grid': self.grid_capital,
                'hedge': self.hedge_capital,
                'total_pnl': self.stats.total_pnl,
                'realized_pnl': self.stats.realized_pnl,
            },

            # Grid status
            'grid': {
                'upper': self.upper_price,
                'lower': self.lower_price,
                'levels': len(self.grid.grid_levels),
                'positions': len(self.grid.positions),
                'trades': self.stats.grid_trades,
                'cycles': self.stats.grid_cycles,
                'pnl': self.stats.grid_pnl,
            },

            # Hedge status
            'hedge': {
                'active': self.hedge_position is not None,
                'side': self.hedge_position.side if self.hedge_position else None,
                'entry_price': self.hedge_position.entry_price if self.hedge_position else 0,
                'unrealized_pnl': self.hedge_position.unrealized_pnl if self.hedge_position else 0,
                'leverage': self.hedge_position.leverage if self.hedge_position else self.hedge_leverage,
                'trailing_active': self.trailing_stop_active,
                'trailing_price': self.trailing_stop_price,
                'trades': self.stats.hedge_trades,
                'wins': self.stats.hedge_winning,
                'losses': self.stats.hedge_losing,
                'win_rate': hedge_win_rate,
                'false_breakouts': self.stats.false_breakouts,
                'pnl': self.stats.hedge_pnl,
            },

            # Dynamic triggers
            'triggers': {
                'long': long_trigger,
                'short': short_trigger,
                'atr': self.current_atr,
                'session_mult': self._get_session_multiplier(),
            },

            # Risk metrics
            'risk': {
                'net_delta': self.stats.net_delta,
                'max_drawdown': self.stats.max_drawdown,
                'recenters': self.stats.recenters,
            },

            # Overall stats
            'total_trades': self.stats.total_trades,
            'winning_trades': self.stats.winning_trades,
            'losing_trades': self.stats.losing_trades,
            'win_rate': (self.stats.winning_trades / max(self.stats.total_trades, 1)) * 100,
            'total_pnl': self.stats.total_pnl,
        }


class TimeWeightedGrid:
    """
    Strategy 7: Time-Weighted Grid (Session-Based) - Phase 27 Enhanced

    Adjusts grid parameters based on trading session volatility with TWAP-style execution.

    Phase 27 Improvements:
    - Corrected session hours based on actual market activity
    - Session overlap detection (EU-US "Golden Hours" 12:00-16:00 UTC)
    - ATR-based dynamic sizing adjustment
    - TWAP-style order slicing for reduced market impact
    - Weekend/holiday low-liquidity detection
    - Proper interface for GridStrategyWrapper compatibility
    - Compound profit support
    - Day-of-week volatility awareness

    Sessions (UTC) - Corrected:
    - Asia (23:00-08:00): Lower vol, tighter grid, smaller size (wraps midnight)
    - Europe (07:00-16:00): Medium vol, standard parameters
    - US (13:00-22:00): Higher vol, wider grid, full size
    - EU-US Overlap (12:00-16:00): Peak liquidity "Golden Hours"
    - Overnight (22:00-23:00): Transition period
    """

    def __init__(self, config: Dict[str, Any]):
        self.symbol = config.get('symbol', 'BTC/USDT')
        self.total_capital = config.get('total_capital', 5000)
        self.initial_capital = self.total_capital
        self.leverage = config.get('leverage', 1)

        # Dynamic price range - centered on current price
        self.base_upper = config.get('upper_price', 100000)
        self.base_lower = config.get('lower_price', 94000)
        self.num_grids = config.get('num_grids', 15)
        self.range_pct = config.get('range_pct', 0.04)  # 4% default range

        # Phase 27: Corrected session configs based on actual market hours
        # Sessions include overlap detection and wrapping support
        default_sessions = {
            'asia': {
                'start': 23, 'end': 8, 'wraps': True,  # Wraps around midnight
                'spacing_mult': 0.8, 'size_mult': 0.6,
                'description': 'Asian session - lower volatility'
            },
            'europe': {
                'start': 7, 'end': 16, 'wraps': False,
                'spacing_mult': 1.0, 'size_mult': 0.85,
                'description': 'European session - medium volatility'
            },
            'us': {
                'start': 13, 'end': 22, 'wraps': False,
                'spacing_mult': 1.3, 'size_mult': 1.0,
                'description': 'US session - higher volatility'
            },
            'overlap_eu_us': {
                'start': 12, 'end': 16, 'wraps': False,
                'spacing_mult': 1.2, 'size_mult': 1.15,
                'priority': 10,  # Higher priority than individual sessions
                'description': 'EU-US overlap - peak liquidity "Golden Hours"'
            },
            'overnight': {
                'start': 22, 'end': 23, 'wraps': False,
                'spacing_mult': 0.9, 'size_mult': 0.5,
                'description': 'Overnight transition - reduced activity'
            }
        }
        # Use config value if provided and not None, otherwise use defaults
        self.sessions = config.get('session_configs') or default_sessions

        # Phase 27: Day-of-week volatility multipliers
        # Research shows mid-week has highest volatility
        default_day_multipliers = {
            0: 0.9,   # Monday - building up
            1: 1.1,   # Tuesday - high activity
            2: 1.2,   # Wednesday - peak (4PM UTC historically highest)
            3: 1.1,   # Thursday - high activity
            4: 1.0,   # Friday - winding down
            5: 0.7,   # Saturday - weekend low liquidity
            6: 0.7,   # Sunday - weekend low liquidity
        }
        self.day_multipliers = config.get('day_multipliers') or default_day_multipliers

        # Phase 27: ATR-based dynamic adjustment
        self.use_atr_adjustment = config.get('use_atr_adjustment', True)
        self.atr_period = config.get('atr_period', 14)
        self.target_atr_pct = config.get('target_atr_pct', 0.02)  # 2% baseline volatility
        self.current_atr = 0.0
        self.atr_adjustment_factor = 1.0

        # Phase 27: TWAP execution settings
        self.use_twap = config.get('use_twap', True)
        self.twap_slice_interval = config.get('twap_slice_interval', 300)  # 5 minutes
        self.twap_randomize_pct = config.get('twap_randomize_pct', 0.1)  # ±10% timing randomization
        self.pending_twap_orders: List[Dict[str, Any]] = []
        self.twap_execution_log: List[Dict[str, Any]] = []

        # Phase 27: Weekend/holiday handling
        self.reduce_weekend_size = config.get('reduce_weekend_size', True)
        self.weekend_size_mult = config.get('weekend_size_mult', 0.5)
        self.holidays_utc = config.get('holidays_utc', [
            (12, 24), (12, 25), (12, 31), (1, 1),  # Christmas/New Year
        ])

        # Phase 27: Compound profit settings
        self.compound_profits = config.get('compound_profits', True)
        self.max_compound = config.get('max_compound', 1.5)
        self.profit_distribution = config.get('profit_distribution', {
            'reinvest': 0.6,
            'realized': 0.4
        })

        # Phase 27: Risk management
        self.fee_rate = config.get('fee_rate', 0.001)
        self.max_drawdown = config.get('max_drawdown', 0.10)
        self.stop_loss = config.get('stop_loss', None)
        self.take_profit = config.get('take_profit', None)

        # State tracking
        self.current_session = None
        self.previous_session = None
        self.grid: Optional[ArithmeticGrid] = None
        self.current_price = 0.0
        self.last_grid_rebuild = None
        self.session_change_count = 0
        self.peak_capital = self.total_capital

        # Phase 27: Statistics tracking (compatible with GridStats interface)
        self._stats = GridStats()

        # Initialize grid
        self._build_grid_for_session()

    # ========== Phase 27: Property proxies for GridStrategyWrapper compatibility ==========

    @property
    def stats(self) -> GridStats:
        """Proxy to internal grid stats for wrapper compatibility."""
        if self.grid:
            return self.grid.stats
        return self._stats

    @property
    def lower_price(self) -> float:
        """Proxy to grid lower_price for wrapper compatibility."""
        if self.grid:
            return self.grid.lower_price
        return self.base_lower

    @property
    def upper_price(self) -> float:
        """Proxy to grid upper_price for wrapper compatibility."""
        if self.grid:
            return self.grid.upper_price
        return self.base_upper

    @property
    def grid_levels(self) -> List[GridLevel]:
        """Proxy to grid levels for wrapper compatibility."""
        if self.grid:
            return self.grid.grid_levels
        return []

    @property
    def positions(self) -> List[GridPosition]:
        """Proxy to positions for wrapper compatibility."""
        if self.grid:
            return self.grid.positions
        return []

    @property
    def grid_spacing(self) -> float:
        """Proxy to grid spacing for wrapper compatibility."""
        if self.grid:
            return getattr(self.grid, 'grid_spacing', 0)
        return 0

    # ========== Session Detection (Phase 27 Enhanced) ==========

    def _get_current_session(self) -> str:
        """
        Determine current trading session with overlap priority.

        Phase 27: Handles wrapping sessions (Asia) and overlap detection.
        Returns the most relevant session based on priority.
        """
        now = datetime.utcnow()
        hour = now.hour

        # Check sessions by priority (overlaps first)
        sessions_by_priority = sorted(
            self.sessions.items(),
            key=lambda x: x[1].get('priority', 0),
            reverse=True
        )

        for name, sess in sessions_by_priority:
            start = sess['start']
            end = sess['end']
            wraps = sess.get('wraps', False)

            if wraps:
                # Session wraps around midnight (e.g., Asia 23:00-08:00)
                if hour >= start or hour < end:
                    return name
            else:
                # Normal session
                if start <= hour < end:
                    return name

        return 'overnight'

    def _is_session_overlap(self) -> Tuple[bool, str]:
        """Check if we're in a session overlap period."""
        now = datetime.utcnow()
        hour = now.hour

        # EU-US overlap: 12:00-16:00 UTC (Golden Hours)
        if 12 <= hour < 16:
            return True, 'overlap_eu_us'

        # Asia-Europe overlap: 07:00-08:00 UTC
        if 7 <= hour < 8:
            return True, 'overlap_asia_eu'

        return False, ''

    def _is_low_liquidity_period(self) -> Tuple[bool, str]:
        """
        Detect weekends and holidays with lower liquidity.

        Phase 27: Returns (is_low_liquidity, reason)
        """
        now = datetime.utcnow()

        # Weekend check (Saturday=5, Sunday=6)
        if now.weekday() >= 5:
            return True, f'weekend_{["saturday", "sunday"][now.weekday()-5]}'

        # Holiday check
        for month, day in self.holidays_utc:
            if now.month == month and now.day == day:
                return True, 'holiday'

        # Late Sunday night (preparing for Monday)
        if now.weekday() == 6 and now.hour >= 20:
            return True, 'sunday_night'

        return False, ''

    def _get_day_multiplier(self) -> float:
        """Get volatility multiplier based on day of week."""
        day = datetime.utcnow().weekday()
        return self.day_multipliers.get(day, 1.0)

    # ========== Grid Management (Phase 27 Enhanced) ==========

    def _build_grid_for_session(self, force_rebuild: bool = False):
        """
        Build grid with session-appropriate parameters.

        Phase 27 Enhanced:
        - Dynamic center price based on current market
        - ATR-adjusted spacing
        - Day-of-week volatility awareness
        - Low-liquidity period handling
        """
        session = self._get_current_session()

        # Check if rebuild is needed
        if not force_rebuild and session == self.current_session and self.grid is not None:
            return False  # No change needed

        self.previous_session = self.current_session
        self.current_session = session
        sess_config = self.sessions.get(session, self.sessions['overnight'])

        # Get base multipliers
        spacing_mult = sess_config['spacing_mult']
        size_mult = sess_config['size_mult']

        # Phase 27: Apply day-of-week adjustment
        day_mult = self._get_day_multiplier()
        size_mult *= day_mult

        # Phase 27: Apply ATR adjustment if enabled
        if self.use_atr_adjustment and self.current_atr > 0:
            atr_pct = self.current_atr / self.current_price if self.current_price > 0 else 0
            self.atr_adjustment_factor = min(max(atr_pct / self.target_atr_pct, 0.5), 2.0)
            spacing_mult *= self.atr_adjustment_factor
            # Reduce size in high volatility, increase in low volatility
            size_mult *= (2.0 - self.atr_adjustment_factor) / 2 + 0.5

        # Phase 27: Apply low-liquidity adjustment
        is_low_liq, liq_reason = self._is_low_liquidity_period()
        if is_low_liq and self.reduce_weekend_size:
            size_mult *= self.weekend_size_mult

        # Calculate grid range
        # Use current price as center if available, otherwise use config midpoint
        if self.current_price > 0:
            center_price = self.current_price
        else:
            center_price = (self.base_upper + self.base_lower) / 2

        # Dynamic range based on range_pct config
        base_range = center_price * self.range_pct * 2
        adjusted_range = base_range * spacing_mult

        upper = center_price + adjusted_range / 2
        lower = center_price - adjusted_range / 2

        # Phase 27: Calculate effective capital with compounding
        effective_capital = self.total_capital
        if self.compound_profits and self._stats.profit_reinvested > 0:
            effective_capital += self._stats.profit_reinvested
            effective_capital = min(effective_capital, self.initial_capital * self.max_compound)

        # Build the underlying ArithmeticGrid
        grid_config = {
            'symbol': self.symbol,
            'upper_price': upper,
            'lower_price': lower,
            'num_grids': self.num_grids,
            'total_capital': effective_capital * size_mult,
            'leverage': self.leverage,
            'fee_rate': self.fee_rate,
            'compound_profits': self.compound_profits,
            'max_compound': self.max_compound,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'max_drawdown': self.max_drawdown,
        }

        # Preserve stats from previous grid
        old_stats = self.grid.stats if self.grid else None

        self.grid = ArithmeticGrid(grid_config)

        # Initialize with current price for proper buy/sell placement
        if self.current_price > 0:
            self.grid._setup_grid(self.current_price)

        # Restore accumulated stats
        if old_stats:
            self.grid.stats.total_trades = old_stats.total_trades
            self.grid.stats.winning_trades = old_stats.winning_trades
            self.grid.stats.losing_trades = old_stats.losing_trades
            self.grid.stats.total_pnl = old_stats.total_pnl
            self.grid.stats.realized_pnl = old_stats.realized_pnl
            self.grid.stats.total_fees = old_stats.total_fees
            self.grid.stats.cycles_completed = old_stats.cycles_completed
            self.grid.stats.profit_reinvested = old_stats.profit_reinvested

        self.last_grid_rebuild = datetime.utcnow()
        self.session_change_count += 1

        # Log session change
        liq_info = f" [{liq_reason}]" if is_low_liq else ""
        print(f"[TimeWeightedGrid] Session: {session}{liq_info}, "
              f"Range: ${lower:,.0f}-${upper:,.0f}, "
              f"Size mult: {size_mult:.2f}, Day mult: {day_mult:.1f}")

        return True

    def _calculate_atr(self, data: pd.DataFrame) -> float:
        """Calculate ATR from OHLCV data."""
        if data is None or len(data) < self.atr_period + 1:
            return 0.0

        high = data['high'].iloc[-self.atr_period:]
        low = data['low'].iloc[-self.atr_period:]
        close = data['close'].iloc[-self.atr_period:]

        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.mean()

    # ========== TWAP Execution (Phase 27 New) ==========

    def _create_twap_slices(self, signal: Dict[str, Any], duration_seconds: int = 1800) -> List[Dict[str, Any]]:
        """
        Split a large order into TWAP slices for reduced market impact.

        Phase 27: Implements time-weighted order slicing with randomization.

        Args:
            signal: Original signal to slice
            duration_seconds: Total duration to spread order (default 30 min)

        Returns:
            List of sliced orders with scheduled execution times
        """
        if not self.use_twap:
            return [signal]

        total_size = signal.get('size', 0)
        if total_size <= 0:
            return []

        # Calculate number of slices
        num_slices = max(1, duration_seconds // self.twap_slice_interval)
        slice_size = total_size / num_slices

        slices = []
        base_time = datetime.utcnow()

        for i in range(num_slices):
            # Add randomization to timing (±10% default)
            random_offset = 0
            if self.twap_randomize_pct > 0:
                max_offset = int(self.twap_slice_interval * self.twap_randomize_pct)
                random_offset = np.random.randint(-max_offset, max_offset + 1)

            scheduled_time = base_time + pd.Timedelta(
                seconds=self.twap_slice_interval * i + random_offset
            )

            slice_signal = {
                **signal,
                'size': slice_size,
                'twap_slice': i + 1,
                'twap_total_slices': num_slices,
                'scheduled_time': scheduled_time,
                'original_size': total_size,
            }
            slices.append(slice_signal)

        return slices

    def _process_pending_twap(self) -> List[Dict[str, Any]]:
        """Process any pending TWAP orders that are due."""
        now = datetime.utcnow()
        ready_orders = []
        remaining_orders = []

        for order in self.pending_twap_orders:
            scheduled = order.get('scheduled_time', now)
            if isinstance(scheduled, str):
                scheduled = datetime.fromisoformat(scheduled)

            if scheduled <= now:
                ready_orders.append(order)
                self.twap_execution_log.append({
                    'time': now.isoformat(),
                    'order': order,
                    'status': 'executed'
                })
            else:
                remaining_orders.append(order)

        self.pending_twap_orders = remaining_orders
        return ready_orders

    # ========== Main Update Loop (Phase 27 Enhanced) ==========

    def update(self, current_price: float, data: pd.DataFrame = None) -> List[Dict[str, Any]]:
        """
        Update grid with session awareness and TWAP execution.

        Phase 27 Enhanced:
        - ATR calculation and dynamic adjustment
        - Session change detection and grid rebuild
        - TWAP order processing
        - Low-liquidity period awareness
        """
        self.current_price = current_price
        signals = []

        # Update ATR if data provided
        if data is not None and self.use_atr_adjustment:
            self.current_atr = self._calculate_atr(data)

        # Check if session changed and rebuild grid if needed
        session_changed = self._build_grid_for_session()

        if self.grid is None:
            return signals

        # Process any pending TWAP orders first
        if self.use_twap and self.pending_twap_orders:
            twap_signals = self._process_pending_twap()
            signals.extend(twap_signals)

        # Update underlying grid
        grid_signals = self.grid.update(current_price, data)

        # Enhance signals with session and TWAP info
        for sig in grid_signals:
            sig['strategy'] = 'TimeWeightedGrid'
            sig['session'] = self.current_session
            sig['session_changed'] = session_changed

            # Check for low liquidity period
            is_low_liq, liq_reason = self._is_low_liquidity_period()
            sig['low_liquidity'] = is_low_liq
            sig['liquidity_reason'] = liq_reason

            # Add overlap info
            is_overlap, overlap_name = self._is_session_overlap()
            sig['in_overlap'] = is_overlap
            sig['overlap_session'] = overlap_name

            # Add ATR info
            sig['current_atr'] = self.current_atr
            sig['atr_adjustment'] = self.atr_adjustment_factor

            # Apply TWAP slicing for larger orders
            if self.use_twap and sig.get('action') in ['buy', 'sell']:
                order_value = sig.get('size', 0) * current_price
                # TWAP for orders over 1% of capital
                if order_value > self.total_capital * 0.01:
                    slices = self._create_twap_slices(sig)
                    if len(slices) > 1:
                        # First slice executes now, rest are queued
                        signals.append(slices[0])
                        self.pending_twap_orders.extend(slices[1:])
                        continue

            signals.append(sig)

        # Update peak capital tracking
        current_value = self._calculate_portfolio_value()
        if current_value > self.peak_capital:
            self.peak_capital = current_value

        return signals

    def _calculate_portfolio_value(self) -> float:
        """Calculate current portfolio value."""
        if not self.grid:
            return self.total_capital

        unrealized = sum(p.unrealized_pnl for p in self.grid.positions)
        return self.total_capital + self.stats.realized_pnl + unrealized

    # ========== Order Management ==========

    def fill_order(self, level_idx: int, fill_price: float, fill_time: datetime = None, order_id: str = None):
        """
        Process filled order - delegate to underlying grid.

        Phase 27: Enhanced signature for wrapper compatibility.
        """
        if self.grid is not None:
            self.grid.fill_order(level_idx, fill_price, fill_time, order_id)

            # Sync stats
            self._stats = self.grid.stats

    def _setup_grid(self, current_price: float = None):
        """
        Setup grid interface for wrapper compatibility.

        Phase 27: Required by GridStrategyWrapper._initialize_grid()
        """
        if current_price:
            self.current_price = current_price
        self._build_grid_for_session(force_rebuild=True)

    # ========== Status and Monitoring ==========

    def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status with session info.

        Phase 27: Extended status for monitoring and debugging.
        """
        grid_status = self.grid.get_status() if self.grid else {}

        is_low_liq, liq_reason = self._is_low_liquidity_period()
        is_overlap, overlap_name = self._is_session_overlap()

        return {
            'strategy': 'TimeWeightedGrid',
            'version': 'Phase27',

            # Session info
            'current_session': self.current_session,
            'previous_session': self.previous_session,
            'session_config': self.sessions.get(self.current_session, {}),
            'session_change_count': self.session_change_count,
            'last_grid_rebuild': self.last_grid_rebuild.isoformat() if self.last_grid_rebuild else None,

            # Overlap and liquidity
            'in_overlap': is_overlap,
            'overlap_session': overlap_name,
            'low_liquidity': is_low_liq,
            'liquidity_reason': liq_reason,
            'day_multiplier': self._get_day_multiplier(),

            # ATR info
            'current_atr': self.current_atr,
            'atr_adjustment_factor': self.atr_adjustment_factor,
            'target_atr_pct': self.target_atr_pct,

            # TWAP status
            'twap_enabled': self.use_twap,
            'pending_twap_orders': len(self.pending_twap_orders),
            'twap_executions': len(self.twap_execution_log),

            # Capital and performance
            'total_capital': self.total_capital,
            'effective_capital': self._calculate_portfolio_value(),
            'peak_capital': self.peak_capital,
            'compound_multiplier': (self.total_capital + self._stats.profit_reinvested) / self.initial_capital if self.initial_capital > 0 else 1.0,

            # Grid status (from underlying grid)
            **grid_status
        }

    def should_recenter(self, current_price: float) -> bool:
        """Check if grid should be recentered."""
        if not self.grid:
            return True

        center = (self.grid.upper_price + self.grid.lower_price) / 2
        deviation = abs(current_price - center) / center

        # Recenter if price moved more than half the range
        return deviation > self.range_pct

    def recenter_grid(self, new_center_price: float):
        """Recenter grid around new price."""
        self.current_price = new_center_price
        self._build_grid_for_session(force_rebuild=True)


class LiquidationHuntScalper:
    """
    Strategy 8: Liquidation Hunt Scalper (Phase 29 Enhanced)
    Targets areas where leveraged traders get liquidated.

    Phase 29 Improvements:
    - ATR-based dynamic liquidation zones (not fixed %)
    - Proper swing high/low detection (not simple max/min)
    - Volume spike confirmation for liquidation cascades
    - RSI filter for oversold/overbought confirmation
    - Multi-pattern reversal detection (hammer, engulfing, doji, pin bar)
    - Dynamic leverage based on volatility
    - Trailing stops after partial profit
    - Session awareness for trigger adjustment
    - Compound profit reinvestment
    - Compatible stats object for orchestrator integration
    - on_order_filled() method for proper position tracking
    """

    def __init__(self, config: Dict[str, Any]):
        self.symbol = config.get('symbol', 'BTC/USD')
        self.total_capital = config.get('total_capital', 2000)
        self.initial_capital = self.total_capital

        # ===== Leverage Configuration =====
        self.base_leverage = config.get('leverage', 3)
        self.leverage = self.base_leverage
        self.use_dynamic_leverage = config.get('use_dynamic_leverage', True)
        self.max_leverage = config.get('max_leverage', 5)
        self.min_leverage = config.get('min_leverage', 2)
        self.high_vol_threshold = config.get('high_vol_threshold', 0.03)  # 3% ATR
        self.low_vol_threshold = config.get('low_vol_threshold', 0.015)  # 1.5% ATR

        # ===== Position Sizing =====
        self.size_pct = config.get('size_pct', 0.08)  # 8% per trade
        self.min_size_pct = config.get('min_size_pct', 0.04)
        self.max_size_pct = config.get('max_size_pct', 0.15)
        self.max_positions = config.get('max_positions', 2)

        # ===== ATR-Based Liquidation Zone Detection =====
        self.use_atr_zones = config.get('use_atr_zones', True)
        self.liq_zone_atr_mult = config.get('liq_zone_atr_mult', 1.5)  # 1.5x ATR from swing
        self.liq_zone_pct = config.get('liq_zone_pct', 0.02)  # Fallback: 2% fixed
        self.atr_period = config.get('atr_period', 14)

        # ===== Swing Detection =====
        self.lookback_bars = config.get('lookback_bars', 50)
        self.swing_lookback = config.get('swing_lookback', 5)  # Bars to confirm swing
        self.min_swing_size_atr = config.get('min_swing_size_atr', 0.5)  # Min swing = 0.5x ATR

        # ===== Volume Confirmation =====
        self.use_volume_filter = config.get('use_volume_filter', True)
        self.volume_spike_mult = config.get('volume_spike_mult', 1.5)  # 1.5x avg volume
        self.volume_window = config.get('volume_window', 20)

        # ===== RSI Filter =====
        self.use_rsi_filter = config.get('use_rsi_filter', True)
        self.rsi_period = config.get('rsi_period', 14)
        self.rsi_oversold = config.get('rsi_oversold', 30)
        self.rsi_overbought = config.get('rsi_overbought', 70)
        self.rsi_extreme_oversold = config.get('rsi_extreme_oversold', 20)
        self.rsi_extreme_overbought = config.get('rsi_extreme_overbought', 80)

        # ===== Reversal Pattern Settings =====
        self.min_wick_ratio = config.get('min_wick_ratio', 2.0)  # Wick >= 2x body for hammer
        self.engulfing_body_ratio = config.get('engulfing_body_ratio', 1.1)  # 10% larger body
        self.require_pattern = config.get('require_pattern', True)  # Must have reversal pattern
        self.pattern_confidence_boost = config.get('pattern_confidence_boost', 0.15)

        # ===== ATR-Based Risk Management =====
        self.use_atr_stops = config.get('use_atr_stops', True)
        self.stop_loss_atr_mult = config.get('stop_loss_atr_mult', 1.5)
        self.take_profit_atr_mult = config.get('take_profit_atr_mult', 2.5)
        self.stop_loss_pct = config.get('stop_loss_pct', 0.015)  # Fallback: 1.5%
        self.take_profit_pct = config.get('take_profit_pct', 0.03)  # Fallback: 3%

        # ===== Trailing Stop =====
        self.use_trailing_stop = config.get('use_trailing_stop', True)
        self.trailing_activation_atr = config.get('trailing_activation_atr', 1.5)  # Activate after 1.5x ATR profit
        self.trailing_distance_atr = config.get('trailing_distance_atr', 1.0)  # Trail by 1x ATR

        # ===== Session Awareness =====
        self.use_session_adjustment = config.get('use_session_adjustment', True)
        self.session_multipliers = config.get('session_multipliers', {
            'asia': 0.8,      # 00:00-08:00 UTC - tighter zones
            'europe': 1.0,    # 08:00-14:00 UTC - standard
            'us': 1.2,        # 14:00-21:00 UTC - wider (more volatile)
            'overnight': 0.9  # 21:00-00:00 UTC - slightly tighter
        })

        # ===== Compound Profits =====
        self.compound_profits = config.get('compound_profits', True)
        self.max_compound = config.get('max_compound', 1.5)
        self.profit_reinvest_ratio = config.get('profit_reinvest_ratio', 0.6)

        # ===== Cooldown =====
        self.cooldown_bars = config.get('cooldown_bars', 3)
        self.bars_since_trade = 0

        # ===== State =====
        self.positions: List[MarginPosition] = []
        self.current_price = 0.0
        self.current_atr = 0.0
        self.current_atr_pct = 0.0
        self.current_rsi = 50.0
        self.current_volume_ratio = 1.0
        self.current_session = 'unknown'

        # Swing points
        self.swing_highs: List[Dict] = []  # [{price, index, strength}]
        self.swing_lows: List[Dict] = []
        self.local_high = 0.0
        self.local_low = float('inf')

        # Liquidation zones (calculated dynamically)
        self.long_liq_zone = 0.0  # Below swing high - where longs get liquidated
        self.short_liq_zone = 0.0  # Above swing low - where shorts get liquidated

        # Last signal info
        self.last_signal = None
        self.last_pattern = None

        # ===== Stats (Compatible with orchestrator) =====
        self.stats = GridStats()
        self.stats.total_trades = 0
        self.stats.winning_trades = 0
        self.stats.losing_trades = 0
        self.stats.total_pnl = 0.0
        self.stats.realized_pnl = 0.0
        self.stats.total_fees = 0.0
        self.stats.cycles_completed = 0
        self.stats.profit_reinvested = 0.0

        # Price/volume history
        self._price_history: List[float] = []
        self._volume_history: List[float] = []
        self._high_history: List[float] = []
        self._low_history: List[float] = []
        self._close_history: List[float] = []

    def update(self, current_price: float, data: pd.DataFrame = None) -> List[Dict[str, Any]]:
        """
        Update strategy and generate signals.

        Phase 29: Full signal generation with all confirmations.
        """
        self.current_price = current_price
        signals = []
        self.bars_since_trade += 1

        # Update price history
        self._price_history.append(current_price)
        if len(self._price_history) > self.lookback_bars:
            self._price_history = self._price_history[-self.lookback_bars:]

        # Need OHLCV data for proper analysis
        if data is None or len(data) < self.atr_period + 5:
            return signals

        # Update from dataframe
        self._update_from_data(data)

        # Calculate indicators
        self._calculate_atr(data)
        self._calculate_rsi(data)
        self._calculate_volume_ratio(data)
        self._detect_session()

        # Detect swing points and liquidation zones
        self._detect_swing_points(data)
        self._calculate_liq_zones()

        # Update dynamic leverage
        if self.use_dynamic_leverage:
            self._update_leverage()

        # Check for entry signals (with cooldown)
        if len(self.positions) < self.max_positions and self.bars_since_trade >= self.cooldown_bars:
            entry_signal = self._check_entry_conditions(data)
            if entry_signal:
                signals.append(entry_signal)

        # Manage existing positions (trailing stop, exit conditions)
        for pos in self.positions[:]:
            exit_signal = self._manage_position(pos, data)
            if exit_signal:
                signals.append(exit_signal)

        return signals

    def _update_from_data(self, data: pd.DataFrame):
        """Update internal history from dataframe."""
        if 'high' in data.columns:
            self._high_history = data['high'].tail(self.lookback_bars).tolist()
        if 'low' in data.columns:
            self._low_history = data['low'].tail(self.lookback_bars).tolist()
        if 'close' in data.columns:
            self._close_history = data['close'].tail(self.lookback_bars).tolist()
        if 'volume' in data.columns:
            self._volume_history = data['volume'].tail(self.lookback_bars).tolist()

    def _calculate_atr(self, data: pd.DataFrame):
        """Calculate ATR for dynamic zones and stops."""
        if len(data) < self.atr_period + 1:
            return

        high = data['high'].iloc[-self.atr_period:]
        low = data['low'].iloc[-self.atr_period:]
        close = data['close'].iloc[-self.atr_period:]

        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        self.current_atr = tr.mean()
        self.current_atr_pct = self.current_atr / self.current_price if self.current_price > 0 else 0

    def _calculate_rsi(self, data: pd.DataFrame):
        """Calculate RSI for confirmation."""
        if len(data) < self.rsi_period + 1:
            return

        close = data['close'].iloc[-(self.rsi_period + 1):]
        delta = close.diff()

        gain = delta.clip(lower=0)
        loss = (-delta.clip(upper=0))

        avg_gain = gain.rolling(self.rsi_period).mean().iloc[-1]
        avg_loss = loss.rolling(self.rsi_period).mean().iloc[-1]

        if avg_loss == 0:
            self.current_rsi = 100
        else:
            rs = avg_gain / avg_loss
            self.current_rsi = 100 - (100 / (1 + rs))

    def _calculate_volume_ratio(self, data: pd.DataFrame):
        """Calculate current volume vs average."""
        if 'volume' not in data.columns or len(data) < self.volume_window + 1:
            self.current_volume_ratio = 1.0
            return

        current_vol = data['volume'].iloc[-1]
        avg_vol = data['volume'].iloc[-self.volume_window-1:-1].mean()

        self.current_volume_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0

    def _detect_session(self):
        """Detect current trading session."""
        hour = datetime.now().hour

        if 0 <= hour < 8:
            self.current_session = 'asia'
        elif 8 <= hour < 14:
            self.current_session = 'europe'
        elif 14 <= hour < 21:
            self.current_session = 'us'
        else:
            self.current_session = 'overnight'

    def _detect_swing_points(self, data: pd.DataFrame):
        """
        Detect swing highs and lows using proper peak/trough algorithm.

        A swing high is a high with lower highs on both sides.
        A swing low is a low with higher lows on both sides.
        """
        if len(data) < self.swing_lookback * 2 + 1:
            return

        highs = data['high'].values
        lows = data['low'].values
        n = self.swing_lookback

        self.swing_highs = []
        self.swing_lows = []

        # Scan for swing points (excluding most recent bars to confirm)
        for i in range(n, len(highs) - n):
            # Check for swing high
            is_swing_high = True
            for j in range(1, n + 1):
                if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                    is_swing_high = False
                    break

            if is_swing_high:
                # Calculate strength (how much higher than neighbors)
                left_diff = highs[i] - max(highs[i-n:i])
                right_diff = highs[i] - max(highs[i+1:i+n+1])
                strength = min(left_diff, right_diff) / self.current_atr if self.current_atr > 0 else 0

                if strength >= self.min_swing_size_atr:
                    self.swing_highs.append({
                        'price': highs[i],
                        'index': i,
                        'strength': strength
                    })

            # Check for swing low
            is_swing_low = True
            for j in range(1, n + 1):
                if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                    is_swing_low = False
                    break

            if is_swing_low:
                left_diff = min(lows[i-n:i]) - lows[i]
                right_diff = min(lows[i+1:i+n+1]) - lows[i]
                strength = min(left_diff, right_diff) / self.current_atr if self.current_atr > 0 else 0

                if strength >= self.min_swing_size_atr:
                    self.swing_lows.append({
                        'price': lows[i],
                        'index': i,
                        'strength': strength
                    })

        # Keep only recent swing points
        self.swing_highs = self.swing_highs[-10:]
        self.swing_lows = self.swing_lows[-10:]

        # Update local high/low from recent swings
        if self.swing_highs:
            self.local_high = max(s['price'] for s in self.swing_highs[-3:])
        else:
            self.local_high = max(self._high_history[-20:]) if self._high_history else self.current_price

        if self.swing_lows:
            self.local_low = min(s['price'] for s in self.swing_lows[-3:])
        else:
            self.local_low = min(self._low_history[-20:]) if self._low_history else self.current_price

    def _calculate_liq_zones(self):
        """
        Calculate liquidation zones based on ATR and swing points.

        Long liquidation zone: Below recent swing high (where longs from the top get stopped out)
        Short liquidation zone: Above recent swing low (where shorts from the bottom get stopped out)
        """
        # Get session multiplier
        session_mult = 1.0
        if self.use_session_adjustment:
            session_mult = self.session_multipliers.get(self.current_session, 1.0)

        if self.use_atr_zones and self.current_atr > 0:
            # ATR-based zones
            zone_distance = self.current_atr * self.liq_zone_atr_mult * session_mult

            self.long_liq_zone = self.local_high - zone_distance
            self.short_liq_zone = self.local_low + zone_distance
        else:
            # Fallback to percentage-based
            zone_pct = self.liq_zone_pct * session_mult

            self.long_liq_zone = self.local_high * (1 - zone_pct)
            self.short_liq_zone = self.local_low * (1 + zone_pct)

    def _update_leverage(self):
        """Update leverage based on current volatility."""
        if self.current_atr_pct <= 0:
            self.leverage = self.base_leverage
            return

        if self.current_atr_pct > self.high_vol_threshold:
            # High volatility - reduce leverage
            self.leverage = self.min_leverage
        elif self.current_atr_pct < self.low_vol_threshold:
            # Low volatility - increase leverage
            self.leverage = self.max_leverage
        else:
            # Scale linearly between thresholds
            vol_range = self.high_vol_threshold - self.low_vol_threshold
            vol_position = (self.current_atr_pct - self.low_vol_threshold) / vol_range
            lev_range = self.max_leverage - self.min_leverage
            self.leverage = self.max_leverage - (vol_position * lev_range)

        self.leverage = max(self.min_leverage, min(self.max_leverage, self.leverage))

    def _check_entry_conditions(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Check for entry conditions with full confirmation.

        Entry conditions for LONG (after long liquidation sweep):
        1. Price dropped into long liquidation zone (below swing high)
        2. Bullish reversal candle pattern detected
        3. RSI oversold (optional but increases confidence)
        4. Volume spike (liquidation cascade confirmation)

        Entry conditions for SHORT (after short liquidation sweep):
        1. Price pumped into short liquidation zone (above swing low)
        2. Bearish reversal candle pattern detected
        3. RSI overbought (optional but increases confidence)
        4. Volume spike confirmation
        """
        if data is None or len(data) < 3:
            return None

        # Check for long entry (price in long liq zone)
        if self.current_price <= self.long_liq_zone and self.current_price > self.local_low:
            pattern, confidence = self._detect_reversal_pattern(data, 'bullish')

            if pattern or not self.require_pattern:
                # Check additional confirmations
                rsi_confirmed = not self.use_rsi_filter or self.current_rsi <= self.rsi_oversold
                volume_confirmed = not self.use_volume_filter or self.current_volume_ratio >= self.volume_spike_mult

                # Calculate overall confidence
                base_confidence = 0.5
                if pattern:
                    base_confidence += self.pattern_confidence_boost
                if self.current_rsi <= self.rsi_extreme_oversold:
                    base_confidence += 0.15
                elif rsi_confirmed:
                    base_confidence += 0.10
                if volume_confirmed:
                    base_confidence += 0.10

                # Strength from how deep into liq zone
                zone_depth = (self.long_liq_zone - self.current_price) / self.current_atr if self.current_atr > 0 else 0
                base_confidence += min(0.15, zone_depth * 0.1)

                if base_confidence >= 0.55 and (pattern or (rsi_confirmed and volume_confirmed)):
                    self.last_pattern = pattern
                    return self._create_entry_signal('long', pattern or 'liq_zone_bounce', base_confidence)

        # Check for short entry (price in short liq zone)
        if self.current_price >= self.short_liq_zone and self.current_price < self.local_high:
            pattern, confidence = self._detect_reversal_pattern(data, 'bearish')

            if pattern or not self.require_pattern:
                rsi_confirmed = not self.use_rsi_filter or self.current_rsi >= self.rsi_overbought
                volume_confirmed = not self.use_volume_filter or self.current_volume_ratio >= self.volume_spike_mult

                base_confidence = 0.5
                if pattern:
                    base_confidence += self.pattern_confidence_boost
                if self.current_rsi >= self.rsi_extreme_overbought:
                    base_confidence += 0.15
                elif rsi_confirmed:
                    base_confidence += 0.10
                if volume_confirmed:
                    base_confidence += 0.10

                zone_depth = (self.current_price - self.short_liq_zone) / self.current_atr if self.current_atr > 0 else 0
                base_confidence += min(0.15, zone_depth * 0.1)

                if base_confidence >= 0.55 and (pattern or (rsi_confirmed and volume_confirmed)):
                    self.last_pattern = pattern
                    return self._create_entry_signal('short', pattern or 'liq_zone_rejection', base_confidence)

        return None

    def _detect_reversal_pattern(self, data: pd.DataFrame, direction: str) -> tuple:
        """
        Detect reversal candle patterns with confidence scoring.

        Returns: (pattern_name, confidence) or (None, 0)

        Patterns detected:
        - Hammer / Inverted Hammer (bullish)
        - Shooting Star / Hanging Man (bearish)
        - Bullish / Bearish Engulfing
        - Doji at extremes
        - Pin Bar
        """
        if len(data) < 3:
            return None, 0

        curr = data.iloc[-1]
        prev = data.iloc[-2]
        prev2 = data.iloc[-3]

        curr_open = curr['open']
        curr_high = curr['high']
        curr_low = curr['low']
        curr_close = curr['close']
        curr_body = abs(curr_close - curr_open)
        curr_range = curr_high - curr_low

        prev_open = prev['open']
        prev_high = prev['high']
        prev_low = prev['low']
        prev_close = prev['close']
        prev_body = abs(prev_close - prev_open)

        if curr_range == 0:
            return None, 0

        if direction == 'bullish':
            # === HAMMER ===
            # Long lower wick, small body at top
            lower_wick = min(curr_open, curr_close) - curr_low
            upper_wick = curr_high - max(curr_open, curr_close)

            if lower_wick >= curr_body * self.min_wick_ratio and upper_wick <= curr_body * 0.5:
                if curr_close > curr_open:  # Green hammer (stronger)
                    return 'hammer', 0.8
                else:  # Red hammer
                    return 'hammer', 0.65

            # === BULLISH ENGULFING ===
            if (prev_close < prev_open and  # Previous red
                curr_close > curr_open and  # Current green
                curr_body > prev_body * self.engulfing_body_ratio and
                curr_close > prev_open and
                curr_open < prev_close):
                return 'bullish_engulfing', 0.85

            # === DOJI AT SUPPORT ===
            if curr_body <= curr_range * 0.1:  # Doji (body < 10% of range)
                if curr_low < prev_low:  # Made new low then reversed
                    return 'doji_reversal', 0.6

            # === PIN BAR ===
            if lower_wick >= curr_range * 0.6 and curr_body <= curr_range * 0.3:
                return 'pin_bar', 0.7

            # === SIMPLE REVERSAL ===
            if (curr_close > curr_open and  # Green candle
                curr_close > prev_close and  # Higher close
                curr_low < prev_low):  # Made lower low first (sweep)
                return 'reversal_candle', 0.55

        else:  # bearish
            # === SHOOTING STAR ===
            lower_wick = min(curr_open, curr_close) - curr_low
            upper_wick = curr_high - max(curr_open, curr_close)

            if upper_wick >= curr_body * self.min_wick_ratio and lower_wick <= curr_body * 0.5:
                if curr_close < curr_open:  # Red shooting star (stronger)
                    return 'shooting_star', 0.8
                else:
                    return 'shooting_star', 0.65

            # === BEARISH ENGULFING ===
            if (prev_close > prev_open and  # Previous green
                curr_close < curr_open and  # Current red
                curr_body > prev_body * self.engulfing_body_ratio and
                curr_close < prev_open and
                curr_open > prev_close):
                return 'bearish_engulfing', 0.85

            # === DOJI AT RESISTANCE ===
            if curr_body <= curr_range * 0.1:
                if curr_high > prev_high:
                    return 'doji_reversal', 0.6

            # === PIN BAR ===
            if upper_wick >= curr_range * 0.6 and curr_body <= curr_range * 0.3:
                return 'pin_bar', 0.7

            # === SIMPLE REVERSAL ===
            if (curr_close < curr_open and
                curr_close < prev_close and
                curr_high > prev_high):
                return 'reversal_candle', 0.55

        return None, 0

    def _create_entry_signal(self, direction: str, reason: str, confidence: float) -> Dict[str, Any]:
        """Create entry signal with dynamic sizing and ATR-based stops."""
        # Calculate position size with compound adjustment
        effective_capital = self.total_capital
        if self.compound_profits and self.stats.realized_pnl > 0:
            compound_mult = min(self.max_compound,
                              1 + (self.stats.realized_pnl / self.initial_capital) * self.profit_reinvest_ratio)
            effective_capital = self.initial_capital * compound_mult

        # Volatility-adjusted sizing
        size_mult = 1.0
        if self.current_atr_pct > 0:
            if self.current_atr_pct > self.high_vol_threshold:
                size_mult = 0.7  # Reduce size in high vol
            elif self.current_atr_pct < self.low_vol_threshold:
                size_mult = 1.2  # Increase size in low vol

        size_pct = min(self.max_size_pct, max(self.min_size_pct, self.size_pct * size_mult))
        size_usd = effective_capital * size_pct
        size_btc = size_usd / self.current_price

        # ATR-based stops
        if self.use_atr_stops and self.current_atr > 0:
            stop_distance = self.current_atr * self.stop_loss_atr_mult
            profit_distance = self.current_atr * self.take_profit_atr_mult
        else:
            stop_distance = self.current_price * self.stop_loss_pct
            profit_distance = self.current_price * self.take_profit_pct

        if direction == 'long':
            stop_loss = self.current_price - stop_distance
            take_profit = self.current_price + profit_distance
        else:
            stop_loss = self.current_price + stop_distance
            take_profit = self.current_price - profit_distance

        self.bars_since_trade = 0

        return {
            'action': 'buy' if direction == 'long' else 'short',
            'symbol': self.symbol,
            'price': self.current_price,
            'size': size_btc,
            'size_usd': size_usd,
            'leverage': int(self.leverage),
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'confidence': min(0.95, confidence),
            'strategy': 'LiquidationHuntScalper',
            'side': direction,
            'reason': f'Liq hunt {direction}: {reason} at ${self.current_price:,.0f} '
                     f'(RSI={self.current_rsi:.0f}, Vol={self.current_volume_ratio:.1f}x, '
                     f'Lev={int(self.leverage)}x)',
            # Metadata for tracking
            'pattern': reason,
            'rsi': self.current_rsi,
            'atr': self.current_atr,
            'volume_ratio': self.current_volume_ratio,
            'session': self.current_session,
            'liq_zone': self.long_liq_zone if direction == 'long' else self.short_liq_zone,
            'swing_point': self.local_high if direction == 'long' else self.local_low,
        }

    def _manage_position(self, pos: MarginPosition, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Manage open position with trailing stop support.
        """
        # Update unrealized PnL
        if pos.side == 'long':
            price_diff = self.current_price - pos.entry_price
            pos.unrealized_pnl = price_diff * pos.size * pos.leverage

            # Check stop loss
            if self.current_price <= pos.stop_loss:
                return self._create_exit_signal(pos, 'stop_loss')

            # Check take profit
            if self.current_price >= pos.take_profit:
                return self._create_exit_signal(pos, 'take_profit')

            # Trailing stop logic
            if self.use_trailing_stop and self.current_atr > 0:
                profit_in_atr = price_diff / self.current_atr
                if profit_in_atr >= self.trailing_activation_atr:
                    new_stop = self.current_price - (self.current_atr * self.trailing_distance_atr)
                    if new_stop > pos.stop_loss:
                        pos.stop_loss = new_stop

        else:  # short
            price_diff = pos.entry_price - self.current_price
            pos.unrealized_pnl = price_diff * pos.size * pos.leverage

            if self.current_price >= pos.stop_loss:
                return self._create_exit_signal(pos, 'stop_loss')

            if self.current_price <= pos.take_profit:
                return self._create_exit_signal(pos, 'take_profit')

            if self.use_trailing_stop and self.current_atr > 0:
                profit_in_atr = price_diff / self.current_atr
                if profit_in_atr >= self.trailing_activation_atr:
                    new_stop = self.current_price + (self.current_atr * self.trailing_distance_atr)
                    if new_stop < pos.stop_loss:
                        pos.stop_loss = new_stop

        return None

    def _create_exit_signal(self, pos: MarginPosition, exit_reason: str) -> Dict[str, Any]:
        """Create exit signal."""
        return {
            'action': 'close',
            'symbol': self.symbol,
            'price': self.current_price,
            'size': pos.size,
            'leverage': pos.leverage,
            'pnl': pos.unrealized_pnl,
            'confidence': 0.9,
            'strategy': 'LiquidationHuntScalper',
            'side': pos.side,
            'reason': f'Exit {pos.side} ({exit_reason}): PnL ${pos.unrealized_pnl:.2f}',
            'exit_reason': exit_reason,
            'position': pos,
            'entry_price': pos.entry_price,
        }

    def on_order_filled(self, order_info: Dict[str, Any]):
        """
        Process filled order - compatible with orchestrator interface.

        This method is called by the orchestrator when an order is executed.
        """
        action = order_info.get('action', '')
        fill_price = order_info.get('price', self.current_price)

        if action in ['buy', 'short']:
            # Opening a new position
            side = 'long' if action == 'buy' else 'short'
            size = order_info.get('amount', order_info.get('size', 0))
            leverage = order_info.get('leverage', self.leverage)

            # Get stop/take profit from signal or calculate
            stop_loss = order_info.get('stop_loss', 0)
            take_profit = order_info.get('take_profit', 0)

            if stop_loss == 0:
                stop_distance = self.current_atr * self.stop_loss_atr_mult if self.current_atr > 0 else fill_price * self.stop_loss_pct
                stop_loss = fill_price - stop_distance if side == 'long' else fill_price + stop_distance

            if take_profit == 0:
                profit_distance = self.current_atr * self.take_profit_atr_mult if self.current_atr > 0 else fill_price * self.take_profit_pct
                take_profit = fill_price + profit_distance if side == 'long' else fill_price - profit_distance

            pos = MarginPosition(
                entry_price=fill_price,
                size=size,
                side=side,
                leverage=leverage,
                entry_time=datetime.now(),
                stop_loss=stop_loss,
                take_profit=take_profit,
                margin_used=size * fill_price / leverage
            )
            self.positions.append(pos)
            self.stats.total_trades += 1

        elif action in ['close', 'sell', 'cover']:
            # Closing a position
            pnl = order_info.get('pnl', 0)

            # Find and remove the position
            pos = order_info.get('position')
            if pos and pos in self.positions:
                self.positions.remove(pos)
                pnl = pos.unrealized_pnl
            elif self.positions:
                # Close most recent position if not specified
                pos = self.positions.pop()
                pnl = pos.unrealized_pnl

            # Update stats
            self.stats.realized_pnl += pnl
            self.stats.total_pnl = self.stats.realized_pnl

            if pnl > 0:
                self.stats.winning_trades += 1
            else:
                self.stats.losing_trades += 1

            # Compound profits
            if self.compound_profits and pnl > 0:
                reinvest = pnl * self.profit_reinvest_ratio
                self.total_capital += reinvest
                self.stats.profit_reinvested += reinvest

            self.stats.cycles_completed += 1

    def fill_order(self, signal: Dict[str, Any], fill_price: float):
        """
        Legacy fill_order method - wraps on_order_filled for compatibility.
        """
        order_info = {
            'action': signal.get('action'),
            'price': fill_price,
            'size': signal.get('size'),
            'amount': signal.get('size'),
            'leverage': signal.get('leverage', self.leverage),
            'stop_loss': signal.get('stop_loss'),
            'take_profit': signal.get('take_profit'),
            'pnl': signal.get('pnl', 0),
            'position': signal.get('position'),
        }
        self.on_order_filled(order_info)

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive strategy status."""
        unrealized_pnl = sum(p.unrealized_pnl for p in self.positions)

        return {
            'strategy': 'LiquidationHuntScalper',
            'version': 'Phase29',

            # Price info
            'current_price': self.current_price,
            'local_high': self.local_high,
            'local_low': self.local_low,

            # Liquidation zones
            'long_liq_zone': self.long_liq_zone,
            'short_liq_zone': self.short_liq_zone,
            'zone_type': 'ATR-based' if self.use_atr_zones else 'Percentage',

            # Swing points
            'swing_highs_count': len(self.swing_highs),
            'swing_lows_count': len(self.swing_lows),

            # Indicators
            'current_atr': self.current_atr,
            'current_atr_pct': self.current_atr_pct * 100,
            'current_rsi': self.current_rsi,
            'current_volume_ratio': self.current_volume_ratio,

            # Leverage
            'leverage': int(self.leverage),
            'base_leverage': self.base_leverage,
            'dynamic_leverage': self.use_dynamic_leverage,

            # Session
            'session': self.current_session,

            # Positions
            'open_positions': len(self.positions),
            'max_positions': self.max_positions,
            'unrealized_pnl': unrealized_pnl,

            # Stats
            'total_pnl': self.stats.total_pnl,
            'realized_pnl': self.stats.realized_pnl,
            'total_trades': self.stats.total_trades,
            'winning_trades': self.stats.winning_trades,
            'losing_trades': self.stats.losing_trades,
            'win_rate': self.stats.winning_trades / max(self.stats.total_trades, 1) * 100,
            'cycles_completed': self.stats.cycles_completed,
            'profit_reinvested': self.stats.profit_reinvested,

            # Capital
            'total_capital': self.total_capital,
            'initial_capital': self.initial_capital,
            'compound_multiplier': self.total_capital / self.initial_capital,

            # Last signal
            'last_pattern': self.last_pattern,
            'bars_since_trade': self.bars_since_trade,
            'cooldown_bars': self.cooldown_bars,
        }
