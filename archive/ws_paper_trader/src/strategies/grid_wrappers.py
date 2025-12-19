"""
Phase 26: Grid Strategy Wrappers - Enhanced
Adapts grid-based strategies to the BaseStrategy interface for unified orchestrator.

Phase 26 Improvements:
- Smart initialization without skipping initial signals
- ATR-based dynamic grid spacing
- Multi-asset support (BTC + XRP)
- Proper position tracking sync with orchestrator
- Cycle-based recentering (not price-based)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from strategies.base_strategy import BaseStrategy
from strategies.grid_base import (
    ArithmeticGrid,
    GeometricGrid,
    RSIMeanReversionGrid,
    BBSqueezeGrid,
    TrendFollowingMargin,
    DualGridHedge,
    TimeWeightedGrid,
    LiquidationHuntScalper
)
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime


class GridStrategyWrapper(BaseStrategy):
    """
    Phase 26 Enhanced wrapper that adapts grid strategies to BaseStrategy interface.

    Key improvements:
    - Doesn't skip signals after initialization
    - Uses ATR for dynamic spacing
    - Multi-asset support
    - Proper position sync
    """

    grid_class = None  # Override in subclasses

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.symbol = config.get('symbol', 'BTC/USDT')

        # Phase 26: Dynamic range parameters
        self.range_pct = config.get('range_pct', 0.04)  # 4% above/below current price
        self.recenter_threshold = config.get('recenter_threshold', 0.02)  # 2% threshold

        # Phase 26: Smarter recentering control
        self.recenter_after_cycles = config.get('recenter_after_cycles', 5)  # Recenter after N cycles
        self.min_recenter_interval = config.get('min_recenter_interval', 3600)  # Min 1 hour between recenters
        self.last_recenter_time = None
        self.cycles_since_recenter = 0

        # Phase 26: ATR-based spacing
        self.use_atr_spacing = config.get('use_atr_spacing', True)
        self.atr_multiplier = config.get('atr_multiplier', 0.3)
        self.atr_period = config.get('atr_period', 14)

        # Phase 26: Multi-asset configuration
        self.secondary_symbol = config.get('secondary_symbol', 'XRP/USDT')
        self.secondary_allocation = config.get('secondary_allocation', 0.3)  # 30% to secondary
        self.secondary_grid = None

        # Phase 26: Compound settings
        self.compound_profits = config.get('compound_profits', True)
        self.max_compound = config.get('max_compound', 1.5)

        # Store config for grid reinitialization
        self.base_config = {
            'symbol': self.symbol,
            'num_grids': config.get('num_grids', 15),
            'total_capital': config.get('total_capital', 1000),
            'leverage': config.get('leverage', 1),
            'stop_loss': config.get('stop_loss'),
            'take_profit': config.get('take_profit'),
            'max_drawdown': config.get('max_drawdown', 0.10),
            'fee_rate': config.get('fee_rate', 0.001),
            'use_atr_spacing': self.use_atr_spacing,
            'atr_multiplier': self.atr_multiplier,
            'atr_period': self.atr_period,
            'compound_profits': self.compound_profits,
            'max_compound': self.max_compound,
        }
        self.base_config.update(self._get_strategy_params(config))

        # Grid will be initialized on first price update
        self.grid_strategy = None
        self.grid_center_price = 0.0
        self.pending_signals: List[Dict] = []
        self.last_price = 0.0
        self.initialized = False
        self.initialization_in_progress = False

    def _initialize_grid(self, current_price: float, atr: float = 0.0):
        """
        Initialize or reinitialize grid centered on current price.

        Phase 26: Includes ATR for dynamic spacing.
        """
        self.grid_center_price = current_price
        upper_price = current_price * (1 + self.range_pct)
        lower_price = current_price * (1 - self.range_pct)

        grid_config = {
            **self.base_config,
            'upper_price': upper_price,
            'lower_price': lower_price,
        }

        if self.grid_class:
            self.grid_strategy = self.grid_class(grid_config)

            # Set ATR if available
            if atr > 0:
                self.grid_strategy.current_atr = atr

            # Initialize grid with current price for proper buy/sell placement
            if hasattr(self.grid_strategy, '_setup_grid'):
                self.grid_strategy._setup_grid(current_price)

            self.initialized = True
            self.last_recenter_time = datetime.now()
            self.cycles_since_recenter = 0
            return True
        return False

    def _should_recenter_grid(self, current_price: float) -> bool:
        """
        Check if grid should be recentered.

        Phase 26: Uses multiple criteria:
        - Price deviation from center
        - Number of cycles completed
        - Time since last recenter
        """
        if not self.initialized or self.grid_center_price == 0:
            return True

        # Check if enough time has passed
        if self.last_recenter_time:
            time_since = (datetime.now() - self.last_recenter_time).total_seconds()
            if time_since < self.min_recenter_interval:
                return False

        # Check price deviation
        deviation = abs(current_price - self.grid_center_price) / self.grid_center_price
        if deviation > (self.range_pct + self.recenter_threshold):
            return True

        # Check cycles completed
        if self.grid_strategy and self.grid_strategy.stats.cycles_completed > 0:
            cycles_since = self.grid_strategy.stats.cycles_completed - self.cycles_since_recenter
            if cycles_since >= self.recenter_after_cycles:
                return True

        return False

    def _get_strategy_params(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Override in subclasses to add strategy-specific params."""
        return {}

    def _calculate_atr(self, df: pd.DataFrame) -> float:
        """Calculate ATR from dataframe."""
        if df is None or len(df) < self.atr_period + 1:
            return 0.0

        high = df['high'].iloc[-self.atr_period:]
        low = df['low'].iloc[-self.atr_period:]
        close = df['close'].iloc[-self.atr_period:]

        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.mean()

    def _convert_size_to_pct(self, size_asset: float, price: float) -> float:
        """
        Convert absolute asset size to percentage of capital.

        The orchestrator expects size as a percentage (0.0-1.0) of USDT balance,
        but grid strategies return size in absolute asset amounts (e.g., 0.00011 BTC).

        Args:
            size_asset: Absolute asset amount (e.g., 0.00011 BTC)
            price: Current price of the asset

        Returns:
            Size as percentage of capital (e.g., 0.05 for 5%)
        """
        if size_asset <= 0 or price <= 0:
            return 0.0

        # Convert to USD value
        size_usd = size_asset * price

        # Get total capital (use USDT balance as reference, default $2000)
        total_capital = self.base_config.get('total_capital', 1000)
        # Scale up to match typical orchestrator USDT balance (~2000)
        reference_capital = max(total_capital * 2, 2000)

        # Calculate percentage (clamp to reasonable range)
        size_pct = size_usd / reference_capital
        return max(0.01, min(0.25, size_pct))  # 1% to 25% range

    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Adapt grid update() to generate_signals() interface.

        Phase 26 Enhanced:
        - Doesn't skip signals after initialization
        - Uses ATR for dynamic spacing
        - Multi-asset signal generation
        - Proper position tracking
        """
        # Find BTC data for grid strategies
        btc_key = None
        xrp_key = None
        for key in data.keys():
            if 'BTC' in key.upper() and '_' not in key:  # Avoid 5m/15m keys
                btc_key = key
            if 'XRP' in key.upper() and '_' not in key:
                xrp_key = key

        if not btc_key or btc_key not in data:
            return self._hold_signal('No BTC data available')

        df = data[btc_key]
        if df is None or len(df) == 0:
            return self._hold_signal('Empty BTC data')

        current_price = df['close'].iloc[-1]
        self.last_price = current_price

        # Calculate ATR for dynamic spacing
        atr = self._calculate_atr(df) if self.use_atr_spacing else 0.0

        # Phase 26: Smart initialization - don't skip signals
        needs_init = self._should_recenter_grid(current_price)

        if needs_init:
            old_center = self.grid_center_price
            was_initialized = self.initialized

            self._initialize_grid(current_price, atr)

            if was_initialized and old_center > 0:
                # Track cycles since last recenter
                if self.grid_strategy:
                    self.cycles_since_recenter = self.grid_strategy.stats.cycles_completed

            # Phase 26: Continue to check for signals even after init
            # Only return hold if this is truly the first initialization
            if not was_initialized:
                reason = f'Grid initialized at ${current_price:,.0f} (±{self.range_pct*100:.0f}%)'
                return self._hold_signal(reason)

        if not self.grid_strategy:
            return self._hold_signal('Grid strategy not initialized')

        # Call the grid's update method
        try:
            signals = self.grid_strategy.update(current_price, df)
        except Exception as e:
            return self._hold_signal(f'Grid update error: {e}')

        # Phase 26: Also check secondary asset (XRP)
        secondary_signals = []
        if self.secondary_grid and xrp_key and xrp_key in data:
            xrp_df = data[xrp_key]
            if xrp_df is not None and len(xrp_df) > 0:
                xrp_price = xrp_df['close'].iloc[-1]
                try:
                    secondary_signals = self.secondary_grid.update(xrp_price, xrp_df)
                except:
                    pass

        # Combine signals - prioritize by confidence
        all_signals = []
        for sig in signals:
            sig['asset'] = 'BTC'
            sig['symbol'] = btc_key
            all_signals.append(sig)

        for sig in secondary_signals:
            sig['asset'] = 'XRP'
            sig['symbol'] = xrp_key
            all_signals.append(sig)

        if not all_signals:
            range_info = f'${self.grid_strategy.lower_price:,.0f}-${self.grid_strategy.upper_price:,.0f}'
            return self._hold_signal(f'No grid signals (price=${current_price:,.0f}, range={range_info})')

        # Sort by confidence and return highest
        all_signals.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        signal = all_signals[0]

        # Convert grid signal format to unified format
        action = signal.get('action', 'hold')
        if action in ['close_all', 'exit']:
            action = 'sell'

        # Convert absolute asset size to percentage for orchestrator
        signal_price = signal.get('price', current_price)
        size_pct = self._convert_size_to_pct(signal.get('size', 0), signal_price)

        return {
            'action': action,
            'symbol': signal.get('symbol', btc_key),
            'size': size_pct,
            'leverage': signal.get('leverage', self.grid_strategy.leverage),
            'confidence': signal.get('confidence', 0.7),
            'reason': signal.get('reason', 'Grid signal'),
            'strategy': self.name,
            'grid_level': signal.get('grid_level'),
            'order_id': signal.get('order_id'),
            'price': signal_price,
            'target_price': signal.get('target_sell_price', 0),
            'asset': signal.get('asset', 'BTC')
        }

    def on_order_filled(self, order: Dict[str, Any]) -> None:
        """
        Phase 26: Callback when order is filled - sync grid state.
        """
        if not self.grid_strategy:
            return

        grid_level = order.get('grid_level')
        fill_price = order.get('price', 0)
        order_id = order.get('order_id')

        if grid_level is not None and fill_price > 0:
            self.grid_strategy.fill_order(grid_level, fill_price, order_id=order_id)

    def _hold_signal(self, reason: str) -> Dict[str, Any]:
        """Create a hold signal."""
        return {
            'action': 'hold',
            'symbol': self.symbol,
            'confidence': 0.0,
            'reason': reason,
            'strategy': self.name
        }

    def update_model(self, data: Optional[Dict[str, pd.DataFrame]] = None) -> bool:
        """Grid strategies are rule-based, no model to update."""
        return True

    def initialize_secondary_grid(self, xrp_price: float = None):
        """
        Phase 26: Initialize secondary grid for XRP.
        """
        if self.secondary_allocation <= 0 or not self.grid_class:
            return False

        # Calculate capital allocation
        primary_capital = self.base_config.get('total_capital', 1000) * (1 - self.secondary_allocation)
        secondary_capital = self.base_config.get('total_capital', 1000) * self.secondary_allocation

        # Update primary grid capital
        if self.grid_strategy:
            self.grid_strategy.total_capital = primary_capital
            self.grid_strategy.initial_capital = primary_capital

        # Create secondary grid config for XRP
        if xrp_price and xrp_price > 0:
            xrp_range_pct = self.range_pct * 1.5  # Wider range for XRP volatility

            secondary_config = {
                **self.base_config,
                'symbol': self.secondary_symbol,
                'total_capital': secondary_capital,
                'upper_price': xrp_price * (1 + xrp_range_pct),
                'lower_price': xrp_price * (1 - xrp_range_pct),
                'num_grids': max(10, self.base_config.get('num_grids', 15) - 5),  # Fewer grids for XRP
            }

            self.secondary_grid = self.grid_class(secondary_config)
            if hasattr(self.secondary_grid, '_setup_grid'):
                self.secondary_grid._setup_grid(xrp_price)

            return True

        return False

    def get_status(self) -> Dict[str, Any]:
        """Get strategy status including grid state with Phase 26 enhancements."""
        base_status = super().get_status()

        if self.grid_strategy:
            grid_status = self.grid_strategy.get_status()
            base_status.update({
                'grid_levels': len(self.grid_strategy.grid_levels),
                'positions': len(self.grid_strategy.positions),
                'stats': {
                    'total_trades': self.grid_strategy.stats.total_trades,
                    'winning_trades': self.grid_strategy.stats.winning_trades,
                    'losing_trades': self.grid_strategy.stats.losing_trades,
                    'total_pnl': self.grid_strategy.stats.total_pnl,
                    'realized_pnl': self.grid_strategy.stats.realized_pnl,
                    'cycles_completed': self.grid_strategy.stats.cycles_completed,
                    'profit_reinvested': self.grid_strategy.stats.profit_reinvested,
                    'total_fees': self.grid_strategy.stats.total_fees,
                },
                'last_price': self.last_price,
                'grid_center': self.grid_center_price,
                'upper_price': self.grid_strategy.upper_price,
                'lower_price': self.grid_strategy.lower_price,
                'grid_spacing': getattr(self.grid_strategy, 'grid_spacing', 0),
                'current_atr': self.grid_strategy.current_atr,
                'compound_multiplier': grid_status.get('compound_multiplier', 1.0),
            })

        # Add secondary grid status
        if self.secondary_grid:
            secondary_status = self.secondary_grid.get_status()
            base_status['secondary_grid'] = {
                'symbol': self.secondary_symbol,
                'positions': len(self.secondary_grid.positions),
                'total_pnl': self.secondary_grid.stats.total_pnl,
                'cycles_completed': self.secondary_grid.stats.cycles_completed,
            }

        return base_status


class ArithmeticGridWrapper(GridStrategyWrapper):
    """
    Phase 26 Enhanced wrapper for ArithmeticGrid strategy.

    Features:
    - Dynamic ATR-based spacing
    - Multi-asset support (BTC + XRP)
    - Compound position sizing
    """
    grid_class = ArithmeticGrid
    name = 'grid_arithmetic'

    def _get_strategy_params(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Add arithmetic grid specific params."""
        return {
            'slippage_tolerance': config.get('slippage_tolerance', 0.005),  # 0.5% for live trading
            'min_profit_per_cycle': config.get('min_profit_per_cycle', 0.001),
        }


class GeometricGridWrapper(GridStrategyWrapper):
    """
    Phase 26 Enhanced wrapper for GeometricGrid strategy.

    Features:
    - Dynamic ATR-based grid ratio calculation
    - Multi-asset support (BTC + XRP)
    - Compound position sizing with extreme bias
    - Smart recentering based on cycles and price deviation
    """
    grid_class = GeometricGrid
    name = 'grid_geometric'

    def __init__(self, config: Dict[str, Any]):
        # Phase 26: GeometricGrid specific defaults
        # Use wider range for geometric grids (better for volatile markets)
        if 'range_pct' not in config:
            config['range_pct'] = 0.06  # ±6% (wider than arithmetic)

        # Geometric grids recenter less frequently (larger natural range)
        if 'recenter_after_cycles' not in config:
            config['recenter_after_cycles'] = 8  # More cycles before recenter

        super().__init__(config)

    def _get_strategy_params(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get geometric grid specific parameters.

        Phase 26 Enhanced with all new parameters.
        """
        return {
            # Core grid ratio (default 2% for crypto volatility)
            'grid_ratio': config.get('grid_ratio', 1.02),

            # Position sizing at extremes
            'size_multiplier': config.get('size_multiplier', 1.3),
            'extreme_multiplier': config.get('extreme_multiplier', 1.5),

            # Phase 26: ATR-based dynamic ratio
            'use_atr_ratio': config.get('use_atr_ratio', True),
            'atr_ratio_mult': config.get('atr_ratio_mult', 0.4),
            'min_grid_ratio': config.get('min_grid_ratio', 1.005),
            'max_grid_ratio': config.get('max_grid_ratio', 1.05),

            # Phase 26: Slippage and profit thresholds
            'slippage_tolerance': config.get('slippage_tolerance', 0.005),  # 0.5% for live trading
            'min_profit_per_cycle': config.get('min_profit_per_cycle', 0.002),  # 0.2% min

            # Phase 26: Compound settings
            'compound_profits': config.get('compound_profits', True),
            'max_compound': config.get('max_compound', 1.5),
            'profit_distribution': config.get('profit_distribution', {
                'reinvest': 0.7,  # 70% reinvested
                'realized': 0.3   # 30% taken
            }),

            # Phase 26: Recentering control
            'recenter_after_cycles': config.get('recenter_after_cycles', 8),
            'min_recenter_interval': config.get('min_recenter_interval', 3600),
        }

    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Generate signals with geometric grid enhancements.

        Phase 26: Includes multi-asset support and improved signal handling.
        """
        # Find data keys
        btc_key = None
        xrp_key = None
        for key in data.keys():
            if 'BTC' in key.upper() and '_' not in key:
                btc_key = key
            if 'XRP' in key.upper() and '_' not in key:
                xrp_key = key

        if not btc_key or btc_key not in data:
            return self._hold_signal('No BTC data available')

        df = data[btc_key]
        if df is None or len(df) == 0:
            return self._hold_signal('Empty BTC data')

        current_price = df['close'].iloc[-1]
        self.last_price = current_price

        # Calculate ATR for dynamic ratio
        atr = self._calculate_atr(df) if self.use_atr_spacing else 0.0

        # Initialize or recenter grid if needed
        if not self.initialized:
            self._initialize_grid(current_price, atr)
            # Continue to check signals even after init (don't return hold)

        # Check if grid strategy needs recentering
        if self.grid_strategy and hasattr(self.grid_strategy, 'should_recenter'):
            if self.grid_strategy.should_recenter(current_price):
                self.grid_strategy.recenter_grid(current_price)

        if not self.grid_strategy:
            return self._hold_signal('Grid strategy not initialized')

        # Call the grid's update method
        try:
            signals = self.grid_strategy.update(current_price, df)
        except Exception as e:
            return self._hold_signal(f'Grid update error: {e}')

        # Phase 26: Also check secondary asset (XRP) if configured
        secondary_signals = []
        if self.secondary_grid and xrp_key and xrp_key in data:
            xrp_df = data[xrp_key]
            if xrp_df is not None and len(xrp_df) > 0:
                xrp_price = xrp_df['close'].iloc[-1]
                try:
                    secondary_signals = self.secondary_grid.update(xrp_price, xrp_df)
                except:
                    pass

        # Combine signals
        all_signals = []
        for sig in signals:
            sig['asset'] = 'BTC'
            sig['symbol'] = btc_key
            all_signals.append(sig)

        for sig in secondary_signals:
            sig['asset'] = 'XRP'
            sig['symbol'] = xrp_key
            all_signals.append(sig)

        if not all_signals:
            # Include grid status in hold reason
            status_info = ""
            if self.grid_strategy:
                effective_ratio = self.grid_strategy._calculate_effective_ratio()
                status_info = f", ratio={effective_ratio:.4f}"
                if hasattr(self.grid_strategy, 'grid_levels') and self.grid_strategy.grid_levels:
                    status_info += f", levels={len(self.grid_strategy.grid_levels)}"

            range_info = f'${self.grid_strategy.lower_price:,.0f}-${self.grid_strategy.upper_price:,.0f}'
            return self._hold_signal(f'No grid signals (price=${current_price:,.0f}, range={range_info}{status_info})')

        # Sort by confidence and return highest
        all_signals.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        signal = all_signals[0]

        # Convert grid signal format to unified format
        action = signal.get('action', 'hold')
        if action in ['close_all', 'exit']:
            action = 'sell'

        # Convert absolute asset size to percentage for orchestrator
        signal_price = signal.get('price', current_price)
        size_pct = self._convert_size_to_pct(signal.get('size', 0), signal_price)

        return {
            'action': action,
            'symbol': signal.get('symbol', btc_key),
            'size': size_pct,
            'leverage': signal.get('leverage', self.grid_strategy.leverage),
            'confidence': signal.get('confidence', 0.7),
            'reason': signal.get('reason', 'Geometric grid signal'),
            'strategy': self.name,
            'grid_level': signal.get('grid_level'),
            'order_id': signal.get('order_id'),
            'price': signal_price,
            'target_price': signal.get('target_sell_price', 0),
            'asset': signal.get('asset', 'BTC'),
            'effective_ratio': self.grid_strategy._calculate_effective_ratio() if self.grid_strategy else 1.0,
        }

    def get_status(self) -> Dict[str, Any]:
        """Get strategy status with geometric-specific metrics."""
        base_status = super().get_status()

        if self.grid_strategy:
            # Add geometric-specific metrics
            base_status.update({
                'grid_ratio': self.grid_strategy.grid_ratio,
                'effective_ratio': self.grid_strategy._calculate_effective_ratio(),
                'size_multiplier': self.grid_strategy.size_multiplier,
                'extreme_multiplier': getattr(self.grid_strategy, 'extreme_multiplier', 1.0),
                'use_atr_ratio': getattr(self.grid_strategy, 'use_atr_ratio', False),
            })

        return base_status


class RSIMeanReversionGridWrapper(GridStrategyWrapper):
    """
    Phase 26 Enhanced wrapper for RSIMeanReversionGrid strategy.

    Features:
    - RSI as confidence modifier (not hard filter)
    - Dynamic ATR-based grid spacing
    - Adaptive RSI zones based on volatility
    - Compound position sizing with profit reinvestment
    - Smart recentering based on cycles and price deviation
    - RSI extreme multiplier for position sizing
    """
    grid_class = RSIMeanReversionGrid
    name = 'grid_rsi_reversion'

    def __init__(self, config: Dict[str, Any]):
        # Phase 26: RSI-appropriate defaults
        # Wider range for mean reversion (RSI extremes imply bigger moves)
        if 'range_pct' not in config:
            config['range_pct'] = 0.05  # 5% range (wider than arithmetic)

        # Fewer cycles before recenter (RSI strategies want fresh levels)
        if 'recenter_after_cycles' not in config:
            config['recenter_after_cycles'] = 4

        # Enable ATR spacing by default for RSI strategy
        if 'use_atr_spacing' not in config:
            config['use_atr_spacing'] = True

        super().__init__(config)

        # Track RSI-specific metrics
        self.rsi_signals_count = {'oversold': 0, 'neutral': 0, 'overbought': 0}

    def _get_strategy_params(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get RSI strategy specific parameters.

        Phase 26 Enhanced with all new parameters.
        """
        return {
            # Core RSI parameters (relaxed from original)
            'rsi_period': config.get('rsi_period', 14),
            'rsi_oversold': config.get('rsi_oversold', 30),  # Relaxed from 25
            'rsi_overbought': config.get('rsi_overbought', 70),  # Relaxed from 75

            # Phase 26: RSI behavior mode
            # 'confidence_only' = RSI modifies confidence, doesn't block trades (recommended)
            # 'filter' = RSI must be in zone to trade (legacy behavior)
            'rsi_mode': config.get('rsi_mode', 'confidence_only'),

            # Phase 26: Adaptive RSI zones
            'use_adaptive_rsi': config.get('use_adaptive_rsi', True),
            'rsi_zone_expansion': config.get('rsi_zone_expansion', 5),

            # Phase 26: Position sizing at RSI extremes
            'rsi_extreme_multiplier': config.get('rsi_extreme_multiplier', 1.3),

            # Phase 26: ATR-based dynamic spacing
            'use_atr_spacing': config.get('use_atr_spacing', True),
            'atr_multiplier': config.get('atr_multiplier', 0.4),
            'atr_period': config.get('atr_period', 14),

            # Phase 26: Recentering control
            'recenter_after_cycles': config.get('recenter_after_cycles', 4),
            'min_recenter_interval': config.get('min_recenter_interval', 3600),

            # Phase 26: Slippage handling
            'slippage_tolerance': config.get('slippage_tolerance', 0.005),  # 0.5% for live trading
            'min_profit_per_cycle': config.get('min_profit_per_cycle', 0.001),

            # Phase 26: Compound settings
            'compound_profits': config.get('compound_profits', True),
            'max_compound': config.get('max_compound', 1.4),
            'profit_distribution': config.get('profit_distribution', {
                'reinvest': 0.6,   # 60% reinvested
                'realized': 0.4    # 40% taken as profit
            }),
        }

    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Generate signals with RSI-enhanced grid logic.

        Phase 26: Includes RSI zone tracking and multi-asset support.
        """
        # Find data keys
        btc_key = None
        xrp_key = None
        for key in data.keys():
            if 'BTC' in key.upper() and '_' not in key:
                btc_key = key
            if 'XRP' in key.upper() and '_' not in key:
                xrp_key = key

        if not btc_key or btc_key not in data:
            return self._hold_signal('No BTC data available')

        df = data[btc_key]
        if df is None or len(df) == 0:
            return self._hold_signal('Empty BTC data')

        current_price = df['close'].iloc[-1]
        self.last_price = current_price

        # Calculate ATR for dynamic spacing
        atr = self._calculate_atr(df) if self.use_atr_spacing else 0.0

        # Initialize grid if needed
        if not self.initialized:
            self._initialize_grid(current_price, atr)

        # Check if grid strategy needs recentering
        if self.grid_strategy and hasattr(self.grid_strategy, 'should_recenter'):
            if self.grid_strategy.should_recenter(current_price):
                self.grid_strategy.recenter_grid(current_price)

        if not self.grid_strategy:
            return self._hold_signal('Grid strategy not initialized')

        # Call the grid's update method
        try:
            signals = self.grid_strategy.update(current_price, df)
        except Exception as e:
            return self._hold_signal(f'Grid update error: {e}')

        # Phase 26: Track RSI zone signals
        for sig in signals:
            rsi_zone = sig.get('rsi_zone', 'neutral')
            self.rsi_signals_count[rsi_zone] = self.rsi_signals_count.get(rsi_zone, 0) + 1

        # Phase 26: Also check secondary asset (XRP) if configured
        secondary_signals = []
        if self.secondary_grid and xrp_key and xrp_key in data:
            xrp_df = data[xrp_key]
            if xrp_df is not None and len(xrp_df) > 0:
                xrp_price = xrp_df['close'].iloc[-1]
                try:
                    secondary_signals = self.secondary_grid.update(xrp_price, xrp_df)
                except:
                    pass

        # Combine signals
        all_signals = []
        for sig in signals:
            sig['asset'] = 'BTC'
            sig['symbol'] = btc_key
            all_signals.append(sig)

        for sig in secondary_signals:
            sig['asset'] = 'XRP'
            sig['symbol'] = xrp_key
            all_signals.append(sig)

        if not all_signals:
            # Include RSI info in hold reason
            rsi_info = ""
            if self.grid_strategy:
                rsi = self.grid_strategy.current_rsi
                rsi_zone = self.grid_strategy.get_status().get('rsi_zone', 'neutral')
                rsi_info = f", RSI={rsi:.1f} ({rsi_zone})"

            range_info = f'${self.grid_strategy.lower_price:,.0f}-${self.grid_strategy.upper_price:,.0f}'
            return self._hold_signal(f'No grid signals (price=${current_price:,.0f}, range={range_info}{rsi_info})')

        # Sort by confidence and return highest
        all_signals.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        signal = all_signals[0]

        # Convert grid signal format to unified format
        action = signal.get('action', 'hold')
        if action in ['close_all', 'exit']:
            action = 'sell'

        # Convert absolute asset size to percentage for orchestrator
        signal_price = signal.get('price', current_price)
        size_pct = self._convert_size_to_pct(signal.get('size', 0), signal_price)

        return {
            'action': action,
            'symbol': signal.get('symbol', btc_key),
            'size': size_pct,
            'leverage': signal.get('leverage', self.grid_strategy.leverage),
            'confidence': signal.get('confidence', 0.5),
            'reason': signal.get('reason', 'RSI grid signal'),
            'strategy': self.name,
            'grid_level': signal.get('grid_level'),
            'order_id': signal.get('order_id'),
            'price': signal_price,
            'target_price': signal.get('target_sell_price', 0),
            'asset': signal.get('asset', 'BTC'),
            'rsi': signal.get('rsi', 50),
            'rsi_zone': signal.get('rsi_zone', 'neutral'),
        }

    def get_status(self) -> Dict[str, Any]:
        """Get strategy status with RSI-specific metrics."""
        base_status = super().get_status()

        if self.grid_strategy:
            grid_status = self.grid_strategy.get_status()
            base_status.update({
                # RSI metrics
                'current_rsi': grid_status.get('current_rsi', 50),
                'rsi_oversold': grid_status.get('rsi_oversold', 30),
                'rsi_overbought': grid_status.get('rsi_overbought', 70),
                'adaptive_oversold': grid_status.get('adaptive_oversold', 30),
                'adaptive_overbought': grid_status.get('adaptive_overbought', 70),
                'rsi_zone': grid_status.get('rsi_zone', 'neutral'),
                'rsi_mode': grid_status.get('rsi_mode', 'confidence_only'),
                'use_adaptive_rsi': grid_status.get('use_adaptive_rsi', True),
                'rsi_extreme_multiplier': grid_status.get('rsi_extreme_multiplier', 1.3),

                # Signal tracking
                'rsi_signals_count': self.rsi_signals_count,

                # Recentering
                'cycles_since_recenter': grid_status.get('cycles_since_recenter', 0),
            })

        return base_status


class BBSqueezeGridWrapper(GridStrategyWrapper):
    """
    Wrapper for BBSqueezeGrid (TTM Squeeze Enhanced) strategy.

    Phase 27 Enhanced with:
    - Keltner Channel parameters for TTM Squeeze detection
    - Momentum histogram settings
    - Multi-level squeeze classification
    - ATR-based risk management
    """
    grid_class = BBSqueezeGrid
    name = 'grid_bb_squeeze'

    def _get_strategy_params(self, config: Dict[str, Any]) -> Dict[str, Any]:
        return {
            # Bollinger Band parameters
            'bb_period': config.get('bb_period', 20),
            'bb_std': config.get('bb_std', 2.0),

            # Keltner Channel parameters (TTM Squeeze)
            'kc_period': config.get('kc_period', 20),
            'kc_atr_mult_high': config.get('kc_atr_mult_high', 1.0),   # Tightest squeeze
            'kc_atr_mult_mid': config.get('kc_atr_mult_mid', 1.5),     # Standard squeeze
            'kc_atr_mult_low': config.get('kc_atr_mult_low', 2.0),     # Loose squeeze

            # Momentum parameters
            'momentum_period': config.get('momentum_period', 12),

            # Squeeze detection
            'squeeze_min_bars': config.get('squeeze_min_bars', 6),
            'squeeze_confirmation': config.get('squeeze_confirmation', True),

            # Breakout parameters
            'breakout_size_pct': config.get('breakout_size_pct', 0.15),
            'breakout_leverage_mult': config.get('breakout_leverage_mult', 2.0),
            'max_breakout_leverage': config.get('max_breakout_leverage', 5),

            # ATR-based risk management
            'stop_loss_atr_mult': config.get('stop_loss_atr_mult', 1.5),
            'take_profit_atr_mult': config.get('take_profit_atr_mult', 3.0),
        }

    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Generate signals with enhanced squeeze status reporting.
        """
        signal = super().generate_signals(data)

        # Add squeeze state to signal for logging
        if self.grid_strategy:
            status = self.grid_strategy.get_status()
            signal['squeeze_info'] = {
                'in_squeeze': status.get('in_squeeze', False),
                'squeeze_level': status.get('squeeze_level', 'none'),
                'squeeze_bars': status.get('squeeze_bars', 0),
                'momentum': status.get('momentum', 0),
                'momentum_rising': status.get('momentum_rising', False),
                'squeeze_direction': status.get('squeeze_direction', 'neutral'),
                'bb_width_percentile': status.get('bb_width_percentile', 0),
            }

        return signal


class TrendFollowingMarginWrapper(GridStrategyWrapper):
    """
    Phase 27 Enhanced: Wrapper for TrendFollowingMargin strategy.

    Exposes all new parameters including:
    - Dynamic trendline calculation
    - ATR-based risk management
    - Multi-position grid entries
    - Volatility-adjusted leverage
    - Short position support
    """
    grid_class = TrendFollowingMargin
    name = 'grid_trend_margin'

    def __init__(self, config: Dict[str, Any]):
        # Phase 27: Wider range for trend margin strategy
        if 'range_pct' not in config:
            config['range_pct'] = 0.06  # 6% range

        # Phase 27: Fewer recenters for trend-following (let trends run)
        if 'recenter_after_cycles' not in config:
            config['recenter_after_cycles'] = 10

        super().__init__(config)

    def _get_strategy_params(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get all strategy parameters with Phase 27 enhancements.

        Exposes full parameter set for tuning and experimentation.
        """
        return {
            # ===== Dynamic Trendline Parameters =====
            'use_dynamic_trendline': config.get('use_dynamic_trendline', True),
            'trendline_lookback': config.get('trendline_lookback', 72),  # Hours
            'trendline_recalc_interval': config.get('trendline_recalc_interval', 4),  # Hours
            'min_swing_points': config.get('min_swing_points', 3),

            # Fallback static trendline (used if dynamic fails)
            'trendline_start_price': config.get('trendline_start_price', 0),  # 0 = auto
            'trendline_slope': config.get('trendline_slope', 0),

            # ===== Grid Entry Parameters =====
            'num_grid_levels': config.get('num_grid_levels', 5),
            'grid_spacing_pct': config.get('grid_spacing_pct', 0.015),  # 1.5%
            'max_positions': config.get('max_positions', 3),

            # ===== Entry Conditions (Relaxed) =====
            'entry_tolerance': config.get('entry_tolerance', 0.01),  # 1% (relaxed from 0.3%)
            'rsi_threshold': config.get('rsi_threshold', 40),  # Relaxed from 35
            'rsi_oversold': config.get('rsi_oversold', 35),
            'rsi_overbought': config.get('rsi_overbought', 65),
            'entry_mode': config.get('entry_mode', 'or'),  # 'or' or 'and'

            # ===== ATR-Based Risk Management =====
            'use_atr_stops': config.get('use_atr_stops', True),
            'atr_period': config.get('atr_period', 14),
            'stop_loss_atr_mult': config.get('stop_loss_atr_mult', 2.0),
            'take_profit_atr_mult': config.get('take_profit_atr_mult', 3.0),

            # Fallback fixed percentages
            'stop_loss_pct': config.get('stop_loss_pct', 0.02),  # 2%
            'take_profit_1': config.get('take_profit_1', 0.025),  # 2.5%
            'take_profit_2': config.get('take_profit_2', 0.04),  # 4%
            'trailing_stop_pct': config.get('trailing_stop_pct', 0.012),  # 1.2%

            # ===== Dynamic Leverage =====
            'use_dynamic_leverage': config.get('use_dynamic_leverage', True),
            'max_leverage': config.get('max_leverage', 5),
            'min_leverage': config.get('min_leverage', 1),
            'high_vol_threshold': config.get('high_vol_threshold', 0.03),  # 3% ATR
            'low_vol_threshold': config.get('low_vol_threshold', 0.015),  # 1.5% ATR

            # ===== Direction & Shorts =====
            'allow_shorts': config.get('allow_shorts', True),

            # ===== Position Sizing =====
            'size_pct': config.get('size_pct', 0.08),  # 8% per trade

            # ===== Fees & Slippage =====
            'fee_rate': config.get('fee_rate', 0.001),
            'slippage_tolerance': config.get('slippage_tolerance', 0.005),  # 0.5% for live trading
        }

    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Generate signals with enhanced status reporting.

        Phase 27: Includes trendline, ATR, and trend direction info.
        """
        # Find BTC data
        btc_key = None
        for key in data.keys():
            if 'BTC' in key.upper() and '_' not in key:
                btc_key = key
                break

        if not btc_key or btc_key not in data:
            return self._hold_signal('No BTC data available')

        df = data[btc_key]
        if df is None or len(df) == 0:
            return self._hold_signal('Empty BTC data')

        current_price = df['close'].iloc[-1]
        self.last_price = current_price

        # Initialize or update grid
        if not self.initialized:
            # For TrendFollowingMargin, initialization happens in update()
            self.initialized = True

        if not self.grid_strategy:
            # Initialize the underlying strategy
            self._initialize_grid(current_price, 0)

        if not self.grid_strategy:
            return self._hold_signal('Grid strategy not initialized')

        # Call the strategy's update method
        try:
            signals = self.grid_strategy.update(current_price, df)
        except Exception as e:
            return self._hold_signal(f'Strategy update error: {e}')

        if not signals:
            # Include detailed status in hold reason
            status = self.grid_strategy.get_status()
            trendline = status.get('trendline', 0)
            distance = status.get('distance_to_trendline_pct', 0)
            rsi = status.get('current_rsi', 50)
            trend = status.get('trend_direction', 'unknown')
            leverage = status.get('leverage', 1)
            positions = status.get('open_positions', 0)

            reason = (f'No signal | Price=${current_price:,.0f}, '
                     f'Trend={trend}, RSI={rsi:.1f}, '
                     f'Trendline=${trendline:,.0f} ({distance:+.1f}%), '
                     f'Lev={leverage}x, Pos={positions}')

            return self._hold_signal(reason)

        # Return the highest confidence signal
        signals.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        signal = signals[0]

        action = signal.get('action', 'hold')
        if action in ['close_all', 'exit']:
            action = 'sell'

        # Convert absolute asset size to percentage for orchestrator
        signal_price = signal.get('price', current_price)
        size_pct = self._convert_size_to_pct(signal.get('size', 0), signal_price)

        return {
            'action': action,
            'symbol': signal.get('symbol', btc_key),
            'size': size_pct,
            'leverage': signal.get('leverage', self.grid_strategy.leverage),
            'confidence': signal.get('confidence', 0.5),
            'reason': signal.get('reason', 'Trend margin signal'),
            'strategy': self.name,
            'price': signal_price,
            'stop_loss': signal.get('stop_loss', 0),
            'take_profit': signal.get('take_profit', 0),
            'side': signal.get('side', 'long'),
            'trendline': signal.get('trendline', 0),
            'atr': signal.get('atr', 0),
            'rsi': signal.get('rsi', 50),
        }

    def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive strategy status.

        Note: TrendFollowingMargin doesn't have traditional grid attributes,
        so we build status directly rather than calling parent.
        """
        # Build status without calling parent (which expects grid attributes)
        base_status = {
            'name': self.name,
            'symbol': self.symbol,
            'initialized': self.initialized,
            'last_price': self.last_price,
        }

        if self.grid_strategy:
            grid_status = self.grid_strategy.get_status()
            base_status.update({
                # Trendline info
                'trendline': grid_status.get('trendline', 0),
                'trendline_type': grid_status.get('trendline_type', 'unknown'),
                'trendline_slope': grid_status.get('trendline_slope', 0),
                'distance_to_trendline_pct': grid_status.get('distance_to_trendline_pct', 0),

                # Indicators
                'current_rsi': grid_status.get('current_rsi', 50),
                'current_atr': grid_status.get('current_atr', 0),
                'current_atr_pct': grid_status.get('current_atr_pct', 0),
                'trend_direction': grid_status.get('trend_direction', 'unknown'),

                # Positions
                'open_positions': grid_status.get('open_positions', 0),
                'max_positions': grid_status.get('max_positions', 3),
                'unrealized_pnl': grid_status.get('unrealized_pnl', 0),

                # Stats
                'total_pnl': grid_status.get('total_pnl', 0),
                'realized_pnl': grid_status.get('realized_pnl', 0),
                'total_trades': grid_status.get('total_trades', 0),
                'winning_trades': grid_status.get('winning_trades', 0),
                'win_rate': grid_status.get('win_rate', 0),

                # Leverage
                'leverage': grid_status.get('leverage', 1),
                'base_leverage': grid_status.get('base_leverage', 1),

                # Grid levels
                'grid_levels': grid_status.get('grid_levels', []),
                'swing_lows_count': grid_status.get('swing_lows_count', 0),

                # Mode
                'entry_mode': grid_status.get('entry_mode', 'or'),
                'use_atr_stops': grid_status.get('use_atr_stops', True),
            })

        return base_status


class DualGridHedgeWrapper(GridStrategyWrapper):
    """
    Phase 28 Enhanced: Wrapper for DualGridHedge strategy.

    Exposes all parameters for the enhanced dual-grid with margin hedge strategy:
    - ATR-based dynamic hedge triggers
    - Correct stop loss/take profit placement
    - Trailing stops for hedges
    - Volatility-adjusted position sizing
    - Session-aware trigger adjustments
    - Grid recentering after hedge closes
    - Compound profit reinvestment
    - Comprehensive statistics tracking
    """
    grid_class = DualGridHedge
    name = 'grid_dual_hedge'

    def __init__(self, config: Dict[str, Any]):
        # Phase 28: Appropriate defaults for dual hedge strategy
        if 'range_pct' not in config:
            config['range_pct'] = 0.05  # 5% range (wider for hedge strategy)

        # Fewer recenters - let hedges run
        if 'recenter_after_cycles' not in config:
            config['recenter_after_cycles'] = 10

        super().__init__(config)

    def _get_strategy_params(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get all Phase 28 enhanced parameters for DualGridHedge.
        """
        return {
            # ===== Capital Allocation =====
            'grid_allocation': config.get('grid_allocation', 0.80),
            'hedge_allocation': config.get('hedge_allocation', 0.20),

            # ===== Grid Parameters =====
            'num_grids': config.get('num_grids', 18),
            'fee_rate': config.get('fee_rate', 0.001),

            # ===== ATR-Based Dynamic Triggers =====
            'use_atr_triggers': config.get('use_atr_triggers', True),
            'atr_period': config.get('atr_period', 14),
            'atr_trigger_mult': config.get('atr_trigger_mult', 1.5),

            # Fallback static triggers
            'static_trigger_pct': config.get('static_trigger_pct', 0.015),
            'long_hedge_trigger': config.get('long_hedge_trigger', 0),
            'short_hedge_trigger': config.get('short_hedge_trigger', 0),

            # ===== Hedge Leverage =====
            'hedge_leverage': config.get('hedge_leverage', 2),
            'max_hedge_leverage': config.get('max_hedge_leverage', 3),
            'min_hedge_leverage': config.get('min_hedge_leverage', 1),

            # ===== Stop Loss / Take Profit =====
            'stop_loss_atr_mult': config.get('stop_loss_atr_mult', 2.0),
            'take_profit_atr_mult': config.get('take_profit_atr_mult', 3.0),
            'stop_loss_pct': config.get('stop_loss_pct', 0.02),
            'take_profit_pct': config.get('take_profit_pct', 0.03),

            # ===== Trailing Stop =====
            'use_trailing_stop': config.get('use_trailing_stop', True),
            'trailing_activation_atr': config.get('trailing_activation_atr', 1.5),
            'trailing_distance_atr': config.get('trailing_distance_atr', 1.0),

            # ===== Volatility-Adjusted Sizing =====
            'use_vol_sizing': config.get('use_vol_sizing', True),
            'base_size_pct': config.get('base_size_pct', 0.5),
            'min_size_pct': config.get('min_size_pct', 0.25),
            'max_size_pct': config.get('max_size_pct', 0.75),
            'target_vol': config.get('target_vol', 0.02),

            # ===== Grid Recentering =====
            'recenter_after_hedge': config.get('recenter_after_hedge', True),
            'recenter_threshold': config.get('recenter_threshold', 0.03),
            'min_recenter_interval': config.get('min_recenter_interval', 3600),

            # ===== Session Awareness =====
            'use_session_adjustment': config.get('use_session_adjustment', True),
            'session_multipliers': config.get('session_multipliers', {
                'asia': 0.8,
                'europe': 1.0,
                'us': 1.2,
                'overnight': 0.9
            }),

            # ===== Compound Profits =====
            'compound_profits': config.get('compound_profits', True),
            'profit_reinvest_ratio': config.get('profit_reinvest_ratio', 0.6),
        }

    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Generate signals with Phase 28 enhanced status reporting.
        """
        # Find BTC data
        btc_key = None
        for key in data.keys():
            if 'BTC' in key.upper() and '_' not in key:
                btc_key = key
                break

        if not btc_key or btc_key not in data:
            return self._hold_signal('No BTC data available')

        df = data[btc_key]
        if df is None or len(df) == 0:
            return self._hold_signal('Empty BTC data')

        current_price = df['close'].iloc[-1]
        self.last_price = current_price

        # Initialize grid if needed
        if not self.initialized:
            atr = self._calculate_atr(df)
            self._initialize_grid(current_price, atr)

        if not self.grid_strategy:
            return self._hold_signal('Grid strategy not initialized')

        # Call the strategy's update method
        try:
            signals = self.grid_strategy.update(current_price, df)
        except Exception as e:
            return self._hold_signal(f'Strategy update error: {e}')

        if not signals:
            # Include comprehensive status in hold reason
            status = self.grid_strategy.get_status()
            hedge_status = status.get('hedge', {})
            triggers = status.get('triggers', {})

            hedge_info = ""
            if hedge_status.get('active'):
                hedge_info = f", Hedge: {hedge_status.get('side')} @ ${hedge_status.get('entry_price', 0):,.0f}"
            else:
                hedge_info = f", Triggers: ${triggers.get('short', 0):,.0f}-${triggers.get('long', 0):,.0f}"

            grid_info = status.get('grid', {})
            session = status.get('session', 'unknown')

            reason = (f'No signal | Price=${current_price:,.0f}, '
                     f"Grid=${grid_info.get('lower', 0):,.0f}-${grid_info.get('upper', 0):,.0f}, "
                     f"ATR=${triggers.get('atr', 0):.0f}, Session={session}{hedge_info}")

            return self._hold_signal(reason)

        # Return the highest confidence signal
        signals.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        signal = signals[0]

        action = signal.get('action', 'hold')
        if action in ['close_all', 'exit']:
            action = 'sell'

        # Convert absolute asset size to percentage for orchestrator
        signal_price = signal.get('price', current_price)
        size_pct = self._convert_size_to_pct(signal.get('size', 0), signal_price)

        return {
            'action': action,
            'symbol': signal.get('symbol', btc_key),
            'size': size_pct,
            'leverage': signal.get('leverage', self.grid_strategy.hedge_leverage),
            'confidence': signal.get('confidence', 0.7),
            'reason': signal.get('reason', 'Dual grid hedge signal'),
            'strategy': self.name,
            'component': signal.get('component', 'grid'),
            'grid_level': signal.get('grid_level'),
            'price': signal_price,
            'stop_loss': signal.get('stop_loss', 0),
            'take_profit': signal.get('take_profit', 0),
            'session': signal.get('session', 'unknown'),
            'atr': signal.get('atr', 0),
        }

    def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status with Phase 28 enhancements.

        Note: DualGridHedge has its own stats structure, so we build
        status directly rather than calling parent.
        """
        base_status = {
            'name': self.name,
            'version': 'Phase28',
            'symbol': self.symbol,
            'initialized': self.initialized,
            'last_price': self.last_price,
        }

        if self.grid_strategy:
            strategy_status = self.grid_strategy.get_status()
            base_status.update({
                # Capital
                'capital': strategy_status.get('capital', {}),

                # Grid info
                'grid': strategy_status.get('grid', {}),

                # Hedge info
                'hedge': strategy_status.get('hedge', {}),

                # Triggers
                'triggers': strategy_status.get('triggers', {}),

                # Risk metrics
                'risk': strategy_status.get('risk', {}),

                # Session
                'session': strategy_status.get('session', 'unknown'),

                # Overall stats
                'total_trades': strategy_status.get('total_trades', 0),
                'winning_trades': strategy_status.get('winning_trades', 0),
                'losing_trades': strategy_status.get('losing_trades', 0),
                'win_rate': strategy_status.get('win_rate', 0),
                'total_pnl': strategy_status.get('total_pnl', 0),
            })

        return base_status


class TimeWeightedGridWrapper(GridStrategyWrapper):
    """
    Phase 27 Enhanced: Wrapper for TimeWeightedGrid strategy.

    Exposes all new parameters including:
    - Corrected session hours with overlap detection
    - ATR-based dynamic sizing
    - TWAP-style order execution
    - Weekend/holiday low-liquidity handling
    - Day-of-week volatility awareness
    - Compound profit support
    """
    grid_class = TimeWeightedGrid
    name = 'grid_time_weighted'

    def __init__(self, config: Dict[str, Any]):
        # Phase 27: TimeWeightedGrid specific defaults
        if 'range_pct' not in config:
            config['range_pct'] = 0.04  # 4% range

        # Enable ATR adjustment by default
        if 'use_atr_spacing' not in config:
            config['use_atr_spacing'] = True

        super().__init__(config)

        # Track session transitions
        self.session_history: List[Dict[str, Any]] = []
        self.last_session = None

    def _get_strategy_params(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get TimeWeightedGrid specific parameters.

        Phase 27 Enhanced with all new parameters.
        """
        return {
            # ===== Session Configuration =====
            'session_configs': config.get('session_configs', None),  # Use defaults if not specified

            # ===== Day-of-Week Multipliers =====
            'day_multipliers': config.get('day_multipliers', None),  # Use defaults if not specified

            # ===== ATR-Based Dynamic Adjustment =====
            'use_atr_adjustment': config.get('use_atr_adjustment', True),
            'atr_period': config.get('atr_period', 14),
            'target_atr_pct': config.get('target_atr_pct', 0.02),  # 2% baseline

            # ===== TWAP Execution Settings =====
            'use_twap': config.get('use_twap', True),
            'twap_slice_interval': config.get('twap_slice_interval', 300),  # 5 minutes
            'twap_randomize_pct': config.get('twap_randomize_pct', 0.1),  # ±10%

            # ===== Weekend/Holiday Handling =====
            'reduce_weekend_size': config.get('reduce_weekend_size', True),
            'weekend_size_mult': config.get('weekend_size_mult', 0.5),
            'holidays_utc': config.get('holidays_utc', None),  # Use defaults

            # ===== Compound Profit Settings =====
            'compound_profits': config.get('compound_profits', True),
            'max_compound': config.get('max_compound', 1.5),
            'profit_distribution': config.get('profit_distribution', {
                'reinvest': 0.6,
                'realized': 0.4
            }),

            # ===== Risk Management =====
            'fee_rate': config.get('fee_rate', 0.001),
            'max_drawdown': config.get('max_drawdown', 0.10),
            'stop_loss': config.get('stop_loss', None),
            'take_profit': config.get('take_profit', None),

            # ===== Recentering Control =====
            'recenter_after_cycles': config.get('recenter_after_cycles', 5),
            'min_recenter_interval': config.get('min_recenter_interval', 3600),
        }

    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Generate signals with TimeWeightedGrid session awareness.

        Phase 27: Includes session tracking, TWAP info, and low-liquidity warnings.
        """
        # Find BTC data
        btc_key = None
        for key in data.keys():
            if 'BTC' in key.upper() and '_' not in key:
                btc_key = key
                break

        if not btc_key or btc_key not in data:
            return self._hold_signal('No BTC data available')

        df = data[btc_key]
        if df is None or len(df) == 0:
            return self._hold_signal('Empty BTC data')

        current_price = df['close'].iloc[-1]
        self.last_price = current_price

        # Calculate ATR for dynamic spacing
        atr = self._calculate_atr(df) if self.use_atr_spacing else 0.0

        # Initialize or recenter grid if needed
        if not self.initialized:
            self._initialize_grid(current_price, atr)

        # Check if grid strategy needs recentering
        if self.grid_strategy and hasattr(self.grid_strategy, 'should_recenter'):
            if self.grid_strategy.should_recenter(current_price):
                self.grid_strategy.recenter_grid(current_price)

        if not self.grid_strategy:
            return self._hold_signal('Grid strategy not initialized')

        # Track session changes
        current_session = self.grid_strategy.current_session
        if current_session != self.last_session:
            self.session_history.append({
                'from': self.last_session,
                'to': current_session,
                'time': datetime.now().isoformat(),
                'price': current_price
            })
            self.last_session = current_session

        # Call the grid's update method
        try:
            signals = self.grid_strategy.update(current_price, df)
        except Exception as e:
            return self._hold_signal(f'Grid update error: {e}')

        if not signals:
            # Include detailed status in hold reason
            status = self.grid_strategy.get_status()

            session_info = f"Session={status.get('current_session', 'unknown')}"
            if status.get('in_overlap'):
                session_info += f" (overlap: {status.get('overlap_session')})"
            if status.get('low_liquidity'):
                session_info += f" [LOW LIQ: {status.get('liquidity_reason')}]"

            atr_info = ""
            if status.get('current_atr', 0) > 0:
                atr_info = f", ATR=${status.get('current_atr', 0):.0f}"

            twap_info = ""
            if status.get('pending_twap_orders', 0) > 0:
                twap_info = f", TWAP pending={status.get('pending_twap_orders')}"

            range_info = f"${self.grid_strategy.lower_price:,.0f}-${self.grid_strategy.upper_price:,.0f}"

            reason = (f'No signal | Price=${current_price:,.0f}, {session_info}, '
                     f'Range={range_info}{atr_info}{twap_info}')

            return self._hold_signal(reason)

        # Return the highest confidence signal
        signals.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        signal = signals[0]

        action = signal.get('action', 'hold')
        if action in ['close_all', 'exit']:
            action = 'sell'

        # Convert absolute asset size to percentage for orchestrator
        signal_price = signal.get('price', current_price)
        size_pct = self._convert_size_to_pct(signal.get('size', 0), signal_price)

        return {
            'action': action,
            'symbol': signal.get('symbol', btc_key),
            'size': size_pct,
            'leverage': signal.get('leverage', self.grid_strategy.leverage),
            'confidence': signal.get('confidence', 0.7),
            'reason': signal.get('reason', 'Time-weighted grid signal'),
            'strategy': self.name,
            'grid_level': signal.get('grid_level'),
            'order_id': signal.get('order_id'),
            'price': signal_price,
            'target_price': signal.get('target_sell_price', 0),

            # Session info
            'session': signal.get('session', 'unknown'),
            'session_changed': signal.get('session_changed', False),
            'in_overlap': signal.get('in_overlap', False),
            'overlap_session': signal.get('overlap_session', ''),

            # Liquidity info
            'low_liquidity': signal.get('low_liquidity', False),
            'liquidity_reason': signal.get('liquidity_reason', ''),

            # ATR info
            'current_atr': signal.get('current_atr', 0),
            'atr_adjustment': signal.get('atr_adjustment', 1.0),

            # TWAP info
            'twap_slice': signal.get('twap_slice'),
            'twap_total_slices': signal.get('twap_total_slices'),
        }

    def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive strategy status with Phase 27 enhancements.
        """
        base_status = {
            'name': self.name,
            'version': 'Phase27',
            'symbol': self.symbol,
            'initialized': self.initialized,
            'last_price': self.last_price,
        }

        if self.grid_strategy:
            strategy_status = self.grid_strategy.get_status()
            base_status.update({
                # Session info
                'current_session': strategy_status.get('current_session'),
                'previous_session': strategy_status.get('previous_session'),
                'session_change_count': strategy_status.get('session_change_count', 0),
                'session_history': self.session_history[-10:],  # Last 10 transitions

                # Overlap and liquidity
                'in_overlap': strategy_status.get('in_overlap', False),
                'overlap_session': strategy_status.get('overlap_session', ''),
                'low_liquidity': strategy_status.get('low_liquidity', False),
                'liquidity_reason': strategy_status.get('liquidity_reason', ''),
                'day_multiplier': strategy_status.get('day_multiplier', 1.0),

                # Grid info
                'grid_levels': len(self.grid_strategy.grid_levels),
                'positions': len(self.grid_strategy.positions),
                'upper_price': self.grid_strategy.upper_price,
                'lower_price': self.grid_strategy.lower_price,
                'grid_spacing': self.grid_strategy.grid_spacing,

                # ATR info
                'current_atr': strategy_status.get('current_atr', 0),
                'atr_adjustment_factor': strategy_status.get('atr_adjustment_factor', 1.0),

                # TWAP info
                'twap_enabled': strategy_status.get('twap_enabled', False),
                'pending_twap_orders': strategy_status.get('pending_twap_orders', 0),
                'twap_executions': strategy_status.get('twap_executions', 0),

                # Stats
                'stats': {
                    'total_trades': self.grid_strategy.stats.total_trades,
                    'winning_trades': self.grid_strategy.stats.winning_trades,
                    'losing_trades': self.grid_strategy.stats.losing_trades,
                    'total_pnl': self.grid_strategy.stats.total_pnl,
                    'realized_pnl': self.grid_strategy.stats.realized_pnl,
                    'cycles_completed': self.grid_strategy.stats.cycles_completed,
                    'profit_reinvested': self.grid_strategy.stats.profit_reinvested,
                    'total_fees': self.grid_strategy.stats.total_fees,
                },

                # Capital
                'total_capital': strategy_status.get('total_capital', 0),
                'effective_capital': strategy_status.get('effective_capital', 0),
                'peak_capital': strategy_status.get('peak_capital', 0),
                'compound_multiplier': strategy_status.get('compound_multiplier', 1.0),
            })

        return base_status


class LiquidationHuntScalperWrapper(GridStrategyWrapper):
    """
    Phase 29 Enhanced: Wrapper for LiquidationHuntScalper strategy.

    Liquidation hunting targets areas where leveraged traders get stopped out:
    - After strong move up, late longs get liquidated on pullback
    - After strong move down, late shorts get liquidated on bounce
    - We enter on the reversal after the sweep

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
    - Compatible with orchestrator interface
    """
    grid_class = LiquidationHuntScalper
    name = 'grid_liq_hunter'

    def __init__(self, config: Dict[str, Any]):
        # Phase 29: Configure for liquidation hunting
        if 'range_pct' not in config:
            config['range_pct'] = 0.04  # 4% range

        # Fewer recenters - let the strategy manage its own zones
        if 'recenter_after_cycles' not in config:
            config['recenter_after_cycles'] = 20

        super().__init__(config)

    def _get_strategy_params(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get all strategy parameters for Phase 29 LiquidationHuntScalper.

        Exposes full parameter set for tuning and experimentation.
        """
        return {
            # ===== Leverage Configuration =====
            'leverage': config.get('leverage', 3),
            'use_dynamic_leverage': config.get('use_dynamic_leverage', True),
            'max_leverage': config.get('max_leverage', 5),
            'min_leverage': config.get('min_leverage', 2),
            'high_vol_threshold': config.get('high_vol_threshold', 0.03),  # 3% ATR = reduce leverage
            'low_vol_threshold': config.get('low_vol_threshold', 0.015),  # 1.5% ATR = increase leverage

            # ===== Position Sizing =====
            'size_pct': config.get('size_pct', 0.08),  # 8% per trade
            'min_size_pct': config.get('min_size_pct', 0.04),
            'max_size_pct': config.get('max_size_pct', 0.15),
            'max_positions': config.get('max_positions', 2),

            # ===== ATR-Based Liquidation Zone Detection =====
            'use_atr_zones': config.get('use_atr_zones', True),
            'liq_zone_atr_mult': config.get('liq_zone_atr_mult', 1.5),  # 1.5x ATR from swing
            'liq_zone_pct': config.get('liq_zone_pct', 0.02),  # Fallback: 2% fixed
            'atr_period': config.get('atr_period', 14),

            # ===== Swing Detection =====
            'lookback_bars': config.get('lookback_bars', 50),
            'swing_lookback': config.get('swing_lookback', 5),  # Bars to confirm swing
            'min_swing_size_atr': config.get('min_swing_size_atr', 0.5),  # Min swing = 0.5x ATR

            # ===== Volume Confirmation =====
            'use_volume_filter': config.get('use_volume_filter', True),
            'volume_spike_mult': config.get('volume_spike_mult', 1.5),  # 1.5x avg volume
            'volume_window': config.get('volume_window', 20),

            # ===== RSI Filter =====
            'use_rsi_filter': config.get('use_rsi_filter', True),
            'rsi_period': config.get('rsi_period', 14),
            'rsi_oversold': config.get('rsi_oversold', 30),
            'rsi_overbought': config.get('rsi_overbought', 70),
            'rsi_extreme_oversold': config.get('rsi_extreme_oversold', 20),
            'rsi_extreme_overbought': config.get('rsi_extreme_overbought', 80),

            # ===== Reversal Pattern Settings =====
            'min_wick_ratio': config.get('min_wick_ratio', 2.0),  # Wick >= 2x body for hammer
            'engulfing_body_ratio': config.get('engulfing_body_ratio', 1.1),  # 10% larger body
            'require_pattern': config.get('require_pattern', True),  # Must have reversal pattern
            'pattern_confidence_boost': config.get('pattern_confidence_boost', 0.15),

            # ===== ATR-Based Risk Management =====
            'use_atr_stops': config.get('use_atr_stops', True),
            'stop_loss_atr_mult': config.get('stop_loss_atr_mult', 1.5),
            'take_profit_atr_mult': config.get('take_profit_atr_mult', 2.5),
            'stop_loss_pct': config.get('stop_loss_pct', 0.015),  # Fallback: 1.5%
            'take_profit_pct': config.get('take_profit_pct', 0.03),  # Fallback: 3%

            # ===== Trailing Stop =====
            'use_trailing_stop': config.get('use_trailing_stop', True),
            'trailing_activation_atr': config.get('trailing_activation_atr', 1.5),  # Activate after 1.5x ATR profit
            'trailing_distance_atr': config.get('trailing_distance_atr', 1.0),  # Trail by 1x ATR

            # ===== Session Awareness =====
            'use_session_adjustment': config.get('use_session_adjustment', True),
            'session_multipliers': config.get('session_multipliers', {
                'asia': 0.8,      # 00:00-08:00 UTC - tighter zones
                'europe': 1.0,    # 08:00-14:00 UTC - standard
                'us': 1.2,        # 14:00-21:00 UTC - wider (more volatile)
                'overnight': 0.9  # 21:00-00:00 UTC - slightly tighter
            }),

            # ===== Compound Profits =====
            'compound_profits': config.get('compound_profits', True),
            'max_compound': config.get('max_compound', 1.5),
            'profit_reinvest_ratio': config.get('profit_reinvest_ratio', 0.6),

            # ===== Cooldown =====
            'cooldown_bars': config.get('cooldown_bars', 3),
        }

    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Generate signals using the LiquidationHuntScalper strategy.

        Phase 29: Full signal generation with all confirmations including:
        - ATR-based liquidation zones
        - Swing point detection
        - Volume spike confirmation
        - RSI filtering
        - Reversal pattern detection
        """
        # Find BTC data
        btc_key = None
        for key in data.keys():
            if 'BTC' in key.upper() and '_' not in key:
                btc_key = key
                break

        if not btc_key or btc_key not in data:
            return self._hold_signal('No BTC data available')

        df = data[btc_key]
        if df is None or len(df) == 0:
            return self._hold_signal('Empty BTC data')

        current_price = df['close'].iloc[-1]
        self.last_price = current_price

        # Initialize if needed
        if not self.initialized:
            self.initialized = True

        if not self.grid_strategy:
            # Initialize the underlying strategy
            self._initialize_grid(current_price, 0)

        if not self.grid_strategy:
            return self._hold_signal('Grid strategy not initialized')

        # Call the strategy's update method with full OHLCV data
        try:
            signals = self.grid_strategy.update(current_price, df)
        except Exception as e:
            return self._hold_signal(f'Strategy update error: {e}')

        if not signals:
            # Include detailed status in hold reason
            status = self.grid_strategy.get_status()
            local_high = status.get('local_high', 0)
            local_low = status.get('local_low', 0)
            long_liq = status.get('long_liq_zone', 0)
            short_liq = status.get('short_liq_zone', 0)
            rsi = status.get('current_rsi', 50)
            atr_pct = status.get('current_atr_pct', 0)
            leverage = status.get('leverage', 3)
            positions = status.get('open_positions', 0)
            session = status.get('session', 'unknown')
            vol_ratio = status.get('current_volume_ratio', 1.0)

            reason = (f'No liq hunt signal | Price=${current_price:,.0f}, '
                     f'Session={session}, RSI={rsi:.0f}, ATR={atr_pct:.2f}%, '
                     f'Vol={vol_ratio:.1f}x, '
                     f'LiqZone=[${long_liq:,.0f}, ${short_liq:,.0f}], '
                     f'Lev={leverage}x, Pos={positions}')

            return self._hold_signal(reason)

        # Return the highest confidence signal
        signals.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        signal = signals[0]

        action = signal.get('action', 'hold')
        if action in ['close_all', 'exit', 'close']:
            action = 'close'

        # Convert absolute asset size to percentage for orchestrator
        signal_price = signal.get('price', current_price)
        size_pct = self._convert_size_to_pct(signal.get('size', 0), signal_price)

        return {
            'action': action,
            'symbol': signal.get('symbol', btc_key),
            'size': size_pct,
            'leverage': signal.get('leverage', self.grid_strategy.leverage),
            'confidence': signal.get('confidence', 0.5),
            'reason': signal.get('reason', 'Liq hunt signal'),
            'strategy': self.name,
            'price': signal_price,
            'stop_loss': signal.get('stop_loss', 0),
            'take_profit': signal.get('take_profit', 0),
            'side': signal.get('side', 'long'),

            # Phase 29 metadata
            'pattern': signal.get('pattern', ''),
            'rsi': signal.get('rsi', 50),
            'atr': signal.get('atr', 0),
            'volume_ratio': signal.get('volume_ratio', 1.0),
            'session': signal.get('session', 'unknown'),
            'liq_zone': signal.get('liq_zone', 0),
            'swing_point': signal.get('swing_point', 0),
        }

    def on_order_filled(self, order_info: Dict[str, Any]):
        """
        Process filled order - delegate to underlying strategy.

        Phase 29: Proper position tracking and compound profit handling.
        """
        if self.grid_strategy and hasattr(self.grid_strategy, 'on_order_filled'):
            self.grid_strategy.on_order_filled(order_info)

    def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive strategy status.

        Note: LiquidationHuntScalper doesn't have traditional grid attributes,
        so we build status directly rather than calling parent.
        """
        # Build status without calling parent (which expects grid attributes)
        base_status = {
            'name': self.name,
            'symbol': self.symbol,
            'initialized': self.initialized,
            'last_price': self.last_price,
            'version': 'Phase29',
        }

        if self.grid_strategy:
            grid_status = self.grid_strategy.get_status()
            base_status.update({
                # Price info
                'current_price': grid_status.get('current_price', 0),
                'local_high': grid_status.get('local_high', 0),
                'local_low': grid_status.get('local_low', 0),

                # Liquidation zones
                'long_liq_zone': grid_status.get('long_liq_zone', 0),
                'short_liq_zone': grid_status.get('short_liq_zone', 0),
                'zone_type': grid_status.get('zone_type', 'unknown'),

                # Swing points
                'swing_highs_count': grid_status.get('swing_highs_count', 0),
                'swing_lows_count': grid_status.get('swing_lows_count', 0),

                # Indicators
                'current_atr': grid_status.get('current_atr', 0),
                'current_atr_pct': grid_status.get('current_atr_pct', 0),
                'current_rsi': grid_status.get('current_rsi', 50),
                'current_volume_ratio': grid_status.get('current_volume_ratio', 1.0),

                # Leverage
                'leverage': grid_status.get('leverage', 3),
                'base_leverage': grid_status.get('base_leverage', 3),
                'dynamic_leverage': grid_status.get('dynamic_leverage', True),

                # Session
                'session': grid_status.get('session', 'unknown'),

                # Positions
                'open_positions': grid_status.get('open_positions', 0),
                'max_positions': grid_status.get('max_positions', 2),
                'unrealized_pnl': grid_status.get('unrealized_pnl', 0),

                # Stats
                'total_pnl': grid_status.get('total_pnl', 0),
                'realized_pnl': grid_status.get('realized_pnl', 0),
                'total_trades': grid_status.get('total_trades', 0),
                'winning_trades': grid_status.get('winning_trades', 0),
                'losing_trades': grid_status.get('losing_trades', 0),
                'win_rate': grid_status.get('win_rate', 0),
                'cycles_completed': grid_status.get('cycles_completed', 0),
                'profit_reinvested': grid_status.get('profit_reinvested', 0),

                # Capital
                'total_capital': grid_status.get('total_capital', 0),
                'initial_capital': grid_status.get('initial_capital', 0),
                'compound_multiplier': grid_status.get('compound_multiplier', 1.0),

                # Last signal
                'last_pattern': grid_status.get('last_pattern', None),
                'bars_since_trade': grid_status.get('bars_since_trade', 0),
                'cooldown_bars': grid_status.get('cooldown_bars', 3),
            })

        return base_status
