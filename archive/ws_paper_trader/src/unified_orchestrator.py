"""
Phase 31: Unified Orchestrator with Dual Portfolio Support
Combines all trading strategies with weighted voting, per-strategy logging,
and experiment parameter tracking.

NEW: Dual Portfolio Mode
- USDT Portfolio (70%): Quick trades with forced exits, profit in USDT
- Crypto Portfolio (30%): Long-term accumulation of BTC/XRP

Features:
- Dynamic strategy loading from registry
- Weighted voting across all strategies
- Per-strategy logging with experiment tracking
- Regime detection for strategy weight adjustment
- Easy experiment parameter injection
- Position tracking with forced exits (max hold time, TP/SL)
- Separate metrics for USDT and Crypto portfolios
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(__file__))

from strategy_registry import StrategyRegistry
from utils.strategy_logger import StrategyLoggerManager
from utils.strategy_diagnostics import StrategyDiagnostics, get_strategy_diagnostics
from portfolio import Portfolio
from data_fetcher import DataFetcher
from risk_manager import get_risk_profile, StrategyRiskProfile, DEFAULT_RISK_PROFILES
from strategy_portfolio import StrategyPortfolioManager


@dataclass
class RegimeState:
    """Current market regime."""
    name: str = "neutral"
    volatility: float = 0.0
    correlation: float = 0.0
    trend: str = "sideways"  # up, down, sideways
    rsi: Dict[str, float] = field(default_factory=dict)
    atr: Dict[str, float] = field(default_factory=dict)


@dataclass
class Position:
    """Track an open position for USDT accumulation strategies."""
    symbol: str
    side: str  # 'long' or 'short'
    entry_price: float
    entry_time: datetime
    size: float
    strategy: str
    portfolio: str  # 'usdt' or 'crypto'
    take_profit_pct: float = 0.02
    stop_loss_pct: float = 0.015
    max_hold_hours: float = 24.0
    trailing_stop_pct: float = 0.0
    highest_price: float = 0.0  # For trailing stop
    lowest_price: float = float('inf')  # For trailing stop on shorts

    def check_exit(self, current_price: float) -> Optional[str]:
        """Check if position should be exited."""
        # Update highest/lowest for trailing stop
        self.highest_price = max(self.highest_price, current_price)
        self.lowest_price = min(self.lowest_price, current_price)

        # Calculate P&L
        if self.side == 'long':
            pnl_pct = (current_price - self.entry_price) / self.entry_price
        else:  # short
            pnl_pct = (self.entry_price - current_price) / self.entry_price

        # Check take profit
        if pnl_pct >= self.take_profit_pct:
            return 'take_profit'

        # Check stop loss
        if pnl_pct <= -self.stop_loss_pct:
            return 'stop_loss'

        # Check trailing stop (if enabled)
        if self.trailing_stop_pct > 0:
            if self.side == 'long':
                trail_price = self.highest_price * (1 - self.trailing_stop_pct)
                if current_price <= trail_price and pnl_pct > 0:
                    return 'trailing_stop'
            else:  # short
                trail_price = self.lowest_price * (1 + self.trailing_stop_pct)
                if current_price >= trail_price and pnl_pct > 0:
                    return 'trailing_stop'

        # Check max hold time
        hold_duration = (datetime.now() - self.entry_time).total_seconds() / 3600
        if hold_duration >= self.max_hold_hours:
            return 'time_exit'

        return None


class UnifiedOrchestrator:
    """
    Master orchestrator for all trading strategies.

    Combines:
    - General strategies (defensive, mean reversion, pair trading)
    - Scalper strategies (intraday, EMA-9)
    - Grid strategies (arithmetic, geometric, RSI, BB)
    - Margin strategies (trend, hedge, liquidation hunter)

    Features:
    - Weighted voting with regime-based adjustment
    - Per-strategy logging
    - Experiment parameter tracking
    - Paper trading execution
    """

    # Phase 27: Default strategy weights with enhanced margin grid support
    DEFAULT_WEIGHTS = {
        # General strategies
        'defensive_yield': 0.16,
        'mean_reversion_vwap': 0.12,
        'xrp_btc_pair_trading': 0.12,
        'ma_trend_follow': 0.10,
        'xrp_btc_leadlag': 0.10,

        # Scalpers
        'intraday_scalper': 0.10,
        'ema9_scalper': 0.04,

        # Phase 26: Grid strategies with increased weights
        'grid_arithmetic': 0.08,  # Enhanced strategy
        'grid_geometric': 0.03,
        'grid_rsi_reversion': 0.03,
        'grid_bb_squeeze': 0.02,

        # Phase 27: Margin strategies - increased weight for trend_margin
        'grid_trend_margin': 0.06,  # Increased from 0.02 - Phase 27 enhanced
        'grid_dual_hedge': 0.02,
        'grid_liq_hunter': 0.02,
    }

    # Phase 27: Regime-based weight presets with enhanced grid and margin support
    REGIME_WEIGHTS = {
        'chop': {  # Sideways/ranging market - GRID STRATEGIES EXCEL HERE
            'mean_reversion_vwap': 0.20,
            'xrp_btc_pair_trading': 0.15,
            'grid_arithmetic': 0.20,  # Grids work well in ranging markets
            'grid_rsi_reversion': 0.15,
            'grid_trend_margin': 0.10,  # Phase 27: Works in chop too
            'defensive_yield': 0.10,
            'grid_geometric': 0.10,
        },
        'trend_up': {  # Bull market - TREND MARGIN EXCELS
            'ma_trend_follow': 0.25,
            'xrp_btc_leadlag': 0.20,
            'grid_trend_margin': 0.20,  # Phase 27: Primary use case
            'defensive_yield': 0.15,
            'grid_arithmetic': 0.10,
            'grid_liq_hunter': 0.10,
        },
        'trend_down': {  # Bear market
            'defensive_yield': 0.35,
            'xrp_btc_pair_trading': 0.15,
            'grid_dual_hedge': 0.15,
            'grid_trend_margin': 0.15,  # Phase 27: Can short in downtrends
            'grid_arithmetic': 0.10,
            'mean_reversion_vwap': 0.10,
        },
        'high_volatility': {  # High ATR
            'intraday_scalper': 0.25,
            'ema9_scalper': 0.15,
            'defensive_yield': 0.20,
            'grid_bb_squeeze': 0.15,
            'grid_trend_margin': 0.10,  # Phase 27: Dynamic leverage adjusts
            'grid_arithmetic': 0.05,
            'grid_liq_hunter': 0.10,
        },
        'low_volatility': {  # Grinding/low activity - IDEAL FOR GRIDS
            'grid_arithmetic': 0.25,  # Perfect for low vol
            'defensive_yield': 0.25,  # Yield accrual
            'mean_reversion_vwap': 0.15,
            'grid_geometric': 0.15,
            'grid_trend_margin': 0.10,  # Phase 27: Max leverage in low vol
            'grid_rsi_reversion': 0.10,
        },
    }

    def __init__(self,
                 portfolio: Portfolio,
                 config_path: str = None,
                 experiment_id: str = None,
                 dual_portfolio_mode: bool = False,
                 usdt_allocation: float = 0.70,
                 crypto_allocation: float = 0.30,
                 use_isolated_portfolios: bool = True,
                 per_strategy_capital: float = 1000.0):
        """
        Initialize unified orchestrator.

        Args:
            portfolio: Portfolio instance for balance tracking
            config_path: Path to unified YAML config
            experiment_id: Optional experiment identifier
            dual_portfolio_mode: Enable 70/30 USDT/Crypto split
            usdt_allocation: Percentage allocated to USDT strategies
            crypto_allocation: Percentage allocated to Crypto strategies
            use_isolated_portfolios: Phase 32 - Give each strategy $1000 isolated portfolio
            per_strategy_capital: Starting capital per strategy (default $1000)
        """
        self.portfolio = portfolio

        # Dual portfolio mode
        self.dual_portfolio_mode = dual_portfolio_mode
        self.usdt_allocation = usdt_allocation
        self.crypto_allocation = crypto_allocation

        # Initialize registry
        self.registry = StrategyRegistry(config_path)

        # Initialize logging
        self.experiment_id = experiment_id or f"unified_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.logger_manager = StrategyLoggerManager(
            experiment_id=self.experiment_id,
            log_dir="logs"
        )

        # Strategy instances
        self.strategies: Dict[str, Any] = {}
        self.strategy_weights: Dict[str, float] = {}

        # Market state
        self.regime = RegimeState()
        self.current_prices: Dict[str, float] = {}
        self.data: Dict[str, pd.DataFrame] = {}

        # Performance tracking
        self.decision_count = 0
        self.execution_count = 0
        self.last_signals: Dict[str, Dict] = {}

        # Position tracking for USDT accumulation (forced exits)
        self.open_positions: Dict[str, Position] = {}  # symbol -> Position

        # Dual portfolio metrics
        self.usdt_metrics = {
            'total_profit': 0.0,
            'trade_count': 0,
            'win_count': 0,
            'loss_count': 0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'total_hold_time': 0.0,
        }
        self.crypto_metrics = {
            'btc_accumulated': 0.0,
            'xrp_accumulated': 0.0,
            'btc_avg_cost': 0.0,
            'xrp_avg_cost': 0.0,
            'dca_buys': 0,
            'dip_buys': 0,
        }

        # Data fetcher
        self.fetcher = DataFetcher()

        # Phase 32: Per-strategy isolated portfolios
        self.use_isolated_portfolios = use_isolated_portfolios
        self.per_strategy_capital = per_strategy_capital
        self.portfolio_manager = None  # Initialized after strategies load

        # Phase 32: Strategy diagnostics for debugging non-trading strategies
        self.enable_diagnostics = True
        self.diagnostics: Optional[StrategyDiagnostics] = None

        self._initialize()

    def _initialize(self):
        """Initialize strategies and weights."""
        # Log experiment config
        self.logger_manager.log_experiment_config({
            'experiment_id': self.experiment_id,
            'registry_status': self.registry.get_status(),
            'timestamp': datetime.now().isoformat()
        })

        # Load enabled strategies
        enabled = self.registry.get_enabled_strategies()
        print(f"\n[UnifiedOrchestrator] Initializing with {len(enabled)} strategies")

        for name in enabled:
            try:
                strategy = self.registry.instantiate(name)
                if strategy:
                    self.strategies[name] = strategy

                    # Get logger for this strategy
                    logger = self.logger_manager.get_logger(name)
                    logger.log_config(self.registry.get_params(name))

                    # Set initial weight
                    self.strategy_weights[name] = self.DEFAULT_WEIGHTS.get(name, 0.05)

                    print(f"  + {name} (weight: {self.strategy_weights[name]:.2f})")
            except Exception as e:
                print(f"  ! Failed to initialize {name}: {e}")

        # Normalize weights
        self._normalize_weights()

        # Phase 32: Initialize per-strategy isolated portfolios
        if self.use_isolated_portfolios:
            strategy_names = list(self.strategies.keys())
            self.portfolio_manager = StrategyPortfolioManager(
                strategy_names=strategy_names,
                per_strategy_capital=self.per_strategy_capital,
                fee_rate=0.001  # 0.1% fee
            )
            total_capital = len(strategy_names) * self.per_strategy_capital
            print(f"\n[Phase 32] Isolated portfolios: {len(strategy_names)} strategies × ${self.per_strategy_capital:,.0f} = ${total_capital:,.0f}")

        # Phase 32: Initialize strategy diagnostics
        if self.enable_diagnostics:
            self.diagnostics = get_strategy_diagnostics(log_dir="logs/diagnostics", enabled=True)
            print(f"[Phase 32] Strategy diagnostics enabled")

        print(f"\n[UnifiedOrchestrator] Ready with {len(self.strategies)} active strategies")

    def _normalize_weights(self):
        """Normalize strategy weights to sum to 1.0."""
        total = sum(self.strategy_weights.values())
        if total > 0:
            for name in self.strategy_weights:
                self.strategy_weights[name] /= total

    def add_strategy(self, name: str, config: Dict[str, Any] = None) -> bool:
        """Add a strategy at runtime."""
        if not self.registry.enable(name):
            return False

        strategy = self.registry.instantiate(name, config)
        if strategy:
            self.strategies[name] = strategy
            self.strategy_weights[name] = self.DEFAULT_WEIGHTS.get(name, 0.05)

            logger = self.logger_manager.get_logger(name)
            logger.log_config(self.registry.get_params(name))

            self._normalize_weights()
            return True

        return False

    def remove_strategy(self, name: str) -> bool:
        """Remove a strategy at runtime."""
        if name in self.strategies:
            del self.strategies[name]
            del self.strategy_weights[name]
            self.registry.disable(name)
            self._normalize_weights()
            return True
        return False

    def set_weight(self, name: str, weight: float):
        """Set a specific strategy's weight."""
        if name in self.strategy_weights:
            self.strategy_weights[name] = weight
            self._normalize_weights()

    def apply_regime_weights(self, regime: str):
        """Apply regime-specific weights."""
        if regime in self.REGIME_WEIGHTS:
            preset = self.REGIME_WEIGHTS[regime]
            for name in self.strategy_weights:
                self.strategy_weights[name] = preset.get(name, 0.02)
            self._normalize_weights()

    def set_experiment_param(self, strategy_name: str, param: str, value: Any):
        """Set an experiment parameter override."""
        self.registry.override_param(strategy_name, param, value)

        # Log the override
        if strategy_name in self.strategies:
            logger = self.logger_manager.get_logger(strategy_name)
            logger.log_experiment_param(param, value, f"Override for experiment {self.experiment_id}")

    def update_data(self, symbols: List[str] = None, include_scalping_timeframes: bool = True,
                    include_1m: bool = False):
        """
        Fetch fresh market data.

        Phase 25: Now fetches multiple timeframes (5m, 15m, 1h) for scalper strategies.
        Phase 30: Added 1m data support for 60-second polling intervals.

        Args:
            symbols: List of symbols to fetch
            include_scalping_timeframes: If True, also fetch 5m and 15m data for scalpers
            include_1m: If True, also fetch 1m data for high-frequency strategies
        """
        symbols = symbols or ['XRP/USDT', 'BTC/USDT']

        for symbol in symbols:
            # Map to Kraken symbols if needed
            kraken_symbol = symbol.replace('USDT', 'USD')

            try:
                # Fetch 1h data (primary timeframe)
                df = self.fetcher.fetch_ohlcv('kraken', kraken_symbol, '1h', 500)
                if df is not None and not df.empty:
                    self.data[symbol] = df

                # Fetch scalping timeframes if enabled
                if include_scalping_timeframes:
                    # 1-minute data for high-frequency strategies (60s polling)
                    if include_1m:
                        df_1m = self.fetcher.fetch_ohlcv('kraken', kraken_symbol, '1m', 200)
                        if df_1m is not None and not df_1m.empty:
                            self.data[f'{symbol}_1m'] = df_1m

                    # 5-minute data for scalpers
                    df_5m = self.fetcher.fetch_ohlcv('kraken', kraken_symbol, '5m', 500)
                    if df_5m is not None and not df_5m.empty:
                        self.data[f'{symbol}_5m'] = df_5m

                    # 15-minute data as fallback
                    df_15m = self.fetcher.fetch_ohlcv('kraken', kraken_symbol, '15m', 300)
                    if df_15m is not None and not df_15m.empty:
                        self.data[f'{symbol}_15m'] = df_15m

            except Exception as e:
                print(f"Warning: Failed to fetch {symbol}: {e}")

    def update_prices(self, prices: Dict[str, float] = None):
        """Update current prices."""
        if prices:
            self.current_prices = prices
        else:
            # Fetch current prices
            for symbol in ['XRP/USDT', 'BTC/USDT']:
                kraken_symbol = symbol.replace('USDT', 'USD')
                try:
                    p = self.fetcher.get_best_price(kraken_symbol)
                    if p:
                        base = symbol.split('/')[0]
                        self.current_prices[base] = list(p.values())[0]
                except:
                    pass

            self.current_prices['USDT'] = 1.0

    def get_live_data(self) -> Dict[str, pd.DataFrame]:
        """
        Get data with live prices injected into the last candle.
        This fixes the stale indicator bug where hourly candles don't reflect
        current price movements between candle closes.

        Phase 25: Now also handles multi-timeframe data (5m, 15m suffixes).
        """
        live_data = {}

        for symbol, df in self.data.items():
            if df is None or df.empty:
                continue

            # Make a copy to avoid modifying original
            df_copy = df.copy()

            # Get base asset (XRP, BTC) - handle timeframe suffixes
            base_symbol = symbol.split('_')[0]  # Remove _5m, _15m suffix
            base = base_symbol.split('/')[0]     # Get base asset (XRP, BTC)
            current_price = self.current_prices.get(base)

            if current_price and len(df_copy) > 0:
                # Update the last candle's close/high/low with live price
                last_idx = df_copy.index[-1]
                df_copy.loc[last_idx, 'close'] = current_price
                df_copy.loc[last_idx, 'high'] = max(df_copy.loc[last_idx, 'high'], current_price)
                df_copy.loc[last_idx, 'low'] = min(df_copy.loc[last_idx, 'low'], current_price)

            live_data[symbol] = df_copy

        return live_data

    def check_position_exits(self) -> List[Dict[str, Any]]:
        """
        Check all open positions for forced exits (USDT accumulation mode).

        Returns list of exit signals for positions that need to be closed.
        """
        exit_signals = []

        for symbol, position in list(self.open_positions.items()):
            base = symbol.split('/')[0]
            current_price = self.current_prices.get(base, 0)

            if current_price <= 0:
                continue

            exit_reason = position.check_exit(current_price)

            if exit_reason:
                # Calculate P&L
                if position.side == 'long':
                    pnl_pct = (current_price - position.entry_price) / position.entry_price
                else:
                    pnl_pct = (position.entry_price - current_price) / position.entry_price

                pnl_usd = position.size * position.entry_price * pnl_pct

                # Create exit signal
                exit_signal = {
                    'action': 'sell' if position.side == 'long' else 'cover',
                    'symbol': symbol,
                    'size': 1.0,  # Close full position
                    'confidence': 0.95,
                    'reason': f'Forced exit ({exit_reason}): {pnl_pct*100:.2f}% P&L',
                    'strategy': position.strategy,
                    'portfolio': position.portfolio,
                    'exit_reason': exit_reason,
                    'pnl_pct': pnl_pct,
                    'pnl_usd': pnl_usd,
                    'entry_price': position.entry_price,
                    'exit_price': current_price,
                    'hold_hours': (datetime.now() - position.entry_time).total_seconds() / 3600,
                }

                exit_signals.append(exit_signal)

                # Update metrics
                if position.portfolio == 'usdt':
                    self.usdt_metrics['trade_count'] += 1
                    self.usdt_metrics['total_profit'] += pnl_usd
                    self.usdt_metrics['total_hold_time'] += exit_signal['hold_hours']

                    if pnl_usd > 0:
                        self.usdt_metrics['win_count'] += 1
                        self.usdt_metrics['largest_win'] = max(self.usdt_metrics['largest_win'], pnl_usd)
                    else:
                        self.usdt_metrics['loss_count'] += 1
                        self.usdt_metrics['largest_loss'] = min(self.usdt_metrics['largest_loss'], pnl_usd)

                # Remove from tracking
                del self.open_positions[symbol]

                print(f"  [FORCED EXIT] {symbol} {position.side}: {exit_reason}, P&L: ${pnl_usd:.2f} ({pnl_pct*100:.2f}%)")

        return exit_signals

    def track_position(self, signal: Dict[str, Any], execution_result: Dict[str, Any]):
        """Track a new position for USDT accumulation strategies."""
        if not self.dual_portfolio_mode:
            return

        symbol = signal.get('symbol', '')
        strategy = signal.get('strategy', signal.get('primary_strategy', ''))
        action = signal.get('action', '')

        # Get strategy config to determine portfolio
        params = self.registry.get_params(strategy) if strategy else {}
        portfolio = params.get('portfolio', 'usdt')

        # Only track USDT portfolio positions (crypto positions are held long-term)
        if portfolio != 'usdt':
            return

        if action in ['buy', 'short'] and execution_result.get('executed'):
            # Phase 31: Get exit parameters from risk profile (with config overrides)
            risk_profile = get_risk_profile(strategy, params)

            position = Position(
                symbol=symbol,
                side='long' if action == 'buy' else 'short',
                entry_price=execution_result.get('price', 0),
                entry_time=datetime.now(),
                size=execution_result.get('amount', 0),
                strategy=strategy,
                portfolio=portfolio,
                take_profit_pct=risk_profile.take_profit_pct,
                stop_loss_pct=risk_profile.stop_loss_pct,
                max_hold_hours=risk_profile.max_hold_hours,
                trailing_stop_pct=risk_profile.trailing_distance_pct if risk_profile.use_trailing_stop else 0.0,
                highest_price=execution_result.get('price', 0),
                lowest_price=execution_result.get('price', 0),
            )

            self.open_positions[symbol] = position
            print(f"  [TRACKING] {symbol} {position.side} @ ${position.entry_price:.4f}, "
                  f"TP:{risk_profile.take_profit_pct*100:.1f}%, SL:{risk_profile.stop_loss_pct*100:.1f}%, "
                  f"Max:{risk_profile.max_hold_hours}h ({risk_profile.category.value})")

    def update_crypto_metrics(self, signal: Dict[str, Any], execution_result: Dict[str, Any]):
        """Update crypto accumulation metrics."""
        if not execution_result.get('executed'):
            return

        symbol = signal.get('symbol', '')
        strategy = signal.get('strategy', signal.get('primary_strategy', ''))
        action = signal.get('action', '')

        # Get strategy config
        params = self.registry.get_params(strategy) if strategy else {}
        portfolio = params.get('portfolio', 'usdt')

        if portfolio != 'crypto' or action != 'buy':
            return

        base = symbol.split('/')[0]
        amount = execution_result.get('amount', 0)
        price = execution_result.get('price', 0)

        if base == 'BTC':
            # Update running average cost
            old_total = self.crypto_metrics['btc_accumulated'] * self.crypto_metrics['btc_avg_cost']
            new_total = old_total + (amount * price)
            self.crypto_metrics['btc_accumulated'] += amount
            if self.crypto_metrics['btc_accumulated'] > 0:
                self.crypto_metrics['btc_avg_cost'] = new_total / self.crypto_metrics['btc_accumulated']

        elif base == 'XRP':
            old_total = self.crypto_metrics['xrp_accumulated'] * self.crypto_metrics['xrp_avg_cost']
            new_total = old_total + (amount * price)
            self.crypto_metrics['xrp_accumulated'] += amount
            if self.crypto_metrics['xrp_accumulated'] > 0:
                self.crypto_metrics['xrp_avg_cost'] = new_total / self.crypto_metrics['xrp_accumulated']

        # Track DCA vs dip buys
        if 'dca' in strategy.lower():
            self.crypto_metrics['dca_buys'] += 1
        elif 'dip' in strategy.lower():
            self.crypto_metrics['dip_buys'] += 1

    def detect_regime(self) -> RegimeState:
        """Detect current market regime from data."""
        regime = RegimeState()

        if not self.data:
            return regime

        # Use live data with current prices injected
        live_data = self.get_live_data()

        # Calculate indicators from BTC data
        btc_data = live_data.get('BTC/USDT')
        xrp_data = live_data.get('XRP/USDT')

        if btc_data is not None and len(btc_data) > 20:
            # ATR for volatility
            high = btc_data['high'].iloc[-14:]
            low = btc_data['low'].iloc[-14:]
            close = btc_data['close'].iloc[-14:]

            tr = pd.concat([
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs()
            ], axis=1).max(axis=1)

            atr = tr.mean()
            atr_pct = atr / close.iloc[-1]
            regime.atr['BTC'] = atr_pct
            regime.volatility = atr_pct

            # RSI
            delta = close.diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rs = gain / loss.replace(0, 1e-10)
            rsi = 100 - (100 / (1 + rs))
            regime.rsi['BTC'] = rsi.iloc[-1] if not rsi.empty else 50

            # Trend detection
            sma_20 = close.rolling(20).mean().iloc[-1]
            sma_50 = close.rolling(50).mean().iloc[-1] if len(btc_data) > 50 else sma_20

            current_price = close.iloc[-1]
            if current_price > sma_20 > sma_50:
                regime.trend = "up"
            elif current_price < sma_20 < sma_50:
                regime.trend = "down"
            else:
                regime.trend = "sideways"

        # XRP indicators
        if xrp_data is not None and len(xrp_data) > 20:
            close = xrp_data['close'].iloc[-14:]
            delta = close.diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rs = gain / loss.replace(0, 1e-10)
            rsi = 100 - (100 / (1 + rs))
            regime.rsi['XRP'] = rsi.iloc[-1] if not rsi.empty else 50

        # Correlation
        if btc_data is not None and xrp_data is not None and len(btc_data) > 20:
            btc_returns = btc_data['close'].pct_change().iloc[-20:]
            xrp_returns = xrp_data['close'].pct_change().iloc[-20:]

            if len(btc_returns) == len(xrp_returns):
                regime.correlation = btc_returns.corr(xrp_returns)

        # Determine regime name
        if regime.volatility > 0.04:
            regime.name = "high_volatility"
        elif regime.volatility < 0.015:
            regime.name = "low_volatility"
        elif regime.trend == "up":
            regime.name = "trend_up"
        elif regime.trend == "down":
            regime.name = "trend_down"
        else:
            regime.name = "chop"

        self.regime = regime
        return regime

    def get_all_signals(self) -> Dict[str, Dict[str, Any]]:
        """Get signals from all enabled strategies."""
        signals = {}

        # Get data with live prices injected (fixes stale indicator bug)
        live_data = self.get_live_data()

        for name, strategy in self.strategies.items():
            try:
                # Generate signal using live-updated data
                signal = strategy.generate_signals(live_data)

                # Validate signal
                if signal and strategy.validate_signal(signal):
                    # Phase 32: Enhance signal with diagnostics
                    if self.diagnostics:
                        signal = self.diagnostics.diagnose_signal(
                            strategy_name=name,
                            signal=signal,
                            data=live_data,
                            strategy_instance=strategy
                        )

                    signals[name] = signal

                    # Log to strategy-specific logger
                    logger = self.logger_manager.get_logger(name)
                    logger.log_signal(
                        action=signal.get('action', 'hold'),
                        symbol=signal.get('symbol', ''),
                        confidence=signal.get('confidence', 0),
                        leverage=signal.get('leverage', 1),
                        size=signal.get('size', 0),
                        reason=signal.get('reason', ''),
                        indicators=signal.get('indicators', {}),
                        price=self.current_prices.get(signal.get('symbol', '').split('/')[0], 0)
                    )
                else:
                    # Default hold signal
                    hold_signal = {
                        'action': 'hold',
                        'confidence': 0,
                        'symbol': '',
                        'reason': 'No valid signal'
                    }
                    # Phase 32: Diagnose why no valid signal
                    if self.diagnostics:
                        hold_signal = self.diagnostics.diagnose_signal(
                            strategy_name=name,
                            signal=hold_signal,
                            data=live_data,
                            strategy_instance=strategy
                        )
                    signals[name] = hold_signal

            except Exception as e:
                print(f"Warning: {name} signal error: {e}")
                error_signal = {
                    'action': 'hold',
                    'confidence': 0,
                    'symbol': '',
                    'reason': f'Error: {e}'
                }
                # Phase 32: Diagnose errors too
                if self.diagnostics:
                    error_signal = self.diagnostics.diagnose_signal(
                        strategy_name=name,
                        signal=error_signal,
                        data=live_data,
                        strategy_instance=strategy if 'strategy' in dir() else None
                    )
                signals[name] = error_signal

        self.last_signals = signals
        return signals

    def weighted_vote(self, signals: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combine signals via weighted voting.

        Returns the action with highest weighted confidence.
        """
        self.decision_count += 1

        # Aggregate votes by action
        # Phase 25: Added 'cover' for closing short positions
        action_scores = {
            'buy': {'score': 0, 'signals': [], 'total_confidence': 0},
            'sell': {'score': 0, 'signals': [], 'total_confidence': 0},
            'short': {'score': 0, 'signals': [], 'total_confidence': 0},
            'cover': {'score': 0, 'signals': [], 'total_confidence': 0},  # Close short position
            'hold': {'score': 0, 'signals': [], 'total_confidence': 0},
            'close': {'score': 0, 'signals': [], 'total_confidence': 0},
        }

        for name, signal in signals.items():
            action = signal.get('action', 'hold')
            confidence = signal.get('confidence', 0)
            weight = self.strategy_weights.get(name, 0)

            if action in action_scores:
                weighted_score = confidence * weight
                action_scores[action]['score'] += weighted_score
                action_scores[action]['signals'].append(name)
                action_scores[action]['total_confidence'] += confidence

        # Find winning action
        best_action = 'hold'
        best_score = 0
        best_confidence = 0

        for action, data in action_scores.items():
            if data['score'] > best_score:
                best_score = data['score']
                best_action = action
                if data['signals']:
                    best_confidence = data['total_confidence'] / len(data['signals'])

        # Build result signal
        contributing = action_scores[best_action]['signals']
        result = {
            'action': best_action,
            'confidence': best_confidence,
            'weighted_score': best_score,
            'contributing_strategies': contributing,
            'regime': self.regime.name,
            'reason': f"Weighted vote: {', '.join(contributing)}"
        }

        # If a specific action, pick symbol/size from highest confidence contributor
        if best_action != 'hold' and contributing:
            best_contributor = max(contributing, key=lambda n: signals[n].get('confidence', 0))
            contributor_signal = signals[best_contributor]
            result['symbol'] = contributor_signal.get('symbol', '')
            result['leverage'] = contributor_signal.get('leverage', 1)
            result['size'] = contributor_signal.get('size', 0.1)
            result['primary_strategy'] = best_contributor

        # Log orchestrator decision
        self.logger_manager.log_orchestrator_decision(
            strategy_signals=signals,
            final_action=best_action,
            final_confidence=best_confidence,
            weights=self.strategy_weights,
            regime=self.regime.name
        )

        return result

    def decide(self) -> Dict[str, Any]:
        """
        Main decision loop (legacy weighted voting).

        1. Update prices and data
        2. Check for forced exits (USDT accumulation mode)
        3. Detect regime
        4. Get all strategy signals
        5. Weighted vote
        6. Return final decision
        """
        # Update market data
        self.update_prices()

        # Check for forced position exits (dual portfolio mode)
        if self.dual_portfolio_mode and self.open_positions:
            exit_signals = self.check_position_exits()
            if exit_signals:
                # Return the first exit signal (highest priority)
                self.last_signals = {}  # No normal signals on forced exit
                return exit_signals[0]

        # Detect regime and adjust weights
        regime = self.detect_regime()
        self.apply_regime_weights(regime.name)

        # Get all signals
        signals = self.get_all_signals()

        # Phase 31: Store signals for logging
        self.last_signals = signals

        # Weighted vote
        decision = self.weighted_vote(signals)

        return decision

    def decide_independent(self) -> List[Dict[str, Any]]:
        """
        Phase 32: Independent strategy decisions for isolated portfolios.

        Each strategy that meets its confidence threshold returns a decision.
        No weighted voting - strategies act independently.

        Returns:
            List of decisions, one per ready strategy
        """
        # Update market data
        self.update_prices()

        # Check for forced position exits first
        exit_decisions = []
        if self.dual_portfolio_mode and self.open_positions:
            exit_signals = self.check_position_exits()
            exit_decisions = exit_signals  # These get executed first

        # Detect regime (for logging, not for weighting)
        self.detect_regime()

        # Get all signals
        signals = self.get_all_signals()
        self.last_signals = signals

        # Build list of ready decisions (strategies that meet their thresholds)
        ready_decisions = []

        for strategy_name, signal in signals.items():
            action = signal.get('action', 'hold')
            confidence = signal.get('confidence', 0)

            # Skip hold actions
            if action == 'hold':
                continue

            # Get this strategy's threshold
            params = self.registry.get_params(strategy_name)
            risk_profile = get_risk_profile(strategy_name, params)

            # Check if meets threshold
            if confidence >= risk_profile.min_confidence:
                decision = {
                    'action': action,
                    'symbol': signal.get('symbol', ''),
                    'size': signal.get('size', 0.1),
                    'leverage': signal.get('leverage', 1),
                    'confidence': confidence,
                    'strategy': strategy_name,
                    'primary_strategy': strategy_name,
                    'reason': signal.get('reason', ''),
                    'regime': self.regime.name,
                    # Preserve grid-specific fields for state sync
                    'grid_level': signal.get('grid_level'),
                    'order_id': signal.get('order_id'),
                    'price': signal.get('price', 0),
                }
                ready_decisions.append(decision)

        # Exit signals take priority, then ready decisions
        return exit_decisions + ready_decisions

    def execute_paper(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Execute decision in paper trading mode with per-strategy risk thresholds."""
        action = decision.get('action', 'hold')
        confidence = decision.get('confidence', 0)

        result = {
            'executed': False,
            'action': action,
            'reason': '',
            'decision': decision
        }

        # Phase 31: Per-strategy confidence threshold
        strategy_name = decision.get('strategy', decision.get('primary_strategy', ''))
        strategy_params = self.registry.get_params(strategy_name) if strategy_name else {}

        # Get risk profile for this strategy (with config overrides)
        risk_profile = get_risk_profile(strategy_name, strategy_params)
        min_confidence = risk_profile.min_confidence

        if confidence < min_confidence:
            result['reason'] = f"Confidence {confidence:.2f} below {strategy_name} threshold {min_confidence:.2f}"
            return result

        if action == 'hold':
            result['reason'] = "Hold action - no execution needed"
            return result

        # Execute based on action
        symbol = decision.get('symbol', '')
        size = decision.get('size', 0.1)
        leverage = decision.get('leverage', 1)
        base = symbol.split('/')[0] if symbol else ''
        price = self.current_prices.get(base, 0)

        # Dual portfolio mode: constrain size by portfolio allocation
        if self.dual_portfolio_mode:
            strategy = strategy_name
            params = strategy_params
            portfolio_type = params.get('portfolio', 'usdt')

            # Get max allocation for this portfolio
            if portfolio_type == 'usdt':
                max_portfolio_pct = self.usdt_allocation
            else:
                max_portfolio_pct = self.crypto_allocation

            # Further constrain by strategy's allocation within its portfolio
            strategy_alloc = params.get('allocation_pct', 0.10)

            # Calculate max trade size as pct of total portfolio
            max_trade_pct = max_portfolio_pct * strategy_alloc
            size = min(size, max_trade_pct)

            # Log the constraint
            if size < decision.get('size', 0.1):
                print(f"  [CONSTRAINED] Size reduced to {size*100:.1f}% ({portfolio_type} portfolio, {strategy_alloc*100:.0f}% allocation)")

        if not price:
            result['reason'] = f"No price available for {base}"
            return result

        # Phase 32: Execute through per-strategy isolated portfolio if enabled
        if self.use_isolated_portfolios and self.portfolio_manager and strategy_name:
            strategy_portfolio = self.portfolio_manager.get_portfolio(strategy_name)

            # Calculate trade cost based on signal size
            trade_cost = strategy_portfolio.get_available_usdt() * size

            # Check if strategy has enough capital
            if not strategy_portfolio.can_trade(trade_cost) and action in ['buy', 'short']:
                result['reason'] = f"Strategy '{strategy_name}' has insufficient capital (${strategy_portfolio.get_available_usdt():.2f})"
                return result

            # Execute through strategy portfolio
            if action == 'buy':
                exec_result = strategy_portfolio.execute_buy(symbol, trade_cost, price, leverage)
            elif action == 'short':
                exec_result = strategy_portfolio.execute_short(symbol, trade_cost, price, leverage)
            elif action in ['sell', 'close']:
                exec_result = strategy_portfolio.execute_sell(symbol, price, size)
            elif action == 'cover':
                exec_result = strategy_portfolio.execute_cover(symbol, price, size)
            else:
                exec_result = {'executed': False, 'reason': f'Unknown action: {action}'}

            if exec_result.get('executed'):
                result['executed'] = True
                result['amount'] = exec_result.get('size', 0)
                result['price'] = price
                result['cost'] = exec_result.get('cost', 0)
                result['pnl'] = exec_result.get('pnl', 0)
                result['reason'] = f"[{strategy_name}] {exec_result.get('action', action)} {exec_result.get('size', 0):.4f} {base} @ ${price:.2f}"
                result['strategy_balance'] = strategy_portfolio.get_available_usdt()

                # Also update shared portfolio for compatibility
                if action == 'buy':
                    self.portfolio.balances['USDT'] -= exec_result.get('cost', 0)
                    self.portfolio.balances[base] = self.portfolio.balances.get(base, 0) + exec_result.get('size', 0)
                elif action in ['sell', 'close']:
                    self.portfolio.balances[base] -= exec_result.get('size', 0)
                    self.portfolio.balances['USDT'] += exec_result.get('proceeds', 0)
            else:
                result['reason'] = exec_result.get('reason', 'Execution failed')

            # Log execution
            primary = decision.get('primary_strategy')
            if primary and primary in self.strategies:
                logger = self.logger_manager.get_logger(primary)
                logger.log_execution(
                    action=action,
                    executed=result['executed'],
                    price=price,
                    size=result.get('amount', 0),
                    reason=result['reason']
                )

            if result['executed']:
                self.execution_count += 1
                if self.dual_portfolio_mode:
                    self.track_position(decision, result)
                    self.update_crypto_metrics(decision, result)

            return result

        # Legacy: Paper execute using shared portfolio
        # Phase 25: Support for buy, sell, short, cover, close actions
        if action == 'buy':
            usdt_available = self.portfolio.balances.get('USDT', 0)
            trade_amount = usdt_available * size

            if trade_amount < 10:
                result['reason'] = f"Insufficient USDT (${usdt_available:.2f})"
                return result

            # Simulate buy
            amount = (trade_amount / price) * leverage
            self.portfolio.balances['USDT'] -= trade_amount
            self.portfolio.balances[base] = self.portfolio.balances.get(base, 0) + amount

            result['executed'] = True
            result['amount'] = amount
            result['price'] = price
            result['cost'] = trade_amount
            result['reason'] = f"Paper buy {amount:.4f} {base} @ ${price:.2f}"

        elif action == 'short':
            # Open a short position - borrow and sell asset
            # In paper trading, we track short positions as negative balances
            usdt_available = self.portfolio.balances.get('USDT', 0)
            trade_amount = usdt_available * size

            if trade_amount < 10:
                result['reason'] = f"Insufficient USDT for margin (${usdt_available:.2f})"
                return result

            # Short: receive USDT, create negative position in asset
            short_amount = (trade_amount / price) * leverage

            # Track short as negative balance (simplified margin model)
            # Collateral stays in USDT (not deducted for paper trading simplicity)
            current_position = self.portfolio.balances.get(base, 0)
            self.portfolio.balances[base] = current_position - short_amount

            # Track margin used (simplified)
            margin_key = f'{base}_margin'
            self.portfolio.balances[margin_key] = self.portfolio.balances.get(margin_key, 0) + trade_amount

            result['executed'] = True
            result['amount'] = short_amount
            result['price'] = price
            result['margin'] = trade_amount
            result['reason'] = f"Paper short {short_amount:.4f} {base} @ ${price:.2f} ({leverage}x)"

        elif action == 'cover':
            # Close a short position - buy back asset to return
            current_position = self.portfolio.balances.get(base, 0)

            if current_position >= 0:
                result['reason'] = f"No short position in {base} to cover"
                return result

            # Cover amount (negative position becomes positive cover)
            cover_amount = abs(current_position) * size
            cover_cost = cover_amount * price

            usdt_available = self.portfolio.balances.get('USDT', 0)
            margin_key = f'{base}_margin'
            margin_held = self.portfolio.balances.get(margin_key, 0)

            if cover_cost > usdt_available + margin_held:
                result['reason'] = f"Insufficient funds to cover (need ${cover_cost:.2f})"
                return result

            # Cover: reduce negative position, return margin
            self.portfolio.balances[base] = current_position + cover_amount

            # Calculate P&L from short
            # If we shorted at higher price, we profit; if lower, we lose
            # Simplified: just adjust USDT by margin - cover_cost
            if margin_held > 0:
                pnl = margin_held - cover_cost
                self.portfolio.balances['USDT'] = usdt_available + pnl
                self.portfolio.balances[margin_key] = max(0, margin_held - margin_held * size)

            result['executed'] = True
            result['amount'] = cover_amount
            result['price'] = price
            result['cost'] = cover_cost
            result['reason'] = f"Paper cover {cover_amount:.4f} {base} @ ${price:.2f}"

        elif action in ['sell', 'close']:
            asset_available = self.portfolio.balances.get(base, 0)

            if asset_available < 0.0001:
                result['reason'] = f"No {base} to sell"
                return result

            sell_amount = asset_available * size
            proceeds = sell_amount * price

            self.portfolio.balances[base] -= sell_amount
            self.portfolio.balances['USDT'] = self.portfolio.balances.get('USDT', 0) + proceeds

            result['executed'] = True
            result['amount'] = sell_amount
            result['price'] = price
            result['proceeds'] = proceeds
            result['reason'] = f"Paper sell {sell_amount:.4f} {base} @ ${price:.2f}"

        # Log execution to primary strategy's logger
        primary = decision.get('primary_strategy')
        if primary and primary in self.strategies:
            logger = self.logger_manager.get_logger(primary)
            logger.log_execution(
                action=action,
                executed=result['executed'],
                price=price,
                size=result.get('amount', 0),
                reason=result['reason']
            )

        # Phase 26: Sync grid state when order is filled
        if result['executed'] and primary and primary in self.strategies:
            strategy = self.strategies[primary]

            # Call on_order_filled callback for proper position tracking
            if hasattr(strategy, 'on_order_filled'):
                order_info = {
                    'symbol': decision.get('symbol', ''),
                    'action': action,
                    'price': result.get('price', price),
                    'amount': result.get('amount', 0),
                    'grid_level': decision.get('grid_level'),
                    'order_id': decision.get('order_id'),
                }
                try:
                    strategy.on_order_filled(order_info)
                except Exception as e:
                    print(f"Warning: on_order_filled callback error: {e}")

        if result['executed']:
            self.execution_count += 1

            # Track position for USDT accumulation (forced exits)
            if self.dual_portfolio_mode:
                self.track_position(decision, result)
                self.update_crypto_metrics(decision, result)

        return result

    def run_loop(self,
                 duration_minutes: int = 60,
                 interval_seconds: int = 300,
                 on_decision: callable = None,
                 scalper_mode: bool = False):
        """
        Run the main trading loop.

        Args:
            duration_minutes: How long to run
            interval_seconds: Seconds between decisions
            on_decision: Optional callback(decision, result)
            scalper_mode: If True, refresh data more frequently (every 5 min)
        """
        # Phase 30: Auto-enable 1m mode when polling at 60s or less
        use_1m_data = interval_seconds <= 60

        # Determine refresh intervals based on mode and interval
        if use_1m_data:
            # High-frequency mode: refresh 1m data every loop, 5m every 2 min
            data_refresh_interval = 300   # 5 minutes for full refresh
            scalp_data_refresh = 60       # 1 minute for 1m/5m data
            one_min_refresh = interval_seconds  # Refresh 1m every loop
        elif scalper_mode:
            data_refresh_interval = 300   # 5 minutes for scalper mode
            scalp_data_refresh = 120      # 2 minutes for 5m data
            one_min_refresh = 0           # No 1m data
        else:
            data_refresh_interval = 1800  # 30 minutes for normal mode
            scalp_data_refresh = 300      # 5 minutes for 5m data
            one_min_refresh = 0           # No 1m data

        print(f"\n{'='*60}")
        print(f"UNIFIED ORCHESTRATOR - Starting Paper Trading")
        print(f"Duration: {duration_minutes} min | Interval: {interval_seconds}s")
        print(f"Experiment ID: {self.experiment_id}")
        print(f"Strategies: {len(self.strategies)}")
        print(f"Scalper Mode: {'ON' if scalper_mode else 'OFF'}")
        print(f"1m Data Mode: {'ON' if use_1m_data else 'OFF'}")
        print(f"{'='*60}\n")

        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        loop_count = 0
        last_data_refresh = 0
        last_scalp_data_refresh = 0
        last_1m_refresh = 0

        try:
            while time.time() < end_time:
                loop_count += 1
                current_time = time.time()

                # Full data refresh (1h data + optionally 1m)
                if current_time - last_data_refresh > data_refresh_interval:
                    print("\n[Refreshing market data (all timeframes)...]")
                    self.update_data(include_scalping_timeframes=True, include_1m=use_1m_data)
                    last_data_refresh = current_time
                    last_scalp_data_refresh = current_time
                    last_1m_refresh = current_time

                # 1m data refresh - every loop when in high-frequency mode
                elif use_1m_data and current_time - last_1m_refresh >= one_min_refresh:
                    for symbol in ['XRP/USDT', 'BTC/USDT']:
                        kraken_symbol = symbol.replace('USDT', 'USD')
                        try:
                            df_1m = self.fetcher.fetch_ohlcv('kraken', kraken_symbol, '1m', 100, use_cache=False)
                            if df_1m is not None and not df_1m.empty:
                                self.data[f'{symbol}_1m'] = df_1m
                        except:
                            pass
                    last_1m_refresh = current_time

                # Scalper data refresh (5m/15m only) - more frequent
                elif current_time - last_scalp_data_refresh > scalp_data_refresh:
                    print("\n[Refreshing scalper data (5m/15m)...]")
                    for symbol in ['XRP/USDT', 'BTC/USDT']:
                        kraken_symbol = symbol.replace('USDT', 'USD')
                        try:
                            df_5m = self.fetcher.fetch_ohlcv('kraken', kraken_symbol, '5m', 200, use_cache=False)
                            if df_5m is not None and not df_5m.empty:
                                self.data[f'{symbol}_5m'] = df_5m
                        except:
                            pass
                    last_scalp_data_refresh = current_time

                # Phase 32: Independent vs weighted voting mode
                if self.use_isolated_portfolios and self.portfolio_manager:
                    # Independent mode: each ready strategy executes independently
                    decisions = self.decide_independent()
                else:
                    # Legacy mode: weighted voting for single decision
                    decisions = [self.decide()]

                print(f"\n[Loop {loop_count}] {datetime.now().strftime('%H:%M:%S')}")
                print(f"  Regime: {self.regime.name}")
                print(f"  Prices: BTC=${self.current_prices.get('BTC', 0):,.2f}, XRP=${self.current_prices.get('XRP', 0):.4f}")

                # Phase 31: Per-strategy signal report (enhanced with diagnostics)
                print(f"  {'─'*72}")
                print(f"  {'Strategy':<22} {'Signal':<8} {'Conf':>5} {'Thresh':>6} {'Status':<10} {'Blocker':<18}")
                print(f"  {'─'*72}")

                for name, signal in sorted(self.last_signals.items()):
                    action = signal.get('action', 'hold')
                    conf = signal.get('confidence', 0)

                    # Get this strategy's threshold
                    params = self.registry.get_params(name)
                    risk_profile = get_risk_profile(name, params)
                    threshold = risk_profile.min_confidence

                    # Determine status
                    if action == 'hold':
                        status = 'waiting'
                    elif conf >= threshold:
                        status = '✓ READY'
                    else:
                        status = 'low conf'

                    # Phase 32: Get blocking reason from diagnostics
                    blocker = ''
                    diag = signal.get('_diagnostic', {})
                    if diag.get('blocked_by'):
                        blocker = diag['blocked_by'][:18]  # Truncate for display

                    print(f"  {name:<22} {action:<8} {conf:>5.2f} {threshold:>6.2f} {status:<10} {blocker:<18}")

                print(f"  {'─'*72}")

                # Phase 32: Execute all ready strategies independently
                if self.use_isolated_portfolios and self.portfolio_manager:
                    ready_count = len(decisions)
                    exec_count = 0

                    if ready_count == 0:
                        print(f"  No strategies ready to trade")
                    else:
                        print(f"  {ready_count} strategies ready to execute:")

                        for decision in decisions:
                            result = self.execute_paper(decision)
                            strategy_name = decision.get('strategy', decision.get('primary_strategy', ''))

                            if result['executed']:
                                exec_count += 1
                                portfolio = self.portfolio_manager.get_portfolio(strategy_name)
                                print(f"    ✓ {strategy_name}: {decision['action'].upper()} "
                                      f"@ ${result.get('price', 0):.4f} | "
                                      f"Balance: ${portfolio.get_available_usdt():.2f}")

                                # Notify strategy of order fill (includes grid state sync)
                                if strategy_name in self.strategies:
                                    strategy = self.strategies[strategy_name]
                                    if hasattr(strategy, 'on_order_filled'):
                                        strategy.on_order_filled({
                                            'symbol': decision.get('symbol', ''),
                                            'action': decision.get('action', ''),
                                            'price': result.get('price', 0),
                                            'amount': result.get('amount', 0),
                                            # Grid state sync fields
                                            'grid_level': decision.get('grid_level'),
                                            'order_id': decision.get('order_id'),
                                        })

                                # Callback for each execution
                                if on_decision:
                                    on_decision(decision, result)
                            else:
                                print(f"    ✗ {strategy_name}: {result['reason']}")

                        print(f"  Executed: {exec_count}/{ready_count} trades")
                else:
                    # Legacy single decision mode
                    decision = decisions[0]
                    print(f"  Final: {decision['action'].upper()} @ {decision['confidence']:.2f} conf")

                    result = self.execute_paper(decision)
                    if result['executed']:
                        print(f"  ✓ EXECUTED: {result['reason']}")

                        # Notify strategies of order fill
                        primary = decision.get('primary_strategy')
                        if primary and primary in self.strategies:
                            strategy = self.strategies[primary]
                            if hasattr(strategy, 'on_order_filled'):
                                strategy.on_order_filled({
                                    'symbol': decision.get('symbol', ''),
                                    'action': decision.get('action', ''),
                                    'price': result.get('price', 0),
                                    'amount': result.get('amount', 0)
                                })
                    else:
                        print(f"  ✗ Skipped: {result['reason']}")

                    # Callback
                    if on_decision:
                        on_decision(decision, result)

                # Portfolio status
                if self.use_isolated_portfolios and self.portfolio_manager:
                    # Phase 32: Show aggregate from isolated portfolios + actual holdings
                    agg = self.portfolio_manager.get_aggregate_stats(self.current_prices)

                    # Get all open positions across strategies
                    all_positions = self.portfolio_manager.get_all_open_positions()
                    total_btc = sum(
                        pos['size'] for strat_positions in all_positions.values()
                        for pos in strat_positions if 'BTC' in pos['symbol'] and pos['side'] == 'long'
                    )
                    total_xrp = sum(
                        pos['size'] for strat_positions in all_positions.values()
                        for pos in strat_positions if 'XRP' in pos['symbol'] and pos['side'] == 'long'
                    )

                    print(f"  Aggregate: ${agg['total_equity']:,.2f} | "
                          f"P&L: ${agg['total_pnl']:+,.2f} ({agg['total_roi_pct']:+.2f}%) | "
                          f"Trades: {agg['total_trades']}")
                    print(f"  Holdings: BTC: {total_btc:.6f} | XRP: {total_xrp:.2f}")

                    # Per-strategy leaderboard
                    self.portfolio_manager.print_leaderboard(self.current_prices)
                else:
                    # Legacy: shared portfolio tracking
                    total_value = self.portfolio.get_total_usd(self.current_prices)
                    usdt_bal = self.portfolio.balances.get('USDT', 0)
                    btc_bal = self.portfolio.balances.get('BTC', 0)
                    xrp_bal = self.portfolio.balances.get('XRP', 0)
                    print(f"  Portfolio: ${total_value:,.2f} | USDT: {usdt_bal:,.2f} | BTC: {btc_bal:.6f} | XRP: {xrp_bal:,.2f}")

                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            print("\n\nStopping trading loop...")

        finally:
            self.close()

    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status."""
        status = {
            'experiment_id': self.experiment_id,
            'strategies_active': len(self.strategies),
            'weights': self.strategy_weights,
            'regime': {
                'name': self.regime.name,
                'volatility': self.regime.volatility,
                'correlation': self.regime.correlation,
                'trend': self.regime.trend
            },
            'decisions': self.decision_count,
            'executions': self.execution_count,
            'execution_rate': self.execution_count / max(self.decision_count, 1) * 100,
            'portfolio': {
                'balances': dict(self.portfolio.balances),
                'total_usd': self.portfolio.get_total_usd(self.current_prices)
            }
        }

        # Add dual portfolio metrics
        if self.dual_portfolio_mode:
            win_rate = (self.usdt_metrics['win_count'] /
                       max(self.usdt_metrics['trade_count'], 1)) * 100
            avg_hold = (self.usdt_metrics['total_hold_time'] /
                       max(self.usdt_metrics['trade_count'], 1))

            status['dual_portfolio'] = {
                'mode': 'enabled',
                'usdt_allocation': self.usdt_allocation,
                'crypto_allocation': self.crypto_allocation,
                'open_positions': len(self.open_positions),
            }
            status['usdt_metrics'] = {
                'total_profit': self.usdt_metrics['total_profit'],
                'trade_count': self.usdt_metrics['trade_count'],
                'win_rate': win_rate,
                'largest_win': self.usdt_metrics['largest_win'],
                'largest_loss': self.usdt_metrics['largest_loss'],
                'avg_hold_hours': avg_hold,
            }
            status['crypto_metrics'] = {
                'btc_accumulated': self.crypto_metrics['btc_accumulated'],
                'xrp_accumulated': self.crypto_metrics['xrp_accumulated'],
                'btc_avg_cost': self.crypto_metrics['btc_avg_cost'],
                'xrp_avg_cost': self.crypto_metrics['xrp_avg_cost'],
                'dca_buys': self.crypto_metrics['dca_buys'],
                'dip_buys': self.crypto_metrics['dip_buys'],
            }

        return status

    def print_status(self):
        """Print formatted status."""
        status = self.get_status()

        print("\n" + "="*60)
        print("UNIFIED ORCHESTRATOR STATUS")
        print("="*60)
        print(f"Experiment: {status['experiment_id']}")
        print(f"Regime: {status['regime']['name']} (vol: {status['regime']['volatility']*100:.2f}%)")
        print(f"Decisions: {status['decisions']} | Executions: {status['executions']} ({status['execution_rate']:.1f}%)")

        # Dual portfolio metrics
        if self.dual_portfolio_mode:
            print("-"*60)
            print("DUAL PORTFOLIO MODE")
            print(f"  USDT Allocation: {self.usdt_allocation*100:.0f}%")
            print(f"  Crypto Allocation: {self.crypto_allocation*100:.0f}%")
            print(f"  Open Positions: {len(self.open_positions)}")

            print("-"*60)
            print("USDT ACCUMULATION METRICS:")
            um = status.get('usdt_metrics', {})
            print(f"  Total Profit: ${um.get('total_profit', 0):,.2f}")
            print(f"  Trades: {um.get('trade_count', 0)} (Win Rate: {um.get('win_rate', 0):.1f}%)")
            print(f"  Largest Win: ${um.get('largest_win', 0):,.2f}")
            print(f"  Largest Loss: ${um.get('largest_loss', 0):,.2f}")
            print(f"  Avg Hold Time: {um.get('avg_hold_hours', 0):.1f}h")

            print("-"*60)
            print("CRYPTO ACCUMULATION METRICS:")
            cm = status.get('crypto_metrics', {})
            print(f"  BTC Accumulated: {cm.get('btc_accumulated', 0):.6f} (avg: ${cm.get('btc_avg_cost', 0):,.2f})")
            print(f"  XRP Accumulated: {cm.get('xrp_accumulated', 0):.2f} (avg: ${cm.get('xrp_avg_cost', 0):.4f})")
            print(f"  DCA Buys: {cm.get('dca_buys', 0)} | Dip Buys: {cm.get('dip_buys', 0)}")

        print("-"*60)
        print("Strategy Weights:")
        for name, weight in sorted(status['weights'].items(), key=lambda x: -x[1])[:10]:
            print(f"  {name}: {weight*100:.1f}%")
        if len(status['weights']) > 10:
            print(f"  ... and {len(status['weights']) - 10} more")

        print("-"*60)
        print(f"Portfolio: ${status['portfolio']['total_usd']:,.2f}")
        for asset, bal in status['portfolio']['balances'].items():
            if bal > 0.0001 and not asset.endswith('_margin'):
                print(f"  {asset}: {bal:.4f}")

        # Phase 32: Per-strategy portfolio leaderboard
        if self.use_isolated_portfolios and self.portfolio_manager:
            self.portfolio_manager.print_leaderboard(self.current_prices)

        print("="*60 + "\n")

    def get_diagnostic_summary(self, strategy_name: str = None) -> Dict[str, Any]:
        """
        Get diagnostic summary for one or all strategies.

        Args:
            strategy_name: Optional specific strategy, or None for all

        Returns:
            Diagnostic summary dict
        """
        if not self.diagnostics:
            return {'error': 'Diagnostics not enabled'}

        if strategy_name:
            return self.diagnostics.get_strategy_summary(strategy_name)
        return self.diagnostics.get_all_summaries()

    def print_diagnostic_report(self):
        """Print detailed diagnostic report for all strategies."""
        if self.diagnostics:
            self.diagnostics.print_diagnostic_report()
        else:
            print("Diagnostics not enabled")

    def print_inactive_strategies(self):
        """Print analysis of strategies that haven't traded."""
        if not self.diagnostics:
            print("Diagnostics not enabled")
            return

        print("\n" + "=" * 70)
        print("INACTIVE STRATEGY ANALYSIS")
        print("=" * 70)

        summaries = self.diagnostics.get_all_summaries()
        inactive = []

        for name, summary in summaries.get('strategy_summaries', {}).items():
            if summary.get('signals', 0) == 0:
                inactive.append((name, summary))

        if not inactive:
            print("All strategies have generated signals!")
            return

        print(f"\n{len(inactive)} strategies have not traded:\n")

        for name, summary in inactive:
            print(f"[{name}]")

            # Show blocking reasons
            if summary.get('blocking_reasons'):
                reasons = summary['blocking_reasons']
                total_blocks = sum(reasons.values())
                print(f"  Blocked {total_blocks} times:")
                for reason, count in sorted(reasons.items(), key=lambda x: -x[1])[:3]:
                    pct = count / total_blocks * 100
                    print(f"    - {reason}: {count} ({pct:.0f}%)")

            # Show threshold failures
            if summary.get('threshold_failures'):
                failures = summary['threshold_failures']
                print(f"  Failed thresholds:")
                for threshold, count in sorted(failures.items(), key=lambda x: -x[1])[:3]:
                    print(f"    - {threshold}: {count} times")

            # Show latest indicator values
            indicators = summary.get('latest_indicators', {})
            if indicators:
                key_indicators = {k: v for k, v in indicators.items()
                                 if any(x in k.lower() for x in ['rsi', 'drawdown', 'volume', 'atr'])}
                if key_indicators:
                    formatted = ', '.join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                                         for k, v in list(key_indicators.items())[:4])
                    print(f"  Latest: {formatted}")

            # Show recommendations
            if summary.get('recommendations'):
                print(f"  Recommendation: {summary['recommendations'][0]}")

            print()

        print("=" * 70)

    def close(self):
        """Close orchestrator and generate summaries."""
        print("\n" + "="*60)
        print("CLOSING UNIFIED ORCHESTRATOR")
        print("="*60)

        # Close all loggers
        summary = self.logger_manager.close_all()

        # Phase 32: Print diagnostic report and close diagnostics
        if self.diagnostics:
            self.print_inactive_strategies()
            diag_summary = self.diagnostics.close()
            summary['diagnostics'] = diag_summary

        # Print final status
        self.print_status()

        return summary


if __name__ == "__main__":
    # Test run
    from portfolio import Portfolio

    config = {
        'starting_balance': {'USDT': 2000.0, 'XRP': 0.0, 'BTC': 0.0}
    }
    portfolio = Portfolio(config['starting_balance'])

    # Create config template first
    from strategy_registry import create_unified_config_template
    config_path = create_unified_config_template()

    orchestrator = UnifiedOrchestrator(
        portfolio=portfolio,
        config_path=config_path,
        experiment_id="test_run"
    )

    orchestrator.print_status()
