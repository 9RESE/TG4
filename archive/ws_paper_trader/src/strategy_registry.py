"""
Phase 24: Strategy Registry
Dynamic strategy loading, configuration, and management.

Features:
- Auto-discovery of strategies in src/strategies/
- YAML-based configuration with hot-reload
- Easy enable/disable per strategy
- Parameter override for experiments
"""

import os
import sys
import yaml
import importlib
from pathlib import Path
from typing import Dict, Any, Optional, List, Type
from dataclasses import dataclass, field

# Ensure src is in path
sys.path.insert(0, os.path.dirname(__file__))

from strategies.base_strategy import BaseStrategy


@dataclass
class StrategyInfo:
    """Metadata about a registered strategy."""
    name: str
    class_type: Type[BaseStrategy]
    module_path: str
    description: str = ""
    category: str = "general"  # general, grid, margin, scalper
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)
    default_params: Dict[str, Any] = field(default_factory=dict)


class StrategyRegistry:
    """
    Central registry for all trading strategies.

    Usage:
        registry = StrategyRegistry()
        registry.load_config('strategies_config/unified.yaml')

        # Get all enabled strategies
        strategies = registry.get_enabled_strategies()

        # Override params for experiment
        registry.override_param('mean_reversion_vwap', 'rsi_oversold', 30)

        # Disable a strategy
        registry.disable('defensive_yield')

        # Enable a strategy
        registry.enable('grid_arithmetic')
    """

    # Built-in strategy mappings (name -> (module, class))
    BUILTIN_STRATEGIES = {
        # Phase 21 Ensemble Strategies
        'defensive_yield': ('strategies.defensive_yield.strategy', 'DefensiveYield'),
        'mean_reversion_vwap': ('strategies.mean_reversion_vwap.strategy', 'MeanReversionVWAP'),
        'xrp_btc_pair_trading': ('strategies.xrp_btc_pair_trading.strategy', 'XRPBTCPairTrading'),
        'intraday_scalper': ('strategies.intraday_scalper.strategy', 'IntraDayScalper'),
        'ma_trend_follow': ('strategies.ma_trend_follow.strategy', 'MATrendFollow'),
        'xrp_btc_leadlag': ('strategies.xrp_btc_leadlag.strategy', 'XRPBTCLeadLag'),
        'ema9_scalper': ('strategies.ema9_scalper.strategy', 'EMA9Scalper'),

        # Phase 24 New Strategies
        'xrp_momentum_lstm': ('strategies.xrp_momentum_lstm.strategy', 'XRPMomentumLSTM'),
        'dip_detector': ('strategies.dip_detector.strategy', 'DipDetector'),
        'mean_reversion_short': ('strategies.mean_reversion_short.strategy', 'MeanReversionShort'),
        'portfolio_rebalancer': ('strategies.portfolio_rebalancer.strategy', 'PortfolioRebalancer'),

        # Grid Strategies (using wrappers for BaseStrategy compatibility)
        'grid_arithmetic': ('strategies.grid_wrappers', 'ArithmeticGridWrapper'),
        'grid_geometric': ('strategies.grid_wrappers', 'GeometricGridWrapper'),
        'grid_rsi_reversion': ('strategies.grid_wrappers', 'RSIMeanReversionGridWrapper'),
        'grid_bb_squeeze': ('strategies.grid_wrappers', 'BBSqueezeGridWrapper'),

        # Margin Grid Strategies (using wrappers)
        'grid_trend_margin': ('strategies.grid_wrappers', 'TrendFollowingMarginWrapper'),
        'grid_dual_hedge': ('strategies.grid_wrappers', 'DualGridHedgeWrapper'),
        'grid_time_weighted': ('strategies.grid_wrappers', 'TimeWeightedGridWrapper'),
        'grid_liq_hunter': ('strategies.grid_wrappers', 'LiquidationHuntScalperWrapper'),

        # Phase 30: Research-Based Strategies (2025)
        'supertrend': ('strategies.supertrend.strategy', 'SuperTrendStrategy'),
        'funding_rate_arb': ('strategies.funding_rate_arb.strategy', 'FundingRateArbitrage'),
        'triangular_arb': ('strategies.triangular_arb.strategy', 'TriangularArbitrage'),
        'volatility_breakout': ('strategies.volatility_breakout.strategy', 'VolatilityBreakout'),
        'ichimoku_cloud': ('strategies.ichimoku_cloud.strategy', 'IchimokuCloud'),
        'multi_indicator_confluence': ('strategies.multi_indicator_confluence.strategy', 'MultiIndicatorConfluence'),
        'whale_sentiment': ('strategies.whale_sentiment.strategy', 'WhaleSentiment'),
        'enhanced_dca': ('strategies.enhanced_dca.strategy', 'EnhancedDCA'),
        'twap_accumulator': ('strategies.twap_accumulator.strategy', 'TWAPAccumulator'),
        'volume_profile': ('strategies.volume_profile.strategy', 'VolumeProfile'),
        'wavetrend': ('strategies.wavetrend.strategy', 'WaveTrend'),
        'scalping_1m5m': ('strategies.scalping_1m5m.strategy', 'Scalping1m5m'),
    }

    # Phase 31: Category map aligned with risk_manager.StrategyCategory
    # Categories: arbitrage, scalping, mean_reversion, trend_following,
    #             accumulation, pair_trading, grid, sentiment
    CATEGORY_MAP = {
        # Accumulation strategies
        'defensive_yield': 'accumulation',
        'enhanced_dca': 'accumulation',
        'twap_accumulator': 'accumulation',
        'dip_detector': 'accumulation',
        'portfolio_rebalancer': 'accumulation',

        # Mean reversion strategies
        'mean_reversion_vwap': 'mean_reversion',
        'mean_reversion_short': 'mean_reversion',
        'wavetrend': 'mean_reversion',
        'volume_profile': 'mean_reversion',

        # Scalping strategies
        'intraday_scalper': 'scalping',
        'ema9_scalper': 'scalping',
        'scalping_1m5m': 'scalping',

        # Pair trading strategies
        'xrp_btc_pair_trading': 'pair_trading',
        'xrp_btc_leadlag': 'pair_trading',

        # Trend following strategies
        'ma_trend_follow': 'trend_following',
        'supertrend': 'trend_following',
        'ichimoku_cloud': 'trend_following',
        'volatility_breakout': 'trend_following',
        'multi_indicator_confluence': 'trend_following',
        'xrp_momentum_lstm': 'trend_following',

        # Arbitrage strategies
        'funding_rate_arb': 'arbitrage',
        'triangular_arb': 'arbitrage',

        # Grid strategies
        'grid_arithmetic': 'grid',
        'grid_geometric': 'grid',
        'grid_rsi_reversion': 'grid',
        'grid_bb_squeeze': 'grid',
        'grid_trend_margin': 'grid',
        'grid_dual_hedge': 'grid',
        'grid_time_weighted': 'grid',
        'grid_liq_hunter': 'grid',

        # Sentiment strategies
        'whale_sentiment': 'sentiment',
    }

    def __init__(self, config_path: str = None):
        self.strategies: Dict[str, StrategyInfo] = {}
        self.config_path = config_path
        self.global_config: Dict[str, Any] = {}
        self.experiment_overrides: Dict[str, Dict[str, Any]] = {}

        # Auto-register built-in strategies
        self._register_builtins()

        # Load config if provided
        if config_path:
            self.load_config(config_path)

    def _register_builtins(self):
        """Register all built-in strategies."""
        for name, (module_path, class_name) in self.BUILTIN_STRATEGIES.items():
            try:
                self.strategies[name] = StrategyInfo(
                    name=name,
                    class_type=None,  # Lazy-loaded
                    module_path=f"{module_path}.{class_name}",
                    category=self.CATEGORY_MAP.get(name, 'general'),
                    enabled=False,  # Disabled by default until config loaded
                    default_params={}
                )
            except Exception as e:
                print(f"Warning: Could not register {name}: {e}")

    def _load_class(self, name: str) -> Optional[Type[BaseStrategy]]:
        """Lazy-load a strategy class."""
        if name not in self.BUILTIN_STRATEGIES:
            return None

        module_path, class_name = self.BUILTIN_STRATEGIES[name]

        try:
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            return cls
        except Exception as e:
            print(f"Error loading {name}: {e}")
            return None

    def load_config(self, config_path: str):
        """Load strategy configuration from YAML."""
        self.config_path = config_path

        if not os.path.exists(config_path):
            print(f"Warning: Config file not found: {config_path}")
            return

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.global_config = config.get('global', {})

        # Load strategy configs
        strategies_config = config.get('strategies', {})
        for name, strat_config in strategies_config.items():
            if name in self.strategies:
                info = self.strategies[name]
                info.enabled = strat_config.get('enabled', False)
                info.config = strat_config
                info.description = strat_config.get('description', '')

                # Extract default params (everything except metadata)
                info.default_params = {
                    k: v for k, v in strat_config.items()
                    if k not in ['enabled', 'description', 'notes', 'category']
                }
            else:
                print(f"Warning: Unknown strategy in config: {name}")

        enabled_count = len([s for s in self.strategies.values() if s.enabled])
        print(f"Loaded config: {len(strategies_config)} strategies, {enabled_count} enabled")

    def reload_config(self):
        """Reload configuration from file."""
        if self.config_path:
            self.load_config(self.config_path)

    def register(self,
                 name: str,
                 strategy_class: Type[BaseStrategy],
                 category: str = "custom",
                 description: str = "",
                 default_params: Dict[str, Any] = None):
        """Register a custom strategy."""
        self.strategies[name] = StrategyInfo(
            name=name,
            class_type=strategy_class,
            module_path="custom",
            category=category,
            description=description,
            enabled=True,
            default_params=default_params or {}
        )

    def enable(self, name: str) -> bool:
        """Enable a strategy."""
        if name in self.strategies:
            self.strategies[name].enabled = True
            return True
        return False

    def disable(self, name: str) -> bool:
        """Disable a strategy."""
        if name in self.strategies:
            self.strategies[name].enabled = False
            return True
        return False

    def enable_category(self, category: str):
        """Enable all strategies in a category."""
        for info in self.strategies.values():
            if info.category == category:
                info.enabled = True

    def disable_category(self, category: str):
        """Disable all strategies in a category."""
        for info in self.strategies.values():
            if info.category == category:
                info.enabled = False

    def enable_all(self):
        """Enable all strategies."""
        for info in self.strategies.values():
            info.enabled = True

    def disable_all(self):
        """Disable all strategies."""
        for info in self.strategies.values():
            info.enabled = False

    def override_param(self, strategy_name: str, param_name: str, value: Any):
        """Override a parameter for experiments."""
        if strategy_name not in self.experiment_overrides:
            self.experiment_overrides[strategy_name] = {}
        self.experiment_overrides[strategy_name][param_name] = value

    def clear_overrides(self, strategy_name: str = None):
        """Clear experiment overrides."""
        if strategy_name:
            self.experiment_overrides.pop(strategy_name, None)
        else:
            self.experiment_overrides.clear()

    def get_params(self, name: str) -> Dict[str, Any]:
        """Get merged params (default + config + overrides)."""
        if name not in self.strategies:
            return {}

        info = self.strategies[name]
        params = {**info.default_params, **info.config}

        # Apply experiment overrides
        if name in self.experiment_overrides:
            params.update(self.experiment_overrides[name])

        return params

    def get_enabled_strategies(self) -> List[str]:
        """Get list of enabled strategy names."""
        return [name for name, info in self.strategies.items() if info.enabled]

    def get_strategies_by_category(self, category: str) -> List[str]:
        """Get strategies in a category."""
        return [name for name, info in self.strategies.items() if info.category == category]

    def instantiate(self, name: str, extra_config: Dict[str, Any] = None) -> Optional[BaseStrategy]:
        """
        Instantiate a strategy with its configuration.

        Args:
            name: Strategy name
            extra_config: Additional config to merge

        Returns:
            Instantiated strategy or None if failed
        """
        if name not in self.strategies:
            print(f"Unknown strategy: {name}")
            return None

        info = self.strategies[name]

        # Lazy-load class if needed
        if info.class_type is None:
            info.class_type = self._load_class(name)

        if info.class_type is None:
            print(f"Could not load class for: {name}")
            return None

        # Build config
        config = {
            'name': name,
            **self.global_config,
            **self.get_params(name),
            **(extra_config or {})
        }

        try:
            return info.class_type(config)
        except Exception as e:
            print(f"Error instantiating {name}: {e}")
            return None

    def instantiate_all_enabled(self, extra_config: Dict[str, Any] = None) -> Dict[str, BaseStrategy]:
        """Instantiate all enabled strategies."""
        strategies = {}

        for name in self.get_enabled_strategies():
            strategy = self.instantiate(name, extra_config)
            if strategy:
                strategies[name] = strategy

        return strategies

    def get_status(self) -> Dict[str, Any]:
        """Get registry status."""
        return {
            'total_registered': len(self.strategies),
            'enabled': len(self.get_enabled_strategies()),
            'by_category': {
                cat: len(self.get_strategies_by_category(cat))
                for cat in ['general', 'lstm', 'scalper', 'grid', 'margin', 'utility', 'trend',
                           'arbitrage', 'breakout', 'confluence', 'sentiment', 'accumulation',
                           'volume', 'momentum', 'custom']
            },
            'strategies': {
                name: {
                    'enabled': info.enabled,
                    'category': info.category,
                    'description': info.description
                }
                for name, info in self.strategies.items()
            },
            'experiment_overrides': self.experiment_overrides
        }

    def print_status(self):
        """Print formatted registry status."""
        status = self.get_status()

        print("\n" + "="*60)
        print("STRATEGY REGISTRY STATUS")
        print("="*60)
        print(f"Total registered: {status['total_registered']}")
        print(f"Enabled: {status['enabled']}")
        print("-"*60)

        # Group by category
        for category in ['general', 'lstm', 'scalper', 'grid', 'margin', 'utility', 'trend',
                        'arbitrage', 'breakout', 'confluence', 'sentiment', 'accumulation',
                        'volume', 'momentum', 'custom']:
            strategies = self.get_strategies_by_category(category)
            if strategies:
                print(f"\n[{category.upper()}]")
                for name in strategies:
                    info = self.strategies[name]
                    status_icon = "+" if info.enabled else "-"
                    print(f"  {status_icon} {name}: {info.description or '(no description)'}")

        if self.experiment_overrides:
            print("\n[EXPERIMENT OVERRIDES]")
            for name, overrides in self.experiment_overrides.items():
                print(f"  {name}: {overrides}")

        print("="*60 + "\n")


def create_unified_config_template(output_path: str = "strategies_config/unified.yaml"):
    """Create a template unified configuration file."""

    template = """# Unified Strategy Configuration
# Phase 24: Combined configuration for all strategies
#
# To enable a strategy, set enabled: true
# To experiment with parameters, modify values and re-run

global:
  # Default settings applied to all strategies
  paper_trading: true
  starting_balance:
    USDT: 2000.0
    XRP: 0.0
    BTC: 0.0
  fee_rate: 0.001  # 0.1%
  max_drawdown: 0.10  # 10%

  # Risk management
  risk_per_trade: 0.10  # 10% of capital per trade
  max_concurrent_positions: 5

strategies:
  # ===== GENERAL STRATEGIES =====

  defensive_yield:
    enabled: true
    category: general
    description: "RL-driven defensive strategy with yield accrual"
    max_leverage: 10
    modes:
      defensive_atr: 0.04  # ATR > 4% triggers defensive
      offensive_atr: 0.02  # ATR < 2% triggers offensive
    yield:
      usdt_apy: 0.065  # 6.5% APY

  mean_reversion_vwap:
    enabled: true
    category: general
    description: "VWAP + RSI mean reversion for choppy markets"
    symbol: "XRP/USDT"
    # Tunable parameters
    dev_threshold: 0.003      # VWAP deviation threshold
    rsi_oversold: 35          # RSI oversold level
    rsi_overbought: 65        # RSI overbought level
    volume_mult: 1.3          # Volume filter multiplier
    max_leverage: 5
    stop_loss_pct: 0.04

  xrp_btc_pair_trading:
    enabled: true
    category: general
    description: "XRP/BTC cointegration stat arb"
    xrp_symbol: "XRP/USDT"
    btc_symbol: "BTC/USDT"
    # Tunable parameters
    lookback: 336             # Lookback period (hours)
    entry_z: 1.8              # Z-score entry threshold
    exit_z: 0.5               # Z-score exit threshold
    stop_z: 3.0               # Z-score stop loss
    max_leverage: 10

  ma_trend_follow:
    enabled: true
    category: general
    description: "SMA-9 trend following with breakout detection"
    ma_period: 9
    confirmation_candles: 1
    max_leverage: 5

  xrp_btc_leadlag:
    enabled: true
    category: general
    description: "XRP/BTC correlation lead-lag trading"
    high_corr_threshold: 0.8
    low_corr_threshold: 0.6
    breakout_leverage: 7
    min_btc_move: 0.005

  # ===== SCALPER STRATEGIES =====

  intraday_scalper:
    enabled: true
    category: scalper
    description: "Phase 25 BB squeeze + RSI + ADX volatility scalper"

    # Indicator periods (optimized for scalping)
    bb_period: 12             # Bollinger Band period (faster than default 20)
    bb_std: 2.0               # Standard deviations
    rsi_period: 7             # RSI period (faster than default 14)
    ema_period: 9             # EMA trend filter period
    adx_period: 14            # ADX period for ranging detection
    atr_period: 14            # ATR period for volatility

    # Volatility thresholds
    vol_threshold: 0.5        # Min ATR% to activate scalper
    vol_threshold_high: 3.0   # Max ATR% (too volatile)

    # RSI thresholds (tighter for crypto)
    rsi_oversold: 25          # Oversold level
    rsi_overbought: 75        # Overbought level

    # ADX filter (ranging market detection)
    adx_max: 25               # Only trade when ADX < 25
    use_adx_filter: true      # Enable ADX filter

    # Volume filter
    volume_mult: 1.3          # Volume > 1.3x average
    volume_window: 20         # Volume lookback window
    use_volume_filter: true   # Enable volume filter

    # Squeeze detection
    squeeze_threshold: 0.8    # Band width < 80% of average
    require_squeeze: false    # Strict mode (require squeeze)

    # Position sizing
    base_size_pct: 0.08       # 8% base position
    dynamic_sizing: true      # ATR-based sizing
    min_size_pct: 0.03        # Min 3%
    max_size_pct: 0.15        # Max 15%

    # Risk management
    max_leverage: 3           # Max 3x leverage for scalps
    scalp_target_pct: 0.01    # 1% take profit
    scalp_stop_pct: 0.005     # 0.5% stop loss

    # Trade cooldown
    cooldown_bars: 3          # Min bars between trades
    cooldown_minutes: 15      # Min minutes between trades

    # Symbols
    symbols:
      - "BTC/USDT"
      - "XRP/USDT"

  ema9_scalper:
    enabled: false  # Optional override scalper
    category: scalper
    description: "Phase 25 EMA-9 crossover with multi-filter confirmation"

    # Core EMA parameters
    ema_period: 9             # Fast EMA for crossover
    ema_slow_period: 21       # Slow EMA for trend filter
    timeframe: "5m"           # Primary timeframe

    # Leverage and sizing (reduced for safety)
    leverage: 3               # Reduced from 5x to 3x
    size_pct: 0.08            # 8% base position
    dynamic_sizing: true      # ATR-based sizing
    min_size_pct: 0.03        # Min 3%
    max_size_pct: 0.12        # Max 12%

    # Volatility thresholds
    atr_period: 14
    atr_threshold: 0.5        # Min ATR% to trade
    atr_threshold_high: 3.0   # Max ATR% (too volatile)
    override_atr_threshold: 1.8  # Override RL gate

    # RSI filter
    rsi_period: 14
    rsi_buy_max: 40           # Only buy when RSI < 40
    rsi_sell_min: 60          # Only sell when RSI > 60
    use_rsi_filter: true

    # ADX filter (ranging market)
    adx_period: 14
    adx_max: 30               # Only trade when ADX < 30
    use_adx_filter: true

    # Volume filter
    volume_mult: 1.2
    volume_window: 20
    use_volume_filter: true

    # Trend filter
    use_trend_filter: true
    trend_tolerance: 0.002

    # Risk management (now implemented!)
    stop_loss_pct: 0.01       # 1% stop loss
    take_profit_pct: 0.02     # 2% take profit
    max_positions: 2          # Enforced limit

    # Trade cooldown
    cooldown_bars: 3
    cooldown_minutes: 10

    # Crossover confirmation
    require_confirmation: true
    confirmation_threshold: 0.001

    # Symbols
    symbols:
      - "BTC/USDT"
      - "XRP/USDT"

  # ===== GRID STRATEGIES =====

  grid_arithmetic:
    enabled: true  # Phase 26: Enabled by default
    category: grid
    description: "Phase 26 Enhanced: Dynamic ATR-based grid with compound profits"

    # Grid structure
    num_grids: 15              # Fewer, wider grids for better profit per cycle
    leverage: 1                # Spot only for safety

    # Phase 26: Dynamic range (calculated from live price)
    range_pct: 0.04            # 4% above/below current price
    recenter_threshold: 0.02   # Recenter when 2% outside range
    recenter_after_cycles: 5   # Or after 5 completed cycles
    min_recenter_interval: 3600  # Min 1 hour between recenters

    # Phase 26: ATR-based dynamic spacing
    use_atr_spacing: true
    atr_multiplier: 0.3        # Each grid = 0.3 * ATR
    atr_period: 14

    # Phase 26: Compound position sizing
    compound_profits: true
    max_compound: 1.5          # Max 150% of initial capital
    profit_distribution:
      reinvest: 0.6            # 60% reinvested
      realized: 0.4            # 40% taken as profit

    # Phase 26: Multi-asset support
    secondary_symbol: "XRP/USDT"
    secondary_allocation: 0.3   # 30% to XRP grid

    # Phase 26: Slippage handling
    slippage_tolerance: 0.002   # 0.2% tolerance
    min_profit_per_cycle: 0.001 # 0.1% minimum profit

    # Risk management
    max_drawdown: 0.08          # 8% max drawdown
    fee_rate: 0.001             # 0.1% fee

  grid_geometric:
    enabled: false
    category: grid
    description: "Percentage spacing grid - extreme bias"
    num_grids: 15
    grid_ratio: 1.008
    leverage: 1

  grid_rsi_reversion:
    enabled: false
    category: grid
    description: "Phase 26 RSI-enhanced mean reversion grid"
    num_grids: 12
    leverage: 2
    # RSI parameters (relaxed for more signals)
    rsi_period: 14
    rsi_oversold: 30
    rsi_overbought: 70
    rsi_mode: confidence_only  # RSI modifies confidence, doesn't block trades
    # Phase 26 features
    range_pct: 0.05
    recenter_after_cycles: 4
    use_atr_spacing: true
    atr_multiplier: 0.4
    use_adaptive_rsi: true
    rsi_extreme_multiplier: 1.3
    compound_profits: true
    max_compound: 1.4
    slippage_tolerance: 0.003

  grid_bb_squeeze:
    enabled: false
    category: grid
    description: "TTM Squeeze Enhanced Grid - Keltner Channel + Momentum confirmation"
    num_grids: 10
    range_pct: 0.03
    leverage: 2
    bb_period: 20
    bb_std: 2.0
    kc_period: 20
    kc_atr_mult_high: 1.0
    kc_atr_mult_mid: 1.5
    kc_atr_mult_low: 2.0
    momentum_period: 12
    squeeze_min_bars: 6
    squeeze_confirmation: true
    breakout_size_pct: 0.15
    stop_loss_atr_mult: 1.5
    take_profit_atr_mult: 3.0

  # ===== MARGIN STRATEGIES =====

  grid_trend_margin:
    enabled: true
    category: margin
    description: "Phase 27: Dynamic trend-following margin grid with ATR-based risk"
    # Dynamic trendline - auto-calculates from swing lows
    use_dynamic_trendline: true
    trendline_lookback: 72
    trendline_recalc_interval: 4
    min_swing_points: 3
    # Grid entry levels
    num_grid_levels: 5
    grid_spacing_pct: 0.015
    max_positions: 3
    # Entry conditions (relaxed)
    entry_tolerance: 0.01
    rsi_threshold: 40
    entry_mode: or
    # ATR-based stops
    use_atr_stops: true
    stop_loss_atr_mult: 2.0
    take_profit_atr_mult: 3.0
    # Dynamic leverage
    use_dynamic_leverage: true
    leverage: 3
    max_leverage: 5
    min_leverage: 1
    # Sizing
    size_pct: 0.08
    allow_shorts: true

  grid_dual_hedge:
    enabled: true
    category: margin
    description: "Phase 28: Enhanced dual-grid with dynamic margin hedging"

    # Capital allocation (80% grid, 20% hedge reserve)
    grid_allocation: 0.80
    hedge_allocation: 0.20

    # Grid parameters
    num_grids: 18
    fee_rate: 0.001

    # ATR-based dynamic hedge triggers
    use_atr_triggers: true
    atr_period: 14
    atr_trigger_mult: 1.5      # Trigger hedge at 1.5x ATR beyond grid

    # Fallback static triggers (used if ATR unavailable)
    static_trigger_pct: 0.015  # 1.5% beyond grid

    # Hedge leverage (dynamically adjusted based on volatility)
    hedge_leverage: 2
    max_hedge_leverage: 3      # Max in low volatility
    min_hedge_leverage: 1      # Min in high volatility

    # Stop loss / Take profit (ATR-based)
    stop_loss_atr_mult: 2.0    # 2x ATR stop loss
    take_profit_atr_mult: 3.0  # 3x ATR take profit
    stop_loss_pct: 0.02        # Fallback: 2%
    take_profit_pct: 0.03      # Fallback: 3%

    # Trailing stop for hedges
    use_trailing_stop: true
    trailing_activation_atr: 1.5  # Activate after 1.5x ATR profit
    trailing_distance_atr: 1.0    # Trail by 1x ATR

    # Volatility-adjusted position sizing
    use_vol_sizing: true
    base_size_pct: 0.5         # 50% of hedge capital per trade
    min_size_pct: 0.25         # Min 25% (high volatility)
    max_size_pct: 0.75         # Max 75% (low volatility)
    target_vol: 0.02           # Target 2% volatility

    # Grid recentering after hedge closes
    recenter_after_hedge: true
    recenter_threshold: 0.03   # 3% price move threshold
    min_recenter_interval: 3600  # Min 1 hour between recenters

    # Session-aware trigger adjustments
    use_session_adjustment: true
    session_multipliers:
      asia: 0.8                # 00:00-08:00 UTC - tighter triggers
      europe: 1.0              # 08:00-14:00 UTC - standard
      us: 1.2                  # 14:00-21:00 UTC - wider (more volatile)
      overnight: 0.9           # 21:00-00:00 UTC - slightly tighter

    # Compound profit reinvestment
    compound_profits: true
    profit_reinvest_ratio: 0.6  # 60% reinvested, 40% realized

  grid_time_weighted:
    enabled: false
    category: grid
    description: "Phase 27: Session-aware TWAP grid with ATR adjustment"
    num_grids: 15
    range_pct: 0.04
    use_atr_adjustment: true
    use_twap: true
    reduce_weekend_size: true
    compound_profits: true

  grid_liq_hunter:
    enabled: true
    category: margin
    description: "Phase 29: Enhanced liquidation hunt scalper with ATR zones & pattern detection"

    # ===== Leverage Configuration =====
    leverage: 3                      # Base leverage
    use_dynamic_leverage: true       # Adjust based on volatility
    max_leverage: 5                  # Low volatility = higher leverage
    min_leverage: 2                  # High volatility = lower leverage
    high_vol_threshold: 0.03         # 3% ATR = reduce leverage
    low_vol_threshold: 0.015         # 1.5% ATR = increase leverage

    # ===== Position Sizing =====
    size_pct: 0.08                   # 8% per trade
    min_size_pct: 0.04               # Minimum 4%
    max_size_pct: 0.15               # Maximum 15%
    max_positions: 2                 # Max concurrent positions

    # ===== ATR-Based Liquidation Zone Detection =====
    use_atr_zones: true              # Use ATR for zone calculation
    liq_zone_atr_mult: 1.5           # 1.5x ATR from swing point
    liq_zone_pct: 0.02               # Fallback: 2% fixed if ATR unavailable
    atr_period: 14                   # ATR calculation period

    # ===== Swing Detection =====
    lookback_bars: 50                # Price history for swing detection
    swing_lookback: 5                # Bars to confirm swing high/low
    min_swing_size_atr: 0.5          # Minimum swing = 0.5x ATR

    # ===== Volume Confirmation =====
    use_volume_filter: true          # Require volume spike for entry
    volume_spike_mult: 1.5           # Volume > 1.5x average
    volume_window: 20                # Volume averaging window

    # ===== RSI Filter =====
    use_rsi_filter: true             # Require RSI confirmation
    rsi_period: 14                   # RSI calculation period
    rsi_oversold: 30                 # Long entry when RSI < 30
    rsi_overbought: 70               # Short entry when RSI > 70
    rsi_extreme_oversold: 20         # Extra confidence boost
    rsi_extreme_overbought: 80       # Extra confidence boost

    # ===== Reversal Pattern Settings =====
    min_wick_ratio: 2.0              # Wick >= 2x body for hammer
    engulfing_body_ratio: 1.1        # Engulfing body 10% larger
    require_pattern: true            # Must have reversal pattern to enter
    pattern_confidence_boost: 0.15   # Pattern adds 15% confidence

    # ===== ATR-Based Risk Management =====
    use_atr_stops: true              # Use ATR for stops
    stop_loss_atr_mult: 1.5          # Stop = 1.5x ATR
    take_profit_atr_mult: 2.5        # TP = 2.5x ATR
    stop_loss_pct: 0.015             # Fallback: 1.5%
    take_profit_pct: 0.03            # Fallback: 3%

    # ===== Trailing Stop =====
    use_trailing_stop: true          # Enable trailing stops
    trailing_activation_atr: 1.5     # Activate after 1.5x ATR profit
    trailing_distance_atr: 1.0       # Trail by 1x ATR

    # ===== Session Awareness =====
    use_session_adjustment: true     # Adjust zones by session
    session_multipliers:
      asia: 0.8                      # 00:00-08:00 UTC - tighter zones
      europe: 1.0                    # 08:00-14:00 UTC - standard
      us: 1.2                        # 14:00-21:00 UTC - wider zones
      overnight: 0.9                 # 21:00-00:00 UTC - slightly tighter

    # ===== Compound Profits =====
    compound_profits: true           # Reinvest profits
    max_compound: 1.5                # Max 150% of initial capital
    profit_reinvest_ratio: 0.6       # 60% reinvested, 40% realized

    # ===== Trade Cooldown =====
    cooldown_bars: 3                 # Min bars between trades

# Experiment presets - quick parameter sets for A/B testing
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
        rsi_overbought: 62
      xrp_btc_pair_trading:
        entry_z: 2.2

  grid_focus:
    description: "Enable grid strategies only"
    enable_only:
      - grid_arithmetic
      - grid_geometric
      - grid_rsi_reversion
      - grid_bb_squeeze
"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(template)

    print(f"Created unified config template: {output_path}")
    return output_path


if __name__ == "__main__":
    # Create template and test registry
    config_path = create_unified_config_template()

    registry = StrategyRegistry(config_path)
    registry.print_status()
