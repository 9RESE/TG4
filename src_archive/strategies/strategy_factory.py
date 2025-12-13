"""
Strategy Factory - Modular Strategy Loader
Phase 15: Modular Strategy Factory

Loads and initializes strategies based on configuration.
Provides a unified interface for strategy management.
"""
import os
import yaml
from typing import Dict, Any, Optional, Type

from strategies.base_strategy import BaseStrategy
from strategies.defensive_yield import DefensiveYield
from strategies.mean_reversion_vwap import MeanReversionVWAP
from strategies.xrp_btc_pair_trading import XRPBTCPairTrading


# Registry of available strategies
STRATEGY_REGISTRY: Dict[str, Type[BaseStrategy]] = {
    'defensive_yield': DefensiveYield,
    'mean_reversion_vwap': MeanReversionVWAP,
    'xrp_btc_pair_trading': XRPBTCPairTrading,
}


class StrategyFactory:
    """
    Factory for creating and managing trading strategies.

    Usage:
        factory = StrategyFactory(config_path='strategies_config/active_strategy.yaml')
        strategy = factory.get_active_strategy()
        signal = strategy.generate_signals(data)
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize factory with configuration.

        Args:
            config_path: Path to YAML config file. If None, uses default.
        """
        self.config_path = config_path or self._find_config()
        self.config = self._load_config()
        self.strategies: Dict[str, BaseStrategy] = {}

    def _find_config(self) -> str:
        """Find the strategy config file."""
        # Check multiple possible locations
        possible_paths = [
            'strategies_config/active_strategy.yaml',
            '../strategies_config/active_strategy.yaml',
            os.path.join(os.path.dirname(__file__), '../../strategies_config/active_strategy.yaml'),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        # Return default path (will create if doesn't exist)
        return 'strategies_config/active_strategy.yaml'

    def _load_config(self) -> Dict[str, Any]:
        """Load strategy configuration from YAML."""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f) or {}

        # Default config
        return {
            'active': 'defensive_yield',
            'strategies': {
                'defensive_yield': {
                    'enabled': True,
                    'max_leverage': 10,
                },
                'mean_reversion_vwap': {
                    'enabled': True,
                    'symbol': 'XRP/USDT',
                    'max_leverage': 5,
                },
                'xrp_btc_pair_trading': {
                    'enabled': True,
                    'max_leverage': 10,
                },
            }
        }

    def get_strategy(self, name: str) -> Optional[BaseStrategy]:
        """
        Get or create a strategy by name.

        Args:
            name: Strategy name (must be in STRATEGY_REGISTRY)

        Returns:
            BaseStrategy instance or None if not found
        """
        if name in self.strategies:
            return self.strategies[name]

        if name not in STRATEGY_REGISTRY:
            print(f"StrategyFactory: Unknown strategy '{name}'")
            return None

        # Get strategy-specific config
        strategy_config = self.config.get('strategies', {}).get(name, {})
        strategy_config['name'] = name

        # Create strategy instance
        strategy_class = STRATEGY_REGISTRY[name]
        strategy = strategy_class(strategy_config)

        self.strategies[name] = strategy
        print(f"StrategyFactory: Created strategy '{name}'")

        return strategy

    def get_active_strategy(self) -> Optional[BaseStrategy]:
        """
        Get the currently active strategy.

        Returns:
            Active BaseStrategy instance
        """
        active_name = self.config.get('active', 'defensive_yield')
        return self.get_strategy(active_name)

    def set_active_strategy(self, name: str) -> bool:
        """
        Set the active strategy.

        Args:
            name: Strategy name to activate

        Returns:
            True if successful
        """
        if name not in STRATEGY_REGISTRY:
            print(f"StrategyFactory: Cannot set unknown strategy '{name}'")
            return False

        self.config['active'] = name
        self._save_config()
        print(f"StrategyFactory: Active strategy set to '{name}'")
        return True

    def _save_config(self) -> None:
        """Save current config to file."""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

    def list_strategies(self) -> Dict[str, bool]:
        """
        List all available strategies and their enabled status.

        Returns:
            Dict of strategy_name -> is_enabled
        """
        result = {}
        for name in STRATEGY_REGISTRY:
            strategy_config = self.config.get('strategies', {}).get(name, {})
            result[name] = strategy_config.get('enabled', True)
        return result

    def initialize_all(self, portfolio: Any, exchanges: Dict[str, Any]) -> None:
        """
        Initialize all enabled strategies with portfolio and exchanges.

        Args:
            portfolio: Portfolio object
            exchanges: Dict of exchange name -> handler
        """
        for name, enabled in self.list_strategies().items():
            if enabled:
                strategy = self.get_strategy(name)
                if strategy:
                    strategy.initialize(portfolio, exchanges)


def load_strategy(name: str = None, config: Dict[str, Any] = None) -> BaseStrategy:
    """
    Convenience function to load a single strategy.

    Args:
        name: Strategy name (default: 'defensive_yield')
        config: Optional config dict (overrides file config)

    Returns:
        BaseStrategy instance
    """
    if name is None:
        name = 'defensive_yield'

    if config is None:
        config = {'name': name}

    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {name}")

    return STRATEGY_REGISTRY[name](config)
