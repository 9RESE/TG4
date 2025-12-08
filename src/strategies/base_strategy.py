"""
Base Strategy Abstract Class
Phase 15: Modular Strategy Factory - Scalable Architecture

Provides a common interface for all trading strategies.
Each strategy implements generate_signals() and update_model().
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
import pandas as pd


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.

    All strategies must implement:
    - generate_signals(): Generate trading signals from market data
    - update_model(): Train/update any ML components (can be no-op)

    Strategies can optionally override:
    - initialize(): Setup before trading starts
    - on_order_filled(): Callback when order is executed
    - get_status(): Return strategy status/metrics
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize strategy with configuration.

        Args:
            config: Strategy-specific configuration dict
        """
        self.config = config
        self.name = config.get('name', self.__class__.__name__)
        self.enabled = config.get('enabled', True)
        self.max_leverage = config.get('max_leverage', 5)
        self.position_size_pct = config.get('position_size_pct', 0.10)  # 10% default
        self.active_positions: Dict[str, Any] = {}

    @abstractmethod
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Generate trading signals from market data.

        Args:
            data: Dict of symbol -> DataFrame with OHLCV data
                  e.g. {'XRP/USDT': df, 'BTC/USDT': df}

        Returns:
            dict: Signal details including:
                - action: 'buy', 'sell', 'short', 'cover', 'hold'
                - size: Position size as fraction of available capital
                - leverage: Leverage to use (1 for spot)
                - symbol: Target trading pair
                - confidence: 0.0 to 1.0 signal strength
                - reason: Human-readable explanation
        """
        pass

    @abstractmethod
    def update_model(self, data: Optional[Dict[str, pd.DataFrame]] = None) -> bool:
        """
        Train/update any ML components.

        Args:
            data: Optional training data. If None, use stored data.

        Returns:
            bool: True if update successful
        """
        pass

    def initialize(self, portfolio: Any, exchanges: Dict[str, Any]) -> None:
        """
        Initialize strategy with portfolio and exchange references.
        Override this for strategy-specific setup.

        Args:
            portfolio: Portfolio object for balance checking
            exchanges: Dict of exchange name -> exchange handler
        """
        self.portfolio = portfolio
        self.exchanges = exchanges

    def on_order_filled(self, order: Dict[str, Any]) -> None:
        """
        Callback when an order is filled.
        Override for strategy-specific position tracking.

        Args:
            order: Filled order details
        """
        pass

    def get_status(self) -> Dict[str, Any]:
        """
        Get current strategy status and metrics.

        Returns:
            dict: Status information
        """
        return {
            'name': self.name,
            'enabled': self.enabled,
            'active_positions': len(self.active_positions),
            'config': {k: v for k, v in self.config.items()
                      if k not in ['api_key', 'secret']}  # Exclude secrets
        }

    def validate_signal(self, signal: Dict[str, Any]) -> bool:
        """
        Validate a signal before execution.

        Args:
            signal: Signal dict from generate_signals()

        Returns:
            bool: True if signal is valid and can be executed
        """
        required_keys = ['action', 'symbol']
        for key in required_keys:
            if key not in signal:
                return False

        valid_actions = ['buy', 'sell', 'short', 'cover', 'hold']
        if signal.get('action') not in valid_actions:
            return False

        return True
