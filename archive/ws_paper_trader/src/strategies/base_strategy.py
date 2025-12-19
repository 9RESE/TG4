"""
Base Strategy Abstract Class
Phase 15: Modular Strategy Factory - Scalable Architecture
Phase 31: Per-strategy risk profiles with unified framework

Provides a common interface for all trading strategies.
Each strategy implements generate_signals() and update_model().
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, Tuple
import pandas as pd
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from risk_manager import StrategyRiskProfile, get_risk_profile


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

    Phase 31: Each strategy now has a risk_profile with:
    - min_confidence: Minimum confidence to execute trades
    - stop_loss_pct, take_profit_pct: Exit targets
    - max_hold_hours: Maximum position hold time
    - Trailing stop configuration
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

        # Phase 31: Initialize risk profile
        self.risk_profile = get_risk_profile(self.name, config)

        # Convenience accessors for commonly used risk params
        self.min_confidence = self.risk_profile.min_confidence
        self.stop_loss_pct = self.risk_profile.stop_loss_pct
        self.take_profit_pct = self.risk_profile.take_profit_pct
        self.max_hold_hours = self.risk_profile.max_hold_hours

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
            'risk_profile': {
                'min_confidence': self.min_confidence,
                'stop_loss_pct': self.stop_loss_pct,
                'take_profit_pct': self.take_profit_pct,
                'max_hold_hours': self.max_hold_hours,
                'category': self.risk_profile.category.value,
            },
            'config': {k: v for k, v in self.config.items()
                      if k not in ['api_key', 'secret']}  # Exclude secrets
        }

    def check_position_exit(self, entry_price: float, current_price: float,
                           entry_time_hours: float, peak_price: float = None,
                           side: str = 'long') -> Tuple[bool, str]:
        """
        Check if an open position should be exited based on risk rules.

        Args:
            entry_price: Entry price of position
            current_price: Current market price
            entry_time_hours: Hours since entry
            peak_price: Peak price since entry (for trailing stop)
            side: 'long' or 'short'

        Returns:
            Tuple[bool, str]: (should_exit, reason)
        """
        return self.risk_profile.check_exit(
            entry_price, current_price, entry_time_hours, peak_price, side
        )

    def meets_confidence_threshold(self, confidence: float) -> bool:
        """
        Check if a signal's confidence meets the strategy's minimum threshold.

        Args:
            confidence: Signal confidence (0.0 to 1.0)

        Returns:
            bool: True if confidence meets or exceeds threshold
        """
        return confidence >= self.min_confidence

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

        # Phase 25: Extended valid actions for all strategy types
        # - buy: Open long position
        # - sell: Close long position
        # - short: Open short position
        # - cover: Close short position (alias for buy to cover)
        # - hold: No action
        # - close: Generic close (used by some strategies)
        valid_actions = ['buy', 'sell', 'short', 'cover', 'hold', 'close']
        if signal.get('action') not in valid_actions:
            return False

        return True
