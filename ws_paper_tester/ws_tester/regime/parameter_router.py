"""
Parameter Router

Routes regime-based parameter adjustments to strategies.

The router takes the current regime and returns appropriate parameter
multipliers that strategies can apply to their base configurations.

Example:
    base_config = {'position_size_usd': 20, 'stop_loss_pct': 1.0}
    adjustments = router.get_adjustments('mean_reversion', regime)
    adjusted = router.apply_to_config(base_config, adjustments)
    # adjusted = {'position_size_usd': 10, 'stop_loss_pct': 1.5}
"""
from typing import Dict, Optional

from .types import (
    MarketRegime,
    VolatilityState,
    RegimeSnapshot,
    RegimeAdjustments,
    DEFAULT_REGIME_ADJUSTMENTS,
)


class ParameterRouter:
    """
    Route regime-based parameter adjustments to strategies.

    Combines regime-specific adjustments with volatility modifiers
    to produce final parameter multipliers.
    """

    # Volatility-based position size modifiers
    VOLATILITY_MODIFIERS = {
        VolatilityState.LOW: 1.2,      # Can size up in low volatility
        VolatilityState.MEDIUM: 1.0,   # Baseline
        VolatilityState.HIGH: 0.7,     # Reduce size
        VolatilityState.EXTREME: 0.3,  # Significant reduction
    }

    def __init__(self, adjustments_config: Optional[Dict] = None):
        """
        Initialize the parameter router.

        Args:
            adjustments_config: Optional custom adjustments mapping.
                Format: {strategy_name: {MarketRegime: RegimeAdjustments}}
        """
        self.adjustments = adjustments_config or DEFAULT_REGIME_ADJUSTMENTS

    def get_adjustments(
        self,
        strategy_name: str,
        regime: RegimeSnapshot
    ) -> RegimeAdjustments:
        """
        Get parameter adjustments for a strategy given current regime.

        Combines regime-specific adjustments with volatility modifiers.

        Args:
            strategy_name: Name of the strategy (e.g., 'mean_reversion')
            regime: Current RegimeSnapshot

        Returns:
            RegimeAdjustments with multipliers to apply
        """
        # Get strategy-specific adjustments
        strategy_adjustments = self.adjustments.get(strategy_name, {})
        current_regime = regime.overall_regime

        # Get regime-specific adjustments or defaults
        base = strategy_adjustments.get(
            current_regime,
            RegimeAdjustments()  # Default: no adjustments
        )

        # Apply volatility modifiers
        vol_multiplier = self._get_volatility_modifier(regime.volatility_state)

        # Adjust confidence-based scaling
        # Lower confidence = more conservative adjustments
        confidence_factor = regime.overall_confidence
        if confidence_factor < 0.5:
            # Scale down position sizes when uncertain
            vol_multiplier *= (0.5 + confidence_factor)  # 0.5-1.0 range

        # Check for regime instability
        if regime.recent_transitions > 5:
            # Many recent transitions = unstable, be more conservative
            vol_multiplier *= 0.7

        return RegimeAdjustments(
            position_size_multiplier=base.position_size_multiplier * vol_multiplier,
            stop_loss_multiplier=base.stop_loss_multiplier,
            take_profit_multiplier=base.take_profit_multiplier,
            entry_threshold_shift=base.entry_threshold_shift,
            strategy_enabled=base.strategy_enabled and regime.volatility_state != VolatilityState.EXTREME,
            cooldown_multiplier=base.cooldown_multiplier,
            max_position_multiplier=base.max_position_multiplier * vol_multiplier,
        )

    def apply_to_config(
        self,
        base_config: dict,
        adjustments: RegimeAdjustments
    ) -> dict:
        """
        Apply adjustments to a strategy configuration dictionary.

        Common config keys that are adjusted:
        - position_size_usd: Multiplied by position_size_multiplier
        - stop_loss_pct: Multiplied by stop_loss_multiplier
        - take_profit_pct: Multiplied by take_profit_multiplier
        - max_position: Multiplied by max_position_multiplier
        - cooldown_seconds: Multiplied by cooldown_multiplier
        - entry_threshold: Shifted by entry_threshold_shift

        Args:
            base_config: Strategy's base configuration
            adjustments: RegimeAdjustments to apply

        Returns:
            Adjusted configuration dictionary
        """
        adjusted = base_config.copy()

        # Position sizing
        if 'position_size_usd' in adjusted:
            adjusted['position_size_usd'] *= adjustments.position_size_multiplier

        if 'position_size' in adjusted:
            adjusted['position_size'] *= adjustments.position_size_multiplier

        if 'size' in adjusted:
            adjusted['size'] *= adjustments.position_size_multiplier

        # Stop loss
        if 'stop_loss_pct' in adjusted:
            adjusted['stop_loss_pct'] *= adjustments.stop_loss_multiplier

        if 'stop_loss' in adjusted:
            adjusted['stop_loss'] *= adjustments.stop_loss_multiplier

        # Take profit
        if 'take_profit_pct' in adjusted:
            adjusted['take_profit_pct'] *= adjustments.take_profit_multiplier

        if 'take_profit' in adjusted:
            adjusted['take_profit'] *= adjustments.take_profit_multiplier

        # Max position
        if 'max_position' in adjusted:
            adjusted['max_position'] *= adjustments.max_position_multiplier

        if 'max_exposure' in adjusted:
            adjusted['max_exposure'] *= adjustments.max_position_multiplier

        # Cooldown
        if 'cooldown_seconds' in adjusted:
            adjusted['cooldown_seconds'] *= adjustments.cooldown_multiplier

        if 'cooldown' in adjusted:
            adjusted['cooldown'] *= adjustments.cooldown_multiplier

        # Entry threshold shift
        if 'entry_threshold' in adjusted and adjustments.entry_threshold_shift != 0:
            adjusted['entry_threshold'] += adjustments.entry_threshold_shift

        if 'deviation_threshold' in adjusted and adjustments.entry_threshold_shift != 0:
            adjusted['deviation_threshold'] += adjustments.entry_threshold_shift

        # Add metadata
        adjusted['_regime_adjusted'] = True
        adjusted['_strategy_enabled'] = adjustments.strategy_enabled

        return adjusted

    def should_trade(
        self,
        strategy_name: str,
        regime: RegimeSnapshot
    ) -> bool:
        """
        Check if a strategy should trade in the current regime.

        Args:
            strategy_name: Strategy name
            regime: Current regime snapshot

        Returns:
            True if strategy should be allowed to trade
        """
        adjustments = self.get_adjustments(strategy_name, regime)
        return adjustments.strategy_enabled

    def _get_volatility_modifier(self, volatility_state: VolatilityState) -> float:
        """
        Get position size modifier based on volatility.

        Args:
            volatility_state: Current volatility classification

        Returns:
            Position size multiplier
        """
        return self.VOLATILITY_MODIFIERS.get(volatility_state, 1.0)

    def register_strategy_adjustments(
        self,
        strategy_name: str,
        adjustments: Dict[MarketRegime, RegimeAdjustments]
    ) -> None:
        """
        Register custom adjustments for a strategy.

        Args:
            strategy_name: Strategy name
            adjustments: Dict mapping MarketRegime to RegimeAdjustments
        """
        self.adjustments[strategy_name] = adjustments

    def get_regime_summary(self, regime: RegimeSnapshot) -> str:
        """
        Get a human-readable summary of the current regime.

        Args:
            regime: Current regime snapshot

        Returns:
            Summary string
        """
        return (
            f"Regime: {regime.overall_regime.name} "
            f"(confidence: {regime.overall_confidence:.0%}) | "
            f"Volatility: {regime.volatility_state.name} | "
            f"Trending: {'Yes' if regime.is_trending else 'No'} "
            f"({regime.trend_direction})"
        )
