"""
Phase 24: Portfolio Rebalancer Strategy
Maintains target asset allocations by generating buy/sell signals when
portfolio drifts from targets.

Features:
- Configurable target weights per asset
- Deviation threshold triggering
- Priority-based rebalancing
- Works with unified orchestrator's portfolio
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from strategies.base_strategy import BaseStrategy
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List


class PortfolioRebalancer(BaseStrategy):
    """
    Portfolio rebalancing strategy that generates signals to maintain
    target asset allocations.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Target weights (must sum to 1.0)
        self.target_weights = config.get('target_weights', {
            'BTC': 0.40,
            'XRP': 0.30,
            'USDT': 0.30
        })

        # Rebalance parameters
        self.deviation_threshold = config.get('deviation_threshold', 0.05)  # 5%
        self.min_trade_usd = config.get('min_trade_usd', 50.0)
        self.rebalance_cooldown = config.get('rebalance_cooldown', 60)  # minutes

        # Portfolio reference (will be set externally)
        self.portfolio_balances: Dict[str, float] = {}
        self.current_prices: Dict[str, float] = {}

        # State
        self.last_rebalance_time = 0
        self.pending_trades: List[Dict] = []

    def set_portfolio(self, balances: Dict[str, float], prices: Dict[str, float]):
        """
        Update portfolio state from external source.

        Args:
            balances: Current asset balances
            prices: Current asset prices in USD
        """
        self.portfolio_balances = balances.copy()
        self.current_prices = prices.copy()

    def _get_total_usd(self) -> float:
        """Calculate total portfolio value in USD."""
        total = 0.0
        for asset, balance in self.portfolio_balances.items():
            price = self.current_prices.get(asset, 0)
            if asset == 'USDT':
                price = 1.0
            total += balance * price
        return total

    def _get_current_weights(self) -> Dict[str, float]:
        """Calculate current portfolio weights."""
        total_usd = self._get_total_usd()
        if total_usd <= 0:
            return {}

        weights = {}
        for asset in self.target_weights.keys():
            balance = self.portfolio_balances.get(asset, 0)
            price = self.current_prices.get(asset, 0)
            if asset == 'USDT':
                price = 1.0
            weights[asset] = (balance * price) / total_usd

        return weights

    def _calculate_rebalance_trades(self) -> List[Dict[str, Any]]:
        """
        Calculate trades needed to rebalance to targets.

        Returns:
            List of trade dicts with asset, action, usd_amount
        """
        current_weights = self._get_current_weights()
        total_usd = self._get_total_usd()

        trades = []

        for asset, target in self.target_weights.items():
            current = current_weights.get(asset, 0)
            diff = target - current

            # Check if deviation exceeds threshold
            if abs(diff) > self.deviation_threshold:
                usd_amount = diff * total_usd

                if abs(usd_amount) >= self.min_trade_usd:
                    trades.append({
                        'asset': asset,
                        'action': 'buy' if usd_amount > 0 else 'sell',
                        'usd_amount': abs(usd_amount),
                        'current_weight': current,
                        'target_weight': target,
                        'deviation': diff
                    })

        # Sort by absolute deviation (largest first)
        trades.sort(key=lambda x: abs(x['deviation']), reverse=True)

        return trades

    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Generate rebalancing signals.

        Note: This strategy needs portfolio data set via set_portfolio()
        before generating meaningful signals.

        Args:
            data: Dict with symbol -> DataFrame (used for price extraction)

        Returns:
            Signal dict with rebalance action
        """
        # Extract current prices from data if not set
        if not self.current_prices:
            for key, df in data.items():
                if df is not None and len(df) > 0:
                    base = key.split('/')[0]
                    self.current_prices[base] = df['close'].iloc[-1]

        # If no portfolio data, can't rebalance
        if not self.portfolio_balances:
            return {
                'action': 'hold',
                'symbol': 'PORTFOLIO',
                'confidence': 0.0,
                'reason': 'No portfolio data set',
                'strategy': 'portfolio_rebalancer'
            }

        # Calculate needed trades
        trades = self._calculate_rebalance_trades()

        if not trades:
            current_weights = self._get_current_weights()
            max_deviation = max(
                abs(current_weights.get(a, 0) - t)
                for a, t in self.target_weights.items()
            ) if current_weights else 0.0

            return {
                'action': 'hold',
                'symbol': 'PORTFOLIO',
                'confidence': 0.0,
                'reason': f'Portfolio balanced, drift {max_deviation*100:.1f}% < threshold {self.deviation_threshold*100:.0f}%',
                'strategy': 'portfolio_rebalancer',
                'current_weights': current_weights,
                'target_weights': self.target_weights,
                'indicators': {
                    'portfolio_drift': max_deviation,
                    'drift_threshold': self.deviation_threshold,
                    'no_drift': max_deviation < self.deviation_threshold,
                    'total_usd': self._get_total_usd()
                }
            }

        # Take the highest priority trade (largest deviation)
        top_trade = trades[0]
        asset = top_trade['asset']
        action = top_trade['action']
        usd_amount = top_trade['usd_amount']

        # Map to symbol
        symbol = f"{asset}/USDT" if asset != 'USDT' else 'USDT'
        price = self.current_prices.get(asset, 1.0)

        # Calculate size in asset terms
        size = usd_amount / price if price > 0 else 0

        # Confidence based on deviation size
        confidence = min(0.5 + abs(top_trade['deviation']) * 5, 0.90)

        return {
            'action': action,
            'symbol': symbol,
            'size': size,
            'leverage': 1,  # No leverage for rebalancing
            'confidence': confidence,
            'reason': f"Rebalance {asset}: {top_trade['current_weight']*100:.1f}% -> {top_trade['target_weight']*100:.1f}% (${usd_amount:.2f})",
            'strategy': 'portfolio_rebalancer',
            'rebalance_details': {
                'asset': asset,
                'usd_amount': usd_amount,
                'current_weight': top_trade['current_weight'],
                'target_weight': top_trade['target_weight'],
                'deviation': top_trade['deviation'],
                'all_pending_trades': len(trades)
            },
            'current_weights': self._get_current_weights(),
            'target_weights': self.target_weights
        }

    def _format_weights(self, weights: Dict[str, float]) -> str:
        """Format weights dict as string."""
        return ', '.join([f"{k}:{v*100:.0f}%" for k, v in weights.items()])

    def update_model(self, data: Optional[Dict[str, pd.DataFrame]] = None) -> bool:
        """Rebalancer is rule-based, no model to update."""
        return True

    def get_status(self) -> Dict[str, Any]:
        """Get strategy status."""
        base_status = super().get_status()
        base_status.update({
            'target_weights': self.target_weights,
            'current_weights': self._get_current_weights(),
            'total_usd': self._get_total_usd(),
            'deviation_threshold': self.deviation_threshold,
            'min_trade_usd': self.min_trade_usd,
            'pending_trades': len(self._calculate_rebalance_trades())
        })
        return base_status
