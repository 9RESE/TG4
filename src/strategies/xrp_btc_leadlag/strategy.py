"""
Phase 20: XRP/BTC Lead-Lag Strategy
Correlation-aware trading: follows BTC when highly correlated.

Rules:
- High correlation (>0.8): XRP follows BTC with lag → trade XRP in BTC direction
- Low correlation (<0.6): Pair divergence → mean reversion or skip
- BTC leads: if BTC moves first, position XRP same direction
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from strategies.base_strategy import BaseStrategy
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional


class XRPBTCLeadLag(BaseStrategy):
    """
    Correlation-aware XRP/BTC lead-lag strategy.
    When BTC and XRP are highly correlated, XRP tends to follow BTC moves.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.corr_high = config.get('corr_high', 0.8)
        self.corr_low = config.get('corr_low', 0.6)
        self.lookback = config.get('lookback', 50)
        self.btc_lead_bars = config.get('btc_lead_bars', 3)  # BTC leads by N bars
        self.min_btc_move = config.get('min_btc_move', 0.01)  # 1% BTC move threshold

        # Track state
        self.last_correlation = 0.0
        self.btc_trend = 'none'
        self.position_taken = False

    def _calculate_returns(self, df: pd.DataFrame) -> pd.Series:
        """Calculate log returns."""
        return np.log(df['close'] / df['close'].shift(1))

    def _get_correlation(self, btc_df: pd.DataFrame, xrp_df: pd.DataFrame) -> float:
        """
        Calculate rolling correlation between BTC and XRP returns.
        """
        if len(btc_df) < self.lookback or len(xrp_df) < self.lookback:
            return 0.5  # Neutral if insufficient data

        btc_returns = self._calculate_returns(btc_df).tail(self.lookback)
        xrp_returns = self._calculate_returns(xrp_df).tail(self.lookback)

        # Align by index if possible, otherwise by position
        if len(btc_returns) != len(xrp_returns):
            min_len = min(len(btc_returns), len(xrp_returns))
            btc_returns = btc_returns.tail(min_len)
            xrp_returns = xrp_returns.tail(min_len)

        corr = btc_returns.corr(xrp_returns)
        return corr if not np.isnan(corr) else 0.5

    def _detect_btc_trend(self, btc_df: pd.DataFrame) -> str:
        """
        Detect recent BTC trend based on last N bars.
        Returns 'up', 'down', or 'none'.
        """
        if len(btc_df) < self.btc_lead_bars + 1:
            return 'none'

        recent = btc_df.tail(self.btc_lead_bars + 1)
        start_price = recent['close'].iloc[0]
        end_price = recent['close'].iloc[-1]
        pct_change = (end_price - start_price) / start_price

        if pct_change > self.min_btc_move:
            return 'up'
        elif pct_change < -self.min_btc_move:
            return 'down'
        return 'none'

    def _check_divergence(self, btc_df: pd.DataFrame, xrp_df: pd.DataFrame) -> Optional[str]:
        """
        Check for BTC/XRP divergence when correlation is low.
        Returns signal direction for mean reversion or None.
        """
        if len(btc_df) < 10 or len(xrp_df) < 10:
            return None

        btc_change = (btc_df['close'].iloc[-1] - btc_df['close'].iloc[-10]) / btc_df['close'].iloc[-10]
        xrp_change = (xrp_df['close'].iloc[-1] - xrp_df['close'].iloc[-10]) / xrp_df['close'].iloc[-10]

        # Significant divergence: one up, one down
        if btc_change > 0.02 and xrp_change < -0.02:
            return 'long_xrp'  # XRP lagging, should catch up
        if btc_change < -0.02 and xrp_change > 0.02:
            return 'short_xrp'  # XRP leading, should fall

        return None

    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Generate trading signals based on BTC/XRP correlation and lead-lag.

        Args:
            data: Dict with symbol -> DataFrame (OHLCV)

        Returns:
            Signal dict with action, size, leverage, etc.
        """
        # Need both BTC and XRP data
        btc_key = None
        xrp_key = None
        for key in data.keys():
            if 'BTC' in key.upper():
                btc_key = key
            if 'XRP' in key.upper():
                xrp_key = key

        if not btc_key or not xrp_key:
            return {
                'action': 'hold',
                'symbol': 'XRP/USDT',
                'confidence': 0.0,
                'reason': 'Missing BTC or XRP data',
                'strategy': 'leadlag'
            }

        btc_df = data[btc_key]
        xrp_df = data[xrp_key]

        if len(btc_df) < self.lookback + 5 or len(xrp_df) < self.lookback + 5:
            return {
                'action': 'hold',
                'symbol': xrp_key,
                'confidence': 0.0,
                'reason': 'Insufficient data for correlation',
                'strategy': 'leadlag'
            }

        # Calculate correlation
        corr = self._get_correlation(btc_df, xrp_df)
        self.last_correlation = corr

        # Detect BTC trend
        btc_trend = self._detect_btc_trend(btc_df)
        self.btc_trend = btc_trend

        # High correlation mode: follow BTC
        if corr > self.corr_high and btc_trend != 'none':
            if btc_trend == 'up':
                return {
                    'action': 'buy',
                    'symbol': xrp_key,
                    'size': self.position_size_pct,
                    'leverage': min(self.max_leverage, 5),
                    'confidence': 0.7 + (corr - 0.8) * 0.5,  # Higher corr = higher confidence
                    'reason': f'Lead-lag: BTC up +{self.min_btc_move*100:.0f}%, corr={corr:.2f}, XRP follow',
                    'strategy': 'leadlag'
                }
            else:  # btc_trend == 'down'
                return {
                    'action': 'short',
                    'symbol': xrp_key,
                    'size': self.position_size_pct * 0.8,
                    'leverage': min(self.max_leverage, 5),
                    'confidence': 0.65 + (corr - 0.8) * 0.5,
                    'reason': f'Lead-lag: BTC down -{self.min_btc_move*100:.0f}%, corr={corr:.2f}, XRP follow',
                    'strategy': 'leadlag'
                }

        # Low correlation mode: check for divergence mean reversion
        if corr < self.corr_low:
            divergence = self._check_divergence(btc_df, xrp_df)
            if divergence == 'long_xrp':
                return {
                    'action': 'buy',
                    'symbol': xrp_key,
                    'size': self.position_size_pct * 0.6,  # Smaller size for divergence plays
                    'leverage': min(self.max_leverage, 3),
                    'confidence': 0.55,
                    'reason': f'Divergence: XRP lagging BTC, corr={corr:.2f}, expect catch-up',
                    'strategy': 'leadlag'
                }
            elif divergence == 'short_xrp':
                return {
                    'action': 'short',
                    'symbol': xrp_key,
                    'size': self.position_size_pct * 0.5,
                    'leverage': min(self.max_leverage, 3),
                    'confidence': 0.50,
                    'reason': f'Divergence: XRP leading BTC, corr={corr:.2f}, expect pullback',
                    'strategy': 'leadlag'
                }

        # No clear signal
        return {
            'action': 'hold',
            'symbol': xrp_key,
            'confidence': 0.0,
            'reason': f'No lead-lag signal (corr={corr:.2f}, btc_trend={btc_trend})',
            'strategy': 'leadlag'
        }

    def update_model(self, data: Optional[Dict[str, pd.DataFrame]] = None) -> bool:
        """Lead-lag is rule-based, no model to update."""
        return True

    def get_status(self) -> Dict[str, Any]:
        """Get strategy status."""
        base_status = super().get_status()
        base_status.update({
            'correlation': self.last_correlation,
            'btc_trend': self.btc_trend,
            'corr_high_threshold': self.corr_high,
            'corr_low_threshold': self.corr_low,
            'lookback': self.lookback
        })
        return base_status
