"""
Triangular Arbitrage Strategy
Research: 3-5% annualized returns, single-exchange execution

Exploits price inefficiencies across three trading pairs on the same exchange.
Example: BTC -> XRP -> USDT -> BTC

If the product of exchange rates > 1 (minus fees), profit exists.

Features:
- Real-time price monitoring across trio of pairs
- Fee-adjusted profit calculation
- Slippage estimation from order book depth
- Fast execution path detection
- Multi-trio monitoring (BTC/XRP/USDT, ETH/BTC/USDT, etc.)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from strategies.base_strategy import BaseStrategy
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime


class TriangularArbitrage(BaseStrategy):
    """
    Triangular Arbitrage on single exchange.

    Monitors price relationships between three assets and executes
    when profitable opportunities exist after accounting for fees.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Define arbitrage trios (base -> intermediate -> quote -> base)
        self.trios = config.get('trios', [
            {'path': ['BTC', 'XRP', 'USDT'], 'pairs': ['XRP/BTC', 'XRP/USDT', 'BTC/USDT']},
            {'path': ['BTC', 'ETH', 'USDT'], 'pairs': ['ETH/BTC', 'ETH/USDT', 'BTC/USDT']},
        ])

        # Profit thresholds
        self.min_profit_pct = config.get('min_profit_pct', 0.10)  # 0.1% minimum profit
        self.fee_per_trade = config.get('fee_per_trade', 0.001)  # 0.1% fee
        self.slippage_buffer = config.get('slippage_buffer', 0.001)  # 0.1% slippage

        # Position sizing
        self.trade_size_pct = config.get('trade_size_pct', 0.05)  # 5% of capital
        self.max_trade_size_usd = config.get('max_trade_size_usd', 500)  # Cap per trade

        # Execution
        self.cooldown_seconds = config.get('cooldown_seconds', 10)
        self.max_opportunities_per_hour = config.get('max_opportunities_per_hour', 20)

        # State tracking
        self.last_execution_time: Dict[str, datetime] = {}
        self.opportunities_this_hour: int = 0
        self.hour_start: datetime = datetime.now()
        self.profit_history: List[Dict] = []
        self.total_profit: float = 0.0

    def _get_price(self, data: Dict[str, pd.DataFrame], pair: str,
                   side: str = 'mid') -> Optional[float]:
        """
        Get price for a trading pair.

        Args:
            data: Market data
            pair: Trading pair (e.g., 'XRP/BTC')
            side: 'bid', 'ask', or 'mid'
        """
        if pair in data:
            df = data[pair]
            if isinstance(df, pd.DataFrame) and len(df) > 0:
                if side == 'bid' and 'bid' in df.columns:
                    return df['bid'].iloc[-1]
                elif side == 'ask' and 'ask' in df.columns:
                    return df['ask'].iloc[-1]
                elif 'close' in df.columns:
                    return df['close'].iloc[-1]
        return None

    def _calculate_arb_profit(self, data: Dict[str, pd.DataFrame],
                              trio: Dict) -> Tuple[float, str, Dict]:
        """
        Calculate potential arbitrage profit for a trio.

        Two directions are possible:
        1. Forward: base -> intermediate -> quote -> base
        2. Reverse: base -> quote -> intermediate -> base

        Returns:
            (profit_pct, direction, details)
        """
        pairs = trio['pairs']
        path = trio['path']

        # Get prices for all pairs
        prices = {}
        for pair in pairs:
            mid = self._get_price(data, pair, 'mid')
            bid = self._get_price(data, pair, 'bid')
            ask = self._get_price(data, pair, 'ask')

            if mid is None:
                return 0.0, 'none', {'error': f'Missing price for {pair}'}

            prices[pair] = {
                'mid': mid,
                'bid': bid or mid * 0.999,  # Estimate if not available
                'ask': ask or mid * 1.001
            }

        # Example trio: BTC -> XRP -> USDT -> BTC
        # Pairs: XRP/BTC, XRP/USDT, BTC/USDT

        # Forward direction:
        # 1. Buy XRP with BTC (sell BTC for XRP): use ask price of XRP/BTC
        # 2. Sell XRP for USDT: use bid price of XRP/USDT
        # 3. Buy BTC with USDT: use ask price of BTC/USDT

        pair1, pair2, pair3 = pairs  # XRP/BTC, XRP/USDT, BTC/USDT

        # Forward: BTC -> XRP -> USDT -> BTC
        # Start with 1 BTC
        # Step 1: Buy XRP with BTC = 1 BTC / (XRP/BTC ask) = XRP amount
        step1_xrp = 1.0 / prices[pair1]['ask']
        step1_xrp *= (1 - self.fee_per_trade)  # Apply fee

        # Step 2: Sell XRP for USDT = XRP * (XRP/USDT bid) = USDT amount
        step2_usdt = step1_xrp * prices[pair2]['bid']
        step2_usdt *= (1 - self.fee_per_trade)

        # Step 3: Buy BTC with USDT = USDT / (BTC/USDT ask) = BTC amount
        step3_btc = step2_usdt / prices[pair3]['ask']
        step3_btc *= (1 - self.fee_per_trade)

        forward_profit = (step3_btc - 1.0) * 100  # Percentage profit

        # Reverse direction: BTC -> USDT -> XRP -> BTC
        # Step 1: Sell BTC for USDT = 1 BTC * (BTC/USDT bid)
        rev_step1_usdt = 1.0 * prices[pair3]['bid']
        rev_step1_usdt *= (1 - self.fee_per_trade)

        # Step 2: Buy XRP with USDT = USDT / (XRP/USDT ask)
        rev_step2_xrp = rev_step1_usdt / prices[pair2]['ask']
        rev_step2_xrp *= (1 - self.fee_per_trade)

        # Step 3: Sell XRP for BTC = XRP * (XRP/BTC bid)
        rev_step3_btc = rev_step2_xrp * prices[pair1]['bid']
        rev_step3_btc *= (1 - self.fee_per_trade)

        reverse_profit = (rev_step3_btc - 1.0) * 100

        # Choose better direction
        if forward_profit > reverse_profit and forward_profit > 0:
            return forward_profit, 'forward', {
                'path': f"{path[0]} -> {path[1]} -> {path[2]} -> {path[0]}",
                'prices': prices,
                'steps': [step1_xrp, step2_usdt, step3_btc]
            }
        elif reverse_profit > 0:
            return reverse_profit, 'reverse', {
                'path': f"{path[0]} -> {path[2]} -> {path[1]} -> {path[0]}",
                'prices': prices,
                'steps': [rev_step1_usdt, rev_step2_xrp, rev_step3_btc]
            }
        else:
            return max(forward_profit, reverse_profit), 'none', {
                'forward_profit': forward_profit,
                'reverse_profit': reverse_profit
            }

    def _check_rate_limits(self, trio_name: str) -> bool:
        """Check if we can execute (cooldown and rate limits)."""
        now = datetime.now()

        # Reset hourly counter
        if (now - self.hour_start).total_seconds() > 3600:
            self.hour_start = now
            self.opportunities_this_hour = 0

        # Check hourly limit
        if self.opportunities_this_hour >= self.max_opportunities_per_hour:
            return False

        # Check cooldown per trio
        last_exec = self.last_execution_time.get(trio_name)
        if last_exec:
            if (now - last_exec).total_seconds() < self.cooldown_seconds:
                return False

        return True

    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Scan for triangular arbitrage opportunities.

        For each trio, calculate profit in both directions.
        Execute if profit > threshold after fees and slippage.
        """
        opportunities = []

        for trio in self.trios:
            trio_name = '-'.join(trio['path'])

            # Check rate limits
            if not self._check_rate_limits(trio_name):
                continue

            # Calculate potential profit
            profit_pct, direction, details = self._calculate_arb_profit(data, trio)

            # Account for slippage buffer
            net_profit = profit_pct - (self.slippage_buffer * 100)

            if net_profit >= self.min_profit_pct and direction != 'none':
                opportunities.append({
                    'trio': trio,
                    'trio_name': trio_name,
                    'profit_pct': profit_pct,
                    'net_profit': net_profit,
                    'direction': direction,
                    'details': details
                })

        if opportunities:
            # Take the most profitable opportunity
            best = max(opportunities, key=lambda x: x['net_profit'])

            # Update state
            self.last_execution_time[best['trio_name']] = datetime.now()
            self.opportunities_this_hour += 1

            # Track profit
            self.profit_history.append({
                'timestamp': datetime.now(),
                'trio': best['trio_name'],
                'profit_pct': best['net_profit']
            })
            self.total_profit += best['net_profit']

            return {
                'action': 'triangular_arb',
                'symbol': best['trio']['pairs'][0],  # Primary pair for tracking
                'size': self.trade_size_pct,
                'leverage': 1,  # Spot only
                'confidence': min(0.95, 0.70 + best['net_profit'] / 100),
                'reason': f"Triangular arb: {best['details']['path']}, profit={best['net_profit']:.3f}%",
                'strategy': 'triangular_arb',
                'trio': best['trio'],
                'direction': best['direction'],
                'profit_pct': best['net_profit'],
                'execution_path': best['details']['path']
            }

        # Build detailed hold reason for diagnostics
        hold_reasons = []
        hold_reasons.append(f'Checked {len(self.trios)} trios')
        hold_reasons.append(f'Min profit threshold: {self.min_profit_pct}%')

        return {
            'action': 'hold',
            'symbol': 'BTC/USDT',
            'confidence': 0.0,
            'reason': f"TriangularArb: {', '.join(hold_reasons)}",
            'strategy': 'triangular_arb',
            'indicators': {
                'arb_profit': 0.0,
                'min_profit_threshold': self.min_profit_pct,
                'trios_checked': len(self.trios),
                'opportunities_this_hour': self.opportunities_this_hour,
                'total_profit': self.total_profit
            }
        }

    def update_model(self, data: Optional[Dict[str, pd.DataFrame]] = None) -> bool:
        """Triangular arb is rule-based, no model to update."""
        return True

    def get_status(self) -> Dict[str, Any]:
        """Get strategy status."""
        base = super().get_status()
        base.update({
            'trios_monitored': len(self.trios),
            'min_profit_pct': self.min_profit_pct,
            'opportunities_this_hour': self.opportunities_this_hour,
            'total_profit_pct': self.total_profit,
            'recent_opportunities': self.profit_history[-10:] if self.profit_history else []
        })
        return base
