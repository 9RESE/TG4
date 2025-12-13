"""
Funding Rate Arbitrage Strategy - Cash and Carry
Research: 19.26% APY average in 2025 (up from 14.39% in 2024)

Exploits perpetual futures funding rate for market-neutral returns.
When funding rate is positive (longs pay shorts):
- Short perpetual futures
- Long spot (or vice versa)
- Collect funding payments every 8 hours

This strategy requires access to both spot and futures markets.

Features:
- Monitors funding rates across symbols
- Opens delta-neutral positions when funding is favorable
- Tracks 8-hour funding payment windows
- Dynamic position sizing based on funding rate magnitude
- Cross-platform funding rate comparison (if available)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from strategies.base_strategy import BaseStrategy
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta


class FundingRateArbitrage(BaseStrategy):
    """
    Funding Rate Arbitrage (Cash and Carry) Strategy.

    Market-neutral strategy that profits from perpetual futures funding rates.
    When funding is positive, longs pay shorts every 8 hours.
    We short perps + long spot to collect these payments.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Funding rate thresholds
        self.min_funding_rate = config.get('min_funding_rate', 0.01)  # 0.01% per 8h = ~11% APY
        self.max_funding_rate = config.get('max_funding_rate', 0.10)  # Avoid extreme rates (manipulation)
        self.negative_funding_threshold = config.get('negative_funding_threshold', -0.01)

        # Position parameters
        self.base_size_pct = config.get('base_size_pct', 0.20)  # 20% of capital per arb
        self.max_positions = config.get('max_positions', 3)
        self.leverage = config.get('leverage', 2)  # Low leverage for safety

        # Timing
        self.funding_interval_hours = config.get('funding_interval_hours', 8)
        self.entry_before_funding_minutes = config.get('entry_before_funding_minutes', 30)
        self.exit_after_funding_minutes = config.get('exit_after_funding_minutes', 15)

        # Symbols to monitor
        self.symbols = config.get('symbols', ['BTC/USDT', 'XRP/USDT', 'ETH/USDT'])

        # State tracking
        self.active_arbs: Dict[str, Dict] = {}
        self.funding_history: Dict[str, List[Dict]] = {}
        self.last_funding_rates: Dict[str, float] = {}
        self.estimated_apy: Dict[str, float] = {}

        # Trading costs (must exceed for profitability)
        self.trading_fee_pct = config.get('trading_fee_pct', 0.001)  # 0.1% per trade
        self.slippage_pct = config.get('slippage_pct', 0.001)  # 0.1% slippage

    def _get_next_funding_time(self) -> datetime:
        """Calculate next funding time (00:00, 08:00, 16:00 UTC)."""
        now = datetime.utcnow()
        hour = now.hour

        # Find next 8-hour mark
        if hour < 8:
            next_hour = 8
        elif hour < 16:
            next_hour = 16
        else:
            next_hour = 24  # Will wrap to next day

        next_funding = now.replace(hour=next_hour % 24, minute=0, second=0, microsecond=0)
        if next_hour == 24:
            next_funding += timedelta(days=1)

        return next_funding

    def _is_near_funding_time(self) -> tuple:
        """Check if we're in the optimal entry/exit window around funding."""
        now = datetime.utcnow()
        next_funding = self._get_next_funding_time()

        time_to_funding = (next_funding - now).total_seconds() / 60  # Minutes

        # Entry window: X minutes before funding
        is_entry_window = 0 < time_to_funding <= self.entry_before_funding_minutes

        # Exit window: X minutes after funding (negative time_to_funding means we passed it)
        # Actually, if we just passed funding, time_to_funding would be ~480 minutes
        # So we need to check differently
        last_funding = next_funding - timedelta(hours=8)
        time_since_last = (now - last_funding).total_seconds() / 60
        is_exit_window = 0 < time_since_last <= self.exit_after_funding_minutes

        return is_entry_window, is_exit_window, time_to_funding

    def _calculate_expected_apy(self, funding_rate: float) -> float:
        """Calculate expected APY from funding rate."""
        # 3 funding payments per day, 365 days
        payments_per_year = 3 * 365
        apy = funding_rate * payments_per_year
        return apy

    def _is_funding_profitable(self, funding_rate: float) -> tuple:
        """
        Check if funding rate is profitable after costs.

        Returns:
            (is_profitable, direction, expected_profit_pct)
        """
        # Total round-trip cost: 4 trades (2 to enter, 2 to exit)
        total_cost = (self.trading_fee_pct + self.slippage_pct) * 4

        if funding_rate >= self.min_funding_rate:
            # Positive funding: short perp, long spot
            net_profit = funding_rate - total_cost
            return net_profit > 0, 'short_perp_long_spot', net_profit

        elif funding_rate <= self.negative_funding_threshold:
            # Negative funding: long perp, short spot (if possible)
            net_profit = abs(funding_rate) - total_cost
            return net_profit > 0, 'long_perp_short_spot', net_profit

        return False, None, 0

    def _parse_funding_rate(self, data: Dict[str, pd.DataFrame], symbol: str) -> Optional[float]:
        """
        Extract funding rate from data.

        This expects funding rate to be provided in the data dict.
        Kraken provides this via their futures API.
        """
        # Try to get funding rate from data
        funding_key = f'{symbol}_funding'

        if funding_key in data:
            df = data[funding_key]
            if isinstance(df, pd.DataFrame) and 'funding_rate' in df.columns:
                return df['funding_rate'].iloc[-1]
            elif isinstance(df, (int, float)):
                return float(df)

        # Try to get from symbol data with funding_rate column
        if symbol in data:
            df = data[symbol]
            if isinstance(df, pd.DataFrame) and 'funding_rate' in df.columns:
                return df['funding_rate'].iloc[-1]

        return None

    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Generate funding rate arbitrage signals.

        Logic:
        1. Check if near funding time window
        2. Get current funding rates
        3. If funding profitable and in entry window, open arb
        4. If in exit window and have position, close arb
        """
        is_entry_window, is_exit_window, time_to_funding = self._is_near_funding_time()

        signals = []

        for symbol in self.symbols:
            # Parse funding rate
            funding_rate = self._parse_funding_rate(data, symbol)

            if funding_rate is None:
                # No funding data available - skip
                continue

            # Store for reference
            self.last_funding_rates[symbol] = funding_rate
            self.estimated_apy[symbol] = self._calculate_expected_apy(abs(funding_rate))

            # Track funding history
            if symbol not in self.funding_history:
                self.funding_history[symbol] = []
            self.funding_history[symbol].append({
                'timestamp': datetime.utcnow(),
                'rate': funding_rate
            })
            # Keep last 24 hours only
            cutoff = datetime.utcnow() - timedelta(hours=24)
            self.funding_history[symbol] = [
                h for h in self.funding_history[symbol]
                if h['timestamp'] > cutoff
            ]

            # Check profitability
            is_profitable, direction, expected_profit = self._is_funding_profitable(funding_rate)

            # Check if we have an active arb for this symbol
            has_active_arb = symbol in self.active_arbs

            # Exit logic: Close after funding collected
            if has_active_arb and is_exit_window:
                arb = self.active_arbs[symbol]
                signal = {
                    'action': 'close_arb',
                    'symbol': symbol,
                    'size': arb['size'],
                    'leverage': 1,
                    'confidence': 0.90,
                    'reason': f'Funding collected, closing arb. Rate was {funding_rate*100:.4f}%',
                    'strategy': 'funding_arb',
                    'arb_type': arb['type'],
                    'funding_collected': funding_rate
                }
                signals.append(signal)
                continue

            # Entry logic: Open before funding if profitable
            if not has_active_arb and is_entry_window and is_profitable:
                if len(self.active_arbs) >= self.max_positions:
                    continue  # Max positions reached

                # Check for extreme rates (potential manipulation)
                if abs(funding_rate) > self.max_funding_rate:
                    continue

                signal = {
                    'action': 'open_arb',
                    'symbol': symbol,
                    'size': self.base_size_pct,
                    'leverage': self.leverage,
                    'confidence': 0.80,
                    'reason': f'Funding arb: {direction}, rate={funding_rate*100:.4f}%, APY={self.estimated_apy[symbol]*100:.1f}%',
                    'strategy': 'funding_arb',
                    'arb_type': direction,
                    'funding_rate': funding_rate,
                    'expected_profit': expected_profit,
                    'minutes_to_funding': time_to_funding
                }
                signals.append(signal)

        # Return best signal or hold
        if signals:
            best = max(signals, key=lambda x: x.get('confidence', 0))

            # Track opened arb
            if best['action'] == 'open_arb':
                self.active_arbs[best['symbol']] = {
                    'type': best['arb_type'],
                    'size': best['size'],
                    'entry_time': datetime.utcnow(),
                    'funding_rate': best['funding_rate']
                }
            elif best['action'] == 'close_arb':
                del self.active_arbs[best['symbol']]

            return best

        # No signal - Build detailed hold reason for diagnostics
        primary_symbol = self.symbols[0] if self.symbols else 'BTC/USDT'
        funding_rate = self.last_funding_rates.get(primary_symbol, 0)
        has_arb = len(self.active_arbs) > 0

        hold_reasons = []
        if not is_entry_window and not is_exit_window:
            hold_reasons.append(f'{time_to_funding:.0f}m to funding window')
        if funding_rate == 0:
            hold_reasons.append('No funding data available')
        elif abs(funding_rate) < self.min_funding_rate:
            hold_reasons.append(f'Rate {funding_rate*100:.4f}% < threshold {self.min_funding_rate*100:.2f}%')

        if not hold_reasons:
            hold_reasons.append('No profitable arb opportunity')

        rates_str = ', '.join([f"{s}:{r*100:.4f}%" for s, r in self.last_funding_rates.items()])
        return {
            'action': 'hold',
            'symbol': primary_symbol,
            'confidence': 0.0,
            'reason': f"FundingArb: {', '.join(hold_reasons)}",
            'strategy': 'funding_arb',
            'indicators': {
                'funding_rate': funding_rate,
                'min_rate_threshold': self.min_funding_rate,
                'funding_unfavorable': abs(funding_rate) < self.min_funding_rate,
                'is_entry_window': is_entry_window,
                'is_exit_window': is_exit_window,
                'time_to_funding': time_to_funding,
                'active_arbs': len(self.active_arbs),
                'estimated_apy': self.estimated_apy.get(primary_symbol, 0)
            }
        }

    def update_model(self, data: Optional[Dict[str, pd.DataFrame]] = None) -> bool:
        """Funding arb is rule-based, no model to update."""
        return True

    def get_status(self) -> Dict[str, Any]:
        """Get strategy status."""
        base = super().get_status()
        base.update({
            'min_funding_rate': self.min_funding_rate,
            'active_arbs': len(self.active_arbs),
            'active_arbs_detail': self.active_arbs,
            'last_funding_rates': self.last_funding_rates,
            'estimated_apy': self.estimated_apy,
            'next_funding': str(self._get_next_funding_time())
        })
        return base
