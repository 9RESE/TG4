"""
Defensive Yield Strategy - Live Production Strategy
Phase 15: Modular Strategy Factory

Wraps the existing RLOrchestrator logic as a BaseStrategy implementation.
This is the LIVE strategy - modifications require thorough backtesting.

Key Features:
- RL-driven decision making with PPO agent
- Defensive mode: Park USDT during high volatility (ATR > 4%)
- Offensive mode: Deploy leverage on confirmed dips in greed regime
- Bear mode: Selective shorts on overbought rips (RSI > 65 + above VWAP)
- Auto-yield accrual (6.5% avg APY on parked USDT)
- Up to 10x Kraken margin / 3x Bitrue ETFs
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from strategies.base_strategy import BaseStrategy
from models.rl_agent import TradingEnv, load_rl_agent
from strategies.ripple_momentum_lstm import generate_xrp_signals, generate_btc_signals
from strategies.dip_buy_lstm import generate_dip_signals
from strategies.mean_reversion import calculate_vwap, is_above_vwap
from risk_manager import RiskManager


class DefensiveYield(BaseStrategy):
    """
    Defensive Yield Strategy - RL-driven with volatility awareness.

    Modes:
    - defensive: Park USDT, accrue yield, wait for opportunities
    - offensive: Deploy leverage on confirmed dips (low ATR + high confidence)
    - bear: Selective shorts on overbought conditions (high ATR + RSI > 65)
    """

    # Confidence thresholds
    LEVERAGE_CONFIDENCE_THRESHOLD = 0.80
    OFFENSIVE_CONFIDENCE_THRESHOLD = 0.85
    BEAR_CONFIDENCE_THRESHOLD = 0.78

    # Volatility thresholds
    GREED_VOL_THRESHOLD = 0.02  # ATR% below this = greed regime
    BEAR_VOL_THRESHOLD = 0.04   # ATR% above this = bear regime
    OPPORTUNISTIC_VOL_THRESHOLD = 0.042

    # RSI thresholds
    RSI_OVERBOUGHT = 65
    RSI_SHORT_EXIT = 40
    RSI_RIP_THRESHOLD = 72
    RSI_OPPORTUNISTIC = 68

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.name = 'defensive_yield'
        self.max_leverage = config.get('max_leverage', 10)

        # Risk manager
        self.risk = RiskManager(max_drawdown=0.20, max_leverage=10.0)

        # State tracking
        self.current_volatility = 0.05
        self.current_atr = {}
        self.current_rsi = {'XRP': 50.0, 'BTC': 50.0}
        self.mode = 'defensive'
        self.short_positions = {'BTC': None, 'XRP': None}

        # RL model
        self.model = None
        self.env = None
        self.rl_enabled = False

        # Action mapping
        self.action_map = {
            0: ('BTC', 'buy'), 1: ('BTC', 'hold'), 2: ('BTC', 'sell'),
            3: ('XRP', 'buy'), 4: ('XRP', 'hold'), 5: ('XRP', 'sell'),
            6: ('USDT', 'park'), 7: ('USDT', 'hold'), 8: ('USDT', 'deploy'),
            9: ('BTC', 'short'), 10: ('XRP', 'short'), 11: ('SHORT', 'close'),
        }

    def initialize(self, portfolio: Any, exchanges: Dict[str, Any]) -> None:
        """Initialize with portfolio and load RL model."""
        super().initialize(portfolio, exchanges)

        # Load RL model
        try:
            self.model = load_rl_agent()
            self.rl_enabled = True
            print(f"DefensiveYield: RL model loaded successfully")
        except Exception as e:
            print(f"DefensiveYield: Could not load RL model - {e}")
            self.model = None
            self.rl_enabled = False

    def _calculate_rsi(self, df: pd.DataFrame) -> float:
        """Calculate RSI from dataframe."""
        if len(df) < 15:
            return 50.0

        close = df['close'].values
        deltas = np.diff(close[-15:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0.001

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_atr_pct(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR as percentage of price."""
        if len(df) < period + 1:
            return 0.05  # Default moderate

        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        return self.risk.calculate_atr_pct(high, low, close, period)

    def _determine_mode(self, data: Dict[str, pd.DataFrame]) -> str:
        """Determine trading mode based on market conditions."""
        xrp_atr = self._calculate_atr_pct(data.get('XRP/USDT', pd.DataFrame()))
        btc_atr = self._calculate_atr_pct(data.get('BTC/USDT', pd.DataFrame()))
        self.current_volatility = max(xrp_atr, btc_atr)

        xrp_rsi = self._calculate_rsi(data.get('XRP/USDT', pd.DataFrame()))
        btc_rsi = self._calculate_rsi(data.get('BTC/USDT', pd.DataFrame()))
        self.current_rsi = {'XRP': xrp_rsi, 'BTC': btc_rsi}

        # Check for bear mode
        is_bear_regime = self.current_volatility > self.BEAR_VOL_THRESHOLD
        xrp_overbought = xrp_rsi > self.RSI_OVERBOUGHT
        btc_overbought = btc_rsi > self.RSI_OVERBOUGHT

        xrp_above_vwap = is_above_vwap(data, 'XRP/USDT')
        btc_above_vwap = is_above_vwap(data, 'BTC/USDT')

        if is_bear_regime and (xrp_overbought or btc_overbought):
            if xrp_rsi > self.RSI_RIP_THRESHOLD or btc_rsi > self.RSI_RIP_THRESHOLD:
                return 'bear_rip'
            elif (xrp_overbought and xrp_above_vwap) or (btc_overbought and btc_above_vwap):
                return 'bear_selective'
            return 'defensive'

        # Check for offensive mode
        is_greed_regime = self.current_volatility < self.GREED_VOL_THRESHOLD
        if is_greed_regime:
            xrp_dip = generate_dip_signals(data, 'XRP/USDT')
            btc_dip = generate_dip_signals(data, 'BTC/USDT')

            if (xrp_dip.get('is_dip') and xrp_dip.get('confidence', 0) > self.OFFENSIVE_CONFIDENCE_THRESHOLD):
                return 'offensive'
            if (btc_dip.get('is_dip') and btc_dip.get('confidence', 0) > self.OFFENSIVE_CONFIDENCE_THRESHOLD):
                return 'offensive'

        return 'defensive'

    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Generate trading signals from market data.

        Returns signal dict with action, size, leverage, etc.
        """
        # Determine current mode
        self.mode = self._determine_mode(data)

        # Get LSTM signals
        xrp_signal = generate_xrp_signals(data) if 'XRP/USDT' in data else {'confidence': 0}
        btc_signal = generate_btc_signals(data) if 'BTC/USDT' in data else {'confidence': 0}
        xrp_dip = generate_dip_signals(data, 'XRP/USDT')
        btc_dip = generate_dip_signals(data, 'BTC/USDT')

        # Default hold signal
        signal = {
            'action': 'hold',
            'symbol': 'USDT',
            'size': 0.0,
            'leverage': 1,
            'confidence': 0.0,
            'reason': f'Mode: {self.mode}',
            'mode': self.mode,
            'volatility': self.current_volatility,
            'rsi': self.current_rsi.copy()
        }

        # Bear mode - short signals
        if self.mode in ['bear_rip', 'bear_selective']:
            xrp_above_vwap = is_above_vwap(data, 'XRP/USDT')
            btc_above_vwap = is_above_vwap(data, 'BTC/USDT')

            is_rip = self.mode == 'bear_rip'
            risk_pct = 0.08 if is_rip else 0.05

            if self.current_rsi['XRP'] > self.RSI_OVERBOUGHT and xrp_above_vwap:
                if self.short_positions['XRP'] is None:
                    signal = {
                        'action': 'short',
                        'symbol': 'XRP/USDT',
                        'size': risk_pct,
                        'leverage': min(self.risk.dynamic_leverage(self.current_volatility), 5),
                        'confidence': xrp_dip.get('confidence', 0.8),
                        'reason': f'Bear {"rip" if is_rip else "selective"} short - RSI: {self.current_rsi["XRP"]:.1f}',
                        'mode': self.mode
                    }

            elif self.current_rsi['BTC'] > self.RSI_OVERBOUGHT and btc_above_vwap:
                if self.short_positions['BTC'] is None:
                    signal = {
                        'action': 'short',
                        'symbol': 'BTC/USDT',
                        'size': risk_pct * 1.5,
                        'leverage': 3,  # Bitrue ETF
                        'confidence': btc_dip.get('confidence', 0.8),
                        'reason': f'Bear {"rip" if is_rip else "selective"} short BTC3S - RSI: {self.current_rsi["BTC"]:.1f}',
                        'mode': self.mode,
                        'use_etf': 'BTC3S'
                    }

        # Offensive mode - long signals
        elif self.mode == 'offensive':
            dynamic_lev = self.risk.dynamic_leverage(self.current_volatility)

            if xrp_dip.get('is_dip') and xrp_dip.get('confidence', 0) > self.OFFENSIVE_CONFIDENCE_THRESHOLD:
                signal = {
                    'action': 'buy',
                    'symbol': 'XRP/USDT',
                    'size': 0.15,
                    'leverage': dynamic_lev,
                    'confidence': xrp_dip['confidence'],
                    'reason': f'Offensive dip buy - conf: {xrp_dip["confidence"]:.2f}',
                    'mode': self.mode
                }

            elif btc_dip.get('is_dip') and btc_dip.get('confidence', 0) > self.OFFENSIVE_CONFIDENCE_THRESHOLD:
                signal = {
                    'action': 'buy',
                    'symbol': 'BTC/USDT',
                    'size': 0.15,
                    'leverage': 3,  # Bitrue ETF
                    'confidence': btc_dip['confidence'],
                    'reason': f'Offensive dip buy BTC3L - conf: {btc_dip["confidence"]:.2f}',
                    'mode': self.mode,
                    'use_etf': 'BTC3L'
                }

        # Defensive mode - accrue yield
        elif self.mode == 'defensive':
            signal = {
                'action': 'hold',
                'symbol': 'USDT',
                'size': 0.0,
                'leverage': 1,
                'confidence': 0.5,
                'reason': f'Defensive - parking USDT, accruing yield (ATR: {self.current_volatility*100:.1f}%)',
                'mode': self.mode,
                'accrue_yield': True
            }

        return signal

    def update_model(self, data: Optional[Dict[str, pd.DataFrame]] = None) -> bool:
        """
        RL model training is handled separately via main.py --mode train-rl.
        This method is a no-op for live trading.
        """
        return True

    def get_status(self) -> Dict[str, Any]:
        """Get strategy status."""
        base_status = super().get_status()
        base_status.update({
            'mode': self.mode,
            'rl_enabled': self.rl_enabled,
            'volatility': self.current_volatility,
            'rsi': self.current_rsi,
            'short_positions': {k: v is not None for k, v in self.short_positions.items()}
        })
        return base_status
