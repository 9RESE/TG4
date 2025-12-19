from strategies.ripple_momentum_lstm import generate_ripple_signals
from strategies.stablecoin_arb import StableArb
from strategies.rebalancer import rebalance
from portfolio import Portfolio
import numpy as np

class Ensemble:
    """
    Ensemble strategy combining multiple signal sources.

    Weights:
    - 40% LSTM momentum
    - 30% Stablecoin arbitrage
    - 20% Rebalancing signals
    - 10% RL agent (when available)
    """

    def __init__(self, data: dict, portfolio: Portfolio):
        self.data = data
        self.portfolio = portfolio
        self.arb = StableArb()
        self.rl_model = None
        self._try_load_rl()

    def _try_load_rl(self):
        """Try to load RL model if available"""
        try:
            from models.rl_agent import load_rl_agent
            self.rl_model = load_rl_agent()
            print("Ensemble: RL model loaded")
        except:
            print("Ensemble: No RL model, using rule-based signals")

    def get_signal(self, symbol: str = 'XRP/USDT'):
        """
        Get combined trading signal from all strategies.

        Returns:
            dict: Signal with action and confidence
        """
        signals = {
            'lstm': 0.0,
            'arb': 0.0,
            'rebalance': 0.0,
            'rl': 0.0
        }

        # 1. LSTM momentum signal (40% weight)
        if symbol in self.data:
            try:
                lstm_signals = generate_ripple_signals(self.data, symbol)
                if len(lstm_signals) > 0:
                    signals['lstm'] = 1.0 if lstm_signals.iloc[-1] else -0.5
            except Exception as e:
                print(f"LSTM signal error: {e}")

        # 2. Arbitrage opportunities (30% weight)
        arb_opps = self.arb.find_opportunities()
        if arb_opps:
            # Positive signal if profitable arb exists
            best_spread = max(opp.get('spread', 0) for opp in arb_opps)
            signals['arb'] = min(best_spread * 100, 1.0)  # Scale spread to 0-1

        # 3. Rebalance signal (20% weight)
        prices = self._get_current_prices()
        total_usd = self.portfolio.get_total_usd(prices)
        if total_usd > 0:
            # Check if XRP is underweight (target: 30%)
            xrp_weight = self.portfolio.balances.get('XRP', 0) * prices.get('XRP', 2.0) / total_usd
            if xrp_weight < 0.25:
                signals['rebalance'] = 0.5  # Buy signal
            elif xrp_weight > 0.35:
                signals['rebalance'] = -0.5  # Sell signal

        # 4. RL signal (10% weight) - placeholder for now
        # TODO: Get actual RL prediction when model is trained

        # Combine signals with weights
        weights = {'lstm': 0.4, 'arb': 0.3, 'rebalance': 0.2, 'rl': 0.1}
        combined = sum(signals[k] * weights[k] for k in signals)

        # Determine action
        if combined > 0.3:
            action = 'long_xrp'
            confidence = min(combined, 1.0)
        elif combined < -0.3:
            action = 'reduce_xrp'
            confidence = min(abs(combined), 1.0)
        else:
            action = 'hold'
            confidence = 0.5

        # Check for arb-specific action
        if arb_opps and any(opp.get('spread', 0) > 0.005 for opp in arb_opps):
            action = 'arb_rlusd'
            confidence = 0.8

        return {
            'action': action,
            'confidence': confidence,
            'signals': signals,
            'arb_opportunities': arb_opps
        }

    def _get_current_prices(self):
        """Get current prices from data"""
        prices = {'USDT': 1.0, 'USDC': 1.0, 'RLUSD': 1.0}
        for sym, df in self.data.items():
            if len(df) > 0:
                base = sym.split('/')[0]
                prices[base] = df['close'].iloc[-1]
        return prices

    def update_data(self, new_data: dict):
        """Update market data for fresh signals"""
        self.data = new_data
