"""
RL Orchestrator - Master controller for RL-driven trading decisions
Phase 7: USDT-focused with 10x Kraken margin and 3x Bitrue ETFs
"""
import numpy as np
from models.rl_agent import TradingEnv, load_rl_agent
from executor import Executor
from portfolio import Portfolio
from exchanges.kraken_margin import KrakenMargin
from exchanges.bitrue_etf import BitrueETF
from strategies.ripple_momentum_lstm import generate_xrp_signals, generate_btc_signals


class RLOrchestrator:
    """
    Orchestrates RL agent decisions with leverage execution.
    Maps RL actions to actual paper trades with proper sizing and leverage.
    """

    def __init__(self, portfolio: Portfolio, data_dict: dict):
        self.portfolio = portfolio
        self.data = data_dict
        self.executor = Executor(portfolio)

        # Initialize exchange modules
        self.kraken = KrakenMargin(portfolio, max_leverage=10.0)
        self.bitrue = BitrueETF(portfolio)

        # Try to load trained RL model
        try:
            self.model = load_rl_agent()
            self.env = TradingEnv(data_dict, portfolio.balances.copy())
            self.env.portfolio = portfolio  # Share portfolio state
            print("RLOrchestrator: Model loaded successfully")
            self.enabled = True
        except Exception as e:
            print(f"RLOrchestrator: Could not load model - {e}")
            self.model = None
            self.env = None
            self.enabled = False

        # Action mapping: action_id -> (asset, action_type)
        # 0-2: BTC (buy/hold/sell), 3-5: XRP (buy/hold/sell), 6-8: USDT (park/hold/deploy)
        self.action_map = {
            0: ('BTC', 'buy'),
            1: ('BTC', 'hold'),
            2: ('BTC', 'sell'),
            3: ('XRP', 'buy'),
            4: ('XRP', 'hold'),
            5: ('XRP', 'sell'),
            6: ('USDT', 'park'),    # Sell assets to USDT
            7: ('USDT', 'hold'),    # Keep USDT
            8: ('USDT', 'deploy'),  # Deploy USDT to margin
        }

        # Risk parameters
        self.max_leverage_risk = 0.20  # Max 20% of USDT for margin
        self.spot_trade_size = 0.15    # 15% of USDT per spot trade

    def get_observation(self):
        """Get current observation for RL model"""
        if not self.enabled or self.env is None:
            return None
        return self.env._get_obs()

    def get_target_allocation(self):
        """Get target allocation from environment"""
        if self.enabled and self.env is not None:
            return self.env.targets
        return {'BTC': 0.45, 'XRP': 0.35, 'USDT': 0.20}

    def get_current_allocation(self, prices: dict):
        """Calculate current portfolio allocation"""
        total = self.portfolio.get_total_usd(prices)
        if total == 0:
            return {}

        allocation = {}
        for asset in ['BTC', 'XRP', 'USDT']:
            value = self.portfolio.balances.get(asset, 0) * prices.get(asset, 1.0)
            allocation[asset] = value / total

        return allocation

    def get_alignment_score(self, prices: dict):
        """Calculate how well current allocation matches targets"""
        targets = self.get_target_allocation()
        current = self.get_current_allocation(prices)

        score = 0.0
        for asset, target in targets.items():
            current_weight = current.get(asset, 0)
            score += min(current_weight, target)

        return score  # Max score is 1.0 (perfect alignment)

    def decide_and_execute(self, prices: dict = None):
        """
        Get RL model prediction and execute corresponding trade.
        Integrates leverage on strong momentum signals.

        Args:
            prices: Current prices dict. If None, uses env's internal prices.

        Returns:
            dict: Action taken and details
        """
        if not self.enabled:
            return {'action': 'disabled', 'reason': 'No RL model loaded'}

        # Get observation
        obs = self.get_observation()
        if obs is None:
            return {'action': 'error', 'reason': 'Could not get observation'}

        # Get model prediction
        action, _ = self.model.predict(obs, deterministic=True)
        action = int(action)

        # Map action to asset and type
        asset, action_type = self.action_map.get(action, ('XRP', 'hold'))

        result = {
            'action_id': action,
            'asset': asset,
            'action_type': action_type,
            'executed': False,
            'leverage_used': False
        }

        # Get momentum signals for leverage decisions
        xrp_signal = generate_xrp_signals(self.data) if 'XRP/USDT' in self.data else {'leverage_ok': False}
        btc_signal = generate_btc_signals(self.data) if 'BTC/USDT' in self.data else {'leverage_ok': False}

        usdt_available = self.portfolio.balances.get('USDT', 0)

        # Execute based on action
        if action_type == 'buy' and asset in ['BTC', 'XRP']:
            symbol = f"{asset}/USDT"
            price = prices.get(asset, 1.0) if prices else self.env._current_prices().get(asset, 1.0)

            # Check if leverage is appropriate
            signal = xrp_signal if asset == 'XRP' else btc_signal
            use_leverage = signal.get('leverage_ok', False) and signal.get('is_dip', False)

            if use_leverage and usdt_available > 100:
                # 10x leverage on dip
                collateral = usdt_available * self.max_leverage_risk
                success = self.kraken.open_long(asset, collateral, price)
                if success:
                    result['executed'] = True
                    result['leverage_used'] = True
                    result['leverage'] = 10
                    result['collateral'] = collateral
            elif usdt_available > 50:
                # Spot buy
                trade_value = usdt_available * self.spot_trade_size
                amount = trade_value / price
                success = self.executor.place_paper_order(symbol, 'buy', amount)
                result['executed'] = success
                result['amount'] = amount

        elif action_type == 'sell' and asset in ['BTC', 'XRP']:
            # Close any margin positions first
            if asset in self.kraken.positions:
                price = prices.get(asset, 1.0) if prices else self.env._current_prices().get(asset, 1.0)
                pnl = self.kraken.close_position(asset, price)
                result['margin_pnl'] = pnl

            # Sell spot holdings
            asset_balance = self.portfolio.balances.get(asset, 0)
            if asset_balance > 0:
                symbol = f"{asset}/USDT"
                sell_amount = asset_balance * 0.2  # Sell 20% of holdings
                success = self.executor.place_paper_order(symbol, 'sell', sell_amount)
                result['executed'] = success
                result['amount'] = sell_amount

        elif action_type == 'park':
            # Park: sell assets to USDT for safety
            for sell_asset in ['XRP', 'BTC']:
                holding = self.portfolio.balances.get(sell_asset, 0)
                if holding > 0:
                    price = prices.get(sell_asset, 1.0) if prices else 1.0
                    sell_amount = holding * 0.1  # Sell 10%
                    self.portfolio.update(sell_asset, -sell_amount)
                    self.portfolio.update('USDT', sell_amount * price)
                    result['executed'] = True
                    result['parked'] = True

        elif action_type == 'deploy':
            # Deploy: open leveraged positions on momentum
            if usdt_available > 200:
                # Check which asset has better momentum
                xrp_mom = xrp_signal.get('momentum', 0)
                btc_mom = btc_signal.get('momentum', 0)

                if xrp_mom > btc_mom and xrp_signal.get('leverage_ok', False):
                    target_asset = 'XRP'
                    price = prices.get('XRP', 2.0) if prices else 2.0
                elif btc_signal.get('leverage_ok', False):
                    target_asset = 'BTC'
                    price = prices.get('BTC', 90000.0) if prices else 90000.0
                else:
                    return result

                collateral = usdt_available * self.max_leverage_risk
                success = self.kraken.open_long(target_asset, collateral, price)
                result['executed'] = success
                result['leverage_used'] = True
                result['deployed_to'] = target_asset

        return result

    def update_env_step(self):
        """Advance environment step (call after each decision cycle)"""
        if self.enabled and self.env is not None:
            self.env.current_step = min(
                self.env.current_step + 1,
                self.env.max_steps - 1
            )

    def check_and_manage_positions(self, prices: dict):
        """
        Check margin positions for liquidation and take profit.
        Call this regularly during paper trading.
        """
        # Check liquidations
        liquidated = self.kraken.check_liquidations(prices)
        if liquidated:
            print(f"WARNING: Positions liquidated: {liquidated}")

        # Take profit on large gains (>20%)
        for asset, pos in list(self.kraken.positions.items()):
            current_price = prices.get(asset, pos['entry'])
            pnl_pct = self.kraken.get_unrealized_pnl(asset, current_price) / pos['collateral']

            if pnl_pct > 0.20:  # 20% profit
                print(f"TAKE PROFIT: {asset} at {pnl_pct*100:.1f}%")
                self.kraken.close_position(asset, current_price)

            elif pnl_pct < -0.10:  # 10% loss - reduce position
                print(f"STOP LOSS: {asset} at {pnl_pct*100:.1f}%")
                self.kraken.close_position(asset, current_price)

    def get_status(self) -> dict:
        """Get full orchestrator status"""
        return {
            'rl_enabled': self.enabled,
            'kraken': self.kraken.get_status(),
            'bitrue': self.bitrue.get_status(),
            'targets': self.get_target_allocation()
        }
