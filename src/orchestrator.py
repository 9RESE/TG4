"""
RL Orchestrator - Master controller for RL-driven trading decisions
"""
import numpy as np
from models.rl_agent import TradingEnv, load_rl_agent
from executor import Executor
from portfolio import Portfolio


class RLOrchestrator:
    """
    Orchestrates RL agent decisions and execution.
    Maps RL actions to actual paper trades with proper sizing and leverage.
    """

    def __init__(self, portfolio: Portfolio, data_dict: dict):
        self.portfolio = portfolio
        self.data = data_dict
        self.executor = Executor(portfolio)

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
        # 0-2: BTC (buy/hold/sell), 3-5: XRP, 6-8: RLUSD
        self.action_map = {
            0: ('BTC', 'buy'),
            1: ('BTC', 'hold'),
            2: ('BTC', 'sell'),
            3: ('XRP', 'buy'),
            4: ('XRP', 'hold'),
            5: ('XRP', 'sell'),
            6: ('RLUSD', 'buy'),
            7: ('RLUSD', 'hold'),
            8: ('RLUSD', 'sell'),
        }

        # Leverage settings per asset
        self.leverage = {
            'BTC': 1.0,   # Spot only for BTC (safer)
            'XRP': 3.0,   # 3x leverage for XRP accumulation
            'RLUSD': 1.0  # Spot for stablecoin
        }

        # Position sizing (% of available USDT)
        self.position_size = {
            'BTC': 0.05,   # 5% per trade
            'XRP': 0.10,   # 10% per trade (more aggressive)
            'RLUSD': 0.15  # 15% per trade (accumulate stablecoin)
        }

    def get_observation(self):
        """Get current observation for RL model"""
        if not self.enabled or self.env is None:
            return None
        return self.env._get_obs()

    def decide_and_execute(self, prices: dict = None):
        """
        Get RL model prediction and execute corresponding trade.

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
            'executed': False
        }

        # Execute trade based on action
        if action_type == 'buy':
            symbol = f"{asset}/USDT"
            leverage = self.leverage.get(asset, 1.0)
            size_pct = self.position_size.get(asset, 0.05)

            usdt_available = self.portfolio.balances.get('USDT', 0)
            if usdt_available > 50:  # Min $50 for trade
                # Get price
                if prices and asset in prices:
                    price = prices[asset]
                else:
                    price = self.env._current_prices().get(asset, 1.0)

                trade_value = usdt_available * size_pct
                amount = trade_value / price

                success = self.executor.place_paper_order(
                    symbol, 'buy', amount, leverage=leverage
                )
                result['executed'] = success
                result['amount'] = amount
                result['leverage'] = leverage

        elif action_type == 'sell':
            asset_balance = self.portfolio.balances.get(asset, 0)
            if asset_balance > 0:
                symbol = f"{asset}/USDT"
                sell_amount = asset_balance * 0.2  # Sell 20% of holdings

                success = self.executor.place_paper_order(
                    symbol, 'sell', sell_amount
                )
                result['executed'] = success
                result['amount'] = sell_amount

        return result

    def update_env_step(self):
        """Advance environment step (call after each decision cycle)"""
        if self.enabled and self.env is not None:
            self.env.current_step = min(
                self.env.current_step + 1,
                self.env.max_steps - 1
            )

    def get_target_allocation(self):
        """Get target allocation from environment"""
        if self.enabled and self.env is not None:
            return self.env.targets
        return {'BTC': 0.4, 'XRP': 0.3, 'RLUSD': 0.2, 'USDT': 0.05, 'USDC': 0.05}

    def get_current_allocation(self, prices: dict):
        """Calculate current portfolio allocation"""
        total = self.portfolio.get_total_usd(prices)
        if total == 0:
            return {}

        allocation = {}
        for asset in ['BTC', 'XRP', 'RLUSD', 'USDT', 'USDC']:
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
            # Score based on minimum of current and target
            score += min(current_weight, target)

        return score  # Max score is 1.0 (perfect alignment)
