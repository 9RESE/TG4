"""
RL Orchestrator - Master controller for RL-driven trading decisions
Phase 8: Defensive Leverage + Volatility Filters + Selective Execution
"""
import numpy as np
from models.rl_agent import TradingEnv, load_rl_agent
from executor import Executor
from portfolio import Portfolio
from exchanges.kraken_margin import KrakenMargin
from exchanges.bitrue_etf import BitrueETF
from strategies.ripple_momentum_lstm import generate_xrp_signals, generate_btc_signals
from risk_manager import RiskManager


class RLOrchestrator:
    """
    Phase 8: Orchestrates RL agent decisions with volatility-aware leverage.
    - Smart, selective leverage (Kraken 10x only when probability >80%)
    - Bitrue 3x ETFs for asymmetric upside
    - USDT parking during high-fear periods
    """

    # Confidence threshold for leverage trades
    LEVERAGE_CONFIDENCE_THRESHOLD = 0.80

    def __init__(self, portfolio: Portfolio, data_dict: dict):
        self.portfolio = portfolio
        self.data = data_dict
        self.executor = Executor(portfolio)

        # Phase 8: Initialize risk manager with volatility awareness
        self.risk = RiskManager(max_drawdown=0.20, max_leverage=10.0)

        # Initialize exchange modules
        self.kraken = KrakenMargin(portfolio, max_leverage=10.0)
        self.bitrue = BitrueETF(portfolio)

        # Current volatility state
        self.current_volatility = 0.05  # Default moderate
        self.current_atr = {}  # Per-asset ATR tracking

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

        # Risk parameters (Phase 8: reduced from 20% to 10% max)
        self.max_leverage_risk = 0.10  # Max 10% of USDT for margin
        self.spot_trade_size = 0.10    # 10% of USDT per spot trade

    def update_volatility(self, symbol: str = 'XRP/USDT'):
        """Update current volatility from market data."""
        if symbol not in self.data:
            return

        df = self.data[symbol]
        if len(df) < 20:
            return

        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        atr_pct = self.risk.calculate_atr_pct(high, low, close, period=14)
        self.current_volatility = atr_pct
        self.current_atr[symbol] = atr_pct

        return atr_pct

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
        Phase 8: Get RL model prediction and execute with volatility-aware leverage.

        - Only leverage when confidence > 80% AND volatility permits
        - Use Bitrue 3x ETFs for asymmetric upside on BTC
        - Park USDT during high-fear periods

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

        # Update volatility from market data
        self.update_volatility('XRP/USDT')
        self.update_volatility('BTC/USDT')

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
            'leverage_used': False,
            'volatility': self.current_volatility,
            'market_state': self.risk.get_market_state()
        }

        # Get momentum signals for leverage decisions
        xrp_signal = generate_xrp_signals(self.data) if 'XRP/USDT' in self.data else {'leverage_ok': False, 'confidence': 0}
        btc_signal = generate_btc_signals(self.data) if 'BTC/USDT' in self.data else {'leverage_ok': False, 'confidence': 0}

        usdt_available = self.portfolio.balances.get('USDT', 0)

        # Phase 8: Check if we should park USDT (defensive mode)
        if self.risk.should_park_usdt(self.current_volatility):
            result['defensive_mode'] = True
            if action_type not in ['park', 'hold']:
                print(f"DEFENSIVE: High volatility ({self.current_volatility*100:.1f}%) - parking USDT")
                # Override to park mode
                action_type = 'park'
                result['action_type'] = 'park'
                result['override_reason'] = 'high_volatility'

        # Execute based on action
        if action_type == 'buy' and asset in ['BTC', 'XRP']:
            symbol = f"{asset}/USDT"
            price = prices.get(asset, 1.0) if prices else self.env._current_prices().get(asset, 1.0)

            # Get signal for this asset
            signal = xrp_signal if asset == 'XRP' else btc_signal
            confidence = signal.get('confidence', 0.5)
            is_dip = signal.get('is_dip', False)
            leverage_ok = signal.get('leverage_ok', False)

            # Phase 8: Dynamic leverage based on volatility
            dynamic_lev = self.risk.dynamic_leverage(self.current_volatility)

            # Only use leverage if confidence > 80% AND dip detected AND volatility permits
            use_leverage = (
                leverage_ok and
                is_dip and
                confidence > self.LEVERAGE_CONFIDENCE_THRESHOLD and
                dynamic_lev > 1
            )

            if use_leverage and usdt_available > 100:
                # Calculate position size based on volatility and confidence
                collateral = self.risk.dynamic_position_size(
                    usdt_available, self.current_volatility, confidence
                )

                if asset == 'XRP':
                    # Kraken 10x margin for XRP (scaled by dynamic_lev)
                    success = self.kraken.open_long(asset, collateral, price, leverage=dynamic_lev)
                    if success:
                        result['executed'] = True
                        result['leverage_used'] = True
                        result['leverage'] = dynamic_lev
                        result['collateral'] = collateral
                        result['confidence'] = confidence
                else:
                    # BTC: Use Bitrue 3x ETF for asymmetric upside
                    success = self.bitrue.buy_etf('BTC3L', collateral, price)
                    if success:
                        result['executed'] = True
                        result['leverage_used'] = True
                        result['leverage'] = 3
                        result['etf'] = 'BTC3L'
                        result['collateral'] = collateral

            elif usdt_available > 50 and not self.risk.should_park_usdt(self.current_volatility):
                # Spot buy (only if not in defensive mode)
                trade_value = usdt_available * self.spot_trade_size
                amount = trade_value / price
                success = self.executor.place_paper_order(symbol, 'buy', amount)
                result['executed'] = success
                result['amount'] = amount
            else:
                print(f"DEFENSIVE: Holding USDT - no high-conviction signal (conf: {confidence:.2f})")

        elif action_type == 'sell' and asset in ['BTC', 'XRP']:
            # Close any margin positions first
            if asset in self.kraken.positions:
                price = prices.get(asset, 1.0) if prices else self.env._current_prices().get(asset, 1.0)
                pnl = self.kraken.close_position(asset, price)
                result['margin_pnl'] = pnl

            # Close any ETF positions
            if asset == 'BTC':
                btc_price = prices.get('BTC', 90000.0) if prices else 90000.0
                for etf in ['BTC3L', 'BTC3S']:
                    if self.bitrue.etf_holdings.get(etf, 0) > 0:
                        self.bitrue.sell_etf(etf, underlying_price=btc_price)

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
            # Deploy: open leveraged positions on momentum (Phase 8: with vol filter)
            if usdt_available > 200 and not self.risk.should_park_usdt(self.current_volatility):
                # Check which asset has better momentum
                xrp_conf = xrp_signal.get('confidence', 0)
                btc_conf = btc_signal.get('confidence', 0)
                xrp_dip = xrp_signal.get('is_dip', False)
                btc_dip = btc_signal.get('is_dip', False)

                dynamic_lev = self.risk.dynamic_leverage(self.current_volatility)

                # Only deploy if confidence > 80%
                if xrp_conf > btc_conf and xrp_conf > self.LEVERAGE_CONFIDENCE_THRESHOLD and xrp_dip:
                    target_asset = 'XRP'
                    price = prices.get('XRP', 2.0) if prices else 2.0
                    collateral = self.risk.dynamic_position_size(usdt_available, self.current_volatility, xrp_conf)
                    success = self.kraken.open_long(target_asset, collateral, price, leverage=dynamic_lev)
                    result['executed'] = success
                    result['leverage_used'] = True
                    result['leverage'] = dynamic_lev
                    result['deployed_to'] = target_asset

                elif btc_conf > self.LEVERAGE_CONFIDENCE_THRESHOLD and btc_dip:
                    # Use Bitrue 3x ETF for BTC
                    price = prices.get('BTC', 90000.0) if prices else 90000.0
                    collateral = self.risk.dynamic_position_size(usdt_available, self.current_volatility, btc_conf)
                    success = self.bitrue.buy_etf('BTC3L', collateral, price)
                    result['executed'] = success
                    result['leverage_used'] = True
                    result['leverage'] = 3
                    result['etf'] = 'BTC3L'
                    result['deployed_to'] = 'BTC'
                else:
                    print(f"DEFENSIVE: No high-conviction deploy signal (XRP: {xrp_conf:.2f}, BTC: {btc_conf:.2f})")

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
