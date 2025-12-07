"""
RL Orchestrator - Master controller for RL-driven trading decisions
Phase 11: Bear Profit Tuning - Lower Short Thresholds + Mean Reversion + USDT Yield
"""
import numpy as np
from models.rl_agent import TradingEnv, load_rl_agent
from executor import Executor
from portfolio import Portfolio
from exchanges.kraken_margin import KrakenMargin
from exchanges.bitrue_etf import BitrueETF
from strategies.ripple_momentum_lstm import generate_xrp_signals, generate_btc_signals
from strategies.dip_buy_lstm import generate_dip_signals
from strategies.mean_reversion import calculate_vwap, is_above_vwap
from risk_manager import RiskManager


class RLOrchestrator:
    """
    Phase 11: Orchestrates RL agent with tuned bear market shorting.
    - Defensive in high-vol (park USDT + earn yield) OR deploy selective shorts
    - Aggressive offense in greed regime (low ATR + RSI oversold)
    - LSTM dip detector must align for leverage execution
    - Up to 10x Kraken / 3x Bitrue ETFs on confirmed dips
    - Phase 11: Earlier overbought entry (RSI>65, ATR>4%) + VWAP mean reversion filter
    """

    # Confidence threshold for leverage trades
    LEVERAGE_CONFIDENCE_THRESHOLD = 0.80

    # Phase 9: Offensive thresholds
    OFFENSIVE_CONFIDENCE_THRESHOLD = 0.85  # Higher bar for offense
    GREED_VOL_THRESHOLD = 0.02  # ATR% below this = greed regime

    # Phase 11: Tuned bear mode thresholds (lowered for earlier entry)
    BEAR_CONFIDENCE_THRESHOLD = 0.80  # Lowered from 0.82 for earlier shorts
    BEAR_VOL_THRESHOLD = 0.04  # Lowered from 0.05 - ATR% above this = bear regime
    RSI_OVERBOUGHT = 65  # Lowered from 70 - earlier overbought detection
    RSI_SHORT_EXIT = 40  # Auto-exit shorts when RSI drops below this (mean reversion complete)

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

        # Phase 9: RSI tracking
        self.current_rsi = {'XRP': 50.0, 'BTC': 50.0}

        # Phase 10: Trading mode (defensive/offensive/bear)
        self.mode = 'defensive'  # 'defensive', 'offensive', or 'bear'

        # Phase 10: Short position tracking
        self.short_positions = {'BTC': None, 'XRP': None}  # Track active shorts

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
        # Phase 10: 9-11: Short actions
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
            9: ('BTC', 'short'),    # Phase 10: Short BTC
            10: ('XRP', 'short'),   # Phase 10: Short XRP
            11: ('SHORT', 'close'), # Phase 10: Close all shorts
        }

        # Risk parameters (Phase 9: increased for offense)
        self.max_leverage_risk = 0.15  # Max 15% of USDT for margin (up from 10%)
        self.spot_trade_size = 0.10    # 10% of USDT per spot trade

    def calculate_rsi(self, symbol: str = 'XRP/USDT') -> float:
        """Calculate RSI for a symbol."""
        if symbol not in self.data:
            return 50.0

        df = self.data[symbol]
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

        asset = symbol.split('/')[0]
        self.current_rsi[asset] = rsi
        return rsi

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
        Phase 10: Get RL model prediction and execute with bear market shorting.

        - Defensive in high-vol (park USDT) OR deploy selective shorts
        - Aggressive offense in greed regime (low ATR <2% + RSI oversold)
        - LSTM dip detector must align for leverage execution
        - Up to 10x Kraken / 3x Bitrue ETFs on confirmed dips
        - Phase 10: Selective shorts on bear signals (high ATR + RSI overbought)

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

        # Update volatility and RSI from market data
        self.update_volatility('XRP/USDT')
        self.update_volatility('BTC/USDT')
        xrp_rsi = self.calculate_rsi('XRP/USDT')
        btc_rsi = self.calculate_rsi('BTC/USDT')

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
            'market_state': self.risk.get_market_state(),
            'mode': self.mode,
            'rsi': self.current_rsi.copy()
        }

        # Get LSTM dip signals for leverage decisions
        xrp_dip_signal = generate_dip_signals(self.data, 'XRP/USDT')
        btc_dip_signal = generate_dip_signals(self.data, 'BTC/USDT')

        # Get momentum signals
        xrp_signal = generate_xrp_signals(self.data) if 'XRP/USDT' in self.data else {'leverage_ok': False, 'confidence': 0}
        btc_signal = generate_btc_signals(self.data) if 'BTC/USDT' in self.data else {'leverage_ok': False, 'confidence': 0}

        usdt_available = self.portfolio.balances.get('USDT', 0)

        # Phase 9: Determine trading mode (defensive vs offensive)
        is_greed_regime = self.current_volatility < self.GREED_VOL_THRESHOLD
        xrp_oversold = xrp_rsi < 30
        btc_oversold = btc_rsi < 30

        # Check for offensive trigger: low ATR + LSTM dip signal + high confidence
        xrp_offensive = (
            is_greed_regime and
            xrp_dip_signal.get('is_dip', False) and
            xrp_dip_signal.get('confidence', 0) > self.OFFENSIVE_CONFIDENCE_THRESHOLD
        )
        btc_offensive = (
            is_greed_regime and
            btc_dip_signal.get('is_dip', False) and
            btc_dip_signal.get('confidence', 0) > self.OFFENSIVE_CONFIDENCE_THRESHOLD
        )

        # Phase 11: Check for bear mode (tuned: ATR>4% + RSI>65 + above VWAP)
        xrp_overbought = xrp_rsi > self.RSI_OVERBOUGHT
        btc_overbought = btc_rsi > self.RSI_OVERBOUGHT
        is_bear_regime = self.current_volatility > self.BEAR_VOL_THRESHOLD

        # Phase 11: Add VWAP mean reversion filter - only short when price above VWAP
        xrp_above_vwap = is_above_vwap(self.data, 'XRP/USDT') if 'XRP/USDT' in self.data else False
        btc_above_vwap = is_above_vwap(self.data, 'BTC/USDT') if 'BTC/USDT' in self.data else False

        xrp_bear = (
            is_bear_regime and
            xrp_overbought and
            xrp_above_vwap and  # Phase 11: VWAP filter
            xrp_dip_signal.get('confidence', 0) > self.BEAR_CONFIDENCE_THRESHOLD
        )
        btc_bear = (
            is_bear_regime and
            btc_overbought and
            btc_above_vwap and  # Phase 11: VWAP filter
            btc_dip_signal.get('confidence', 0) > self.BEAR_CONFIDENCE_THRESHOLD
        )

        # Update mode
        if xrp_bear or btc_bear:
            self.mode = 'bear'
            result['mode'] = 'bear'
            print(f"BEAR PROFIT MODE: Overbought deviation (ATR: {self.current_volatility*100:.2f}%, XRP RSI: {xrp_rsi:.1f}, BTC RSI: {btc_rsi:.1f}, VWAP: XRP={xrp_above_vwap}, BTC={btc_above_vwap})")
        elif xrp_offensive or btc_offensive:
            self.mode = 'offensive'
            result['mode'] = 'offensive'
            print(f"OFFENSIVE MODE: Greed regime detected (ATR: {self.current_volatility*100:.2f}%, XRP RSI: {xrp_rsi:.1f}, BTC RSI: {btc_rsi:.1f})")
        else:
            self.mode = 'defensive'
            result['mode'] = 'defensive'

        # Phase 11: Bear mode - deploy selective shorts with tuned sizing
        if self.mode == 'bear' and usdt_available > 100:
            short_leverage = self.risk.dynamic_leverage(self.current_volatility)
            short_leverage = min(short_leverage, 5)  # Cap shorts at 5x

            if xrp_bear and self.short_positions['XRP'] is None:
                price = prices.get('XRP', 2.0) if prices else 2.0
                collateral = usdt_available * 0.06  # Phase 11: 6% for shorts (from 8%)
                success = self.kraken.open_short('XRP', collateral, price, leverage=short_leverage)
                if success:
                    self.short_positions['XRP'] = {'entry': price, 'collateral': collateral}
                    result['executed'] = True
                    result['leverage_used'] = True
                    result['leverage'] = short_leverage
                    result['short'] = 'XRP'
                    result['collateral'] = collateral
                    print(f"BEAR PROFIT MODE: Short deployed on overbought deviation - {short_leverage}x XRP @ ${price:.4f}")
                    return result

            elif btc_bear and self.short_positions['BTC'] is None:
                price = prices.get('BTC', 90000.0) if prices else 90000.0
                # Use Bitrue 3x short ETF for BTC
                collateral = usdt_available * 0.10  # 10% for BTC shorts
                success = self.bitrue.buy_etf('BTC3S', collateral, price)
                if success:
                    self.short_positions['BTC'] = {'entry': price, 'collateral': collateral}
                    result['executed'] = True
                    result['leverage_used'] = True
                    result['leverage'] = 3
                    result['etf'] = 'BTC3S'
                    result['short'] = 'BTC'
                    result['collateral'] = collateral
                    print(f"BEAR MODE: Deployed 3x BTC3S ETF @ ${price:.2f}")
                    return result

        # Phase 8: Check if we should park USDT (defensive mode)
        if self.mode == 'defensive' and self.risk.should_park_usdt(self.current_volatility):
            result['defensive_mode'] = True
            if action_type not in ['park', 'hold', 'short', 'close']:
                print(f"DEFENSIVE: High volatility ({self.current_volatility*100:.1f}%) - parking USDT")
                # Override to park mode
                action_type = 'park'
                result['action_type'] = 'park'
                result['override_reason'] = 'high_volatility'

        # Phase 9: Offensive override - deploy leverage on confirmed dips
        elif self.mode == 'offensive' and action_type in ['buy', 'deploy']:
            # Aggressive leverage deployment in greed regime
            dynamic_lev = self.risk.dynamic_leverage(self.current_volatility)

            if xrp_offensive and usdt_available > 100:
                price = prices.get('XRP', 2.0) if prices else 2.0
                collateral = usdt_available * self.max_leverage_risk
                success = self.kraken.open_long('XRP', collateral, price, leverage=dynamic_lev)
                if success:
                    result['executed'] = True
                    result['leverage_used'] = True
                    result['leverage'] = dynamic_lev
                    result['collateral'] = collateral
                    result['offensive_trigger'] = 'xrp_dip'
                    print(f"OFFENSIVE: Deploying {dynamic_lev}x leverage on XRP dip (conf: {xrp_dip_signal.get('confidence', 0):.2f})")
                    return result

            elif btc_offensive and usdt_available > 100:
                price = prices.get('BTC', 90000.0) if prices else 90000.0
                collateral = usdt_available * self.max_leverage_risk
                success = self.bitrue.buy_etf('BTC3L', collateral, price)
                if success:
                    result['executed'] = True
                    result['leverage_used'] = True
                    result['leverage'] = 3
                    result['etf'] = 'BTC3L'
                    result['collateral'] = collateral
                    result['offensive_trigger'] = 'btc_dip'
                    print(f"OFFENSIVE: Deploying 3x BTC3L ETF on dip (conf: {btc_dip_signal.get('confidence', 0):.2f})")
                    return result

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

        # Phase 10: Handle explicit short actions from RL agent
        elif action_type == 'short' and asset in ['BTC', 'XRP']:
            # Validate bear conditions before opening short
            if is_bear_regime and usdt_available > 100:
                short_leverage = min(self.risk.dynamic_leverage(self.current_volatility), 5)

                if asset == 'XRP' and self.short_positions['XRP'] is None:
                    price = prices.get('XRP', 2.0) if prices else 2.0
                    collateral = usdt_available * 0.08
                    success = self.kraken.open_short('XRP', collateral, price, leverage=short_leverage)
                    if success:
                        self.short_positions['XRP'] = {'entry': price, 'collateral': collateral, 'leverage': short_leverage}
                        result['executed'] = True
                        result['leverage_used'] = True
                        result['leverage'] = short_leverage
                        result['short'] = 'XRP'
                        result['collateral'] = collateral
                        print(f"SHORT: Deployed {short_leverage}x short on XRP @ ${price:.4f}")

                elif asset == 'BTC' and self.short_positions['BTC'] is None:
                    price = prices.get('BTC', 90000.0) if prices else 90000.0
                    collateral = usdt_available * 0.10
                    success = self.bitrue.buy_etf('BTC3S', collateral, price)
                    if success:
                        self.short_positions['BTC'] = {'entry': price, 'collateral': collateral, 'leverage': 3}
                        result['executed'] = True
                        result['leverage_used'] = True
                        result['leverage'] = 3
                        result['etf'] = 'BTC3S'
                        result['short'] = 'BTC'
                        result['collateral'] = collateral
                        print(f"SHORT: Deployed BTC3S ETF @ ${price:.2f}")
            else:
                print(f"SHORT BLOCKED: Not in bear regime (ATR: {self.current_volatility*100:.2f}%)")

        # Phase 10: Handle close shorts action
        elif action_type == 'close' and asset == 'SHORT':
            closed_any = False

            # Close XRP short (Kraken margin)
            if self.short_positions['XRP'] is not None:
                price = prices.get('XRP', 2.0) if prices else 2.0
                pnl = self.kraken.close_short('XRP', price)
                entry = self.short_positions['XRP']['entry']
                pnl_pct = ((entry - price) / entry) * 100
                print(f"CLOSED XRP SHORT: Entry ${entry:.4f} -> Exit ${price:.4f} (PnL: {pnl_pct:+.2f}%)")
                self.short_positions['XRP'] = None
                result['short_pnl_xrp'] = pnl
                closed_any = True

            # Close BTC short (Bitrue ETF)
            if self.short_positions['BTC'] is not None:
                price = prices.get('BTC', 90000.0) if prices else 90000.0
                if self.bitrue.etf_holdings.get('BTC3S', 0) > 0:
                    self.bitrue.sell_etf('BTC3S', underlying_price=price)
                entry = self.short_positions['BTC']['entry']
                pnl_pct = ((entry - price) / entry) * 100
                print(f"CLOSED BTC SHORT (BTC3S): Entry ${entry:.2f} -> Exit ${price:.2f} (PnL: {pnl_pct:+.2f}%)")
                self.short_positions['BTC'] = None
                closed_any = True

            if closed_any:
                result['executed'] = True
                result['closed_shorts'] = True

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
        Phase 10: Also manages short positions.
        """
        # Check long liquidations
        liquidated = self.kraken.check_liquidations(prices)
        if liquidated:
            print(f"WARNING: Long positions liquidated: {liquidated}")

        # Take profit on large gains (>20%) for longs
        for asset, pos in list(self.kraken.positions.items()):
            current_price = prices.get(asset, pos['entry'])
            pnl_pct = self.kraken.get_unrealized_pnl(asset, current_price) / pos['collateral']

            if pnl_pct > 0.20:  # 20% profit
                print(f"TAKE PROFIT: {asset} at {pnl_pct*100:.1f}%")
                self.kraken.close_position(asset, current_price)

            elif pnl_pct < -0.10:  # 10% loss - reduce position
                print(f"STOP LOSS: {asset} at {pnl_pct*100:.1f}%")
                self.kraken.close_position(asset, current_price)

        # Phase 10: Manage short positions
        self._manage_short_positions(prices)

    def _manage_short_positions(self, prices: dict):
        """
        Phase 11: Manage short positions - take profit, stop loss, RSI mean reversion exit.
        """
        # Get current RSI for mean reversion exit
        xrp_rsi = self.current_rsi.get('XRP', 50)
        btc_rsi = self.current_rsi.get('BTC', 50)

        # Check XRP short (Kraken margin)
        if self.short_positions['XRP'] is not None:
            pos = self.short_positions['XRP']
            current_price = prices.get('XRP', 2.0)
            entry_price = pos['entry']
            collateral = pos['collateral']
            leverage = pos.get('leverage', 5)

            # Calculate P&L for short (profit when price drops)
            pnl_pct = ((entry_price - current_price) / entry_price) * leverage

            # Take profit at 15% gain
            if pnl_pct > 0.15:
                pnl = self.kraken.close_short('XRP', current_price)
                print(f"SHORT TAKE PROFIT XRP: {pnl_pct*100:.1f}% gain")
                self.short_positions['XRP'] = None

            # Stop loss at 8% loss (price went UP)
            elif pnl_pct < -0.08:
                pnl = self.kraken.close_short('XRP', current_price)
                print(f"SHORT STOP LOSS XRP: {pnl_pct*100:.1f}% loss")
                self.short_positions['XRP'] = None

            # Phase 11: Mean reversion exit - close when RSI drops below 40
            elif xrp_rsi < self.RSI_SHORT_EXIT:
                pnl = self.kraken.close_short('XRP', current_price)
                print(f"SHORT MEAN REVERSION EXIT XRP: RSI {xrp_rsi:.1f} < {self.RSI_SHORT_EXIT} (PnL: {pnl_pct*100:.1f}%)")
                self.short_positions['XRP'] = None

        # Check BTC short (Bitrue ETF)
        if self.short_positions['BTC'] is not None:
            pos = self.short_positions['BTC']
            current_price = prices.get('BTC', 90000.0)
            entry_price = pos['entry']

            # Calculate P&L for short ETF (3x)
            pnl_pct = ((entry_price - current_price) / entry_price) * 3

            # Take profit at 12% gain
            if pnl_pct > 0.12:
                if self.bitrue.etf_holdings.get('BTC3S', 0) > 0:
                    self.bitrue.sell_etf('BTC3S', underlying_price=current_price)
                print(f"SHORT TAKE PROFIT BTC: {pnl_pct*100:.1f}% gain")
                self.short_positions['BTC'] = None

            # Stop loss at 10% loss
            elif pnl_pct < -0.10:
                if self.bitrue.etf_holdings.get('BTC3S', 0) > 0:
                    self.bitrue.sell_etf('BTC3S', underlying_price=current_price)
                print(f"SHORT STOP LOSS BTC: {pnl_pct*100:.1f}% loss")
                self.short_positions['BTC'] = None

            # Phase 11: Mean reversion exit - close when RSI drops below 40
            elif btc_rsi < self.RSI_SHORT_EXIT:
                if self.bitrue.etf_holdings.get('BTC3S', 0) > 0:
                    self.bitrue.sell_etf('BTC3S', underlying_price=current_price)
                print(f"SHORT MEAN REVERSION EXIT BTC: RSI {btc_rsi:.1f} < {self.RSI_SHORT_EXIT} (PnL: {pnl_pct*100:.1f}%)")
                self.short_positions['BTC'] = None

    def get_status(self) -> dict:
        """Get full orchestrator status"""
        return {
            'rl_enabled': self.enabled,
            'kraken': self.kraken.get_status(),
            'bitrue': self.bitrue.get_status(),
            'targets': self.get_target_allocation()
        }
