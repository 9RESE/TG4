from data_fetcher import DataFetcher
from portfolio import Portfolio
from risk_manager import RiskManager
import time
from typing import Dict, Any, Optional, Literal
from enum import Enum


class OrderType(Enum):
    """Order types supported by the executor."""
    MARKET = 'market'
    LIMIT = 'limit'
    LIMIT_IOC = 'limit_ioc'  # Immediate or Cancel


class OrderStatus(Enum):
    """Order execution status."""
    FILLED = 'filled'
    PARTIAL = 'partial'
    PENDING = 'pending'
    CANCELLED = 'cancelled'
    REJECTED = 'rejected'


class Executor:
    """
    Paper/Live trading executor with realistic simulation.

    Phase 32: Enhanced with limit order support for better fill prices
    and reduced slippage in live trading.

    Features:
    - Market orders: Immediate execution with slippage simulation
    - Limit orders: Execute only at specified price or better
    - Limit IOC: Immediate fill or cancel (no partial)
    - Order logging with detailed fill information
    - Callback support for strategy position sync
    """

    # Default slippage simulation (0.15% = 0.05% spread + 0.10% fees)
    DEFAULT_SLIPPAGE_BUY = 0.0015
    DEFAULT_SLIPPAGE_SELL = 0.0015

    # Limit order tolerance (how close market needs to be for fill simulation)
    LIMIT_FILL_TOLERANCE = 0.001  # 0.1% - fill if market within this of limit

    def __init__(self, portfolio: Portfolio):
        self.fetcher = DataFetcher()
        self.portfolio = portfolio
        self.risk = RiskManager()
        self.trade_log = []
        self.pending_orders: Dict[str, Dict[str, Any]] = {}  # order_id -> order details
        self._order_counter = 0

        # Callbacks for strategy position sync
        self.on_fill_callbacks = []

    def register_fill_callback(self, callback):
        """Register a callback to be called when orders are filled."""
        self.on_fill_callbacks.append(callback)

    def _notify_fill(self, fill_info: Dict[str, Any]):
        """Notify all registered callbacks about an order fill."""
        for callback in self.on_fill_callbacks:
            try:
                callback(fill_info)
            except Exception as e:
                print(f"[Executor] Callback error: {e}")

    def _generate_order_id(self) -> str:
        """Generate unique order ID."""
        self._order_counter += 1
        return f"ORD-{int(time.time())}-{self._order_counter:04d}"

    def place_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        leverage: float = 1.0,
        order_type: str = 'limit',
        limit_price: Optional[float] = None,
        strategy_name: str = None,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Place an order with specified type.

        Args:
            symbol: Trading pair (e.g., 'XRP/USDT')
            side: 'buy' or 'sell'
            amount: Amount of base asset to trade
            leverage: Leverage multiplier (1.0 = spot)
            order_type: 'market', 'limit', or 'limit_ioc'
            limit_price: Price for limit orders (required for limit types)
            strategy_name: Name of strategy placing order (for logging)
            metadata: Additional order metadata (grid_level, etc.)

        Returns:
            Dict with order status and fill details
        """
        order_id = self._generate_order_id()

        if order_type == 'market':
            return self._execute_market_order(
                order_id, symbol, side, amount, leverage, strategy_name, metadata
            )
        elif order_type in ['limit', 'limit_ioc']:
            if limit_price is None:
                return {
                    'order_id': order_id,
                    'status': OrderStatus.REJECTED.value,
                    'reason': 'Limit price required for limit orders'
                }
            return self._execute_limit_order(
                order_id, symbol, side, amount, leverage, limit_price,
                order_type == 'limit_ioc', strategy_name, metadata
            )
        else:
            return {
                'order_id': order_id,
                'status': OrderStatus.REJECTED.value,
                'reason': f'Unknown order type: {order_type}'
            }

    def _execute_market_order(
        self,
        order_id: str,
        symbol: str,
        side: str,
        amount: float,
        leverage: float,
        strategy_name: str = None,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Execute a market order with slippage simulation."""
        prices = self.fetcher.get_best_price(symbol)
        if not prices:
            return {
                'order_id': order_id,
                'status': OrderStatus.REJECTED.value,
                'reason': f'No price data for {symbol}'
            }

        avg_price = sum(prices.values()) / len(prices)

        # Simulate slippage
        if side == 'buy':
            execution_price = avg_price * (1 + self.DEFAULT_SLIPPAGE_BUY)
        else:
            execution_price = avg_price * (1 - self.DEFAULT_SLIPPAGE_SELL)

        return self._fill_order(
            order_id, symbol, side, amount, execution_price, leverage,
            'market', strategy_name, metadata
        )

    def _execute_limit_order(
        self,
        order_id: str,
        symbol: str,
        side: str,
        amount: float,
        leverage: float,
        limit_price: float,
        immediate_or_cancel: bool,
        strategy_name: str = None,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Execute a limit order.

        For paper trading: Simulates fill if market price is favorable.
        For live trading: Would submit to exchange API.
        """
        prices = self.fetcher.get_best_price(symbol)
        if not prices:
            return {
                'order_id': order_id,
                'status': OrderStatus.REJECTED.value,
                'reason': f'No price data for {symbol}'
            }

        market_price = sum(prices.values()) / len(prices)

        # Check if limit price is favorable for immediate fill
        can_fill = False
        execution_price = limit_price

        if side == 'buy':
            # Buy limit fills if market <= limit price (+ small tolerance)
            if market_price <= limit_price * (1 + self.LIMIT_FILL_TOLERANCE):
                can_fill = True
                # Fill at better of limit or market
                execution_price = min(limit_price, market_price * (1 + 0.0005))
        else:
            # Sell limit fills if market >= limit price (- small tolerance)
            if market_price >= limit_price * (1 - self.LIMIT_FILL_TOLERANCE):
                can_fill = True
                # Fill at better of limit or market
                execution_price = max(limit_price, market_price * (1 - 0.0005))

        if can_fill:
            result = self._fill_order(
                order_id, symbol, side, amount, execution_price, leverage,
                'limit', strategy_name, metadata
            )
            result['limit_price'] = limit_price
            result['price_improvement'] = abs(execution_price - limit_price)
            return result
        else:
            if immediate_or_cancel:
                return {
                    'order_id': order_id,
                    'status': OrderStatus.CANCELLED.value,
                    'reason': f'IOC order not filled - market {market_price:.4f} vs limit {limit_price:.4f}',
                    'market_price': market_price,
                    'limit_price': limit_price
                }
            else:
                # Store pending order (for live trading, would be on exchange)
                self.pending_orders[order_id] = {
                    'order_id': order_id,
                    'symbol': symbol,
                    'side': side,
                    'amount': amount,
                    'leverage': leverage,
                    'limit_price': limit_price,
                    'strategy': strategy_name,
                    'metadata': metadata,
                    'created_at': time.time()
                }
                print(f"[Executor] PENDING: {side.upper()} {amount:.4f} {symbol} @ ${limit_price:.4f} (limit)")
                return {
                    'order_id': order_id,
                    'status': OrderStatus.PENDING.value,
                    'limit_price': limit_price,
                    'market_price': market_price
                }

    def _fill_order(
        self,
        order_id: str,
        symbol: str,
        side: str,
        amount: float,
        execution_price: float,
        leverage: float,
        order_type: str,
        strategy_name: str = None,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Execute the actual fill and update portfolio."""
        base_asset = symbol.split('/')[0]
        quote_asset = symbol.split('/')[1]
        cost = amount * execution_price / leverage

        success = False

        if side == 'buy':
            if self.portfolio.balances.get(quote_asset, 0) >= cost:
                if leverage > 1:
                    success = self.portfolio.open_margin_position(
                        base_asset, amount, leverage, execution_price, 'long'
                    )
                else:
                    self.portfolio.update(quote_asset, -cost)
                    self.portfolio.update(base_asset, amount)
                    success = True

        elif side == 'sell':
            holding = self.portfolio.balances.get(base_asset, 0)
            if holding >= amount:
                self.portfolio.update(base_asset, -amount)
                self.portfolio.update(quote_asset, amount * execution_price * 0.999)
                success = True

        if success:
            fill_info = {
                'order_id': order_id,
                'status': OrderStatus.FILLED.value,
                'symbol': symbol,
                'action': side,
                'side': side,
                'amount': amount,
                'price': execution_price,
                'leverage': leverage,
                'order_type': order_type,
                'strategy': strategy_name,
                'value': amount * execution_price,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'metadata': metadata or {}
            }

            self._log_trade(symbol, side, amount, execution_price, leverage, order_type, strategy_name)
            self._notify_fill(fill_info)

            print(f"[Executor] FILLED: {side.upper()} {amount:.4f} {base_asset} @ ${execution_price:.4f} "
                  f"({order_type}, lev={leverage}x) [{strategy_name or 'manual'}]")

            return fill_info
        else:
            return {
                'order_id': order_id,
                'status': OrderStatus.REJECTED.value,
                'reason': f'Insufficient balance for {side} order'
            }

    def check_pending_orders(self) -> list:
        """
        Check and fill any pending limit orders that are now fillable.

        Call this periodically to simulate limit order execution.
        """
        filled = []
        to_remove = []

        for order_id, order in self.pending_orders.items():
            prices = self.fetcher.get_best_price(order['symbol'])
            if not prices:
                continue

            market_price = sum(prices.values()) / len(prices)
            side = order['side']
            limit_price = order['limit_price']

            can_fill = False
            if side == 'buy' and market_price <= limit_price:
                can_fill = True
            elif side == 'sell' and market_price >= limit_price:
                can_fill = True

            if can_fill:
                result = self._fill_order(
                    order_id, order['symbol'], side, order['amount'],
                    limit_price, order['leverage'], 'limit',
                    order['strategy'], order['metadata']
                )
                filled.append(result)
                to_remove.append(order_id)

        for order_id in to_remove:
            del self.pending_orders[order_id]

        return filled

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        if order_id in self.pending_orders:
            order = self.pending_orders.pop(order_id)
            print(f"[Executor] CANCELLED: {order['side'].upper()} {order['symbol']} @ ${order['limit_price']:.4f}")
            return True
        return False

    # Legacy method for backwards compatibility
    def place_paper_order(self, symbol: str, side: str, amount: float, leverage: float = 1.0):
        """
        Execute a paper trade with simulated fees and slippage.

        Args:
            symbol: Trading pair (e.g., 'XRP/USDT')
            side: 'buy' or 'sell'
            amount: Amount of base asset to trade
            leverage: Leverage multiplier (1.0 = spot)

        Returns:
            bool: True if order executed successfully
        """
        prices = self.fetcher.get_best_price(symbol)
        if not prices:
            print(f"No price data for {symbol}")
            return False

        avg_price = sum(prices.values()) / len(prices)

        # Simulate fees 0.1%, slippage 0.05%
        if side == 'buy':
            execution_price = avg_price * 1.0015  # Pay more when buying
        else:
            execution_price = avg_price * 0.9985  # Get less when selling

        base_asset = symbol.split('/')[0]
        quote_asset = symbol.split('/')[1]
        cost = amount * execution_price / leverage

        if side == 'buy':
            if self.portfolio.balances.get(quote_asset, 0) >= cost:
                if leverage > 1:
                    success = self.portfolio.open_margin_position(base_asset, amount, leverage, execution_price, 'long')
                else:
                    self.portfolio.update(quote_asset, -cost)
                    self.portfolio.update(base_asset, amount)
                    success = True

                if success:
                    self._log_trade(symbol, side, amount, execution_price, leverage)
                    print(f"EXECUTED: BUY {amount:.4f} {base_asset} @ {execution_price:.4f} (lev {leverage}x)")
                    return True
            else:
                print(f"Insufficient {quote_asset} balance for buy order")

        elif side == 'sell':
            holding = self.portfolio.balances.get(base_asset, 0)
            if holding >= amount:
                self.portfolio.update(base_asset, -amount)
                self.portfolio.update(quote_asset, amount * execution_price * 0.999)  # Fee on proceeds
                self._log_trade(symbol, side, amount, execution_price, leverage)
                print(f"EXECUTED: SELL {amount:.4f} {base_asset} @ {execution_price:.4f}")
                return True
            else:
                print(f"Insufficient {base_asset} balance for sell order")

        return False

    def close_position(self, asset: str, direction: str = 'long'):
        """Close an open margin position"""
        prices = self.fetcher.get_best_price(f"{asset}/USDT")
        if prices:
            avg_price = sum(prices.values()) / len(prices)
            self.portfolio.close_margin_position(asset, avg_price, direction)
            print(f"CLOSED: {direction.upper()} position on {asset} @ {avg_price:.4f}")

    def _log_trade(self, symbol: str, side: str, amount: float, price: float, leverage: float,
                   order_type: str = 'market', strategy: str = None):
        """Log trade for history tracking with enhanced details."""
        self.trade_log.append({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'symbol': symbol,
            'side': side,
            'amount': amount,
            'price': price,
            'leverage': leverage,
            'value': amount * price,
            'order_type': order_type,
            'strategy': strategy
        })

    def get_trade_history(self):
        """Return trade history"""
        return self.trade_log
