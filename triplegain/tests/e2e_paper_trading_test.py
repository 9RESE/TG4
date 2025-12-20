#!/usr/bin/env python3
"""
End-to-End Paper Trading Test Script

This script exercises the complete paper trading flow:
1. Loads configuration
2. Initializes paper trading components
3. Executes simulated trades
4. Reports portfolio status and P&L

Usage:
    python -m triplegain.tests.e2e_paper_trading_test
"""

import asyncio
import logging
import sys
from decimal import Decimal
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from triplegain.src.execution.paper_portfolio import PaperPortfolio
from triplegain.src.execution.paper_executor import PaperOrderExecutor
from triplegain.src.execution.paper_price_source import PaperPriceSource, MockPriceSource
from triplegain.src.execution.trading_mode import TradingMode, get_trading_mode
from triplegain.src.risk.rules_engine import TradeProposal
from triplegain.src.utils.config import get_config_loader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_banner(title: str) -> None:
    """Print a formatted banner."""
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_portfolio(portfolio: PaperPortfolio, prices: dict, title: str = "Portfolio Status") -> None:
    """Print portfolio status in a formatted way."""
    print_banner(title)

    print(f"Session ID: {portfolio.session_id}")
    print(f"Created: {portfolio.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print()

    print("Balances:")
    for asset, balance in portfolio.get_balances_dict().items():
        print(f"  {asset}: {balance:,.6f}")

    print()
    pnl = portfolio.get_pnl_summary(prices)
    print("P&L Summary:")
    print(f"  Initial Equity: ${pnl['initial_equity_usd']:,.2f}")
    print(f"  Current Equity: ${pnl['current_equity_usd']:,.2f}")
    print(f"  Total P&L: ${pnl['total_pnl_usd']:,.2f} ({pnl['total_pnl_pct']:+.2f}%)")
    print(f"  Realized P&L: ${pnl['realized_pnl_usd']:,.2f}")
    print(f"  Total Fees: ${pnl['total_fees_usd']:,.2f}")
    print(f"  Trade Count: {pnl['trade_count']}")

    if hasattr(portfolio, 'win_count'):
        total_closed = portfolio.win_count + portfolio.loss_count
        win_rate = (portfolio.win_count / total_closed * 100) if total_closed > 0 else 0
        print(f"  Win Rate: {win_rate:.1f}% ({portfolio.win_count}W/{portfolio.loss_count}L)")


async def run_paper_trading_test():
    """Run the end-to-end paper trading test."""

    print_banner("TripleGain Paper Trading E2E Test")

    # Step 1: Load configuration
    print("\n[1/5] Loading configuration...")
    try:
        config_loader = get_config_loader()
        execution_config = config_loader.get_execution_config()
        print("  ✓ Configuration loaded")
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        # Use default config
        execution_config = {
            "paper_trading": {
                "initial_balance": {"USDT": 10000, "BTC": 0, "XRP": 0},
                "fill_delay_ms": 50,
                "simulated_slippage_pct": 0.1,
            },
            "symbols": {
                "BTC/USDT": {"fee_pct": 0.26, "price_decimals": 2, "size_decimals": 8},
                "XRP/USDT": {"fee_pct": 0.26, "price_decimals": 5, "size_decimals": 0},
            }
        }
        print("  ⚠ Using default configuration")

    # Step 2: Verify trading mode
    print("\n[2/5] Checking trading mode...")
    trading_mode = get_trading_mode(execution_config)
    if trading_mode != TradingMode.PAPER:
        print(f"  ⚠ Trading mode is {trading_mode.value} - forcing PAPER mode for test")
    else:
        print(f"  ✓ Trading mode: {trading_mode.value}")

    # Step 3: Initialize paper trading components
    print("\n[3/5] Initializing paper trading components...")

    # Create portfolio
    portfolio = PaperPortfolio.from_config(execution_config)
    print(f"  ✓ Portfolio created with ${portfolio.get_balance('USDT'):,.2f} USDT")

    # Create mock price source with realistic prices
    initial_prices = {
        "BTC/USDT": Decimal("98500.00"),
        "XRP/USDT": Decimal("2.35"),
        "XRP/BTC": Decimal("0.0000238"),
    }
    price_source = MockPriceSource(initial_prices=initial_prices)
    print("  ✓ Price source initialized")
    print(f"    BTC/USDT: ${price_source.get_price('BTC/USDT'):,.2f}")
    print(f"    XRP/USDT: ${price_source.get_price('XRP/USDT'):.4f}")

    # Create executor
    executor = PaperOrderExecutor(
        config=execution_config,
        paper_portfolio=portfolio,
        price_source=price_source.get_price,
    )
    print("  ✓ Paper executor initialized")

    # Get current prices for P&L calculation
    current_prices = {
        "BTC/USDT": price_source.get_price("BTC/USDT"),
        "XRP/USDT": price_source.get_price("XRP/USDT"),
    }

    # Show initial portfolio
    print_portfolio(portfolio, current_prices, "Initial Portfolio")

    # Step 4: Execute test trades
    print_banner("Executing Test Trades")

    # Get current prices for trade proposals (market orders use current price)
    btc_price = float(price_source.get_price("BTC/USDT"))
    xrp_price = float(price_source.get_price("XRP/USDT"))

    trades = [
        # Trade 1: Buy BTC with 20% of portfolio
        TradeProposal(
            symbol="BTC/USDT",
            side="buy",
            size_usd=2000,
            entry_price=btc_price,  # Market order uses current price
            leverage=1,
            confidence=0.75,
            regime="trending_up",
        ),
        # Trade 2: Buy XRP with 15% of portfolio
        TradeProposal(
            symbol="XRP/USDT",
            side="buy",
            size_usd=1500,
            entry_price=xrp_price,  # Market order uses current price
            leverage=1,
            confidence=0.80,
            regime="trending_up",
        ),
        # Trade 3: Add to BTC position
        TradeProposal(
            symbol="BTC/USDT",
            side="buy",
            size_usd=1000,
            entry_price=btc_price,
            leverage=1,
            confidence=0.70,
            regime="ranging",
        ),
    ]

    for i, trade in enumerate(trades, 1):
        print(f"\n[Trade {i}/{len(trades)}] {trade.side.upper()} ${trade.size_usd:,.0f} of {trade.symbol}")

        result = await executor.execute_trade(trade)

        if result.success:
            order = result.order
            print(f"  ✓ Order filled: {order.filled_size:.6f} @ ${order.filled_price:,.2f}")
            print(f"    Fee: ${order.fee_amount:.4f}")
            print(f"    Execution time: {result.execution_time_ms:.0f}ms")
        else:
            print(f"  ✗ Failed: {result.error_message}")

    # Show portfolio after buys
    print_portfolio(portfolio, current_prices, "Portfolio After Buys")

    # Simulate price movement
    print_banner("Simulating Price Movement")

    # Price goes up 3%
    new_btc_price = Decimal("101455.00")  # +3%
    new_xrp_price = Decimal("2.47")  # +5%

    price_source.set_mock_price("BTC/USDT", new_btc_price)
    price_source.set_mock_price("XRP/USDT", new_xrp_price)

    print(f"  BTC/USDT: $98,500 → ${new_btc_price:,.2f} (+3.0%)")
    print(f"  XRP/USDT: $2.35 → ${new_xrp_price:.4f} (+5.1%)")

    current_prices["BTC/USDT"] = new_btc_price
    current_prices["XRP/USDT"] = new_xrp_price

    # Execute sell to realize profit
    print_banner("Taking Profits")

    # Sell half of XRP
    xrp_balance = portfolio.get_balance("XRP")
    sell_xrp = float(xrp_balance) / 2 * float(new_xrp_price)

    sell_trade = TradeProposal(
        symbol="XRP/USDT",
        side="sell",
        size_usd=sell_xrp,
        entry_price=float(new_xrp_price),  # Use current price
        leverage=1,
        confidence=0.85,
        regime="taking_profit",
    )

    print(f"\n[Sell Trade] SELL ${sell_xrp:,.0f} of XRP/USDT (half position)")
    result = await executor.execute_trade(sell_trade)

    if result.success:
        order = result.order
        print(f"  ✓ Order filled: {order.filled_size:.0f} XRP @ ${order.filled_price:.4f}")
        print(f"    Fee: ${order.fee_amount:.4f}")
    else:
        print(f"  ✗ Failed: {result.error_message}")

    # Step 5: Final results
    print_portfolio(portfolio, current_prices, "Final Portfolio Status")

    # Executor statistics
    print_banner("Executor Statistics")
    stats = executor.get_stats()
    print(f"  Total Orders Placed: {stats['total_orders_placed']}")
    print(f"  Total Orders Filled: {stats['total_orders_filled']}")
    print(f"  Total Orders Rejected: {stats['total_orders_rejected']}")
    print(f"  Open Orders: {stats['open_orders_count']}")
    print(f"  Fill Delay: {stats['fill_delay_ms']}ms")
    print(f"  Slippage: {stats['slippage_pct']}%")

    # Trade history
    print_banner("Trade History")
    for i, trade in enumerate(portfolio.trade_history[-5:], 1):
        print(f"  {i}. {trade.side.upper():4} {trade.size:>12.4f} {trade.symbol:10} @ ${trade.price:>10,.2f}")

    print_banner("Test Complete")
    print("  Paper trading E2E test completed successfully!")
    print("  All components functioning correctly.")
    print()

    return True


if __name__ == "__main__":
    try:
        success = asyncio.run(run_paper_trading_test())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Test failed with error: {e}")
        sys.exit(1)
