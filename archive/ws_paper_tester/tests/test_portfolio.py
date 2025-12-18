"""Tests for portfolio management."""

import pytest
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ws_tester.portfolio import StrategyPortfolio, PortfolioManager, STARTING_CAPITAL
from ws_tester.types import Fill


class TestStrategyPortfolio:
    def test_portfolio_creation(self):
        portfolio = StrategyPortfolio(
            strategy_name='test',
            starting_capital=100.0
        )
        assert portfolio.usdt == 100.0
        assert portfolio.strategy_name == 'test'

    def test_get_equity_no_positions(self):
        portfolio = StrategyPortfolio(
            strategy_name='test',
            starting_capital=100.0
        )
        equity = portfolio.get_equity({'XRP/USD': 2.35})
        assert equity == 100.0

    def test_get_equity_with_position(self):
        portfolio = StrategyPortfolio(
            strategy_name='test',
            starting_capital=100.0,
            usdt=50.0
        )
        portfolio.assets['XRP'] = 20.0  # 20 XRP

        prices = {'XRP/USD': 2.50}
        equity = portfolio.get_equity(prices)

        # 50 USDT + 20 XRP * 2.50 = 50 + 50 = 100
        assert equity == 100.0

    def test_win_rate_calculation(self):
        portfolio = StrategyPortfolio(
            strategy_name='test',
            total_trades=10,
            winning_trades=6
        )
        assert portfolio.get_win_rate() == 60.0

    def test_win_rate_no_trades(self):
        portfolio = StrategyPortfolio(strategy_name='test')
        assert portfolio.get_win_rate() == 0.0

    def test_roi_calculation(self):
        portfolio = StrategyPortfolio(
            strategy_name='test',
            starting_capital=100.0,
            usdt=110.0
        )
        roi = portfolio.get_roi({})
        assert roi == 10.0  # 10% return

    def test_drawdown_tracking(self):
        portfolio = StrategyPortfolio(
            strategy_name='test',
            starting_capital=100.0,
            usdt=100.0,
            peak_equity=100.0
        )

        # Simulate equity going up
        portfolio.usdt = 120.0
        portfolio.update_drawdown({})
        assert portfolio.peak_equity == 120.0

        # Simulate equity going down
        portfolio.usdt = 96.0
        portfolio.update_drawdown({})
        # Drawdown = (120 - 96) / 120 = 0.2 = 20%
        assert portfolio.max_drawdown == 0.2

    def test_to_dict(self):
        portfolio = StrategyPortfolio(
            strategy_name='test',
            starting_capital=100.0,
            usdt=95.0,
            total_trades=5,
            winning_trades=3,
            total_pnl=-5.0
        )
        data = portfolio.to_dict({})

        assert data['strategy'] == 'test'
        assert data['usdt'] == 95.0
        assert data['trades'] == 5
        assert data['win_rate'] == 60.0
        assert data['pnl'] == -5.0


class TestPortfolioManager:
    def test_manager_creation(self):
        pm = PortfolioManager(['strategy_a', 'strategy_b'])
        assert len(pm.portfolios) == 2
        assert 'strategy_a' in pm.portfolios
        assert 'strategy_b' in pm.portfolios

    def test_add_strategy(self):
        pm = PortfolioManager(['strategy_a'])
        pm.add_strategy('strategy_b')

        assert len(pm.portfolios) == 2
        assert pm.portfolios['strategy_b'].usdt == STARTING_CAPITAL

    def test_get_portfolio(self):
        pm = PortfolioManager(['test'])
        portfolio = pm.get_portfolio('test')

        assert portfolio is not None
        assert portfolio.strategy_name == 'test'

    def test_get_nonexistent_portfolio(self):
        pm = PortfolioManager(['test'])
        portfolio = pm.get_portfolio('nonexistent')

        assert portfolio is None

    def test_aggregate_stats(self):
        pm = PortfolioManager(['a', 'b', 'c'], starting_capital=100.0)

        # Simulate some P&L
        pm.portfolios['a'].total_pnl = 10.0
        pm.portfolios['a'].total_trades = 5
        pm.portfolios['a'].winning_trades = 3

        pm.portfolios['b'].total_pnl = -5.0
        pm.portfolios['b'].total_trades = 3
        pm.portfolios['b'].winning_trades = 1

        pm.portfolios['c'].total_pnl = 0.0
        pm.portfolios['c'].total_trades = 2
        pm.portfolios['c'].winning_trades = 1

        agg = pm.get_aggregate({})

        assert agg['total_strategies'] == 3
        assert agg['total_capital'] == 300.0
        assert agg['total_pnl'] == 5.0  # 10 - 5 + 0
        assert agg['total_trades'] == 10  # 5 + 3 + 2
        assert agg['win_rate'] == 50.0  # 5/10

    def test_leaderboard(self):
        pm = PortfolioManager(['a', 'b', 'c'])

        pm.portfolios['a'].total_pnl = 10.0
        pm.portfolios['b'].total_pnl = -5.0
        pm.portfolios['c'].total_pnl = 15.0

        leaderboard = pm.get_leaderboard({})

        assert len(leaderboard) == 3
        assert leaderboard[0]['strategy'] == 'c'  # Highest P&L
        assert leaderboard[1]['strategy'] == 'a'
        assert leaderboard[2]['strategy'] == 'b'  # Lowest P&L

    def test_reset_strategy(self):
        pm = PortfolioManager(['test'])
        pm.portfolios['test'].usdt = 50.0
        pm.portfolios['test'].total_pnl = -50.0

        pm.reset_strategy('test')

        assert pm.portfolios['test'].usdt == STARTING_CAPITAL
        assert pm.portfolios['test'].total_pnl == 0.0

    def test_get_all_fills(self):
        pm = PortfolioManager(['a', 'b'])

        # Add some fills
        fill_a = Fill(
            fill_id='fill1',
            timestamp=datetime.now(),
            symbol='XRP/USD',
            side='buy',
            size=42.0,
            price=2.35,
            fee=0.1,
            signal_reason='Test',
            strategy='a'
        )
        fill_b = Fill(
            fill_id='fill2',
            timestamp=datetime.now(),
            symbol='BTC/USD',
            side='sell',
            size=0.001,
            price=104500.0,
            fee=0.1,
            signal_reason='Test',
            strategy='b'
        )

        pm.portfolios['a'].fills.append(fill_a)
        pm.portfolios['b'].fills.append(fill_b)

        all_fills = pm.get_all_fills()

        assert len(all_fills) == 2
