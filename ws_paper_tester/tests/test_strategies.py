"""Tests for strategy loader and strategies."""

import pytest
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ws_tester.types import DataSnapshot, OrderbookSnapshot, Candle, Trade
from ws_tester.strategy_loader import discover_strategies, StrategyWrapper, get_all_symbols


def create_rich_snapshot(price: float = 2.35) -> DataSnapshot:
    """Create a test snapshot with candles, trades, and orderbook."""
    now = datetime.now()

    # Create candles
    candles = []
    for i in range(25):
        offset = (25 - i) * 0.001
        candles.append(Candle(
            timestamp=now,
            open=price - offset,
            high=price - offset + 0.005,
            low=price - offset - 0.005,
            close=price - offset + 0.002,
            volume=100.0
        ))

    # Create trades
    trades = []
    for i in range(60):
        trades.append(Trade(
            timestamp=now,
            price=price + (i % 2 - 0.5) * 0.002,
            size=50.0 + i,
            side='buy' if i % 3 != 0 else 'sell'
        ))

    # Create orderbook
    ob = OrderbookSnapshot(
        bids=tuple((price - 0.001 - i*0.001, 100.0 + i*10) for i in range(10)),
        asks=tuple((price + 0.001 + i*0.001, 100.0 + i*10) for i in range(10))
    )

    return DataSnapshot(
        timestamp=now,
        prices={'XRP/USDT': price, 'BTC/USDT': 104500.0},
        candles_1m={'XRP/USDT': tuple(candles)},
        candles_5m={'XRP/USDT': tuple(candles)},
        orderbooks={'XRP/USDT': ob, 'BTC/USDT': ob},
        trades={'XRP/USDT': tuple(trades), 'BTC/USDT': tuple(trades)}
    )


class TestStrategyLoader:
    def test_discover_strategies(self):
        strategies_path = Path(__file__).parent.parent / "strategies"
        strategies = discover_strategies(str(strategies_path))

        assert len(strategies) >= 3  # market_making, order_flow, mean_reversion
        assert 'market_making' in strategies
        assert 'order_flow' in strategies
        assert 'mean_reversion' in strategies

    def test_strategy_wrapper(self):
        strategies_path = Path(__file__).parent.parent / "strategies"
        strategies = discover_strategies(str(strategies_path))

        mm = strategies.get('market_making')
        assert mm is not None
        assert isinstance(mm, StrategyWrapper)
        assert mm.name == 'market_making'
        assert mm.version.startswith('1.')  # Allow minor/patch version bumps
        assert len(mm.symbols) > 0

    def test_get_all_symbols(self):
        strategies_path = Path(__file__).parent.parent / "strategies"
        strategies = discover_strategies(str(strategies_path))
        symbols = get_all_symbols(strategies)

        assert 'XRP/USDT' in symbols
        assert len(symbols) >= 1


class TestMarketMakingStrategy:
    def test_generate_signal(self):
        strategies_path = Path(__file__).parent.parent / "strategies"
        strategies = discover_strategies(str(strategies_path))
        mm = strategies.get('market_making')

        assert mm is not None

        snapshot = create_rich_snapshot(2.35)
        mm.on_start()

        # May or may not generate signal depending on conditions
        signal = mm.generate_signal(snapshot)

        # Check state was updated
        assert 'indicators' in mm.state

    def test_on_fill_updates_inventory(self):
        strategies_path = Path(__file__).parent.parent / "strategies"
        strategies = discover_strategies(str(strategies_path))
        mm = strategies.get('market_making')

        mm.on_start()
        assert mm.state.get('inventory', 0) == 0

        # Simulate a fill
        fill = {
            'side': 'buy',
            'size': 42.5,
            'price': 2.35
        }
        mm.on_fill(fill)

        # Inventory should increase
        assert mm.state.get('inventory', 0) > 0


class TestOrderFlowStrategy:
    def test_generate_signal(self):
        strategies_path = Path(__file__).parent.parent / "strategies"
        strategies = discover_strategies(str(strategies_path))
        of = strategies.get('order_flow')

        assert of is not None

        snapshot = create_rich_snapshot(2.35)
        of.on_start()

        signal = of.generate_signal(snapshot)

        # Check state was updated
        assert 'indicators' in of.state or of.state.get('total_trades_seen', 0) > 0


class TestMeanReversionStrategy:
    def test_generate_signal(self):
        strategies_path = Path(__file__).parent.parent / "strategies"
        strategies = discover_strategies(str(strategies_path))
        mr = strategies.get('mean_reversion')

        assert mr is not None

        snapshot = create_rich_snapshot(2.35)
        mr.on_start()

        signal = mr.generate_signal(snapshot)

        # Check indicators were calculated
        assert 'indicators' in mr.state
        indicators = mr.state['indicators']
        assert 'sma' in indicators
        assert 'rsi' in indicators


class TestStrategyValidation:
    def test_all_strategies_have_required_attrs(self):
        strategies_path = Path(__file__).parent.parent / "strategies"
        strategies = discover_strategies(str(strategies_path))

        for name, strategy in strategies.items():
            assert strategy.name, f"{name} missing name"
            assert strategy.version, f"{name} missing version"
            assert callable(strategy._generate_signal), f"{name} generate_signal not callable"
            assert len(strategy.symbols) > 0, f"{name} has no symbols"

    def test_strategies_dont_crash_on_empty_data(self):
        strategies_path = Path(__file__).parent.parent / "strategies"
        strategies = discover_strategies(str(strategies_path))

        # Empty snapshot
        snapshot = DataSnapshot(
            timestamp=datetime.now(),
            prices={},
            candles_1m={},
            candles_5m={},
            orderbooks={},
            trades={}
        )

        for name, strategy in strategies.items():
            strategy.on_start()
            try:
                signal = strategy.generate_signal(snapshot)
                # Should return None, not crash
                assert signal is None or hasattr(signal, 'action')
            except Exception as e:
                pytest.fail(f"{name} crashed on empty data: {e}")
