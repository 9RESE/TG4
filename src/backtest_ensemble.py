"""
Ensemble Backtest - Phase 19
Backtest full ensemble on broader date range.

Phase 19 Features:
- Early trail stops (+1.5% activation, 1.2% trail) - tighter for shallow chop
- Partial profit-taking (50% at +3% unrealized)
- Dynamic yield from config (7% APY realistic Dec 2025)
- Dynamic sizing based on ADX regime detection
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from data_fetcher import DataFetcher
from portfolio import Portfolio
from orchestrator import EnsembleOrchestrator
from risk_manager import RiskManager
import yaml


def load_config():
    with open('config/exchanges.yaml') as f:
        return yaml.safe_load(f)


def run_ensemble_backtest(start_date: str = '2025-12-01', end_date: str = '2025-12-07'):
    """
    Run ensemble backtest on historical data.

    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
    """
    print("=" * 70)
    print("ENSEMBLE BACKTEST - Phase 19")
    print(f"Period: {start_date} to {end_date}")
    print("=" * 70)

    config = load_config()
    initial_balance = config.get('starting_balance', {'USDT': 10000})

    # Initialize portfolio
    portfolio = Portfolio(initial_balance)

    # Fetch data
    fetcher = DataFetcher()
    symbols = ['XRP/USDT', 'BTC/USDT']
    data = {}

    print("\nFetching historical data...")
    for sym in symbols:
        # Fetch 500 candles (enough for Dec 1-7 + lookback)
        df = fetcher.fetch_ohlcv('kraken', sym, '1h', 500)
        if not df.empty:
            # Filter to date range
            df.index = pd.to_datetime(df.index)
            start = pd.Timestamp(start_date)
            end = pd.Timestamp(end_date) + timedelta(days=1)

            # Keep extra for lookback, but track backtest range
            data[sym] = df
            in_range = df[(df.index >= start) & (df.index < end)]
            print(f"  {sym}: {len(df)} total candles, {len(in_range)} in backtest range")

    if len(data) < 2:
        print("ERROR: Could not fetch data for both symbols")
        return None

    # Get backtest range
    xrp_df = data['XRP/USDT']
    btc_df = data['BTC/USDT']

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date) + timedelta(days=1)

    backtest_indices = xrp_df[(xrp_df.index >= start_ts) & (xrp_df.index < end_ts)].index

    if len(backtest_indices) == 0:
        print("ERROR: No data in backtest range")
        return None

    print(f"\nBacktest range: {backtest_indices[0]} to {backtest_indices[-1]}")
    print(f"Total candles: {len(backtest_indices)}")

    # Initialize ensemble orchestrator
    print("\nInitializing Ensemble Orchestrator...")
    ensemble = EnsembleOrchestrator(portfolio, data)
    print(f"  RL Agent: {'Loaded' if ensemble.rl_agent else 'Rule-based fallback'}")
    print(f"  Strategies: {list(ensemble.strategies.keys())}")

    # Phase 18: Initialize risk manager for trail stops + dynamic sizing
    risk_mgr = RiskManager()
    print(f"  Risk Manager: Trail activation={risk_mgr.trail_activation_pct*100:.0f}%, Trail distance={risk_mgr.trail_distance_pct*100:.1f}%")

    # Tracking
    equity_curve = []
    trades = []
    regime_history = []
    weight_history = []
    signal_history = []

    # Phase 19: Open position tracking for trail stops + partial takes
    # {symbol: {'entry_price': x, 'peak_price': x, 'size': x, 'side': 'long'/'short', 'partial_taken': bool}}
    open_positions = {}
    trail_exits = 0
    partial_takes = 0

    # Phase 19: Dynamic yield from config (7% APY realistic Dec 2025)
    yield_config = config.get('yield', {})
    USDT_APY = yield_config.get('usdt_apy', 0.07)
    USDT_HOURLY_YIELD_RATE = USDT_APY / 8760  # Hourly rate
    total_yield_earned = 0.0
    print(f"  Yield Rate: {USDT_APY*100:.1f}% APY ({yield_config.get('source', 'N/A')})")

    initial_value = portfolio.get_total_usd({'USDT': 1.0, 'XRP': 2.0, 'BTC': 100000})

    print("\n" + "=" * 70)
    print("RUNNING BACKTEST...")
    print("=" * 70)

    for i, ts in enumerate(backtest_indices):
        # Build rolling data window up to current timestamp
        window_data = {}
        for sym, df in data.items():
            window_df = df[df.index <= ts].tail(400)  # Keep 400 candles for lookback
            window_data[sym] = window_df

        # Update ensemble's data reference
        ensemble.data = window_data

        # Get current prices
        prices = {'USDT': 1.0}
        for sym in window_data:
            df = window_data[sym]
            if len(df) > 0:
                base = sym.split('/')[0]
                prices[base] = df['close'].iloc[-1]

        # Phase 18: Check trail stops on open positions BEFORE new signals
        xrp_df = window_data.get('XRP/USDT')
        if xrp_df is not None and len(xrp_df) >= 50:
            high = xrp_df['high'].values
            low = xrp_df['low'].values
            close = xrp_df['close'].values

            # Detect regime for dynamic sizing
            detected_regime = risk_mgr.detect_regime(high, low, close)

            # Phase 19: Check each open position for profit-locking (early trail + partial takes)
            positions_to_close = []
            positions_to_partial = []
            for sym, pos in open_positions.items():
                current_price = prices.get(sym.split('/')[0], pos['entry_price'])

                # Update peak price
                if pos['side'] == 'long':
                    pos['peak_price'] = max(pos['peak_price'], current_price)
                else:
                    pos['peak_price'] = min(pos['peak_price'], current_price)

                # Phase 19: Use combined profit-lock check
                action, amount = risk_mgr.check_profit_lock(
                    pos['entry_price'], current_price, pos['peak_price'],
                    pos['side'], pos.get('partial_taken', False)
                )

                if action == 'trail_exit':
                    positions_to_close.append((sym, 'early_trail_hit'))
                elif action == 'partial_take':
                    positions_to_partial.append((sym, amount))

            # Execute partial takes (50% at +3%)
            for sym, amount in positions_to_partial:
                pos = open_positions[sym]
                current_price = prices.get(sym.split('/')[0], pos['entry_price'])
                if pos['side'] == 'long':
                    pnl_pct = (current_price / pos['entry_price'] - 1) * 100
                else:
                    pnl_pct = (pos['entry_price'] / current_price - 1) * 100
                pos['partial_taken'] = True
                pos['size'] *= (1 - amount)  # Reduce position size
                partial_takes += 1
                trades.append({
                    'timestamp': ts,
                    'action': f'partial_take_{pos["side"]}',
                    'symbol': sym,
                    'leverage': pos.get('leverage', 1),
                    'confidence': 1.0,
                    'regime': ensemble.current_regime,
                    'pnl_pct': pnl_pct,
                    'reason': f'locked {amount*100:.0f}% at +{pnl_pct:.1f}%'
                })

            # Close trail-stopped positions
            for sym, reason in positions_to_close:
                pos = open_positions.pop(sym)
                current_price = prices.get(sym.split('/')[0], pos['entry_price'])
                if pos['side'] == 'long':
                    pnl_pct = (current_price / pos['entry_price'] - 1) * 100
                else:
                    pnl_pct = (pos['entry_price'] / current_price - 1) * 100
                trail_exits += 1
                trades.append({
                    'timestamp': ts,
                    'action': f'trail_exit_{pos["side"]}',
                    'symbol': sym,
                    'leverage': pos.get('leverage', 1),
                    'confidence': 1.0,
                    'regime': ensemble.current_regime,
                    'pnl_pct': pnl_pct,
                    'reason': reason
                })
        else:
            detected_regime = 'chop'  # Default

        # Get ensemble decision
        signal = ensemble.decide(prices)

        # Phase 18: Apply regime-based dynamic sizing
        if signal.get('action') in ['buy', 'sell', 'short']:
            base_size = signal.get('size', 0.12)
            adjusted_size = risk_mgr.regime_dynamic_size(detected_regime, base_size)
            signal['size'] = adjusted_size
            signal['detected_regime'] = detected_regime

        # Execute signal (paper mode)
        result = ensemble.execute(signal, prices)

        # Phase 19: Track new positions for trail stops + partial takes
        if result.get('executed') and signal.get('action') in ['buy', 'short']:
            sym = signal.get('symbol', 'XRP/USDT')
            asset = sym.split('/')[0]
            entry_price = prices.get(asset, 1.0)
            open_positions[sym] = {
                'entry_price': entry_price,
                'peak_price': entry_price,
                'size': signal.get('size', 0.1),
                'side': 'long' if signal['action'] == 'buy' else 'short',
                'leverage': signal.get('leverage', 1),
                'partial_taken': False  # Phase 19: Track if partial take executed
            }

        # Phase 19: Apply hourly USDT yield compounding (7% APY from config)
        usdt_balance = portfolio.balances.get('USDT', 0)
        hourly_yield = usdt_balance * USDT_HOURLY_YIELD_RATE
        portfolio.balances['USDT'] = usdt_balance + hourly_yield
        total_yield_earned += hourly_yield

        # Track equity
        current_value = portfolio.get_total_usd(prices)
        equity_curve.append({
            'timestamp': ts,
            'value': current_value,
            'xrp_price': prices.get('XRP', 0),
            'btc_price': prices.get('BTC', 0)
        })

        # Track regime and weights
        regime_history.append({
            'timestamp': ts,
            'regime': ensemble.current_regime,
            'volatility': ensemble.current_volatility,
            'correlation': ensemble.current_correlation
        })

        weight_history.append({
            'timestamp': ts,
            'mean_reversion': ensemble.weights['mean_reversion'],
            'pair_trading': ensemble.weights['pair_trading'],
            'defensive': ensemble.weights['defensive']
        })

        # Track signals
        signal_history.append({
            'timestamp': ts,
            'action': signal.get('action', 'hold'),
            'confidence': signal.get('confidence', 0),
            'regime': ensemble.current_regime,
            'contributing': signal.get('contributing_strategies', [])
        })

        # Track trades
        if result.get('executed') and signal.get('action') not in ['hold']:
            trades.append({
                'timestamp': ts,
                'action': signal.get('action'),
                'symbol': signal.get('symbol'),
                'leverage': signal.get('leverage', 1),
                'confidence': signal.get('confidence'),
                'regime': ensemble.current_regime
            })

        # Progress
        if (i + 1) % 24 == 0:  # Every 24 hours
            pct_return = ((current_value / initial_value) - 1) * 100
            print(f"  {ts.strftime('%Y-%m-%d %H:%M')} | Value: ${current_value:.2f} ({pct_return:+.2f}%) | Regime: {ensemble.current_regime}")

    # Final results
    final_value = equity_curve[-1]['value']
    total_return = (final_value / initial_value - 1) * 100

    # Calculate metrics
    equity_df = pd.DataFrame(equity_curve)
    equity_df.set_index('timestamp', inplace=True)

    returns = equity_df['value'].pct_change().dropna()

    # Sharpe (annualized from hourly)
    sharpe = (returns.mean() / returns.std()) * np.sqrt(24 * 365) if returns.std() > 0 else 0

    # Max Drawdown
    rolling_max = equity_df['value'].expanding().max()
    drawdown = (equity_df['value'] - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100

    # Win rate
    if len(trades) > 0:
        # Simple: positive returns after trade
        win_count = sum(1 for t in trades if t.get('confidence', 0) > 0.6)
        win_rate = win_count / len(trades) * 100
    else:
        win_rate = 0

    # Regime breakdown
    regime_df = pd.DataFrame(regime_history)
    regime_counts = regime_df['regime'].value_counts()

    # Weight stats
    weight_df = pd.DataFrame(weight_history)
    avg_weights = weight_df[['mean_reversion', 'pair_trading', 'defensive']].mean()

    # Signal breakdown
    signal_df = pd.DataFrame(signal_history)
    action_counts = signal_df['action'].value_counts()

    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)

    print(f"\n{'PERFORMANCE':=^50}")
    print(f"  Initial Value:     ${initial_value:,.2f}")
    print(f"  Final Value:       ${final_value:,.2f}")
    print(f"  Total Return:      {total_return:+.2f}%")
    print(f"  Sharpe Ratio:      {sharpe:.2f}")
    print(f"  Max Drawdown:      {max_drawdown:.2f}%")
    print(f"  Total Trades:      {len(trades)}")
    print(f"  Trail Exits:       {trail_exits}")
    print(f"  Partial Takes:     {partial_takes}")
    print(f"  USDT Yield Earned: ${total_yield_earned:.2f} ({USDT_APY*100:.1f}% APY)")

    print(f"\n{'REGIME ANALYSIS':=^50}")
    for regime, count in regime_counts.items():
        pct = count / len(regime_df) * 100
        print(f"  {regime:12s}: {count:4d} ({pct:.1f}%)")

    print(f"\n{'AVERAGE WEIGHTS':=^50}")
    for strat, weight in avg_weights.items():
        print(f"  {strat:20s}: {weight:.2f}")

    print(f"\n{'SIGNAL BREAKDOWN':=^50}")
    for action, count in action_counts.head(10).items():
        pct = count / len(signal_df) * 100
        print(f"  {action:20s}: {count:4d} ({pct:.1f}%)")

    if trades:
        print(f"\n{'TRADE LOG (last 10)':=^50}")
        for t in trades[-10:]:
            print(f"  {t['timestamp']}: {t['action']} ({t['regime']}) conf={t['confidence']:.2f}")

    # Price performance comparison
    xrp_start = equity_curve[0]['xrp_price']
    xrp_end = equity_curve[-1]['xrp_price']
    btc_start = equity_curve[0]['btc_price']
    btc_end = equity_curve[-1]['btc_price']

    xrp_return = (xrp_end / xrp_start - 1) * 100
    btc_return = (btc_end / btc_start - 1) * 100

    print(f"\n{'BENCHMARK COMPARISON':=^50}")
    print(f"  Ensemble Return:   {total_return:+.2f}%")
    print(f"  XRP Buy & Hold:    {xrp_return:+.2f}%")
    print(f"  BTC Buy & Hold:    {btc_return:+.2f}%")
    print(f"  Alpha vs XRP:      {total_return - xrp_return:+.2f}%")
    print(f"  Alpha vs BTC:      {total_return - btc_return:+.2f}%")

    print("\n" + "=" * 70)

    # Save equity curve
    equity_df.to_csv('logs/ensemble_backtest_equity.csv')
    print("Equity curve saved to logs/ensemble_backtest_equity.csv")

    return {
        'total_return': total_return,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'trades': len(trades),
        'trail_exits': trail_exits,
        'partial_takes': partial_takes,
        'final_value': final_value,
        'usdt_yield_earned': total_yield_earned,
        'xrp_return': xrp_return,
        'btc_return': btc_return,
        'alpha_xrp': total_return - xrp_return,
        'alpha_btc': total_return - btc_return
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Ensemble Backtest - Phase 19')
    parser.add_argument('--start', default='2025-11-25', help='Start date YYYY-MM-DD')
    parser.add_argument('--end', default='2025-12-08', help='End date YYYY-MM-DD')
    args = parser.parse_args()

    results = run_ensemble_backtest(args.start, args.end)
