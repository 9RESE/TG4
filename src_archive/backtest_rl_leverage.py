#!/usr/bin/env python3
"""
Phase 7 RL-Orchestrated Leverage Backtest
Uses trained PPO model with 10x Kraken margin
"""
import numpy as np
import pandas as pd
from data_fetcher import DataFetcher
from portfolio import Portfolio
from orchestrator import RLOrchestrator
from exchanges.kraken_margin import KrakenMargin
from strategies.ripple_momentum_lstm import generate_xrp_signals, generate_btc_signals


def run_rl_backtest():
    print("=" * 70)
    print("TG4 Phase 7: RL-Orchestrated 10x Leverage Backtest")
    print("=" * 70)

    # Initialize
    starting = {'USDT': 1000.0, 'XRP': 500.0, 'BTC': 0.0}
    portfolio = Portfolio(starting.copy())
    fetcher = DataFetcher()

    # Fetch data
    print("\nFetching market data...")
    data = {}
    for sym in ['XRP/USDT', 'BTC/USDT']:
        df = fetcher.fetch_ohlcv('kraken', sym, '1h', 2000)
        if not df.empty:
            data[sym] = df
            print(f"  {sym}: {len(df)} candles")

    if not data:
        print("No data fetched")
        return

    # Initialize RL Orchestrator
    print("\nInitializing RL Orchestrator...")
    orchestrator = RLOrchestrator(portfolio, data)

    if not orchestrator.enabled:
        print("RL model not loaded - train first with --mode train-rl")
        return

    print(f"  Model loaded: {orchestrator.enabled}")
    print(f"  Targets: {orchestrator.get_target_allocation()}")

    # Get price arrays
    xrp_df = data.get('XRP/USDT')
    btc_df = data.get('BTC/USDT')

    if xrp_df is None:
        print("No XRP data")
        return

    # Align dataframes
    min_len = min(len(xrp_df), len(btc_df)) if btc_df is not None else len(xrp_df)
    xrp_df = xrp_df.iloc[-min_len:]
    if btc_df is not None:
        btc_df = btc_df.iloc[-min_len:]

    initial_xrp = xrp_df['close'].iloc[0]
    final_xrp = xrp_df['close'].iloc[-1]
    initial_btc = btc_df['close'].iloc[0] if btc_df is not None else 0
    final_btc = btc_df['close'].iloc[-1] if btc_df is not None else 0

    print(f"\n  XRP: ${initial_xrp:.4f} → ${final_xrp:.4f} ({(final_xrp/initial_xrp - 1)*100:+.1f}%)")
    print(f"  BTC: ${initial_btc:.0f} → ${final_btc:.0f} ({(final_btc/initial_btc - 1)*100:+.1f}%)")

    # Backtest loop
    print("\n" + "=" * 70)
    print("RL DECISIONS & LEVERAGE TRADES")
    print("=" * 70)

    actions_taken = []
    trade_log = []
    margin_pnl_total = 0.0

    # Sample every 4 hours to simulate trading
    step_size = 4
    for i in range(100, min_len, step_size):
        timestamp = xrp_df.index[i]
        xrp_price = xrp_df['close'].iloc[i]
        btc_price = btc_df['close'].iloc[i] if btc_df is not None else 90000.0

        prices = {
            'XRP': xrp_price,
            'BTC': btc_price,
            'USDT': 1.0
        }

        # Update env data window
        if orchestrator.env is not None:
            orchestrator.env.current_step = min(i, orchestrator.env.max_steps - 1)

        # Get RL decision
        result = orchestrator.decide_and_execute(prices)

        # Check for margin position management
        orchestrator.check_and_manage_positions(prices)

        # Log significant actions
        if result.get('executed'):
            action_str = f"{result['asset']} {result['action_type']}"
            if result.get('leverage_used'):
                action_str += f" (10x, ${result.get('collateral', 0):.0f} collateral)"
                trade_log.append({
                    'time': timestamp,
                    'action': action_str,
                    'price': prices.get(result['asset'], 0),
                    'leverage': True
                })
            else:
                if result.get('amount', 0) > 0:
                    action_str += f" ({result.get('amount', 0):.4f})"

            actions_taken.append({
                'time': timestamp,
                'action': action_str,
                'xrp': xrp_price,
                'btc': btc_price
            })

            if len(actions_taken) <= 20:  # Print first 20
                print(f"{timestamp} | {action_str}")

        # Track margin P&L
        if result.get('margin_pnl'):
            margin_pnl_total += result['margin_pnl']
            print(f"  → Margin P&L: ${result['margin_pnl']:+.2f}")

        # Update step counter
        orchestrator.update_env_step()

    # Final portfolio snapshot
    final_prices = {
        'XRP': final_xrp,
        'BTC': final_btc,
        'USDT': 1.0
    }

    # Close remaining margin positions
    remaining_margin = 0.0
    for asset in list(orchestrator.kraken.positions.keys()):
        pnl = orchestrator.kraken.close_position(asset, final_prices.get(asset, 1.0))
        remaining_margin += pnl
        margin_pnl_total += pnl

    print("\n" + "=" * 70)
    print("BACKTEST SUMMARY")
    print("=" * 70)

    # Calculate final values
    final_usdt = portfolio.balances.get('USDT', 0)
    final_xrp_holdings = portfolio.balances.get('XRP', 0)
    final_btc_holdings = portfolio.balances.get('BTC', 0)

    total_value = (
        final_usdt +
        final_xrp_holdings * final_xrp +
        final_btc_holdings * final_btc
    )

    initial_value = (
        starting['USDT'] +
        starting['XRP'] * initial_xrp +
        starting['BTC'] * initial_btc
    )

    pnl = total_value - initial_value
    roi = (pnl / initial_value) * 100

    print(f"\nPortfolio:")
    print(f"  USDT:  ${final_usdt:.2f}")
    print(f"  XRP:   {final_xrp_holdings:.4f} (${final_xrp_holdings * final_xrp:.2f})")
    print(f"  BTC:   {final_btc_holdings:.6f} (${final_btc_holdings * final_btc:.2f})")
    print(f"")
    print(f"Initial Value:   ${initial_value:.2f}")
    print(f"Final Value:     ${total_value:.2f}")
    print(f"Total P&L:       ${pnl:+.2f} ({roi:+.1f}%)")
    print(f"")
    print(f"Margin P&L:      ${margin_pnl_total:+.2f}")
    print(f"Total Actions:   {len(actions_taken)}")
    print(f"Leverage Trades: {len([t for t in trade_log if t.get('leverage')])}")

    # Allocation analysis
    print(f"\nFinal Allocation:")
    targets = orchestrator.get_target_allocation()
    for asset in ['BTC', 'XRP', 'USDT']:
        if asset == 'USDT':
            value = final_usdt
        elif asset == 'XRP':
            value = final_xrp_holdings * final_xrp
        else:
            value = final_btc_holdings * final_btc

        pct = (value / total_value) * 100 if total_value > 0 else 0
        target = targets.get(asset, 0) * 100
        print(f"  {asset}: {pct:.1f}% (target: {target:.0f}%)")

    alignment = orchestrator.get_alignment_score(final_prices)
    print(f"\nAlignment Score: {alignment:.2f}/1.0")

    print("=" * 70)

    return {
        'pnl': pnl,
        'roi': roi,
        'margin_pnl': margin_pnl_total,
        'actions': len(actions_taken)
    }


if __name__ == "__main__":
    run_rl_backtest()
