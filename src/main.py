import argparse
import warnings
import os

# Suppress ROCm/HIP warnings for cleaner output
warnings.filterwarnings('ignore', message='.*expandable_segments.*')
warnings.filterwarnings('ignore', message='.*hipBLASLt.*')
warnings.filterwarnings('ignore', category=UserWarning, module='torch')

from data_fetcher import DataFetcher
from portfolio import Portfolio
from backtester import Backtester
from strategies.ripple_momentum_lstm import generate_ripple_signals, generate_xrp_signals, generate_btc_signals
from strategies.rebalancer import rebalance
from risk_manager import RiskManager
from executor import Executor
from orchestrator import RLOrchestrator
import yaml
import time


def load_config():
    with open('config/exchanges.yaml') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="TG4 Local AI Crypto Trader - Phase 16")
    parser.add_argument('--mode', choices=['fetch', 'backtest', 'paper', 'train-rl', 'train-ensemble', 'paper-ensemble'], default='fetch')
    parser.add_argument('--timesteps', type=int, default=100000, help='RL training timesteps')
    parser.add_argument('--device', type=str, default='cuda', help='Training device (cuda/cpu)')
    args = parser.parse_args()

    config = load_config()
    portfolio = Portfolio(config['starting_balance'])
    fetcher = DataFetcher()

    print("=" * 60)
    print("TG4 Platform - Phase 16: RL Ensemble Orchestrator")
    print("=" * 60)
    print(f"Targets: BTC {config['targets']['BTC']*100:.0f}% | XRP {config['targets']['XRP']*100:.0f}% | USDT {config['targets']['USDT']*100:.0f}%")
    print(portfolio)

    if args.mode == 'fetch':
        # Fetch latest prices for core pairs
        symbols = ['BTC/USDT', 'XRP/USDT']
        print("\nFetching current prices...")
        for symbol in symbols:
            prices = fetcher.get_best_price(symbol)
            if prices:
                print(f"  {symbol}: {prices}")

    elif args.mode == 'backtest':
        # Phase 7: USDT-based backtest
        symbols = ['XRP/USDT', 'BTC/USDT']
        data = {}
        for sym in symbols:
            print(f"Fetching {sym}...")
            df = fetcher.fetch_ohlcv('kraken', sym, '1h', 2000)
            if not df.empty:
                data[sym] = df

        def print_backtest_results(pf, symbol):
            print(f"\n{'='*50}")
            print(f"BACKTEST RESULTS: {symbol}")
            print(f"{'='*50}")
            print(f"Total Return:    {pf.total_return():.2%}")
            print(f"Sharpe Ratio:    {pf.sharpe_ratio():.2f}")
            print(f"Max Drawdown:    {pf.max_drawdown():.2%}")
            print(f"Win Rate:        {pf.trades.win_rate():.2%}" if pf.trades.count() > 0 else "Win Rate:        N/A")
            print(f"Total Trades:    {pf.trades.count()}")
            print(f"Final Value:     ${pf.value().iloc[-1]:.2f}")
            print(f"{'='*50}\n")

        if 'XRP/USDT' in data:
            signals = generate_ripple_signals(data, 'XRP/USDT')
            bt = Backtester(data)
            pf = bt.run_with_lstm_signals('XRP/USDT', signals)
            print_backtest_results(pf, 'XRP/USDT')

    elif args.mode == 'train-rl':
        from models.rl_agent import train_rl_agent

        print("\n[RL TRAINING MODE - Phase 7]")
        print(f"Training for {args.timesteps} timesteps on {args.device.upper()}...")
        print(f"Target allocation: BTC 45% | XRP 35% | USDT 20%")

        # Fetch data for RL training
        symbols = ['XRP/USDT', 'BTC/USDT']
        data = {}
        for sym in symbols:
            print(f"Fetching {sym}...")
            df = fetcher.fetch_ohlcv('kraken', sym, '1h', 2000)
            if not df.empty:
                data[sym] = df

        if data:
            model = train_rl_agent(data, timesteps=args.timesteps, device=args.device)
            print("\nRL training complete!")
            print("Model saved to models/rl_ppo_agent.zip")
        else:
            print("No data available for training")

    elif args.mode == 'train-ensemble':
        from models.ensemble_env import train_ensemble_agent

        print("\n[ENSEMBLE RL TRAINING MODE - Phase 16]")
        print(f"Training ensemble agent for {args.timesteps} timesteps on {args.device.upper()}...")
        print(f"Strategies: Mean Reversion VWAP + XRP/BTC Pair Trading + Defensive Yield")

        # Fetch data for ensemble training
        symbols = ['XRP/USDT', 'BTC/USDT']
        data = {}
        for sym in symbols:
            print(f"Fetching {sym}...")
            df = fetcher.fetch_ohlcv('kraken', sym, '1h', 2000)
            if not df.empty:
                data[sym] = df

        if data:
            model = train_ensemble_agent(data, timesteps=args.timesteps, device=args.device)
            print("\nEnsemble RL training complete!")
            print("Model saved to models/rl_ensemble_agent.zip")
        else:
            print("No data available for training")

    elif args.mode == 'paper-ensemble':
        from orchestrator import EnsembleOrchestrator

        print("\n[ENSEMBLE PAPER TRADING MODE - Phase 16]")
        print("Dynamic strategy weighting: Mean Reversion + Pair Trading + Defensive")
        print("Press Ctrl+C to stop\n")

        # Fetch initial data
        print("Fetching initial market data...")
        symbols = ['XRP/USDT', 'BTC/USDT']
        data = {}
        for sym in symbols:
            df = fetcher.fetch_ohlcv('kraken', sym, '1h', 500)
            if not df.empty:
                data[sym] = df
                print(f"  {sym}: {len(df)} candles")

        # Initialize Ensemble Orchestrator
        ensemble = EnsembleOrchestrator(portfolio, data)
        print(f"\n[ENSEMBLE ORCHESTRATOR] Initialized")
        print(f"  RL agent loaded: {ensemble.rl_agent is not None}")
        print(f"  Current weights: {ensemble.weights}")

        loop_count = 0
        last_data_refresh = time.time()

        while True:
            try:
                loop_count += 1
                print(f"\n{'='*60}")
                print(f"Loop {loop_count} @ {time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"{'='*60}")

                # Fetch current prices
                prices = {'USDT': 1.0}
                for sym in ['XRP/USDT', 'BTC/USDT']:
                    p = fetcher.get_best_price(sym)
                    if p:
                        base = sym.split('/')[0]
                        prices[base] = list(p.values())[0]
                        print(f"{sym}: ${prices[base]:.4f}")

                # Record portfolio snapshot
                portfolio.record_snapshot(prices)
                total_value = portfolio.get_total_usd(prices)
                print(f"\nPortfolio Value: ${total_value:.2f}")

                # Get ensemble status
                status = ensemble.get_status()
                print(f"\n[ENSEMBLE STATUS]")
                print(f"  Regime: {status['regime']}")
                print(f"  Weights: MR={status['weights']['mean_reversion']:.2f}, PT={status['weights']['pair_trading']:.2f}, DEF={status['weights']['defensive']:.2f}")
                print(f"  Volatility: {status['volatility']*100:.2f}%")
                print(f"  Correlation: {status['correlation']:.2f}")

                # Refresh data every 30 minutes
                if time.time() - last_data_refresh > 1800:
                    print("\n[Refreshing market data...]")
                    for sym in symbols:
                        df = fetcher.fetch_ohlcv('kraken', sym, '1h', 500)
                        if not df.empty:
                            data[sym] = df
                            ensemble.data = data
                    last_data_refresh = time.time()

                # Get ensemble decision
                signal = ensemble.decide(prices)
                print(f"\n[ENSEMBLE DECISION]")
                print(f"  Action: {signal['action']}")
                print(f"  Confidence: {signal['confidence']:.2f}")
                print(f"  Contributing: {signal.get('contributing_strategies', [])}")
                print(f"  Reason: {signal['reason']}")

                # Execute signal
                result = ensemble.execute(signal, prices)
                if result['executed']:
                    print(f"  EXECUTED: {result}")

                time.sleep(300)  # 5min loop

            except KeyboardInterrupt:
                print("\n\n" + "=" * 60)
                print("Stopping ensemble paper trading...")
                print("=" * 60)

                final_status = ensemble.get_status()
                print(f"\nFinal Stats:")
                print(f"  Total signals: {final_status['total_signals']}")
                print(f"  Executed signals: {final_status['executed_signals']}")
                print(f"  Final weights: {final_status['weights']}")

                break

    elif args.mode == 'paper':
        print("\n[PAPER TRADING MODE - Phase 7]")
        print("10x Kraken margin + 3x Bitrue ETFs enabled")
        print("Press Ctrl+C to stop\n")

        # Initialize execution engine
        executor = Executor(portfolio)

        # Fetch initial data
        print("Fetching initial market data...")
        symbols = ['XRP/USDT', 'BTC/USDT']
        data = {}
        for sym in symbols:
            df = fetcher.fetch_ohlcv('kraken', sym, '1h', 500)
            if not df.empty:
                data[sym] = df
                print(f"  {sym}: {len(df)} candles")

        # Initialize RL Orchestrator with leverage support
        orchestrator = RLOrchestrator(portfolio, data)
        rl_enabled = orchestrator.enabled

        if rl_enabled:
            print("\n[RL ORCHESTRATOR] Enabled - using trained PPO model")
            print(f"  Target allocation: {orchestrator.get_target_allocation()}")
            print(f"  Kraken margin: 10x max")
            print(f"  Risk per trade: 20% of USDT")
        else:
            print("\n[RL ORCHESTRATOR] Disabled - no trained model found")
            print("  Run 'python main.py --mode train-rl --timesteps 500000' to train")

        loop_count = 0
        last_data_refresh = time.time()

        while True:
            try:
                loop_count += 1
                print(f"\n{'='*60}")
                print(f"Loop {loop_count} @ {time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"{'='*60}")

                # Fetch current prices
                prices = {'USDT': 1.0}
                for sym in ['XRP/USDT', 'BTC/USDT']:
                    p = fetcher.get_best_price(sym)
                    if p:
                        base = sym.split('/')[0]
                        prices[base] = list(p.values())[0]
                        print(f"{sym}: ${prices[base]:.4f}")

                # Record portfolio snapshot
                portfolio.record_snapshot(prices)
                total_value = portfolio.get_total_usd(prices)
                print(f"\nPortfolio Value: ${total_value:.2f}")
                print(f"Spot Holdings: {portfolio}")

                # Show margin positions
                if orchestrator.kraken.positions:
                    print("\nMargin Positions:")
                    for asset, pos in orchestrator.kraken.positions.items():
                        unrealized = orchestrator.kraken.get_unrealized_pnl(asset, prices.get(asset, pos['entry']))
                        pnl_pct = (unrealized / pos['collateral']) * 100 if pos['collateral'] > 0 else 0
                        print(f"  {asset}: {pos['size']:.4f} @ ${pos['entry']:.2f} ({pos['leverage']}x) | P&L: ${unrealized:.2f} ({pnl_pct:+.1f}%)")

                # Show allocation vs targets
                if rl_enabled:
                    current_alloc = orchestrator.get_current_allocation(prices)
                    targets = orchestrator.get_target_allocation()
                    alignment = orchestrator.get_alignment_score(prices)
                    print(f"\nAllocation (target in brackets):")
                    for asset in ['BTC', 'XRP', 'USDT']:
                        curr = current_alloc.get(asset, 0) * 100
                        tgt = targets.get(asset, 0) * 100
                        status = "OK" if abs(curr - tgt) < 5 else "LOW" if curr < tgt else "HIGH"
                        print(f"  {asset}: {curr:.1f}% [{tgt:.0f}%] {status}")
                    print(f"  Alignment Score: {alignment:.2f}/1.0")

                # Refresh data every 30 minutes
                if time.time() - last_data_refresh > 1800:
                    print("\n[Refreshing market data...]")
                    for sym in symbols:
                        df = fetcher.fetch_ohlcv('kraken', sym, '1h', 500)
                        if not df.empty:
                            data[sym] = df
                    last_data_refresh = time.time()

                # Get momentum signals
                xrp_signal = generate_xrp_signals(data)
                btc_signal = generate_btc_signals(data)

                print(f"\nMomentum Signals:")
                print(f"  XRP: {xrp_signal['signal']} (conf: {xrp_signal['confidence']:.2f}, dip: {xrp_signal['is_dip']}, lev_ok: {xrp_signal['leverage_ok']})")
                print(f"  BTC: {btc_signal['signal']} (conf: {btc_signal['confidence']:.2f}, dip: {btc_signal['is_dip']}, lev_ok: {btc_signal['leverage_ok']})")

                # RL Orchestrator decision (every loop)
                if rl_enabled:
                    print(f"\n[RL DECISION]")
                    rl_result = orchestrator.decide_and_execute(prices)
                    print(f"  Action: {rl_result['asset']} {rl_result['action_type']}")
                    if rl_result['executed']:
                        if rl_result.get('leverage_used'):
                            print(f"  LEVERAGED: {rl_result.get('leverage', 10)}x with ${rl_result.get('collateral', 0):.2f} collateral")
                        else:
                            print(f"  Spot: {rl_result.get('amount', 0):.4f}")
                    orchestrator.update_env_step()

                    # Check and manage margin positions
                    orchestrator.check_and_manage_positions(prices)

                # Manual leverage entry on strong dip signals (if RL disabled)
                elif xrp_signal['leverage_ok'] and xrp_signal['is_dip']:
                    usdt = portfolio.balances.get('USDT', 0)
                    if usdt > 100:
                        collateral = usdt * 0.2
                        orchestrator.kraken.open_long('XRP', collateral, prices['XRP'])

                # Rebalance check (every 12 loops = ~1 hour with 5min loop)
                if loop_count % 12 == 0:
                    print("\n[REBALANCE CHECK]")
                    rebalance(portfolio, prices)

                # Trade history
                if executor.trade_log:
                    print(f"\nRecent trades: {len(executor.trade_log)}")

                time.sleep(300)  # 5min loop for live trading

            except KeyboardInterrupt:
                print("\n\n" + "=" * 60)
                print("Stopping paper trading...")
                print("=" * 60)

                # Close all margin positions
                if orchestrator.kraken.positions:
                    print("\nClosing margin positions...")
                    for asset in list(orchestrator.kraken.positions.keys()):
                        orchestrator.kraken.close_position(asset, prices.get(asset, 1.0))

                print(f"\nFinal {portfolio}")
                print(f"Total trades executed: {len(executor.trade_log)}")

                if executor.trade_log:
                    print("\nTrade History (last 10):")
                    for trade in executor.trade_log[-10:]:
                        print(f"  {trade['timestamp']}: {trade['side'].upper()} {trade['amount']:.4f} {trade['symbol']} @ {trade['price']:.4f}")

                if rl_enabled:
                    final_alignment = orchestrator.get_alignment_score(prices)
                    print(f"\nFinal RL Alignment Score: {final_alignment:.2f}/1.0")

                break


if __name__ == "__main__":
    main()
