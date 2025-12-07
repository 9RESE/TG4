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
from strategies.ripple_momentum_lstm import generate_ripple_signals
from strategies.stablecoin_arb import StableArb
from strategies.rebalancer import rebalance
from risk_manager import RiskManager
from executor import Executor
from ensemble import Ensemble
from orchestrator import RLOrchestrator
import yaml
import time

def load_config():
    with open('config/exchanges.yaml') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="TG4 Local AI Crypto Trader")
    parser.add_argument('--mode', choices=['fetch', 'backtest', 'paper', 'train-rl'], default='fetch')
    parser.add_argument('--timesteps', type=int, default=100000, help='RL training timesteps')
    args = parser.parse_args()

    config = load_config()
    portfolio = Portfolio(config['starting_balance'])
    fetcher = DataFetcher()

    print("TG4 Platform Initialized")
    print(portfolio)

    if args.mode == 'fetch':
        # Example: fetch latest prices
        symbols = ['BTC/USDT', 'XRP/USDT', 'RLUSD/USDT', 'XRP/RLUSD']
        for symbol in symbols:
            prices = fetcher.get_best_price(symbol)
            if prices:
                print(f"{symbol}: {prices}")

        # Also fetch RLUSD pairs
        print("\nFetching RLUSD pairs...")
        rlusd_data = fetcher.fetch_rlusd_pairs()
        for sym, df in rlusd_data.items():
            print(f"{sym}: {len(df)} candles, latest close: {df['close'].iloc[-1]:.4f}")

    elif args.mode == 'backtest':
        symbols = ['XRP/USDT', 'XRP/RLUSD', 'BTC/USDT']
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

        if 'XRP/RLUSD' in data:
            signals = generate_ripple_signals(data, 'XRP/RLUSD')
            bt = Backtester(data)
            pf = bt.run_with_lstm_signals('XRP/RLUSD', signals)
            print_backtest_results(pf, 'XRP/RLUSD')
        elif 'XRP/USDT' in data:
            # Fallback to XRP/USDT if RLUSD pair not available
            signals = generate_ripple_signals(data, 'XRP/USDT')
            bt = Backtester(data)
            pf = bt.run_with_lstm_signals('XRP/USDT', signals)
            print_backtest_results(pf, 'XRP/USDT')

    elif args.mode == 'train-rl':
        from models.rl_agent import train_rl_agent

        print("\n[RL TRAINING MODE]")
        print(f"Training for {args.timesteps} timesteps...")

        # Fetch data for RL training
        symbols = ['XRP/USDT', 'BTC/USDT']
        data = {}
        for sym in symbols:
            print(f"Fetching {sym}...")
            df = fetcher.fetch_ohlcv('kraken', sym, '1h', 2000)
            if not df.empty:
                data[sym] = df

        if data:
            model = train_rl_agent(data, timesteps=args.timesteps)
            print("RL training complete!")
        else:
            print("No data available for training")

    elif args.mode == 'paper':
        print("\n[PAPER TRADING MODE] Starting live monitoring...")
        print("Press Ctrl+C to stop\n")

        # Initialize execution engine
        executor = Executor(portfolio)

        # Fetch initial data for ensemble
        print("Fetching initial market data...")
        symbols = ['XRP/USDT', 'BTC/USDT', 'RLUSD/USDT']
        data = {}
        for sym in symbols:
            df = fetcher.fetch_ohlcv('kraken', sym, '1h', 500)
            if not df.empty:
                data[sym] = df
                print(f"  {sym}: {len(df)} candles")

        # Initialize ensemble strategy
        ensemble = Ensemble(data, portfolio)

        # Initialize RL Orchestrator (uses trained model if available)
        orchestrator = RLOrchestrator(portfolio, data)
        rl_enabled = orchestrator.enabled
        if rl_enabled:
            print("\n[RL ORCHESTRATOR] Enabled - using trained PPO model")
            print(f"  Target allocation: {orchestrator.get_target_allocation()}")
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
                prices = {'USDT': 1.0, 'USDC': 1.0, 'RLUSD': 1.0}
                for sym in ['XRP/USDT', 'BTC/USDT', 'RLUSD/USDT']:
                    p = fetcher.get_best_price(sym)
                    if p:
                        base = sym.split('/')[0]
                        prices[base] = list(p.values())[0]
                        print(f"{sym}: ${prices[base]:.4f}")

                # Record portfolio snapshot
                portfolio.record_snapshot(prices)
                total_value = portfolio.get_total_usd(prices)
                print(f"\nPortfolio Value: ${total_value:.2f}")
                print(f"Holdings: {portfolio}")

                # Show allocation vs targets
                if rl_enabled:
                    current_alloc = orchestrator.get_current_allocation(prices)
                    targets = orchestrator.get_target_allocation()
                    alignment = orchestrator.get_alignment_score(prices)
                    print(f"\nAllocation (target in brackets):")
                    for asset in ['BTC', 'XRP', 'RLUSD', 'USDT']:
                        curr = current_alloc.get(asset, 0) * 100
                        tgt = targets.get(asset, 0) * 100
                        status = "✓" if abs(curr - tgt) < 5 else "↓" if curr < tgt else "↑"
                        print(f"  {asset}: {curr:.1f}% [{tgt:.0f}%] {status}")
                    print(f"  Alignment Score: {alignment:.2f}/1.0")

                # Refresh data every 30 minutes
                if time.time() - last_data_refresh > 1800:
                    print("\n[Refreshing market data...]")
                    for sym in symbols:
                        df = fetcher.fetch_ohlcv('kraken', sym, '1h', 500)
                        if not df.empty:
                            data[sym] = df
                    ensemble.update_data(data)
                    last_data_refresh = time.time()

                # Get ensemble signal
                signal = ensemble.get_signal('XRP/USDT')
                print(f"\nEnsemble Signal: {signal['action']} (confidence: {signal['confidence']:.2f})")
                print(f"  LSTM: {signal['signals']['lstm']:.2f}")
                print(f"  Arb:  {signal['signals']['arb']:.2f}")
                print(f"  Rebal: {signal['signals']['rebalance']:.2f}")

                # RL Orchestrator decision (if enabled and every other loop)
                if rl_enabled and loop_count % 2 == 0:
                    print(f"\n[RL DECISION]")
                    rl_result = orchestrator.decide_and_execute(prices)
                    print(f"  Action: {rl_result['asset']} {rl_result['action_type']}")
                    if rl_result['executed']:
                        print(f"  Executed: {rl_result.get('amount', 0):.4f} @ {rl_result.get('leverage', 1.0)}x")
                    orchestrator.update_env_step()

                # Ensemble-based execution (fallback or complement to RL)
                elif signal['action'] == 'long_xrp' and signal['confidence'] > 0.5:
                    xrp_price = prices.get('XRP', 2.0)
                    usdt_available = portfolio.balances.get('USDT', 0)
                    if usdt_available > 50:  # Min $50 trade
                        trade_size = min(usdt_available * 0.1, 100) / xrp_price  # 10% or max $100
                        executor.place_paper_order('XRP/USDT', 'buy', trade_size, leverage=2.0)

                elif signal['action'] == 'arb_rlusd' and signal['arb_opportunities']:
                    best_arb = signal['arb_opportunities'][0]
                    print(f"\n[ARB OPPORTUNITY] {best_arb}")

                # Rebalance check (every 12 loops = ~1 hour with 5min loop)
                if loop_count % 12 == 0:
                    print("\n[REBALANCE CHECK]")
                    rebalance(portfolio, prices)

                # Trade history
                if executor.trade_log:
                    print(f"\nRecent trades: {len(executor.trade_log)}")

                time.sleep(300)  # 5min loop for live trading

            except KeyboardInterrupt:
                print("\n\n" + "="*60)
                print("Stopping paper trading...")
                print("="*60)
                print(f"Final {portfolio}")
                print(f"Total trades executed: {len(executor.trade_log)}")
                if executor.trade_log:
                    print("\nTrade History:")
                    for trade in executor.trade_log[-10:]:  # Last 10 trades
                        print(f"  {trade['timestamp']}: {trade['side'].upper()} {trade['amount']:.4f} {trade['symbol']} @ {trade['price']:.4f}")
                if rl_enabled:
                    final_alignment = orchestrator.get_alignment_score(prices)
                    print(f"\nFinal RL Alignment Score: {final_alignment:.2f}/1.0")
                break

if __name__ == "__main__":
    main()
