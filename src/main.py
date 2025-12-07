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
        risk = RiskManager()
        arb = StableArb()
        print("\n[PAPER TRADING MODE] Starting live monitoring...")
        print("Press Ctrl+C to stop\n")

        # Try to load RL model if available
        rl_model = None
        try:
            from models.rl_agent import load_rl_agent
            rl_model = load_rl_agent()
            print("RL model loaded for signal generation")
        except:
            print("No RL model found, using rule-based signals only")

        loop_count = 0
        while True:
            try:
                loop_count += 1
                print(f"\n--- Loop {loop_count} @ {time.strftime('%H:%M:%S')} ---")

                # Fetch current prices
                prices = {}
                for sym in ['XRP/USDT', 'BTC/USDT']:
                    p = fetcher.get_best_price(sym)
                    if p:
                        # Use first exchange price as reference
                        prices[sym.split('/')[0]] = list(p.values())[0]
                        print(f"{sym}: {p}")

                # Add stablecoin prices (assumed 1:1)
                prices['USDT'] = 1.0
                prices['USDC'] = 1.0
                prices['RLUSD'] = 1.0

                # Record portfolio snapshot
                portfolio.record_snapshot(prices)
                print(f"Portfolio Value: ${portfolio.get_total_usd(prices):.2f}")

                # Check for arb opportunities
                opps = arb.find_opportunities()
                if opps:
                    print(f"ARB OPPORTUNITIES: {opps}")

                # Rebalance check (every 60 loops = ~1 hour)
                if loop_count % 60 == 0:
                    print("\n[REBALANCE CHECK]")
                    rebalance(portfolio, prices)

                time.sleep(60)  # 1min loop

            except KeyboardInterrupt:
                print("\n\nStopping paper trading...")
                print(f"Final {portfolio}")
                break

if __name__ == "__main__":
    main()
