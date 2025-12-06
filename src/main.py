import argparse
from data_fetcher import DataFetcher
from portfolio import Portfolio
from backtester import Backtester
from strategies.ripple_momentum_lstm import generate_ripple_signals
import yaml
import matplotlib.pyplot as plt

def load_config():
    with open('config/exchanges.yaml') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="TG4 Local AI Crypto Trader")
    parser.add_argument('--mode', choices=['fetch', 'backtest', 'paper'], default='fetch')
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

    elif args.mode == 'backtest':
        symbols = ['XRP/USDT', 'XRP/RLUSD', 'BTC/USDT']
        data = {}
        for sym in symbols:
            print(f"Fetching {sym}...")
            df = fetcher.fetch_ohlcv('kraken', sym, '1h', 2000)
            if not df.empty:
                data[sym] = df

        if 'XRP/RLUSD' in data:
            signals = generate_ripple_signals(data, 'XRP/RLUSD')
            bt = Backtester(data)
            pf = bt.run_with_lstm_signals('XRP/RLUSD', signals)

            print(f"Total Return: {pf.total_return():.2%}")
            print(f"Sharpe Ratio: {pf.sharpe_ratio():.2f}")
            pf.plot().show()
        elif 'XRP/USDT' in data:
            # Fallback to XRP/USDT if RLUSD pair not available
            signals = generate_ripple_signals(data, 'XRP/USDT')
            bt = Backtester(data)
            pf = bt.run_with_lstm_signals('XRP/USDT', signals)

            print(f"Total Return: {pf.total_return():.2%}")
            print(f"Sharpe Ratio: {pf.sharpe_ratio():.2f}")
            pf.plot().show()

if __name__ == "__main__":
    main()
