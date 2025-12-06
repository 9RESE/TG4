import argparse
from data_fetcher import DataFetcher
from portfolio import Portfolio
import yaml

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

    # Example: fetch latest prices
    symbols = ['BTC/USDT', 'XRP/USDT', 'RLUSD/USDT', 'XRP/RLUSD']
    for symbol in symbols:
        prices = fetcher.get_best_price(symbol)
        if prices:
            print(f"{symbol}: {prices}")

if __name__ == "__main__":
    main()
