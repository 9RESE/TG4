from data_fetcher import DataFetcher
import time

class StableArb:
    def __init__(self, threshold=0.002):  # 0.2% spread after fees
        self.fetcher = DataFetcher()
        self.threshold = threshold
        self.stables = ['USDT', 'USDC', 'RLUSD']

    def find_opportunities(self):
        opportunities = []
        for base in ['BTC', 'XRP']:
            for stable1 in self.stables:
                for stable2 in self.stables:
                    if stable1 == stable2: continue
                    symbol1 = f"{base}/{stable1}"
                    symbol2 = f"{base}/{stable2}"
                    prices1 = self.fetcher.get_best_price(symbol1)
                    prices2 = self.fetcher.get_best_price(symbol2)
                    if not prices1 or not prices2: continue

                    # Implicit cross-rate arb
                    for ex1 in prices1:
                        for ex2 in prices2:
                            implied = prices2[ex2] / prices1[ex1] if base == 'BTC' else prices1[ex1] / prices2[ex2]
                            spread = abs(implied - 1) - 0.002  # fees
                            if spread > self.threshold:
                                opportunities.append({
                                    'pair': f"{stable1}-{stable2}",
                                    'spread': spread,
                                    'ex1': ex1, 'ex2': ex2
                                })
        return opportunities
