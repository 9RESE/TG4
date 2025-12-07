from data_fetcher import DataFetcher
import time

class StableArb:
    """Stablecoin arbitrage detector with RLUSD focus"""

    def __init__(self, threshold=0.002):  # 0.2% spread after fees
        self.fetcher = DataFetcher()
        self.threshold = threshold
        self.stables = ['USDT', 'USDC', 'RLUSD']

    def find_opportunities(self):
        """Find arbitrage opportunities, prioritizing RLUSD deviations"""
        opportunities = []

        # Priority 1: RLUSD/USDT peg deviation on Kraken
        rlusd_prices = self.fetcher.get_best_price('RLUSD/USDT')
        if rlusd_prices:
            for ex, price in rlusd_prices.items():
                deviation = (price - 1.0) / 1.0
                if abs(deviation) > self.threshold:
                    opportunities.append({
                        'type': 'peg_deviation',
                        'action': 'buy' if deviation < 0 else 'sell',
                        'asset': 'RLUSD',
                        'deviation': deviation,
                        'spread': abs(deviation) - 0.002,
                        'exchange': ex,
                        'price': price
                    })

        # Priority 2: Cross-pair triangular arb
        for base in ['BTC', 'XRP']:
            for stable1 in self.stables:
                for stable2 in self.stables:
                    if stable1 == stable2:
                        continue
                    symbol1 = f"{base}/{stable1}"
                    symbol2 = f"{base}/{stable2}"
                    prices1 = self.fetcher.get_best_price(symbol1)
                    prices2 = self.fetcher.get_best_price(symbol2)
                    if not prices1 or not prices2:
                        continue

                    # Implicit cross-rate arb
                    for ex1 in prices1:
                        for ex2 in prices2:
                            if base == 'BTC':
                                implied = prices2[ex2] / prices1[ex1]
                            else:
                                implied = prices1[ex1] / prices2[ex2]
                            spread = abs(implied - 1) - 0.002  # fees
                            if spread > self.threshold:
                                opportunities.append({
                                    'type': 'cross_pair',
                                    'pair': f"{stable1}-{stable2}",
                                    'base': base,
                                    'spread': spread,
                                    'implied_rate': implied,
                                    'ex1': ex1,
                                    'ex2': ex2
                                })

        # Sort by spread (best opportunities first)
        opportunities.sort(key=lambda x: x.get('spread', 0), reverse=True)
        return opportunities

    def get_best_opportunity(self):
        """Get the single best arbitrage opportunity"""
        opps = self.find_opportunities()
        return opps[0] if opps else None
