from portfolio import Portfolio

def rebalance(portfolio: Portfolio, prices: dict, targets: dict = {'BTC': 0.4, 'XRP': 0.3, 'RLUSD': 0.2, 'USDT': 0.05, 'USDC': 0.05}):
    total_usd = portfolio.get_total_usd(prices)
    current_weights = {asset: portfolio.balances.get(asset, 0) * prices.get(asset, 0) / total_usd for asset in targets}

    for asset, target in targets.items():
        diff = target - current_weights.get(asset, 0)
        if abs(diff) > 0.05:  # 5% deviation
            usd_amount = diff * total_usd
            # Simplified: trade to USDT first then to target
            print(f"Rebalancing {asset}: {'Buy' if usd_amount > 0 else 'Sell'} {abs(usd_amount):.2f}$")
