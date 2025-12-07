#!/usr/bin/env python3
"""
Phase 7 Leverage Backtest
Pure USDT-collateralized 10x XRP longs during dips
"""
import numpy as np
import pandas as pd
from data_fetcher import DataFetcher
from portfolio import Portfolio
from exchanges.kraken_margin import KrakenMargin
from strategies.ripple_momentum_lstm import generate_xrp_signals, detect_dip

def run_leverage_backtest():
    print("=" * 70)
    print("TG4 Phase 7: 10x Leveraged XRP Backtest - Dec 2025 Dip Strategy")
    print("=" * 70)

    # Initialize
    portfolio = Portfolio({'USDT': 1000.0, 'XRP': 500.0, 'BTC': 0.0})
    kraken = KrakenMargin(portfolio, max_leverage=10.0)
    fetcher = DataFetcher()

    # Fetch XRP data
    print("\nFetching XRP/USDT hourly data from Kraken...")
    df = fetcher.fetch_ohlcv('kraken', 'XRP/USDT', '1h', 2000)

    if df.empty:
        print("No data fetched")
        return

    print(f"Loaded {len(df)} candles from {df.index[0]} to {df.index[-1]}")

    # Get initial XRP price
    initial_xrp_price = df['close'].iloc[0]
    final_xrp_price = df['close'].iloc[-1]
    print(f"\nXRP price: ${initial_xrp_price:.4f} → ${final_xrp_price:.4f} ({(final_xrp_price/initial_xrp_price - 1)*100:+.1f}%)")

    # Backtest parameters
    collateral_per_trade = 200.0  # $200 USDT per leveraged position
    take_profit = 0.08  # 8% gain = close (conservative in volatile market)
    stop_loss = -0.05   # 5% loss = close (tight stops)
    min_bars_between = 24  # Wait 24 hours between trades

    # Trading state
    trades = []
    active_position = None
    bars_since_last = 0
    total_pnl = 0.0

    print(f"\nBacktest parameters:")
    print(f"  Collateral per trade: ${collateral_per_trade}")
    print(f"  Leverage: 10x (exposure: ${collateral_per_trade * 10})")
    print(f"  Take profit: +{take_profit*100:.0f}%")
    print(f"  Stop loss: {stop_loss*100:.0f}%")
    print(f"  Cooldown: {min_bars_between}h between trades")

    print("\n" + "=" * 70)
    print("TRADE LOG")
    print("=" * 70)

    for i in range(100, len(df)):  # Start after warmup
        bars_since_last += 1
        price = df['close'].iloc[i]
        close_arr = df['close'].iloc[:i+1].values
        timestamp = df.index[i]

        # Check active position
        if active_position:
            entry = active_position['entry']
            size = active_position['size']
            pnl_pct = (price - entry) / entry * 10  # 10x leverage
            pnl_usd = (price - entry) * size

            # Take profit or stop loss
            if pnl_pct >= take_profit or pnl_pct <= stop_loss:
                reason = "TAKE PROFIT" if pnl_pct >= take_profit else "STOP LOSS"
                total_pnl += pnl_usd

                print(f"{timestamp} | {reason} @ ${price:.4f} | P&L: ${pnl_usd:+.2f} ({pnl_pct*100:+.1f}%)")

                trades.append({
                    'entry_time': active_position['time'],
                    'exit_time': timestamp,
                    'entry': entry,
                    'exit': price,
                    'size': size,
                    'pnl': pnl_usd,
                    'pnl_pct': pnl_pct,
                    'reason': reason
                })

                active_position = None
                bars_since_last = 0
            continue

        # Look for entry on dip
        if bars_since_last >= min_bars_between:
            is_dip = detect_dip(close_arr, lookback=20, threshold=-0.05)

            # Additional momentum filter - RSI oversold
            delta = np.diff(close_arr[-15:])
            gains = np.where(delta > 0, delta, 0)
            losses = np.where(delta < 0, -delta, 0)
            avg_gain = np.mean(gains) if len(gains) > 0 else 0
            avg_loss = np.mean(losses) if len(losses) > 0 else 0.001
            rsi = 100 - (100 / (1 + avg_gain / avg_loss))

            # Entry on dip + heavily oversold (stricter entry)
            if is_dip and rsi < 25:
                size = (collateral_per_trade * 10) / price  # 10x leverage

                active_position = {
                    'entry': price,
                    'size': size,
                    'time': timestamp,
                    'collateral': collateral_per_trade
                }

                drawdown = (price - np.max(close_arr[-20:])) / np.max(close_arr[-20:]) * 100
                print(f"{timestamp} | OPEN 10X LONG @ ${price:.4f} | {size:.2f} XRP | RSI: {rsi:.0f} | Drawdown: {drawdown:.1f}%")

    # Close any remaining position at final price
    if active_position:
        entry = active_position['entry']
        size = active_position['size']
        pnl_usd = (final_xrp_price - entry) * size
        pnl_pct = (final_xrp_price - entry) / entry * 10
        total_pnl += pnl_usd

        trades.append({
            'entry_time': active_position['time'],
            'exit_time': df.index[-1],
            'entry': entry,
            'exit': final_xrp_price,
            'size': size,
            'pnl': pnl_usd,
            'pnl_pct': pnl_pct,
            'reason': 'END'
        })
        print(f"{df.index[-1]} | CLOSE (END) @ ${final_xrp_price:.4f} | P&L: ${pnl_usd:+.2f} ({pnl_pct*100:+.1f}%)")

    # Summary
    print("\n" + "=" * 70)
    print("BACKTEST SUMMARY")
    print("=" * 70)

    if not trades:
        print("No trades executed")
        return

    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]

    print(f"Total Trades:       {len(trades)}")
    print(f"Winning Trades:     {len(wins)} ({len(wins)/len(trades)*100:.0f}%)")
    print(f"Losing Trades:      {len(losses)} ({len(losses)/len(trades)*100:.0f}%)")
    print(f"")
    print(f"Total P&L:          ${total_pnl:+.2f}")
    print(f"Avg Win:            ${np.mean([t['pnl'] for t in wins]):.2f}" if wins else "Avg Win:            N/A")
    print(f"Avg Loss:           ${np.mean([t['pnl'] for t in losses]):.2f}" if losses else "Avg Loss:           N/A")
    print(f"")
    print(f"Starting Capital:   $1,000 USDT + 500 XRP")
    print(f"Collateral Used:    ${collateral_per_trade} per trade (10x = ${collateral_per_trade*10} exposure)")
    print(f"")

    # ROI calculation
    capital_at_risk = collateral_per_trade * len(trades)
    roi = (total_pnl / 1000) * 100  # ROI on starting USDT

    print(f"Total ROI on USDT:  {roi:+.1f}%")
    print(f"Final USDT:         ${1000 + total_pnl:.2f}")
    print("=" * 70)

    # Trade detail table
    print("\nDETAILED TRADES:")
    print("-" * 90)
    print(f"{'Entry Time':<20} {'Entry':>10} {'Exit':>10} {'Size':>10} {'P&L':>12} {'Reason':<12}")
    print("-" * 90)

    for t in trades:
        entry_str = str(t['entry_time'])[:16]
        print(f"{entry_str:<20} ${t['entry']:.4f}  ${t['exit']:.4f}  {t['size']:.2f} XRP  ${t['pnl']:+.2f}  {t['reason']}")

    return trades, total_pnl


if __name__ == "__main__":
    run_leverage_backtest()
