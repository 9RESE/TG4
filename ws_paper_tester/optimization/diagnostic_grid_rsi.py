#!/usr/bin/env python3
"""
Grid RSI Diagnostic Script

Comprehensive diagnostic to identify why the Grid RSI strategy is not generating signals.
Tests multiple ADX thresholds, timeframes, and logs detailed rejection reasons.

Usage:
    python diagnostic_grid_rsi.py --symbol XRP/USDT --period 1y
    python diagnostic_grid_rsi.py --symbol XRP/USDT --period 1y --parallel
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Add project root and ws_paper_tester to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.kraken_db import HistoricalDataProvider
from ws_tester.types import DataSnapshot, Candle, Signal, Fill, OrderbookSnapshot

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('diagnostic_grid_rsi.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class DiagnosticResult:
    """Result of a single diagnostic run."""
    adx_threshold: int
    timeframe_minutes: int
    use_trend_filter: bool

    # Signal statistics
    total_candles: int = 0
    signals_generated: int = 0
    trades_executed: int = 0

    # Rejection breakdown
    rejection_counts: Dict[str, int] = field(default_factory=dict)

    # Grid statistics
    grid_initializations: int = 0
    grid_recenters: int = 0
    grid_levels_hit: int = 0

    # Grid analysis
    grid_center_price: float = 0.0
    grid_upper_price: float = 0.0
    grid_lower_price: float = 0.0
    grid_buy_levels: List[float] = field(default_factory=list)

    # Price analysis during backtest
    price_min: float = 0.0
    price_max: float = 0.0
    price_at_init: float = 0.0
    times_price_below_grid: int = 0  # Times price dropped into grid buy zone
    times_price_in_buy_range: int = 0  # Times price was within buy level tolerance

    # Performance
    total_pnl: float = 0.0
    win_rate: float = 0.0
    max_drawdown_pct: float = 0.0

    # Detailed logs
    first_signal_candle: Optional[int] = None
    warmup_candles: int = 0

    # Market conditions
    avg_adx: float = 0.0
    max_adx: float = 0.0
    min_adx: float = 0.0
    pct_time_trending: float = 0.0  # % of time ADX > threshold


@dataclass
class DiagnosticConfig:
    """Configuration for diagnostic run."""
    symbol: str
    period: str
    db_url: str
    output_dir: str = "diagnostic_results"

    # Test parameters
    adx_thresholds: List[int] = field(default_factory=lambda: [0, 25, 30, 35, 40, 50])  # 0 = disabled
    timeframes: List[int] = field(default_factory=lambda: [1, 5, 60, 1440])

    # Fixed strategy parameters (reasonable defaults)
    num_grids: int = 15
    grid_spacing_pct: float = 1.5
    range_pct: float = 7.5
    rsi_period: int = 14
    rsi_oversold: int = 30
    rsi_overbought: int = 70
    stop_loss_pct: float = 8.0
    max_accumulation_levels: int = 5
    position_size_usd: float = 20.0
    starting_capital: float = 100.0


class DiagnosticRunner:
    """Runs diagnostic tests on Grid RSI strategy."""

    def __init__(self, config: DiagnosticConfig):
        self.config = config
        self.results: List[DiagnosticResult] = []

    async def run_all_diagnostics(self, provider: HistoricalDataProvider) -> List[DiagnosticResult]:
        """Run all diagnostic combinations."""
        total_tests = len(self.config.adx_thresholds) * len(self.config.timeframes)
        logger.info(f"Starting diagnostic: {total_tests} test combinations")
        logger.info(f"ADX thresholds: {self.config.adx_thresholds}")
        logger.info(f"Timeframes: {self.config.timeframes}")

        test_num = 0
        for timeframe in self.config.timeframes:
            for adx_threshold in self.config.adx_thresholds:
                test_num += 1
                logger.info(f"\n{'='*60}")
                logger.info(f"TEST {test_num}/{total_tests}: TF={timeframe}m, ADX={adx_threshold}")
                logger.info(f"{'='*60}")

                result = await self.run_single_diagnostic(
                    provider,
                    timeframe,
                    adx_threshold
                )
                self.results.append(result)

                # Log summary for this test
                logger.info(f"Result: {result.signals_generated} signals, "
                           f"{result.trades_executed} trades, "
                           f"P&L: ${result.total_pnl:.2f}")

        return self.results

    async def run_single_diagnostic(
        self,
        provider: HistoricalDataProvider,
        timeframe_minutes: int,
        adx_threshold: int
    ) -> DiagnosticResult:
        """Run a single diagnostic test."""
        use_trend_filter = adx_threshold > 0

        result = DiagnosticResult(
            adx_threshold=adx_threshold,
            timeframe_minutes=timeframe_minutes,
            use_trend_filter=use_trend_filter
        )

        # Calculate date range
        end_time = datetime.now(timezone.utc)
        start_time = self._parse_period(self.config.period, end_time)

        logger.info(f"Loading data: {start_time.date()} to {end_time.date()}")

        # Load candle data
        candles = await provider.get_candles(
            self.config.symbol,
            timeframe_minutes,
            start_time,
            end_time
        )

        if not candles:
            logger.warning(f"No candles loaded for {timeframe_minutes}m timeframe")
            return result

        result.total_candles = len(candles)
        logger.info(f"Loaded {len(candles)} candles")

        # Import strategy components
        from strategies.grid_rsi_reversion.indicators import (
            calculate_rsi, calculate_atr, calculate_adx, calculate_volatility
        )
        from strategies.grid_rsi_reversion.grid import (
            check_price_at_grid_level, calculate_grid_stats
        )
        from strategies.grid_rsi_reversion.lifecycle import (
            initialize_state, initialize_grid_for_symbol
        )
        from strategies.grid_rsi_reversion.risk import check_trend_filter
        from strategies.grid_rsi_reversion.regimes import classify_volatility_regime
        from strategies.grid_rsi_reversion.config import VolatilityRegime

        # Build strategy config for this test
        strategy_config = {
            'num_grids': self.config.num_grids,
            'grid_spacing_pct': self.config.grid_spacing_pct,
            'range_pct': self.config.range_pct,
            'rsi_period': self.config.rsi_period,
            'rsi_oversold': self.config.rsi_oversold,
            'rsi_overbought': self.config.rsi_overbought,
            'stop_loss_pct': self.config.stop_loss_pct,
            'max_accumulation_levels': self.config.max_accumulation_levels,
            'position_size_usd': self.config.position_size_usd,
            'adx_threshold': adx_threshold,
            'use_trend_filter': use_trend_filter,
            'adx_period': 14,
            'atr_period': 14,
            'candle_timeframe_minutes': timeframe_minutes,
            'use_volatility_regimes': True,
            'use_adaptive_rsi': True,
            'track_rejections': True,
            'cooldown_seconds': 60,
            'slippage_tolerance_pct': 0.5,
        }

        # Initialize state
        state = {}
        initialize_state(state)

        # Track ADX values for analysis
        adx_values = []

        # Minimum candles for indicators (NO WARMUP PENALTY - we have historic data)
        min_candles = max(
            strategy_config['rsi_period'],
            strategy_config['atr_period'],
            strategy_config['adx_period']
        ) + 5

        result.warmup_candles = min_candles
        logger.info(f"Minimum candles for indicators: {min_candles}")

        # Process candles
        rejection_counts = defaultdict(int)
        signals_by_reason = defaultdict(list)

        # Track price range
        all_prices = [float(c.close) for c in candles]
        result.price_min = min(all_prices)
        result.price_max = max(all_prices)
        logger.info(f"Price range: ${result.price_min:.4f} - ${result.price_max:.4f}")

        # Grid tracking vars
        grid_initialized = False
        symbol = self.config.symbol

        total_to_process = len(candles) - min_candles
        log_interval = max(1000, total_to_process // 20)  # Log every 5% or 1000 candles

        # Fixed lookback for efficiency - no need to use entire history for indicators
        indicator_lookback = 100  # More than enough for RSI(14), ATR(14), ADX(14)

        for i in range(min_candles, len(candles)):
            # Progress logging
            processed = i - min_candles
            if processed > 0 and processed % log_interval == 0:
                pct = (processed / total_to_process) * 100
                logger.info(f"  Progress: {processed}/{total_to_process} ({pct:.0f}%) - Signals: {result.signals_generated}")

            # Use fixed lookback window for O(n) instead of O(n^2)
            start_idx = max(0, i - indicator_lookback)
            window = candles[start_idx:i+1]
            current_candle = candles[i]
            current_price = float(current_candle.close)

            # Calculate indicators (now O(lookback) per candle, not O(i))
            closes = [float(c.close) for c in window]

            rsi = calculate_rsi(closes, strategy_config['rsi_period'])
            atr = calculate_atr(window, strategy_config['atr_period'])
            adx = calculate_adx(window, strategy_config['adx_period'])
            volatility = calculate_volatility(window, lookback=20)

            if adx is not None:
                adx_values.append(adx)

            # Check each rejection reason
            rejection_reason = None

            # 1. Check trend filter
            if use_trend_filter and adx is not None:
                is_trending, _ = check_trend_filter(adx, strategy_config)
                if is_trending:
                    rejection_reason = f"trend_filter_adx_{int(adx)}"
                    rejection_counts['trend_filter'] += 1
                    if i < min_candles + 10:
                        logger.debug(f"Candle {i}: Rejected - ADX={adx:.1f} > {adx_threshold}")
                    continue

            # 2. Check volatility regime
            if volatility is not None:
                regime = classify_volatility_regime(volatility, strategy_config)
                if regime == VolatilityRegime.EXTREME:
                    rejection_reason = "extreme_volatility"
                    rejection_counts['extreme_volatility'] += 1
                    continue

            # 3. Initialize grid if needed
            if not state.get('grids_initialized', {}).get(symbol, False):
                initialize_grid_for_symbol(symbol, current_price, strategy_config, state, atr)
                result.grid_initializations += 1
                result.price_at_init = current_price
                grid_initialized = True

                # Capture grid details for analysis
                grid_metadata = state.get('grid_metadata', {}).get(symbol, {})
                result.grid_center_price = grid_metadata.get('center_price', 0)
                result.grid_upper_price = grid_metadata.get('upper_price', 0)
                result.grid_lower_price = grid_metadata.get('lower_price', 0)

                # Extract buy level prices
                grid_levels_list = state.get('grid_levels', {}).get(symbol, [])
                buy_levels = [l['price'] for l in grid_levels_list if l['side'] == 'buy']
                result.grid_buy_levels = sorted(buy_levels, reverse=True)

                logger.info(f"Grid initialized at candle {i}, price ${current_price:.4f}")
                logger.info(f"  Grid center: ${result.grid_center_price:.4f}")
                logger.info(f"  Grid upper: ${result.grid_upper_price:.4f}")
                logger.info(f"  Grid lower: ${result.grid_lower_price:.4f}")
                logger.info(f"  Buy levels ({len(buy_levels)}): ${buy_levels[0]:.4f} to ${buy_levels[-1]:.4f}" if buy_levels else "  Buy levels: NONE")
                logger.info(f"  Price must DROP to reach buy levels!")

                # Calculate if price ever reaches buy zone
                future_prices = [float(c.close) for c in candles[i:]]
                prices_below_center = sum(1 for p in future_prices if p < result.grid_center_price)
                logger.info(f"  Future candles below grid center: {prices_below_center}/{len(future_prices)}")

            # Track price vs grid
            if grid_initialized and result.grid_buy_levels:
                highest_buy = result.grid_buy_levels[0]  # Closest to center
                tolerance = highest_buy * (strategy_config['slippage_tolerance_pct'] / 100)

                # Count times price was below grid center
                if current_price < result.grid_center_price:
                    result.times_price_below_grid += 1

                # Count times price was within tolerance of a buy level
                if current_price <= highest_buy + tolerance:
                    result.times_price_in_buy_range += 1

            # 4. Check if price is at grid level
            grid_levels = state.get('grid_levels', {}).get(symbol, [])
            if not grid_levels:
                rejection_reason = "no_grid_levels"
                rejection_counts['no_grid_levels'] += 1
                continue

            buy_level = check_price_at_grid_level(grid_levels, current_price, 'buy', strategy_config)

            if buy_level:
                result.grid_levels_hit += 1

                # 5. Check RSI conditions
                if rsi is None:
                    rejection_reason = "no_rsi"
                    rejection_counts['no_rsi'] += 1
                    continue

                # Signal would be generated!
                result.signals_generated += 1
                if result.first_signal_candle is None:
                    result.first_signal_candle = i
                    logger.info(f"FIRST SIGNAL at candle {i}!")
                    logger.info(f"  Price: ${current_price:.4f}")
                    logger.info(f"  RSI: {rsi:.1f}")
                    logger.info(f"  ADX: {adx:.1f}" if adx else "  ADX: N/A")
                    logger.info(f"  Grid level: ${buy_level['price']:.4f}")

                # For diagnostic, count as a trade
                result.trades_executed += 1
            else:
                rejection_reason = "price_not_at_level"
                rejection_counts['price_not_at_level'] += 1

        # Calculate ADX statistics
        if adx_values:
            result.avg_adx = sum(adx_values) / len(adx_values)
            result.max_adx = max(adx_values)
            result.min_adx = min(adx_values)
            if use_trend_filter:
                trending_count = sum(1 for v in adx_values if v > adx_threshold)
                result.pct_time_trending = (trending_count / len(adx_values)) * 100

        # Store rejection counts
        result.rejection_counts = dict(rejection_counts)

        # Log summary
        logger.info(f"\n--- Test Summary ---")
        logger.info(f"Candles processed: {result.total_candles - min_candles}")
        logger.info(f"Signals generated: {result.signals_generated}")
        logger.info(f"Grid levels hit: {result.grid_levels_hit}")
        logger.info(f"ADX stats: avg={result.avg_adx:.1f}, min={result.min_adx:.1f}, max={result.max_adx:.1f}")
        logger.info(f"Time trending (ADX>{adx_threshold}): {result.pct_time_trending:.1f}%")

        # Price vs Grid analysis
        logger.info(f"\n--- Price vs Grid Analysis ---")
        logger.info(f"Price range: ${result.price_min:.4f} - ${result.price_max:.4f}")
        if result.grid_center_price > 0:
            logger.info(f"Grid center: ${result.grid_center_price:.4f}")
            logger.info(f"Grid buy zone: ${result.grid_lower_price:.4f} - ${result.grid_center_price:.4f}")
            logger.info(f"Times price below grid center: {result.times_price_below_grid}")
            logger.info(f"Times price in buy range: {result.times_price_in_buy_range}")

            # Critical insight: did price ever reach buy levels?
            if result.grid_buy_levels:
                highest_buy = result.grid_buy_levels[0]
                lowest_buy = result.grid_buy_levels[-1]
                logger.info(f"Highest buy level: ${highest_buy:.4f}")
                logger.info(f"Lowest buy level: ${lowest_buy:.4f}")
                logger.info(f"Price min vs highest buy: ${result.price_min:.4f} vs ${highest_buy:.4f}")
                if result.price_min > highest_buy:
                    logger.warning(f"ISSUE: Price NEVER dropped to buy levels! Min price ${result.price_min:.4f} > highest buy ${highest_buy:.4f}")
        logger.info(f"Rejection breakdown:")
        for reason, count in sorted(rejection_counts.items(), key=lambda x: -x[1]):
            pct = (count / (result.total_candles - min_candles)) * 100
            logger.info(f"  {reason}: {count} ({pct:.1f}%)")

        return result

    def _parse_period(self, period: str, end_time: datetime) -> datetime:
        """Parse period string to start datetime."""
        periods = {
            '1w': timedelta(weeks=1),
            '2w': timedelta(weeks=2),
            '1m': timedelta(days=30),
            '3m': timedelta(days=90),
            '6m': timedelta(days=180),
            '1y': timedelta(days=365),
            '2y': timedelta(days=730),
        }
        delta = periods.get(period.lower(), timedelta(days=365))
        return end_time - delta

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive diagnostic report."""
        report = {
            'config': {
                'symbol': self.config.symbol,
                'period': self.config.period,
                'adx_thresholds': self.config.adx_thresholds,
                'timeframes': self.config.timeframes,
            },
            'summary': {},
            'results_by_timeframe': {},
            'results_by_adx': {},
            'detailed_results': [],
            'recommendations': []
        }

        # Group results
        by_timeframe = defaultdict(list)
        by_adx = defaultdict(list)

        for r in self.results:
            by_timeframe[r.timeframe_minutes].append(r)
            by_adx[r.adx_threshold].append(r)

            report['detailed_results'].append({
                'timeframe_minutes': r.timeframe_minutes,
                'adx_threshold': r.adx_threshold,
                'use_trend_filter': r.use_trend_filter,
                'total_candles': r.total_candles,
                'signals_generated': r.signals_generated,
                'trades_executed': r.trades_executed,
                'grid_initializations': r.grid_initializations,
                'grid_levels_hit': r.grid_levels_hit,
                'rejection_counts': r.rejection_counts,
                'avg_adx': r.avg_adx,
                'pct_time_trending': r.pct_time_trending,
                'first_signal_candle': r.first_signal_candle,
                'warmup_candles': r.warmup_candles,
                # Grid analysis
                'grid_center_price': r.grid_center_price,
                'grid_upper_price': r.grid_upper_price,
                'grid_lower_price': r.grid_lower_price,
                'grid_buy_levels_count': len(r.grid_buy_levels),
                'grid_highest_buy': r.grid_buy_levels[0] if r.grid_buy_levels else 0,
                'grid_lowest_buy': r.grid_buy_levels[-1] if r.grid_buy_levels else 0,
                # Price analysis
                'price_min': r.price_min,
                'price_max': r.price_max,
                'price_at_init': r.price_at_init,
                'times_price_below_grid': r.times_price_below_grid,
                'times_price_in_buy_range': r.times_price_in_buy_range,
            })

        # Summarize by timeframe
        for tf, results in by_timeframe.items():
            total_signals = sum(r.signals_generated for r in results)
            total_trades = sum(r.trades_executed for r in results)
            avg_adx = sum(r.avg_adx for r in results) / len(results) if results else 0

            report['results_by_timeframe'][str(tf)] = {
                'total_signals': total_signals,
                'total_trades': total_trades,
                'avg_adx': avg_adx,
                'tests': len(results)
            }

        # Summarize by ADX threshold
        for adx, results in by_adx.items():
            total_signals = sum(r.signals_generated for r in results)
            total_trades = sum(r.trades_executed for r in results)
            avg_trending_pct = sum(r.pct_time_trending for r in results) / len(results) if results else 0

            report['results_by_adx'][str(adx)] = {
                'total_signals': total_signals,
                'total_trades': total_trades,
                'avg_trending_pct': avg_trending_pct,
                'tests': len(results)
            }

        # Overall summary
        total_signals = sum(r.signals_generated for r in self.results)
        total_trades = sum(r.trades_executed for r in self.results)

        report['summary'] = {
            'total_tests': len(self.results),
            'total_signals_all_tests': total_signals,
            'total_trades_all_tests': total_trades,
            'best_timeframe': max(by_timeframe.items(),
                                  key=lambda x: sum(r.signals_generated for r in x[1]))[0] if by_timeframe else None,
            'best_adx_threshold': max(by_adx.items(),
                                      key=lambda x: sum(r.signals_generated for r in x[1]))[0] if by_adx else None,
        }

        # Generate recommendations
        recommendations = []

        # Check if ADX filter is the problem
        adx_disabled_signals = sum(r.signals_generated for r in by_adx.get(0, []))
        adx_30_signals = sum(r.signals_generated for r in by_adx.get(30, []))

        if adx_disabled_signals > adx_30_signals * 2:
            recommendations.append(
                f"ADX filter is blocking signals. With ADX disabled: {adx_disabled_signals} signals, "
                f"with ADX=30: {adx_30_signals} signals. Consider raising ADX threshold or disabling."
            )

        # Check timeframe impact
        best_tf = report['summary']['best_timeframe']
        if best_tf:
            best_tf_signals = report['results_by_timeframe'][str(best_tf)]['total_signals']
            recommendations.append(f"Best performing timeframe: {best_tf}m with {best_tf_signals} signals")

        # Check if grid initialization is working
        grid_init_count = sum(r.grid_initializations for r in self.results)
        if grid_init_count == 0:
            recommendations.append("WARNING: No grid initializations occurred. Check grid initialization logic.")

        # Check if price is reaching grid levels
        total_levels_hit = sum(r.grid_levels_hit for r in self.results)
        if total_levels_hit == 0:
            recommendations.append(
                "WARNING: No grid levels were hit. Grid spacing may be too wide or "
                "grid center may be far from current price."
            )

        report['recommendations'] = recommendations

        return report


async def main():
    parser = argparse.ArgumentParser(
        description='Grid RSI Diagnostic Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full diagnostic (all timeframes, all ADX thresholds, 1 year)
    python diagnostic_grid_rsi.py --symbol XRP/USDT --period 1y

    # Quick diagnostic (5m only, 3 months)
    python diagnostic_grid_rsi.py --symbol XRP/USDT --period 3m --timeframes 5

    # Test specific ADX thresholds
    python diagnostic_grid_rsi.py --symbol XRP/USDT --period 1y --adx-thresholds 0,30,50
        """
    )

    parser.add_argument('--symbol', type=str, default='XRP/USDT',
                        help='Trading pair (default: XRP/USDT)')
    parser.add_argument('--period', type=str, default='1y',
                        help='Test period: 1w, 1m, 3m, 6m, 1y, 2y (default: 1y)')
    parser.add_argument('--db-url', type=str, default=os.getenv('DATABASE_URL'),
                        help='Database URL')
    parser.add_argument('--output', type=str, default='diagnostic_results',
                        help='Output directory')
    parser.add_argument('--timeframes', type=str, default='1,5,60,1440',
                        help='Comma-separated timeframes in minutes (default: 1,5,60,1440)')
    parser.add_argument('--adx-thresholds', type=str, default='0,25,30,35,40,50',
                        help='Comma-separated ADX thresholds, 0=disabled (default: 0,25,30,35,40,50)')

    args = parser.parse_args()

    if not args.db_url:
        print("ERROR: Database URL required. Set DATABASE_URL or use --db-url")
        print("Example: DATABASE_URL=postgresql://trading:password@localhost:5433/kraken_data")
        sys.exit(1)

    # Parse parameters
    timeframes = [int(t.strip()) for t in args.timeframes.split(',')]
    adx_thresholds = [int(t.strip()) for t in args.adx_thresholds.split(',')]

    # Create config
    config = DiagnosticConfig(
        symbol=args.symbol,
        period=args.period,
        db_url=args.db_url,
        output_dir=args.output,
        adx_thresholds=adx_thresholds,
        timeframes=timeframes,
    )

    # Show test plan
    total_tests = len(timeframes) * len(adx_thresholds)
    print(f"\n{'='*60}")
    print(f"GRID RSI DIAGNOSTIC")
    print(f"{'='*60}")
    print(f"Symbol: {args.symbol}")
    print(f"Period: {args.period}")
    print(f"Timeframes: {timeframes}")
    print(f"ADX Thresholds: {adx_thresholds}")
    print(f"Total test combinations: {total_tests}")
    print(f"Output: {args.output}/")
    print(f"Log file: diagnostic_grid_rsi.log")
    print(f"{'='*60}\n")

    # Initialize provider
    provider = HistoricalDataProvider(args.db_url)
    await provider.connect()

    try:
        # Run diagnostics
        runner = DiagnosticRunner(config)
        results = await runner.run_all_diagnostics(provider)

        # Generate report
        report = runner.generate_report()

        # Save report
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = output_dir / f"diagnostic_{args.symbol.replace('/', '_')}_{timestamp}.json"

        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        # Print summary
        print(f"\n{'='*60}")
        print("DIAGNOSTIC COMPLETE")
        print(f"{'='*60}")
        print(f"\nSUMMARY:")
        print(f"  Total tests: {report['summary']['total_tests']}")
        print(f"  Total signals (all tests): {report['summary']['total_signals_all_tests']}")
        print(f"  Best timeframe: {report['summary']['best_timeframe']}m")
        print(f"  Best ADX threshold: {report['summary']['best_adx_threshold']}")

        print(f"\nRESULTS BY TIMEFRAME:")
        for tf, data in sorted(report['results_by_timeframe'].items(), key=lambda x: int(x[0])):
            print(f"  {tf}m: {data['total_signals']} signals, avg ADX={data['avg_adx']:.1f}")

        print(f"\nRESULTS BY ADX THRESHOLD:")
        for adx, data in sorted(report['results_by_adx'].items(), key=lambda x: int(x[0])):
            threshold_str = "disabled" if adx == "0" else f">{adx}"
            print(f"  ADX {threshold_str}: {data['total_signals']} signals, "
                  f"trending {data['avg_trending_pct']:.1f}% of time")

        print(f"\nRECOMMENDATIONS:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"  {i}. {rec}")

        print(f"\nReport saved: {report_file}")
        print(f"Full log: diagnostic_grid_rsi.log")

    finally:
        await provider.close()


if __name__ == '__main__':
    asyncio.run(main())
