#!/usr/bin/env python3
"""
WebSocket Paper Trading Tester with Historical Data Support.

Extended main entry point that integrates the historical data system:
1. Runs gap filler on startup to sync historical data
2. Initializes database writer for real-time data persistence
3. Starts ws_paper_tester with historical data provider for strategy warmup

Usage:
    python main_with_historical.py                  # Run with historical data support
    python main_with_historical.py --skip-gap-fill  # Skip gap filling
    python main_with_historical.py --duration 60    # Run for 60 minutes
    python main_with_historical.py --simulated      # Use simulated data
"""

import asyncio
import argparse
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root and ws_paper_tester to path
import sys
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

# Import main tester (defer to avoid import errors if not fully set up)
try:
    from paper_tester import WebSocketPaperTester, load_config
    TESTER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import WebSocketPaperTester: {e}")
    TESTER_AVAILABLE = False

# Import historical data components
try:
    from data.kraken_db import (
        HistoricalDataProvider,
        DatabaseWriter,
        run_gap_filler,
        integrate_db_writer,
    )
    HISTORICAL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Historical data module not available: {e}")
    HISTORICAL_AVAILABLE = False

logger = logging.getLogger(__name__)


class HistoricalPaperTester:
    """
    Extended paper tester with historical data integration.

    Startup sequence:
    1. Initialize database connection
    2. Run gap filler to sync historical data
    3. Start WebSocket connection with DB writer
    4. Run paper tester with historical data support
    """

    def __init__(
        self,
        db_url: str,
        config_path: Optional[str] = None,
        skip_gap_fill: bool = False,
        symbols: Optional[list] = None,
        duration_minutes: int = 0,
        simulated: bool = False,
        enable_dashboard: bool = True,
    ):
        """
        Initialize HistoricalPaperTester.

        Args:
            db_url: PostgreSQL connection URL
            config_path: Path to config.yaml
            skip_gap_fill: Skip gap filling on startup
            symbols: List of symbols to trade
            duration_minutes: Run duration (0 = indefinite)
            simulated: Use simulated data instead of live WebSocket
            enable_dashboard: Enable web dashboard
        """
        self.db_url = db_url
        self.config_path = config_path
        self.skip_gap_fill = skip_gap_fill
        self.symbols = symbols
        self.duration_minutes = duration_minutes
        self.simulated = simulated
        self.enable_dashboard = enable_dashboard

        # Components (initialized in run())
        self.db_writer: Optional[DatabaseWriter] = None
        self.historical_provider: Optional[HistoricalDataProvider] = None
        self.tester: Optional[WebSocketPaperTester] = None
        self.integration = None

    async def run(self):
        """Main entry point with historical data integration."""
        logger.info("=" * 60)
        logger.info("Starting ws_paper_tester with Historical Data Support")
        logger.info("=" * 60)

        try:
            # =========================================
            # PHASE 1: Gap Filler (Startup Sync)
            # =========================================
            if not self.skip_gap_fill and HISTORICAL_AVAILABLE:
                logger.info("")
                logger.info("=" * 60)
                logger.info("PHASE 1: Running gap filler...")
                logger.info("=" * 60)

                try:
                    gap_results = await run_gap_filler(self.db_url, symbols=self.symbols)

                    if gap_results.get('errors'):
                        logger.warning(f"Gap filler completed with errors: {gap_results['errors']}")
                    else:
                        logger.info(f"Gap filler complete: {gap_results}")
                except Exception as e:
                    logger.warning(f"Gap filler failed (continuing anyway): {e}")
            else:
                logger.info("Skipping gap fill (--skip-gap-fill or historical module unavailable)")

            # =========================================
            # PHASE 2: Initialize Database Writer
            # =========================================
            if HISTORICAL_AVAILABLE and not self.simulated:
                logger.info("")
                logger.info("=" * 60)
                logger.info("PHASE 2: Starting database writer...")
                logger.info("=" * 60)

                try:
                    self.db_writer = DatabaseWriter(self.db_url)
                    await self.db_writer.start()
                    logger.info("Database writer started")
                except Exception as e:
                    logger.warning(f"Database writer failed (continuing without persistence): {e}")
                    self.db_writer = None

            # =========================================
            # PHASE 3: Initialize Historical Provider
            # =========================================
            if HISTORICAL_AVAILABLE:
                logger.info("")
                logger.info("=" * 60)
                logger.info("PHASE 3: Initializing historical data provider...")
                logger.info("=" * 60)

                try:
                    self.historical_provider = HistoricalDataProvider(self.db_url)
                    await self.historical_provider.connect()

                    # Log available data
                    health = await self.historical_provider.health_check()
                    logger.info(f"Historical data status:")
                    logger.info(f"  Symbols: {health.get('symbols', [])}")
                    logger.info(f"  Total candles: {health.get('total_candles', 0):,}")
                    logger.info(f"  Data range: {health.get('oldest_data')} to {health.get('newest_data')}")
                except Exception as e:
                    logger.warning(f"Historical provider failed (continuing without warmup): {e}")
                    self.historical_provider = None

            # =========================================
            # PHASE 4: Start Paper Tester
            # =========================================
            logger.info("")
            logger.info("=" * 60)
            logger.info("PHASE 4: Starting ws_paper_tester...")
            logger.info("=" * 60)

            if not TESTER_AVAILABLE:
                logger.error("WebSocketPaperTester not available")
                return

            # Load config
            config = load_config(self.config_path)

            # Initialize tester
            self.tester = WebSocketPaperTester(
                symbols=self.symbols,
                config=config,
                simulated=self.simulated,
                enable_dashboard=self.enable_dashboard,
            )

            # =========================================
            # PHASE 4b: Warm up strategies with historical data
            # =========================================
            if self.historical_provider and hasattr(self.tester, 'data_manager'):
                try:
                    logger.info("Loading historical candles for strategy warmup...")
                    total_loaded = 0

                    for symbol in self.symbols:
                        # Load 1m candles (400 = MAX_CANDLES in DataManager)
                        candles_1m = await self.historical_provider.get_latest_candles(
                            symbol, interval_minutes=1, count=400
                        )
                        if candles_1m:
                            loaded = await self.tester.data_manager.prefill_candles(
                                candles_1m, interval_minutes=1
                            )
                            total_loaded += loaded
                            logger.info(f"  {symbol}: Loaded {loaded} 1m candles")

                        # Load 5m candles (400 = ~33 hours of data)
                        candles_5m = await self.historical_provider.get_latest_candles(
                            symbol, interval_minutes=5, count=400
                        )
                        if candles_5m:
                            loaded = await self.tester.data_manager.prefill_candles(
                                candles_5m, interval_minutes=5
                            )
                            total_loaded += loaded
                            logger.info(f"  {symbol}: Loaded {loaded} 5m candles")

                    logger.info(f"Strategy warmup complete: {total_loaded:,} total candles loaded")
                except Exception as e:
                    logger.warning(f"Historical warmup failed (continuing anyway): {e}")

            # Integrate DB writer with WebSocket client (if available)
            if self.db_writer and not self.simulated:
                try:
                    ws_client = getattr(self.tester, 'ws_client', None)
                    if ws_client is None:
                        # WS client may be created lazily
                        pass
                    else:
                        self.integration = integrate_db_writer(ws_client, self.db_writer)
                        logger.info("Database writer integrated with WebSocket client")
                except Exception as e:
                    logger.warning(f"Could not integrate DB writer: {e}")

            # Run tester
            await self.tester.run(duration_minutes=self.duration_minutes)

        finally:
            # =========================================
            # CLEANUP
            # =========================================
            logger.info("")
            logger.info("=" * 60)
            logger.info("Shutting down...")
            logger.info("=" * 60)

            # Flush current candles
            if self.integration:
                try:
                    await self.integration.flush_current_candles()
                except Exception as e:
                    logger.error(f"Error flushing candles: {e}")

            # Stop database writer
            if self.db_writer:
                try:
                    await self.db_writer.stop()
                    stats = self.db_writer.get_stats()
                    logger.info(f"Database writer stats: {stats}")
                except Exception as e:
                    logger.error(f"Error stopping DB writer: {e}")

            # Close historical provider
            if self.historical_provider:
                try:
                    await self.historical_provider.close()
                except Exception as e:
                    logger.error(f"Error closing historical provider: {e}")

            logger.info("Shutdown complete")


async def main():
    """Entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description='WebSocket Paper Trading Tester with Historical Data Support'
    )
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config.yaml')
    # REC-004: No default password - require explicit configuration
    parser.add_argument('--db-url', type=str,
                        default=os.getenv('DATABASE_URL'),
                        help='PostgreSQL connection URL (required, or set DATABASE_URL env var)')
    parser.add_argument('--skip-gap-fill', action='store_true',
                        help='Skip gap filling on startup')
    parser.add_argument('--symbols', nargs='+', default=None,
                        help='Trading pairs to trade')
    parser.add_argument('--duration', type=int, default=0,
                        help='Run duration in minutes (0 = indefinite)')
    parser.add_argument('--simulated', action='store_true',
                        help='Use simulated data instead of live WebSocket')
    parser.add_argument('--no-dashboard', action='store_true',
                        help='Disable web dashboard')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')

    args = parser.parse_args()

    # REC-004: Require database URL - no default credentials
    if not args.db_url:
        parser.error(
            "--db-url or DATABASE_URL environment variable is required.\n"
            "Example: DATABASE_URL=postgresql://trading:YOUR_PASSWORD@localhost:5432/kraken_data"
        )

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create and run tester
    tester = HistoricalPaperTester(
        db_url=args.db_url,
        config_path=args.config,
        skip_gap_fill=args.skip_gap_fill,
        symbols=args.symbols,
        duration_minutes=args.duration,
        simulated=args.simulated,
        enable_dashboard=not args.no_dashboard,
    )

    await tester.run()


if __name__ == '__main__':
    asyncio.run(main())
