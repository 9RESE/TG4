"""
Structured logging for WebSocket Paper Trading Tester.
Provides separate log streams for system, strategies, and trades.
"""

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from queue import Queue, Empty


@dataclass
class LogConfig:
    """Configuration for the logging system."""
    base_dir: str = "logs"
    compress: bool = False           # Gzip old logs
    max_file_size_mb: int = 100      # Rotate at 100MB
    buffer_size: int = 100           # Flush every N entries
    enable_aggregated: bool = True   # Write unified log
    console_output: bool = True      # Print to console


class TesterLogger:
    """
    Structured logging for WebSocket paper tester.

    Features:
    - Separate streams for system, strategies, trades
    - Correlation IDs to link related events
    - Buffered async writing
    - Optional aggregated view
    """

    def __init__(self, session_id: str, config: LogConfig = None):
        self.session_id = session_id
        self.config = config or LogConfig()
        self.sequence = 0
        self._lock = threading.Lock()
        self._write_queue: Queue = Queue()
        self._running = True

        # Create log directories
        self.log_dir = Path(self.config.base_dir)
        self.system_dir = self.log_dir / "system"
        self.strategy_dir = self.log_dir / "strategies"
        self.trades_dir = self.log_dir / "trades"
        self.aggregated_dir = self.log_dir / "aggregated"

        for d in [self.system_dir, self.strategy_dir, self.trades_dir, self.aggregated_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Open file handles
        self._files: Dict[str, Any] = {}
        self._open_files()

        # Start async writer thread
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()

    def _open_files(self):
        """Open log file handles."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._files['system'] = open(self.system_dir / f"ws_tester_{ts}.jsonl", 'a')
        self._files['trades'] = open(self.trades_dir / f"fills_{ts}.jsonl", 'a')
        if self.config.enable_aggregated:
            self._files['aggregated'] = open(self.aggregated_dir / f"unified_{ts}.jsonl", 'a')

    def _get_strategy_file(self, strategy: str):
        """Get or create file handle for a strategy."""
        if strategy not in self._files:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._files[strategy] = open(
                self.strategy_dir / f"{strategy}_{ts}.jsonl", 'a'
            )
        return self._files[strategy]

    def _writer_loop(self):
        """Background thread that writes log entries."""
        buffer = []
        while self._running:
            try:
                entry = self._write_queue.get(timeout=1.0)
                buffer.append(entry)

                if len(buffer) >= self.config.buffer_size:
                    self._flush_buffer(buffer)
                    buffer = []
            except Empty:
                if buffer:
                    self._flush_buffer(buffer)
                    buffer = []
            except Exception as e:
                print(f"[Logger] Writer error: {e}")

        # Final flush
        if buffer:
            self._flush_buffer(buffer)

    def _flush_buffer(self, buffer):
        """Write buffered entries to files."""
        for stream, data in buffer:
            try:
                f = self._files.get(stream) or self._get_strategy_file(stream)
                f.write(json.dumps(data, default=str) + '\n')
                f.flush()
            except Exception as e:
                print(f"[Logger] Write error: {e}")

    def _generate_correlation_id(self) -> str:
        """Generate unique correlation ID."""
        with self._lock:
            self.sequence += 1
            return f"{self.session_id[:8]}-{self.sequence:06d}"

    def _console_log(self, level: str, message: str):
        """Print to console if enabled."""
        if self.config.console_output:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            print(f"[{timestamp}] [{level}] {message}")

    def log_system(self, event: str, level: str = "INFO", details: Dict = None):
        """Log system event."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event": event,
            "level": level,
            "session_id": self.session_id,
            "details": details or {}
        }
        self._write_queue.put(('system', entry))

        if level in ("ERROR", "WARNING"):
            self._console_log(level, f"{event}: {details}")

    def log_signal(
        self,
        strategy: str,
        signal: Optional[Any],
        data_hash: str,
        indicators: Dict,
        latency_us: int
    ) -> str:
        """
        Log strategy signal. Returns correlation_id.

        Args:
            strategy: Strategy name
            signal: Signal object or None
            data_hash: Hash of data snapshot for reproducibility
            indicators: Strategy indicators at time of signal
            latency_us: Signal generation latency in microseconds
        """
        correlation_id = self._generate_correlation_id()

        signal_dict = None
        if signal:
            if hasattr(signal, '__dataclass_fields__'):
                signal_dict = asdict(signal)
            elif hasattr(signal, '__dict__'):
                signal_dict = signal.__dict__
            else:
                signal_dict = str(signal)

        entry = {
            "timestamp": datetime.now().isoformat(),
            "strategy": strategy,
            "event": "signal_generated" if signal else "no_signal",
            "correlation_id": correlation_id,
            "data_snapshot_hash": data_hash,
            "signal": signal_dict,
            "indicators": indicators,
            "latency_us": latency_us
        }
        self._write_queue.put((strategy, entry))

        return correlation_id

    def log_fill(
        self,
        fill: Any,
        correlation_id: str,
        strategy: str,
        portfolio: Dict,
        position: Dict = None
    ):
        """Log trade execution."""
        fill_dict = None
        if fill:
            if hasattr(fill, '__dataclass_fields__'):
                fill_dict = asdict(fill)
            elif hasattr(fill, '__dict__'):
                fill_dict = fill.__dict__
            else:
                fill_dict = str(fill)

        entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "fill",
            "correlation_id": correlation_id,
            "fill": fill_dict,
            "strategy": strategy,
            "portfolio_after": portfolio,
            "position": position
        }
        self._write_queue.put(('trades', entry))

        # Console output for fills
        if fill and self.config.console_output:
            side = fill.side if hasattr(fill, 'side') else 'unknown'
            symbol = fill.symbol if hasattr(fill, 'symbol') else 'unknown'
            price = fill.price if hasattr(fill, 'price') else 0
            pnl = fill.pnl if hasattr(fill, 'pnl') else 0
            pnl_str = f" P&L: ${pnl:+.2f}" if pnl != 0 else ""
            print(f"[FILL] [{strategy}] {side.upper()} {symbol} @ {price:.6f}{pnl_str}")

    def log_aggregated(
        self,
        correlation_id: str,
        data: Dict,
        strategy: str,
        signal: Any,
        execution: Dict,
        portfolio: Dict
    ):
        """Log unified aggregated entry."""
        if not self.config.enable_aggregated:
            return

        signal_dict = None
        if signal:
            if hasattr(signal, '__dataclass_fields__'):
                signal_dict = asdict(signal)
            elif hasattr(signal, '__dict__'):
                signal_dict = signal.__dict__
            else:
                signal_dict = str(signal)

        entry = {
            "timestamp": datetime.now().isoformat(),
            "correlation_id": correlation_id,
            "sequence": self.sequence,
            "data": data,
            "strategy": strategy,
            "signal": signal_dict,
            "execution": execution,
            "portfolio": portfolio
        }
        self._write_queue.put(('aggregated', entry))

    def log_error(self, strategy: str, error: str, details: Dict = None):
        """Log strategy error."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "strategy": strategy,
            "event": "error",
            "error": error,
            "details": details or {}
        }
        self._write_queue.put((strategy, entry))
        self._console_log("ERROR", f"[{strategy}] {error}")

    def log_status(
        self,
        tick_count: int,
        signal_count: int,
        fill_count: int,
        prices: Dict[str, float],
        portfolios: list
    ):
        """Log periodic status update."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "status",
            "tick_count": tick_count,
            "signal_count": signal_count,
            "fill_count": fill_count,
            "prices": prices,
            "portfolios": portfolios
        }
        self._write_queue.put(('system', entry))

    def close(self):
        """Flush and close all log files."""
        self._running = False

        # Wait for writer thread
        if self._writer_thread.is_alive():
            self._writer_thread.join(timeout=5.0)

        # Close files
        for name, f in self._files.items():
            try:
                f.flush()
                f.close()
            except Exception:
                pass

        print(f"[Logger] Closed all log files")


class NullLogger:
    """No-op logger for testing."""

    def __init__(self, *args, **kwargs):
        self.session_id = "null"
        self.sequence = 0

    def log_system(self, *args, **kwargs):
        pass

    def log_signal(self, *args, **kwargs) -> str:
        self.sequence += 1
        return f"null-{self.sequence:06d}"

    def log_fill(self, *args, **kwargs):
        pass

    def log_aggregated(self, *args, **kwargs):
        pass

    def log_error(self, *args, **kwargs):
        pass

    def log_status(self, *args, **kwargs):
        pass

    def close(self):
        pass
