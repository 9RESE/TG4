"""
Structured logging for WebSocket Paper Trading Tester.
Provides separate log streams for system, strategies, and trades.

Features:
- Log rotation when file size exceeds limit
- Optional gzip compression of rotated logs
- Immediate flush for critical events (trades/fills)
- Graceful shutdown with buffer flush
"""

import atexit
import gzip
import json
import os
import shutil
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Set
from dataclasses import dataclass, asdict
from queue import Queue, Empty


@dataclass
class LogConfig:
    """Configuration for the logging system."""
    base_dir: str = "logs"
    compress: bool = True            # Gzip old logs
    max_file_size_mb: int = 100      # Rotate at 100MB
    buffer_size: int = 100           # Flush every N entries
    enable_aggregated: bool = True   # Write unified log
    console_output: bool = True      # Print to console
    max_rotated_files: int = 10      # Max number of rotated files to keep
    immediate_flush_events: Set[str] = None  # Events to flush immediately

    def __post_init__(self):
        """Set default immediate flush events."""
        if self.immediate_flush_events is None:
            self.immediate_flush_events = {'fill', 'trade', 'error'}


class TesterLogger:
    """
    Structured logging for WebSocket paper tester.

    Features:
    - Separate streams for system, strategies, trades
    - Correlation IDs to link related events
    - Buffered async writing with immediate flush for critical events
    - Log rotation when file size exceeds limit
    - Optional gzip compression of rotated logs
    - Graceful shutdown with buffer flush
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
        self.archive_dir = self.log_dir / "archive"

        for d in [self.system_dir, self.strategy_dir, self.trades_dir,
                  self.aggregated_dir, self.archive_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Track file sizes and paths for rotation
        self._file_sizes: Dict[str, int] = {}
        self._file_paths: Dict[str, Path] = {}
        self._files_to_close: list = []  # Track files for cleanup

        # Open file handles
        self._files: Dict[str, Any] = {}
        self._open_files()

        # Start async writer thread
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()

        # Register atexit handler for graceful shutdown
        atexit.register(self._atexit_handler)

    def _atexit_handler(self):
        """Ensure logs are flushed on exit."""
        if self._running:
            self.close()

    def _open_files(self):
        """Open log file handles."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        system_path = self.system_dir / f"ws_tester_{ts}.jsonl"
        trades_path = self.trades_dir / f"fills_{ts}.jsonl"

        self._files['system'] = open(system_path, 'a')
        self._file_paths['system'] = system_path
        self._file_sizes['system'] = system_path.stat().st_size if system_path.exists() else 0

        self._files['trades'] = open(trades_path, 'a')
        self._file_paths['trades'] = trades_path
        self._file_sizes['trades'] = trades_path.stat().st_size if trades_path.exists() else 0

        if self.config.enable_aggregated:
            agg_path = self.aggregated_dir / f"unified_{ts}.jsonl"
            self._files['aggregated'] = open(agg_path, 'a')
            self._file_paths['aggregated'] = agg_path
            self._file_sizes['aggregated'] = agg_path.stat().st_size if agg_path.exists() else 0

    def _get_strategy_file(self, strategy: str):
        """Get or create file handle for a strategy."""
        if strategy not in self._files:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            strategy_path = self.strategy_dir / f"{strategy}_{ts}.jsonl"
            self._files[strategy] = open(strategy_path, 'a')
            self._file_paths[strategy] = strategy_path
            self._file_sizes[strategy] = strategy_path.stat().st_size if strategy_path.exists() else 0
            self._files_to_close.append(strategy)  # Track for cleanup
        return self._files[strategy]

    def _rotate_file(self, stream: str):
        """Rotate a log file when it exceeds the size limit."""
        if stream not in self._file_paths:
            return

        old_path = self._file_paths[stream]
        old_file = self._files.get(stream)

        if old_file:
            try:
                old_file.flush()
                old_file.close()
            except (IOError, OSError):
                # File may already be closed or inaccessible during rotation
                pass

        # Create rotation filename with timestamp
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        rotated_name = f"{old_path.stem}_rotated_{ts}{old_path.suffix}"
        rotated_path = self.archive_dir / rotated_name

        # Move old file to archive
        try:
            shutil.move(str(old_path), str(rotated_path))
            print(f"[Logger] Rotated {old_path.name} -> {rotated_path.name}")

            # Compress if enabled
            if self.config.compress:
                self._compress_file(rotated_path)

            # Cleanup old rotated files
            self._cleanup_old_files(stream)

        except Exception as e:
            print(f"[Logger] Rotation error: {e}")

        # Open new file
        new_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if stream == 'system':
            new_path = self.system_dir / f"ws_tester_{new_ts}.jsonl"
        elif stream == 'trades':
            new_path = self.trades_dir / f"fills_{new_ts}.jsonl"
        elif stream == 'aggregated':
            new_path = self.aggregated_dir / f"unified_{new_ts}.jsonl"
        else:
            # Strategy file
            new_path = self.strategy_dir / f"{stream}_{new_ts}.jsonl"

        self._files[stream] = open(new_path, 'a')
        self._file_paths[stream] = new_path
        self._file_sizes[stream] = 0

    def _compress_file(self, filepath: Path):
        """Compress a file using gzip in a background thread."""
        def compress():
            try:
                gz_path = filepath.with_suffix(filepath.suffix + '.gz')
                with open(filepath, 'rb') as f_in:
                    with gzip.open(gz_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                os.remove(filepath)
                print(f"[Logger] Compressed {filepath.name}")
            except Exception as e:
                print(f"[Logger] Compression error: {e}")

        # Run compression in background thread
        threading.Thread(target=compress, daemon=True).start()

    def _cleanup_old_files(self, stream: str):
        """Remove old rotated files beyond the retention limit."""
        try:
            # Find all rotated files for this stream in archive
            pattern = f"*_rotated_*"
            rotated_files = sorted(
                self.archive_dir.glob(pattern),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )

            # Keep only the most recent files
            files_to_remove = rotated_files[self.config.max_rotated_files:]
            for f in files_to_remove:
                try:
                    os.remove(f)
                    print(f"[Logger] Removed old log: {f.name}")
                except (IOError, OSError, PermissionError):
                    # File may be locked or already removed
                    pass
        except (IOError, OSError) as e:
            print(f"[Logger] Cleanup error: {e}")

    def _writer_loop(self):
        """Background thread that writes log entries."""
        buffer = []
        immediate_buffer = []  # For critical events that need immediate flush

        while self._running:
            try:
                entry = self._write_queue.get(timeout=0.5)  # Reduced timeout for faster response
                stream, data = entry

                # Check if this is a critical event that needs immediate flush
                event_type = data.get('event', '')
                if event_type in self.config.immediate_flush_events:
                    immediate_buffer.append(entry)
                else:
                    buffer.append(entry)

                # Flush immediate buffer right away
                if immediate_buffer:
                    self._flush_buffer(immediate_buffer)
                    immediate_buffer = []

                # Flush regular buffer when full
                if len(buffer) >= self.config.buffer_size:
                    self._flush_buffer(buffer)
                    buffer = []

            except Empty:
                # Flush any remaining entries on timeout
                if buffer:
                    self._flush_buffer(buffer)
                    buffer = []
                if immediate_buffer:
                    self._flush_buffer(immediate_buffer)
                    immediate_buffer = []
            except Exception as e:
                print(f"[Logger] Writer error: {e}")

        # Final flush
        if buffer:
            self._flush_buffer(buffer)
        if immediate_buffer:
            self._flush_buffer(immediate_buffer)

    def _flush_buffer(self, buffer):
        """Write buffered entries to files with rotation check."""
        max_size_bytes = self.config.max_file_size_mb * 1024 * 1024

        for stream, data in buffer:
            try:
                f = self._files.get(stream) or self._get_strategy_file(stream)
                entry_json = json.dumps(data, default=str) + '\n'
                entry_bytes = len(entry_json.encode('utf-8'))

                f.write(entry_json)
                f.flush()

                # Update file size tracking
                self._file_sizes[stream] = self._file_sizes.get(stream, 0) + entry_bytes

                # Check for rotation
                if self._file_sizes.get(stream, 0) >= max_size_bytes:
                    self._rotate_file(stream)

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

        # Unregister atexit handler to prevent double close
        try:
            atexit.unregister(self._atexit_handler)
        except (TypeError, ValueError):
            # Handler may not be registered or already unregistered
            pass

        # Wait for writer thread
        if self._writer_thread.is_alive():
            self._writer_thread.join(timeout=5.0)

        # Close all files including strategy files
        closed_count = 0
        for name, f in list(self._files.items()):
            try:
                f.flush()
                f.close()
                closed_count += 1
            except (IOError, OSError, ValueError):
                # File may already be closed or invalid
                pass

        # Clear file tracking
        self._files.clear()
        self._file_paths.clear()
        self._file_sizes.clear()
        self._files_to_close.clear()

        print(f"[Logger] Closed {closed_count} log files")

    def remove_strategy(self, strategy: str):
        """
        Remove and close a strategy's log file.
        Call this when a strategy is removed at runtime.
        """
        if strategy in self._files:
            try:
                self._files[strategy].flush()
                self._files[strategy].close()
            except (IOError, OSError, ValueError):
                # File may already be closed
                pass
            del self._files[strategy]
            if strategy in self._file_paths:
                del self._file_paths[strategy]
            if strategy in self._file_sizes:
                del self._file_sizes[strategy]
            if strategy in self._files_to_close:
                self._files_to_close.remove(strategy)
            print(f"[Logger] Closed log file for strategy: {strategy}")


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
