"""Tests for logger (HIGH-006)."""

import pytest
import tempfile
import time
import json
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from ws_tester.logger import TesterLogger, LogConfig, NullLogger
from ws_tester.types import Signal, Fill


class TestLogConfig:
    """Tests for LogConfig class."""

    def test_default_config(self):
        """Test default LogConfig values."""
        config = LogConfig()

        assert config.base_dir == "logs"
        assert config.compress is True
        assert config.max_file_size_mb == 100
        assert config.buffer_size == 100
        assert config.enable_aggregated is True
        assert config.console_output is True

    def test_custom_config(self):
        """Test custom LogConfig values."""
        config = LogConfig(
            base_dir="custom_logs",
            compress=False,
            max_file_size_mb=50,
            buffer_size=50
        )

        assert config.base_dir == "custom_logs"
        assert config.compress is False
        assert config.max_file_size_mb == 50

    def test_immediate_flush_events(self):
        """Test default immediate flush events."""
        config = LogConfig()

        assert 'fill' in config.immediate_flush_events
        assert 'trade' in config.immediate_flush_events
        assert 'error' in config.immediate_flush_events


class TestTesterLogger:
    """Tests for TesterLogger class."""

    @pytest.fixture
    def temp_logger(self):
        """Create a logger with temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LogConfig(
                base_dir=tmpdir,
                buffer_size=1,  # Small buffer for testing
                console_output=False
            )
            logger = TesterLogger("test_session", config)
            yield logger
            logger.close()

    def test_logger_creation(self, temp_logger):
        """Test logger initialization."""
        assert temp_logger.session_id == "test_session"
        assert temp_logger.sequence == 0

    def test_log_system(self, temp_logger):
        """Test system logging."""
        temp_logger.log_system("test_event", details={"key": "value"})

        # Wait for async writer
        time.sleep(0.2)

        # Check that log was written
        log_files = list(temp_logger.system_dir.glob("*.jsonl"))
        assert len(log_files) > 0

        # Read and verify content
        with open(log_files[0]) as f:
            content = f.read()
            assert "test_event" in content

    def test_log_signal(self, temp_logger):
        """Test signal logging."""
        signal = Signal(
            action='buy',
            symbol='XRP/USD',
            size=100.0,
            price=2.35,
            reason='Test signal'
        )

        correlation_id = temp_logger.log_signal(
            strategy='test_strategy',
            signal=signal,
            data_hash='abc123',
            indicators={'rsi': 50},
            latency_us=100
        )

        assert correlation_id is not None
        # Correlation ID is truncated, just check it starts with session prefix
        assert correlation_id.startswith('test_ses')

    def test_log_fill(self, temp_logger):
        """Test fill logging."""
        fill = Fill(
            fill_id='test123',
            timestamp=datetime.now(),
            symbol='XRP/USD',
            side='buy',
            size=100.0,
            price=2.35,
            fee=0.235,
            signal_reason='Test',
            pnl=5.0,
            strategy='test_strategy'
        )

        temp_logger.log_fill(
            fill=fill,
            correlation_id='corr123',
            strategy='test_strategy',
            portfolio={'usdt': 100.0}
        )

        # Wait for async writer
        time.sleep(0.2)

        # Check that log was written
        log_files = list(temp_logger.trades_dir.glob("*.jsonl"))
        assert len(log_files) > 0

    def test_log_error(self, temp_logger):
        """Test error logging."""
        temp_logger.log_error(
            strategy='test_strategy',
            error='Test error message'
        )

        # Wait for async writer
        time.sleep(0.2)

        # Check that log was written
        log_files = list(temp_logger.strategy_dir.glob("*.jsonl"))
        assert len(log_files) > 0

    def test_correlation_id_uniqueness(self, temp_logger):
        """Test that correlation IDs are unique."""
        ids = set()
        for _ in range(100):
            cid = temp_logger.log_signal(
                strategy='test',
                signal=None,
                data_hash='hash',
                indicators={},
                latency_us=0
            )
            ids.add(cid)

        assert len(ids) == 100  # All IDs should be unique

    def test_close_flushes_buffer(self):
        """Test that close() flushes the buffer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LogConfig(
                base_dir=tmpdir,
                buffer_size=1000,  # Large buffer
                console_output=False
            )
            logger = TesterLogger("test_session", config)

            # Log some events
            for i in range(10):
                logger.log_system(f"event_{i}")

            logger.close()

            # Check that logs were written
            log_files = list(Path(tmpdir).glob("**/*.jsonl"))
            assert len(log_files) > 0

            # Verify content - check that at least some events were written
            total_content_length = 0
            for log_file in log_files:
                with open(log_file) as f:
                    content = f.read()
                    total_content_length += len(content)

            # Just verify that content was written
            assert total_content_length > 0


class TestLogRotation:
    """Tests for log rotation functionality (HIGH-001/002)."""

    def test_file_size_tracking(self):
        """Test that file sizes are tracked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LogConfig(
                base_dir=tmpdir,
                buffer_size=1,
                console_output=False
            )
            logger = TesterLogger("test_session", config)

            # Log some events
            logger.log_system("test_event")
            time.sleep(0.2)

            # Check that file size is tracked
            assert 'system' in logger._file_sizes
            assert logger._file_sizes['system'] > 0

            logger.close()


class TestNullLogger:
    """Tests for NullLogger (no-op logger)."""

    def test_null_logger_creation(self):
        """Test NullLogger can be created."""
        logger = NullLogger()
        assert logger.session_id == "null"

    def test_null_logger_methods(self):
        """Test NullLogger methods don't raise errors."""
        logger = NullLogger()

        # These should all work without error
        logger.log_system("event")
        cid = logger.log_signal("strategy", None, "hash", {}, 0)
        assert cid is not None

        logger.log_fill(None, "corr", "strategy", {})
        logger.log_aggregated("corr", {}, "strategy", None, {}, {})
        logger.log_error("strategy", "error")
        logger.log_status(0, 0, 0, {}, [])
        logger.close()


class TestImmediateFlush:
    """Tests for immediate flush of critical events (HIGH-004)."""

    def test_fill_immediate_flush(self):
        """Test that fills are flushed immediately."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LogConfig(
                base_dir=tmpdir,
                buffer_size=1000,  # Large buffer
                console_output=False,
                immediate_flush_events={'fill'}
            )
            logger = TesterLogger("test_session", config)

            fill = Fill(
                fill_id='test123',
                timestamp=datetime.now(),
                symbol='XRP/USD',
                side='buy',
                size=100.0,
                price=2.35,
                fee=0.235,
                signal_reason='Test',
                pnl=5.0,
                strategy='test'
            )

            logger.log_fill(fill, "corr", "test", {})

            # Should be flushed quickly due to immediate flush
            time.sleep(0.5)

            log_files = list(Path(tmpdir).glob("**/fills_*.jsonl"))
            assert len(log_files) > 0

            # Verify content is written
            with open(log_files[0]) as f:
                content = f.read()
                assert len(content) > 0

            logger.close()


class TestStrategyFileManagement:
    """Tests for strategy file management (MED-011)."""

    def test_strategy_file_cleanup(self):
        """Test that strategy files are cleaned up on close."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LogConfig(
                base_dir=tmpdir,
                buffer_size=1,
                console_output=False
            )
            logger = TesterLogger("test_session", config)

            # Log to a strategy
            signal = Signal(
                action='buy',
                symbol='XRP/USD',
                size=100.0,
                price=2.35,
                reason='Test'
            )
            logger.log_signal("strategy1", signal, "hash", {}, 0)
            time.sleep(0.2)

            # Verify file was created
            assert 'strategy1' in logger._files

            # Close
            logger.close()

            # Verify files dict is cleared
            assert len(logger._files) == 0
