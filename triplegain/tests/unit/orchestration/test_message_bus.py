"""
Unit tests for MessageBus - Inter-agent communication.

Tests cover:
- Message creation and serialization
- Pub/sub pattern
- Subscription filtering
- TTL expiration
- Message history
- Thread safety
"""

import asyncio
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from triplegain.src.orchestration.message_bus import (
    Message,
    MessageBus,
    MessagePriority,
    MessageTopic,
    Subscription,
    create_message,
)


class TestMessage:
    """Tests for Message dataclass."""

    def test_message_creation(self):
        """Test basic message creation."""
        msg = Message(
            topic=MessageTopic.TA_SIGNALS,
            source="technical_analysis",
            payload={"signal": "buy"},
        )
        assert msg.topic == MessageTopic.TA_SIGNALS
        assert msg.source == "technical_analysis"
        assert msg.payload == {"signal": "buy"}
        assert msg.priority == MessagePriority.NORMAL
        assert msg.id is not None
        assert msg.timestamp is not None

    def test_message_with_priority(self):
        """Test message with custom priority."""
        msg = Message(
            topic=MessageTopic.RISK_ALERTS,
            source="risk_engine",
            payload={"alert": "circuit_breaker"},
            priority=MessagePriority.URGENT,
        )
        assert msg.priority == MessagePriority.URGENT

    def test_message_with_correlation_id(self):
        """Test message with correlation ID for request-response."""
        correlation_id = "request-123"
        msg = Message(
            topic=MessageTopic.TRADING_SIGNALS,
            source="coordinator",
            payload={},
            correlation_id=correlation_id,
        )
        assert msg.correlation_id == correlation_id

    def test_message_ttl(self):
        """Test message TTL."""
        msg = Message(
            topic=MessageTopic.MARKET_DATA,
            source="data_feed",
            payload={},
            ttl_seconds=60,
        )
        assert msg.ttl_seconds == 60

    def test_message_is_expired_false(self):
        """Test fresh message is not expired."""
        msg = Message(
            topic=MessageTopic.MARKET_DATA,
            source="data_feed",
            payload={},
            ttl_seconds=300,
        )
        assert msg.is_expired() is False

    def test_message_is_expired_true(self):
        """Test old message is expired."""
        msg = Message(
            topic=MessageTopic.MARKET_DATA,
            source="data_feed",
            payload={},
            timestamp=datetime.now(timezone.utc) - timedelta(seconds=400),
            ttl_seconds=300,
        )
        assert msg.is_expired() is True

    def test_message_to_dict(self):
        """Test message serialization to dictionary."""
        msg = Message(
            topic=MessageTopic.TA_SIGNALS,
            source="technical_analysis",
            payload={"signal": "buy", "confidence": 0.8},
        )
        d = msg.to_dict()
        assert d["topic"] == "ta_signals"
        assert d["source"] == "technical_analysis"
        assert d["payload"] == {"signal": "buy", "confidence": 0.8}
        assert "id" in d
        assert "timestamp" in d

    def test_message_from_dict(self):
        """Test message deserialization from dictionary."""
        data = {
            "id": "test-123",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "topic": "ta_signals",
            "source": "technical_analysis",
            "priority": 2,
            "payload": {"signal": "buy"},
            "ttl_seconds": 300,
        }
        msg = Message.from_dict(data)
        assert msg.id == "test-123"
        assert msg.topic == MessageTopic.TA_SIGNALS
        assert msg.source == "technical_analysis"
        assert msg.payload == {"signal": "buy"}


class TestSubscription:
    """Tests for Subscription dataclass."""

    @pytest.mark.asyncio
    async def test_subscription_creation(self):
        """Test subscription creation."""
        async def handler(msg):
            pass

        sub = Subscription(
            subscriber_id="test-subscriber",
            topic=MessageTopic.TA_SIGNALS,
            handler=handler,
        )
        assert sub.subscriber_id == "test-subscriber"
        assert sub.topic == MessageTopic.TA_SIGNALS

    def test_subscription_matches_no_filter(self):
        """Test subscription matches without filter."""
        async def handler(msg):
            pass

        sub = Subscription(
            subscriber_id="test",
            topic=MessageTopic.TA_SIGNALS,
            handler=handler,
        )
        msg = Message(topic=MessageTopic.TA_SIGNALS, source="test", payload={})
        assert sub.matches(msg) is True

    def test_subscription_matches_with_filter(self):
        """Test subscription matches with filter function."""
        async def handler(msg):
            pass

        def filter_fn(msg):
            return msg.payload.get("symbol") == "BTC/USDT"

        sub = Subscription(
            subscriber_id="test",
            topic=MessageTopic.TA_SIGNALS,
            handler=handler,
            filter_fn=filter_fn,
        )

        msg_btc = Message(
            topic=MessageTopic.TA_SIGNALS,
            source="test",
            payload={"symbol": "BTC/USDT"},
        )
        msg_xrp = Message(
            topic=MessageTopic.TA_SIGNALS,
            source="test",
            payload={"symbol": "XRP/USDT"},
        )

        assert sub.matches(msg_btc) is True
        assert sub.matches(msg_xrp) is False

    def test_subscription_filter_error_returns_false(self):
        """Test subscription filter error returns False."""
        async def handler(msg):
            pass

        def bad_filter(msg):
            raise ValueError("Filter error")

        sub = Subscription(
            subscriber_id="test",
            topic=MessageTopic.TA_SIGNALS,
            handler=handler,
            filter_fn=bad_filter,
        )
        msg = Message(topic=MessageTopic.TA_SIGNALS, source="test", payload={})
        assert sub.matches(msg) is False


class TestMessageBus:
    """Tests for MessageBus."""

    @pytest.fixture
    def message_bus(self):
        """Create a MessageBus instance for testing."""
        return MessageBus({"max_history_size": 100, "cleanup_interval_seconds": 60})

    @pytest.mark.asyncio
    async def test_publish_no_subscribers(self, message_bus):
        """Test publishing with no subscribers."""
        msg = create_message(
            topic=MessageTopic.TA_SIGNALS,
            source="test",
            payload={"signal": "buy"},
        )
        delivered = await message_bus.publish(msg)
        assert delivered == 0

    @pytest.mark.asyncio
    async def test_subscribe_and_publish(self, message_bus):
        """Test subscribe and publish workflow."""
        received_messages = []

        async def handler(msg):
            received_messages.append(msg)

        await message_bus.subscribe(
            subscriber_id="test-sub",
            topic=MessageTopic.TA_SIGNALS,
            handler=handler,
        )

        msg = create_message(
            topic=MessageTopic.TA_SIGNALS,
            source="test",
            payload={"signal": "buy"},
        )
        delivered = await message_bus.publish(msg)

        assert delivered == 1
        assert len(received_messages) == 1
        assert received_messages[0].payload == {"signal": "buy"}

    @pytest.mark.asyncio
    async def test_multiple_subscribers(self, message_bus):
        """Test publishing to multiple subscribers."""
        received_1 = []
        received_2 = []

        async def handler1(msg):
            received_1.append(msg)

        async def handler2(msg):
            received_2.append(msg)

        await message_bus.subscribe("sub1", MessageTopic.TA_SIGNALS, handler1)
        await message_bus.subscribe("sub2", MessageTopic.TA_SIGNALS, handler2)

        msg = create_message(
            topic=MessageTopic.TA_SIGNALS,
            source="test",
            payload={},
        )
        delivered = await message_bus.publish(msg)

        assert delivered == 2
        assert len(received_1) == 1
        assert len(received_2) == 1

    @pytest.mark.asyncio
    async def test_filtered_subscription(self, message_bus):
        """Test subscription with filter."""
        received = []

        async def handler(msg):
            received.append(msg)

        def only_urgent(msg):
            return msg.priority == MessagePriority.URGENT

        await message_bus.subscribe(
            subscriber_id="urgent-only",
            topic=MessageTopic.RISK_ALERTS,
            handler=handler,
            filter_fn=only_urgent,
        )

        # Normal priority - should not be received
        msg1 = create_message(
            topic=MessageTopic.RISK_ALERTS,
            source="test",
            payload={"type": "warning"},
        )
        await message_bus.publish(msg1)

        # Urgent priority - should be received
        msg2 = create_message(
            topic=MessageTopic.RISK_ALERTS,
            source="test",
            payload={"type": "critical"},
            priority=MessagePriority.URGENT,
        )
        await message_bus.publish(msg2)

        assert len(received) == 1
        assert received[0].payload["type"] == "critical"

    @pytest.mark.asyncio
    async def test_unsubscribe_from_topic(self, message_bus):
        """Test unsubscribing from a topic."""
        received = []

        async def handler(msg):
            received.append(msg)

        await message_bus.subscribe("test", MessageTopic.TA_SIGNALS, handler)

        # First message - should be received
        msg1 = create_message(topic=MessageTopic.TA_SIGNALS, source="test", payload={})
        await message_bus.publish(msg1)
        assert len(received) == 1

        # Unsubscribe
        removed = await message_bus.unsubscribe("test", MessageTopic.TA_SIGNALS)
        assert removed == 1

        # Second message - should not be received
        msg2 = create_message(topic=MessageTopic.TA_SIGNALS, source="test", payload={})
        await message_bus.publish(msg2)
        assert len(received) == 1  # Still 1

    @pytest.mark.asyncio
    async def test_unsubscribe_all_topics(self, message_bus):
        """Test unsubscribing from all topics."""
        async def handler(msg):
            pass

        await message_bus.subscribe("test", MessageTopic.TA_SIGNALS, handler)
        await message_bus.subscribe("test", MessageTopic.RISK_ALERTS, handler)

        removed = await message_bus.unsubscribe("test")
        assert removed == 2

    @pytest.mark.asyncio
    async def test_get_latest_message(self, message_bus):
        """Test getting the latest message."""
        msg1 = create_message(
            topic=MessageTopic.TA_SIGNALS,
            source="ta1",
            payload={"signal": "first"},
        )
        msg2 = create_message(
            topic=MessageTopic.TA_SIGNALS,
            source="ta2",
            payload={"signal": "second"},
        )

        await message_bus.publish(msg1)
        await message_bus.publish(msg2)

        latest = await message_bus.get_latest(MessageTopic.TA_SIGNALS)
        assert latest is not None
        assert latest.payload["signal"] == "second"

    @pytest.mark.asyncio
    async def test_get_latest_with_source_filter(self, message_bus):
        """Test getting latest message filtered by source."""
        msg1 = create_message(
            topic=MessageTopic.TA_SIGNALS,
            source="agent1",
            payload={"from": "agent1"},
        )
        msg2 = create_message(
            topic=MessageTopic.TA_SIGNALS,
            source="agent2",
            payload={"from": "agent2"},
        )

        await message_bus.publish(msg1)
        await message_bus.publish(msg2)

        latest = await message_bus.get_latest(MessageTopic.TA_SIGNALS, source="agent1")
        assert latest is not None
        assert latest.payload["from"] == "agent1"

    @pytest.mark.asyncio
    async def test_get_latest_with_max_age(self, message_bus):
        """Test getting latest message with max age filter."""
        # Old message
        old_msg = Message(
            topic=MessageTopic.TA_SIGNALS,
            source="test",
            payload={"age": "old"},
            timestamp=datetime.now(timezone.utc) - timedelta(seconds=120),
            ttl_seconds=300,
        )
        await message_bus.publish(old_msg)

        # Should not find message older than 60 seconds
        latest = await message_bus.get_latest(MessageTopic.TA_SIGNALS, max_age_seconds=60)
        assert latest is None

    @pytest.mark.asyncio
    async def test_get_history(self, message_bus):
        """Test getting message history."""
        for i in range(5):
            msg = create_message(
                topic=MessageTopic.TA_SIGNALS,
                source="test",
                payload={"index": i},
            )
            await message_bus.publish(msg)

        history = await message_bus.get_history(MessageTopic.TA_SIGNALS, limit=3)
        assert len(history) == 3
        # Newest first
        assert history[0].payload["index"] == 4

    @pytest.mark.asyncio
    async def test_get_subscriber_count(self, message_bus):
        """Test getting subscriber count."""
        async def handler(msg):
            pass

        await message_bus.subscribe("sub1", MessageTopic.TA_SIGNALS, handler)
        await message_bus.subscribe("sub2", MessageTopic.TA_SIGNALS, handler)
        await message_bus.subscribe("sub3", MessageTopic.RISK_ALERTS, handler)

        # Per topic
        ta_count = await message_bus.get_subscriber_count(MessageTopic.TA_SIGNALS)
        assert ta_count == 2

        # All topics
        total_count = await message_bus.get_subscriber_count()
        assert total_count == 3

    @pytest.mark.asyncio
    async def test_message_history_trimming(self, message_bus):
        """Test message history is trimmed when exceeding max size."""
        bus = MessageBus({"max_history_size": 5})

        for i in range(10):
            msg = create_message(
                topic=MessageTopic.MARKET_DATA,
                source="test",
                payload={"index": i},
            )
            await bus.publish(msg)

        history = await bus.get_history(limit=100)
        assert len(history) == 5

    @pytest.mark.asyncio
    async def test_clear_history_all(self, message_bus):
        """Test clearing all message history."""
        for i in range(5):
            msg = create_message(
                topic=MessageTopic.TA_SIGNALS,
                source="test",
                payload={},
            )
            await message_bus.publish(msg)

        cleared = await message_bus.clear_history()
        assert cleared == 5

        history = await message_bus.get_history()
        assert len(history) == 0

    @pytest.mark.asyncio
    async def test_clear_history_by_topic(self, message_bus):
        """Test clearing history for a specific topic."""
        for i in range(3):
            msg1 = create_message(topic=MessageTopic.TA_SIGNALS, source="test", payload={})
            msg2 = create_message(topic=MessageTopic.RISK_ALERTS, source="test", payload={})
            await message_bus.publish(msg1)
            await message_bus.publish(msg2)

        cleared = await message_bus.clear_history(MessageTopic.TA_SIGNALS)
        assert cleared == 3

        remaining = await message_bus.get_history()
        assert len(remaining) == 3
        assert all(m.topic == MessageTopic.RISK_ALERTS for m in remaining)

    @pytest.mark.asyncio
    async def test_handler_error_does_not_stop_delivery(self, message_bus):
        """Test that handler error doesn't stop delivery to other subscribers."""
        received = []

        async def bad_handler(msg):
            raise ValueError("Handler error")

        async def good_handler(msg):
            received.append(msg)

        await message_bus.subscribe("bad", MessageTopic.TA_SIGNALS, bad_handler)
        await message_bus.subscribe("good", MessageTopic.TA_SIGNALS, good_handler)

        msg = create_message(topic=MessageTopic.TA_SIGNALS, source="test", payload={})
        delivered = await message_bus.publish(msg)

        # One delivery failed, one succeeded
        assert delivered == 1
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_stats(self, message_bus):
        """Test getting message bus statistics."""
        async def handler(msg):
            pass

        await message_bus.subscribe("sub", MessageTopic.TA_SIGNALS, handler)

        msg = create_message(topic=MessageTopic.TA_SIGNALS, source="test", payload={})
        await message_bus.publish(msg)

        stats = message_bus.get_stats()
        assert stats["total_published"] == 1
        assert stats["total_delivered"] == 1
        assert stats["history_size"] == 1
        assert stats["subscriber_count"] == 1

    @pytest.mark.asyncio
    async def test_start_stop(self, message_bus):
        """Test starting and stopping the message bus."""
        await message_bus.start()
        assert message_bus._running is True
        assert message_bus._cleanup_task is not None

        await message_bus.stop()
        assert message_bus._running is False

    @pytest.mark.asyncio
    async def test_cleanup_expired_messages(self, message_bus):
        """Test cleanup of expired messages."""
        # Add an expired message
        expired_msg = Message(
            topic=MessageTopic.MARKET_DATA,
            source="test",
            payload={},
            timestamp=datetime.now(timezone.utc) - timedelta(seconds=400),
            ttl_seconds=300,
        )
        await message_bus.publish(expired_msg)

        # Add a fresh message
        fresh_msg = create_message(
            topic=MessageTopic.MARKET_DATA,
            source="test",
            payload={},
        )
        await message_bus.publish(fresh_msg)

        # Run cleanup
        removed = await message_bus._cleanup_expired()
        assert removed == 1

        history = await message_bus.get_history()
        assert len(history) == 1


class TestCreateMessage:
    """Tests for create_message helper function."""

    def test_create_message_basic(self):
        """Test basic message creation with helper."""
        msg = create_message(
            topic=MessageTopic.TA_SIGNALS,
            source="test_agent",
            payload={"signal": "buy"},
        )
        assert msg.topic == MessageTopic.TA_SIGNALS
        assert msg.source == "test_agent"
        assert msg.payload == {"signal": "buy"}
        assert msg.priority == MessagePriority.NORMAL
        assert msg.ttl_seconds == 300

    def test_create_message_with_options(self):
        """Test message creation with all options."""
        msg = create_message(
            topic=MessageTopic.RISK_ALERTS,
            source="risk_engine",
            payload={"alert": "circuit_breaker"},
            priority=MessagePriority.URGENT,
            ttl_seconds=60,
            correlation_id="request-456",
        )
        assert msg.priority == MessagePriority.URGENT
        assert msg.ttl_seconds == 60
        assert msg.correlation_id == "request-456"


class TestMessageTopic:
    """Tests for MessageTopic enum."""

    def test_all_topics_have_string_value(self):
        """Test all topics have string values."""
        for topic in MessageTopic:
            assert isinstance(topic.value, str)
            assert len(topic.value) > 0

    def test_expected_topics_exist(self):
        """Test expected topics exist."""
        expected = [
            "market_data",
            "ta_signals",
            "regime_updates",
            "sentiment_updates",
            "trading_signals",
            "risk_alerts",
            "execution_events",
            "portfolio_updates",
        ]
        topic_values = [t.value for t in MessageTopic]
        for expected_topic in expected:
            assert expected_topic in topic_values


class TestMessagePriority:
    """Tests for MessagePriority enum."""

    def test_priority_ordering(self):
        """Test priority levels have correct ordering."""
        assert MessagePriority.LOW.value < MessagePriority.NORMAL.value
        assert MessagePriority.NORMAL.value < MessagePriority.HIGH.value
        assert MessagePriority.HIGH.value < MessagePriority.URGENT.value
