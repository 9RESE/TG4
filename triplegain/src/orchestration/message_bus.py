"""
Message Bus - Inter-agent communication with pub/sub pattern.

Thread-safe async implementation for agent coordination.
Features:
- Pub/sub pattern with topic-based routing
- Message priority levels
- TTL-based message expiration
- Subscription filtering
- Message history with cleanup
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Optional, Awaitable

logger = logging.getLogger(__name__)


class MessagePriority(Enum):
    """Message priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


class MessageTopic(Enum):
    """Standard message topics for inter-agent communication."""
    MARKET_DATA = "market_data"
    TA_SIGNALS = "ta_signals"
    REGIME_UPDATES = "regime_updates"
    SENTIMENT_UPDATES = "sentiment_updates"
    TRADING_SIGNALS = "trading_signals"
    RISK_ALERTS = "risk_alerts"
    EXECUTION_EVENTS = "execution_events"
    PORTFOLIO_UPDATES = "portfolio_updates"
    SYSTEM_EVENTS = "system_events"
    COORDINATOR_COMMANDS = "coordinator_commands"


@dataclass
class Message:
    """
    Standard message format for inter-agent communication.

    All messages include:
    - Unique identifier
    - Timestamp and TTL
    - Source agent and topic
    - Priority level
    - Payload data
    - Optional correlation ID for request-response patterns
    """
    topic: MessageTopic
    source: str
    payload: dict = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    priority: MessagePriority = MessagePriority.NORMAL
    correlation_id: Optional[str] = None
    ttl_seconds: int = 300

    def is_expired(self) -> bool:
        """Check if message has exceeded its TTL."""
        now = datetime.now(timezone.utc)
        age = (now - self.timestamp).total_seconds()
        return age > self.ttl_seconds

    def to_dict(self) -> dict:
        """Serialize message to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "topic": self.topic.value,
            "source": self.source,
            "priority": self.priority.value,
            "payload": self.payload,
            "correlation_id": self.correlation_id,
            "ttl_seconds": self.ttl_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Message':
        """Deserialize message from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else datetime.now(timezone.utc),
            topic=MessageTopic(data["topic"]),
            source=data["source"],
            priority=MessagePriority(data.get("priority", 2)),
            payload=data.get("payload", {}),
            correlation_id=data.get("correlation_id"),
            ttl_seconds=data.get("ttl_seconds", 300),
        )


@dataclass
class Subscription:
    """Subscription to a message topic."""
    subscriber_id: str
    topic: MessageTopic
    handler: Callable[[Message], Awaitable[None]]
    filter_fn: Optional[Callable[[Message], bool]] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def matches(self, message: Message) -> bool:
        """Check if message matches subscription filter."""
        if self.filter_fn is None:
            return True
        try:
            return self.filter_fn(message)
        except Exception as e:
            logger.error(f"Subscription filter error for {self.subscriber_id}: {e}")
            return False


class MessageBus:
    """
    In-memory message bus for agent communication.

    Thread-safe async implementation with pub/sub pattern.

    Features:
    - Topic-based message routing
    - Priority-aware delivery (higher priority first)
    - TTL-based message expiration
    - Subscription filtering
    - Message history for late subscribers
    """

    def __init__(self, config: Optional[dict] = None):
        """
        Initialize MessageBus.

        Args:
            config: Optional configuration with:
                - max_history_size: Maximum messages to keep in history
                - cleanup_interval_seconds: How often to clean expired messages
                - enable_persistence: Whether to persist messages to database
        """
        self.config = config or {}
        self._subscriptions: dict[MessageTopic, list[Subscription]] = {}
        self._message_history: list[Message] = []
        self._lock = asyncio.Lock()

        # Configuration
        self._max_history_size = self.config.get('max_history_size', 1000)
        self._cleanup_interval = self.config.get('cleanup_interval_seconds', 60)

        # Statistics
        self._total_published = 0
        self._total_delivered = 0
        self._delivery_errors = 0

        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        """Start the message bus and cleanup task."""
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("MessageBus started")

    async def stop(self) -> None:
        """Stop the message bus and cleanup task."""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("MessageBus stopped")

    async def publish(self, message: Message) -> int:
        """
        Publish a message to all subscribers of the topic.

        Thread-safe: Handlers are called OUTSIDE the lock to prevent deadlock
        if a handler tries to publish a message.

        Args:
            message: Message to publish

        Returns:
            Number of subscribers notified
        """
        # Phase 1: Under lock - update history and get subscribers snapshot
        async with self._lock:
            # Store in history
            self._message_history.append(message)
            self._total_published += 1

            # Trim history if too large
            if len(self._message_history) > self._max_history_size:
                self._message_history = self._message_history[-self._max_history_size:]

            # Get COPY of subscriptions to avoid holding lock during callbacks
            # This prevents deadlock if handler tries to publish/subscribe
            subscriptions = list(self._subscriptions.get(message.topic, []))

        # Phase 2: Outside lock - call handlers
        # CRITICAL: Handlers are called without holding the lock
        # This allows handlers to publish messages without deadlock
        delivered = 0
        for sub in subscriptions:
            if sub.matches(message):
                try:
                    await sub.handler(message)
                    delivered += 1
                    # Update stats under lock
                    async with self._lock:
                        self._total_delivered += 1
                except Exception as e:
                    async with self._lock:
                        self._delivery_errors += 1
                    logger.error(
                        f"Error delivering message to {sub.subscriber_id}: {e}",
                        exc_info=True
                    )

        logger.debug(
            f"Published message {message.id[:8]} to {message.topic.value}: "
            f"{delivered} subscribers notified"
        )

        return delivered

    async def subscribe(
        self,
        subscriber_id: str,
        topic: MessageTopic,
        handler: Callable[[Message], Awaitable[None]],
        filter_fn: Optional[Callable[[Message], bool]] = None,
    ) -> str:
        """
        Subscribe to a message topic.

        Args:
            subscriber_id: Unique identifier for the subscriber
            topic: Topic to subscribe to
            handler: Async function to call when message received
            filter_fn: Optional filter function to filter messages

        Returns:
            Subscription ID
        """
        subscription = Subscription(
            subscriber_id=subscriber_id,
            topic=topic,
            handler=handler,
            filter_fn=filter_fn,
        )

        async with self._lock:
            if topic not in self._subscriptions:
                self._subscriptions[topic] = []

            # Remove existing subscription for same subscriber/topic
            self._subscriptions[topic] = [
                s for s in self._subscriptions[topic]
                if s.subscriber_id != subscriber_id
            ]

            self._subscriptions[topic].append(subscription)

        logger.debug(f"Subscriber {subscriber_id} subscribed to {topic.value}")
        return f"{subscriber_id}:{topic.value}"

    async def unsubscribe(
        self,
        subscriber_id: str,
        topic: Optional[MessageTopic] = None,
    ) -> int:
        """
        Unsubscribe from topic(s).

        Args:
            subscriber_id: Subscriber identifier
            topic: Specific topic to unsubscribe from, or None for all

        Returns:
            Number of subscriptions removed
        """
        removed = 0
        async with self._lock:
            if topic:
                # Unsubscribe from specific topic
                if topic in self._subscriptions:
                    original_count = len(self._subscriptions[topic])
                    self._subscriptions[topic] = [
                        s for s in self._subscriptions[topic]
                        if s.subscriber_id != subscriber_id
                    ]
                    removed = original_count - len(self._subscriptions[topic])
            else:
                # Unsubscribe from all topics
                for t in self._subscriptions:
                    original_count = len(self._subscriptions[t])
                    self._subscriptions[t] = [
                        s for s in self._subscriptions[t]
                        if s.subscriber_id != subscriber_id
                    ]
                    removed += original_count - len(self._subscriptions[t])

        logger.debug(f"Subscriber {subscriber_id} unsubscribed: {removed} removed")
        return removed

    async def get_latest(
        self,
        topic: MessageTopic,
        source: Optional[str] = None,
        max_age_seconds: Optional[int] = None,
    ) -> Optional[Message]:
        """
        Get the most recent message for a topic.

        Args:
            topic: Message topic
            source: Optional source agent filter
            max_age_seconds: Maximum message age (None = no limit)

        Returns:
            Most recent matching message or None
        """
        async with self._lock:
            for msg in reversed(self._message_history):
                if msg.topic != topic:
                    continue

                if source is not None and msg.source != source:
                    continue

                if max_age_seconds is not None:
                    age = (datetime.now(timezone.utc) - msg.timestamp).total_seconds()
                    if age > max_age_seconds:
                        continue

                if not msg.is_expired():
                    return msg

            return None

    async def get_history(
        self,
        topic: Optional[MessageTopic] = None,
        source: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[Message]:
        """
        Get message history with filtering.

        Args:
            topic: Optional topic filter
            source: Optional source filter
            since: Only messages after this time
            limit: Maximum messages to return

        Returns:
            List of matching messages (newest first)
        """
        async with self._lock:
            results = []
            for msg in reversed(self._message_history):
                if len(results) >= limit:
                    break

                if topic is not None and msg.topic != topic:
                    continue

                if source is not None and msg.source != source:
                    continue

                if since is not None and msg.timestamp < since:
                    continue

                if not msg.is_expired():
                    results.append(msg)

            return results

    async def get_subscriber_count(self, topic: Optional[MessageTopic] = None) -> int:
        """Get number of subscribers for a topic or all topics."""
        async with self._lock:
            if topic:
                return len(self._subscriptions.get(topic, []))
            return sum(len(subs) for subs in self._subscriptions.values())

    def get_stats(self) -> dict:
        """Get message bus statistics."""
        return {
            "total_published": self._total_published,
            "total_delivered": self._total_delivered,
            "delivery_errors": self._delivery_errors,
            "history_size": len(self._message_history),
            "subscriber_count": sum(len(s) for s in self._subscriptions.values()),
            "topics_active": len([t for t, s in self._subscriptions.items() if s]),
        }

    async def _cleanup_loop(self) -> None:
        """Background task to clean up expired messages."""
        while self._running:
            try:
                await asyncio.sleep(self._cleanup_interval)
                await self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup error: {e}")

    async def _cleanup_expired(self) -> int:
        """Remove expired messages from history."""
        async with self._lock:
            original_count = len(self._message_history)
            self._message_history = [
                m for m in self._message_history
                if not m.is_expired()
            ]
            removed = original_count - len(self._message_history)

        if removed > 0:
            logger.debug(f"Cleaned up {removed} expired messages")

        return removed

    async def clear_history(self, topic: Optional[MessageTopic] = None) -> int:
        """
        Clear message history.

        Args:
            topic: Clear only this topic, or None for all

        Returns:
            Number of messages cleared
        """
        async with self._lock:
            if topic:
                original = len(self._message_history)
                self._message_history = [
                    m for m in self._message_history
                    if m.topic != topic
                ]
                return original - len(self._message_history)
            else:
                count = len(self._message_history)
                self._message_history = []
                return count


# Convenience function for creating messages
def create_message(
    topic: MessageTopic,
    source: str,
    payload: dict,
    priority: MessagePriority = MessagePriority.NORMAL,
    ttl_seconds: int = 300,
    correlation_id: Optional[str] = None,
) -> Message:
    """Create a new message with default values."""
    return Message(
        topic=topic,
        source=source,
        payload=payload,
        priority=priority,
        ttl_seconds=ttl_seconds,
        correlation_id=correlation_id,
    )
