"""
Orchestration module - Agent communication and coordination.

Phase 3 Components:
- MessageBus: Inter-agent pub/sub communication
- CoordinatorAgent: Agent scheduling and conflict resolution
"""

from .message_bus import (
    Message,
    MessageBus,
    MessagePriority,
    MessageTopic,
    Subscription,
)
from .coordinator import (
    CoordinatorAgent,
    CoordinatorState,
    ConflictResolution,
    ScheduledTask,
)

__all__ = [
    'Message',
    'MessageBus',
    'MessagePriority',
    'MessageTopic',
    'Subscription',
    'CoordinatorAgent',
    'CoordinatorState',
    'ConflictResolution',
    'ScheduledTask',
]
