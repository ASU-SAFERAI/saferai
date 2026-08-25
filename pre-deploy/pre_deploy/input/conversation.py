from typing import List, Dict, Hashable, Iterator, Union
from dataclasses import dataclass, field

from .message import Message
from .validation import ValidationUtilities
from .enums import Role


@dataclass
class Conversation:
    """
    Represents a conversation containing multiple messages.

    Attributes:
        id: Unique identifier for the conversation
        messages: List of messages in the conversation
        metadata: Additional metadata for the conversation
    """
    id: str
    messages: List[Message]
    metadata: Dict[str, Hashable] = field(default_factory=dict)

    def __post_init__(self):
        """Post-initialization validation for conversation attributes."""
        self.id = ValidationUtilities._validate_id(self.id, "id")
        self.messages = ValidationUtilities._validate_type(self.messages, list, "messages")
        self.metadata = ValidationUtilities._validate_type(self.metadata, dict, "metadata")

    @classmethod
    def from_dict(cls, data: Dict) -> 'Conversation':
        """Create a Conversation instance from a dictionary."""
        ValidationUtilities._validate_dict(data)
        required_keys = ['id', 'messages']
        ValidationUtilities._validate_keys_exist(data, required_keys)

        messages = [Message.from_dict(msg) for msg in data['messages']]
        return cls(
            id=data['id'],
            messages=messages,
            metadata=data.get('metadata', {})
        )

    def to_dict(self) -> Dict:
        """Convert the Conversation instance to a dictionary."""
        return {
            'id': self.id,
            'messages': [message.to_dict() for message in self.messages],
            'metadata': self.metadata
        }

    def __iter__(self) -> Iterator[Message]:
        """Make Conversation iterable, yielding each message."""
        return iter(self.messages)

    def __getitem__(self, index: Union[int, slice]) -> Union[Message, List[Message]]:
        """Allow indexing and slicing to get messages."""
        return self.messages[index]

    def __len__(self) -> int:
        """Return the number of messages in the conversation."""
        return len(self.messages)

    def __str__(self) -> str:
        """Return a human-readable string representation."""
        return f"Conversation(id='{self.id}', messages={len(self.messages)})"

    def __repr__(self) -> str:
        """Return a detailed string representation for debugging."""
        return (f"Conversation(id='{self.id}', "
                f"messages_count={len(self.messages)}, "
                f"metadata_keys={list(self.metadata.keys())})")
