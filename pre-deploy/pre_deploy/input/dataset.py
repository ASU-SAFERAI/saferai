from dataclasses import dataclass, field
from typing import List, Dict, Hashable, Iterator, Union
import random
from .conversation import Conversation

from .validation import ValidationUtilities


@dataclass
class EvalDataset:
    """
    Represents a dataset containing multiple conversations.

    Attributes:
        id: Unique identifier for the dataset
        conversations: List of conversations in the dataset
        metadata: Additional metadata for the dataset
    """
    id: str
    conversations: List[Conversation]
    metadata: Dict[str, Hashable] = field(default_factory=dict)

    def __post_init__(self):
        """Post-initialization validation for dataset attributes."""
        self.id = ValidationUtilities._validate_id(self.id, "id")
        self.conversations = ValidationUtilities._validate_type(self.conversations, list, "conversations")
        self.metadata = ValidationUtilities._validate_type(self.metadata, dict, "metadata")

    @classmethod
    def from_dict(cls, data: Dict) -> 'EvalDataset':
        """Create an EvalDataset instance from a dictionary."""
        ValidationUtilities._validate_dict(data)
        required_keys = ['id', 'conversations']
        ValidationUtilities._validate_keys_exist(data, required_keys)

        conversations = [Conversation.from_dict(conv) for conv in data['conversations']]
        return cls(
            id=data['id'],
            conversations=conversations,
            metadata=data.get('metadata', {})
        )

    def to_dict(self) -> Dict:
        """Convert the EvalDataset instance to a dictionary."""
        return {
            'id': self.id,
            'conversations': [conv.to_dict() for conv in self.conversations],
            'metadata': self.metadata
        }

    def sample(self, n: int) -> "EvalDataset":
        sampled_conversations = random.sample(self.conversations, min(n, len(self.conversations)))
        return EvalDataset(
            id=self.id,
            conversations=sampled_conversations,
            metadata=self.metadata
        )

    def __iter__(self) -> Iterator[Conversation]:
        """Make EvalDataset iterable, yielding each conversation."""
        return iter(self.conversations)

    def __getitem__(self, index: Union[int, slice]) -> Union[Conversation, List[Conversation]]:
        """Allow indexing to get a specific conversation."""
        return self.conversations[index]

    def __len__(self) -> int:
        """Return the number of conversations in the dataset."""
        return len(self.conversations)

    def __str__(self) -> str:
        """Return a human-readable string representation."""
        return f"EvalDataset(id='{self.id}', conversations={len(self.conversations)})"

    def __repr__(self) -> str:
        """Return a detailed string representation for debugging."""
        return (f"EvalDataset(id='{self.id}', "
                f"conversations_count={len(self.conversations)}, "
                f"metadata_keys={list(self.metadata.keys())})")
