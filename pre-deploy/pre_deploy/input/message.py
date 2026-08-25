from dataclasses import dataclass, field
from typing import List, Dict, Hashable, Optional

from .message_content import MessageContent
from .validation import ValidationUtilities
from .enums import Role

@dataclass
class Message:
    """
    Represents a single message in a conversation.

    Attributes:
        sequence: The order of the message in the conversation
        role: The role of the message sender (user, assistant, system, tool)
        contents: List of content items in the message
        metadata: Additional metadata for the message
    """
    sequence: int
    role: str
    contents: List[MessageContent]
    metadata: Dict[str, Hashable] = field(default_factory=dict)

    def __post_init__(self):
        """Post-initialization validation for message level attributes. MessageContent
        validation is handled in its own class."""
        self.sequence = ValidationUtilities._validate_sequence(self.sequence, "sequence")
        self.role = ValidationUtilities._validate_enum(self.role, Role, "role")
        self.contents = ValidationUtilities._validate_type(self.contents, list, "contents")
        self.metadata = ValidationUtilities._validate_type(self.metadata, dict, "metadata")
        self.text = " ".join(
            [content.content for content in self.contents if content.type == 'text']
        )

    @classmethod
    def from_dict(cls, data: dict) -> 'Message':
        """Create a Message instance from a dictionary."""
        ValidationUtilities._validate_dict(data)
        required_keys = ['sequence', 'role']
        ValidationUtilities._validate_keys_exist(data, required_keys)
        if 'contents' in data:
            contents = [MessageContent.from_dict(elem) for elem in data['contents']]
            return cls(
                sequence=data['sequence'],
                role=data['role'],
                contents=contents,
                metadata=data.get('metadata', {})
            )
        elif 'text' in data:
            content = MessageContent.from_dict({
                'type': 'text',
                'content': data['text'],
                'metadata': data.get('metadata', {})
            })
            return cls(
                sequence=data['sequence'],
                role=data['role'],
                contents=[content],
                metadata=data.get('metadata', {})
            )
        else:
            raise KeyError("Message dictionary must contain either 'contents' or 'text' key.")

    @classmethod
    def from_text(cls,
                  text: str,
                  role: str,
                  sequence: int,
                  metadata: Optional[Dict] = None) -> 'Message':
        """Create a simple text message."""
        return cls(
            sequence=sequence,
            role=role,
            contents=[MessageContent(type='text', content=text)],
            metadata=metadata or {}
        )

    def to_dict(self) -> Dict:
        """Convert the Message instance to a dictionary."""
        return {
            'sequence': self.sequence,
            'role': self.role,
            'contents': [content.to_dict() for content in self.contents],
            'text': self.text,
            'metadata': self.metadata
        }

    def __str__(self) -> str:
        """Return a human-readable string representation."""
        return (f"Message(sequence={self.sequence}, role='{self.role}', contents={len(self.contents)}), "
                f"text={self.text[:30]}...")

    def __repr__(self) -> str:
        """Return a detailed string representation for debugging."""
        return (f"Message(sequence={self.sequence}, role='{self.role}', "
                f"contents_count={len(self.contents)}, "
                f"metadata_keys={list(self.metadata.keys())})")
