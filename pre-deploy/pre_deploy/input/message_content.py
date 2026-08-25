from typing import Optional, Dict, Hashable
from dataclasses import dataclass, field

from .validation import ValidationUtilities
from .enums import MessageType

@dataclass
class MessageContent:
    """
    Represents content within a message, supporting various media types.

    Attributes:
        type: The type of content (image, video, audio, text_file, text, tool_call)
        uri: URI pointing to the content resource
        content: Raw content data
        metadata: Additional metadata for the content
    """
    type: str
    uri: Optional[str] = None
    content: Optional[str] = None
    metadata: Dict[str, Hashable] = field(default_factory=dict)

    def __post_init__(self):
        """Validation functions that run upon the instantiation of the factory dataclass."""
        self.type = ValidationUtilities._validate_enum(self.type, MessageType, "type")
        if self.uri is not None:
            self.uri = ValidationUtilities._validate_uri(self.uri, "uri")
        if self.content is not None:
            self.content = ValidationUtilities._validate_type(self.content, str, "content")
        self.metadata = ValidationUtilities._validate_type(self.metadata, dict, "metadata")

    @classmethod
    def from_dict(cls, data: dict) -> 'MessageContent':
        """Create a MessageContent instance from a dictionary."""
        ValidationUtilities._validate_dict(data)
        ValidationUtilities._validate_keys_exist(data, ['type'])

        return cls(
            type=data['type'],
            uri=data.get('uri'),
            content=data.get('content'),
            metadata=data.get('metadata', {})
        )

    def to_dict(self) -> Dict:
        """Convert the MessageContent instance to a dictionary."""
        return {
            'type': self.type,
            'uri': self.uri,
            'content': self.content,
            'metadata': self.metadata
        }

    def __str__(self) -> str:
        """Return a human-readable string representation."""
        return f"MessageContent(type='{self.type}', uri='{self.uri}')"

    def __repr__(self) -> str:
        """Return a detailed string representation for debugging."""
        return (f"MessageContent(type='{self.type}', uri='{self.uri}', "
                f"content_length={len(self.content) if self.content else 0}, "
                f"metadata_keys={list(self.metadata.keys())})")
