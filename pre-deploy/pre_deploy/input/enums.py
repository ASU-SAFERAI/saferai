from enum import Enum

class MessageType(Enum):
    """Enumeration of supported message content types."""
    IMAGE = 'image'
    VIDEO = 'video'
    AUDIO = 'audio'
    TEXT_FILE = 'text_file'
    TEXT = 'text'
    TOOL_CALL = 'tool_call'


class Role(Enum):
    """Enumeration of message roles in a conversation."""
    USER = 'user'
    ASSISTANT = 'assistant'
