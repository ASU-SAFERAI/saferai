from dataclasses import dataclass
from ..input import Message


@dataclass
class Turn:
    """
    Represents a single turn in a conversation, consisting of a user message
    and an assistant response.

    Attributes:
        dataset_id: The ID of the dataset this turn belongs to
        conversation_id: The ID of the conversation this turn belongs to
        turn_number: The sequential number of this turn within the conversation
        user_message: The user's message in this turn
        assistant_message: The assistant's response message in this turn
    """
    dataset_id: str
    conversation_id: str
    turn_number: int
    user_message: Message
    assistant_message: Message

    def __str__(self) -> str:
        """Return a human-readable string representation."""
        return (f"Turn(dataset_id='{self.dataset_id}', "
                f"conversation_id='{self.conversation_id}', "
                f"turn_number={self.turn_number})")

    def __repr__(self) -> str:
        """Return a detailed string representation for debugging."""
        return (f"Turn(dataset_id='{self.dataset_id}', "
                f"conversation_id='{self.conversation_id}', "
                f"turn_number={self.turn_number}, "
                f"user_sequence={self.user_message.sequence}, "
                f"assistant_sequence={self.assistant_message.sequence})")
