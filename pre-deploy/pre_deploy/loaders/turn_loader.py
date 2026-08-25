from typing import List, Dict, Optional
import warnings

from ..input.dataset import EvalDataset
from ..input.conversation import Conversation
from ..input.enums import Role
from .turn import Turn


class TurnLoader:
    """
    Loader class for converting EvalDataset into a list of turns.

    Each turn consists of a user message followed by an assistant message.
    """

    @staticmethod
    def load_turns_from_dataset(dataset: EvalDataset) -> Dict[str, List[Turn]]:
        """
        Convert an EvalDataset into a list of turns.
        """
        if not isinstance(dataset, EvalDataset):
            raise ValueError("Input must be an EvalDataset instance")

        turns = {}

        for conversation in dataset.conversations:
            conversation_turns = TurnLoader._extract_turns_from_conversation(
                dataset.id, conversation
            )
            turns[conversation.id] = conversation_turns

        return turns

    @staticmethod
    def _extract_turns_from_conversation(dataset_id: str, conversation: Conversation) -> List[Turn]:
        """
        Extract turns from a single conversation.
        """
        if not isinstance(conversation, Conversation):
            raise ValueError("Input must be a Conversation instance")

        # Sort messages by sequence to ensure proper order
        sorted_messages = sorted(conversation.messages, key=lambda msg: msg.sequence)

        turns = []
        turn_number = 1

        # Look for user-assistant pairs
        i = 0
        while i < len(sorted_messages):
            current_message = sorted_messages[i]

            if len(sorted_messages) == 1:
                warnings.warn(
                    f"Conversation {conversation.id} has only one message, skipping turn extraction."
                )
                break

            if current_message.role == Role.USER.value:
                if i + 1 < len(sorted_messages) and sorted_messages[i + 1].role == Role.ASSISTANT.value:
                    turn = Turn(
                        dataset_id=dataset_id,
                        conversation_id=conversation.id,
                        turn_number=turn_number,
                        user_message=current_message,
                        assistant_message=sorted_messages[i + 1]
                    )
                    turns.append(turn)
                    turn_number += 1
                    i += 2
                else:
                    warnings.warn(
                        f"User message {current_message.sequence} in conversation {conversation.id} "
                        f"does not have a following assistant message."
                    )
                    i += 1
            else:
                warnings.warn(
                    f"Skipping message {current_message.sequence} in conversation {conversation.id} "
                    f"because it is not a user message."
                )
                i += 1

        return turns
