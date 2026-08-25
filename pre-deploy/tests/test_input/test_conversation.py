"""
Unit tests for Conversation validation.

This test suite validates Conversation with both good and bad data scenarios
to ensure the validation logic correctly accepts valid inputs and rejects invalid ones.
"""

import unittest
import sys
from pathlib import Path

sys.path.append(Path(__file__).resolve().parent.parent.parent.as_posix())

from pre_deploy.input.enums import MessageType, Role
from pre_deploy.input.message_content import MessageContent
from pre_deploy.input.message import Message
from pre_deploy.input.conversation import Conversation


class TestConversationGoodData(unittest.TestCase):
    """Test cases for Conversation validation with good data."""

    def test_validate_conversation_single_message(self):
        """Test validation of conversation with single message."""
        content = MessageContent(
            type=MessageType.TEXT.value,
            content="Hello"
        )

        message = Message(
            sequence=0,
            role=Role.USER.value,
            contents=[content]
        )

        conversation = Conversation(
            id="test-conversation-001",
            messages=[message]
        )

        self.assertEqual(conversation.id, "test-conversation-001")
        self.assertEqual(len(conversation.messages), 1)
        self.assertEqual(conversation.messages[0].sequence, 0)

    def test_validate_conversation_multiple_messages(self):
        """Test validation of conversation with multiple messages."""
        # User message
        user_content = MessageContent(
            type=MessageType.TEXT.value,
            content="What is the weather like?"
        )
        user_message = Message(
            sequence=0,
            role=Role.USER.value,
            contents=[user_content]
        )

        # Assistant message
        assistant_content = MessageContent(
            type=MessageType.TEXT.value,
            content="I'd be happy to help you check the weather."
        )
        assistant_message = Message(
            sequence=1,
            role=Role.ASSISTANT.value,
            contents=[assistant_content]
        )

        conversation = Conversation(
            id="weather-conversation-123",
            messages=[user_message, assistant_message]
        )

        self.assertEqual(conversation.id, "weather-conversation-123")
        self.assertEqual(len(conversation.messages), 2)
        self.assertEqual(conversation.messages[0].role, Role.USER.value)
        self.assertEqual(conversation.messages[1].role, Role.ASSISTANT.value)

    def test_validate_conversation_complex_scenario(self):
        """Test validation of complex conversation with mixed content types."""
        # User message with text and image
        user_contents = [
            MessageContent(type=MessageType.TEXT.value, content="Can you analyze this image?"),
            MessageContent(type=MessageType.IMAGE.value, uri="https://example.com/chart.png")
        ]
        user_message = Message(
            sequence=0,
            role=Role.USER.value,
            contents=user_contents
        )

        # Assistant message with text and tool call
        assistant_contents = [
            MessageContent(type=MessageType.TEXT.value, content="I'll analyze the image for you."),
            MessageContent(type=MessageType.TOOL_CALL.value, content="analyze_image(url='https://example.com/chart.png')")
        ]
        assistant_message = Message(
            sequence=1,
            role=Role.ASSISTANT.value,
            contents=assistant_contents
        )

        # Follow-up user message
        followup_content = MessageContent(
            type=MessageType.TEXT.value,
            content="Thank you for the analysis!"
        )
        followup_message = Message(
            sequence=2,
            role=Role.USER.value,
            contents=[followup_content]
        )

        conversation = Conversation(
            id="complex-conversation-456",
            messages=[user_message, assistant_message, followup_message]
        )

        self.assertEqual(len(conversation.messages), 3)
        # Verify first message (user with text + image)
        self.assertEqual(len(conversation.messages[0].contents), 2)
        self.assertEqual(conversation.messages[0].contents[0].type, MessageType.TEXT.value)
        self.assertEqual(conversation.messages[0].contents[1].type, MessageType.IMAGE.value)

        # Verify second message (assistant with text + tool call)
        self.assertEqual(len(conversation.messages[1].contents), 2)
        self.assertEqual(conversation.messages[1].contents[0].type, MessageType.TEXT.value)
        self.assertEqual(conversation.messages[1].contents[1].type, MessageType.TOOL_CALL.value)

        # Verify third message (user with text)
        self.assertEqual(len(conversation.messages[2].contents), 1)
        self.assertEqual(conversation.messages[2].contents[0].type, MessageType.TEXT.value)

    def test_conversation_with_single_character_id(self):
        """Test Conversation with single character ID."""
        content = MessageContent(type=MessageType.TEXT.value, content="First message")
        message = Message(sequence=0, role=Role.USER.value, contents=[content])
        single_char_conversation = Conversation(id="x", messages=[message])
        self.assertEqual(single_char_conversation.id, "x")

    def test_conversation_iteration_indexing_and_slicing(self):
        """Test that Conversation supports iteration, indexing, and slicing."""
        # Create messages
        messages = []
        for i in range(5):
            content = MessageContent(type=MessageType.TEXT.value, content=f"Message {i}")
            role = Role.USER.value if i % 2 == 0 else Role.ASSISTANT.value
            message = Message(sequence=i, role=role, contents=[content])
            messages.append(message)

        conversation = Conversation(id="test-conv", messages=messages)
        self.assertEqual(len(conversation), 5)

        sequences = [msg.sequence for msg in conversation]
        self.assertEqual(sequences, [0, 1, 2, 3, 4])
        self.assertEqual(conversation[2].sequence, 2)
        self.assertEqual(conversation[-2].sequence, 3)
        first_three = conversation[:3]
        self.assertEqual([m.sequence for m in first_three], [0, 1, 2])
        last_two = conversation[-2:]
        self.assertEqual([m.sequence for m in last_two], [3, 4])
        every_other = conversation[::2]
        self.assertEqual([m.sequence for m in every_other], [0, 2, 4])


class TestConversationBadData(unittest.TestCase):
    """Test cases for Conversation validation with bad data inputs."""

    def test_conversation_missing_id(self):
        """Test Conversation with missing id."""
        with self.assertRaises(TypeError) as context:
            Conversation(messages=[], metadata={})
        self.assertIn("Conversation.__init__() missing 1 required positional argument", str(context.exception))

    def test_conversation_missing_messages(self):
        """Test Conversation with missing messages."""
        with self.assertRaises(TypeError) as context:
            Conversation(id="conv1", metadata={})
        self.assertIn("Conversation.__init__() missing 1 required positional argument", str(context.exception))

    def test_conversation_invalid_id(self):
        """Test Conversation with invalid id."""
        with self.assertRaises(ValueError) as context:
            Conversation(id="", messages=[], metadata={})
        self.assertIn("ID `id` must be a non-empty string.", str(context.exception))

    def test_conversation_invalid_messages(self):
        """Test Conversation with invalid messages."""
        with self.assertRaises(TypeError) as context:
            Conversation(id="conv1", messages="not_a_list", metadata={})
        self.assertIn("`messages`: Expected type list, got str.", str(context.exception))

    def test_conversation_from_dict_missing_id(self):
        """Test Conversation from_dict with missing id."""
        with self.assertRaises(KeyError) as context:
            Conversation.from_dict({"messages": []})
        self.assertIn("Missing required key", str(context.exception))

    def test_conversation_from_dict_missing_messages(self):
        """Test Conversation from_dict with missing messages."""
        with self.assertRaises(KeyError) as context:
            Conversation.from_dict({"id": "conv1"})
        self.assertIn("Missing required key", str(context.exception))

    def test_conversation_from_dict_invalid_id(self):
        """Test Conversation from_dict with invalid id."""
        with self.assertRaises(ValueError) as context:
            Conversation.from_dict({"id": "", "messages": []})
        self.assertIn("ID `id` must be a non-empty string.", str(context.exception))

    def test_conversation_from_dict_invalid_messages(self):
        """Test Conversation from_dict with invalid messages."""
        with self.assertRaises(TypeError) as context:
            Conversation.from_dict({"id": "conv1", "messages": "not_a_list"})
        self.assertIn("Data must be a dictionary", str(context.exception))


if __name__ == '__main__':
    unittest.main(verbosity=2)
