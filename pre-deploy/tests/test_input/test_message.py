"""
Unit tests for Message validation.

This test suite validates Message with both good and bad data scenarios
to ensure the validation logic correctly accepts valid inputs and rejects invalid ones.
"""

import unittest
import sys
from pathlib import Path

sys.path.append(Path(__file__).resolve().parent.parent.parent.as_posix())

from pre_deploy.input.enums import MessageType, Role
from pre_deploy.input.message_content import MessageContent
from pre_deploy.input.message import Message


class TestMessageGoodData(unittest.TestCase):
    """Test cases for Message validation with good data."""

    def test_validate_message_user_role(self):
        """Test validation of user message."""
        content = MessageContent(
            type=MessageType.TEXT.value,
            content="User question"
        )

        message = Message(
            sequence=0,
            role=Role.USER.value,
            contents=[content]
        )

        self.assertEqual(message.sequence, 0)
        self.assertEqual(message.role, Role.USER.value)
        self.assertEqual(len(message.contents), 1)
        self.assertEqual(message.contents[0].type, MessageType.TEXT.value)
        self.assertEqual(message.text, "User question")

    def test_validate_message_assistant_role(self):
        """Test validation of assistant message."""
        content = MessageContent(
            type=MessageType.TEXT.value,
            content="Assistant response"
        )

        message = Message(
            sequence=1,
            role=Role.ASSISTANT.value,
            contents=[content]
        )

        self.assertEqual(message.sequence, 1)
        self.assertEqual(message.role, Role.ASSISTANT.value)
        self.assertEqual(len(message.contents), 1)
        self.assertEqual(message.text, "Assistant response")

    def test_validate_message_multiple_contents(self):
        """Test validation of message with multiple content items."""
        contents = [
            MessageContent(type=MessageType.TEXT.value, content="Text content"),
            MessageContent(type=MessageType.IMAGE.value, uri="https://example.com/image.jpg"),
            MessageContent(type=MessageType.TOOL_CALL.value, content="function_call()")
        ]

        message = Message(
            sequence=2,
            role=Role.USER.value,
            contents=contents
        )

        self.assertEqual(len(message.contents), 3)
        self.assertEqual(message.contents[0].type, MessageType.TEXT.value)
        self.assertEqual(message.contents[1].type, MessageType.IMAGE.value)
        self.assertEqual(message.contents[2].type, MessageType.TOOL_CALL.value)
        self.assertEqual(message.text, "Text content")

    def test_message_with_minimum_sequence(self):
        """Test Message with sequence 0 (minimum valid value)."""
        content = MessageContent(type=MessageType.TEXT.value, content="First message")
        message = Message(sequence=0, role=Role.USER.value, contents=[content])
        self.assertEqual(message.sequence, 0)
        self.assertEqual(message.text, "First message")

    def test_message_from_text(self):
        """Test Message creation using from_text class method."""
        message = Message.from_text(
            text="Hello, how can I help you?",
            role=Role.ASSISTANT.value,
            sequence=1
        )

        self.assertEqual(message.sequence, 1)
        self.assertEqual(message.role, Role.ASSISTANT.value)
        self.assertEqual(len(message.contents), 1)
        self.assertEqual(message.contents[0].type, MessageType.TEXT.value)
        self.assertEqual(message.contents[0].content, "Hello, how can I help you?")
        self.assertEqual(message.text, "Hello, how can I help you?")
        self.assertEqual(message.metadata, {})

    def test_message_from_text_with_metadata(self):
        """Test Message creation using from_text with metadata."""
        metadata = {"source": "test", "priority": "high"}
        message = Message.from_text(
            text="Test message",
            role=Role.USER.value,
            sequence=0,
            metadata=metadata
        )

        self.assertEqual(message.text, "Test message")
        self.assertEqual(message.metadata, metadata)

    def test_message_from_dict_with_contents(self):
        """Test Message creation from dict with contents."""
        data = {
            "sequence": 1,
            "role": Role.USER.value,
            "contents": [
                {"type": MessageType.TEXT.value, "content": "Hello"}
            ]
        }
        message = Message.from_dict(data)

        self.assertEqual(message.sequence, 1)
        self.assertEqual(message.role, Role.USER.value)
        self.assertEqual(len(message.contents), 1)
        self.assertEqual(message.text, "Hello")

    def test_message_from_dict_with_text(self):
        """Test Message creation from dict with text field."""
        data = {
            "sequence": 2,
            "role": Role.ASSISTANT.value,
            "text": "Response text"
        }
        message = Message.from_dict(data)

        self.assertEqual(message.sequence, 2)
        self.assertEqual(message.role, Role.ASSISTANT.value)
        self.assertEqual(len(message.contents), 1)
        self.assertEqual(message.contents[0].type, MessageType.TEXT.value)
        self.assertEqual(message.text, "Response text")

    def test_message_text_filters_non_text_contents(self):
        """Test message.text only includes text type contents."""
        contents = [
            MessageContent(type=MessageType.TEXT.value, content="Text part"),
            MessageContent(type=MessageType.IMAGE.value, uri="https://example.com/image.jpg"),
            MessageContent(type=MessageType.TEXT.value, content="More text")
        ]
        message = Message(
            sequence=0,
            role=Role.USER.value,
            contents=contents
        )

        self.assertEqual(message.text, "Text part More text")


class TestMessageBadData(unittest.TestCase):
    """Test cases for Message validation with bad data inputs."""

    def test_message_missing_sequence(self):
        """Test Message with missing sequence."""
        with self.assertRaises(TypeError) as context:
            Message(role=Role.USER.value, contents=[], metadata={})
        self.assertIn("Message.__init__() missing 1 required positional argument", str(context.exception))

    def test_message_missing_role(self):
        """Test Message with missing role."""
        with self.assertRaises(TypeError) as context:
            Message(sequence=1, contents=[], metadata={})
        self.assertIn("Message.__init__() missing 1 required positional argument", str(context.exception))

    def test_message_missing_contents(self):
        """Test Message with missing contents."""
        with self.assertRaises(TypeError) as context:
            Message(sequence=1, role=Role.USER.value, metadata={})
        self.assertIn("Message.__init__() missing 1 required positional argument", str(context.exception))

    def test_message_invalid_sequence(self):
        """Test Message with invalid sequence."""
        with self.assertRaises(ValueError) as context:
            Message(sequence=-1, role=Role.USER.value, contents=[], metadata={})
        self.assertIn("Sequence `sequence` must be a non-negative integer.", str(context.exception))

    def test_message_invalid_role(self):
        """Test Message with invalid role."""
        with self.assertRaises(ValueError) as context:
            Message(sequence=1, role="invalid_role", contents=[], metadata={})
        self.assertIn("`role`: Invalid value 'invalid_role'. Must be one of:", str(context.exception))

    def test_message_invalid_contents(self):
        """Test Message with invalid contents."""
        with self.assertRaises(TypeError) as context:
            Message(sequence=1, role=Role.USER.value, contents="not_a_list", metadata={})
        self.assertIn("`contents`: Expected type list, got str.", str(context.exception))

    def test_message_from_dict_missing_sequence(self):
        """Test Message from_dict with missing sequence."""
        with self.assertRaises(KeyError) as context:
            Message.from_dict({"role": Role.USER.value, "contents": []})
        self.assertIn("Missing required key", str(context.exception))

    def test_message_from_dict_missing_role(self):
        """Test Message from_dict with missing role."""
        with self.assertRaises(KeyError) as context:
            Message.from_dict({"sequence": 1, "contents": []})
        self.assertIn("Missing required key", str(context.exception))

    def test_message_from_dict_missing_contents_and_text(self):
        """Test Message from_dict with neither contents nor text."""
        with self.assertRaises(KeyError) as context:
            Message.from_dict({"sequence": 1, "role": Role.USER.value})
        self.assertIn("Message dictionary must contain either 'contents' or 'text' key", str(context.exception))

    def test_message_from_dict_invalid_sequence(self):
        """Test Message from_dict with invalid sequence."""
        with self.assertRaises(ValueError) as context:
            Message.from_dict({"sequence": -1, "role": Role.USER.value, "text": "test"})
        self.assertIn("Sequence `sequence` must be a non-negative integer.", str(context.exception))

    def test_message_from_dict_invalid_role(self):
        """Test Message from_dict with invalid role."""
        with self.assertRaises(ValueError) as context:
            Message.from_dict({"sequence": 1, "role": "invalid_role", "text": "test"})
        self.assertIn("`role`: Invalid value 'invalid_role'. Must be one of:", str(context.exception))

    def test_message_from_dict_invalid_contents_type(self):
        """Test Message from_dict with invalid contents type."""
        with self.assertRaises(TypeError) as context:
            Message.from_dict({"sequence": 1, "role": Role.USER.value, "contents": "not_a_list"})
        self.assertIn("Data must be a dictionary.", str(context.exception))

    def test_message_from_text_invalid_sequence(self):
        """Test Message from_text with invalid sequence."""
        with self.assertRaises(ValueError) as context:
            Message.from_text(
                text="Test",
                role=Role.USER.value,
                sequence=-1
            )
        self.assertIn("Sequence `sequence` must be a non-negative integer.", str(context.exception))

    def test_message_from_text_invalid_role(self):
        """Test Message from_text with invalid role."""
        with self.assertRaises(ValueError) as context:
            Message.from_text(
                text="Test",
                role="invalid_role",
                sequence=0
            )
        self.assertIn("`role`: Invalid value 'invalid_role'. Must be one of:", str(context.exception))


if __name__ == '__main__':
    unittest.main(verbosity=2)
