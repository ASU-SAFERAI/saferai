"""
Unit tests for EvalDataset validation.

This test suite validates EvalDataset with both good and bad data scenarios
to ensure the validation logic correctly accepts valid inputs and rejects invalid ones.
"""

import unittest
import tempfile
import os
import sys
from pathlib import Path

sys.path.append(Path(__file__).resolve().parent.parent.parent.as_posix())

from pre_deploy.input.enums import MessageType, Role
from pre_deploy.input.message_content import MessageContent
from pre_deploy.input.message import Message
from pre_deploy.input.conversation import Conversation
from pre_deploy.input.dataset import EvalDataset


class TestEvalDatasetGoodData(unittest.TestCase):
    """Test cases for EvalDataset validation with good data."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a temporary file for URI validation tests
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
        self.temp_file.write("Test content")
        self.temp_file.close()
        self.temp_file_path = self.temp_file.name

    def tearDown(self):
        """Clean up after each test method."""
        # Clean up temporary file
        if os.path.exists(self.temp_file_path):
            os.unlink(self.temp_file_path)

    def test_validate_dataset_single_conversation(self):
        """Test validation of dataset with single conversation."""
        # Create a simple conversation
        content = MessageContent(
            type=MessageType.TEXT.value,
            content="Hello world"
        )
        message = Message(
            sequence=0,
            role=Role.USER.value,
            contents=[content]
        )
        conversation = Conversation(
            id="conv-001",
            messages=[message]
        )

        # Create dataset
        dataset = EvalDataset(
            id="dataset-001",
            conversations=[conversation]
        )

        self.assertEqual(dataset.id, "dataset-001")
        self.assertEqual(len(dataset.conversations), 1)
        self.assertEqual(dataset.conversations[0].id, "conv-001")
        self.assertEqual(len(dataset.conversations[0].messages), 1)

    def test_validate_dataset_multiple_conversations(self):
        """Test validation of dataset with multiple conversations."""
        # First conversation
        content1 = MessageContent(type=MessageType.TEXT.value, content="First conversation")
        message1 = Message(sequence=0, role=Role.USER.value, contents=[content1])
        conversation1 = Conversation(id="conv-001", messages=[message1])

        # Second conversation with multiple messages
        user_content = MessageContent(type=MessageType.TEXT.value, content="User question")
        user_message = Message(sequence=0, role=Role.USER.value, contents=[user_content])

        assistant_content = MessageContent(type=MessageType.TEXT.value, content="Assistant response")
        assistant_message = Message(sequence=1, role=Role.ASSISTANT.value, contents=[assistant_content])

        conversation2 = Conversation(id="conv-002", messages=[user_message, assistant_message])

        # Create dataset
        dataset = EvalDataset(
            id="multi-conv-dataset",
            conversations=[conversation1, conversation2]
        )

        self.assertEqual(dataset.id, "multi-conv-dataset")
        self.assertEqual(len(dataset.conversations), 2)

        # Verify first conversation
        self.assertEqual(dataset.conversations[0].id, "conv-001")
        self.assertEqual(len(dataset.conversations[0].messages), 1)

        # Verify second conversation
        self.assertEqual(dataset.conversations[1].id, "conv-002")
        self.assertEqual(len(dataset.conversations[1].messages), 2)
        self.assertEqual(dataset.conversations[1].messages[0].role, Role.USER.value)
        self.assertEqual(dataset.conversations[1].messages[1].role, Role.ASSISTANT.value)

    def test_validate_dataset_complex_scenario(self):
        """Test validation of complex dataset with various content types."""
        # Conversation 1: Multi-modal user input
        user_contents = [
            MessageContent(type=MessageType.TEXT.value, content="Please analyze this data"),
            MessageContent(type=MessageType.IMAGE.value, uri="https://example.com/chart.png"),
            MessageContent(type=MessageType.TEXT_FILE.value, uri=self.temp_file_path)
        ]
        user_message = Message(sequence=0, role=Role.USER.value, contents=user_contents)

        assistant_contents = [
            MessageContent(type=MessageType.TEXT.value, content="I'll analyze the data for you"),
            MessageContent(type=MessageType.TOOL_CALL.value, content="analyze_data(image_url='https://example.com/chart.png')")
        ]
        assistant_message = Message(sequence=1, role=Role.ASSISTANT.value, contents=assistant_contents)

        conversation1 = Conversation(id="analysis-conv", messages=[user_message, assistant_message])

        # Conversation 2: Simple text exchange
        simple_user = MessageContent(type=MessageType.TEXT.value, content="How are you?")
        simple_user_msg = Message(sequence=0, role=Role.USER.value, contents=[simple_user])

        simple_assistant = MessageContent(type=MessageType.TEXT.value, content="I'm doing well, thank you!")
        simple_assistant_msg = Message(sequence=1, role=Role.ASSISTANT.value, contents=[simple_assistant])

        conversation2 = Conversation(id="greeting-conv", messages=[simple_user_msg, simple_assistant_msg])

        # Create complex dataset
        dataset = EvalDataset(
            id="complex-dataset-789",
            conversations=[conversation1, conversation2],
            metadata={"version": "1.0", "created_by": "test_suite"}
        )

        # Verify dataset structure
        self.assertEqual(dataset.id, "complex-dataset-789")
        self.assertEqual(len(dataset.conversations), 2)
        self.assertEqual(dataset.metadata["version"], "1.0")

        # Verify first conversation (multi-modal)
        conv1 = dataset.conversations[0]
        self.assertEqual(conv1.id, "analysis-conv")
        self.assertEqual(len(conv1.messages), 2)
        self.assertEqual(len(conv1.messages[0].contents), 3)  # Text + Image + File
        self.assertEqual(len(conv1.messages[1].contents), 2)  # Text + Tool call

        # Verify content types in first conversation
        user_msg_contents = conv1.messages[0].contents
        self.assertEqual(user_msg_contents[0].type, MessageType.TEXT.value)
        self.assertEqual(user_msg_contents[1].type, MessageType.IMAGE.value)
        self.assertEqual(user_msg_contents[2].type, MessageType.TEXT_FILE.value)

        assistant_msg_contents = conv1.messages[1].contents
        self.assertEqual(assistant_msg_contents[0].type, MessageType.TEXT.value)
        self.assertEqual(assistant_msg_contents[1].type, MessageType.TOOL_CALL.value)

        # Verify second conversation (simple)
        conv2 = dataset.conversations[1]
        self.assertEqual(conv2.id, "greeting-conv")
        self.assertEqual(len(conv2.messages), 2)
        self.assertEqual(len(conv2.messages[0].contents), 1)
        self.assertEqual(len(conv2.messages[1].contents), 1)

    def test_validate_dataset_empty_conversations(self):
        """Test validation of dataset with empty conversations list."""
        dataset = EvalDataset(
            id="empty-dataset",
            conversations=[]
        )

        self.assertEqual(dataset.id, "empty-dataset")
        self.assertEqual(len(dataset.conversations), 0)
        self.assertIsInstance(dataset.conversations, list)

    def test_dataset_iteration_indexing_and_slicing(self):
        """Test that EvalDataset supports iteration, indexing, and slicing."""
        # Create conversations
        conversations = []
        for i in range(5):
            content = MessageContent(type=MessageType.TEXT.value, content=f"Message {i}")
            message = Message(sequence=0, role=Role.USER.value, contents=[content])
            conversation = Conversation(id=f"conv-{i}", messages=[message])
            conversations.append(conversation)

        dataset = EvalDataset(id="test-dataset", conversations=conversations)
        self.assertEqual(len(dataset), 5)

        conv_ids = [conv.id for conv in dataset]
        self.assertEqual(conv_ids, ["conv-0", "conv-1", "conv-2", "conv-3", "conv-4"])
        self.assertEqual(dataset[0].id, "conv-0")
        self.assertEqual(dataset[-2].id, "conv-3")
        first_three = dataset[:3]
        self.assertEqual([c.id for c in first_three], ["conv-0", "conv-1", "conv-2"])
        last_two = dataset[-2:]
        self.assertEqual([c.id for c in last_two], ["conv-3", "conv-4"])
        every_other = dataset[::2]
        self.assertEqual([c.id for c in every_other], ["conv-0", "conv-2", "conv-4"])


class TestEvalDatasetBadData(unittest.TestCase):
    """Test cases for EvalDataset validation with bad data inputs."""

    def test_eval_dataset_missing_id(self):
        """Test EvalDataset with missing id."""
        with self.assertRaises(TypeError) as context:
            EvalDataset(conversations=[], metadata={})
        self.assertIn("EvalDataset.__init__() missing 1 required positional argument", str(context.exception))

    def test_eval_dataset_missing_conversations(self):
        """Test EvalDataset with missing conversations."""
        with self.assertRaises(TypeError) as context:
            EvalDataset(id="dataset1", metadata={})
        self.assertIn("EvalDataset.__init__() missing 1 required positional argument", str(context.exception))

    def test_eval_dataset_invalid_id(self):
        """Test EvalDataset with invalid id."""
        with self.assertRaises(ValueError) as context:
            EvalDataset(id="", conversations=[], metadata={})
        self.assertIn("ID `id` must be a non-empty string.", str(context.exception))

    def test_eval_dataset_invalid_conversations(self):
        """Test EvalDataset with invalid conversations."""
        with self.assertRaises(TypeError) as context:
            EvalDataset(id="dataset1", conversations="not_a_list", metadata={})
        self.assertIn("`conversations`: Expected type list, got str.", str(context.exception))

    def test_eval_dataset_from_dict_missing_id(self):
        """Test EvalDataset from_dict with missing id."""
        with self.assertRaises(KeyError) as context:
            EvalDataset.from_dict({"conversations": []})
        self.assertIn("Missing required key", str(context.exception))

    def test_eval_dataset_from_dict_missing_conversations(self):
        """Test EvalDataset from_dict with missing conversations."""
        with self.assertRaises(KeyError) as context:
            EvalDataset.from_dict({"id": "dataset1"})
        self.assertIn("Missing required key", str(context.exception))

    def test_eval_dataset_from_dict_invalid_id(self):
        """Test EvalDataset from_dict with invalid id."""
        with self.assertRaises(ValueError) as context:
            EvalDataset.from_dict({"id": "", "conversations": []})
        self.assertIn("ID `id` must be a non-empty string.", str(context.exception))

    def test_eval_dataset_from_dict_invalid_conversations(self):
        """Test EvalDataset from_dict with invalid conversations."""
        with self.assertRaises(TypeError) as context:
            EvalDataset.from_dict({"id": "dataset1", "conversations": "not_a_list"})
        self.assertIn("Data must be a dictionary", str(context.exception))


if __name__ == '__main__':
    unittest.main(verbosity=2)
