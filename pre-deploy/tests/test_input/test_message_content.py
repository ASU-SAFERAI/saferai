"""
Unit tests for MessageContent validation.

This test suite validates MessageContent with both good and bad data scenarios
to ensure the validation logic correctly accepts valid inputs and rejects invalid ones.
"""

import unittest
import tempfile
import os
import sys
from pathlib import Path

sys.path.append(Path(__file__).resolve().parent.parent.parent.as_posix())

from pre_deploy.input.enums import MessageType
from pre_deploy.input.message_content import MessageContent


class TestMessageContentGoodData(unittest.TestCase):
    """Test cases for MessageContent validation with good data."""

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

    def test_validate_message_content_text(self):
        """Test validation of text message content."""
        content = MessageContent(
            type=MessageType.TEXT.value,
            content="Hello, this is a test message."
        )

        self.assertEqual(content.type, MessageType.TEXT.value)
        self.assertEqual(content.content, "Hello, this is a test message.")
        self.assertIsNone(content.uri)

    def test_validate_message_content_with_uri(self):
        """Test validation of message content with various URI types."""
        # Test with HTTP URL
        content_http = MessageContent(
            type=MessageType.IMAGE.value,
            uri="https://example.com/image.jpg"
        )
        self.assertEqual(content_http.uri, "https://example.com/image.jpg")

        # Test with file path
        content_file = MessageContent(
            type=MessageType.TEXT_FILE.value,
            uri=self.temp_file_path
        )
        self.assertEqual(content_file.uri, self.temp_file_path)

        # Test with file:// URI
        content_file_uri = MessageContent(
            type=MessageType.AUDIO.value,
            uri="file:///path/to/audio.mp3"
        )
        self.assertEqual(content_file_uri.uri, "file:///path/to/audio.mp3")

    def test_validate_message_content_all_types(self):
        """Test validation of all supported message content types."""
        for msg_type in MessageType:
            content = MessageContent(
                type=msg_type.value,
                content="Sample content",
                uri="https://example.com/resource"
            )

            self.assertEqual(content.type, msg_type.value)

    def test_message_content_with_none_uri(self):
        """Test MessageContent with None URI (should be valid)."""
        content_no_uri = MessageContent(type=MessageType.TEXT.value, uri=None, content="Text only")
        self.assertIsNone(content_no_uri.uri)


class TestMessageContentBadData(unittest.TestCase):
    """Test cases for MessageContent validation with bad data inputs."""

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

    def test_message_content_missing_type(self):
        """Test MessageContent with missing type."""
        with self.assertRaises(TypeError) as context:
            MessageContent(uri="http://example.com", content=None, metadata={})
        self.assertIn("MessageContent.__init__() missing 1 required positional argument", str(context.exception))

    def test_message_content_invalid_type(self):
        """Test MessageContent with invalid type."""
        with self.assertRaises(ValueError) as context:
            MessageContent(type="invalid_type", uri=None, content=None, metadata={})
        self.assertIn("Invalid value 'invalid_type'", str(context.exception))

    def test_message_content_invalid_uri(self):
        """Test MessageContent with invalid URI."""
        with self.assertRaises(ValueError) as context:
            MessageContent(type=MessageType.TEXT.value, uri="invalid_uri", content=None, metadata={})
        self.assertIn("URI 'invalid_uri' must start with one of", str(context.exception))

    def test_message_content_from_dict_missing_type(self):
        """Test MessageContent from_dict with missing type."""
        with self.assertRaises(KeyError) as context:
            MessageContent.from_dict({"uri": "http://example.com"})
        self.assertIn("Missing required key", str(context.exception))

    def test_message_content_from_dict_invalid_type(self):
        """Test MessageContent from_dict with invalid type."""
        with self.assertRaises(ValueError) as context:
            MessageContent.from_dict({"type": "invalid_type", "uri": "http://example.com"})
        self.assertIn("Invalid value 'invalid_type'", str(context.exception))

    def test_message_content_from_dict_invalid_uri(self):
        """Test MessageContent from_dict with invalid URI."""
        with self.assertRaises(ValueError) as context:
            MessageContent.from_dict({"type": MessageType.TEXT.value, "uri": "invalid_uri"})
        self.assertIn("URI 'invalid_uri' must start with one of", str(context.exception))


if __name__ == '__main__':
    unittest.main(verbosity=2)
