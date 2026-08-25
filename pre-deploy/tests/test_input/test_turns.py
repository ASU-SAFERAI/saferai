import unittest
import sys
from pathlib import Path
from warnings import catch_warnings, simplefilter

sys.path.append(Path(__file__).resolve().parent.parent.parent.as_posix())

from pre_deploy.input.dataset import EvalDataset
from pre_deploy.input.conversation import Conversation
from pre_deploy.input.message import Message
from pre_deploy.input.message_content import MessageContent
from pre_deploy.input.enums import Role, MessageType
from pre_deploy.loaders.turn_loader import TurnLoader, Turn

def create_sample_dataset() -> EvalDataset:
    user_content1 = MessageContent(type=MessageType.TEXT.value, content="Hello, how are you?")
    assistant_content1 = MessageContent(type=MessageType.TEXT.value, content="I'm doing well, thank you!")
    user_content2 = MessageContent(type=MessageType.TEXT.value, content="What's the weather like?")
    assistant_content2 = MessageContent(type=MessageType.TEXT.value, content="I don't have access to current weather data.")
    messages = [
        Message(sequence=1, role=Role.USER.value, contents=[user_content1]),
        Message(sequence=2, role=Role.ASSISTANT.value, contents=[assistant_content1]),
        Message(sequence=3, role=Role.USER.value, contents=[user_content2]),
        Message(sequence=4, role=Role.ASSISTANT.value, contents=[assistant_content2]),
    ]
    conversation = Conversation(id="conv_001", messages=messages)
    dataset = EvalDataset(id="dataset_foo", conversations=[conversation])
    return dataset

class TestTurnLoader(unittest.TestCase):
    def setUp(self):
        self.dataset = create_sample_dataset()
        self.turns_dict = TurnLoader.load_turns_from_dataset(self.dataset)
        self.turns = self.turns_dict["conv_001"]

    def test_turn_extraction(self):
        self.assertEqual(len(self.turns), 2)
        self.assertIsInstance(self.turns[0], Turn)
        self.assertEqual(self.turns[0].user_message.role, Role.USER.value)
        self.assertEqual(self.turns[0].assistant_message.role, Role.ASSISTANT.value)
        self.assertEqual(self.turns[0].user_message.contents[0].content, "Hello, how are you?")
        self.assertEqual(self.turns[0].assistant_message.contents[0].content, "I'm doing well, thank you!")

    def test_invalid_dataset_type(self):
        with self.assertRaises(ValueError):
            TurnLoader.load_turns_from_dataset("not_a_dataset")

    def test_invalid_conversation_type(self):
        with self.assertRaises(ValueError):
            TurnLoader._extract_turns_from_conversation("dataset_foo", "not_a_conversation")

    def test_conversation_of_len_one(self):
        user_content = MessageContent(type=MessageType.TEXT.value, content="Single message")
        messages = [Message(sequence=1, role=Role.USER.value, contents=[user_content])]
        conversation = Conversation(id="conv_single", messages=messages)
        dataset = EvalDataset(id="dataset_single", conversations=[conversation])

        with self.assertWarns(UserWarning):
            turns_dict = TurnLoader.load_turns_from_dataset(dataset)
            self.assertEqual(len(turns_dict["conv_single"]), 0)

    def test_unmatched_user_message_triggers_warning(self):
        user_content = MessageContent(type=MessageType.TEXT.value, content="Matched user message")
        asst_content = MessageContent(type=MessageType.TEXT.value, content="Matched assistant message")
        unmatched_user_content = MessageContent(type=MessageType.TEXT.value, content="Unmatched user message")
        messages = [
            Message(sequence=1, role=Role.USER.value, contents=[user_content]),
            Message(sequence=2, role=Role.ASSISTANT.value, contents=[asst_content]),
            Message(sequence=3, role=Role.USER.value, contents=[unmatched_user_content])
        ]
        conversation = Conversation(id="conv_warn", messages=messages)
        dataset = EvalDataset(id="dataset_warn", conversations=[conversation])

        with catch_warnings(record=True) as w:
            simplefilter("always")
            turns_dict = TurnLoader.load_turns_from_dataset(dataset)
            self.assertEqual(len(turns_dict["conv_warn"]), 1)
            self.assertTrue(any("User message" in str(warn.message) for warn in w))

    def test_unmatched_asst_message_triggers_warning(self):
        user_content = MessageContent(type=MessageType.TEXT.value, content="Matched user message")
        asst_content = MessageContent(type=MessageType.TEXT.value, content="Matched assistant message")
        unmatched_asst_content = MessageContent(type=MessageType.TEXT.value, content="Unmatched assistant message")
        messages = [
            Message(sequence=1, role=Role.USER.value, contents=[user_content]),
            Message(sequence=2, role=Role.ASSISTANT.value, contents=[asst_content]),
            Message(sequence=3, role=Role.ASSISTANT.value, contents=[unmatched_asst_content])
        ]
        conversation = Conversation(id="conv_warn", messages=messages)
        dataset = EvalDataset(id="dataset_warn", conversations=[conversation])

        with catch_warnings(record=True) as w:
            simplefilter("always")
            turns_dict = TurnLoader.load_turns_from_dataset(dataset)
            self.assertEqual(len(turns_dict["conv_warn"]), 1)
            self.assertTrue(any("Skipping message" in str(warn.message) for warn in w))

    def test_unmatched_user_message_mid_convo_triggers_warning(self):
        user_content = MessageContent(type=MessageType.TEXT.value, content="Matched user message")
        asst_content = MessageContent(type=MessageType.TEXT.value, content="Matched assistant message")
        unmatched_user_content = MessageContent(type=MessageType.TEXT.value, content="Unmatched user message")
        messages = [
            Message(sequence=1, role=Role.USER.value, contents=[user_content]),
            Message(sequence=2, role=Role.ASSISTANT.value, contents=[asst_content]),
            Message(sequence=3, role=Role.USER.value, contents=[unmatched_user_content]),  # Unmatched user message
            Message(sequence=4, role=Role.USER.value, contents=[user_content]),
            Message(sequence=5, role=Role.ASSISTANT.value, contents=[asst_content]),
        ]
        conversation = Conversation(id="conv_warn", messages=messages)
        dataset = EvalDataset(id="dataset_warn", conversations=[conversation])

        with catch_warnings(record=True) as w:
            simplefilter("always")
            turns = TurnLoader.load_turns_from_dataset(dataset)
            self.assertTrue(any("User message" in str(warn.message) for warn in w))

    def test_unmatched_asst_message_mid_convo_triggers_warning(self):
        user_content = MessageContent(type=MessageType.TEXT.value, content="Matched user message")
        asst_content = MessageContent(type=MessageType.TEXT.value, content="Matched assistant message")
        unmatched_user_content = MessageContent(type=MessageType.TEXT.value, content="Unmatched user message")
        messages = [
            Message(sequence=1, role=Role.USER.value, contents=[user_content]),
            Message(sequence=2, role=Role.ASSISTANT.value, contents=[asst_content]),
            Message(sequence=3, role=Role.ASSISTANT.value, contents=[unmatched_user_content]),  # Unmatched assistant message
            Message(sequence=4, role=Role.USER.value, contents=[user_content]),
            Message(sequence=5, role=Role.ASSISTANT.value, contents=[asst_content]),
        ]
        conversation = Conversation(id="conv_warn", messages=messages)
        dataset = EvalDataset(id="dataset_warn", conversations=[conversation])

        with catch_warnings(record=True) as w:
            simplefilter("always")
            turns = TurnLoader.load_turns_from_dataset(dataset)
            self.assertTrue(any("Skipping message" in str(warn.message) for warn in w))


if __name__ == "__main__":
    unittest.main()
