import unittest
import sys
import os

# Ensure project root on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from pre_deploy.loaders.deepeval_test_cases import convert_turns_to_deepeval_conversational_test_cases, dataset_to_deepeval_conversational_test_cases
from pre_deploy.loaders.turn import Turn
from pre_deploy.loaders.turn_loader import TurnLoader
from pre_deploy.input.dataset import EvalDataset
from pre_deploy.input.conversation import Conversation
from pre_deploy.input.message import Message
from pre_deploy.input.message_content import MessageContent


class TestDeepEvalLoaders(unittest.TestCase):
    def _build_message(self, seq: int, role: str, text: str):
        return Message(sequence=seq, role=role, contents=[MessageContent(type='text', content=text)])

    def test_convert_turns_single_conversation_single_turn(self):
        user_msg = self._build_message(1, 'user', 'Hello')
        asst_msg = self._build_message(2, 'assistant', 'Hi there')
        turn = Turn(dataset_id='ds1', conversation_id='c1', turn_number=1, user_message=user_msg, assistant_message=asst_msg)
        raw = {'c1': [turn]}
        test_cases = convert_turns_to_deepeval_conversational_test_cases(raw)
        self.assertIn('c1', test_cases)
        tc = test_cases['c1']
        # Expect two turns (user + assistant)
        self.assertEqual(len(tc.turns), 2)
        self.assertEqual(tc.turns[0].role, 'user')
        self.assertEqual(tc.turns[0].content, 'Hello')
        self.assertEqual(tc.turns[1].role, 'assistant')
        self.assertEqual(tc.turns[1].content, 'Hi there')

    def test_dataset_to_deepeval_multiple_turns(self):
        # Build conversation with 2 user-assistant pairs (4 messages)
        msgs = [
            self._build_message(1, 'user', 'Q1'),
            self._build_message(2, 'assistant', 'A1'),
            self._build_message(3, 'user', 'Q2'),
            self._build_message(4, 'assistant', 'A2'),
        ]
        conv = Conversation(id='c1', messages=msgs)
        dataset = EvalDataset(id='dsX', conversations=[conv])
        turns_dict = TurnLoader.load_turns_from_dataset(dataset)
        self.assertEqual(len(turns_dict['c1']), 2)  # two internal turns
        test_cases = dataset_to_deepeval_conversational_test_cases(dataset)
        self.assertIn('c1', test_cases)
        tc = test_cases['c1']
        # 2 pairs -> 4 turns for deepeval
        self.assertEqual(len(tc.turns), 4)
        roles = [t.role for t in tc.turns]
        self.assertEqual(roles, ['user', 'assistant', 'user', 'assistant'])


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
