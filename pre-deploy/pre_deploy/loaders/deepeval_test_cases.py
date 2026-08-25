"""Helpers for converting internal Turn representations into DeepEval test cases."""
from typing import Dict, List, cast

from deepeval.test_case import ConversationalTestCase, LLMTestCase, Turn as DeepEvalTurn

from ..input.dataset import EvalDataset
from ..input import Message, Conversation, MessageContent
from .turn_loader import TurnLoader
from .turn import Turn


def convert_turns_to_deepeval_conversational_test_cases(raw_turns: Dict[str, List[Turn]]) -> Dict[str, ConversationalTestCase]:
    """Convert a mapping of conversation id -> list[Turn] into DeepEval ConversationalTestCase objects."""
    test_cases: Dict[str, ConversationalTestCase] = {}
    for convo_id, turns in raw_turns.items():
        deepeval_turns: List[DeepEvalTurn] = []
        for turn in turns:
            # Each internal Turn has a user_message and assistant_message; map them into DeepEval's Turn objects.
            # Safely coerce role strings into the expected literal set (DeepEval usually expects 'user'/'assistant').
            user_role = str(turn.user_message.role)
            asst_role = str(turn.assistant_message.role)
            user_content = cast(str, (turn.user_message.contents[0].content or ""))
            asst_content = cast(str, (turn.assistant_message.contents[0].content or ""))
            user_turn = DeepEvalTurn(role=user_role, content=user_content)
            assistant_turn = DeepEvalTurn(role=asst_role, content=asst_content)
            deepeval_turns.append(user_turn)
            deepeval_turns.append(assistant_turn)
        test_cases[convo_id] = ConversationalTestCase(turns=deepeval_turns)
    return test_cases


def dataset_to_deepeval_conversational_test_cases(dataset: EvalDataset) -> Dict[str, ConversationalTestCase]:
    """Loads internal Turns from an EvalDataset and converts them
    to DeepEval ConversationalTestCase objects."""
    raw_turns = TurnLoader.load_turns_from_dataset(dataset)
    return convert_turns_to_deepeval_conversational_test_cases(raw_turns)


def dataset_to_deepeval_llm_test_cases(eval_dataset: EvalDataset) -> Dict[str, LLMTestCase]:
    """Convert a mapping of conversation id -> LLMTestCase into DeepEval LLMTestCase objects."""
    test_cases: Dict[str, LLMTestCase] = {}

    for convo in eval_dataset.conversations:

        input = convo.messages[0].contents[0].content
        actual_output = convo.messages[1].contents[0].content
        expected_output = convo.metadata.get("expected_output") if convo.metadata is not None else None
        context = convo.metadata.get("context") if convo.metadata is not None else None

        test_cases[convo.id] = LLMTestCase(input=input,
                                           expected_output=expected_output,
                                           actual_output=actual_output,
                                           context=context)

    return test_cases


def deepeval_llm_test_cases_to_dataset(test_cases: List[LLMTestCase], dataset_id: str) -> EvalDataset:
    """Convert DeepEval LLMTestCase objects into an EvalDataset."""
    conversations = []

    for idx, test_case in enumerate(test_cases):
        messages = []

        sequence = 0

        # Add input as user message
        if test_case.input:
            messages.append(Message(
                sequence=sequence,
                role="user",
                contents=[MessageContent(type="text", content=test_case.input)]
            ))
            sequence += 1

        # Add actual output as assistant message
        if test_case.actual_output:
            messages.append(Message(
                sequence=sequence,
                role="assistant",
                contents=[MessageContent(type="text", content=test_case.actual_output)]
            ))
            sequence += 1

        # Create conversation for this test case
        tmp_convo = Conversation(id=str(idx), messages=messages)
        if test_case.expected_output is not None:
            tmp_convo.metadata = {"expected_output": test_case.expected_output}

        conversations.append(tmp_convo)

    eval_dataset = EvalDataset(id=dataset_id, conversations=conversations)
    return eval_dataset


__all__ = [
    "dataset_to_deepeval_conversational_test_cases",
    "convert_turns_to_deepeval_conversational_test_cases",
    "dataset_to_deepeval_llm_test_cases",
    "deepeval_llm_test_cases_to_dataset",
]
