"""
Simple production-style comparison script for answer_relevancy implementations.

Runs both the legacy batch and staged implementations sequentially
against the same dataset to observe their outputs side-by-side.
No mocking, no assertions - just execution and output inspection.
"""
import sys
from pathlib import Path
import uuid
import time

sys.path.append(Path(__file__).resolve().parent.parent.parent.as_posix())

from pre_deploy.query_processor import RequestDict
from pre_deploy.metrics.answer_relevancy import (
    answer_relevancy_batch_generate,
    start_answer_relevancy,
    advance_answer_relevancy,
    finalize_answer_relevancy,
)
from pre_deploy.output.results import MetricsResults
from tests.utils import create_eval_dataset_for_testing


def create_test_dataset():
    """Create a small test dataset."""
    test_set = [
        {
            "input": "What is Python?",
            "actual_output": "Python is a high-level, interpreted programming language known for its simplicity.",
        },
        {
            "input": "When was Python released?",
            "actual_output": "Python was first released in 1991 by Guido van Rossum.",
        },
    ]

    return create_eval_dataset_for_testing(
        test_set,
        metadata={"dataset_version_id": "comparison-test-v1"}
    )


def poll_until_ready(action, action_name, poll_interval_seconds=5, max_attempts=60):
    """Poll a staged action until it is ready or raise on timeout."""
    for attempt in range(1, max_attempts + 1):
        result = action()
        blocked_status = result if (
            isinstance(result, dict)
            and "is_complete" in result
            and not result.get("is_complete", True)
        ) else None

        if result is not False and blocked_status is None:
            return result

        if blocked_status is not None:
            completed = blocked_status.get("completed_items", 0)
            total = blocked_status.get("total_items", completed + blocked_status.get("pending_items", 0))
            print(
                f"  {action_name}: waiting on prerequisite phase "
                f"({completed}/{total}) "
                f"(attempt {attempt}/{max_attempts}); sleeping {poll_interval_seconds}s"
            )
            time.sleep(poll_interval_seconds)
            continue

        print(
            f"  {action_name}: not ready yet "
            f"(attempt {attempt}/{max_attempts}); sleeping {poll_interval_seconds}s"
        )
        time.sleep(poll_interval_seconds)

    raise TimeoutError(
        f"Timed out waiting for {action_name} after {max_attempts} attempts"
    )


def main():
    """Run both implementations and compare outputs."""
    print("\n" + "="*80)
    print("ANSWER_RELEVANCY: BATCH vs STAGED IMPLEMENTATION COMPARISON")
    print("="*80 + "\n")

    # Create shared dataset
    eval_dataset = create_test_dataset()
    print(f"Dataset: {len(eval_dataset.conversations)} conversations")
    print(f"Dataset version: {eval_dataset.metadata.get('dataset_version_id')}\n")

    # ============================================================================
    # RUN 1: BATCH IMPLEMENTATION
    # ============================================================================
    print("-" * 80)
    print("1. BATCH IMPLEMENTATION (Legacy Synchronous)")
    print("-" * 80)

    batch_run_id = str(uuid.uuid4())
    batch_run_info = RequestDict(
        metric_name="answer_relevancy",
        metric_phase="",
        run_id=batch_run_id,
        username="test_user",
        model_name="gpt4o_mini",
        model_provider="openai",
    )

    print(f"Run ID: {batch_run_id}")
    print("Executing: answer_relevancy_batch_generate()")

    try:
        batch_results = answer_relevancy_batch_generate(
            evaluator_info=batch_run_info,
            eval_dataset=eval_dataset,
            threshold=0.5,
            include_reason=True,
            strict_mode=False,
            verbose_mode=False,
        )

        if not isinstance(batch_results, MetricsResults):
            raise TypeError("Expected MetricsResults-like object from batch implementation")
        batch_dict = batch_results.to_dict()

        print(f"Status: SUCCESS\n")
        print("Results:")
        for test_id, result in batch_dict["results"].items():
            print(f"  Test {test_id}:")
            print(f"    Score:   {result['score']:.3f}")
            print(f"    Success: {result['success']}")
            print(f"    Reason:  {result['reason'][:100]}...")

    except Exception as e:
        print(f"Status: FAILED")
        print(f"Error: {type(e).__name__}: {e}\n")
        batch_results = None
        batch_dict = None

    # ============================================================================
    # RUN 2: STAGED IMPLEMENTATION
    # ============================================================================
    print("\n" + "-" * 80)
    print("2. STAGED IMPLEMENTATION (Distributed Phases)")
    print("-" * 80)

    staged_run_id = str(uuid.uuid4())
    staged_run_info = RequestDict(
        metric_name="answer_relevancy",
        metric_phase="",
        run_id=staged_run_id,
        username="test_user",
        model_name="gpt4o_mini",
        model_provider="openai",
    )

    print(f"Run ID: {staged_run_id}")

    try:
        # Phase 1: Start (enqueue statements)
        print("\nPhase 1: start_answer_relevancy()")
        start_result = start_answer_relevancy(
            evaluator_info=staged_run_info,
            eval_dataset=eval_dataset,
        )
        print(f"  Result: {start_result}")

        # Phase 2: Poll until statements complete, then enqueue verdicts.
        print("\nPhase 2: poll and advance to verdicts")
        advance_result = poll_until_ready(
            action=lambda: advance_answer_relevancy(
                evaluator_info=staged_run_info,
                eval_dataset=eval_dataset,
                phase="verdicts",
            ),
            action_name="advance_answer_relevancy(phase='verdicts')",
        )
        print(f"  Result: {advance_result}")

        # Phase 3: Poll until verdicts complete, then enqueue reasons.
        print("\nPhase 3: poll and advance to reasons")
        reasons_result = poll_until_ready(
            action=lambda: advance_answer_relevancy(
                evaluator_info=staged_run_info,
                eval_dataset=eval_dataset,
                phase="reasons",
                include_reason=True,
            ),
            action_name="advance_answer_relevancy(phase='reasons')",
        )
        print(f"  Result: {reasons_result}")

        # Phase 4: Poll finalize until all required stages are complete.
        print("\nPhase 4: poll and finalize answer_relevancy")
        staged_results = poll_until_ready(
            action=lambda: finalize_answer_relevancy(
                evaluator_info=staged_run_info,
                eval_dataset=eval_dataset,
                threshold=0.5,
                include_reason=True,
                strict_mode=False,
                verbose_mode=False,
            ),
            action_name="finalize_answer_relevancy()",
        )

        if not isinstance(staged_results, MetricsResults):
            raise TypeError("Expected MetricsResults-like object from finalize action")
        staged_dict = staged_results.to_dict()
        print(f"Status: SUCCESS\n")
        print("Results:")
        for test_id, result in staged_dict["results"].items():
            print(f"  Test {test_id}:")
            print(f"    Score:   {result['score']:.3f}")
            print(f"    Success: {result['success']}")
            print(f"    Reason:  {result['reason'][:100]}...")

    except Exception as e:
        print(f"Status: FAILED")
        print(f"Error: {type(e).__name__}: {e}\n")
        staged_results = None
        staged_dict = None

    # ============================================================================
    # COMPARISON
    # ============================================================================
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)

    if batch_dict and staged_dict:
        print("\nScores comparison:")
        for test_id in batch_dict["results"].keys():
            batch_score = batch_dict["results"][test_id]["score"]
            staged_score = staged_dict["results"][test_id]["score"]
            diff = abs(batch_score - staged_score)
            print(f"  Test {test_id}:")
            print(f"    Batch:  {batch_score:.3f}")
            print(f"    Staged: {staged_score:.3f}")
            print(f"    Diff:   {diff:.6f}")

        print("\nSuccess flags comparison:")
        for test_id in batch_dict["results"].keys():
            batch_success = batch_dict["results"][test_id]["success"]
            staged_success = staged_dict["results"][test_id]["success"]
            match = "✓" if batch_success == staged_success else "✗"
            print(f"  Test {test_id}: {match} (batch={batch_success}, staged={staged_success})")

    elif batch_dict:
        print("\nBatch succeeded, staged did not complete for comparison")

    elif staged_dict:
        print("\nStaged succeeded, batch did not complete for comparison")

    else:
        print("\nBoth implementations failed - no comparison possible")

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
