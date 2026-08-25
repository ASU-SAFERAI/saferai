"""
A test script to assess the determinism and consistency of specific judge models
when running accuracy_with_reference evaluations on a set of edge case conversations.
The script outputs a report summarizing the performance of each model on the test cases,
highlighting any inconsistencies or failures.
"""
import uuid
from time import sleep

from pre_deploy.query_processor.request_dict import RequestDict
from pre_deploy.query_processor.environment import AWSEnvironment
from pre_deploy.output import MetricsResults
from pre_deploy.metrics.accuracy_with_reference import start_accuracy_with_reference, finalize_accuracy_with_reference
import pandas as pd

from dataset import dataset_fixture

CANDIDATE_MODELS = [
    {"name": "gpt5_4_nano", "provider": "openai"},
    {"name": "gpt5_4_mini", "provider": "openai"},
    {"name": "gpt-oss-120b", "provider": "aws"},
    {"name": "llama4_maverick-17b", "provider": "aws"},
    {"name": "llama4_scout-17b", "provider": "aws"},
    {"name": "geminiflash3_1_lite", "provider": "gcp-deepmind"},
    {"name": "geminiflash3", "provider": "gcp-deepmind"},
    {"name": "gemma4_31b_it", "provider": "sol"}
]


def convert_results_to_report(model_scores):
    report_rows = []
    for model_name, iteration, scores in model_scores:
        for idx, result in scores["results"].items():
            report_rows.append({
                "model": model_name,
                "iteration": iteration,
                "test_case_id": idx,
                "score": result["score"],
                "reason": result["reason"],
                "success": result["success"]
            })
    return pd.DataFrame(report_rows)


def test_accuracy_with_reference_on_edge_cases():
    dataset = dataset_fixture
    environment = AWSEnvironment(None, None)

    model_scores = []

    for i in range(10):  # Run multiple iterations to check for consistency
        for model_info in CANDIDATE_MODELS:
            evaluator_info = RequestDict(
                username="test_user",
                run_id=str(uuid.uuid4()),
                model_name=model_info["name"],
                model_provider=model_info["provider"],
                model_temperature=0.0,
            )

            results = start_accuracy_with_reference(
                evaluator_info=evaluator_info,
                eval_dataset=dataset,
                threshold=0.5,
                environment=environment,
                force_rerun=True,
            )

            print(f"Initial evaluation complete for model {model_info['name']}.\n\n{results}...")

            finalized_results = None

            while True:
                finalized_results = finalize_accuracy_with_reference(
                    evaluator_info=evaluator_info,
                    eval_dataset=dataset,
                    threshold=0.5,
                    environment=environment
                )
                if isinstance(finalized_results, MetricsResults):
                    break
                print(f"Finalization not complete for model {model_info['name']}.\nStatus:{finalized_results}\nRetrying...")

                sleep(10)  # Wait for 10 seconds before retrying finalization

            scores = finalized_results.to_dict()
            model_scores.append((model_info["name"], i+1, scores))

    # Output the scores for all models
    report_df = convert_results_to_report(model_scores)
    report_df.to_csv("accuracy_with_reference_edge_case_report.csv", index=False)


def post_process_report():
    report_df = pd.read_csv("accuracy_with_reference_edge_case_report.csv")
    # Multi-tier indexing by model, conversation, and iteration
    report_df.set_index(["model", "iteration", "test_case_id"], inplace=True)
    # Calculate mean and standard deviation of scores for each model and test case
    summary_df = report_df.groupby(["model", "test_case_id"]).agg(
        mean_score=pd.NamedAgg(column="score", aggfunc="mean"),
        std_score=pd.NamedAgg(column="score", aggfunc="std"),
    )
    summary_df.to_csv("accuracy_with_reference_edge_case_summary.csv")



if __name__ == "__main__":
    # test_accuracy_with_reference_on_edge_cases()
    post_process_report()
