"""Download the zero-shot classification model for local inference.

Usage:
    python scripts/download_model.py [--output-dir ./models/zeroshot]

This saves the model and tokenizer locally so ZeroShotMetric can run
without internet access (e.g., inside a container).
"""

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
DEFAULT_OUTPUT_DIR = "./models/zeroshot_offline"


def save_model_locally(model_name: str = DEFAULT_MODEL, save_directory: str = DEFAULT_OUTPUT_DIR):
    """
    Download and save the zero-shot classification model and tokenizer locally.

    See:
    https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.save_pretrained
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    logger.info("Downloading model: %s", model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    logger.info("Saving to: %s", save_directory)
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    logger.info("Done. Model and tokenizer saved at: %s", save_directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download the zero-shot classification model.")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help=f"HuggingFace model name (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR, help=f"Local save directory (default: {DEFAULT_OUTPUT_DIR})"
    )
    args = parser.parse_args()
    save_model_locally(model_name=args.model, save_directory=args.output_dir)
