import logging
import os
import traceback

import torch
from torch.nn.functional import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from environs import bucket_name, model_id, model_provider, promptguard_label_index
from environs import huggingface_model_url as configured_huggingface_model_url
from src.logs import LogEvent
from utils import download_model_from_hf, download_model_from_s3, get_model_id_from_ddb

logger = logging.getLogger(__name__)
log_event = LogEvent()

local_model_path = f"/tmp/{model_id}"

MODEL_SESSION = None
TOKENIZER = None
MODEL_INIT = False
LABEL_INDEX = None

# Class labels that represent a prompt-injection / jailbreak attempt. Checked in
# order, which is what lets one implementation serve both model generations:
# Prompt-Guard v1 is 3-class (BENIGN / INJECTION / JAILBREAK -> index 2) and
# Prompt-Guard 2 is 2-class (benign / malicious -> index 1).
MALICIOUS_LABELS = ("jailbreak", "malicious", "injection")


def model_is_present(local_model_path):
    """Whether a loadable model already sits on local disk.

    Tests for `config.json` rather than the directory, because a failed S3
    download leaves the directory behind: `download_model_from_s3` creates it
    before it discovers there is nothing to fetch. Checking the directory would
    make a later attempt try to load an empty path instead of falling back to
    HuggingFace.
    """
    return os.path.exists(os.path.join(local_model_path, "config.json"))


def resolve_huggingface_model_url():
    """Where to pull weights from: environment first, model registry second.

    Setting `huggingface_model_url` avoids the DynamoDB registry entirely, which
    is how a deployment without a model catalog runs.
    """
    if configured_huggingface_model_url:
        return configured_huggingface_model_url

    huggingface_model_url, context_length = get_model_id_from_ddb(
        model_id, model_provider
    )
    logger.info(
        log_event.format(
            "model_registry_lookup",
            model_id=model_id,
            model_provider=model_provider,
            context_length=context_length,
        )
    )
    return huggingface_model_url


def resolve_label_index(model):
    """Which softmax index holds the malicious probability for this model."""
    if promptguard_label_index:
        return int(promptguard_label_index)

    id2label = dict(getattr(model.config, "id2label", {}) or {})
    for wanted in MALICIOUS_LABELS:
        for index, label in id2label.items():
            if str(label).strip().lower() == wanted:
                return int(index)

    # No recognisable label name: the malicious class is the last one by
    # convention in both Prompt-Guard generations.
    num_labels = int(getattr(model.config, "num_labels", len(id2label) or 2))
    return num_labels - 1


def initialize_model(model_id, chat_format=None, context_length=4096):
    global MODEL_SESSION
    global TOKENIZER
    global MODEL_INIT
    global LABEL_INDEX
    MODEL_INIT = True
    # Check if model exists locally
    local_model_path = f"/tmp/{model_id}"
    if model_is_present(local_model_path):
        logger.info(log_event.format("model_found", model_id=model_id))
    else:
        # Check if model exists in S3
        try:
            logger.info(log_event.format("model_check", model_id=model_id))
            download_model_from_s3(bucket_name, model_id, local_model_path)
            if not model_is_present(local_model_path):
                raise ValueError("S3 cache did not contain a loadable model.")
        except Exception as e:
            logger.info(log_event.format("s3_download_failed", error=str(e)))
            MODEL_INIT = False

    # Initialize the model session
    if MODEL_INIT:
        logger.info(log_event.format("model_loading", model_id=model_id))
        logger.info(log_event.format("model_path", local_model_path=local_model_path))
        TOKENIZER = AutoTokenizer.from_pretrained(local_model_path)
        MODEL_SESSION = AutoModelForSequenceClassification.from_pretrained(
            local_model_path
        )
        MODEL_SESSION.eval()
        LABEL_INDEX = resolve_label_index(MODEL_SESSION)
        logger.info(
            log_event.format(
                "model_loaded", model_id=model_id, label_index=LABEL_INDEX
            )
        )


try:
    initialize_model(model_id)
except Exception as e:
    raise RuntimeError(f"Failed to initialize model {model_id}: {e}")


def get_class_probabilities(model, tokenizer, text, temperature=1.0, device="cpu"):
    """
    Evaluate the model on the given text with temperature-adjusted softmax.
    Note, as this is a DeBERTa model, the input text should have a maximum length of 512.

    Args:
            text (str): The input text to classify.
            temperature (float): The temperature for the softmax function. Default is 1.0.
            device (str): The device to evaluate the model on.

    Returns:
            torch.Tensor: The probability of each class adjusted by the temperature.
    """
    # Encode the text
    inputs = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=512
    ).to(device)
    # Get logits from the model
    with torch.no_grad():
        logits = model(**inputs).logits
    # Apply temperature scaling
    scaled_logits = logits / temperature
    # Apply softmax to get probabilities
    probabilities = softmax(scaled_logits, dim=-1)
    return probabilities


def eval_jailbreak_PI(model, tokenizer, text, temperature=1.0, device="cpu"):
    """
    Evaluate the probability that a given string contains malicious jailbreak or prompt injection.
    Appropriate for filtering dialogue between a user and an LLM.

    Args:
            text (str): The input text to evaluate.
            temperature (float): The temperature for the softmax function. Default is 1.0.
            device (str): The device to evaluate the model on.

    Returns:
            float: The probability of the text containing malicious content.
    """
    probabilities = get_class_probabilities(model, tokenizer, text, temperature, device)
    return probabilities[0, LABEL_INDEX].item()


def eval_promptguard(query):
    # If model was not initialized, download from HF and init model
    if MODEL_INIT is False:
        huggingface_model_url = resolve_huggingface_model_url()
        logger.info(
            log_event.format(
                "model_download_hf",
                huggingface_model_url=huggingface_model_url,
                model_id=model_id,
            )
        )
        download_model_from_hf(huggingface_model_url, model_id, bucket_name)
        # Load the weights that were just fetched. Without this the globals below
        # are still unset and scoring fails.
        logger.info(log_event.format("model_init_retry", model_id=model_id))
        initialize_model(model_id)
        if MODEL_INIT is False:
            raise RuntimeError(
                f"Failed to initialize model {model_id} after downloading from Hugging Face."
            )
    try:
        probability = eval_jailbreak_PI(MODEL_SESSION, TOKENIZER, text=query)
        return probability
    except Exception as e:
        logger.error(
            log_event.format(
                "pipeline_failed", error=e, error_trace=traceback.format_exc()
            )
        )
        raise RuntimeError(f"Error generating model response: {e}")
