import json
import logging
import os
from datetime import timedelta
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Dict, Optional, Union

import boto3
import botocore.session
from aws_secretsmanager_caching import SecretCache, SecretCacheConfig

from environs import (
    HUGGINGFACE_KEY_NAME,
    OPENAI_KEY_NAME,
    api,
    region,
)
from src.logs import LogEvent, setup_logging

logger = logging.getLogger(__name__)
log_event = LogEvent()

model_config = "model_config_dev" if api == "dev" else "model_config"
s3_client = boto3.client("s3")

# Weight files pulled from HuggingFace. Both weight formats are allowed so a
# repo that ships only `pytorch_model.bin` still resolves.
MODEL_FILE_PATTERNS = [
    "config.json",
    "model.safetensors",
    "pytorch_model.bin",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
]


def configure_logging(query_id: str, request_id: Optional[str]):
    setup_logging(log_level=log_level_upper(), local=False)
    LogEvent.query_id = query_id
    LogEvent.request_id = request_id


def log_level_upper() -> str:
    return os.environ.get("log_level", "DEBUG").upper()


def round_decimal(val, precision=10):
    return Decimal(val).quantize(Decimal(10) ** -precision, rounding=ROUND_HALF_UP)


def get_secret(secret_search_key: str) -> Union[Dict[str, Any], str]:
    client = botocore.session.get_session().create_client(
        "secretsmanager", region_name=region
    )
    cache_config = SecretCacheConfig(
        secret_refresh_interval=int(timedelta(hours=5).total_seconds())
    )
    cache = SecretCache(config=cache_config, client=client)
    try:
        secret = cache.get_secret_string(secret_search_key)

        if secret is None:
            secret = client.get_secret_value(SecretId=secret_search_key)["SecretString"]
            cache.put_secret_string(secret_search_key, secret)
    except Exception as e:
        print(e)
        if "ResourceNotFoundException" in str(type(e)):
            return {}
        raise e

    try:
        return json.loads(secret)
    except json.JSONDecodeError:
        return secret


class Auth:
    @staticmethod
    def setup_openai():
        if not os.environ.get(OPENAI_KEY_NAME):
            kw = get_secret(f"openai-{api}")
            if kw:
                os.environ[OPENAI_KEY_NAME] = kw["api_key"]

    @staticmethod
    def setup_huggingface():
        if not os.environ.get(HUGGINGFACE_KEY_NAME):
            kw = get_secret(f"huggingface-{api}")
            if kw:
                os.environ[HUGGINGFACE_KEY_NAME] = kw["api_key"]


# -----------------------------------------------------------------------------
# Model weights
# -----------------------------------------------------------------------------


def download_model_from_s3(bucket_name, key, local_path):
    """
    Downloads a model or an entire folder from S3 to a local directory.

    Args:
            bucket_name (str): The name of the S3 bucket.
            key (str): The key (file or folder path) in the S3 bucket.
            local_path (str): The local path to save the downloaded file(s).
    """
    if not bucket_name:
        raise ValueError("No S3 weight cache configured (bucket_name is unset).")

    # Ensure the local directory exists
    os.makedirs(local_path, exist_ok=True)

    try:
        # List all objects in the specified S3 folder
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=key)
        if "Contents" not in response:
            raise ValueError("No files found in the specified S3 folder.")

        # Log the files being downloaded
        logger.info(
            log_event.format(
                f"Files found in S3 bucket '{bucket_name}' under '{key}': {response['Contents']}"
            )
        )

        for obj in response["Contents"]:
            s3_file_key = obj["Key"]
            relative_path = os.path.relpath(s3_file_key, key)
            local_file_path = os.path.join(local_path, relative_path)

            # Ensure the local directory structure exists
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

            # Download each file if it doesn't already exist locally
            if not os.path.exists(local_file_path):
                logger.info(
                    log_event.format(f"Downloading {s3_file_key} to {local_file_path}")
                )
                s3_client.download_file(bucket_name, s3_file_key, local_file_path)
                logger.info(log_event.format(f"Downloaded: {local_file_path}"))
            else:
                logger.info(log_event.format(f"File already exists: {local_file_path}"))
    except Exception as e:
        raise Exception(f"Error while downloading folder: {e}")


def hf_repo_id(huggingface_model_url):
    """Normalise a model reference into a HuggingFace repo id.

    Accepts a full URL (`https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M/`,
    which is the form stored in the DynamoDB model registry) or a bare
    `org/model` id, and returns `org/model`.
    """
    if not huggingface_model_url:
        return None

    repo = str(huggingface_model_url).strip().rstrip("/")
    for prefix in (
        "https://huggingface.co/",
        "http://huggingface.co/",
        "huggingface.co/",
    ):
        if repo.startswith(prefix):
            repo = repo[len(prefix) :]
            break
    return repo.strip("/")


def download_model_from_hf(huggingface_model_url, model_id, bucket_name):
    """
    Downloads a model from Hugging Face and uploads it to an S3 bucket.

    Args:
            huggingface_model_url (str): HuggingFace URL or `org/model` repo id.
            model_id (str): The unique identifier for the model.
            bucket_name (str): The name of the S3 bucket to upload the model to.
                    Falsy skips the upload, leaving the weights in /tmp only.

    Raises on failure. A partially-downloaded model directory cannot be loaded,
    so there is nothing useful to continue with.
    """
    from huggingface_hub import snapshot_download

    # Grab HF token from AWS-SM
    Auth.setup_huggingface()
    HUGGINGFACE_KEY = os.environ.get(HUGGINGFACE_KEY_NAME)

    repo_id = hf_repo_id(huggingface_model_url)
    if not repo_id:
        raise ValueError(
            "No HuggingFace model configured: set the huggingface_model_url "
            "environment variable, or add a row to the model registry table."
        )

    local_path = f"/tmp/{model_id}"
    logger.info(
        log_event.format("hf_download_start", repo_id=repo_id, local_path=local_path)
    )

    # Prompt-Guard is a gated repo, so a missing or unauthorised token surfaces
    # here as GatedRepoError / RepositoryNotFoundError rather than a partial
    # download.
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_path,
        token=HUGGINGFACE_KEY or None,
        allow_patterns=MODEL_FILE_PATTERNS,
    )
    logger.info(log_event.format("hf_download_complete", repo_id=repo_id))

    if not bucket_name:
        logger.info(log_event.format("hf_cache_upload_skipped", reason="no bucket_name"))
        return

    # Mirror into S3 so later cold starts load from the account, not the internet.
    for root, _dirs, files in os.walk(local_path):
        for file_name in files:
            local_file = os.path.join(root, file_name)
            relative_path = os.path.relpath(local_file, local_path)
            s3_key = f"{model_id}/{relative_path}"
            logger.info(
                log_event.format(
                    "hf_cache_upload", bucket_name=bucket_name, s3_key=s3_key
                )
            )
            s3_client.upload_file(local_file, bucket_name, s3_key)


def get_model_id_from_ddb(model_id, model_provider):
    dynamodb = boto3.resource("dynamodb", region_name=region)
    table = dynamodb.Table(model_config)
    response = table.get_item(Key={"id": model_id, "api": model_provider})
    item = response.get("Item")
    if not item:
        raise ValueError(f"Model {model_id} not found in DynamoDB")
    huggingface_model_url = item.get("huggingface_model_url")
    context_length = item.get("context_length", 4096)
    context_length = int(context_length)
    return huggingface_model_url, context_length


# -----------------------------------------------------------------------------
# DynamoDB streams event parsing
# -----------------------------------------------------------------------------


def get_dynamodb_eval_body(event, index=0):
    return event.get("Records")[index]


def get_dynamodb_new_image(event, index=0):
    return get_dynamodb_eval_body(event, index)["dynamodb"]["NewImage"]


def get_dynamodb_event_api(event, index=0):
    return get_dynamodb_eval_body(event, index)["dynamodb"]["Keys"]["api"]["S"]


def get_dynamodb_event_query_id(event, index=0):
    return get_dynamodb_eval_body(event, index)["dynamodb"]["Keys"]["id"]["S"]


def hydrate_s3_fields(new_image: dict):
    """Fetch full payload from S3 if s3_offload_key is present in the DDB stream image.

    Returns the full payload dict if offloaded, or None if not.
    """
    s3_uri = new_image.get("s3_offload_key", {}).get("S")
    if not s3_uri:
        return None

    try:
        parts = s3_uri.replace("s3://", "").split("/", 1)
        s3_bucket, s3_key = parts[0], parts[1]
        response = s3_client.get_object(Bucket=s3_bucket, Key=s3_key)
        full_payload = json.loads(response["Body"].read().decode("utf-8"))
        return full_payload
    except Exception as e:
        logger.error(log_event.format("s3_hydrate_failed", error=str(e)))
        return None


def _join_hydrated(raw, provider=None):
    """Flatten a field from an S3-hydrated (plain JSON) payload into a string."""
    if raw is None:
        return ""
    if isinstance(raw, list):
        if provider == "aws":
            return "".join(
                item["text"] if isinstance(item, dict) else item for item in raw
            )
        return "".join(item if isinstance(item, str) else str(item) for item in raw)
    return raw or ""


def _join_attribute(attribute, provider=None):
    """Flatten a field from a raw DynamoDB stream image into a string.

    Long strings are chunked into a list by the producer to stay under
    DynamoDB's item size limit, and Bedrock system prompts arrive as a list of
    `{"M": {"text": {"S": ...}}}` blocks.
    """
    if not attribute:
        return ""

    value = attribute.get("L")
    if value is None:
        return attribute.get("S", "")

    if isinstance(value, list):
        if provider == "aws":
            return "".join(
                item["M"]["text"]["S"] if "M" in item else item["S"] for item in value
            )
        return "".join(item["S"] for item in value)
    return value.get("S", "")


def extract_dynamodb_inputs(new_image: dict, provider=None):
    """Return the three text fields GUARD scores, as plain strings.

    Transparently resolves rows whose text was offloaded to S3 because it
    exceeded DynamoDB's 400 KB item limit.
    """
    hydrated = hydrate_s3_fields(new_image)

    if hydrated:
        return {
            "query": _join_hydrated(hydrated.get("query", [])),
            "response": _join_hydrated(hydrated.get("response", [])),
            "sys_prompt": _join_hydrated(
                hydrated.get("system_prompt", []), provider=provider
            ),
        }

    return {
        "query": _join_attribute(new_image.get("query")),
        "response": _join_attribute(new_image.get("response")),
        "sys_prompt": _join_attribute(
            new_image.get("system_prompt"), provider=provider
        ),
    }
