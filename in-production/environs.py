import os

# --- Deployment identity -----------------------------------------------------
api = os.environ.get("api", "dev")
env = os.environ.get("env", "poc")
region = os.environ.get("region", os.environ.get("AWS_REGION", "us-west-2"))

# --- Credential environment-variable names -----------------------------------
OPENAI_KEY_NAME = "OPENAI_API_KEY"
HUGGINGFACE_KEY_NAME = "HUGGINGFACE_KEY"

# --- Logging -----------------------------------------------------------------
local_run = "log_level" not in os.environ
log_level = os.environ.get("log_level", "DEBUG")

# --- Prompt-Guard classifier -------------------------------------------------
# `model_id` doubles as the S3 prefix and the /tmp directory name.
model_id = os.environ.get("model_id", "Prompt-Guard-2-86M")
# Sort key used when looking the model up in the DynamoDB model registry.
model_provider = os.environ.get("model_provider", "lambda")
# S3 cache for the weights. Unset means "no cache": go straight to HuggingFace.
bucket_name = os.environ.get("bucket_name")
# Set this to skip the DynamoDB registry entirely. Accepts a full HuggingFace URL
# or a bare `org/model` repo id.
huggingface_model_url = os.environ.get("huggingface_model_url")
# Override which softmax class is treated as the malicious probability. Normally
# left unset: the index is read from the model's own id2label mapping, which is
# what makes the same code correct for both Prompt-Guard v1 (3-class) and
# Prompt-Guard 2 (2-class).
promptguard_label_index = os.environ.get("promptguard_label_index")

# --- Content moderation ------------------------------------------------------
moderation_model = os.environ.get("moderation_model", "omni-moderation-latest")

# --- Score persistence (DynamoDB streams path) -------------------------------
eval_table = os.environ.get("model_eval_table", "model_eval_dev")
alert_threshold = float(os.environ.get("model_eval_threshold", "0.85"))

# --- Alerting (DynamoDB streams path) ----------------------------------------
critical_content_mod = os.environ.get("critical_content_mod_sns_topic")
general_content_mod = os.environ.get("general_content_mod_sns_topic")
prompt_injection_mod = os.environ.get("prompt_injection_content_mod_sns_topic")
alert_timezone = os.environ.get("alert_timezone", "America/Phoenix")
