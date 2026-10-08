"""Data-mart credential loading (ported/adapted from the pre-release container).

Resolution order, highest precedence first:

1. Explicit environment overrides (``DATA_MART_USERNAME``/``_PASSWORD``/``_HOST``/
   ``_DATABASE``/``_PORT``).
2. A ``DATA_MART_NONPROD`` / ``DATA_MART_PROD`` section in a credentials conf
   file (defaults to the checked-in agentic conf path, overridable via
   ``DATA_MART_CONFIG_PATH`` / ``AGENTIC_CONFIG_PATH``).
3. AWS Secrets Manager (``DATA_MART_SECRET_NAME`` or the pre-release default),
   used only when no conf file is present.

NONPROD is the default target; PROD must be selected explicitly. Credentials are
never logged, and the ``DataMartCredentials`` repr redacts the password.

Security note: the checked-in ``varun_credentials.conf`` currently holds
plaintext data-mart passwords. Prefer environment overrides or Secrets Manager;
the committed plaintext secrets should be rotated and removed from source
control as a follow-up.
"""
from __future__ import annotations

import configparser
import os
from dataclasses import dataclass
from pathlib import Path

from custom_benchmarks.tool_calling.agentic.config import DEFAULT_CONFIG_PATH

# Environment identifiers for the two Redshift serverless workgroups.
ENV_NONPROD = "NONPROD"
ENV_PROD = "PROD"
DEFAULT_ENVIRONMENT = ENV_NONPROD

_SECTION_FOR_ENV = {
    ENV_NONPROD: "DATA_MART_NONPROD",
    ENV_PROD: "DATA_MART_PROD",
}

# Secrets Manager fallbacks (match the pre-release container's secret names).
_SECRET_FOR_ENV = {
    ENV_NONPROD: "ai-dm-ai-madkit-app-dev",
    ENV_PROD: "ai-dm-ai-madkit-app-prod",
}

_REQUIRED_KEYS = ("username", "password", "host", "database", "port")


@dataclass(frozen=True)
class DataMartCredentials:
    """Redshift connection parameters. The password is kept out of ``repr``."""

    username: str
    password: str
    host: str
    database: str
    port: str
    environment: str = DEFAULT_ENVIRONMENT

    def __repr__(self) -> str:  # pragma: no cover - defensive secret handling
        return (
            "DataMartCredentials(username=%r, password=<redacted>, host=%r, "
            "database=%r, port=%r, environment=%r)"
            % (self.username, self.host, self.database, self.port, self.environment)
        )


def _normalize_environment(environment: str | None) -> str:
    selected = (environment or os.getenv("DATA_MART_ENV") or DEFAULT_ENVIRONMENT).strip().upper()
    if selected not in _SECTION_FOR_ENV:
        raise ValueError(
            f"Unknown data-mart environment {selected!r}; expected one of "
            f"{sorted(_SECTION_FOR_ENV)}"
        )
    return selected


def _env_overrides() -> dict[str, str]:
    overrides = {}
    for key in _REQUIRED_KEYS:
        value = os.getenv(f"DATA_MART_{key.upper()}")
        if value is not None and value.strip():
            overrides[key] = value.strip()
    return overrides


def _from_secrets_manager(environment: str) -> dict[str, str]:
    """Load credentials from AWS Secrets Manager (lazy boto3 import)."""
    import base64
    import json

    import boto3
    from botocore.exceptions import ClientError

    secret_name = os.getenv("DATA_MART_SECRET_NAME") or _SECRET_FOR_ENV[environment]
    region = os.getenv("DATA_MART_SECRET_REGION", "us-west-2")
    client = boto3.session.Session().client("secretsmanager", region_name=region)
    try:
        response = client.get_secret_value(SecretId=secret_name)
    except ClientError as exc:  # pragma: no cover - network/credential dependent
        raise RuntimeError(f"Unable to retrieve data-mart secret {secret_name!r}") from exc
    if "SecretString" in response:
        secret = json.loads(response["SecretString"])
    else:
        secret = json.loads(base64.b64decode(response["SecretBinary"]))
    return {key: str(secret[key]) for key in _REQUIRED_KEYS if key in secret}


def load_data_mart_credentials(
    config_path: str | Path | None = None,
    environment: str | None = None,
) -> DataMartCredentials:
    """Resolve data-mart credentials for ``environment`` (default NONPROD)."""
    selected_env = _normalize_environment(environment)
    values: dict[str, str] = {}

    selected_path = Path(
        config_path
        or os.getenv("DATA_MART_CONFIG_PATH")
        or os.getenv("AGENTIC_CONFIG_PATH")
        or DEFAULT_CONFIG_PATH
    ).expanduser()

    if selected_path.exists():
        parser = configparser.ConfigParser(interpolation=None)
        parser.read(selected_path, encoding="utf-8")
        section = _SECTION_FOR_ENV[selected_env]
        if parser.has_section(section):
            for key in _REQUIRED_KEYS:
                if parser.has_option(section, key):
                    values[key] = parser.get(section, key).strip()

    # Environment overrides take precedence over the conf file.
    values.update(_env_overrides())

    missing = [key for key in _REQUIRED_KEYS if not values.get(key)]
    if missing:
        # Fall back to Secrets Manager only when the conf file could not supply
        # a complete credential set (mirrors the pre-release behavior).
        if not selected_path.exists():
            values = {**_from_secrets_manager(selected_env), **_env_overrides()}
            missing = [key for key in _REQUIRED_KEYS if not values.get(key)]
        if missing:
            raise ValueError(
                "Missing data-mart credential field(s) "
                f"{missing} for environment {selected_env}. Set DATA_MART_* "
                "environment variables, provide a credentials conf section "
                f"[{_SECTION_FOR_ENV[selected_env]}], or configure Secrets Manager."
            )

    return DataMartCredentials(
        username=values["username"],
        password=values["password"],
        host=values["host"],
        database=values["database"],
        port=str(values["port"]),
        environment=selected_env,
    )
