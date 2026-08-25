from typing import Optional

from pydantic import BaseModel


class ModelRequest(BaseModel):
    """Envelope fields shared by every request.

    Both are optional: neither GUARD check uses a judge model, so they are
    accepted for backwards compatibility with callers that still send them.
    """

    model_provider: Optional[str] = None
    model_name: Optional[str] = None


class PromptGuardPayload(BaseModel):
    query: str


class ContentModerationPayload(BaseModel):
    user_input: str
