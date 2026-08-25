from dataclasses import dataclass, asdict
from typing import Optional
import uuid


@dataclass
class RequestDict:
    username: str
    run_id: str
    metric_name: Optional[str] = None
    metric_phase: Optional[str] = None
    num_eval: Optional[int] = None
    max_task: int = 3
    model_name: Optional[str] = None
    model_provider: Optional[str] = None
    project_id: Optional[str] = None
    project_version: Optional[str] = None
    dataset_version_id: Optional[str] = "null"
    system_prompt: str = ""
    enable_search: bool = False
    search_collection: Optional[str] = None
    search_top_k: Optional[int] = None
    model_temperature: Optional[float] = None
    model_top_p: Optional[float] = None
    model_top_k: Optional[int] = None
    model_max_tokens: Optional[int] = None
    # A field that will be used to override existing project's model_name if provided. This is useful for testing new models without creating a new project.
    override_model_name: Optional[str] = None
    override_model_provider: Optional[str] = None

    def __post_init__(self):
        if not self.username:
            raise ValueError("RequestDict must include a valid username.")
        if not (self.model_name and self.model_provider) and not self.project_id:
            raise ValueError("RequestDict must include either model_name and model_provider, or project_id.")

    @classmethod
    def from_dict(cls, data: dict) -> 'RequestDict':
        return cls(**data)

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        return str(asdict(self))


def copy_request_dict(original: RequestDict) -> RequestDict:
    """Create a copy of the given RequestDict instance with a new run_id."""
    new_dict = asdict(original)
    new_dict['run_id'] = str(uuid.uuid4())
    return RequestDict.from_dict(new_dict)
