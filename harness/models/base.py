from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ModelResult:
    text: str
    status: str = "ok"
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseModel(ABC):
    def __init__(self, model_id: str, **parameters):
        self.model_id = model_id
        self.parameters = parameters

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> ModelResult:
        """Return output; never perform scoring here."""
        raise NotImplementedError
