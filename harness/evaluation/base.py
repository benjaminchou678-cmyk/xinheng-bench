from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class EvaluationResult:
    status: str = "not_implemented"
    score: Optional[float] = None
    reason: str = "Phase 0: scoring criteria are not defined."
    details: Dict[str, Any] = field(default_factory=dict)


class BaseEvaluator(ABC):
    name = "base"

    @abstractmethod
    def evaluate(self, case: dict, response: dict, *, context: dict) -> EvaluationResult:
        """context contains all cases/responses for comparisons within this run."""
        raise NotImplementedError
