from models.base import BaseModel, ModelResult


class PlaceholderModel(BaseModel):
    def generate(self, prompt: str, **kwargs) -> ModelResult:
        return ModelResult(
            text="[PLACEHOLDER RESPONSE: no model API was called]",
            status="placeholder",
            metadata={"is_placeholder": True},
        )
