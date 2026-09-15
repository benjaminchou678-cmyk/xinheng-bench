"""Gemini through Google's OpenAI-compatible Chat Completions endpoint."""
import os

from models.base import BaseModel, ModelResult


class GeminiOpenAIModel(BaseModel):
    def __init__(self, model_id, *, model, max_output_tokens=1200, timeout=60):
        super().__init__(model_id, model=model, max_output_tokens=max_output_tokens, timeout=timeout)
        key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not key:
            raise ValueError("GEMINI_API_KEY is missing. Use the local hidden-input runner.")
        if not isinstance(max_output_tokens, int) or not 1 <= max_output_tokens <= 4096:
            raise ValueError("Pilot max_output_tokens must be between 1 and 4096.")

        from openai import OpenAI

        # Keep the credential destination fixed so ambient base-URL variables cannot redirect it.
        self.client = OpenAI(
            api_key=key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            timeout=timeout,
            max_retries=0,
        )
        self.model = model
        self.max_output_tokens = max_output_tokens

    def generate(self, prompt, **kwargs):
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_output_tokens,
                temperature=0,
            )
            choice = completion.choices[0]
            text = choice.message.content or ""
            usage = completion.usage
            metadata = {
                "provider": "google-gemini",
                "model": completion.model,
                "provider_response_id": completion.id,
                "provider_finish_reason": choice.finish_reason,
                "usage": {
                    "input_tokens": usage.prompt_tokens,
                    "output_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                } if usage else {},
            }
            if choice.finish_reason not in ("stop", None):
                return ModelResult(
                    text=text,
                    status="error",
                    error="Gemini response did not complete normally; excluded from scoring.",
                    metadata=metadata,
                )
            if not text:
                return ModelResult(text="", status="error", error="Gemini returned no text.", metadata=metadata)
            return ModelResult(text=text, metadata=metadata)
        except Exception as exc:
            # Avoid persisting provider response bodies, headers, or exception messages.
            code = getattr(exc, "status_code", None)
            suffix = f" (HTTP {code})" if isinstance(code, int) else ""
            return ModelResult(
                text="",
                status="error",
                error="Gemini request failed: " + type(exc).__name__ + suffix,
                metadata={
                    "provider": "google-gemini",
                    "model": self.model,
                    "stop_run": code in (400, 401, 403, 429),
                },
            )
