"""Tencent Hunyuan TokenHub through its OpenAI-compatible Chat Completions API."""
import os

from models.base import BaseModel, ModelResult


class HunyuanOpenAIModel(BaseModel):
    def __init__(self, model_id, *, model="hy3", max_output_tokens=None, timeout=None):
        super().__init__(model_id, model=model, max_output_tokens=max_output_tokens, timeout=timeout)
        key = os.environ.get("HUNYUAN_API_KEY", "").strip()
        if not key:
            raise ValueError("HUNYUAN_API_KEY is missing. Use the local hidden-input runner.")
        from openai import OpenAI
        # This endpoint matches keys issued by the mainland China TokenHub.
        base_url = "https://tokenhub.tencentmaas.com/v1"
        self.model = os.environ.get("HUNYUAN_MODEL", model).strip()
        self.max_output_tokens = max_output_tokens
        self.client = OpenAI(api_key=key, base_url=base_url, timeout=timeout, max_retries=0)

    def generate(self, prompt, **kwargs):
        try:
            request = {
                "model": self.model,
                "messages": kwargs.get("messages") or [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
            }
            if self.max_output_tokens is not None:
                request["max_tokens"] = self.max_output_tokens
            response = self.client.chat.completions.create(**request)
            choice = response.choices[0]
            answer = choice.message.content or ""
            usage = response.usage
            metadata = {
                "provider": "hunyuan",
                "model": getattr(response, "model", self.model),
                "provider_response_id": getattr(response, "id", None),
                "provider_finish_reason": choice.finish_reason,
                "usage": {
                    "input_tokens": getattr(usage, "prompt_tokens", 0),
                    "output_tokens": getattr(usage, "completion_tokens", 0),
                    "total_tokens": getattr(usage, "total_tokens", 0),
                } if usage else {},
            }
            if choice.finish_reason not in ("stop", None):
                return ModelResult(text=answer, status="error",
                                   error="Hunyuan response did not complete normally.", metadata=metadata)
            if not answer:
                return ModelResult(text="", status="error", error="Hunyuan returned no answer text.", metadata=metadata)
            return ModelResult(text=answer, metadata=metadata)
        except Exception as exc:
            code = getattr(exc, "status_code", None)
            suffix = f" (HTTP {code})" if isinstance(code, int) else ""
            body = getattr(exc, "body", None)
            provider_error = body.get("error", body) if isinstance(body, dict) else {}
            if not isinstance(provider_error, dict):
                provider_error = {}
            return ModelResult(
                text="", status="error",
                error="Hunyuan request failed: " + type(exc).__name__ + suffix,
                metadata={"provider": "hunyuan", "model": self.model,
                          "provider_error_code": provider_error.get("code"),
                          "provider_error_message": None,  # Avoid persisting provider text that may echo credentials.
                          "stop_run": code in (400, 401, 402, 403, 429)},
            )
