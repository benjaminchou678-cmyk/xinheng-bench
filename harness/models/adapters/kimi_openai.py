"""Kimi through Moonshot's OpenAI-compatible Chat Completions API."""
import os

from models.base import BaseModel, ModelResult


class KimiOpenAIModel(BaseModel):
    def __init__(self, model_id, *, model="kimi-k2.6", max_output_tokens=None, timeout=None):
        super().__init__(model_id, model=model, max_output_tokens=max_output_tokens, timeout=timeout)
        key = os.environ.get("KIMI_API_KEY", "").strip()
        if not key:
            raise ValueError("KIMI_API_KEY is missing. Use the local hidden-input runner.")
        from openai import OpenAI
        base_url = os.environ.get("KIMI_BASE_URL", "https://api.moonshot.cn/v1").strip()
        self.model = os.environ.get("KIMI_MODEL", model).strip()
        self.max_output_tokens = max_output_tokens
        self.client = OpenAI(api_key=key, base_url=base_url, timeout=timeout, max_retries=0)

    def generate(self, prompt, **kwargs):
        try:
            request = {
                "model": self.model,
                "messages": kwargs.get("messages") or [{"role": "user", "content": prompt}],
                "stream": False,
            }
            if self.max_output_tokens is not None:
                request["max_tokens"] = self.max_output_tokens
            completion = self.client.chat.completions.create(**request)
            choice = completion.choices[0]
            answer = choice.message.content or ""
            usage = completion.usage
            metadata = {
                "provider": "kimi",
                "model": completion.model,
                "provider_response_id": completion.id,
                "provider_finish_reason": choice.finish_reason,
                "usage": {
                    "input_tokens": usage.prompt_tokens,
                    "output_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                } if usage else {},
            }
            if not answer:
                return ModelResult(text="", status="error", error="Kimi returned no answer text.", metadata=metadata)
            return ModelResult(text=answer, metadata=metadata)
        except Exception as exc:
            code = getattr(exc, "status_code", None)
            suffix = f" (HTTP {code})" if isinstance(code, int) else ""
            return ModelResult(
                text="", status="error",
                error="Kimi request failed: " + type(exc).__name__ + suffix,
                metadata={"provider": "kimi", "model": self.model,
                          "stop_run": code in (400, 401, 402, 403, 429)},
            )
