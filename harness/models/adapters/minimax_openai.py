"""MiniMax through its official OpenAI-compatible Chat Completions API."""
import os
import re

from models.base import BaseModel, ModelResult


class MiniMaxOpenAIModel(BaseModel):
    def __init__(self, model_id, *, model="MiniMax-M2.7-highspeed", max_output_tokens=700, timeout=60):
        super().__init__(model_id, model=model, max_output_tokens=max_output_tokens, timeout=timeout)
        key = os.environ.get("MINIMAX_API_KEY", "").strip()
        if not key:
            raise ValueError("MINIMAX_API_KEY is missing. Use the local hidden-input runner.")
        from openai import OpenAI
        self.client = OpenAI(api_key=key, base_url="https://api.minimaxi.com/v1",
                             timeout=timeout, max_retries=0)
        self.model, self.max_output_tokens = model, max_output_tokens

    def generate(self, prompt, **kwargs):
        try:
            request = {"model": self.model, "messages": kwargs.get("messages") or [{"role": "user", "content": prompt}],
                       "temperature": 0.1, "stream": False,
                       "extra_body": {"reasoning_split": True}}
            if self.max_output_tokens is not None:
                request["max_tokens"] = self.max_output_tokens
            completion = self.client.chat.completions.create(**request)
            choice = completion.choices[0]
            text = re.sub(r"^\s*<think>.*?</think>\s*", "", choice.message.content or "", flags=re.S)
            usage = completion.usage
            metadata = {"provider": "minimax", "model": completion.model,
                        "provider_response_id": completion.id, "provider_finish_reason": choice.finish_reason,
                        "usage": {"input_tokens": usage.prompt_tokens, "output_tokens": usage.completion_tokens,
                                  "total_tokens": usage.total_tokens} if usage else {}}
            if choice.finish_reason not in ("stop", None):
                return ModelResult(text=text, status="error", error="MiniMax response did not complete normally; excluded from scoring.", metadata=metadata)
            if not text:
                return ModelResult(text="", status="error", error="MiniMax returned no answer text.", metadata=metadata)
            return ModelResult(text=text, metadata=metadata)
        except Exception as exc:
            code = getattr(exc, "status_code", None)
            suffix = f" (HTTP {code})" if isinstance(code, int) else ""
            return ModelResult(text="", status="error", error="MiniMax request failed: " + type(exc).__name__ + suffix,
                               metadata={"provider": "minimax", "model": self.model,
                                         "stop_run": code in (400, 401, 402, 403, 429)})
