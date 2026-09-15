"""Doubao through Volcengine Ark's OpenAI-compatible Responses API."""
import os
import time

from models.base import BaseModel, ModelResult


class DoubaoResponsesModel(BaseModel):
    def __init__(self, model_id, *, model, max_output_tokens=None, timeout=60,
                 rate_limit_retries=3, retry_wait_seconds=20):
        super().__init__(model_id, model=model, max_output_tokens=max_output_tokens, timeout=timeout)
        key = os.environ.get("ARK_API_KEY", "").strip()
        if not key:
            raise ValueError("ARK_API_KEY is missing. Use the local hidden-input runner.")
        if max_output_tokens is not None and (not isinstance(max_output_tokens, int) or max_output_tokens < 1):
            raise ValueError("max_output_tokens must be a positive integer or null.")
        from openai import OpenAI
        self.client = OpenAI(api_key=key, base_url="https://ark.cn-beijing.volces.com/api/v3",
                             timeout=timeout, max_retries=0)
        self.model = os.environ.get("ARK_MODEL_ID", "").strip() or model
        self.max_output_tokens = max_output_tokens
        self.rate_limit_retries = rate_limit_retries
        self.retry_wait_seconds = retry_wait_seconds

    def generate(self, prompt, **kwargs):
        request = {"model": self.model, "input": kwargs.get("messages") or prompt,
                   "extra_body": {"thinking": {"type": "disabled"}}}
        if self.max_output_tokens is not None:
            request["max_output_tokens"] = self.max_output_tokens
        retries = 0
        while True:
            try:
                response = self.client.responses.create(**request)
                break
            except Exception as exc:
                code = getattr(exc, "status_code", None)
                if code == 429 and retries < self.rate_limit_retries:
                    retries += 1
                    time.sleep(self.retry_wait_seconds * retries)
                    continue
                suffix = f" (HTTP {code})" if isinstance(code, int) else ""
                return ModelResult(text="", status="error",
                                   error="Doubao request failed: " + type(exc).__name__ + suffix,
                                   metadata={"provider": "volcengine-ark", "model": self.model,
                                             "retry_count": retries,
                                             "stop_run": code in (400, 401, 402, 403, 404, 429)})
        try:
            usage = getattr(response, "usage", None)
            metadata = {
                "provider": "volcengine-ark",
                "model": getattr(response, "model", self.model),
                "provider_response_id": getattr(response, "id", None),
                "provider_status": getattr(response, "status", None),
                "retry_count": retries,
                "usage": {
                    "input_tokens": getattr(usage, "input_tokens", 0),
                    "output_tokens": getattr(usage, "output_tokens", 0),
                    "total_tokens": getattr(usage, "total_tokens", 0),
                } if usage else {},
            }
            text = getattr(response, "output_text", "") or ""
            if getattr(response, "status", None) not in ("completed", None):
                return ModelResult(text=text, status="error",
                                   error="Doubao response did not complete normally; excluded from scoring.",
                                   metadata=metadata)
            if not text:
                return ModelResult(text="", status="error", error="Doubao returned no text.", metadata=metadata)
            return ModelResult(text=text, metadata=metadata)
        except Exception as exc:
            code = getattr(exc, "status_code", None)
            suffix = f" (HTTP {code})" if isinstance(code, int) else ""
            return ModelResult(text="", status="error",
                               error="Doubao request failed: " + type(exc).__name__ + suffix,
                               metadata={"provider": "volcengine-ark", "model": self.model, "retry_count": retries,
                                         "stop_run": code in (400, 401, 402, 403, 404, 429)})
