"""Official OpenAI Responses endpoint; credentials never enter configuration."""
import os
from models.base import BaseModel, ModelResult


class OpenAIResponsesModel(BaseModel):
    def __init__(self, model_id, *, model, max_output_tokens=1200, timeout=60):
        super().__init__(model_id, model=model, max_output_tokens=max_output_tokens, timeout=timeout)
        key = os.environ.get('OPENAI_API_KEY', '').strip()
        if not key:
            raise ValueError('OPENAI_API_KEY is missing. Use the local hidden-input runner.')
        if not isinstance(max_output_tokens, int) or not 1 <= max_output_tokens <= 4096:
            raise ValueError('Pilot max_output_tokens must be between 1 and 4096.')
        from openai import OpenAI
        # Fixed destination; environment base-url overrides cannot redirect the credential.
        self.client = OpenAI(api_key=key, base_url='https://api.openai.com/v1',
                             timeout=timeout, max_retries=0)
        self.model = model
        self.max_output_tokens = max_output_tokens

    def generate(self, prompt, **kwargs):
        try:
            response = self.client.responses.create(
                model=self.model, input=prompt,
                max_output_tokens=self.max_output_tokens, store=False,
            )
            usage = response.usage
            metadata = {'provider': 'openai', 'model': response.model,
                        'provider_response_id': response.id, 'provider_status': response.status,
                        'usage': {'input_tokens': usage.input_tokens, 'output_tokens': usage.output_tokens,
                                  'total_tokens': usage.total_tokens} if usage else {}}
            if response.status != 'completed':
                return ModelResult(text=response.output_text or '', status='error',
                                   error='OpenAI response did not complete; excluded from scoring.',
                                   metadata=metadata)
            if not response.output_text:
                return ModelResult(text='', status='error', error='OpenAI returned no text.', metadata=metadata)
            return ModelResult(text=response.output_text, metadata=metadata)
        except Exception as exc:
            # Do not persist exception text, request headers or response bodies: they may contain secrets.
            code = getattr(exc, 'status_code', None)
            suffix = f' (HTTP {code})' if isinstance(code, int) else ''
            fatal = code in (401, 403, 429)
            return ModelResult(text='', status='error',
                               error='OpenAI request failed: ' + type(exc).__name__ + suffix,
                               metadata={'provider': 'openai', 'model': self.model,
                                         'stop_run': fatal})
