from models.adapters.placeholder import PlaceholderModel

from models.adapters.openai_responses import OpenAIResponsesModel
from models.adapters.gemini_openai import GeminiOpenAIModel
from models.adapters.deepseek_responses import DeepSeekResponsesModel
from models.adapters.baichuan_openai import BaichuanOpenAIModel
from models.adapters.doubao_responses import DoubaoResponsesModel
from models.adapters.minimax_openai import MiniMaxOpenAIModel
from models.adapters.hunyuan_openai import HunyuanOpenAIModel
from models.adapters.kimi_openai import KimiOpenAIModel

ADAPTERS = {
    "placeholder": PlaceholderModel,
    "openai_responses": OpenAIResponsesModel,
    "gemini_openai": GeminiOpenAIModel,
    "deepseek_responses": DeepSeekResponsesModel,
    "baichuan_openai": BaichuanOpenAIModel,
    "doubao_responses": DoubaoResponsesModel,
    "minimax_openai": MiniMaxOpenAIModel,
    "hunyuan_openai": HunyuanOpenAIModel,
    "kimi_openai": KimiOpenAIModel,
}


def register_adapter(name, adapter_class):
    if name in ADAPTERS:
        raise ValueError("Adapter already registered: " + name)
    ADAPTERS[name] = adapter_class


def create_model(spec):
    name = spec["adapter"]
    if name not in ADAPTERS:
        raise ValueError("Unknown adapter: " + name)
    return ADAPTERS[name](model_id=spec["id"], **spec.get("parameters", {}))
