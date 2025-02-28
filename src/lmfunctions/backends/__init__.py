from .litellm import LiteLLMBackend
from .llamacpp import LlamaCppBackend
from .transformers import TransformersBackend
from .vllm import VLLMBackend

LMBackend = LiteLLMBackend | LlamaCppBackend | TransformersBackend | VLLMBackend

__all__ = [
    "LiteLLMBackend",
    "LlamaCppBackend",
    "TransformersBackend",
    "LMBackend",
]
