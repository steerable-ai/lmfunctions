from .litellm import LiteLLMBackend
from .llamacpp import LlamaCppBackend
from .transformers import TransformersBackend

LMBackend = LiteLLMBackend | LlamaCppBackend | TransformersBackend

__all__ = [
    "LiteLLMBackend",
    "LlamaCppBackend",
    "TransformersBackend",
    "LMBackend",
]
