from lmfunctions.backends import LiteLLMBackend, LlamaCppBackend, TransformersBackend
from lmfunctions.base import Base
from lmfunctions.eventmanager import EventManager
from lmfunctions.retrypolicy import RetryPolicy

from .chat import chat
from .default import default
from .lmfunc import LMFunc, lmdef
from .lmresponse import LMResponse

# Shortcuts
from_string = LMFunc.from_string
from_store = LMFunc.from_store
from_file = LMFunc.from_file


def complete(*args, **kwargs):
    return default.backend.complete(*args, *kwargs)


class BackendSetter:
    @staticmethod
    def llamacpp(*args, **kwargs):
        default.backend = LlamaCppBackend(*args, **kwargs)

    @staticmethod
    def transformers(*args, **kwargs):
        default.backend = TransformersBackend(*args, **kwargs)

    @staticmethod
    def litellm(*args, **kwargs):
        default.backend = LiteLLMBackend(*args, **kwargs)


set_backend = BackendSetter()


__all__ = [
    "lmdef",
    "LMFunc",
    "chat",
    "from_string",
    "from_store",
    "from_file",
    "LMResponse",
    "EventManager",
    "Base",
    "RetryPolicy",
    "set_backend",
    "complete",
    "default",
]
