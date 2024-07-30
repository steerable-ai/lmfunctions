from lmfunctions.base.base import Base
from lmfunctions.eventmanager import (
    EventManager,
    consoleRich,
    fileLog,
    panelPrint,
    tokenStream,
)
from lmfunctions.eventmanager.managers import timeEvents
from lmfunctions.lmbackend import (
    LiteLLMBackend,
    LlamaCppBackend,
    TransformersBackend,
)
from lmfunctions.retrypolicy import RetryPolicy

from .chat import chat
from .default import default
from .lmbackend import LMBackend
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


class EventManagerSetter:
    @staticmethod
    def panelprint():
        default.event_manager = panelPrint()

    @staticmethod
    def consolerich():
        default.event_manager = consoleRich()

    @staticmethod
    def filelogger(*args, **kwargs):
        default.event_manager = fileLog(*args, **kwargs)

    @staticmethod
    def tokenstream():
        default.event_manager = tokenStream()

    @staticmethod
    def time_events():
        default.event_manager = timeEvents()

    @staticmethod
    def default():
        default.event_manager = EventManager()


set_event_manager = EventManagerSetter()

__all__ = [
    "lmdef",
    "LMFunc",
    "chat",
    "from_string",
    "from_store",
    "from_file",
    "LMBackend",
    "LMResponse",
    "EventManager",
    "Base",
    "RetryPolicy",
    "set_backend",
    "set_event_manager",
    "complete",
    "default",
]
