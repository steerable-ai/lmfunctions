import logging
from importlib.util import find_spec

from lmfunctions.backends import (
    LiteLLMBackend,
    LlamaCppBackend,
    LMBackend,
    TransformersBackend,
)
from lmfunctions.base import Base
from lmfunctions.eventmanager import EventManager
from lmfunctions.managers import (
    consoleRich,
    fileLog,
    panelPrint,
    timeEvents,
    tokenStream,
)
from lmfunctions.retrypolicy import RetryPolicy
from lmfunctions.utils import cuda_check

from .chat import chat
from .default import default
from .lmfunc import LMFunc, lmdef
from .message import Message

logger = logging.getLogger(__name__)

# Shortcuts
from_string = LMFunc.from_string
from_store = LMFunc.from_store
from_file = LMFunc.from_file


def complete(*args, **kwargs):
    return default.backend(*args, *kwargs)


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

if cuda_check()["cuda_available"]:
    logger.info("✔ CUDA available")
if find_spec("llama_cpp"):
    logger.info("✔ Llama-CPP backend available")
if find_spec("transformers"):
    logger.info("✔ Transformers backend available")
if find_spec("litellm"):
    logger.info("✔ Lite-LLM backend available")


class EventManagerSetter:
    @staticmethod
    def panelprint():
        default.event_manager = panelPrint

    @staticmethod
    def consolerich():
        default.event_manager = consoleRich

    @staticmethod
    def filelogger(*args, **kwargs):
        default.event_manager = fileLog(*args, **kwargs)

    @staticmethod
    def tokenstream():
        default.event_manager = tokenStream

    @staticmethod
    def timeevents():
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
    "Message",
    "EventManager",
    "LMBackend",
    "Base",
    "RetryPolicy",
    "set_backend",
    "complete",
    "default",
]
