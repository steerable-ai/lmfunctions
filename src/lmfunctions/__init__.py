import logging
from importlib.util import find_spec

from . import backends, base, eventmanager, managers, retrypolicy, utils
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
    return default.backend(*args, **kwargs)


class BackendSetter:
    @staticmethod
    def llamacpp(*args, **kwargs):
        default.backend = backends.LlamaCppBackend(*args, **kwargs)

    @staticmethod
    def transformers(*args, **kwargs):
        default.backend = backends.TransformersBackend(*args, **kwargs)

    @staticmethod
    def litellm(*args, **kwargs):
        default.backend = backends.LiteLLMBackend(*args, **kwargs)

    @staticmethod
    def vllm(*args, **kwargs):
        default.backend = backends.VLLMBackend(*args, **kwargs)


set_backend = BackendSetter()

if utils.cuda_check()["cuda_available"]:
    logger.info("✔ CUDA available")
if find_spec("llama_cpp"):
    logger.info("✔ Llama-CPP backend available")
if find_spec("transformers"):
    logger.info("✔ Transformers backend available")
if find_spec("litellm"):
    logger.info("✔ Lite-LLM backend available")
if find_spec("vllm"):
    logger.info("✔ VLLM backend available")


class EventManagerSetter:
    @staticmethod
    def panelprint():
        default.event_manager = managers.panelPrint

    @staticmethod
    def consolerich():
        default.event_manager = managers.consoleRich

    @staticmethod
    def filelogger(*args, **kwargs):
        default.event_manager = managers.fileLog(*args, **kwargs)

    @staticmethod
    def tokenstream():
        default.event_manager = managers.tokenStream

    @staticmethod
    def timeevents():
        default.event_manager = managers.timeEvents()

    @staticmethod
    def default():
        default.event_manager = eventmanager.EventManager()


set_event_manager = EventManagerSetter()

__all__ = [
    "lmdef",
    "LMFunc",
    "chat",
    "from_string",
    "from_store",
    "from_file",
    "Message",
    "eventmanager",
    "base",
    "retrypolicy",
    "set_backend",
    "complete",
    "default",
    "backends",
    "managers",
    "utils",
]
