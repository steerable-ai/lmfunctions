from lmfunctions.eventmanager import EventManager
from lmfunctions.lmbackend import LMBackend, LlamaCppBackend
from lmfunctions.retrypolicy import RetryPolicy


class default:
    backend: LMBackend = LlamaCppBackend()
    backend.generation.stop = [
        "<|end_header_id|>",
        "<|eot_id|>",
        "<|reserved_special_token",
    ]
    event_manager = EventManager()
    retry_policy = RetryPolicy()
