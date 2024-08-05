from lmfunctions.backends import LlamaCppBackend, LMBackend
from lmfunctions.eventmanager import EventManager
from lmfunctions.retrypolicy import RetryPolicy


class default:
    backend: LMBackend = LlamaCppBackend(
        generation=dict(
            stop=["<|end_header_id|>", "<|eot_id|>", "<|reserved_special_token"]
        )
    )
    event_manager = EventManager()
    retry_policy = RetryPolicy()
