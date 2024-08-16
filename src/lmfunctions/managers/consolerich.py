import logging

from lmfunctions.eventmanager import EventManager
from lmfunctions.handlers import LoggingHandler

# EventManager preset that logs selected internal variables to the console
# using the rich library.
consoleRich = EventManager(
    handlers={
        "call_start": [
            LoggingHandler(
                message="Call started with vars:",
                log_keys=["self", "args", "kwargs", "examples"],
            )
        ],
        "input_render": [
            LoggingHandler(
                message="Backend input ready",
                log_keys=["backend_input"],
            )
        ],
        "token_or_char": [
            LoggingHandler(
                message="New token/char processed:",
                log_keys=["token_or_char"],
            )
        ],
        "retry": [
            LoggingHandler(message="Retrying with vars:", log_keys=["retry_call_state"])
        ],
        "exception": [
            LoggingHandler(
                log_level=logging.ERROR,
                message="Exception occurred:",
                log_keys=["exception"],
            )
        ],
        "success": [
            LoggingHandler(
                message="Process succeeded with vars:",
                log_keys=["input", "completion", "output"],
            )
        ],
    }
)
