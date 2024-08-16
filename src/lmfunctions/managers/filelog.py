import logging

from lmfunctions.eventmanager import EventManager
from lmfunctions.handlers import LoggingHandler


def fileLog(log_file="lmdef.log") -> EventManager:
    """EventManager factory that logs selected internal variables to a file."""

    return EventManager(
        handlers={
            "call_start": [
                LoggingHandler(
                    log_file=log_file,
                    message="Call started with vars:",
                    log_keys=["self", "args", "kwargs", "examples"],
                )
            ],
            "input_render": [
                LoggingHandler(
                    log_file=log_file,
                    message="Prompt rendered with vars:",
                    log_keys=["backend_input"],
                )
            ],
            "token_or_char": [
                LoggingHandler(
                    log_file=log_file,
                    message="New token/char processed:",
                    log_keys=["token_or_char"],
                )
            ],
            "retry": [
                LoggingHandler(
                    message="Retrying with vars:", log_keys=["retry_call_state"]
                )
            ],
            "exception": [
                LoggingHandler(
                    log_file=log_file,
                    log_level=logging.ERROR,
                    message="Exception occurred:",
                    log_keys=["exception"],
                )
            ],
            "success": [
                LoggingHandler(
                    log_file=log_file,
                    message="Process succeeded with vars:",
                    log_keys=["input", "completion", "output"],
                )
            ],
        }
    )
