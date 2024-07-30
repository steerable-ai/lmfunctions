import logging

from lmfunctions.eventmanager import EventManager
from lmfunctions.handler import LoggingHandler


def consoleRich() -> EventManager:
    """EventManager preset that logs selected internal variables to the console
    using the rich library."""

    return EventManager(
        handlers={
            "call_start": [
                LoggingHandler(
                    message="Call started with vars:",
                    log_keys=["self", "input", "examples", "runtime", "kwargs"],
                )
            ],
            "prompt_render": [
                LoggingHandler(
                    message="Prompt rendered with vars:",
                    log_keys=["input_string", "examples_string", "prompt"],
                )
            ],
            "token_or_char": [
                LoggingHandler(
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
                    log_level=logging.ERROR,
                    message="Exception occurred:",
                    log_keys=["exception"],
                )
            ],
            "success": [
                LoggingHandler(
                    message="Process succeeded with vars:",
                    log_keys=["input", "completion", "parsed_completion", "output"],
                )
            ],
        }
    )
