from lmfunctions.eventmanager import EventManager
from lmfunctions.handler import PrintHandler


def tokenStream() -> EventManager:
    """
    EventManager preset that prints individual tokens or characters to the console as
    they are processed.
    """
    return EventManager(
        handlers={
            "token_or_char": [
                PrintHandler(varnames=["token_or_char"], end="", flush=True)
            ]
        }
    )
