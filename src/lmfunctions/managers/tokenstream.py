from lmfunctions.eventmanager import EventManager
from lmfunctions.handlers import PrintHandler

# EventManager preset that prints individual tokens or characters to the console as
# they are processed.

tokenStream = EventManager(
    handlers={
        "token_or_char": [PrintHandler(varnames=["token_or_char"], end="", flush=True)]
    }
)
