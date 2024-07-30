from typing import Callable, Dict, List

from pydantic import Field
from typing_extensions import Annotated

from lmfunctions.base import Base
from lmfunctions.handler import (
    ExceptionRaiseHandler,
    LoggingHandler,
    PanelPrintHandler,
    PrintHandler,
)

StandardHandler = Annotated[
    LoggingHandler | PanelPrintHandler | PrintHandler | ExceptionRaiseHandler,
    Field(discriminator="name"),
]

Handler = StandardHandler | Callable


class EventManager(Base):
    """
    An EventManager defines a map between event names and a list of handler functions.
    Each handler is a callable that is executed when the corresponding event occurs.

    There are two types of handlers: standard handlers and custom handlers. Standard
    handlers are predefined functions which can be serialized, whereas custom
    handlers are arbitrary callables.
    """

    handlers: Dict[str, List[Handler]] = dict(
        exception=[ExceptionRaiseHandler(varname="exception")],
    )

    def __call__(self, event_name, **kwargs):
        [handler(**kwargs) for handler in self.handlers.get(event_name, [])]
        return None

    def __add__(self, other):
        if isinstance(other, EventManager):
            handlers = {}
            for d in (self.handlers, other.handlers):
                for key, value in d.items():
                    handlers.setdefault(key, []).extend(value)
            return EventManager(handlers=handlers)
        return NotImplemented
