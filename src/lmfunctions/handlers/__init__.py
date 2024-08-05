from typing import Callable

from pydantic import Field
from typing_extensions import Annotated

from .logging import LoggingHandler
from .otelevent import OtelEventHandler
from .panelprint import PanelPrintHandler
from .print import PrintHandler

StandardHandler = Annotated[
    LoggingHandler | PanelPrintHandler | PrintHandler | OtelEventHandler,
    Field(discriminator="name"),
]

Handler = StandardHandler | Callable

__all__ = [
    "LoggingHandler",
    "PanelPrintHandler",
    "PrintHandler",
    "OtelEventHandler",
    "Handler",
    "StandardHandler",
]
