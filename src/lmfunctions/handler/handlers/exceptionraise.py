from typing import Literal

from lmfunctions.handler import Handler


class ExceptionRaiseHandler(Handler):
    """Event handler that raises an exception extracted from the event variables."""

    name: Literal["exceptionraise"] = "exceptionraise"
    varname: str = "exception"

    def __call__(self, **kwargs):
        raise kwargs[self.varname]
