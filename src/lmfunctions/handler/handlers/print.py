from typing import List, Literal

from lmfunctions.handler import Handler


class PrintHandler(Handler):
    """Callback handler that prints the specified variables to the console."""

    name: Literal["print"] = "print"
    sep: str | None = " "
    end: str | None = "\n"
    flush: bool = False
    varnames: List[str] = []

    def __call__(self, **kwargs):
        print(
            *(kwargs.get(variable, None) for variable in self.varnames),
            sep=self.sep,
            end=self.end,
            flush=self.flush,
        )
