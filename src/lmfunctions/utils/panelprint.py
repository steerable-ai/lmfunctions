from typing import Any, get_args

from rich import print
from rich.console import RenderableType
from rich.panel import Panel
from rich.pretty import Pretty


def panelprint(_object: Any, title: str = ""):

    print(
        "\n",
        Panel(
            (
                _object
                if isinstance(_object, get_args(RenderableType))
                else Pretty(_object)
            ),
            title=title,
        ),
    )
