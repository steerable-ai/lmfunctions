from typing import Dict, List, Literal

from lmfunctions.base import Base
from lmfunctions.utils import panelprint


class PanelPrintHandler(Base):
    """Callback handler that prints the specified variables to the console using titled panels."""

    name: Literal["panelprint"] = "panelprint"
    variables: List[Dict] = []

    def __call__(self, **kwargs):
        for variable in self.variables:
            name = variable.get("name", None)
            title = variable.get("title", name)
            panelprint(kwargs.get(name, None), title=title)
