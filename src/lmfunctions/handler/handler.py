from abc import abstractmethod
from typing import Any

from lmfunctions.base import Base


class Handler(Base):

    @abstractmethod
    def __call__(self, **kwargs) -> Any:
        pass  # pragma: no cover
