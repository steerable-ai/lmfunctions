from abc import abstractmethod
from typing import Dict


class ObjectStore:
    """Generic interface for an object store."""

    @abstractmethod
    def push(self, obj, path, **kwargs):
        pass  # pragma: no cover

    @abstractmethod
    def pull(self, path, **kwargs) -> Dict:
        pass  # pragma: no cover
