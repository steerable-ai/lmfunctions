from abc import abstractmethod
from typing import Dict, List, Optional

from lmfunctions.base import Base
from lmfunctions.lmresponse import LMResponse


class LMBackend(Base):
    """
    A class providing methods for interacting with a language model backend,
    such as completing prompts and replying to chat messages.
    """

    @abstractmethod
    def complete(
        self, prompt: str, schema: Optional[Dict] = None, *args, **kwargs
    ) -> LMResponse:
        pass  # pragma: no cover

    @abstractmethod
    def chat_complete(self, messages: List, **kwargs) -> LMResponse:
        pass  # pragma: no cover
