import json
from typing import Any, Callable, Dict, Iterator, List, Optional, TypeGuard, Union

from lmfunctions.base import Base
from lmfunctions.handlers import PrintHandler


class Message(Base):
    """
    A class representing a message to/from a language model.
    """

    role: str = "assistant"
    content: str = ""

    _unprocessed: Union[str, Iterator[str]] = ""

    def __init__(
        self, unprocessed: Optional[Union[str, Iterator[str]]] = None, **kwargs
    ):
        """
        The message can be initialized with an unprocessed string or an iterator.
        Executing __call__ on the object will iterate over tokens/characters
        in the message, optionally invoke a callback function for each token/character
        processed, and parse the response according to the given schema.
        """
        super().__init__(**kwargs)
        if unprocessed is not None:
            self._unprocessed = unprocessed

    def __repr__(self):
        self.process()
        print()
        return super().__repr__()

    def process(
        self,
        schema: Optional[Dict] = None,
        handle_token_or_char: Optional[Callable] = PrintHandler(
            varnames=["token_or_char"], end="", flush=True
        ),
        **kwargs
    ) -> Any:
        """
        Iterates over tokens/characters in the message, optionally parsing an output according to the given schema.
        For each token or character processed, calls the `handle_token_or_chat` function if provided.

        Args:
            schema (Optional[Dict]): A JSON schema to parse the response.
            handle_token_or_chat (Optional[Callable]): A callback function to be called when a new token/character is processed. Defaults to PrintHandler.

        Returns:
            Any: The (optionally parsed) response content.

        Raises:
            JSONDecodeError: If parsing is required but it fails.
        """
        if not self.content:
            # Check if the schema defines a JSON object
            json_object = (schema is not None) and (
                schema.get("type", None) == "object"
            )
            # The response is unprocessed
            depth, in_json = 0, False
            for token_or_char in self._unprocessed:
                open_braces, closed_braces = token_or_char.count(
                    "{"
                ), token_or_char.count("}")
                if not (json_object) or in_json or open_braces:
                    self.content += token_or_char
                    # New token or character callback
                    if handle_token_or_char:
                        handle_token_or_char(
                            schema=schema,
                            json_object=json_object,
                            token_or_char=token_or_char,
                            depth=depth,
                            in_json=in_json,
                            open_braces=open_braces,
                            closed_braces=closed_braces,
                            **kwargs
                        )
                if json_object:
                    if in_json:
                        depth += open_braces - closed_braces
                        if depth == 0:
                            try:
                                return json.loads(self.content)
                            except json.JSONDecodeError:
                                self.content, in_json = "", False
                    elif open_braces:
                        depth, in_json = 1, True
        return (
            json.loads(self.content)
            if (schema and schema.get("type", None) != "string")
            else self.content
        )

    @classmethod
    def from_openai_v1(cls, response: Union[Any, Iterator[Any]]):
        """
        Creates a Message object from a response in OpenAI-v1 format.
        """
        if isinstance(response, Iterator):
            response_generator = (c.choices[0].delta.content or "" for c in response)
            return cls(response_generator)
        else:
            message = response.choices[0].message
            role = message.role if hasattr(message, "role") else ""
            content = message.content if isinstance(message.content, str) else ""
            return cls(unprocessed=content, role=role)


def is_message_list(obj: Any) -> TypeGuard[List[Message]]:
    """
    Checks if the input is a list of Message objects.
    """
    return isinstance(obj, list) and all(isinstance(m, Message) for m in obj)
