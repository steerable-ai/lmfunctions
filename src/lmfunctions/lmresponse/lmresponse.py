import json
from typing import Any, Callable, Dict, Iterator, Optional, Union

from lmfunctions.base import Base
from lmfunctions.handler import PrintHandler


class LMResponse(Base):
    """
    A class representing a response from a language model.

    The response can be initialized with a string or an iterator of strings.

    Executing __call__ on the object will iterate over tokens/characters
    in the response, optionally call a callback function for each token/character
    processed, and parse the response according to a given schema.
    """

    text: str = ""

    _unprocessed: Union[str, Iterator[str]]

    def __init__(self, unprocessed: Union[str, Iterator[str]], **kwargs):
        super().__init__(**kwargs)
        self._unprocessed = unprocessed

    def __repr__(self):
        self()
        print()
        return super().__repr__()

    def __call__(
        self,
        schema: Optional[Dict] = None,
        new_token_or_char_callback: Optional[Callable] = PrintHandler(
            varnames=["token_or_char"], end="", flush=True
        ),
        **kwargs
    ) -> Any:
        """
        Processes the response, optionally parsing it according to the given schema.
        For each token or character processed, calls the `new_token_or_char_callback` function
        if provided.

        Args:
            schema (Optional[Dict]): A JSON schema to parse the response.
            new_token_or_char_callback (Optional[Callable]): A callback function
            to be called when a new token/character is processed. Defaults to PrintHandler.

        Returns:
            Any: The processed response.

        Raises:
            JSONDecodeError: If JSON parsing is required but no valid JSON is found.
        """
        if not self.text:
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
                    self.text += token_or_char
                    # New token or character callback
                    if new_token_or_char_callback:
                        new_token_or_char_callback(
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
                                return json.loads(self.text)
                            except json.JSONDecodeError:
                                self.text, in_json = "", False
                    elif open_braces:
                        depth, in_json = 1, True
        return (
            json.loads(self.text)
            if (schema and schema.get("type", None) != "string")
            else self.text
        )

    @classmethod
    def from_openai_v1(cls, response: Union[Any, Iterator[Any]]):
        """
        Creates an LMResponse object from a response in OpenAI-v1 format.
        """
        if isinstance(response, Iterator):
            response_generator = (c.choices[0].delta.content or "" for c in response)
            return cls(response_generator)
        else:
            content = response.choices[0].message.content
            return cls(content if isinstance(content, str) else "")
