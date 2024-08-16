from typing import Any, Dict, List, Literal, Optional

from lmfunctions.base import Base
from lmfunctions.message import Message, is_message_list
from lmfunctions.utils import lazy_import, model_from_schema


class LiteLLMBackend(Base):

    name: Literal["litellm"] = "litellm"
    model: str = "gpt-4o-mini"
    timeout: float | None = None
    temperature: float | None = None
    top_p: float | None = None
    n: int | None = None
    stream: bool | None = True
    stream_options: Dict | None = None
    stop: str | None = None
    max_tokens: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    logit_bias: Dict | None = None
    user: str | None = None
    response_format: Dict | None = None
    seed: int | None = None
    tools: list[Any] | None = None
    tool_choice: str | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    extra_headers: Dict | None = None
    functions: List | None = None
    function_call: str | None = None
    base_url: str | None = None
    api_version: str | None = None
    api_key: str | None = None
    max_retries: int = 2
    mock_response: str | None = None
    drop_params: bool = True

    def __call__(
        self, input: Any = "", schema: Optional[Dict] = None, **kwargs
    ) -> Message:
        lazy_import("litellm")
        import litellm

        messages = (
            input if is_message_list(input) else [Message(role="user", content=input)]
        )
        params = (
            self.model_dump(exclude={"name"})
            | dict(response_format=model_from_schema(schema) if schema else None)
            | kwargs
        )
        response = litellm.completion(
            messages=[message.dump() for message in messages], **params
        )
        return Message.from_openai_v1(response)
