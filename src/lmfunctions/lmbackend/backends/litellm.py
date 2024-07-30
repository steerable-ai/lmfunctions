from typing import Any, Dict, Iterable, List, Literal, Optional, Union

from lmfunctions.base import Base
from lmfunctions.lmbackend.lmbackend import LMBackend
from lmfunctions.lmresponse import LMResponse
from lmfunctions.utils import lazy_import


class LiteLLMBackend(LMBackend, Base):

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

    def complete(
        self, prompt: str = "", schema: Optional[Dict] = None, **kwargs
    ) -> LMResponse:
        messages = [{"role": "user", "content": prompt}]
        response_format = dict(type="json_object") if schema else None
        return self.chat_complete(messages, response_format=response_format, **kwargs)

    def chat_complete(self, messages: List, **kwargs) -> LMResponse:
        response = self.chat_complete_openai_v1(messages, **kwargs)
        return LMResponse.from_openai_v1(response)

    def chat_complete_openai_v1(
        self, messages: List, **kwargs
    ) -> Union[Any, Iterable[Any]]:
        params = self.model_dump(exclude={"name"}) | kwargs
        lazy_import("litellm")
        import litellm

        litellm.drop_params = True
        return litellm.completion(messages=messages, **params)
