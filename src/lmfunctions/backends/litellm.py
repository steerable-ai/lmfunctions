from typing import Any, Dict, Iterator, List, Literal, Optional

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
    stream: bool | None = False
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
    chat: bool = True

    def __call__(
        self,
        input: str | List[str] | List[Message] | List[List[Message]] = "",
        schema: Optional[Dict] = None,
        **kwargs
    ) -> Message | List[Message]:
        lazy_import("litellm")
        import litellm

        params = (
            self.model_dump(exclude={"name", "chat"})
            | dict(response_format=model_from_schema(schema) if schema else None)
            | kwargs
        )
        if self.chat:
            # Chat mode
            if is_message_list(input):
                output = [
                    litellm.completion(
                        messages=[message.dump() for message in input], **params
                    )
                ]
            elif isinstance(input, list):
                output = litellm.batch_completion(
                    messages=[[dict(role="user", content=_in)] for _in in input],
                    **params
                )
            else:
                output = [
                    litellm.completion(
                        messages=[dict(role="user", content=input)], **params
                    )
                ]
            output = [Message.from_openai_v1(_out) for _out in output]
        else:
            # Text generation mode
            if isinstance(input, str):
                response = litellm.text_completion(input, **params)
                if isinstance(response, Iterator):
                    output = Message((c["choices"][0]["text"] or "" for c in response))
                else:
                    content = response["choices"][0]["text"]
                    output = Message(content if isinstance(content, str) else "")
            else:
                raise ValueError("The input must be a string")
        if isinstance(output, list) and len(output) == 1:
            return output[0]
        return output
