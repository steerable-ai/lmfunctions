import gc
from importlib import import_module
from typing import Any, Dict, List, Literal, Optional

from lmformatenforcer import JsonSchemaParser
from pydantic import model_validator

from lmfunctions.base import Base
from lmfunctions.message import Message, is_message_list
from lmfunctions.utils import cuda_check, lazy_import, pip_install


class TransformersBackend(Base):
    name: Literal["transformers"] = "transformers"
    model: str = "Qwen/Qwen2-0.5B-Instruct"
    config: str | None = None
    feature_extractor: str | None = None
    image_processor: str | None = None
    framework: str | None = None
    revision: str | None = None
    use_fast: bool = True
    token: str | bool | None = None
    device: int | str | None = None
    device_map: str | None = None
    trust_remote_code: bool = False
    model_kwargs: Dict[str, Any] = {}
    pipeline_class: Any | None = None
    chat: bool = True
    generation: Dict[str, Any] = {}

    _pipeline: Any = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.device is None:
            self.device = "cpu"
            gpu_info = cuda_check()
            if gpu_info["cuda_available"]:
                self.device = "cuda"

    @property
    def pipeline(self):
        def import_error_callback(name, package):
            if pip_install(["transformers[torch]"]):
                return import_module(name, package=package)

        if self._pipeline is None:
            if lazy_import("transformers", import_error_callback=import_error_callback):
                import transformers

                tokenizer = transformers.AutoTokenizer.from_pretrained(self.model)

                self._pipeline = transformers.pipeline(
                    task="text-generation",
                    tokenizer=tokenizer,
                    **self.model_dump(
                        exclude={"name", "generation", "chat"},
                        exclude_none=True,
                    ),
                )
                self._pipeline.model.generation_config.pad_token_id = (
                    tokenizer.eos_token_id
                )

            else:
                raise ImportError(
                    "The package 'transformers' is required"
                )  # pragma: no cover
        return self._pipeline

    def _unload(self):
        self._pipeline = None
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover
            pass  # pragma: no cover
        return self

    @model_validator(mode="after")
    def unload(self):
        # Force the model to be reloaded when the parameters are changed
        if self._pipeline:
            self._unload()
        return self

    @property
    def prefix_fn(self):
        from lmformatenforcer.integrations.transformers import (
            build_transformers_prefix_allowed_tokens_fn,
        )

        return build_transformers_prefix_allowed_tokens_fn

    def __call__(
        self,
        input: str | List[str] | List[Message] | List[List[Message]] = "",
        schema: Optional[Dict] = None,
        **kwargs
    ) -> Message | List[Message]:

        generation_config = self.pipeline.model.generation_config
        if (
            generation_config.max_length == 20
            and generation_config.max_new_tokens is None
            and "max_new_tokens" not in self.generation
            and "max_length" not in self.generation
        ):
            self.generation["max_new_tokens"] = 4096

        if schema and self.pipeline.tokenizer:
            prefix_function = self.prefix_fn(
                self.pipeline.tokenizer, JsonSchemaParser(schema)
            )
        else:
            prefix_function = None

        params = (
            self.generation | dict(prefix_allowed_tokens_fn=prefix_function) | kwargs
        )

        if (
            self.chat
            and hasattr(self.pipeline.tokenizer, "chat_template")
            and self.pipeline.tokenizer.chat_template
        ):
            # Chat mode
            if is_message_list(input):
                # Message list
                output = self.pipeline([message.dump() for message in input], **params)
            elif isinstance(input, list):
                # List of strings
                output = [
                    self.pipeline(
                        [dict(role="user", content=_in)],
                        **params,
                    )[0]
                    for _in in input
                ]
            else:
                # Single string
                output = self.pipeline([dict(role="user", content=input)], **params)
            output = [response["generated_text"][-1] for response in output]
            output = [
                Message(unprocessed=response["content"], role=response["role"])
                for response in output
            ]
        else:
            # Text generation mode
            if isinstance(input, str) or isinstance(input, list):
                output = self.pipeline(input, **params | {"return_full_text": False})
                if len(output) == 1:
                    output = [output]
                output = [Message(_out[0]["generated_text"]) for _out in output]
            else:
                raise ValueError("The input must be a string or a list of strings.")

        if len(output) == 1:
            return output[0]
        return output
