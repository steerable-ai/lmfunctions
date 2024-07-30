import gc
from typing import Any, Dict, List, Literal, Optional

from lmformatenforcer import JsonSchemaParser
from pydantic import model_validator

from lmfunctions.lmbackend.lmbackend import LMBackend
from lmfunctions.lmresponse import LMResponse
from lmfunctions.utils import cuda_check, lazy_import


class TransformersBackend(LMBackend):
    name: Literal["transformers"] = "transformers"
    model: str = "Qwen/Qwen2-0.5B"
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
    generation: Dict[str, Any] = {}

    _pipeline: Any = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.device is None:
            gpu_info = cuda_check()
            if not gpu_info["cuda_available"]:
                self.device = "cpu"
            elif gpu_info["num_gpus"] > 1 and self.device_map is None:
                try:
                    __import__("accelerate")

                    self.device_map = "auto"
                except ImportError:
                    self.device = "cuda"
            else:
                self.device = "cuda"

    @property
    def pipeline(self):
        if self._pipeline is None:
            if lazy_import("transformers", install_packages=["transformers['torch']"]):
                import transformers

                tokenizer = transformers.AutoTokenizer.from_pretrained(self.model)
                self._pipeline = transformers.pipeline(
                    task="text-generation",
                    tokenizer=tokenizer,
                    **self.model_dump(exclude={"name", "generation"})
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
        if self._pipeline:
            self._unload()
        return self

    @property
    def prefix_fn(self):
        from lmformatenforcer.integrations.transformers import (
            build_transformers_prefix_allowed_tokens_fn,
        )

        return build_transformers_prefix_allowed_tokens_fn

    def complete(
        self, prompt: str = "", schema: Optional[Dict] = None, **kwargs
    ) -> LMResponse:
        messages = [{"role": "user", "content": prompt}]
        pipeline = self.pipeline
        if schema and pipeline.tokenizer:
            prefix_function = self.prefix_fn(
                pipeline.tokenizer, JsonSchemaParser(schema)
            )
            return self.chat_complete(
                messages, prefix_allowed_tokens_fn=prefix_function, **kwargs
            )
        else:
            return self.chat_complete(messages, **kwargs)

    def chat_complete(self, messages: List, **kwargs) -> LMResponse:
        generation_config = self.pipeline.model.generation_config
        if (
            generation_config.max_length == 20
            and generation_config.max_new_tokens is None
            and "max_new_tokens" not in self.generation
            and "max_length" not in self.generation
        ):
            self.generation["max_new_tokens"] = 4096

        params = self.generation | kwargs
        response = self.pipeline(messages, **params)
        return LMResponse(response[0]["generated_text"][-1].get("content", ""))
