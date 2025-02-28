import gc
from importlib import import_module
from typing import Any, Dict, List, Literal, Optional

from pydantic import model_validator

from lmfunctions.base import Base
from lmfunctions.message import Message, is_message_list
from lmfunctions.utils import lazy_import, pip_install


class VLLMBackend(Base):

    name: Literal["vllm"] = "vllm"
    model: str = "Qwen/Qwen2-0.5B-Instruct"
    tokenizer: Optional[str] = None
    tokenizer_mode: str = "auto"
    skip_tokenizer_init: bool = False
    trust_remote_code: bool = False
    allowed_local_media_path: str = ""
    tensor_parallel_size: int = 1
    dtype: str = "auto"
    quantization: Optional[str] = None
    revision: Optional[str] = None
    tokenizer_revision: Optional[str] = None
    seed: int = 0
    gpu_memory_utilization: float = 0.9
    swap_space: float = 4
    cpu_offload_gb: float = 0
    enforce_eager: Optional[bool] = None
    max_seq_len_to_capture: int = 8192
    disable_custom_all_reduce: bool = False
    disable_async_output_proc: bool = False
    mm_processor_kwargs: Optional[Dict[str, Any]] = None
    enable_prefix_caching: bool = True
    chat: bool = True
    sampling_params: Dict[str, Any] = {}

    _lm: Any = None

    @property
    def lm(self):
        def import_error_callback(name, package):
            if pip_install(["vllm"]):
                return import_module(name, package=package)

        if self._lm is None:
            if lazy_import("vllm", import_error_callback=import_error_callback):
                from vllm import LLM

                params = self.model_dump(
                    exclude={"name", "sampling_params", "chat"},
                    exclude_none=True,
                )
                self._lm = LLM(**params)
            else:
                raise ImportError("The package 'vllm' is required")  # pragma: no cover
        return self._lm

    def _unload(self):
        self._lm = None
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
        if self._lm:
            self._unload()
        return self

    def __call__(
        self,
        input: str | List[str] | List[Message] | List[List[Message]] = "",
        schema: Optional[Dict] = None,
        **kwargs
    ) -> Message | List[Message]:

        lazy_import("vllm")
        from vllm import SamplingParams
        from vllm.sampling_params import GuidedDecodingParams

        params = (
            self.sampling_params
            | (
                dict(guided_decoding=GuidedDecodingParams(json=schema))
                if schema
                else {}
            )
            | kwargs
        )

        params = SamplingParams(**params)
        tokenizer = self.lm.get_tokenizer()
        if (
            self.chat
            and hasattr(tokenizer, "chat_template")
            and tokenizer.chat_template
        ):
            # Chat mode
            if is_message_list(input):
                # Message list
                output = self.lm.chat([message.dump() for message in input], params)
            elif isinstance(input, list):
                # List of strings
                output = self.lm.chat(
                    [[dict(role="user", content=_in)] for _in in input],
                    params,
                )
            else:
                # Single string
                output = self.lm.chat([dict(role="user", content=input)], params)
            output = [Message(unprocessed=_out.outputs[-1].text) for _out in output]
        else:
            # Text generation mode
            if not isinstance(input, str) and not isinstance(input, list):
                raise ValueError("The input must be a string or a list of strings.")
            output = self.lm.generate(input, params)
            output = [Message(_out.outputs[0].text) for _out in output]
        if len(output) == 1:
            return output[0]
        return output
