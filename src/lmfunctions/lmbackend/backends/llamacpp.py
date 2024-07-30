import gc
import json
import multiprocessing
from typing import Any, Dict, Iterator, List, Literal, Optional, Union

import huggingface_hub
from pydantic import model_validator

from lmfunctions.base import Base
from lmfunctions.lmbackend.lmbackend import LMBackend
from lmfunctions.lmresponse import LMResponse
from lmfunctions.utils import cuda_check, lazy_import


class LLamaCppGenerationParams(Base):
    """
    Parameters controlling generation with the LLamaCpp model.
    """

    max_tokens: int | None = None
    temperature: float = 0.8
    top_p: float = 0.95
    min_p: float = 0.05
    typical_p: float = 1
    logprobs: int | None = None
    stop: str | List[str] | None = []
    frequency_penalty: float = 0
    presence_penalty: float = 0
    repeat_penalty: float = 1.1
    top_k: int = 40
    stream: bool = True
    seed: int | None = None
    tfs_z: float = 1
    mirostat_mode: int = 0
    mirostat_tau: float = 5
    mirostat_eta: float = 0.1
    model: str | None = None
    logit_bias: Dict[str, float] | None = None


class LlamaCppBackend(LMBackend):
    """
    LLamaCpp model configuration.  The model is loaded lazily, only when it is accessed.
    If any of the arguments get modified, the current model is unloaded.
    """

    name: Literal["llamacpp"] = "llamacpp"
    model: str = (
        "hf://QuantFactory/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf"
    )
    n_gpu_layers: Optional[int] = None
    split_mode: int = 1
    main_gpu: int = 0
    tensor_split: List[float] | None = None
    vocab_only: bool = False
    use_mmap: bool = True
    use_mlock: bool = False
    kv_overrides: Dict[str, bool | int | float] | None = None
    seed: int = 4294967295
    n_ctx: int = 0
    n_batch: int = 512
    n_threads: int | None = None
    n_threads_batch: int | None = None
    rope_scaling_type: int | None = -1
    rope_freq_base: float = 0
    rope_freq_scale: float = 0
    yarn_ext_factor: float = -1
    yarn_attn_factor: float = 1
    yarn_beta_fast: float = 32
    yarn_beta_slow: float = 1
    yarn_orig_ctx: int = 0
    mul_mat_q: bool = True
    logits_all: bool = False
    embedding: bool = False
    offload_kqv: bool = True
    last_n_tokens_size: int = 64
    lora_base: str | None = None
    lora_scale: float = 1
    lora_path: str | None = None
    numa: bool = False
    chat_format: str | None = None
    verbose: bool = False
    generation: LLamaCppGenerationParams = LLamaCppGenerationParams()

    _llama: Any = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.n_gpu_layers is None:
            gpu_info = cuda_check()
            if gpu_info["cuda_available"]:
                self.n_gpu_layers = -1
            else:
                self.n_gpu_layers = 0
        if self.n_threads is None:
            self.n_threads = multiprocessing.cpu_count() // 2

    @property
    def llama(self):
        if self._llama is None:
            lazy_import("llama_cpp", install_packages=["llama-cpp-python"])
            from llama_cpp import Llama

            model_reference = self.model
            if model_reference.startswith("hf://"):
                components = model_reference.split("://")[1].split("/")
                repo_id, filename = "/".join(components[0:2]), components[-1]
                model_path = huggingface_hub.hf_hub_download(repo_id, filename)
            else:
                model_path = model_reference
            self._llama = Llama(
                model_path=model_path,
                **self.model_dump(exclude={"name", "model", "generation"})
            )
        return self._llama

    def _unload(self):
        self._llama = None
        gc.collect()

    @model_validator(mode="after")
    def unload(self):
        if self._llama:
            self._unload()
        return self

    def complete(
        self, prompt: str = "", schema: Optional[Dict] = None, **kwargs
    ) -> LMResponse:

        if "tokenizer.chat_template" in self.llama.metadata:
            messages = [{"role": "user", "content": prompt}]
            response_format = (
                dict(type="json_object", schema=schema) if schema else None
            )
            return self.chat_complete(
                messages, response_format=response_format, **kwargs
            )
        else:
            lazy_import("llama_cpp", install_packages=["llama-cpp-python"])
            from llama_cpp.llama_chat_format import _grammar_for_json_schema

            grammar = _grammar_for_json_schema(json.dumps(schema)) if schema else None
            params = self.generation.model_dump() | kwargs
            response = self.llama.create_completion(prompt, grammar=grammar, **params)
            if isinstance(response, Iterator):
                response_generator = (c["choices"][0]["text"] or "" for c in response)
                return LMResponse(response_generator)
            else:
                content = response["choices"][0]["text"]
                return LMResponse(content if isinstance(content, str) else "")

    def chat_complete(self, messages: List, **kwargs) -> LMResponse:
        response = self.chat_complete_openai_v1(messages, **kwargs)
        return LMResponse.from_openai_v1(response)

    def chat_complete_openai_v1(
        self, messages: List, **kwargs
    ) -> Union[Any, Iterator[Any]]:
        params = self.generation.model_dump() | kwargs
        return self.llama.create_chat_completion_openai_v1(messages=messages, **params)
