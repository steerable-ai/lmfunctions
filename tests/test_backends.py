import lmfunctions as lmf
from lmfunctions.backends import LlamaCppBackend

from .models import test_models

prompt = "1, 2, 3, "

schema = test_models[0].model_json_schema()


TEST_CHAT_BACKEND = LlamaCppBackend(
    model="hf://Qwen/Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-q4_k_m.gguf",
    n_ctx=1024,
)


def test_transformers():
    lmf.set_backend.transformers(generation=dict(max_new_tokens=5))
    lmf.complete(prompt)
    # Test model unload and reload
    setattr(lmf.default.backend, "model", getattr(lmf.default.backend, "model"))
    lmf.complete("", schema)
    # Test non-chat template code path
    lmf.default.backend.pipeline.tokenizer.chat_template = None
    lmf.complete(prompt)


def test_llamacpp():
    lmf.default.backend = TEST_CHAT_BACKEND
    lmf.complete(prompt)
    # Test loading model from local path
    lmf.default.backend.model = lmf.default.backend.llama.model_path
    lmf.default.backend.generation.stream = False
    lmf.complete("", schema)
    # # Test code path for base models without chat template
    chat_template = TEST_CHAT_BACKEND.llama.metadata["tokenizer.chat_template"]
    del TEST_CHAT_BACKEND.llama.metadata["tokenizer.chat_template"]
    lmf.default.backend.generation.stream = True
    lmf.complete(prompt)
    lmf.default.backend.generation.stream = False
    lmf.complete(" ", schema)  # Currently, the length of the prompt needs to be > 0
    lmf.set_backend.llamacpp()
    TEST_CHAT_BACKEND.llama.metadata["tokenizer.chat_template"] = chat_template


def test_litellm():
    lmf.set_backend.litellm(mock_response="4")
    lmf.complete(prompt)
    lmf.complete("", schema)
