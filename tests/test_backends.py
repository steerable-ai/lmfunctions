import lmfunctions as lmf

from .models import test_models

prompt = "1, 2, 3, "
conversation = [
    lmf.Message(role="user", content="Hello"),
    lmf.Message(role="assistant", content="Hi"),
]

schema = test_models[0].model_json_schema()


TEST_CHAT_BACKEND = lmf.backends.LlamaCppBackend(
    model="hf://Qwen/Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-q4_k_m.gguf",
    n_ctx=1024,
)


def test_vllm():
    lmf.set_backend.vllm()
    # Test chat mode (default)
    out = lmf.complete(prompt)  # Single string
    assert isinstance(out, lmf.Message)
    out = lmf.complete(prompt, schema)  # Single string with schema
    assert isinstance(out, lmf.Message)
    out = lmf.complete([prompt] * 2)  # List of strings
    assert (
        isinstance(out, list)
        and len(out) == 2
        and all(isinstance(m, lmf.Message) for m in out)
    )
    out = lmf.complete([prompt] * 2, schema)  # List of strings with schema
    assert (
        isinstance(out, list)
        and len(out) == 2
        and all(isinstance(m, lmf.Message) for m in out)
    )
    out = lmf.complete(conversation)  # Message list
    assert isinstance(out, lmf.Message)
    out = lmf.complete(conversation, schema)  # Message list with schema
    assert isinstance(out, lmf.Message)
    # Test text generation mode
    lmf.default.backend.chat = False
    out = lmf.complete(prompt)  # Single string
    assert isinstance(out, lmf.Message)
    out = lmf.complete(prompt, schema)  # Single string with schema
    assert isinstance(out, lmf.Message)
    out = lmf.complete([prompt] * 2)  # List of strings
    assert (
        isinstance(out, list)
        and len(out) == 2
        and all(isinstance(m, lmf.Message) for m in out)
    )
    out = lmf.complete([prompt] * 2, schema)  # List of strings with schema
    assert (
        isinstance(out, list)
        and len(out) == 2
        and all(isinstance(m, lmf.Message) for m in out)
    )


def test_transformers():
    lmf.set_backend.transformers(generation=dict(max_new_tokens=5))
    # Test chat mode (default)
    out = lmf.complete(prompt)  # Single string
    assert isinstance(out, lmf.Message)
    out = lmf.complete(prompt, schema)  # Single string with schema
    assert isinstance(out, lmf.Message)
    out = lmf.complete([prompt] * 2)  # List of strings
    assert (
        isinstance(out, list)
        and len(out) == 2
        and all(isinstance(m, lmf.Message) for m in out)
    )
    out = lmf.complete([prompt] * 2, schema)  # List of strings with schema
    assert (
        isinstance(out, list)
        and len(out) == 2
        and all(isinstance(m, lmf.Message) for m in out)
    )
    out = lmf.complete(conversation)  # Message list
    assert isinstance(out, lmf.Message)
    out = lmf.complete(conversation, schema)  # Message list with schema
    assert isinstance(out, lmf.Message)
    # Test text generation mode
    lmf.default.backend.chat = False
    out = lmf.complete(prompt)  # Single string
    assert isinstance(out, lmf.Message)
    out = lmf.complete(prompt, schema)  # Single string with schema
    assert isinstance(out, lmf.Message)
    out = lmf.complete([prompt] * 2)  # List of strings
    assert (
        isinstance(out, list)
        and len(out) == 2
        and all(isinstance(m, lmf.Message) for m in out)
    )
    out = lmf.complete([prompt] * 2, schema)  # List of strings with schema
    assert (
        isinstance(out, list)
        and len(out) == 2
        and all(isinstance(m, lmf.Message) for m in out)
    )


def test_litellm():
    lmf.set_backend.litellm(mock_response="4")
    # Test chat mode (default)
    out = lmf.complete(prompt)  # Single string
    assert isinstance(out, lmf.Message)
    out = lmf.complete(prompt, schema)  # Single string with schema
    assert isinstance(out, lmf.Message)
    out = lmf.complete([prompt] * 2)  # List of strings
    assert (
        isinstance(out, list)
        and len(out) == 2
        and all(isinstance(m, lmf.Message) for m in out)
    )
    out = lmf.complete([prompt] * 2, schema)  # List of strings with schema
    assert (
        isinstance(out, list)
        and len(out) == 2
        and all(isinstance(m, lmf.Message) for m in out)
    )
    out = lmf.complete(conversation)  # Message list
    assert isinstance(out, lmf.Message)
    out = lmf.complete(conversation, schema)  # Message list with schema
    assert isinstance(out, lmf.Message)
    # Test text generation mode
    lmf.default.backend.chat = False
    out = lmf.complete(prompt)  # Single string
    assert isinstance(out, lmf.Message)
    out = lmf.complete(prompt, schema)  # Single string with schema
    assert isinstance(out, lmf.Message)


def test_llamacpp():
    lmf.default.backend = TEST_CHAT_BACKEND
    # Test chat mode (default)
    lmf.complete(prompt)
    # Test loading model from local path
    lmf.default.backend.model = lmf.default.backend.llama.model_path
    lmf.default.backend.generation.stream = False
    lmf.complete("", schema)
    # Test text generation mode
    chat_template = TEST_CHAT_BACKEND.llama.metadata["tokenizer.chat_template"]
    del TEST_CHAT_BACKEND.llama.metadata["tokenizer.chat_template"]
    lmf.default.backend.generation.stream = True
    lmf.complete(prompt)
    lmf.default.backend.generation.stream = False
    lmf.complete(prompt, schema)
    lmf.set_backend.llamacpp()
    TEST_CHAT_BACKEND.llama.metadata["tokenizer.chat_template"] = chat_template
