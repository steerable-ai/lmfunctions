from functools import partial
from typing import Any, Callable, List, Optional

from rich import print

from lmfunctions.default import default
from lmfunctions.lmbackend.lmbackend import LMBackend


def multiline_input(terminators=tuple()):
    """
    Simple input function that allows entering multiple lines
    """
    lines = []
    prompt = ">>> "
    while True:
        line = input(prompt)
        lines.append(line)
        if line in terminators:
            break
        prompt = "... "
    return "\n".join(lines)


def default_setup_callback(backend: LMBackend):
    backend.complete("warmup json", schema={"type": "null"})
    print(
        f"Backend: {backend.name}\nModel: {backend.model}\n"
        "Type /exit to exit.\n"
        "Type /history to see all messages\n"
        "Type /clear to clear the chat history\n"
        "Interrupt generation with Ctrl+C (streaming-mode only)\n"
    )


def chat(
    backend: Optional[LMBackend] = None,
    setup_callback: Optional[Callable] = default_setup_callback,
    input_callback: Callable[[], str] = partial(
        multiline_input, terminators=("", "\n", "/exit", "/clear", "/history")
    ),
    new_token_or_char_callback: Callable[[Any], None] = lambda **kwargs: print(
        kwargs["token_or_char"], end="", flush=True
    ),
):
    """
    A minimal chat loop that interacts with the default runtime language model.
    """
    if backend is None:
        backend = default.backend
    if setup_callback:
        setup_callback(backend)
    messages: List = []
    while True:
        user_message = input_callback()
        if not user_message:
            continue
        if user_message in ("/exit", "/clear"):
            if user_message == "/exit":
                return
            else:
                messages = []
        elif user_message == "/history":
            print(messages)
        else:
            messages.append({"role": "user", "content": user_message})
            try:
                response = backend.chat_complete(messages)
                response(new_token_or_char_callback=new_token_or_char_callback)
            except KeyboardInterrupt:  # pragma: no cover
                pass  # pragma: no cover
            finally:
                print()
                messages.append({"role": "assistant", "content": response()})
