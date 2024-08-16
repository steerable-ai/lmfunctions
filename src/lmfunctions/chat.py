from functools import partial
from typing import Callable, List, Optional

from rich import print

from lmfunctions.backends import LMBackend
from lmfunctions.default import default
from lmfunctions.eventmanager import EventManager
from lmfunctions.lmfunc import lmdef
from lmfunctions.managers import tokenStream
from lmfunctions.message import Message


def multiline_input(terminators=tuple()):
    """
    Simple input function that allows entering multiple lines
    """
    lines = []
    line = input(">>> ")
    lines.append(line)
    # Check if the line contains both opening and closing """ on the same line
    if line.count('"""') >= 2:
        return line  # Return immediately if both are on the same line
    # Check if the line contains an opening """
    if '"""' in line:
        while True:
            line = input("... ")
            lines.append(line)
            # Check if the current line contains the closing triple quote
            if '"""' in line:
                break
    elif line in terminators:
        return line
    return "\n".join(lines)


def initialize_chat(backend: LMBackend):
    # Call the backend to load the model
    backend(
        schema={
            "properties": {"message": {"enum": ["warmup"], "type": "string"}},
            "type": "object",
        },
    )
    print(
        f"Backend: {backend.name}\nModel: {backend.model}\n"
        "Type /exit to exit.\n"
        "Type /history to see all messages\n"
        "Type /clear to clear the chat history\n"
        "Interrupt generation with Ctrl+C (streaming-mode only)\n"
        'Use triple quotes """ to enter multiple lines\n'
    )


@lmdef
def chatmessage(messages: List[Message]) -> Message: ...  # type: ignore


def chat(
    backend: Optional[LMBackend] = None,
    event_manager: Optional[EventManager] = None,
    initialize_chat: Optional[Callable] = initialize_chat,
    user_input: Callable[[], str] = partial(
        multiline_input, terminators=("", "\n", "/exit", "/clear", "/history")
    ),
    system_message="",
):
    """
    A minimal chat loop that interacts with the specified backend
    """
    backend = backend or default.backend
    event_manager = event_manager or tokenStream
    if initialize_chat:
        initialize_chat(backend)
    messages: List = []
    if system_message:
        messages.append(Message(role="system", content=system_message))
    while True:
        user_message = user_input()
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
            messages.append(Message(role="user", content=user_message))
            try:
                response = chatmessage(
                    messages,
                    event_manager=event_manager,
                )
            except KeyboardInterrupt:  # pragma: no cover
                pass  # pragma: no cover
            finally:
                print()
                messages.append(response)
