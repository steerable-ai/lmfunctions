from json import JSONDecodeError

import pytest

import lmfunctions as lmf
from lmfunctions import Message

from .test_backends import TEST_CHAT_BACKEND


def test_init_response():
    response = Message("This is a string response.")
    str(response)
    repr(response)
    print(response)
    response = Message(iter("This is an iterator response."))
    str(response)
    repr(response)
    print(response)


def test_init_from_openai_v1():
    lmf.default.backend = TEST_CHAT_BACKEND
    lmf.complete("1, 2, 3, 4, ").process()
    lmf.default.backend.generation.stream = False
    lmf.complete("1, 2, 3, 4, ").process()


def test_parse_response():
    Message('{"first_valid": "json"}').process(schema={"type": "object"})
    Message('{+}   {"first_valid": "json"}').process(schema={"type": "object"})
    with pytest.raises(JSONDecodeError):
        Message("  ").process(schema={"type": "object"})
