from json import JSONDecodeError

import pytest

import lmfunctions as lmf
from lmfunctions import LMResponse

from .test_lmbackend import TEST_CHAT_BACKEND


def test_init_response():
    response = LMResponse("This is a string response.")
    str(response)
    repr(response)
    print(response)
    response = LMResponse(iter("This is an iterator response."))
    str(response)
    repr(response)
    print(response)


def test_init_from_openai_v1():
    lmf.default.backend = TEST_CHAT_BACKEND
    lmf.complete("1, 2, 3, 4, ")()
    lmf.default.backend.generation.stream = False
    lmf.complete("1, 2, 3, 4, ")()


def test_parse_response():
    LMResponse('{"first_valid": "json"}')(schema={"type": "object"})
    LMResponse('{+}   {"first_valid": "json"}')(schema={"type": "object"})
    with pytest.raises(JSONDecodeError):
        LMResponse("  ")(schema={"type": "object"})
