import time
from multiprocessing import Process
from typing import Dict, Literal, Tuple

import pytest
import requests

import lmfunctions as lmf
from lmfunctions import LMFunc, from_store, from_string, lmdef

from .models import CityInfo, Entities, FlightRoute, NERInput, Plan, TwoCities
from .test_backends import TEST_CHAT_BACKEND


@lmdef
def none(): ...


@lmdef
def anagram(sentence: str) -> str:
    """
    Returns an anagram of the input string that is a valid English sentence
    """
    ...  # pragma: no cover


@lmdef
def city_info(city: str) -> CityInfo:
    """
    Returns information about the city
    """
    ...  # pragma: no cover


@lmdef
def plan(goal: str) -> Plan:
    """
    Given an input goal to achieve,
    - A detailed list of steps to accomplish the goal
    - Assumptions being made when generating the list of steps
    """
    ...  # pragma: no cover


@lmdef
def contextual_qa(context: str, query: str) -> str:
    """
    Answer the question using information from the context
    """
    ...  # pragma: no cover


@lmdef
def route(input: TwoCities) -> FlightRoute:
    """
    Given two cities (origin) and destination, outputs a minimal cost route
    consisting of a sequence of airport codes to visit,
    and an approximate cost of the flight in US dollars.
    """
    ...  # pragma: no cover


@lmdef
def ner(input: NERInput) -> Entities:
    """
    Given an input text and a list of entity names to extract,
    find all words in the text that can be associated to one of
    the entity names listed, and output a map between the word
    and the entity name
    """
    ...  # pragma: no cover


@lmdef
def sum(x: int, y: int) -> int:
    """
    Returns the sum of the two integers
    """
    ...  # pragma: no cover


@lmdef
def sentiment(comment: str) -> Literal["positive", "negative", "neutral"]:
    """Analyze the sentiment of the given comment"""
    ...  # pragma: no cover


@lmdef
def chat(message: str) -> str: ...


truthmachine = LMFunc(
    name="truthmachine",
    description="Returns whether the statement is true or false",
    input_schema=dict(type="string"),
    output_schema=dict(type="boolean"),
)

noneraw = LMFunc(
    name="noneraw",
    description="Returns nothing",
    input_schema=None,
    output_schema=dict(type="null"),
)

test_functions: Dict[str, Tuple[LMFunc, Tuple, Dict]] = {
    "none": (none, tuple(), {}),
    "anagram": (anagram, ("dormitory",), {}),
    "city_info": (city_info, ("New York",), {}),
    "route": (
        route,
        (TwoCities(origin="New York", destination="Los Angeles"),),
        {},
    ),
    "sum": (sum, tuple(), dict(x=3, y=4)),
    "sentiment": (sentiment, ("I love this!",), dict()),
    "truthmachine": (truthmachine, ("This is a test",), dict()),
    "noneraw": (noneraw, tuple(), {}),
    "chat": (chat, ("Hello",), {}),
}


def test_lmfuncs():
    lmf.default.backend = TEST_CHAT_BACKEND
    for func, args, kwargs in test_functions.values():
        func(*args, **kwargs)
    func.info()


def test_serialize_deserialize():
    lmf.default.backend = TEST_CHAT_BACKEND
    for format in ["json", "yaml"]:
        for func, args, kwargs in test_functions.values():
            func.info()
            serialized = func.dumps(format=format)
            deserialized = from_string(serialized, format=format)
            assert deserialized.dumps(format=format) == serialized
            deserialized(*args, **kwargs)

    anagram = from_store("steerable/lmfunc/anagram")

    with pytest.raises(ValueError):
        serialized = anagram.dumps(format="unsupported")
    with pytest.raises(ValueError):
        from_string("string", format="unsupported")

    anagram.loads(anagram.dumps())


test_url = "http://localhost:8000/docs"
func = from_store("steerable/lmfunc/route")


def serve_func():
    func.serve()


@pytest.fixture(scope="module")
def server():
    timeout = 20
    from pytest_cov.embed import cleanup_on_sigterm

    cleanup_on_sigterm()
    proc = Process(target=serve_func, args=(), daemon=True)
    proc.start()
    start_time = time.time()
    while True:
        try:
            response = requests.head(test_url)
            if response.status_code == 200:
                break
        except requests.ConnectionError:
            pass
        time.sleep(0.1)
        if time.time() - start_time > timeout:
            raise TimeoutError(
                f"Server didn't start within {timeout} seconds"
            )  # pragma: no cover
    yield proc
    proc.terminate()
    proc.join()
    proc.close()


def test_serve(server):
    response = requests.get(test_url)
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_fastapi_app():
    lmf.default.backend = TEST_CHAT_BACKEND
    for func, args, kwargs in test_functions.values():
        if kwargs:
            await func.async_handler()(kwargs)
            assert func.fastapi_app()
            assert from_string(func.dumps()).fastapi_app()
        else:
            await func.async_handler()(*args)
            assert func.fastapi_app()
            assert from_string(func.dumps()).fastapi_app()
