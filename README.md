![CI](https://github.com/steerable-ai/lmfunctions/actions/workflows/ci.yml/badge.svg)
[![codecov](https://codecov.io/gh/steerable-ai/lmfunctions/graph/badge.svg?token=JHZLMOYX86)](https://codecov.io/gh/steerable-ai/lmfunctions)
[![PyPI](https://img.shields.io/pypi/v/lmfunctions)](https://pypi.org/project/lmfunctions/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/lmfunctions)](https://pypi.org/project/lmfunctions/)
[![PyPI - License](https://img.shields.io/pypi/l/lmfunctions)](https://pypi.org/project/lmfunctions/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/lmfunctions)](https://pypi.org/project/lmfunctions/)

# lmfunctions

Express language model tasks as Python functions. Just define the signature and docstring and add the `@lmdef` decorator:

```python
from lmfunctions import lmdef

@lmdef
def qa(context: str, query: str) -> str:
    """
    Answer the question using information from the context
    """
```

Calling the function will invoke a language model under the hood:

```python
context = """John started his first job right after graduating from college in 2005.
He spent five years working in that company before deciding to pursue a master's degree,
which took him two years to complete. After obtaining his master's degree, he worked
in various companies for another decade before landing his current job, which he has been in
for the past three years. John mentioned that he entered college at the typical age of 18"""

query = "How old is John?"

qa(context,query)
```
```plaintext
Based on the given context, ...
```

Easily serve the function as an API (using [fastapi](https://fastapi.tiangolo.com/) and [uvicorn](https://www.uvicorn.org/)):

```python
qa.serve()
```
```plaintext
INFO:     Started server process [152003]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```


## What does this package do?

* Streamlines **prompt engineering** by reducing it to a function definition
* Makes it easy to perform **structured data generation** and **function calling**
* Enforces **constraints and guardrails** via constrained generation whenever possible
* Supports both **local and remote** language models
* Provides **event managers** to handle events via callbacks (predefined or custom)
* Provides a **retry policy** to handle exceptions 
* Every component (functions, backend settings, event managers) can be **serialized and deserialized**
* Tasks for language models can be **expressed intuitively** and mixed with regular Python code

## QuickStart

- You will need [Python](https://www.python.org/downloads/) (version at least 3.10) and at least one of the supported [language model backends](#language-model-backends) installed.

- Install the package (preferably in a virtual environment) with

    ```console
    pip install lmfunctions
    ```

- Test the installation with

    ```console
    python -c "import lmfunctions as lmf; print(lmf.from_store('steerable/lmfunc/plan')('save the world'))"
    ```

  If the default language model backend `llama-cpp-python` is not available in your environment, you will be prompted to install it. The installation will attempt to autodetect CUDA GPU availability and use it accordingly, otherwise
  it will install a version with CPU support only. For best performance tailored to your hardware and platform, it is recommended to install local backends such as llama.cpp by following the instructions in the corresponding documentation. 

- If you are starting a new project with `lmfunctions`, a possible way to set up your environment is as follows:

    ```console
    mkdir <project_name>
    cd <project_name>
    python -m venv .venv
    source .venv/bin/activate 
    pip install --upgrade pip
    pip install lmfunctions 
    ```

    On Windows, run `venv\Scripts\activate` instead of `source .venv/bin/activate`.

## Tasks

- Simple tasks (e.g. **classification**) can be expressed by using type hints:

    ```python
    from lmfunctions import lmdef
    from typing import Literal
    
    @lmdef
    def sentiment(comment: str) -> Literal["negative","neutral","positive"]:
        """ Analyze the sentiment of the given comment """
    ```
    
    ```python
    sentiment("Even though it was raining, we had a good time")
    ```
    ```plaintext
    <Output.positive: 'positive'>
    ```

- To specify more complex tasks, use **Pydantic models** or **JSON schemas**. This allows to inject information about the fields:

    ```python
    from lmfunctions import lmdef
    from pydantic import BaseModel, Field
    
    class CityInfo(BaseModel):
        country: str
        population: float = Field(description="Population expressed in Millions")
        languages_spoken: list[str]
    
    @lmdef
    def info(cityname: str) -> CityInfo:
        """
        Returns information about the city
        """
    ```
    ```python
    info("Paris")
    ```
    ```plaintext
    CityInfo(country='France', population=2.16, languages_spoken=['French'])
    ```
    

- **Structured data** can be generated by simply defining a language function without input arguments:

    ```python
    from lmfunctions import lmdef
    from pydantic import BaseModel
    
    class Cocktail(BaseModel):
        name: str
        glass_type: str
        ingredients: list[str]
        instructions: list[str]
    
    @lmdef
    def cocktail() -> Cocktail:
        """Invent a new cocktail"""
    ```
    ```python
    cocktail()
    ```
    ```
    Cocktail(name='Sakura Sunset', glass_type='Coupe glass', ingredients=['1 1/2 oz Japanese whiskey' ...
    ```
    

- Language functions can be serialized:

    ```python
    sentiment_yaml = sentiment.dumps(format='yaml')
    ```

    deserialized:

    ```python
    from lmfunctions import from_string
    sentiment_deserialized = from_string(sentiment_yaml)
    sentiment_deserialized("This is an excellent Python package")
    ```
    ```plaintext
    <Output.positive: 'positive'>
    ```

    and dynamically loaded from remote artifacts

    ```python
    from lmfunctions import from_store
    route = from_store("steerable/lmfunc/route")
    route(origin="Seattle",destination="New York")
    ```
    ```plaintext
    FlightRoute(airports=['SEA', 'ORD', 'JFK'], cost_of_flight=350)
    ```

## Language Model Backends

The backends currently supported are 

* [llamacpp](https://github.com/ggerganov/llama.cpp): lean backend that runs locally hosted quantized models (GGUF format) on a variety of CPU and GPU devices
* [transformers](https://github.com/huggingface/transformers): runs models in HF transformers format (high flexibility but heavy dependencies such as PyTorch)
* [litellm](https://github.com/BerriAI/litellm): provides a common wrapper interface (OpenAI-compatible) for several language model providers

The default backend can be set using the `set_backend` function. For example, the following sets a locally hosted model backed by `llamacpp` where the model weights are retrieved from HuggingFace Hub:

```python
import lmfunctions as lmf
lmf.set_backend.llamacpp(model="hf://Qwen/Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-q4_k_m.gguf")
```

To invoke a remote language model via API (OpenAI, Anthropic, Cohere, etc), obtain the corresponding API key by creating an account with these providers, then use the `litellm` backend

```python
import os 
import lmfunctions as lmf
os.environ['OPENAI_API_KEY'] = <YOUR_OPENAI_API_KEY>
lmf.set_backend.litellm(model="gpt-4o-mini")
```

The API keys can be also set as environment variables (preferable so they are not in the code):

```python
import lmfunctions as lmf
lmf.set_backend.litellm(model="gpt-4o-mini")
```

When making individual calls to a language function, the default backend can be overridden:

```python
from lmfunctions.backends import LlamaCppBackend
llama3_1_8B = LlamaCppBackend()
qa(context,query,backend=llama3_1_8B)
```

To display information about the current backend settings, use the command

```python
lmf.default.backend.info()
```

## Event Manager

Execution of a language function proceeds through several steps:

* Call start
* Prompt template render
* Token or character processed
* Retry in case of exceptions
* Failure
* Success in obtaining and parsing the output

Event Managers can be used to introduce callback handlers for each of these events. For example they can be used to instrument all execution stages, gaining visibility into internal variables and metrics.

## Retry Policy

A retry policy specifies what to do when an exception occurs while executing the language function, for example when when the language model is unable to generate an output in the desired format. [Tenacity](https://tenacity.readthedocs.io/en/latest/) is used to implement the retries callbacks, with the class `RetryPolicy` wrapping some tenacity's input arguments in a serializable format

```python
from lmfunctions import RetryPolicy
retrypolicy = RetryPolicy(stop_max_attempt= 2, wait="fixed")
retrypolicy.info()
```

The default RetryPolicy can be set as follows:

```python
lmf.default.retry_policy = retrypolicy
```

Individual fields can be also modified:

```python
import lmfunctions as lmf
lmf.default.retry_policy.stop_max_attempt = 10
```
