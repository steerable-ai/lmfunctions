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

# Based on the given context, ...
```

<details> <summary>Backends</summary>

The default backend can be configured to invoke a remote API supported by litellm (such as OpenAI's GPT):

```python
import lmfunctions as lmf
lmf.set_backend.litellm(model="gpt-4o-mini")
```
or a local model via llama.cpp or HF Transformers

```python
import lmfunctions as lmf
lmf.set_backend.llamacpp(model="hf://Qwen/Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-q4_k_m.gguf")
```

</details>

<details>
<summary>Tasks</summary>

Constraints on inputs and outputs can be enforced via type hints. For instance, a text classification task can be expressed as follows:

```python
from typing import Literal

@lmdef
def sentiment(comment: str) -> Literal["negative","neutral","positive"]:
    """ Analyze the sentiment of the given comment """
```

```python
sentiment("I feel under the weather today")
# <Output.negative: 'negative'>
```

[Pydantic](https://docs.pydantic.dev/latest/) models or JSON schemas can be used to specify more complex constraints and inject information about the fields:

```python
from lmfunctions import lmdef
from pydantic import BaseModel, Field

class CityInfo(BaseModel):
    country: str
    population: float = Field(description="Population expressed in Millions")
    languages_spoken: list[str]

@lmdef
def city_info(input: str) -> CityInfo:
    """
    Returns information about the city
    """

city_info("Paris")
# CityInfo(country='France', population=2.16, languages_spoken=['French'])
```

Generating structured data can be accomplished by simply defining a language function without input arguments:

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

cocktail()
# Cocktail(name='Sakura Sunset', glass_type='Coupe glass', ingredients=['1 1/2 oz Japanese whiskey' ...
```

</details>

<details> <summary>Serialization</summary>

Language functions can be serialized

```python
from lmfunctions import from_string, lmdef
from typing import Literal

@lmdef
def sentiment(comment: str) -> Literal["negative","neutral","positive"]:
    """ Analyze the sentiment of the given comment """

sentiment_yaml = sentiment.dumps(format='yaml')
```

and deserialized

```python
sentiment_deserialized = from_string(sentiment_yaml)
sentiment_deserialized("This is an excellent Python package")
# <Output.positive: 'positive'>
```

This allows to store them in text files and dynamically load them from remote artifacts:

```python
from lmfunctions import from_store
route = from_store("steerable/lmfunc/route")
route(origin="Seattle",destination="New York")
# FlightRoute(airports=['SEA', 'ORD', 'JFK'], cost_of_flight=350)
```

</details>

<details> <summary>Observability</summary>
Event managers and callbacks allow to instrument all execution stages, gaining visibility into internal variables and metrics.

</details>

## QuickStart

* Requirements: Python > 3.10

* Install the package (preferably in a virtual environment):

    ```console
    pip install lmfunctions
    ```
    
* Test the installation with

   ```console
   python -c "import lmfunctions as lmf; print(lmf.from_store('steerable/lmfunc/plan')('save the world'))"
   ```
   
   This will prompt you to install the default language model backend `llama-cpp-python` if not yet present in your environment. The installation will try to detect CUDA availability and use it accordingly, otherwise it will install the default build with CPU support only.

* If you are starting a new project with `lmfunctions`, you can set up the environment as follows:

  ```console
  mkdir new_project
  cd new_project
  python -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install lmfunctions 
  ```
    
## Language Model Backend

The backends currently supported are 

* [llamacpp](https://github.com/ggerganov/llama.cpp)
* [transformers](https://github.com/huggingface/transformers)
* [litellm](https://github.com/BerriAI/litellm)

The default language model can be set using shorcuts. For example, the following sets `llamacpp` and retrieves a model from from HuggingFace Hub:

```python
import lmfunctions as lmf
lmf.set_backend.llamacpp(model="hf://Qwen/Qwen2-1.5B-Instruct-GGUF/qwen2-1_5b-instruct-q4_k_m.gguf")
```

API providers such as OpenAI (GPT), Anthropic (Claude), Cohere, and many others can be accessed using the [litellm](https://github.com/BerriAI/litellm) backend. For example,
to use OpenaAI's GPT-4o API:

```python
lmf.set_backend.litellm(model="gpt-4o-mini")
```
The necessary API keys (in this case OpenAI's) need to be set as environment variables.

###

The default backend can be overridden when calling the language function:

```python
from lmfunctions.backends import LiteLLMBackend
gpt_4o = LiteLLMBackend(model="gpt-4o")
qa(context,query,backend=gpt_4o)
```

To display information about the current language model backend settings:

```python
lmf.default.backend.info()
# ...
```

## Retry Policy

A retry policy specifies what to do when an exception occurs while executing the language function, for example when when the language model is unable to generate an output in the desired format. [Tenacity](https://tenacity.readthedocs.io/en/latest/) is used to implement the retries callbacks, with the class `RetryPolicy` wrapping some tenacity's input arguments in a serializable format

```python
from lmfunctions import RetryPolicy
retrypolicy = RetryPolicy(stop_max_attempt= 2, wait="fixed")
retrypolicy.info()
```

The default RetryPolicy can be modified as follows:

```python
import lmfunctions as lmf
lmf.default.retry_policy.stop_max_attempt = 10
```

## Event Manager

Execution of a language function proceeds through several steps:

* Call start
* Prompt template render
* Token or character processed
* Retry in case of exceptions
* Failure
* Success in obtaining and parsing the output

Event Managers can be used to introduce callback handlers for each of these events.
