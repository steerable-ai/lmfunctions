# LMdef

![CI](https://github.com/steerable-ai/lmdef/actions/workflows/ci.yml/badge.svg)

Easily create Python functions backed by language models. Just define the signature and docstring and add the `@lmdef` decorator:

```python
from lmfunctions import lmdef

@lmdef
def contextual_qa(context: str, query: str) -> str:
    """
    Answer the question using information from the context
    """
```

The resulting `language function` invokes a language model backend under the hood, but can be used just like a regular function:

```python
context = """John is planning a vacation. He wants to visit a country with a rich history,
             delicious cuisine, and beautiful beaches. He also prefers places where English
             is commonly spoken."""
query = "Where should John go?"

contextual_qa(context,query)

# Based on the given context, ...
```

<details> <summary>Language model backend</summary>

The default backend can be configured to invoke a remote API (such as OpenAI's GPT):

```python
lmf.set_backend.litellm(model="gpt-4o")
```
or run a local model (for example via llama.cpp):

```python
import lmfunctions as lmf
lmf.set_backend.llamacpp(model="hf://Qwen/Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-q4_k_m.gguf")
```

See all supported [language model backends](#language-model-backend).

</details>

<details>
<summary>Add Constraints</summary>

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

</details>

<details> <summary>Complex Constraints</summary>

[Pydantic](https://docs.pydantic.dev/latest/) models or JSON schemas can be used to specify complex constraints and inject additional information about fields, useful to guide the model

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

</details>

<details> <summary>Structured Data Generation</summary>

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
```

```python
cocktail()
# Cocktail(name='Sakura Sunset', glass_type='Coupe glass', ingredients=['1 1/2 oz Japanese whiskey', '1/2 oz cherry liqueur', ...
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

This allows to store them in text files and dynamically load them from a remote artifact:

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

<details> <summary>Additional examples</summary>

See this [notebook](notebooks/Examples.ipynb).

</details>

## Installation

* Install at least one of the supported [language model backend](#language-model-backend):

    * llama.cpp (CPU only):

    ```console
    pip install llama-cpp-python
    ```
    * llama.cpp (GPU with CUDA support):

    ```console
    CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python
    ```
    
    * Other options to build llama.cpp are listed [here](https://llama-cpp-python.readthedocs.io/en/latest/)

    * Transformers 

    ```console
    pip install transformers[torch]
    ```

    * Litellm (API-based language models)

    ```console
    pip install litellm
    ```


* Install the package with `pip` or  [`Poetry`](https://python-poetry.org/docs)
  
```console
pip install lmdef
```

```console
poetry add lmdef
```

## Language Model Backend

The backends currently supported are 

* [llamacpp](https://github.com/ggerganov/llama.cpp)
* [transformers](https://github.com/huggingface/transformers)
* [litellm](https://github.com/BerriAI/litellm)

The default backend can be set using shorcuts. For example, the following sets `llamacpp` and retrieves a model from from HuggingFace Hub:

```python
import lmfunctions as lmf

lmf.set_backend.llamacpp(model="hf://Qwen/Qwen2-1.5B-Instruct-GGUF/qwen2-1_5b-instruct-q4_k_m.gguf")
```

API providers such as OpenAI (GPT), Anthropic (Claude), Cohere, and many others can be accessed using the [litellm](https://github.com/BerriAI/litellm) backend. For example,
to use OpenaAI's GPT-4o API:

```python
lmf.set_backend.litellm(model="gpt-4o")
```

This requires setting suitable API keys in the environment (in this case an OpenAI API key obtainable by creating an OpenAI account).

###

The default backend can be overridden when calling the language function:

```python
from lmfunctions.lmbackend import LiteLLMBackend
gpt4omini = LiteLLMBackend(model="gpt-4o-mini")
contextual_qa(context,query,backend=gpt4omini)
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
import lmfunctions import lmf
lmf.retrypolicy.stop_max_attempt=10
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
