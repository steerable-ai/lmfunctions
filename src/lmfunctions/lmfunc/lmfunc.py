import inspect
import json
import os
import re
from types import NoneType
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    ParamSpec,
    Type,
    TypeVar,
    get_type_hints,
)

from jinja2 import Template
from opentelemetry import trace
from pydantic import BaseModel, create_model
from tenacity import Retrying

from lmfunctions.base import Base
from lmfunctions.default import default
from lmfunctions.eventmanager import EventManager
from lmfunctions.lmbackend import LMBackend
from lmfunctions.retrypolicy import RetryPolicy

# from lmfunctions.tracer import tracer
from lmfunctions.utils import from_jsonschema, lazy_import

curdir = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(curdir, "metaprompt.jinja"), "r") as f:
    default_metaprompt = f.read()

InputArgs = ParamSpec("InputArgs")
ReturnType = TypeVar("ReturnType")


class LMFunc(Base, Generic[InputArgs, ReturnType]):
    """
    A class that represents a function implemented via a language model computation (language function). Language functions are serializable objects identified by the following attributes.

    Attributes:
        name: str
            A unique identifier for the language function.
        description: str
            A brief summary of what the language function does.
        input_schema: dict
            A JSON schema that defines the expected structure and constraints
            for input data.
        output_schema: dict
            A JSON schema that defines the expected structure and constraints
            for output data.
        metaprompt: str
            A Jinja template that combines the description, input schema, and output
            schema to generate a dynamic prompt at call time.

    Methods:
        __call__(*args, examples=[], backend=None, retry_policy=None, event_manager=None, extra_args={}, **kwargs) -> ReturnType:
            Calls the language function to generate a response based on the given inputs.
            When called, the language function relies on the following components:
            - A language model backend used to perform the computation.
            - A retry policy to handle exceptions and ensure reliable execution.
            - An event manager that invokes callback functions in correspondence to events.
    """

    name: str
    description: str = ""
    input_schema: Optional[Dict] = None
    output_schema: Optional[Dict] = None
    metaprompt: str = default_metaprompt

    _input_model: Optional[Type[BaseModel]] = None
    _output_model: Optional[Type[BaseModel]] = None
    _template: Optional[Template] = None

    @staticmethod
    def to_json_str(obj: Any) -> Optional[str]:
        """
        Returns a string representation of the given object which is
        a JSON dumps for BaseModel or dict objects, and a str otherwise.
        """
        if obj is None:
            return None
        elif isinstance(obj, dict):
            return json.dumps(obj, default=str)
        elif isinstance(obj, BaseModel):
            return obj.model_dump_json()
        else:
            return str(obj)

    @property
    def input_model(self) -> Type[BaseModel]:
        if self._input_model is None:
            self._input_model = from_jsonschema(self.input_schema or {})
        return self._input_model

    @property
    def output_model(self) -> Type[BaseModel]:
        if self._output_model is None:
            self._output_model = from_jsonschema(self.output_schema or {})
        return self._output_model

    @property
    def template(self) -> Template:
        if self._template is None:
            template = Template(self.metaprompt).render(
                description=self.description,
                input_schema=self.to_json_str(self.input_schema),
                output_schema=self.to_json_str(self.output_schema),
            )
            self._template = Template(template)
        return self._template

    def __init__(
        self, func: Optional[Callable[InputArgs, ReturnType]] = None, **kwargs
    ):
        """
        Initializes the language function with the provided model fields or a Python function.

        Args:
            func (Callable[InputArgs, ReturnType], optional): A Python function used to extract the name, description, input schema, and output schema of the language function. Defaults to None.
            **kwargs: If a Python function is not provided, we initialize directly the model fields: name, description, input_schema, output_schema, metaprompt.
        """
        # If no function is provided, initialize the model fields
        if not func:
            return super().__init__(**kwargs)

        # Extract name and description
        name = func.__name__
        description = re.sub(r"\s+", " ", (func.__doc__ or "").strip())

        # Extract input and output schemas
        signature = inspect.signature(func)
        type_hints = get_type_hints(func)
        return_hint = type_hints.pop("return", None)
        fields = {
            name: (
                type_hints.get(name, str),
                ... if bool(param.empty) else param.default,
            )
            for name, param in signature.parameters.items()
        }

        # Input Schema
        if len(fields) == 0:
            input_schema = None
        else:
            first_field = next(iter(fields.items()))
            first_type_hint = first_field[1][0]
            if (
                len(fields) == 1
                and type(first_type_hint) is type(BaseModel)
                and issubclass(first_type_hint, BaseModel)
            ):
                # If the only argument is a Pydantic model, get the input schema from it
                input_schema = first_type_hint.model_json_schema()
            elif len(fields) == 1 and first_type_hint is str:
                # If the only argument is a string, input_schema is None
                input_schema = None
            else:
                # Otherwise, create a wrapper model with the input arguments as fields
                input_schema = create_model(
                    "InputWrapper", **fields
                ).model_json_schema()

        # Output Schema
        if type(return_hint) is type(BaseModel) and issubclass(return_hint, BaseModel):
            # If the return hint is a Pydantic model, get the output schema from it
            output_schema = return_hint.model_json_schema()
        elif return_hint is str and not (description):
            # If the return hint is a string and the description is empty,
            # output_schema is None. This allows to express a plain language model
            # call as a language function.
            output_schema = None
        else:
            # Otherwise, create a wrapper model with the return hint as the output type
            output_schema = create_model(
                "OutputWrapper", output=(return_hint or NoneType, ...)
            ).model_json_schema()

        super().__init__(
            name=name,
            description=description,
            input_schema=input_schema,
            output_schema=output_schema,
        )

    def load(self, *args, **kwargs):
        """On load, reset the private variables."""
        super().load(*args, **kwargs)
        self._input_model = None
        self._output_model = None
        self._template = None

    def __call__(
        self,
        *args,
        examples: List = [],
        backend: Optional[LMBackend] = None,
        retry_policy: Optional[RetryPolicy] = None,
        event_manager: Optional[EventManager] = None,
        extra_args={},
        **kwargs,
    ) -> ReturnType:
        """
        Calls the language function to generate a response based on the given inputs.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            examples (List[Tuple[Any, ReturnType]], optional): A list of examples to provide to the language model. Defaults to [].
            backend (LMBackend, optional): The language model backend to use for the computation.
            retry_policy (RetryPolicy, optional): The retry policy to use for handling exceptions.
            event_manager (EventManager, optional): The event manager to use for handling callbacks.
            extra_args (Dict, optional): Additional arguments that may be used by the callback handlers. Defaults to {}.
        """
        tracer = trace.get_tracer(__name__)
        with tracer.start_span("lmfunc.__call__") as span:
            backend = backend or default.backend
            event_manager = event_manager or default.event_manager
            retry_policy = retry_policy or default.retry_policy

            # Call Start
            event_manager(
                "call_start",
                func=self,
                args=args,
                kwargs=kwargs,
                examples=examples,
                backend=backend,
                retry_policy=retry_policy,
                event_manager=event_manager,
                extra_args=extra_args,
                tracer=tracer,
                span=span,
            )

            # Assemble all input arguments into a single input object.
            # The object can only be a Pydantic model, a dictionary, a string, or None.
            input = None
            if args or kwargs:
                first_arg = args[0] if args else next(iter(kwargs.values()), None)
                if len(args) + len(kwargs) == 1 and (
                    isinstance(first_arg, BaseModel) or isinstance(first_arg, str)
                ):
                    # If the only argument is a Pydantic model or a string, use it as input
                    input = first_arg
                else:
                    # Otherwise, create a dictionary with the input arguments
                    input = {
                        **dict(zip(self.input_model.model_fields.keys(), args)),
                        **kwargs,
                    }
            try:
                for attempt in Retrying(
                    **retry_policy.args,
                    before_sleep=lambda x: event_manager("retry", retry_call_state=x),
                ):
                    with attempt:
                        # Render Input as a string
                        input_string = self.to_json_str(input)

                        # Render Examples as strings
                        examples_string = [
                            (self.to_json_str(i), self.to_json_str(o))
                            for i, o in examples
                        ]

                        # Render the Prompt
                        prompt = self.template.render(
                            inputs=input_string,
                            examples=examples_string,
                        )

                        # Language Model Prompt Template Render Callback
                        event_manager(
                            "prompt_render",
                            func=self,
                            args=args,
                            kwargs=kwargs,
                            examples=examples,
                            backend=backend,
                            retry_policy=retry_policy,
                            event_manager=event_manager,
                            extra_args=extra_args,
                            tracer=tracer,
                            span=span,
                            ##
                            input=input,
                            attempt=attempt,
                            input_string=input_string,
                            examples_string=examples_string,
                            prompt=prompt,
                        )

                        # Language Model Call
                        response = backend.complete(prompt, self.output_schema)

                        # Parse the response
                        parsed_completion = response(
                            self.output_schema,
                            new_token_or_char_callback=lambda **kwargs: event_manager(
                                "token_or_char", span=span, **kwargs
                            ),
                        )
                        completion = response.text
                        if isinstance(parsed_completion, dict):
                            # If the completion is a dictionary, build a Pydantic object
                            output = self.output_model(**parsed_completion)
                            if self.output_model.__name__ == "OutputWrapper":
                                # If the object is a wrapper, unwrap it
                                output = getattr(
                                    output, next(iter(output.model_fields.keys()))
                                )
                        else:
                            # Otherwise, the output is the parsed completion
                            output = parsed_completion

                        # Success Callback
                        event_manager(
                            "success",
                            func=self,
                            args=args,
                            kwargs=kwargs,
                            examples=examples,
                            backend=backend,
                            retry_policy=retry_policy,
                            event_manager=event_manager,
                            extra_args=extra_args,
                            tracer=tracer,
                            span=span,
                            input=input,
                            attempt=attempt,
                            input_string=input_string,
                            examples_string=examples_string,
                            prompt=prompt,
                            ##
                            response=response,
                            parsed_completion=parsed_completion,
                            completion=completion,
                            output=output,
                        )

            except Exception as exception:
                # Exception Callback
                event_manager("exception", exception=exception, vars=locals())

            return output

    def async_handler(self):
        """
        Returns an async route handler for the language function that can be used with
        FastAPI.

        Returns:
            Callable: A FastAPI route handler for the language function.
        """

        async def handler(input=None):
            return self(**input) if isinstance(input, dict) else self(input)

        handler.__annotations__["input"] = self.input_model
        if self.output_model.__name__ == "OutputWrapper":
            handler.__annotations__["return"] = next(
                iter(self.output_model.model_fields.values())
            ).annotation
        else:
            handler.__annotations__["return"] = self.output_model
        return handler

    def fastapi_app(self, fast_api_params: Dict = {}):
        """
        Creates a FastAPI application with the specified parameters and registers
        a POST route for the current instance.

        Args:
            fast_api_params (Dict, optional): Additional parameters to be passed to the
            FastAPI application. Defaults to {}.

        Returns:
            FastAPI: The created FastAPI application.
        """
        lazy_import("fastapi")
        import fastapi

        app = fastapi.FastAPI(**fast_api_params)
        app.post(
            f"/{self.name}",
            name=self.name,
            description=self.description,
        )(self.async_handler())

        return app

    def serve(self, fast_api_params={}, uvicorn_params={}):
        """
        Serves the lmfunc using FastAPI and Uvicorn.

        Args:
            fast_api_params (dict): Parameters to be passed to the FastAPI application.
            uvicorn_params (dict): Parameters to be passed to the Uvicorn server.

        Returns:
            None
        """

        def start_uvicorn_server(app, **kwargs):
            lazy_import("uvicorn")
            import uvicorn

            return uvicorn.run(app, **kwargs)

        app = self.fastapi_app(fast_api_params)
        if app:
            return start_uvicorn_server(app, **uvicorn_params)


def lmdef(func: Callable[InputArgs, ReturnType]):
    """
    Decorator that creates an LMFunc instance from a regular Python function.
    """
    return LMFunc[InputArgs, ReturnType](func)
