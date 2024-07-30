from typing import Optional

from opentelemetry import trace
from opentelemetry.sdk.trace import SpanLimits, TracerProvider
from pandas import DataFrame
from rich import print

from lmfunctions.eventmanager import EventManager
from lmfunctions.handler import ExceptionRaiseHandler, OtelEventHandler


def time_diff(df, event_type1, event_type2):
    return (
        df.loc[event_type1]["timestamp"].min() - df.loc[event_type2]["timestamp"].max()
    ) / 10**9


def print_stats(**kwargs):
    prompt = kwargs["prompt"]
    completion = kwargs["completion"]
    span = kwargs["span"]
    events = DataFrame.from_records(
        [
            {"name": e.name, "timestamp": e.timestamp, "attributes": e.attributes}
            for e in span.events
        ],
        index="name",
    )
    prompt_length = len(prompt)
    completion_length = len(completion)
    time_to_prompt_render = time_diff(events, "prompt_render", "call_start")
    time_to_first_token = time_diff(events, "token_or_char", "call_start")
    time_to_first_token_lm_only = time_diff(events, "token_or_char", "prompt_render")
    lm_call_time = time_diff(events, "success", "prompt_render")
    total_time = time_diff(events, "success", "call_start")
    stats = {
        "Prompt Characters": prompt_length,
        "Completion Characters": completion_length,
        "Time to Prompt Render": time_to_prompt_render,
        "Time to First Token": time_to_first_token,
        "Time to First Token (LM Only)": time_to_first_token_lm_only,
        "LM Call Time": lm_call_time,
        "Total Time": total_time,
        "Completion Characters per Second": completion_length / total_time,
        "Completion Characters / LM Call Time": completion_length / lm_call_time,
        "Prompt Characters / LM Call Time": prompt_length / lm_call_time,
    }
    print(stats)


def timeEvents(tracer_provider: Optional[TracerProvider] = None) -> EventManager:
    """ """

    if tracer_provider is None:
        tracer_provider = trace.get_tracer_provider()
        if not isinstance(tracer_provider, TracerProvider):
            # Set the tracer provider with custom limits
            tracer_provider = TracerProvider(span_limits=SpanLimits(max_events=1000000))
            trace.set_tracer_provider(tracer_provider)

    return EventManager(
        handlers={
            "call_start": [OtelEventHandler(event_name="call_start")],
            "prompt_render": [OtelEventHandler(event_name="prompt_render")],
            "token_or_char": [
                OtelEventHandler(
                    event_name="token_or_char",
                    attributes_vars={"token_or_char": "token_or_char"},
                )
            ],
            "success": [OtelEventHandler(event_name="success"), print_stats],
            "exception": [
                OtelEventHandler(event_name="exception"),
                ExceptionRaiseHandler(varname="exception"),
            ],
        }
    )
