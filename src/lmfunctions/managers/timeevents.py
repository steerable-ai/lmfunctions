from pandas import DataFrame
from rich import print

from lmfunctions.eventmanager import EventManager
from lmfunctions.handlers import OtelEventHandler
from lmfunctions.utils.tracing import get_or_create_tracer_provider


def time_diff(df, event_type1, event_type2):
    return (
        float(
            df.loc[event_type1]["timestamp"].min()
            - df.loc[event_type2]["timestamp"].max()
        )
        / 10**9
    )


def print_stats(completion, span, **kwargs):
    events = DataFrame.from_records(
        [
            {"name": e.name, "timestamp": e.timestamp, "attributes": e.attributes}
            for e in span.events
        ],
        index="name",
    )
    completion_length = len(completion)
    time_to_input_render = time_diff(events, "input_render", "call_start")
    time_to_first_token = time_diff(events, "token_or_char", "call_start")
    time_to_first_token_lm_only = time_diff(events, "token_or_char", "input_render")
    lm_call_time = time_diff(events, "success", "input_render")
    total_time = time_diff(events, "success", "call_start")
    stats = {
        "Completion Characters": completion_length,
        "Time to Input Render": time_to_input_render,
        "Time to First Token": time_to_first_token,
        "Time to First Token (LM Only)": time_to_first_token_lm_only,
        "LM Call Time": lm_call_time,
        "Total Time": total_time,
        "Completion Characters per Second": completion_length / total_time,
        "Completion Characters / LM Call Time": completion_length / lm_call_time,
    }
    print(stats)


def timeEvents() -> EventManager:

    get_or_create_tracer_provider()

    return EventManager(
        handlers={
            "call_start": [OtelEventHandler(event_name="call_start")],
            "input_render": [OtelEventHandler(event_name="input_render")],
            "token_or_char": [
                OtelEventHandler(
                    event_name="token_or_char",
                    attributes_vars={"token_or_char": "token_or_char"},
                )
            ],
            "success": [OtelEventHandler(event_name="success"), print_stats],
            "exception": [
                OtelEventHandler(event_name="exception"),
            ],
        }
    )
