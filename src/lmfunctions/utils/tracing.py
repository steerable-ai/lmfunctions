from opentelemetry import trace
from opentelemetry.sdk.trace import SpanLimits, TracerProvider

MAX_EVENTS = 1000000


def get_or_create_tracer_provider() -> TracerProvider:
    provider = trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        # Set the tracer provider with custom limits
        provider = TracerProvider(span_limits=SpanLimits(max_events=MAX_EVENTS))
        trace.set_tracer_provider(provider)
    return provider
