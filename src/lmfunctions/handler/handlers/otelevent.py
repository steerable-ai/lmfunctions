from typing import Dict, Literal

from lmfunctions.handler import Handler


class OtelEventHandler(Handler):
    """Callback handler that sends events to an OpenTelemetry span."""

    name: Literal["otelevent"] = "otelevent"
    span_name: str = "span"
    event_name: str = "event"
    attributes_vars: Dict[str, str] = {}

    def __call__(self, **kwargs):
        span = kwargs.get(self.span_name, None)
        attributes = {
            key: kwargs.get(varname, None)
            for key, varname in self.attributes_vars.items()
        }
        if span:
            span.add_event(self.event_name, attributes=attributes)
