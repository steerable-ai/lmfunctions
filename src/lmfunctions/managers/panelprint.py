from lmfunctions.eventmanager import EventManager
from lmfunctions.handlers import PanelPrintHandler, PrintHandler

# EventManager preset that prints selected internal variables to the console
# using panels.

panelPrint = EventManager(
    handlers={
        "input_render": [
            PanelPrintHandler(
                variables=[dict(name="backend_input", title="Language Model Input")]
            )
        ],
        "token_or_char": [PrintHandler(varnames=["token_or_char"], end="", flush=True)],
        "retry": [
            PanelPrintHandler(variables=[dict(name="retry_call_state", title="Retry")])
        ],
        "exception": [
            PanelPrintHandler(variables=[dict(name="exception", title="Exception")])
        ],
        "success": [
            PanelPrintHandler(
                variables=[
                    dict(name="completion", title="Completion"),
                    dict(name="output", title="Output"),
                ]
            )
        ],
    }
)
