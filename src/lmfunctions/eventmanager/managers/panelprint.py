from lmfunctions.eventmanager import EventManager
from lmfunctions.handler import PanelPrintHandler, PrintHandler


def panelPrint() -> EventManager:
    """
    EventManager preset that prints selected internal variables to the console
    using panels.
    """
    return EventManager(
        handlers={
            "prompt_render": [
                PanelPrintHandler(variables=[dict(name="prompt", title="Prompt")])
            ],
            "token_or_char": [
                PrintHandler(varnames=["token_or_char"], end="", flush=True)
            ],
            "retry": [
                PanelPrintHandler(
                    variables=[dict(name="retry_call_state", title="Retry")]
                )
            ],
            "exception": [
                PanelPrintHandler(variables=[dict(name="exception", title="Exception")])
            ],
            "success": [
                PanelPrintHandler(
                    variables=[
                        dict(name="completion", title="Completion"),
                        dict(name="parsed_completion", title="Parsed Completion"),
                        dict(name="output", title="Output"),
                    ]
                )
            ],
        }
    )
