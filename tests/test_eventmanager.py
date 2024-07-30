from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

import lmfunctions as lmf
from lmfunctions.eventmanager import EventManager
from lmfunctions.eventmanager.managers import timeEvents

from .test_lmbackend import TEST_CHAT_BACKEND
from .test_lmfuncs import test_functions

route, args, kwargs = test_functions["route"]


def test_eventmanager(mocker):
    lmf.default.backend = TEST_CHAT_BACKEND
    lmf.set_event_manager.panelprint()
    route(*args, **kwargs)
    lmf.set_event_manager.consolerich()
    route(*args, **kwargs)
    lmf.default.event_manager += timeEvents()
    route(*args, **kwargs)
    lmf.set_event_manager.time_events()

    with TemporaryDirectory(ignore_cleanup_errors=True) as temporary_directory_name:
        temporary_directory = Path(temporary_directory_name)
        temporary_file_path = Path(temporary_directory / "logs.log")
        lmf.set_event_manager.filelogger(log_file=str(temporary_file_path))
        route(*args, **kwargs)

    lmf.set_event_manager.tokenstream()
    lmf.default.event_manager += EventManager()
    mocker.patch(
        "lmfunctions.lmbackend.backends.llamacpp.LlamaCppBackend.complete",
        return_value=lmf.LMResponse("{{"),
    )
    with pytest.raises(ValueError):
        route(*args, **kwargs)
    lmf.set_event_manager.default()
