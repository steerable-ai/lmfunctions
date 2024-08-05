from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

import lmfunctions as lmf
from lmfunctions.eventmanager import EventManager
from lmfunctions.managers import (
    consoleRich,
    fileLog,
    panelPrint,
    timeEvents,
    tokenStream,
)

from .test_backends import TEST_CHAT_BACKEND
from .test_lmfuncs import test_functions

route, args, kwargs = test_functions["route"]


def test_eventmanager(mocker):
    lmf.default.backend = TEST_CHAT_BACKEND
    lmf.default.event_manager = panelPrint()
    route(*args, **kwargs)
    lmf.default.event_manager = consoleRich()
    route(*args, **kwargs)
    lmf.default.event_manager += timeEvents()
    route(*args, **kwargs)

    with TemporaryDirectory(ignore_cleanup_errors=True) as temporary_directory_name:
        temporary_directory = Path(temporary_directory_name)
        temporary_file_path = Path(temporary_directory / "logs.log")
        lmf.default.event_manager = fileLog(log_file=str(temporary_file_path))
        route(*args, **kwargs)

    lmf.default.event_manager = EventManager() + tokenStream()
    mocker.patch(
        "lmfunctions.backends.llamacpp.LlamaCppBackend.complete",
        return_value=lmf.LMResponse("{{"),
    )
    with pytest.raises(ValueError):
        route(*args, **kwargs)
    lmf.default.event_manager = EventManager()
