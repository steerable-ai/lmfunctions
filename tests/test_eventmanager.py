from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

import lmfunctions as lmf

from .test_backends import TEST_CHAT_BACKEND
from .test_lmfuncs import test_functions

route, args, kwargs = test_functions["route"]


def test_eventmanager():
    lmf.default.backend = TEST_CHAT_BACKEND
    lmf.set_event_manager.panelprint()
    route(*args, **kwargs)
    lmf.set_event_manager.consolerich()
    route(*args, **kwargs)
    lmf.set_event_manager.timeevents()
    route(*args, **kwargs)

    with TemporaryDirectory(ignore_cleanup_errors=True) as temporary_directory_name:
        temporary_directory = Path(temporary_directory_name)
        temporary_file_path = Path(temporary_directory / "logs.log")
        lmf.set_event_manager.filelogger(log_file=str(temporary_file_path))
        route(*args, **kwargs)

    lmf.set_event_manager.default()
    lmf.set_event_manager.tokenstream()
    lmf.default.event_manager += lmf.eventmanager.EventManager()
    with pytest.raises(NotImplementedError):
        lmf.default.event_manager += "not a manager"
