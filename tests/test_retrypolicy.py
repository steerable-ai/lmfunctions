import lmfunctions as lmf
from lmfunctions.retrypolicy import StopType, WaitType

from .test_backends import TEST_CHAT_BACKEND
from .test_lmfuncs import test_functions


def test_retry_policy():
    lmf.default.backend = TEST_CHAT_BACKEND
    route, args, kwargs = test_functions["route"]
    lmf.default.retry_policy.stop = StopType.after_delay
    route(*args, **kwargs)
    for waittype in WaitType:
        lmf.default.retry_policy.wait = waittype
        route(*args, **kwargs)
