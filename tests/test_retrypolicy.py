import lmfunctions as lmf

from .test_backends import TEST_CHAT_BACKEND
from .test_lmfuncs import test_functions


def test_retry_policy():
    lmf.default.backend = TEST_CHAT_BACKEND
    route, args, kwargs = test_functions["route"]
    lmf.default.retry_policy.stop = lmf.retrypolicy.StopType.after_delay
    route(*args, **kwargs)
    for waittype in lmf.retrypolicy.WaitType:
        lmf.default.retry_policy.wait = waittype
        route(*args, **kwargs)
