import lmfunctions as lmf

from .test_lmbackend import TEST_CHAT_BACKEND


def test_chat(mocker):
    lmf.default.backend = TEST_CHAT_BACKEND
    inputs = ["1, 2, 3, 4, ", "\n", "", "/history", "/clear", "/exit"]
    mocker.patch("builtins.input", side_effect=inputs)
    lmf.chat()
