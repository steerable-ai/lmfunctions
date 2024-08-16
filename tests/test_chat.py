import lmfunctions as lmf

from .test_backends import TEST_CHAT_BACKEND


def test_chat(mocker):
    lmf.default.backend = TEST_CHAT_BACKEND
    lmf.default.backend()
    inputs = [
        "1, 2, 3, 4, ",
        'Starting a multiline """ statement',
        'ending a multi-line""" statement',
        '""" Starting and ending """ on the same line',
        "\n",
        "",
        "/history",
        "/clear",
        "/exit",
    ]
    mocker.patch("builtins.input", side_effect=inputs)
    lmf.chat(system_message="You are a helpful assistant")
