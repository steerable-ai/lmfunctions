# conftest.py

import pytest


# This fixture will be used by all tests in the test suite
# to mock the input() function and always return "Y" when
# the prompt contains "Do you want to run". This happens
# when the user is asked if they want to install a missing
# package.
@pytest.fixture(autouse=True)
def mock_input_y_for_specific_prompt(monkeypatch):
    def mock_input(prompt):
        if "Do you want to run" in prompt:
            return "Y"

    monkeypatch.setattr("builtins.input", mock_input)
