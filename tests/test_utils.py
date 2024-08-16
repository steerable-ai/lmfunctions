import pytest

from lmfunctions.utils import lazy_import, model_from_schema

from .models import test_models


def test_lazy_import(mocker):
    mocker.patch("builtins.input", side_effect=["n"])
    with pytest.raises(ImportError):
        lazy_import("unexistent_package")
    mocker.patch("builtins.input", side_effect=["y"])
    mocker.patch("subprocess.run", return_value=None)
    with pytest.raises(ModuleNotFoundError):
        lazy_import("unexistent_package")
    assert lazy_import("numpy") is not None


def test_from_jsonschema():
    for model in test_models:
        schema = model.model_json_schema()
        newmodel = model_from_schema(schema)
        assert newmodel.model_json_schema()

    # Test that the model can be created and dumped to json
    model = model_from_schema(test_models[0].model_json_schema())
    assert model(
        country="USA", population=1, languages_spoken=["English"]
    ).model_dump_json()
