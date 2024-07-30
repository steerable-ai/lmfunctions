import importlib.util
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Type

from datamodel_code_generator import DataModelType, InputFileType, generate
from pydantic import BaseModel


def from_jsonschema(schema: Dict) -> Type[BaseModel]:
    """Generate a Pydantic Model from a json schema.

    Args:
    schema: Source json schema to create Pydantic model from

    Returns:
    The newly created and loaded Pydantic class
    """
    # Ref: https://github.com/koxudaxi/datamodel-code-generator/issues/278
    class_name = schema.get("title", "Model")
    json_schema = json.dumps(schema)
    with TemporaryDirectory() as temporary_directory_name:
        temporary_directory = Path(temporary_directory_name)
        temporary_file_path = Path(temporary_directory / "tempmodel.py")
        generate(
            json_schema,
            input_file_type=InputFileType.JsonSchema,
            class_name=class_name,
            output=temporary_file_path,
            output_model_type=DataModelType.PydanticV2BaseModel,
        )
        spec = importlib.util.spec_from_file_location(
            "models", str(temporary_file_path)
        )
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return getattr(module, class_name)
        raise ImportError("Failed to import generated model")  # pragma: no cover
