import json
import os
from typing import Any, Dict

import fsspec
import yaml


def dumps(data, format="yaml"):
    """
    Serialize the given data into a string representation.

    Args:
        data: The data to be serialized.
        format (str): The format in which to serialize the data.
        Supported formats are "yaml" and "json".

    Returns:
        str: The serialized data as a string.

    Raises:
        ValueError: If the specified format is not supported.
    """
    if format == "yaml":
        return yaml.safe_dump(data, default_flow_style=False)
    elif format == "json":
        return json.dumps(data)
    else:
        raise ValueError("Unsupported format")


def loads(data, format="yaml") -> Dict[str, Any]:
    """
    Load a dictionary from a string representation using the specified format.

    Args:
        data (str): The string representation of the data.
        format (str, optional): The format of the data. Defaults to "yaml".

    Returns:
        object: The loaded data.

    Raises:
        ValueError: If the specified format is unsupported.
    """
    if format == "yaml":
        return yaml.safe_load(data)
    elif format == "json":
        return json.loads(data)
    else:
        raise ValueError("Unsupported format")


def loadf(url: str, **kwargs) -> Dict[str, Any]:
    """
    Load a file from the given generalized URL and return its contents.
    Determines the format of the file from its extension.

    Parameters:
    url (str): The generalized URL of the file to load.
    **kwargs: Additional keyword arguments to pass to the `fsspec.open` function.

    Returns:
    The contents of the file.

    """
    extension = os.path.splitext(url)[1][1:]
    with fsspec.open(url, **kwargs) as file:
        return loads(file, format=extension)
