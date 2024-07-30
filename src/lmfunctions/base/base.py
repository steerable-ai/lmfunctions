from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict

from lmfunctions.objectstore import Hub, ObjectStore
from lmfunctions.utils import dumps, loadf, loads, panelprint

hub = Hub()


class Base(BaseModel):
    """
    A customized Pydantic model with convenience methods for serialization,
    deserialization, storage. Enforces validation of the model's data on
    assignment to fields.
    """

    model_config = ConfigDict(
        protected_namespaces=(), validate_assignment=True, arbitrary_types_allowed=True
    )

    def dump(self, **kwargs) -> Dict[str, Any]:
        """
        Returns a dump of the model.

        This method returns a dump of the current state of the model.
        It can be used for debugging or saving the model's state.

        Returns:
            str: A string representation of the model's dump.
        """
        return self.model_dump(**kwargs)

    def dumps(self, format: str = "yaml", **kwargs) -> str:
        """
        Serialize the data model object to a string representation.

        Args:
            format (str, optional): The serialization format. Defaults to "yaml".

        Returns:
            str: The serialized string representation of the data model object.
        """
        return dumps(self.dump(**kwargs), format=format)

    def push(self, path: str, objectstore: ObjectStore = hub, **kwargs) -> None:
        """
        Pushes the serialized object to an object store.

        Args:
        - path (str): The path to save the object to in the object store.
        """
        return objectstore.push(path, self.dumps(), **kwargs)

    def load(self, data: Optional[Dict] = None, **kwargs) -> None:
        """
        Loads data into the object, checking that it conforms to the model.

        Args:
            data (dict): The data to be loaded.

        Returns:
            None
        """
        new_data = self.dump() | (data or kwargs)
        self.__dict__.update(self.model_validate(new_data))

    def loads(self, data, format: str = "yaml") -> None:
        """
        Deserializes a string in the given format to an object.

        Args:
        - data (str): The string to deserialize.
        - format (str): The format to deserialize from.
        """
        self.load(loads(data, format=format))

    def loadf(self, url: str, **kwargs) -> None:
        """
        Loads the object from a generalized URL.

        The format is inferred from the file extension.

        Args:
        - url (str): The URL to load the configuration from.
        """
        self.load(loadf(url, **kwargs))

    def pull(self, path: str, objectstore: ObjectStore = hub, **kwargs) -> None:
        """
        Pulls the object from the object store.

        Args:
        - path (str): The path to the object in the object store.
        """
        self.load(objectstore.pull(path, **kwargs))

    @classmethod
    def from_string(cls, data, format: str = "yaml"):
        """
        Creates an instance from a string.

        Args:
        - data (str): The string to create the instance from.
        - format (str): The format of the string.

        Returns:
        - An instance created from the string.
        """
        return cls(**loads(data, format=format))

    @classmethod
    def from_file(cls, url: str, **kwargs):
        """
        Creates an instance by reading a file-like object from a generalized URL.

        Args:
        - url (str): The URL to create the instance from.

        Returns:
        - An instance created from the URL.
        """
        return cls(**loadf(url, **kwargs))

    @classmethod
    def from_store(cls, path: str, objectstore: ObjectStore = hub, **kwargs):
        """
        Creates an instance by pulling it from an object store.

        Args:
        - path (str): The path to the object in the object store.

        Returns:
        - An instance created from the object store.
        """
        return cls(**objectstore.pull(path, **kwargs))

    def info(self) -> None:
        """
        Prints information about the object.

        This method uses the `panel_print` function to print information about
        the object. The title of the printed information is set to the name of
        the class.
        """
        panelprint(self, title=self.__class__.__name__)
