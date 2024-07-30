import os

from lmfunctions.objectstore.objectstore import ObjectStore
from lmfunctions.utils import loadf

BASE_URL: str = os.environ.get("LMHUB_URL", "https://lmhub.steerable.co/lmhub/")


class Hub(ObjectStore):
    def push(self, obj, path, **kwargs):
        raise NotImplementedError(
            "Pushing objects is not supported yet."
        )  # pragma: no cover

    @staticmethod
    def pull(path, format="yaml", **kwargs):
        url = f"{BASE_URL}{path}.{format}"
        return loadf(url, **kwargs)
