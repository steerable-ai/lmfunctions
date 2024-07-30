from enum import Enum

import yaml
from tenacity import (
    stop_after_attempt,
    stop_after_delay,
    wait_exponential,
    wait_fixed,
    wait_none,
    wait_random,
)

from lmfunctions.base import Base


class StopType(str, Enum):
    never = "never"
    after_attempt = "after_attempt"
    after_delay = "after_delay"


class WaitType(str, Enum):
    none = "none"
    fixed = "fixed"
    random = "random"
    exponential = "exponential"


class RetryPolicy(Base):
    """
    A retry policy in a serializable format. It generates retry arguments
    for the tenacity library. See https://tenacity.readthedocs.io/en/latest/.
    """

    stop: StopType = StopType.after_attempt
    wait: WaitType = WaitType.none
    stop_max_attempt: int = 3
    stop_max_delay: int = 1
    wait_fixed: int = 1
    wait_random_min: int = 1
    wait_random_max: int = 2
    wait_exponential_min: int = 1
    wait_exponential_max: int = 2
    wait_exponential_multiplier: float = 2.0
    reraise: bool = True

    @property
    def args(self) -> dict:
        """
        Get the retry arguments based on the configured retry policy.

        Returns:
            dict: The retry arguments to be used for retrying.
        """
        retry_args = {}
        if self.stop == StopType.after_attempt:
            retry_args["stop"] = stop_after_attempt(self.stop_max_attempt)
        elif self.stop == StopType.after_delay:
            retry_args["stop"] = stop_after_delay(self.stop_max_delay)
        if self.wait == WaitType.none:
            retry_args["wait"] = wait_none()
        elif self.wait == WaitType.fixed:
            retry_args["wait"] = wait_fixed(self.wait_fixed)
        elif self.wait == WaitType.random:
            retry_args["wait"] = wait_random(self.wait_random_min, self.wait_random_max)
        elif self.wait == WaitType.exponential:
            retry_args["wait"] = wait_exponential(
                min=self.wait_exponential_min,
                max=self.wait_exponential_max,
                multiplier=self.wait_exponential_multiplier,
            )
        retry_args["reraise"] = self.reraise
        return retry_args


# Workaround for issue with YAML serialization of enums
# See https://github.com/yaml/pyyaml/issues/722
for enumtype in [StopType, WaitType]:
    yaml.SafeDumper.add_representer(
        enumtype,  # type: ignore
        yaml.representer.SafeRepresenter.represent_str,
    )
