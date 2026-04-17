import ssl
import urllib.request
from contextlib import AbstractContextManager
from functools import partial
from typing import Any

import certifi


class EnforceCertificates(AbstractContextManager):
    """Context manager to enforce SSL certificates."""

    def __enter__(self) -> None:
        """Enter the context."""
        self.urlopen = urllib.request.urlopen  # noqa: S310
        urllib.request.urlopen = partial(
            urllib.request.urlopen,  # noqa: S310
            context=ssl.create_default_context(cafile=certifi.where()),
        )

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:  # noqa: ANN401
        """Exit the context. Restore default functionality."""
        urllib.request.urlopen = self.urlopen
        return exc_type is None
