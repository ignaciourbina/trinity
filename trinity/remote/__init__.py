"""Run the model on one machine (e.g. a Colab GPU) and the experiments on another."""

from .client import RemoteBackend, RemoteError
from .server import TrinityServer

__all__ = ["RemoteBackend", "RemoteError", "TrinityServer"]
