"""Language-model backends. Heavy dependencies (torch) are imported lazily."""

from __future__ import annotations

from .base import Backend, ModelInfo
from .synthetic import SyntheticBackend


def get_backend(kind: str = "hf", *, model: str = "gpt2", device: str = "auto",
                dtype: str = "float32", batch_tokens: int = 1 << 16, compile: bool = False,
                url: str | None = None, token: str | None = None) -> Backend:
    """Build a backend by name: ``hf`` (local HuggingFace), ``remote`` (a trinity server), ``synthetic``."""
    if kind == "hf":
        from .hf import HFBackend
        return HFBackend.from_pretrained(model, device=device, dtype=dtype,
                                         batch_tokens=batch_tokens, compile=compile)
    if kind == "remote":
        from ..remote.client import RemoteBackend
        if not url:
            raise ValueError("the remote backend needs a server URL")
        return RemoteBackend(url, token=token)
    if kind == "synthetic":
        return SyntheticBackend()
    raise ValueError(f"unknown backend {kind!r}; choose hf, remote or synthetic")


__all__ = ["Backend", "ModelInfo", "SyntheticBackend", "get_backend"]
