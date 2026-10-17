"""The interface every language-model backend implements.

The experiments only ever need three batched primitives, which keeps backends
small and lets the same code drive a local GPU, a CPU, or a remote Colab runtime:

* ``topk_next``        top-K next-token log-probs for a batch of equal-length sequences
* ``tv_at_positions``  TV distance between next-token distributions of perturbed and base text
* ``hidden_states``    last-position hidden states per layer (for probing experiments)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ModelInfo:
    name: str
    vocab_size: int
    eos_id: int
    hidden_dim: int
    n_layers: int
    device: str = "cpu"

    def to_dict(self) -> dict:
        return asdict(self)


class Backend(ABC):
    info: ModelInfo

    @abstractmethod
    def encode(self, text: str) -> list[int]: ...

    @abstractmethod
    def decode(self, ids: Sequence[int]) -> str: ...

    @abstractmethod
    def topk_next(self, seqs: np.ndarray, k: int, force_id: int | None = None
                  ) -> tuple[np.ndarray, np.ndarray]:
        """Top-k of log pi_ref(. | seq) for each row of ``seqs`` (N, L).

        If ``force_id`` is not among a row's top-k it replaces the k-th entry.
        Returns ``(ids (N, k) int64, logp (N, k) float64)``.
        """

    @abstractmethod
    def tv_at_positions(self, base: np.ndarray, perturbed: np.ndarray,
                        positions: np.ndarray) -> np.ndarray:
        """TV(p(. | perturbed[i, :t+1]), p(. | base[:t+1])) with t = positions[i]. Returns (P,)."""

    @abstractmethod
    def hidden_states(self, seqs: np.ndarray, layers: Sequence[int] | None = None) -> np.ndarray:
        """Last-position hidden states, shape (N, n_selected_layers, hidden_dim), float32."""

    def close(self) -> None:
        pass


def as_seqs(seqs) -> np.ndarray:
    arr = np.asarray(seqs, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] == 0:
        raise ValueError(f"expected a non-empty (N, L) token array, got shape {arr.shape}")
    return arr


def force_token(ids: np.ndarray, logp: np.ndarray, full_logp_of, force_id: int | None):
    """Shared top-k post-processing: make sure ``force_id`` is a candidate in every row.

    ``full_logp_of(rows)`` returns log p(force_id) for the given row indices.
    """
    if force_id is None:
        return ids, logp
    missing = np.flatnonzero(~(ids == force_id).any(axis=1))
    if missing.size:
        ids[missing, -1] = force_id
        logp[missing, -1] = full_logp_of(missing)
    return ids, logp
