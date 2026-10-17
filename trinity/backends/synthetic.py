"""A small deterministic numpy language model for offline runs and tests.

Its hidden state is an exponentially discounted sum of token embeddings, so it
has the same fading-memory structure the paper attributes to transformers,
and it needs no download and no torch.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .base import Backend, ModelInfo, as_seqs, force_token


class SyntheticBackend(Backend):
    def __init__(self, vocab_size: int = 96, hidden_dim: int = 32, decay: float = 0.7,
                 temperature: float = 1.5, seed: int = 0):
        rng = np.random.RandomState(seed)
        self.emb = rng.randn(vocab_size, hidden_dim)
        self.w_out = rng.randn(hidden_dim, vocab_size) * temperature / np.sqrt(hidden_dim)
        self.decay = decay
        self.info = ModelInfo(name=f"synthetic-{vocab_size}", vocab_size=vocab_size,
                              eos_id=vocab_size - 1, hidden_dim=hidden_dim, n_layers=1)

    # Printable ASCII maps onto ids 0..vocab-2; the last id is EOS.
    def encode(self, text: str) -> list[int]:
        return [(ord(c) - 32) % (self.info.vocab_size - 1) for c in text]

    def decode(self, ids: Sequence[int]) -> str:
        return "".join("<eos>" if i == self.info.eos_id else chr(32 + int(i)) for i in ids)

    def _hidden(self, seqs: np.ndarray) -> np.ndarray:
        weights = self.decay ** np.arange(seqs.shape[1] - 1, -1, -1)
        return np.einsum("l,nlh->nh", weights, self.emb[seqs])

    def _logp(self, seqs: np.ndarray) -> np.ndarray:
        logits = self._hidden(seqs) @ self.w_out
        logits -= logits.max(axis=1, keepdims=True)
        return logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))

    def topk_next(self, seqs, k, force_id=None):
        logp = self._logp(as_seqs(seqs))
        ids = np.argsort(-logp, axis=1, kind="stable")[:, :k].astype(np.int64)
        vals = np.take_along_axis(logp, ids, axis=1)
        return force_token(ids, vals, lambda rows: logp[rows, force_id], force_id)

    def tv_at_positions(self, base, perturbed, positions):
        base = np.asarray(base, dtype=np.int64)
        perturbed = as_seqs(perturbed)
        positions = np.asarray(positions, dtype=np.int64)
        out = np.empty(len(positions))
        for t in np.unique(positions):
            rows = np.flatnonzero(positions == t)
            p = np.exp(self._logp(perturbed[rows, : t + 1]))
            q = np.exp(self._logp(base[None, : t + 1]))
            out[rows] = 0.5 * np.abs(p - q).sum(axis=1)
        return out

    def hidden_states(self, seqs, layers=None):
        return self._hidden(as_seqs(seqs))[:, None, :].astype(np.float32)
