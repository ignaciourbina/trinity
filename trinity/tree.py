"""Batched soft dynamic programming over the top-K lookahead tree of a language model.

The tree is expanded breadth-first: every node at depth d is a sequence of the
same length, so a whole level costs one batched backend call (instead of one
forward pass per node). Z is then computed bottom-up in log space with numpy.

Node semantics, matching the paper's free-boundary formulation:
* each node expands its top-K next tokens (EOS always included);
* EOS terminates with value exp(R / beta);
* at the horizon (d == depth) every candidate terminates, i.e. a forced stop.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from . import identities
from .backends.base import Backend
from .rewards import Reward


@dataclass
class Level:
    seqs: np.ndarray          # (N, L0 + d) full token sequences
    cont_tokens: np.ndarray   # (N, d) continuation tokens after the prompt
    cont_logp: np.ndarray     # (N, d) their log pi_ref
    cand_ids: np.ndarray      # (N, K) candidate next tokens
    cand_logp: np.ndarray     # (N, K) log pi_ref of candidates
    child: np.ndarray         # (N, K) index of the child node in the next level, -1 if terminal
    log_terminal: np.ndarray  # (N, K) R / beta for terminal candidates, nan otherwise
    log_child: np.ndarray = field(default=None)  # (N, K) log Z(child) or R / beta
    log_z: np.ndarray = field(default=None)      # (N,)

    @property
    def terminal(self) -> np.ndarray:
        return self.child < 0


@dataclass
class SoftTree:
    levels: list[Level]
    beta: float
    eos_id: int
    build_seconds: float

    @property
    def log_z(self) -> float:
        return float(self.levels[0].log_z[0])

    @property
    def n_nodes(self) -> int:
        return sum(len(lv.seqs) for lv in self.levels)

    def _stack(self):
        lp = np.concatenate([lv.cand_logp for lv in self.levels])
        lz = np.concatenate([lv.log_z for lv in self.levels])
        lc = np.concatenate([lv.log_child for lv in self.levels])
        return lp, lz, lc

    def root_gammas(self) -> dict[str, np.ndarray]:
        root = self.levels[0]
        g = identities.gamma_three_ways(root.cand_logp[0], root.log_z[0], root.log_child[0], self.beta)
        g.update(token_id=root.cand_ids[0], p_ref=np.exp(root.cand_logp[0]),
                 is_eos=root.cand_ids[0] == self.eos_id,
                 z_child=np.exp(root.log_child[0]))
        return g

    def checks(self) -> dict[str, float]:
        """Identity checks over *every* node in the tree, not only the root."""
        lp, lz, lc = self._stack()
        g = identities.gamma_three_ways(lp, lz, lc, self.beta)
        return {
            "gamma_error": float(g["error"].max()),
            "bellman_residual": float(identities.bellman_residual(lp, lz, lc).max()),
            "pi_star_normalization": float(identities.pi_star_normalization(lp, lz, lc).max()),
        }

    def sample(self, n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        """Sample n trajectories from pi* and check the telescoping identity.

        prod_t Gamma(h_t, x_t) should equal exp(R / beta) / Z(h_0).
        """
        depth = len(self.levels)
        node = np.zeros(n, dtype=np.int64)
        alive = np.ones(n, dtype=bool)
        gamma_prod = np.ones(n)
        expected = np.full(n, np.nan)
        tokens = np.full((n, depth), -1, dtype=np.int64)
        z0 = np.exp(self.log_z)

        for d, lv in enumerate(self.levels):
            idx = np.flatnonzero(alive)
            if idx.size == 0:
                break
            nd = node[idx]
            logits = lv.cand_logp[nd] + lv.log_child[nd] - lv.log_z[nd, None]
            probs = np.exp(logits)
            probs /= probs.sum(axis=1, keepdims=True)
            choice = (rng.random(len(idx))[:, None] > np.cumsum(probs, axis=1)).sum(axis=1)
            choice = np.minimum(choice, probs.shape[1] - 1)

            log_child = lv.log_child[nd, choice]
            gamma_prod[idx] *= np.exp(log_child) / np.exp(lv.log_z[nd])
            tokens[idx, d] = lv.cand_ids[nd, choice]

            stop = lv.child[nd, choice] < 0
            expected[idx[stop]] = np.exp(lv.log_terminal[nd[stop], choice[stop]]) / z0
            alive[idx[stop]] = False
            node[idx[~stop]] = lv.child[nd[~stop], choice[~stop]]

        rel_error = np.abs(gamma_prod - expected) / np.maximum(np.abs(expected), 1e-10)
        return {"tokens": tokens, "gamma_product": gamma_prod, "expected": expected,
                "rel_error": rel_error}


def build_tree(backend: Backend, prompt_ids, *, top_k: int, depth: int, beta: float,
               reward: Reward) -> SoftTree:
    eos = backend.info.eos_id
    t0 = time.perf_counter()

    seqs = np.asarray([prompt_ids], dtype=np.int64)
    cont_tokens = np.zeros((1, 0), dtype=np.int64)
    cont_logp = np.zeros((1, 0))
    levels: list[Level] = []

    # Forward pass: expand one level per batched backend call.
    for d in range(depth + 1):
        ids, logp = backend.topk_next(seqs, top_k, force_id=eos)
        terminal = (ids == eos) | (d == depth)

        log_terminal = np.full(ids.shape, np.nan)
        r, c = np.nonzero(terminal)
        if r.size:
            toks = np.concatenate([cont_tokens[r], ids[r, c, None]], axis=1)
            lps = np.concatenate([cont_logp[r], logp[r, c, None]], axis=1)
            log_terminal[r, c] = reward(toks, lps, eos) / beta

        child = np.full(ids.shape, -1, dtype=np.int64)
        r, c = np.nonzero(~terminal)
        child[r, c] = np.arange(r.size)
        levels.append(Level(seqs, cont_tokens, cont_logp, ids, logp, child, log_terminal))
        if r.size == 0:
            break
        seqs = np.concatenate([seqs[r], ids[r, c, None]], axis=1)
        cont_tokens = np.concatenate([cont_tokens[r], ids[r, c, None]], axis=1)
        cont_logp = np.concatenate([cont_logp[r], logp[r, c, None]], axis=1)

    # Backward pass: soft Bellman recursion, bottom-up.
    next_log_z = None
    for lv in reversed(levels):
        if next_log_z is None:
            lv.log_child = lv.log_terminal
        else:
            lv.log_child = np.where(lv.terminal, lv.log_terminal, next_log_z[np.maximum(lv.child, 0)])
        lv.log_z = identities.soft_bellman(lv.cand_logp, lv.log_child)
        next_log_z = lv.log_z

    return SoftTree(levels, beta, eos, time.perf_counter() - t0)


def tree_size(top_k: int, depth: int) -> int:
    """Upper bound on the node count (each node has at most K - 1 non-EOS children)."""
    b = top_k - 1
    return sum(b ** d for d in range(depth + 1))
