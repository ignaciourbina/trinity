"""The Trinity on a small discrete model with exact, fully vectorised DP.

States s, tokens v (last one is EOS), deterministic transitions F(s, v),
softmax reference policy pi_ref(v | s), stopping reward R(s, t) = r_s - c * t.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import identities
from ..config import ToyConfig


@dataclass
class ToyModel:
    cfg: ToyConfig
    F: np.ndarray        # (S, V - 1) next state for each non-EOS token
    pi_ref: np.ndarray   # (S, V)

    @classmethod
    def build(cls, cfg: ToyConfig) -> "ToyModel":
        # Same draw order as the original script, so the model is identical for a given seed.
        rng = np.random.RandomState(cfg.seed)
        S, V = cfg.n_states, cfg.n_tokens
        F = rng.randint(0, S, size=(S, V - 1))
        logits = rng.randn(S, V) * 1.5
        logits[:, -1] = rng.randn(S) * 0.5 - 0.5
        logits -= logits.max(axis=1, keepdims=True)
        pi_ref = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        return cls(cfg, F, pi_ref)

    @property
    def eos(self) -> int:
        return self.cfg.n_tokens - 1

    def r_stop(self, t: int) -> np.ndarray:
        """(S,) reward for stopping in each state after t steps."""
        return np.asarray(self.cfg.state_rewards) - self.cfg.step_cost * t

    def solve(self) -> dict[str, np.ndarray]:
        """Backward DP. Returns log Z (S, T+1) and log_child (T, S, V) for every decision."""
        cfg = self.cfg
        S, T, beta = cfg.n_states, cfg.t_max, cfg.beta
        log_p = np.log(self.pi_ref)
        log_z = np.empty((S, T + 1))
        log_z[:, T] = self.r_stop(T) / beta
        log_child = np.empty((T, S, cfg.n_tokens))
        for t in range(T - 1, -1, -1):
            log_child[t, :, :-1] = log_z[self.F, t + 1]
            log_child[t, :, -1] = self.r_stop(t) / beta
            log_z[:, t] = identities.soft_bellman(log_p, log_child[t])
        return {"log_z": log_z, "log_child": log_child, "log_p": log_p}


@dataclass
class ToyResults:
    model: ToyModel
    log_z: np.ndarray
    gammas: dict[str, np.ndarray]     # each (T, S, V)
    bellman: np.ndarray               # (T, S)
    normalization: np.ndarray         # (T, S)
    telescope_errors: np.ndarray      # (n_trajectories,)
    pi_star: np.ndarray               # (T, S, V)
    influence_lags: np.ndarray
    influence_tv: np.ndarray
    decay_rate: float

    @property
    def v_star(self) -> np.ndarray:
        return self.model.cfg.beta * self.log_z

    def summary(self) -> dict:
        return {
            "gamma_max_error": float(self.gammas["error"].max()),
            "bellman_max_residual": float(self.bellman.max()),
            "pi_star_max_normalization_error": float(self.normalization.max()),
            "telescope_max_error": float(self.telescope_errors.max()),
            "influence_decay_rate": self.decay_rate,
        }


def _telescope(m: ToyModel, log_z, pi_star, rng, n: int) -> np.ndarray:
    T, eos, beta = m.cfg.t_max, m.eos, m.cfg.beta
    z = np.exp(log_z)
    s = rng.randint(m.cfg.n_states, size=n)
    s0 = s.copy()
    prod = np.ones(n)
    err = np.full(n, np.nan)
    alive = np.ones(n, dtype=bool)
    for t in range(T):
        idx = np.flatnonzero(alive)
        if idx.size == 0:
            break
        p = pi_star[t, s[idx]]
        v = (rng.random_sample(idx.size)[:, None] > np.cumsum(p, axis=1)).sum(axis=1)
        v = np.minimum(v, m.cfg.n_tokens - 1)
        stop = v == eos
        nxt = np.where(stop, 0, m.F[s[idx], np.minimum(v, eos - 1)])
        z_child = np.where(stop, np.exp(m.r_stop(t)[s[idx]] / beta), z[nxt, t + 1])
        prod[idx] *= z_child / z[s[idx], t]
        done = idx[stop]
        err[done] = np.abs(prod[done] - np.exp(m.r_stop(t)[s[done]] / beta) / z[s0[done], 0])
        alive[done] = False
        s[idx[~stop]] = nxt[~stop]
    left = np.flatnonzero(alive)  # forced stop at the horizon
    err[left] = np.abs(prod[left] - np.exp(m.r_stop(T)[s[left]] / beta) / z[s0[left], 0])
    return err


def _influence(m: ToyModel, pi_star, rng, n: int, max_lag: int):
    S = m.cfg.n_states
    lags = np.arange(1, min(max_lag + 1, m.cfg.t_max))
    tv = np.empty(len(lags))
    for i, lag in enumerate(lags):
        a = rng.randint(S, size=n)
        b = (a + rng.randint(1, S, size=n)) % S       # a distinct second state
        tokens = rng.randint(m.cfg.n_tokens - 1, size=(n, lag))
        for j in range(lag):                          # same token sequence for both
            a, b = m.F[a, tokens[:, j]], m.F[b, tokens[:, j]]
        tv[i] = 0.5 * np.abs(pi_star[lag, a] - pi_star[lag, b]).sum(axis=1).mean()
    return lags, tv


def run_toy_suite(cfg: ToyConfig | None = None) -> ToyResults:
    cfg = cfg or ToyConfig()
    m = ToyModel.build(cfg)
    sol = m.solve()
    T = cfg.t_max
    log_z, log_child = sol["log_z"], sol["log_child"]
    log_p = np.broadcast_to(sol["log_p"], log_child.shape)
    log_zt = log_z[:, :T].T                                        # (T, S)

    gammas = identities.gamma_three_ways(log_p, log_zt, log_child, cfg.beta)
    bellman = identities.bellman_residual(log_p, log_zt, log_child)
    norm = identities.pi_star_normalization(log_p, log_zt, log_child)
    pi_star = np.exp(log_p + log_child - log_zt[..., None])

    rng = np.random.RandomState(cfg.seed + 1)
    tele = _telescope(m, log_z, pi_star, rng, cfg.n_trajectories)
    lags, tv = _influence(m, pi_star, rng, cfg.influence_samples, cfg.influence_max_lag)

    return ToyResults(m, log_z, gammas, bellman, norm, tele, pi_star, lags, tv,
                      identities.exp_decay_rate(lags, tv))
