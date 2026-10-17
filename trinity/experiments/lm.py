"""The Trinity on a real language model, through any Backend."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .. import identities
from ..backends.base import Backend
from ..config import LMConfig
from ..rewards import get_reward
from ..tree import build_tree


@dataclass
class GammaResult:
    prompt: str
    log_z: float
    v_star: float
    n_nodes: int
    seconds: float
    candidates: list[dict]       # root-level table, one row per candidate token
    root_max_error: float
    checks: dict[str, float]     # max errors over every node of the tree


@dataclass
class TelescopeResult:
    prompt: str
    log_z0: float
    rel_errors: np.ndarray
    examples: list[dict]


@dataclass
class InfluenceResult:
    prompt: str
    lags: np.ndarray
    mean_tv: np.ndarray
    decay_rate: float


@dataclass
class LMResults:
    model: dict
    config: dict
    gamma: list[GammaResult]
    telescope: TelescopeResult
    influence: InfluenceResult
    seconds: float = 0.0
    extra: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "gamma_max_error": max(g.checks["gamma_error"] for g in self.gamma),
            "bellman_max_residual": max(g.checks["bellman_residual"] for g in self.gamma),
            "pi_star_max_normalization_error": max(g.checks["pi_star_normalization"] for g in self.gamma),
            "telescope_max_rel_error": float(self.telescope.rel_errors.max()),
            "influence_decay_rate": self.influence.decay_rate,
        }

    def to_dict(self) -> dict:
        def clean(x):
            if isinstance(x, np.ndarray):
                return x.tolist()
            if isinstance(x, (np.floating, np.integer, np.bool_)):
                return x.item()
            if isinstance(x, dict):
                return {k: clean(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return [clean(v) for v in x]
            if hasattr(x, "__dataclass_fields__"):
                return {k: clean(getattr(x, k)) for k in x.__dataclass_fields__}
            return x
        return clean({**{k: getattr(self, k) for k in self.__dataclass_fields__},
                      "summary": self.summary()})


def _token_str(backend: Backend, tid: int, width: int = 12) -> str:
    return backend.decode([tid]).replace("\n", "\\n")[:width]


def verify_gamma(backend: Backend, prompt: str, cfg: LMConfig) -> GammaResult:
    tree = build_tree(backend, backend.encode(prompt), top_k=cfg.top_k, depth=cfg.depth, beta=cfg.beta,
                      reward=get_reward(cfg.reward))
    g = tree.root_gammas()
    rows = [{
        "token_id": int(g["token_id"][i]),
        "token_str": _token_str(backend, int(g["token_id"][i])),
        "is_eos": bool(g["is_eos"][i]),
        "p_ref": float(g["p_ref"][i]),
        "pi_star": float(g["pi_star"][i]),
        "gamma_control": float(g["gamma_control"][i]),
        "gamma_bayes": float(g["gamma_bayes"][i]),
        "gamma_hmm": float(g["gamma_hmm"][i]),
        "error": float(g["error"][i]),
        "A_star": float(g["advantage"][i]),
        "Z_next": float(g["z_child"][i]),
    } for i in range(len(g["token_id"]))]
    return GammaResult(prompt=prompt, log_z=tree.log_z, v_star=cfg.beta * tree.log_z,
                       n_nodes=tree.n_nodes, seconds=tree.build_seconds, candidates=rows,
                       root_max_error=float(g["error"].max()), checks=tree.checks())


def verify_telescoping(backend: Backend, prompt: str, cfg: LMConfig,
                       rng: np.random.Generator) -> TelescopeResult:
    tree = build_tree(backend, backend.encode(prompt), top_k=cfg.top_k, depth=cfg.telescope_depth,
                      beta=cfg.beta, reward=get_reward(cfg.reward))
    s = tree.sample(cfg.telescope_trajectories, rng)
    examples = [{
        "text": backend.decode([t for t in s["tokens"][i] if t >= 0]).replace("\n", "\\n")[:40],
        "gamma_product": float(s["gamma_product"][i]),
        "expected": float(s["expected"][i]),
        "rel_error": float(s["rel_error"][i]),
    } for i in range(min(8, len(s["rel_error"])))]
    return TelescopeResult(prompt, tree.log_z, s["rel_error"], examples)


def measure_influence(backend: Backend, prompt: str, cfg: LMConfig,
                      rng: np.random.Generator) -> InfluenceResult:
    """Mean TV shift of p(. | x_{<=t}) when the token at t - lag is replaced at random."""
    base = np.asarray(backend.encode(prompt), dtype=np.int64)
    T, n = len(base), cfg.influence_perturbations
    lags = np.arange(1, min(T - 1, cfg.influence_max_lag + 1))
    lag_of = np.repeat(lags, n)
    s = (rng.random(lag_of.size) * (T - lag_of)).astype(np.int64)
    perturbed = np.repeat(base[None], lag_of.size, axis=0)
    perturbed[np.arange(lag_of.size), s] = rng.integers(0, backend.info.vocab_size, lag_of.size)

    tv = backend.tv_at_positions(base, perturbed, s + lag_of)
    mean_tv = tv.reshape(len(lags), n).mean(axis=1)
    return InfluenceResult(prompt, lags, mean_tv, identities.exp_decay_rate(lags, mean_tv))


def run_lm_suite(backend: Backend, cfg: LMConfig | None = None, log=print) -> LMResults:
    cfg = cfg or LMConfig()
    rng = np.random.default_rng(cfg.seed)
    t0 = time.perf_counter()

    gamma = []
    for prompt in cfg.prompts:
        res = verify_gamma(backend, prompt, cfg)
        gamma.append(res)
        log(f"  Gamma     {prompt!r}: {res.n_nodes} nodes in {res.seconds:.2f}s, "
            f"max err {res.checks['gamma_error']:.1e}")

    tele_prompt = cfg.prompts[cfg.telescope_prompt]
    telescope = verify_telescoping(backend, tele_prompt, cfg, rng)
    log(f"  Telescope {tele_prompt!r}: max rel err {telescope.rel_errors.max():.1e}")

    influence = measure_influence(backend, cfg.influence_prompt, cfg, rng)
    log(f"  Influence: lambda ~ {influence.decay_rate:.3f} over {len(influence.lags)} lags")

    return LMResults(model=backend.info.to_dict(), config=cfg.to_dict(), gamma=gamma,
                     telescope=telescope, influence=influence,
                     seconds=time.perf_counter() - t0)
