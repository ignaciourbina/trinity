"""Trajectory-level rewards R(x_{1:tau}), vectorised over a batch of continuations.

A reward is any callable ``reward(tokens, logps, eos_id) -> (N,) float array`` where
``tokens`` and ``logps`` are (N, L) arrays holding the continuation tokens and their
log pi_ref. Every row in a batch has the same length L >= 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np


class Reward(Protocol):
    def __call__(self, tokens: np.ndarray, logps: np.ndarray, eos_id: int) -> np.ndarray: ...


@dataclass(frozen=True)
class FluencyReward:
    """Fluency plus a diversity penalty, a length bonus and a stopping bonus.

    R = sum_t [fluency * lp_t - diversity * max(lp_t - threshold, 0)]
        + length_bonus * L + eos_bonus * [x_L == EOS]
    """

    fluency: float = 0.5
    diversity: float = 0.1
    threshold: float = -3.0
    length_bonus: float = 0.2
    eos_bonus: float = 1.0

    def __call__(self, tokens: np.ndarray, logps: np.ndarray, eos_id: int) -> np.ndarray:
        per_token = self.fluency * logps - self.diversity * np.maximum(logps - self.threshold, 0.0)
        r = per_token.sum(axis=1) + self.length_bonus * tokens.shape[1]
        return r + self.eos_bonus * (tokens[:, -1] == eos_id)


@dataclass(frozen=True)
class LengthReward:
    """Rewards stopping after exactly ``target`` tokens; a sanity-check reward."""

    target: int = 2
    scale: float = 1.0

    def __call__(self, tokens: np.ndarray, logps: np.ndarray, eos_id: int) -> np.ndarray:
        return np.full(tokens.shape[0], -self.scale * abs(tokens.shape[1] - self.target), dtype=float)


REWARDS: dict[str, Callable[[], Reward]] = {
    "fluency": FluencyReward,
    "length": LengthReward,
}


def get_reward(name_or_reward: str | Reward) -> Reward:
    if callable(name_or_reward):
        return name_or_reward
    try:
        return REWARDS[name_or_reward]()
    except KeyError:
        raise ValueError(f"unknown reward {name_or_reward!r}; choose from {sorted(REWARDS)}") from None
