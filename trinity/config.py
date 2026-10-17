"""Experiment configuration. Defaults reproduce the original scripts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

DEFAULT_PROMPTS = (
    "The meaning of life is",
    "In 1969, the first human",
    "The capital of France is",
    "Once upon a time there was a",
)

LONG_PROMPT = ("In the beginning there was nothing but darkness and silence "
               "and the vast empty void stretched endlessly in every direction "
               "until one day a small spark appeared")


@dataclass
class LMConfig:
    prompts: tuple[str, ...] = DEFAULT_PROMPTS
    top_k: int = 12               # candidates per node (EOS always included)
    depth: int = 3                # lookahead depth of the backward DP
    beta: float = 1.0             # KL penalty / inverse temperature
    reward: str = "fluency"
    telescope_prompt: int = 0     # index into prompts
    telescope_depth: int = 2
    telescope_trajectories: int = 15
    influence_prompt: str = LONG_PROMPT
    influence_perturbations: int = 40
    influence_max_lag: int = 14
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ToyConfig:
    n_tokens: int = 4             # last token is EOS
    t_max: int = 10
    beta: float = 2.0
    seed: int = 42
    n_trajectories: int = 1000
    influence_samples: int = 500
    influence_max_lag: int = 7
    state_rewards: tuple[float, ...] = field(
        default=(0.5, 1.2, 0.8, -0.3, 1.5, 0.2, -0.5, 0.9))
    step_cost: float = 0.1

    @property
    def n_states(self) -> int:
        return len(self.state_rewards)

    def to_dict(self) -> dict:
        return asdict(self)
