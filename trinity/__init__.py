"""The Trinity: optimal control, Bayesian inference and HMMs on language models.

Layers, from the bottom up:
    identities   the three formulas for Gamma, plus Bellman / normalisation checks
    rewards      vectorised trajectory rewards R(x_{1:tau})
    tree         batched soft-DP over a model's top-K lookahead tree
    backends     where log-probs come from: HuggingFace, synthetic, or a remote server
    experiments  the toy and language-model verification suites
    remote       HTTP server/client and the Colab tunnel helper
"""

from .backends import Backend, ModelInfo, SyntheticBackend, get_backend
from .config import LMConfig, ToyConfig
from .experiments import run_lm_suite, run_toy_suite
from .tree import build_tree

__version__ = "0.2.0"

__all__ = ["Backend", "LMConfig", "ModelInfo", "SyntheticBackend", "ToyConfig",
           "build_tree", "get_backend", "run_lm_suite", "run_toy_suite"]
