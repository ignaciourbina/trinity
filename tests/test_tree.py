import numpy as np
import pytest

from trinity.config import LMConfig
from trinity.experiments.lm import run_lm_suite
from trinity.experiments.toy import run_toy_suite
from trinity.rewards import FluencyReward
from trinity.tree import build_tree, tree_size

from .conftest import brute_force_log_z


@pytest.mark.parametrize("depth,top_k", [(0, 5), (1, 4), (3, 6)])
def test_tree_matches_brute_force(synthetic, depth, top_k):
    reward, beta = FluencyReward(), 1.3
    prompt = synthetic.encode("hello")
    tree = build_tree(synthetic, prompt, top_k=top_k, depth=depth, beta=beta, reward=reward)
    expected = brute_force_log_z(synthetic, prompt, depth, top_k, beta, reward)
    assert tree.log_z == pytest.approx(expected, abs=1e-12)
    assert tree.n_nodes <= tree_size(top_k, depth)


def test_tree_identities_hold_everywhere(synthetic):
    tree = build_tree(synthetic, synthetic.encode("abc"), top_k=8, depth=3, beta=0.7,
                      reward=FluencyReward())
    checks = tree.checks()
    assert max(checks.values()) < 1e-10
    root = tree.root_gammas()
    assert (root["token_id"] == synthetic.info.eos_id).any()  # EOS is always a candidate


def test_telescoping(synthetic):
    tree = build_tree(synthetic, synthetic.encode("xy"), top_k=6, depth=3, beta=1.0,
                      reward=FluencyReward())
    s = tree.sample(500, np.random.default_rng(0))
    assert np.all(np.isfinite(s["expected"]))
    assert s["rel_error"].max() < 1e-10


def test_toy_suite():
    s = run_toy_suite().summary()
    assert s["gamma_max_error"] < 1e-12
    assert s["bellman_max_residual"] < 1e-12
    assert s["pi_star_max_normalization_error"] < 1e-12
    assert s["telescope_max_error"] < 1e-10
    assert 0 < s["influence_decay_rate"] < 1


def test_lm_suite_synthetic(synthetic):
    cfg = LMConfig(prompts=("abc", "hello there"), top_k=5, depth=2, telescope_trajectories=50,
                   influence_perturbations=10)
    res = run_lm_suite(synthetic, cfg, log=lambda *a: None)
    s = res.summary()
    assert s["gamma_max_error"] < 1e-10
    assert s["telescope_max_rel_error"] < 1e-10
    assert 0 < s["influence_decay_rate"] < 1
    assert res.to_dict()["summary"] == s
