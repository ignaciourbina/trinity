import numpy as np
import pytest

from trinity.config import LMConfig
from trinity.experiments.lm import run_lm_suite
from trinity.remote import RemoteBackend, RemoteError, TrinityServer


@pytest.fixture
def served(synthetic):
    server = TrinityServer(synthetic, port=0, token="s3cret").start()
    yield synthetic, server
    server.stop()


def test_remote_matches_local(served):
    local, server = served
    remote = RemoteBackend(server.address, token="s3cret")
    assert remote.info == local.info
    assert remote.encode("hi there") == local.encode("hi there")
    assert remote.decode([1, 2, 3]) == local.decode([1, 2, 3])

    seqs = np.random.default_rng(0).integers(0, 39, size=(17, 5))
    for got, want in zip(remote.topk_next(seqs, 4, force_id=39), local.topk_next(seqs, 4, force_id=39)):
        np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(remote.hidden_states(seqs), local.hidden_states(seqs))

    cfg = LMConfig(prompts=("abc",), top_k=5, depth=2, telescope_trajectories=20,
                   influence_perturbations=5)
    quiet = dict(log=lambda *a: None)
    assert run_lm_suite(remote, cfg, **quiet).summary() == run_lm_suite(local, cfg, **quiet).summary()


def test_rejects_bad_token(served):
    _, server = served
    with pytest.raises(RemoteError, match="401"):
        RemoteBackend(server.address, token="wrong", retries=0)


def test_rejects_out_of_vocab_ids(served):
    _, server = served
    remote = RemoteBackend(server.address, token="s3cret", retries=0)
    with pytest.raises(RemoteError, match="400"):
        remote.topk_next(np.array([[0, 999]]), 3)
