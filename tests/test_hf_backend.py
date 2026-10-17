import numpy as np
import pytest

torch = pytest.importorskip("torch")


def _full_logp(backend, seqs):
    with torch.no_grad():
        logits = backend.model(input_ids=torch.as_tensor(seqs)).logits[:, -1].float()
    return torch.log_softmax(logits, -1).double().numpy()


def test_topk_matches_full_forward(tiny_hf):
    rng = np.random.default_rng(0)
    seqs = rng.integers(0, 63, size=(23, 7))  # 23 rows with a 37-token budget forces many chunks
    ids, logp = tiny_hf.topk_next(seqs, 5)
    ref = _full_logp(tiny_hf, seqs)
    np.testing.assert_allclose(logp, np.take_along_axis(ref, ids, 1), atol=1e-5)
    np.testing.assert_array_equal(ids[:, 0], ref.argmax(1))


def test_force_eos(tiny_hf):
    seqs = np.random.default_rng(1).integers(0, 63, size=(9, 4))
    eos = tiny_hf.info.eos_id
    ids, logp = tiny_hf.topk_next(seqs, 3, force_id=eos)
    assert (ids == eos).any(1).all()
    ref = _full_logp(tiny_hf, seqs)
    np.testing.assert_allclose(logp, np.take_along_axis(ref, ids, 1), atol=1e-5)


def test_tv_matches_naive(tiny_hf):
    rng = np.random.default_rng(2)
    base = rng.integers(0, 63, size=10)
    perturbed = np.repeat(base[None], 6, 0)
    perturbed[np.arange(6), [0, 1, 2, 3, 4, 5]] = rng.integers(0, 63, 6)
    positions = np.array([3, 5, 9, 4, 9, 7])
    tv = tiny_hf.tv_at_positions(base, perturbed, positions)
    with torch.no_grad():
        pb = torch.softmax(tiny_hf.model(input_ids=torch.as_tensor(base[None])).logits[0], -1)
        pp = torch.softmax(tiny_hf.model(input_ids=torch.as_tensor(perturbed)).logits, -1)
    naive = [0.5 * (pp[i, t] - pb[t]).abs().sum().item() for i, t in enumerate(positions)]
    np.testing.assert_allclose(tv, naive, atol=1e-5)


def test_hidden_states(tiny_hf):
    seqs = np.random.default_rng(3).integers(0, 63, size=(4, 6))
    h = tiny_hf.hidden_states(seqs, layers=[0, 2])
    assert h.shape == (4, 2, tiny_hf.info.hidden_dim)
    with torch.no_grad():
        ref = tiny_hf.model(input_ids=torch.as_tensor(seqs), output_hidden_states=True).hidden_states
    np.testing.assert_allclose(h[:, 1], ref[2][:, -1].numpy(), atol=1e-5)


def test_suite_on_tiny_gpt2(tiny_hf):
    from trinity.config import LMConfig
    from trinity.experiments.lm import run_lm_suite

    cfg = LMConfig(prompts=("abcdef",), top_k=6, depth=2, telescope_trajectories=30,
                   influence_prompt="the quick brown fox jumps", influence_perturbations=8)
    s = run_lm_suite(tiny_hf, cfg, log=lambda *a: None).summary()
    assert s["gamma_max_error"] < 1e-8
    assert s["bellman_max_residual"] < 1e-8
    assert s["telescope_max_rel_error"] < 1e-8


def test_prefix_cache_matches_full_recompute(tiny_hf):
    rng = np.random.default_rng(4)
    prompt = rng.integers(0, 63, size=9)
    seqs = np.concatenate([np.repeat(prompt[None], 31, 0), rng.integers(0, 63, size=(31, 3))], 1)
    ids, logp = tiny_hf.topk_next(seqs, 6, force_id=tiny_hf.info.eos_id)
    assert tiny_hf.prefix_cache and tiny_hf._kv_key == prompt.tobytes()  # the cached path ran
    ref = _full_logp(tiny_hf, seqs)
    np.testing.assert_allclose(logp, np.take_along_axis(ref, ids, 1), atol=1e-5)
