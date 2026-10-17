import numpy as np
import pytest

from trinity.backends.synthetic import SyntheticBackend


@pytest.fixture
def synthetic():
    return SyntheticBackend(vocab_size=40, hidden_dim=16, seed=3)


class CharTokenizer:
    """Stand-in for a HF tokenizer so tests never download anything."""

    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.eos_token_id = vocab_size - 1

    def encode(self, text):
        return [ord(c) % (self.vocab_size - 1) for c in text]

    def decode(self, ids):
        return "".join(chr(97 + i % 26) for i in ids)


@pytest.fixture(scope="session")
def tiny_hf():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from trinity.backends.hf import HFBackend

    torch.manual_seed(0)
    cfg = transformers.GPT2Config(vocab_size=64, n_positions=64, n_embd=32, n_layer=2, n_head=2)
    model = transformers.GPT2LMHeadModel(cfg)
    return HFBackend(model, CharTokenizer(64), device="cpu", batch_tokens=37)


def brute_force_log_z(backend, seq, depth, top_k, beta, reward, cont=(), cont_lp=()):
    """Reference recursion, one sequence at a time, sharing nothing with trinity.tree."""
    eos = backend.info.eos_id
    ids, logp = backend.topk_next(np.asarray([seq]), top_k, force_id=eos)
    terms = []
    for v, lp in zip(ids[0], logp[0]):
        toks, lps = (*cont, int(v)), (*cont_lp, float(lp))
        if v == eos or depth == 0:
            child = reward(np.asarray([toks]), np.asarray([lps]), eos)[0] / beta
        else:
            child = brute_force_log_z(backend, [*seq, int(v)], depth - 1, top_k, beta, reward, toks, lps)
        terms.append(lp + child)
    return float(np.logaddexp.reduce(terms))
