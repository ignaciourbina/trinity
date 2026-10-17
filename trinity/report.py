"""Console reports for the experiment results."""

from __future__ import annotations

import numpy as np

from .experiments.lm import LMResults
from .experiments.toy import ToyResults


def _status(ok: bool) -> str:
    return "PASS" if ok else "CHECK"


def print_lm(res: LMResults) -> None:
    m, cfg = res.model, res.config
    print(f"\n{'=' * 72}\nTHE TRINITY ON {m['name']} ({m['n_layers']} layers, "
          f"d={m['hidden_dim']}, vocab={m['vocab_size']}, {m['device']})\n{'=' * 72}")
    for g in res.gamma:
        print(f"\nPrompt: {g.prompt!r}   [{g.n_nodes} nodes, {g.seconds:.2f}s]")
        print(f"  Z(h_t) = {np.exp(g.log_z):.6f}   V*(h_t) = {g.v_star:.6f}")
        print(f"  {'Token':>15} {'Gamma_Ctrl':>12} {'Gamma_Bayes':>12} {'Gamma_HMM':>12} "
              f"{'Error':>9} {'pi_ref':>8} {'pi*':>8}")
        for r in g.candidates:
            label = r["token_str"] + (" [EOS]" if r["is_eos"] else "")
            print(f"  {label:>15} {r['gamma_control']:>12.8f} {r['gamma_bayes']:>12.8f} "
                  f"{r['gamma_hmm']:>12.8f} {r['error']:>9.1e} {r['p_ref']:>8.4f} {r['pi_star']:>8.4f}")
        c = g.checks
        print(f"  All nodes: Gamma err {c['gamma_error']:.1e} | Bellman {c['bellman_residual']:.1e}"
              f" | pi* norm {c['pi_star_normalization']:.1e}")

    t = res.telescope
    print(f"\nTelescoping on {t.prompt!r} (depth {cfg['telescope_depth']}):")
    for e in t.examples:
        print(f"  {e['text']!r:<44} prod(Gamma)={e['gamma_product']:.6f}  "
              f"exp(R/b)/Z0={e['expected']:.6f}  rel_err={e['rel_error']:.1e}")
    print(f"  Relative errors: max={t.rel_errors.max():.1e}, median={np.median(t.rel_errors):.1e}")

    inf = res.influence
    print("\nInfluence decay:")
    for lag, tv in zip(inf.lags, inf.mean_tv):
        print(f"  lag {lag:>2d}  TV {tv:.6f}  {'#' * int(tv * 50)}")

    s = res.summary()
    print(f"\n{'-' * 72}\nSUMMARY  ({res.seconds:.1f}s total)")
    print(f"  Gamma equivalence     {s['gamma_max_error']:.2e}  {_status(s['gamma_max_error'] < 1e-8)}")
    print(f"  Bellman equation      {s['bellman_max_residual']:.2e}  {_status(s['bellman_max_residual'] < 1e-8)}")
    print(f"  pi* normalisation     {s['pi_star_max_normalization_error']:.2e}  "
          f"{_status(s['pi_star_max_normalization_error'] < 1e-8)}")
    print(f"  Telescoping identity  {s['telescope_max_rel_error']:.2e}  "
          f"{_status(s['telescope_max_rel_error'] < 1e-6)}")
    print(f"  Influence decay       lambda ~ {s['influence_decay_rate']:.3f}")


def print_toy(res: ToyResults) -> None:
    cfg = res.model.cfg
    s = res.summary()
    print(f"\n{'=' * 72}\nTHE TRINITY ON A TOY MODEL ({cfg.n_states} states, {cfg.n_tokens} tokens, "
          f"T={cfg.t_max}, beta={cfg.beta})\n{'=' * 72}")
    g = res.gammas
    names = [*"abcdefghijklmnopqrstuvwxyz"[: cfg.n_tokens - 1], "EOS"]
    print(f"\nGamma at t=3 (first 4 states):\n  {'State':>5} {'Token':>5} {'Ctrl':>12} "
          f"{'Bayes':>12} {'HMM':>12} {'Error':>9}")
    for st in range(min(4, cfg.n_states)):
        for v in range(cfg.n_tokens):
            print(f"  {st:>5d} {names[v]:>5} {g['gamma_control'][3, st, v]:>12.8f} "
                  f"{g['gamma_bayes'][3, st, v]:>12.8f} {g['gamma_hmm'][3, st, v]:>12.8f} "
                  f"{g['error'][3, st, v]:>9.1e}")
    print("\nInfluence decay:")
    for lag, tv in zip(res.influence_lags, res.influence_tv):
        print(f"  lag {lag:>2d}  TV {tv:.6f}  {'#' * int(tv * 40)}")
    print(f"\n{'-' * 72}\nSUMMARY  ({g['error'].size} (t, s, v) triples)")
    print(f"  Gamma equivalence     {s['gamma_max_error']:.2e}  {_status(s['gamma_max_error'] < 1e-12)}")
    print(f"  Bellman equation      {s['bellman_max_residual']:.2e}  {_status(s['bellman_max_residual'] < 1e-12)}")
    print(f"  pi* normalisation     {s['pi_star_max_normalization_error']:.2e}  "
          f"{_status(s['pi_star_max_normalization_error'] < 1e-12)}")
    print(f"  Telescoping identity  {s['telescope_max_error']:.2e}  {_status(s['telescope_max_error'] < 1e-10)}")
    print(f"  Influence decay       lambda ~ {s['influence_decay_rate']:.3f}")
