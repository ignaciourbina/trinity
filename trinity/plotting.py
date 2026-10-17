"""Figures for the experiment results. matplotlib is imported only when plotting."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .experiments.lm import LMResults
from .experiments.toy import ToyResults


def _figure(rows: int, title: str, height: float):
    import matplotlib.pyplot as plt  # falls back to Agg on headless machines
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(16, height))
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.98)
    gs = GridSpec(rows, 2, figure=fig, hspace=0.35, wspace=0.3,
                  left=0.08, right=0.95, top=0.94, bottom=0.05)
    return plt, fig, gs


def _save(plt, fig, out: Path, stem: str, show: bool) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / f"{stem}.png", out / f"{stem}.pdf"]
    fig.savefig(paths[0], dpi=150, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return paths


def _gamma_scatter(ax, gc, gb, gh, title):
    ax.scatter(gc, gb, alpha=0.6, s=30, c="#2563eb", label="Bayes vs Control", zorder=3)
    ax.scatter(gc, gh, alpha=0.4, s=20, c="#dc2626", marker="x", label="HMM vs Control", zorder=3)
    lims = [min(gc.min(), 0) * 0.9, max(gc.max(), 1.5) * 1.1]
    ax.plot(lims, lims, "k--", alpha=0.4, linewidth=1, label="y = x")
    ax.set(xlabel="Γ (Optimal Control)", ylabel="Γ (Bayes / HMM)")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)


def _decay(ax, lags, tv, rate, title):
    ax.plot(lags, tv, "o-", color="#ea580c", linewidth=2, markersize=6)
    if np.isfinite(rate):
        slope, icpt = np.polyfit(lags, np.log(np.asarray(tv) + 1e-10), 1)
        ax.plot(lags, np.exp(icpt + slope * np.asarray(lags)), "--", color="gray",
                label=f"Exp fit: λ ≈ {rate:.3f}")
        ax.legend(fontsize=9)
    ax.set(xlabel="Lag", ylabel="Mean TV distance")
    ax.set_title(title, fontsize=11, fontweight="bold")


def _summary(ax, text):
    ax.axis("off")
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=9.5, va="top",
            fontfamily="monospace", bbox=dict(boxstyle="round", facecolor="#f0f9ff", alpha=0.8))


def plot_lm(res: LMResults, out: str | Path = "outputs", show: bool = False) -> list[Path]:
    name = res.model["name"]
    plt, fig, gs = _figure(3, f"The Trinity: Empirical Verification on {name}", 18)

    rows = [r for g in res.gamma for r in g.candidates]
    gc, gb, gh = (np.array([r[k] for r in rows]) for k in ("gamma_control", "gamma_bayes", "gamma_hmm"))
    _gamma_scatter(fig.add_subplot(gs[0, 0]), gc, gb, gh, "Γ Identity: All Prompts")

    ax = fig.add_subplot(gs[0, 1])
    first = res.gamma[0].candidates
    x = np.arange(len(first))
    ax.bar(x - 0.175, [r["p_ref"] for r in first], 0.35, label="π_ref (pretrained)", color="#94a3b8")
    ax.bar(x + 0.175, [r["pi_star"] for r in first], 0.35, label="π* (optimal)", color="#2563eb")
    ax.set_xticks(x, [r["token_str"] for r in first], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Probability")
    ax.set_title(f"Policy Shift: {res.gamma[0].prompt[:25]!r}", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 0])
    width = 0.8 / len(res.gamma)
    for i, g in enumerate(res.gamma):
        adv = [r["A_star"] for r in g.candidates]
        ax.bar(np.arange(len(adv)) + i * width - 0.4, adv, width * 0.9, alpha=0.7,
               label=f"{g.prompt[:15]!r}")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set(xlabel="Token index (top-K)", ylabel="A*(h_t, v)")
    ax.set_title("Soft Advantage Function", fontsize=11, fontweight="bold")
    ax.legend(fontsize=6, ncol=2)

    ax = fig.add_subplot(gs[1, 1])
    ax.semilogy(res.telescope.rel_errors + 1e-16, "o-", color="#7c3aed", markersize=6)
    ax.axhline(1e-6, color="red", linestyle="--", alpha=0.5, label="10⁻⁶ threshold")
    ax.set(xlabel="Trajectory index", ylabel="|∏Γ - exp(R/β)/Z₀| / (exp(R/β)/Z₀)")
    ax.set_title("Telescoping Identity", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)

    inf = res.influence
    _decay(fig.add_subplot(gs[2, 0]), inf.lags, inf.mean_tv, inf.decay_rate, f"Influence Decay on {name}")

    s = res.summary()
    _summary(fig.add_subplot(gs[2, 1]), f"""
  VERIFICATION SUMMARY ({name})
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Γ equivalence (every tree node): {s['gamma_max_error']:.2e}
  Bellman residual:                {s['bellman_max_residual']:.2e}
  π* normalisation:                {s['pi_star_max_normalization_error']:.2e}
  Telescoping (max rel. error):    {s['telescope_max_rel_error']:.2e}
  Influence decay:                 λ ≈ {s['influence_decay_rate']:.3f}
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  exp(A*/β) = π*/π_ref = dκ*/dκ_ref
  [Control]   [Bayes]    [HMM]
""")
    stem = "trinity_" + "".join(c if c.isalnum() else "_" for c in name) + "_verification"
    return _save(plt, fig, Path(out), stem, show)


def plot_toy(res: ToyResults, out: str | Path = "outputs", show: bool = False) -> list[Path]:
    m, cfg = res.model, res.model.cfg
    plt, fig, gs = _figure(4, "Computational Verification of the Trinity", 22)
    g = res.gammas
    gc, gb, gh = (g[k].ravel()[::10] for k in ("gamma_control", "gamma_bayes", "gamma_hmm"))
    _gamma_scatter(fig.add_subplot(gs[0, 0]), gc, gb, gh, "Test 1: Γ Identity Across Frameworks")

    ax = fig.add_subplot(gs[0, 1])
    ax.semilogy(res.bellman.ravel() + 1e-16, ".", markersize=3, color="#059669")
    ax.axhline(1e-12, color="red", linestyle="--", alpha=0.5, label="Machine ε threshold")
    ax.set(xlabel="(Time, State) index", ylabel="|LHS - RHS|")
    ax.set_title("Test 2: Bellman Equation Residuals", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 0])
    for st in range(cfg.n_states):
        ax.plot(range(cfg.t_max + 1), res.v_star[st], "o-", markersize=3, label=f"State {st}", alpha=0.7)
    ax.set(xlabel="Time step t", ylabel="V*(s, t)")
    ax.set_title("Value Function Landscape", fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, ncol=2, loc="upper right")

    ax = fig.add_subplot(gs[1, 1])
    p0 = m.pi_ref
    stop = p0[:, -1] * np.exp(m.r_stop(0) / cfg.beta)
    cont = np.exp(res.log_z[:, 0]) - stop
    x = np.arange(cfg.n_states)
    ax.bar(x - 0.175, stop, 0.35, label="S(h): Stop value", color="#dc2626", alpha=0.7)
    ax.bar(x + 0.175, cont, 0.35, label="C(h): Continue value", color="#2563eb", alpha=0.7)
    ax.set(xlabel="Hidden State", ylabel="Value", xticks=x)
    ax.set_title("Stop vs Continue (t=0)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[2, 0])
    ax.semilogy(res.telescope_errors + 1e-16, ".", markersize=3, color="#7c3aed")
    ax.axhline(1e-10, color="red", linestyle="--", alpha=0.5, label="Threshold")
    ax.set(xlabel="Trajectory index", ylabel="|∏Γ - exp(R/β)/Z₀|")
    ax.set_title("Test 4: Telescoping Identity Errors", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)

    _decay(fig.add_subplot(gs[2, 1]), res.influence_lags, res.influence_tv, res.decay_rate,
           "Influence Decay (Memorylessness)")

    ax = fig.add_subplot(gs[3, 0])
    t_comp, width = 2, 0.15
    names = [*"abcdefghijklmnopqrstuvwxyz"[: cfg.n_tokens - 1], "EOS"]
    for i, st in enumerate(range(min(4, cfg.n_states))):
        pos = np.arange(cfg.n_tokens) + i * width
        ax.bar(pos - 0.1, m.pi_ref[st], width * 0.45, alpha=0.5, color=f"C{i}")
        ax.bar(pos + 0.1, res.pi_star[t_comp, st], width * 0.45, alpha=0.9, color=f"C{i}",
               edgecolor="black", linewidth=0.5)
    ax.set_xticks(np.arange(cfg.n_tokens) + width * 1.5, names)
    ax.set_ylabel("Probability")
    ax.set_title(f"Policy Shift: π_ref (light) vs π* (dark), t={t_comp}", fontsize=11, fontweight="bold")

    s = res.summary()
    _summary(fig.add_subplot(gs[3, 1]), f"""
VERIFICATION SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Γ equivalence:      {s['gamma_max_error']:.2e}
Bellman residual:   {s['bellman_max_residual']:.2e}
π* normalisation:   {s['pi_star_max_normalization_error']:.2e}
Telescoping error:  {s['telescope_max_error']:.2e}
Influence decay:    λ ≈ {s['influence_decay_rate']:.3f}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
    return _save(plt, fig, Path(out), "trinity_verification", show)
