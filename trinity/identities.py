"""The three faces of Gamma, each computed along its own numerical path.

Every function here is pure numpy and broadcasts over arbitrary leading axes,
so the same code verifies the toy model (states x time x tokens) and the
transformer search tree (nodes x top-K candidates).

Notation, for a node h and candidate token v:
    log_p_ref   log pi_ref(v | h)
    log_z       log Z(h)
    log_child   log Z(h') if v continues, or R(x_{1:tau}) / beta if v terminates
"""

from __future__ import annotations

import numpy as np


def logsumexp(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    return np.squeeze(m, axis=axis) + np.log(np.sum(np.exp(x - m), axis=axis))


def soft_bellman(log_p_ref: np.ndarray, log_child: np.ndarray) -> np.ndarray:
    """log Z(h) = log sum_v pi_ref(v|h) * exp(log_child(h, v))  (free-boundary Bellman)."""
    return logsumexp(log_p_ref + log_child, axis=-1)


def gamma_three_ways(log_p_ref: np.ndarray, log_z: np.ndarray,
                     log_child: np.ndarray, beta: float) -> dict[str, np.ndarray]:
    """Gamma via (I) optimal control, (II) Bayes, (III) HMM kernel ratio.

    Shapes: log_p_ref, log_child (..., K); log_z (...,). All outputs are (..., K).
    """
    log_z = log_z[..., None]

    # (I) Optimal control: exponentiated soft advantage A* = V*(h') - V*(h).
    v_star = beta * log_z
    v_next = beta * log_child
    advantage = v_next - v_star
    gamma_control = np.exp(advantage / beta)

    # (II) Bayesian: likelihood ratio pi* / pi_ref, with pi* built in probability space.
    p_ref = np.exp(log_p_ref)
    pi_star = p_ref * np.exp(log_child) / np.exp(log_z)
    with np.errstate(divide="ignore", invalid="ignore"):
        gamma_bayes = np.where(p_ref > 0, pi_star / p_ref, 0.0)

    # (III) HMM: Radon-Nikodym derivative of the killed kernel, Z(h') / Z(h).
    gamma_hmm = np.exp(log_child) / np.exp(log_z)

    error = np.maximum.reduce([np.abs(gamma_control - gamma_bayes),
                               np.abs(gamma_control - gamma_hmm),
                               np.abs(gamma_bayes - gamma_hmm)])
    return {
        "gamma_control": gamma_control,
        "gamma_bayes": gamma_bayes,
        "gamma_hmm": gamma_hmm,
        "pi_star": pi_star,
        "advantage": advantage,
        "error": error,
    }


def bellman_residual(log_p_ref: np.ndarray, log_z: np.ndarray,
                     log_child: np.ndarray) -> np.ndarray:
    """|Z(h) - sum_v pi_ref(v|h) Z(h'_v)| evaluated in probability space."""
    rhs = np.sum(np.exp(log_p_ref) * np.exp(log_child), axis=-1)
    return np.abs(np.exp(log_z) - rhs)


def pi_star_normalization(log_p_ref: np.ndarray, log_z: np.ndarray,
                          log_child: np.ndarray) -> np.ndarray:
    """|sum_v pi*(v|h) - 1|."""
    pi_star = np.exp(log_p_ref + log_child - log_z[..., None])
    return np.abs(pi_star.sum(axis=-1) - 1.0)


def exp_decay_rate(lags, values) -> float:
    """Fit values ~ c * lambda^lag and return lambda (nan if too few points)."""
    lags = np.asarray(lags, dtype=float)
    values = np.asarray(values, dtype=float)
    if len(lags) < 3:
        return float("nan")
    slope, _ = np.polyfit(lags, np.log(values + 1e-10), 1)
    return float(np.exp(slope))
