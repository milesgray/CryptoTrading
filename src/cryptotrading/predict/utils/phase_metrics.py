#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
phase_metrics.py
================
Reference implementation for "The Spectrum Is Not Enough: When Context Helps
Time-Series Forecasting."

What this file provides
-----------------------
1. Surrogates that hold the power spectrum (and, for IAAFT, the marginal) fixed
   while destroying beyond-second-order structure:
       phase_randomize(...)   - FT phase randomization (preserves periodogram)
       iaaft(...)             - amplitude-adjusted version (preserves marginal too)

2. The "opponent" series-level predictability indices, all functions of the
   power spectrum / autocovariance (hence invariant under phase randomization,
   Proposition 1 of the paper):
       omega_spectral_predictability(...)   - spectral concentration  (Omega-like)
       scp_coherence(...)                   - 2nd-order coherence      (SCP-like)

3. The configuration-level diagnostic (the paper's coverage deficit Gamma):
       analog_gain(...)     - Delta_nl: gain of nearest-neighbour over linear
                              prediction = the beyond-spectrum structure term
       coverage_u(...)      - operating-point term u(S) = max(0,(m-S)/m)
       gamma_coverage(...)  - Gamma_cov = Delta_nl * u(S)
       symbolic_oov_rate(...) - Gamma_oov novelty term

4. A context-value estimator for a beyond-spectrum mechanism:
       context_value_analog(...) - analog retrieval (reaches the whole training
                                   memory) vs a short within-window linear model.

5. bootstrap_ci(...) and a SMOKE TEST (run `python -m phasepower.metrics`) that
   reproduces, in miniature, experiment E1: on a nonlinear deterministic system
   the spectral indices are (nearly) identical across a surrogate pair while the
   diagnostic and the measured context value collapse on the surrogate.

Dependencies: numpy, scipy only.  CPU, seconds to run.

NOTE: This module is the analysis/diagnostic core. It is intentionally
self-contained and does NOT train the deep backbones; those runs are wired in
the experiment harness referenced by RUNBOOK.txt (E1-E6). The numbers printed
here are a positive control, not the paper's benchmark results.
"""

from __future__ import annotations
import numpy as np
from numpy.fft import rfft, irfft
from scipy.spatial import cKDTree

# --------------------------------------------------------------------------- #
# 1. Surrogates                                                               #
# --------------------------------------------------------------------------- #

def phase_randomize(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """FT phase randomization: identical periodogram, random Fourier phases."""
    x = np.asarray(x, float)
    n = x.size
    X = rfft(x)
    mag = np.abs(X)
    phases = np.angle(X)
    # randomize interior phases; keep DC (and Nyquist if present) real
    new_phases = rng.uniform(0, 2 * np.pi, size=phases.shape)
    new_phases[0] = phases[0]
    if n % 2 == 0:
        new_phases[-1] = phases[-1]
    Xs = mag * np.exp(1j * new_phases)
    return irfft(Xs, n=n)


def iaaft(x: np.ndarray, rng: np.random.Generator, n_iter: int = 200,
          tol: float = 1e-8) -> np.ndarray:
    """Iterative amplitude-adjusted FT surrogate (Schreiber & Schmitz, 2000).

    Preserves the periodogram (up to a small residual) AND the empirical
    marginal distribution, while destroying deterministic phase structure.
    """
    x = np.asarray(x, float)
    n = x.size
    sorted_x = np.sort(x)
    target_mag = np.abs(rfft(x))
    # start from a random shuffle of the data (preserves marginal exactly)
    s = rng.permutation(x).astype(float)
    prev_err = np.inf
    for _ in range(n_iter):
        # 1) impose the target power spectrum
        S = rfft(s)
        phases = np.angle(S)
        s = irfft(target_mag * np.exp(1j * phases), n=n)
        # 2) impose the empirical marginal (rank remap)
        ranks = np.argsort(np.argsort(s))
        s = sorted_x[ranks]
        err = np.linalg.norm(np.abs(rfft(s)) - target_mag) / np.linalg.norm(target_mag)
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    return s


def iaaft_residual(x: np.ndarray, xs: np.ndarray) -> float:
    """Relative periodogram mismatch between a series and its IAAFT surrogate."""
    a, b = np.abs(rfft(x)), np.abs(rfft(xs))
    denom = np.linalg.norm(a)
    if denom <= 1e-12:        # constant series has an empty spectrum
        return 0.0
    return float(np.linalg.norm(a - b) / denom)


# --------------------------------------------------------------------------- #
# 2. Series-level predictability indices (power-spectrum functionals)         #
# --------------------------------------------------------------------------- #

def _periodogram(x: np.ndarray) -> np.ndarray:
    X = rfft(x - x.mean())
    p = (np.abs(X) ** 2)
    p = p[1:]  # drop DC
    s = p.sum()
    return p / s if s > 0 else p


def omega_spectral_predictability(x: np.ndarray) -> float:
    """Spectral predictability (Omega-like): 1 - normalized spectral entropy.

    Pure function of the periodogram => invariant under phase randomization.
    High when power is concentrated (peaked spectrum), low when flat.
    """
    p = _periodogram(np.asarray(x, float))
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    h = -(p * np.log(p)).sum()
    return float(1.0 - h / np.log(p.size))


def scp_coherence(x: np.ndarray, S: int, H: int, n_lags: int | None = None) -> float:
    """Second-order coherence proxy (SCP-like): normalized predictive power of a
    linear predictor of horizon H from a length-S window, computed from the
    autocovariance only. Pure 2nd-order => invariant under phase randomization.
    """
    x = np.asarray(x, float)
    x = x - x.mean()
    n_lags = n_lags or (S + H)
    ac = np.correlate(x, x, mode="full")
    mid = ac.size // 2
    if ac[mid] <= 1e-12:      # constant series: no variance, define R^2 = 0
        return 0.0
    g = ac[mid: mid + n_lags + 1] / ac[mid]  # normalized autocovariance
    # predictive R^2 of an AR(S) one-step predictor via Yule-Walker
    R = np.array([[g[abs(i - j)] for j in range(S)] for i in range(S)])
    r = g[1:S + 1]
    try:
        phi = np.linalg.solve(R + 1e-8 * np.eye(S), r)
        r2 = float(np.clip(phi @ r, 0.0, 1.0))
    except np.linalg.LinAlgError:
        r2 = 0.0
    return r2


def dominant_period(x: np.ndarray, pmin: int = 2, pmax: int | None = None) -> int:
    x = np.asarray(x, float)
    p = _periodogram(x)
    freqs = np.arange(1, p.size + 1)
    periods = x.size / freqs
    mask = (periods >= pmin)
    if pmax:
        mask &= (periods <= pmax)
    if not mask.any():
        return pmin
    k = np.argmax(np.where(mask, p, -np.inf))
    return int(round(periods[k]))


# --------------------------------------------------------------------------- #
# 3. Beyond-spectrum structure: analog vs linear prediction (Delta_nl)        #
# --------------------------------------------------------------------------- #

def _delay_embed(x: np.ndarray, d: int) -> tuple[np.ndarray, np.ndarray]:
    """Return delay vectors of dimension d and the next-step targets."""
    n = x.size - d
    V = np.stack([x[i:i + d] for i in range(n)], axis=0)
    y = x[d:d + n]
    return V, y


def _linear_onestep_mse(x: np.ndarray, d: int, split: float = 0.6) -> float:
    V, y = _delay_embed(x, d)
    k = int(len(y) * split)
    A = np.hstack([V[:k], np.ones((k, 1))])
    coef, *_ = np.linalg.lstsq(A, y[:k], rcond=None)
    At = np.hstack([V[k:], np.ones((len(y) - k, 1))])
    pred = At @ coef
    return float(np.mean((pred - y[k:]) ** 2))


def _analog_onestep_mse(x: np.ndarray, d: int, split: float = 0.6,
                        n_neighbors: int = 4) -> float:
    V, y = _delay_embed(x, d)
    k = int(len(y) * split)
    lib, libt = V[:k], y[:k]
    tree = cKDTree(lib)
    nn = min(n_neighbors, k)
    _, idx = tree.query(V[k:], k=nn)
    idx = np.asarray(idx)
    if idx.ndim == 1:                       # k=1 -> shape (Ntest,)
        pred = libt[idx]
    else:
        pred = libt[idx].mean(axis=1)
    return float(np.mean((pred - y[k:]) ** 2))


def _delay_embed_multi(x: np.ndarray, d: int, H: int) -> tuple[np.ndarray, np.ndarray]:
    """Delay vectors of dimension d and their next-H continuations."""
    n = x.size - d - H + 1
    V = np.stack([x[i:i + d] for i in range(n)], axis=0)
    Y = np.stack([x[d + i:d + i + H] for i in range(n)], axis=0)
    return V, Y


def _test_idx(n_test: int, max_queries: int = 4000) -> np.ndarray:
    """Deterministic stride over the test windows (same set for every predictor,
    so paired MSE ratios stay paired)."""
    if n_test <= max_queries:
        return np.arange(n_test)
    return np.linspace(0, n_test - 1, max_queries).astype(int)


def _linear_direct_mse(x: np.ndarray, d: int, H: int, split: float = 0.6) -> float:
    """Direct multi-step least squares: window(d) -> next H steps."""
    if H == 1:
        return _linear_onestep_mse(x, d, split)
    V, Y = _delay_embed_multi(x, d, H)
    k = int(len(Y) * split)
    A = np.hstack([V[:k], np.ones((k, 1))])
    coef, *_ = np.linalg.lstsq(A, Y[:k], rcond=None)
    ti = _test_idx(len(Y) - k)
    At = np.hstack([V[k:][ti], np.ones((ti.size, 1))])
    return float(np.mean((At @ coef - Y[k:][ti]) ** 2))


def _analog_direct_mse(x: np.ndarray, d: int, H: int, split: float = 0.6,
                       n_neighbors: int = 4) -> float:
    """Analog/simplex multi-step: kNN on the key, average the neighbours'
    H-step continuations (Sugihara-May, direct form)."""
    if H == 1:
        return _analog_onestep_mse(x, d, split, n_neighbors)
    V, Y = _delay_embed_multi(x, d, H)
    k = int(len(Y) * split)
    tree = cKDTree(V[:k])
    ti = _test_idx(len(Y) - k)
    _, idx = tree.query(V[k:][ti], k=min(n_neighbors, k))
    idx = np.atleast_2d(np.asarray(idx).T).T
    pred = Y[:k][idx].mean(axis=1)
    return float(np.mean((pred - Y[k:][ti]) ** 2))


def analog_gain(x: np.ndarray, d: int | None = None, split: float = 0.6,
                n_neighbors: int = 4, H: int = 1) -> float:
    """Delta_nl = 1 - MSE_analog / MSE_linear, the beyond-spectrum structure term.

    ~0 when the best predictor is linear (e.g. a Gaussian/IAAFT surrogate),
    large when recurring nonlinear motifs make analog forecasting win.
    Configuration-level: evaluate at the deployment horizon H (both predictors
    share the same embedding d and the same test windows).
    """
    x = np.asarray(x, float)
    if d is None:
        d = max(4, min(32, dominant_period(x)))
    mse_lin = _linear_direct_mse(x, d, H, split)
    mse_ana = _analog_direct_mse(x, d, H, split, n_neighbors)
    if mse_lin <= 0:
        return 0.0
    return float(np.clip(1.0 - mse_ana / mse_lin, 0.0, 1.0))


def motif_length(x: np.ndarray, pmin: int = 4, pmax: int = 1024) -> int:
    """Motif/state length m for the coverage term.

    Two-step, trend-robust: (1) the periodogram of the FIRST DIFFERENCES gives a
    trend-free period estimate m0 (differencing removes the level/trend power
    that pins the raw peak to the lowest frequency on real benchmarks), but it
    can land on a harmonic of the true cycle; (2) snap to the integer multiple
    k*m0 (<= pmax) whose neighbourhood carries the most power in the ORIGINAL
    periodogram -- the fundamental. Trend power sits at periods >> pmax, so it
    cannot capture the search."""
    x = np.asarray(x, float)
    n = x.size
    pmax = min(pmax, n // 4)
    m0 = dominant_period(np.diff(x), pmin=pmin, pmax=pmax)
    p = _periodogram(x)
    best_m, best_pow = m0, -1.0
    k = 1
    while k * m0 <= pmax:
        m = k * m0
        j = int(round(n / m)) - 1          # frequency bin for period m (DC dropped)
        if 0 <= j < p.size:
            w = float(p[max(0, j - 1): j + 2].max())
            if w > best_pow:
                best_m, best_pow = m, w
        k += 1
    return int(best_m)


# --------------------------------------------------------------------------- #
# 4. Coverage deficit Gamma                                                   #
# --------------------------------------------------------------------------- #

def coverage_u(S: int, m: int) -> float:
    """Operating-point term: fraction of the state-identifying motif (length m)
    that a window of length S does not observe."""
    if m <= 0:
        return 0.0
    return float(max(0.0, (m - S) / m))


def gamma_coverage(x: np.ndarray, S: int, m: int | None = None,
                   d: int | None = None, H: int = 1) -> float:
    """Gamma_cov = Delta_nl * u(S). Large only when beyond-spectrum structure
    exists AND the window is too short to reach it. Configuration-level:
    read at the operating point (S, H)."""
    x = np.asarray(x, float)
    if m is None:
        m = dominant_period(x)
    return analog_gain(x, d=d, H=H) * coverage_u(S, m)


def symbolic_oov_rate(train: np.ndarray, deploy: np.ndarray, S: int,
                      n_bins: int = 8, tuple_len: int = 3) -> float:
    """Gamma_oov: fraction of deploy-window symbol tuples absent from the
    training index (memory staleness). High => retrieval cannot help."""
    edges = np.quantile(train, np.linspace(0, 1, n_bins + 1)[1:-1])
    def symbolize(a):
        return np.digitize(a, edges)
    st, sd = symbolize(train), symbolize(deploy)
    def tuples(s):
        return {tuple(s[i:i + tuple_len]) for i in range(len(s) - tuple_len + 1)}
    train_idx = tuples(st)
    dep = [tuple(sd[i:i + tuple_len]) for i in range(len(sd) - tuple_len + 1)]
    if not dep:
        return 0.0
    miss = sum(t not in train_idx for t in dep)
    return float(miss / len(dep))


# --------------------------------------------------------------------------- #
# 5. Context value of a beyond-spectrum mechanism (analog retrieval)          #
# --------------------------------------------------------------------------- #

def context_value_analog(x: np.ndarray, S: int, d_context: int | None = None,
                         split: float = 0.6, n_neighbors: int = 4,
                         H: int = 1, m: int | None = None) -> float:
    """V = (MSE_base - MSE_context)/MSE_base at the operating point (S, H).

    base    : a within-window linear predictor of the next H steps from S lags.
    context : analog retrieval that matches a longer key (reaching memory
              beyond the window) and copies the neighbours' continuations.
    Positive when reaching beyond the window via similarity pays off. Both
    predictors are scored on the same test windows (paired).
    """
    x = np.asarray(x, float)
    if d_context is None:
        mm = m if m is not None else motif_length(x)
        d_context = int(min(max(S + 1, 2 * mm), 168))
    mse_base = _linear_direct_mse(x, S, H, split)
    mse_ctx = _analog_direct_mse(x, d_context, H, split, n_neighbors)
    if mse_base <= 0:
        return 0.0
    return float((mse_base - mse_ctx) / mse_base)


# --------------------------------------------------------------------------- #
# 6. Bootstrap                                                                #
# --------------------------------------------------------------------------- #

def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05,
                 rng: np.random.Generator | None = None) -> tuple[float, float, float]:
    rng = rng or np.random.default_rng(0)
    values = np.asarray(values, float)
    means = [rng.choice(values, values.size, replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(values.mean()), float(lo), float(hi)


# --------------------------------------------------------------------------- #
# 7. Synthetic generators                                                     #
# --------------------------------------------------------------------------- #

def motif_grammar(T: int, n_motifs: int = 8, motif_len: int = 24,
                  noise: float = 0.05, rng: np.random.Generator | None = None) -> np.ndarray:
    """Nonlinear structure spread over a motif longer than a short window.

    A small alphabet of smooth templates (length motif_len) is emitted in a
    DETERMINISTIC-CHAOTIC order. Predicting across a motif boundary needs (a)
    enough context to identify the current motif (so u(S)>0 for S<motif_len)
    and (b) the nonlinear transition rule (so analog beats linear). An IAAFT
    surrogate keeps the spectrum but scrambles the motifs into noise, so the
    structure, and the value of context, vanish.
    """
    rng = rng or np.random.default_rng(0)
    base_t = np.linspace(0, 2 * np.pi, motif_len, endpoint=False)
    templates = []
    for _ in range(n_motifs):
        c, s = rng.standard_normal(4), rng.standard_normal(4)
        temp = sum(c[h] * np.sin((h + 1) * base_t) + s[h] * np.cos((h + 1) * base_t)
                   for h in range(4))
        templates.append((temp - temp.mean()) / (temp.std() + 1e-8))
    templates = np.asarray(templates)
    seq, idx, xc = [], int(rng.integers(n_motifs)), float(rng.uniform(0.1, 0.9))
    for _ in range(T // motif_len + 2):
        seq.append(idx)
        xc = 4.0 * xc * (1.0 - xc)                  # logistic chaos (hidden state)
        idx = int(xc * n_motifs) % n_motifs         # nonlinear next-motif rule
    out = np.concatenate([templates[i] for i in seq])[:T]
    return out + noise * rng.standard_normal(out.size)


def henon(T: int, a: float = 1.4, b: float = 0.3, burn: int = 1000,
          rng: np.random.Generator | None = None) -> np.ndarray:
    """Henon map x-coordinate: deterministic chaos, broadband spectrum, strong
    one-step nonlinear (analog) predictability that no global linear model has."""
    rng = rng or np.random.default_rng(0)
    x, y = rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1)
    out = []
    for _ in range(T + burn):
        x, y = 1.0 - a * x * x + y, b * x
        out.append(x)
    return np.asarray(out[burn:], float)


def noisy_sine(T: int, period: int = 24, noise: float = 0.15,
               rng: np.random.Generator | None = None) -> np.ndarray:
    """Linear/second-order control: perfectly predictable but LINEARLY, so the
    analog gain is ~0 and the diagnostic correctly reports no beyond-spectrum
    value to import."""
    rng = rng or np.random.default_rng(0)
    t = np.arange(T)
    x = np.sin(2 * np.pi * t / period) + 0.4 * np.sin(2 * np.pi * t / (period / 2))
    return x + noise * rng.standard_normal(T)


# --------------------------------------------------------------------------- #
# 8. Smoke test (miniature E1)                                                #
# --------------------------------------------------------------------------- #

def smoke_test(seed: int = 0):
    rng = np.random.default_rng(seed)
    S = 8
    cases = [
        ("motif-grammar (m=24)", motif_grammar(7200, motif_len=24, rng=rng), 24),
        ("motif-grammar (m=36)", motif_grammar(7200, n_motifs=6, motif_len=36, rng=rng), 36),
        ("Henon map (nonlinear)", henon(7000, rng=rng), None),
        ("white noise (control)", rng.standard_normal(7000), None),
    ]
    print("=" * 90)
    print(f"SMOKE TEST (miniature E1):  window S = {S};  mechanism = analog retrieval")
    print("Claim: Omega is ~identical across the IAAFT surrogate pair, while the")
    print("       structure term Delta_nl, the diagnostic Gamma, and the measured")
    print("       context value V collapse on the surrogate.")
    print("=" * 90)
    print(f"{'series':<26}{'Omega x/x~':>14}{'Delta_nl x/x~':>16}"
          f"{'Gamma x/x~':>14}{'V x/x~':>16}{'iaaft res':>10}")
    print("-" * 90)
    for name, x, m in cases:
        xs = iaaft(x, rng)
        om_x, om_s = omega_spectral_predictability(x), omega_spectral_predictability(xs)
        dnl_x, dnl_s = analog_gain(x), analog_gain(xs)
        g_x = gamma_coverage(x, S, m=m)
        g_s = gamma_coverage(xs, S, m=m)
        v_x = context_value_analog(x, S)
        v_s = context_value_analog(xs, S)
        print(f"{name:<26}"
              f"{om_x:>6.3f}/{om_s:<6.3f}"
              f"{dnl_x:>7.3f}/{dnl_s:<7.3f}"
              f"{g_x:>6.3f}/{g_s:<6.3f}"
              f"{v_x:>7.3f}/{v_s:<7.3f}"
              f"{iaaft_residual(x, xs):>9.3f}")
    print("-" * 90)
    print("Read: in the nonlinear rows, Omega is ~unchanged (periodogram preserved)")
    print("while Delta_nl, Gamma, and V collapse toward 0 on the surrogate x~.")
    print("In the linear control row, Delta_nl/Gamma/V are ~0 on BOTH x and x~:")
    print("the diagnostic does not fire where there is no beyond-spectrum value.")
    print("=" * 90)
    diffs = []
    for s in range(8):
        rg = np.random.default_rng(100 + s)
        xx = motif_grammar(4800, motif_len=24, rng=rg)
        diffs.append(gamma_coverage(xx, S, m=24) - gamma_coverage(iaaft(xx, rg), S, m=24))
    mn, lo, hi = bootstrap_ci(np.array(diffs))
    print(f"Gamma(x) - Gamma(x~) over 8 seeds: mean {mn:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
    print("(Expected strictly positive: structure present in x, absent in x~.)")


if __name__ == "__main__":
    smoke_test()