from __future__ import annotations
import numpy as np
from numpy.fft import rfft

def bicoherence_index(x, seg=64, step=32):
    """Segment-averaged squared bicoherence, summed to a scalar in [0,1].
    b^2(f1,f2) = |<X1 X2 X12*>|^2 / (<|X1 X2|^2> <|X12|^2>); index = mean over the
    non-redundant triangle f1>=f2, f1+f2<=seg/2."""
    x = np.asarray(x, float)
    n = x.size
    if n < seg * 4:
        return 0.0
    win = np.hanning(seg)
    starts = range(0, n - seg + 1, step)
    F = np.array([rfft((x[s:s + seg] - x[s:s + seg].mean()) * win) for s in starts])
    if len(F) < 8:
        return 0.0
    nf = seg // 2
    num = np.zeros((nf, nf), complex)
    d1 = np.zeros((nf, nf)); d2 = np.zeros((nf, nf))
    for f1 in range(nf):
        for f2 in range(f1 + 1):
            if f1 + f2 >= nf:
                continue
            tri = F[:, f1] * F[:, f2] * np.conj(F[:, f1 + f2])
            num[f1, f2] = tri.mean()
            d1[f1, f2] = (np.abs(F[:, f1] * F[:, f2]) ** 2).mean()
            d2[f1, f2] = (np.abs(F[:, f1 + f2]) ** 2).mean()
    denom = d1 * d2
    mask = denom > 1e-20
    b2 = np.zeros_like(d1)
    b2[mask] = (np.abs(num[mask]) ** 2) / denom[mask]
    return float(b2[mask].mean()) if mask.any() else 0.0

def permutation_entropy(x, order=4, delay=1):
    x = np.asarray(x, float)
    n = x.size - (order - 1) * delay
    if n <= 0:
        return 1.0
    # ordinal patterns
    patt = {}
    from math import factorial, log
    for i in range(n):
        window = x[i:i + order * delay:delay]
        key = tuple(np.argsort(window, kind="stable"))
        patt[key] = patt.get(key, 0) + 1
    counts = np.array(list(patt.values()), float)
    p = counts / counts.sum()
    H = -(p * np.log(p)).sum()
    return float(H / log(factorial(order)))

def acc_law_complexity(x, P=96, F=96, step=96):
    """Accuracy-law window-wise pattern complexity (Wang et al. 2025,
    arXiv:2510.02729, Eq. 3): trace of the covariance of per-window amplitude
    spectra, Complexity = (1/N) sum_i ||A_i - Abar||^2, A_i = |FFT| of the i-th
    length-(P+F) window (phase discarded). A power-spectrum-family, marginal-
    reading index -> the impossibility (Cor. 1) predicts it cannot rank context
    value; E2/E4 tests that empirically."""
    x = np.asarray(x, float)
    W = P + F
    n = x.size - W + 1
    if n < 8:
        return 0.0
    starts = range(0, n, step)
    A = np.array([np.abs(rfft(x[s:s + W] - x[s:s + W].mean())) for s in starts])
    Abar = A.mean(axis=0)
    return float(np.mean(np.sum((A - Abar) ** 2, axis=1)))

def _ksg_mi(X, Y, k=3):
    """Kraskov-Stogberger-Grassberger (KSG, estimator 1) mutual information
    I(X;Y) for X in R^dx, Y in R^dy, via Chebyshev-metric kNN counts."""
    from scipy.special import digamma
    from sklearn.neighbors import NearestNeighbors
    n = len(X)
    XY = np.hstack([X, Y])
    dk, _ = NearestNeighbors(metric="chebyshev", n_neighbors=k + 1).fit(XY).kneighbors(XY)
    eps = dk[:, k]
    def cnt(Z):
        nn = NearestNeighbors(metric="chebyshev").fit(Z)
        idx = nn.radius_neighbors(Z, radius=eps, return_distance=False)  # inclusive
        return np.maximum(np.array([len(a) - 1 for a in idx]), 0)   # exclude self, clamp
    mi = digamma(k) + digamma(n) - np.mean(digamma(cnt(X) + 1) + digamma(cnt(Y) + 1))
    return max(float(mi), 0.0)

def catt_forecastability(x, p=3, k=3, cap=2000, seed=0):
    """Catt forecastability (arXiv:2603.27074): F = I(Y_{t+1}; past p lags),
    the mutual information between the one-step future and the length-p lag
    vector, KSG-estimated. A non-spectrum, series-level index (outside Cor. 1)."""
    x = np.asarray(x, float)
    n = x.size - p
    if n < 200:
        return 0.0
    Xlag = np.stack([x[i:i + p] for i in range(n)])   # (n, p) past
    y = x[p:p + n].reshape(-1, 1)                      # one-step future
    rng = np.random.default_rng(seed)
    if n > cap:                                        # subsample for KSG speed
        sel = rng.choice(n, cap, replace=False)
        Xlag, y = Xlag[sel], y[sel]
    # tiny jitter breaks exact ties (repeated/quantized values) that otherwise
    # give a zero kth-NN radius and a divergent KSG estimate (sklearn does this).
    Xlag = Xlag + 1e-10 * (Xlag.std() + 1e-12) * rng.standard_normal(Xlag.shape)
    y = y + 1e-10 * (y.std() + 1e-12) * rng.standard_normal(y.shape)
    return _ksg_mi(Xlag, y, k=k)
