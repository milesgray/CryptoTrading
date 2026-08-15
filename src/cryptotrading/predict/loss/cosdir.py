"""
CosDir / CosDir-UW: Direction-Aware Loss for Time Series Forecasting
======================================================================
Implementation of the loss proposed in:
    "Beyond Magnitude and Shape: A Direction-Aware Loss for Time Series
    Forecasting" (Lee et al., LG AI Research, arXiv:2608.01857)

Core idea
---------
Standard magnitude losses (MSE/MAE) have gradient ∝ (ŷ - y), so small-
amplitude moves get almost no gradient and their *direction* is often
wrong even when overall error is low. CosDir fixes this by comparing the
horizon-length first-difference vectors of prediction and target via
cosine similarity — a term that depends only on orientation, not scale,
so it keeps a directional gradient even on tiny moves.

    Δŷ_c = first differences of the prediction, over the horizon, channel c
    Δy_c = first differences of the target,     over the horizon, channel c

    L_CosDir  = L_MSE + λ · (1/C) Σ_c [ 1 - cos(Δŷ_c, Δy_c) ]         (Eq. 2)

    L_CosDir-UW = e^{-s1}·L_MSE + e^{-s2}·L_dir + 0.5·(s1 + s2)        (Eq. 3)

where s1, s2 are learned log-variance (uncertainty) parameters and
λ_eff = e^{s1 - s2} is the ratio the optimizer discovers automatically
(no λ hyperparameter to tune; paper reports this matches a per-dataset
tuned fixed λ, and beats any single fixed λ on average).

Notes on this implementation
-----------------------------
- Fully vectorized (no python loop over channels/horizon).
- Supports two ways of forming the first-difference vector at h=1:
    (a) pass `y_last` (the last observed value before the forecast
        window) so the first horizon step's direction is included
        (this is what the paper does, Sec. "Preliminaries": y_0,c is
        set to the last observed input value x_{L,c}).
    (b) omit `y_last` -> uses only the H-1 consecutive differences
        *within* the predicted window (drops the "did we get the very
        first jump right" signal, but needs no extra context tensor).
- `reduction='mean'|'sum'|'none'` matches the base MSE reduction.
- Numerically safe: eps stabilizer in the cosine denominator (paper's ϵ),
  and diff vectors that are exactly zero (flat target window) contribute
  ~0 directional gradient rather than NaN.

Example
-------
    loss_fn = CosDirLoss(lambda_=0.5)
    loss, logs = loss_fn(y_hat, y_true, y_last=x[:, -1, :])
    loss.backward()

    # or the hyperparameter-free variant:
    loss_fn = CosDirUWLoss()
    ...
    loss, logs = loss_fn(y_hat, y_true, y_last=x[:, -1, :])
    print(logs['lambda_eff'])   # what the model is currently weighting direction at
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _first_differences(
    y: torch.Tensor,
    y_last: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute horizon-wise first differences.

    Args:
        y: (B, H, C) tensor — predictions or targets.
        y_last: optional (B, C) tensor, the last observed value before the
            forecast window (x_{L,c} in the paper). If given, the returned
            tensor has shape (B, H, C) with step 0 = y[:,0] - y_last.
            If omitted, returns (B, H-1, C): consecutive diffs within y only.

    Returns:
        Δy of shape (B, H, C) if y_last is given, else (B, H-1, C).
    """
    if y_last is not None:
        prev = torch.cat([y_last.unsqueeze(1), y[:, :-1, :]], dim=1)  # (B, H, C)
        return y - prev
    return y[:, 1:, :] - y[:, :-1, :]  # (B, H-1, C)


def cosdir_directional_term(
    y_hat: torch.Tensor,
    y_true: torch.Tensor,
    y_last: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Standalone computation of the CosDir directional penalty
    L_dir = (1/C) * sum_c (1 - cos(Δŷ_c, Δy_c))   [Eq. 2's second term, unweighted]

    Args:
        y_hat, y_true: (B, H, C) prediction and target.
        y_last: optional (B, C) last observed value; see `_first_differences`.
        eps: cosine-denominator stabilizer (paper's ϵ).
        reduction: 'mean' | 'sum' | 'none'. 'none' returns per-sample (B,).

    Returns:
        Scalar (or (B,) if reduction='none') directional loss, in [0, 2].
    """
    d_hat = _first_differences(y_hat, y_last)   # (B, H', C)
    d_true = _first_differences(y_true, y_last)  # (B, H', C)

    # Cosine similarity along the horizon axis (dim=1), per (batch, channel).
    num = (d_hat * d_true).sum(dim=1)                       # (B, C)
    denom = d_hat.norm(dim=1) * d_true.norm(dim=1) + eps     # (B, C)
    cos_sim = num / denom                                    # (B, C)

    per_sample = (1.0 - cos_sim).mean(dim=1)  # mean over channels -> (B,)

    if reduction == "mean":
        return per_sample.mean()
    elif reduction == "sum":
        return per_sample.sum()
    elif reduction == "none":
        return per_sample
    else:
        raise ValueError(f"Unknown reduction: {reduction!r}")


class CosDirLoss(nn.Module):
    """
    CosDir: MSE + λ · directional cosine-alignment term (Eq. 2).

    Args:
        lambda_: weight on the directional term. Paper default 0.5;
            best fixed λ is dataset-dependent, ranging ~0.3-3.0
            (see Appendix H/L of the paper) — sweep if you can, or use
            CosDirUWLoss to avoid tuning this at all.
        base_loss: 'mse' or 'mae'. Paper shows CosDir is agnostic to the
            base loss (Appendix F): both benefit.
        eps: cosine-denominator stabilizer.
    """

    def __init__(
        self,
        lambda_: float = 0.5,
        base_loss: str = "mse",
        eps: float = 1e-8,
    ):
        super().__init__()
        if base_loss not in ("mse", "mae"):
            raise ValueError("base_loss must be 'mse' or 'mae'")
        self.lambda_ = lambda_
        self.base_loss = base_loss
        self.eps = eps

    def forward(
        self,
        y_hat: torch.Tensor,
        y_true: torch.Tensor,
        y_last: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            y_hat, y_true: (B, H, C) prediction and target.
            y_last: optional (B, C) last observed value before the forecast
                window, to include the first horizon step's direction.

        Returns:
            (total_loss, logs) where logs has 'mse'/'mae', 'dir', 'lambda'.
        """
        if self.base_loss == "mse":
            base = F.mse_loss(y_hat, y_true)
        else:
            base = F.l1_loss(y_hat, y_true)

        dir_term = cosdir_directional_term(y_hat, y_true, y_last=y_last, eps=self.eps)

        total = base + self.lambda_ * dir_term
        logs = {
            self.base_loss: base.detach(),
            "dir": dir_term.detach(),
            "lambda": torch.as_tensor(self.lambda_),
            "total": total.detach(),
        }
        return total, logs


class CosDirUWLoss(nn.Module):
    """
    CosDir-UW: learns the magnitude/direction balance via homoscedastic
    uncertainty weighting (Kendall, Gal & Cipolla 2018), Eq. 3:

        L = e^{-s1}·L_MSE + e^{-s2}·L_dir + 0.5·(s1 + s2)

    s1, s2 are trainable scalar log-variance parameters — register this
    module so its parameters are included in your optimizer
    (e.g. `optimizer = Adam(list(model.parameters()) + list(loss_fn.parameters()))`,
    or just put the loss module as a submodule of your model).

    No λ hyperparameter: the effective directional weight
        λ_eff = e^{s1 - s2}
    is discovered automatically and empirically tracks a per-dataset
    tuned fixed λ (paper Fig. 6, Spearman ρ=0.60), while matching or
    beating the best fixed λ on both DA and MSE (paper Table 8).

    Args:
        base_loss: 'mse' or 'mae'.
        eps: cosine-denominator stabilizer.
        init_s1, init_s2: initial log-variances. 0.0 means both terms
            start with precision 1.0 (i.e. λ_eff starts at 1.0).
    """

    def __init__(
        self,
        base_loss: str = "mse",
        eps: float = 1e-8,
        init_s1: float = 0.0,
        init_s2: float = 0.0,
    ):
        super().__init__()
        if base_loss not in ("mse", "mae"):
            raise ValueError("base_loss must be 'mse' or 'mae'")
        self.base_loss = base_loss
        self.eps = eps
        self.s1 = nn.Parameter(torch.tensor(float(init_s1)))
        self.s2 = nn.Parameter(torch.tensor(float(init_s2)))

    @property
    def lambda_eff(self) -> torch.Tensor:
        """Effective directional weight e^{s1 - s2}, read-only convenience."""
        return torch.exp(self.s1 - self.s2).detach()

    def forward(
        self,
        y_hat: torch.Tensor,
        y_true: torch.Tensor,
        y_last: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if self.base_loss == "mse":
            base = F.mse_loss(y_hat, y_true)
        else:
            base = F.l1_loss(y_hat, y_true)

        dir_term = cosdir_directional_term(y_hat, y_true, y_last=y_last, eps=self.eps)

        precision1 = torch.exp(-self.s1)
        precision2 = torch.exp(-self.s2)
        total = precision1 * base + precision2 * dir_term + 0.5 * (self.s1 + self.s2)

        logs = {
            self.base_loss: base.detach(),
            "dir": dir_term.detach(),
            "s1": self.s1.detach(),
            "s2": self.s2.detach(),
            "lambda_eff": self.lambda_eff,
            "total": total.detach(),
        }
        return total, logs


def directional_accuracy(
    y_hat: torch.Tensor,
    y_true: torch.Tensor,
    y_last: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    DA metric from the paper (Eq. 1) — NOT a loss (non-differentiable,
    uses sign()), use only for evaluation/logging.

        DA = (1/HC) * sum_{h,c} 1[sign(Δŷ_h,c) == sign(Δy_h,c)]

    Returns a scalar in [0, 1], with 0.5 = chance.
    """
    d_hat = _first_differences(y_hat, y_last)
    d_true = _first_differences(y_true, y_last)
    agree = (torch.sign(d_hat) == torch.sign(d_true)).float()
    return agree.mean()


# ----------------------------------------------------------------------------
# Self-test / demo
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    B, H, C = 32, 24, 8
    y_true = torch.randn(B, H, C).cumsum(dim=1)
    y_last = torch.zeros(B, C)  # last observed value before horizon

    # A "bad" prediction: right-ish magnitude, wrong small-move directions
    y_hat_bad = y_true + torch.randn(B, H, C) * 0.5
    # A "good" prediction: same MSE budget, but nudged toward correct direction
    y_hat_good = y_true + 0.1 * torch.sign(y_true - y_true.roll(1, dims=1))

    cosdir = CosDirLoss(lambda_=0.5)
    loss_bad, logs_bad = cosdir(y_hat_bad, y_true, y_last=y_last)
    loss_good, logs_good = cosdir(y_hat_good, y_true, y_last=y_last)

    print("=== CosDir (fixed lambda=0.5) ===")
    print(f"bad : total={loss_bad.item():.4f}  mse={logs_bad['mse'].item():.4f}  "
          f"dir={logs_bad['dir'].item():.4f}  "
          f"DA={directional_accuracy(y_hat_bad, y_true, y_last).item():.3f}")
    print(f"good: total={loss_good.item():.4f}  mse={logs_good['mse'].item():.4f}  "
          f"dir={logs_good['dir'].item():.4f}  "
          f"DA={directional_accuracy(y_hat_good, y_true, y_last).item():.3f}")

    # Quick gradient check + a few optimization steps with CosDir-UW
    print("\n=== CosDir-UW: a few training steps ===")
    pred = nn.Parameter(y_hat_bad.clone())
    uw = CosDirUWLoss()
    opt = torch.optim.Adam([pred] + list(uw.parameters()), lr=0.05)
    for step in range(200):
        opt.zero_grad()
        loss, logs = uw(pred, y_true, y_last=y_last)
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == 199:
            da = directional_accuracy(pred, y_true, y_last).item()
            print(f"step {step:3d}  total={loss.item():.4f}  mse={logs['mse'].item():.4f}  "
                  f"dir={logs['dir'].item():.4f}  lambda_eff={logs['lambda_eff'].item():.3f}  "
                  f"DA={da:.3f}")