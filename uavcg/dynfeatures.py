"""Group-B dynamic dual features (spec §2.2), computed from a WarmupTrace.

The trace carries exactly the first W iterations' duals / reduced costs /
columns, so these features physically cannot see past the warm-up.  All
matrices are ``(n+1, n+1)`` indexed by node, following the solver's own
convention: depot = 0, customers = 1..n, reduced cost matrix
``c~_ij = c_ij - pi_j`` (the tail-attribution form, R-6 -- the trace's
``reduced_cost`` entries are built by `pricers.base.modified_cost`, never
recomputed here).

Feature layout (B1..B12, order fixed):
    B1  mean_t c~          B7  mean_t theta_i
    B2  min_t  c~          B8  mean_t theta_j
    B3  std_t  c~          B9  std_t  theta_i
    B4  frac_t c~ < 0      B10 std_t  theta_j
    B5  first t with c~<0  B11 slope of (t, c~)
         (W+1 if never,
          / (W+1))         B12 warm-up use count
    B6  mean over last ceil(W/2)

theta is the node dual with the depot folded to sigma (spec §2.2:
theta_0 = theta_{n+1} = sigma, theta_k = pi_k for customers).
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from .cg import WarmupTrace

B_FEATURE_NAMES = [
    "rc_mean", "rc_min", "rc_std", "rc_neg_frac", "rc_first_neg",
    "rc_tail_mean", "theta_i_mean", "theta_j_mean", "theta_i_std",
    "theta_j_std", "rc_slope", "warm_use_count",
]


def _theta_matrices(trace: WarmupTrace, n: int) -> np.ndarray:
    """(W, n+1) node duals: theta[t, 0] = sigma[t], theta[t, k] = pi[t][k-1]."""
    W = trace.W
    th = np.empty((W, n + 1), dtype=np.float64)
    for t in range(W):
        th[t, 0] = trace.sigma[t]
        th[t, 1:] = np.asarray(trace.pi[t], dtype=np.float64)
    return th


def compute_b_features(trace: WarmupTrace, n: int) -> Dict[str, np.ndarray]:
    """All twelve group-B features as named (n+1, n+1) matrices."""
    W = trace.W
    if W < 2:
        raise ValueError(f"need W >= 2 warm-up iterations, got {W}")
    rc = np.stack([np.asarray(m, dtype=np.float64)
                   for m in trace.reduced_cost])          # (W, n+1, n+1)
    if rc.shape[1:] != (n + 1, n + 1):
        raise ValueError(f"reduced_cost shape {rc.shape[1:]}, "
                         f"expected {(n + 1, n + 1)}")
    th = _theta_matrices(trace, n)                        # (W, n+1)

    neg = rc < 0.0
    first_neg = np.full((n + 1, n + 1), W + 1, dtype=np.float64)
    any_neg = neg.any(axis=0)
    first_neg[any_neg] = np.argmax(neg, axis=0)[any_neg] + 1  # t is 1-based
    first_neg /= (W + 1)

    tail = rc[-((W + 1) // 2):]                           # last ceil(W/2)
    tt = np.arange(1, W + 1, dtype=np.float64)
    tc = tt - tt.mean()
    slope = (tc[:, None, None] * rc).sum(axis=0) / (tc * tc).sum()

    return {
        "rc_mean": rc.mean(axis=0),
        "rc_min": rc.min(axis=0),
        "rc_std": rc.std(axis=0),
        "rc_neg_frac": neg.mean(axis=0),
        "rc_first_neg": first_neg,
        "rc_tail_mean": tail.mean(axis=0),
        "theta_i_mean": np.repeat(th.mean(axis=0)[:, None], n + 1, axis=1),
        "theta_j_mean": np.repeat(th.mean(axis=0)[None, :], n + 1, axis=0),
        "theta_i_std": np.repeat(th.std(axis=0)[:, None], n + 1, axis=1),
        "theta_j_std": np.repeat(th.std(axis=0)[None, :], n + 1, axis=0),
        "rc_slope": slope,
        "warm_use_count": trace.arc_use_counts(n).astype(np.float64),
    }


def b_feature_rows(trace: WarmupTrace, n: int,
                   arcs: np.ndarray) -> np.ndarray:
    """(len(arcs), 12) feature rows for the given (i, j) arc array."""
    feats = compute_b_features(trace, n)
    ii = arcs[:, 0].astype(int)
    jj = arcs[:, 1].astype(int)
    return np.column_stack([feats[name][ii, jj]
                            for name in B_FEATURE_NAMES])
