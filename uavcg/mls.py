"""ML-S arm (spec §0.2 / §5): a learned static arc mask for column generation.

Offline: `train_rf_arc.py` trains a calibrated RF on the 22 static arc
features.  Online: before the solve starts (ML-S) every candidate arc is
scored once; arcs with calibrated probability >= tau are kept, depot arcs
are ALWAYS kept (they carry vehicle-count feasibility; scoring them is how
a mask learns to break the master -- see collect_arc_labels.candidate_arcs).

The mask is mounted through `CGConfig.mask_after` + `mask_hook`; the CG loop
itself owns the G/G_r switching (reduced_min/max_columns) and the
certification separation, both self-tested.  This module only produces the
boolean matrix -- it never touches the loop.

Feature provenance (R-6 lesson: one definition, one place): the static
block is obtained by calling collect_arc_labels.build_arc_features with a
zero dual and slicing the first 22 columns, so training and serving can
never drift apart on the static definitions.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np

from .cg import WarmupTrace
from .instance import Instance

N_STATIC = 22


def static_arc_table(inst: Instance) -> Tuple[np.ndarray, np.ndarray]:
    """(arcs (m,2) int, X (m,22)) for the candidate arc set of `inst`."""
    from scripts.collect_arc_labels import (FeatureContext,
                                            build_arc_features,
                                            candidate_arcs,
                                            feasible_arc_mask,
                                            per_node_stats)
    ctx = FeatureContext(inst)
    mask = feasible_arc_mask(inst)
    ii, jj = candidate_arcs(mask)
    stats = per_node_stats(inst, ii, jj)
    zero_pi = np.zeros(inst.n, dtype=float)
    X34 = build_arc_features(ctx, inst, zero_pi, 0.0, 0, 1, 0.0, ii, jj,
                             stats)
    arcs = np.column_stack([ii, jj]).astype(np.int64)
    return arcs, X34[:, :N_STATIC]


def load_mask_model(path: str | Path) -> dict:
    with Path(path).open("rb") as fh:
        bundle = pickle.load(fh)
    if "features" not in bundle:
        raise ValueError(f"{path}: not a trained arc-scorer bundle")
    return bundle


def predict_keep_mask(inst: Instance, bundle: dict,
                      tau: float = 0.5) -> np.ndarray:
    """(n+1, n+1) bool keep-mask: scored arcs above tau, depot arcs kept,
    everything static-infeasible dropped (the pricer never sees those arcs
    in a certificate either -- masking them is a no-op semantically)."""
    from scripts.collect_arc_labels import (candidate_arcs,
                                            feasible_arc_mask)
    arcs, X = static_arc_table(inst)
    p = bundle["calibrated"].predict_proba(X)[:, 1]
    keep = np.zeros((inst.n + 1, inst.n + 1), dtype=bool)
    keep[arcs[:, 0], arcs[:, 1]] = p >= tau
    keep[0, :] = True          # depot out-arcs: always kept
    keep[:, 0] = True          # depot return arcs: always kept
    keep &= feasible_arc_mask(inst) | np.eye(inst.n + 1, dtype=bool)
    keep[0, :] = True
    keep[:, 0] = True
    return keep


def make_mls_hook(inst: Instance, model_path: str | Path,
                  tau: float = 0.5
                  ) -> Callable[[WarmupTrace], Optional[np.ndarray]]:
    """A `CGConfig.mask_hook` that ignores the trace (ML-S is static) and
    returns the learned keep-mask.  Use with `mask_after=1` so the mask is
    mounted on the first well-defined dual, and the certification machinery
    sees the reduced network as what it is."""
    bundle = load_mask_model(model_path)
    keep = predict_keep_mask(inst, bundle, tau)

    def hook(trace: WarmupTrace) -> Optional[np.ndarray]:
        return keep

    return hook


# ---------------------------------------------------------------------------
# DynML-S: static + group-B dynamic features (spec §0.3, change 1; §2.2)
# ---------------------------------------------------------------------------

def _per_instance_minmax(X: np.ndarray) -> np.ndarray:
    """§2.3: normalise every column on this instance's own min/max."""
    lo = X.min(axis=0)
    hi = X.max(axis=0)
    span = np.where(hi - lo > 1e-12, hi - lo, 1.0)
    return (X - lo) / span


def predict_keep_mask_dynmls(inst: Instance, trace: "WarmupTrace",
                             bundle: dict, tau: float = 0.5) -> np.ndarray:
    """(n+1, n+1) bool keep-mask from static(22) + B(12) features.

    The B block is computed from the warm-up trace via dynfeatures -- the
    same code build_dynmls_dataset.py uses to assemble training rows, so the
    feature definitions cannot drift between training and serving.
    """
    from uavcg.dynfeatures import b_feature_rows
    from scripts.collect_arc_labels import candidate_arcs, feasible_arc_mask
    if bundle.get("features") != "all":
        raise ValueError(f"DynML-S needs an 'all'-feature model, got "
                         f"{bundle.get('features')!r}")
    arcs, X_static = static_arc_table(inst)
    X_B = b_feature_rows(trace, inst.n, arcs)
    X = np.column_stack([X_static, X_B])
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    X = _per_instance_minmax(X)
    p = bundle["calibrated"].predict_proba(X)[:, 1]
    keep = np.zeros((inst.n + 1, inst.n + 1), dtype=bool)
    keep[arcs[:, 0], arcs[:, 1]] = p >= tau
    keep[0, :] = True
    keep[:, 0] = True
    keep &= feasible_arc_mask(inst) | np.eye(inst.n + 1, dtype=bool)
    keep[0, :] = True
    keep[:, 0] = True
    return keep


def make_dynmls_hook(inst: Instance, model_path: str | Path,
                     tau: float = 0.5
                     ) -> "Callable[[WarmupTrace], Optional[np.ndarray]]":
    """A `CGConfig.mask_hook` for DynML-S: static+B features from the live
    warm-up trace.  Use with `mask_after=W` (spec §1, W=20)."""
    bundle = load_mask_model(model_path)

    def hook(trace: "WarmupTrace") -> Optional[np.ndarray]:
        if trace.W < 2:
            # not enough warm-up to compute B features -- fall back to no mask
            # (full network) rather than crash; the certification machinery
            # handles the rest.
            return None
        return predict_keep_mask_dynmls(inst, trace, bundle, tau)

    return hook
