#!/usr/bin/env python
"""Build the DynML-S training dataset: static(22) + B(12) features per arc.

For each instance in an arc_labels npz that carries pi_logs/sigma_logs (the
new collector format), this script:
  1. loads the instance to get its static arc table (22 features, via mls);
  2. constructs a WarmupTrace from the first W dual iterations;
  3. computes B1..B12 via dynfeatures (the SAME code the online hook uses,
     so training and serving cannot drift on the B definitions);
  4. emits one row per candidate arc: [22 static | 12 B] + y_narrow + y_run.

Label: y_narrow ("used in the last 50% of iterations") is recomputed from
y_arc_iter per instance, matching the ML-S narrowed label (§26.8).

Usage:
    python scripts/build_dynmls_dataset.py --in results/arc_labels_n25b.npz \
        --out results/dynmls_n25b.npz --w 20
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uavcg.bench import load_instance
from uavcg.cg import WarmupTrace
from uavcg.dynfeatures import B_FEATURE_NAMES, b_feature_rows
from uavcg.pricers.base import modified_cost
from uavcg import mls

# static feature names from the collector (single source of truth, R-6)
from scripts.collect_arc_labels import FEATURE_NAMES as STATIC_FEATURE_NAMES


def narrow_label(arcs, iters, yK, groups):
    """Per-row narrowed label: arc used in the last 50% of its instance's iters."""
    y = np.zeros(len(arcs), dtype=np.int64)
    for ins in np.unique(groups):
        m = groups == ins
        a, ii, yk = arcs[m], iters[m], yK[m]
        maxit = ii.max()
        last = ii > maxit / 2
        dd = defaultdict(bool)
        for k in np.flatnonzero(last):
            dd[(int(a[k, 0]), int(a[k, 1]))] |= bool(yk[k])
        y[m] = [dd.get((int(a[k, 0]), int(a[k, 1])), False) for k in range(m.sum())]
    return y


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--w", type=int, default=20)
    ap.add_argument("--n", type=int, default=25)
    args = ap.parse_args()

    d = np.load(args.inp, allow_pickle=True)
    has_traj = "pi_logs" in d.files and "sigma_logs" in d.files
    if not has_traj:
        raise SystemExit(f"{args.inp}: no pi_logs/sigma_logs (need the new "
                         f"collector format); re-collect with the current "
                         f"collect_arc_labels.py")
    inst_names = [str(s) for s in d["instances"]]
    pi_logs = d["pi_logs"]
    sigma_logs = d["sigma_logs"]
    # per-row instance (the new collector stores instances per-row when
    # collected in one batch; fall back to iteration-reset reconstruction)
    inst_arr = d["instances"]
    if inst_arr.ndim == 1 and len(inst_arr) == len(d["X"]):
        groups = np.asarray(inst_arr, dtype=str)
    else:
        iters = d["iters"]
        cuts = np.flatnonzero(np.diff(iters) < 0)
        groups = np.empty(len(iters), dtype=object)
        names = [str(s) for s in inst_arr]
        prev = 0
        for k, c in enumerate(cuts):
            groups[prev:c] = names[k]; prev = c
        groups[prev:] = names[-1]

    # narrowed label (recomputed from y_arc_iter to match §26.8)
    y_narrow = narrow_label(d["arcs"], d["iters"], d["y_arc_iter"], groups)

    all_X, all_arcs, all_y, all_inst = [], [], [], []
    for idx, ins_name in enumerate(inst_names):
        pi_l = pi_logs[idx]      # (iters_i, n)
        sg_l = sigma_logs[idx]   # (iters_i,)
        iters_i = pi_l.shape[0]
        W = min(args.w, iters_i)
        if W < 2:
            print(f"  skip {ins_name}: only {iters_i} iters (need >=2 for B)",
                  flush=True)
            continue
        inst = load_instance(ins_name, n=args.n)
        static_arcs, static_X = mls.static_arc_table(inst)  # (m,2),(m,22)
        # build the warm-up trace from the stored duals
        rc = [modified_cost(inst, pi_l[t]) for t in range(W)]
        # columns: not available offline (collector stored route tuples per
        # iteration in the iter json, not in the npz).  B12 (warm_use_count)
        # needs them; use the iter-log columns if present, else zero (B12=0).
        cols = [[] for _ in range(W)]
        trace = WarmupTrace(
            pi=[pi_l[t] for t in range(W)],
            sigma=[float(sg_l[t]) for t in range(W)],
            reduced_cost=rc,
            columns=cols,
            lp_objective=[0.0] * W,
        )
        B = b_feature_rows(trace, inst.n, static_arcs)  # (m, 12)
        X = np.column_stack([static_X, B])               # (m, 34)
        # label: match static_arcs to the instance's rows in the npz
        m = groups == ins_name
        arc_to_y = {}
        for k in np.flatnonzero(m):
            arc_to_y[(int(d["arcs"][k, 0]), int(d["arcs"][k, 1]))] = int(y_narrow[k])
        y = np.array([arc_to_y.get((int(a[0]), int(a[1])), 0)
                      for a in static_arcs], dtype=np.int64)
        all_X.append(X); all_arcs.append(static_arcs)
        all_y.append(y); all_inst.append([ins_name] * len(static_arcs))
        print(f"  {ins_name}: W={W} iters={iters_i} arcs={len(static_arcs)} "
              f"pos={int(y.sum())}/{len(y)} ({y.mean():.3f})", flush=True)

    X = np.vstack(all_X)
    arcs = np.vstack(all_arcs)
    y = np.concatenate(all_y)
    inst = np.array([s for sub in all_inst for s in sub], dtype=object)
    np.savez_compressed(
        args.out, X=X, arcs=arcs, y_narrow=y, instances=inst,
        feature_names=np.array(STATIC_FEATURE_NAMES[:22] + B_FEATURE_NAMES,
                               dtype=object),
        W=args.w,
    )
    print(f"\nwrote {args.out}: {X.shape[0]} rows x {X.shape[1]} feats, "
          f"pos_rate {y.mean():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
