"""Quality check + summary for a collected arc-label dataset.

Reads `results/arc_labels_<tag>.npz` (written by `collect_arc_labels.py`) and
answers the questions that decide whether the file can be used for training:

  1. **Integrity** - every array loads, lengths agree, no unexpected NaN/Inf
     (NaN in `y_opt` is legitimate: it means "not sampled this iteration").
  2. **Label consistency** - `y_arc_run` must dominate `y_arc_iter` row by row
     (an arc used at some iteration must also count as used over the run).
     A violation means the two labels were built from different column sets.
  3. **Coverage of the run** - how many instances, how many iterations each,
     and the size of the candidate arc set, so the "scale is the first independent variable" argument
     can be made from the data rather than asserted.
  4. **Signal** - univariate ROC AUC of each feature against each label, as a
     first check that the features carry information at all.  This is a
     screening number, not a model result.

Usage:
    python scripts/check_arc_labels.py --tag n25
    python scripts/check_arc_labels.py --npz results/arc_labels_n25.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def auc_1d(x: np.ndarray, y: np.ndarray) -> float:
    """Rank-based AUC via the Mann-Whitney U statistic (ties handled)."""
    ok = np.isfinite(x)
    x, y = x[ok], y[ok]
    pos, neg = int(y.sum()), int((1 - y).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(x, kind="stable")
    ranks = np.empty(x.size, dtype=float)
    ranks[order] = np.arange(1, x.size + 1, dtype=float)
    # average ranks over ties
    xs = x[order]
    i = 0
    while i < xs.size:
        j = i + 1
        while j < xs.size and xs[j] == xs[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = ranks[order[i:j]].mean()
        i = j
    return float((ranks[y > 0].sum() - pos * (pos + 1) / 2) / (pos * neg))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="n25")
    ap.add_argument("--npz", default="")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    npz_path = Path(args.npz) if args.npz else ROOT / f"results/arc_labels_{args.tag}.npz"
    summ_path = npz_path.with_name(npz_path.stem + "_summary.json")
    d = np.load(npz_path, allow_pickle=True)
    X = d["X"]
    yK, yR, yO = d["y_arc_iter"], d["y_arc_run"], d["y_opt"]
    arcs, iters = d["arcs"], d["iters"]
    names = [str(s) for s in d["feature_names"]]
    insts = [str(s) for s in d["instances"]]

    print(f"file       : {npz_path}")
    print(f"rows       : {X.shape[0]:,}   features: {X.shape[1]}   instances: {len(insts)}")
    print(f"instances  : {', '.join(insts)}")

    # ---- 1. integrity -------------------------------------------------
    lengths = {k: len(d[k]) for k in ("X", "y_arc_iter", "y_arc_run", "y_opt", "arcs", "iters")}
    assert len(set(lengths.values())) == 1, f"length mismatch: {lengths}"
    bad_inf = int((~np.isfinite(X)).sum())
    nan_by_col = np.isnan(X).sum(axis=0)
    print(f"\n[integrity] lengths consistent: {lengths}")
    print(f"[integrity] non-finite entries in X: {bad_inf}")
    if bad_inf:
        cols = [names[i] for i in np.flatnonzero(nan_by_col)]
        print(f"            affected columns: {cols}")

    # ---- 2. label consistency -----------------------------------------
    viol = int(((yK > 0.5) & (yR < 0.5)).sum())
    print(f"\n[label] y_arc_iter positive rate: {yK.mean():.4f}")
    print(f"[label] y_arc_run  positive rate: {yR.mean():.4f}")
    fin = np.isfinite(yO)
    print(f"[label] y_opt      positive rate: "
          f"{(yO[fin] > 0).mean() if fin.any() else 0.0:.4f} "
          f"(coverage {fin.mean():.2%})")
    print(f"[label] y_arc_run does NOT dominate y_arc_iter on {viol} rows "
          f"({'OK' if viol == 0 else 'BUG'})")

    # ---- 3. per-instance coverage -------------------------------------
    print("\n[coverage] per instance")
    per_inst = []
    for s in json.loads(summ_path.read_text(encoding="utf-8")) if summ_path.exists() else []:
        per_inst.append(s)
        print(f"   {s['instance']:7s} candidate arcs={s['arcs_candidate']:5d} "
              f"iters={s['iterations']:3d} rows={s['rows']:6d} "
              f"P3={s['promising_frac_of_cand']:.2%} "
              f"{'(time-capped)' if s.get('time_capped') else ''}")

    # ---- 4. univariate signal -----------------------------------------
    print("\n[signal] univariate AUC (screening only)")
    signal = {}
    for target_name, y in (("P3-run", yR), ("P3-iter", yK), ("opt", (yO > 0).astype(float))):
        okt = np.isfinite(yO) if target_name == "opt" else np.ones(len(y), bool)
        aucs = [(n, auc_1d(X[okt, i], y[okt])) for i, n in enumerate(names)]
        aucs = [a for a in aucs if np.isfinite(a[1])]
        aucs.sort(key=lambda t: -abs(t[1] - 0.5))
        signal[target_name] = {n: round(v, 4) for n, v in aucs}
        top = ", ".join(f"{n}={v:.3f}" for n, v in aucs[:6])
        print(f"   {target_name:8s} top-6 by |AUC-0.5|: {top}")

    report = {
        "file": str(npz_path), "rows": int(X.shape[0]), "features": int(X.shape[1]),
        "instances": insts, "nonfinite_in_X": bad_inf,
        "label_rates": {"iter": float(yK.mean()), "run": float(yR.mean()),
                        "opt": float((yO[fin] > 0).mean()) if fin.any() else None,
                        "opt_coverage": float(fin.mean())},
        "run_dominates_iter_violations": viol,
        "per_instance": per_inst, "univariate_auc": signal,
    }
    out = Path(args.json_out) if args.json_out else npz_path.with_name(
        npz_path.stem + "_check.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
