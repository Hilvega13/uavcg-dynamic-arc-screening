#!/usr/bin/env python
"""Train the random-forest arc scorer on collect_arc_labels.py output (spec §4).

Implements, per DynML-S spec:
  §3.4  RandomForestClassifier, hyperparameter grid from the paper,
        5-fold cross-validation split BY INSTANCE (no arc of a test
        instance leaks into its training fold).
  §2.3  per-instance min-max normalisation of every feature column.
  §4.2  isotonic probability calibration (CalibratedClassifierCV) with
        instance-level folds -- the calibration set never contains a
        test instance; tau sweep over {0.3,...,0.7}.
  §4.3  class-imbalance metrics: Recall/TNR/BalAcc/PR-AUC/ROC-AUC/Brier,
        Gini + permutation importances.

Label: y_arc_run (Morabit "promising" -- arc used at any iteration of the
run).  Features: --features static (cols 0-21, for ML-S, known before
iteration 0) or all (0-33, adds the dual-dependent block, for DynML-S
offline evaluation).

Usage:
    python scripts/train_rf_arc.py --train results/arc_labels_n60_train.npz \
        --holdout results/arc_labels_n60_test.npz --tag n60
    python scripts/train_rf_arc.py --train results/arc_labels_n25.npz --tag n25
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, GroupKFold

ROOT = Path(__file__).resolve().parents[1]
N_STATIC = 22  # static feature block length (collect_arc_labels.FEATURE_NAMES[:22])
TAUS = [0.3, 0.4, 0.5, 0.6, 0.7]
GRID = {
    "max_depth": [5, 10],
    "max_features": [5, 7, "sqrt"],
}
BASE_RF = dict(
    n_estimators=500,
    min_samples_leaf=50,
    min_samples_split=100,
    bootstrap=True,
    class_weight="balanced",
    n_jobs=-1,
    random_state=0,
)


def load(path: str, label: str = "run"):
    d = np.load(path, allow_pickle=True)
    X = d["X"].astype(np.float64)
    key = {"run": "y_arc_run", "narrow": "y_narrow", "opt": "y_opt"}[label]
    raw = d[key]
    if label == "opt":
        y = (raw > 0).astype(np.int64)            # near-optimal arcs (graded -> binary)
        y[np.isnan(raw)] = 0                       # unsampled rows -> negative
    else:
        y = raw.astype(np.int64)
    inst = d["instances"]
    # per-row instance: the narrowed dump stores it per-row; older dumps store
    # one entry per instance and need iteration-reset reconstruction.
    if inst.ndim == 1 and len(inst) == len(y):
        groups = np.asarray(inst, dtype=str)
    elif "row_instance" in d.files:
        groups = d["row_instance"].astype(str)
    else:
        iters = d["iters"]
        cuts = np.flatnonzero(np.diff(iters) < 0)
        groups = np.empty(len(iters), dtype=object)
        names = inst.astype(str)
        prev = 0
        for k, c in enumerate(cuts):
            groups[prev:c] = names[k]; prev = c
        groups[prev:] = names[len(names) - 1]
    names = d["feature_names"].astype(str)
    return X, y, groups, names


def per_instance_minmax(X: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """§2.3: each instance normalised on its own min/max (per column)."""
    Xn = X.copy()
    for g in np.unique(groups):
        m = groups == g
        lo = Xn[m].min(axis=0)
        hi = Xn[m].max(axis=0)
        span = np.where(hi - lo > 1e-12, hi - lo, 1.0)
        Xn[m] = (Xn[m] - lo) / span
    return Xn


def metrics_at(y_true, p, tau):
    yhat = (p >= tau).astype(int)
    tp = int(((yhat == 1) & (y_true == 1)).sum())
    fn = int(((yhat == 0) & (y_true == 1)).sum())
    tn = int(((yhat == 0) & (y_true == 0)).sum())
    fp = int(((yhat == 1) & (y_true == 0)).sum())
    return {
        "tau": tau,
        "recall": float(tp / max(tp + fn, 1)),
        "tnr": float(tn / max(tn + fp, 1)),
        "bal_acc": float(balanced_accuracy_score(y_true, yhat)),
        "kept_frac": float(yhat.mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--holdout", default="")
    ap.add_argument("--features", choices=["static", "all"], default="static")
    ap.add_argument("--label", choices=["run","narrow","opt"], default="run")
    ap.add_argument("--tag", default="arc")
    ap.add_argument("--out-model", default="")
    ap.add_argument("--out-report", default="")
    args = ap.parse_args()

    t0 = time.perf_counter()
    X, y, groups, feat_names = load(args.train, args.label)
    ncol = N_STATIC if args.features == "static" else X.shape[1]
    X = X[:, :ncol]
    feat_names = feat_names[:ncol]
    X = per_instance_minmax(X, groups)
    print(f"train {X.shape[0]} rows x {ncol} feats, "
          f"{len(np.unique(groups))} instances, pos-rate {y.mean():.3f}")

    uniq = np.unique(groups)
    if len(uniq) < 2:
        raise SystemExit(
            "need >= 2 instances for instance-level CV (spec §3.4); "
            f"got {len(uniq)} in {args.train}")
    # materialise instance-level folds as index lists -- works for both
    # GridSearchCV and CalibratedClassifierCV without sklearn's metadata
    # routing (CalibratedClassifierCV.fit drops `groups` otherwise).
    gkf = GroupKFold(n_splits=min(5, len(uniq)))
    folds = list(gkf.split(X, y, groups))

    gs = GridSearchCV(
        RandomForestClassifier(**BASE_RF), GRID,
        scoring="balanced_accuracy", cv=folds, n_jobs=1, refit=True,
    )
    gs.fit(X, y)
    print(f"best params {gs.best_params_}  bal_acc={gs.best_score_:.4f}")

    # ---- isotonic calibration, instance-level folds (§4.2) ----
    cal = CalibratedClassifierCV(gs.best_estimator_, method="isotonic",
                                 cv=folds, ensemble=False)
    # NOTE: ensemble=False -> one calibrated model per fold; probability =
    # fold-model average.  The calibration folds are the same instance-level
    # splits used above, so no test-instance arc ever touches calibration.
    cal.fit(X, y)

    report = {
        "train_file": args.train,
        "label": args.label,
        "n_rows": int(X.shape[0]),
        "n_instances": int(len(uniq)),
        "features": args.features,
        "feature_names": list(feat_names),
        "pos_rate": float(y.mean()),
        "best_params": {k: (str(v) if not isinstance(v, (int, float)) else v)
                        for k, v in gs.best_params_.items()},
        "cv_bal_acc": float(gs.best_score_),
        "cv_results": [
            {"params": {k: str(v) for k, v in p.items()},
             "bal_acc": float(m)}
            for p, m in zip(gs.cv_results_["params"],
                            gs.cv_results_["mean_test_score"])
        ],
    }

    # ---- holdout evaluation ----
    if args.holdout:
        Xh, yh, gh, _ = load(args.holdout, args.label)
        Xh = per_instance_minmax(Xh[:, :ncol], gh)
        ph = cal.predict_proba(Xh)[:, 1]
        report["holdout"] = {
            "file": args.holdout,
            "n_rows": int(Xh.shape[0]),
            "instances": sorted(set(gh.tolist())),
            "pos_rate": float(yh.mean()),
            "roc_auc": float(roc_auc_score(yh, ph)),
            "pr_auc": float(average_precision_score(yh, ph)),
            "brier": float(brier_score_loss(yh, ph)),
            "brier_uncal": float(brier_score_loss(
                yh, gs.best_estimator_.predict_proba(Xh)[:, 1])),
            "tau_sweep": [metrics_at(yh, ph, t) for t in TAUS],
        }
        print(f"holdout roc_auc={report['holdout']['roc_auc']:.4f} "
              f"pr_auc={report['holdout']['pr_auc']:.4f} "
              f"brier {report['holdout']['brier_uncal']:.4f}"
              f"->{report['holdout']['brier']:.4f}")

    # ---- importances (on the uncalibrated best RF) ----
    rf = gs.best_estimator_
    gini = rf.feature_importances_
    perm = permutation_importance(rf, X, y, n_repeats=3,
                                  random_state=0, n_jobs=-1,
                                  scoring="balanced_accuracy")
    order = np.argsort(-gini)
    report["importances"] = {
        "gini": [{"feature": str(feat_names[i]), "value": float(gini[i])}
                 for i in order],
        "permutation": sorted(
            [{"feature": str(feat_names[i]),
              "value": float(perm.importances_mean[i])}
             for i in range(ncol)],
            key=lambda d: -d["value"]),
    }
    top = report["importances"]["gini"][:8]
    print("top gini:", ", ".join(f"{d['feature']}={d['value']:.3f}"
                                 for d in top))

    # ---- persist ----
    import pickle
    model_path = Path(args.out_model or
                      ROOT / "models" / f"rf_arc_{args.tag}_{args.features}_{args.label}.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as fh:
        pickle.dump({"calibrated": cal, "rf": rf, "features": args.features,
                     "feature_names": list(feat_names),
                     "normalisation": "per-instance-minmax",
                     "tau_default": 0.5}, fh)
    report["model"] = str(model_path)
    report["elapsed_s"] = round(time.perf_counter() - t0, 1)
    rep_path = Path(args.out_report or
                    ROOT / "results" / f"rf_report_{args.tag}_{args.features}_{args.label}.json")
    rep_path.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    print(f"model -> {model_path}\nreport -> {rep_path} "
          f"({report['elapsed_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
