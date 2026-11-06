"""Collect labels from column generation and train the arc scorer.

Two evaluation numbers are reported and they answer different questions:

    in-instance  rows held out at random from the same instances
    cross-instance  rows from instances the model never saw

The second one is the honest one.  Rows produced from the same dual solution -
and even from consecutive dual solutions of the same run - are strongly
correlated, so a random split flatters any model.  Splitting by *instance* is
what tells you whether the score generalises, which is exactly what the
experiments downstream need it to do.

Usage:
    python scripts/train_model.py --n 25
    python scripts/train_model.py --n 25 --train RC101,C101,R102 --holdout R105,RC104
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from uavcg.bench import DEFAULT_DATA, load_instance
from uavcg.features import FEATURE_NAMES
from uavcg.labeling import collect_from_cg, default_label_source
from uavcg.learning import ArcScorer, evaluate_scorer

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"


def gather(args, names: List[str], cache: Path | None = None) -> Dict[str, np.ndarray]:
    if cache is not None and cache.exists():
        blob = np.load(cache, allow_pickle=True)
        print(f"  [cached] {cache.name}: {blob['X'].shape[0]} rows")
        return {
            "X": blob["X"], "y": blob["y"],
            "instances": [tuple(t) for t in blob["instances"]],
        }
    Xs, ys, meta = [], [], []
    for name in names:
        inst = load_instance(name, args.n, args.uav, args.load_factor, args.no_tw,
                             Path(args.data))
        src = default_label_source(inst, args.exact_max_n, args.ng_k)
        t0 = time.perf_counter()
        rep = collect_from_cg(
            inst, label_source=src,
            max_iterations=args.max_iters, sample_every=args.sample_every,
            samples_per_beta=args.samples_per_beta, rel_tol=args.rel_tol,
            max_rows=args.max_rows, seed=args.seed, graded=args.graded,
        )
        batch = rep.stacked()
        dt = time.perf_counter() - t0
        print(f"  {name:8s} n={inst.n:>3d} source={rep.label_source:7s} "
              f"batches={len(rep.batches):>2d} rows={len(batch):>6d} "
              f"pos={100 * batch.hard_positive_rate:>5.2f}% "
              f"partial={100 * batch.partial_rate:>5.2f}% "
              f"skipped={rep.skipped_no_negative:>2d} {dt:>6.2f}s")
        if len(batch):
            Xs.append(batch.X); ys.append(batch.y)
            meta.append((name, len(batch), batch.hard_positive_rate,
                         batch.partial_rate, rep.label_source))
    if not Xs:
        raise SystemExit("no labels collected - check that the pricer finds negative columns")
    out = {"X": np.vstack(Xs), "y": np.concatenate(ys), "instances": meta}
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, X=out["X"], y=out["y"],
                            instances=np.array(meta, dtype=object))
        print(f"  [saved] {cache}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--uav", type=float, default=0.0)
    ap.add_argument("--load-factor", type=float, default=1.0)
    ap.add_argument("--no-tw", action="store_true")
    ap.add_argument("--train", default="RC101,C101,R102,RC201")
    ap.add_argument("--holdout", default="R105,RC104")
    ap.add_argument("--kinds", default="logistic,svm,gbdt")
    ap.add_argument("--max-iters", type=int, default=24)
    ap.add_argument("--sample-every", type=int, default=4)
    ap.add_argument("--samples-per-beta", type=int, default=120)
    ap.add_argument("--max-rows", type=int, default=6000)
    ap.add_argument("--rel-tol", type=float, default=0.02)
    ap.add_argument("--exact-max-n", type=int, default=14)
    ap.add_argument("--graded", dest="graded", action="store_true", default=True,
                    help="grade labels by reduced-cost margin (default)")
    ap.add_argument("--binary", dest="graded", action="store_false",
                    help="reproduce the earlier binary label for comparison")
    ap.add_argument("--ng-k", type=int, default=None,
                    help="ng-route neighbourhood size for the label source "
                         "(default: min(n, 12); k = n makes it exact)")
    ap.add_argument("--cache-suffix", default="",
                    help="disambiguate the label cache, e.g. when the same "
                         "instance set is labelled by two different sources")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", default=str(RESULT_DIR / "train_report.json"))
    args = ap.parse_args()

    train_names = [s for s in args.train.split(",") if s]
    hold_names = [s for s in args.holdout.split(",") if s]

    print("=== collecting training labels ===")
    tag = (f"n{args.n}{'_uav' if args.uav else ''}{'_notw' if args.no_tw else ''}"
           f"{'_graded' if args.graded else ''}{args.cache_suffix}")
    tr = gather(args, train_names, RESULT_DIR / f"labels_train_{tag}.npz")
    print("=== collecting held-out labels ===")
    ho = gather(args, hold_names, RESULT_DIR / f"labels_holdout_{tag}.npz")

    Xtr, ytr = tr["X"], tr["y"]
    Xho, yho = ho["X"], ho["y"]
    print(f"\ntraining rows {len(ytr)} (pos {100 * ytr.mean():.2f}%)   "
          f"holdout rows {len(yho)} (pos {100 * yho.mean():.2f}%)")
    print(f"feature count {Xtr.shape[1]} = {len(FEATURE_NAMES)} named features\n")

    # in-instance split for reference only
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(ytr))
    cut = int(0.8 * len(idx))
    in_tr, in_va = idx[:cut], idx[cut:]

    rows = []
    header = (f"{'kind':9s} {'fit(s)':>7s} {'in_auc':>8s} {'in_f1':>7s} "
              f"{'x_auc':>8s} {'x_f1':>7s} {'x_prec':>8s} {'x_rec':>7s} {'x_spear':>8s}")
    print(header)
    print("-" * len(header))

    best = None
    for kind in [k for k in args.kinds.split(",") if k]:
        scorer = ArcScorer(kind=kind, seed=args.seed).fit(
            Xtr[in_tr], ytr[in_tr], soft=args.graded)
        m_in = evaluate_scorer(scorer, Xtr[in_va], ytr[in_va])
        m_x = evaluate_scorer(scorer, Xho, yho)
        rows.append({"kind": kind, "in_instance": m_in, "cross_instance": m_x,
                     "fit_seconds": scorer.scores.train_seconds})
        print(f"{kind:9s} {scorer.scores.train_seconds:>7.2f} {m_in['roc_auc']:>8.3f} "
              f"{m_in['f1']:>7.3f} {m_x['roc_auc']:>8.3f} {m_x['f1']:>7.3f} "
              f"{m_x['precision']:>8.3f} {m_x['recall']:>7.3f} "
              f"{m_x.get('spearman', float('nan')):>8.3f}")
        if best is None or m_x.get("roc_auc", 0) > best[1]:
            best = (kind, m_x.get("roc_auc", 0), scorer)

    # final model on everything
    kind, _auc, _ = best
    final = ArcScorer(kind=kind, seed=args.seed).fit(Xtr, ytr, soft=args.graded)
    print(f"\nbest by cross-instance AUC: {kind}")
    if final.scores and final.scores.importance:
        print("top features (signed):")
        for nm, val in final.scores.top(10):
            print(f"    {nm:18s} {val:+.3f}")
    print("\nNOTE: the model family that wins here is a result, not a default. "
          "If the linear model matches the tree model, that is worth reporting too.")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_graded" if args.graded else ""
    out_model = MODEL_DIR / f"arc_scorer_n{args.n}{'_uav' if args.uav else ''}{suffix}.pkl"
    final.save(out_model)
    print(f"saved {out_model}")

    if final.affine is not None:
        # The folded coefficients are the model that actually runs: printing
        # them keeps the interpretable version and the deployed version from
        # drifting apart.
        print("folded linear coefficients (raw feature scale, standardised y):")
        for nm, c in sorted(final.coefficients(), key=lambda kv: -abs(kv[1]))[:10]:
            print(f"    {nm:18s} {c:+.4f}")

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps({
            "n": args.n, "uav": args.uav, "no_tw": args.no_tw,
            "graded": bool(args.graded), "label_tag": tag,
            "exact_max_n": args.exact_max_n, "ng_k": args.ng_k,
            "train_instances": tr["instances"], "holdout_instances": ho["instances"],
            "models": rows, "best": kind,
            "model_path": str(out_model),
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
