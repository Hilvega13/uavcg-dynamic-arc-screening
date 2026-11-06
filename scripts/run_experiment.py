"""The main comparison: one column generation loop, six pricing subproblems.

Only the pricer changes between rows of the output table.  The master problem,
the seeding heuristic, the fleet limit, the stopping rule and the solver are
identical, so the differences are attributable to the pricing subproblem - and
to nothing else.  That is the whole reason the Pricer interface exists.

Speed is reported two ways, because they can disagree and the disagreement is
informative:

    time to target   wall-clock until the LP reaches within `--target-gap` of
                     the best bound any pricer reached on that instance
    iterations       the same progress counted in pricing rounds rather than
                     seconds, which separates "this pricer is better" from
                     "this pricer is cheaper per call"

Usage:
    python scripts/run_experiment.py --n 25
    python scripts/run_experiment.py --n 50 --uav 0.7
    python scripts/run_experiment.py --n 25 --pricers ngroute,rollout-dual,rollout-model
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from uavcg.bench import DEFAULT_DATA, load_instance
from uavcg.cg import CGConfig, ColumnGeneration
from uavcg.heuristics import singleton_routes
from uavcg.learning import ArcScorer
from uavcg.pricers import (
    ExactLabelingPricer,
    GreedyPricer,
    NgRoutePricer,
    RolloutPricer,
)

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results"


def build_pricer(kind: str, inst, args, models: Dict[str, ArcScorer]):
    """Return (pricer, fallback).  The fallback is what certifies optimality.

    `kind` may carry a per-pricer sample count as `kind:samples`, e.g.
    `rollout-model:60`.  That exists so configs can be compared at *equal
    per-call compute* instead of equal sample count: a learned rollout costs
    several times more per sample than a dual-guided one, so matching sample
    counts would hand the heuristic a several-fold compute advantage and call
    the resulting difference a quality difference.
    """
    samples = args.samples
    if ":" in kind:
        kind, _, tail = kind.partition(":")
        try:
            samples = int(tail)
        except ValueError:
            raise SystemExit(f"bad pricer spec {kind!r}:{tail!r} - expected kind:samples")

    # `rollout-model@<key>` selects one of possibly several loaded scorers, so
    # two labelling strategies can be compared inside a single run.  A bare
    # `rollout-model` means "the only model" and errors out loudly if there is
    # not exactly one, rather than silently picking one of them.
    model_key = None
    if kind.startswith("rollout-model") and "@" in kind:
        kind, _, model_key = kind.partition("@")
    cheap_fallback = (
        ExactLabelingPricer(inst) if inst.n <= args.exact_max_n
        else NgRoutePricer(inst, k=args.ng_k)
    )
    if kind == "exact":
        return ExactLabelingPricer(inst), None
    if kind == "ngroute":
        return NgRoutePricer(inst, k=args.ng_k), None
    if kind == "greedy":
        return GreedyPricer(inst, restarts=args.greedy_restarts, seed=0), cheap_fallback
    if kind == "rollout-uniform":
        return RolloutPricer(
            inst, scoring="uniform", n_samples=samples, seed=args.seed,
            record_prefixes=not args.no_prefix,
            name=f"rollout-uniform:{samples}",
            candidate_pool=args.pool or None,
        ), cheap_fallback
    if kind == "rollout-dual":
        return RolloutPricer(
            inst, scoring="dual", n_samples=samples, beta=args.beta,
            seed=args.seed, record_prefixes=not args.no_prefix,
            name=f"rollout-dual:{samples}",
            candidate_pool=args.pool or None,
        ), cheap_fallback
    if kind == "rollout-model":
        if model_key is None:
            if len(models) != 1:
                raise SystemExit(
                    f"rollout-model needs an explicit @key when {len(models)} "
                    f"models are loaded: {sorted(models)}"
                )
            model_key = next(iter(models))
        if model_key not in models:
            raise SystemExit(f"no model {model_key!r}; loaded: {sorted(models)}")
        return RolloutPricer(
            inst, scoring="model", scorer=models[model_key], n_samples=samples,
            beta=args.beta, seed=args.seed, record_prefixes=not args.no_prefix,
            warmup_samples=args.warmup_samples,
            candidate_pool=args.pool or None,
            name=f"rollout-model[{model_key}]:{samples}",
        ), cheap_fallback
    raise SystemExit(f"unknown pricer {kind!r}")


def run_one(kind: str, name: str, args, models: Dict[str, ArcScorer]) -> Dict:
    inst = load_instance(name, args.n, args.uav, args.load_factor, args.no_tw,
                         Path(args.data))
    pricer, fallback = build_pricer(kind, inst, args, models)
    cfg = CGConfig(
        max_iterations=args.max_iters,
        max_columns_per_iter=args.max_cols,
        exact_confirmation_max_n=args.exact_max_n,
        fallback_max_n=args.fallback_max_n,
        stall_limit=args.stall,
    )
    cg = ColumnGeneration(inst, pricer, config=cfg, fallback_pricer=fallback,
                          seed_routes=singleton_routes(inst))
    t0 = time.perf_counter()
    res = cg.run()
    dt = time.perf_counter() - t0

    negs = [s.extra.get("negative_found", 0) for s in res.snapshots]
    deads = [s.extra.get("dead_end_rate") for s in res.snapshots
             if s.extra.get("dead_end_rate") is not None]
    return {
        "pricer": kind,
        "instance": name,
        "n": inst.n,
        "mode": "UAV" if inst.params.energy_enabled else "VRPTW",
        "lp_objective": res.lp_objective,
        "lp_certified": res.optimal_lp,
        "ip_objective": res.ip_objective,
        "num_vehicles": res.num_vehicles,
        "iterations": res.iterations,
        "columns_in_pool": res.columns_in_pool,
        "pricing_calls": res.pricing_calls,
        "exact_confirmations": res.exact_confirmation_calls,
        "confirmation_skipped": res.exact_confirmation_skipped,
        "stalled": res.stalled,
        "mean_negative_per_call": float(np.mean(negs)) if negs else 0.0,
        "mean_dead_end_rate": float(np.mean(deads)) if deads else None,
        "wall_clock": dt,
        "curve": [(s.iteration, s.elapsed, s.lp_objective, s.num_columns)
                  for s in res.snapshots],
        "fallback": fallback.name if fallback else None,
        "model": model_key_of(kind, models),
    }


def model_key_of(kind: str, models: Dict[str, ArcScorer]):
    """Which loaded scorer an arm used, for the results table."""
    if kind.startswith("rollout-model") and "@" in kind:
        return kind.partition("@")[2].partition(":")[0]
    if kind == "rollout-model" and len(models) == 1:
        return next(iter(models))
    return None


def time_to_target(row: Dict, target: float, gap: float) -> tuple:
    """(wall-clock, iteration) at which the LP first gets within `gap` of target."""
    if not (target < float("inf")) or target <= 0:
        return float("inf"), -1
    need = target * (1.0 + gap)
    for it, elapsed, obj, _cols in row["curve"]:
        if obj <= need:
            return elapsed, it
    return float("inf"), -1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--instances", default="R105,RC104,R107,C105,RC107")
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--uav", type=float, default=0.0)
    ap.add_argument("--load-factor", type=float, default=1.0)
    ap.add_argument("--no-tw", action="store_true")
    ap.add_argument("--pricers", default="ngroute,greedy,rollout-uniform,rollout-dual,rollout-model")
    ap.add_argument("--model", action="append", default=None,
                    help="path to a trained scorer, optionally key=path; "
                         "repeatable, and each key is addressable as "
                         "rollout-model@key:samples")
    ap.add_argument("--ng-k", type=int, default=10)
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--beta", type=float, default=2.0,
                    help="sampling temperature; 2.0 is the sweep optimum "
                         "(see RolloutPricer)")
    ap.add_argument("--warmup-samples", type=int, default=0,
                    help="rank-feature warm-up rollouts; must match training")
    ap.add_argument("--pool", type=int, default=0,
                    help="prefilter candidates to the K most promising before scoring "
                         "(0 = score every candidate)")
    ap.add_argument("--no-prefix", action="store_true",
                    help="record only maximal routes instead of every prefix")
    ap.add_argument("--greedy-restarts", type=int, default=60)
    ap.add_argument("--max-iters", type=int, default=300)
    ap.add_argument("--max-cols", type=int, default=25)
    ap.add_argument("--stall", type=int, default=40,
                    help="stop after N iterations with no LP improvement (0 = never)")
    ap.add_argument("--exact-max-n", type=int, default=20)
    ap.add_argument("--fallback-max-n", type=int, default=120)
    ap.add_argument("--target-gap", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    models: Dict[str, ArcScorer] = {}
    for spec in (args.model or []):
        if "=" in spec:
            key, _, path = spec.partition("=")
        else:
            key, path = "default", spec
        models[key] = ArcScorer.load(path)
        print(f"loaded scorer [{key}] {models[key]}  <- {path}")

    names = [s for s in args.instances.split(",") if s]
    kinds = [s for s in args.pricers.split(",") if s]

    rows: List[Dict] = []
    for name in names:
        print(f"\n=== {name} (n={args.n}) ===")
        per_instance: List[Dict] = []
        for kind in kinds:
            try:
                r = run_one(kind, name, args, models)
            except Exception as exc:
                print(f"  {kind:16s} FAILED {type(exc).__name__}: {exc}")
                continue
            per_instance.append(r)
            print(f"  {kind:16s} LP={r['lp_objective']:>10.3f} "
                  f"cert={str(r['lp_certified']):5s} IP={r['ip_objective']:>10.2f} "
                  f"it={r['iterations']:>3d} cols={r['columns_in_pool']:>5d} "
                  f"neg/call={r['mean_negative_per_call']:>6.1f} "
                  f"{r['wall_clock']:>8.2f}s")

        best = min((r["lp_objective"] for r in per_instance
                    if r["lp_certified"]), default=None)
        if best is None:
            best = min((r["lp_objective"] for r in per_instance), default=0.0)
        for r in per_instance:
            r["reference_lp"] = best
            r["instance_best_certified"] = any(
                x["lp_certified"] and abs(x["lp_objective"] - best) < 1e-6
                for x in per_instance
            )
            t, it = time_to_target(r, best, args.target_gap)
            r["time_to_target"] = t
            r["iter_to_target"] = it
        rows.extend(per_instance)

    print("\n=== summary ===")
    header = (f"{'instance':9s} {'pricer':16s} {'LP':>10s} {'cert':>5s} "
              f"{'it':>4s} {'t_iter':>7s} {'time':>9s} {'t_target':>9s} "
              f"{'cols':>6s} {'neg/call':>9s} {'dead%':>7s}")
    print(header)
    print("-" * len(header))
    for r in rows:
        dead = r["mean_dead_end_rate"]
        t = r["time_to_target"]
        print(f"{r['instance']:9s} {r['pricer']:16s} {r['lp_objective']:>10.3f} "
              f"{'yes' if r['lp_certified'] else 'no':>5s} {r['iterations']:>4d} "
              f"{r['iter_to_target']:>7d} {r['wall_clock']:>8.2f}s "
              f"{(f'{t:.2f}s' if np.isfinite(t) else '  --'):>9s} "
              f"{r['columns_in_pool']:>6d} {r['mean_negative_per_call']:>9.1f} "
              f"{(f'{100*dead:.1f}' if dead is not None else '--'):>7s}")

    # aggregate per pricer
    print("\n=== per pricer (mean over instances) ===")
    print(f"{'pricer':16s} {'LPgap%':>8s} {'certified':>10s} {'iters':>7s} "
          f"{'time(s)':>9s} {'t_target':>9s} {'neg/call':>9s}")
    agg = {}
    for kind in kinds:
        sel = [r for r in rows if r["pricer"] == kind]
        if not sel:
            continue
        gaps = [
            100.0 * (r["lp_objective"] - r["reference_lp"]) / max(abs(r["reference_lp"]), 1e-9)
            for r in sel
        ]
        tt = [r["time_to_target"] for r in sel if np.isfinite(r["time_to_target"])]
        rec = {
            "mean_lp_gap_pct": float(np.mean(gaps)),
            "certified": f"{sum(1 for r in sel if r['lp_certified'])}/{len(sel)}",
            "mean_iterations": float(np.mean([r["iterations"] for r in sel])),
            "mean_wall_clock": float(np.mean([r["wall_clock"] for r in sel])),
            "mean_time_to_target": float(np.mean(tt)) if tt else None,
            "mean_negative_per_call": float(np.mean([r["mean_negative_per_call"] for r in sel])),
        }
        agg[kind] = rec
        tt_str = f"{rec['mean_time_to_target']:.2f}s" if rec["mean_time_to_target"] else "  --"
        print(f"{kind:16s} {rec['mean_lp_gap_pct']:>8.3f} {rec['certified']:>10s} "
              f"{rec['mean_iterations']:>7.1f} {rec['mean_wall_clock']:>8.2f}s "
              f"{tt_str:>9s} {rec['mean_negative_per_call']:>9.1f}")

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps({
            "config": vars(args), "rows": rows, "aggregate": agg,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
