"""Measure dual degeneracy, then try the classical remedy on it.

Why this exists
---------------
The project measured that on n=25 Solomon instances the LP sits in a state
where a pricer keeps returning genuinely negative columns that cannot move the
bound - the "plateau" that Vanderbeck (2005) names.  That state matters here
for a reason beyond speed: the dual vector is the shared input of three
downstream components (the exact pricing objective, the `pi_*` features of the
learned scorer, and the dual context that labels are collected under).  While
the duals wander, all three are measuring a moving target, so a weak learned
signal may be a symptom rather than a cause.

This script does two things in the order the plan calls for:

  1. Quantify.  `DegeneracyReport` turns one run into numbers - what share of
     the column-adding iterations were wasted, how long the longest plateau
     ran, how far the duals moved per iteration, and how many iterations were
     needed to capture 99% of the total bound improvement.
  2. Remedy.  Convex-combination smoothing, pi_stab = a*center + (1-a)*pi,
     with the centre the dual of the best LP so far.

The hard constraint, checked and reported rather than assumed: stabilization
may change the path, never the destination.  Every configuration must report
the *same* LP bound as the unstabilized run, and every configuration must
still certify it.  A row where that fails is a bug, not a result.

Examples
--------
    python scripts/run_stabilization.py
    python scripts/run_stabilization.py --names RC104,R107 --n 25
    python scripts/run_stabilization.py --pricer rollout-dual --alphas 0,0.5,0.7,0.9
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uavcg.bench import load_instance
from uavcg.cg import CGConfig, ColumnGeneration
from uavcg.heuristics import singleton_routes
from uavcg.pricers import ExactLabelingPricer, NgRoutePricer, RolloutPricer

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results"

#: R and RC instances.  C-class is excluded on a measured ground: truncated to
#: n=25 the cheapest-insertion seed is already LP optimal, so there is no
#: degeneracy to study there - and no column generation either.
DEFAULT_NAMES = ["RC104", "R107", "R105", "RC107"]


def build_pricer(kind: str, inst, args):
    """Return (pricer, fallback).  The fallback is what certifies optimality."""
    cheap_fallback = (
        ExactLabelingPricer(inst) if inst.n <= args.exact_max_n
        else NgRoutePricer(inst, k=args.ng_k)
    )
    if kind == "exact":
        return ExactLabelingPricer(inst), None
    if kind == "ngroute":
        return NgRoutePricer(inst, k=args.ng_k), None
    if kind == "rollout-dual":
        return RolloutPricer(
            inst, scoring="dual", n_samples=args.samples, beta=args.beta,
            seed=args.seed, record_prefixes=not args.no_prefix,
            name=f"rollout-dual:{args.samples}", candidate_pool=args.pool or None,
        ), cheap_fallback
    raise SystemExit(f"unknown pricer {kind!r}")


def run_one(kind, name, alpha, args) -> dict:
    inst = load_instance(name, args.n, args.uav, args.load_factor, args.no_tw,
                         Path(args.data))
    pricer, fallback = build_pricer(kind, inst, args)
    stabilized = alpha > 0.0
    cfg = CGConfig(
        max_iterations=args.max_iters,
        max_columns_per_iter=args.max_cols,
        exact_confirmation_max_n=args.exact_max_n,
        fallback_max_n=args.fallback_max_n,
        stall_limit=args.stall,
        stabilization="smooth" if stabilized else "off",
        stabilization_alpha=alpha,
        stabilization_warmup=args.warmup,
        log_every=args.log_every,
    )
    if args.verbose:
        print(f"    .. {name} alpha={alpha:g} start", flush=True)
    t0 = time.perf_counter()
    res = ColumnGeneration(
        inst, pricer, config=cfg, fallback_pricer=fallback,
        seed_routes=singleton_routes(inst), max_vehicles=args.fleet,
    ).run()
    d = res.degeneracy(target=args.target)
    return {
        "instance": name,
        "n": inst.n,
        "pricer": kind,
        "alpha": alpha,
        "seed_lp": d.seed_lp,
        "lp": res.lp_objective,
        "certified": bool(res.optimal_lp),
        "stalled": bool(res.stalled),
        "ip": res.ip_objective,
        "iterations": res.iterations,
        "columns": res.columns_in_pool,
        "wall": time.perf_counter() - t0,
        "calls": res.pricing_calls,
        "stabilized_iters": res.stabilized_iterations,
        "filtered": res.filtered_columns,
        "max_iters": args.max_iters,
        "degeneracy": d.as_dict(),
    }


def report(rows, alphas, names, args) -> dict:
    """Per-instance and per-alpha tables, plus the consistency verdict.

    The verdict is deliberately convergence-aware.  A run cut off by
    `max_iterations` or by the stall rule reports whatever LP value it happened
    to be sitting on.  The RMP is a restriction of the full LP, so that value
    is an *upper* bound on the true optimum rather than the optimum itself -
    two truncated runs sitting at different values is the expected outcome of
    having explored different columns, not evidence of a broken stabilization.
    Flagging it as a bug would be wrong, and it would bury the real signal:
    on a truncated instance the interesting number is how far each
    configuration got inside the same iteration budget.
    """
    for name in names:
        per = [r for r in rows if r["instance"] == name]
        if not per:
            continue
        ref = next((r for r in per if r["alpha"] == 0.0), per[0])
        d0 = ref["degeneracy"]

        print()
        print(f"=== {name} (n={ref['n']}) ===")
        print(f"  seed LP = {ref['seed_lp']:.3f}   gain over the run = {d0['total_gain']:.3f}"
              f"   final LP = {ref['lp']:.3f}")
        if d0["total_gain"] > 1e-6:
            print(f"  {d0['wasted_iterations']}/{d0['wasted_iterations'] + d0['improving_iterations']}"
                  f" column-adding iterations moved no bound"
                  f"  ->  wasted {d0['wasted_fraction']:.0%},"
                  f" longest plateau {d0['longest_plateau']},"
                  f" dual move {d0['dual_move_mean']:.3f}/iter")
            print(f"  99% of the gain needed {d0['iters_to_target']} of {ref['iterations']}"
                  f" iterations;  {d0['gain_per_column']:.4f} bound per column")
        else:
            print("  ** no bound improvement at all: the seed relaxation already "
                  "equals the final LP, so this instance is fully degenerate and "
                  "has nothing to converge to **")

        print()
        hdr = (f"  {'alpha':>6} {'LP':>12} {'cert':>5} {'it':>4} {'cols':>5} "
               f"{'waste':>6} {'plateau':>7} {'dpi/it':>7} {f't{args.target:g}':>5} "
               f"{'stab':>5} {'filt':>5} {'wall':>8}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for r in per:
            d = r["degeneracy"]
            tag = "off" if r["alpha"] == 0 else f"{r['alpha']:g}"
            print(f"  {tag:>6} {r['lp']:12.3f} {str(r['certified'])[:1]:>5} "
                  f"{r['iterations']:4d} {r['columns']:5d} {d['wasted_fraction']:6.0%} "
                  f"{d['longest_plateau']:7d} {d['dual_move_mean']:7.3f} "
                  f"{d['iters_to_target']:5d} {r['stabilized_iters']:5d} "
                  f"{r['filtered']:5d} {r['wall']:7.2f}s")

        spread = max(abs(r["lp"] - ref["lp"]) for r in per)
        print()
        if all(r["certified"] for r in per):
            if spread < 1e-4:
                print(f"  [OK] every alpha converged to the same certified LP bound "
                      f"(max deviation {spread:.2e}) -> stabilization is bound-neutral here")
            else:
                print(f"  [BUG] every alpha certified, yet the bounds differ by "
                      f"{spread:.4e}. Stabilization must be bound-neutral; the "
                      f"true-dual verification in the loop is not doing its job.")
        else:
            best = min(per, key=lambda r: r["lp"])
            gain = ref["lp"] - best["lp"]
            print(f"  [n/a] not every row converged, so the LP column is an upper bound "
                  f"on the optimum and agreement is not expected or required.")
            if spread < 1e-6:
                print(f"        All rows landed on the same value anyway "
                      f"({ref['lp']:.3f}) - agreement, though not a certificate.")
            elif gain > 1e-6:
                print(f"        At the equal {args.max_iters}-iteration budget the best "
                      f"alpha ({best['alpha']:g}) got to {best['lp']:.3f} against off's "
                      f"{ref['lp']:.3f}, i.e. {gain:+.3f} further.")
            else:
                print(f"        At the equal {args.max_iters}-iteration budget no alpha "
                      f"beat off ({ref['lp']:.3f}); best was {best['lp']:.3f}.")
        if not all(r["certified"] for r in per):
            print("  [NOTE] some rows did not certify optimality (instance size or the "
                  "iteration/stall cap); no optimality claim attaches to those numbers")

    print()
    print("=" * 108)
    print("per-alpha mean over instances")
    print("=" * 108)
    hdr = (f"{'alpha':>6} {'LPdev':>9} {'cert':>6} {'iters':>6} {'cols':>6} "
           f"{'waste':>6} {'plateau':>7} {'dpi/it':>7} {'t_target':>8} {'wall':>8}")
    print(hdr)
    print("-" * len(hdr))

    summary = {}
    for alpha in alphas:
        sel = [r for r in rows if r["alpha"] == alpha]
        if not sel:
            continue
        off_lp = {r["instance"]: r["lp"] for r in rows if r["alpha"] == 0.0}
        conv = [r for r in sel if r["certified"] and r["instance"] in off_lp]
        dev = max((abs(r["lp"] - off_lp[r["instance"]]) for r in conv), default=0.0)
        m = lambda key: sum(r[key] for r in sel) / len(sel)
        md = lambda key: sum(r["degeneracy"][key] for r in sel) / len(sel)
        print(f"{alpha:6g} {dev:9.1e} {sum(r['certified'] for r in sel):3d}/{len(sel):<2d} "
              f"{m('iterations'):6.1f} {m('columns'):6.1f} {md('wasted_fraction'):6.0%} "
              f"{md('longest_plateau'):7.1f} {md('dual_move_mean'):7.3f} "
              f"{md('iters_to_target'):8.1f} {m('wall'):7.2f}s")
        summary[str(alpha)] = {
            "lp_max_deviation_over_certified": dev,
            "certified": sum(r["certified"] for r in sel),
            "instances": len(sel),
            "mean_iterations": m("iterations"),
            "mean_columns": m("columns"),
            "mean_wasted_fraction": md("wasted_fraction"),
            "mean_longest_plateau": md("longest_plateau"),
            "mean_dual_move": md("dual_move_mean"),
            "mean_iters_to_target": md("iters_to_target"),
            "mean_wall": m("wall"),
        }
    if len(alphas) > 1:
        base_cols = summary.get("0.0", {}).get("mean_columns")
        base_move = summary.get("0.0", {}).get("mean_dual_move")
        base_waste = summary.get("0.0", {}).get("mean_wasted_fraction")
        if base_cols:
            print()
            print("read this as: does smoothing move the three numbers that matter?")
            print(f"  {'alpha':>6} {'cols vs off':>12} {'dpi vs off':>11} {'waste vs off':>13}")
            for alpha in alphas:
                key = str(alpha)
                if key not in summary:
                    continue
                s = summary[key]
                dc = s["mean_columns"] / base_cols - 1.0
                dm = s["mean_dual_move"] / base_move - 1.0 if base_move else 0.0
                dw = s["mean_wasted_fraction"] - base_waste
                print(f"  {alpha:6g} {dc:+11.1%} {dm:+11.1%} {dw:+13.1%}")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", default=",".join(DEFAULT_NAMES))
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--pricer", default="ngroute",
                    choices=["exact", "ngroute", "rollout-dual"])
    ap.add_argument("--alphas", default="0,0.3,0.5,0.7,0.9",
                    help="0 is the unstabilized reference row")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--target", type=float, default=0.99,
                    help="fraction of the total bound gain used for iters_to_target")
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--pool", type=int, default=0)
    ap.add_argument("--no-prefix", action="store_true")
    ap.add_argument("--ng-k", type=int, default=10)
    ap.add_argument("--max-iters", type=int, default=400)
    ap.add_argument("--max-cols", type=int, default=25)
    ap.add_argument("--stall", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=0,
                    help="print the CG loop every N iterations (0 = quiet)")
    ap.add_argument("--verbose", action="store_true",
                    help="announce each (instance, alpha) run before starting it")
    ap.add_argument("--exact-max-n", type=int, default=20,
                    help="default 20, matching run_experiment.  At n=25 this "
                         "makes the fallback an ng-route relaxation instead of "
                         "the exact labeller.  Measured on R107 the two reach "
                         "the same certified bound (425.269) with the same "
                         "iteration and column counts, but the exact labeller "
                         "costs 204.6 s against the relaxation's 12.6 s, which "
                         "is what made the long sweeps look like they had hung")
    ap.add_argument("--fallback-max-n", type=int, default=120,
                    help="ceiling for calling the fallback at all.  Do not leave "
                         "this unset while lowering --exact-max-n: the effective "
                         "limit becomes the smaller value, and at n=25 that "
                         "silently disables the fallback, turning a converging "
                         "run into a stalled one without raising anything")
    ap.add_argument("--fleet", type=int, default=None)
    ap.add_argument("--uav", type=float, default=0.0)
    ap.add_argument("--load-factor", type=float, default=1.0)
    ap.add_argument("--no-tw", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default=str(ROOT / "data" / "solomon100"))
    ap.add_argument("--out", default=str(RESULT_DIR / "stabilization.json"))
    ap.add_argument("--from-json", default=None,
                    help="skip the solves and re-report an existing result file")
    args = ap.parse_args()

    alphas = [float(a) for a in args.alphas.split(",") if a.strip()]
    names = [s.strip() for s in args.names.split(",") if s.strip()]

    rows = []
    if args.from_json:
        with open(args.from_json, encoding="utf-8") as fh:
            rows = json.load(fh)["rows"]
        # a stored file may cover more instances than --names asks for; report
        # exactly what was asked for and nothing else
        have = {r["instance"] for r in rows}
        names = [n for n in names if n in have] or sorted(have)
        print("=" * 108)
        print(f"re-reporting {args.from_json}  ({len(rows)} runs)")
        print("=" * 108)
    else:
        print("=" * 108)
        print(f"duality degeneracy and convex-combination smoothing  "
              f"(n={args.n}, pricer={args.pricer})")
        print("=" * 108)
        for name in names:
            for alpha in alphas:
                try:
                    rows.append(run_one(args.pricer, name, alpha, args))
                except Exception as exc:  # keep a long sweep going
                    print(f"  [{name}] alpha={alpha:g} failed: {exc}", flush=True)

    summary = report(rows, alphas, names, args)

    if not args.from_json:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump({"args": vars(args), "summary": summary, "rows": rows},
                      fh, indent=2)
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
