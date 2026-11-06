"""Step-3 viability probe: does the rollout actually produce column batches?

This is the one experiment in the whole project that can kill the design, so
it runs before any learning is attached.  The reference paper's sampler is
feasible by construction ("add a vertex to an independent set" can never
fail); a route sampler is not, and if the resource constraints make most
samples die then "produce many columns per pricing call" - the entire reason
for copying that paradigm - does not survive the migration.

What is measured, per pricing call:

    neg_cols    unique negative-reduced-cost columns produced
    dead_ends   share of samples that produced no closable route at all
    best_rc     most negative reduced cost found
    lp_gain     how much the master LP objective drops when those columns
                are added to a throw-away copy of the master
    coverage    lp_gain as a share of what a relaxed exact pricer achieves
                on the same dual solution

`coverage` is the honest number.  A sampler can produce a hundred columns and
still capture nothing if they are all clones of routes the master already has.

Usage:
    python scripts/probe_rollout.py --name C101 --n 25
    python scripts/probe_rollout.py --name C101 --n 50 --no-tw --uav 0.7
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from uavcg.bench import DEFAULT_DATA, load_instance
from uavcg.heuristics import cheapest_insertion
from uavcg.instance import evaluate_route
from uavcg.pricers import NgRoutePricer, RolloutPricer
from uavcg.rmp import MasterProblem

ROOT = Path(__file__).resolve().parents[1]


def seed_master(inst) -> MasterProblem:
    m = MasterProblem(inst)
    for r in cheapest_insertion(inst):
        ev = evaluate_route(inst, r)
        if ev.feasible:
            m.add(r, ev.cost)
    if len(m) == 0:
        raise RuntimeError("seeding produced no feasible column")
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="C101")
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--uav", type=float, default=0.0)
    ap.add_argument("--load-factor", type=float, default=1.0)
    ap.add_argument("--no-tw", action="store_true", help="open the time windows (CVRP mode)")
    ap.add_argument("--warmup", type=int, default=20, help="CG iterations with ng-route first")
    ap.add_argument("--warm-cols", type=int, default=8,
                    help="columns the warm-up pricer may add per iteration")
    ap.add_argument("--probe-upto", type=int, default=10,
                    help="probe the first N iterations (the LP converges after that)")
    ap.add_argument("--ng-k", type=int, default=10)
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--betas", default="0.5,1.0,2.0")
    ap.add_argument("--scoring", default="uniform,dual")
    ap.add_argument("--prefix", default="on", choices=["on", "off"])
    args = ap.parse_args()

    inst = load_instance(
        args.name, args.n, args.uav, args.load_factor, args.no_tw, Path(args.data)
    )
    mode = "UAV" if inst.params.energy_enabled else "VRPTW"
    if args.no_tw:
        mode += "/no-TW"
    print(f"=== {inst.name}  n={inst.n}  mode={mode} ===")

    master = seed_master(inst)
    warm = NgRoutePricer(inst, k=args.ng_k)
    scoring_modes = [s for s in args.scoring.split(",") if s]
    betas = [float(b) for b in args.betas.split(",") if b]
    configs = [(s, b) for s in scoring_modes for b in betas]

    rows: List[Dict[str, float]] = []
    t0 = time.perf_counter()

    for it in range(1, args.warmup + 1):
        lp = master.solve_lp()

        if it <= args.probe_upto:
            # reference: everything a relaxed exact pricer can find here
            ref = warm.price(lp.pi, sigma=lp.sigma, fixed_cost=0.0,
                             max_columns=10_000_000)
            ref_m = master.clone()
            ref_new = ref_m.add_many(ref.columns)
            ref_lp = ref_m.solve_lp().objective
            ref_gain = max(lp.objective - ref_lp, 0.0)

            for scoring, beta in configs:
                pricer = RolloutPricer(
                    inst, scoring=scoring, n_samples=args.samples, beta=beta,
                    record_prefixes=(args.prefix == "on"), seed=0,
                    deterministic_passes=1,
                )
                t1 = time.perf_counter()
                out = pricer.price(lp.pi, sigma=lp.sigma, fixed_cost=0.0,
                                   max_columns=10_000_000)
                dt = time.perf_counter() - t1
                trial = master.clone()
                added = trial.add_many(out.columns)
                new_lp = trial.solve_lp().objective
                gain = max(lp.objective - new_lp, 0.0)
                rows.append({
                    "iter": it, "scoring": scoring, "beta": beta,
                    "lp_before": lp.objective,
                    "cols": len(out.columns), "added": added,
                    "ref_cols": len(ref.columns), "ref_new": ref_new,
                    "ref_gain": ref_gain,
                    "lp_gain": gain,
                    "best_rc": out.min_reduced_cost,
                    "dead_end_rate": out.stats["dead_end_rate"],
                    "stuck": out.stats["stuck"],
                    "rejected_tw": out.stats["rejected_tw"],
                    "rejected_energy": out.stats["rejected_energy"],
                    "seconds": dt,
                })

        res = warm.price(lp.pi, sigma=lp.sigma, fixed_cost=0.0,
                         max_columns=args.warm_cols)
        if master.add_many(res.columns) == 0:
            print(f"warm-up converged at iteration {it}")
            break

    print(f"warm-up: {len(master)} columns, {time.perf_counter() - t0:.2f}s, "
          f"LP={master.solve_lp().objective:.4f}")
    print(f"probed {len(rows)} (iteration, configuration) pairs\n")

    header = (f"{'scoring':9s} {'beta':>5s} {'iters':>5s} {'cols':>6s} {'added':>6s} "
              f"{'dead%':>7s} {'lp_gain':>9s} {'ref_gain':>9s} {'cover%':>7s} {'t/s':>7s}")
    print(header)
    print("-" * len(header))

    summary = []
    for scoring, beta in configs:
        sel = [r for r in rows if r["scoring"] == scoring and r["beta"] == beta]
        if not sel:
            continue
        n_it = len(sel)
        gain = sum(r["lp_gain"] for r in sel) / n_it
        ref_gain = sum(r["ref_gain"] for r in sel) / n_it
        cover = gain / ref_gain if ref_gain > 1e-9 else float("nan")
        rec = {
            "scoring": scoring, "beta": beta,
            "cols": sum(r["cols"] for r in sel) / n_it,
            "added": sum(r["added"] for r in sel) / n_it,
            "dead": sum(r["dead_end_rate"] for r in sel) / n_it,
            "gain": gain, "ref_gain": ref_gain, "cover": cover,
            "seconds": sum(r["seconds"] for r in sel) / n_it,
            "best_rc": min(r["best_rc"] for r in sel),
        }
        summary.append(rec)
        print(f"{scoring:9s} {beta:>5.1f} {n_it:>5d} {rec['cols']:>6.1f} {rec['added']:>6.1f} "
              f"{100 * rec['dead']:>7.1f} {rec['gain']:>9.4f} {rec['ref_gain']:>9.4f} "
              f"{100 * cover:>7.1f} {rec['seconds']:>6.3f}s")

    print()
    ok = [r for r in summary if r["cover"] == r["cover"]]
    if not ok:
        print("no configuration produced a measurable one-step LP gain; "
              "the instance is already converged at the probed duals")
        return 0

    best = max(ok, key=lambda r: r["cover"])
    print(f"best: scoring={best['scoring']} beta={best['beta']} -> "
          f"coverage {100 * best['cover']:.1f}% of the reference one-step gain, "
          f"{best['cols']:.1f} columns and {best['added']:.1f} new per call, "
          f"{best['seconds']:.3f}s per call")
    print(f"worst dead-end rate: {100 * max(r['dead'] for r in ok):.1f}%")
    print()
    if best["cover"] < 0.5:
        print("VERDICT: coverage below 50% - fix the sampler before attaching a model.")
    else:
        print("VERDICT: the sampler produces useful batches; proceed to "
              "feature/label collection and learning.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
