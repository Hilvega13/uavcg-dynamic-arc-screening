"""Run column generation on a benchmark instance.

Examples
--------
    python scripts/run_cg.py --name C101 --n 25 --pricer exact --trace
    python scripts/run_cg.py --name C101 --pricer greedy --fleet 25
    python scripts/run_cg.py --name R101 --n 50 --pricer exact --uav 1.0
    python scripts/run_cg.py --sweep --n 25 --pricer exact

The point of this script is to answer, before any learning is involved:
how large an instance can the exact pricer still handle, and what does the
column generation baseline look like on it? Everything the learned pricer
will later be compared against comes out of here.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from uavcg.cg import CGConfig, ColumnGeneration
from uavcg.heuristics import cheapest_insertion, report, singleton_routes
from uavcg.instance import (
    EPS,
    Instance,
    as_uav,
    evaluate_route,
    narrow_time_windows,
    parse_solomon,
    truncate,
)
from uavcg.pricers import (
    ExactLabelingPricer,
    GreedyPricer,
    NgRoutePricer,
    RedCostArcPricer,
)

ROOT = Path(__file__).resolve().parents[1]


def make_uav(inst: Instance, factor: float, load_factor: float) -> Instance:
    """Energy budget derived from the seed solution, so the constraint bites.

    A budget that is generous enough never to bind would make the UAV mode
    indistinguishable from plain VRPTW, and every experiment would silently
    be measuring the wrong thing.

    NOTE (v3.4): this is *instance definition*, not the column-generation seed.
    The v3.4 singleton-seed decision governs which columns seed the RMP; it
    deliberately does not touch the energy budget, because changing how the
    budget is derived would silently redefine the instance and move every
    number ever recorded on it.  Left on `cheapest_insertion` on purpose --
    do not "fix" this without re-declaring the instance.
    """
    routes = cheapest_insertion(inst)
    seed_max = max(evaluate_route(inst, r).energy for r in routes)
    floor = max(
        evaluate_route(inst, (i,)).energy * 1.05 for i in inst.customers
    )
    return as_uav(
        inst,
        energy_capacity=max(factor * seed_max, floor),
        load_energy_factor=load_factor,
    )


def load_instance(
    name: str,
    n: int,
    uav: float,
    load_factor: float,
    data_dir: Path,
    tw_scale: float = 1.0,
) -> Instance:
    path = data_dir / f"{name}.txt"
    if not path.exists():
        raise SystemExit(f"instance not found: {path}\nrun scripts/fetch_instances.py first")
    inst, _meta = parse_solomon(path)
    if n and n < inst.n:
        inst = truncate(inst, n)
    # Truncate first, then narrow: the narrowing protocol shifts windows toward
    # their midpoint, so it has to see the same geometry the solver will use.
    if tw_scale != 1.0:
        inst, _rep = narrow_time_windows(inst, tw_scale)
    if uav > 0:
        inst = make_uav(inst, uav, load_factor)
    return inst


def run_one(args, name: str) -> dict:
    inst = load_instance(
        name, args.n, args.uav, args.load_factor, Path(args.data), args.tw_scale
    )
    fleet = args.fleet if args.fleet > 0 else None

    if args.pricer == "exact":
        pricer = ExactLabelingPricer(inst)
        fallback = None
    elif args.pricer == "ngroute":
        pricer = NgRoutePricer(inst, k=args.ng_k)
        fallback = ExactLabelingPricer(inst)
    elif args.pricer == "redcost":
        ladder = tuple(
            None if r.lower() in ("none", "full", "*") else int(r)
            for r in args.redcost_ladder.split(",")
            if r.strip()
        )
        pricer = RedCostArcPricer(inst, ladder=ladder, k=args.ng_k)
        fallback = ExactLabelingPricer(inst)
    elif args.pricer == "greedy":
        pricer = GreedyPricer(inst, restarts=args.greedy_restarts, noise=args.greedy_noise, seed=0)
        fallback = ExactLabelingPricer(inst)
    else:
        raise SystemExit(f"unknown pricer: {args.pricer}")

    cfg = CGConfig(
        max_iterations=args.max_iters,
        max_columns_per_iter=args.max_cols,
        log_every=1 if args.trace else 0,
        exact_confirmation_max_n=args.exact_max_n,
        stall_limit=args.stall,
    )
    cg = ColumnGeneration(
        inst, pricer, config=cfg, fallback_pricer=fallback,
        # v3.4 decision ①/⑥: singleton seed for every formal arm.  The W
        # warm-up is the first `W` iterations of this same run, so it inherits
        # the seed here -- there is no separate warm-up seed to keep in sync.
        seed_routes=singleton_routes(inst),
        max_vehicles=fleet, vehicle_fixed_cost=args.fixed_cost,
    )

    print(
        f"\n--- {inst.name}  n={inst.n}  mode={'UAV' if inst.params.energy_enabled else 'VRPTW'}"
        f"  pricer={args.pricer}  fleet={fleet if fleet else '-'}"
    )
    res = cg.run()
    # Diagnostic only (spec v3.4: the cheapest_insertion value is "recorded as a diagnostic only").
    # It is not the pool that was solved, so it must never be read as a bound.
    seed = report(inst, cheapest_insertion(inst))

    labels = sum(s.extra.get("labels_created", 0) for s in res.snapshots)
    neg = sum(s.extra.get("negative_found", 0) for s in res.snapshots)
    relaxed_last = res.snapshots[-1].extra.get("relaxed_min", None) if res.snapshots else None

    print(f"    seed cost (heuristic) : {seed['cost']:.2f} in {seed['num_routes']} routes")
    print(f"    LP bound              : {res.lp_objective:.4f}   certified={res.optimal_lp}")
    print(f"    IP solution           : {res.ip_objective:.4f} in {res.num_vehicles} routes")
    print(f"    iterations            : {res.iterations}")
    print(f"    columns in pool       : {res.columns_in_pool}")
    print(f"    pricing calls         : {res.pricing_calls}  (exact confirmations: {res.exact_confirmation_calls})")
    print(f"    labels created        : {labels}")
    print(f"    negative cols found   : {neg}")
    if relaxed_last is not None:
        print(f"    final relaxed bound   : {relaxed_last:+.6f}  (ng-route lower bound)")
    if res.exact_confirmation_skipped:
        print("    NOTE                  : exact confirmation skipped (instance too large)")
    if res.degraded:
        print(f"    DEGRADED              : label cap hit "
              f"({res.truncated_pricing_calls} calls truncated, peak "
              f"{res.peak_labels_created}/{res.label_cap} = "
              f"{res.peak_labels_created / max(1, res.label_cap or 1):.1%}) "
              f"-> NOT usable as evidence (spec §3.5)")
    print(f"    wall clock            : {res.wall_clock:.3f} s")
    print(f"      PP / RMP / other    : {res.pp_seconds:.3f} / {res.rmp_seconds:.3f} "
          f"/ {res.other_seconds:.3f} s   (§6.2)")
    if res.masked_pricing_calls or res.mask_hook_fired:
        # `arc_reduction` is the hook mask's size (one number for the run);
        # `mean_arc_reduction` is what the calls actually withheld, which is the
        # only meaningful figure when the pricer picks its own reduction.
        share = res.mean_arc_reduction
        print(f"    arc reduction         : "
              f"{'n/a' if share is None else f'{share:.1%}'} mean over "
              f"{res.masked_pricing_calls} masked calls "
              f"(mean {res.masked_arcs_mean:.0f} of {res.arcs_total} arcs)")
        if res.arcs_total and res.masked_arcs:
            print(f"      hook mask          : {res.arc_reduction:.1%} "
                  f"({res.masked_arcs}/{res.arcs_total} arcs)")
        print(f"    network iters         : full {res.full_network_iterations} "
              f"({res.reduced_network_iterations} reduced)")

    return {
        "instance": inst.name,
        "n": inst.n,
        "mode": "UAV" if inst.params.energy_enabled else "VRPTW",
        "pricer": args.pricer,
        "cg_seed": "singletons",  # v3.4: the pool that was actually solved
        "fleet": fleet,
        "seed_cost": seed["cost"],
        "seed_routes": seed["num_routes"],
        "lp_objective": res.lp_objective,
        "lp_optimal": res.optimal_lp,
        "confirmation_skipped": res.exact_confirmation_skipped,
        "ip_objective": res.ip_objective,
        "ip_routes": res.num_vehicles,
        "iterations": res.iterations,
        "columns_in_pool": res.columns_in_pool,
        "pricing_calls": res.pricing_calls,
        "exact_confirmations": res.exact_confirmation_calls,
        "labels_created": labels,
        "peak_labels_created": res.peak_labels_created,
        "label_cap": res.label_cap,
        "truncated_pricing_calls": res.truncated_pricing_calls,
        "degraded": res.degraded,
        "negative_found": neg,
        "final_relaxed_bound": relaxed_last,
        "wall_clock": res.wall_clock,
        "pp_seconds": res.pp_seconds,
        "rmp_seconds": res.rmp_seconds,
        "other_seconds": res.other_seconds,
        "full_network_iterations": res.full_network_iterations,
        "reduced_network_iterations": res.reduced_network_iterations,
        "masked_pricing_calls": res.masked_pricing_calls,
        "masked_arcs": res.masked_arcs,
        "arcs_total": res.arcs_total,
        "arc_reduction": res.arc_reduction,
        "arcs_removed_total": res.arcs_removed_total,
        "masked_arcs_mean": res.masked_arcs_mean,
        "mean_arc_reduction": res.mean_arc_reduction,
        "tw_scale": args.tw_scale,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="C101")
    ap.add_argument("--sweep", action="store_true", help="run over every instance in the data dir")
    ap.add_argument("--data", default=str(ROOT / "data" / "solomon100"))
    ap.add_argument("--n", type=int, default=0, help="truncate to N customers (0 = all)")
    ap.add_argument("--pricer", default="exact",
                    choices=["exact", "ngroute", "redcost", "greedy"])
    ap.add_argument("--ng-k", type=int, default=10, help="ng-route neighbourhood size")
    ap.add_argument("--tw-scale", type=float, default=1.0,
                    help="time-window narrowing factor, spec §3.3 (1.0 = untouched). "
                         "Record it: a run at another factor is a run on another instance.")
    ap.add_argument("--redcost-ladder", default="10,20,None",
                    help="RedCost-S rung ladder; must end at None (the full network), "
                         "which is what keeps certification possible")
    ap.add_argument("--exact-max-n", type=int, default=40,
                    help="skip exact confirmation above this instance size")
    ap.add_argument("--fleet", type=int, default=0, help="max vehicles (0 = unconstrained)")
    ap.add_argument("--fixed-cost", type=float, default=0.0)
    ap.add_argument("--max-iters", type=int, default=400)
    ap.add_argument("--stall", type=int, default=0,
                    help="stop after N consecutive non-improving iterations (0 = off). "
                         "Without this a degenerate set-covering LP can grind through "
                         "the full --max-iters budget for no gain.")
    ap.add_argument("--max-cols", type=int, default=25)
    ap.add_argument("--uav", type=float, default=0.0, help=">0 enables UAV energy mode (budget multiplier)")
    ap.add_argument("--load-factor", type=float, default=1.0)
    ap.add_argument("--greedy-restarts", type=int, default=60)
    ap.add_argument("--greedy-noise", type=float, default=0.0)
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    t0 = time.perf_counter()
    if args.sweep:
        files = sorted(Path(args.data).glob("*.txt"))
        rows = []
        for f in files:
            try:
                rows.append(run_one(args, f.stem))
            except Exception as exc:
                print(f"    {f.stem}: FAILED {type(exc).__name__}: {exc}")
        print("\n=== summary ===")
        print(f"{'inst':10s} {'n':>4s} {'LP':>12s} {'IP':>12s} {'it':>4s} {'cols':>6s} {'time':>9s}")
        for r in rows:
            print(
                f"{r['instance']:10s} {r['n']:>4d} {r['lp_objective']:>12.2f} "
                f"{r['ip_objective']:>12.2f} {r['iterations']:>4d} "
                f"{r['columns_in_pool']:>6d} {r['wall_clock']:>8.2f}s"
            )
        out = rows
    else:
        out = [run_one(args, args.name)]

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nwrote {args.json_out}")
    print(f"\ntotal elapsed {time.perf_counter() - t0:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
