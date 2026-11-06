"""Spec §7.5 step 5: land the RedCost-S baseline on one instance per family.

What this produces
------------------
For each of the three landing instances frozen in `data/splits/n60_cat2_v1.json`
(C201, RC201, R201) a paired measurement of two arms on the *same* narrowed
instance:

    baseline  NgRoutePricer(k=10)                     -- plain CG
    redcost   RedCostArcPricer(ladder=10,20,None, k=10)

and the §6.2 columns for each: `#Itr`, PP time, RMP time, total, and -- for the
reduced arm -- the share of arcs removed and the number of iterations that ran
on the full network (the bracketed figure §6.2 asks for).

Why RedCost-S gets its own landing before any learning starts
-------------------------------------------------------------
The paper this project follows reports that RedCost-S -- a rule with no training
at all -- beats its learned ML-S in every class at 200 customers, and that at 400
customers on C2 the learned version is *slower than the baseline*.  A learned
method that is not measured against the free greedy is therefore not measured at
all.  Landing these numbers first also fixes the order of the report: the floor
is established before the thing being compared to it exists.

Fairness rules, enforced rather than intended
---------------------------------------------
Both arms run the same instance, the same `ng_k`, the same iteration and column
caps, the same stopping rule, and the same fallback policy.  At n=60 the exact
fallback is deliberately skipped (`fallback_max_n=40`), so both arms certify the
same way: through the ng-route relaxation bound published on the full network.
That is the only certification available at this size, and having both arms use
it keeps the comparison about the arc reduction rather than about who got the
stronger certificate.

A run that does not certify is reported as such and excluded from the gain
table.  An uncertified run's total time is a measurement of "how long it took to
stop making progress", which is not a quantity the two arms can be compared on.

Usage
-----
    python scripts/run_redcost_baseline.py --name C201 --arm baseline
    python scripts/run_redcost_baseline.py --name C201 --arm redcost
    python scripts/run_redcost_baseline.py --all
    python scripts/run_redcost_baseline.py --report

One run per process is the intended interface: a shell `timeout` around a single
(instance, arm) is then a complete safety net, and the JSONL is appended before
the process exits.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uavcg.cg import CGConfig, ColumnGeneration
from uavcg.pricers import ExactLabelingPricer, NgRoutePricer, RedCostArcPricer
from uavcg.splits import Split, load_split

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "redcost_baseline.jsonl"

#: Identical for both arms. Anything that differs between the arms turns this
#: from a comparison of arc reductions into a comparison of configurations.
#:
#: v3.4 values (spec §3.3 / §6.2).  `STALL_LIMIT=30` with `PLATEAU_TOL=1e-9` is
#: the plateau rule, identical for all four arms; `MAX_COLUMNS_PER_ITER=200` is
#: the batch decision, and it is deliberately *also* identical, because the
#: batch size changes which columns reach the pool first and a per-arm batch
#: would make the arms incomparable for a reason that has nothing to do with
#: arc selection.
MAX_ITERATIONS = 200
MAX_COLUMNS_PER_ITER = 200
STALL_LIMIT = 30
PLATEAU_TOL = 1e-9
NG_K = 10
FALLBACK_MAX_N = 40

#: The v3.4 seed, for every arm.  `cheapest_insertion` induces a
#: high-dimensional optimal dual face on these instances and the run stalls
#: without ever certifying (see `uavcg/heuristics.py::singleton_routes` and
#: the internal seed-diagnosis note, §4.5, not shipped here).  The insertion seed's value is
#: still recorded, as a diagnostic only -- it does not take part in stopping or
#: certification.
SEED = "singletons"


def build_pricers(inst, arm: str):
    """(pricer, fallback) for one arm. The only difference between the arms."""
    if arm == "baseline":
        return NgRoutePricer(inst, k=NG_K), ExactLabelingPricer(inst)
    if arm == "redcost":
        return RedCostArcPricer(inst, k=NG_K), ExactLabelingPricer(inst)
    raise SystemExit(f"unknown arm: {arm}")


def build_seed(inst):
    """The v3.4 seed for every arm, plus its diagnostic counterpart."""
    from uavcg.heuristics import cheapest_insertion, singleton_routes

    diag = 0.0
    for r in cheapest_insertion(inst):
        from uavcg.instance import evaluate_route
        ev = evaluate_route(inst, r)
        if ev.feasible:
            diag += ev.cost
    return singleton_routes(inst), diag


def run_one(split: Split, name: str, arm: str, out: Path) -> dict:
    entry = split.entry(name)
    inst = split.load(entry)
    pricer, fallback = build_pricers(inst, arm)
    seed_routes, insertion_value = build_seed(inst)

    cfg = CGConfig(
        max_iterations=MAX_ITERATIONS,
        max_columns_per_iter=MAX_COLUMNS_PER_ITER,
        stall_limit=STALL_LIMIT,
        plateau_tol=PLATEAU_TOL,
        stabilization="off",
        fallback_max_n=FALLBACK_MAX_N,
        log_every=0,
    )
    cg = ColumnGeneration(inst, pricer, config=cfg, fallback_pricer=fallback,
                          seed_routes=seed_routes)
    res = cg.run()

    labels = sum(s.extra.get("labels_created", 0) for s in res.snapshots)
    rung = getattr(pricer, "rung_n_min", None)
    rec = {
        "instance": inst.name,
        "family": entry.family,
        "bucket": entry.bucket,
        "arm": arm,
        "n": inst.n,
        "tw_scale": entry.tw_scale,
        "ng_k": NG_K,
        # --- v3.4 seed / status ---------------------------------------
        "seed": SEED,
        "seed_columns": len(seed_routes),
        "insertion_value_diagnostic": insertion_value,
        "status": res.status,
        "counts_toward_main_table": res.counts_toward_main_table,
        "degraded": bool(res.degraded),
        # --- §6.2 -----------------------------------------------------
        "iterations": res.iterations,
        "pp_seconds": res.pp_seconds,
        "rmp_seconds": res.rmp_seconds,
        "other_seconds": res.other_seconds,
        "total_seconds": res.wall_clock,
        "full_network_iterations": res.full_network_iterations,
        "reduced_network_iterations": res.reduced_network_iterations,
        "arc_reduction": res.mean_arc_reduction,
        "masked_arcs_mean": res.masked_arcs_mean,
        "arcs_removed_total": res.arcs_removed_total,
        "masked_arcs": res.masked_arcs,
        "arcs_total": res.arcs_total,
        # --- validity -------------------------------------------------
        "lp_objective": res.lp_objective,
        "optimal_lp": bool(res.optimal_lp),
        "degraded": bool(res.degraded),
        "truncated_pricing_calls": res.truncated_pricing_calls,
        "exact_confirmation_skipped": bool(res.exact_confirmation_skipped),
        "peak_labels_created": res.peak_labels_created,
        "label_cap": res.label_cap,
        "pricing_calls": res.pricing_calls,
        "labels_created": labels,
        "columns_in_pool": res.columns_in_pool,
        "ip_objective": res.ip_objective,
        "ip_routes": res.num_vehicles,
        # --- reduced-arm diagnostics ----------------------------------
        "masked_pricing_calls": res.masked_pricing_calls,
        "redcost_final_rung_n_min": rung,
        "redcost_escalations": getattr(pricer, "escalations", None),
        "config": {
            "max_iterations": MAX_ITERATIONS,
            "max_columns_per_iter": MAX_COLUMNS_PER_ITER,
            "stall_limit": STALL_LIMIT,
            "fallback_max_n": FALLBACK_MAX_N,
        },
    }
    with out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def fmt(rec: dict) -> str:
    arc = rec["arc_reduction"]
    return (
        f"  {rec['arm']:9s} itr={rec['iterations']:4d}"
        f"  PP={rec['pp_seconds']:8.2f}s  RMP={rec['rmp_seconds']:7.2f}s"
        f"  tot={rec['total_seconds']:8.2f}s"
        f"  full={rec['full_network_iterations']:4d}"
        f"({rec['reduced_network_iterations']:4d}red)"
        f"  arcs-={'-' if arc is None else f'{arc:6.1%}'}"
        f"  LP={rec['lp_objective']:11.4f}"
        f"  cert={rec['optimal_lp']}"
        f"{'  DEGRADED' if rec['degraded'] else ''}"
    )


def report(out: Path) -> int:
    if not out.exists():
        print(f"no results at {out}")
        return 1
    recs = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    # Keep the last run of each (instance, arm): a re-run supersedes rather than
    # duplicates, so a crash-and-retry does not silently double-count.
    latest: dict = {}
    for r in recs:
        latest[(r["instance"], r["arm"])] = r
    by_inst: dict = {}
    for (name, arm), r in latest.items():
        by_inst.setdefault(name, {})[arm] = r

    print("=" * 108)
    print("§7.5 step 5 — RedCost-S baseline landing (n=60, frozen split n60_cat2_v1)")
    print("=" * 108)
    print(f"{'instance':9s} {'arm':9s} {'#Itr':>5s} {'PP':>8s} {'RMP':>7s} "
          f"{'total':>9s} {'full':>6s} {'red':>5s} {'arc-':>7s} {'LP':>12s} "
          f"{'cert':>5s} {'Gain':>8s}")
    rows = []
    for name in sorted(by_inst):
        arms = by_inst[name]
        base = arms.get("baseline")
        for arm in ("baseline", "redcost"):
            r = arms.get(arm)
            if r is None:
                continue
            gain = ""
            if base is not None and base["total_seconds"] > 0:
                g = 1.0 - r["total_seconds"] / base["total_seconds"]
                gain = f"{g:+7.1%}" if r["optimal_lp"] and base["optimal_lp"] else f"({g:+5.1%})"
            arc = r["arc_reduction"]
            print(f"{name:9s} {arm:9s} {r['iterations']:>5d} {r['pp_seconds']:>7.2f}s "
                  f"{r['rmp_seconds']:>6.2f}s {r['total_seconds']:>8.2f}s "
                  f"{r['full_network_iterations']:>6d} {r['reduced_network_iterations']:>5d} "
                  f"{'-' if arc is None else f'{arc:6.1%}':>7s} "
                  f"{r['lp_objective']:>12.4f} {str(r['optimal_lp']):>5s} {gain:>8s}")
            rows.append((name, arm, r))

    print("-" * 108)
    # Certification is the gate: a gain between two runs where one did not prove
    # its bound is a gain between two different things.
    uncert = [(n, a) for n, a, r in rows if r["status"] != "CERTIFIED"]
    if uncert:
        print(f"NOT CERTIFIED (excluded from any gain claim): {uncert}")

    # v3.4 §6.2: the three states are reported separately and their wall clocks
    # are never averaged together.  A PLATEAU run's total is "how long it took to
    # stop making progress", which is not the same quantity as "how long it took
    # to prove the bound" -- one mean over both would be a number that describes
    # neither.
    for state in ("CERTIFIED", "PLATEAU", "INCOMPLETE"):
        pool = [(n, a, r) for n, a, r in rows if r["status"] == state]
        if not pool:
            continue
        mean_pp = sum(r["pp_seconds"] for _, _, r in pool) / len(pool)
        mean_rmp = sum(r["rmp_seconds"] for _, _, r in pool) / len(pool)
        tag = "enters the main table" if state == "CERTIFIED" else "reported separately"
        print(f"{state:11s} n={len(pool):2d}  mean PP={mean_pp:8.2f}s  "
              f"mean RMP={mean_rmp:7.2f}s   ({tag})")
    mixed = len({r["status"] for _, _, r in rows}) > 1
    if mixed:
        print("** statuses differ across runs: the means above are per-status "
              "on purpose; do NOT collapse them into one **")

    for name in sorted(by_inst):
        arms = by_inst[name]
        if "baseline" in arms and "redcost" in arms:
            b, r = arms["baseline"], arms["redcost"]
            same_lp = abs(b["lp_objective"] - r["lp_objective"]) < 1e-6
            both_ok = (b["status"] == "CERTIFIED" and r["status"] == "CERTIFIED")
            if both_ok:
                print(f"{name}: both CERTIFIED, LP "
                      f"{'agree' if same_lp else 'DIFFER'} "
                      f"({b['lp_objective']:.6f} vs {r['lp_objective']:.6f})")
            else:
                print(f"{name}: comparison withheld — status "
                      f"baseline={b['status']} redcost={r['status']}")
    print(f"\nsource: {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="", help="one instance, e.g. C201")
    ap.add_argument("--arm", default="", choices=["", "baseline", "redcost"])
    ap.add_argument("--all", action="store_true",
                    help="all three landing instances, both arms, in order")
    ap.add_argument("--report", action="store_true", help="print the table and exit")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    out = Path(args.out)

    if args.report:
        return report(out)

    split = load_split()

    if args.all:
        print(f"landing instances: {', '.join(split.landing_instances)}")
        for name in split.landing_instances:
            for arm in ("baseline", "redcost"):
                t = time.perf_counter()
                rec = run_one(split, name, arm, out)
                print(f"[{time.perf_counter() - t:7.2f}s] {name} {fmt(rec)}", flush=True)
        return report(out)

    if not args.name or not args.arm:
        raise SystemExit("give --name and --arm, or --all, or --report")
    t = time.perf_counter()
    rec = run_one(split, args.name, args.arm, out)
    print(f"[{time.perf_counter() - t:7.2f}s] {rec['instance']} {fmt(rec)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
