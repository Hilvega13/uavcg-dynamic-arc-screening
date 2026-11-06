"""Decompose the "same bound, smaller pool" effect into its two possible causes.

Where this comes from
---------------------
The smoothing sweep produced one solid, repeatable result and one negative one.
The solid result: at an identical, still-certified LP bound, convex-combination
smoothing left 20-38% fewer columns in the pool.  The negative one: it did not
reduce the wasted-iteration fraction, and on the worst instance it made the
iteration count worse.  So smoothing is not a cure for the plateau, but it does
appear to be a cure for pool bloat.

Pool bloat matters here for a reason that has nothing to do with elegance.  It
is the documented bottleneck of this project at n=50 and beyond, where the cost
of a run is dominated by re-solving an LP over a pool that keeps growing.

Two mechanisms could produce the smaller pool, and they are cheap to separate:

  A.  Smoothing changes WHICH columns the pricer is asked to find.  Pricing
      against pi_stab explores a different neighbourhood than pricing against
      pi, so the batches themselves are different.
  B.  The true-dual verification in the loop is, incidentally, a column
      selection: every column of a stabilized batch is recomputed against the
      real duals and dropped when it is not genuinely negative.  Earlier runs
      measured that dropping directly - R107 alpha=0.5 discarded 1012 columns.

The arms below cross the two switches, which is what makes the attribution
possible:

  off              no stabilization, add everything          (reference)
  smooth           stabilization + true-dual filter
  smooth-nofilter  stabilization, filter disabled            (isolates A)
  partial          no stabilization, keep the k most negative (a third cause:
                   plain reduced-cost selection, no smoothing involved)
  diverse          no stabilization, keep only columns that are not near
                   duplicates of what the LP already uses

A = cols(smooth-nofilter) vs cols(off)
B = cols(smooth) vs cols(smooth-nofilter)

The arms are not just diagnostics.  `partial` and `diverse` are the reference
paper's own column-selection modes (Shen et al., AAAI 2022, section on column
management, where add-all is reported to slow the RMP down), reproduced on this
problem.  If plain reduced-cost selection accounts for the same compression at
no cost, then the interesting statement is not "smoothing compresses the pool"
but "smoothing is an expensive way to do what column selection does directly" -
which is a far more useful thing for the thesis to say.

Correctness
-----------
Every arm is a restriction of the column pool, so none of them can move the LP
optimum; they only change the path taken to it.  The script therefore asserts
the same invariant the stabilization sweep asserted: every arm must report the
same certified bound, and a row that fails is a bug rather than a finding.

Disabling the filter is safe for the *bound* - the loop still ends by pricing
against the true duals, so the certificate is real - but it is not safe for the
pool: columns that are not actually negative do enter it.  That is the point of
the arm, and it is why the no-filter rows must never be the ones a bound is
quoted from in the write-up.

Usage
-----
    python scripts/run_ablation.py --names R105 --pricer rollout-dual
    python scripts/run_ablation.py --names R105,R107 --arms off,smooth,partial
    python scripts/run_ablation.py --from-json results/ablation.json
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

DEFAULT_NAMES = ["R105", "R107", "RC107"]

#: arm -> config overrides.  `k` and `frac` are alternative ways of specifying
#: the "partial" budget; the paper's own mode is a fraction of the batch, but a
#: fixed count is the fairer comparison against the stabilized arms, whose
#: filter drops a variable number per iteration.  Both are therefore run.
ARMS = {
    "off": dict(stab="off", verify=True, sel="all"),
    "smooth": dict(stab="smooth", verify=True, sel="all"),
    "smooth-nofilter": dict(stab="smooth", verify=False, sel="all"),
    "partial-k2": dict(stab="off", verify=True, sel="partial", k=2),
    "partial-half": dict(stab="off", verify=True, sel="partial", frac=0.5),
    # Boxstep at two widths.  One knob, so two points are enough to see
    # whether the arm is in the "still too loose" or the "already frozen"
    # regime; a single point could not tell those apart.
    "boxstep-mid": dict(stab="boxstep", verify=True, sel="all"),
    "boxstep-tight": dict(stab="boxstep", verify=True, sel="all", box_delta=0.15),
    "diverse": dict(stab="off", verify=True, sel="diverse"),
}


def build_pricer(kind: str, inst, args):
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


def run_one(kind, name, arm, args) -> dict:
    inst = load_instance(name, args.n, args.uav, args.load_factor, args.no_tw,
                         Path(args.data))
    pricer, fallback = build_pricer(kind, inst, args)
    spec = ARMS[arm]
    stab = spec["stab"]
    verify = spec["verify"]
    sel = spec["sel"]
    cfg = CGConfig(
        max_iterations=args.max_iters,
        max_columns_per_iter=args.max_cols,
        exact_confirmation_max_n=args.exact_max_n,
        fallback_max_n=args.fallback_max_n,
        stall_limit=args.stall,
        stabilization=stab,
        stabilization_alpha=args.alpha,
        box_delta=spec.get("box_delta", args.box_delta),
        stabilization_warmup=args.warmup,
        verify_against_true_duals=verify,
        column_selection=sel,
        column_selection_k=spec.get("k", 0),
        column_selection_frac=spec.get("frac", 0.5),
        diversity_max_jaccard=args.jaccard,
        log_every=args.log_every,
    )
    if args.verbose:
        print(f"    .. {name} arm={arm} start", flush=True)
    t0 = time.perf_counter()
    res = ColumnGeneration(
        inst, pricer, config=cfg, fallback_pricer=fallback,
        seed_routes=singleton_routes(inst),
        max_vehicles=args.fleet,
    ).run()
    d = res.degeneracy(target=args.target)
    return {
        "instance": name,
        "n": inst.n,
        "pricer": kind,
        "arm": arm,
        "stabilization": stab,
        "verify": verify,
        "selection": sel,
        "k": spec.get("k"),
        "frac": spec.get("frac"),
        "alpha": args.alpha if stab == "smooth" else 0.0,
        "box_delta": spec.get("box_delta", args.box_delta) if stab == "boxstep" else None,
        "seed_lp": d.seed_lp,
        "lp": res.lp_objective,
        "certified": bool(res.optimal_lp),
        "stalled": bool(res.stalled),
        "ip": res.ip_objective,
        "iterations": res.iterations,
        "columns": res.columns_in_pool,
        "offered": res.offered_columns,
        "filtered": res.filtered_columns,
        "sel_dropped": res.dropped_by_selection,
        "wall": time.perf_counter() - t0,
        "max_iters": args.max_iters,
        "degeneracy": d.as_dict(),
    }


def key_of(row) -> str:
    return f"{row['pricer']}|{row['instance']}|{row['arm']}|{row['alpha']}|{row['n']}"


def report(rows, arms, names, args) -> dict:
    summary = {"args": vars(args), "decomposition": {}, "arms": {}}

    for name in names:
        for kind in sorted({r["pricer"] for r in rows}):
            per = [r for r in rows
                   if r["instance"] == name and r["pricer"] == kind]
            if not per:
                continue
            by_arm = {r["arm"]: r for r in per}
            ref = by_arm.get("off") or per[0]

            print()
            print("=" * 112)
            print(f"=== {name}  n={ref['n']}  pricer={kind}  "
                  f"alpha={ref['alpha'] if ref['alpha'] else args.alpha} ===")
            print("=" * 112)
            print(f"  seed LP = {ref['seed_lp']:.3f}   final LP = {ref['lp']:.3f}   "
                  f"gain = {ref['degeneracy']['total_gain']:.3f}   "
                  f"wasted = {ref['degeneracy']['wasted_fraction']:.0%} of "
                  f"column-adding iterations")

            hdr = (f"  {'arm':>16} {'LP':>11} {'cert':>5} {'it':>4} {'cols':>6} "
                   f"{'offered':>8} {'filt':>6} {'seldrop':>8} {'waste':>6} "
                   f"{'plateau':>7} {'wall':>9}")
            print()
            print(hdr)
            print("  " + "-" * (len(hdr) - 2))
            for arm in arms:
                r = by_arm.get(arm)
                if r is None:
                    continue
                d = r["degeneracy"]
                print(f"  {arm:>16} {r['lp']:11.3f} {str(r['certified'])[:1]:>5} "
                      f"{r['iterations']:4d} {r['columns']:6d} {r['offered']:8d} "
                      f"{r['filtered']:6d} {r['sel_dropped']:8d} "
                      f"{d['wasted_fraction']:6.0%} {d['longest_plateau']:7d} "
                      f"{r['wall']:8.2f}s")

            # ---- the attribution itself
            print()
            base = by_arm.get("off")
            nofilt = by_arm.get("smooth-nofilter")
            filt = by_arm.get("smooth")
            if base and nofilt and filt:
                c0, cA, cB = base["columns"], nofilt["columns"], filt["columns"]
                a = (cA / c0 - 1.0) if c0 else 0.0
                b = (cB / cA - 1.0) if cA else 0.0
                tot = (cB / c0 - 1.0) if c0 else 0.0
                print("  decomposition of the pool change (columns vs the off arm)")
                print(f"    A  smoothing changes the batches        {a:+7.1%}   "
                      f"({c0} -> {cA})")
                print(f"    B  true-dual filter acts as selection   {b:+7.1%}   "
                      f"({cA} -> {cB})")
                print(f"    =  total                                {tot:+7.1%}   "
                      f"({c0} -> {cB})")
                summary["decomposition"][f"{kind}|{name}"] = {
                    "off_columns": c0,
                    "nofilter_columns": cA,
                    "filtered_columns": cB,
                    "effect_smoothing_batches": a,
                    "effect_filter_as_selection": b,
                    "effect_total": tot,
                    "off_wall": base["wall"],
                    "nofilter_wall": nofilt["wall"],
                    "filtered_wall": filt["wall"],
                    "off_wasted": base["degeneracy"]["wasted_fraction"],
                    "filtered_wasted": filt["degeneracy"]["wasted_fraction"],
                }
            for alt in ("partial-k2", "partial-half", "diverse"):
                r = by_arm.get(alt)
                if base and r and base["columns"]:
                    print(f"    {alt:<13} (no smoothing at all)     "
                          f"{(r['columns'] / base['columns'] - 1.0):+7.1%}   "
                          f"({base['columns']} -> {r['columns']}), "
                          f"waste {r['degeneracy']['wasted_fraction']:.0%}, "
                          f"{r['wall']:.1f}s")

            # ---- the invariant
            print()
            cert = [r for r in per if r["certified"]]
            vals = {round(r["lp"], 4) for r in cert}
            if len(cert) == len(per) and len(vals) == 1:
                print(f"  [OK] all {len(per)} arms certified the same LP bound "
                      f"({cert[0]['lp']:.3f}) -> every arm is bound-neutral")
            elif len(cert) == len(per):
                print(f"  [BUG] all arms certified but the bounds differ: "
                      f"{sorted(vals)}. A restriction of the pool cannot move the "
                      f"optimum, so this means the loop is leaking a heuristic "
                      f"dual into its stopping rule.")
            else:
                nc = [r["arm"] for r in per if not r["certified"]]
                best = min(per, key=lambda r: r["lp"])
                print(f"  [n/a] {len(per) - len(cert)} arm(s) did not certify "
                      f"({', '.join(nc)}); their LP value is an upper bound from a "
                      f"truncated run and no optimality claim attaches to it.")
                if base:
                    print(f"        at the equal {args.max_iters}-iteration budget the "
                          f"best arm was {best['arm']} at {best['lp']:.3f} vs off's "
                          f"{base['lp']:.3f} ({base['lp'] - best['lp']:+.3f})")

    print()
    print("=" * 112)
    print("per-arm means (only rows that certified, so all bounds are comparable)")
    print("=" * 112)
    hdr = (f"{'arm':>16} {'n':>3} {'iters':>6} {'cols':>7} {'cols vs off':>12} "
           f"{'offered':>8} {'dropped':>8} {'waste':>6} {'wall':>9}")
    print(hdr)
    print("-" * len(hdr))
    for arm in arms:
        sel = [r for r in rows if r["arm"] == arm and r["certified"]]
        if not sel:
            continue
        offs = {f"{r['pricer']}|{r['instance']}": r["columns"]
                for r in rows if r["arm"] == "off" and r["certified"]}
        ratios = [r["columns"] / offs[f"{r['pricer']}|{r['instance']}"]
                  for r in sel if f"{r['pricer']}|{r['instance']}" in offs
                  and offs[f"{r['pricer']}|{r['instance']}"]]
        m = lambda k: sum(r[k] for r in sel) / len(sel)
        md = lambda k: sum(r["degeneracy"][k] for r in sel) / len(sel)
        ratio = (sum(ratios) / len(ratios) - 1.0) if ratios else float("nan")
        dropped = m("filtered") + m("sel_dropped")
        print(f"{arm:>16} {len(sel):3d} {m('iterations'):6.1f} {m('columns'):7.1f} "
              f"{ratio:+11.1%} {m('offered'):8.1f} {dropped:8.1f} "
              f"{md('wasted_fraction'):6.0%} {m('wall'):8.2f}s")
        summary["arms"][arm] = {
            "n_certified": len(sel),
            "mean_iterations": m("iterations"),
            "mean_columns": m("columns"),
            "mean_columns_vs_off": ratio,
            "mean_offered": m("offered"),
            "mean_filtered": m("filtered"),
            "mean_sel_dropped": m("sel_dropped"),
            "mean_wasted_fraction": md("wasted_fraction"),
            "mean_wall": m("wall"),
        }
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", default=",".join(DEFAULT_NAMES))
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--pricer", default="rollout-dual",
                    choices=["exact", "ngroute", "rollout-dual"])
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--box-delta", type=float, default=0.5,
                    help="boxstep half-width in dual-scale units; see CGConfig "
                         "for the definition and the +scale floor")
    ap.add_argument("--alpha", type=float, default=0.3,
                    help="the smoothing weight used by the stabilized arms; "
                         "0.3 measured best in the alpha sweep")
    ap.add_argument("--jaccard", type=float, default=0.5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--target", type=float, default=0.99)
    ap.add_argument("--samples", type=int, default=120)
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--pool", type=int, default=0)
    ap.add_argument("--no-prefix", action="store_true")
    ap.add_argument("--ng-k", type=int, default=10)
    ap.add_argument("--max-iters", type=int, default=100)
    ap.add_argument("--max-cols", type=int, default=25)
    ap.add_argument("--stall", type=int, default=50)
    ap.add_argument("--log-every", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--exact-max-n", type=int, default=20,
                    help="default 20, matching run_experiment.  At n=25 this "
                         "makes the fallback an ng-route relaxation rather than "
                         "the exact labeller; measured on R107 the two reach the "
                         "same certified bound (425.269) with the same iteration "
                         "and column counts, but the exact one costs 204.6 s "
                         "against the relaxation's 12.6 s")
    ap.add_argument("--fallback-max-n", type=int, default=120,
                    help="ceiling for calling the fallback at all.  Do not leave "
                         "this unset while lowering --exact-max-n: the effective "
                         "limit becomes the smaller value, and at n=25 that "
                         "silently disables the fallback entirely, which turns a "
                         "converging run into a stalled one with no error")
    ap.add_argument("--fleet", type=int, default=None)
    ap.add_argument("--uav", type=float, default=0.0)
    ap.add_argument("--load-factor", type=float, default=1.0)
    ap.add_argument("--no-tw", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default=str(ROOT / "data" / "solomon100"))
    ap.add_argument("--out", default=str(RESULT_DIR / "ablation.json"))
    ap.add_argument("--append", action="store_true",
                    help="merge into an existing --out file instead of replacing it, "
                         "so a long sweep can be split across short foreground calls")
    ap.add_argument("--from-json", default=None)
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in ARMS:
            raise SystemExit(f"unknown arm {a!r}; known: {sorted(ARMS)}")
    names = [s.strip() for s in args.names.split(",") if s.strip()]

    rows: list = []
    if args.from_json:
        with open(args.from_json, encoding="utf-8") as fh:
            rows = json.load(fh)["rows"]
        have = {r["instance"] for r in rows}
        names = [n for n in names if n in have] or sorted(have)
        print(f"re-reporting {args.from_json} ({len(rows)} runs)")
    else:
        if args.append and Path(args.out).exists():
            with open(args.out, encoding="utf-8") as fh:
                rows = json.load(fh)["rows"]
            print(f"appending to {args.out} (already {len(rows)} runs)")
        print("=" * 112)
        print(f"column-selection / stabilization ablation  "
              f"(n={args.n}, pricer={args.pricer}, alpha={args.alpha})")
        print("=" * 112)
        for name in names:
            for arm in arms:
                try:
                    r = run_one(args.pricer, name, arm, args)
                except Exception as exc:  # keep a partial sweep usable
                    print(f"  [{name}] arm={arm} failed: {exc}", flush=True)
                    continue
                rows = [x for x in rows if key_of(x) != key_of(r)] + [r]
                print(f"  done {name} {arm}: LP={r['lp']:.3f} "
                      f"cert={r['certified']} it={r['iterations']} "
                      f"cols={r['columns']} wall={r['wall']:.1f}s", flush=True)

    summary = report(rows, arms, names, args)
    if not args.from_json:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump({"args": vars(args), "summary": summary, "rows": rows},
                      fh, indent=2)
        print(f"\nwrote {out}  ({len(rows)} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
