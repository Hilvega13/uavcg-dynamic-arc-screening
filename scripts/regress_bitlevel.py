"""The §8 dual anchor: invariance anchors plus a BKS red line.

Run:  python scripts/regress_bitlevel.py            # both halves (~80 s)
      python scripts/regress_bitlevel.py --fast     # invariance only (~15 s)

Two anchors, two different jobs
-------------------------------
(1) **Invariance.**  Every change to the labelling pricer, the dominance rule,
    the reduced-cost convention or the master's dual handling is supposed to be
    *neutral*: it may reorder work, it may make a call cheaper, it must not move
    a single digit of the answer.  "Neutral" is a claim about the whole
    pipeline, so the only way to test it is to run the pipeline and compare
    against numbers frozen before the change.  R105 and R107 at n=25, recorded
    to full precision on 2026-09-30 -- before the §3.7 dominance rewrite and
    before the §7.2 arc mask.  The comparison is `==`, not `abs(...) < tol`: 15
    and 43 iterations of accumulated floating-point work each touch millions of
    doubles, so a reordering that mattered would show in the 7th digit, and a
    tolerance wide enough to hide that would hide everything.

(2) **Red line.**  A certified LP may never exceed the same instance's best
    known integer solution.  Unlike (1) this is an *external* number, which
    costs a great deal to get right and is worth the cost exactly once:

    * the instance must be **un-narrowed**, because shortening the time windows
      makes it a different instance whose BKS no longer applies;
    * it must be stated in the **benchmark's own distance convention** -- see
      `uavcg.instance.with_rounded_distances`.  This solver keeps exact
      Euclidean distances; Solomon's published optima round every arc to the
      nearest integer.  Measured on the 100-customer set the two differ by 0.223
      per arc on average, which over the ~110 arcs of a ten-route solution is a
      difference of the same order as the margins this check is meant to police.
      Comparing without the transform would be comparing two different problems;
    * the instance must be **light enough to certify**, which rules out the
      category-2 instances the main study uses.  Probed on 2026-10-02 at n=100,
      un-narrowed, seed RMP duals, a single ng-route k=10 call costs 0.13 s /
      43 k labels on R101, 0.29 s / 91 k on RC101, and 2.07 s (truncating) on
      R102; C101 is cheap per call (0.59 s) but degeneracy-dominated -- 180
      iterations and 36 k columns without certification -- so it is unusable
      here.  R101 certifies in 6.4 s, RC101 in 58.5 s.

    What (2) catches is a gross error: a bug that certifies the wrong LP, a
    wrong reduced-cost sign, a network reduction that leaks into the
    certification path.  It cannot catch a 1e-9 drift, and it is not asked to --
    (1) is what covers that.  The two are why §8 calls this a *dual* anchor.

Protocols below must not drift, or the recorded values stop meaning anything.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uavcg.bench import load_instance
from uavcg.cg import CGConfig, ColumnGeneration
from uavcg.instance import with_rounded_distances
from uavcg.pricers import ExactLabelingPricer, NgRoutePricer

ROOT = Path(__file__).resolve().parents[1]
DATA = str(ROOT / "data" / "solomon100")

#: ---- (1) invariance anchors ------------------------------------------
#: n = 25 customers, Solomon R-class.  Pricing: ng-route k=10 with exact
#: labelling as the fallback pricer; exact_confirmation_max_n = 20,
#: fallback_max_n = 120; max_iterations = 400, max_columns_per_iter = 25,
#: no stall limit.
BIT_ANCHORS = {
    "R105": {"n": 25, "lp": 531.5386420627178, "iterations": 15, "columns": 277},
    "R107": {"n": 25, "lp": 425.26852495439266, "iterations": 43, "columns": 886},
}

#: ---- (2) BKS red line -------------------------------------------------
#: Un-narrowed (tw = 1.0), rounded distances, ng-route k=10 with NO fallback
#: (an exact fallback at n=100 would never return; certification comes from the
#: ng-route relaxation bound, which is the invariant the solver is built on).
#: max_iterations = 400, max_columns_per_iter = 100.
#: `bks` is from data/solomon100/minimum.csv, fetched by
#: scripts/fetch_solomon_cat2.py; `lp` is this project's own reading, recorded
#: on 2026-10-02.
BKS_ANCHORS = {
    "R101": {"bks": 1637.7, "lp": 1596.6666666666663, "iterations": 44},
    "RC101": {"bks": 1619.8, "lp": 1578.3141025641035, "iterations": 38},
}


def run_bit_anchor(name: str, n: int):
    inst = load_instance(name, n=n, data_dir=DATA)
    cfg = CGConfig(
        max_iterations=400,
        max_columns_per_iter=25,
        exact_confirmation_max_n=20,
        fallback_max_n=120,
    )
    cg = ColumnGeneration(
        inst,
        NgRoutePricer(inst, k=10),
        fallback_pricer=ExactLabelingPricer(inst),
        config=cfg,
    )
    return cg.run()


def run_bks_anchor(name: str):
    inst = with_rounded_distances(load_instance(name, n=100, data_dir=DATA))
    cfg = CGConfig(
        max_iterations=400,
        max_columns_per_iter=100,
        exact_confirmation_max_n=20,
        fallback_max_n=None,
    )
    cg = ColumnGeneration(inst, NgRoutePricer(inst, k=10), config=cfg)
    return cg.run()


def check_bits() -> int:
    print("### anchor 1/2: bit-level invariance (no change may move these)")
    fails = 0
    for name, want in BIT_ANCHORS.items():
        t0 = time.perf_counter()
        res = run_bit_anchor(name, want["n"])
        dt = time.perf_counter() - t0

        ok_lp = res.lp_objective == want["lp"]
        ok_it = res.iterations == want["iterations"]
        ok_cols = res.columns_in_pool == want["columns"]
        ok_cert = res.optimal_lp and res.truncated_pricing_calls == 0

        print(f"  {name} n={want['n']}  ({dt:5.1f}s)")
        print(f"    LP       {res.lp_objective!r}  vs {want['lp']!r}"
              f"   {'OK' if ok_lp else 'MOVED'}")
        print(f"    iters    {res.iterations}  vs {want['iterations']}"
              f"   {'OK' if ok_it else 'MOVED'}")
        print(f"    columns  {res.columns_in_pool}  vs {want['columns']}"
              f"   {'OK' if ok_cols else 'MOVED'}")
        print(f"    certified={res.optimal_lp} truncated_calls="
              f"{res.truncated_pricing_calls}   {'OK' if ok_cert else 'FAIL'}")
        if not (ok_lp and ok_it and ok_cols and ok_cert):
            fails += 1
    return fails


def check_bks() -> int:
    print("### anchor 2/2: certified LP <= published BKS, un-narrowed, rounded")
    fails = 0
    for name, want in BKS_ANCHORS.items():
        t0 = time.perf_counter()
        res = run_bks_anchor(name)
        dt = time.perf_counter() - t0

        ok_cert = res.optimal_lp and res.truncated_pricing_calls == 0
        ok_lp = res.lp_objective <= want["bks"]

        print(f"  {name} n=100 tw=1.0 rounded  ({dt:5.1f}s)")
        print(f"    LP       {res.lp_objective!r}  vs recorded {want['lp']!r}"
              f"   {'OK' if res.lp_objective == want['lp'] else 'MOVED'}")
        print(f"    BKS      {want['bks']}   LP <= BKS: {ok_lp}"
              f"   margin {want['bks'] - res.lp_objective:+.4f}")
        print(f"    iters    {res.iterations}  vs {want['iterations']}")
        print(f"    certified={res.optimal_lp}  truncated_calls="
              f"{res.truncated_pricing_calls}   {'OK' if ok_cert else 'FAIL'}"
              "   (ng-route relaxation bound, no exact fallback at n=100)")
        if not (ok_lp and ok_cert):
            fails += 1
    return fails


def main(argv: list | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    fast = "--fast" in argv
    only = ""
    for a in argv:
        if a.startswith("--only="):
            only = a.split("=", 1)[1]

    fails = 0
    if only in ("", "bits"):
        fails += check_bits()
    if not fast and only in ("", "bks"):
        fails += check_bks()

    print("-" * 72)
    if fails:
        print(f"{fails} anchor(s) MOVED -- the change under test is not neutral.")
        return 1
    print("all anchors reproduce."
          + ("" if fast else " Red line holds."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
