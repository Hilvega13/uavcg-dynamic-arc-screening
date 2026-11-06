"""Where does a pricing call actually spend its time?

The n=100 main tier lives or dies on one number: microseconds per label.  At
n=60 the measured spread is 15 us (C201) to 45 us (C208), and labels scale as
~n^3.5, so a 3x constant-factor win is the difference between "one instance
takes 30 minutes" and "one instance takes 10 minutes".

There are two candidate explanations and they have very different fixes:

  1. per-label bookkeeping -- the `NgLabel` dataclass, `lab.path + (j,)` tuple
     copy (ngroute.py:140), the dict/stack churn.  Fix: parent pointers.
  2. the dominance scan -- `any(_dominates(other, lab) for other in
     per_node[lab.node])` (ngroute.py:114) is a linear scan with no bucketing,
     so its cost per label grows with the number of labels already stored at
     that node.  Fix: bucket by time or load.

Measured on C201 n=60, k=10 (191,499 labels, 2.65 s):

    ngroute.py:114 <genexpr>   32,653,271 calls   4.811 s tottime
    ngroute.py:55  _dominates  32,461,771 calls   2.736 s tottime
    builtins.any                  191,500 calls   2.365 s tottime
    -> dominance checking is 9.887 s of 10.669 s cumulative = 92.7%

    NgLabel.__init__ (slots)      191,500 calls   0.014 s      = 0.1%
    tuple copy `lab.path + (j,)`  does not appear in the top 15 at all

So it is (2), and it is not close: **170 dominance comparisons per label**.
Hypothesis (1) -- the one Phase 0 parent-pointer work addresses, and the one I
had argued was on the critical path -- is 0.1% of the call.  I had inferred
from the n=20 -> n=60 scaling (per-label cost grew only ~2x while labels grew
48x, and a quadratic term should grow linearly) that the scan could not
dominate.  That inference was wrong and the profile is the reason to distrust
it: the scan's cost per label depends on labels *per node*, not on the total,
and the earlier measurements mixed different duals.

The prize: a 5x constant-factor win here moves C201 from ~18 s to ~4 s per call
at n=100, which is the difference between an affordable main tier and an
unaffordable one.

    python scripts/profile_price.py --name C201 --n 60 --ng-k 10
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uavcg.bench import load_instance
from uavcg.heuristics import cheapest_insertion
from uavcg.instance import evaluate_route
from uavcg.pricers import NgRoutePricer
from uavcg.rmp import MasterProblem

ROOT = Path(__file__).resolve().parents[1]


def seed_duals(inst):
    master = MasterProblem(inst)
    for r in cheapest_insertion(inst):
        ev = evaluate_route(inst, r)
        if ev.feasible:
            master.add(r, ev.cost)
    lp = master.solve_lp()
    return np.asarray(lp.pi, dtype=float), float(lp.sigma)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="C201")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--ng-k", type=int, default=10)
    ap.add_argument("--tw-scale", type=float, default=1.0)
    ap.add_argument("--top", type=int, default=18)
    ap.add_argument("--data", default=str(ROOT / "data" / "solomon100"))
    args = ap.parse_args()

    inst = load_instance(args.name, n=args.n, data_dir=Path(args.data),
                         tw_scale=args.tw_scale)
    pi, sigma = seed_duals(inst)
    pricer = NgRoutePricer(inst, k=args.ng_k)

    # Warm once outside the profiler: module-level numpy/scipy import effects
    # would otherwise be attributed to the first call and hide everything else.
    t0 = time.perf_counter()
    res = pricer.price(pi, sigma=sigma, max_columns=25)
    dt = time.perf_counter() - t0
    labels = res.labels_created
    print(f"instance={inst.name} n={inst.n} k={args.ng_k} tw={args.tw_scale:g}")
    print(f"plain call: {dt:.2f}s  labels={labels}  "
          f"{dt / max(labels, 1) * 1e6:.1f} us/label  cols={len(res.columns)}")
    dom = res.stats.get("dom_checks", 0)
    print(f"dominance: {dom:,} comparisons, "
          f"{res.stats.get('dom_checks_per_label', 0.0):.1f} per label "
          f"(was ~170/label before the time-sorted store, spec §3.7)")
    print(f"min_rc={res.min_reduced_cost:.6f}  "
          f"relaxed_min={res.stats.get('relaxed_min')!r}  "
          f"truncated={res.truncated}  clean={res.is_clean()}")

    prof = cProfile.Profile()
    prof.enable()
    pricer.price(pi, sigma=sigma, max_columns=25)
    prof.disable()

    buf = io.StringIO()
    stats = pstats.Stats(prof, stream=buf)
    stats.sort_stats("tottime")
    print(f"\n--- top {args.top} by tottime (own time, excludes callees) ---")
    stats.print_stats(args.top)
    lines = buf.getvalue().splitlines()
    start = next((i for i, ln in enumerate(lines) if "ncalls" in ln), 0)
    for ln in lines[start:start + args.top + 6]:
        print(ln)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
