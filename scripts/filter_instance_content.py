"""v3.6 content filter + time-window recalibration over a tier's instances.

Two questions per instance, both answered by the same bounded run
---------------------------------------------------------------
1. **Content** (spec v3.4 §3.3, unchanged).  Does this instance have any column
   generation to do at all?  The filter is direction-safe and needs no
   certificate:

       the restricted master LP is an UPPER bound on the true LP, so if the
       singleton-seeded run's pool LP ever drops below the cheapest_insertion
       value by more than 1e-6, the insertion seed was NOT LP-optimal and the
       instance provably has content.

2. **Admissibility** (spec v3.6 ④, CHANGED).  Under the singleton seed and the
   capped protocol, does the Baseline finish **CERTIFIED within a WORK budget**
   -- total labels created, not seconds -- inside the caps?

Why the criterion moved from seconds to labels
---------------------------------------------
The wall clock on this machine is not an instrument.  Two runs of the same
configuration produced *bit-identical* output -- 108 iterations, LP_min
397.1891082438668 -- and took 935.61 s and 1573.73 s: **1.68x apart**.  On top of
that, when every call is pinned at a time cap the seconds series is a constant, so
"cost decreases as the dual improves" reads the cap rather than the work (C206
scored +0.137 at one cap and -0.974 at another, from the same instance, off a
1.1 s monotone drift inside a 30 s cap).  Labels created are reproducible, so
they are what admissibility is decided on.  Seconds are recorded, and reported,
as an auxiliary column only.

Why the descent is cut at 300,000 LABELS and the certification is not cut at all
--------------------------------------------------------------------------------
The descent handle is a LABEL cap (300,000 labels per call) with a 120 s clock
ceiling that is deliberately loose enough never to bind.  A time cap does not
merely add noise, it destroys determinism, because the cap decides how much work
happens: two back-to-back repeats of RC201(n40,x0.8) at a 1 s cap gave 10
iterations and the same seed LP 3320.8965 but DIFFERENT work (581,265 labels /
LP_min 1123.7996 vs 478,816 / 1132.4247).  Truncating on a label count
reproduced `work_identical=True` with a 0.0% seconds spread.

The certification attempt has no per-call cap at all -- only the hard total.
Measured: a COMPLETE full-network call needs >=1250 s on these instances, so an
attempt bounded by the phase's nominal 1200 s was cut off every single time, came
back with columns instead of proving emptiness, and left the certification half
of the verdict unreachable.  `2400 + 1200 = 3600 s` is still a BOUND (the budget
is checked before each call and each call is handed at most the total reserve
left), and at most `MAX_LAPS` full-network attempts are made per run.

The work gate closes the DESCENT phase, not the run
---------------------------------------------------
Measured defect: with the gate enforced as a wind-down, the n=40 gate (2e7
labels) always closed during descent -- roughly 67 capped calls -- so no
certification attempt was ever made, `certified` was False for EVERY attempt, and
no instance of the tier could ever be admissible.  A certification attempt is
not extra search, it is the measurement.  The gate now closes the descent phase;
the run then spends at most `MAX_LAPS` full-network attempts, and a later lap is
only earned by a previous lap that came back WITH columns (proof of content).
The gate itself stays what the ruling asked for -- a bound on total labels --
and is re-adjudicated against the first measured batch, once, with a record kept
(`--label-gate` in report mode re-reads the same JSONL under a new gate, so no
row has to be re-run to re-decide it).

Bounded and resumable
---------------------
Each (instance, tw_scale) attempt carries a work gate, a wall fuse and the hard
phase budgets, all enforced inside the pricer; one flushed line per call goes to
the log, and one record per attempt to JSONL.  Re-running skips attempts already
present, so a budget overrun can be resumed instead of restarted.  Log goes to a
FILE, never a pipe -- a `| tail` buffered this measurement into oblivion once.

Usage
-----
    python -u scripts/filter_instance_content.py --split data/splits/n40_cat2_v1.json \\
        --out results/_v36_filter_n40.jsonl > results/_v36_filter_n40.log 2>&1
    python -u scripts/filter_instance_content.py --report --out results/_v36_filter_n40.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uavcg.cg import CGConfig, ColumnGeneration
from uavcg.degeneracy import seed_dual_face
from uavcg.heuristics import cheapest_insertion, singleton_routes
from uavcg.instance import evaluate_route
from uavcg.pricers import NgRoutePricer
from uavcg.splits import load_split

from _probe_n60_cost import (
    CERT_BUDGET,
    CERT_LABELS,
    CERT_SECONDS,
    DESCENT_BUDGET,
    DESCENT_LABELS,
    DESCENT_SECONDS,
    MAX_LAPS,
    WALL_FUSE_SECONDS,
    TimedPricer,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT = ROOT / "data" / "splits" / "n40_cat2_v1.json"
DEFAULT_OUT = ROOT / "results" / "_v36_filter.jsonl"

#: spec v3.6 §3.3 / §3.5 item 3
MAX_ITERATIONS = 200
STALL_LIMIT = 30
PLATEAU_TOL = 1e-9
MAX_COLS = 200            # spec v3.4 §6.2 batch decision
#: v3.6 ④: admissibility is decided on WORK.  A single knob per tier, and the
#: value is explicitly provisional -- the first re-screen batch may move it once,
#: with the change written into the diagnostic doc.
LABEL_GATE_BY_N = {40: 2e7, 60: 1e8}
#: v3.6 ④ ladder, by tier.  The main tier (n=40) starts at x0.8 and narrows with
#: a floor of x0.5.  The challenge tier (n=60) starts lower: n=60 is
#: intrinsically more expensive, and its entries are pre-decided.
LADDER_BY_N = {40: (0.8, 0.6, 0.5), 60: (0.5, 0.4, 0.35)}
CONTENT_TOL = 1e-6
#: v3.6 ④: content+admissible target band, and the "rework" trigger.
MIN_CONTENT_ADMISSIBLE = 6


def insertion_value(inst) -> float:
    """Cost of the cheapest_insertion seed. Diagnostic only (spec v3.4)."""
    total = 0.0
    for r in cheapest_insertion(inst):
        ev = evaluate_route(inst, r)
        if ev.feasible:
            total += ev.cost
    return total


def label_gate_for(n: int, override: float = 0.0) -> float:
    if override:
        return float(override)
    return float(LABEL_GATE_BY_N.get(int(n), LABEL_GATE_BY_N[40]))


def ladder_for(n: int, override=None) -> tuple:
    if override:
        return tuple(float(x) for x in override)
    return LADDER_BY_N.get(int(n), LADDER_BY_N[40])


def attempts_for(entry, ladder=None) -> list:
    """tw_scale rungs to try: start at the frozen factor, then tighten."""
    start = entry.tw_scale
    rungs = ladder if ladder else ladder_for(entry.n)
    return [s for s in rungs if s <= start + 1e-12]


def run_one(split, entry, tw_scale, out, *, label_gate: float,
            descent_seconds: float = DESCENT_SECONDS,
            cert_seconds: float = CERT_SECONDS,
            descent_budget: float = DESCENT_BUDGET,
            cert_budget: float = CERT_BUDGET,
            wall_fuse: float = WALL_FUSE_SECONDS,
            descent_labels: float = DESCENT_LABELS,
            max_laps: int = MAX_LAPS):
    inst = split.load(entry.with_tw_scale(tw_scale))

    ins_val = insertion_value(inst)
    routes = singleton_routes(inst)
    face = seed_dual_face(inst, routes)

    # v3.6 protocol: descent calls cut at 300,000 LABELS that KEEP their columns,
    # a certification attempt bounded only by the hard total when a descent call
    # comes back empty and blind (or when the work gate closes the descent), a
    # work gate on total labels, a 2 h fuse, and 2400 + 1200 = 3600 s enforced as
    # a bound by the pricer itself.
    pricer = TimedPricer(NgRoutePricer(inst, k=10),
                         descent_seconds=descent_seconds,
                         cert_seconds=cert_seconds,
                         descent_budget=descent_budget,
                         cert_budget=cert_budget,
                         descent_labels=descent_labels,
                         label_budget=label_gate,
                         gate_seconds=wall_fuse,
                         max_laps=max_laps,
                         label=f"{entry.name}/x{tw_scale:g}")
    cfg = CGConfig(
        max_iterations=MAX_ITERATIONS,
        max_columns_per_iter=MAX_COLS,
        stall_limit=STALL_LIMIT,
        plateau_tol=PLATEAU_TOL,
        fallback_max_n=40,
        stabilization="off",
        log_every=0,
    )
    cg = ColumnGeneration(inst, pricer, config=cfg, fallback_pricer=None,
                          seed_routes=routes)

    base = {
        "instance": entry.name, "family": entry.family, "bucket": entry.bucket,
        "n": entry.n, "tw_scale": tw_scale, "seed": "singletons",
        #: The split's frozen factor, carried so the write-back knows what is
        #: being replaced without re-reading the split.
        "frozen_tw_scale": entry.tw_scale,
        "seed_columns": len(routes), "seed_lp": face.objective,
        "seed_face_is_point": face.is_point,
        "seed_face_free": face.free_coordinates,
        "seed_face_max_span": face.max_span,
        "insertion_value": ins_val,
        # ---- v3.6 budget/goal accounting ------------------------------------
        "label_gate": label_gate,
        "descent_seconds": descent_seconds, "cert_seconds": cert_seconds,
        "descent_budget": descent_budget, "cert_budget": cert_budget,
        "descent_labels": descent_labels, "max_laps": max_laps,
        "cert_labels": CERT_LABELS,
        "wall_fuse": wall_fuse,
    }

    t0 = time.perf_counter()
    try:
        res = cg.run()
    except TimeoutError as exc:
        # Safety net only: `TimedPricer` winds the run down gracefully instead of
        # raising, precisely so this branch is not reached (the raise used to
        # discard every snapshot, which is how a 16 h sweep produced nine
        # verdict-free rows).  Kept so a budgeted attempt can never crash a sweep.
        lp_min = min((s.lp_objective for s in getattr(res, "snapshots", [])
                      if isinstance(s.lp_objective, float)), default=None)
        rec = dict(base, status="BUDGET", crash=str(exc)[:80],
                   pp_seconds=pricer.spent, pricing_calls=pricer.calls,
                   total_labels=pricer.labels,
                   untruncated_empty_calls=pricer.untruncated_empty_calls,
                   laps=pricer.laps,
                   unscheduled_truncations=0,
                   wind_down_reason=pricer.wind_down_reason,
                   wall_seconds=time.perf_counter() - t0,
                   lp_min=lp_min, lp_last=lp_min,
                   content=bool(lp_min is not None
                                and lp_min < ins_val - CONTENT_TOL),
                   zero_content=False, admissible=False,
                   certified=False, time_capped=True)
        append(out, rec)
        return rec

    trace = [s.lp_objective for s in res.snapshots]
    lp_min = min(trace) if trace else res.lp_objective
    content = bool(lp_min < ins_val - CONTENT_TOL)
    # v3.6 ③: "certified" means an empty call that COMPLETED, not an empty call
    # that was cut off.  `optimal_lp` is the master's own verdict and already
    # requires `certifies_no_negative` on real duals; the trace counter is the
    # independent cross-check, and both must agree before a certificate is
    # recorded.  Deliberately NOT `res.status == "CERTIFIED"`: under a scheduled
    # truncation protocol the status carries §3.5's training-data rule, which is
    # a different question (see `CGResult.degraded`).
    certified = bool(res.optimal_lp
                     and pricer.untruncated_empty_calls >= 1)
    zero_content = bool(certified
                        and abs(res.lp_objective - ins_val) < CONTENT_TOL)
    # v3.6 ④: admissibility is a property of the WORK spent, with the seconds
    # kept only as an auxiliary column.  `wind_down_reason` is deliberately NOT
    # part of the test any more (P3).  Under this protocol the descent phase is
    # *supposed* to end on the work gate -- that is what a work gate is for --
    # so disqualifying that reason disqualified every run that ever certified.
    admissible = bool(certified and pricer.labels <= label_gate)

    rec = dict(
        base,
        status=res.status,
        certified=certified, stalled=bool(res.stalled),
        degraded=bool(res.degraded),
        #: Truncations the protocol did NOT schedule: the clock, or the
        #: intrinsic label cap.  These are what §3.5 `degraded` sees.
        unscheduled_truncations=res.unscheduled_truncations,
        #: Did the wall-clock cap fire anywhere in this run?  Distinct from
        #: `degraded`, which also covers the label cap.
        time_capped=any(t.get("time_truncated") for t in pricer.trace),
        #: v3.6 protocol accounting.
        cert_escalations=pricer.cert_escalations,
        #: Full-network certification attempts actually launched (P3) and
        #: whether the work gate closed the descent phase.
        laps=pricer.laps, descent_closed=bool(pricer.descent_closed),
        untruncated_empty_calls=pricer.untruncated_empty_calls,
        wind_down_reason=pricer.wind_down_reason,
        iterations=res.iterations, pricing_calls=res.pricing_calls,
        lp_first=trace[0] if trace else None,
        lp_last=res.lp_objective, lp_min=lp_min,
        moved=sum(1 for a, b in zip(trace, trace[1:]) if abs(b - a) > 1e-9),
        columns_in_pool=res.columns_in_pool,
        # ---- PRIMARY cost measure (v3.6): work, not seconds ----------------
        total_labels=pricer.labels,
        max_labels=max((t["labels"] for t in pricer.trace), default=0),
        descent_labels=sum(t["labels"] for t in pricer.trace
                           if t["phase"] == "desc"),
        cert_labels=sum(t["labels"] for t in pricer.trace
                        if t["phase"] == "cert"),
        # ---- auxillary only: recorded, never used to decide ----------------
        pp_seconds=res.pp_seconds, rmp_seconds=res.rmp_seconds,
        wall_seconds=res.wall_clock,
        max_call_seconds=max((t["seconds"] for t in pricer.trace), default=0.0),
        truncated=res.truncated_pricing_calls,
        content=content, zero_content=zero_content, admissible=admissible,
    )
    append(out, rec)
    return rec


def append(out: Path, rec: dict) -> None:
    with out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_records(out: Path):
    if not out.exists():
        return []
    return [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()
            if l.strip()]


def report(out: Path, gate_override: float = 0.0) -> int:
    rows = load_records(out)
    if not rows:
        print(f"no records at {out}")
        return 1
    latest = {}
    for r in rows:
        latest[(r["instance"], r["tw_scale"])] = r
    tier_n = rows[-1].get("n", 40)
    gate = float(gate_override or rows[-1].get("label_gate")
                 or label_gate_for(tier_n))
    # Re-adjudicate against the gate being reported under, straight from the
    # stored work columns.  This is what makes the once-permitted gate adjustment
    # (spec v3.6 ④) cost nothing: the same JSONL is re-decided under the new
    # value without re-running a single attempt, and every row still records the
    # gate it was actually run under.
    if gate_override:
        changed = 0
        for r in latest.values():
            new = bool(r.get("certified")
                       and (r.get("total_labels") or 0) <= gate)
            changed += (new != bool(r.get("admissible")))
            r["admissible_under_report_gate"] = new
        print(f"** gate adjusted to {gate:.1e} labels for this report "
              f"({changed} rows change verdict; run gate was "
              f"{rows[-1].get('label_gate'):.1e}) **")

    print("=" * 140)
    print(f"v3.6 content filter + recalibration -- tier n={tier_n}, singleton seed, "
          f"stall={STALL_LIMIT}/tol={PLATEAU_TOL:g}, max_cols={MAX_COLS}, "
          f"descent {rows[-1].get('descent_labels', DESCENT_LABELS):,.0f} labels/"
          f"{rows[-1].get('descent_seconds', DESCENT_SECONDS):.0f}s, "
          f"cert <= {rows[-1].get('max_laps', MAX_LAPS)} laps (hard total "
          f"{rows[-1].get('descent_budget', DESCENT_BUDGET) + rows[-1].get('cert_budget', CERT_BUDGET):.0f}s), "
          f"WORK gate {gate:.1e} labels, <= {MAX_ITERATIONS} iters")
    print("=" * 140)

    def _adm(r):
        """Admissible under the gate this report prints, not the run-time one."""
        return r.get("admissible_under_report_gate", r.get("admissible"))
    print(f"{'inst':6s} {'fam':4s} {'x':>5s} {'face':>5s} {'status':10s} "
          f"{'#Itr':>5s} {'ins_val':>11s} {'LP_min':>11s} {'TOTAL LABELS':>13s} "
          f"{'gate%':>7s} {'empty#':>6s} {'laps':>4s} {'content':>8s} "
          f"{'admis':>6s} {'PP(s) aux':>10s}")
    for key in sorted(latest):
        r = latest[key]
        f = lambda v: "n/a" if v is None else f"{v:11.4f}"  # noqa: E731
        labels = r.get("total_labels")
        pct = "-" if labels is None else f"{100.0 * labels / gate:6.0f}%"
        print(f"{r['instance']:6s} {r['family']:4s} {r['tw_scale']:>5.2f} "
              f"{'pt' if r.get('seed_face_is_point') else 'wide':>5s} "
              f"{r['status']:10s} {r.get('iterations', 0):>5d} "
              f"{r['insertion_value']:>11.4f} {f(r.get('lp_min'))} "
              f"{labels if labels is not None else 'n/a':>13} {pct:>7s} "
              f"{r.get('untruncated_empty_calls', '-'):>6} "
              f"{r.get('laps', 0):>4d} "
              f"{'YES' if r.get('content') else ('ZERO' if r.get('zero_content') else '?'):>8s} "
              f"{'yes' if _adm(r) else 'no':>6s} "
              f"{r.get('pp_seconds', 0):>9.1f}s")
    print("-" * 140)

    # Work split, for the once-permitted gate adjustment (spec v3.6 ④).  The gate
    # is on TOTAL work, which is descent search + the fixed cost of the
    # certification laps; printing the two separately is what lets the PI see
    # whether the number that moved is the instance or the protocol.
    print(f"{'inst':6s} {'x':>5s} {'total labels':>13s} {'descent':>13s} "
          f"{'cert laps':>13s} {'laps':>4s} {'desc share':>10s} {'wind':>10s}")
    for key in sorted(latest):
        r = latest[key]
        tot = r.get("total_labels")
        des = r.get("descent_labels")
        crt = r.get("cert_labels")
        share = "-" if not tot else f"{100.0 * (des or 0) / tot:9.0f}%"
        print(f"{r['instance']:6s} {r['tw_scale']:>5.2f} "
              f"{tot if tot is not None else 'n/a':>13} "
              f"{des if des is not None else 'n/a':>13} "
              f"{crt if crt is not None else 'n/a':>13} "
              f"{r.get('laps', 0):>4d} {share:>10s} "
              f"{str(r.get('wind_down_reason') or '-'):>10s}")
    print("-" * 140)

    # One verdict per instance.  Content and admissibility are properties of the
    # INSTANCE, not of one rung: a looser rung that proved content proves it for
    # the instance, and throwing that away because a tighter rung -- a different,
    # more narrowed instance -- stayed indeterminate is how C201 and RC205 were
    # first misread as unknown.  Both are "any rung".  The rung reported
    # alongside is the LOOSEST admissible one: the least narrowed instance that
    # still passes the gate, which is the one to freeze.
    by_inst = {}
    for (name, _x), r in latest.items():
        by_inst.setdefault(name, []).append(r)
    verdict = {}
    for name, rs in by_inst.items():
        adm = [r for r in rs if _adm(r)]
        chosen = (max(adm, key=lambda r: r["tw_scale"]) if adm
                  else max(rs, key=lambda r: r["tw_scale"]))
        verdict[name] = dict(
            chosen=chosen,
            content=any(r.get("content") for r in rs),
            zero=any(r.get("zero_content") for r in rs),
            admissible=bool(adm),
        )
    have = sorted(n for n, v in verdict.items() if v["content"])
    zero = sorted(n for n, v in verdict.items() if v["zero"])
    unk = sorted(n for n, v in verdict.items()
                 if not v["content"] and not v["zero"])
    adm = sorted(n for n, v in verdict.items() if v["admissible"])
    notadm = sorted(n for n, v in verdict.items() if not v["admissible"])
    print(f"instances measured: {len(verdict)}")
    print(f"  has content      ({len(have):2d}): {', '.join(have) or '-'}")
    print(f"  ZERO content     ({len(zero):2d}): {', '.join(zero) or '-'}")
    print(f"  indeterminate    ({len(unk):2d}): {', '.join(unk) or '-'}")
    print(f"  admissible       ({len(adm):2d}): {', '.join(adm) or '-'}")
    print(f"  NOT admissible   ({len(notadm):2d}): {', '.join(notadm) or '-'}")
    if adm:
        print(f"\n  recalibrated tw_scale (loosest admissible rung):")
        for n in adm:
            r = verdict[n]["chosen"]
            frozen = r.get("frozen_tw_scale")
            frozen_s = "n/a" if frozen is None else f"x{frozen:.2f}"
            labs = r.get("total_labels")
            lp_last = r.get("lp_last")
            lp_s = "n/a" if lp_last is None else f"{lp_last:.4f}"
            print(f"    {n:6s} x{r['tw_scale']:.2f}  (frozen {frozen_s})"
                  f"  labels={labs if labs is not None else 'n/a':>12}"
                  f"  iters={r.get('iterations', 0)}"
                  f"  LP={lp_s}  ins={r['insertion_value']:.4f}"
                  f"  laps={r.get('laps', 0)}"
                  f"  PP={r.get('pp_seconds', 0.0):7.1f}s (aux)")
    both = sorted(set(have) & set(adm))
    print(f"  content + admissible ({len(both):2d}): {', '.join(both) or '-'}")
    if len(both) < MIN_CONTENT_ADMISSIBLE:
        print(f"  ** content+admissible < {MIN_CONTENT_ADMISSIBLE} -> MATRIX REWORK "
              f"triggered (spec v3.6 ④); report to the PI **")
    else:
        print(f"  matrix target met: {len(both)} in the required "
              f"{MIN_CONTENT_ADMISSIBLE}-8 band (spec v3.6 ④)")
    print(f"\nsource: {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=str(DEFAULT_SPLIT))
    ap.add_argument("--names", default="")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--label-gate", type=float, default=0.0,
                    help="work gate: max total labels created (0 = tier default).  "
                         "In --report mode this RE-ADJUDICATES the stored rows "
                         "under the new value without re-running anything")
    ap.add_argument("--ladder", default="", help="comma list, e.g. 0.8,0.6,0.5")
    ap.add_argument("--max-call-seconds", type=float, default=DESCENT_SECONDS,
                    help="descent-round per-call TIME ceiling (v3.6: 120 s, and it "
                         "is deliberately loose -- the label cap below is what "
                         "actually truncates a descent round)")
    ap.add_argument("--descent-labels", type=float, default=DESCENT_LABELS,
                    help="descent-round per-call LABEL cap (v3.6: 300000) -- the "
                         "deterministic truncation knob")
    ap.add_argument("--cert-seconds", type=float, default=CERT_SECONDS,
                    help="NOMINAL certification per-call cap; a certification "
                         "attempt is really bounded by the hard total")
    ap.add_argument("--max-laps", type=int, default=MAX_LAPS,
                    help="max full-network certification attempts per run (v3.6: 3)")
    ap.add_argument("--descent-budget", type=float, default=DESCENT_BUDGET)
    ap.add_argument("--cert-budget", type=float, default=CERT_BUDGET)
    ap.add_argument("--wall-fuse", type=float, default=WALL_FUSE_SECONDS,
                    help="anti-hang fuse, seconds (v3.6: 7200)")
    ap.add_argument("--wall-budget", type=float, default=24 * 3600.0,
                    help="whole-sweep guard, checked between attempts")
    args = ap.parse_args()

    out = Path(args.out)
    if args.report:
        return report(out, gate_override=args.label_gate)

    split = load_split(Path(args.split))
    entries = ([split.entry(n) for n in args.names.split(",") if n.strip()]
               if args.names else list(split.entries))
    # cheapest first: the frozen blind-pricing probe reading is a lower bound on
    # the honest cost, so it orders the sweep without pretending to predict it.
    entries = sorted(entries, key=lambda e: e.probe_seconds)
    #: `--ladder` used to be passed through as a list of STRINGS, so any use of
    #: the flag the help text advertises ("0.8,0.6,0.5") died in `attempts_for`
    #: with `TypeError: '<=' not supported between 'str' and 'float'`.  The
    #: default ladder never went through here (`ladder_for()` already floats it),
    #: which is why the bug never surfaced in a sweep — only on the documented
    #: flag.  Measured 2026-10-07.
    ladder = ([float(x) for x in args.ladder.split(",") if x.strip()]
              if args.ladder.strip() else None)

    done = {(r["instance"], r["tw_scale"]) for r in load_records(out)}
    gates = {label_gate_for(e.n, args.label_gate) for e in entries}
    print(f"{len(entries)} instances from {Path(args.split).name}, "
          f"work gate {sorted(gates)}, descent cap "
          f"{args.descent_labels:,.0f} labels ({args.max_call_seconds:.0f}s "
          f"ceiling), cert <= {args.max_laps} full-network laps, reserves "
          f"{args.descent_budget:.0f}+{args.cert_budget:.0f}="
          f"{args.descent_budget + args.cert_budget:.0f}s (hard), "
          f"fuse {args.wall_fuse:.0f}s, sweep guard {args.wall_budget:.0f}s, "
          f"{len(done)} attempts already on disk", flush=True)

    t_start = time.perf_counter()
    stop = False
    for entry in entries:
        if stop:
            break
        gate = label_gate_for(entry.n, args.label_gate)
        decided = any(r["instance"] == entry.name and r.get("admissible")
                      for r in load_records(out))
        for x in attempts_for(entry, ladder):
            if (entry.name, x) in done:
                continue
            if decided:
                break
            elapsed = time.perf_counter() - t_start
            if elapsed > args.wall_budget:
                print(f"\n!! sweep guard {args.wall_budget:.0f}s exhausted after "
                      f"{elapsed:.0f}s -- stopping.  JSONL is resumable: re-run "
                      f"the same command to continue.", flush=True)
                stop = True
                break
            t = time.perf_counter()
            rec = run_one(split, entry, x, out, label_gate=gate,
                          descent_seconds=args.max_call_seconds,
                          cert_seconds=args.cert_seconds,
                          descent_budget=args.descent_budget,
                          cert_budget=args.cert_budget,
                          descent_labels=args.descent_labels,
                          max_laps=args.max_laps,
                          wall_fuse=args.wall_fuse)
            lp_l = rec.get("lp_last")
            labs = rec.get("total_labels")
            print(f"[{time.perf_counter() - t:7.1f}s] {entry.name:6s} "
                  f"x{x:<5g} {rec['status']:10s} "
                  f"face={'point' if rec['seed_face_is_point'] else 'WIDE'} "
                  f"del={rec['insertion_value'] - (lp_l if lp_l is not None else float('nan')):+11.4f} "
                  f"LP_min={lp_l if lp_l is None else f'{lp_l:11.4f}'} "
                  f"labels={labs if labs is not None else 'n/a':>12} "
                  f"({'-' if labs is None else f'{100.0 * labs / gate:4.0f}%'}) "
                  f"empty#={rec.get('untruncated_empty_calls', '-')} "
                  f"wind={rec.get('wind_down_reason') or '-'} "
                  f"content={'YES' if rec.get('content') else ('ZERO' if rec.get('zero_content') else '?')} "
                  f"admissible={'yes' if rec.get('admissible') else 'no'}",
                  flush=True)
            if rec.get("admissible"):
                decided = True
    return report(out)


if __name__ == "__main__":
    raise SystemExit(main())
