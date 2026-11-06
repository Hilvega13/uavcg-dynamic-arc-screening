#!/usr/bin/env python
"""§7.5 step 10 -- ML-S integration acceptance (three checks).

Runs column generation with a static learned arc mask (ML-S) on the n=60
landing instances and reports:
  1. LP agreement with the RedCost-S baseline (bound-neutral: the mask must
     not drop any arc of an optimal route -> LP unchanged).
  2. arc reduction (kept fraction of the candidate set).
  3. iterations / labels / time, no fixed speed-up threshold (R-5).

The mask is mounted through CGConfig.mask_after + mask_hook; the CG loop
owns G/G_r switching and certification.  This script only produces the mask.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uavcg.cg import CGConfig, ColumnGeneration
from uavcg.heuristics import singleton_routes
from uavcg.instance import evaluate_route
from uavcg.pricers import ExactLabelingPricer, NgRoutePricer
from uavcg.splits import load_split
from uavcg import mls

MAX_ITERATIONS = 200
MAX_COLUMNS_PER_ITER = 200
STALL_LIMIT = 30
PLATEAU_TOL = 1e-9
NG_K = 10
FALLBACK_MAX_N = 40


def run(name: str, model_path: str, tau: float, out: Path,
        arm: str = "mls", w: int = 20) -> dict:
    split = load_split()
    entry = split.entry(name)
    inst = split.load(entry)
    if arm == "mls":
        bundle = mls.load_mask_model(model_path)
        keep = mls.predict_keep_mask(inst, bundle, tau)
        mask_after = 1
        hook = mls.make_mls_hook(inst, model_path, tau)
    elif arm == "dynmls":
        keep = None  # computed at hook time from the live trace
        mask_after = w
        hook = mls.make_dynmls_hook(inst, model_path, tau)
    else:
        raise SystemExit(f"unknown arm: {arm}")
    pricer = NgRoutePricer(inst, k=NG_K)
    fallback = ExactLabelingPricer(inst)
    seed = singleton_routes(inst)
    cfg = CGConfig(
        max_iterations=MAX_ITERATIONS,
        max_columns_per_iter=MAX_COLUMNS_PER_ITER,
        stall_limit=STALL_LIMIT,
        plateau_tol=PLATEAU_TOL,
        stabilization="off",
        fallback_max_n=FALLBACK_MAX_N,
        mask_after=mask_after,
        mask_hook=hook,
        log_every=0,
    )
    t = time.perf_counter()
    cg = ColumnGeneration(inst, pricer, config=cfg, fallback_pricer=fallback,
                          seed_routes=seed)
    res = cg.run()
    wall = time.perf_counter() - t
    n_arcs_total = (inst.n + 1) * inst.n
    masked = int(keep.sum()) if keep is not None else res.masked_arcs_mean or 0
    rec = {
        "instance": inst.name, "n": inst.n, "tw_scale": entry.tw_scale,
        "arm": arm, "tau": tau, "model": model_path, "w": w,
        "status": res.status, "iterations": res.iterations,
        "lp_objective": res.lp_objective, "optimal_lp": bool(res.optimal_lp),
        "degraded": bool(res.degraded),
        "pp_seconds": res.pp_seconds, "total_seconds": res.wall_clock,
        "wall_run": round(wall, 1),
        "labels_created": sum(s.extra.get("labels_created", 0)
                              for s in res.snapshots),
        "peak_labels": res.peak_labels_created,
        "arcs_kept": masked, "arcs_total": n_arcs_total,
        "arc_reduction": 1.0 - masked / max(1, n_arcs_total),
        "reduced_network_iterations": res.reduced_network_iterations,
    }
    with out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[{wall:7.1f}s] {name:6s} {arm} tau={tau} itr={res.iterations:3d} "
          f"LP={res.lp_objective:10.4f} arcs_cut={1.0 - masked / max(1, n_arcs_total):.1%} "
          f"status={res.status}", flush=True)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", default="C201,RC201,R201")
    ap.add_argument("--model", default="models/rf_arc_n25_static_narrow.pkl")
    ap.add_argument("--arm", choices=["mls", "dynmls"], default="mls")
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--w", type=int, default=20)
    ap.add_argument("--out", default="results/mls_landing.jsonl")
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # idempotent: drop prior rows for these instances at this tau
    names = set(s.strip() for s in args.names.split(",") if s.strip())
    kept = []
    if out.exists():
        for l in out.read_text(encoding="utf-8").splitlines():
            r = json.loads(l)
            if r.get("tau") == args.tau and r.get("instance") in names:
                continue
            kept.append(l)
    out.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    for name in [s.strip() for s in args.names.split(",") if s.strip()]:
        run(name, args.model, args.tau, out, args.arm, args.w)
    # LP-agreement check vs redcost baseline
    base = {}
    if Path("results/redcost_baseline.jsonl").exists():
        for l in Path("results/redcost_baseline.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(l)
            if r.get("arm") == "redcost":
                base[r["instance"]] = r["lp_objective"]
    print("\n=== LP agreement vs RedCost-S ===")
    for l in out.read_text(encoding="utf-8").splitlines():
        r = json.loads(l)
        b = base.get(r["instance"])
        if b is not None:
            ok = abs(b - r["lp_objective"]) < 1e-4
            print(f"  {r['instance']}: mls={r['lp_objective']:.4f} "
                  f"redcost={b:.4f} {'AGREE' if ok else 'DIFFER'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
