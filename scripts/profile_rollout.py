"""Where does a learned rollout actually spend its time?

The learned pricing arm costs several times the dual-guided one per sample, and
the experiments show the extra compute buys nothing.  Before optimising
anything it is worth measuring which part is slow, because the intuitive answer
("model inference") is usually wrong and the actual answer changes what to fix.

Three counters, all measured on one pricing call at n=25:

    mask        the feasibility prune, one call per decision step
    features    the arc feature block, one call per decision step
    inference   the scorer's `logit`, one call per decision step

Usage:
    python scripts/profile_rollout.py --names R105 --n 25
    python scripts/profile_rollout.py --names R105,RC107 --n 25 --repeats 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from uavcg.bench import DEFAULT_DATA, load_instance
from uavcg.cg import CGConfig, ColumnGeneration
from uavcg.features import FeatureContext, RolloutState, arc_features
from uavcg.heuristics import cheapest_insertion
from uavcg.learning import ArcScorer
from uavcg.pricers import NgRoutePricer, RolloutPricer
from uavcg.rmp import MasterProblem

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results"


def warm_dual(inst, seed_routes):
    """A dual solution that is not all-zero, so the profile is representative."""
    m = MasterProblem(inst)
    for r in seed_routes:
        from uavcg.instance import evaluate_route
        ev = evaluate_route(inst, r)
        if ev.feasible:
            m.add(r, ev.cost)
    if len(m) == 0:
        raise RuntimeError("no feasible seed column")
    return m.solve_lp()


def timeit(fn, repeats: int) -> float:
    """Seconds per call, best of `repeats` (best-of beats mean for micro work)."""
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def profile_instance(name: str, args) -> dict:
    inst = load_instance(name, args.n, args.uav, args.load_factor, args.no_tw,
                         Path(args.data))
    routes = [r for r in cheapest_insertion(inst)]
    lp = warm_dual(inst, routes)
    pi, sigma, fixed = lp.pi, lp.sigma, 0.0

    ctx = FeatureContext(inst)
    rng = np.random.default_rng(0)
    scorer = ArcScorer.load(args.model) if args.model else None

    # ---- a representative mid-route state -------------------------------
    path = list(routes[0])[: max(1, len(routes[0]) // 2)]
    node, t, load, dist_so_far, pi_sum, energy_lb = 0, 0.0, 0.0, 0.0, 0.0, 0.0
    visited = np.zeros(inst.n + 1, dtype=bool)
    visited[0] = True
    for j in path:
        t += inst.service[node] + inst.travel_time[node, j]
        t = max(t, float(inst.ready[j]))
        load += float(inst.demand[j])
        dist_so_far += float(inst.dist[node, j])
        pi_sum += float(pi[j - 1])
        visited[j] = True
        node = j
    st = RolloutState(node=node, time=t, load=load, dist_so_far=dist_so_far,
                      pi_sum=pi_sum, energy_lb=energy_lb, step=len(path),
                      path=tuple(path), visited=visited)
    cand = ctx.mask(st)

    R = args.repeats
    t_mask = timeit(lambda: ctx.mask(st), R)
    # Both feature paths are measured.  They are not the same measurement and
    # the difference is the point of the cache: `arc_features` without `dual`
    # rebuilds the dual-dependent tables on every call, which is what the
    # labelling and training paths do, while the pricer builds them once per
    # pricing call and gathers from them several hundred times.  Profiling only
    # the uncached path is how a cache can look like it does nothing.
    t_feat = timeit(lambda: arc_features(ctx, st, cand, pi, None), R)
    dual_block = ctx.dual_block(pi)
    t_feat_cached = timeit(
        lambda: arc_features(ctx, st, cand, pi, dual=dual_block), R)
    t_dual_block = timeit(lambda: ctx.dual_block(pi), R)
    t_succ = timeit(lambda: ctx.succ_counts(st, cand), R)
    t_inf = float("nan")
    if scorer is not None:
        X = arc_features(ctx, st, cand, pi, None)
        t_inf = timeit(lambda: scorer.logit(X), R)

    # ---- a whole pricing call, by arm -----------------------------------
    calls = {}
    for kind, pricer in (
        ("dual", RolloutPricer(inst, scoring="dual", n_samples=args.samples,
                               beta=args.beta, seed=0)),
    ):
        calls[kind] = timeit(
            lambda p=pricer: p.price(pi, sigma=sigma, fixed_cost=fixed,
                                     max_columns=25), max(3, R // 4))
    if scorer is not None:
        p = RolloutPricer(inst, scoring="model", scorer=scorer,
                          n_samples=args.samples, beta=args.beta, seed=0)
        calls["model"] = timeit(
            lambda: p.price(pi, sigma=sigma, fixed_cost=fixed, max_columns=25),
            max(3, R // 4))
    ng = NgRoutePricer(inst, k=args.ng_k)
    calls["ngroute"] = timeit(
        lambda: ng.price(pi, sigma=sigma, fixed_cost=fixed, max_columns=25),
        max(2, R // 8))

    return {
        "instance": name, "n": inst.n,
        "cand_size": int(cand.size),
        "mask_ms": 1e3 * t_mask,
        "features_ms": 1e3 * t_feat,
        "features_cached_ms": 1e3 * t_feat_cached,
        "dual_block_ms": 1e3 * t_dual_block,
        "of_which_succ_ms": 1e3 * t_succ,
        "inference_ms": 1e3 * t_inf if np.isfinite(t_inf) else None,
        "call_dual_ms": 1e3 * calls["dual"],
        "call_model_ms": 1e3 * calls.get("model", float("nan")),
        "call_ngroute_ms": 1e3 * calls["ngroute"],
        "call_ratio_model_over_dual": (calls.get("model", float("nan"))
                                      / calls["dual"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--names", default="R105,RC107")
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--uav", type=float, default=0.0)
    ap.add_argument("--load-factor", type=float, default=1.0)
    ap.add_argument("--no-tw", action="store_true")
    ap.add_argument("--model", default=str(ROOT / "models" / "arc_scorer_n25.pkl"))
    ap.add_argument("--ng-k", type=int, default=10)
    ap.add_argument("--samples", type=int, default=60)
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--repeats", type=int, default=40)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    rows = []
    hdr = (f"{'inst':7s} {'|C|':>4s} {'mask':>8s} {'feat':>8s} {'f+cache':>8s} "
           f"{'succ':>7s} {'infer':>7s} {'dual/call':>10s} {'model/call':>11s} "
           f"{'ng/call':>9s} {'x':>6s}")
    print(hdr)
    print("-" * len(hdr))
    for name in [s for s in args.names.split(",") if s]:
        r = profile_instance(name, args)
        rows.append(r)
        inf = f"{r['inference_ms']:.3f}" if r["inference_ms"] else "--"
        print(f"{r['instance']:7s} {r['cand_size']:4d} {r['mask_ms']:7.3f}ms "
              f"{r['features_ms']:7.3f}ms {r['features_cached_ms']:7.3f}ms "
              f"{r['of_which_succ_ms']:6.3f}ms {inf:>7s} "
              f"{r['call_dual_ms']:9.1f}ms {r['call_model_ms']:10.1f}ms "
              f"{r['call_ngroute_ms']:8.1f}ms {r['call_ratio_model_over_dual']:5.2f}x")

    print()
    print("per-decision-step feature cost, in JSON units (ms):")
    for r in rows:
        feat = r["features_ms"]
        share = 100.0 * r["of_which_succ_ms"] / feat if feat else 0.0
        cached = r["features_cached_ms"]
        gain = 100.0 * (feat - cached) / feat if feat else 0.0
        print(f"  {r['instance']:7s} features {feat:.3f}ms uncached vs "
              f"{cached:.3f}ms cached ({gain:+.0f}%), of which succ_counts "
              f"{r['of_which_succ_ms']:.3f}ms ({share:.0f}% of uncached); "
              f"dual_block {r['dual_block_ms']:.3f}ms, mask {r['mask_ms']:.3f}ms, "
              f"inference {r['inference_ms']:.3f}ms" if r["inference_ms"] else "")

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(
            json.dumps({"args": vars(args), "rows": rows}, indent=2),
            encoding="utf-8")
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
