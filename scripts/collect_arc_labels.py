"""Arc-level label collection for the arc-selection (Morabit-style) pipeline.

What this script is for
-----------------------
The project spent eight rounds attacking the pricing subproblem from the wrong
side ("predict the optimal column").  The measured conclusion was that the
pricing *call* itself is the bottleneck (82.7-99.4% of wall clock) and that the
only lever left is to make each call cheaper.  Morabit, Desaulniers & Lodi
(arXiv:2201.02535) do exactly that: train an arc-level classifier offline, then
shrink the pricing network to the arcs the classifier keeps.

This script produces the training set for that classifier.  It runs full
column generation to convergence on one instance and, at every iteration,
emits one row per *candidate arc*:

    X  (N, 34)    arc features (22 static + 12 dual-dependent)
    y_arc_iter    did a negative column generated *this iteration* use the arc?
    y_arc_run     did a negative column generated *at any iteration* use it?
                  (Morabit's "promising arc" definition; per instance)
    y_opt         graded "is the arc on a near-optimal route for this dual",
                  margin in [0, 1] - the project's older label, kept so the
                  head-to-head the plan asks for is possible

Three decisions worth spelling out
----------------------------------
1. **Rows are per (iteration, arc), not per arc.**  Morabit's features are all
   static, so one row per arc suffices for them.  We add a dual-dependent block
   that changes every iteration: the same arc is a different sample at
   iteration 2 and at iteration 20.  Collapsing to one row per arc would throw
   that away - and the "dual drift" trap (early duals look nothing like late
   ones) is exactly what the plan warns about.

2. **Labels come from the pricing call we already made.**  `max_columns` is
   raised to its ceiling so the pricer returns *every* negative column, not the
   50 the CG loop would pool, and `y_arc_iter` is then free.  Pooling that
   subset would systematically drop late-iteration columns and bias the label
   toward negative.  Note the distinction: we *generate* every negative column
   but still *pool* only `--pool-cap` of them per iteration, so the run stays
   cheap and the iteration count stays comparable with earlier project runs.

3. **`y_opt` costs an extra full pricing call per iteration.**  It comes from
   `labeling.near_optimal_arcs`, which re-solves the pricing problem.  It is on
   by default (the plan wants all three labels co-collected) and can be sampled
   with `--opt-every`, or switched off with `--no-opt-label`.

Outputs (all under `results/`, prefix given by `--out-prefix`):
    <prefix>_<tag>.npz           full arrays, for training
    <prefix>_<tag>.csv           full arrays, human-readable (--no-csv to skip)
    <prefix>_<tag>_iter.json     per-iteration log: lp, #neg cols, new arcs, ...
    <prefix>_<tag>_summary.json  per-instance totals + label positive rates

Usage
-----
    python scripts/collect_arc_labels.py --names R105,RC105,C105 --n 25 --tag n25
    python scripts/collect_arc_labels.py --names R105,RC105 --n 50 --tag n50 \
        --max-iters 40 --stall 25 --opt-every 2
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uavcg.bench import load_instance
from uavcg.features import FeatureContext
from uavcg.heuristics import singleton_routes
from uavcg.instance import DEPOT, EPS, evaluate_route
from uavcg.labeling import near_optimal_arcs
from uavcg.pricers import NgRoutePricer
from uavcg.pricers.base import CLEAN_LABEL_FRACTION
from uavcg.rmp import MasterProblem

ROOT = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------
# Feature block.  Order is fixed and mirrored in the plan's section 3.3 table.
# --------------------------------------------------------------------------
FEATURE_NAMES: List[str] = [
    # ---- static (22) ----
    "cost_ij",           # 1  d_ij / dmean
    "travel_time_ij",    # 2  t_ij / tmean
    "demand_j",          # 3  q_j / Q
    "out_deg_i",         # 4  |out(i) in A_cand| / n
    "in_deg_j",          # 5  |in(j) in A_cand| / n
    "out_min_t_i",       # 6
    "out_max_t_i",       # 7
    "out_avg_t_i",       # 8
    "in_min_t_j",        # 9
    "in_max_t_j",        # 10
    "in_avg_t_j",        # 11
    "tw_lo_i",           # 12 ready_i / H
    "tw_hi_i",           # 13 due_i / H
    "tw_lo_j",           # 14
    "tw_hi_j",           # 15
    "tw_width_j",        # 16 (due_j - ready_j) / H
    "slack_at_j",        # 17 (due_j - max(ready_i+s_i+t_ij, ready_j)) / H
    "wait_at_j",         # 18 max(0, ready_j - (ready_i+s_i+t_ij)) / H
    "lb_back_j",         # 19 shortest path j -> depot, / dmean
    "detour_gain_ij",    # 20 (lb_back_i - d_ij - lb_back_j) / dmean
    "tw_overlap_ij",     # 21 max(0, min(due_i,due_j) - max(ready_i,ready_j)) / H
    "spatial_rank_ij",   # 22 rank of d_ij among i's out-arcs / count
    # ---- dual-dependent (12) ----
    "m_ij",              # 23 (d_ij - pi_j) / mean|d - pi|
    "pi_j",              # 24 pi_j / max(pi)
    "pi_rt_j",           # 25 pi_j / max(round_trip_j, EPS)
    "pi_marg_ij",        # 26 pi_j / max(d_ij + lb_back_j, EPS)
    "rc_rank_ij",        # 27 rank of (d_ij - pi_j) among i's out-arcs / count
    "rc_gap_ij",         # 28 (cbar_ij - min_i cbar) / (max_i cbar - min_i cbar)
    "pi_top_frac_j",     # 29 1 if pi_j in the top 25%
    "rc_neg_frac_i",     # 30 fraction of i's out-arcs with cbar < 0
    "sigma_norm",        # 31 sigma / max(pi)       (constant within an iteration)
    "pi_mean_norm",      # 32 mean(pi) / max(pi)    (constant)
    "iter_frac",         # 33 iteration / max_iterations (constant)
    "neg_col_rate_prev", # 34 previous iteration's #neg cols / |A_cand| (constant)
]
N_FEATURES = len(FEATURE_NAMES)   # 34

#: human-readable columns, in order, before the feature block
RAW_COLUMNS = ["instance", "n", "iter", "i", "j",
               "y_arc_iter", "y_arc_run", "y_opt",
               "cost", "rc_ij", "pi_i", "pi_j"]


# --------------------------------------------------------------------------
def feasible_arc_mask(inst) -> np.ndarray:
    """Necessary-condition feasible arcs, i.e. `A_static`.

    Deliberately the weakest filter - capacity and time window feasibility
    only, both necessary conditions - plus the depot-return rule.  Anything
    stronger would already be a heuristic and would confound what the learned
    model adds on top of it.  Identical to `_probe_arcpool.static_network`, so
    the two produce comparable counts.
    """
    n = inst.n
    cap = inst.params.capacity
    mask = np.zeros((n + 1, n + 1), dtype=bool)
    for i in range(n + 1):
        for j in range(1, n + 1):
            if i == j:
                continue
            if inst.demand[j] > cap + EPS:
                continue
            arr = inst.ready[i] + inst.service[i] + inst.travel_time[i, j]
            if arr > inst.due[j] + EPS:
                continue
            start = max(arr, inst.ready[j])
            if start + inst.service[j] + inst.travel_time[j, DEPOT] > inst.due[DEPOT] + EPS:
                continue
            mask[i, j] = True
    return mask


def candidate_arcs(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """`A_cand`: static-feasible arcs minus every depot-related arc.

    Depot arcs are excluded from the *scored* set because the plan keeps them
    unconditionally at mask-build time - they are the only entry point for
    vehicle-count feasibility, so dropping one can make the master infeasible.
    Scoring them would be wasted work, and a model that learned to drop one
    would be actively harmful.
    """
    ii, jj = np.nonzero(mask)
    keep = (ii != DEPOT) & (jj != DEPOT)
    return ii[keep].astype(np.int64), jj[keep].astype(np.int64)


def per_node_stats(inst, ii: np.ndarray, jj: np.ndarray) -> Dict[str, np.ndarray]:
    """Out/in degree and travel-time min/max/avg per node, over `A_cand`."""
    n = inst.n
    T = inst.travel_time
    t = T[ii, jj]
    out_cnt = np.bincount(ii, minlength=n + 1).astype(float)
    in_cnt = np.bincount(jj, minlength=n + 1).astype(float)
    out_sum = np.bincount(ii, weights=t, minlength=n + 1)
    in_sum = np.bincount(jj, weights=t, minlength=n + 1)
    out_min = np.full(n + 1, np.inf)
    out_max = np.full(n + 1, -np.inf)
    in_min = np.full(n + 1, np.inf)
    in_max = np.full(n + 1, -np.inf)
    np.minimum.at(out_min, ii, t)
    np.maximum.at(out_max, ii, t)
    np.minimum.at(in_min, jj, t)
    np.maximum.at(in_max, jj, t)
    out_avg = np.where(out_cnt > 0, out_sum / np.maximum(out_cnt, 1), 0.0)
    in_avg = np.where(in_cnt > 0, in_sum / np.maximum(in_cnt, 1), 0.0)
    return dict(
        out_cnt=out_cnt, in_cnt=in_cnt,
        out_min=np.where(np.isfinite(out_min), out_min, 0.0),
        out_max=np.where(np.isfinite(out_max), out_max, 0.0),
        out_avg=out_avg,
        in_min=np.where(np.isfinite(in_min), in_min, 0.0),
        in_max=np.where(np.isfinite(in_max), in_max, 0.0),
        in_avg=in_avg,
    )


def group_rank(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """0-based rank of each value inside its group (stable on ties).

    Ties take the order they appear in, which is deterministic and is all a
    rank feature needs.  Groups number at most a few hundred, so the per-group
    loop is not worth vectorising.
    """
    out = np.zeros(values.size, dtype=float)
    for g in np.unique(groups):
        sel = groups == g
        v = values[sel]
        order = np.argsort(v, kind="stable")
        r = np.empty(v.size, dtype=float)
        r[order] = np.arange(v.size, dtype=float)
        out[sel] = r
    return out


def build_arc_features(ctx: FeatureContext, inst, pi: np.ndarray, sigma: float,
                       it: int, max_iters: int, neg_rate_prev: float,
                       ii: np.ndarray, jj: np.ndarray,
                       stats: dict) -> np.ndarray:
    """(len(ii), N_FEATURES) features for every candidate arc, vectorised.

    Careful with `pi`: it has one entry per *customer* (length n), while every
    per-node table in `FeatureContext` is indexed 0..n with node 0 the depot.
    All node-indexed lookups below therefore go through `pi_flat`, which has
    the depot slot prepended, to match `features.FeatureContext.dual_block`.
    """
    n = inst.n
    D = inst.dist
    T = inst.travel_time
    H = ctx.horizon
    dmean = ctx.dmean
    tmean = float(T[1:, 1:].mean())
    if not np.isfinite(tmean) or tmean <= EPS:
        tmean = 1.0

    pi_flat = np.concatenate([[0.0], np.asarray(pi, dtype=float)])
    pi_max = float(max(np.max(pi_flat) if pi_flat.size else 0.0, 1e-9))
    dual = ctx.dual_block(pi)

    d_ij = D[ii, jj]
    t_ij = T[ii, jj]

    # arrival / window quantities along the arc
    arr = inst.ready[ii] + inst.service[ii] + t_ij
    start = np.maximum(arr, inst.ready[jj])
    slack_at_j = (inst.due[jj] - start) / H
    wait_at_j = np.maximum(0.0, inst.ready[jj] - arr) / H
    tw_overlap = np.maximum(
        0.0,
        np.minimum(inst.due[ii], inst.due[jj])
        - np.maximum(inst.ready[ii], inst.ready[jj])) / H

    # reduced-cost structure: cbar_ij = d_ij - pi_j, grouped by leaving node i
    cbar = d_ij - pi_flat[jj]
    m_scale = float(np.mean(np.abs(D[1:, 1:] - pi_flat[None, 1:]))) or 1.0
    cbar_cnt = np.maximum(np.bincount(ii, minlength=n + 1).astype(float), 1.0)
    cbar_rank = group_rank(cbar, ii)
    cbar_min = np.full(n + 1, np.inf)
    cbar_max = np.full(n + 1, -np.inf)
    np.minimum.at(cbar_min, ii, cbar)
    np.maximum.at(cbar_max, ii, cbar)
    rc_neg_frac = np.bincount(
        ii, weights=(cbar < 0).astype(float), minlength=n + 1) / cbar_cnt
    rc_gap = (cbar - cbar_min[ii]) / (cbar_max[ii] - cbar_min[ii] + EPS)

    pi_cust = pi_flat[1:]
    q75 = float(np.quantile(pi_cust, 0.75)) if pi_cust.size else 0.0

    X = np.column_stack([
        # ---- static (22)
        d_ij / dmean,
        t_ij / tmean,
        ctx.demand_norm[jj],
        stats["out_cnt"][ii] / max(n, 1),
        stats["in_cnt"][jj] / max(n, 1),
        stats["out_min"][ii] / tmean,
        stats["out_max"][ii] / tmean,
        stats["out_avg"][ii] / tmean,
        stats["in_min"][jj] / tmean,
        stats["in_max"][jj] / tmean,
        stats["in_avg"][jj] / tmean,
        inst.ready[ii] / H,
        inst.due[ii] / H,
        inst.ready[jj] / H,
        inst.due[jj] / H,
        ctx.tw_width_norm[jj],
        slack_at_j,
        wait_at_j,
        ctx.dist_home_norm[jj],
        ctx.home_gain[ii, jj],
        tw_overlap,
        group_rank(d_ij, ii) / cbar_cnt[ii],
        # ---- dual-dependent (12)
        cbar / m_scale,
        dual.pi_norm[jj],
        dual.pi_rt[jj],
        dual.pi_marg[ii, jj],
        cbar_rank / cbar_cnt[ii],
        rc_gap,
        (pi_flat[jj] >= q75).astype(float),
        rc_neg_frac[ii],
        np.full(ii.size, sigma / pi_max),
        np.full(ii.size, float(pi_cust.mean()) / pi_max),
        np.full(ii.size, it / max(max_iters, 1)),
        np.full(ii.size, neg_rate_prev),
    ])
    assert X.shape[1] == N_FEATURES, (X.shape, N_FEATURES)
    return np.ascontiguousarray(X, dtype=np.float64)


# --------------------------------------------------------------------------
def collect_instance(name: str, n: int, ng_k: int, max_iters: int, stall: int,
                     pool_cap: int, opt_label: bool, opt_every: int,
                     rel_tol: float, verbose: bool, max_seconds: float = 0.0,
                     tw_scale: float = 1.0, label_cap: int = 3_000_000,
                     allow_sliced: bool = False):
    """One instance -> arrays, arcs, per-iteration log, summary."""
    t_inst = time.perf_counter()
    inst = load_instance(name, n=n, tw_scale=tw_scale)
    nn = inst.n
    ctx = FeatureContext(inst)
    mask = feasible_arc_mask(inst)
    ii, jj = candidate_arcs(mask)
    cand_set = set(zip(ii.tolist(), jj.tolist()))
    stats = per_node_stats(inst, ii, jj)
    n_static = int(mask.sum())

    pricer = NgRoutePricer(inst, k=ng_k, max_labels=label_cap)
    master = MasterProblem(inst)
    # v3.4 decision ⑥: the training labels must come off the *same* seed the
    # online solves use, or the dynamic features learned here describe a
    # trajectory the solver will never walk.  `cheapest_insertion` induced a
    # high-dimensional optimal dual face (60/60 coordinates free on R201), which
    # left the labels drawn from a degenerate vertex; the singleton seed pins the
    # dual uniquely.  See `uavcg/heuristics.py::singleton_routes`.
    for r in singleton_routes(inst):
        ev = evaluate_route(inst, r)
        if ev.feasible:
            master.add(r, ev.cost)
    if len(master) == 0:
        raise RuntimeError("seeding produced no feasible column")

    seen_run: set = set()
    pi_log: List[np.ndarray] = []   # duals USED at each iteration (§3.2 dual_log;
    sigma_log: List[float] = []     # offline B1..B12 reconstruction needs these
    Xs, yKs, yOs, raws = [], [], [], []
    log: List[dict] = []
    prev_neg_rate = 0.0
    best_lp = np.inf
    stall_count = 0
    time_capped = False
    #: Spec §3.5 bookkeeping.  A pricing call that hits the label cap sees only
    #: part of the label space, so the columns it reports are a subset of the
    #: real ones -- labelling from it would teach the model that arcs the pricer
    #: never got to see are unpromising.
    degraded = False
    truncated_iters = 0
    peak_labels = 0
    label_cap_seen = None
    it = 0
    lp = master.solve_lp()

    while it < max_iters:
        # Wall-clock budget.  Execution is capped at roughly two minutes per
        # call, so a slow instance must be cut off *by us* (and its finished
        # iterations kept) rather than killed mid-flight by the harness.
        if max_seconds and (time.perf_counter() - t_inst) > max_seconds:
            time_capped = True
            if log:
                log[-1]["time_capped"] = True
            break
        it += 1
        t0 = time.perf_counter()
        res = pricer.price(lp.pi, sigma=lp.sigma,
                           fixed_cost=master.vehicle_fixed_cost,
                           max_columns=10 ** 7)
        pi_log.append(np.asarray(lp.pi, dtype=float).copy())
        sigma_log.append(float(lp.sigma))
        price_s = time.perf_counter() - t0
        if res.labels_created > peak_labels:
            peak_labels = res.labels_created
        if res.label_cap is not None:
            label_cap_seen = (res.label_cap if label_cap_seen is None
                              else min(label_cap_seen, res.label_cap))
        if res.truncated and not allow_sliced:
            # Stop this instance rather than keep labelling from a truncated
            # search: everything after this point is measured against a pricer
            # that cannot see the whole label space (spec §3.5.2).
            degraded = True
            print(f"  [{inst.name}] DEGRADED at iteration {it}: label cap hit "
                  f"({res.labels_created}/{res.label_cap}) -> instance dropped "
                  f"from the training set (spec §3.5)", flush=True)
            break
        if res.truncated:
            # --allow-sliced (n=60 fallback, doc §26.3): a truncated call's
            # RETURNED columns are true observations, so positive labels stay
            # valid (the arc was seen and used); only run-level zeros carry
            # false-negative risk, which the C201 dual-trajectory experiment
            # quantifies.  Iteration rows from truncated calls are flagged and
            # must not train per-iteration (y_arc_iter) models.
            truncated_iters += 1
            print(f"  [{inst.name}] sliced at iteration {it} "
                  f"({res.labels_created}/{res.label_cap}); positives kept, "
                  f"iteration flagged", flush=True)
        neg = [(r, rc) for r, rc in res.columns if rc < -EPS]

        neg_arcs: set = set()
        for r, _rc in neg:
            prev = DEPOT
            for j in r:
                neg_arcs.add((prev, j))
                prev = j
            neg_arcs.add((prev, DEPOT))
        # Count only candidate arcs.  `seen_run` is the numerator of the
        # "promising fraction" whose denominator is |A_cand|; mixing the depot
        # arcs in (they are always kept at mask-build time, so they are never
        # scored) is how that fraction ends up above 1.
        neg_cand = {a for a in neg_arcs if a in cand_set}
        new_arcs = len(neg_cand - seen_run)

        yK = np.fromiter(((int(a), int(b)) in neg_arcs for a, b in zip(ii, jj)),
                         dtype=float, count=ii.size)
        if opt_label and neg and (it - 1) % max(1, opt_every) == 0:
            margins, _best, _tied = near_optimal_arcs(
                inst, lp.pi, lp.sigma, master.vehicle_fixed_cost, pricer, rel_tol)
            yO = np.fromiter((margins.get((int(a), int(b)), 0.0)
                              for a, b in zip(ii, jj)), dtype=float, count=ii.size)
        else:
            # NaN, not 0.  "not sampled this iteration" has to stay
            # distinguishable from "sampled, and the arc is not on any
            # near-optimal route" - otherwise `--opt-every 2` would inject
            # silent false negatives into the label.
            yO = np.full(ii.size, np.nan)

        pi_flat = np.concatenate([[0.0], np.asarray(lp.pi, dtype=float)])
        Xs.append(build_arc_features(ctx, inst, lp.pi, lp.sigma, it, max_iters,
                                     prev_neg_rate, ii, jj, stats))
        yKs.append(yK)
        yOs.append(yO)
        raws.append(np.column_stack([
            inst.dist[ii, jj],             # cost
            inst.dist[ii, jj] - pi_flat[jj],  # rc_ij = c_ij - pi_j
            pi_flat[ii],                   # pi_i
            pi_flat[jj],                   # pi_j
        ]))
        seen_run |= neg_cand

        log.append({
            "instance": name, "iteration": it, "lp": float(lp.objective),
            "neg_columns": len(neg), "price_seconds": round(price_s, 4),
            "relaxed_min_rc": (None if res.relaxed_min_reduced_cost is None
                               else float(res.relaxed_min_reduced_cost)),
            "neg_arcs_all": len(neg_arcs), "neg_cand_arcs": len(neg_cand),
            "new_arcs": new_arcs,
            "cum_arcs": len(seen_run),
            "cum_frac_of_cand": len(seen_run) / max(1, ii.size),
            "pool_cols": len(master),
            "truncated": bool(res.truncated),
        })
        if verbose:
            print(f"    it {it:3d}  lp={lp.objective:10.3f}  neg={len(neg):6d}  "
                  f"new={new_arcs:5d}  cum={len(seen_run):5d}/{ii.size}  "
                  f"{price_s:.3f}s", flush=True)

        prev_neg_rate = len(neg) / max(1, ii.size)

        if not neg:
            break
        to_pool = neg if pool_cap <= 0 else neg[:pool_cap]
        added = master.add_many(to_pool)
        if added == 0 and len(to_pool) < len(neg):
            # The capped prefix was nothing but duplicate columns.  Fall back to
            # the whole batch before concluding the pricer had nothing new:
            # reading "nothing was added" as "the pricer found nothing" is a
            # mistake this project has already made once.
            added = master.add_many(neg)
        if added == 0:
            log[-1]["no_new_columns"] = True
            break
        lp_new = master.solve_lp()
        if lp_new.objective < best_lp - 1e-6:
            best_lp = lp_new.objective
            stall_count = 0
        else:
            stall_count += 1
        lp = lp_new
        if stall_count >= stall:
            log[-1]["stalled"] = True
            break

    X_all = np.vstack(Xs)
    yK_all = np.concatenate(yKs)
    yO_all = np.concatenate(yOs)
    raw_all = np.vstack(raws)
    arcs_all = np.column_stack(
        [np.tile(ii, len(Xs)), np.tile(jj, len(Xs))]).astype(np.int64)
    it_all = np.repeat(np.arange(1, len(Xs) + 1, dtype=np.int64), ii.size)
    # Run-level (Morabit) label: back-filled once the run is over, because
    # "used at least once at any iteration" is not knowable before then.
    yRun_all = np.fromiter(
        (((int(a), int(b)) in seen_run) for a, b in arcs_all),
        dtype=float, count=arcs_all.shape[0])

    summary = {
        "instance": name, "n": nn, "ng_k": ng_k, "tw_scale": tw_scale,
        "label_cap_configured": int(label_cap),
        "sliced_iters": int(truncated_iters),
        "arcs_all_directed": int(nn * (nn + 1)),
        "arcs_static_feasible": n_static,
        "arcs_candidate": int(ii.size),
        "arcs_promising_run": len(seen_run),
        "promising_frac_of_cand": len(seen_run) / max(1, ii.size),
        "iterations": len(Xs),
        "rows": int(X_all.shape[0]),
        "positive_rate_iter": float(yK_all.mean()),
        "positive_rate_run": float(yRun_all.mean()),
        "positive_rate_opt": (float((yO_all[np.isfinite(yO_all)] > 0).mean())
                              if np.isfinite(yO_all).any() else 0.0),
        "mean_opt_margin": (float(np.nanmean(yO_all))
                            if np.isfinite(yO_all).any() else 0.0),
        "opt_label_coverage": float(np.isfinite(yO_all).mean()),
        "final_lp": float(lp.objective),
        "time_capped": bool(time_capped),
        #: Spec §3.5.2: a run is clean only while the peak label count stays
        #: below `CLEAN_LABEL_FRACTION` of the cap.  `clean_run` is the single
        #: flag the experiment layer is allowed to filter on.
        "degraded": bool(degraded),
        "peak_labels_created": int(peak_labels),
        "label_cap": (int(label_cap_seen) if label_cap_seen else None),
        "clean_run": bool(not degraded and (label_cap_seen is None or
                                            peak_labels < CLEAN_LABEL_FRACTION * label_cap_seen)),
    }
    summary["warmup_duals"] = True  # pi_log/sigma_log present in the npz
    return (X_all, yK_all, yRun_all, yO_all, raw_all, arcs_all, it_all, log,
            summary, np.stack(pi_log), np.asarray(sigma_log))


# --------------------------------------------------------------------------
def _stack_arrays(prev, allX, allK, allR, allO, allRaw, allA, allI,
                  allPI=(), allSG=()):
    """Concatenate this session's chunks, then any pre-existing arrays."""
    if allX:
        X = np.vstack(allX)
        yK = np.concatenate(allK)
        yR = np.concatenate(allR)
        yO = np.concatenate(allO)
        raw = np.vstack(allRaw)
        arcs = np.vstack(allA)
        iters = np.concatenate(allI)
    else:
        X = np.zeros((0, N_FEATURES))
        yK = yR = yO = np.zeros(0)
        raw = np.zeros((0, 4))
        arcs = np.zeros((0, 2), dtype=np.int64)
        iters = np.zeros(0, dtype=np.int64)
    if prev is not None:
        X = np.vstack([prev["X"], X])
        yK = np.concatenate([prev["y_arc_iter"], yK])
        yR = np.concatenate([prev["y_arc_run"], yR])
        yO = np.concatenate([prev["y_opt"], yO])
        raw = np.vstack([prev["raw"], raw])
        arcs = np.vstack([prev["arcs"], arcs])
        iters = np.concatenate([prev["iters"], iters])
    def _as_object_array(list_):
        # np.array(list, dtype=object) COLLAPSES same-shaped entries into a
        # higher-dim array; assign elementwise instead.
        arr = np.empty(len(list_), dtype=object)
        arr[:] = list_
        return arr
    pi_logs = _as_object_array(allPI)
    sigma_logs = _as_object_array(allSG)
    if prev is not None and "pi_logs" in prev.files:
        pi_logs = np.concatenate([prev["pi_logs"], pi_logs])
        sigma_logs = np.concatenate([prev["sigma_logs"], sigma_logs])
    return X, yK, yR, yO, raw, arcs, iters, pi_logs, sigma_logs


def _dump(npz_path, iter_path, summ_path, X, yK, yR, yO, raw, arcs, iters,
          summaries, log, pi_logs, sigma_logs) -> None:
    """Write the dataset, the per-iteration log and the summary in one go.

    The archive is written straight to its final path on purpose.  An earlier
    version wrote a temporary file and moved it into place atomically, which is
    the textbook fix - but on this machine `os.replace` onto the existing file
    raises `PermissionError: [WinError 5]`, and because that happened *inside*
    the per-instance flush it aborted the whole run after the first instance.
    The log and summary files are small text and are rewritten unconditionally.
    """
    np.savez_compressed(
        npz_path, X=X, y_arc_iter=yK, y_arc_run=yR, y_opt=yO,
        arcs=arcs, iters=iters, raw=raw,
        feature_names=np.array(FEATURE_NAMES, dtype=object),
        instances=np.array([s["instance"] for s in summaries], dtype=object),
        pi_logs=pi_logs, sigma_logs=sigma_logs,
    )
    iter_path.write_text(json.dumps(log, indent=1), encoding="utf-8")
    summ_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--names", default="R105,RC105,C105,R107,RC107",
                    help="comma-separated Solomon instance names")
    ap.add_argument("--split", action="store_true",
                    help="resolve names through the frozen split: n and "
                         "tw_scale come from the split entry, not --n")
    ap.add_argument("--n", type=int, default=25, help="truncate to n customers")
    ap.add_argument("--ng-k", type=int, default=10)
    ap.add_argument("--allow-sliced", action="store_true",
                    help="do not drop the instance on a truncated call; "
                         "positives stay valid, run-level zeros carry "
                         "quantified false-negative risk (doc §26.3)")
    ap.add_argument("--label-cap", type=int, default=3_000_000,
                    help="per-call label cap; spec §3.5 requires UNTRUNCATED "
                         "calls, so at n=60 this must clear the true first-call "
                         "demand (3M default only suffices for C201)")
    ap.add_argument("--max-iters", type=int, default=60)
    ap.add_argument("--stall", type=int, default=40)
    ap.add_argument("--pool-cap", type=int, default=50,
                    help="columns pooled per iteration (<=0 pools every negative column)")
    ap.add_argument("--opt-every", type=int, default=1,
                    help="sample the (expensive) y_opt label every k iterations")
    ap.add_argument("--no-opt-label", action="store_true",
                    help="skip y_opt entirely (saves one full pricing call per iteration)")
    ap.add_argument("--rel-tol", type=float, default=0.02)
    ap.add_argument("--tag", default="n25")
    ap.add_argument("--out-prefix", default="results/arc_labels")
    ap.add_argument("--no-csv", action="store_true")
    ap.add_argument("--max-seconds", type=float, default=90.0,
                    help="wall-clock budget per instance; on expiry the run "
                         "stops cleanly and keeps the iterations it finished "
                         "(0 disables the cap)")
    ap.add_argument("--total-seconds", type=float, default=100.0,
                    help="wall-clock budget for the whole call; collection "
                         "stops before the next instance once it is exhausted, "
                         "so a long sweep can be split across short calls")
    ap.add_argument("--append", action="store_true",
                    help="merge into an existing <prefix>_<tag>.npz, skipping any "
                         "instance already collected there (idempotent re-runs; "
                         "lets a long sweep be split across short batches)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    names = [s.strip() for s in args.names.split(",") if s.strip()]
    verbose = not args.quiet
    prefix = ROOT / args.out_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    npz_path = Path(str(prefix) + f"_{args.tag}.npz")
    iter_path = Path(str(prefix) + f"_{args.tag}_iter.json")
    summ_path = Path(str(prefix) + f"_{args.tag}_summary.json")
    csv_path = Path(str(prefix) + f"_{args.tag}.csv")

    prev = None
    prev_summaries: List[dict] = []
    prev_log: List[dict] = []
    if args.append and npz_path.exists():
        prev = np.load(npz_path, allow_pickle=True)
        prev_summaries = (json.loads(summ_path.read_text(encoding="utf-8"))
                          if summ_path.exists() else [])
        prev_log = (json.loads(iter_path.read_text(encoding="utf-8"))
                    if iter_path.exists() else [])
        done = {str(s) for s in prev["instances"]}
        before = len(names)
        names = [n for n in names if n not in done]
        print(f"[append] {len(done)} instance(s) already in {npz_path.name}; "
              f"{len(names)} of {before} left to collect", flush=True)

    allX, allK, allR, allO, allRaw, allA, allI = [], [], [], [], [], [], []
    allPI: List[np.ndarray] = []   # per-instance (iters_i, n) dual trajectory
    allSG: List[np.ndarray] = []   # per-instance (iters_i,) sigma trajectory
    all_log, summaries = [], []
    t_start = time.perf_counter()

    tw_map = {}
    if args.split:
        from uavcg.splits import load_split
        sp = load_split()
        tw_map = {e.name: (e.n, e.tw_scale) for e in sp.entries}
        missing = [n_ for n_ in names if n_ not in tw_map]
        if missing:
            raise SystemExit(f"not in frozen split {sp.version}: {missing}")
        print(f"[split] {sp.version}: " +
              ", ".join(f"{n_}@x{tw_map[n_][1]:g}" for n_ in names),
              flush=True)

    for name in names:
        spent = time.perf_counter() - t_start
        remaining = args.total_seconds - spent if args.total_seconds else 1e9
        if remaining <= 5.0:
            left = names[names.index(name):]
            print(f"[budget] {spent:.0f}s spent, {remaining:.0f}s left; "
                  f"stopping. Re-run with the same --append to collect: "
                  f"{','.join(left)}", flush=True)
            break
        per_inst = remaining if not args.max_seconds else min(args.max_seconds,
                                                             remaining)
        n_eff, tw_eff = tw_map.get(name, (args.n, 1.0))
        print(f"=== {name} n={n_eff} tw={tw_eff:g} ===", flush=True)
        try:
            (X, yK, yR, yO, raw, arcs, iters, log, summ,
             pi_l, sg_l) = collect_instance(
                name, n_eff, args.ng_k, args.max_iters, args.stall,
                args.pool_cap, not args.no_opt_label, args.opt_every,
                args.rel_tol, verbose, per_inst, tw_eff, args.label_cap,
                args.allow_sliced)
        except Exception as exc:      # keep the sweep going
            print(f"  FAILED {type(exc).__name__}: {exc}", flush=True)
            continue
        allX.append(X); allK.append(yK); allR.append(yR); allO.append(yO)
        allPI.append(pi_l); allSG.append(sg_l)
        allRaw.append(raw); allA.append(arcs); allI.append(iters)
        all_log.extend(log); summaries.append(summ)
        print(f"  -> rows={summ['rows']:7d}  iters={summ['iterations']:3d}  "
              f"promising={summ['arcs_promising_run']:5d}"
              f"/{summ['arcs_candidate']} ({summ['promising_frac_of_cand']:.2%})  "
              f"pos_iter={summ['positive_rate_iter']:.4f} "
              f"pos_run={summ['positive_rate_run']:.4f} "
              f"pos_opt={summ['positive_rate_opt']:.4f}", flush=True)
        # Flush after every instance.  Execution is capped at about two minutes
        # per call, and a batch interrupted at that cap used to lose everything
        # it had already finished; now it keeps all but the in-flight instance.
        (_X, _yK, _yR, _yO, _raw, _arcs, _iters,
         _piL, _sgL) = _stack_arrays(prev, allX, allK, allR, allO, allRaw,
                                     allA, allI, allPI, allSG)
        _dump(npz_path, iter_path, summ_path, _X, _yK, _yR, _yO, _raw,
              _arcs, _iters, prev_summaries + summaries, prev_log + all_log,
              _piL, _sgL)

    if not allX and prev is None:
        print("nothing collected", flush=True)
        return 1

    (X, yK, yR, yO, raw, arcs, iters,
     pi_logs, sigma_logs) = _stack_arrays(prev, allX, allK, allR, allO,
                                          allRaw, allA, allI, allPI, allSG)
    summaries = prev_summaries + summaries
    all_log = prev_log + all_log
    _dump(npz_path, iter_path, summ_path, X, yK, yR, yO, raw, arcs, iters,
          summaries, all_log, pi_logs, sigma_logs)
    print(f"\nwrote {npz_path}  ({X.shape[0]} rows x {X.shape[1]} features)",
          flush=True)
    print(f"wrote {iter_path}  ({len(all_log)} iteration records)", flush=True)
    print(f"wrote {summ_path}", flush=True)

    if not args.no_csv and X.shape[0]:
        inst_of_row = np.concatenate(
            [np.full(s["rows"], s["instance"], dtype=object) for s in summaries])
        num = np.column_stack([
            np.full(X.shape[0], args.n, dtype=float),
            iters.astype(float), arcs[:, 0].astype(float), arcs[:, 1].astype(float),
            yK, yR, yO, raw, X,
        ])
        num = np.round(num, 6)
        num_rows = num.tolist()
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(RAW_COLUMNS + FEATURE_NAMES)
            w.writerows(
                [inst_of_row[k], *num_rows[k]] for k in range(len(num_rows)))
        print(f"wrote {csv_path}  ({len(RAW_COLUMNS) + N_FEATURES} cols)", flush=True)

    fin = np.isfinite(yO)
    print(f"\nTOTAL rows={X.shape[0]}  features={X.shape[1]}  "
          f"instances={len(summaries)}  wall={time.perf_counter() - t_start:.1f}s")
    print(f"positive rate: iter={yK.mean():.4f}  run={yR.mean():.4f}  "
          f"opt>0={(float((yO[fin] > 0).mean()) if fin.any() else 0.0):.4f} "
          f"(opt coverage {float(fin.mean()):.2%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
