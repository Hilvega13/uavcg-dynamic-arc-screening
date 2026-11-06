"""Arc-level features for the learned pricing heuristic.

Why arcs and not customers
--------------------------
The reference paper (Shen et al., AAAI 2022) predicts, for every *vertex* of
the graph, whether it belongs to an optimal maximum-weight independent set.
That works because a column of their master problem is an *unordered* set of
vertices: "the set of chosen vertices" determines the column completely.

Our column is an ordered route, so the same target is not enough.  Knowing
"customer 7 is in the optimal route" does not tell us where in the route it
sits, and two routes over the same customers in different orders are two
different columns with different costs.  A predictor over customers can
therefore not be turned into a route at all.

The target is lifted from the vertex to the arc: does arc (i, j) belong to an
optimal route?  A route is *defined* as its arc set, so this is the natural
generalisation, it strictly dominates the vertex formulation in information
content, and it carries the sequencing decision for free - no artificial
"visit the nearest customer first" tie-breaking rule is needed.

Everything below is computed for one decision step of a rollout: given the
partial route, produce one feature vector per candidate next customer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from .instance import DEPOT, EPS, Instance

#: Feature order is fixed; the model is trained on whatever comes out of
#: `arc_features`, so `FEATURE_NAMES` and the column order in that function
#: must be kept in sync.  `scripts/selftest.py` checks the two lengths agree.
FEATURE_NAMES: List[str] = [
    "m_ij",             # (dist_ij - pi_j) / dmean      modified arc cost
    "dist_ij",          # dist_ij / dmean
    "pi",               # pi_j / max pi                 dual weight
    "pi_roundtrip",     # pi_j / (d_0j + d_j0)          dual reward per detour
    "pi_marginal",      # pi_j / (dist_ij + lb_back_j)  dual reward per extra km
    "demand",           # q_j / Q                       load pressure
    "tw_width",         # (due_j - ready_j) / horizon
    "tw_position",      # where the arrival sits inside the window
    "tw_slack",         # (due_j - arrival) / horizon
    "cap_residual",     # (Q - load - q_j) / Q
    "dist_home",        # lb_back_j / dmean             distance still to fly home
    "home_gain",        # (lb_back_i - dist_ij - lb_back_j) / dmean
    "succ_count",       # feasible successors of j / n  neighbourhood degree
    "rank_freq",        # arc usage in warm-up rollouts rank-style feature
    "progress",         # step / n
    "load_state",       # load / Q
    "energy_pressure",  # energy lower bound after this arc / E
]

NUM_FEATURES = len(FEATURE_NAMES)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def all_pairs_shortest(dist: np.ndarray) -> np.ndarray:
    """Floyd-Warshall.  Used to bound the cost of getting back to the depot.

    Necessary conditions in the feasibility mask need a *lower* bound on the
    remaining travel, and with asymmetric or non-triangular costs the direct
    arc value is not one.  The shortest path in the graph always is.
    """
    N = dist.shape[0]
    D = np.array(dist, dtype=float, copy=True)
    np.fill_diagonal(D, 0.0)
    for k in range(N):
        np.minimum(D, D[:, k][:, None] + D[k, :][None, :], out=D)
    return D


@dataclass(frozen=True)
class DualFeat:
    """Everything in the feature block that depends on the dual solution.

    Built once per *pricing call* rather than once per decision step.  A call
    makes a few hundred decision steps and the duals are fixed for all of them,
    so the same four tables were previously rebuilt several hundred times.
    Each field is a full `(n+1, n+1)` or `(n+1,)` table so that a decision step
    only has to gather from it.
    """

    m: np.ndarray         # (n+1, n+1)  (d_ij - pi_j) / dmean
    pi_norm: np.ndarray   # (n+1,)      pi_j / max(pi)
    pi_rt: np.ndarray     # (n+1,)      pi_j / max(round_trip_j, EPS)
    pi_marg: np.ndarray   # (n+1, n+1)  pi_j / max(d_ij + lb_back_j, EPS)


@dataclass
class RolloutState:
    """The partial route at one decision step (everything the mask needs)."""

    node: int
    time: float
    load: float
    dist_so_far: float
    pi_sum: float
    energy_lb: float
    step: int
    path: tuple
    visited: np.ndarray          # (n+1,) bool, index 0 unused


class FeatureContext:
    """Per-instance constants, computed once and reused by every rollout."""

    def __init__(self, inst: Instance) -> None:
        self.inst = inst
        n = inst.n
        self.n = n
        self.cap = float(inst.params.capacity) if inst.params.capacity else 1.0
        self.base_rate = float(inst.params.base_energy_rate)
        self.load_factor = float(inst.params.load_energy_factor)
        self.energy_cap = inst.params.energy_capacity

        d = inst.dist
        if n:
            self.dmean = float(d[1:, 1:].mean()) if n > 1 else float(d[1:, 1:].mean())
        else:
            self.dmean = 1.0
        if not np.isfinite(self.dmean) or self.dmean <= EPS:
            self.dmean = 1.0
        self.horizon = float(inst.due[DEPOT])
        if not np.isfinite(self.horizon) or self.horizon <= EPS:
            self.horizon = 1.0

        # customer-indexed (length n) copies of the per-node data
        self.demand_c = inst.demand[1:].astype(float)
        self.ready_c = inst.ready[1:].astype(float)
        self.due_c = inst.due[1:].astype(float)
        self.service_c = inst.service[1:].astype(float)
        self.tw_width_c = np.maximum(self.due_c - self.ready_c, EPS)
        self.round_trip_c = d[0, 1:] + d[1:, 0]

        self.sd = all_pairs_shortest(d)
        self.lb_dist_back_c = self.sd[1:, DEPOT]
        self.lb_time_back_c = all_pairs_shortest(inst.travel_time)[1:, DEPOT]
        self.depot_due = float(inst.due[DEPOT]) + EPS
        self.customers = np.arange(1, n + 1)

        # ---- tables that do not depend on the dual solution or the state ---
        #
        # These are the columns of `arc_features` that are fixed once the
        # instance is fixed.  Precomputing them turns per-step arithmetic into
        # per-step gathers.  Node index 0 is the depot and its entry is a
        # placeholder (`lb_vec[0] = 0`, matching the `i != DEPOT` branch the
        # original expression used) because no column is ever wanted for it.
        lb_vec = np.concatenate([[0.0], self.lb_dist_back_c])
        self.lb_vec = lb_vec
        self.dist_norm = d / self.dmean                          # d_ij / dmean
        self.home_gain = (lb_vec[:, None] - d - lb_vec[None, :]) / self.dmean
        self.demand_norm = np.concatenate([[0.0], self.demand_c / self.cap])
        self.tw_width_norm = np.concatenate(
            [[0.0], self.tw_width_c / self.horizon])
        self.dist_home_norm = lb_vec / self.dmean
        #: Denominator of `pi_marginal`; the numerator is the dual, so this is
        #: the only part that can be precomputed before the duals are known.
        self.marg_den = np.maximum(d + lb_vec[None, :], EPS)
        #: Round-trip distance, as a divisor.  Stored un-normalised so the
        #: per-call division `pi_j / round_trip_j` keeps the original order.
        self.round_trip_den = np.concatenate(
            [[1.0], np.maximum(self.round_trip_c, EPS)])

    # ------------------------------------------------------------------
    def dual_block(self, pi: np.ndarray) -> DualFeat:
        """Precompute the dual-dependent feature tables for one pricing call.

        `(n+1)^2` doubles, built three times per call - trivial next to the
        several hundred decision steps that then read it.
        """
        flat = np.concatenate([[0.0], np.asarray(pi, dtype=float)])
        pi_max = float(max(np.max(pi) if np.size(pi) else 0.0, 1e-9))
        return DualFeat(
            m=(self.inst.dist - flat[None, :]) / self.dmean,
            pi_norm=flat / pi_max,
            pi_rt=flat / self.round_trip_den,
            pi_marg=flat[None, :] / self.marg_den,
        )

    # ------------------------------------------------------------------
    def arc_energy_lb(self, carry_lb: np.ndarray | float) -> np.ndarray | float:
        """Energy rate on an arc when the carried load is at least `carry_lb`.

        energy(arc, carried) = base_rate * dist * (1 + lf * carried / Q) is
        increasing in `carried`, so a lower bound on the load gives a lower
        bound on the energy.  During a forward rollout the load carried on the
        next arc is at least the demand of the arc's own head node, because
        that demand has to be on board to be delivered.
        """
        r = 1.0 + self.load_factor * np.asarray(carry_lb, dtype=float) / self.cap
        return self.base_rate * r

    # ------------------------------------------------------------------
    def mask(self, st: RolloutState) -> np.ndarray:
        """Customers that could still be visited without breaking feasibility.

        Every test below is a *necessary* condition for a feasible completion,
        so the mask can never remove a candidate that some feasible route uses.
        It is a weak prune, deliberately: pruning is only useful if it is also
        safe, and an unsafe prune would silently destroy the lower bound.
        """
        inst = self.inst
        i = st.node
        if self.n == 0:
            return np.empty(0, dtype=int)

        arr = st.time + inst.service[i] + inst.travel_time[i, 1:]
        start = np.maximum(arr, self.ready_c)
        ok = arr <= self.due_c + EPS                                  # time window
        ok &= (start + self.service_c + self.lb_time_back_c) <= self.depot_due
        ok &= (st.load + self.demand_c) <= self.cap + EPS              # capacity
        ok &= ~st.visited[1:]                                          # elementary

        if inst.params.energy_enabled:
            rate = self.arc_energy_lb(self.demand_c)
            need = st.energy_lb + rate * inst.dist[i, 1:]
            need = need + self.base_rate * self.lb_dist_back_c
            ok &= need <= float(self.energy_cap) + EPS

        return np.flatnonzero(ok) + 1

    # ------------------------------------------------------------------
    def succ_counts(self, st: RolloutState, cand: np.ndarray) -> np.ndarray:
        """Feasible successors of each candidate - a cheap "vertex degree".

        This is the arc-level counterpart of the reference paper's vertex
        degree feature: how much freedom a customer leaves after it is served.
        """
        inst = self.inst
        cand = np.asarray(cand, dtype=int)
        if cand.size == 0:
            return np.zeros(0)
        row = inst.travel_time[st.node]
        arr = st.time + inst.service[st.node] + row[cand]
        start = np.maximum(arr, self.ready_c[cand - 1])
        load = st.load + self.demand_c[cand - 1]

        # `np.take(..., axis=0)` then slice beats `np.ix_` fancy indexing: the
        # latter builds two broadcast index arrays and does a full outer
        # gather, while the former copies contiguous rows.  Same values, and
        # `[:, 1:]` is exactly `customers`.
        table = np.take(inst.travel_time, cand, axis=0)[:, 1:]
        # `start + service` is needed by both tests below; computing it once
        # removes an (m, n) addition from the hot path.
        start_svc = start[:, None] + self.service_c[None, :]
        ok = start_svc + table <= self.due_c[None, :] + EPS
        ok &= start_svc + self.lb_time_back_c[None, :] <= self.depot_due
        ok &= (load[:, None] + self.demand_c[None, :]) <= self.cap + EPS
        ok &= ~st.visited[1:][None, :]
        ok[np.arange(cand.size), cand - 1] = False
        return ok.sum(axis=1).astype(float)


# --------------------------------------------------------------------------
def arc_features(
    ctx: FeatureContext,
    st: RolloutState,
    cand: np.ndarray,
    pi: np.ndarray,
    rank_freq: Optional[np.ndarray] = None,
    dual: Optional[DualFeat] = None,
) -> np.ndarray:
    """(len(cand), NUM_FEATURES) feature block for one decision step.

    `dual` is the per-call cache from `FeatureContext.dual_block`.  Passing it
    is optional: when it is omitted the block is built here, which is what the
    labelling and profiling scripts do.  Both paths are asserted to produce
    bit-identical output (`scripts/selftest.py`), because a feature that
    differs in the last ulp changes the sampler and would silently decouple
    this round's numbers from every earlier table.
    """
    inst = ctx.inst
    i = st.node
    cand = np.asarray(cand, dtype=int)
    m = cand.size
    if m == 0:
        return np.zeros((0, NUM_FEATURES))
    if dual is None:
        dual = ctx.dual_block(pi)

    k = cand - 1
    d_ij = inst.dist[i, cand]

    arr = st.time + inst.service[i] + inst.travel_time[i, cand]
    start = np.maximum(arr, ctx.ready_c[k])

    tw_position = np.clip((start - ctx.ready_c[k]) / ctx.tw_width_c[k], 0.0, 1.0)
    tw_slack = (ctx.due_c[k] - arr) / ctx.horizon
    cap_residual = (ctx.cap - st.load - ctx.demand_c[k]) / ctx.cap

    if inst.params.energy_enabled:
        rate = ctx.arc_energy_lb(ctx.demand_c[k])
        e_after = st.energy_lb + rate * d_ij + ctx.base_rate * ctx.lb_vec[cand]
        energy_pressure = e_after / float(inst.params.energy_capacity)
    else:
        energy_pressure = np.zeros(m)

    if rank_freq is not None:
        rank = np.asarray(rank_freq, dtype=float)[i, cand]
    else:
        rank = np.zeros(m)

    X = np.column_stack([
        dual.m[i, cand],                    # (d_ij - pi_j) / dmean
        ctx.dist_norm[i, cand],             # d_ij / dmean
        dual.pi_norm[cand],                 # pi_j / max(pi)
        dual.pi_rt[cand],                   # pi_j / round_trip_j
        dual.pi_marg[i, cand],              # pi_j / (d_ij + lb_back_j)
        ctx.demand_norm[cand],              # q_j / Q
        ctx.tw_width_norm[cand],            # (due_j - ready_j) / horizon
        tw_position,
        tw_slack,
        cap_residual,
        ctx.dist_home_norm[cand],           # lb_back_j / dmean
        ctx.home_gain[i, cand],             # (lb_back_i - d_ij - lb_back_j)/dmean
        ctx.succ_counts(st, cand) / max(ctx.n, 1),
        rank,
        np.full(m, st.step / max(ctx.n, 1)),
        np.full(m, st.load / ctx.cap),
        energy_pressure,
    ])
    return np.ascontiguousarray(X, dtype=np.float64)
