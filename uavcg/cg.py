"""The column generation loop.

    solve master LP  ->  duals (pi, sigma)  ->  price  ->  add negative columns
    ... repeated until no column with negative reduced cost can be found.

The loop is deliberately agnostic about how the pricing problem is solved, so
the exact labelling algorithm, the greedy baseline and the learned pricer all
drop into the same slot. That is what makes the comparison in the experimental
chapter a like-for-like one: only the pricer changes.

Optimality bookkeeping
----------------------
A heuristic pricer cannot prove that no negative column exists. When such a
pricer returns nothing, the loop therefore calls the exact pricer once to
confirm. This mirrors the standard "heuristic pricing + exact pricing at the
end" pattern and is what keeps the reported LP bound valid.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from .heuristics import cheapest_insertion
from .instance import Instance, evaluate_route
from .pricers.base import (
    CLEAN_LABEL_FRACTION,
    PricingResult,
    Pricer,
    modified_cost,
)
from .rmp import MasterProblem


def reduced_cost_of(
    inst: Instance,
    nodes: Sequence[int],
    pi: np.ndarray,
    sigma: float,
    fixed_cost: float = 0.0,
) -> float:
    """Recompute a route's reduced cost from first principles.

        rc(r) = cost(r) + V - sum_{i in r} pi_i - sigma

    This is the definition every pricer works against (see `pricers/base.py`)
    and the only place the loop trusts it.  Everything a pricer reports about
    its own columns is a self-assessment, and under stabilized duals it is an
    assessment against the *wrong* objective; this function is the independent
    check that keeps the pool honest.  It costs one `evaluate_route` per route,
    against a batch the pricer has already spent far more on.
    """
    key = tuple(int(v) for v in nodes)
    if not key:
        return 0.0 + float(fixed_cost) - float(sigma)
    cost = evaluate_route(inst, key).cost
    cover = float(np.sum(np.asarray(pi, dtype=float)[np.asarray(key) - 1]))
    return float(cost) + float(fixed_cost) - cover - float(sigma)


@dataclass
class WarmupTrace:
    """Everything the first `W` iterations observed -- and nothing after them.

    This object is the whole mechanism of the warm-up protocol (spec §1).  The
    dynamic arc features depend on the solve trajectory, while the arc decision
    has to be taken before the solve finishes; handing the hook a log that stops
    at iteration ``W`` turns "do not use future information" from a promise into
    a property of the data structure.  There is no route from this object to
    iteration ``W+1``.

    All four logs have exactly `W` entries, so the ``t`` index in the feature
    definitions of §2.2 is ``0..W-1`` here.  `reduced_cost[t]` is the full arc
    matrix ``c~_ij^(t) = c_ij - pi_j^(t)`` built with the same
    `pricers.base.modified_cost` the pricer uses -- the features and the solver
    must not be able to disagree about the convention.
    """

    #: RMP duals of each warm-up iteration, the master's own and not a
    #: stabilized surrogate.  Combined with the assertion that masking and
    #: stabilization are never enabled together, that means there is exactly one
    #: dual in play per iteration.
    pi: List[np.ndarray] = field(default_factory=list)
    sigma: List[float] = field(default_factory=list)
    reduced_cost: List[np.ndarray] = field(default_factory=list)
    #: Columns generated in each warm-up iteration, as node tuples.  B12 needs
    #: "how often was this arc used in the warm-up", which is a fold over these.
    columns: List[List[Tuple[int, ...]]] = field(default_factory=list)
    #: The LP objective of each warm-up iteration, for diagnosing a warm-up that
    #: was still moving when the features were frozen.
    lp_objective: List[float] = field(default_factory=list)

    @property
    def W(self) -> int:
        return len(self.pi)

    def arc_use_counts(self, n: int) -> np.ndarray:
        """B12's numerator: uses of arc (i, j) over the warm-up columns.

        Returned as an ``(n+1, n+1)`` integer matrix indexed by node, with the
        depot arcs the columns actually contain (a route's first and last arc).
        """
        counts = np.zeros((n + 1, n + 1), dtype=np.int64)
        for batch in self.columns:
            for route in batch:
                prev = 0
                for j in route:
                    counts[prev, j] += 1
                    prev = j
                counts[prev, 0] += 1
        return counts


@dataclass
class CGConfig:
    max_iterations: int = 400
    max_columns_per_iter: int = 25
    eps: float = 1e-6
    confirm_with_exact: bool = True
    log_every: int = 0          # 0 = silent
    #: Exact labelling is only affordable on small instances: measured on
    #: Solomon C101 it costs 0.07 s at n=10, 1.5 s at n=15, 105 s at n=20 and
    #: 264 s at n=25. Beyond this size the loop declines to certify rather
    #: than hang, and reports optimal_lp=False.
    exact_confirmation_max_n: int = 40
    #: Separate ceiling for how large an instance the *fallback* pricer may be
    #: called on.  When the fallback is an ng-route relaxation rather than the
    #: exact labeller it stays affordable far beyond 40 customers, so tying
    #: the two limits together would silently disable certification on exactly
    #: the medium-sized instances the study is about.
    fallback_max_n: Optional[int] = None
    #: Stop after this many consecutive iterations in which the LP objective
    #: fails to improve.  Set covering LPs are frequently degenerate: a pricer
    #: can keep finding genuinely negative columns that cannot move the bound,
    #: so "no column left" is a much weaker stopping signal than it looks.
    #: The rule is applied identically to every pricer, which is what keeps the
    #: comparison fair; 0 disables it and runs to max_iterations.
    stall_limit: int = 0
    #: Improvement threshold that `stall_limit` counts against, in objective
    #: units (spec v3.4 §6.2: PLATEAU = 30 consecutive iterations improving the
    #: LP by less than 1e-9).  It is a *separate* knob from `eps` on purpose:
    #: `eps` (1e-6) gates pricing and dominance decisions where a looser
    #: tolerance is right, and tightening it globally would change pricing
    #: behaviour and move the bit-level anchors.  This one only decides whether
    #: an iteration counts as progress for the stopping rule, so it can be set
    #: to the spec's 1e-9 without touching anything else.
    plateau_tol: float = 1e-9
    #: Dual stabilization.  "off" hands the pricer the master's own duals.
    #: "smooth" hands it a convex combination
    #:     pi_stab = alpha * center + (1 - alpha) * pi
    #: with `center` the dual of the best LP seen so far.  This is Vanderbeck's
    #: (2005) convex-combination smoothing in its simplest form, aimed exactly
    #: at the pathology this project measured: the LP accepts negative columns
    #: iteration after iteration without moving the bound, because a degenerate
    #: optimum leaves the dual solution free to wander.  Vanderbeck names that
    #: state a *plateau*, and stabilization is the standard remedy for it.
    #: "off" | "smooth" | "boxstep".  The third is du Merle et al.'s (1999)
    #: alternative to smoothing, and the difference is structural rather than
    #: a matter of tuning: smoothing *always* pulls the price towards the
    #: center - including on the coordinates where the current dual is
    #: informative - while boxstep only *clamps* the coordinates that leave a
    #: box.  Measured on this project's instances, smoothing does not cure the
    #: plateau, and the mechanism is identifiable: it keeps feeding the pricer
    #: columns that look good against the center, so `added == 0` never fires
    #: and the handover to the fallback pricer is never triggered.  Boxstep
    #: damps the drift without freezing the signal, which is the shape a
    #: plateau calls for.  Reported as a measurement, not as a fix - if it
    #: fails too, the reason gets written down the same way.
    stabilization: str = "off"
    #: Box half-width for `stabilization="boxstep"`, in dual-scale units:
    #:
    #:     scale   = mean(|center|)
    #:     delta_i = box_delta * (|center_i| + scale)
    #:
    #: The `+ scale` floor matters: without it the box collapses precisely on
    #: the coordinates whose dual is near zero, which is common on a degenerate
    #: LP, and the arm would freeze rather than clamp.  Scale-free by
    #: construction, so one knob covers every instance size - the alternative
    #: (an absolute box width) would have to be recalibrated per instance and
    #: would make the arm look tuned rather than measured.
    box_delta: float = 0.5
    #: Weight on the center.  alpha = 0 is by construction identical to "off"
    #: (pi_stab collapses to pi), which is the invariant the self-test pins
    #: down; alpha = 1 would price against a frozen center and make no progress.
    stabilization_alpha: float = 0.7
    #: Leave the first `stabilization_warmup` iterations unstabilized so the
    #: center is formed from real LP solves rather than from the seeding
    #: columns.  Vanderbeck calls this the heading-in phase.
    stabilization_warmup: int = 2
    #: A stabilized dual is a *heuristic* objective: a route that looks
    #: attractive against pi_stab may have non-negative reduced cost against
    #: the master's true duals.  With this on - and it belongs on - every
    #: batch is re-checked before it enters the pool, and a stabilized
    #: iteration that produces nothing re-prices with the true duals before
    #: the loop is allowed to conclude anything.  Turning it off is only
    #: useful to demonstrate the failure mode; it cannot be part of a run
    #: whose LP bound is reported as exact.
    verify_against_true_duals: bool = True
    #: Column selection policy.  The reference paper (Shen et al., AAAI 2022)
    #: compares three ways of moving a pricing batch into the master, and
    #: reports that "add everything" slows the RMP down.  Our own measurements
    #: saw the same thing from the other side: `dual:500` was *worse* than
    #: `dual:200`.  So the three modes are reproduced here.
    #:
    #:   "all"      - add every column the pricer returns (baseline)
    #:   "partial"  - score by TRUE reduced cost, add the most negative k only
    #:   "diverse"  - greedy by true reduced cost, but skip a column that is
    #:                too similar (Jaccard on the customer set) to one already
    #:                accepted in this batch or to one the current LP is using
    #:
    #: Selection is a restriction of the column pool, so it can never change
    #: the LP optimum - only how many iterations it takes to reach it.  The
    #: one hazard is dropped-everything, which would look like "no negative
    #: column exists"; the code always keeps at least one column whenever the
    #: pricer offered any, so that state is unreachable.
    column_selection: str = "all"
    #: For "partial": keep this many.  0 means use `column_selection_frac`.
    column_selection_k: int = 0
    #: For "partial" when `column_selection_k` is 0.
    column_selection_frac: float = 0.5
    #: For "diverse": a candidate is rejected when its customer set overlaps
    #: an accepted one by more than this Jaccard ratio.
    diversity_max_jaccard: float = 0.5

    # ---- §7.2 item 2: the warm-up / freeze / mask interface ---------------
    #: Iterations to run on the FULL network before arc selection begins.
    #: This is `W` of spec §1.  0 disables arc selection, which is also what a
    #: missing hook does: the baseline arm is "same loop, no hook", not a
    #: separate implementation.
    mask_after: int = 0
    #: Called exactly once, at the end of iteration `mask_after`, with a
    #: `WarmupTrace` covering iterations 1..mask_after.  Returns a boolean
    #: ``(n+1, n+1)`` arc mask (True = arc may be used) or None for "this run
    #: reduces nothing".  `compute_features -> predict -> set_mask` of §7.2.2 is
    #: this callback; the loop owns the timing and the pricer owns the effect.
    #:
    #: A callable on the config rather than an argument to `run()` because it is
    #: part of the *protocol*, not of one invocation: ML-S, DynML-S and the
    #: baseline differ here and essentially nowhere else.
    mask_hook: Optional[Callable[[WarmupTrace], Optional[np.ndarray]]] = None
    #: §5 switch thresholds.  Fewer than `reduced_min_columns` columns from the
    #: reduced network means it is too weak and the loop returns to G; at least
    #: `reduced_max_columns` from G means it is worth re-entering G_r.
    reduced_min_columns: int = 30
    reduced_max_columns: int = 100

    # ---- optimality-gap diagnostic ---------------------------------------
    #: Record, per run, the value of the best known feasible integer solution
    #: and whether the restricted master LP reached it.
    #:
    #: This is a *diagnostic*, not a certificate, and the distinction is worth
    #: being precise about because the inequality points the unintuitive way.
    #: Adding a column relaxes the master, so the restricted master LP is an
    #: UPPER bound on the full LP optimum, not a lower one:
    #:
    #:     opt(full LP)  <=  opt(restricted LP)  <=  opt(restricted IP)
    #:
    #: Reaching the incumbent therefore says the incumbent is optimal *among the
    #: columns collected so far* -- nothing more.  Counterexample: customers
    #: {1,2}, pool {(1,2) at cost 10}, which gives restricted LP = 10 = the
    #: incumbent, while the full problem also has (1) and (2) at cost 3 each and
    #: a true optimum of 6.  A rule that stopped on "bound met the incumbent"
    #: would have certified 10 and been wrong.
    #:
    #: What it IS good for: a run whose gap is still open is demonstrably not
    #: finished, and a run whose LP is stuck while the gap is open is the exact
    #: signature of a dual-degeneracy plateau.  Reported, not acted on.
    diagnose_incumbent: bool = True
    #: Tolerance for "the bound reached the incumbent".  Both sides are sums of
    #: floating-point costs, so the comparison cannot be exact; 1e-6 is far below
    #: any cost difference the models can represent and far above the ~1e-13
    #: residue seen in practice.
    incumbent_tol: float = 1e-6


@dataclass
class CGSnapshot:
    iteration: int
    lp_objective: float
    num_columns: int
    added: int
    min_reduced_cost: float
    pricer: str
    elapsed: float
    dual_source: str = ""
    extra: dict = field(default_factory=dict)


@dataclass
class CGResult:
    snapshots: List[CGSnapshot]
    lp_objective: float
    ip_objective: float
    ip_routes: List[tuple]
    num_vehicles: int
    iterations: int
    columns_in_pool: int
    pricing_calls: int
    exact_confirmation_calls: int
    wall_clock: float
    optimal_lp: bool
    exact_confirmation_skipped: bool = False
    stalled: bool = False
    #: Iterations that priced against a smoothed dual rather than the master's.
    stabilized_iterations: int = 0
    #: Columns a stabilized batch proposed that turned out not to be negative
    #: against the true duals, and were therefore dropped.
    filtered_columns: int = 0
    #: Columns the pricer offered over the whole run (sum of batch sizes).
    offered_columns: int = 0
    #: Columns the selection policy discarded before they reached the pool.
    dropped_by_selection: int = 0
    #: Pricing calls that hit the label cap and therefore returned no valid
    #: relaxation bound.  Any non-zero value makes the whole run unusable as
    #: evidence: the LP bound is not certified and the iteration/column counts
    #: were measured under a pricer that was silently degraded (spec §3.5).
    truncated_pricing_calls: int = 0
    #: Of those, the ones the protocol did NOT schedule: a cap the clock forced,
    #: or the pricer's own intrinsic label cap.  v3.6 ③: only these degrade a run.
    #: A descent round cut on purpose at `DESCENT_LABELS` is the protocol working,
    #: not the pricer failing -- and counting it as degradation would make
    #: `degraded` true for every run under the capped protocol, put every run's
    #: status at INCOMPLETE, and make a certificate unreachable.
    unscheduled_truncations: int = 0
    #: Largest label count seen in a single pricing call.
    peak_labels_created: int = 0
    #: Smallest label cap among the pricers this run used, or None if none of
    #: them is bounded.
    label_cap: Optional[int] = None
    #: Iterations priced on the full network / on the reduced one (§7.2 item 3).
    #: `full_network_iterations` is the parenthesised number §6.2 asks for in
    #: the main table; the two sum to `iterations`.
    full_network_iterations: int = 0
    reduced_network_iterations: int = 0
    #: Pricing calls issued while a mask that actually removed arcs was in
    #: force.  These calls publish no bound and cannot certify anything.
    masked_pricing_calls: int = 0
    #: Arcs the installed mask removed, and the size of the maskable set (n^2
    #: for n customers, spec §2).  `1 - masked_arcs / arcs_total` is the
    #: network reduction §6.2 wants alongside the gain.
    masked_arcs: int = 0
    arcs_total: int = 0
    #: Sum, over the masked pricing calls that were actually made, of the arcs
    #: each of those calls removed; and the mean that implies.  This is what
    #: §6.2's arc-share column must be read from.  `masked_arcs` above is the
    #: size of the mask the run's *hook* installed -- one number for the whole
    #: run -- and it is 0 for a pricer that reduces the network on its own
    #: schedule (RedCost-S), because such a pricer installs nothing: it hands a
    #: different reduction to each call.  Reporting `masked_arcs / arcs_total`
    #: for RedCost-S would print a confident 0.0% for the arm whose entire
    #: mechanism is arc removal.
    arcs_removed_total: int = 0
    masked_arcs_mean: float = 0.0
    #: True once the reduced network returned nothing and the VRPTW variant of
    #: the switching rule disabled it for the rest of the run (§0.2).  After
    #: this the loop is plain full-network column generation.
    reduced_dead: bool = False
    #: True when the hook ran.  False with `arcs_total > 0` means the run
    #: finished before the warm-up length, so arc selection never got a chance
    #: to act -- reported rather than silently counted as "no gain".
    mask_hook_fired: bool = False
    #: Reduced-network calls that returned nothing but were truncated, and so
    #: were NOT allowed to trigger the permanent `reduced_dead` flag (spec §5
    #: line 8, the R-8 fix).  Expected to be 0; a non-zero value says the
    #: label cap is binding on the reduced network, which is itself informative.
    reduced_kill_blocked_by_truncation: int = 0
    #: Cumulative wall time inside pricers (PP) and inside the master and the
    #: column pool (RMP).  §6.2's main table is written in these three columns,
    #: so they are *measured* at the call sites rather than inferred.  Reporting
    #: the split as `total - PP` would silently fold every other cost into RMP
    #: time and flatter whichever arm does more bookkeeping -- and the two arms
    #: being compared here differ by exactly that.  Whatever is left over is
    #: `other_seconds`, reported instead of hidden.
    pp_seconds: float = 0.0
    rmp_seconds: float = 0.0
    #: True when the restricted master LP reached the value of the best known
    #: feasible integer solution.  A diagnostic only -- see
    #: `CGConfig.diagnose_incumbent` for why this is not an optimality proof.
    bound_meets_incumbent: bool = False
    #: The feasible integer solution value the bound was compared against.
    incumbent_value: float = float("inf")
    #: Iterations that priced against an unverified dual.  Those iterations can
    #: still generate useful columns, but nothing they produce may be certified.
    unverified_dual_iterations: int = 0

    @property
    def optimality_gap(self) -> float:
        """`incumbent - LP`, the gap a certificate would have to close.

        Zero means the LP has caught up with the best known feasible solution;
        positive and constant across iterations is the signature of a plateau.
        """
        if not np.isfinite(self.incumbent_value):
            return float("nan")
        return float(self.incumbent_value - self.lp_objective)

    @property
    def mean_arc_reduction(self) -> Optional[float]:
        """Arc share removed per masked call, averaged over masked calls.

        None when no masked call was made, which is a different statement from
        "removed nothing": the reduced arm that never reduced and the arm that
        was never asked to are not the same measurement.
        """
        if not self.arcs_total or not self.masked_arcs_mean:
            return None
        return self.masked_arcs_mean / self.arcs_total

    @property
    def other_seconds(self) -> float:
        """Wall time outside pricing and the master: bookkeeping and logging."""
        return self.wall_clock - self.pp_seconds - self.rmp_seconds

    @property
    def arc_reduction(self) -> Optional[float]:
        """Fraction of maskable arcs removed, or None when no mask was used."""
        if not self.arcs_total:
            return None
        return self.masked_arcs / self.arcs_total

    # ------------------------------------------------------------------
    @property
    def status(self) -> str:
        """One of CERTIFIED / PLATEAU / INCOMPLETE (spec v3.4 §6.2).

        The three labels are not a convenience wrapper around the existing
        flags, they are the reporting contract, and the order of the tests is
        what makes them mean something:

        CERTIFIED   the run proved its LP value: the pricer certified that no
                    negative column exists, against the master's own verified
                    duals, on the full network.  Only these runs may enter the
                    main table's timing comparison.
        PLATEAU     the stall rule fired -- 30 consecutive iterations with an LP
                    improvement below `plateau_tol`.  The value reached is
                    probably right but is *not* a proof, so its wall clock is
                    reported separately and never averaged with CERTIFIED runs.
        INCOMPLETE  neither: the iteration/column budget ran out, or the run is
                    `degraded` (spec §3.5.2).  `degraded` outranks the other two
                    deliberately -- a run whose pricer hit the label cap is
                    excluded from evidence even if it happens to certify, so
                    letting it report CERTIFIED would smuggle it into the main
                    table through the back door.
        """
        if self.degraded:
            return "INCOMPLETE"
        if self.optimal_lp:
            return "CERTIFIED"
        if self.stalled:
            return "PLATEAU"
        return "INCOMPLETE"

    @property
    def counts_toward_main_table(self) -> bool:
        """True only for CERTIFIED runs (spec v3.4 §6.2 / §8).

        Exists as a named property rather than an inline comparison so the
        "only CERTIFIED enters the timing comparison" rule has exactly one
        definition to audit, and so a future status label cannot silently join
        the main table by being spelled differently.
        """
        return self.status == "CERTIFIED"

    # ------------------------------------------------------------------
    @property
    def degraded(self) -> bool:
        """True when this run may not be used as training or experimental data.

        Two independent ways in, both from spec §3.5: a pricing call hit a cap the
        protocol had NOT scheduled, or the peak label count came within
        `CLEAN_LABEL_FRACTION` of the pricer's intrinsic cap -- truncation
        narrowly missed is still too close to the edge.

        v3.6 ③ narrowed the first way.  It used to be "any call was truncated",
        which was right when truncation could only mean an accident; under the
        capped protocol every descent round is truncated by design, so the
        original test made `degraded` true for every run and a certificate
        unreachable.  What still degrades a run is a truncation nobody scheduled
        -- the wall clock (whose timing, and therefore whose work, is not
        reproducible) or the intrinsic label cap.
        """
        if self.unscheduled_truncations:
            return True
        if self.label_cap:
            return self.peak_labels_created >= CLEAN_LABEL_FRACTION * self.label_cap
        return False

    def summary(self) -> dict:
        return {
            "lp_objective": self.lp_objective,
            "ip_objective": self.ip_objective,
            "num_vehicles": self.num_vehicles,
            "iterations": self.iterations,
            "columns_in_pool": self.columns_in_pool,
            "pricing_calls": self.pricing_calls,
            "exact_confirmation_calls": self.exact_confirmation_calls,
            "wall_clock": self.wall_clock,
            "optimal_lp": self.optimal_lp,
            #: The reporting contract from spec v3.4 §6.2.  Carried in the
            #: summary so a results file can be grouped by it without every
            #: consumer re-deriving the precedence rules from the flags.
            "status": self.status,
            "counts_toward_main_table": self.counts_toward_main_table,
            "exact_confirmation_skipped": self.exact_confirmation_skipped,
            "stalled": self.stalled,
            "stabilized_iterations": self.stabilized_iterations,
            "filtered_columns": self.filtered_columns,
            "offered_columns": self.offered_columns,
            "dropped_by_selection": self.dropped_by_selection,
            "truncated_pricing_calls": self.truncated_pricing_calls,
            "unscheduled_truncations": self.unscheduled_truncations,
            "peak_labels_created": self.peak_labels_created,
            "label_cap": self.label_cap,
            "degraded": self.degraded,
            "full_network_iterations": self.full_network_iterations,
            "reduced_network_iterations": self.reduced_network_iterations,
            "masked_pricing_calls": self.masked_pricing_calls,
            "masked_arcs": self.masked_arcs,
            "arcs_total": self.arcs_total,
            "arc_reduction": self.arc_reduction,
            "reduced_dead": self.reduced_dead,
            "mask_hook_fired": self.mask_hook_fired,
        }

    # ------------------------------------------------------------------
    def degeneracy(self, target: float = 0.99) -> "DegeneracyReport":
        """How much of this run was spent on a degenerate LP.

        Derived entirely from `snapshots`, which the loop already records, so
        asking for it costs nothing.
        """
        return degeneracy_report(self.snapshots, self.lp_objective, target=target)


@dataclass
class DegeneracyReport:
    """Degeneracy diagnostics for one column generation run.

    Motivation
    ----------
    "The pricing problem found no negative column" is a much weaker stopping
    signal on a covering LP than it looks.  Our own measurements on n=25
    Solomon instances show the LP sitting in a state where a pricer keeps
    returning genuinely negative columns that cannot move the bound at all.
    A run like that burns iterations and columns for no progress, and - more
    importantly for this project - it hands the duals a long stretch in which
    they are free to drift, which is what stabilization is meant to remove.

    The numbers below make that visible instead of anecdotal.
    """

    iterations: int
    seed_lp: float
    final_lp: float
    #: seed_lp - final_lp.  Positive means the bound improved (minimisation).
    total_gain: float
    #: Iterations that added at least one column *and* moved the bound.
    improving_iterations: int
    #: Iterations that added at least one column and did *not* move the bound.
    #: The signature of degeneracy.
    wasted_iterations: int
    #: `wasted_iterations` over the iterations that added any column at all.
    #: Restricted to column-adding iterations on purpose: an iteration that
    #: adds nothing is a termination or a fallback, not waste.
    wasted_fraction: float
    #: Longest run of consecutive wasted iterations.
    longest_plateau: int
    #: Relative L1 movement of the dual vector between consecutive iterations,
    #: ||pi_t - pi_{t-1}||_1 / ||pi_t||_1.  Large values are Vanderbeck's
    #: "yo-yo"; the mean is the headline number.
    dual_move_mean: float
    dual_move_max: float
    #: Fraction of `total_gain` used as the target, and the number of pricing
    #: iterations needed to capture it.  This is the number the comparison in
    #: the experimental chapter should quote: two pricers can reach the same
    #: final bound while one of them gets there in half the iterations.
    target: float
    iters_to_target: int
    #: Bound improvement per column added.  Falls when the pool bloats.
    gain_per_column: float

    def as_dict(self) -> dict:
        return {
            "iterations": self.iterations,
            "seed_lp": self.seed_lp,
            "final_lp": self.final_lp,
            "total_gain": self.total_gain,
            "improving_iterations": self.improving_iterations,
            "wasted_iterations": self.wasted_iterations,
            "wasted_fraction": self.wasted_fraction,
            "longest_plateau": self.longest_plateau,
            "dual_move_mean": self.dual_move_mean,
            "dual_move_max": self.dual_move_max,
            "target": self.target,
            "iters_to_target": self.iters_to_target,
            "gain_per_column": self.gain_per_column,
        }


def degeneracy_report(
    snapshots: Sequence[CGSnapshot],
    final_lp: float,
    target: float = 0.99,
    eps: float = 1e-6,
) -> DegeneracyReport:
    """Summarise a run's snapshots into a `DegeneracyReport`.

    Timing detail that the indices below encode: iteration i solves the LP
    (recording `vals[i]`) and only then adds its columns, so the effect of
    iteration i's batch shows up in `vals[i + 1]`.
    """
    if not snapshots:
        return DegeneracyReport(
            iterations=0, seed_lp=float(final_lp), final_lp=float(final_lp),
            total_gain=0.0, improving_iterations=0, wasted_iterations=0,
            wasted_fraction=0.0, longest_plateau=0, dual_move_mean=0.0,
            dual_move_max=0.0, target=target, iters_to_target=0,
            gain_per_column=0.0,
        )

    vals = [float(s.lp_objective) for s in snapshots] + [float(final_lp)]
    added = [int(s.added) for s in snapshots]
    seed = vals[0]
    total = seed - float(final_lp)
    gains = [vals[i] - vals[i + 1] for i in range(len(added))]

    improving = 0
    wasted = 0
    with_columns = 0
    longest = 0
    run = 0
    for n_add, gain in zip(added, gains):
        if n_add <= 0:
            run = 0
            continue
        with_columns += 1
        if gain > eps:
            improving += 1
            run = 0
        else:
            wasted += 1
            run += 1
            longest = max(longest, run)

    moves = [float(s.extra.get("pi_move_rel", 0.0)) for s in snapshots[1:]]

    iters_to_target = len(vals) - 1
    if total > eps:
        need = target * total
        for k in range(1, len(vals)):
            if seed - vals[k] >= need - eps:
                iters_to_target = k
                break

    return DegeneracyReport(
        iterations=len(snapshots),
        seed_lp=seed,
        final_lp=float(final_lp),
        total_gain=total,
        improving_iterations=improving,
        wasted_iterations=wasted,
        wasted_fraction=(wasted / with_columns) if with_columns else 0.0,
        longest_plateau=longest,
        dual_move_mean=(sum(moves) / len(moves)) if moves else 0.0,
        dual_move_max=max(moves) if moves else 0.0,
        target=target,
        iters_to_target=iters_to_target,
        gain_per_column=(total / sum(added)) if sum(added) else 0.0,
    )


def boxstep_dual(
    pi: np.ndarray,
    sigma: float,
    center_pi: np.ndarray,
    center_sigma: float,
    box_delta: float,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, float]:
    """Clamp a dual solution into a box around the stabilization centre.

    du Merle, Villeneuve, Desrosiers & Hansen (1999); the form is the one
    Vanderbeck (2005, section 3.2) describes as boxstep.  The contrast with
    convex-combination smoothing is the whole point of having it here:

        smooth    pi_price = alpha * centre + (1 - alpha) * pi
        boxstep   pi_price = clip(pi, centre - d, centre + d)

    Smoothing *always* pulls the price towards the centre, including on the
    coordinates where the current dual carries the information that would let
    the loop make progress.  Boxstep leaves those alone and only clamps the
    coordinates that run away, so it damps the drift on a plateau without
    freezing the signal.

    The half-width is scale-free - one knob:

        scale   = mean(|centre|)
        d_i     = box_delta * (|centre_i| + scale)

    The `+ scale` floor is load-bearing.  A degenerate LP routinely leaves some
    duals at or near zero, and a purely relative box would be empty exactly on
    those coordinates, which would freeze the arm instead of clamping it.

    Written as a pure function rather than inlined in the loop so the
    self-test can assert the two properties that matter - the result is always
    inside the box, and a wide enough box is the identity - against the same
    code path the run uses.
    """
    pi = np.asarray(pi, dtype=float)
    scale = max(float(np.abs(center_pi).mean()), eps)
    half = box_delta * (np.abs(center_pi) + scale)
    s_half = box_delta * (abs(center_sigma) + scale)
    return (
        np.clip(pi, center_pi - half, center_pi + half),
        float(np.clip(sigma, center_sigma - s_half, center_sigma + s_half)),
    )


class ColumnGeneration:
    def __init__(
        self,
        inst: Instance,
        pricer: Pricer,
        master: Optional[MasterProblem] = None,
        config: Optional[CGConfig] = None,
        fallback_pricer: Optional[Pricer] = None,
        seed_routes: Optional[Sequence[Sequence[int]]] = None,
        max_vehicles: Optional[int] = None,
        vehicle_fixed_cost: float = 0.0,
    ) -> None:
        self.inst = inst
        self.pricer = pricer
        self.config = config or CGConfig()
        self.master = master or MasterProblem(
            inst, max_vehicles=max_vehicles, vehicle_fixed_cost=vehicle_fixed_cost
        )
        self.fallback = fallback_pricer
        self.seed_routes = seed_routes
        #: running totals behind `CGResult.offered_columns` /
        #: `CGResult.dropped_by_selection`.  Reset at the start of every run so
        #: a reused ColumnGeneration object cannot carry stale counters.
        self._offered = 0
        self._dropped = 0
        #: The arc mask the warm-up hook installed, or None.  Held here rather
        #: than left on the pricer because the pricer is unmasked again after
        #: every masked call -- see `_price`.
        self._mask = None

    # ------------------------------------------------------------------
    def _price(
        self,
        reduced: bool,
        pi: np.ndarray,
        sigma: float,
        max_columns: int,
    ) -> PricingResult:
        """Price on the heuristic path, leaving the pricer unmasked afterwards.

        The `finally` block is the entire point of this method.  Everything else
        in the loop that calls a pricer -- the stabilized re-price, the exact
        fallback, and therefore every call that is allowed to certify -- runs
        with the mask off *by construction* rather than by remembering to turn
        it off.  Spec §7.2 item 1 makes "certification only on the full network"
        a red line; a red line that depends on call-site discipline is a bug
        waiting for a refactor.
        """
        pricer = self.pricer
        if not reduced:
            return pricer.price(
                pi,
                sigma=sigma,
                fixed_cost=self.master.vehicle_fixed_cost,
                max_columns=max_columns,
            )
        pricer.set_mask(self._mask)
        try:
            return pricer.price(
                pi,
                sigma=sigma,
                fixed_cost=self.master.vehicle_fixed_cost,
                max_columns=max_columns,
            )
        finally:
            pricer.set_mask(None)

    # ------------------------------------------------------------------
    def _admit(
        self,
        columns: Sequence[Tuple[Sequence[int], float]],
        pi: np.ndarray,
        sigma: float,
        active: Sequence[Sequence[int]] = (),
    ) -> int:
        """Run the selection policy on a batch, then add what survives.

        The escalation below is not a refinement, it is a correctness guard.
        `added == 0` is how the loop is told "the pricer found nothing", and
        that is a much stronger statement than "the policy declined everything
        offered".  A selection rule that keeps only the single most negative
        column will, sooner or later, keep one the pool already holds - and
        then the run would stop mid-convergence while genuinely negative
        columns were still on the table.  That was observed, not imagined: on
        R107 with ng-route, `partial-k2` and the filter-disabled arm both
        stopped at iteration 51 reporting a bound equal to the seed value.

        So when the selected subset turns out to be entirely duplicates, the
        rest of the batch is admitted.  Selection stays a *preference* about
        what to add first; it never gets to decide that the run is finished.
        """
        self._offered += len(columns)
        kept = self._select_columns(columns, pi, sigma, active=active)
        self._dropped += len(columns) - len(kept)
        added = self.master.add_many(kept)
        if added == 0 and len(kept) < len(columns):
            chosen = {k for k, _ in kept}
            rest = [
                (tuple(int(v) for v in n), c)
                for n, c in columns
                if tuple(int(v) for v in n) not in chosen
            ]
            extra = self.master.add_many(rest)
            self._dropped -= len(rest)
            added += extra
        return added

    # ------------------------------------------------------------------
    def _seed_master(self) -> int:
        from .instance import evaluate_route

        if self.master.routes:
            return 0
        routes = (
            list(self.seed_routes)
            if self.seed_routes is not None
            else cheapest_insertion(self.inst)
        )
        added = 0
        for r in routes:
            ev = evaluate_route(self.inst, r)
            if not ev.feasible:
                continue
            if self.master.add(r, ev.cost):
                added += 1
        if len(self.master) == 0:
            raise RuntimeError("seeding produced no feasible column")
        return added

    # ------------------------------------------------------------------
    def _keep_truly_negative(
        self,
        columns: Sequence[Tuple[Sequence[int], float]],
        pi: np.ndarray,
        sigma: float,
    ) -> Tuple[List[Tuple[Tuple[int, ...], float]], int]:
        """Split a stabilized batch into genuinely negative columns and noise.

        A smoothed dual is deliberately *not* the master's dual, so a route the
        pricer found attractive need not be attractive to the real master.
        Adding it anyway is not a correctness error - extra columns can only
        help a covering LP - but it is not free either: it bloats the pool and
        it makes the pricer's own reduced-cost report describe the wrong
        objective.  Recomputing costs one `evaluate_route` per candidate, on a
        batch the pricer has already spent far more than that on.

        Returns the surviving `(route, true reduced cost)` pairs and the number
        of columns dropped.
        """
        kept: List[Tuple[Tuple[int, ...], float]] = []
        for nodes, _ in columns:
            rc = reduced_cost_of(
                self.inst, nodes, pi, sigma, self.master.vehicle_fixed_cost
            )
            if rc < -self.config.eps:
                kept.append((tuple(int(v) for v in nodes), rc))
        return kept, len(columns) - len(kept)

    # ------------------------------------------------------------------
    def _score_batch(
        self,
        columns: Sequence[Tuple[Sequence[int], float]],
        pi: np.ndarray,
        sigma: float,
    ) -> List[Tuple[float, Tuple[int, ...]]]:
        """Rank a batch by *true* reduced cost, most negative first.

        Ranking has to use the real duals even during a stabilized iteration:
        the whole point of the selection is to keep the columns that pay off
        for the master we actually have, and pi_stab is not that master.
        Equal reduced costs fall back on the route as a tie-break so the order
        is deterministic and two runs stay comparable.
        """
        scored: List[Tuple[float, Tuple[int, ...]]] = []
        for nodes, _ in columns:
            key = tuple(int(v) for v in nodes)
            rc = reduced_cost_of(
                self.inst, key, pi, sigma, self.master.vehicle_fixed_cost
            )
            scored.append((rc, key))
        scored.sort(key=lambda t: (t[0], t[1]))
        return scored

    @staticmethod
    def _jaccard(a: Sequence[int], b: Sequence[int]) -> float:
        sa, sb = set(a), set(b)
        union = len(sa | sb)
        return len(sa & sb) / union if union else 0.0

    def _select_columns(
        self,
        columns: Sequence[Tuple[Sequence[int], float]],
        pi: np.ndarray,
        sigma: float,
        active: Sequence[Sequence[int]] = (),
    ) -> List[Tuple[Tuple[int, ...], float]]:
        """Apply the configured column-selection policy to one pricing batch.

        Any policy here is a *restriction* of the pool, so none of them can
        move the LP optimum; they only change how many iterations it takes to
        get there and how big the pool grows on the way.  That is exactly why
        they are safe to compare against stabilization on equal footing - both
        are path heuristics, and both must land on the same certified bound.
        """
        mode = self.config.column_selection
        if mode == "all" or not columns:
            return [(tuple(int(v) for v in n), c) for n, c in columns]

        scored = self._score_batch(columns, pi, sigma)

        if mode == "partial":
            k = self.config.column_selection_k
            if k <= 0:
                k = int(round(self.config.column_selection_frac * len(scored)))
            # never return an empty batch when negative columns were offered:
            # an empty batch is read by the loop as "the pricer found nothing",
            # which is a strictly stronger statement than "we chose nothing".
            k = max(1, min(k, len(scored)))
            chosen = scored[:k]
        elif mode == "diverse":
            limit = float(self.config.diversity_max_jaccard)
            # Reference sets: what the current LP is already using, plus what
            # this batch has accepted so far.  Using the LP's active columns
            # rather than the whole pool keeps the comparison cheap and targets
            # the thing that actually matters - not restating a column the
            # master is already built on.
            seen: List[Tuple[int, ...]] = [tuple(int(v) for v in r) for r in active]
            chosen = []
            for rc, key in scored:
                if any(self._jaccard(key, s) > limit for s in seen):
                    continue
                chosen.append((rc, key))
                seen.append(key)
            if not chosen:
                chosen = scored[:1]
        else:
            raise ValueError(f"unknown column_selection {mode!r}")

        return [(key, rc) for rc, key in chosen]

    # ------------------------------------------------------------------
    def run(self) -> CGResult:
        cfg = self.config
        t0 = time.perf_counter()
        self._offered = 0
        self._dropped = 0
        self._seed_master()

        #: Best known feasible integer solution.  The seed is a feasible cover,
        #: so summing the cost vector over the seed columns gives a valid
        #: incumbent before a single pricing call is made.
        incumbent = float(np.sum(self.master.cost_vector))
        unverified_iters = 0

        snapshots: List[CGSnapshot] = []
        pricing_calls = 0
        exact_calls = 0
        optimal = False
        confirmation_skipped = False
        stalled = False
        best_lp = float("inf")
        stall = 0
        #: How many stall rechecks have produced columns without the LP moving
        #: (v8).  Two such rechecks is a degenerate plateau, not convergence
        #: worth chasing; C201-n50 burned 4.7h / 120 iterations when unbounded.
        stall_rechecks = 0
        #: Reference for the `stall_limit` rule, tracked separately from
        #: `best_lp`.  `best_lp` (and therefore `center_pi`) only moves on
        #: improvements larger than `cfg.eps`; the plateau rule counts
        #: improvements larger than `cfg.plateau_tol`.  Running the stall
        #: counter off `best_lp` would let a stream of improvements in the band
        #: [plateau_tol, eps) reset the counter every iteration without ever
        #: updating the reference -- the counter would never fire.  Tying it to
        #: its own reference makes "30 consecutive iterations improving by less
        #: than 1e-9" mean exactly that (spec v3.4 §6.2).
        plateau_ref = float("inf")
        it = 0
        stabilized_iters = 0
        filtered_out = 0
        #: Spec §3.5 bookkeeping.  A pricing call that hit its label cap has no
        #: valid relaxation bound, so the run it belongs to proves nothing.
        truncated_calls = 0
        #: Calls that ran to completion.  Needed by the stall rule below: under
        #: the v3.6 capped protocol EVERY call can be truncated, and then the
        #: stall counter is measuring the cap rather than the instance.
        complete_pricing_calls = 0
        unscheduled_truncations = 0
        peak_labels = 0
        used_label_cap: Optional[int] = None
        #: §6.2 timing columns, accumulated at the call sites below.
        pp_seconds = 0.0
        rmp_seconds = 0.0
        #: §6.2 arc-share column, accumulated per pricing result.
        arcs_removed_total = 0
        masked_mean_calls = 0

        #: ---- §7.2 items 1-3: warm-up, mask, network accounting ------------
        mask_available = cfg.mask_after > 0 and cfg.mask_hook is not None
        warm_pi: List[np.ndarray] = []
        warm_sigma: List[float] = []
        warm_rc: List[np.ndarray] = []
        warm_cols: List[List[Tuple[int, ...]]] = []
        warm_obj: List[float] = []
        hook_fired = False
        use_reduced = False
        reduced_dead = False
        full_iters = 0
        reduced_iters = 0
        masked_calls = 0
        masked_arcs = 0
        kill_blocked = 0

        if mask_available and not hasattr(self.pricer, "set_mask"):
            raise ValueError(
                "mask_hook was given but the pricer cannot honour a mask (no "
                "set_mask). Arc selection may only be mounted on a pricer that "
                "supports it -- the exact pricer is the certification circuit "
                "and must stay on the full network (spec §7.2 item 1)."
            )
        if mask_available and cfg.stabilization != "off":
            raise ValueError(
                "arc masking and dual stabilization are not combined here. "
                "§5's switching rule -- including the permanent death of G_r -- "
                "is defined against the master's own duals, and a call priced "
                "against a smoothed dual cannot support that decision any more "
                "than it can publish a bound. Run the two as separate arms."
            )
        if self.fallback is not None and getattr(self.fallback, "masked_arcs", 0):
            raise ValueError(
                "the fallback pricer must run on the full network; it currently "
                "arrives with a mask installed (spec §7.2 item 1)"
            )

        def _note_price(result: PricingResult) -> None:
            """Record cap hits so the run can be refused as evidence (§3.5).

            Every pricing call goes through here, including ones whose result
            is later replaced.  That is deliberate: once a pricer is operating
            at its cap the label space is not being explored, so the columns
            the loop did see are not the columns it would have seen -- the run
            is suspect even if the surviving result looks fine.
            """
            nonlocal truncated_calls, peak_labels, used_label_cap
            nonlocal arcs_removed_total, masked_mean_calls
            nonlocal complete_pricing_calls, unscheduled_truncations
            if result.truncated:
                truncated_calls += 1
                #: v3.6 ③: only an UNSCHEDULED cap degrades the run.  See
                #: `CGResult.degraded` for why this distinction is load-bearing.
                if not result.stats.get("scheduled_cap"):
                    unscheduled_truncations += 1
            else:
                complete_pricing_calls += 1
            if result.labels_created > peak_labels:
                peak_labels = result.labels_created
            cap = result.label_cap
            #: v3.6 ③, second half.  Only the pricer's OWN cap may be tracked.
            #: A cap the protocol scheduled for this call (the 300k descent slice)
            #: is not a degradation signal, and tracking it is actively harmful:
            #: `degraded`'s proximity test asks whether the peak got within
            #: CLEAN_LABEL_FRACTION of `label_cap`, and with a scheduled 300k cap
            #: that threshold becomes 240k -- which every run exceeds, so
            #: `degraded` was true for EVERY run, `status` could never be
            #: CERTIFIED, and the §6.2 main table was unreachable even for runs
            #: holding a genuine certificate.  Measured: RC201(n40,x0.6) certified
            #: (empty complete call, LP_min 634.23 vs insertion 1005.41) and still
            #: reported INCOMPLETE.
            if cap is not None and not result.stats.get("protocol_label_cap"):
                used_label_cap = cap if used_label_cap is None else min(used_label_cap, cap)
            # Arc share, read from the call rather than from the loop's intent.
            # Every pricer that can remove arcs reports the count under the same
            # key, so this one line covers both the hook-driven mask and a
            # pricer that decides its own reduction per call.
            if result.masked:
                masked_mean_calls += 1
                arcs_removed_total += int(result.stats.get("masked_arcs", 0) or 0)

        #: Stabilization centre: the dual of the best LP seen so far.  Refreshed
        #: whenever the bound improves, so on an improving iteration it *is* the
        #: current dual and pi_stab collapses onto pi - stabilization only bites
        #: on the iterations that made no progress, which are precisely the ones
        #: whose duals are free to drift.
        center_pi: Optional[np.ndarray] = None
        center_sigma = 0.0
        prev_pi: Optional[np.ndarray] = None

        for it in range(1, cfg.max_iterations + 1):
            _t_rmp = time.perf_counter()
            lp = self.master.solve_lp()
            rmp_seconds += time.perf_counter() - _t_rmp

            # ---- optimality-gap diagnostic ----------------------------------
            # The gap itself is reported at the end from `incumbent` and the
            # final LP value; nothing here changes control flow.  Kept as a
            # comment rather than removed because the temptation to "stop when
            # the bound reaches the incumbent" is real, and it is wrong: the
            # restricted master LP is an UPPER bound on the full LP optimum, so
            # that comparison says only that the incumbent is optimal among the
            # columns collected so far.  See `CGConfig.diagnose_incumbent`.
            if not lp.dual_verified:
                unverified_iters += 1

            #: Which network the *main* pricing call of this iteration went to,
            #: as far as the hook-driven state knows.  For a pricer that manages
            #: its own reduction (RedCost-S) this is always "full"; the honest
            #: answer comes from the result's own `masked` flag, and the
            #: bookkeeping below uses that instead.
            net = "reduced" if use_reduced else "full"

            # ---- §1 warm-up log ------------------------------------------
            # Recorded here, before the pricing call, so the trace holds the
            # dual of iteration t exactly as the master solved it -- which is
            # the definition §2.2's B1..B11 are written in.  The columns of
            # iteration t are appended at the end of the body below; the two
            # stay the same length because nothing can `continue` past the
            # second append while `use_reduced` is still False, and
            # `use_reduced` stays False until the hook has fired.
            if it <= cfg.mask_after and mask_available:
                warm_pi.append(np.array(lp.pi, dtype=float, copy=True))
                warm_sigma.append(float(lp.sigma))
                warm_rc.append(modified_cost(self.inst, lp.pi))
                warm_obj.append(float(lp.objective))

            # ---- degeneracy metric: how far the duals moved this iteration
            pi_move = 0.0
            if prev_pi is not None:
                denom = max(float(np.abs(lp.pi).sum()), cfg.eps)
                pi_move = float(np.abs(lp.pi - prev_pi).sum()) / denom
            prev_pi = np.array(lp.pi, dtype=float, copy=True)

            #: Columns the LP is actually built on right now.  `diverse` uses
            #: them as its reference set: the interesting redundancy to avoid
            #: is a new column that merely restates one the master already
            #: relies on, not one that resembles a dormant column in the pool.
            active_routes = [
                self.master.routes[j] for j in np.flatnonzero(np.asarray(lp.x) > 1e-9)
            ]

            improved = lp.objective < best_lp - cfg.eps
            if improved:
                best_lp = lp.objective
                center_pi = np.array(lp.pi, dtype=float, copy=True)
                center_sigma = float(lp.sigma)

            #: Same question as `improved` but at the spec's plateau tolerance,
            #: with its own reference so a run of sub-eps improvements cannot
            #: hold the stall counter at zero forever.
            plateau_improved = lp.objective < plateau_ref - cfg.plateau_tol
            if plateau_improved:
                plateau_ref = lp.objective

            # ---- pick the dual the pricing problem is written against
            alpha = 0.0
            stabilized = False
            pi_price, sigma_price = lp.pi, lp.sigma
            if center_pi is not None and it > cfg.stabilization_warmup:
                if (cfg.stabilization == "smooth"
                        and cfg.stabilization_alpha > 0.0):
                    alpha = float(cfg.stabilization_alpha)
                    pi_price = alpha * center_pi + (1.0 - alpha) * lp.pi
                    sigma_price = (alpha * center_sigma
                                   + (1.0 - alpha) * lp.sigma)
                    stabilized = True
                elif cfg.stabilization == "boxstep" and cfg.box_delta > 0.0:
                    pi_price, sigma_price = boxstep_dual(
                        lp.pi, lp.sigma, center_pi, center_sigma,
                        cfg.box_delta, eps=cfg.eps,
                    )
                    stabilized = True
            #: Whether the current `res` was produced against the master's true
            #: duals.  Only a result that was can certify anything.
            on_true_duals = not stabilized

            pricing_calls += 1
            _t_pp = time.perf_counter()
            res = self._price(
                use_reduced, pi_price, sigma_price, cfg.max_columns_per_iter
            )
            pp_seconds += time.perf_counter() - _t_pp
            _note_price(res)
            # Counted from the result, not from the loop's own state: a pricer
            # that reduces the network internally (RedCost-S) masks its calls
            # without the loop ever installing a hook mask, and a counter that
            # says "0 masked calls" about such a run would be worse than no
            # counter at all.
            if res.masked:
                masked_calls += 1

            # ---- §5 lines 8-9: the reduced network found nothing ----------
            # C_t = empty is the strongest thing a masked call can say, and it
            # still says nothing about G: the arcs it withheld were never
            # expanded. So the loop goes back to the full network before
            # concluding anything, and the only question left is whether the
            # return is permanent.
            #
            # A truncated call may not answer that question (R-8): its empty
            # column set is an artefact of the label cap, not a property of G_r.
            # Letting it set `reduced_dead` would take an irreversible decision
            # on evidence the run itself declares invalid.
            if use_reduced and not res.columns:
                if res.truncated:
                    kill_blocked += 1
                else:
                    reduced_dead = True
                snapshots.append(
                    CGSnapshot(
                        iteration=it,
                        lp_objective=lp.objective,
                        num_columns=len(self.master),
                        added=0,
                        min_reduced_cost=res.min_reduced_cost,
                        pricer=f"{self.pricer.name}[Gr dead]"
                        if reduced_dead
                        else f"{self.pricer.name}[Gr blocked]",
                        elapsed=time.perf_counter() - t0,
                        dual_source=lp.dual_source,
                        extra=dict(
                            res.stats,
                            network=net,
                            pi_move_rel=pi_move,
                            alpha=alpha,
                            stabilized=float(stabilized),
                            stab_mode="off",
                            reduced_death_switch=1.0,
                        ),
                    )
                )
                use_reduced = False
                reduced_iters += 1
                continue

            # ---- a smoothed dual is a heuristic objective, not a certificate
            if stabilized and res.columns and cfg.verify_against_true_duals:
                kept, dropped = self._keep_truly_negative(res.columns, lp.pi, lp.sigma)
                filtered_out += dropped
                on_true_duals = True
                res = PricingResult(
                    columns=kept,
                    min_reduced_cost=(min(rc for _, rc in kept) if kept else 0.0),
                    exact=res.exact,
                    stats={**res.stats, "stab_filtered": float(dropped)},
                    # the relaxation bound was measured against pi_stab, so it
                    # says nothing about the true duals and must not survive
                    relaxed_min_reduced_cost=None,
                    # ...and a cap hit upstream is still a cap hit: carry it so
                    # the run cannot re-enter as "clean" through this rebuild.
                    truncated=res.truncated,
                    # ...as is a mask.  Stabilization and masking cannot both be
                    # on in this implementation, so this is false in any run that
                    # gets this far; carrying it anyway keeps the rebuild from
                    # being the one place a contract field can go missing.
                    masked=res.masked,
                )

            _t_rmp = time.perf_counter()
            added = self._admit(res.columns, lp.pi, lp.sigma, active=active_routes)
            rmp_seconds += time.perf_counter() - _t_rmp
            used = self.pricer.name
            if stabilized:
                stabilized_iters += 1
                tag = (f"a={alpha:g}" if cfg.stabilization == "smooth"
                       else f"box={cfg.box_delta:g}")
                used = f"{used}[stab {tag}]"

            if added == 0 and stabilized:
                # Finding nothing against pi_stab proves nothing about pi, so
                # ask again with the true duals before concluding anything.
                pricing_calls += 1
                _t_pp = time.perf_counter()
                res = self._price(False, lp.pi, lp.sigma, cfg.max_columns_per_iter)
                pp_seconds += time.perf_counter() - _t_pp
                _note_price(res)
                _t_rmp = time.perf_counter()
                added = self._admit(res.columns, lp.pi, lp.sigma, active=active_routes)
                rmp_seconds += time.perf_counter() - _t_rmp
                used = f"{used}+true"
                on_true_duals = True

            if added == 0 and res.masked:
                # D2: a masked batch that is all duplicates is NOT a proof of
                # convergence.  `certifies_no_negative` is False for masked
                # calls by contract (base.py), and on a degenerate RMP the
                # masked network can keep returning columns the pool already
                # holds.  Ask the pricer for one true-dual full-network pass
                # before the loop is allowed to stop.
                pricing_calls += 1
                _t_pp = time.perf_counter()
                if hasattr(self.pricer, "set_mask"):
                    self.pricer.set_mask(None)
                res = self._price(False, lp.pi, lp.sigma,
                                  cfg.max_columns_per_iter)
                pp_seconds += time.perf_counter() - _t_pp
                _note_price(res)
                _t_rmp = time.perf_counter()
                added = self._admit(res.columns, lp.pi, lp.sigma,
                                    active=active_routes)
                rmp_seconds += time.perf_counter() - _t_rmp
                used = f"{used}+true"
                on_true_duals = True

            if added == 0 and not self.pricer.guarantees_optimality and self.fallback is not None:
                limit = (
                    cfg.fallback_max_n
                    if cfg.fallback_max_n is not None
                    else cfg.exact_confirmation_max_n
                )
                if self.inst.n > limit:
                    confirmation_skipped = True
                else:
                    if getattr(self.fallback, "masked_arcs", 0):
                        raise RuntimeError(
                            "fallback pricer arrived with a mask installed; the "
                            "fallback is part of the certification circuit and "
                            "must see the full network (spec §7.2 item 1)"
                        )
                    exact_calls += 1
                    _t_pp = time.perf_counter()
                    cres = self.fallback.price(
                        lp.pi,
                        sigma=lp.sigma,
                        fixed_cost=self.master.vehicle_fixed_cost,
                        max_columns=cfg.max_columns_per_iter,
                    )
                    pp_seconds += time.perf_counter() - _t_pp
                    _note_price(cres)
                    _t_rmp = time.perf_counter()
                    added = self._admit(cres.columns, lp.pi, lp.sigma, active=active_routes)
                    rmp_seconds += time.perf_counter() - _t_rmp
                    used = f"{self.pricer.name}+{self.fallback.name}"
                    res = cres
                    on_true_duals = True

            # ---- §5 lines 11-12: what network the NEXT iteration uses -----
            # Deliberately placed after the fallback: when the fallback ran, the
            # yield that matters for line 12 is the full-network yield, and
            # `res` is then the fallback's own result.
            #
            # The two branches cannot both fire, and line 12 is closed for good
            # once `reduced_dead` is set -- that is what stops the G_r -> G ->
            # G_r oscillation §5 warns about. While the warm-up is still
            # running, `masked_arcs == 0` closes line 12 instead.
            n_offered = len(res.columns)
            if use_reduced and n_offered < cfg.reduced_min_columns and not res.truncated:
                use_reduced = False
            elif (not use_reduced and not reduced_dead and masked_arcs > 0
                  and n_offered >= cfg.reduced_max_columns):
                use_reduced = True

            # §7.2 item 3, read off the result: "which network did this
            # iteration actually price on" is a fact about the call, not about
            # the loop's intent.
            net_used = "reduced" if res.masked else "full"
            if res.masked:
                reduced_iters += 1
            else:
                full_iters += 1

            snapshots.append(
                CGSnapshot(
                    iteration=it,
                    lp_objective=lp.objective,
                    num_columns=len(self.master),
                    added=added,
                    min_reduced_cost=res.min_reduced_cost,
                    pricer=used,
                    elapsed=time.perf_counter() - t0,
                    dual_source=lp.dual_source,
                    extra=dict(res.stats, pi_move_rel=pi_move, alpha=alpha,
                               stabilized=float(stabilized),
                               stab_mode=(cfg.stabilization if stabilized
                                          else "off"),
                               # §7.2 item 3: which network this iteration
                               # priced on.  §6.2's bracketed full-network
                               # count is read straight off these.
                               network=net_used),
                )
            )
            if cfg.log_every and it % cfg.log_every == 0:
                print(
                    f"  it {it:4d}  LP={lp.objective:12.3f}  cols={len(self.master):5d}  "
                    f"added={added:3d}  min_rc={res.min_reduced_cost:+.6f}  "
                    f"dpi={pi_move:6.3f}  [{used}]  net={net_used}"
                )

            if added == 0:
                # Stop when the pricer can certify that no negative column
                # exists. For an exact pricer that is min_reduced_cost >= 0.
                # For a relaxed pricer the ng-route lower bound works too,
                # because a relaxation can only under-estimate reduced costs,
                # so a non-negative bound is a real certificate.
                # The extra guard matters: a bound measured against a smoothed
                # dual is not a statement about the master, so it must never be
                # allowed to close the run as "optimal".  `certifies_no_negative`
                # is itself False for any masked call, so the verdict below can
                # only ever be reached on the full network (spec §7.2 item 1).
                # `lp.dual_verified` is the other half of the same guard, one
                # level up: a certificate is a claim about the duals, and an
                # unverified dual cannot support one.
                if (res.certifies_no_negative and on_true_duals
                        and lp.dual_verified
                        and res.stats.get("priced_on_true_duals", True)):
                    optimal = True
                break

            if cfg.stall_limit:
                if plateau_improved:
                    stall = 0
                else:
                    stall += 1
                    if stall >= cfg.stall_limit:
                        # v3.6 ③: a truncated round is BLIND, not converged.
                        # When every pricing call the run ever made was cut off
                        # by the cap, "30 consecutive rounds with no LP
                        # improvement" is a fact about the cap, not about the
                        # instance -- and reporting it as PLATEAU would publish
                        # that artefact as a finding about the instance.  Leave
                        # `stalled` False so the three-state status reports
                        # INCOMPLETE instead.
                        if complete_pricing_calls > 0:
                            # v8: a stall seen only through the mask is not a
                            # certificate about the master.  Before stopping,
                            # ask for one true-dual full-network pass: no
                            # negative column there closes the run as optimal;
                            # columns there extend the run (stall resets).
                            pricing_calls += 1
                            _t_pp = time.perf_counter()
                            if hasattr(self.pricer, "set_mask"):
                                self.pricer.set_mask(None)
                            fres = self._price(False, lp.pi, lp.sigma,
                                               cfg.max_columns_per_iter)
                            pp_seconds += time.perf_counter() - _t_pp
                            _note_price(fres)
                            if hasattr(self.pricer, "set_mask"):
                                self.pricer.set_mask(self._mask)
                            if (fres.certifies_no_negative and on_true_duals
                                    and lp.dual_verified
                                    and fres.stats.get("priced_on_true_duals", True)):
                                optimal = True
                                break
                            _t_rmp = time.perf_counter()
                            _added_recheck = self._admit(
                                fres.columns, lp.pi, lp.sigma,
                                active=active_routes)
                            rmp_seconds += time.perf_counter() - _t_rmp
                            if _added_recheck:
                                stall_rechecks += 1
                                if stall_rechecks >= 2:
                                    # Two full-network rechecks in a row returned
                                    # negative columns the LP never uses: that is
                                    # dual degeneracy, not slow convergence.
                                    stalled = True
                                    break
                                stall = 0
                                continue
                            stalled = True
                        break

            # ---- §1: close out this iteration's warm-up log ---------------
            if it <= cfg.mask_after and mask_available:
                warm_cols.append([tuple(int(v) for v in r) for r, _ in res.columns])

            # ---- §7.2 item 2: warm-up ends, features are frozen, mask goes in
            if mask_available and not hook_fired and it >= cfg.mask_after:
                hook_fired = True
                trace = WarmupTrace(
                    pi=list(warm_pi[: cfg.mask_after]),
                    sigma=list(warm_sigma[: cfg.mask_after]),
                    reduced_cost=warm_rc[: cfg.mask_after],
                    columns=warm_cols[: cfg.mask_after],
                    lp_objective=warm_obj[: cfg.mask_after],
                )
                self._mask = cfg.mask_hook(trace)
                if self._mask is not None:
                    masked_arcs = int(self.pricer.set_mask(self._mask))
                else:
                    self._mask = None
                    masked_arcs = 0
                # A mask that removes nothing leaves the pricer on the exact
                # unmasked path, so `use_reduced` stays False and the run is
                # bit-identical to the baseline -- which is the property the
                # full-network control arm rests on.
                use_reduced = masked_arcs > 0

        # finalise
        _t_rmp = time.perf_counter()
        lp = self.master.solve_lp()
        rmp_seconds += time.perf_counter() - _t_rmp
        try:
            ip = self.master.solve_ip()
            ip_obj = ip.objective
            ip_routes = [self.master.routes[j] for j in ip.chosen]
        except Exception:
            ip_obj = float("nan")
            ip_routes = []

        # The incumbent can be tightened at the end by the IP solve, whose value
        # is a feasible cover just like the seed's.  Reporting the tighter of the
        # two makes the gap diagnostic as informative as it can be.
        if cfg.diagnose_incumbent and np.isfinite(ip_obj):
            incumbent = min(float(incumbent), float(ip_obj))
        bound_meets_incumbent = bool(
            cfg.diagnose_incumbent and np.isfinite(incumbent)
            and float(lp.objective) >= incumbent - cfg.incumbent_tol
        )

        return CGResult(
            snapshots=snapshots,
            lp_objective=lp.objective,
            ip_objective=ip_obj,
            ip_routes=ip_routes,
            num_vehicles=len(ip_routes),
            iterations=len(snapshots),
            columns_in_pool=len(self.master),
            pricing_calls=pricing_calls,
            exact_confirmation_calls=exact_calls,
            wall_clock=time.perf_counter() - t0,
            optimal_lp=optimal,
            exact_confirmation_skipped=confirmation_skipped,
            stalled=stalled,
            stabilized_iterations=stabilized_iters,
            filtered_columns=filtered_out,
            offered_columns=self._offered,
            dropped_by_selection=self._dropped,
            truncated_pricing_calls=truncated_calls,
            unscheduled_truncations=unscheduled_truncations,
            peak_labels_created=peak_labels,
            label_cap=used_label_cap,
            full_network_iterations=full_iters,
            reduced_network_iterations=reduced_iters,
            masked_pricing_calls=masked_calls,
            masked_arcs=masked_arcs,
            arcs_total=(self.pricer.arcs_total
                        if hasattr(self.pricer, "arcs_total") else 0),
            arcs_removed_total=arcs_removed_total,
            masked_arcs_mean=(arcs_removed_total / masked_mean_calls
                              if masked_mean_calls else 0.0),
            reduced_dead=reduced_dead,
            mask_hook_fired=hook_fired,
            reduced_kill_blocked_by_truncation=kill_blocked,
            pp_seconds=pp_seconds,
            rmp_seconds=rmp_seconds,
            bound_meets_incumbent=bound_meets_incumbent,
            incumbent_value=float(incumbent),
            unverified_dual_iterations=unverified_iters,
        )
