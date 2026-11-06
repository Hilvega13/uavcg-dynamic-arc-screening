"""Label collection: what makes an arc worth choosing?

The reference paper trains a classifier on the vertices of an optimal
independent set.  The migration of that idea to routes is described in
`features.py`: the target becomes an *arc*.  This module builds the training
set by running column generation, and at chosen iterations asking a strong
pricer which arcs appear in an optimal route for the current duals.

Three decisions worth stating out loud, because each one is a place where a
careless implementation would produce a model that looks fine and is not:

1. **The label source is a relaxation.**  On the instances where training is
   affordable, the exact elementary pricer is used when it fits in the budget
   and the ng-route relaxation otherwise.  A relaxation can only *under*-
   estimate reduced costs, so it may report arcs of routes that are not
   actually elementary; those rows are dropped rather than trusted.  This is
   weak supervision and the training script reports which source was used.

2. **Ties are labelled positively, not arbitrarily.**  Several different
   routes can share the minimum reduced cost.  Picking one of them and
   calling every other arc negative would teach the model to avoid arcs of
   solutions that are just as good.  All routes within a relative tolerance
   of the minimum contribute their arcs as positives.

3. **Only arcs the sampler actually considered become samples.**  A label on
   an arc the search never reaches is worthless, and giving the model rows
   from a different state distribution than the one it will be used in is a
   classic way to get a model that validates well and helps nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from .features import NUM_FEATURES
from .heuristics import singleton_routes
from .instance import DEPOT, EPS, Instance, evaluate_route
from .pricers import ExactLabelingPricer, NgRoutePricer, RolloutPricer
from .pricers.base import Pricer
from .rmp import MasterProblem

Arc = Tuple[int, int]


@dataclass
class LabelBatch:
    X: np.ndarray
    y: np.ndarray
    arcs: np.ndarray
    instance: str = ""
    iteration: int = 0
    graded: bool = False

    def __len__(self) -> int:
        return int(self.X.shape[0])

    @property
    def positive_rate(self) -> float:
        """Mean target.  For a graded batch this is soft positive mass."""
        return float(self.y.mean()) if len(self) else 0.0

    @property
    def hard_positive_rate(self) -> float:
        """Fraction of rows the model would call positive at a 0.5 threshold."""
        return float((self.y > 0.5).mean()) if len(self) else 0.0

    @property
    def partial_rate(self) -> float:
        """Fraction of rows whose target is neither 0 nor 1.

        The point of grading is that this number is not zero.  If it is, the
        graded label has silently collapsed back onto the binary one and the
        comparison between the two is vacuous.
        """
        if not len(self):
            return 0.0
        return float(((self.y > 1e-9) & (self.y < 1.0 - 1e-9)).mean())


@dataclass
class CollectionReport:
    batches: List[LabelBatch] = field(default_factory=list)
    iterations_run: int = 0
    label_source: str = ""
    skipped_no_negative: int = 0
    non_elementary_dropped: int = 0

    def stacked(self) -> LabelBatch:
        if not self.batches:
            return LabelBatch(np.zeros((0, NUM_FEATURES)), np.zeros(0), np.zeros((0, 2), int))
        return LabelBatch(
            X=np.vstack([b.X for b in self.batches]),
            y=np.concatenate([b.y for b in self.batches]),
            arcs=np.vstack([b.arcs for b in self.batches]),
            instance="pooled",
            graded=self.batches[0].graded,
        )


# --------------------------------------------------------------------------
def near_optimal_arcs(
    inst: Instance,
    pi: np.ndarray,
    sigma: float,
    fixed_cost: float,
    source: Pricer,
    rel_tol: float = 0.02,
) -> Tuple[Dict[Arc, float], float, int]:
    """Arcs used by a near-optimal route, each graded by how good that route was.

    Returns `(margins, best_reduced_cost, number_of_tied_routes)`.

    The first label this project used was binary: an arc is on an optimal route
    or it is not, and *every* arc of *every* route within `rel_tol` of the best
    reduced cost was labelled 1 with equal weight.  That discards information
    the pricer had already computed, in two ways.

    Ties are not equally good.  When several routes share (or nearly share) the
    best violation, their common arcs are the real signal and the arcs that
    differ between them are noise - yet all of them were labelled 1.  And an
    arc whose best completing route is 2% off the optimum looks exactly like an
    arc on the optimum itself.

    Grading fixes both with the number already in hand.  For an arc that some
    near-optimal route uses, the grade is

        margin = 1 - (min_rc_through_that_arc - best_rc) / tol   in [0, 1]

    so arcs of the best route score 1.0, arcs that only appear in routes at the
    edge of the tolerance score near 0.0, and an arc no near-optimal route uses
    scores 0.0.  This is the reduced-cost-based target the earlier round of
    analysis argued for, in the cheapest form that needs no new machinery: the
    pricer already returns a reduced cost with every column.

    The sign was wrong once and this docstring kept the wrong formula for a
    round after the code was fixed.  `(best_rc - v) / tol` is <= 0 for every
    arc, so it labels the best route's own arcs 0.0 and cannot express "this
    arc is better" at all.  The self-test assertion that the optimal route's
    arcs score exactly 1.0 is what catches it; keep that assertion.
    """
    res = source.price(pi, sigma=sigma, fixed_cost=fixed_cost, max_columns=10**7)
    cols = [(r, rc) for r, rc in res.columns if rc < -EPS]
    if not cols:
        return {}, 0.0, 0

    best = min(rc for _r, rc in cols)
    tol = max(1e-9, rel_tol * abs(best))
    arc_rc: Dict[Arc, float] = {}
    tied = 0
    for route, rc in cols:
        if rc > best + tol:
            continue
        if len(set(route)) != len(route):
            continue
        tied += 1
        prev = DEPOT
        for j in route:
            k = (prev, j)
            cur = arc_rc.get(k)
            if cur is None or rc < cur:
                arc_rc[k] = rc
            prev = j
        k = (prev, DEPOT)
        cur = arc_rc.get(k)
        if cur is None or rc < cur:
            arc_rc[k] = rc

    margins = {k: 1.0 - (v - best) / tol for k, v in arc_rc.items()}
    return margins, best, tied


def optimal_arcs(
    inst: Instance,
    pi: np.ndarray,
    sigma: float,
    fixed_cost: float,
    source: Pricer,
    rel_tol: float = 0.02,
) -> Tuple[Set[Arc], float, int]:
    """The binary view of `near_optimal_arcs`: the arc set, no grades.

    Kept because a number of earlier results were computed with it, and because
    the graded label has to be able to degrade to the old one exactly.
    """
    margins, best, tied = near_optimal_arcs(inst, pi, sigma, fixed_cost,
                                            source, rel_tol)
    return set(margins), best, tied


def label_batch(
    inst: Instance,
    pi: np.ndarray,
    sigma: float,
    fixed_cost: float,
    source: Pricer,
    betas: Sequence[float] = (0.5, 1.0, 2.0),
    samples_per_beta: int = 120,
    rel_tol: float = 0.02,
    max_rows: int = 6000,
    seed: int = 0,
    iteration: int = 0,
    warmup_samples: int = 0,
    candidate_pool: Optional[int] = None,
    graded: bool = True,
) -> Tuple[Optional[LabelBatch], int]:
    """One dual solution -> a labelled batch of arc decisions.

    `warmup_samples` must be set to the *same* value that the pricer will use
    at inference time.  It controls the `rank_freq` feature, which is built
    from a separate batch of rollouts: leaving it at 0 while collecting and
    switching it on later would feed the model a feature it only ever saw as
    zero during training.  That is a silent train/test skew, so the parameter
    is threaded through explicitly rather than defaulted in one place.

    `graded=True` produces targets in [0, 1] from the reduced-cost margin of
    the best near-optimal route through each arc (see `near_optimal_arcs`);
    `graded=False` reproduces the earlier binary target exactly.
    """
    margins, best_rc, tied = near_optimal_arcs(
        inst, pi, sigma, fixed_cost, source, rel_tol)
    if not margins:
        return None, 0

    X_parts, y_parts, a_parts = [], [], []
    per_beta = max(1, max_rows // max(len(betas), 1))
    for k, beta in enumerate(betas):
        pricer = RolloutPricer(
            inst, scoring="dual", n_samples=samples_per_beta, beta=beta,
            seed=seed + k, deterministic_passes=1,
            warmup_samples=warmup_samples, candidate_pool=candidate_pool,
        )
        X, arcs = pricer.collect_arc_samples(
            pi, sigma=sigma, fixed_cost=fixed_cost,
            n_samples=samples_per_beta, max_samples=per_beta,
        )
        if X.shape[0] == 0:
            continue
        if graded:
            y = np.fromiter(
                (margins.get((int(i), int(j)), 0.0) for i, j in arcs),
                dtype=float, count=arcs.shape[0],
            )
        else:
            y = np.fromiter(
                (1.0 if (int(i), int(j)) in margins else 0.0 for i, j in arcs),
                dtype=float, count=arcs.shape[0],
            )
        X_parts.append(X); y_parts.append(y); a_parts.append(arcs)

    if not X_parts:
        return None, tied
    return (
        LabelBatch(
            X=np.vstack(X_parts), y=np.concatenate(y_parts),
            arcs=np.vstack(a_parts), instance=inst.name, iteration=iteration,
            graded=graded,
        ),
        tied,
    )


# --------------------------------------------------------------------------
def collect_from_cg(
    inst: Instance,
    label_source: Optional[Pricer] = None,
    max_iterations: int = 24,
    sample_every: int = 3,
    warm_columns: int = 10,
    betas: Sequence[float] = (0.5, 1.0, 2.0),
    samples_per_beta: int = 120,
    rel_tol: float = 0.02,
    max_rows: int = 6000,
    seed: int = 0,
    warmup_samples: int = 0,
    candidate_pool: Optional[int] = None,
    graded: bool = True,
) -> CollectionReport:
    """Run column generation and sample labels along the way.

    Sampling every few iterations rather than at every one follows the
    reference paper: consecutive dual solutions are highly correlated, so
    dense sampling buys almost no new information for a lot of labelling cost.

    `warmup_samples` is forwarded to `label_batch` and must match the pricer
    used at inference - see the note there.  `graded` selects between the
    reduced-cost-graded target and the earlier binary one, and is forwarded to
    `label_batch` unchanged so both modes see exactly the same sampled rows.
    """
    if label_source is None:
        label_source = default_label_source(inst)

    master = MasterProblem(inst)
    # v3.4 decision ⑥: training labels must come off the same seed the online
    # solves use.  `cheapest_insertion` left the master LP dual on a
    # high-dimensional optimal face, so the sampled trajectory described a
    # degenerate vertex the solver never actually visits; the singleton seed
    # pins the dual uniquely (`uavcg/heuristics.py::singleton_routes`).
    for r in singleton_routes(inst):
        ev = evaluate_route(inst, r)
        if ev.feasible:
            master.add(r, ev.cost)
    if len(master) == 0:
        raise RuntimeError("seeding produced no feasible column")

    report = CollectionReport(label_source=label_source.name)
    for it in range(1, max_iterations + 1):
        report.iterations_run = it
        lp = master.solve_lp()

        if (it - 1) % sample_every == 0:
            batch, tied = label_batch(
                inst, lp.pi, lp.sigma, master.vehicle_fixed_cost, label_source,
                betas=betas, samples_per_beta=samples_per_beta,
                rel_tol=rel_tol, max_rows=max_rows, seed=seed + it,
                iteration=it, warmup_samples=warmup_samples,
                candidate_pool=candidate_pool, graded=graded,
            )
            if batch is None:
                report.skipped_no_negative += 1
            else:
                report.batches.append(batch)

        res = label_source.price(
            lp.pi, sigma=lp.sigma, fixed_cost=master.vehicle_fixed_cost,
            max_columns=warm_columns,
        )
        if master.add_many(res.columns) == 0:
            break
    return report


def default_label_source(
    inst: Instance, exact_max_n: int = 14, ng_k: Optional[int] = None,
) -> Pricer:
    """Exact labelling where it is affordable, ng-route relaxation above that.

    Measured on Solomon C101 the exact pricer needs 1.5 s at n=15 and 105 s at
    n=20, so a training set at n=25 has to be labelled by the relaxation.
    `scripts/train_model.py` prints which one was used, because the strength of
    the labels bounds the strength of everything downstream.

    `ng_k` exposes the relaxation's neighbourhood size, and it is the knob that
    makes the label-source question testable at all.  Two facts, both measured
    rather than assumed (see the timing probe in the internal design notes, §16.x):

      * `min_reduced_cost` is *identical* for every k and for the exact pricer.
        The relaxation does not cost you the optimum; it costs you the *tie
        set*.  At n=25 on R105, k=6 returns 694 routes and the exact pricer
        returns 1280, all with the same best reduced cost.
      * k = n makes ng-route exact, which is why comparing "exact vs relaxed"
        at n <= 12 is vacuous - the default k is `min(n, 12)`, so at n = 12 the
        neighbourhood already covers every customer.  Any honest test of label
        degradation has to hold n fixed and move k.
    """
    if inst.n <= exact_max_n:
        return ExactLabelingPricer(inst)
    k = min(inst.n, 12) if ng_k is None else min(inst.n, int(ng_k))
    return NgRoutePricer(inst, k=k)
