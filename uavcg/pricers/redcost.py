"""RedCost-S: the train-free dynamic arc reduction (Morabit, Desaulniers & Lodi).

Why this is a first-class deliverable and not a footnote
-------------------------------------------------------
The paper this project reproduces reports its own negative result in Table 7:
``RedCost-S`` -- a completely train-free heuristic that recomputes the reduced
arc set from the current duals at every pricing call -- beats the learned ``ML-S``
in every class and at both sizes (200 customers: 84% / 81% / 66% acceleration on
R2 / RC2 / C2 against 70% / 48% / 16% for ML-S), and at 400 customers on C2 the
learned version is *slower than the baseline* (-4%).

So ``RedCost-S`` is the floor the learned method has to clear. A project that
reports "we made column generation faster" without it has said nothing, because
the cheapest available alternative already did. It is also the reason the whole
approach is worth attempting at all: if a per-iteration greedy beats a
per-instance classifier, the interesting question stops being "can learning
help" and becomes "what does learning see that a greedy cannot" -- which is
exactly the question the two project changes (dynamic dual features, calibrated
probability) are designed to answer.

The rule
--------
At every pricing call, with ``c~_ij = c_ij - pi_j`` from the *current* duals:

    keep arc (i, j) if it is among the N_min cheapest arcs leaving i
                      OR among the N_min cheapest arcs entering j,

with ``N_min`` walking an increasing ladder ``(10, 20, None)``. ``None`` means
the full network. The union (rather than intersection) of the two rules matters:
the outgoing rule alone severs the head of every route, and a route that cannot
be entered cannot be built.

Why the ladder, and why it ends at the full network
---------------------------------------------------
The reduction is a heuristic restriction of the pricing network, so a "no
negative column" answer from a reduced rung proves nothing about the original
problem. Escalating on an empty answer and *ending the ladder at the full
network* is what keeps certification: at that rung the pricer is an ordinary
ng-route pricer on G, it publishes its relaxation bound, and the CG loop can
close. The escalation is monotone -- a rung that returned nothing is not retried,
because the duals only get more restrictive as columns accumulate and a rung that
found nothing once has nothing left to offer.

Mechanically this reuses the §7.2 arc mask: ``set_mask`` on the inner pricer. The
returned ``PricingResult`` therefore carries ``masked=True`` on every reduced
rung, which is the contract field that stops that call from publishing a bound or
certifying anything. No new safety machinery was needed for this baseline -- that
is what the §7.2 item 1 work bought.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional, Sequence, Tuple

import numpy as np

from ..instance import Instance
from .base import (
    Pricer,
    PricingResult,
    depot_open,
    maskable_arc_count,
    modified_cost,
    removed_arcs,
)
from .ngroute import NgRoutePricer

#: Default ladder. 10 and 20 are the paper's values; the final rung is the full
#: network, which is what makes the ladder certification-preserving.
DEFAULT_LADDER: Tuple[Optional[int], ...] = (10, 20, None)


class RedCostArcPricer(Pricer):
    """Per-iteration reduced-cost arc reduction (spec §7.5 step 5, plan §3.9).

    Wraps an `NgRoutePricer`; the wrapping is where the reduction lives, so the
    labelling algorithm itself is untouched and the two arms differ in exactly
    one thing.
    """

    name = "redcost"
    guarantees_optimality = False

    def __init__(
        self,
        inst: Instance,
        ladder: Sequence[Optional[int]] = DEFAULT_LADDER,
        k: int = 10,
        max_labels: int = 3_000_000,
    ) -> None:
        super().__init__(inst)
        self.inner = NgRoutePricer(inst, k=k, max_labels=max_labels)
        self.ladder: Tuple[Optional[int], ...] = tuple(ladder)
        if not self.ladder:
            raise ValueError("the ladder must have at least one rung")
        if self.ladder[-1] is not None:
            raise ValueError(
                "the last rung of the ladder must be the full network (None). "
                "A ladder that never returns to G can never certify, and the "
                "reduced network would then be silently reported as optimal."
            )
        if any(r is not None and r <= 0 for r in self.ladder):
            raise ValueError(f"non-positive rung in ladder {self.ladder!r}")

        #: Current floor of the ladder; index into `ladder`. Monotone.
        self._rung = 0
        #: Optional outer mask (the learned mask of plan §4.1 arm A4). Stored
        #: normalised so the composition below is a plain elementwise AND.
        self._outer: Optional[np.ndarray] = None

        # diagnostics, cumulative over the life of the pricer
        self.calls = 0
        self.escalations = 0
        self.masked_calls = 0

    # ------------------------------------------------------------------
    # ladder / mask plumbing
    # ------------------------------------------------------------------
    @property
    def rung(self) -> int:
        return self._rung

    @property
    def rung_n_min(self) -> Optional[int]:
        return self.ladder[self._rung]

    @property
    def arcs_total(self) -> int:
        return maskable_arc_count(self.inst.n)

    @property
    def masked_arcs(self) -> int:
        """Arcs removed by the outer mask, read by the CG loop's bookkeeping."""
        if self._outer is None:
            return 0
        return removed_arcs(self._outer, self.inst.n)

    def set_mask(self, mask) -> int:
        """Install an outer mask, intersected with the per-iteration reduction.

        Present so that CG's warm-up hook can drive this pricer as well: plan
        §4.1 arm A4 is the composition of the learned mask with RedCost-S, and a
        composition is only well defined if both masks pass through one place.
        For the plain RedCost-S arm this is never called.
        """
        if mask is None:
            self._outer = None
            return 0
        n = self.inst.n
        a = np.asarray(mask)
        if a.shape != (n + 1, n + 1):
            raise ValueError(
                f"arc mask must have shape ({n + 1}, {n + 1}), got {a.shape}"
            )
        self._outer = depot_open(a)
        return removed_arcs(self._outer, n)

    # ------------------------------------------------------------------
    def _reduced_mask(self, pi: np.ndarray, n_min: int) -> np.ndarray:
        """The N_min-cheapest-in-or-out mask for the duals `pi`.

        Vectorised rather than looped: this runs once per pricing call, and at
        n=60 a call is ~0.5 s, so the ranking must not become the bottleneck of
        the *baseline* -- an inflated baseline would flatter the method under
        test. Two argsorts over a 61x61 matrix is ~0.1 ms.
        """
        n = self.inst.n
        m = modified_cost(self.inst, pi)
        # Self-arcs are not arcs. Sending them to +inf before sorting removes
        # the "skip j == i" special case from both rules below; it cannot change
        # any survivor, since a self-arc was never among the cheapest *arcs*.
        np.fill_diagonal(m, np.inf)

        keep = np.zeros((n + 1, n + 1), dtype=bool)
        # n_min cheapest arcs leaving each node (targets are customers only)
        out = np.argsort(m[:, 1:], axis=1, kind="stable")[:, :n_min] + 1
        keep[np.arange(n + 1)[:, None], out] = True
        # n_min cheapest arcs entering each node (tails include the depot)
        into = np.argsort(m, axis=0, kind="stable")[:n_min, :]
        keep[into, np.broadcast_to(np.arange(n + 1), into.shape)] = True
        return keep

    def _effective_mask(self, pi: np.ndarray, n_min: Optional[int]):
        if n_min is None:
            return self._outer
        dyn = depot_open(self._reduced_mask(pi, n_min))
        if self._outer is None:
            return dyn
        return dyn & self._outer

    # ------------------------------------------------------------------
    def price(
        self,
        pi: np.ndarray,
        sigma: float = 0.0,
        fixed_cost: float = 0.0,
        max_columns: int = 50,
    ) -> PricingResult:
        """Price on the current rung, escalating while it yields nothing.

        The escalation loop is inside the call, so one call always does its best:
        it never returns "nothing" while a looser rung it is entitled to use
        could have found a column. The rung reached is remembered, which is what
        makes the ladder monotone across the solve.
        """
        escalations_here = 0
        while True:
            n_min = self.ladder[self._rung]
            mask = self._effective_mask(pi, n_min)
            try:
                self.inner.set_mask(mask)
                res = self.inner.price(
                    pi, sigma=sigma, fixed_cost=fixed_cost,
                    max_columns=max_columns,
                )
            finally:
                # Leave the inner pricer unmasked. Cheap insurance: this pricer
                # hands the same object to nothing else, but an inner pricer that
                # is ever left masked is exactly the failure §7.2 item 1 exists
                # to prevent, so it costs nothing to make it unreachable.
                self.inner.set_mask(None)

            self.calls += 1
            if res.masked:
                self.masked_calls += 1

            if res.columns or n_min is None:
                # Rebuilt through `dataclasses.replace` rather than by mutating
                # `stats`, so `__post_init__` re-checks the masked/truncated
                # contract on the way out. A rebuild is the one place a contract
                # field can quietly go missing, and this pricer rebuilds on
                # every single call.
                return replace(
                    res,
                    stats=dict(
                        res.stats,
                        redcost_rung=self._rung,
                        redcost_n_min=(-1 if n_min is None else n_min),
                        redcost_escalated_here=escalations_here,
                        redcost_arcs_removed=(
                            0 if mask is None else removed_arcs(mask, self.inst.n)
                        ),
                    ),
                )

            # An empty answer from a reduced rung is not a statement about G, so
            # relax the reduction and ask again rather than reporting it.
            self._rung += 1
            self.escalations += 1
            escalations_here += 1
