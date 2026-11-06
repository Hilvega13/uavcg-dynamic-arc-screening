"""Pricing subproblem interface.

Every pricer answers the same question: given the current duals, which routes
have negative reduced cost?

Uniform contract
----------------
A route is an ordered tuple of customer indices, excluding the depot:
    (3, 7, 1)  means  0 -> 3 -> 7 -> 1 -> 0

Its reduced cost against the set covering master is
    rc(r) = sum_{arcs} dist_ij  +  V  -  sum_{i in r} pi_i  -  sigma
with V the fixed cost per route (drone sortie) and sigma <= 0 the dual of the
fleet-size constraint. Pricers minimise
    modified(r) = sum_{arcs} dist_ij - sum_{i in r} pi_i
and compare it against  thresh = sigma - V.

Implementations must set `min_reduced_cost` to the true minimum over ALL
routes they are able to consider, not just the ones they return: the column
generation loop uses that number to decide whether the LP relaxation is
optimal. A pricer may cap how many columns it returns, but it must not cap
what it measures.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..instance import DEPOT, Instance

RouteColumn = Tuple[Tuple[int, ...], float]   # (node sequence, reduced cost)

#: A pricing call counts as "clean" only while its label count stays this far
#: below the pricer's cap.  Hitting the cap invalidates the call outright;
#: coming within 20% of it means truncation was narrowly missed, so the call is
#: too close to the edge to be used as evidence.  Spec DynML-S §3.5.2.
CLEAN_LABEL_FRACTION = 0.8


def modified_cost(inst: Instance, pi: np.ndarray) -> np.ndarray:
    """dist with pi_j subtracted from every arc *entering* customer j.

    This is the single definition of the project's reduced-cost convention: the
    tail-attribution form ``c~_ij = c_ij - pi_j`` of spec §2.2.  It is
    deliberately a module function rather than a method, because the dynamic
    arc features B1..B6 are computed from exactly this matrix and a second copy
    of the convention would eventually drift from the first -- the symmetric
    split ``c_ij - pi_i/2 - pi_j/2`` agrees with this one at the *path* level
    but differs by ``-pi_i/2`` on every arc, which is precisely the kind of
    difference that is invisible in a spot check and fatal in a feature table.
    """
    m = inst.dist.copy()
    m[:, 1:] -= np.asarray(pi, dtype=float)[None, :]
    return m


def depot_open(mask: np.ndarray) -> np.ndarray:
    """Copy of `mask` with the depot arcs forced open.

    Spec §2 keeps `(0, j)` and `(i, 0)` out of arc selection: every route leaves
    and returns to the depot, so disabling either would not reduce the search, it
    would erase it. Applying this inside the mask installers rather than trusting
    each caller is the difference between "the depot cannot be closed" and "the
    depot is not supposed to be closed".
    """
    keep = np.asarray(mask).astype(bool, copy=True)
    keep[:, DEPOT] = True
    keep[DEPOT, :] = True
    return keep


def maskable_arc_count(n: int) -> int:
    """Size of the maskable arc set, which is exactly n^2.

    The set is ``{(i, j) : i in 0..n, j in 1..n, i != j}``: n+1 possible tails
    times n possible heads, less the n self-loops. This is the denominator §3.6's
    positive-rate gate is stated against, which is the only reason the identity
    is worth naming.
    """
    return n * n


def removed_arcs(mask: np.ndarray, n: int) -> int:
    """How many maskable arcs `mask` disables.

    Zeros on depot arcs and on self-loops are not "arcs removed", so an all-open
    mask -- and a mask written the naive way, with False down the depot column --
    both report 0. That is what lets the full-network control arm be recognised
    from the number alone.
    """
    body = np.asarray(mask)[1:, 1:]
    eye = np.eye(n, dtype=bool)
    return int(np.count_nonzero(~body & ~eye))


@dataclass
class PricingResult:
    columns: List[RouteColumn] = field(default_factory=list)
    min_reduced_cost: float = 0.0
    exact: bool = True
    stats: Dict[str, float] = field(default_factory=dict)
    #: A lower bound on the true minimum reduced cost over ALL feasible routes.
    #: For an exact pricer this equals min_reduced_cost. A relaxed pricer such
    #: as ng-route can produce a bound that is below the best elementary route
    #: it managed to build, which is what lets the column generation loop
    #: certify optimality even when its columns are only heuristic.
    relaxed_min_reduced_cost: Optional[float] = None
    #: True when the pricer stopped early because it hit its label cap.
    #:
    #: The surviving labels are a SUBSET of the ones the algorithm would
    #: otherwise have created, so any minimum taken over them is an *upper*
    #: bound on the true relaxed minimum -- the wrong direction for a bound.
    #: Before this field existed the truncated number was published anyway, and
    #: the loop could certify an LP it had never proved (spec §3.5).
    #:
    #: Contract for every consumer: `truncated` means "this call proves
    #: nothing".  Such a result may never certify optimality and must never be
    #: used as training or experimental evidence.
    truncated: bool = False
    #: True when this call priced on a network with at least one arc removed.
    #:
    #: Isomorphic to `truncated`, and for the same reason: a minimum taken over
    #: a *reduced* arc set bounds the reduced-network optimum, not the true one.
    #: A removed arc is one on which no label was ever created, so the value in
    #: `relaxed_min_reduced_cost` stops being a lower bound on the full network
    #: the moment an arc disappears -- and "no negative column exists" stops
    #: being a statement about the problem.  Hence the same treatment: no bound
    #: is published and no certification is possible (spec §7.2 item 1, v3.2 ⑤).
    #:
    #: Deliberately False when the installed mask removes nothing.  An all-True
    #: mask has to reproduce the unmasked solver bit for bit -- bound included --
    #: or the full-network control arm is not a control.
    masked: bool = False

    def __post_init__(self) -> None:
        """Enforce the "proves nothing => publishes nothing" contract in code.

        Spec v3.2 ⑤ asks for this as a code-level assertion rather than an audit
        log, which is the right call: both flags are ways of saying the same
        thing, and the failure they guard against (a bound from a restricted
        search being read as a bound on the unrestricted problem) is a silently
        wrong answer, not a slow one.  A convention would be checked by whoever
        remembers; this is checked by every construction.
        """
        if self.truncated and self.relaxed_min_reduced_cost is not None:
            raise AssertionError(
                "truncated pricing result published a bound: the surviving "
                "labels are a subset, so their minimum is an upper bound on the "
                "true relaxed minimum (spec §3.5.1)"
            )
        if self.masked and self.relaxed_min_reduced_cost is not None:
            raise AssertionError(
                "masked pricing result published a bound: arcs were removed, so "
                "the minimum is a bound on the reduced network only (spec §7.2.1)"
            )

    @property
    def num_negative(self) -> int:
        return len(self.columns)

    @property
    def certifies_no_negative(self) -> bool:
        """True when the pricer can prove no negative column exists.

        Neither a truncated nor a masked call can certify anything, whatever
        `exact` claims: in the first case the discarded labels may well have
        carried a negative reduced cost, in the second the arcs that were
        removed may well have carried one.
        """
        if self.truncated or self.masked:
            return False
        bound = self.relaxed_min_reduced_cost
        if bound is None:
            bound = self.min_reduced_cost if self.exact else None
        return bound is not None and bound >= -1e-6

    # ------------------------------------------------------------------
    @property
    def labels_created(self) -> int:
        """Labels built by this call, or 0 when the pricer does not report."""
        return int(self.stats.get("labels_created", 0) or 0)

    @property
    def label_cap(self) -> Optional[int]:
        """The pricer's label cap for this call, or None when unbounded."""
        cap = self.stats.get("label_cap")
        return int(cap) if cap else None

    def is_clean(self, frac: float = CLEAN_LABEL_FRACTION) -> bool:
        """Whether this call is usable as evidence (spec §3.5.2)."""
        if self.truncated:
            return False
        cap = self.label_cap
        if cap is None:
            return True
        return self.labels_created < frac * cap


class Pricer(ABC):
    name: str = "pricer"
    #: True when min_reduced_cost is guaranteed to be the global minimum.
    guarantees_optimality: bool = True

    def __init__(self, inst: Instance) -> None:
        self.inst = inst

    @abstractmethod
    def price(
        self,
        pi: np.ndarray,
        sigma: float = 0.0,
        fixed_cost: float = 0.0,
        max_columns: int = 50,
    ) -> PricingResult:
        """Return columns with negative reduced cost, cheapest first."""

    # ------------------------------------------------------------------
    def _modified_cost(self, pi: np.ndarray) -> np.ndarray:
        """dist with pi_j subtracted from every arc entering customer j."""
        return modified_cost(self.inst, pi)
