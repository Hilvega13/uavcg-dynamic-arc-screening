"""ng-route relaxed pricing - the workhorse for anything above ~20 customers.

Why this exists
---------------
Measured on Solomon C101, exact elementary labelling needs 3.9k labels at
n=10, 51k at n=15, 765k at n=20 and 2.5M at n=25 (0.07s -> 264s). Each extra
five customers multiplies the effort by roughly fifteen. Full elementarity is
therefore unusable beyond n~20, which is far below the 50-100 customer
instances a VRPTW study has to report on.

ng-route relaxation (Baldacci, Mingozzi & Roberti) replaces the exact
"visited" memory of a label with a neighbourhood memory:

    NG(j)  = {j} plus the k nearest customers of j
    when extending to j:  j must not be in the label's ng-set
    new ng-set          = (ng_set AND NG(j)) OR {j}

A customer may therefore be revisited once it has dropped out of the
neighbourhood memory, which makes the subproblem a relaxation: it can only
under-estimate reduced costs. Two consequences, both used explicitly here:

  * routes handed back to the master must be checked for elementarity, since
    the relaxation may build a route that visits a customer twice;
  * the smallest reduced cost over ALL closed labels is a valid LOWER BOUND
    on the true minimum. When that bound is >= 0, no negative column can
    exist and column generation is genuinely optimal, even though the
    columns themselves were produced heuristically.

That second point is what `relaxed_min_reduced_cost` on PricingResult is for,
and it is what lets the loop certify optimality on instances where the exact
pricer cannot run at all.
"""

from __future__ import annotations

import time
from bisect import bisect_right
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..instance import DEPOT, EPS, Instance, evaluate_route
from .base import (
    Pricer,
    PricingResult,
    RouteColumn,
    depot_open,
    maskable_arc_count,
    removed_arcs,
)


@dataclass(slots=True)
class NgLabel:
    node: int
    cost: float
    time: float
    load: float
    ng: int              # bitmask of customers the label may not revisit
    path: Tuple[int, ...]


def _dominates(a: NgLabel, b: NgLabel) -> bool:
    """Reference dominance predicate.  Must stay equivalent to the inlined one.

    `price()` no longer calls this -- it is the specification, not the hot path.
    Keeping it as a separate, obviously-correct function and testing the fast
    path against it is the only way to make a 170x-comparison-per-label rewrite
    safe, so do not "simplify" it away.
    """
    if a.node != b.node:
        return False
    if a.cost > b.cost + EPS or a.time > b.time + EPS or a.load > b.load + EPS:
        return False
    # anything still addable for b must still be addable for a
    return (a.ng | b.ng) == b.ng


class NgRoutePricer(Pricer):
    name = "ng-route"
    guarantees_optimality = False

    def __init__(
        self,
        inst: Instance,
        k: int = 10,
        max_labels: int = 3_000_000,
        arc_mask=None,
        max_seconds: Optional[float] = None,
    ) -> None:
        super().__init__(inst)
        self.k = int(k)
        self.max_labels = int(max_labels)
        #: Wall-clock cap for a single `price()` call, or None for "no cap".
        #:
        #: SCREENING ONLY -- do not set this in a formal run.  It exists because
        #: the label cap below bounds *work*, not *time*: reaching 3 M labels was
        #: measured at 40 min to 5 h on the heavy n=60 instances, so a budget
        #: checked between calls (which is where the caller can check it) cannot
        #: bound anything.  Only a check inside the label loop can.
        #:
        #: The contract is exactly the label cap's: the surviving labels are a
        #: SUBSET, so the call sets `truncated`, publishes no bound, and can
        #: certify nothing (spec §3.5.1).  Truncation is therefore fail-safe --
        #: a capped run can only ever be INCOMPLETE, never wrongly optimal --
        #: but it *is* wall-clock dependent, so its timings are not reproducible
        #: and it may never be quoted as experimental evidence.
        self.max_seconds = None if max_seconds is None else float(max_seconds)

        n = inst.n
        masks = [0] * (n + 1)
        for i in range(n + 1):
            others = sorted(
                (j for j in range(1, n + 1) if j != i),
                key=lambda j: float(inst.dist[i, j]),
            )[: self.k]
            m = 0
            for j in others:
                m |= 1 << j
            masks[i] = m
        self.ng_masks = masks

        #: Arc selection (spec §7.2 item 1).  `None` means "every arc open",
        #: which is the representation the baseline arm uses -- not a separate
        #: code path, so "no mask" and "all-True mask" cannot drift apart.
        self._all_j: Tuple[int, ...] = tuple(range(1, n + 1))
        self._allowed_rows: Optional[List[Tuple[int, ...]]] = None
        self._installed_mask = None
        self._masked_arcs = 0
        #: Arcs a mask is allowed to touch: see `base.maskable_arc_count`, n^2.
        #: Depot arcs are outside the set by construction, which is what makes
        #: n^2 the right denominator for the positive-rate gate of §3.6.
        self._arcs_total = maskable_arc_count(n)
        self.set_mask(arc_mask)

    # ------------------------------------------------------------------
    @property
    def masked_arcs(self) -> int:
        """How many arcs the installed mask removes (0 when none is installed)."""
        return self._masked_arcs

    @property
    def arcs_total(self) -> int:
        """Size of the maskable arc set, n^2 for n customers."""
        return self._arcs_total

    def set_mask(self, mask) -> int:
        """Install an arc mask; returns the number of arcs it removes.

        `mask` is a boolean array of shape ``(n+1, n+1)`` where ``mask[i, j]``
        is True when arc i->j may be used, or None for "no mask" (an exact
        synonym for an all-True mask).  Depot arcs -- ``(0, j)`` and ``(i, 0)``
        -- are forced open: every route leaves and returns to the depot, and
        spec §2 excludes them from arc selection, so a mask whose only zeros sit
        on depot arcs is equal to no mask at all and returns 0 here.

        Self-loops are not arcs and are never counted as removed.

        A mask that removes nothing leaves ``_allowed_rows`` as None, so the
        pricing loop takes exactly the unmasked path.  That identity is what the
        full-network control arm rests on, and it is tested rather than assumed.
        """
        n = self.inst.n
        if mask is None:
            self._allowed_rows = None
            self._installed_mask = None
            self._masked_arcs = 0
            return 0
        if mask is self._installed_mask:
            # The CG loop re-installs the same mask on every iteration; the
            # rebuild is O(n^2) and the cache makes it free.
            return self._masked_arcs
        a = np.asarray(mask)
        if a.shape != (n + 1, n + 1):
            raise ValueError(
                f"arc mask must have shape ({n + 1}, {n + 1}), got {a.shape}"
            )
        keep = depot_open(a)
        removed = removed_arcs(keep, n)

        self._installed_mask = mask
        self._masked_arcs = removed
        if removed == 0:
            self._allowed_rows = None
            return 0
        self._allowed_rows = [
            tuple(j for j in range(1, n + 1) if keep[i, j]) for i in range(n + 1)
        ]
        return removed

    # ------------------------------------------------------------------
    def price(
        self,
        pi: np.ndarray,
        sigma: float = 0.0,
        fixed_cost: float = 0.0,
        max_columns: int = 50,
    ) -> PricingResult:
        inst = self.inst
        n = inst.n
        cap = inst.params.capacity
        ng = self.ng_masks
        m = self._modified_cost(pi)
        thresh = sigma - fixed_cost

        # Arc selection. `rows` is None when no arc is being withheld, in which
        # case the successor loop below walks `all_j` -- the same tuple, in the
        # same order, as the `range(1, n + 1)` it replaced, so the unmasked run
        # is unchanged down to the order labels are created in.
        rows = self._allowed_rows
        all_j = self._all_j
        masked = self._masked_arcs > 0

        # Plain-Python copies of every array the inner loop touches.  Numpy
        # scalar indexing (`m[i, j]`) costs ~100 ns per element and the label
        # extension loop runs n times per label -- 11.5 M times on C201 n=60 --
        # so the conversion more than pays for itself and the values are the
        # same doubles.
        m_rows = m.tolist()
        tt_rows = inst.travel_time.tolist()
        demand_l = inst.demand.tolist()
        service_l = inst.service.tolist()
        ready_l = inst.ready.tolist()
        due_l = inst.due.tolist()
        load_limit = cap + EPS
        due_depot = inst.due[DEPOT] + EPS

        relaxed_min = 0.0
        non_elementary = 0
        found: Dict[Tuple[int, ...], float] = {}
        exact = True
        truncated = False
        #: Set when the cap that fired was the wall-clock one, so a probe can
        #: tell the two truncation reasons apart in `stats`.  Both mean the same
        #: thing to every consumer; only the diagnosis differs.
        time_truncated = False
        deadline = (
            None if self.max_seconds is None
            else time.perf_counter() + self.max_seconds
        )

        # Per-node label store, kept sorted by `time`, plus the parallel list of
        # times that makes the cut position findable by bisection.
        #
        # Why sorted by time: a label can only be dominated by one with
        # `time <= lab.time + EPS`, so only that prefix of the node's list needs
        # to be examined.  Measured before this change (C201 n=60): 32.5 M
        # dominance comparisons for 191 k labels = 170 per label, and 92.7% of
        # the whole call.  The cut is taken at `lab.time + EPS` rather than
        # `lab.time` because `_dominates` allows exactly that slack, which makes
        # the set of candidates tested a superset of the true dominators -- the
        # predicate result, and therefore every downstream result, is unchanged.
        per_node: List[List[NgLabel]] = [[] for _ in range(n + 1)]
        per_time: List[List[float]] = [[] for _ in range(n + 1)]
        stack: List[NgLabel] = [NgLabel(DEPOT, 0.0, 0.0, 0.0, 0, ())]
        created = 0
        processed = 0
        dom_checks = 0

        while stack:
            lab = stack.pop()
            processed += 1

            # Wall-clock cap; same contract as the label cap further down (a
            # subset of labels => `truncated`, no bound, nothing certifyable).
            # Sampled every 256 labels, which costs nothing next to the
            # dominance scan it is bounding, and short-circuits on `None` so
            # an uncapped call is unchanged.
            if (deadline is not None and not (processed & 0xFF)
                    and time.perf_counter() > deadline):
                exact = False
                truncated = True
                time_truncated = True
                break

            node = lab.node
            node_list = per_node[node]
            times = per_time[node]
            l_cost = lab.cost + EPS
            l_load = lab.load + EPS
            l_ng = lab.ng
            cut = bisect_right(times, lab.time + EPS)
            dom_checks += cut
            dominated = False
            for k in range(cut):
                other = node_list[k]
                if (other.cost <= l_cost and other.load <= l_load
                        and (other.ng | l_ng) == l_ng):
                    dominated = True
                    break
            if dominated:
                continue
            pos = bisect_right(times, lab.time)
            times.insert(pos, lab.time)
            node_list.insert(pos, lab)

            i = lab.node
            m_row = m_rows[i]
            tt_row = tt_rows[i]
            base_time = lab.time + service_l[i]
            base_load = lab.load
            base_ng = lab.ng
            # Arc selection is mounted here and nowhere else: removing an arc
            # only stops labels being created on it. Everything downstream --
            # the depot return below, the relaxed bound, and therefore every
            # certification in the CG loop -- is untouched by the mask, which is
            # why the certification circuit has to run unmasked (spec §7.2.1).
            for j in (all_j if rows is None else rows[i]):
                if (base_ng >> j) & 1:
                    continue
                if base_load + demand_l[j] > load_limit:
                    continue
                arr = base_time + tt_row[j]
                due_j = due_l[j]
                if arr > due_j + EPS:
                    continue
                ready_j = ready_l[j]
                start = arr if arr > ready_j else ready_j
                created += 1
                if created > self.max_labels:
                    exact = False
                    truncated = True
                    break
                stack.append(
                    NgLabel(
                        j,
                        lab.cost + m_row[j],
                        start,
                        base_load + demand_l[j],
                        (base_ng & ng[j]) | (1 << j),
                        lab.path + (j,),
                    )
                )
            if not exact:
                break

            if lab.path:
                back = base_time + tt_row[DEPOT]
                if back <= due_depot:
                    rc = lab.cost + m_row[DEPOT] + fixed_cost - sigma
                    if rc < relaxed_min:
                        relaxed_min = rc
                    # only elementary routes are legitimate columns
                    if rc < -EPS:
                        if len(set(lab.path)) == len(lab.path):
                            if lab.path not in found or rc < found[lab.path]:
                                found[lab.path] = rc
                        else:
                            non_elementary += 1

        feasible: List[RouteColumn] = []
        for path, rc in found.items():
            if not evaluate_route(inst, path).feasible:
                continue
            feasible.append((path, rc))
        feasible.sort(key=lambda kv: kv[1])

        # A truncated search only visited part of the label space, so
        # `relaxed_min` is an UPPER bound on the true relaxed minimum -- the
        # wrong direction for a bound. Publishing it would let the CG loop
        # certify an LP it never proved (spec §3.5.1), so the field goes to
        # None and `truncated` carries the reason. The raw measurement is kept
        # in `stats["relaxed_min"]` for diagnosis only.
        #
        # A masked call fails the same test for a different reason: the arcs it
        # withheld were never expanded, so the minimum is a bound on the reduced
        # network and says nothing about G (spec §7.2.1). Both reasons suppress
        # the bound, and PricingResult refuses to be constructed with either of
        # them set and a bound present.
        bound = None if (truncated or masked) else relaxed_min
        return PricingResult(
            columns=feasible[:max_columns],
            min_reduced_cost=feasible[0][1] if feasible else 0.0,
            exact=False,
            relaxed_min_reduced_cost=bound,
            truncated=truncated,
            masked=masked,
            stats={
                "labels_created": created,
                "labels_processed": processed,
                "negative_found": len(feasible),
                "non_elementary_pruned": non_elementary,
                "relaxed_min": relaxed_min,
                "label_cap": self.max_labels,
                "truncated": truncated,
                #: Present only when the wall-clock cap fired, so a run that
                #: never sets `max_seconds` produces byte-identical stats --
                #: the bit-level anchors fingerprint those dicts.
                **({"time_truncated": True} if time_truncated else {}),
                "k": self.k,
                # Dominance comparisons actually performed, and what they cost
                # per label.  Before the time-sorted store this was ~170 on
                # C201 n=60; it is the quantity spec §3.7 is about, so it is
                # reported rather than inferred.
                "dom_checks": dom_checks,
                "dom_checks_per_label": (dom_checks / created) if created else 0.0,
                # Arc selection, so a masked run can report the masked-arc share
                # (§6.2) from its own snapshots instead of from a side channel.
                "masked": masked,
                "masked_arcs": self._masked_arcs,
                "arcs_total": self._arcs_total,
                "arcs_active": self._arcs_total - self._masked_arcs,
            },
        )
