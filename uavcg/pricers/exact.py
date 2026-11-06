"""Exact pricing: labelling algorithm for the ESPPRC relaxation of the route.

Why energy is *not* a labelling resource
----------------------------------------
A drone lifts its whole payload at the depot and gets lighter as it delivers,
so the load on an arc is the demand still to be delivered - information that
does not exist yet while a route is being built forward. Making energy an
incremental resource would therefore either need a backward labelling scheme
or a conservative (load = capacity) approximation, and the approximation
would silently discard feasible routes, which breaks the optimality guarantee
of column generation.

So energy is evaluated exactly on completed routes instead. Dominance uses
(time, load) only. That is weaker pruning, never incorrect pruning: a label
that dominance keeps is always safe, and every completed route is checked
against the real energy model before it is accepted. The pricing problem
stays exact.

Resources and dominance
-----------------------
label at node i:  (cost, time, load, visited bitmask, path)

L1 dominates L2 when both sit at the same node and
    cost1 <= cost2, time1 <= time2, load1 <= load2, visited1 subset of visited2
because every completion available to L2 is also available to L1 (it can
skip nodes L1 has not visited) and costs no more, since all resource
extension functions are non-decreasing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from ..instance import DEPOT, EPS, Instance, evaluate_route
from .base import Pricer, PricingResult, RouteColumn


@dataclass(slots=True)
class Label:
    node: int
    cost: float
    time: float
    load: float
    visited: int
    path: Tuple[int, ...]


def _dominates(a: Label, b: Label) -> bool:
    if a.node != b.node:
        return False
    if a.cost > b.cost + EPS or a.time > b.time + EPS or a.load > b.load + EPS:
        return False
    # a.visited must be a subset of b.visited
    return (a.visited | b.visited) == b.visited


class ExactLabelingPricer(Pricer):
    """Exact ESPPRC pricing by forward labelling with full elementarity."""

    name = "exact-labeling"
    guarantees_optimality = True

    def __init__(self, inst: Instance, max_labels: int = 4_000_000) -> None:
        super().__init__(inst)
        self.max_labels = int(max_labels)

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
        m = self._modified_cost(pi)
        thresh = float(sigma) - float(fixed_cost)

        raw_min_rc = 0.0
        found: Dict[Tuple[int, ...], float] = {}
        exact = True
        truncated = False

        per_node: List[List[Label]] = [[] for _ in range(n + 1)]
        stack: List[Label] = [Label(DEPOT, 0.0, 0.0, 0.0, 1, ())]
        created = 0
        processed = 0

        while stack:
            lab = stack.pop()
            processed += 1
            if any(_dominates(other, lab) for other in per_node[lab.node]):
                continue
            per_node[lab.node].append(lab)

            i = lab.node
            # ---- extend to a customer -----------------------------------
            for j in range(1, n + 1):
                if (lab.visited >> j) & 1:
                    continue
                if lab.load + inst.demand[j] > cap + EPS:
                    continue
                arr = lab.time + inst.service[i] + inst.travel_time[i, j]
                if arr > inst.due[j] + EPS:
                    continue
                start = arr if arr > inst.ready[j] else inst.ready[j]
                created += 1
                if created > self.max_labels:
                    exact = False
                    truncated = True
                    break
                stack.append(
                    Label(
                        j,
                        lab.cost + float(m[i, j]),
                        start,
                        lab.load + float(inst.demand[j]),
                        lab.visited | (1 << j),
                        lab.path + (j,),
                    )
                )
            if not exact:
                break

            # ---- close the route ----------------------------------------
            if lab.path:
                back = lab.time + inst.service[i] + inst.travel_time[i, DEPOT]
                if back <= inst.due[DEPOT] + EPS:
                    rc = lab.cost + float(m[i, DEPOT]) + fixed_cost - sigma
                    if rc < raw_min_rc:
                        raw_min_rc = rc
                    if rc < -EPS:
                        if lab.path not in found or rc < found[lab.path]:
                            found[lab.path] = rc

        # energy (and every other completed-route check) is applied here
        feasible: List[RouteColumn] = []
        for path, rc in found.items():
            if not evaluate_route(inst, path).feasible:
                continue
            feasible.append((path, rc))

        feasible.sort(key=lambda kv: kv[1])
        # min_reduced_cost must describe FEASIBLE routes only. Reporting the
        # raw minimum here would keep the column generation loop alive forever
        # when the only negative columns turn out to be energy-infeasible.
        min_rc = feasible[0][1] if feasible else 0.0

        # A truncated search saw only part of the label space, so `min_rc` is an
        # UPPER bound on the true minimum, not a lower bound. Publishing it
        # would let the CG loop certify an LP it never proved (spec §3.5.1).
        bound = None if truncated else min_rc
        return PricingResult(
            columns=feasible[:max_columns],
            min_reduced_cost=min_rc,
            exact=exact,
            relaxed_min_reduced_cost=bound,
            truncated=truncated,
            stats={
                "labels_created": created,
                "labels_processed": processed,
                "negative_found": len(feasible),
                "raw_min_reduced_cost": raw_min_rc,
                "label_cap": self.max_labels,
                "truncated": truncated,
            },
        )
