"""Classical pricing baselines.

GreedyPricer walks the modified-cost graph with a nearest-neighbour rule from
several starting customers. It is deliberately weak - that is the point: it
is the "classical" baseline the learned pricer has to beat.

Because it never proves optimality, `guarantees_optimality` is False and
`min_reduced_cost` only reports the best it happened to find. The column
generation loop is responsible for falling back to an exact pricer before it
declares the LP optimal.
"""

from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..instance import DEPOT, EPS, Instance, evaluate_route
from .base import Pricer, PricingResult, RouteColumn


class GreedyPricer(Pricer):
    name = "greedy"
    guarantees_optimality = False

    def __init__(
        self,
        inst: Instance,
        restarts: int = 0,
        noise: float = 0.0,
        seed: int = 0,
    ) -> None:
        """restarts=0 -> start once from each customer (deterministic sweep)."""
        super().__init__(inst)
        self.restarts = int(restarts)
        self.noise = float(noise)
        self.rng = random.Random(seed)

    def _build(self, m: np.ndarray, first: int, pi: np.ndarray) -> Tuple[int, ...]:
        inst = self.inst
        cap = inst.params.capacity
        path: List[int] = []
        visited = 1 << DEPOT
        node, time, load = DEPOT, 0.0, 0.0
        nxt = first
        while True:
            if (visited >> nxt) & 1:
                break
            if load + inst.demand[nxt] > cap + EPS:
                break
            arr = time + inst.service[node] + inst.travel_time[node, nxt]
            if arr > inst.due[nxt] + EPS:
                break
            time = arr if arr > inst.ready[nxt] else inst.ready[nxt]
            load += float(inst.demand[nxt])
            visited |= 1 << nxt
            path.append(nxt)
            node = nxt

            best_j, best_val = -1, float("inf")
            for j in range(1, inst.n + 1):
                if (visited >> j) & 1:
                    continue
                if load + inst.demand[j] > cap + EPS:
                    continue
                arr_j = time + inst.service[node] + inst.travel_time[node, j]
                if arr_j > inst.due[j] + EPS:
                    continue
                val = float(m[node, j])
                if self.noise:
                    val += self.rng.uniform(-self.noise, self.noise)
                if val < best_val:
                    best_j, best_val = j, val
            if best_j < 0:
                break
            nxt = best_j
        return tuple(path)

    def price(
        self,
        pi: np.ndarray,
        sigma: float = 0.0,
        fixed_cost: float = 0.0,
        max_columns: int = 1,
    ) -> PricingResult:
        inst = self.inst
        m = self._modified_cost(pi)
        found: Dict[Tuple[int, ...], float] = {}

        starts: Sequence[int]
        if self.restarts > 0:
            starts = [self.rng.randrange(1, inst.n + 1) for _ in range(self.restarts)]
        else:
            starts = list(range(1, inst.n + 1))

        for s in starts:
            path = self._build(m, s, pi)
            if not path:
                continue
            node = path[-1]
            if evaluate_route(inst, path).feasible:
                back = 0.0
                # reduced cost = modified path cost + return arc - sigma + V
                rc = sum(float(m[a, b]) for a, b in zip((DEPOT,) + path[:-1], path))
                rc += float(m[node, DEPOT]) + fixed_cost - sigma
                if rc < -EPS and (path not in found or rc < found[path]):
                    found[path] = rc

        cols: List[RouteColumn] = sorted(found.items(), key=lambda kv: kv[1])
        min_rc = cols[0][1] if cols else 0.0
        return PricingResult(
            columns=cols[:max_columns],
            min_reduced_cost=min_rc,
            exact=False,
            stats={"negative_found": len(cols)},
        )
