"""Construction heuristics used to seed the master problem.

Column generation needs a feasible starting basis. Singletons alone are not
enough whenever the fleet size is binding, so the master is seeded with a
cheapest-insertion solution that covers every customer.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from .instance import EPS, Instance, evaluate_route

Route = Tuple[int, ...]


def cheapest_insertion(
    inst: Instance,
    customers: Optional[Sequence[int]] = None,
    max_routes: Optional[int] = None,
) -> List[Route]:
    """Sequential cheapest-insertion over an urgency-ordered customer list."""
    order = sorted(
        list(customers if customers is not None else inst.customers),
        key=lambda i: (float(inst.due[i]), float(inst.dist[0, i])),
    )
    routes: List[Route] = []
    route_cost: List[float] = []

    for c in order:
        best = None  # (delta, route_index, position)
        for ri, r in enumerate(routes):
            for p in range(len(r) + 1):
                cand = r[:p] + (c,) + r[p:]
                ev = evaluate_route(inst, cand)
                if not ev.feasible:
                    continue
                delta = ev.cost - route_cost[ri]
                if best is None or delta < best[0] - EPS:
                    best = (delta, ri, p)
        # Insert whenever a feasible position exists; only open a new route
        # when no existing route can absorb this customer. Serving a customer
        # always beats respecting max_routes here, because the fleet limit is
        # enforced by the master problem, not by the seed heuristic.
        if best is not None:
            delta, ri, p = best
            r = routes[ri]
            routes[ri] = r[:p] + (c,) + r[p:]
            route_cost[ri] = route_cost[ri] + delta
        else:
            routes.append((c,))
            route_cost.append(evaluate_route(inst, (c,)).cost)
    return routes


def singleton_routes(inst: Instance) -> List[Route]:
    """One route per customer. The v3.4 standard seed for every formal run.

    Why this and not `cheapest_insertion`
    -------------------------------------
    The seed decides the RMP's first optimal dual *face*, and that face decides
    whether column generation can make progress at all (spec v3.4; see the
    seed-face diagnosis in the internal design notes, §4.5, not shipped here).

    `cheapest_insertion` returns 3-4 giant routes that happen to partition the
    customers.  The first LP then has exactly that many tight constraints, every
    dual of the form "one representative customer per route takes the whole
    route cost, everyone else takes zero" is optimal, and the face has a
    dimension of up to n - k.  With a face that large the solver returns an
    arbitrary vertex, the pricer finds routes that violate *that vertex*,
    the bound does not move, another vertex comes back, and the run spins
    without ever certifying.  Measured on R201: 60/60 dual coordinates free,
    one coordinate spanning [0, 494.59].

    Singletons pin the dual down to a single point.  Each column `(i,)` is the
    constraint `pi_i <= 2*d(0,i)`, and the seed LP value is exactly
    `sum_i 2*d(0,i)`, so `sum_i pi_i = sum_i 2*d(0,i)` with every coordinate at
    its cap -- the optimal dual is unique, reduced costs are trustworthy, and a
    negative reduced cost strictly improves the LP.  Measured on R201: 0/60
    coordinates free.

    The price is real and must not be hidden: pricing recovers the full label
    space (C201 0.7 s -> 249 s per call).  That is the honest cost of a baseline
    whose bound actually means something.

    Infeasible singletons are dropped; a customer that cannot be served on its
    own is a data problem, not something a seed should paper over.
    """
    routes: List[Route] = []
    for i in inst.customers:
        r = (int(i),)
        if evaluate_route(inst, r).feasible:
            routes.append(r)
    if not routes:
        raise RuntimeError("no customer can be served in a singleton route")
    return routes


def nearest_neighbour(inst: Instance) -> List[Route]:
    """Plain nearest-neighbour sweep, useful as a weaker reference solution."""
    unserved = set(inst.customers)
    routes: List[Route] = []
    while unserved:
        seed = min(unserved, key=lambda i: float(inst.due[i]))
        path: List[int] = []
        node, time, load = 0, 0.0, 0.0
        while True:
            nxt, best = -1, float("inf")
            for j in unserved:
                if load + inst.demand[j] > inst.params.capacity + EPS:
                    continue
                arr = time + inst.service[node] + inst.travel_time[node, j]
                if arr > inst.due[j] + EPS:
                    continue
                if float(inst.dist[node, j]) < best:
                    nxt, best = j, float(inst.dist[node, j])
            if nxt < 0:
                break
            arr = time + inst.service[node] + inst.travel_time[node, nxt]
            time = arr if arr > inst.ready[nxt] else inst.ready[nxt]
            load += float(inst.demand[nxt])
            path.append(nxt)
            unserved.discard(nxt)
            node = nxt
        if not path:
            raise RuntimeError(f"customer {seed} cannot be served in a singleton route")
        routes.append(tuple(path))
    return routes


def is_covering(inst: Instance, routes: Sequence[Route]) -> bool:
    seen = set()
    for r in routes:
        if not evaluate_route(inst, r).feasible:
            return False
        seen.update(r)
    return seen >= set(inst.customers)


def report(inst: Instance, routes: Sequence[Route]) -> dict:
    total = 0.0
    energy = 0.0
    for r in routes:
        ev = evaluate_route(inst, r)
        total += ev.cost
        energy += ev.energy
    return {
        "num_routes": len(routes),
        "cost": total,
        "energy": energy,
        "covering": is_covering(inst, routes),
    }
